"""Download datasets from HuggingFace and convert to unified JSONL format.

Usage:
    python -m data.download_and_format                                  # all datasets, all splits
    python -m data.download_and_format --datasets hotpotqa --max-samples 100  # 100 examples per split
    python -m data.download_and_format --datasets gsm8k hotpotqa finqa  # milestone 2 minimum
    python -m data.download_and_format --datasets hotpotqa --splits dev --max-samples 50  # inspection
"""

import argparse
import json
import os
import re
import urllib.request
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def get_data_root() -> Path:
    root = os.environ.get("DATA_ROOT", "./data_local")
    p = Path(root)
    p.mkdir(parents=True, exist_ok=True)
    (p / "processed").mkdir(exist_ok=True)
    (p / "eval_splits").mkdir(exist_ok=True)
    return p


# ── Per-dataset answer extraction ──

def _extract_gsm8k_answer(ex: dict) -> str:
    match = re.search(r"####\s*(.+)", ex["answer"])
    return match.group(1).strip() if match else ex["answer"].strip()


def _extract_math_answer(ex: dict) -> str:
    solution = ex["solution"]
    matches = list(re.finditer(r"\\boxed\{", solution))
    if matches:
        start = matches[-1].end()
        depth, i = 1, start
        while i < len(solution) and depth > 0:
            if solution[i] == "{": depth += 1
            elif solution[i] == "}": depth -= 1
            i += 1
        if depth == 0:
            return solution[start:i - 1].strip()
    return solution.strip()


def _extract_nq_answer(ex: dict) -> list[str]:
    answers = ex.get("answer", [])
    if isinstance(answers, list):
        return answers if answers else [""]
    return [str(answers)]


def _extract_triviaqa_answer(ex: dict) -> list[str]:
    answer_obj = ex.get("answer", {})
    if isinstance(answer_obj, dict):
        aliases = answer_obj.get("aliases", [])
        value = answer_obj.get("value", "")
        all_answers = list(set(aliases + [value])) if value else aliases
        return all_answers if all_answers else [""]
    return [str(answer_obj)]


def _format_finqa_question(ex):
    """Build FinQA question with table context (pre_text + table + post_text + question)."""
    parts = []
    if ex.get("pre_text"):
        parts.append("\n".join(ex["pre_text"]))
    if ex.get("table"):
        rows = []
        for i, row in enumerate(ex["table"]):
            rows.append("| " + " | ".join(str(c) for c in row) + " |")
            if i == 0:
                rows.append("|" + " --- |" * len(row))
        parts.append("\n".join(rows))
    if ex.get("post_text"):
        parts.append("\n".join(ex["post_text"]))
    parts.append(ex["qa"]["question"])
    return "\n\n".join(parts)


# ── Dataset configs ──

_answer_field = lambda ex: ex["answer"]
_finqa_field = lambda ex: str(ex["exe_ans"])

DATASET_CONFIGS = {
    "gsm8k": {
        "hf_path": "openai/gsm8k", "hf_name": "main",
        "extract_fn": _extract_gsm8k_answer, "question_key": "question",
        "splits": {"train": "train", "test": "test"},
    },
    "math": {
        "hf_path": "hendrycks/competition_math", "hf_name": None,
        "extract_fn": _extract_math_answer, "question_key": "problem",
        "splits": {"train": "train", "test": "test"},
    },
    "hotpotqa": {
        "hf_path": "hotpot_qa", "hf_name": "distractor",
        "extract_fn": _answer_field, "question_key": "question",
        "splits": {"train": "train", "validation": "dev"},
    },
    "nq": {
        "hf_path": "google-research-datasets/natural_questions", "hf_name": "default",
        "extract_fn": _extract_nq_answer, "question_key": "question",
        "splits": {"train": "train", "validation": "test"},
    },
    "musique": {
        "hf_path": "bdsaglam/musique", "hf_name": None,
        "extract_fn": _answer_field, "question_key": "question",
        "splits": {"train": "train", "validation": "dev"},
    },
    "2wiki": {
        "hf_path": "scholarly-shadows-syndicate/2WikiMultiHopQA", "hf_name": None,
        "extract_fn": _answer_field, "question_key": "question",
        "splits": {"train": "train", "validation": "dev"},
    },
    "finqa": {
        "github_urls": {
            "train": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json",
            "test": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json",
        },
        "extract_fn": lambda ex: str(ex["qa"]["exe_ans"]),
        "question_fn": _format_finqa_question,
        "splits": {"train": "train", "test": "test"},
    },
    "triviaqa": {
        "hf_path": "trivia_qa", "hf_name": "rc",
        "extract_fn": _extract_triviaqa_answer, "question_key": "question",
        "splits": {"validation": "test"},
    },
}

# Map split names to output directories
_SPLIT_DIRS = {"train": "processed", "dev": "eval_splits", "test": "eval_splits"}


def _get_field(ex, key):
    """Get a field, supporting dotted keys like 'qa.question'."""
    for part in key.split("."):
        ex = ex[part]
    return ex


def _load_github_json(url):
    """Download a JSON file from GitHub and return as list of dicts."""
    print(f"  Downloading from GitHub...")
    data = json.loads(urllib.request.urlopen(url).read())
    return data


def process_dataset(name: str, config: dict, data_root: Path,
                    max_samples: int = 0, splits_filter: list[str] | None = None,
                    verify_n: int = 10):
    print(f"\n{'='*60}\nProcessing: {name}\n{'='*60}")

    for hf_split, our_split in config["splits"].items():
        if splits_filter and our_split not in splits_filter:
            print(f"  {our_split}: skipped (not in --splits)")
            continue

        suffix = f"_{max_samples}" if max_samples else ""
        output_dir = _SPLIT_DIRS[our_split]
        output_path = data_root / output_dir / f"{name}_{our_split}{suffix}.jsonl"

        if output_path.exists():
            n_lines = sum(1 for _ in open(output_path))
            print(f"  {our_split}: already exists ({n_lines} examples), skipping")
            continue

        print(f"  Loading {hf_split} split...")
        try:
            if "github_urls" in config:
                raw_data = _load_github_json(config["github_urls"][hf_split])
                if max_samples and len(raw_data) > max_samples:
                    raw_data = raw_data[:max_samples]
            else:
                hf_kwargs = {"path": config["hf_path"]}
                if config.get("hf_name"):
                    hf_kwargs["name"] = config["hf_name"]
                ds = load_dataset(**hf_kwargs, split=hf_split, trust_remote_code=True)
                if max_samples and len(ds) > max_samples:
                    ds = ds.select(range(max_samples))
                raw_data = ds
        except Exception as e:
            print(f"  ERROR loading {name}/{hf_split}: {e}")
            continue

        print(f"  Converting {len(raw_data)} examples...")
        examples, errors = [], 0

        for ex in tqdm(raw_data, desc=f"  {name}/{our_split}"):
            try:
                if "question_fn" in config:
                    question = config["question_fn"](ex)
                else:
                    question = _get_field(ex, config["question_key"])
                examples.append({
                    "question": question,
                    "gold_answer": config["extract_fn"](ex),
                    "dataset": name,
                    "split": our_split,
                })
            except Exception:
                errors += 1

        with open(output_path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        print(f"  Wrote {len(examples)} examples to {output_path}")
        if errors:
            print(f"  ({errors} examples failed extraction)")

        if verify_n > 0:
            print(f"\n  Verification ({min(verify_n, len(examples))} examples):")
            for ex in examples[:verify_n]:
                print(f"    Q: {ex['question'][:80]}")
                print(f"    A: {str(ex['gold_answer'])[:60]}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download and format datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python -m data.download_and_format --datasets hotpotqa --splits dev --max-samples 50
  python -m data.download_and_format --datasets gsm8k hotpotqa finqa
  python -m data.download_and_format --datasets hotpotqa --max-samples 500""")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="datasets to download (default: all)")
    parser.add_argument("--splits", nargs="*", default=None,
                        help="only download these splits: train, dev, test (default: all)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="cap examples per split (0 = no cap)")
    parser.add_argument("--verify", type=int, default=10,
                        help="examples to print for inspection")
    args = parser.parse_args()

    if args.data_root:
        os.environ["DATA_ROOT"] = args.data_root

    data_root = get_data_root()
    print(f"Data root: {data_root}")
    if args.max_samples:
        print(f"Max samples per split: {args.max_samples}")
    if args.splits:
        print(f"Splits filter: {args.splits}")

    for name in (args.datasets or list(DATASET_CONFIGS.keys())):
        if name not in DATASET_CONFIGS:
            print(f"Unknown dataset: {name}")
            continue
        process_dataset(name, DATASET_CONFIGS[name], data_root,
                        max_samples=args.max_samples,
                        splits_filter=args.splits,
                        verify_n=args.verify)

    print(f"\n{'='*60}\nDone! Summary:")
    for subdir in ["processed", "eval_splits"]:
        path = data_root / subdir
        files = sorted(path.glob("*.jsonl"))
        if files:
            print(f"\n  {subdir}/")
            for f in files:
                print(f"    {f.name}: {sum(1 for _ in open(f))} examples")


if __name__ == "__main__":
    main()
