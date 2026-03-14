"""Download datasets from HuggingFace and convert to unified JSONL format.

Usage: python -m data.download_and_format [--data-root /path] [--datasets gsm8k hotpotqa]
"""

import argparse
import json
import os
import re
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
        "hf_path": "drt/musique", "hf_name": None,
        "extract_fn": _answer_field, "question_key": "question",
        "splits": {"train": "train", "validation": "dev"},
    },
    "2wiki": {
        "hf_path": "scholarly-shadows-syndicate/2WikiMultiHopQA", "hf_name": None,
        "extract_fn": _answer_field, "question_key": "question",
        "splits": {"train": "train", "validation": "dev"},
    },
    "finqa": {
        "hf_path": "ibm/finqa", "hf_name": None,
        "extract_fn": _finqa_field, "question_key": "question",
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


def process_dataset(name: str, config: dict, data_root: Path, verify_n: int = 10):
    print(f"\n{'='*60}\nProcessing: {name}\n{'='*60}")

    hf_kwargs = {"path": config["hf_path"]}
    if config.get("hf_name"):
        hf_kwargs["name"] = config["hf_name"]

    for hf_split, our_split in config["splits"].items():
        output_dir = _SPLIT_DIRS[our_split]
        output_path = data_root / output_dir / f"{name}_{our_split}.jsonl"

        if output_path.exists():
            n_lines = sum(1 for _ in open(output_path))
            print(f"  {our_split}: already exists ({n_lines} examples), skipping")
            continue

        print(f"  Loading {hf_split} split...")
        try:
            ds = load_dataset(**hf_kwargs, split=hf_split, trust_remote_code=True)
        except Exception as e:
            print(f"  ERROR loading {name}/{hf_split}: {e}")
            continue

        print(f"  Converting {len(ds)} examples...")
        examples, errors = [], 0

        for ex in tqdm(ds, desc=f"  {name}/{our_split}"):
            try:
                examples.append({
                    "question": ex[config["question_key"]],
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
    parser = argparse.ArgumentParser(description="Download and format datasets")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--verify", type=int, default=10)
    args = parser.parse_args()

    if args.data_root:
        os.environ["DATA_ROOT"] = args.data_root

    data_root = get_data_root()
    print(f"Data root: {data_root}")

    for name in (args.datasets or list(DATASET_CONFIGS.keys())):
        if name not in DATASET_CONFIGS:
            print(f"Unknown dataset: {name}")
            continue
        process_dataset(name, DATASET_CONFIGS[name], data_root, args.verify)

    print(f"\n{'='*60}\nDone! Summary:")
    for subdir in ["processed", "eval_splits"]:
        path = data_root / subdir
        files = sorted(path.glob("*.jsonl"))
        print(f"\n  {subdir}/")
        for f in files:
            print(f"    {f.name}: {sum(1 for _ in open(f))} examples")


if __name__ == "__main__":
    main()
