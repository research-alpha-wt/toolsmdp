"""Download datasets from HuggingFace and convert to unified JSONL format.

Unified schema per line:
{
    "question": str,
    "gold_answer": str | list[str],
    "dataset": str,
    "split": "train" | "dev" | "test",
    "metadata": { ... dataset-specific fields ... }
}

Usage:
    python -m data.download_and_format [--data-root /path/to/data]
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


def _extract_gsm8k_answer(example: dict) -> str:
    """GSM8K: answer field has chain-of-thought + '#### <number>'."""
    answer = example["answer"]
    match = re.search(r"####\s*(.+)", answer)
    if match:
        return match.group(1).strip()
    return answer.strip()


def _extract_math_answer(example: dict) -> str:
    r"""MATH: solution field contains \boxed{<answer>}."""
    solution = example["solution"]
    matches = list(re.finditer(r"\\boxed\{", solution))
    if matches:
        start = matches[-1].end()
        depth = 1
        i = start
        while i < len(solution) and depth > 0:
            if solution[i] == "{":
                depth += 1
            elif solution[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            return solution[start:i - 1].strip()
    return solution.strip()


def _extract_hotpotqa_answer(example: dict) -> str:
    return example["answer"]


def _extract_nq_answer(example: dict) -> list[str]:
    """NQ: answer field is a list of acceptable answers."""
    answers = example.get("answer", [])
    if isinstance(answers, list):
        return answers if answers else [""]
    return [str(answers)]


def _extract_musique_answer(example: dict) -> str:
    return example["answer"]


def _extract_2wiki_answer(example: dict) -> str:
    return example["answer"]


def _extract_finqa_answer(example: dict) -> str:
    """FinQA: exe_ans is the numeric answer."""
    return str(example["exe_ans"])


def _extract_triviaqa_answer(example: dict) -> list[str]:
    """TriviaQA: answer.aliases is a list of acceptable answers."""
    answer_obj = example.get("answer", {})
    if isinstance(answer_obj, dict):
        aliases = answer_obj.get("aliases", [])
        value = answer_obj.get("value", "")
        all_answers = list(set(aliases + [value])) if value else aliases
        return all_answers if all_answers else [""]
    return [str(answer_obj)]


# ── Dataset configs ──


DATASET_CONFIGS = {
    "gsm8k": {
        "hf_path": "openai/gsm8k",
        "hf_name": "main",
        "extract_fn": _extract_gsm8k_answer,
        "question_key": "question",
        "splits": {"train": "train", "test": "test"},
        "output_splits": {"train": "processed", "test": "eval_splits"},
    },
    "math": {
        "hf_path": "hendrycks/competition_math",
        "hf_name": None,
        "extract_fn": _extract_math_answer,
        "question_key": "problem",
        "splits": {"train": "train", "test": "test"},
        "output_splits": {"train": "processed", "test": "eval_splits"},
    },
    "hotpotqa": {
        "hf_path": "hotpot_qa",
        "hf_name": "distractor",
        "extract_fn": _extract_hotpotqa_answer,
        "question_key": "question",
        "splits": {"train": "train", "validation": "dev"},
        "output_splits": {"train": "processed", "validation": "eval_splits"},
    },
    "nq": {
        "hf_path": "google-research-datasets/natural_questions",
        "hf_name": "default",
        "extract_fn": _extract_nq_answer,
        "question_key": "question",
        "splits": {"train": "train", "validation": "test"},
        "output_splits": {"train": "processed", "validation": "eval_splits"},
        "custom_loader": True,
    },
    "musique": {
        "hf_path": "drt/musique",
        "hf_name": None,
        "extract_fn": _extract_musique_answer,
        "question_key": "question",
        "splits": {"train": "train", "validation": "dev"},
        "output_splits": {"train": "processed", "validation": "eval_splits"},
    },
    "2wiki": {
        "hf_path": "scholarly-shadows-syndicate/2WikiMultiHopQA",
        "hf_name": None,
        "extract_fn": _extract_2wiki_answer,
        "question_key": "question",
        "splits": {"train": "train", "validation": "dev"},
        "output_splits": {"train": "processed", "validation": "eval_splits"},
    },
    "finqa": {
        "hf_path": "ibm/finqa",
        "hf_name": None,
        "extract_fn": _extract_finqa_answer,
        "question_key": "question",
        "splits": {"train": "train", "test": "test"},
        "output_splits": {"train": "processed", "test": "eval_splits"},
    },
    "triviaqa": {
        "hf_path": "trivia_qa",
        "hf_name": "rc",
        "extract_fn": _extract_triviaqa_answer,
        "question_key": "question",
        "splits": {"validation": "test"},
        "output_splits": {"validation": "eval_splits"},
    },
}


def process_dataset(name: str, config: dict, data_root: Path, verify_n: int = 10):
    """Download and convert one dataset."""
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"{'='*60}")

    hf_kwargs = {"path": config["hf_path"]}
    if config.get("hf_name"):
        hf_kwargs["name"] = config["hf_name"]

    for hf_split, our_split in config["splits"].items():
        output_dir = config["output_splits"][hf_split]
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
        examples = []
        errors = 0

        for ex in tqdm(ds, desc=f"  {name}/{our_split}"):
            try:
                question = ex[config["question_key"]]
                gold = config["extract_fn"](ex)

                examples.append({
                    "question": question,
                    "gold_answer": gold,
                    "dataset": name,
                    "split": our_split,
                    "metadata": {},
                })
            except Exception:
                errors += 1
                continue

        with open(output_path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        print(f"  Wrote {len(examples)} examples to {output_path}")
        if errors:
            print(f"  ({errors} examples failed extraction)")

        # Verify a few examples
        if verify_n > 0:
            print(f"\n  Verification ({min(verify_n, len(examples))} examples):")
            for ex in examples[:verify_n]:
                q = ex["question"][:80]
                a = str(ex["gold_answer"])[:60]
                print(f"    Q: {q}")
                print(f"    A: {a}")
                print()


def main():
    parser = argparse.ArgumentParser(description="Download and format datasets")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Specific datasets to process (default: all)")
    parser.add_argument("--verify", type=int, default=10,
                        help="Number of examples to print for verification")
    args = parser.parse_args()

    if args.data_root:
        os.environ["DATA_ROOT"] = args.data_root

    data_root = get_data_root()
    print(f"Data root: {data_root}")

    datasets_to_process = args.datasets or list(DATASET_CONFIGS.keys())

    for name in datasets_to_process:
        if name not in DATASET_CONFIGS:
            print(f"Unknown dataset: {name}")
            continue
        process_dataset(name, DATASET_CONFIGS[name], data_root, args.verify)

    print("\n" + "=" * 60)
    print("Done! Summary:")
    for subdir in ["processed", "eval_splits"]:
        path = data_root / subdir
        files = sorted(path.glob("*.jsonl"))
        print(f"\n  {subdir}/")
        for f in files:
            n = sum(1 for _ in open(f))
            print(f"    {f.name}: {n} examples")


if __name__ == "__main__":
    main()
