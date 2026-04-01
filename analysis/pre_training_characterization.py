"""Pre-training characterization: run base model with tools on eval sets.

Uses vLLM with wavefront batching: all questions × all rollouts are processed
together. Each generation step batches every active rollout into one vLLM call,
maximizing GPU utilization.

Usage:
    python -m analysis.pre_training_characterization \
        --input-dir data_local/eval_splits \
        --output-dir data_local/analysis \
        --num-rollouts 4

    # Quick test
    python -m analysis.pre_training_characterization \
        --input-dir data_local/eval_splits \
        --output-dir data_local/analysis \
        --max-samples 2 --num-rollouts 1
"""

import argparse
import json
import logging
import time
from pathlib import Path

from vllm import LLM, SamplingParams

from core.code_block_detector import detect_code_block
from core.replacement import replace_code_block
from core.reward import compute_reward, extract_answer, normalize_answer
from sandbox.executor import execute_code, extract_search_query_strings

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MAX_TOOL_CALLS = 7
MAX_TOKENS_PER_SEGMENT = 1024

SEARCH_DATASETS = {"hotpotqa", "nq", "musique", "2wiki", "triviaqa"}

SYSTEM_PROMPT = """\
You are a helpful assistant that solves problems step by step.

## Tools
You have access to a Python interpreter for computation and information retrieval. Write code in a fenced block:

```python
your code here
```

The code will be executed and you will see its printed output. Use code blocks ONLY when you need to:
- Compute something you cannot do reliably in your head (arithmetic, logic, data processing)
- Search for information you do not already know using search(query)

Do NOT write code for things you already know the answer to.

## <context> block
After a code block executes and you see the tool output, write a <context> block to extract the key information relevant to answering the question. Only use <context> blocks immediately after tool output. For example:

Tool output: "Paris is the capital of France. It has a population of 2.1 million in the city proper and 12.4 million in the metro area..."
<context>Paris population: 2.1 million (city), 12.4 million (metro)</context>

## <answer> block
When you have the final answer, write ONLY the answer inside an answer block with no extra words:
<answer>42</answer>
<answer>William Shakespeare</answer>
<answer>Paris</answer>
"""

SEARCH_PROMPT_ADDITION = """
A search() function is available. Call it with a query string to retrieve relevant passages:
```python
results = search("your query here")
print(results)
```
"""

log = logging.getLogger(__name__)


def setup_logging(output_dir: Path):
    log_file = output_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )
    log.info("Logging to %s", log_file)


def load_model():
    log.info("Loading %s with vLLM...", MODEL_ID)
    t0 = time.time()
    llm = LLM(model=MODEL_ID, trust_remote_code=True)
    tokenizer = llm.get_tokenizer()
    log.info("Loaded in %.1fs", time.time() - t0)
    return llm, tokenizer


def load_search_fn():
    try:
        from retrieval.search import get_search
        fn = get_search()
        log.info("Search backend loaded (Pyserini)")
        return fn
    except (ImportError, RuntimeError) as e:
        log.warning("No search backend: %s", e)
        return None


def build_prompt(tokenizer, context: str, search_enabled: bool) -> str:
    system = SYSTEM_PROMPT + (SEARCH_PROMPT_ADDITION if search_enabled else "")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": context},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def classify_tool_type(code: str) -> str:
    has_search = "search(" in code
    lines = [l.strip() for l in code.strip().splitlines()
             if l.strip() and not l.strip().startswith("#")]
    non_search_lines = [l for l in lines
                        if "search(" not in l and l not in ("", "print(results)")]
    has_calc = len(non_search_lines) > 0
    if has_search and has_calc:
        return "both"
    return "search" if has_search else "calc"


def compute_tool_output_relevance(tool_output: str, gold_answer) -> float:
    if not tool_output or tool_output.startswith("ERROR:"):
        return 0.0
    gold_str = gold_answer if isinstance(gold_answer, str) else " ".join(gold_answer)
    gold_tokens = set(normalize_answer(gold_str).split())
    if not gold_tokens:
        return 0.0
    output_tokens = set(normalize_answer(tool_output).split())
    return len(gold_tokens & output_tokens) / len(gold_tokens)


# ── Wavefront rollout engine ──

def process_dataset(llm, tokenizer, data_path: Path, output_dir: Path,
                    num_rollouts: int, search_fn, max_samples: int = 0):
    """Process all questions × rollouts using wavefront batching.

    All active rollouts across all questions are batched into each vLLM
    generate call. Rollouts finish independently as they produce final
    answers or exhaust tool calls.
    """
    examples = [json.loads(line) for line in open(data_path)]
    if max_samples:
        examples = examples[:max_samples]

    dataset = examples[0]["dataset"]
    search_enabled = dataset in SEARCH_DATASETS
    log.info("Dataset: %s | N: %d | Rollouts: %d | Wavefront batch size: %d",
             dataset, len(examples), num_rollouts, len(examples) * num_rollouts)

    params = SamplingParams(
        temperature=0.7, top_p=0.9, max_tokens=MAX_TOKENS_PER_SEGMENT,
    )

    # Initialize state for every (question, rollout) pair
    # Each state is identified by (example_idx, rollout_idx)
    states = {}
    for ex_idx, ex in enumerate(examples):
        for r_idx in range(num_rollouts):
            key = (ex_idx, r_idx)
            states[key] = {
                "question": ex["question"],
                "context": ex["question"],
                "segments": [],
                "tool_types": [],
                "tool_outputs": [],
                "full_generated": "",
                "done": False,
                "tool_call_count": 0,
            }

    t0 = time.time()

    for wave in range(MAX_TOOL_CALLS):
        active_keys = [k for k, s in states.items() if not s["done"]]
        if not active_keys:
            break

        log.info("Wave %d: %d active rollouts", wave, len(active_keys))

        # Build prompts for all active rollouts
        prompts = [build_prompt(tokenizer, states[k]["context"], search_enabled)
                   for k in active_keys]

        # Single batched vLLM call for ALL active rollouts
        outputs = llm.generate(prompts, params)

        # Process each output
        for batch_idx, key in enumerate(active_keys):
            state = states[key]
            generated = outputs[batch_idx].outputs[0].text
            state["full_generated"] += generated

            code_detection = detect_code_block(generated)

            if code_detection is None:
                # No code block — rollout finished
                state["segments"].append({"type": "synthesize", "termination": "eos"})
                state["context"] = state["context"] + "\n" + generated
                state["done"] = True
                continue

            # Execute code
            search_results = None
            if search_fn and search_enabled:
                queries = extract_search_query_strings(code_detection.executable)
                if queries:
                    search_results = {q: search_fn(q) for q in queries}

            stdout = execute_code(
                code_detection.executable,
                search_enabled=search_enabled,
                search_results=search_results,
            )
            replaced = replace_code_block(generated, code_detection, stdout)

            tt = classify_tool_type(code_detection.executable)
            state["tool_types"].append(tt)
            state["tool_outputs"].append(stdout)
            state["segments"].append({
                "type": "invoke", "termination": "tool_call",
                "tool_type": tt, "code": code_detection.executable, "output": stdout,
            })

            state["context"] = state["question"] + "\n\n" + replaced
            state["full_generated"] += f"\n[TOOL OUTPUT]\n{stdout}\n"
            state["tool_call_count"] += 1

            if state["tool_call_count"] >= MAX_TOOL_CALLS:
                state["segments"][-1]["termination"] = "truncated"
                state["done"] = True

    # Mark any still-active as done
    for s in states.values():
        s["done"] = True

    # Aggregate results per question
    results = []
    for ex_idx, ex in enumerate(examples):
        rollouts = []
        for r_idx in range(num_rollouts):
            s = states[(ex_idx, r_idx)]
            reward = compute_reward(s["context"], ex["gold_answer"], dataset)
            pred = extract_answer(s["context"], dataset)
            relevances = [compute_tool_output_relevance(out, ex["gold_answer"])
                          for out in s["tool_outputs"]]

            rollouts.append({
                "rollout_idx": r_idx,
                "num_segments": len(s["segments"]),
                "num_tool_calls": sum(1 for seg in s["segments"] if seg["type"] == "invoke"),
                "tool_types": s["tool_types"],
                "tool_output_relevance": relevances,
                "reward": reward,
                "prediction": pred,
                "full_generated": s["full_generated"],
                "final_context": s["context"],
            })

        avg_tool_calls = sum(r["num_tool_calls"] for r in rollouts) / num_rollouts
        any_correct = any(r["reward"] > 0 for r in rollouts)

        if avg_tool_calls < 1.5:
            bucket = "1_call"
        elif avg_tool_calls < 2.5:
            bucket = "2_calls"
        else:
            bucket = "3+_calls"

        result = {
            "question_idx": ex_idx,
            "question": ex["question"],
            "gold_answer": ex["gold_answer"],
            "dataset": dataset,
            "avg_tool_calls": avg_tool_calls,
            "avg_segments": sum(r["num_segments"] for r in rollouts) / num_rollouts,
            "bucket": bucket,
            "any_correct": any_correct,
            "rollouts": rollouts,
        }
        results.append(result)

    # Write results
    out_path = output_dir / f"{dataset}_rollout_stats.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    total = len(results)
    correct = sum(1 for r in results if r["any_correct"])
    elapsed = time.time() - t0
    log.info("=== %s Summary ===", dataset)
    log.info("EM (any_correct): %d/%d = %.1f%%", correct, total, 100 * correct / total)

    from collections import Counter
    buckets = Counter(r["bucket"] for r in results)
    for b in ["1_call", "2_calls", "3+_calls"]:
        n = buckets.get(b, 0)
        if n == 0:
            continue
        bc = sum(1 for r in results if r["bucket"] == b and r["any_correct"])
        log.info("  %s: %d questions, EM = %d/%d = %.1f%%", b, n, bc, n, 100 * bc / n)

    log.info("Avg tool calls: %.2f", sum(r["avg_tool_calls"] for r in results) / total)
    log.info("Total time: %.1fs (%.1fs per example)", elapsed, elapsed / total)
    log.info("Saved %d results to %s", total, out_path)

    return results


def main():
    parser = argparse.ArgumentParser(description="Pre-training characterization (vLLM wavefront)")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    log.info("Args: %s", vars(args))

    input_dir = Path(args.input_dir)
    eval_files = sorted(input_dir.glob("*_*.jsonl"))
    eval_files = [f for f in eval_files if "_results" not in f.stem and "_rollout_stats" not in f.stem]
    if args.datasets:
        eval_files = [f for f in eval_files if any(d in f.stem for d in args.datasets)]
    if not eval_files:
        log.error("No eval files found in %s", input_dir)
        return
    log.info("Found %d eval files: %s", len(eval_files), [f.name for f in eval_files])

    llm, tokenizer = load_model()
    search_fn = load_search_fn()

    for data_path in eval_files:
        log.info("=" * 60)
        log.info("Processing: %s", data_path.name)
        log.info("=" * 60)
        process_dataset(llm, tokenizer, data_path, output_dir,
                        num_rollouts=args.num_rollouts, search_fn=search_fn,
                        max_samples=args.max_samples)

    log.info("All done.")


if __name__ == "__main__":
    main()
