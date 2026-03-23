"""Test that Qwen3.5-4B generates code blocks and verify the full SMDP pipeline.

Three test modes:
  1. code_gen   — Does the model produce ```python blocks when prompted with tools?
  2. pipeline   — Full segment loop: generate → detect → execute → replace → continue
  3. eval       — Run on JSONL eval data, compute EM accuracy with difficulty stats

Usage:
    python -m scripts.test_base_model --mode code_gen
    python -m scripts.test_base_model --mode pipeline
    python -m scripts.test_base_model --mode eval --data data_local/eval_splits/gsm8k_test_50.jsonl
    python -m scripts.test_base_model --mode eval --data data_local/eval_splits/hotpotqa_dev_50.jsonl --num-rollouts 4
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.code_block_detector import detect_code_block
from core.context_block_detector import detect_context_block
from core.replacement import replace_code_block, replace_tool_output_with_context
from core.reward import compute_reward, extract_answer
from core.segment_rollout import Segment, Trajectory
from sandbox.executor import execute_code, extract_search_query_strings

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MAX_SEGMENTS = 15
MAX_TOKENS_PER_SEGMENT = 1024
MAX_ASSIMILATE_TOKENS = 256

SYSTEM_PROMPT = """\
You are a helpful assistant that solves problems step by step.
You have access to a Python interpreter. To use it, write code in a fenced block:

```python
# your code here
result = 2 + 2
print(result)
```

The code will be executed and you will see the output. You can use this to:
- Do arithmetic or complex calculations
- Search for information using search(query)

After seeing tool output, always write the key result in a <context>...</context> block before continuing your reasoning. For example:
<context>The population of France is approximately 67 million.</context>

When you have the final answer, state it clearly as: "The answer is <answer>."
"""

SEARCH_PROMPT_ADDITION = """
A search() function is available. Call it with a query string to retrieve relevant passages:
```python
results = search("your query here")
print(results)
```
"""


def load_model(device: str = "auto"):
    print(f"Loading {MODEL_ID}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    # Use float16 for T4 compatibility (T4 lacks native bfloat16 support).
    # bfloat16 silently falls back to float32 on T4 and OOMs.
    dtype = torch.float16 if not torch.cuda.is_bf16_supported() else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Loaded in {time.time() - t0:.1f}s | dtype={dtype} | device={model.device}")
    return model, tokenizer


def generate_segment(model, tokenizer, context: str, max_new_tokens: int = MAX_TOKENS_PER_SEGMENT) -> str:
    """Generate text until code block completion, EOS, or token limit."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def run_single_rollout(model, tokenizer, question: str, dataset: str,
                       search_enabled: bool = False, search_fn=None) -> Trajectory:
    """Run the full SMDP pipeline for one question.

    Three segment types: invoke (code block), assimilate (<context> block), synthesize (reasoning/EOS).
    Each tool call produces two segments: invoke → assimilate. If the model skips <context>,
    no assimilate segment is created and raw output persists.
    """
    trajectory = Trajectory()
    context = question

    for seg_idx in range(MAX_SEGMENTS):
        generated = generate_segment(model, tokenizer, context)
        code_detection = detect_code_block(generated)

        if code_detection is None:
            # No code block — check if this is an assimilate or synthesize segment
            ctx_detection = detect_context_block(generated)
            if ctx_detection is not None:
                # Assimilate segment — model wrote <context>...</context>
                segment = Segment(
                    start_context=context,
                    generated_text=generated,
                    segment_type="assimilate",
                    termination="context_block",
                )
                trajectory.segments.append(segment)

                # Phase-2 replacement: find the last tool output and replace with context content
                last_invoke = next(
                    (s for s in reversed(trajectory.segments) if s.segment_type == "invoke"),
                    None,
                )
                if last_invoke and last_invoke.tool_output:
                    replaced = replace_tool_output_with_context(
                        context + "\n" + generated,
                        last_invoke.tool_output,
                        ctx_detection.content,
                    )
                    context = replaced
                else:
                    context = context + "\n" + generated
                continue

            # Synthesize segment — final reasoning / answer
            segment = Segment(
                start_context=context,
                generated_text=generated,
                segment_type="synthesize",
                termination="eos",
            )
            trajectory.segments.append(segment)
            trajectory.full_context = context + "\n" + generated
            break

        # Code block found — this is an invoke segment
        # Pre-resolve search calls if we have a search backend
        search_results = None
        if search_fn and search_enabled:
            queries = extract_search_query_strings(code_detection.executable)
            if queries:
                search_results = {q: search_fn(q) for q in queries}

        stdout = execute_code(code_detection.executable, search_enabled=search_enabled,
                              search_results=search_results)
        replaced = replace_code_block(generated, code_detection, stdout)

        segment = Segment(
            start_context=context,
            generated_text=generated,
            segment_type="invoke",
            termination="tool_call",
            tool_code=code_detection.executable,
            tool_comments=code_detection.comments,
            tool_output=stdout,
        )
        trajectory.segments.append(segment)

        # Build next context: original question + replaced output so far
        # This is the transient state s̃ with raw tool output
        context = question + "\n\n" + replaced

        if seg_idx == MAX_SEGMENTS - 1:
            segment.termination = "truncated"
            trajectory.full_context = context
    else:
        trajectory.full_context = context

    return trajectory


# ── Mode: code_gen ──

SMOKE_QUESTIONS = [
    ("What is 347 * 283?", "gsm8k"),
    ("If a train travels 120 miles in 2.5 hours, what is its average speed in mph?", "gsm8k"),
    ("What is the sum of the first 20 prime numbers?", "gsm8k"),
]


def mode_code_gen(model, tokenizer):
    """Check if the model generates code blocks when prompted."""
    print("\n" + "=" * 60)
    print("MODE: code_gen — Testing code block generation")
    print("=" * 60)

    code_block_count = 0
    for question, dataset in SMOKE_QUESTIONS:
        print(f"\nQ: {question}")
        generated = generate_segment(model, tokenizer, question)
        detection = detect_code_block(generated)

        print(f"Generated ({len(generated)} chars):")
        print("-" * 40)
        print(generated[:500])
        if len(generated) > 500:
            print("... (truncated)")
        print("-" * 40)

        if detection:
            code_block_count += 1
            print(f"Code block found: {detection.executable[:100]}")
        else:
            print("No code block detected")

    print(f"\n{'=' * 60}")
    print(f"Result: {code_block_count}/{len(SMOKE_QUESTIONS)} generated code blocks")
    print("=" * 60)
    return code_block_count > 0


# ── Mode: pipeline ──

def mode_pipeline(model, tokenizer):
    """Full segment rollout pipeline on smoke questions."""
    print("\n" + "=" * 60)
    print("MODE: pipeline — Full SMDP segment rollout")
    print("=" * 60)

    for question, dataset in SMOKE_QUESTIONS:
        print(f"\nQ: {question}")
        traj = run_single_rollout(model, tokenizer, question, dataset)

        print(f"  Segments: {traj.num_segments}")
        print(f"  Tool calls: {traj.total_tool_calls}")
        print(f"  Assimilations: {traj.total_assimilations}")
        for i, seg in enumerate(traj.segments):
            print(f"  Segment {i}: {seg.segment_type}/{seg.termination}", end="")
            if seg.tool_code:
                print(f" | code: {seg.tool_code[:60]}...", end="")
            if seg.tool_output:
                print(f" | output: {seg.tool_output[:60]}", end="")
            print()

        # Show final context (no scoring in pipeline mode — no gold answers)
        print(f"  Final context (last 200 chars): ...{traj.full_context[-200:]}")


# ── Mode: eval ──

def mode_eval(model, tokenizer, data_path: str, num_rollouts: int = 1):
    """Run eval on a JSONL file, compute EM accuracy and tool-use stats."""
    print(f"\n{'=' * 60}")
    print(f"MODE: eval — {data_path} ({num_rollouts} rollout(s) per question)")
    print("=" * 60)

    examples = []
    with open(data_path) as f:
        for line in f:
            examples.append(json.loads(line))

    dataset = examples[0]["dataset"]
    split = examples[0]["split"]
    print(f"Dataset: {dataset}, Split: {split}, N: {len(examples)}")

    search_enabled = dataset in ("hotpotqa", "nq", "musique", "2wiki", "triviaqa")

    search_fn = None
    if search_enabled:
        try:
            from retrieval.search import get_search
            search_fn = get_search()
        except (ImportError, RuntimeError) as e:
            print(f"No search backend available ({e}). Using placeholder.")

    results = []
    for i, ex in enumerate(examples):
        question = ex["question"]
        gold = ex["gold_answer"]

        print(f"\n[{i+1}/{len(examples)}] Q: {question[:80]}...")

        best_reward = 0.0
        best_traj = None
        rollout_stats = []

        for r in range(num_rollouts):
            traj = run_single_rollout(model, tokenizer, question, dataset,
                                       search_enabled=search_enabled,
                                       search_fn=search_fn)
            reward = compute_reward(traj.full_context, gold, dataset)
            pred = extract_answer(traj.full_context, dataset)

            rollout_stats.append({
                "num_segments": traj.num_segments,
                "tool_calls": traj.total_tool_calls,
                "reward": reward,
                "prediction": pred,
            })

            if reward > best_reward:
                best_reward = reward
                best_traj = traj

        # Use the first rollout for classification
        stats = rollout_stats[0]
        avg_tool_calls = sum(r["tool_calls"] for r in rollout_stats) / num_rollouts

        # Difficulty bucket
        if avg_tool_calls < 1.5:
            bucket = "1_call"
        elif avg_tool_calls < 2.5:
            bucket = "2_calls"
        else:
            bucket = "3+_calls"

        result = {
            "question": question,
            "gold_answer": gold,
            "prediction": stats["prediction"],
            "reward": best_reward,
            "pass_at_k": best_reward,  # 1.0 if any rollout correct
            "avg_tool_calls": avg_tool_calls,
            "avg_segments": sum(r["num_segments"] for r in rollout_stats) / num_rollouts,
            "bucket": bucket,
            "rollouts": rollout_stats,
        }
        results.append(result)

        print(f"  Gold: {str(gold)[:60]}")
        print(f"  Pred: {stats['prediction']}")
        print(f"  Reward: {best_reward} | Tool calls: {avg_tool_calls:.1f} | Bucket: {bucket}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)

    total = len(results)
    correct = sum(1 for r in results if r["reward"] > 0)
    print(f"Overall EM: {correct}/{total} = {correct/total:.1%}")

    if num_rollouts > 1:
        pass_at_k = sum(1 for r in results if r["pass_at_k"] > 0)
        print(f"Pass@{num_rollouts}: {pass_at_k}/{total} = {pass_at_k/total:.1%}")

    # Per-bucket stats
    from collections import Counter
    buckets = Counter(r["bucket"] for r in results)
    print(f"\nDifficulty distribution:")
    for bucket in ["1_call", "2_calls", "3+_calls"]:
        n = buckets.get(bucket, 0)
        if n == 0:
            continue
        bucket_correct = sum(1 for r in results if r["bucket"] == bucket and r["reward"] > 0)
        print(f"  {bucket}: {n} questions, EM = {bucket_correct}/{n} = {bucket_correct/n:.1%}")

    avg_tools = sum(r["avg_tool_calls"] for r in results) / total
    avg_segs = sum(r["avg_segments"] for r in results) / total
    print(f"\nAvg tool calls: {avg_tools:.2f}")
    print(f"Avg segments: {avg_segs:.2f}")

    # Save results
    out_path = Path(data_path).parent / f"{Path(data_path).stem}_results.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nResults saved to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Test base model with SMDP pipeline")
    parser.add_argument("--mode", choices=["code_gen", "pipeline", "eval"],
                        default="code_gen")
    parser.add_argument("--data", type=str, help="JSONL file for eval mode")
    parser.add_argument("--num-rollouts", type=int, default=1,
                        help="Rollouts per question in eval mode (for pass@k)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device map for model loading")
    args = parser.parse_args()

    if args.mode == "eval" and not args.data:
        parser.error("--data required for eval mode")

    model, tokenizer = load_model(args.device)

    if args.mode == "code_gen":
        mode_code_gen(model, tokenizer)
    elif args.mode == "pipeline":
        mode_pipeline(model, tokenizer)
    elif args.mode == "eval":
        mode_eval(model, tokenizer, args.data, args.num_rollouts)


if __name__ == "__main__":
    main()
