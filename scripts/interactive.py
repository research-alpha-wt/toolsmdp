"""Interactive REPL for testing the SMDP pipeline with Qwen2.5-3B.

Ask questions, watch the invoke→assimilate→synthesize loop in real time.
Type 'quit' to exit, 'search on/off' to toggle search.

Usage:
    python -m scripts.interactive
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.code_block_detector import detect_code_block
from core.context_block_detector import detect_context_block
from core.replacement import replace_code_block, replace_tool_output_with_context
from core.segment_rollout import Segment, Trajectory
from sandbox.executor import execute_code, extract_search_query_strings

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MAX_SEGMENTS = 15
MAX_TOKENS_PER_SEGMENT = 1024

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

When you have the final answer, write it in an answer block:
<answer>your final answer here</answer>
"""

SEARCH_PROMPT_ADDITION = """
A search() function is available. Call it with a query string to retrieve relevant passages:
```python
results = search("your query here")
print(results)
```
"""


def load_model():
    print(f"Loading {MODEL_ID}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    dtype = torch.float16 if not torch.cuda.is_bf16_supported() else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Loaded in {time.time() - t0:.1f}s | dtype={dtype} | device={model.device}")
    return model, tokenizer


def generate(model, tokenizer, context, system_prompt, max_new_tokens=MAX_TOKENS_PER_SEGMENT):
    messages = [
        {"role": "system", "content": system_prompt},
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


def run_question(model, tokenizer, question, search_enabled=False, search_fn=None):
    sys_prompt = SYSTEM_PROMPT + (SEARCH_PROMPT_ADDITION if search_enabled else "")
    context = question

    print(f"\n{'=' * 70}")
    print(f"QUESTION: {question}")
    print("=" * 70)

    for seg_idx in range(MAX_SEGMENTS):
        print(f"\n--- Segment {seg_idx} ---")
        generated = generate(model, tokenizer, context, sys_prompt)

        code_det = detect_code_block(generated)
        ctx_det = detect_context_block(generated)

        if code_det is not None:
            # INVOKE segment
            print(f"[INVOKE] Code block detected:")
            print(f"  Code: {code_det.executable}")

            search_results = None
            if search_fn and search_enabled:
                queries = extract_search_query_strings(code_det.executable)
                if queries:
                    search_results = {q: search_fn(q) for q in queries}
                    for q in queries:
                        print(f"  Search: \"{q}\"")

            stdout = execute_code(code_det.executable, search_enabled=search_enabled,
                                  search_results=search_results)
            print(f"  Output: {stdout[:200]}{'...' if len(stdout) > 200 else ''}")

            replaced = replace_code_block(generated, code_det, stdout)
            context = question + "\n\n" + replaced

        elif ctx_det is not None:
            # ASSIMILATE segment
            print(f"[ASSIMILATE] Context block detected:")
            print(f"  Content: {ctx_det.content}")
            context = context + "\n" + generated

        else:
            # SYNTHESIZE segment
            print(f"[SYNTHESIZE] Final reasoning:")
            print(f"  {generated[:500]}{'...' if len(generated) > 500 else ''}")

            # Check for <answer> tag
            import re
            answer_match = re.search(r"<answer>(.*?)</answer>", generated, re.DOTALL)
            if answer_match:
                print(f"\n  >>> ANSWER: {answer_match.group(1).strip()}")
            break

    print("\n" + "=" * 70)


def main():
    model, tokenizer = load_model()

    search_enabled = False
    search_fn = None

    print("\nInteractive SMDP Pipeline")
    print("Commands: 'quit', 'search on', 'search off'")
    print("-" * 40)

    while True:
        try:
            question = input("\nQ: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() == "quit":
            break
        if question.lower() == "search on":
            if search_fn is None:
                try:
                    from retrieval.search import get_search
                    search_fn = get_search()
                    print("Search backend loaded.")
                except (ImportError, RuntimeError) as e:
                    print(f"Failed to load search: {e}")
                    continue
            search_enabled = True
            print("Search: ON")
            continue
        if question.lower() == "search off":
            search_enabled = False
            print("Search: OFF")
            continue

        run_question(model, tokenizer, question, search_enabled, search_fn)


if __name__ == "__main__":
    main()
