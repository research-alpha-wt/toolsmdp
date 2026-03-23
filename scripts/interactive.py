"""Interactive REPL for testing the SMDP pipeline with Qwen2.5-3B.

Ask questions, watch the model generate → execute → re-generate loop.
Type 'quit' to exit, 'search on/off' to toggle search.

Usage:
    python -m scripts.interactive

How it works (batch generation):
    1. Generate full response from model
    2. If <answer> found → done
    3. If code block found → execute, replace code with stdout, re-generate
    4. If neither → done (model finished without structured answer)

    <context> blocks stay in the text as-is (Phase-2 replacement is a
    training-time concern — needs token-by-token generation).
"""

import re
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.code_block_detector import detect_code_block
from core.replacement import replace_code_block
from sandbox.executor import execute_code, extract_search_query_strings

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MAX_TOOL_CALLS = 7
MAX_TOKENS_PER_GENERATION = 1024

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


def generate(model, tokenizer, context, system_prompt):
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
            max_new_tokens=MAX_TOKENS_PER_GENERATION,
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
    tool_calls = 0

    print(f"\n{'=' * 70}")
    print(f"QUESTION: {question}")
    print("=" * 70)

    while tool_calls < MAX_TOOL_CALLS:
        generated = generate(model, tokenizer, context, sys_prompt)

        print(f"\n--- Generation {tool_calls} ---")
        print(generated)
        print("---")

        # 1. Check for <answer> — if found, we're done
        answer_match = re.search(r"<answer>(.*?)</answer>", generated, re.DOTALL)
        if answer_match:
            print(f"\n>>> ANSWER: {answer_match.group(1).strip()}")
            break

        # 2. Check for code block — execute and re-generate
        code_det = detect_code_block(generated)
        if code_det is None:
            print("\n>>> No <answer> or code block found. Model finished.")
            break

        print(f"\n  [TOOL CALL {tool_calls + 1}]")
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
        print(f"  Output: {stdout[:500]}{'...' if len(stdout) > 500 else ''}")

        # Replace code block with stdout, rebuild context for next generation
        replaced = replace_code_block(generated, code_det, stdout)
        context = question + "\n\n" + replaced
        tool_calls += 1

    print("=" * 70)


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
