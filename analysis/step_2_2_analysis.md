# ToolSMDP — Step 2.2 Detailed Analysis

## Overview

Pre-training characterization of Qwen2.5-3B-Instruct with tool access (code execution + Wikipedia search).
500 samples per dataset, 4 rollouts each, pass@4 metric.

| Dataset | Type | EM (pass@4) | Avg Tool Calls |
|---|---|---|---|
| GSM8K | Math (computation) | **84.2%** | 2.3 |
| 2Wiki | Multi-hop QA (search) | **31.0%** | 3.8 |
| HotpotQA | Multi-hop QA (search) | **21.4%** | 4.4 |
| Musique | Multi-hop QA (search) | **3.8%** | 3.9 |
| FinQA | Financial QA (tables + computation) | **0.6%** | 2.2 |

---

## Why GSM8K Works Well (84.2%)

GSM8K is a **math word problem** dataset. The model excels here because:

1. **Clean tool-task fit.** The model writes Python to compute arithmetic it can't do reliably in its head. This is exactly what code tools are for.
2. **Simple code patterns.** Most code blocks are straightforward: define variables, compute, print. No imports, no external data needed.
3. **Self-contained questions.** All information needed is in the question — no search required.
4. **Consistent across buckets.** 83-86% EM regardless of whether the question needs 1, 2, or 3+ tool calls. The model handles multi-step computation well.

**Failure mode (16% wrong):** Mostly arithmetic/logic errors in the code itself, or the model not printing the result (code runs but produces no output).

---

## Why Search Datasets Struggle

### The Root Cause: `import sys` → ImportError Loop

**This is the single biggest failure mode across all search datasets.** The model frequently generates code that starts with `import sys` or other blocked imports, triggering an ImportError. The sandbox blocks these imports for security. However:

1. The model sees `ERROR: ImportError: Import 'sys' is not allowed`
2. It tries again with a slightly different import
3. Same error → loops until hitting max tool calls (7)
4. All 7 tool outputs are errors → model guesses an answer without any real search results

**Impact by dataset:**

| Dataset | Incorrect examples with ImportError | Explanation |
|---|---|---|
| HotpotQA | 278/357 (78%) | Model tries `import sys` before `search()` |
| Musique | 308/431 (72%) | Same pattern |
| 2Wiki | Similar | Same pattern |

**Why this happens:** The model was trained on general Python code where `import sys` is common. Our system prompt says `search()` is available, but the model's instinct is to write "proper" Python with imports first. It hasn't learned (yet) that `search()` is a bare global function — this is exactly what RL training should fix.

### Tool Output Relevance: Near Zero

Even when search does execute successfully, the tool outputs have ~0% keyword overlap with gold answers:

| Dataset | Rollouts with 0.0 relevance across ALL tool calls |
|---|---|
| HotpotQA | 432/432 (100%) |
| 2Wiki | 422/424 (99.5%) |

This means the search results rarely contain the exact answer keywords. Reasons:
- BM25 retrieves passages by keyword match with the *query*, not the *answer*
- Multi-hop questions require chaining: search for entity A → find fact → search for entity B → combine
- The model's search queries may not retrieve relevant passages

### Per-Dataset Analysis

**HotpotQA (21.4%)**
- 90% of questions land in the 3+ calls bucket — the model keeps searching
- Even correct answers often come from model knowledge, not search results
- The 21% that succeed likely have simple factual answers the model already knows

**2WikiMultiHopQA (31.0%)**
- Better than HotpotQA because 2Wiki questions tend to have clearer entity names
- 56% EM on 2-call questions vs 26% on 3+ calls — simpler questions are easier
- Still dominated by import error loops

**Musique (3.8%)**
- Hardest multi-hop dataset — questions require 2-4 reasoning steps
- 37% of incorrect examples hit max tools (7) — stuck in error loops
- Even the few correct answers are likely lucky guesses

---

## Why FinQA Fails Almost Completely (0.6%)

FinQA is a **financial question answering** dataset where questions reference tables in financial reports. The model fails because:

1. **No table context.** The questions reference tables ("as of december 31, 2010, what was the ratio of...") but the model doesn't see the table — only the question text is provided. Without the table data, it's impossible to compute the answer.
2. **Domain-specific reasoning.** Financial calculations require understanding specific accounting concepts.
3. **Exact numeric match.** Gold answers are precise numbers like `-0.02918` or `299999990.4`. Even small rounding differences cause EM failure.

**FinQA may not be viable for this project** without adding table context to the prompts.

---

## Failure Mode Summary

| Failure Mode | GSM8K | HotpotQA | 2Wiki | Musique | FinQA |
|---|---|---|---|---|---|
| **Import error loop** | Rare | **78%** | High | **72%** | Low |
| **Wrong computation** | **85%** | — | — | — | 79% |
| **No tool use at all** | 14% | 5% | 8% | 12% | **20%** |
| **Hit max tools (7)** | 6% | **44%** | **34%** | **37%** | 4% |
| **No prediction** | 2% | 0% | 1% | 0% | 1% |

---

## Implications for RL Training (Milestones 4-5)

### What RL should fix:
1. **Import error loop** — **FIXED (sandbox update).** Allowed `sys`, `os`, `io`, `pathlib` imports. Only dangerous modules (`subprocess`, `socket`, `requests`, etc.) remain blocked. This alone should significantly improve search datasets.
2. **Search query quality** — RL reward for correct final answers will pressure the model to write better search queries.
3. **Assimilation** — The `<context>` block training will teach the model to extract relevant facts from search results.

### Expected training gains (predictions for Step 2.5):
- **GSM8K:** Small gain (already 84%). Maybe 88-92% as the model learns to avoid unnecessary tool calls.
- **HotpotQA/2Wiki:** Large gain potential. Sandbox fix alone should push from 21-31% to 35-45%. RL + better search + assimilation could reach 50%+.
- **Musique:** Moderate gain. Even with fixes, 4-hop reasoning is hard for 3B model.
- **FinQA:** Needs table context added to prompts (see FinQA plan below).

### Immediate fixes applied:
1. **Sandbox import rules relaxed** — `sys`, `os`, `math`, `io`, `pathlib`, `re`, `json`, `collections`, `itertools`, `functools`, `datetime` etc. all allowed now. 21/21 tests pass.
2. **Next: re-run search datasets** to measure impact of sandbox fix.

---

## FinQA: Table Context Plan

### The Problem

FinQA questions reference financial tables that are NOT included in the question text. Example:

**Question:** "what is the net change in net revenue during 2015 for entergy corporation?"
**Gold answer:** 94.0

The model sees only the question. But the answer requires this table:

| | Amount (in millions) |
|---|---|
| 2014 net revenue | $5735 |
| Retail electric price | 187 |
| Volume/weather | 95 |
| Waterford 3 provision | -32 |

The answer is `5829 - 5735 = 94`. Without the table, there is no way to compute this.

### Raw FinQA Data Structure

Each FinQA example has:
- `pre_text` — paragraphs before the table (context)
- `table` — the actual data table (list of rows)
- `post_text` — paragraphs after the table (context)
- `qa.question` — the question
- `qa.exe_ans` — the gold answer

### Implementation Plan

**Modify `data/download_and_format.py`** to include table context in FinQA questions:

```python
def _format_finqa_question(ex):
    """Combine pre_text + table + post_text + question into a single context."""
    parts = []

    # Pre-text paragraphs
    if ex.get('pre_text'):
        parts.append('\n'.join(ex['pre_text']))

    # Table as markdown
    if ex.get('table'):
        table_lines = []
        for row in ex['table']:
            table_lines.append('| ' + ' | '.join(str(c) for c in row) + ' |')
            if len(table_lines) == 1:  # header separator
                table_lines.append('|' + '---|' * len(row))
        parts.append('\n'.join(table_lines))

    # Post-text paragraphs
    if ex.get('post_text'):
        parts.append('\n'.join(ex['post_text']))

    # Question
    parts.append('Question: ' + ex['qa']['question'])

    return '\n\n'.join(parts)
```

**Effort:** ~20 lines of code change in `download_and_format.py`. Re-generate FinQA eval splits. Re-run characterization.

### Pros of Adding Table Context

1. **Fair evaluation.** Without table context, FinQA EM is 0.6% — this is meaningless. The task is literally impossible without the data.
2. **Multi-tool benchmark.** With table context, FinQA becomes a genuine computation + reasoning task. The model reads the table, writes code to compute ratios/changes, and returns the answer. This is the "multi-tool" use case the paper targets.
3. **Paper completeness.** FinQA is listed as one of our evaluation datasets. Reporting 0.6% adds nothing. With table context, we can show meaningful base → trained improvement.

### Cons / Risks

1. **Long context.** Table + pre/post text can be 500-2000 tokens. Adds to prompt length, may slow inference.
2. **Different from Search-R1.** Search-R1 paper doesn't use FinQA with table context. Our comparison may not be apples-to-apples.
3. **Higher base accuracy.** With table context, the base model might already perform well (just compute from table), leaving less room for RL improvement.

### Recommendation

**Add table context.** The cons are minor. A 0.6% result is useless for the paper. With table context, FinQA becomes a real benchmark for the computation tool. If the base model already does well, that's fine — it validates that the tool works. RL should still improve assimilation and error handling.

---

## Throughput Summary

| Approach | Per example (4 rollouts) | 500 questions total |
|---|---|---|
| HF sequential (old) | ~240s | ~33 hours |
| vLLM wavefront (new) | 0.9-2.5s | 7-21 min |
| **Speedup** | **~100x** | **~100x** |
