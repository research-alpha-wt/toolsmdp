# ToolSMDP — Project Guide

## What This Project Is

Segment-level RL for LLM tool use via the Semi-Markov Decision Process framework. Trains LLMs to use tools (Python code blocks) with per-segment credit assignment using PPO and a learned value function. Three segment types — **invoke** (call tool), **assimilate** (distill tool output into `<context>` block), **synthesize** (reason/answer) — each an SMDP option with independent advantage. Good tool calls get positive advantage, bad assimilations get independently negative advantage, unnecessary calls get near-zero advantage — all from a single binary outcome reward.

Paper draft: `paper_draft/toolsmdp/main.tex`
**Milestone tracker: `milestones.md`** — canonical source for what to do next, step by step.

## What We Did

### Session 1 (2026-03-13): Design Review
No code written. Read both documents, identified 13 spec ambiguities, resolved all with rationale.

### Session 2 (2026-03-13): Milestone 0-1 Implementation
Built all core components and data pipeline:

- **Project setup**: `pyproject.toml`, package inits, container files
- **Container environment**: `Dockerfile`, `docker-compose.yml`, `.devcontainer/devcontainer.json`

### Session 3 (2026-03-13): Container TODOs + Code Cleanup
Resolved all current-milestone container TODOs and simplified code:

- **`.dockerignore`**: Created — excludes `.git`, `__pycache__`, `data_local/`, `checkpoints_local/`, `.devcontainer/`, `*.md` (except `CLAUDE.md`), `.claude/`
- **Dockerfile**: Cleaned up resolved TODOs. Added `sandbox` user + `SANDBOX_USER` env var.
- **`docker-compose.yml`**: Added `stdin_open`/`tty` to `dev`, `--timeout=30` to `test`, new `download` service with HF cache volume.
- **`sandbox/executor.py`**: Simplified — removed `_get_sandbox_uid()` indirection, removed `search_fn` parameter (unused placeholder), removed `_check_imports`/`_build_runner_script` helper split, consolidated into flat `execute_code()` function. 163→87 lines.
- **`data/download_and_format.py`**: Removed trivial one-liner extract functions (hotpotqa, musique, 2wiki → shared `_answer_field` lambda, finqa → `_finqa_field` lambda). Removed per-config `output_splits` dict in favor of shared `_SPLIT_DIRS` mapping. Removed empty `metadata: {}` from output. 275→167 lines.
- **`core/code_block_detector.py`**: Post-hoc `detect_code_block()` + real-time `CodeBlockWatcher` state machine. 17 tests in `tests/test_code_block_detector.py`.
- **`sandbox/executor.py`**: Subprocess execution, 5s timeout, import whitelist, `search()` injection. 21 tests in `tests/test_executor.py`.
- **`core/replacement.py`**: Comment-preserving replacement — code vanishes, comments + stdout remain. 10 tests in `tests/test_replacement.py`.
- **`core/reward.py`**: Per-dataset answer extraction (GSM8K `####`, MATH `\boxed{}`, QA patterns, FinQA numeric) + normalized exact match. 30 tests in `tests/test_reward.py`.
- **`core/segment_rollout.py`**: `Segment` and `Trajectory` dataclasses (training-loop bookkeeping only — model never sees these).
- **`data/download_and_format.py`**: Downloads 8 datasets from HuggingFace, converts to unified JSONL. Not yet run (needs HF access on compute platform).

### Session 4 (2026-03-16): Container Validation + Bug Fixes
- **Container validated**: Docker build + `docker compose run test` working on dev machine.
- **100/100 tests passing** after fixing 5 bugs found during first run:
  - `CodeBlockWatcher` EOS detection: check buffer instead of just current token (char-by-char feeding)
  - `executor.py` import guard: switched from allowlist to blocklist — stdlib internal deps (`codecs`, `decoder`, `_io`) were blocked
  - `reward.py` `_extract_last_number` regex: `\.?\d*` → `(?:\.\d+)?` to avoid matching trailing dots (`42.`)
  - FinQA test expectation: `-3.2` is correct, not `3.2`
- **pyproject.toml**: added `[tool.setuptools.packages.find]` to fix editable install
- **Dockerfile**: split dep install (non-editable) from source install (editable) for layer caching
- **VS Code Dev Container**: working — "Reopen in Container" connects to Linux container with full debugging
- **`.vscode/launch.json`**: added "Debug Tests" and "Debug Current Test File" configurations
- **Sample data downloaded** (50 examples each): `gsm8k_train_50`, `gsm8k_test_50`, `hotpotqa_dev_50`, `finqa_train_50`, `finqa_test_50`

### Session 5 (2026-03-17): Inference Script + Compute Planning
- **`scripts/test_base_model.py`**: Three-mode inference script for Qwen3.5-4B:
  - `code_gen`: Smoke test — does the model produce ```python blocks?
  - `pipeline`: Full SMDP segment loop (generate → detect → execute → replace → continue)
  - `eval`: Run on JSONL data, compute EM accuracy + difficulty bucket classification
- **Model choice**: Qwen3.5-4B (`Qwen/Qwen3.5-4B`), ~10GB VRAM BF16, 256K context
- **Auto dtype**: float16 on T4 (no native bfloat16), bfloat16 on A100/H200
- **Compute plan**: Lightning.ai T4 (79h free) for Milestones 2-3, save A100/H200 for training

### Session 6 (2026-03-21): Search Index + Milestone Planning
- **Milestone tracker**: Created `milestones.md` — canonical step-by-step tracker
- **Decisions**: MATH dataset dropped, no quantization ever, in-process BM25 (not microservice)
- **Retrieval package** (`retrieval/`):
  - `extract_corpus.py`: Extracts passages from HotpotQA/Musique/2Wiki HF data
  - `build_index.py`: Builds BM25Okapi index, saves to pkl
  - `search.py`: `BM25Search` class with `query()` and singleton `get_search()`
- **Pre-resolved search**: `executor.py` updated — `extract_search_query_strings()` extracts
  queries from code, rollout loop resolves via BM25, injects results dict
- **`test_base_model.py`**: `mode_eval` now auto-loads backend, passes `search_fn`
- **Container**: `build-index` service added to docker-compose.yml
- **Local BM25 validated**: 1,855 passages from 150 questions. HotpotQA 54% Recall@5,
  Musique/2Wiki weak (multi-hop answers not in single passages)
- **115/115 tests passing** (15 new search tests)

### Session 7 (2026-03-22): Pyserini Wikipedia Search + Simplification
- **PyseriniSearch**: Pyserini BM25 over `wikipedia-dpr-100w` (21M passages, 9.2 GB, cached)
- **Simplified search.py**: One backend, one function. Removed BM25Search, auto-detection, classes.
  `get_search()` returns a callable `query(str, top_k) -> str`.
- **Removed**: `extract_corpus.py`, `build_index.py`, `build-index` docker service, `rank_bm25` dep,
  `validate_search.py`, `ToolSMDP_Paper_Draft_v3.md`, `ToolSMDP_Implementation_Plan_v3.md`
- **Java 21 installed** via winget. JAVA_HOME: `/c/Program Files/Microsoft/jdk-21.0.10.7-hotspot`
- **`try_search.py`**: Simple script to test queries (edit QUERIES list, run it)
- **Codebase conventions updated**: added "one way to do things", "flat over nested" rules
- **110 passed, 2 skipped**

### Session 8 (2026-03-22): Three-Segment Design (invoke/assimilate/synthesize)
- **Paper updated** with three segment types: invoke, assimilate, synthesize
- **`<context>` block design** (replaces forced assimilation prompt from paper draft):
  - Model learns to write `<context>...</context>` after tool output via system prompt instruction
  - No forced prompt injection — preserves K/V cache, vLLM compatible
  - Always used (both search and math) for consistent format
  - Rollout loop detects `</context>` as segment boundary, replaces raw tool output
- **Two-phase replacement**: (1) code → stdout, (2) raw stdout → `<context>` block
- **Max 15 segments** (was 5). Each tool call = 2 segments (invoke + assimilate). 15 allows up to 7 tool calls.
- **Three independent skills trained**: invoke (what to search), assimilate (what to keep), synthesize (how to answer)
- **Key case analysis**: Case B vs E in paper — good search + bad assimilation separately penalized
- **Implementation plan updated** in milestones.md — new steps for `<context>` detector, two-phase replacement, segment typing

### Session 9 (2026-03-23): Step 2.0.5 — `<context>` Block Implementation
- **`core/context_block_detector.py`** (new): `ContextBlockDetection` dataclass, `detect_context_block()` regex, `ContextBlockWatcher` state machine with 256-token budget. Signals: `"context_block_complete"`, `"budget_exceeded"`, `"eos"`.
- **`core/replacement.py`**: Added `replace_tool_output_with_context()` — Phase-2 replacement (exact string match, swap raw stdout with distilled content).
- **`core/segment_rollout.py`**: Added `segment_type` field (`"invoke"` | `"assimilate"` | `"synthesize"`), `"context_block"` termination, `total_assimilations` property.
- **`scripts/test_base_model.py`**: System prompt with `<context>` instruction, `MAX_SEGMENTS=15`, `MAX_ASSIMILATE_TOKENS=256`, full invoke→assimilate→synthesize rollout loop.
- **Tests**: 23 new detector tests, 5 new replacement tests.
- **138 passed, 2 skipped**

## Key Decisions and Why

### Critic Architecture
- **2-layer MLP head** (hidden dim 1024, ~7.5M params) on shared backbone. Critic gradients flow through backbone.
- **Monte Carlo targets**: V_target = R for all states in episode. Unbiased, bounded variance (binary R, short episodes).
- **3 PPO epochs**, advantages computed once.
- **KL = 0 initially**, add 1e-4 only if EM drops >10% for 50+ steps.

### Segment Mechanics (invoke / assimilate / synthesize)
- **Three segment types**, each an SMDP option:
  - **Invoke**: model generates reasoning + code block. Terminates at closing ``` fence. Code executes, stdout replaces code.
  - **Assimilate**: model reads raw tool output, writes `<context>...</context>` block. Terminates at `</context>`. Raw output replaced by `<context>` block contents.
  - **Synthesize**: free reasoning or final answer. Terminates at EOS or length limit.
- **`<context>` block is learned behavior**, not a forced prompt. System prompt instructs: "After seeing tool output, always write the key result in a `<context>...</context>` block before continuing." Enforced by RL: good assimilations → correct answers → positive advantage.
- **Always use `<context>`**, for both search output (distill 500 tokens) and math output (wrap `42` as `<context>347 * 28 = 9716</context>`). Consistent format across all tools.
- **Max 15 segments**. Each tool call = 2 segments (invoke + assimilate). Typical 2-tool trajectory: invoke, assimilate, invoke, assimilate, synthesize = 5 segments.
- **Assimilation budget**: max 256 tokens per `<context>` block.
- **Two-phase replacement**: (1) code → stdout after invoke, (2) raw stdout → `<context>` block after assimilate. Both code and raw output are transient.
- **Sequential rollout generation** per question, **batched PPO update** across all segments.
- **GRPO variant excluded** from implementation. Focus purely on PPO.

### Tool Interface
- **Search**: Pyserini BM25 over Wikipedia (21M passages). `get_search()` returns a callable.
  Pre-resolved calls: rollout loop extracts queries from code, resolves in parent process, injects results.
- **Raw Python only** — no separate calculator tool. `search()` is a pre-built function declared in the system prompt. The Python interpreter IS the universal tool.
- **Error replacement**: comments + `"ERROR: message"` preserved in context. Enables recovery learning (Case E in the paper).

### Data & Curriculum
- **Per-batch Tier enforcement**: every batch of 128 has ~90 Tier 1 (needs tools) / ~38 Tier 2 (solvable directly).
- **Linear curriculum ramp** over first 30% of training: start 80% single-tool, ramp to natural distribution.
- **Mixing**: 30/70 NQ/HotpotQA for search; GSM8K for math; 50/25/25 FinQA/GSM8K-subset/HotpotQA-subset for multi-tool.

## Gotchas

1. **Model never sees segment boundaries as markers.** The model generates continuously. `Trajectory`/`Segment` objects are training-loop bookkeeping ONLY. The rollout loop detects boundaries (``` fence, `</context>`, EOS) and splits segments — the model just writes text.

2. **Two-phase replacement erases both code AND raw output.** After the full invoke→assimilate cycle, the context contains only the `<context>` block — neither the code that produced the search nor the full passage returned by the search. The critic evaluates post-assimilation states.

3. **`<context>` detection is a segment boundary.** When the rollout loop sees `</context>`, it: (a) marks the assimilate segment complete, (b) replaces raw tool output with `<context>` contents, (c) starts the next segment (synthesize or invoke). This is analogous to how ``` fence detection works for invoke segments.

4. **Critic warmup is essential.** Without it, initial V(s) predictions are random, making early advantages meaningless noise.

5. **K/V cache must be preserved.** The `<context>` block is part of the model's continuous generation — no prompt injection, no prefix modification. This is why we use a learned `<context>` tag instead of a forced assimilation prompt.

## What's Left to Do

See `milestones.md` for the full step-by-step tracker.

**Current:** Milestone 2 (Pre-Training Analysis) — VALIDATED, ready to execute.
**Step 2.0 (Search Backend) DONE. Step 2.0.5 (`<context>` blocks) DONE.** Next up: Step 2.1 (GPU smoke test with Qwen3.5-4B).

### Key decisions:
- MATH dataset dropped entirely
- No quantization — always run full precision (float16 on T4, bfloat16 on A100+)
- Search: Pyserini BM25 over Wikipedia (21M passages). One backend, no fallback chain.
- Pre-resolved search calls (rollout loop resolves queries before sandboxed execution)
- Musique HF path: `bdsaglam/musique` (not `drt/musique`)
- TriviaQA HF path: `mandarjoshi/trivia_qa` config `rc`

## How to Run

```bash
# Run tests (local with Docker)
docker compose run test

# Run tests (local with conda — quick one-liner)
conda run -n toolsmdp pytest tests/ -v

# Download datasets
docker compose run download python -m data.download_and_format --datasets gsm8k hotpotqa finqa --max-samples 50

# Test search locally (edit QUERIES list in the script)
python scripts/try_search.py

# Base model inference (requires GPU — run on Lightning.ai or similar)
python -m scripts.test_base_model --mode code_gen              # smoke test
python -m scripts.test_base_model --mode pipeline              # full SMDP loop
python -m scripts.test_base_model --mode eval --data data_local/eval_splits/hotpotqa_dev_50.jsonl

# Lightning.ai setup
git clone <repo> && cd toolsmdp
pip install -e ".[train,dev]"
pytest tests/ -v
```

See `SETUP.md` for conda environment setup and JAVA_HOME configuration.

## Codebase Conventions
- **This is research code, not production code.** Keep it simple, readable, and small.
- **One way to do things.** Don't build multiple backends, fallback chains, or abstraction layers. Pick one approach and use it directly.
- **Flat over nested.** Functions over classes when a class would just have one method. No inheritance hierarchies. No factory patterns.
- **No unnecessary abstractions.** Three similar lines > a premature helper function. No wrappers around libraries that add no value.
- **Minimal docstrings.** A one-liner is fine. Multi-paragraph docstrings are not. Let function names and signatures speak.
- **No defensive coding** for impossible scenarios. Trust internal code paths.
- Modular package structure: `core/`, `sandbox/`, `data/`, `retrieval/`, `tests/`
- All paths via `$DATA_ROOT` and `$CHECKPOINT_ROOT` env vars (compute-agnostic)
- Container-first: all deps managed via `pyproject.toml`, never `pip install` on bare metal
