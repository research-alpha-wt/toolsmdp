# ToolSMDP — Project Guide

## What This Project Is

Segment-level RL for LLM tool use via the Semi-Markov Decision Process framework. Trains LLMs to use tools (Python code blocks) with per-segment credit assignment using PPO and a learned value function. The key insight: decompose trajectories at tool-call boundaries so good tool calls get positive advantage, bad ones get negative advantage, and unnecessary ones get near-zero advantage — all from a single binary outcome reward.

Paper draft: `ToolSMDP_Paper_Draft_v3.md`
Implementation plan: `ToolSMDP_Implementation_Plan_v3.md`

## What We Did

### Session 1 (2026-03-13): Design Review
No code written. Read both documents, identified 13 spec ambiguities, resolved all with rationale.

### Session 2 (2026-03-13): Milestone 0-1 Implementation
Built all core components and data pipeline:

- **Project setup**: `pyproject.toml`, package inits, container files
- **Container environment**: `Dockerfile`, `docker-compose.yml`, `.devcontainer/devcontainer.json`
  - Not yet tested — Docker not available on dev machine. Must validate on compute platform.
- **`core/code_block_detector.py`**: Post-hoc `detect_code_block()` + real-time `CodeBlockWatcher` state machine. 17 tests in `tests/test_code_block_detector.py`.
- **`sandbox/executor.py`**: Subprocess execution, 5s timeout, import whitelist, `search()` injection. 21 tests in `tests/test_executor.py`.
- **`core/replacement.py`**: Comment-preserving replacement — code vanishes, comments + stdout remain. 10 tests in `tests/test_replacement.py`.
- **`core/reward.py`**: Per-dataset answer extraction (GSM8K `####`, MATH `\boxed{}`, QA patterns, FinQA numeric) + normalized exact match. 30 tests in `tests/test_reward.py`.
- **`core/segment_rollout.py`**: `Segment` and `Trajectory` dataclasses (training-loop bookkeeping only — model never sees these).
- **`data/download_and_format.py`**: Downloads 8 datasets from HuggingFace, converts to unified JSONL. Not yet run (needs HF access on compute platform).

**Tests not yet run** — all written, awaiting container build. Run with `docker compose run test`.

## Key Decisions and Why

### Critic Architecture
- **Backbone frozen for critic** (`.detach()` on hidden states before critic head). Why: critic head is only 3,584 params; letting critic gradients flow into the backbone would fight with policy gradients. Critic warmup also assumes frozen backbone representations.
- **Bootstrap targets with stop-gradient** (`target(s_k) = sg[V(s_{k+1})]`). Why: aligns with the paper's core claim that `V(s_{k+1}) - V(s_k)` provides per-state differentiation. Monte Carlo (target = R for all segments) would weaken this. Standard in OpenRLHF, no custom code needed.
- **3 PPO epochs**, advantages computed once. Why: standard for Search-R1/OpenRLHF; recomputing advantages each epoch requires expensive forward passes at every segment boundary.
- **KL = 0 initially**, add 1e-4 only if EM drops >10% for 50+ steps. Why: matches Search-R1 baseline for fair comparison; base models don't have instruction-following behavior worth preserving; PPO clip provides implicit constraint.

### Segment Mechanics
- **Max 5 segments**, force EOS at limit, extract answer, score normally. Track hit-rate in wandb.
- **1024 tokens per segment** (adjustable per task).
- **Sequential rollout generation** per question, **batched PPO update** across all ~320 segments. Why: tool execution breaks generation pipeline anyway; batching rollouts adds complexity for minimal speedup. PPO update is standard batched training.
- **GRPO variant excluded** from implementation. Focus purely on PPO.

### Tool Interface
- **Search backend**: local BM25 index OR cached retrieval (both options kept open).
- **Raw Python only** — no separate calculator tool. `search()` is a pre-built function declared in the system prompt. The Python interpreter IS the universal tool.
- **Error replacement**: comments + `"ERROR: message"` preserved in context. Enables recovery learning (Case E in the paper).

### Data & Curriculum
- **Per-batch Tier enforcement**: every batch of 128 has ~90 Tier 1 (needs tools) / ~38 Tier 2 (solvable directly).
- **Linear curriculum ramp** over first 30% of training: start 80% single-tool, ramp to natural distribution.
- **Mixing**: 30/70 NQ/HotpotQA for search; 50/50 GSM8K/MATH for math (deferred); 50/25/25 FinQA/GSM8K-subset/HotpotQA-subset for multi-tool.

## Gotchas

1. **Model never sees segment boundaries.** The model receives `full_context` (clean natural text). `Trajectory`/`Segment` objects are training-loop bookkeeping ONLY. Segment count is tracked by the rollout loop as it calls `model.generate()` — never inferred from the text. Don't encode segment markers in the context.

2. **Comment-preserving replacement erases code.** After execution, the context contains comments + stdout but NOT the code that produced the output. This makes the SMDP partially observed. The paper argues this is fine because tool output is more predictive of future success than the code syntax that produced it. The critic evaluates post-replacement context.

3. **Critic warmup is essential.** Without it, initial V(s) predictions are random, making early advantages meaningless noise. Three data sources combined: rollout-based (~80K, binary, free), LLM-labeled (~1K, continuous, ~$3-5), contrastive (~8K, structured, free).

4. **No reference model in memory when KL=0.** Saves ~14GB. Load it only if switching to KL=1e-4.

5. **Container not yet validated.** Dockerfile, compose, and devcontainer were written but Docker is not installed on the dev machine. First action on any compute platform: `docker compose run test`.

## What's Left to Do

### Immediate (next session)
- [ ] Get Docker running and validate container build + test suite
- [ ] Run `data/download_and_format.py` on compute platform to download datasets
- [ ] Verify answer extraction on 10 examples per dataset (script prints these)
- [ ] `scripts/test_base_model_code_generation.py` — verify Qwen generates code blocks

### Milestone 2: Pre-Training Analysis
- [ ] Run base model with tools on eval sets (500 samples per dataset)
- [ ] Build difficulty buckets (1-call, 2-call, 3+-call)
- [ ] Build Tier 1/2 training splits
- [ ] Write predictions document (pre-registration)

### Milestone 3: Critic Warm-Up
- [ ] Generate rollout-based warmup data (~80K pairs)
- [ ] Generate LLM-labeled warmup data (~1K pairs)
- [ ] Generate contrastive warmup pairs (~8K pairs)
- [ ] Train critic head on combined data

### Milestone 4: OpenRLHF Integration
- [ ] Study OpenRLHF internals (ppo_trainer, experience_maker, actor, critic)
- [ ] Modify experience maker for segment rollout generation
- [ ] Implement segment advantage computation
- [ ] PPO loss with segment advantages (all tokens in segment share same scalar)
- [ ] Custom reward function (exact match, no neural RM)
- [ ] Critic head initialization from warmup
- [ ] 10-step integration test on GSM8K

### Milestone 5-7: Training, Evaluation, Paper
- [ ] 3B scale training + validation
- [ ] 7B scale runs (only if 3B validates)
- [ ] Full evaluation on all benchmarks with difficulty bucketing
- [ ] Diagnostic figures and paper tables

## How to Run

```bash
# Run tests
docker compose run test

# Interactive dev shell
docker compose run dev

# Download datasets (inside container)
DATA_ROOT=/data python -m data.download_and_format

# Download specific datasets only
DATA_ROOT=/data python -m data.download_and_format --datasets gsm8k hotpotqa
```

## Codebase Conventions
- Modular package structure: `core/`, `sandbox/`, `data/`, `tests/`
- Minimal abstractions — no over-engineering
- No excessive comments; code should be self-explanatory
- All paths via `$DATA_ROOT` and `$CHECKPOINT_ROOT` env vars (compute-agnostic)
- Container-first: all deps managed via `pyproject.toml`, never `pip install` on bare metal
