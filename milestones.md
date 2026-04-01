# ToolSMDP — Milestone Tracker

**Status key:** DONE | VALIDATED (approved, ready to execute) | DRAFT (not yet reviewed)

---

## Milestone 0: Environment + Data — DONE

Completed in Sessions 1-4. Project setup, container, all core components built and tested.
See CLAUDE.md "Session 1-4" for details.

## Milestone 1: Core ToolSMDP Components — DONE (needs update for `<context>` blocks)

Completed in Sessions 2-4. Code block detector, executor, replacement, reward, segment rollout.
110 passed, 2 skipped. See CLAUDE.md "Session 2-7" for details.

**Needs update (Session 8 design change):** The three-segment design (invoke/assimilate/synthesize)
requires new components. These are tracked as Step 2.0.5 below rather than reopening Milestone 1.

---

## Milestone 2: Pre-Training Analysis — VALIDATED

**Goal:** Build a working search backend, characterize the base model's tool-use behavior before
any training, construct difficulty buckets and tier splits for the paper's key results.

**Prerequisite:** GPU access (Lightning.ai T4 or equivalent).

**Key constraints (confirmed with user):**
- No quantization — run at full precision (float16 on T4, bfloat16 on A100+)
- No MATH dataset — dropped entirely. GSM8K stays for math reasoning.
- Search index must work inside Docker containers AND on bare-metal GPU platforms
- Model downloads via HuggingFace `from_pretrained()` (auto-cached)

---

### Step 2.0: Search Backend — DONE

Wikipedia search via Pyserini BM25 (`wikipedia-dpr-100w`, 21M passages).
First run downloads 9.2 GB index, cached at `~/.cache/pyserini/` after that.

**How it works:**
1. Rollout loop detects a code block with `search("...")`
2. Extracts query strings via `extract_search_query_strings()` (regex)
3. Resolves each query via Pyserini in the parent process
4. Injects results dict into sandboxed executor

**Files:**
- `retrieval/search.py` — `get_search()` returns a callable `query(str, top_k) -> str`
- `sandbox/executor.py` — `search_results` param, pre-resolved injection
- `scripts/try_search.py` — edit QUERIES list, run to test

**Setup:** `pip install pyserini faiss-cpu`, set `JAVA_HOME`, run.
**Tests:** 110 passed, 2 skipped (Pyserini skipped in Docker).

---

### Step 2.0.5: `<context>` Block Detection + Two-Phase Replacement — DONE

Implements the invoke/assimilate/synthesize segment structure from the paper (Session 8).
The model learns to write `<context>...</context>` after tool output. The rollout loop
detects this boundary, replaces raw tool output with the `<context>` contents, and
starts the next segment.

#### Design (agreed in Session 8)

**How a tool interaction works:**

```
1. INVOKE: Model writes reasoning + ```python code block
   → code executes, stdout replaces code (Phase 1 replacement)
   → context now has raw tool output (transient state s̃)

2. ASSIMILATE: Model continues generating, writes <context>key fact</context>
   → rollout loop detects </context> boundary
   → raw tool output replaced by <context> block (Phase 2 replacement)
   → context now has only distilled fact (clean state s)

3. SYNTHESIZE: Model continues reasoning / writes final answer
   → terminates at EOS or next code block (new invoke)
```

**Key design decisions:**
- `<context>` is a learned behavior, NOT a forced prompt (preserves K/V cache)
- System prompt instructs: "After seeing tool output, always write the key result
  in a `<context>...</context>` block before continuing."
- Always use `<context>` for ALL tools (search AND math) — consistent format
- Math example: `<context>347 * 28 = 9716</context>` (self-documenting)
- Assimilation budget: max 256 tokens per `<context>` block
- If model skips `<context>`, no assimilate segment — raw output persists (like Search-R1).
  RL pressure teaches the model that `<context>` leads to better outcomes.

**What was implemented:**

| Component | File | Change |
|---|---|---|
| `<context>` tag detector | `core/context_block_detector.py` (new) | `ContextBlockDetection`, `detect_context_block()`, `ContextBlockWatcher` with 256-token budget |
| Phase-2 replacement | `core/replacement.py` | `replace_tool_output_with_context()` — exact string match + swap |
| Segment dataclass | `core/segment_rollout.py` | `segment_type` field, `"context_block"` termination, `total_assimilations` property |
| Rollout loop | `scripts/test_base_model.py` | Full invoke→assimilate→synthesize flow with Phase-2 replacement |
| System prompt | `scripts/test_base_model.py` | Added `<context>` instruction with example |
| Max segments | `scripts/test_base_model.py` | `MAX_SEGMENTS = 15`, `MAX_ASSIMILATE_TOKENS = 256` |
| Tests | `tests/test_context_block_detector.py` (new) | 23 tests (post-hoc + watcher) |
| Tests | `tests/test_replacement.py` | 5 Phase-2 replacement tests |

**Deliverables:**
- [x] `<context>` tag detector (real-time + post-hoc)
- [x] Phase-2 replacement function
- [x] Updated `Segment` dataclass with `segment_type`
- [x] Updated rollout loop with invoke→assimilate→synthesize flow
- [x] Updated system prompt
- [x] Tests for all new components (138 passed, 2 skipped)

---

### Step 2.1: Validate pipeline on GPU (smoke test) — DONE

- Lightning.ai Studio (1x T4, interruptible)
- Model: Qwen2.5-3B-Instruct (switched from Qwen3.5-4B — too new for transformers 4.x)
- Pyserini search working (21M passages loaded)
- `<context>` blocks generated by base model (observed in interactive testing)
- `<answer>` tag added to system prompt for cleaner answer extraction

**Baseline results (50 samples each):**

| Dataset | EM | Avg Tool Calls | Key Issues |
|---|---|---|---|
| GSM8K | 40% (20/50) | 1.88 | Over-calling tools on simple problems, some wrong computations |
| HotpotQA | 8% (4/50) | 1.50 | Answer format (full sentences vs bare answer), not searching when needed |

**HotpotQA failure breakdown:**
- 16/50 didn't search at all (answered from wrong knowledge)
- 18/50 searched but extracted wrong info
- ~6 had correct info but wrong format ("No, they are not..." vs "no")
- 4 correct, 1 JSON error

**Model decision:** Staying with Qwen2.5-3B-Instruct. Problems are format/behavior issues (fixable by RL), not model capacity. 3B trains faster, fits T4. If SMDP effect doesn't show at Milestone 5, upgrade to 7B on A100.

**Deliverables:**
- [x] Environment working on GPU platform (Lightning.ai T4)
- [x] Code generation rate >30% (100% — 3/3)
- [x] `<context>` block generation observed
- [x] Full pipeline validated with real search results
- [x] Baseline numbers: GSM8K 40%, HotpotQA 8%
- [x] Results saved: `data_local/eval_splits/{gsm8k_test_50,hotpotqa_dev_50}_results.jsonl`

---

### Step 2.2: Run base model WITH tools on eval sets

**Script:** `analysis/pre_training_characterization.py` — WRITTEN
**AML job config:** `az/jobs/step_2_2.py` — WRITTEN (Python SDK, replaces old YAML)

- Full precision (float16 on T4, bfloat16 on A100+). NO quantization.
- 500 samples per eval dataset, 4 rollouts each
- Eval datasets: HotpotQA dev, Musique dev, 2Wiki dev, GSM8K test, FinQA test
  (NO MATH)
- Record per rollout:
  - num_segments, num_tool_calls
  - tool_types used (search, calc, both)
  - tool_output_relevance (keyword overlap with gold answer)
  - final_EM (correct or not)
  - full_generated text + final_context (for post-analysis)
- Logging: Python `logging` to stdout + `run.log` file in output dir
- Output: `$DATA_ROOT/analysis/{dataset}_rollout_stats.jsonl`

#### Step 2.2a: Validate AML infra (50 samples)

Run on existing 50-sample eval data to validate the full AML pipeline works:
1. Upload eval data: `python -m az.upload_data --name eval-data-50 --path data_local/eval_splits`
2. Submit job: `python -m az.submit_job --job step_2_2`
3. Check status: `python -m az.check_job --name <job-name>`
4. Download results: `python -m az.check_job --name <job-name> --download results`
5. Verify: model loads, GPU detected, search works, results saved

**AML Python helpers (`az/`, gitignored):**
- `az/config.py` — `get_ml_client()`, prompts for workspace details on first use, saves to `az/.env`
- `az/upload_data.py` — upload local folders as AML data assets
- `az/submit_job.py` — submit jobs from Python job configs in `az/jobs/`
- `az/check_job.py` — check status, stream logs, download outputs

**Deliverables:**
- [x] `analysis/pre_training_characterization.py` written and tested locally
- [x] `az/jobs/step_2_2.py` written (Python SDK)
- [x] `az/` helpers written (config, upload, submit, check)
- [ ] AML infra validated with 50-sample run (Step 2.2a)

#### Step 2.2b: Full 500-sample characterization

1. Generate 500-sample eval splits: `python -m data.download_and_format --datasets gsm8k hotpotqa finqa musique 2wiki --max-samples 500`
2. Upload: `python -m az.upload_data --name eval-data-500 --path data_local/eval_splits`
3. Duplicate `az/jobs/step_2_2.py` → `step_2_2_500.py`, point at `eval-data-500:1`
4. Submit: `python -m az.submit_job --job step_2_2_500`

**Deliverables:**
- [ ] 500-sample eval splits generated and uploaded
- [ ] Rollout stats for all 5 eval datasets
- [ ] Base model accuracy measured per dataset

---

### Step 2.3: Build difficulty buckets

**Script:** `analysis/build_difficulty_buckets.py` (new)

- From Step 2.2 rollouts, classify each question by avg tool calls:
  - avg < 1.5 → `1_call`
  - 1.5 <= avg < 2.5 → `2_calls`
  - avg >= 2.5 → `3+_calls`
- Per bucket compute: base accuracy, tool relevance rate, mixed-quality fraction
- Output: `$DATA_ROOT/analysis/difficulty_buckets.json`

```json
{
  "hotpotqa": {
    "1_call": { "question_ids": [...], "base_accuracy": 0.15, "n": 120 },
    "2_calls": { "question_ids": [...], "base_accuracy": 0.08, "n": 280 },
    "3+_calls": { "question_ids": [...], "base_accuracy": 0.03, "n": 100 }
  }
}
```

**Deliverables:**
- [ ] Difficulty buckets for all eval datasets
- [ ] Validates that multi-hop datasets actually produce 3+ call questions

---

### Step 2.4: Build Tier 1/2 training splits

**Script:** `data/filter_by_base_model.py` (new)

- Run base model WITHOUT tools on training splits, 4 rollouts per question
- All 4 wrong → Tier 1 (needs tools, ~70% of training batches)
- Any correct → Tier 2 (solvable directly, ~30% of batches)
- Output: `$DATA_ROOT/processed/{dataset}_tier1.jsonl`, `{dataset}_tier2.jsonl`

**Deliverables:**
- [ ] Tier 1/2 splits for: gsm8k, hotpotqa, nq, musique, 2wiki, finqa, triviaqa
- [ ] Per-dataset Tier 1/2 ratio reported

---

### Step 2.5: Write predictions document

- Pre-register expected results BEFORE any training
- Predictions based on difficulty bucket data from Step 2.3:
  - EM gains by bucket (expect gains to increase with tool call count)
  - Tool selectivity (expect fewer unnecessary calls on Tier 2)
  - Training curve shape (expect initial comparable phase, then crossover)
- Output: `predictions.md` in repo root

**Deliverables:**
- [ ] predictions.md written with concrete numeric predictions

---

## Milestone 3: Critic Warm-Up — DRAFT (not yet validated)

**Goal:** Generate (state, outcome) pairs from base model rollouts for critic initialization.
States now include BOTH transient states s̃ (with raw tool output) and clean states s
(with `<context>` blocks). The critic must learn to evaluate both.

### Step 3.1 — Generate rollout-based warmup data (~80K pairs)
- From Milestone 2 rollouts + additional rollouts on training data
- Extract states at ALL segment boundaries: s₀, s̃₁ (post-invoke), s₂ (post-assimilate), etc.
- Pair each state with the trajectory outcome R
- Binary labels (0/1). Free compute.
- Output: `$DATA_ROOT/processed/critic_warmup_rollout.jsonl`

### Step 3.2 — Generate LLM-labeled warmup data (~1K pairs)
- Use a strong model to rate intermediate states on 0-1 scale
- "Given this context, how likely is the model to produce a correct final answer?"
- Provides continuous supervision. ~$3-5 cost.
- Output: `$DATA_ROOT/processed/critic_warmup_llm.jsonl`

### Step 3.3 — Generate contrastive warmup pairs (~8K pairs)
- Same question, one state with relevant tool output, one with irrelevant
- Label relevant higher. Structured, free.
- Output: `$DATA_ROOT/processed/critic_warmup_contrastive.jsonl`

### Step 3.4 — Train critic head
- Freeze backbone, initialize 2-layer MLP (hidden dim 1024, ~7.5M params)
- Train 2 epochs on combined data, MSE loss
- ~15 min on free GPU
- Save head weights: `$CHECKPOINT_ROOT/critic_warmup/critic_head.pt`

### Step 3.5 — Verify critic calibration
- Spot-check: V(state with good search result) > V(state with bad search result)
- Scatter plot: V(s) vs actual R on held-out rollouts

**Deliverables:**
- Critic warmup data from 3 sources
- Trained critic head (~30KB)
- Calibration verified

---

## Milestone 4: OpenRLHF Integration — DRAFT (not yet validated)

**Goal:** Modify OpenRLHF for segment-level PPO with invoke/assimilate/synthesize segments.
Hardest engineering milestone.

### Step 4.1 — Study OpenRLHF internals
- `openrlhf/trainer/ppo_trainer.py` — main loop
- `openrlhf/trainer/ppo_utils/experience_maker.py` — rollout generation
- `openrlhf/models/actor.py`, `critic.py`

### Step 4.2 — Segment rollout generation (invoke/assimilate/synthesize)
- Replace experience maker's single-rollout with three-type segment rollout
- Each tool call produces 2 segments: invoke (code + execute) → assimilate (`<context>` block)
- Detect `</context>` boundary, do Phase-2 replacement, start next segment
- Max 15 segments per question

### Step 4.3 — Segment advantage computation
- A(seg_k) = V(s_{k+1}) - V(s_k) for intermediate segments
- A(seg_N) = R - V(s_N) for final segment
- Critic evaluates both transient states s̃ (raw output) and clean states s (`<context>`)
- Lambda = 0 (no multi-step lookahead)

### Step 4.4 — PPO loss with segment advantages
- All tokens in a segment share same scalar advantage
- PPO clipped loss unchanged, just advantage values differ
- Assimilate segments trained identically to invoke/synthesize

### Step 4.5 — Custom reward function
- Exact match, no neural RM

### Step 4.6 — Critic head initialization
- Load pre-trained head from Milestone 3

### Step 4.7 — Build Search-R1 GRPO baseline
- Same framework, fair comparison
- Single rollout, gradient masking, GRPO advantages

### Step 4.8 — Integration test
- 10 PPO steps on 10 GSM8K questions with Qwen2.5-1.5B
- Manually verify: segment types (invoke/assimilate/synthesize), V(s) values,
  V(s̃) vs V(s) differences, advantages, loss, no NaNs

**Deliverables:**
- Modified OpenRLHF with three-type segment PPO
- Search-R1 GRPO baseline in same framework
- Integration test passing

---

## Milestone 5: First Real Training + Validation (3B) — DRAFT (not yet validated)

**Goal:** Train at 3B scale, validate core claim, go/no-go for 7B.

### Step 5.1 — Train ToolSMDP on GSM8K (3B, cheap GPU)

### Step 5.2 — Train all three variants on GSM8K + HotpotQA subset (3B)

| Run | Method | What to Check |
|---|---|---|
| A | ToolSMDP + PPO (ours) | Does EM improve? Do advantages look right? |
| B | ToolSMDP + GRPO | Same segments but no critic — is EM lower? |
| C | Search-R1 GRPO repro | Single rollout — is EM lower on multi-hop? |

### Step 5.3 — Evaluate on difficulty buckets
- Per-bucket EM for all three runs
- Key test: A > B/C more on 3+ call questions than 1-call?

### Step 5.4 — Decision gate
- A > B on multi-hop → proceed to 7B
- A ≈ B → debug critic
- A < B → stop and rethink

**Deliverables:**
- Three trained 3B models
- Comparative bucket results
- Go/no-go decision

---

## Milestone 6: Commission Compute + 7B Runs — DRAFT (not yet validated)

Only after Milestone 5 passes. 4x A100 80GB.

### Core experiments (fill Tables 1-3)

| Run | Data | ~Hours |
|---|---|---|
| T1: ToolSMDP Search | NQ + HotpotQA (170K) | ~35h |
| T2: ToolSMDP Math | GSM8K (7.5K) | ~8h |
| T3: ToolSMDP Multi-tool | FinQA (6.2K) | ~10h |

### Ablations (fill Tables 4-7)

| Run | Method | ~Hours |
|---|---|---|
| T4: Search-R1 repro (GRPO) | Single rollout | ~28h |
| T5: ToolSMDP + GRPO | Segment, GRPO | ~30h |
| T6: Reward + exec penalty | + penalty | ~15h |
| T7: MC vs bootstrap targets | bootstrap V | ~15h |

### Full evaluation on all benchmarks with difficulty bucketing

**Deliverables:**
- All paper tables filled
- All trained models saved

---

## Milestone 7: Analysis, Figures, Paper — DRAFT (not yet validated)

### Step 7.1 — Generate diagnostic figures
1. Value function calibration (V(s) vs R scatter)
2. Advantage distributions **per segment type** (invoke, assimilate, synthesize)
3. Training curves (EM over steps, all methods)
4. Tool frequency by difficulty over training
5. Unnecessary tool call rate comparison
6. Error recovery rate
7. **Assimilation quality over training** (overlap between `<context>` content and gold answer)

### Step 7.2 — Fill all paper tables
- `analysis/generate_paper_tables.py` → markdown tables from eval results

### Step 7.3 — Finalize paper draft
- Insert tables, figures, actual numbers into main.tex

**Deliverables:**
- All figures generated
- Paper draft complete with real numbers
- Ready for submission
