# ToolSMDP — Results Tracker

Tracks all experimental results across milestones, feeding into the paper tables.

**Model:** Qwen2.5-3B-Instruct
**Inference:** vLLM wavefront batching on MI300X (ROCm)
**Search:** Pyserini BM25 over Wikipedia (21M passages)

---

## Milestone 2: Pre-Training Characterization

### Step 2.1 — Baseline (50 samples, pass@1, HF generate, Lightning.ai T4)

| Dataset | N | EM (pass@1) | Avg Tool Calls | Notes |
|---|---|---|---|---|
| GSM8K | 50 | 40.0% (20/50) | 1.88 | Over-calling tools on simple problems |
| HotpotQA | 50 | 8.0% (4/50) | 1.50 | Format issues, not searching when needed |

### Step 2.2 — Base Model + Tools (500 samples, pass@4, vLLM, MI300X)

**Job:** `bright_spinach_0gkn7pd4xl` (GSM8K, HotpotQA), `happy_hominy_jxxddy3x8p` (FinQA, Musique, 2Wiki)
**Date:** 2026-03-31

#### Overall Results

| Dataset | N | EM (pass@4) | Avg Tool Calls | Avg Segments | Time/example | Total Time |
|---|---|---|---|---|---|---|
| GSM8K | 500 | **84.2%** (421/500) | 2.31 | — | 2.5s | 21 min |
| HotpotQA | 500 | **21.4%** (107/500) | 4.41 | — | 0.9s | 7 min |
| 2Wiki | 500 | **31.0%** (155/500) | 3.82 | — | 2.0s | 16 min |
| FinQA | 500 | **0.6%** (3/500) | 2.21 | — | 1.0s | 9 min |
| Musique | 500 | **3.8%** (19/500) | 3.86 | — | 0.9s | 8 min |

#### Difficulty Bucket Breakdown

**GSM8K** (math — computation tool)

| Bucket | N | EM (pass@4) | % of dataset |
|---|---|---|---|
| 1_call | 105 | 82.9% (87/105) | 21% |
| 2_calls | 187 | 86.1% (161/187) | 37% |
| 3+_calls | 208 | 83.2% (173/208) | 42% |

**HotpotQA** (multi-hop QA — search tool)

| Bucket | N | EM (pass@4) | % of dataset |
|---|---|---|---|
| 1_call | 24 | 33.3% (8/24) | 5% |
| 2_calls | 24 | 16.7% (4/24) | 5% |
| 3+_calls | 452 | 21.0% (95/452) | 90% |

**2WikiMultiHopQA** (multi-hop QA — search tool)

| Bucket | N | EM (pass@4) | % of dataset |
|---|---|---|---|
| 1_call | 22 | 40.9% (9/22) | 4% |
| 2_calls | 66 | 56.1% (37/66) | 13% |
| 3+_calls | 412 | 26.5% (109/412) | 82% |

**FinQA** (financial QA — computation + search)

| Bucket | N | EM (pass@4) | % of dataset |
|---|---|---|---|
| 1_call | 109 | 0.9% (1/109) | 22% |
| 2_calls | 189 | 1.1% (2/189) | 38% |
| 3+_calls | 202 | 0.0% (0/202) | 40% |

**Musique** (multi-hop QA — search tool)

| Bucket | N | EM (pass@4) | % of dataset |
|---|---|---|---|
| 1_call | 35 | 5.7% (2/35) | 7% |
| 2_calls | 61 | 4.9% (3/61) | 12% |
| 3+_calls | 404 | 3.5% (14/404) | 81% |

#### Key Observations

1. **GSM8K is strong** (84% pass@4) — the model uses code blocks effectively for math. Consistent ~83-86% across all bucket sizes.
2. **Multi-hop QA is weak** — HotpotQA 21%, 2Wiki 31%, Musique 4%. The model makes many search calls (3.8-4.4 avg) but struggles to extract/combine information correctly.
3. **FinQA near zero** (0.6%) — financial reasoning requires interpreting tables + arithmetic. The 3B model lacks this domain competence.
4. **Most questions are 3+ calls** — 82-90% of search datasets, 40-42% of math. This validates the multi-tool focus of the paper.
5. **2Wiki > HotpotQA > Musique** for search QA — 2Wiki questions may be simpler than Musique's deeper multi-hop chains.

#### Throughput

| Metric | Old (HF sequential) | New (vLLM wavefront) | Speedup |
|---|---|---|---|
| Per example (4 rollouts) | ~240s | 0.9-2.5s | **~100x** |
| 500 questions × 4 rollouts | ~33 hours | ~10-20 min | **~100x** |

---

## Step 2.3 — Difficulty Buckets (TODO)

Build from Step 2.2 data. Classify questions by avg tool calls across rollouts.

## Step 2.4 — Tier 1/2 Training Splits (TODO)

Run base model WITHOUT tools. All 4 rollouts wrong → Tier 1. Any correct → Tier 2.

## Step 2.5 — Predictions Document (TODO)

Pre-register expected training results before any RL training.

---

## Milestone 3: Critic Warm-Up (TODO)

## Milestone 4: OpenRLHF Integration (TODO)

## Milestone 5: First Training + Validation (TODO)

Paper tables to fill:
- Table 1: Main results (EM by dataset, ToolSMDP vs baselines)
- Table 2: Difficulty bucket breakdown (EM by bucket, ToolSMDP vs baselines)
- Table 3: Tool selectivity (unnecessary calls, Tier 2 behavior)
- Table 4-7: Ablations (GRPO, reward variants, MC vs bootstrap, curriculum)

---

## Result Files

| Job | Datasets | Location |
|---|---|---|
| `bright_spinach_0gkn7pd4xl` | GSM8K, HotpotQA (50+500) | `downloads/vllm_trial/artifacts/outputs/` |
| `happy_hominy_jxxddy3x8p` | FinQA, Musique, 2Wiki (50+500) | `downloads/remaining_datasets/artifacts/outputs/` |
| `mighty_turnip_w7v26j6jr0` | All 5 datasets (50, HF baseline) | Running (old HF approach) |
