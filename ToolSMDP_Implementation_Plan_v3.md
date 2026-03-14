# ToolSMDP — Final Implementation Plan (v3)

## Key Decisions (Settled)

- **Models:** Qwen2.5-3B (dev), Qwen3.5-4B (test if smaller works), Qwen2.5-7B (final results matching baselines)
- **Baselines:** Reimplement Search-R1 GRPO in OpenRLHF
- **Datasets:** HotpotQA + Musique + 2WikiMultiHopQA (primary), GSM8K + MATH (competitiveness), FinQA (multi-tool), NQ + TriviaQA (text mention)
- **Compute:** Lightning.ai (dev) + Colab Pro (backup), paid compute only for final runs
- **Analysis-first:** Pre-training characterization before any training
- **No commissioned compute until pipeline is validated end-to-end**
- **Code is compute-agnostic:** works on Lightning, Colab, RunPod, AWS via $DATA_ROOT

---

## Environment Variable Contract

All code references these variables. Set once per platform:

```bash
# Lightning.ai
export DATA_ROOT="/teamspace/studios/this_studio/toolsmdp_data"
export CHECKPOINT_ROOT="/teamspace/studios/this_studio/toolsmdp_checkpoints"

# Google Colab
export DATA_ROOT="/content/drive/MyDrive/toolsmdp/data"
export CHECKPOINT_ROOT="/content/drive/MyDrive/toolsmdp/checkpoints"

# RunPod (when you commission compute)
export DATA_ROOT="/workspace/toolsmdp/data"
export CHECKPOINT_ROOT="/workspace/toolsmdp/checkpoints"

# AWS
export DATA_ROOT="/mnt/efs/toolsmdp/data"
export CHECKPOINT_ROOT="/mnt/efs/toolsmdp/checkpoints"
```

---

## Milestone 0: Environment + Data (Days 1-2)

**Goal:** All datasets downloaded, formatted, base model confirmed to generate code blocks. No GPU needed for most of this.

### 0.1 Setup (CPU only)

```bash
# Create project
mkdir toolsmdp && cd toolsmdp
git init

# Install dependencies (no GPU needed yet)
pip install datasets transformers tokenizers
pip install pandas jsonlines tqdm

# Create directory structure
mkdir -p data/{raw,processed,eval_splits}
mkdir -p configs
mkdir -p core sandbox retrieval evaluation analysis scripts results
```

### 0.2 Download and format all datasets

```python
# data/download_and_format.py
# Downloads from HuggingFace + converts to unified format
# Output format per example:
# {
#   "question": str,
#   "gold_answer": str or list[str],
#   "dataset": str,
#   "split": "train" or "dev" or "test",
#   "metadata": { ... dataset-specific fields ... }
# }

# TRAINING DATA
# GSM8K train: 7,473 questions → $DATA_ROOT/processed/gsm8k_train.jsonl
# MATH train: 7,500 problems → $DATA_ROOT/processed/math_train.jsonl
# HotpotQA train: 90,447 questions → $DATA_ROOT/processed/hotpotqa_train.jsonl
# NQ train: 79,168 questions → $DATA_ROOT/processed/nq_train.jsonl
# FinQA train: 6,251 questions → $DATA_ROOT/processed/finqa_train.jsonl

# EVAL DATA (NEVER train on these)
# HotpotQA dev: 7,405 → $DATA_ROOT/eval_splits/hotpotqa_dev.jsonl
# Musique dev: 2,417 → $DATA_ROOT/eval_splits/musique_dev.jsonl
# 2WikiMultiHopQA dev: 12,576 → $DATA_ROOT/eval_splits/2wiki_dev.jsonl
# GSM8K test: 1,319 → $DATA_ROOT/eval_splits/gsm8k_test.jsonl
# MATH test: 5,000 → $DATA_ROOT/eval_splits/math_test.jsonl
# FinQA test: 1,147 → $DATA_ROOT/eval_splits/finqa_test.jsonl
# NQ test: 3,610 → $DATA_ROOT/eval_splits/nq_test.jsonl
# TriviaQA test: 11,313 → $DATA_ROOT/eval_splits/triviaqa_test.jsonl
```

**Per-dataset answer extraction (critical to get right):**
- GSM8K: answer field contains chain-of-thought + `#### <number>`. Extract after `####`.
- MATH: solution field contains `\boxed{<answer>}`. Extract from boxed.
- HotpotQA: `answer` field is a string. Direct use.
- NQ: `answer` field is a list of acceptable answers. Any match counts.
- Musique: `answer` field is a string. Direct use.
- 2WikiMultiHopQA: `answer` field is a string. Direct use.
- FinQA: `exe_ans` field is the numeric answer. Handle percentages/currencies.
- TriviaQA: `answer.aliases` is a list of acceptable answers.

### 0.3 Verify base model generates code blocks (needs GPU — use free tier)

```python
# scripts/test_base_model_code_generation.py
# Run on Colab T4 or Lightning free GPU
# Test with multiple models and prompts

SYSTEM_PROMPT = """You can write Python code in code blocks. Code will be executed and you will see the output. Available functions: search(query).
Always begin code blocks with a comment explaining what you are computing."""

TEST_QUESTIONS = [
    "What is 347 × 28?",
    "What is 15% of France's GDP?",
    "What country hosted the 2024 Olympics?",
    "What is the sum of the first 100 positive integers?",
]

MODELS_TO_TEST = [
    "Qwen/Qwen2.5-1.5B",  # Smallest, fastest to test
    "Qwen/Qwen2.5-3B",    # Dev model
    # "Qwen/Qwen3.5-4B",  # If available and fits on T4
]

# For each model × question:
# Generate 4 completions at temperature 0.7
# Record: does output contain ```python? Does it contain a # comment?
# Report: code block generation rate, comment rate per model
```

**What you're looking for:**
- Code block rate > 30% → good, RL can improve this
- Code block rate < 5% → problem, need different prompt or different model
- Comment rate > 50% of code blocks → good, model naturally comments
- Comment rate < 10% → add stronger prompting for comments

### Deliverable
- All datasets downloaded and formatted ✓
- Answer extraction verified on 10 examples per dataset ✓
- Base model code generation rate measured ✓
- Decision: which base model to use for development ✓

---

## Milestone 1: Core ToolSMDP Components (Days 2-4)

**Goal:** All novel components built and unit tested. No training yet.

### 1.1 Code block detector (with state machine for real-time detection)

```python
# core/code_block_detector.py
# Two modes:
# (a) Post-hoc detection: given full generated text, find code blocks
# (b) Real-time detection: during token-by-token generation, detect closing fence

# Post-hoc (for testing):
def detect_code_block(text: str) -> dict | None

# Real-time (for training):
class CodeBlockWatcher:
    """Watches generated tokens and signals when a code block is complete."""
    def __init__(self):
        self.state = "NORMAL"  # NORMAL | IN_CODE_BLOCK
        self.buffer = ""
    
    def feed_token(self, token_text: str) -> str:
        """Returns 'continue', 'code_block_complete', or 'eos'"""
        ...
```

### 1.2 Sandboxed executor

```python
# sandbox/executor.py
# Execute Python code in a subprocess with:
# - 5 second timeout
# - No network access (except search() if enabled)
# - No filesystem access
# - Whitelisted imports: math, numpy, collections, itertools, re, statistics
# - search() function injected if search_enabled=True

def execute_code(code: str, search_enabled: bool = False,
                 search_server_url: str = "http://localhost:8000") -> str:
    """Returns stdout or 'ERROR: <message>'"""
```

### 1.3 Comment-preserving replacement

```python
# core/replacement.py
def replace_code_block(generated_text: str, detection: dict, stdout: str) -> str:
    """Replace code block with comments + stdout."""
```

### 1.4 Reward function with per-dataset answer extraction

```python
# core/reward.py
def extract_answer(text: str, dataset: str) -> str | None
def exact_match(pred: str, gold: str | list[str]) -> bool
def compute_reward(generated_text: str, gold_answer: str | list[str], dataset: str) -> float
```

### 1.5 Segment-level rollout generator (the integration piece)

```python
# core/segment_rollout.py
@dataclass
class Segment:
    start_context: str
    generated_text: str
    generated_ids: list[int]
    log_probs: list[float]
    termination: str  # "tool_call" | "eos" | "truncated"
    tool_code: str | None
    tool_comments: list[str] | None
    tool_output: str | None
    advantage: float | None  # Filled later
    value_estimate: float | None
    value_target: float | None

@dataclass
class Trajectory:
    segments: list[Segment]
    full_context: str
    reward: float | None
```

### 1.6 Comprehensive unit tests

```python
# tests/test_all.py
# 1. Code detection: 15+ test cases including edge cases
#    - Normal code block
#    - Code block with no comments
#    - Multiple code blocks (should detect first)
#    - Unclosed code block
#    - Code fence in natural text discussion
#    - Empty code block
#    - Code block with only comments, no executable code

# 2. Executor: 10+ test cases
#    - Simple arithmetic: print(2+2) → "4"
#    - String operation: print("hello"[::-1]) → "olleh"
#    - Error handling: pritn(42) → "ERROR: ..."
#    - Timeout: while True: pass → "ERROR: Execution timed out"
#    - Import allowed: import math; print(math.pi) → "3.14..."
#    - Import blocked: import os → "ERROR: ..."
#    - Multi-line output
#    - search() function (with mock)

# 3. Replacement: 10+ test cases
#    - Basic replacement
#    - Comment preservation
#    - No comments case
#    - Multiple comments
#    - Context reads naturally after replacement

# 4. Reward: 5+ test cases per dataset
#    - Verify answer extraction on real examples from each dataset
#    - Verify exact match normalization (whitespace, case, punctuation)

# 5. End-to-end segment rollout (with mock model)
#    - Question → 2 segments → correct answer → R=1
#    - Question → 1 segment → wrong answer → R=0
#    - Question → 3 segments with error recovery → R=1
```

### Deliverable
- All core components passing all unit tests ✓
- Can demonstrate end-to-end: question → segments → tool execution → replacement → answer extraction → reward ✓

---

## Milestone 2: Pre-Training Analysis (Days 4-6)

**Goal:** Characterize the datasets, build difficulty buckets, predict where gains should appear. This validates your hypothesis BEFORE spending on training.

### 2.1 Run base model with tools on eval sets

```python
# analysis/pre_training_characterization.py
# Run on free GPU (Colab T4 or Lightning L4)
# Uses 4-bit quantized model for inference (no training, so quantization is fine)

# For each eval dataset:
#   For each question (sample 500 if dataset > 1000):
#     Generate 4 rollouts with tool access
#     Record per rollout:
#       - num_segments
#       - num_tool_calls
#       - tool_types used (search, calc, both)
#       - whether each tool output was relevant (keyword overlap with gold answer)
#       - whether final answer was correct
```

### 2.2 Build difficulty buckets

```python
# analysis/build_difficulty_buckets.py
# From the rollouts above, classify each question:
# - avg_tool_calls < 1.5 → "1_call" bucket
# - 1.5 <= avg_tool_calls < 2.5 → "2_calls" bucket
# - avg_tool_calls >= 2.5 → "3+_calls" bucket
#
# Also compute:
# - base_model_accuracy per bucket (without RL)
# - tool_relevance_rate per bucket
# - fraction of rollouts with mixed-quality tool calls (good + bad in same trajectory)

# Output: $DATA_ROOT/analysis/difficulty_buckets.json
# {
#   "hotpotqa": {
#     "1_call": { "question_ids": [...], "base_accuracy": 0.15, "n": 120 },
#     "2_calls": { "question_ids": [...], "base_accuracy": 0.08, "n": 280 },
#     "3+_calls": { "question_ids": [...], "base_accuracy": 0.03, "n": 100 },
#   },
#   ...
# }
```

### 2.3 Build training data tiers

```python
# data/filter_by_base_model.py
# Run base model WITHOUT tools on training splits
# Generate 4 rollouts per question, no tools
# Questions model gets wrong on all 4 → Tier 1 (needs tools)
# Questions model gets right on any → Tier 2 (can answer directly)

# Output:
# $DATA_ROOT/processed/gsm8k_tier1.jsonl, gsm8k_tier2.jsonl
# $DATA_ROOT/processed/hotpotqa_tier1.jsonl, hotpotqa_tier2.jsonl
# etc.
```

### 2.4 Write predictions document

Before any training, write down:
- "We predict gains of X-Y EM on HotpotQA 3+ call bucket"
- "We predict comparable performance on 1-call bucket"
- "We predict ToolSMDP will reduce unnecessary tool calls on Tier 2 questions"

This becomes a pre-registration of sorts. If results match predictions, it's extremely strong evidence.

### Deliverable
- Difficulty buckets for all eval datasets ✓
- Tier 1/2 splits for all training datasets ✓
- Written predictions document ✓
- Decision: which datasets actually have multi-tool-call questions (validates dataset choice) ✓

---

## Milestone 3: Critic Warm-Up Data (Days 6-7)

**Goal:** Generate (state, outcome) pairs from base model rollouts for critic initialization.

### 3.1 Generate critic warm-up data

```python
# core/critic_warmup.py
# Run base model WITH tools on training questions (10K subset)
# Generate 4 rollouts per question
# For each rollout, extract states at segment boundaries
# Pair each state with the trajectory outcome R

# Output: $DATA_ROOT/processed/critic_warmup_data.jsonl
# Each line: {"state_text": "...", "outcome": 0 or 1}
# Expected: ~80K-120K pairs, mostly outcome=0 (base model fails)
```

### 3.2 Train critic head

```python
# core/train_critic_warmup.py
# Load base model (frozen)
# Initialize linear head (3584 → 1)
# Train on warmup data using MSE loss
# Save only the critic head weights (tiny file)

# Output: $CHECKPOINT_ROOT/critic_warmup/critic_head.pt (~14KB)
```

This takes ~15 minutes on a free GPU. The backbone stays frozen — you're only training 3,584 parameters.

### Deliverable
- Critic warm-up data generated ✓
- Critic head pre-trained ✓
- Verified: V(state with relevant tool output) > V(state with irrelevant output) on a few manual examples ✓

---

## Milestone 4: OpenRLHF Integration (Days 7-12)

**Goal:** Modify OpenRLHF to support segment-level PPO. This is the hardest engineering milestone.

### 4.1 Understand OpenRLHF internals

Study these files:
- `openrlhf/trainer/ppo_trainer.py` — main loop
- `openrlhf/trainer/ppo_utils/experience_maker.py` — rollout generation
- `openrlhf/models/actor.py` — actor with generation
- `openrlhf/models/critic.py` — critic with value head

### 4.2 Key modifications

**Modification 1: Segment rollout generation**
Replace the experience maker's single-rollout generation with your `SegmentRolloutGenerator`. Each question produces N segments instead of one trajectory.

**Modification 2: Segment advantage computation**
Replace GAE with your `compute_segment_advantages`. Each segment gets V(s_{k+1}) − V(s_k) instead of token-level GAE.

**Modification 3: PPO loss with segment advantages**
Ensure all tokens within a segment receive the same advantage scalar. The PPO clipped loss is unchanged — just the advantage values differ.

**Modification 4: Custom reward function**
Use OpenRLHF's `--remote_rm_url` or inline reward function support. Your reward is exact match — no neural reward model.

**Modification 5: Critic head initialization**
Load the pre-trained critic head from Milestone 3 at the start of training.

### 4.3 Also implement: Search-R1 GRPO baseline in OpenRLHF

**Modification for Search-R1 repro:**
- Single-rollout generation (no segments)
- Tool outputs injected into sequence with tags
- Gradient masking on injected tokens
- GRPO advantages (group normalization, no critic)

**Modification for ToolSMDP + GRPO ablation:**
- Segment-level rollout (same as yours)
- But replace advantage computation with GRPO: generate K rollouts per question, normalize rewards across group, all segments in a rollout get the same normalized advantage

### 4.4 Integration test

Run 10 steps on 10 GSM8K questions with Qwen2.5-1.5B. Manually verify:
- Segments are created correctly
- Code blocks are detected and executed
- Replacement produces clean context
- V(s) values are non-trivial (not all same number)
- Advantages are computed correctly (manually verify 2-3 trajectories)
- Loss decreases
- No NaN anywhere

### Deliverable
- Modified OpenRLHF running segment-level PPO ✓
- Search-R1 GRPO baseline running in same framework ✓
- ToolSMDP + GRPO variant running ✓
- 10-step integration test passing ✓

---

## Milestone 5: First Real Training + Validation (Days 12-18)

**Goal:** Train at 3B scale, validate core claim, decide if ready for 7B.

### 5.1 Train ToolSMDP on GSM8K (3B, free or cheap GPU)

Lightning.ai L4 or Colab A100 might handle this. If not, this is the first point to commission cheap compute (1× A40 on RunPod, ~$2 for this run).

### 5.2 Train all three variants on GSM8K + HotpotQA subset (3B)

| Run | Method | What to Check |
|---|---|---|
| A | ToolSMDP + PPO (ours) | Does EM improve? Do advantages look right? |
| B | ToolSMDP + GRPO | Same segments but no critic — is EM lower? |
| C | Search-R1 GRPO repro | Single rollout — is EM lower on multi-hop? |

### 5.3 Evaluate on difficulty buckets

For each trained model, evaluate on HotpotQA dev and GSM8K test, broken down by difficulty bucket. Check: does Run A beat Run B/C more on 3+ call questions than on 1-call questions?

### 5.4 Decision gate

If A > B on multi-hop: ✅ proceed to 7B
If A ≈ B on multi-hop: ⚠️ debug critic, check V(s) calibration, try critic warm-up
If A < B: ❌ fundamental issue, revisit approach before spending on 7B

### Deliverable
- Three trained 3B models ✓
- Comparative results on difficulty buckets ✓
- Go/no-go decision for 7B ✓

---

## Milestone 6: Commission Compute + 7B Runs (Days 18-28)

**Only after Milestone 5 succeeds.** Commission 4× A100 80GB.

### 6.1 Core experiments (Tables 1-3)

| Run | Data | Est. Wall Hours | Fills Table |
|---|---|---|---|
| T1: ToolSMDP Search | NQ + HotpotQA (170K) | ~35 hrs | Table 1 |
| T2: ToolSMDP Math | GSM8K + MATH (15K) | ~12 hrs | Table 2 |
| T3: ToolSMDP Multi-tool | FinQA (6.2K) | ~10 hrs | Table 3 |

### 6.2 Ablations (Tables 4-6)

| Run | Method | Est. Wall Hours | Fills Table |
|---|---|---|---|
| T4: Search-R1 repro (GRPO) | Single rollout, GRPO | ~28 hrs | Tables 1, 4, 5 |
| T5: ToolSMDP + GRPO | Segment, GRPO | ~30 hrs | Tables 4, 5 |
| T6: Reward + exec penalty | Our method + penalty | ~15 hrs | Table 5 |
| T7: MC value targets | Our method + MC targets | ~15 hrs | Table 6 |

### 6.3 Full evaluation

All trained models evaluated on all benchmarks with difficulty bucketing.

---

## Milestone 7: Analysis, Figures, Paper (Days 28-32)

### 7.1 Generate all diagnostic figures

1. Value function calibration (V(s) vs R scatter plot)
2. Advantage distributions (relevant vs irrelevant tool outputs)
3. Training curves (EM over steps, all methods)
4. Tool frequency over training (broken by question difficulty)
5. Unnecessary tool call rate (Tier 2 questions)
6. Error recovery rate

### 7.2 Fill all paper tables

Use `analysis/generate_paper_tables.py` to automatically generate markdown tables from eval results.

### 7.3 Finalize paper draft

Insert tables, figures, and actual numbers into the draft.

---

## Monitoring and Correctness

### Wandb Logging (Every Training Step)

```python
wandb.log({
    # Losses
    "loss/policy": policy_loss,
    "loss/critic": critic_loss,
    
    # Reward
    "reward/mean": mean_reward,
    "reward/fraction_correct": fraction_correct,
    
    # Tool use
    "tools/calls_per_question": avg_tool_calls,
    "tools/fraction_using_tools": frac_using_tools,
    "tools/fraction_unnecessary": frac_unnecessary_on_tier2,
    
    # Critic health
    "critic/v_mean": mean_v,
    "critic/v_std": std_v,
    "critic/v_min": min_v,
    "critic/v_max": max_v,
    
    # Advantage health
    "advantage/mean": mean_adv,
    "advantage/std": std_adv,
    "advantage/fraction_negative": frac_negative,
    
    # Sample trajectories (every 50 steps)
    "samples/trajectory_html": format_trajectory(sample),
})
```

### Red Flags and Fixes

| Red Flag | Indicates | Fix |
|---|---|---|
| All advantages ≈ 0 | Critic head disconnected or V constant | Check critic gradient flow, verify head is attached |
| All advantages positive | Critic predicts same V for all states | Increase critic LR, verify diverse training states |
| Reward always 0 | Answer extraction bug | Test extract_answer on 10 manual examples |
| Tool calls = 0 | Model not generating code blocks | Strengthen system prompt, check model choice |
| Tool calls = max for every Q | Model stuck in code loop | Check termination detection, add max_segments limit |
| Loss NaN | Gradient explosion | Lower LR to 5e-7, enable grad clipping (max_norm=1.0) |
| V values all > 0.9 | Critic overfit or bug | Check critic training targets, verify R distribution has both 0s and 1s |
| EM drops after initial improvement | Reward hacking or catastrophic forgetting | Add KL penalty (1e-4), reduce LR |

### Correctness Verification Protocol

Before scaling to any larger experiment, manually verify on 5 trajectories:

```
For each of 5 randomly sampled training trajectories:
□ Generated text is coherent English
□ Code blocks are syntactically valid Python
□ Tool execution returns correct output (verify by hand)
□ Replacement produces readable context
□ Answer extraction returns the correct substring
□ Reward matches manual judgment (is the answer actually correct?)
□ V(s₀) < V(s₁) when tool output is relevant
□ V(s₀) > V(s₁) when tool output is irrelevant
□ Advantage signs match intuition (good call → positive, bad → negative)
```

---

## Paper Tables to Fill (Complete List)

**Table 1: Multi-Hop QA** (Main result — largest expected gains)

| Method | HotpotQA EM | HotpotQA F1 | 2Wiki EM | Musique EM | Avg EM |
|---|---|---|---|---|---|
| Qwen2.5-7B-Base (no tools) | _ | _ | _ | _ | _ |
| Search-R1 (repro, GRPO) | _ | _ | _ | _ | _ |
| **ToolSMDP (Ours)** | **_** | **_** | **_** | **_** | **_** |

**Table 2: Math Reasoning** (Competitiveness)

| Method | GSM8K Acc | MATH Acc |
|---|---|---|
| Qwen2.5-7B-Base (no tools) | _ | _ |
| ToRL (reported) | _ | _ |
| **ToolSMDP (Ours)** | **_** | **_** |

**Table 3: Multi-Tool Routing** (Novel)

| Method | FinQA EM | FinQA F1 |
|---|---|---|
| Qwen2.5-7B-Base (no tools) | _ | _ |
| Search-R1 (search only) | _ | _ |
| **ToolSMDP (Both tools)** | **_** | **_** |

**Table 4: Gains by Number of Tool Calls** (Key mechanistic result)

| Method | 1 tool call | 2 tool calls | 3+ tool calls |
|---|---|---|---|
| Search-R1 (GRPO) | _ | _ | _ |
| ToolSMDP + GRPO | _ | _ | _ |
| **ToolSMDP + PPO** | **_** | **_** | **_** |

**Table 5: Credit Assignment Ablation** (Core ablation)

| Variant | Episode | Credit | HotpotQA EM | Musique EM | GSM8K Acc |
|---|---|---|---|---|---|
| Search-R1 (repro) | Single | Trajectory GRPO | _ | _ | _ |
| ToolSMDP + GRPO | Segment | Trajectory GRPO | _ | _ | _ |
| **ToolSMDP + PPO** | **Segment** | **Segment V(s)** | **_** | **_** | **_** |

**Table 6: Reward Design Ablation**

| Reward | HotpotQA EM | GSM8K Acc |
|---|---|---|
| Outcome-only (default) | _ | _ |
| Outcome + exec penalty | _ | _ |

**Table 7: Value Function Ablation**

| Value Target | HotpotQA EM | GSM8K Acc |
|---|---|---|
| Bootstrap (default) | _ | _ |
| Monte Carlo | _ | _ |

**Figures:**
1. Training curves: EM over steps (3 methods)
2. Advantage distribution: relevant vs irrelevant tool outputs
3. V(s) calibration: scatter plot
4. Tool selectivity: tool frequency by question difficulty over training
5. Unnecessary tool call rate comparison across methods
