# ToolSMDP

Segment-level RL for LLM tool use via the Semi-Markov Decision Process framework. Decomposes LLM trajectories at tool-call boundaries for per-segment credit assignment using PPO.

## Project Structure

```
core/
  code_block_detector.py   # Detect ```python blocks: post-hoc (detect_code_block) + real-time (CodeBlockWatcher)
  replacement.py           # Replace code blocks with comments + stdout (code vanishes, output remains)
  reward.py                # Answer extraction (GSM8K ####, MATH \boxed, QA patterns, FinQA numeric) + exact match
  segment_rollout.py       # Segment and Trajectory dataclasses (training-loop bookkeeping)

sandbox/
  executor.py              # Execute Python in sandboxed subprocess (5s timeout, import blocklist, search() injection)

data/
  download_and_format.py   # Download 8 datasets from HuggingFace/GitHub → unified JSONL

scripts/
  test_base_model.py       # Inference script: code_gen | pipeline | eval modes (requires GPU)

tests/
  test_code_block_detector.py  # 17 tests: fence detection, watcher state machine, EOS handling
  test_executor.py             # 21 tests: sandboxing, blocked imports, timeouts, search injection
  test_replacement.py          # 10 tests: comment preservation, stdout insertion, edge cases
  test_reward.py               # 30 tests: answer extraction per dataset, normalization, exact match
```

## Quick Start

### Run Tests (Docker)
```bash
docker compose run test              # all 100 tests, 30s timeout
docker compose run dev               # interactive shell inside container
```

### Run Tests (bare metal)
```bash
pip install -e ".[dev]"
pytest tests/ -v
pytest tests/test_reward.py -v -k "TestExtractGSM8K"   # specific test class
```

### Download Data
```bash
# Via Docker
docker compose run download python -m data.download_and_format --datasets gsm8k hotpotqa finqa --max-samples 50

# Bare metal
python -m data.download_and_format --datasets gsm8k hotpotqa --splits dev --max-samples 50
```

### Run Inference (GPU required)
```bash
python -m scripts.test_base_model --mode code_gen       # does Qwen generate code blocks?
python -m scripts.test_base_model --mode pipeline       # full SMDP segment loop
python -m scripts.test_base_model --mode eval --data data_local/eval_splits/gsm8k_test_50.jsonl
python -m scripts.test_base_model --mode eval --data data_local/eval_splits/hotpotqa_dev_50.jsonl --num-rollouts 4
```

### Debug in VS Code
- **Dev Container**: Ctrl+Shift+P → "Reopen in Container" (requires Docker Desktop)
- **Debug Tests**: F5 → select "Debug Tests" or "Debug Current Test File" from launch configs

## Key Config

| Env var | Default | Purpose |
|---------|---------|---------|
| `DATA_ROOT` | `./data_local` | Dataset storage |
| `CHECKPOINT_ROOT` | `./checkpoints_local` | Model checkpoints |
| `SANDBOX_USER` | `sandbox` | Unprivileged user for code execution |
