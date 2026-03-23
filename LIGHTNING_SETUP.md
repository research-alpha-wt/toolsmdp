# Lightning.ai Studio Setup — Step 2.1

## Why no container?

Lightning.ai Studios are already isolated Linux VMs with dedicated GPUs.
Docker-in-Docker adds overhead and complexity for zero benefit here.
The container setup exists for local dev (isolate from Windows).
On Lightning, `pip install -e .` directly is simpler and gives direct GPU access.

## Connect

```bash
ssh s_01kme7kcjmrxjkpchmqf8kd09f@ssh.lightning.ai
```

## Setup (run these in order)

```bash
# 1. Clone repo
git clone https://github.com/research-alpha-wt/toolsmdp.git && cd toolsmdp

# 2. Install Python dependencies (train extras include torch, transformers; dev includes pytest)
pip install -e ".[train,dev]"

# 3. Install Java 21 for Pyserini
sudo apt update && sudo apt install -y openjdk-21-jdk
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
echo 'export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64' >> ~/.bashrc

# 4. Verify GPU access
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

# 5. Run tests (should be 138 passed, 2 skipped)
pytest tests/ -v
```

## Step 2.1 Smoke Tests

```bash
# 6. Smoke test — does Qwen3.5-4B generate code blocks? (~2 min first run for model download)
python -m scripts.test_base_model --mode code_gen

# 7. Full pipeline — invoke → assimilate → synthesize loop (~3 min, downloads Pyserini index on first run)
python -m scripts.test_base_model --mode pipeline

# 8. Eval on sample data (50 questions, ~10-15 min)
python -m scripts.test_base_model --mode eval --data data_local/eval_splits/gsm8k_test_50.jsonl
python -m scripts.test_base_model --mode eval --data data_local/eval_splits/hotpotqa_dev_50.jsonl
```

## What to check

- Step 6: Code block generation rate >30%
- Step 7: Does the model write `<context>...</context>` blocks? (even inconsistently is fine)
- Step 8: EM accuracy numbers + tool call counts per dataset

## Disk usage

- Model (Qwen3.5-4B): ~8 GB (cached at `~/.cache/huggingface/`)
- Pyserini index (wikipedia-dpr-100w): ~9.2 GB (cached at `~/.cache/pyserini/`)
- Total: ~18 GB first run, cached after that

## If preempted (interruptible instance)

Just reconnect via SSH and re-run from where you left off.
Model and index are cached — no re-download needed.
