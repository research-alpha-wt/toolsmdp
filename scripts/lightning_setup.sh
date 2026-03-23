#!/bin/bash
# Lightning.ai Studio setup + Step 2.1 smoke tests
#
# First run:  git clone https://github.com/research-alpha-wt/toolsmdp.git && cd toolsmdp && bash scripts/lightning_setup.sh
# Re-run:     cd toolsmdp && git pull && bash scripts/lightning_setup.sh
#
# Everything is idempotent — safe to re-run after preemption or code changes.
set -e

echo ""
echo "============================================================"
echo "  ToolSMDP — Lightning.ai Setup + Step 2.1 Smoke Tests"
echo "============================================================"
echo ""

# ── 1. Python dependencies ──
echo "=== [1/7] Installing Python dependencies ==="
pip install "transformers>=4.57,<5" accelerate
pip install -e ".[train,dev]"

# ── 2. Java for Pyserini ──
echo "=== [2/7] Installing Java 21 ==="
if ! java -version 2>&1 | grep -q "21"; then
    sudo apt update && sudo apt install -y openjdk-21-jdk
fi
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
grep -q JAVA_HOME ~/.bashrc || echo 'export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64' >> ~/.bashrc

# ── 3. Verify GPU ──
echo "=== [3/7] Verifying GPU ==="
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}, BF16: {torch.cuda.is_bf16_supported()}')"

# ── 4. Unit tests ──
echo "=== [4/7] Running unit tests ==="
pytest tests/ -v

# ── 5. Smoke test: code generation ──
echo "=== [5/7] Smoke test: code_gen ==="
python -m scripts.test_base_model --mode code_gen

# ── 6. Smoke test: pipeline (math, no search) ──
echo "=== [6/7] Smoke test: pipeline ==="
python -m scripts.test_base_model --mode pipeline

# ── 7. Eval on sample data ──
echo "=== [7/7] Eval on sample data ==="
echo "--- GSM8K (math, no search) ---"
python -m scripts.test_base_model --mode eval --data data_local/eval_splits/gsm8k_test_50.jsonl

echo "--- HotpotQA (search enabled, downloads Pyserini index on first run ~9GB) ---"
python -m scripts.test_base_model --mode eval --data data_local/eval_splits/hotpotqa_dev_50.jsonl

echo ""
echo "============================================================"
echo "  Step 2.1 complete. Check results above."
echo "============================================================"
