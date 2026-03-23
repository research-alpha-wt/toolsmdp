#!/bin/bash
# Lightning.ai Studio setup script for ToolSMDP
# Usage: git clone <repo> && cd toolsmdp && bash scripts/lightning_setup.sh
set -e

echo "=== Installing Python dependencies ==="
pip install -e ".[train,dev]"

echo "=== Installing Java 21 for Pyserini ==="
sudo apt update && sudo apt install -y openjdk-21-jdk
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
grep -q JAVA_HOME ~/.bashrc || echo 'export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64' >> ~/.bashrc

echo "=== Verifying GPU ==="
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

echo "=== Running tests ==="
pytest tests/ -v

echo "=== Setup complete ==="
