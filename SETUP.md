# Local Development Setup

## One-time setup

```
# 1. Create conda environment
conda create -n toolsmdp python=3.11 -y

# 2. Activate it
conda activate toolsmdp

# 3. Install everything (takes a few minutes — torch is ~2.5 GB)
cd c:\Users\abhijitkumar\Research\toolsmdp
pip install -e ".[train,dev]"

# 4. Verify
pytest tests/ -v
python scripts/try_search.py
```

## Every new terminal

```
conda activate toolsmdp
```

Java is auto-detected from PATH (installed via winget). If it's not found, set it manually:

```
:: Windows CMD
set JAVA_HOME=C:\Program Files\Microsoft\jdk-21.0.10.7-hotspot

:: Git Bash / WSL
export JAVA_HOME="/c/Program Files/Microsoft/jdk-21.0.10.7-hotspot"
```

## VS Code setup

1. `Ctrl+Shift+P` → "Python: Select Interpreter" → choose `toolsmdp`
2. Now F5, debugging, and test runner all use the right environment

## What's installed

| Package | Why |
|---|---|
| `torch` | Model inference/training |
| `transformers` | Qwen3.5-4B loading |
| `pyserini`, `faiss-cpu` | Wikipedia search (21M passages) |
| `datasets` | HuggingFace dataset download |
| `wandb` | Training metrics logging |
| `pytest` | Tests |

## Test search locally

Edit the `QUERIES` list in `scripts/try_search.py`, then run it:

```
conda activate toolsmdp
python scripts/try_search.py
```

First run downloads the 9.2 GB Wikipedia index (cached at `~/.cache/pyserini/`).

## Run tests

```
pytest tests/ -v                     # all tests
pytest tests/test_search.py -v       # search tests only
pytest tests/test_reward.py -v       # reward tests only
```

## Docker (optional, for clean-room testing)

```
docker compose run test              # runs pytest in container
docker compose run dev               # interactive shell
```
