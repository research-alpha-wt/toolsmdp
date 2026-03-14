FROM python:3.11-slim

# TODO: Switch to a CUDA base image for GPU training (e.g. nvidia/cuda:12.1.0-devel-ubuntu22.04)
#       Python 3.11-slim is fine for unit tests and data processing but won't work for
#       model inference or training. When ready for Milestone 2+:
#       FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
#       RUN apt-get update && apt-get install -y python3.11 python3-pip ...

# TODO: Pin dependency versions in pyproject.toml before training runs.
#       Floating versions are fine for dev but training reproducibility needs pinned deps.
#       Run `pip freeze > requirements.lock` inside the container and commit.

# TODO: Add a .dockerignore to exclude .git, __pycache__, data_local/, checkpoints_local/,
#       *.md (except CLAUDE.md), and other large/unnecessary files from the build context.

WORKDIR /workspace

# System deps for subprocess sandboxing
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# TODO: Add numpy as a system-level dependency here if the sandbox executor needs it
#       for whitelisted imports inside subprocess. Currently numpy is in pyproject.toml
#       but the subprocess spawned by executor.py inherits the same Python env, so it
#       should work. Verify during container testing.

# Copy project definition first (cache layer for deps)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

# Copy source
COPY . .
RUN pip install --no-cache-dir -e .

# TODO: Add a non-root user for sandboxed code execution.
#       Currently the executor subprocess runs as root inside the container.
#       For better isolation:
#       RUN useradd -m sandbox
#       Then run executor subprocesses as that user via subprocess.run(..., user="sandbox")

# Default: run tests
CMD ["pytest", "tests/", "-v"]
