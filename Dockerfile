FROM python:3.11-slim

# TODO: Switch to a CUDA base image for GPU training (e.g. nvidia/cuda:12.1.0-devel-ubuntu22.04)
# TODO: Pin dependency versions in pyproject.toml before training runs.

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install deps first (cache layer — only re-runs when pyproject.toml changes)
COPY pyproject.toml .
RUN pip install --no-cache-dir ".[dev]"

# Copy source and install package
COPY . .
RUN pip install --no-cache-dir -e .

RUN useradd -m sandbox
ENV SANDBOX_USER=sandbox

CMD ["pytest", "tests/", "-v"]
