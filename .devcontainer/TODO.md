# Dev Container TODOs

These changes should be applied to `devcontainer.json` when we reach the relevant milestones.

## Milestone 2+ (GPU inference/training)
- Add `"runArgs": ["--gpus", "all"]` for GPU forwarding (requires CUDA base image in Dockerfile)
- Add `WANDB_API_KEY` to `containerEnv` when training starts

## Data pipeline
- Add volume mount for HuggingFace cache to avoid re-downloading models:
  ```json
  "mounts": [
    "source=${localWorkspaceFolder},target=/workspace,type=bind",
    "source=hf-cache,target=/root/.cache/huggingface,type=volume"
  ]
  ```

## Search server
- Add port forwarding if we run a local BM25 search server:
  ```json
  "forwardPorts": [8000]
  ```
