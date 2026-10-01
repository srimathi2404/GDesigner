#!/usr/bin/env bash
# Step 2 (instructions.md) - serve Qwen2.5-7B-Instruct locally via vLLM.
# Run this in its own terminal/tmux pane and leave it running; run
# scripts/run_smoke_test.sh in a second pane once it's up.
#
# Usage: scripts/serve_qwen.sh [GPU_INDEX]
#   GPU_INDEX defaults to 0. Always re-check `nvidia-smi` yourself first and
#   pick whichever GPU currently has the most free memory / lowest util -
#   this is a shared node.
set -euo pipefail

GPU_INDEX="${1:-0}"

echo "=== nvidia-smi (re-check before trusting old numbers) ==="
nvidia-smi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm-serve

echo "=== Serving Qwen2.5-7B-Instruct on GPU ${GPU_INDEX} ==="
CUDA_VISIBLE_DEVICES="${GPU_INDEX}" vllm serve Qwen/Qwen2.5-7B-Instruct \
  --port 8000 \
  --served-model-name gpt-4o-mini \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192
