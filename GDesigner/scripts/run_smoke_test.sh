#!/usr/bin/env bash
# Step 5 (instructions.md) - tiny end-to-end smoke test against the running
# vLLM server started by scripts/serve_qwen.sh (run that first, in another
# terminal, and wait for it to report "Uvicorn running on http://0.0.0.0:8000").
#
# Usage: scripts/run_smoke_test.sh [GPU_INDEX]
set -euo pipefail

GPU_INDEX="${1:-0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Checking vLLM server is reachable ==="
curl -sf http://localhost:8000/v1/models > /dev/null || {
  echo "ERROR: http://localhost:8000/v1/models is not responding." >&2
  echo "Start the server first: scripts/serve_qwen.sh" >&2
  exit 1
}
echo "vLLM server OK."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gdesigner
cd "${REPO_ROOT}"

echo "=== Running smoke test (small: 3 agents, 2 iterations, 25 eval questions) ==="
CUDA_VISIBLE_DEVICES="${GPU_INDEX}" python experiments/run_mmlu.py \
  --mode FullConnected \
  --batch_size 2 \
  --agent_nums 3 \
  --num_iterations 2 \
  --num_rounds 1 \
  --limit_questions 25 \
  --optimized_spatial \
  --llm_name gpt-4o-mini
