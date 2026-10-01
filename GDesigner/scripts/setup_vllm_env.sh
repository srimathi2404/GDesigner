#!/usr/bin/env bash
# Creates the separate `vllm-serve` conda env and installs vLLM into it.
# Already run once during initial setup - kept for reproducibility.
#
# vLLM lives in its OWN env (not `gdesigner`) deliberately: vllm==0.6.3.post1
# hard-pins torch==2.4.0, which would fight the torch==2.3.0+cu121 build that
# GDesigner's torch_geometric C-extensions (torch_scatter/torch_sparse/
# torch_cluster) are compiled against in the `gdesigner` env. Since the
# server is just an HTTP process, there's no need for it to share a
# Python env with the GDesigner client code at all.
#
# transformers is pinned explicitly: vllm's own metadata only requires
# transformers>=4.45.2 with no upper bound, so an unpinned install picks the
# latest transformers, which now requires torch>=2.5 and silently disables
# itself against vllm's pinned torch==2.4.0 (breaks model loading). 4.46.3 is
# a version from the same release window as vllm 0.6.3.post1 and is known-
# compatible with torch 2.4.0.
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -q '^vllm-serve '; then
  conda create -n vllm-serve python=3.11 -y -c conda-forge --override-channels
fi

conda activate vllm-serve
pip install --upgrade pip
pip install "vllm==0.6.3.post1"
pip install "transformers==4.46.3"

python -c "
import torch, transformers, vllm
print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())
print('transformers', transformers.__version__)
print('vllm', vllm.__version__)
"
