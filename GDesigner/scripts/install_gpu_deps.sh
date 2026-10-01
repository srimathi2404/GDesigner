#!/usr/bin/env bash
# Installs the CUDA-locked pieces of the `gdesigner` env (torch + the
# torch_geometric C-extensions). Already run once during initial setup -
# this script exists for reproducibility (e.g. a fresh clone of this repo).
#
# Note: pyg-lib is deliberately NOT installed - see gpu-requirements.txt for
# why (GLIBC_2.29 required, this host has 2.28).
#
# This does NOT require a free GPU: it only downloads and installs wheels.
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gdesigner

pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
  --index-url https://download.pytorch.org/whl/cu121

pip install torch_geometric==2.6.1

pip install torch_scatter torch_sparse torch_cluster \
  -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

python -c "
import torch, torch_geometric, torch_scatter, torch_sparse, torch_cluster
print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())
print('torch_geometric', torch_geometric.__version__)
print('GPU deps OK')
"
