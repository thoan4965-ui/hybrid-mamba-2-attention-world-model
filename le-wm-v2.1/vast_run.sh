#!/bin/bash
# V2.1 Mamba-2 All-in-One — Vast RTX 5080
# Usage: bash vast_run.sh [task=tworoom] [epochs=100]
# Default: tworoom, 100 epochs

set -euo pipefail
TASK=${1:-tworoom}
EPOCHS=${2:-10}
LR=5e-5
BATCH=128
SEED=3072

echo "=== V2.1 Mamba-2: Install ==="
pip install stable-pretraining stable-worldmodel huggingface_hub hydra-core einops hdf5plugin pygame pymunk -q

echo "=== Install Mamba-2 + causal-conv1d wheel ==="
pip install https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl --no-deps -q
pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/causal_conv1d-1.6.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl -q
pip install einops -q

echo "=== Verify ==="
python -c "from mamba_ssm import Mamba2; print('Mamba-2 OK')"

echo "=== Train $TASK (${EPOCHS} epochs) ==="
python train.py model=mamba2_hybrid data=$TASK \
    trainer.max_epochs=$EPOCHS optimizer.lr=$LR \
    loader.batch_size=$BATCH seed=$SEED

echo "=== Eval $TASK ==="
CKPT=$(python -c "
import os
from pathlib import Path
import stable_worldmodel as swm
d = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'))
c = sorted(d.rglob('*_epoch_*.pt'), key=os.path.getmtime)
print(str(c[-1].parent) if c else str(d))
")
python eval.py task=$TASK policy=$CKPT

echo "=== Done ==="
echo "Results saved in: $CKPT"
