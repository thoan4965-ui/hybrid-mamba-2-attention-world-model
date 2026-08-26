# LeWM Data & Resources

## HuggingFace Datasets (public)
| Task | HF Repo | File |
|------|---------|------|
| TwoRoom | https://huggingface.co/datasets/quentinll/lewm-tworooms | tworoom.tar.zst (3.4 GB) |
| PushT | https://huggingface.co/datasets/quentinll/lewm-pusht | pusht_expert_train.h5.zst (13 GB) |
| Reacher | https://huggingface.co/datasets/quentinll/lewm-reacher | reacher.tar.zst (24 GB) |
| Cube | https://huggingface.co/datasets/quentinll/lewm-cube | cube_single_expert.tar.zst (46 GB) |

## Installation
```bash
pip install stable-pretraining stable-worldmodel huggingface_hub hydra-core einops imageio
pip install https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl --no-deps
pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/causal_conv1d-1.6.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

## Mamba wheels
- Mamba-2 v2.3.1: https://github.com/state-spaces/mamba/releases/tag/v2.3.1
- causal-conv1d v1.6.1.post4: https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.6.1.post4
