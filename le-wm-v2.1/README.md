# V2.1 — Hybrid Mamba-2+Attention

Code chính của dự án. Kiến trúc: 6×{Self-Attention(AdaLN) → Mamba-2}, heads=16, d_state=256, expand=4.

## Cài đặt

```bash
git clone https://github.com/thoan4965-ui/hybrid-mamba-atention-WM.git
cd hybrid-mamba-atention-WM/le-wm-v2.1

# Môi trường + torch
pip install uv
uv pip install --system torch==2.10.0+cu128 torchvision==0.25.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# Mamba-2 wheel
uv pip install --system https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/causal_conv1d-1.6.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
uv pip install --system https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# Dependencies
pip install stable-pretraining stable-worldmodel shapely hdf5plugin pymunk hydra-core huggingface_hub
```

## Checkpoints & Data

HuggingFace: [hhian/checkpoints](https://huggingface.co/hhian/checkpoints)

- `checkpoints/pusht/pusht/ep_10/` — Push-T (epoch 10, best checkpoint)
- `checkpoints/hybrid_mamba_tworoom/ep_10/` — TwoRoom Mamba (epoch 10, best checkpoint)
- `checkpoints/hybrid_cfc_tworoom/ep_10/` — TwoRoom CfC (epoch 10, best checkpoint)

## Eval Push-T

```bash
# Patch eval.py (bypass load_pretrained cho custom model)
sed -i "s|swm.wm.utils.load_pretrained(cfg.policy)|torch.load(cfg.policy, weights_only=False)|" eval.py

# Chạy eval
python eval.py --config-name=pusht policy=/content/datasets/lewm_pusht_model.pt seed=3072 ++eval.num_eval=50
```

## Eval TwoRoom

```bash
sed -i "s|swm.wm.utils.load_pretrained(cfg.policy)|torch.load(cfg.policy, weights_only=False)|" eval.py

python eval.py --config-name=tworoom policy=/content/datasets/hybrid_tworoom_model.pt seed=3072 ++eval.num_eval=50 ++eval.eval_budget=50 ++eval.goal_offset_steps=25 ++plan_config.horizon=5 ++plan_config.receding_horizon=5
```

## File cấu trúc

| File | Chức năng |
|---|---|
| `module.py` | Mamba2Predictor, Mamba2ConditionalBlock, Embedder, MLP |
| `jepa.py` | JEPA — encode + predict |
| `eval.py` | Eval với CEM planning |
| `train.py` | Training script |
| `download_data.py` | Download dataset từ HuggingFace |
| `utils.py` | Save/load checkpoint + HF upload |
| `config/eval/` | Hydra eval configs |
| `config/train/` | Hydra training configs |
