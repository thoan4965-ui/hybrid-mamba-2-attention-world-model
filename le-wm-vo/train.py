import os
from functools import partial
from pathlib import Path

import hydra

# Khắc phục lỗi tương thích Python 3.14 với Hydra (argparse yêu cầu __contains__ cho LazyCompletionHelp)
import argparse
original_expand_help = argparse.HelpFormatter._expand_help
def patched_expand_help(self, action):
    if action.help is not None and not isinstance(action.help, str):
        try:
            action_help_str = str(action.help)
            if action_help_str.startswith("<") and action_help_str.endswith(">"):
                action_help_str = repr(action.help)
            original_help = action.help
            action.help = action_help_str
            res = original_expand_help(self, action)
            action.help = original_help
            return res
        except Exception:
            pass
    return original_expand_help(self, action)
argparse.HelpFormatter._expand_help = patched_expand_help

# Khắc phục lỗi thiếu tín hiệu Unix (SIGUSR1, SIGUSR2, SIGCONT) trên Windows
import sys
import signal
if sys.platform == "win32":
    class MockSignal(int):
        def __new__(cls, value, name):
            obj = super().__new__(cls, value)
            obj._name = name
            return obj
        @property
        def name(self):
            return self._name

    _mock_handlers = {}

    original_getsignal = signal.getsignal
    def patched_getsignal(sig):
        if isinstance(sig, MockSignal) or sig in (30, 31, 32):
            return _mock_handlers.get(int(sig), signal.SIG_DFL)
        try:
            return original_getsignal(sig)
        except ValueError:
            return signal.SIG_DFL
    signal.getsignal = patched_getsignal

    original_signal = signal.signal
    def patched_signal(sig, handler):
        if isinstance(sig, MockSignal) or sig in (30, 31, 32):
            _mock_handlers[int(sig)] = handler
            return signal.SIG_DFL
        try:
            return original_signal(sig, handler)
        except ValueError:
            return signal.SIG_DFL
    signal.signal = patched_signal

    for i, sig_name in enumerate(["SIGUSR1", "SIGUSR2", "SIGCONT"]):
        if not hasattr(signal, sig_name) or not hasattr(getattr(signal, sig_name), "name"):
            setattr(signal, sig_name, MockSignal(30 + i, sig_name))



import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback


# DEPRECATED: chỉ dùng cho ViT cũ, TinyViT không cần remap
def remap_checkpoint_keys(checkpoint_state_dict):
    remapped_sd = {}
    for k, v in checkpoint_state_dict.items():
        new_key = k
        if k.startswith("encoder."):
            sub_key = k[8:]
            if sub_key.startswith("encoder.layer."):
                parts = sub_key.split(".")
                layer_idx = parts[2]
                rest = ".".join(parts[3:])
                if rest == "attention.attention.query.weight":
                    new_key = f"encoder.layers.{layer_idx}.attention.q_proj.weight"
                elif rest == "attention.attention.query.bias":
                    new_key = f"encoder.layers.{layer_idx}.attention.q_proj.bias"
                elif rest == "attention.attention.key.weight":
                    new_key = f"encoder.layers.{layer_idx}.attention.k_proj.weight"
                elif rest == "attention.attention.key.bias":
                    new_key = f"encoder.layers.{layer_idx}.attention.k_proj.bias"
                elif rest == "attention.attention.value.weight":
                    new_key = f"encoder.layers.{layer_idx}.attention.v_proj.weight"
                elif rest == "attention.attention.value.bias":
                    new_key = f"encoder.layers.{layer_idx}.attention.v_proj.bias"
                elif rest == "attention.output.dense.weight":
                    new_key = f"encoder.layers.{layer_idx}.attention.o_proj.weight"
                elif rest == "attention.output.dense.bias":
                    new_key = f"encoder.layers.{layer_idx}.attention.o_proj.bias"
                elif rest == "intermediate.dense.weight":
                    new_key = f"encoder.layers.{layer_idx}.mlp.fc1.weight"
                elif rest == "intermediate.dense.bias":
                    new_key = f"encoder.layers.{layer_idx}.mlp.fc1.bias"
                elif rest == "output.dense.weight":
                    new_key = f"encoder.layers.{layer_idx}.mlp.fc2.weight"
                elif rest == "output.dense.bias":
                    new_key = f"encoder.layers.{layer_idx}.mlp.fc2.bias"
                else:
                    new_key = f"encoder.layers.{layer_idx}.{rest}"
            else:
                new_key = f"encoder.{sub_key}"
        remapped_sd[new_key] = v
    return remapped_sd

def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    
    # Định vị động tệp dataset H5 khi chạy trên Colab / môi trường khác
    if cache_dir:
        dataset_filename = Path(dataset_name).name
        candidate_paths = [
            Path(cache_dir) / dataset_filename,
            Path(cache_dir) / "datasets" / dataset_filename
        ]
        for candidate in candidate_paths:
            if candidate.exists():
                dataset_name = str(candidate)
                print(f"[INFO] Tự động định vị dataset tại: {dataset_name}")
                break

    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

    if cfg.get('augmentation', True):
        from torchvision.transforms import v2
        aug_transform = v2.Compose([
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            v2.RandomResizedCrop(cfg.img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            v2.RandomRotation(degrees=5),
        ])
        transforms.append(
            spt.data.transforms.WrapTorchTransform(
                aug_transform, source='pixels', target='pixels'
            )
        )

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")
        
        # Tối ưu cấu hình dataloader cho CPU local để tránh tràn RAM (8GB)
        if cfg.trainer.accelerator == "cpu":
            print("[INFO] Phát hiện huấn luyện trên CPU local. Đang tối ưu cấu hình Loader tránh tràn RAM (8GB)...")
            cfg.loader.num_workers = 0
            cfg.loader.persistent_workers = False
            cfg.loader.prefetch_factor = None

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    if cfg.get("pretrained_weights_path"):
        print("\n" + "="*80)
        print("🌟 CHẾ ĐỘ: TIẾN HÀNH FINE-TUNING WORLD MODEL (LLRD) ĐÃ ĐƯỢC KÍCH HOẠT 🌟")
        print(f"   Trọng số nền tảng: {cfg.pretrained_weights_path}")
        print("="*80 + "\n")
        optimizers = {
            'encoder_opt': {
                "modules": 'model.encoder',
                "optimizer": {"type": "AdamW", "lr": 5e-6, "weight_decay": 1e-3},
                "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
                "interval": "epoch",
            },
            'predictor_opt': {
                "modules": 'model.predictor|model.projector|model.pred_proj',
                "optimizer": {"type": "AdamW", "lr": 1e-5, "weight_decay": 1e-3},
                "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
                "interval": "epoch",
            },
            'action_opt': {
                "modules": 'model.action_encoder',
                "optimizer": {"type": "AdamW", "lr": 5e-5, "weight_decay": 1e-3},
                "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
                "interval": "epoch",
            },
        }
    else:
        print("\n" + "="*80)
        print("⚡ CHẾ ĐỘ: HUẤN LUYỆN MỚI TỪ ĐẦU (TRAIN FROM SCRATCH) ĐÃ ĐƯỢC KÍCH HOẠT ⚡")
        print("="*80 + "\n")
        optimizers = {
            'model_opt': {
                "modules": 'model',
                "optimizer": dict(cfg.optimizer),
                "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
                "interval": "epoch",
            },
        }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    # Nạp trọng số pretrained để tiến hành Fine-tuning / Thích ứng tại chỗ
    if cfg.get("pretrained_weights_path"):
        pretrained_path = Path(cfg.pretrained_weights_path)
        if pretrained_path.exists():
            print("\n" + "*"*80)
            print(f"🔥 ĐÃ NẠP THÀNH CÔNG TRỌNG SỐ PRETRAINED FINE-TUNE TỪ: {pretrained_path} 🔥")
            print("*"*80 + "\n")
            raw_sd = torch.load(pretrained_path, map_location="cpu")
            first_key = next(iter(raw_sd))
            clean_sd = {k[6:]: v for k, v in raw_sd.items()} if first_key.startswith("model.") else raw_sd
            
            is_resnet_weights = any(".backbone." in k for k in clean_sd.keys())
            if is_resnet_weights:
                remapped_sd = clean_sd
            else:
                remapped_sd = remap_checkpoint_keys(clean_sd)
                
            missing, unexpected = world_model.model.load_state_dict(remapped_sd, strict=False)
            if missing:
                # Lọc các key không ảnh hưởng (như mask_token)
                critical_missing = [k for k in missing if "mask_token" not in k]
                if critical_missing:
                    print(f"[INFO] Trọng số missing: {len(critical_missing)} keys (ví dụ: {critical_missing[:3]})")
            if unexpected:
                print(f"[INFO] Trọng số unexpected: {len(unexpected)} keys (ví dụ: {unexpected[:3]})")
        else:
            raise FileNotFoundError(
                f"❌ [LỖI] Không tìm thấy tệp trọng số pretrained tại: {pretrained_path}\n"
                f"   Hãy chắc chắn tệp tin tồn tại trên Google Drive hoặc ổ đĩa của bạn."
            )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, run_dir=run_dir, epoch_interval=10,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    
    # Đảm bảo các tín hiệu Unix giả lập không bị các thư viện khác đè thành int thường
    if sys.platform == "win32":
        for i, sig_name in enumerate(["SIGUSR1", "SIGUSR2", "SIGCONT"]):
            setattr(signal, sig_name, MockSignal(30 + i, sig_name))

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
