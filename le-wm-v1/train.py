import os, subprocess, hashlib
from datetime import datetime
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback


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

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm_hybrid")
def run(cfg):
    #########################
    ##       seed         ##
    #########################

    pl.seed_everything(cfg.seed, workers=True)

    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

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

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    with open_dict(cfg):
        cfg.repro = {
            "git_commit": subprocess.run(['git', 'rev-parse', 'HEAD'],
                                         capture_output=True, text=True).stdout.strip(),
            "data_hash": hashlib.sha256(dataset_name.encode()).hexdigest()[:16],
            "pytorch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda or "cpu",
            "cudnn_version": torch.backends.cudnn.version() or "none",
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        }

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    # ModelCheckpoint — spt.Manager sẽ redirect path, nhưng vẫn cần để nó qua
    from lightning.pytorch.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir,
        filename="epoch_{epoch:02d}",
        save_top_k=-1,
        every_n_epochs=1,
    )

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, run_dir=run_dir,
        epoch_interval=1, subdir=run_id,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback, checkpoint_callback],
        num_sanity_val_steps=1,
        logger=logger,
    )

    # Resume: tìm .ckpt trong run_dir hoặc spt cache
    ckpt_path = None
    resume_ckpt = run_dir / "resume.ckpt"
    if resume_ckpt.exists():
        ckpt_path = str(resume_ckpt)
    else:
        spt_runs = Path.home() / ".cache" / "stable-pretraining" / "runs"
        if spt_runs.exists():
            candidates = sorted(spt_runs.rglob("checkpoints/last.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                ckpt_path = str(candidates[0])

    if ckpt_path:
        print(f"📦 Resuming from {ckpt_path}")
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path,
    )

    manager()

    # Copy last.ckpt vào run_dir/resume.ckpt cho lần resume sau
    spt_runs = Path.home() / ".cache" / "stable-pretraining" / "runs"
    if spt_runs.exists():
        latest = sorted(spt_runs.rglob("checkpoints/last.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if latest:
            import shutil
            shutil.copy2(str(latest[0]), str(resume_ckpt))
            print(f"📦 Copied resume.ckpt → {resume_ckpt}")

    req_path = run_dir / f"requirements_{datetime.now():%Y%m%d}.txt"
    os.system(f"pip freeze > {req_path}")
    print(f"📋 Requirements saved: {req_path}")
    return


if __name__ == "__main__":
    run()
