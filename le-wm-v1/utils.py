import os
import numpy as np
import torch
from pathlib import Path
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)

class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, run_dir, epoch_interval=1, subdir=None):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.run_dir = run_dir
        self.epoch_interval = epoch_interval
        self.subdir = subdir or run_name

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        from stable_worldmodel.wm.utils import save_pretrained
        # save_pretrained lưu xuống {STABLEWM_HOME}/checkpoints/{run_name}/
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )
        # Copy .pt vào run_dir/ để gom chung với .ckpt
        import shutil
        cache_dir = os.environ.get("STABLEWM_HOME", "/root/.cache/stable-pretraining")
        src = Path(f"{cache_dir}/checkpoints/{self.run_name}/weights_epoch_{epoch}.pt")
        dst = self.run_dir / f"weights_epoch_{epoch}.pt"
        if src.exists():
            shutil.copy2(str(src), str(dst))
        self._upload_to_hf(epoch)

    def _upload_to_hf(self, epoch):
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            return
        from huggingface_hub import HfApi
        api = HfApi()

        base = f"checkpoints/{self.subdir}/{self.run_name}/ep_{epoch}"

        # Upload .pt từ run_dir (đã copy từ save_pretrained)
        pt_path = self.run_dir / f"weights_epoch_{epoch}.pt"
        if pt_path.exists():
            api.upload_file(path_or_fileobj=str(pt_path), path_in_repo=f"{base}/weights_epoch_{epoch}.pt", repo_id="hhian/checkpoints", repo_type="model")
            print(f"✅ Uploaded .pt ep{epoch}")

        # Upload .ckpt — cùng thư mục run_dir/
        for ckpt in sorted(self.run_dir.glob("*.ckpt"), reverse=True):
            api.upload_file(path_or_fileobj=str(ckpt), path_in_repo=f"{base}/weights.ckpt", repo_id="hhian/checkpoints", repo_type="model")
            print(f"✅ Uploaded .ckpt ep{epoch}")
            break