import numpy as np
import torch
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

    def __init__(self, run_name, cfg, run_dir=None, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval
        if run_dir is None:
            try:
                from stable_worldmodel.data.utils import get_cache_dir
                self.run_dir = get_cache_dir(sub_folder='checkpoints')
            except Exception:
                self.run_dir = "/content/dataset/checkpoints"
        else:
            self.run_dir = run_dir

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(trainer, pl_module, trainer.current_epoch + 1)

            elif (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(trainer, pl_module, trainer.current_epoch + 1)

    def _save(self, trainer, pl_module, epoch):
        from stable_worldmodel.wm.utils import save_pretrained
        import os
        import shutil
        from pathlib import Path

        # 1. Save raw weights (.pt)
        save_pretrained(
            pl_module.model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )

        # 2. Save full lightning checkpoint (.ckpt) to restore optimizer and scheduler state
        ckpt_filename = f"{self.run_name}_weights.ckpt"
        local_ckpt_path = Path(self.run_dir) / ckpt_filename
        
        try:
            trainer.save_checkpoint(str(local_ckpt_path))
            print(f"\n[SaveCkptCallback] Successfully saved full lightning checkpoint to {local_ckpt_path}")
        except Exception as e:
            print(f"\n[SaveCkptCallback Warning] Failed to save lightning checkpoint: {e}")

        # 3. Auto-backup to Google Drive if mounted (prevents loss on Colab disconnect)
        drive_base = "/content/drive/MyDrive/Bionic_Hand_LWM"
        if os.path.exists("/content/drive/MyDrive"):
            try:
                real_drive_path = os.path.realpath(drive_base)
                drive_backup_dir = os.path.join(real_drive_path, "checkpoints")
                os.makedirs(drive_backup_dir, exist_ok=True)
                
                # Backup the raw weights .pt
                local_ckpt_dir = os.path.join(os.environ.get("STABLEWM_HOME", "/content/dataset"), "checkpoints")
                src_pt_path = os.path.join(local_ckpt_dir, self.run_name, f"weights_epoch_{epoch}.pt")
                if os.path.exists(src_pt_path):
                    shutil.copy2(src_pt_path, drive_backup_dir)
                    print(f"[Auto-Backup] Successfully synced {os.path.basename(src_pt_path)} to Google Drive!")
                else:
                    print(f"[Auto-Backup Warning] Could not find saved checkpoint at {src_pt_path} to backup.")
                
                # Backup the full lightning checkpoint .ckpt
                if local_ckpt_path.exists():
                    shutil.copy2(str(local_ckpt_path), os.path.join(drive_backup_dir, ckpt_filename))
                    shutil.copy2(str(local_ckpt_path), os.path.join(drive_backup_dir, f"{self.run_name}_epoch_{epoch}.ckpt"))
                    print(f"[Auto-Backup] Successfully synced {ckpt_filename} and epoch_{epoch}.ckpt to Google Drive!")
            except Exception as e:
                print(f"[Auto-Backup Warning] Failed to backup to Drive: {e}")