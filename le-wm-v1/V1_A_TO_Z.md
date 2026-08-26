# V1 — Hybrid CfC+Attention: TwoRoom benchmark

## A-Z Guide

### Trước khi bắt đầu

Tài khoản cần:
- **Kaggle** (GPU A100 free) → https://www.kaggle.com
- **HuggingFace** (lưu checkpoint) → https://huggingface.co

---

### Bước 1: Upload code lên Kaggle

1. Mở Kaggle Notebook: `https://www.kaggle.com → New Notebook → GPU Accelerator`
2. Upload file zip:
   ```
   Add File → chọn D:\ai_training\le-wm-v1\v1_hybrid_full.zip
   ```
3. Copy file `kaggle_notebook.py` ở bên dưới → paste vào notebook

### Bước 2: Copy-paste script này vào Kaggle notebook

```python
# ═══════════════════════════════════════════
# Cell 1: Install dependencies
# ═══════════════════════════════════════════
!pip install stable-worldmodel[train,env] ncps h5py hdf5plugin huggingface_hub --quiet

# ═══════════════════════════════════════════
# Cell 2: Check GPU (đợi A100)
# ═══════════════════════════════════════════
import torch
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
!nvidia-smi | grep "Tesla\|A100\|T4"

# ═══════════════════════════════════════════
# Cell 3: Login HuggingFace
# ═══════════════════════════════════════════
from huggingface_hub import login
login(token="HF_TOKEN_REMOVED")

# ═══════════════════════════════════════════
# Cell 4: Clone LeWM gốc + giải nén Hybrid code
# ═══════════════════════════════════════════
import os, zipfile
os.environ["STABLEWM_HOME"] = "/kaggle/working"
os.environ["LOCAL_DATASET_DIR"] = "/kaggle/working"

!git clone https://github.com/lucas-maes/le-wm.git /kaggle/working/le-wm

# Giải nén Hybrid module.py + configs zip
with zipfile.ZipFile("/kaggle/working/v1_hybrid_full.zip", "r") as z:
    z.extractall("/kaggle/working/le-wm/")
print("Hybrid code ready")

# ═══════════════════════════════════════════
# Cell 5: Download TwoRoom data (3.4GB, ~5 phút)
# ═══════════════════════════════════════════
from huggingface_hub import snapshot_download
snapshot_download("quentinll/lewm-tworooms", repo_type="dataset",
                  local_dir="/kaggle/working/tworoom_data")
!tar --zstd -xvf /kaggle/working/tworoom_data/tworoom.tar.zst -C /kaggle/working/
print("Data ready")

# ═══════════════════════════════════════════
# Cell 6: TRAIN HYBRID (10 epochs, ~30-45 phút)
# ═══════════════════════════════════════════
!cd /kaggle/working/le-wm && python train.py --config-name=lewm_hybrid data=tworoom

# ═══════════════════════════════════════════
# Cell 7: Upload checkpoint lên HuggingFace
# ═══════════════════════════════════════════
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="/kaggle/working/checkpoints",
    repo_id="hhian/checkpoints",
    repo_type="model",
    commit_message="Hybrid TwoRoom training complete"
)
print("Uploaded to HF: https://huggingface.co/hhian/checkpoints")

# ═══════════════════════════════════════════
# Cell 8 (tùy chọn): Eval AR baseline để so sánh
# ═══════════════════════════════════════════
# !cd /kaggle/working/le-wm && python eval.py --config-name=tworoom policy=tworoom/lewm
```

### Bước 3: Chạy

Run từng cell từ trên xuống. Mỗi cell đợi cell trước xong.

### Bước 4: Xem kết quả

Sau khi chạy xong, checkpoint ở:
```
https://huggingface.co/hhian/checkpoints
```

### Troubleshooting

| Vấn đề | Fix |
|---|---|
| `ModuleNotFoundError: ncps` | Chạy lại Cell 1 |
| `CUDA out of memory` | Giảm `batch_size` trong `lewm_hybrid.yaml` xuống 64 |
| `File tworoom.h5 not found` | Cell 5 chưa xong — đợi giải nén |
| Kaggle session die | Chạy lại từ Cell 5 — checkpoint tự resume |
