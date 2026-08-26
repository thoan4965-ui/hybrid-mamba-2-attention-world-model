# ═══════════════════════════════════════════
# Colab V1: Hybrid TwoRoom — copy 1 cell → Run
# ═══════════════════════════════════════════
# Yêu cầu: Set secret HF_TOKEN trong Colab Secrets
# (https://huggingface.co/settings/tokens)

import os, time
os.environ["STABLEWM_HOME"] = "/content"
os.environ["LOCAL_DATASET_DIR"] = "/content"

# 1. Install
!pip install stable-worldmodel[train] ncps h5py hdf5plugin huggingface_hub hydra-core -q
!apt-get install zstd -qq

# 2. GPU check
import torch
print("GPU:", torch.cuda.get_device_name(0), "| VRAM:", round(torch.cuda.get_device_properties(0).total_memory/1e9,1), "GB")

# 3. Login HF + clone code
from huggingface_hub import login, HfApi
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
!git clone https://github.com/thoan4965-ui/hybrid-cfc-atention-WM.git /content/le-wm
print("Clone OK:", os.path.exists("/content/le-wm/train.py"))

# 4. Download TwoRoom data
if not os.path.exists("/content/tworoom.h5"):
    from huggingface_hub import snapshot_download
    snapshot_download("quentinll/lewm-tworooms", repo_type="dataset", local_dir="/content/tworoom_data")
    !tar --zstd -xvf /content/tworoom_data/tworoom.tar.zst -C /content/
    !rm -rf /content/tworoom_data

!mkdir -p /content/datasets
!ln -sf /content/tworoom.h5 /content/datasets/tworoom.h5
print("Data OK:", os.path.exists("/content/tworoom.h5"))

# 5. TRAIN (10 epochs, ~2-3h on T4)
start = time.time()
!cd /content/le-wm && python train.py --config-name=lewm_hybrid data=tworoom
print(f"⏱ Training time: {(time.time()-start)/60:.1f} phut")

# 6. Upload checkpoint
api = HfApi()
if os.path.exists("/content/checkpoints"):
    api.upload_folder(folder_path="/content/checkpoints",
        repo_id="hhian/checkpoints", repo_type="model",
        commit_message="Hybrid TwoRoom done")
    print("✅ https://huggingface.co/hhian/checkpoints")
else:
    print("⚠️ Checkpoints not found — training may have failed")

print("DONE")
