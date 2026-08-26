# ═══════════════════════════════════════════
# Kaggle V1: Hybrid TwoRoom — copy 1 cell → Run
# ═══════════════════════════════════════════
# Set HF_TOKEN via Kaggle notebook secrets

import os
os.environ["STABLEWM_HOME"]="/kaggle/working"
os.environ["LOCAL_DATASET_DIR"]="/kaggle/working"
# os.environ["HF_TOKEN"] = "your_hf_token"

!pip install stable-worldmodel[train] ncps h5py hdf5plugin huggingface_hub hydra-core -q
!apt-get install zstd -qq

!git clone https://github.com/thoan4965-ui/hybrid-cfc-atention-WM.git /kaggle/working/le-wm

if not os.path.exists("/kaggle/working/tworoom.h5"):
    from huggingface_hub import snapshot_download
    snapshot_download("quentinll/lewm-tworooms", repo_type="dataset", local_dir="/kaggle/working/tworoom_data")
    !tar --zstd -xvf /kaggle/working/tworoom_data/tworoom.tar.zst -C /kaggle/working/
    !rm -rf /kaggle/working/tworoom_data

!mkdir -p /kaggle/working/datasets
!ln -sf /kaggle/working/tworoom.h5 /kaggle/working/datasets/tworoom.h5

!cd /kaggle/working/le-wm && python train.py --config-name=lewm_hybrid data=tworoom

from huggingface_hub import HfApi
api = HfApi()
if os.path.exists("/kaggle/working/checkpoints"):
    api.upload_folder(folder_path="/kaggle/working/checkpoints", repo_id="hhian/checkpoints", repo_type="model", commit_message="Hybrid TwoRoom done")
    print("✅ https://huggingface.co/hhian/checkpoints")
