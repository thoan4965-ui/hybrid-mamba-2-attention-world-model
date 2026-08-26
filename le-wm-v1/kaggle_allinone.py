# ═══════════════════════════════════════════
# V1: Hybrid CfC+Attention — TwoRoom benchmark
# ALL-IN-ONE — Copy nguyên cell này vào Kaggle, Run
# ═══════════════════════════════════════════

import os, zipfile, subprocess, sys, time
os.environ["STABLEWM_HOME"] = "/kaggle/working"
os.environ["LOCAL_DATASET_DIR"] = "/kaggle/working"

print("1/7: Install deps")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
    "stable-worldmodel[train,env]", "ncps", "h5py", "hdf5plugin", "huggingface_hub"])

print("2/7: GPU check")
import torch
print(f"{torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

print("3/7: Login HF + clone LeWM")
from huggingface_hub import login; login(token="HF_TOKEN_REMOVED")
if not os.path.exists("/kaggle/working/le-wm/train.py"):
    subprocess.check_call(["git", "clone", "https://github.com/lucas-maes/le-wm.git", "/kaggle/working/le-wm"])

print("4/7: Extract Hybrid code")
if os.path.exists("/kaggle/working/v1_hybrid_full.zip"):
    with zipfile.ZipFile("/kaggle/working/v1_hybrid_full.zip") as z:
        z.extractall("/kaggle/working/le-wm/")

print("5/7: Download TwoRoom (3.4GB, ~5ph)")
if not os.path.exists("/kaggle/working/tworoom.h5"):
    from huggingface_hub import snapshot_download
    snapshot_download("quentinll/lewm-tworooms", repo_type="dataset", local_dir="/kaggle/working/tworoom_data")
    subprocess.check_call(["tar", "--zstd", "-xvf", "/kaggle/working/tworoom_data/tworoom.tar.zst", "-C", "/kaggle/working/"])

print("6/7: TRAIN (10 epochs, ~30-45ph)")
print("=" * 40)
# subprocess.run for live output
p = subprocess.Popen(["python", "train.py", "--config-name=lewm_hybrid", "data=tworoom"],
    cwd="/kaggle/working/le-wm", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True)
for line in p.stdout: print(line, end="")
p.wait()

print("7/7: Upload checkpoint → HF")
from huggingface_hub import HfApi
api = HfApi()
if os.path.exists("/kaggle/working/checkpoints"):
    api.upload_folder(folder_path="/kaggle/working/checkpoints",
        repo_id="hhian/checkpoints", repo_type="model",
        commit_message=f"Hybrid TwoRoom {time.strftime('%Y-%m-%d %H:%M')}")
print("Checkpoint: https://huggingface.co/hhian/checkpoints")
print("DONE 🎉")
