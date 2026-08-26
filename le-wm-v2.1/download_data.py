import os, shutil
from pathlib import Path
from huggingface_hub import snapshot_download

def ensure_dataset(name="tworoom", cache_dir=None):
    """Tải + giải nén dataset từ HF (nếu chưa có)."""
    cache_dir = cache_dir or os.environ.get("STABLEWM_HOME",
                                            str(Path.home() / ".stable_worldmodel"))
    datasets_dir = Path(cache_dir) / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    # map: task name -> (huggingface repo, archive file)
    repos = {
        "tworoom": ("quentinll/lewm-tworooms", "tworoom.tar.zst", "tworoom.h5"),
        "pusht":   ("quentinll/lewm-pusht",   "pusht_expert_train.h5.zst", "pusht_expert_train.h5"),
        "reacher": ("quentinll/lewm-reacher",  "reacher.tar.zst", "reacher.h5"),
        "cube":    ("quentinll/lewm-cube",     "cube_single_expert.tar.zst", "cube_single_expert.h5"),
    }
    if name not in repos:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(repos.keys())}")

    repo_id, archive, out_file = repos[name]
    dst = datasets_dir / out_file
    if dst.exists():
        print(f"✅ Data exists: {dst}")
        return

    print(f"⬇ Downloading {repo_id}...")
    tmp = Path(cache_dir) / f"tmp_{name}"
    if tmp.exists(): shutil.rmtree(tmp)

    snapshot_download(repo_id, repo_type="dataset", local_dir=str(tmp),
                      token=os.environ.get("HF_TOKEN"))

    # Giải nén .tar.zst hoặc .h5.zst
    archive_path = tmp / archive
    if archive.endswith(".tar.zst"):
        os.system(f"tar --zstd -xf {archive_path} -C {datasets_dir}")
    elif archive.endswith(".h5.zst"):
        os.system(f"zstd -d {archive_path} -o {dst}")
    else:
        shutil.copy2(str(archive_path), str(dst))

    shutil.rmtree(tmp)
    print(f"✅ Dataset ready: {dst}")

if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "tworoom"
    ensure_dataset(name)
