"""
V2 All-in-One — Mamba-3+Attention hybrid predictor
Chạy trên RTX 5080 (Vast.ai) hoặc L40S

Usage:
  python v2_allinone.py [--config [tworoom|pusht|cube|reacher]] [--lambda 0.09]
"""

import os, sys, json, argparse, subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="tworoom", choices=["tworoom", "pusht", "cube", "reacher"])
parser.add_argument("--lambda", dest="sigreg_weight", type=float, default=0.09)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--dry-run", action="store_true", help="Only print commands, don't run")
args = parser.parse_args()

# === Config ===
TASK = args.config
SIGREG_WEIGHT = args.sigreg_weight
EPOCHS = args.epochs
BATCH = 128
LR = 5e-5
PRECISION = "bf16-mixed"  # RTX 5080 BF16 native
NUM_WORKERS = 6
T = 4  # history_size=3, num_preds=1

CFG_OVERRIDES = [
    f"data={TASK}",
    f"model=mamba3_hybrid",
    f"trainer.max_epochs={EPOCHS}",
    f"trainer.precision={PRECISION}",
    f"loader.batch_size={BATCH}",
    f"loader.num_workers={NUM_WORKERS}",
    f"loader.persistent_workers=True",
    f"loader.prefetch_factor=3",
    f"optimizer.lr={LR}",
    f"loss.sigreg.weight={SIGREG_WEIGHT}",
    f"seed=3072",
    f"subdir=v2_{TASK}_Mamba3",
]

# === Eval after training ===
EVAL_CMD = [
    "python", "eval.py",
    f"world.env_name=swm/{TASK.capitalize().replace('pusht','PushT').replace('tworoom','TwoRoom').replace('cube','OGBCube').replace('reacher','Reacher')}",
    f"eval.eval_budget={'150' if TASK=='tworoom' else '50'}",
    "seed=42",
    f"plan_config.horizon=5",
]

# === Commands ===
print("=" * 60)
print("V2 — Mamba-3+Attention Hybrid Predictor")
print(f"Task: {TASK}, λ={SIGREG_WEIGHT}, precision={PRECISION}")
print(f"Config: T={T}, batch={BATCH}, lr={LR}, workers={NUM_WORKERS}")
print("=" * 60)

def run(cmd, desc):
    print(f"\n>>> {desc}")
    print(f"    {' '.join(cmd)}")
    if not args.dry_run:
        subprocess.run(cmd, check=True)

# Step 1: Train
run(["python", "train.py", f"--config-name=lewm"] + CFG_OVERRIDES,
    f"Training V2 on {TASK}")

# Step 2: Eval
run(EVAL_CMD, f"Evaluating V2 on {TASK} (budget={'150' if TASK=='tworoom' else '50'})")

print("\n✅ Done!")
