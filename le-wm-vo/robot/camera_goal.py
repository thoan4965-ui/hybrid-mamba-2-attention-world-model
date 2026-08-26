"""
Camera capture + goal state management for bionic hand.
Saves: goal.png (for visual inspection) + goal.npy (latent vector).

Usage:
  python camera_goal.py goal_name           # capture & save as goal_name.png + .npy
  python camera_goal.py goal_name --model cfc --checkpoint path/v4.ckpt  # capture + auto-encode
"""
import cv2
import numpy as np
import argparse
import json
import torch
from pathlib import Path


IMG_SIZE = 96

# ─── Config ───────────────────────────────────────────────────────────
def _load_camera_config():
    """Load camera config, returns dict with defaults."""
    cfg_path = Path(__file__).parent.parent / "data/config/camera_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    return {}

CAM_CFG = _load_camera_config()
DEFAULT_CAM_ID = CAM_CFG.get("camera_idx", 1)
CROP_X = CAM_CFG.get("crop_x", 264)
CROP_Y = CAM_CFG.get("crop_y", 8)
CROP_SIZE = CAM_CFG.get("crop_size", 364)
IMG_SIZE = CAM_CFG.get("img_size", 96)
WARMUP = CAM_CFG.get("warmup_frames", 5)


# ─── Capture ──────────────────────────────────────────────────────────
_STABILIZED = False  # once-per-session flag

def capture(camera_id=None, stabilize=True):
    """Capture single frame from USB camera. Returns (H, W, C) float32 [0,1].
    
    Args:
        stabilize: If True, wait 20s on first call for camera sensor to stabilize.
                   Subsequent calls in same session skip the delay.
    """
    if camera_id is None:
        camera_id = DEFAULT_CAM_ID
    
    # Use DSHOW on Windows (required for most USB cameras)
    import os as _os
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW if _os.name == 'nt' else cv2.CAP_ANY)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_id)  # fallback
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}. Try --cam 0 or --cam 1.")
    
    global _STABILIZED
    if stabilize and not _STABILIZED:
        print(f"  Camera stabilizing 30s...")
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        import time; time.sleep(30)
        _STABILIZED = True
    
    # Set fixed exposure + white balance for consistent latent across sessions
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = off (DSHOW quirk)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)         # -6 ≈ 1/64s, tune if needed
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)           # disable auto white balance
    for _ in range(WARMUP):
        cap.read()
    
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to read frame. Check camera connection or try --cam 0.")
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Apply crop (same as training data collection)
    h, w = frame.shape[:2]
    if CROP_Y + CROP_SIZE <= h and CROP_X + CROP_SIZE <= w:
        frame = frame[CROP_Y:CROP_Y + CROP_SIZE, CROP_X:CROP_X + CROP_SIZE]
    
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    result = frame.astype(np.float32) / 255.0
    print(f"  Camera {camera_id}: {result.shape}, range [{result.min():.3f}, {result.max():.3f}]")
    return result


def read_image(path):
    """Read image from file. Returns (H, W, C) float32 [0,1]."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    frame = cv2.imread(str(path))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    return frame.astype(np.float32) / 255.0


def save_image(img, path):
    """Save image as PNG."""
    img_uint8 = (img * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img_bgr)


def save_goal(img, name, enc=None, device="cpu"):
    """Save goal: PNG for visual check + optional .npy latent.
    
    Args:
        img: (H, W, C) float32 [0,1]
        name: base filename (without extension)
        enc: encoder model (optional — if provided, saves latent too)
        device: cpu or cuda
    """
    base = Path(name)
    png_path = base.with_suffix(".png")
    npy_path = base.with_suffix(".npy")
    
    save_image(img, png_path)
    print(f"✓ Saved: {png_path}")
    
    if enc is not None:
        # Encode to latent and save
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            latent = enc(img_tensor).squeeze(0).cpu().numpy()
        np.save(npy_path, latent)
        print(f"✓ Saved: {npy_path} (latent norm={np.linalg.norm(latent):.3f})")
    
    return img


def load_goal(name, device="cpu"):
    """Load goal: prefer .npy latent, fallback to .png + encode.
    
    Args:
        name: base filename (without extension)
        device: cpu or cuda
    
    Returns:
        latent: (32,) numpy array on cpu, or None if only image exists
        img: (H, W, C) float32 numpy array
    """
    base = Path(name)
    npy_path = base.with_suffix(".npy")
    png_path = base.with_suffix(".png")
    
    latent = None
    if npy_path.exists():
        latent = np.load(npy_path)
    
    img = None
    if png_path.exists():
        img = read_image(png_path)
    
    return latent, img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture goal state for bionic hand")
    parser.add_argument("name", type=str, help="Goal name (saved as name.png + name.npy)")
    parser.add_argument("--cam", type=int, default=None, help=f"Camera ID (default: {DEFAULT_CAM_ID})")
    parser.add_argument("--model", type=str, default=None, choices=["cfc", "ar"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .ckpt for auto-encode")
    args = parser.parse_args()
    
    # Capture
    cam_id = args.cam if args.cam is not None else DEFAULT_CAM_ID
    img = capture(camera_id=cam_id)
    
    # Optionally encode
    enc = None
    if args.model and args.checkpoint:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "le-wm"))
        from module import TinyViT
        
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        sd = {}
        for k, v in ckpt["state_dict"].items():
            sd[k[6:] if k.startswith("model.") else k] = v
        enc = TinyViT(IMG_SIZE, 8, 4, 64, 4, 256, 32)
        enc.load_state_dict({k.replace("encoder.", ""): v for k, v in sd.items()
                             if k.startswith("encoder.")}, strict=True)
        enc.eval()
        print(f"Encoder loaded: {sum(p.numel() for p in enc.parameters()):,} params")
    
    save_goal(img, args.name, enc=enc)
    print(f"\nUsage: python robot_planner.py --goal {args.name} ...")
