"""
Robot planner for bionic hand 8-DOF grasp.
CfC + AR predictor planning via CEM, with MPC loop for real-time execution.

Usage:
  python robot_planner.py --model cfc --checkpoint path/to/v4.ckpt --goal path/to/goal.png
  python robot_planner.py --model ar  --checkpoint path/to/ar.ckpt  --goal path/to/goal.png
  python robot_planner.py --model cfc --checkpoint path/to/v4.ckpt --goal path/to/goal.png --source file --data h5_path
"""
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import h5py
import hdf5plugin

from pathlib import Path

# LeWM modules
import sys; sys.path.insert(0, str(Path(__file__).parent.parent / "le-wm-vo"))
from module import TinyViT, CfCPredictorV2, ARPredictor, Embedder
from camera_goal import read_image, load_goal

# ─── CONFIG ──────────────────────────────────────────────────────────
HIST_SIZE = 3
PRED_SIZE = 3
SKIP = 3
SEQ = HIST_SIZE + PRED_SIZE
IMG_SIZE = 96
LATENT_DIM = 32

SERVO_IDS = [1, 2, 4, 5, 6, 7, 8, 9]

# CEM config
CEM_ITERATIONS = 5
CEM_SAMPLES = 100
CEM_TOP_K = 15
CEM_HORIZON = 3           # plan 3 steps ahead
MPC_EXECUTE_STEPS = 3     # execute 3 actions before replanning


# ─── MODEL LOADING ───────────────────────────────────────────────────
def load_model(ckpt_path, model_type, device="cpu"):
    """Load encoder + predictor + action_encoder from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["state_dict"]
    cleaned = {}
    for k, v in sd.items():
        nk = k[6:] if k.startswith("model.") else k
        cleaned[nk] = v

    # Encoder (shared)
    enc = TinyViT(IMG_SIZE, patch_size=8, num_layers=4, hidden_dim=64,
                  num_heads=4, mlp_dim=256, output_dim=LATENT_DIM).to(device)
    enc.load_state_dict({k.replace("encoder.", ""): v for k, v in cleaned.items()
                         if k.startswith("encoder.")}, strict=True)
    enc.eval()

    # Predictor
    if model_type == "ar":
        pred = ARPredictor(num_frames=3, depth=1, heads=2, mlp_dim=96,
                           input_dim=32, hidden_dim=64, output_dim=32, dim_head=16).to(device)
        ae_dim = 24
    else:  # cfc
        pred = CfCPredictorV2(num_frames=6, input_dim=32, hidden_dim=96, output_dim=32,
                              action_dim=32, backbone_layers=1, backbone_units=96).to(device)
        ae_dim = 8
    pred.load_state_dict({k.replace("predictor.", ""): v for k, v in cleaned.items()
                          if k.startswith("predictor.")}, strict=False)
    pred.eval()

    # Action encoder
    has_norm = any("action_encoder.norm." in k for k in cleaned)
    aenc = Embedder(input_dim=ae_dim, smoothed_dim=8, emb_dim=32,
                    mlp_scale=2, use_norm=has_norm).to(device)
    aenc.load_state_dict({k.replace("action_encoder.", ""): v for k, v in cleaned.items()
                          if k.startswith("action_encoder.")}, strict=True)
    aenc.eval()

    return enc, pred, aenc, has_norm


# ─── ROLLOUT ─────────────────────────────────────────────────────────
@torch.no_grad()
def rollout_cfc(enc, pred, aenc, start_img, actions, context_emb=None, context_aemb=None):
    """CfC step-by-step rollout of H-step action sequence."""
    device = next(pred.parameters()).device
    H = actions.shape[0]

    with torch.no_grad():
        emb = enc(start_img.to(device))

    h = None
    if context_emb is not None and context_aemb is not None:
        for t in range(2):
            _, h = pred.step(context_emb[:, t:t+1], context_aemb[:, t:t+1], h)

    lp = emb.unsqueeze(1)

    for t in range(H):
        a = aenc(actions[t:t+1].unsqueeze(0))
        out, h = pred.step(lp, a, h)
        lp = out

    return lp.squeeze(1)


@torch.no_grad()
def rollout_ar(enc, pred, aenc, start_img, actions, context_emb=None, context_aemb=None):
    """AR sliding window rollout of H-step action sequence.
    
    Args:
        enc, pred, aenc: model components
        start_img: (1, 3, H, W) — starting image
        actions: (H, 8) — raw servo actions
        context_emb: (1, 3, 32) — optional 3-frame context embeddings
        context_aemb: (1, 3, 32) — optional 3-frame context action embeddings
    
    Returns:
        latent_end: (1, 32)
    """
    device = next(pred.parameters()).device
    H = actions.shape[0]

    # Encode start image
    with torch.no_grad():
        emb = enc(start_img.to(device))  # (1, 32)

    # Use provided context or repeat start image
    if context_emb is not None:
        ctx_emb = context_emb.to(device)
    else:
        ctx_emb = emb.unsqueeze(0).repeat(1, 3, 1)  # (1, 3, 32)

    # Build context actions
    if context_aemb is not None:
        ctx_aemb = context_aemb.to(device)
    else:
        # Fake 24-dim from repeated action
        first_8 = actions[0:1]
        first_24 = torch.cat([first_8] * 3, dim=-1)
        ctx_act_raw = first_24.unsqueeze(0).repeat(1, 3, 1)
        with torch.no_grad():
            ctx_aemb = aenc(ctx_act_raw.to(device))

    for t in range(H):
        preds = pred(ctx_emb, ctx_aemb)  # (1, 3, 32)
        next_p = preds[:, -1:]  # (1, 1, 32)
        ctx_emb = torch.cat([ctx_emb[:, 1:], next_p], dim=1)

        if t < H - 1:
            next_8 = actions[t+1:t+2]
            next_24 = torch.cat([next_8] * 3, dim=-1)
            next_act_raw = next_24.unsqueeze(1)
            with torch.no_grad():
                next_act_emb = aenc(next_act_raw.to(device))
            ctx_aemb = torch.cat([ctx_aemb[:, 1:], next_act_emb], dim=1)

    return ctx_emb[:, -1:, :].squeeze(1)  # (1, 32)


def build_history_from_h5(enc, pred, aenc, model_type, h5_path, h5_idx, device="cpu"):
    """Build initial embedding + hidden state from H5 data at given index.
    Used for file-based testing without camera.
    """
    with h5py.File(h5_path, 'r') as f:
        pix = f['pixels']
        act = f['action']

        # Get 3 history frames
        px = torch.from_numpy(pix[h5_idx:h5_idx+3]).float().to(device) / 255.0
        px = px.permute(0, 3, 1, 2)  # (3, C, H, W)

        with torch.no_grad():
            emb = enc(px).unsqueeze(0)  # (1, 3, 32)

            if model_type == "ar":
                # AR: need 24-dim actions
                ac = []
                for k in range(3):
                    stacked = torch.from_numpy(act[h5_idx+k:h5_idx+k+3]).float().reshape(-1)
                    ac.append(stacked)
                ac = torch.stack(ac).unsqueeze(0).to(device)  # (1, 3, 24)
                aemb = aenc(ac)
                return emb, aemb, None
            else:
                # CfC: need to build hidden state
                ac = torch.from_numpy(act[h5_idx:h5_idx+3]).float().unsqueeze(0).to(device)
                aemb = aenc(ac)
                h = None
                for t in range(HIST_SIZE - 1):
                    _, h = pred.step(emb[:, t:t+1], aemb[:, t:t+1], h)
                return emb, aemb, h


# ─── CEM PLANNER ─────────────────────────────────────────────────────
def cem_planner(enc, pred, aenc, model_type, latent_goal, start_img=None,
                start_latent=None, context_emb=None, context_aemb=None,
                current_positions=None,
                n_iter=CEM_ITERATIONS, n_samples=CEM_SAMPLES,
                top_k=CEM_TOP_K, horizon=CEM_HORIZON, action_dim=8, device="cpu"):
    """CEM planner: find action sequence to minimize ||latent_end - latent_goal||².
    
    Args:
        current_positions: (action_dim,) array of current servo values.
                           If provided, CEM starts exploring from here.
    """

    # Load per-servo limits from calibration (neutral & grasp define safe range)
    import json
    base = Path(__file__).parent.parent
    servo_min = np.full(action_dim, np.nan, dtype=np.float32)
    servo_max = np.full(action_dim, np.nan, dtype=np.float32)
    for calib_name in ["calib_neutral.json", "calib_grasp.json"]:
        calib_path = base / "data/calib" / calib_name
        if calib_path.exists():
            with open(calib_path) as f:
                data = json.load(f)
            pos_key = [k for k in data if "_pos" in k][0]
            for i, sid in enumerate(SERVO_IDS):
                val = data[pos_key].get(str(sid))
                if val is not None:
                    servo_min[i] = min(servo_min[i], val) if not np.isnan(servo_min[i]) else val
                    servo_max[i] = max(servo_max[i], val) if not np.isnan(servo_max[i]) else val
    
    # Fallback: servos without calib data use [val-100, val+100] or [51, 1018]
    for i in range(action_dim):
        if np.isnan(servo_min[i]) or np.isnan(servo_max[i]):
            servo_min[i] = 51; servo_max[i] = 1018
        elif servo_max[i] == servo_min[i]:
            servo_max[i] = min(1018, servo_min[i] + 50)
            servo_min[i] = max(51, servo_min[i] - 50)

    action_mean = (servo_min + servo_max) / 2
    action_std = (servo_max - servo_min) / 4

    # Load grasp values as CEM target direction (steer toward grasp, not midpoint)
    grasp_path = base / "data/calib/calib_grasp.json"
    grasp_values = action_mean.copy()  # fallback
    if grasp_path.exists():
        with open(grasp_path) as f:
            gdata = json.load(f)
        gpos = gdata.get("grasp_pos", gdata)
        grasp_values = np.array([float(gpos.get(str(sid), 500)) for sid in SERVO_IDS], dtype=np.float32)

    mean = np.full((horizon, action_dim), action_mean, dtype=np.float32)
    std = np.full((horizon, action_dim), action_std, dtype=np.float32)

    # Anchor to current position → steer toward GRASP (not midpoint)
    if current_positions is not None:
        mean[0] = current_positions
        for t in range(1, horizon):
            alpha = t / (horizon - 1)
            mean[t] = current_positions * (1 - alpha) + grasp_values * alpha
        std *= 0.7

    rollout_fn = rollout_cfc if model_type == "cfc" else rollout_ar

    for it in range(n_iter):
        samples = np.random.randn(n_samples, horizon, action_dim) * std + mean
        samples = np.clip(samples, servo_min, servo_max)

        costs = np.zeros(n_samples)
        for i in range(n_samples):
            actions = torch.from_numpy(samples[i]).float()
            latent_end = rollout_fn(enc, pred, aenc, start_img, actions,
                                    context_emb=context_emb, context_aemb=context_aemb)
            lg = latent_goal.to(device)
            if lg.dim() == 1: lg = lg.unsqueeze(0)
            if latent_end.dim() == 1: latent_end = latent_end.unsqueeze(0)
            cost = F.mse_loss(latent_end, lg).item()
            costs[i] = cost

        elite_idx = np.argsort(costs)[:top_k]
        elites = samples[elite_idx]
        mean = elites.mean(axis=0)
        std = elites.std(axis=0) + 1e-6

        min_idx = np.argmin(costs)
        if it == 0:
            best_cost = costs[min_idx]
            best_actions = samples[min_idx]
        elif costs[min_idx] < best_cost:
            best_cost = costs[min_idx]
            best_actions = samples[min_idx]

        if it % 2 == 0 or it == n_iter - 1:
            print(f"  CEM iter {it+1}/{n_iter}: cost={costs[elite_idx[0]]:.6f} (best={best_cost:.6f})")

    return best_actions, best_cost


# ─── DRY-RUN TEST ────────────────────────────────────────────────────
def dryrun_test(enc, pred, aenc, model_type, latent_goal, h5_path, device="cpu"):
    """Single CEM test with proper multi-frame context from H5."""
    print(f"\n{'='*60}")
    print(f"DRY-RUN — {model_type.upper()}")
    print(f"{'='*60}")

    # Pick random 3-frame context from H5 (ensures fair comparison)
    import h5py, hdf5plugin
    with h5py.File(h5_path, 'r') as f:
        N = f['pixels'].shape[0]
        idx = np.random.randint(0, max(0, N - (3 + CEM_HORIZON) * SKIP))
        
        # Get 3 context frames + actions
        px = [f['pixels'][idx + k * SKIP][:] for k in range(3)]
        ac_raw = [f['action'][idx + k * SKIP][:] for k in range(3)]
        start_img_np = f['pixels'][idx + 3 * SKIP][:]
        
        # AR 24-dim context actions (need 6 extra raw frames before context start)
        ar_actions_24 = []
        # We need 3 positions, each position = 3 consecutive raw actions
        # Position k uses actions at raw indices: idx + k*SKIP, +1, +2
        for k in range(3):
            base = idx + k * SKIP
            stack = [f['action'][base + j][:] for j in range(3)]  # 3 raw frames per position
            ar_actions_24.append(np.concatenate(stack))

    # Build tensors
    context_frames = torch.from_numpy(np.stack(px)).permute(0, 3, 1, 2).float().to(device) / 255.0
    start_img = torch.from_numpy(start_img_np).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    # Encode
    with torch.no_grad():
        context_emb = enc(context_frames).unsqueeze(0)  # (1, 3, 32)
        start_latent = enc(start_img)  # (1, 32)

    # Build proper initial state for each model
    if model_type == "cfc":
        context_act = torch.from_numpy(np.stack(ac_raw)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            context_aemb = aenc(context_act)
    else:
        # AR: build 24-dim context actions (proper 3-frame stacks)
        context_act = torch.from_numpy(np.stack(ar_actions_24)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            context_aemb = aenc(context_act)

    # Cost before planning
    lg = latent_goal.to(device)
    if lg.dim() == 1: lg = lg.unsqueeze(0)
    if start_latent.dim() == 1: start_latent = start_latent.unsqueeze(0)
    cost_before = F.mse_loss(start_latent, lg).item()
    print(f"  Start→Goal before plan: {cost_before:.6f}")

    # CEM plan (passes context for fair AR comparison)
    actions, cost = cem_planner(enc, pred, aenc, model_type, latent_goal,
                                start_img=start_img,
                                context_emb=context_emb, context_aemb=context_aemb,
                                device=device)
    print(f"\n  Start→Goal after plan:  {cost:.6f}")
    if cost_before > 0:
        improvement = (1 - cost / cost_before) * 100
        print(f"  Improvement:            {improvement:.1f}%")

    return {"cost_before": cost_before, "cost_after": cost}


# ─── MAIN LOOP ───────────────────────────────────────────────────────
def robot_loop(enc, pred, aenc, model_type, latent_goal, device="cpu",
               source="camera", h5_path=None, h5_idx=None, serial_port=None,
               live_goal=False, task_type="grasp"):
    """MPC loop: camera → CEM plan → execute action → repeat."""
    print(f"\n{'='*60}")
    print(f"ROBOT LOOP — {model_type.upper()}")
    print(f"{'='*60}")

    if serial_port:
        import serial_servo
        pH, pkt = serial_servo.connect(port=serial_port)
        print(f"  Serial: {serial_port} connected")
    else:
        pH = pkt = None
        print("  Serial: disabled (test mode)")
    
    latent_goal = latent_goal.to(device)
    MAX_STEPS = 30
    cost_history = []
    # Load per-servo thresholds from file
    load_config_path = Path(__file__).parent / "load_threshold.json"
    if load_config_path.exists():
        with open(load_config_path) as f:
            cfg = json.load(f)
        per_servo_threshold = cfg["thresholds"]
        wait_first = cfg.get("wait_first", 1.5)
        wait_confirm = cfg.get("wait_confirm", 0.5)
        min_blocked = cfg.get("min_blocked", 2)
    else:
        per_servo_threshold = {str(sid): 500 for sid in SERVO_IDS}
        wait_first = 1.5
        wait_confirm = 0.5
        min_blocked = 1

    # Load neutral position as starting point for CEM
    base = Path(__file__).parent.parent
    neutral_path = base / "data/calib/calib_neutral.json"
    if neutral_path.exists():
        with open(neutral_path) as f:
            data = json.load(f)
        current_pos = np.array([data["neutral_pos"].get(str(sid), 500) 
                                for sid in SERVO_IDS], dtype=np.float32)
    else:
        current_pos = np.full(len(SERVO_IDS), 500, dtype=np.float32)

    # Context history: always maintain 3 frames (replicate step 0 if needed)
    ctx_frames = []
    ctx_actions_list = []

    for step in range(MAX_STEPS):
        t0 = time.time()

        # Camera warmup stabilization handled by camera_goal.capture() (20s on first call)

        # 1. Capture current image
        if source == "camera":
            import camera_goal
            img_np = camera_goal.capture()
            start_img = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
        elif source == "file" and h5_path is not None:
            import h5py
            with h5py.File(h5_path, 'r') as f:
                img_np = f['pixels'][step * 3][:]
                start_img = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        else:
            raise ValueError("Need camera or file source")

        # Build context: always 3 frames (replicate first frame for step 0)
        ctx_frames.append(start_img)
        if len(ctx_frames) > 3:
            ctx_frames.pop(0)
        
        # Ensure we always have 3 context frames (pad with first frame)
        while len(ctx_frames) < 3:
            ctx_frames.insert(0, start_img)
        
        # Encode 3 context frames
        ctx_tensor = torch.cat(ctx_frames, dim=0)
        with torch.no_grad():
            context_emb = enc(ctx_tensor.to(device)).unsqueeze(0)
        
        # Ensure we always have 2 context actions (pad with current_pos)
        cp_tensor = torch.from_numpy(current_pos).float().to(device)
        ctx_actions_list.append(cp_tensor)
        if len(ctx_actions_list) > 2:
            ctx_actions_list.pop(0)
        while len(ctx_actions_list) < 2:
            ctx_actions_list.insert(0, cp_tensor)
        
        # Build 3-frame action context: [ctx_act_0, ctx_act_1, current_pos]
        ctx_act = torch.stack(list(ctx_actions_list) + [cp_tensor]).unsqueeze(0)  # (1, 3, 8)
        with torch.no_grad():
            context_aemb = aenc(ctx_act)

        # Live goal: step 0 frame = goal
        if live_goal and step == 0:
            with torch.no_grad():
                latent_goal = enc(start_img.to(device))
            print(f"  Live goal captured (norm={latent_goal.norm().item():.3f})")

        # 2. Encode current state
        with torch.no_grad():
            latent_current = enc(start_img)

        # 3. Compute current cost
        goal_cost = F.mse_loss(latent_current, latent_goal).item()
        cost_history.append(goal_cost)

        # 4. CEM planning
        print(f"\n  Step {step}: cost={goal_cost:.6f}")

        actions, cost = cem_planner(enc, pred, aenc, model_type, latent_goal,
                                    start_img=start_img, current_positions=current_pos,
                                    context_emb=context_emb, context_aemb=context_aemb,
                                    device=device)

        # 6. Execute K actions before replanning
        exec_steps = min(MPC_EXECUTE_STEPS, CEM_HORIZON)
        print(f"  Plan (H={CEM_HORIZON}): {dict(zip(SERVO_IDS, actions[0].astype(int).tolist()))}")

        for k in range(exec_steps):
            act = actions[k].astype(int).tolist()
            action_dict = {str(sid): int(val) for sid, val in zip(SERVO_IDS, act)}

            if pkt is not None:
                import serial_servo
                serial_servo.move_all(pkt, action_dict)
            
            current_pos = actions[k].astype(np.float32)
            if k < exec_steps - 1:
                time.sleep(0.8)

        # Track executed action for context history
        ctx_actions_list.append(torch.from_numpy(current_pos).float().to(device))
        if len(ctx_actions_list) > 2:
            ctx_actions_list.pop(0)

        # ─── POSITION ERROR GRASP DETECTION
        if pkt is not None:
            import serial_servo
            GRASP_SERVOS = [2, 4, 7]
            POS_THRESH = 100
            time.sleep(2.0)
            blocked = 0
            info = []
            for sid in GRASP_SERVOS:
                actual = serial_servo.read_position(pkt, sid) or 0
                target = float(current_pos[SERVO_IDS.index(sid)])
                error = abs(target - actual) if target > 0 else 0
                ok = error < POS_THRESH
                if ok: blocked += 1
                info.append(f"S{sid}:cmd={target:.0f} actual={actual} err={error:.0f} {'GRASP' if ok else '-'}")
            print(f"  Position: {' | '.join(info)}")
            if blocked >= 3:
                print(f"  ✅ All 3 servos at target — grasp detected")
                time.sleep(1.0)
                break

        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s")

    # Report
    if cost_history:
        print(f"\n  Cost history: min={min(cost_history):.4f}, final={cost_history[-1]:.4f}")

    if pH is not None:
        import serial_servo
        serial_servo.disconnect(pH)
    print("\n✓ Robot loop done")


# ─── MAIN ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Robot planner for bionic hand grasp")
    parser.add_argument("--model", type=str, required=True, choices=["cfc", "ar"])
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .ckpt file")
    parser.add_argument("--goal", type=str, default=None,
                        help="Path to goal image file (not needed if --live-goal)")
    parser.add_argument("--source", type=str, default="camera", choices=["camera", "file", "dryrun"])
    parser.add_argument("--data", type=str, default=None,
                        help="H5 data path (only if --source file)")
    parser.add_argument("--serial", type=str, default=None,
                        help="Serial port (e.g. COM13)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu or cuda")
    parser.add_argument("--live-goal", action="store_true",
                        help="Capture goal from camera at step 0 (same domain)")
    args = parser.parse_args()

    # 1. Load model
    print(f"Loading {args.model.upper()} from {args.checkpoint}...")
    enc, pred, aenc, has_norm = load_model(args.checkpoint, args.model, args.device)
    print(f"  Loaded: enc={sum(p.numel() for p in enc.parameters()):,} params")

    # 2. Load goal (skip if --live-goal, will capture from camera at step 0)
    latent_goal = None
    if args.goal:
        latent_np, goal_img_np = load_goal(args.goal)
        
        if latent_np is not None:
            latent_goal = torch.from_numpy(latent_np).unsqueeze(0).float().to(args.device)
            print(f"  Goal latent loaded from .npy (norm={latent_goal.norm().item():.3f})")
        elif goal_img_np is not None:
            goal_tensor = torch.from_numpy(goal_img_np).permute(2, 0, 1).unsqueeze(0).float()
            with torch.no_grad():
                latent_goal = enc(goal_tensor.to(args.device))
            print(f"  Goal encoded from PNG (norm={latent_goal.norm().item():.3f})")
            np.save(Path(args.goal).with_suffix(".npy"), latent_goal.cpu().numpy())
            print(f"  ✓ Cached latent to {Path(args.goal).with_suffix('.npy')}")
        else:
            raise FileNotFoundError(f"No goal found: {args.goal} (.npy or .png)")
    elif args.live_goal:
        print("  --live-goal: goal will be captured at step 0")
    else:
        raise ValueError("Need --goal or --live-goal")

    # 3. Run
    if args.source == "dryrun":
        if not args.data:
            raise ValueError("--data H5_PATH required for dryrun mode")
        dryrun_test(enc, pred, aenc, args.model, latent_goal,
                    h5_path=args.data, device=args.device)
    else:
        robot_loop(enc, pred, aenc, args.model, latent_goal,
                   device=args.device, source=args.source,
                   h5_path=args.data, serial_port=args.serial,
                   live_goal=args.live_goal)


if __name__ == "__main__":
    main()
