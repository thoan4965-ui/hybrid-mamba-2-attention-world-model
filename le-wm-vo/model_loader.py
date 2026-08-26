"""
Model loader for LeWM checkpoints.
Loads TinyViT + CfC/AR predictor models from .ckpt files.
"""
import torch
from pathlib import Path
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
import sys

# Add le-wm to path
sys.path.insert(0, str(Path(__file__).parent))

from module import TinyViT, CfCPredictorV2, ARPredictor, Embedder, SIGReg


def load_model(ckpt_path: str, model_type: str = "cfc", device: str = "cpu"):
    """
    Load a LeWM model from checkpoint.
    
    Args:
        ckpt_path: Path to .ckpt file
        model_type: "cfc" or "ar"
        device: "cpu" or "cuda"
    
    Returns:
        dict with keys: encoder, predictor, action_encoder, sigreg, epoch
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"]
    epoch = ckpt.get("epoch", -1)
    
    # Remove "model." prefix from state_dict keys
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            cleaned_state_dict[k[6:]] = v  # Remove "model." prefix
    
    # Instantiate encoder (TinyViT - same for both CfC and AR)
    encoder = TinyViT(
        img_size=96,
        patch_size=8,
        num_layers=4,
        hidden_dim=64,
        num_heads=4,
        mlp_dim=256,
        output_dim=32,
    )
    
    # Instantiate predictor based on model_type
    if model_type == "cfc_v3":
        predictor = CfCPredictorV2(
            num_frames=6,
            input_dim=32,
            hidden_dim=96,
            output_dim=32,
            action_dim=32,
            backbone_layers=1,
            backbone_units=96,
        )
    elif model_type == "cfc_v2":
        predictor = CfCPredictorV2(
            num_frames=6,
            input_dim=32,
            hidden_dim=96,
            output_dim=32,
            action_dim=32,
            backbone_layers=1,
            backbone_units=96,
        )
    elif model_type == "cfc":
        predictor = CfCPredictorV2(
            num_frames=6,
            input_dim=32,
            hidden_dim=64,
            output_dim=32,
            action_dim=32,
            backbone_layers=1,
            backbone_units=64,
        )
    elif model_type == "ar":
        predictor = ARPredictor(
            num_frames=3,  # history_size only (AR processes context frames)
            depth=1,
            heads=2,
            mlp_dim=96,
            input_dim=32,
            hidden_dim=64,
            output_dim=32,
            dim_head=16,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'cfc_v3', 'cfc_v2', 'cfc', or 'ar'.")
    
    # Detect norm from checkpoint keys
    aenc_norm_keys = [k for k in cleaned_state_dict.keys() if k.startswith("action_encoder.norm.")]
    has_aenc_norm = len(aenc_norm_keys) > 0

    # Instantiate action encoder
    cfc_input_dim = 8   # Single-frame action dim for CfC
    ar_input_dim = 24   # frameskip * action_dim = 3 * 8 for AR
    input_dim = cfc_input_dim if model_type in ("cfc_v3", "cfc_v2", "cfc") else ar_input_dim
    action_encoder = Embedder(
        input_dim=input_dim,
        smoothed_dim=8,
        emb_dim=32,
        mlp_scale=2,
        use_norm=has_aenc_norm,
    )
    
    # Instantiate SIGReg
    sigreg = SIGReg(knots=9, num_proj=256)
    
    # Load encoder weights
    encoder_state = {k.replace("encoder.", ""): v 
                     for k, v in cleaned_state_dict.items() 
                     if k.startswith("encoder.")}
    encoder.load_state_dict(encoder_state, strict=True)
    
    # Load predictor weights
    predictor_state = {k.replace("predictor.", ""): v 
                       for k, v in cleaned_state_dict.items() 
                       if k.startswith("predictor.")}
    predictor.load_state_dict(predictor_state, strict=True)
    
    # Load action encoder weights
    action_encoder_state = {k.replace("action_encoder.", ""): v 
                            for k, v in cleaned_state_dict.items() 
                            if k.startswith("action_encoder.")}
    action_encoder.load_state_dict(action_encoder_state, strict=True)
    
    # Load SIGReg weights
    sigreg_state = {k.replace("sigreg.", ""): v 
                    for k, v in state_dict.items() 
                    if k.startswith("sigreg.")}
    sigreg.load_state_dict(sigreg_state, strict=True)
    
    # Move to device
    encoder = encoder.to(device)
    predictor = predictor.to(device)
    action_encoder = action_encoder.to(device)
    sigreg = sigreg.to(device)
    
    # Set to eval mode
    encoder.eval()
    predictor.eval()
    action_encoder.eval()
    sigreg.eval()
    
    return {
        "encoder": encoder,
        "predictor": predictor,
        "action_encoder": action_encoder,
        "sigreg": sigreg,
        "epoch": epoch,
        "model_type": model_type,
    }


def predict_next_embedding(model_dict, pixel_seq, action_seq, history_size=3, num_preds=3):
    """
    Predict next embedding given pixel and action sequences.
    
    Args:
        model_dict: Output from load_model()
        pixel_seq: (B, T, 3, H, W) - pixel sequence
        action_seq: (B, T, action_dim) - action sequence
        history_size: Number of history frames
        num_preds: Number of prediction frames
    
    Returns:
        (B, T, output_dim) - predicted embeddings
    """
    encoder = model_dict["encoder"]
    predictor = model_dict["predictor"]
    action_encoder = model_dict["action_encoder"]
    model_type = model_dict["model_type"]
    
    B, T, C, H, W = pixel_seq.shape
    
    # Encode pixels: (B, T, 3, H, W) -> (B, T, D)
    pixel_seq_flat = pixel_seq.reshape(B * T, C, H, W)
    with torch.no_grad():
        pixel_emb = encoder(pixel_seq_flat)  # (B*T, D)
    pixel_emb = pixel_emb.reshape(B, T, -1)  # (B, T, D)
    
    # Encode actions: (B, T, action_dim) -> (B, T, D)
    with torch.no_grad():
        action_emb = action_encoder(action_seq)  # (B, T, D)
    
    with torch.no_grad():
        if model_type in ("cfc_v3",):
            # Sequential RNN loop: build history, then predict future
            h = None
            # Phase 1: Build hidden state from history
            for t in range(history_size - 1):
                _, h = predictor.step(
                    pixel_emb[:, t:t+1], action_emb[:, t:t+1], h
                )
            # Phase 2: Predict future frames
            predictions = []
            for t in range(num_preds):
                feed_idx = history_size - 1 + t
                out, h = predictor.step(
                    pixel_emb[:, feed_idx:feed_idx+1],
                    action_emb[:, feed_idx:feed_idx+1],
                    h,
                )
                predictions.append(out)
            pred_emb = torch.cat(predictions, dim=1)
        else:
            # Batch-style: feed all frames at once
            pred_emb = predictor(pixel_emb, action_emb)
    
    return pred_emb


if __name__ == "__main__":
    # Test loader
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python model_loader.py <ckpt_path> [model_type]")
        print("  model_type: 'cfc' or 'ar' (default: auto-detect)")
        sys.exit(1)
    
    ckpt_path = sys.argv[1]
    
    # Auto-detect model type from path
    if "cfc" in ckpt_path.lower():
        model_type = "cfc"
    elif "ar" in ckpt_path.lower():
        model_type = "ar"
    else:
        model_type = sys.argv[2] if len(sys.argv) > 2 else "cfc"
    
    print(f"Loading {model_type} model from {ckpt_path}...")
    model = load_model(ckpt_path, model_type=model_type)
    
    print(f"✓ Loaded successfully!")
    print(f"  Epoch: {model['epoch']}")
    print(f"  Encoder params: {sum(p.numel() for p in model['encoder'].parameters()):,}")
    print(f"  Predictor params: {sum(p.numel() for p in model['predictor'].parameters()):,}")
    print(f"  Action encoder params: {sum(p.numel() for p in model['action_encoder'].parameters()):,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    B = 2
    # AR uses history_size=3, CfC uses history_size+num_preds=6
    T = 3 if model_type == "ar" else 6
    pixel_seq = torch.randn(B, T, 3, 96, 96)
    action_seq = torch.randn(B, T, 24)
    
    pred_emb = predict_next_embedding(model, pixel_seq, action_seq)
    print(f"✓ Forward pass successful!")
    print(f"  Input: pixels {pixel_seq.shape}, actions {action_seq.shape}")
    print(f"  Output: predictions {pred_emb.shape}")
