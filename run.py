"""
KLA Hackathon 2026 — Three Musketeers Submission

Usage:
    python run.py <input-dir> <output-dir>

Restores degraded semiconductor inspection images using SwinIR.
"""

import os
import sys
import glob
import numpy as np
import torch
import yaml

# Add models to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.swinir import build_swinir


def load_model(device):
    """Load trained SwinIR model from models/best_model.pt."""
    checkpoint_path = os.path.join(os.path.dirname(__file__), "models", "best_model.pt")
    
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Model not found: {checkpoint_path}")
    
    print(f"[run.py] Loading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract config from checkpoint
    if "config" in checkpoint and "model" in checkpoint["config"]:
        model_config = checkpoint["config"]["model"]
    else:
        # Fallback config for SwinIR 1.1M
        model_config = {
            "name": "SwinIR",
            "in_channels": 1,
            "out_channels": 1,
            "embed_dim": 60,
            "depths": [6, 6, 6, 6],
            "num_heads": [6, 6, 6, 6],
            "window_size": 8,
            "mlp_ratio": 2.0,
            "upscale": 2,
            "img_size": 64,
            "resi_connection": "1conv",
        }
    
    model = build_swinir(model_config)
    
    # Load weights
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    params = sum(p.numel() for p in model.parameters())
    print(f"[run.py] Model loaded: {params:,} parameters")
    return model


def restore(model, img_np, device):
    """Restore a single grayscale image."""
    # Handle input shapes
    if img_np.ndim == 2:
        tensor = torch.from_numpy(img_np).float().unsqueeze(0).unsqueeze(0).to(device)
    elif img_np.ndim == 3 and img_np.shape[0] == 1:
        tensor = torch.from_numpy(img_np).float().unsqueeze(0).to(device)
    elif img_np.ndim == 3 and img_np.shape[2] == 1:
        tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    else:
        tensor = torch.from_numpy(img_np).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            output = model(tensor)
    
    output = torch.clamp(output, 0.0, 1.0)
    output_np = output.squeeze().cpu().numpy()
    
    # Clean NaN/Inf
    output_np = np.nan_to_num(output_np, nan=0.0, posinf=1.0, neginf=0.0)
    
    return output_np.astype(np.float32)


def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not os.path.isdir(input_dir):
        print(f"[ERROR] Input directory not found: {input_dir}")
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run.py] Device: {device}")
    
    model = load_model(device)
    
    input_files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    if len(input_files) == 0:
        input_files = sorted(glob.glob(os.path.join(input_dir, "**", "*.npy"), recursive=True))
    
    if len(input_files) == 0:
        print(f"[ERROR] No .npy files found in: {input_dir}")
        sys.exit(1)
    
    print(f"[run.py] Found {len(input_files)} .npy files")
    
    for i, fpath in enumerate(input_files):
        img = np.load(fpath).astype(np.float32)
        restored = restore(model, img, device)
        
        out_path = os.path.join(output_dir, os.path.basename(fpath))
        np.save(out_path, restored)
        
        if (i + 1) % 50 == 0 or (i + 1) == len(input_files):
            print(f"[run.py] {i + 1}/{len(input_files)} restored")
    
    print(f"[run.py] Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
