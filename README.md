# AI-Based Restoration of Degraded Images for Semiconductor Inspection

**Team:** Three Musketeers  
**Competition:** KLA Hackathon 2026

---

## Problem

Semiconductor inspection images suffer from three degradations applied in random order:

1. **Speckle noise** — multiplicative pixel-level noise that pushes values beyond [0, 1]
2. **Gaussian noise** — additive noise that softens edges and reduces sharpness
3. **Spatial resolution reduction** — 2× downsampling

The model must restore these degraded images in a single forward pass.

---

## Solution

- **Architecture:** Lightweight SwinIR transformer
- **Parameters:** 1,085,373 (1.1M)
- **Input:** 128×128 grayscale NoisyLR
- **Output:** 256×256 grayscale restored
- **Approach:** Joint denoising + 2× super-resolution
- **Checkpoint:** 26.8 MB

---

## Repository Structure

    three_musketeers/
    ├── run.py
    ├── requirements.txt
    ├── README.md
    └── models/
        ├── best_model.pt
        ├── swinir.py
        ├── losses.py
        └── __init__.py

---

## Setup

### Prerequisites
- Python 3.8+
- pip

### Installation

    pip install -r requirements.txt

---

## Usage

    python run.py <input-dir> <output-dir>

**Arguments:**
- `<input-dir>`: Directory containing .npy NoisyLR images
- `<output-dir>`: Directory where restored .npy images will be saved

**Example:**

    python run.py ./test_input ./test_output

---

## Results

### Metrics (Validated on 320-image hold-out set)

| Metric | Value |
|--------|-------|
| PSNR | 28.08 dB |
| SSIM | 0.7725 |
| LPIPS | 0.2688 |

### Baseline Comparison

| Method | PSNR | SSIM | Params |
|--------|------|------|--------|
| Bicubic + Median Filter | 24.94 dB | 0.6227 | 0 |
| Our SwinIR | **28.08 dB** | **0.7725** | **1.1M** |

---

## Runtime Performance

| Setting | Value |
|---------|-------|
| Batch size | 1 (per image) |
| CPU | ~1.3 img/s |
| GPU (H100 estimated) | ~180 img/s |

---

## Model Details

- **Architecture:** SwinIR
- **Embed dim:** 60
- **RSTB blocks:** 4
- **Window size:** 8
- **Upscale factor:** 2×
- **Loss:** Charbonnier + SSIM + Edge
- **Optimizer:** AdamW
- **Training data:** 3,200 paired images
- **Epochs:** 45
