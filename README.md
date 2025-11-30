# 🎨 Toy-Jasper: Lightweight Diffusion Model for Shadow Relighting

This is a **small, efficient diffusion model** trained to predict realistic shadow maps given an object image, its mask, and lighting parameters (angle, azimuth, size).

---

## 🎯 What Does It Do?

**Goal:** Given an object and lighting conditions, generate spatially-correct shadow maps that can be composited back onto the scene for realistic relighting.

**Input:**
- RGB object image (256×256)
- Object mask (256×256)
- Lighting parameters: theta (elevation), phi (azimuth), size (intensity/scale)

**Output:**
- Predicted shadow map (latent space, ~32×32 after decoding)

---

## 🏗️ Architecture Overview

### 1. **VAE Encoding** (Frozen, from Stable Diffusion)
- Encodes full-resolution images (256×256) → 4-channel latent space (32×32)
- Used to compress object image and shadow maps into compact representations
- **Not trained**; kept frozen from pre-trained weights

### 2. **Toy-Jasper UNet** (Trainable)
A small U-Net that performs **latent-space diffusion**:
- **Input:** noisy latent + object latent + mask + spatial conditioning
- **Output:** noise prediction (denoising step)

**Key conditioning mechanisms:**
- **Sinusoidal embeddings** of lighting params → time-embedding injection
- **Spatial conditioning** (learned linear projection) → concatenated to model input

### 3. **Denoising Process**
The model learns to denoise shadow latents in a diffusion framework:
- `x_t = (1 - t) × x_0 + t × x_1` (linear interpolation between clean and noise)
- UNet predicts `v = x_1 - x_0` (velocity/difference)
- Loss: MSE between predicted and true `v`

---

## 📊 Data Flow (Training Step-by-Step)

```
Batch Data:
  - RGB (256×256)
  - Mask (256×256)
  - Shadow full-res (256×256)
  - Theta, Phi, Size (scalar params)

        ↓

[Step 1: Encode RGB object]
  RGB → VAE encoder → z_obj (4, 32, 32) → scale → downsample to (4, 32, 32)

[Step 2: Encode shadow at full resolution]
  Shadow_full (1, 256, 256) → tripled to (3, 256, 256)
                           → VAE encoder → z_sh_full (4, 32, 32)
                           → interpolate down to (4, 32, 32) ✓ (matches model input size)

[Step 3: Add noise and create noisy latent]
  x_0 = z_sh (ground truth shadow latent, 32×32)
  x_1 = random noise, same shape as x_0
  t = random time (0.01 to 0.99)
  x_t = (1-t) × x_0 + t × x_1  ← noisy version

[Step 4: Downsample mask to latent space]
  Mask (1, 256, 256) → nearest-neighbor resize → (1, 32, 32)

[Step 5: Create spatial conditioning]
  Theta, Phi, Size → linear layer → (COND_SPATIAL_CHANNELS, 32, 32)
  This tells the model "where the light is" and "how strong"

[Step 6: Build model input]
  model_input = concat([x_t, z_obj, mask_ds, spatial_cond], dim=1)
              = (batch, 4+4+1+8, 32, 32)  ← all same spatial size ✓

[Step 7: Create sinusoidal embeddings of lighting params]
  Theta → sin/cos embeddings → (batch, 256)
  Phi   → sin/cos embeddings → (batch, 256)
  Size  → sin/cos embeddings → (batch, 256)
  cond_emb = concat(all three) → (batch, 768)
  
  Optional: classifier-free training
    - Zero out cond_emb with probability CFG_PROB
    - Forces model to sometimes predict without conditioning

[Step 8: UNet forward pass]
  model_input, cond_emb → UNet → predicted_v (batch, 4, 32, 32)

[Step 9: Compute losses]
  Loss 1: Diffusion loss
    v_target = x_1 - x_0
    loss_rf = MSE(predicted_v, v_target)
  
  Loss 2: Pixel reconstruction loss
    pred_x0 = x_t - predicted_v  ← estimated ground truth
    pred_rgb = VAE.decode(pred_x0) → (batch, 3, 256, 256)
    gt_shadow3 = concat([shadow_full]*3) → same resolution
    loss_pixel = MSE(pred_rgb, gt_shadow3)
  
  total_loss = loss_rf + 0.5 × loss_pixel

[Step 10: Backprop and update]
  Optimize: UNet + cond_to_spatial + cond_proj
```

---

## 📁 Files

| File | Purpose |
|------|---------|
| `train.py` | Main training loop; handles data loading, model training, validation, checkpoints |
| `dataset.py` | Dataset class for loading RGB/mask/shadow/metadata |
| `clean_dataset.py` | Utility to clean/validate dataset files |
| `infer.py` | Inference script (DDPM sampling, VAE decoding) |
| `blendify/dataset_generator.py` | Generate synthetic training data |

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Training
```bash
# Use defaults (256×256 images, batch=4, 60 epochs)
python train.py

# Or customize with env vars:
RES=256 BATCH=8 EPOCHS=100 LR=1e-4 python train.py

# Common options:
#   RES: image resolution (256, 512, etc.)
#   BATCH: batch size
#   EPOCHS: number of training epochs
#   LR: learning rate
#   SHADOW_RES: latent-space resolution (default 32)
#   CFG_PROB: classifier-free guidance probability (default 0.2)
#   EARLY_STOP_PATIENCE: early stopping after N epochs no improvement (default 10)
```

### Inference
```bash
python infer.py --checkpoint unet_rf_best.pth --output outputs/
```

---

## 🔑 Key Concepts

### **Diffusion Training**
The model learns to denoise noisy shadow latents. At inference time, you start with pure noise and iteratively denoise, refining the shadow map step-by-step.

### **Latent Space**
All images are encoded into a 4D latent space (32×32) using a frozen VAE. This is ~8× smaller than pixel space, making training faster.

### **Sinusoidal Conditioning**
Lighting parameters (theta, phi, size) are encoded as sinusoidal waves, then injected into the UNet's time embedding. This is inspired by Jasper's architecture.

### **Spatial Conditioning**
A learned projection maps lighting params to a spatial feature map (32×32), concatenated to the model input. This helps the UNet "know where the light is spatially."

### **Classifier-Free Guidance**
During training, we randomly zero-out the conditioning embedding. At inference, this allows explicit control of "how strongly" the model obeys the lighting conditions.

---

## 📈 Training Details

- **Loss function:**
  - Diffusion: predict velocity `v = x_1 - x_0` (mean-squared-error)
  - Reconstruction: decode predicted latent, compare pixel-space shadow
- **Optimizer:** AdamW
- **Gradient clipping:** 1.0
- **Learning schedule:** constant (no decay)
- **Validation:** every epoch, early stopping if no improvement for `EARLY_STOP_PATIENCE` epochs

### Checkpoints
- `unet_rf_best.pth`: best validation loss
- `unet_rf_last.pth`: latest checkpoint
- `training_loss.png`: loss curves
- `training_progress.json`: epoch history

---

## 🐛 Important Implementation Notes

### Full-Resolution Shadow Encoding (Recent Fix)
The VAE expects full-resolution images (256×256) to produce correct latent shapes. Previously, encoding low-res shadows (32×32) produced 4×4 latents, causing shape mismatches. 

**Current approach (correct):**
1. Take full-res shadow (256×256)
2. Encode with VAE → (4, 32, 32)
3. Interpolate down to exact target size (32×32) if needed
4. Use in model

This ensures all model inputs (`x_t`, `z_obj`, `mask`, spatial_cond) are exactly (batch, C, 32, 32).

---

## 📊 Expected Results

After training on a large shadow dataset:
- **Train loss:** 0.001 – 0.01 (diffusion) + 0.001 – 0.01 (pixel)
- **Val loss:** similar or slightly higher (overfitting is normal)
- **Generated shadows:** should follow lighting direction and respect object mask

---

## 🔗 Dependencies

- `torch`, `torchvision`: model and data
- `diffusers`: VAE and UNet models
- `tqdm`: progress bars
- `matplotlib`: loss visualization
- `Pillow`: image I/O
- `numpy`: array ops

See `requirements.txt` for exact versions.

