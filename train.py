#!/usr/bin/env python3
"""
train_toy_jasper.py

Rewritten training script: uses the large dataset loading from your original script
but replaces the model architecture with the toy-Jasper architecture (small UNet,
sinusoidal conditioning injected into the UNet time embedding, spatial cond concat).
Inference has been removed per request.
"""

import os, json, random, sys, types, math
from pathlib import Path
from glob import glob

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusers import AutoencoderKL, UNet2DModel

# -------------------------
# Config (keeps your original env vars and defaults)
# -------------------------
DATA_DIR = os.environ.get("SHADOW_DATA_DIR", "/home/razz/Downloads/output_cleaned")
RES = int(os.environ.get("RES", "256"))
BATCH = int(os.environ.get("BATCH", "4"))
EPOCHS = int(os.environ.get("EPOCHS", "60"))
LR = float(os.environ.get("LR", "1e-4"))
BEST_CKPT = os.environ.get("BEST_CKPT", "unet_rf_best.pth")
LATEST_CKPT = os.environ.get("LATEST_CKPT", "unet_rf_last.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_OUTPUT_DIR = os.environ.get("VAL_OUTPUT_DIR", "val_output")
SEED = int(os.environ.get("SEED", "42"))
CFG_PROB = float(os.environ.get("CFG_PROB", "0.35"))  # prob to drop cond (classifier-free training)
EARLY_STOP_PATIENCE = int(os.environ.get("EARLY_STOP_PATIENCE", "10"))

# Toy UNet downsample latent size (we use same SHADOW_RES as your model_utils used)
SHADOW_RES = int(os.environ.get("SHADOW_RES", "32"))
COND_SPATIAL_CHANNELS = int(os.environ.get("COND_SPATIAL_CHANNELS", "8"))
SIN_EMB_DIM = 256
COND_EMB_DIM = SIN_EMB_DIM * 3

os.makedirs(VAL_OUTPUT_DIR, exist_ok=True)
random.seed(SEED); torch.manual_seed(SEED); np.random.seed(SEED)

# -------------------------
# Dataset (from your original script)
# -------------------------
class Samples(torch.utils.data.Dataset):
    def __init__(self, root, res=RES, shadow_res=SHADOW_RES):
        self.root = Path(root)
        rgb = sorted(glob(str(self.root / "rgb_*.png")))
        rnd = sorted(glob(str(self.root / "render_*.png")))
        keys = [Path(p).stem.split("_")[-1] for p in rgb + rnd]
        good = []
        for k in keys:
            ok_img = (self.root / f"rgb_{k}.png").exists() or (self.root / f"render_{k}.png").exists()
            ok_mask = (self.root / f"mask_{k}.png").exists()
            ok_shadow = (self.root / f"shadow_{k}.png").exists()
            ok_meta = (self.root / f"meta_{k}.json").exists()
            if ok_img and ok_mask and ok_shadow and ok_meta:
                good.append(k)
        self.keys = sorted(set(good))
        self.tRGB = T.Compose([T.Resize((res, res)), T.ToTensor()])
        self.tGray = T.Compose([T.Resize((res, res)), T.ToTensor()])
        self.tDown = T.Compose([T.Resize((shadow_res, shadow_res)), T.ToTensor()])
        print(f"Dataset: {len(self.keys)} samples in {root}")

    def __len__(self): return len(self.keys)
    
    def __getitem__(self, idx):
        k = self.keys[idx]
        rgb_path = self.root / f"rgb_{k}.png"
        if not rgb_path.exists():
            rgb_path = self.root / f"render_{k}.png"
        mask_path = self.root / f"mask_{k}.png"
        shadow_path = self.root / f"shadow_{k}.png"
        meta_path = self.root / f"meta_{k}.json"
        
        rgb = Image.open(rgb_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        shadow = Image.open(shadow_path).convert("L")
        meta = json.load(open(meta_path))
        
        return {
            "RGB": self.tRGB(rgb),
            "mask": self.tGray(mask)[0:1],
            "shadow": self.tDown(shadow)[0:1],       # small shadow used for latent training reference
            "shadow_full": self.tGray(shadow)[0:1],  # full-res shadow for pixel recon
            "theta": torch.tensor(meta.get("theta", 45.0)),
            "phi": torch.tensor(meta.get("phi", 0.0)),
            "size": torch.tensor(meta.get("size", 1.0)),
        }

# -------------------------
# Helpers: save preview & loss plot
# -------------------------
def save_relighting(tag, rgb_np, mask_np, gt_full, preds, out_dir=VAL_OUTPUT_DIR):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    row = []
    row.append(rgb_np)
    row.append(np.stack([mask_np]*3, axis=-1))
    row.append(np.stack([gt_full]*3, axis=-1))
    for p in preds:
        row.append(np.stack([p]*3, axis=-1))
    concat = np.concatenate(row, axis=1)
    Image.fromarray(concat.astype(np.uint8)).save(Path(out_dir) / f"{tag}.png")
    print(f"Saved: {tag}.png")

def update_loss_plot(train_losses, val_losses, current_epoch):
    plt.figure(figsize=(10, 6))
    epochs = range(len(train_losses))
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training Progress (Epoch {current_epoch})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=100, bbox_inches='tight')
    plt.close()
    with open('training_progress.json', 'w') as f:
        json.dump({'train_losses': train_losses, 'val_losses': val_losses, 'epochs': list(range(len(train_losses)))}, f, indent=2)

# -------------------------
# Sinusoidal embedding (Jasper-style)
# -------------------------
def sinusoidalEmbedding(x, dim=SIN_EMB_DIM):
    # x: tensor [B] or scalar tensor
    if x.dim() == 0:
        x = x.unsqueeze(0)
    if dim % 2 != 0:
        raise ValueError("dim must be even")
    half = dim // 2
    device = x.device
    freqs = 10000.0 ** (-torch.arange(0, half, device=device, dtype=torch.float32) / half)
    angles = x.unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(angles), torch.sin(angles)], dim=1)
    return emb  # [B, dim]

# -------------------------
# VAE helpers (same as before)
# -------------------------
def load_vae():
    print("Loading VAE (may download)...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float32).to(DEVICE)
    vae.eval()
    for p in vae.parameters(): p.requires_grad = False
    return vae

@torch.no_grad()
def encode_with_vae(vae, img):
    return vae.encode(img * 2.0 - 1.0).latent_dist.mode()

@torch.no_grad()
def decode_with_vae(vae, lat):
    img = vae.decode(lat / float(vae.config.scaling_factor)).sample
    return ((img + 1.0) / 2.0).clamp(0,1)

# -------------------------
# Toy-Jasper model bits
# -------------------------
def build_small_unet(latent_ch=4, cond_spatial_channels=COND_SPATIAL_CHANNELS):
    # in_channels = xt + z_obj + mask_ds + cond_spatial
    in_ch = latent_ch + latent_ch + 1 + cond_spatial_channels
    unet = UNet2DModel(
        sample_size=SHADOW_RES,
        in_channels=in_ch,
        out_channels=latent_ch,
        layers_per_block=2,
        block_out_channels=(64,128,128),
        down_block_types=("DownBlock2D","DownBlock2D","DownBlock2D"),
        up_block_types=("UpBlock2D","UpBlock2D","UpBlock2D"),
        attention_head_dim=None
    )
    return unet.to(DEVICE)

class CondToSpatial(nn.Module):
    def __init__(self, cond_dim=3, out_ch=COND_SPATIAL_CHANNELS):
        super().__init__()
        self.lin = nn.Linear(cond_dim, out_ch)
        nn.init.normal_(self.lin.weight, std=0.02)
        nn.init.zeros_(self.lin.bias)
    def forward(self, cond, H, W):
        B = cond.shape[0]
        x = self.lin(cond)  # [B,out_ch]
        x = x * 10.0  # amplify so the network cannot trivially ignore it
        x = x.view(B, -1, 1, 1).expand(-1, -1, H, W)
        return x

def wrap_time_embedding_with_cond(unet):
    if getattr(unet, "_jasper_cond_wrapped", False):
        return
    try:
        time_emb_dim = unet.time_embedding.linear_1.out_features
    except Exception:
        time_emb_dim = getattr(unet.time_embedding, "temb_dim", getattr(unet.time_embedding, "out_dim", 512))
    unet.cond_proj = nn.Linear(COND_EMB_DIM, time_emb_dim).to(DEVICE)
    nn.init.normal_(unet.cond_proj.weight, std=0.02)
    nn.init.zeros_(unet.cond_proj.bias)

    orig_forward = unet.time_embedding.forward
    def emb_forward(self, t_emb, *args, **kwargs):
        t_out = orig_forward(t_emb, *args, **kwargs)
        custom = getattr(self, "_custom_light_cond", None)
        if custom is not None:
            if custom.device != t_out.device or custom.dtype != t_out.dtype:
                custom = custom.to(device=t_out.device, dtype=t_out.dtype)
            t_out = t_out*5 + custom
        return t_out

    unet.time_embedding.forward = types.MethodType(emb_forward, unet.time_embedding)
    unet._jasper_cond_wrapped = True

def unet_call_with_cond(unet, model_in, timestep, cond_emb=None):
    if cond_emb is not None:
        with torch.no_grad():
            proj = unet.cond_proj(cond_emb)
        unet.time_embedding._custom_light_cond = proj
    else:
        unet.time_embedding._custom_light_cond = None
    try:
        out = unet(model_in, timestep=timestep)
    finally:
        unet.time_embedding._custom_light_cond = None
    return out

# -------------------------
# Training loop
# -------------------------
def train():
    ds = Samples(DATA_DIR)
    N = len(ds)
    if N == 0:
        raise RuntimeError("No data found!")
    n_train = int(0.75 * N)
    n_val = int(0.20 * N)
    n_test = N - n_train - n_val
    print(f"Split: train={n_train}, val={n_val}, test={n_test}")

    g = torch.Generator().manual_seed(SEED)
    train_set, val_set, test_set = torch.utils.data.random_split(ds, [n_train, n_val, n_test], generator=g)

    tr_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
    va_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)

    print("Loading VAE (for object encoding only)...")
    vae = load_vae()
    scaling = float(vae.config.scaling_factor)
    latent_ch = vae.config.latent_channels if hasattr(vae.config, "latent_channels") else 4

    print("Building small toy-Jasper UNet...")
    unet = build_small_unet(latent_ch=latent_ch, cond_spatial_channels=COND_SPATIAL_CHANNELS)
    wrap_time_embedding_with_cond(unet)
    cond_to_spatial = CondToSpatial(cond_dim=3, out_ch=COND_SPATIAL_CHANNELS).to(DEVICE)

    # optimizer must include cond_to_spatial and cond_proj
    params = list(unet.parameters()) + list(cond_to_spatial.parameters())
    if hasattr(unet, "cond_proj"):
        params += list(unet.cond_proj.parameters())
    opt = torch.optim.AdamW(params, lr=LR)

    best_val = float("inf")
    epochs_no_improve = 0
    train_losses = []
    val_losses = []

    step = 0
    for ep in range(EPOCHS):
        unet.train()
        pbar = tqdm(tr_loader, desc=f"Epoch {ep}")
        epoch_train_loss = 0.0
        train_steps = 0

        for batch_idx, b in enumerate(pbar):
            rgb = b["RGB"].to(DEVICE)
            mask = b["mask"].to(DEVICE)
            shadow = b["shadow"].to(DEVICE)            # small shadow map (SHADOW_RES)
            shadow_full = b["shadow_full"].to(DEVICE)  # full-res shadow for pixel recon
            theta = b["theta"].to(DEVICE)
            phi = b["phi"].to(DEVICE)
            size = b["size"].to(DEVICE)

            B = rgb.shape[0]

            # encode object image -> latents (frozen VAE)
            with torch.no_grad():
                z_obj = encode_with_vae(vae, rgb * 1.0)  # [B, C, h, w]
                z_obj = z_obj * scaling
                # ensure z_obj spatial matches SHADOW_RES
                z_obj_ds = F.interpolate(z_obj, size=(SHADOW_RES, SHADOW_RES), mode="bilinear", align_corners=False).clamp(-3,3)

            # ---- Encode shadow full-res, then downsample latent ----
            shadow3_full = torch.cat([shadow_full, shadow_full, shadow_full], dim=1)  # [B,3,RES,RES]

            with torch.no_grad():
                z_sh_full = encode_with_vae(vae, shadow3_full) * scaling  # output [B,C,32,32]

            # resize to SHADOW_RES exactly
            z_sh = F.interpolate(z_sh_full, size=(SHADOW_RES, SHADOW_RES),
                                 mode="bilinear", align_corners=False).clamp(-3,3)

            x0 = z_sh
            x1 = torch.randn_like(x0, device=DEVICE)
            t = torch.rand(B, 1, 1, 1, device=DEVICE).clamp(0.01, 0.99)
            xt = (1.0 - t) * x0 + t * x1

            # downsample mask to latent size (nearest)
            mask_ds = F.interpolate(mask, size=(SHADOW_RES, SHADOW_RES), mode="nearest")

            # spatial cond map -> broadcast
            sp = cond_to_spatial(torch.stack([theta/60.0, phi/360.0, size/6.0], dim=1), xt.shape[-2], xt.shape[-1])

            # build model input
            model_in = torch.cat([xt, z_obj_ds, mask_ds, sp], dim=1)

            # create sinusoidal cond embedding
            theta_emb = sinusoidalEmbedding(theta * 10, dim=SIN_EMB_DIM)
            phi_emb   = sinusoidalEmbedding(phi * 10, dim=SIN_EMB_DIM)
            size_emb  = sinusoidalEmbedding(size * 10, dim=SIN_EMB_DIM)
            cond_emb = torch.cat([theta_emb, phi_emb, size_emb], dim=1)  # [B,768]

            # classifier-free training: occasionally zero-out cond_emb
            cfg_mask = torch.rand(cond_emb.shape[0], 1, device=DEVICE) < CFG_PROB
            if cfg_mask.any():
                cond_emb = cond_emb * (~cfg_mask).float()

            timestep = torch.zeros(B, dtype=torch.long, device=DEVICE)

            # forward with injected cond
            out = unet_call_with_cond(unet, model_in, timestep=timestep, cond_emb=cond_emb)
            pred_v = out.sample
            v_target = x1 - x0
            loss_rf = F.mse_loss(pred_v, v_target)

            # reconstruction pixel loss: decode predicted x0 and compare to full-res shadow
            pred_x0 = (xt - pred_v)  # latent
            pred_rgb = decode_with_vae(vae, pred_x0)  # [B,3,H',W']
            # ground truth shadow rgb (tripled) at same resolution as pred_rgb
            gt_shadow3 = torch.cat([shadow_full, shadow_full, shadow_full], dim=1)
            gt_resized = F.interpolate(gt_shadow3, size=pred_rgb.shape[-2:], mode="bilinear")
            loss_pixel = F.mse_loss(pred_rgb, gt_resized)

            loss = loss_rf + 0.5 * loss_pixel

            # backprop
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            opt.step()

            if step % 100 == 0:
                print(f"ep {ep} step {step} loss_rf {loss_rf.item():.6f} loss_px {loss_pixel.item():.6f}")
            step += 1

            epoch_train_loss += loss.item()
            train_steps += 1
            pbar.set_postfix({"loss": f"{loss.item():.6f}", "rf": f"{loss_rf.item():.4f}", "px": f"{loss_pixel.item():.4f}"})

        # validation (simple latent RF numeric)
        unet.eval()
        val_loss = 0.0; cnt = 0
        with torch.no_grad():
            for i, b in enumerate(va_loader):
                rgb = b["RGB"].to(DEVICE)
                mask = b["mask"].to(DEVICE)
                shadow = b["shadow"].to(DEVICE)
                shadow_full = b["shadow_full"].to(DEVICE)
                theta = b["theta"].to(DEVICE)
                phi = b["phi"].to(DEVICE)
                size = b["size"].to(DEVICE)

                z_obj = encode_with_vae(vae, rgb) * scaling
                z_obj_ds = F.interpolate(z_obj, size=(SHADOW_RES, SHADOW_RES), mode="bilinear", align_corners=False).clamp(-3,3)
                shadow3_full = torch.cat([shadow_full, shadow_full, shadow_full], dim=1)
                with torch.no_grad():
                    z_sh_full = encode_with_vae(vae, shadow3_full) * scaling

                z_sh = F.interpolate(z_sh_full, size=(SHADOW_RES, SHADOW_RES),
                                     mode="bilinear", align_corners=False).clamp(-3,3)

                x0 = z_sh
                x1 = torch.randn_like(x0)
                t = torch.rand(1,1,1,1, device=DEVICE).clamp(0.01, 0.99)
                xt = (1 - t) * x0 + t * x1

                mask_ds = F.interpolate(mask, size=(SHADOW_RES, SHADOW_RES), mode="nearest")
                sp = cond_to_spatial(torch.stack([theta/60.0, phi/360.0, size/6.0], dim=1), xt.shape[-2], xt.shape[-1])
                model_in = torch.cat([xt, z_obj_ds, mask_ds, sp], dim=1)

                theta_emb = sinusoidalEmbedding(theta * 10, dim=SIN_EMB_DIM)
                phi_emb   = sinusoidalEmbedding(phi * 10, dim=SIN_EMB_DIM)
                size_emb  = sinusoidalEmbedding(size * 10, dim=SIN_EMB_DIM)
                cond_emb = torch.cat([theta_emb, phi_emb, size_emb], dim=1)

                timestep = torch.zeros(1, dtype=torch.long, device=DEVICE)
                out = unet_call_with_cond(unet, model_in, timestep=timestep, cond_emb=cond_emb)
                pred_v = out.sample
                loss_val = F.mse_loss(pred_v, x1 - x0)
                if torch.isfinite(loss_val):
                    val_loss += float(loss_val)
                    cnt += 1

        avg_val = val_loss / max(1, cnt)
        avg_train = epoch_train_loss / max(1, train_steps)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        print(f"Epoch {ep}: train_loss={avg_train:.6f}, val_loss={avg_val:.6f}")

        # Save checkpoint
        ckpt = {"unet": unet.state_dict(), "epoch": ep, "best_val": avg_val, "cond_to_spatial": cond_to_spatial.state_dict(), "cond_proj": unet.cond_proj.state_dict()}
        torch.save(ckpt, LATEST_CKPT)
        if avg_val < best_val:
            best_val = avg_val
            epochs_no_improve = 0
            torch.save(ckpt, BEST_CKPT)
            print(f"✓ New best: {BEST_CKPT} (val_loss={avg_val:.6f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{EARLY_STOP_PATIENCE} epochs")

        update_loss_plot(train_losses, val_losses, ep)

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n🛑 Early stopping")
            break

    print("Training complete!")
    print(f"Final best val loss: {best_val:.6f}")

if __name__ == "__main__":
    train()
