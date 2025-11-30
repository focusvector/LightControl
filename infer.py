#!/usr/bin/env python3
"""
infer_toy_jasper.py

Inference for the toy-Jasper training script.

- Loads checkpoint (LATEST_CKPT by default)
- Loads VAE and small UNet (same builder as train)
- Runs multi-step Rectified Flow ODE in latent space with CFG
- Upsamples predicted latent to pixel space via VAE decode
- Composites prediction with RGB and saves visualizations
"""

import os, json, math, types
from pathlib import Path
from glob import glob

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusers import AutoencoderKL, UNet2DModel

# -------------------------
# Config (match train_toy_jasper.py)
# -------------------------
DATA_DIR = os.environ.get("SHADOW_DATA_DIR", "/home/razz/Downloads/output_cleaned")
RES = int(os.environ.get("RES", "256"))
SHADOW_RES = int(os.environ.get("SHADOW_RES", "32"))
COND_SPATIAL_CHANNELS = int(os.environ.get("COND_SPATIAL_CHANNELS", "8"))
SIN_EMB_DIM = 256
COND_EMB_DIM = SIN_EMB_DIM * 3

CKPT_PATH = os.environ.get("CKPT_PATH", "unet_rf_last.pth")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "infer_output")
TEST_KEYS = os.environ.get("TEST_KEYS", "test_keys.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Helpers (same logic as train file)
# -------------------------
def sinusoidalEmbedding(x, dim=SIN_EMB_DIM):
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

class CondToSpatial(nn.Module):
    def __init__(self, cond_dim=3, out_ch=COND_SPATIAL_CHANNELS):
        super().__init__()
        self.lin = nn.Linear(cond_dim, out_ch)
        nn.init.normal_(self.lin.weight, std=0.02)
        nn.init.zeros_(self.lin.bias)
    def forward(self, cond, H, W):
        B = cond.shape[0]
        x = self.lin(cond)  # [B,out_ch]
        x = x * 4.0
        x = x.view(B, -1, 1, 1).expand(-1, -1, H, W)
        return x

def build_small_unet(latent_ch=4, cond_spatial_channels=COND_SPATIAL_CHANNELS):
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
            t_out = t_out + custom
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

@torch.no_grad()
def decode_with_vae(vae, lat):
    img = vae.decode(lat / float(vae.config.scaling_factor)).sample
    return ((img + 1.0) / 2.0).clamp(0,1)

def composite(rgb, mask, shadow):
    shadow_3ch = np.stack([shadow]*3, axis=-1)
    lit = rgb * shadow_3ch
    bg = np.ones_like(rgb)
    mask_3ch = np.stack([mask]*3, axis=-1)
    result = lit * mask_3ch + bg * (1 - mask_3ch)
    return (result * 255).clip(0,255).astype(np.uint8)

def visualize_and_save(key, rgb_np, mask_np, gt_shadow_np, pred_shadow_np, relight_shadows, theta_angles, phi_angles, out_dir):
    H, W = rgb_np.shape[:2]
    PAD = 60
    rgb_vis = (rgb_np * 255).astype(np.uint8)
    mask_vis = np.stack([mask_np * 255]*3, axis=-1).astype(np.uint8)
    gt_shadow_vis = np.stack([gt_shadow_np * 255]*3, axis=-1).astype(np.uint8)
    pred_shadow_vis = np.stack([pred_shadow_np * 255]*3, axis=-1).astype(np.uint8)
    row1 = np.concatenate([rgb_vis, mask_vis, gt_shadow_vis, pred_shadow_vis], axis=1)
    row2 = np.concatenate([ np.stack([s*255]*3, axis=-1).astype(np.uint8) for s in relight_shadows ], axis=1)
    row1_padded = np.pad(row1, ((PAD,0),(0,0),(0,0)), constant_values=255)
    row2_padded = np.pad(row2, ((PAD,0),(0,0),(0,0)), constant_values=255)
    canvas = np.concatenate([row1_padded, row2_padded], axis=0)
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        font = ImageFont.load_default(); font_small = font
    draw.text((W//2 - 30, 10), "RGB", fill=(0,0,0), font=font)
    draw.text((W + W//2 - 30, 10), "Mask", fill=(0,0,0), font=font)
    draw.text((2*W + W//2 - 80, 10), "GT Shadow", fill=(0,0,0), font=font)
    draw.text((3*W + W//2 - 100, 10), "Pred Shadow", fill=(0,0,0), font=font)
    y_offset = H + PAD
    for i in range(len(phi_angles)):
        x_pos = i*W + 10
        label = f"θ={theta_angles[i]:.0f}° φ={phi_angles[i]:.0f}°"
        draw.text((x_pos, y_offset + 10), label, fill=(255,0,255), font=font_small)
    out_path = Path(out_dir) / f"full_viz_{key}.png"
    img.save(out_path)
    print(f"Saved viz: {out_path}")

# -------------------------
# Main inference logic
# -------------------------
def main():
    if not os.path.exists(CKPT_PATH):
        print(f"Checkpoint not found: {CKPT_PATH}")
        return

    print("Loading checkpoint:", CKPT_PATH)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float32).to(DEVICE)
    vae.eval()

    latent_ch = vae.config.latent_channels if hasattr(vae.config, "latent_channels") else 4

    print("Building small UNet and cond projection...")
    unet = build_small_unet(latent_ch=latent_ch, cond_spatial_channels=COND_SPATIAL_CHANNELS)
    wrap_time_embedding_with_cond(unet)

    cond_to_spatial = CondToSpatial(cond_dim=3, out_ch=COND_SPATIAL_CHANNELS).to(DEVICE)

    # load weights (be permissive)
    if "unet" in ckpt:
        unet.load_state_dict(ckpt["unet"], strict=False)
    if "cond_to_spatial" in ckpt:
        cond_to_spatial.load_state_dict(ckpt["cond_to_spatial"], strict=False)
    if "cond_proj" in ckpt and hasattr(unet, "cond_proj"):
        try:
            unet.cond_proj.load_state_dict(ckpt["cond_proj"])
        except Exception:
            # older/newer shapes: fallback
            for k,v in ckpt.get("cond_proj", {}).items():
                if hasattr(unet.cond_proj, k):
                    setattr(unet.cond_proj, k, v)

    unet.eval()

    # Load test keys
    if os.path.exists(TEST_KEYS):
        with open(TEST_KEYS, "r") as f:
            test_keys = json.load(f)
        print(f"Loaded {len(test_keys)} test keys from {TEST_KEYS}")
    else:
        # fallback: pick first few keys from DATA_DIR
        keys_rgb = sorted(glob(str(Path(DATA_DIR) / "rgb_*.png")))
        keys = [Path(p).stem.split("_")[-1] for p in keys_rgb]
        test_keys = keys[:20]
        print(f"No test_keys.json found; using {len(test_keys)} keys from DATA_DIR (first 20)")

    tRGB = T.Compose([T.Resize((RES, RES)), T.ToTensor()])
    tGray = T.Compose([T.Resize((RES, RES)), T.ToTensor()])

    STEPS = int(os.environ.get("STEPS", "40"))
    CFG_SCALE = float(os.environ.get("CFG_SCALE", "2.0"))

    for idx, key in enumerate(test_keys[:50]):  # limit to 50 samples to avoid hogging
        print(f"\n[{idx}] {key}")
        rgb_path = Path(DATA_DIR) / f"rgb_{key}.png"
        if not rgb_path.exists():
            rgb_path = Path(DATA_DIR) / f"render_{key}.png"
        mask_path = Path(DATA_DIR) / f"mask_{key}.png"
        meta_path = Path(DATA_DIR) / f"meta_{key}.json"

        if not (rgb_path.exists() and mask_path.exists() and meta_path.exists()):
            print("  missing files, skipping")
            continue

        rgb = Image.open(rgb_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        meta = json.load(open(meta_path))

        rgb_t = tRGB(rgb).unsqueeze(0).to(DEVICE)
        mask_t = tGray(mask)[0:1].unsqueeze(0).to(DEVICE)

        theta = meta.get("theta", 45.0)
        phi = meta.get("phi", 0.0)
        size = meta.get("size", 1.0)

        # ground truth shadow
        shadow_path = Path(DATA_DIR) / f"shadow_{key}.png"
        if shadow_path.exists():
            gt_shadow = Image.open(shadow_path).convert("L")
            gt_t = tGray(gt_shadow)[0:1].unsqueeze(0).to(DEVICE)
            gt_np = gt_t[0,0].cpu().numpy()
        else:
            gt_np = np.ones((RES, RES))

        # object latents
        with torch.no_grad():
            z_obj = vae.encode(rgb_t * 2.0 - 1.0).latent_dist.mode() * float(vae.config.scaling_factor)
            z_obj_ds = F.interpolate(z_obj, size=(SHADOW_RES, SHADOW_RES), mode="bilinear", align_corners=False).clamp(-3,3)

        mask_ds = F.interpolate(mask_t, size=(SHADOW_RES, SHADOW_RES), mode="nearest")

        # conditioning embeddings
        cond_norm = torch.tensor([[theta/60.0, phi/360.0, size/6.0]], device=DEVICE, dtype=torch.float32)
        sp = cond_to_spatial(cond_norm, SHADOW_RES, SHADOW_RES)

        theta_emb = sinusoidalEmbedding(torch.tensor([theta*10.0], device=DEVICE), dim=SIN_EMB_DIM)
        phi_emb = sinusoidalEmbedding(torch.tensor([phi*10.0], device=DEVICE), dim=SIN_EMB_DIM)
        size_emb = sinusoidalEmbedding(torch.tensor([size*10.0], device=DEVICE), dim=SIN_EMB_DIM)
        cond_emb = torch.cat([theta_emb, phi_emb, size_emb], dim=1)  # [1,768]
        cond_null = torch.zeros_like(cond_emb)

        # start from latent noise
        gen = torch.Generator(device=DEVICE).manual_seed(42)
        xt = torch.randn(1, z_obj.shape[1], SHADOW_RES, SHADOW_RES, device=DEVICE, generator=gen)

        dt = 1.0 / max(1, STEPS)
        for step in range(STEPS):
            t_scalar = 1.0 - step * dt
            timestep = torch.zeros(1, dtype=torch.long, device=DEVICE)  # toy uses zeros

            model_in_cond = torch.cat([xt, z_obj_ds, mask_ds, sp], dim=1)
            model_in_uncond = torch.cat([xt, z_obj_ds, mask_ds, torch.zeros_like(sp)], dim=1)

            out_c = unet_call_with_cond(unet, model_in_cond, timestep=timestep, cond_emb=cond_emb).sample
            out_u = unet_call_with_cond(unet, model_in_uncond, timestep=timestep, cond_emb=None).sample

            v = out_u + CFG_SCALE * (out_c - out_u)
            xt = xt - v * dt
            # clamp to keep numerical stability
            xt = xt.clamp(-10, 10)

        pred_lat = xt.clamp(-10,10)
        with torch.no_grad():
            pred_img = decode_with_vae(vae, pred_lat)[0]  # [3,H_lat,W_lat] in [0,1]
        # extract single channel shadow (we trained on grey shadow replicated to 3 ch)
        pred_shadow = pred_img[0].cpu().numpy()  # single channel
        # upsample to full RES
        pred_shadow_up = F.interpolate(torch.from_numpy(pred_shadow).unsqueeze(0).unsqueeze(0).to(DEVICE),
                                       size=(RES, RES), mode="bilinear", align_corners=False)[0,0].cpu().numpy()
        rgb_np = (rgb_t[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        mask_np = mask_t[0,0].cpu().numpy()

        # relighting examples (original + 3 rotated angles)
        relight_configs = [
            (theta, phi),
            (theta, (phi + 90) % 360),
            (theta, (phi + 180) % 360),
            (theta, (phi + 270) % 360),
        ]
        relight_shadows = []
        phi_angles = []
        for th, ph in relight_configs:
            phi_angles.append(ph)
            cond_norm_r = torch.tensor([[th/60.0, ph/360.0, size/6.0]], device=DEVICE, dtype=torch.float32)
            sp_r = cond_to_spatial(cond_norm_r, SHADOW_RES, SHADOW_RES)
            t_emb_r = sinusoidalEmbedding(torch.tensor([th*10.0], device=DEVICE), dim=SIN_EMB_DIM)
            p_emb_r = sinusoidalEmbedding(torch.tensor([ph*10.0], device=DEVICE), dim=SIN_EMB_DIM)
            s_emb_r = sinusoidalEmbedding(torch.tensor([size*10.0], device=DEVICE), dim=SIN_EMB_DIM)
            cond_emb_r = torch.cat([t_emb_r, p_emb_r, s_emb_r], dim=1)

            xt_r = torch.randn(1, z_obj.shape[1], SHADOW_RES, SHADOW_RES, device=DEVICE, generator=gen)
            for step in range(STEPS):
                timestep = torch.zeros(1, dtype=torch.long, device=DEVICE)
                mi_c = torch.cat([xt_r, z_obj_ds, mask_ds, sp_r], dim=1)
                mi_u = torch.cat([xt_r, z_obj_ds, mask_ds, torch.zeros_like(sp_r)], dim=1)
                out_c = unet_call_with_cond(unet, mi_c, timestep=timestep, cond_emb=cond_emb_r).sample
                out_u = unet_call_with_cond(unet, mi_u, timestep=timestep, cond_emb=None).sample
                v = out_u + CFG_SCALE * (out_c - out_u)
                xt_r = xt_r - v * dt
                xt_r = xt_r.clamp(-10,10)
            pred_lat_r = xt_r
            with torch.no_grad():
                pred_img_r = decode_with_vae(vae, pred_lat_r)[0]
            pred_sh_r = pred_img_r[0].cpu().numpy()
            pred_sh_r_up = F.interpolate(torch.from_numpy(pred_sh_r).unsqueeze(0).unsqueeze(0).to(DEVICE),
                                         size=(RES, RES), mode="bilinear", align_corners=False)[0,0].cpu().numpy()
            relight_shadows.append(pred_sh_r_up)

        # save visualization
        visualize_and_save(key, rgb_np, mask_np, gt_np, pred_shadow_up, relight_shadows, [c[0] for c in relight_configs], phi_angles, OUTPUT_DIR)

    print("\nInference complete.")

if __name__ == "__main__":
    main()
