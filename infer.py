#!/usr/bin/env python3

import os
import random
import tarfile
import json
from contextlib import nullcontext
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image
import matplotlib.pyplot as plt

from diffusers import AutoencoderKL, UNet2DConditionModel
from helpers import sinusoidalEmbedding


# ============================================================
# CONFIG
# ============================================================
DATA_TAR = "/home/razz/Downloads/jasperTestData/controllable-shadow-generation-benchmark.tar"
CHECKPOINT = "checkpoint_best.pth"   # change as needed
RES = 256

def build_custom_lightings(meta, count=3):
    """Construct lighting variations that significantly diverge from the ground truth."""
    theta = float(meta.get("theta", 30.0))
    phi = float(meta.get("phi", 0.0)) % 360.0
    size = float(meta.get("size", 2.0))

    variations = []

    if count >= 1:
        variations.append(
            {
                "label": "custom_light_overhead",
                "theta": 85.0,
                "phi": round((phi + 180.0) % 360.0, 3),
                "size": round(min(max(size * 3.5, size + 4.0), 8.0), 3),
            }
        )

    if count >= 2:
        variations.append(
            {
                "label": "custom_light_grazing",
                "theta": 5.0,
                "phi": round((phi + 270.0) % 360.0, 3),
                "size": round(max(size * 0.2, 0.2), 3),
            }
        )

    if count >= 3:
        theta_offset = 50.0 if theta <= 35.0 else -50.0
        variations.append(
            {
                "label": "custom_light_lateral",
                "theta": round(float(np.clip(theta + theta_offset, 5.0, 85.0)), 3),
                "phi": round((phi + 120.0) % 360.0, 3),
                "size": round(max(size * 0.6, size - 0.8), 3),
            }
        )

    return variations[:count]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Load a RANDOM sample from the benchmark tar
# ============================================================
def load_random_sample(tar_path):

    with tarfile.open(tar_path, "r:*") as tf:
        members = [m.name for m in tf.getmembers() if m.isfile()]
        prefixes = [
            name.replace(".image.png", "")
            for name in members
            if name.endswith(".image.png")
        ]

    if not prefixes:
        raise RuntimeError("No image entries found inside tar archive")

    chosen = random.choice(prefixes)

    with tarfile.open(tar_path, "r:*") as tf:
        def read(name):
            fileobj = tf.extractfile(name)
            if fileobj is None:
                raise FileNotFoundError(f"Missing {name} in archive")
            return fileobj.read()

        img = Image.open(BytesIO(read(chosen + ".image.png"))).convert("RGB")
        mask = Image.open(BytesIO(read(chosen + ".mask.png"))).convert("L")
        sh = Image.open(BytesIO(read(chosen + ".shadow.png"))).convert("L")
        meta = json.loads(read(chosen + ".metadata.json").decode("utf-8"))

    return img, mask, sh, meta, chosen


# ============================================================
# Load models (UNet + VAE)
# ============================================================
def load_models(checkpoint):

    print("Loading VAE…")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    vae.requires_grad_(False)
    scaling = vae.config.scaling_factor

    print(f"Loading UNet checkpoint: {checkpoint}")
    uNet = UNet2DConditionModel(
        sample_size=RES // 8,
        in_channels=9,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(64,128,128,128),
        attention_head_dim=(64,64,64,64),
        cross_attention_dim=768
    ).to(device)

    state = torch.load(checkpoint, map_location=device)
    uNet.load_state_dict(state)
    uNet.eval()

    return vae, uNet, scaling


# ============================================================
# Predict shadow (Rectified Flow)
# ============================================================
@torch.no_grad()
def predict_shadow(obj_rgb, mask, theta, phi, size, vae, uNet, scaling):

    # Encode object + mask
    Zobj = vae.encode(obj_rgb * 2 - 1).latent_dist.mode() * scaling
    Zmsk = F.interpolate(mask, size=Zobj.shape[-2:])

    # very small t for inference
    t = torch.tensor([0.001], device=device).view(1, 1, 1, 1)
    noise = torch.randn_like(Zobj)
    latent = t * noise

    # Concatenate latents
    Zin = torch.cat([latent, Zobj, Zmsk], dim=1)

    embed = torch.cat([
        sinusoidalEmbedding(torch.tensor([theta], device=device)),
        sinusoidalEmbedding(torch.tensor([phi], device=device)),
        sinusoidalEmbedding(torch.tensor([size], device=device)),
    ], dim=1).unsqueeze(1)

    # FP16 safe UNet forward
    if device.type == "cuda":
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    else:
        autocast_ctx = nullcontext()

    with autocast_ctx:
        pred = uNet(
            sample=Zin,
            timestep=(t[:, 0, 0, 0] * 1000).long(),
            encoder_hidden_states=embed
        ).sample

    # Flow velocity -> latent
    Zsh = -pred

    z32 = (Zsh / scaling).float()
    original_dtype = next(vae.parameters()).dtype
    if original_dtype != torch.float32:
        vae.to(dtype=torch.float32)
    shadow_3 = vae.decode(z32).sample
    if original_dtype != torch.float32:
        vae.to(dtype=original_dtype)

    shadow_1 = shadow_3.mean(dim=1, keepdim=True)
    shadow_1 = (shadow_1.clamp(-1, 1) + 1) / 2

    return shadow_1  # 1×1×128×128


# ============================================================
# Overlay: shadow BEHIND object
# ============================================================
def composite_shadow(obj, mask, pred_shadow):
    """
    obj:   H×W×3  uint8
    mask:  H×W    {0..255}
    pred:  H×W    float 0..1
    """
    obj = obj.astype(np.float32)
    if mask.max() > 1:
        mask = mask / 255.0
    mask = mask[..., None]  # H×W×1

    floor = np.ones_like(obj) * 200
    shadow_floor = floor * (1 - pred_shadow[..., None])

    final = obj * mask + shadow_floor * (1 - mask)
    return final.clip(0, 255).astype("uint8")


# ============================================================
# Visualization
# ============================================================
def visualize(lighting_info, entries, prefix):
    cols = 4
    rows = len(entries)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    if rows == 1:
        axes = np.expand_dims(axes, 0)

    for row_idx, entry in enumerate(entries):
        ax = axes[row_idx]
        img = entry["object"]
        mask = entry["mask"]
        pred_shadow = entry["pred_shadow"]
        composite = entry["composite"]
        label = entry["label"]
        gt_shadow = entry.get("gt_shadow")
        theta = entry.get("theta")
        phi = entry.get("phi")
        size = entry.get("size")

        ax[0].imshow(img)
        if theta is not None and phi is not None and size is not None:
            ax[0].set_title(
                f"Object\n{label}\nθ={theta:.1f}, φ={phi:.1f}, size={size:.2f}"
            )
        else:
            ax[0].set_title(f"Object\n{label}")

        if gt_shadow is not None:
            ax[1].imshow(gt_shadow.squeeze(), cmap="gray")
            ax[1].set_title("GT Shadow")
        else:
            ax[1].imshow(np.zeros_like(pred_shadow.squeeze()), cmap="gray")
            ax[1].set_title("GT Shadow (N/A)")

        pred_vis = (pred_shadow.squeeze() * 255).astype("uint8")
        ax[2].imshow(pred_vis, cmap="gray")
        ax[2].set_title("Predicted Shadow")

        ax[3].imshow(composite)
        ax[3].set_title("Overlay")

        for a in ax:
            a.axis("off")

    plt.tight_layout()
    out_path = f"{prefix}_all_vis.png"
    plt.savefig(out_path)
    print("Saved visualization:", out_path)
    plt.show()

    print("\nLighting Parameters:")
    for info in lighting_info:
        print(
            f" - {info['label']}: theta={info['theta']}, phi={info['phi']}, size={info['size']}"
        )

    gt_entry = next((e for e in entries if "gt_shadow" in e), None)
    if gt_entry is None:
        return

    fig, ax = plt.subplots(1, 4, figsize=(16, 5))

    ax[0].imshow(gt_entry["object"])
    ax[0].set_title(
        "Object\n" +
        f"θ={gt_entry['theta']:.1f}, φ={gt_entry['phi']:.1f}, size={gt_entry['size']:.2f}"
    )

    ax[1].imshow(gt_entry["gt_shadow"], cmap="gray")
    ax[1].set_title("GT Shadow")

    pred_vis = (gt_entry["pred_shadow"] * 255).astype("uint8")
    ax[2].imshow(pred_vis, cmap="gray")
    ax[2].set_title("Predicted Shadow")

    ax[3].imshow(gt_entry["composite"])
    ax[3].set_title("Overlay (Shadow behind object)")

    for a in ax:
        a.axis("off")

    plt.tight_layout()
    out = f"{prefix}_vis.png"
    plt.savefig(out)
    print("Saved visualization:", out)
    plt.show()


# ============================================================
# MAIN
# ============================================================
def main():

    img, mask, gt_sh, meta, prefix = load_random_sample(DATA_TAR)

    print("\nPicked sample:", prefix)
    print("Metadata:", meta)

    tf_img = T.Compose([T.Resize((RES, RES)), T.ToTensor()])
    tf_mask = T.Compose([T.Resize((RES, RES)), T.ToTensor()])

    obj_t = tf_img(img).unsqueeze(0).to(device)
    mask_t = tf_mask(mask)[0:1].unsqueeze(0).to(device)

    vae, uNet, scaling = load_models(CHECKPOINT)

    entries = []
    lighting_records = []

    # Original sample with ground truth
    pred_sh = predict_shadow(
        obj_t,
        mask_t,
        theta=meta["theta"],
        phi=meta["phi"],
        size=meta["size"],
        vae=vae,
        uNet=uNet,
        scaling=scaling,
    )

    gt_t = torch.Tensor(np.array(gt_sh.resize((RES, RES)))).unsqueeze(0).unsqueeze(0)
    obj_np = np.array(img.resize((RES, RES))).astype(np.uint8)
    mask_np = np.array(mask.resize((RES, RES))).astype(np.uint8)
    pred_np = pred_sh.squeeze().cpu().numpy().astype(np.float32)
    composite = composite_shadow(obj_np, mask_np, pred_np)

    entries.append(
        {
            "label": prefix,
            "object": obj_np,
            "mask": mask_np,
            "pred_shadow": pred_np,
            "composite": composite,
            "gt_shadow": gt_t.squeeze().numpy(),
            "theta": float(meta["theta"]),
            "phi": float(meta["phi"]),
            "size": float(meta["size"]),
        }
    )
    lighting_records.append(
        {
            "label": prefix,
            "theta": meta["theta"],
            "phi": meta["phi"],
            "size": meta["size"],
        }
    )

    # Additional custom lighting setups
    custom_light_setups = build_custom_lightings(meta, count=3)

    for custom in custom_light_setups:
        label = custom["label"]
        theta = float(custom["theta"])
        phi = float(custom["phi"])
        size = float(custom["size"])

        mask_use = mask_t
        obj_use = obj_t

        pred_custom = predict_shadow(
            obj_use,
            mask_use,
            theta=theta,
            phi=phi,
            size=size,
            vae=vae,
            uNet=uNet,
            scaling=scaling,
        )

        obj_np = np.array(img.resize((RES, RES))).astype(np.uint8)
        mask_np = np.array(mask.resize((RES, RES))).astype(np.uint8)
        pred_np = pred_custom.squeeze().cpu().numpy().astype(np.float32)
        composite = composite_shadow(obj_np, mask_np, pred_np)

        entries.append(
            {
                "label": label,
                "object": obj_np,
                "mask": mask_np,
                "pred_shadow": pred_np,
                "composite": composite,
                "theta": theta,
                "phi": phi,
                "size": size,
            }
        )
        lighting_records.append(
            {"label": label, "theta": theta, "phi": phi, "size": size}
        )

    visualize(lighting_records, entries, prefix)


if __name__ == "__main__":
    main()
