#!/usr/bin/env python3

import os
import tarfile
import json
from io import BytesIO

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from diffusers import AutoencoderKL, UNet2DConditionModel
from helpers import sinusoidalEmbedding


# ============================================================
# CONFIG
# ============================================================
RES = 256
BATCH_SIZE = 6
EPOCHS = 30
LR = 1e-4
DATA_TAR = "/home/razz/Downloads/jasperTestData/controllable-shadow-generation-benchmark.tar"
BEST_MODEL_PATH = "checkpoint_best.pth"
PROGRESS_PATH = "training_progress.json"
EARLY_STOP_PATIENCE = 3
MIN_LOSS_DELTA = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# DATASET (prefix-based grouping)
# ============================================================
class JasperBenchmark(Dataset):
    def __init__(self, tar_path):
        self.tar_path = tar_path
        self.members = []

        # Read all member names from tar
        with tarfile.open(tar_path, "r:*") as tf:
            for m in tf.getmembers():
                if m.isfile():
                    self.members.append(m.name)

        # Build groups by prefix
        # Example prefix: "softness_control_053"
        prefixes = {}
        for name in self.members:
            if name.endswith(".image.png"):
                prefix = name.replace(".image.png", "")
                prefixes.setdefault(prefix, {})["image"] = name
            elif name.endswith(".mask.png"):
                prefix = name.replace(".mask.png", "")
                prefixes.setdefault(prefix, {})["mask"] = name
            elif name.endswith(".shadow.png"):
                prefix = name.replace(".shadow.png", "")
                prefixes.setdefault(prefix, {})["shadow"] = name
            elif name.endswith(".metadata.json"):
                prefix = name.replace(".metadata.json", "")
                prefixes.setdefault(prefix, {})["meta"] = name

        # Keep only valid samples
        self.samples = [
            p for p, files in prefixes.items()
            if {"image", "mask", "shadow", "meta"}.issubset(files.keys())
        ]

        print(f"Dataset detected {len(self.samples)} valid samples.")

        # Transforms
        self.tf_img = T.Compose([T.Resize((RES, RES)), T.ToTensor()])
        self.tf_gray = T.Compose([T.Resize((RES, RES)), T.ToTensor()])

    def _load(self, tar, member):
        f = tar.extractfile(member)
        return f.read()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prefix = self.samples[idx]

        with tarfile.open(self.tar_path, "r:*") as tf:
            files = {
                "image": self._load(tf, prefix + ".image.png"),
                "mask": self._load(tf, prefix + ".mask.png"),
                "shadow": self._load(tf, prefix + ".shadow.png"),
                "meta": self._load(tf, prefix + ".metadata.json"),
            }

        # Decode images
        img = Image.open(BytesIO(files["image"])).convert("RGB")
        mask = Image.open(BytesIO(files["mask"])).convert("L")
        sh = Image.open(BytesIO(files["shadow"])).convert("L")

        # Apply transforms
        img = self.tf_img(img)
        mask = self.tf_gray(mask)[0:1]
        sh = self.tf_gray(sh)[0:1]

        # Metadata
        meta = json.loads(files["meta"].decode("utf-8"))
        theta = torch.tensor(meta["theta"], dtype=torch.float32)
        phi = torch.tensor(meta["phi"], dtype=torch.float32)
        size = torch.tensor(meta["size"], dtype=torch.float32)

        return {
            "objectRGB": img,
            "mask": mask,
            "shadow": sh,
            "theta": theta,
            "phi": phi,
            "size": size,
        }


# ============================================================
# TRAINER
# ============================================================
def main():
    print("Loading dataset...")
    dataset = JasperBenchmark(DATA_TAR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    vae.requires_grad_(False)
    scaling = vae.config.scaling_factor

    print("Building UNet...")
    uNet = UNet2DConditionModel(
        sample_size=RES // 8,
        in_channels=9,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(64,128,128,128),
        attention_head_dim=(64,64,64,64),
        cross_attention_dim=768,
    ).to(device)

    uNet.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(uNet.parameters(), lr=LR)
    mse = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()

    print("Training...")
    best_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        running_loss = 0.0
        batch_count = 0

        for batch in pbar:
            obj = batch["objectRGB"].to(device)
            msk = batch["mask"].to(device)
            shw = batch["shadow"].to(device)

            theta = batch["theta"].to(device)
            phi   = batch["phi"].to(device)
            size  = batch["size"].to(device)

            # Encode with VAE
            with torch.no_grad():
                sh_rgb = torch.cat([shw, shw, shw], 1)
                Zsh = vae.encode(sh_rgb * 2 - 1).latent_dist.mode() * scaling
                Zobj = vae.encode(obj * 2 - 1).latent_dist.mode() * scaling

            Zmsk = F.interpolate(msk, size=Zsh.shape[-2:])

            noise = torch.randn_like(Zsh)
            t = torch.rand(Zsh.shape[0], 1, 1, 1, device=device)
            latent = t * noise + (1 - t) * Zsh

            Zin = torch.cat([latent, Zobj, Zmsk], dim=1)

            embed = torch.cat([
                sinusoidalEmbedding(theta),
                sinusoidalEmbedding(phi),
                sinusoidalEmbedding(size)
            ], dim=1)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                pred = uNet(
                    sample=Zin,
                    timestep=(t[:,0,0,0] * 1000).long(),
                    encoder_hidden_states=embed.unsqueeze(1)
                ).sample

                target = noise - Zsh
                loss = mse(pred, target)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_item = loss.item()
            running_loss += loss_item
            batch_count += 1
            avg_so_far = running_loss / batch_count
            pbar.set_postfix(loss=loss_item, avg=avg_so_far)

        if batch_count == 0:
            print("⚠️ No batches processed this epoch; stopping.")
            break

        epoch_loss = running_loss / batch_count
        history.append({"epoch": epoch, "avg_loss": epoch_loss})
        with open(PROGRESS_PATH, "w", encoding="utf-8") as progress_file:
            json.dump(history, progress_file, indent=2)

        if best_loss - epoch_loss > MIN_LOSS_DELTA:
            best_loss = epoch_loss
            epochs_without_improvement = 0
            torch.save(uNet.state_dict(), BEST_MODEL_PATH)
            print(f"Saved new best model to {BEST_MODEL_PATH} (avg_loss={epoch_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"No improvement (avg_loss={epoch_loss:.4f}, best={best_loss:.4f}) - patience {epochs_without_improvement}/{EARLY_STOP_PATIENCE}")

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print("Early stopping triggered; halting training.")
            break

    print("Training complete.")


if __name__ == "__main__":
    main()
