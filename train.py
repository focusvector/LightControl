import torch, torch.nn as nn
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL, UNet2DConditionModel
from tqdm import tqdm
from helpers import sinusoidalEmbedding
from dataset import ToyShadowDataset # my dataset

# ---- config ----
BATCH_SIZE = 2
EPOCHS = 100
NUM_WORKERS = 2
####
device = "cuda" if torch.cuda.is_available() else "cpu"
####

def main():
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval(); vae.requires_grad_(False)
    scaling = vae.config.scaling_factor

    uNet = UNet2DConditionModel(
        sample_size=16,
        in_channels=9,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(64, 128, 128, 128),
        attention_head_dim=(64, 64, 64, 64),
        cross_attention_dim=768  # match three 256-dim sinusoidal embeddings
    ).to(device)

    optimizer = torch.optim.AdamW(uNet.parameters(), lr = 1e-4)
    mse = nn.MSELoss()

    loader = DataLoader(
        ToyShadowDataset(n=2000, size=128),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    # ---- training loop (Rectified Flow) ----
    for epoch in range(EPOCHS):
        for batch in tqdm(loader, desc=f"epoch {epoch}"):
            obj = batch["objectRGB"].to(device) #cropped object image (no background)
            msk = batch["mask"].to(device).float() # binary mask 
            shw = batch["shadow"].to(device) # target shadow map (ground truth)
            theta, phi, size = [batch[k].to(device).float().view(-1) for k in ("theta","phi","size")]

            with torch.no_grad():
                Zshw = vae.encode(torch.cat([shw,shw,shw],1)*2-1).latent_dist.mode() * scaling
                Zobj = vae.encode(obj*2-1).latent_dist.mode() * scaling
            Zmsk = nn.functional.interpolate(msk, size=Zshw.shape[-2:])   

            # random noise and interpolation
            noise = torch.randn_like(Zshw)
            factor = torch.rand(Zshw.shape[0],1,1,1, device=device)
            latent = factor*noise + (1-factor)*Zshw

            # concatenate conditioning channels (2c+1)
            Zin = torch.cat([latent, Zobj, Zmsk], dim=1)

            # embed light parameters
            embed = torch.cat([
                sinusoidalEmbedding(theta),
                sinusoidalEmbedding(phi),
                sinusoidalEmbedding(size)],
                dim=1
            )

            # forward pass
            pred = uNet(sample=Zin,
                        timestep = (factor[:,0,0,0]*1000).long(),
                        encoder_hidden_states = embed.unsqueeze(1)).sample
        
            target = noise - Zshw
            loss = mse(pred, target)

            optimizer.zero_grad(); loss.backward(); optimizer.step()

        print(f"Epoch {epoch} || Loss: {loss.item():.4f}")
        torch.save(uNet.state_dict(), f"toyJasperE{epoch}.pth")

if __name__ == "__main__":
    main()