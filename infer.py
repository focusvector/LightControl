import torch
import matplotlib.pyplot as plt
from helpers import sinusoidalEmbedding
from dataset import ToyShadowDataset
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- load models ----
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
vae.eval(); vae.requires_grad_(False)
scaling = vae.config.scaling_factor

uNet = UNet2DConditionModel(
    sample_size=16,
    in_channels=9,
    out_channels=4,
    layers_per_block=1,
    block_out_channels=(64,128,128,128),
    attention_head_dim=(64,64,64,64),
    cross_attention_dim=768  # align with concatenated sinusoidal embeddings
).to(device)

uNet.load_state_dict(torch.load("toyJasperE10.pth", map_location=device))
uNet.eval()

# ---- sample one example ----
data = ToyShadowDataset(n=1)
x = data[1]
obj = x["objectRGB"].unsqueeze(0).to(device)
msk = x["mask"].unsqueeze(0).to(device)
theta, phi, size = [x[k].to(device) for k in ("theta","phi","size")]

with torch.no_grad():
    Zobj = vae.encode(obj*2-1).latent_dist.mode() * scaling
    Zmsk = F.interpolate(msk, size=Zobj.shape[-2:])
    noise = torch.randn_like(Zobj)
    Zin = torch.cat([noise, Zobj, Zmsk], 1)

    embed = torch.cat([
            sinusoidalEmbedding(theta),
            sinusoidalEmbedding(phi),
            sinusoidalEmbedding(size)],
            dim=1
        )
    embed = embed.to(device)
    pred = uNet(sample = Zin, timestep = torch.tensor([999], device=device),
                encoder_hidden_states = embed.unsqueeze(1)).sample
    
    z0 = noise - pred
    shadow_rgb = vae.decode(z0 / scaling).sample
    shadow_pred = shadow_rgb[:,0:1]

# --- visualize ---
plt.subplot(1,3,1); plt.imshow(obj[0].permute(1,2,0).cpu()); plt.title("object")
plt.subplot(1,3,2); plt.imshow(x["shadow"][0], cmap="gray"); plt.title("GT shadow")
plt.subplot(1,3,3); plt.imshow(shadow_pred[0,0].cpu(), cmap="gray"); plt.title("Pred shadow")
plt.show()