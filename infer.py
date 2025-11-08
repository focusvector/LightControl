import torch
import matplotlib.pyplot as plt
from helpers import sinusoidalEmbedding
from dataset import ToyShadowDataset, ToyShadowDataset3D
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel

DATASET_CLASS = ToyShadowDataset3D  # switch to 3D dataset for depth-aware sampling

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

try:
    state_dict = torch.load("toyJasperE7.pth", map_location=device, weights_only=True)
except TypeError:
    state_dict = torch.load("toyJasperE7.pth", map_location=device)
uNet.load_state_dict(state_dict)
uNet.eval()

# ---- sample one example ----
data = DATASET_CLASS(n=1)
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
panels = [
    {
        "image": obj[0].permute(1, 2, 0).detach().cpu(),
        "title": "object",
        "cmap": None
    },
    {
        "image": x["shadow"][0].detach().cpu(),
        "title": "GT shadow",
        "cmap": "gray"
    },
    {
        "image": shadow_pred[0, 0].detach().cpu(),
        "title": "Pred shadow",
        "cmap": "gray"
    }
]

depth_map = None
if "depth" in x:
    depth_map = x["depth"][0].detach().cpu()
    panels.append({
        "image": depth_map,
        "title": "Depth",
        "cmap": "magma"
    })

fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
if len(panels) == 1:
    axes = [axes]

for ax, panel in zip(axes, panels):
    if panel["cmap"]:
        ax.imshow(panel["image"], cmap=panel["cmap"])
    else:
        ax.imshow(panel["image"])
    ax.set_title(panel["title"])
    ax.axis("off")

plt.tight_layout()
plt.show()

if depth_map is not None:
    y_coords = torch.linspace(0, depth_map.shape[0] - 1, depth_map.shape[0])
    x_coords = torch.linspace(0, depth_map.shape[1] - 1, depth_map.shape[1])
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

    fig3d = plt.figure(figsize=(6, 5))
    ax3d = fig3d.add_subplot(111, projection="3d")
    ax3d.plot_surface(
        grid_x.numpy(),
        grid_y.numpy(),
        depth_map.numpy(),
        cmap="viridis",
        edgecolor="none"
    )
    ax3d.set_title("Depth surface")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Depth")
    plt.tight_layout()
    plt.show()