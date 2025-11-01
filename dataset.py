import torch
from torch.utils.data import Dataset
import random

class ToyShadowDataset(Dataset):
    """
    Synthetic dataset: simple squares with analytic shadow maps.
    Returns: object RGB, mask, shadow map, and light parameters (θ, φ, s)
    """
    def __init__(self,n=2000,size=128):
        self.n, self.size = n, size

    def __len__(self): return self.n

    def __getitem__(self, idx):
        S = self.size
        obj = torch.zeros(3, S, S)
        mask = torch.zeros(1, S, S)

        # random square
        x, y  = random.randint(20,80), random.randint(20,80)
        w = random.randint(15,30)
        obj[:, y:y+w, x:x+w] = 1.0
        mask[:, y:y+w, x:x+w] = 1.0

        # random light direction + softness
        theta = torch.rand(1)*3.14
        phi = torch.rand(1)*6.28
        size = torch.rand(1)*6.28

        # simple shadow
        dx, dy = int(15*torch.cos(phi)), int(15*torch.sin(phi))
        shadow = torch.zeros_like(mask)
        y0, y1 = max(0, y + dy), min(S, y + dy + w)
        x0, x1 = max(0, x + dx), min(S, x + dx + w)
        if y0 < y1 and x0 < x1:
            shadow[:, y0:y1, x0:x1] = 1.0

        return {
            "objectRGB": obj.float(),
            "mask": mask.float(),
            "shadow": shadow.float(),
            "theta": theta.float(),
            "phi": phi.float(),
            "size": size.float()
        }