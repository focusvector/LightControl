import torch
from torch.utils.data import Dataset
import random
import math
class ToyShadowDataset(Dataset):
    """
    Synthetic dataset: simple squares with analytic shadow maps.
    Returns: object RGB, mask, shadow map, and light parameters (θ, φ, s)
    """
    def __init__(self,n=2000,size=128):
        self.n, self.size = n, size
        self.pi = math.pi
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
        theta = torch.rand(1)*self.pi #pie
        phi = torch.rand(1)*self.pi*2
        size = torch.rand(1)*self.pi*2

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
    
class ToyShadowDataset3D(Dataset):
    def __init__(self,n=2000, size=128):
        self.n = n
        self.size = size
        self.shapes = ["cube","sphere","pyramid"]
        self.pi = math.pi

    def __len__(self):
        return self.n
        
    def __getitem__(self, index):
        S =self.size
        obj = torch.zeros(3, S, S)
        mask = torch.zeros(1, S, S)
        depth = torch.zeros(3, S, S)

        # choosing random shape
        shape = random.choice(self.shapes)

        # random centre and scale
        cx, cy = random.randint(40, 80), random.randint(40, 80)
        r = random.randint(10, 25) 

        Y, X = torch.meshgrid(
            torch.arange(S, dtype=torch.float32),
            torch.arange(S, dtype=torch.float32),
            indexing="ij"
        )

        # --- create height/depth field depending on shape type ---
        if shape == "cube":
            inside = (X > cx - r) & (X < cx + r) & (Y > cy - r) & (Y < cy + r)
            depth[0, inside] = r

        elif shape == "sphere":
            Z = r**2 - ((X - cx) ** 2 + (Y - cy) ** 2)
            Z = torch.clamp(Z, min=0).sqrt()
            depth[0] = Z

        elif shape == "pyramid":
            dx = torch.abs(X - cx) / r
            dy = torch.abs(Y - cy) / r
            depth[0] = torch.clamp(r * (1 - (dx + dy)), min=0)

        mask[0] = (depth[0] > 0).float()
        obj[0] = mask[0] * 0.8  # gray cube
        obj[1] = mask[0] * 0.8
        obj[2] = mask[0] * 0.8

        # --- light direction ---
        theta = torch.rand(1) * (self.pi / 2)  # elevation (0–90°)
        phi = torch.rand(1) * (2 * self.pi)    # azimuth (0–360°)
        size = torch.rand(1) * self.pi * 2     # softness parameter

        theta_val = theta.item()
        phi_val = phi.item()

        Lx = math.cos(phi_val) * math.cos(theta_val)
        Ly = math.sin(phi_val) * math.cos(theta_val)
        Lz = math.sin(theta_val)
        Lz = Lz if abs(Lz) > 1e-3 else 1e-3  # avoid division by ~0

        # --- shadow projection ---
        shadow = torch.zeros_like(mask)
        for yy in range(S):
            for xx in range(S):
                z = depth[0, yy, xx]
                if z > 0:
                    xs = int(xx - (z / Lz) * Lx)
                    ys = int(yy - (z / Lz) * Ly)
                    if 0 <= xs < S and 0 <= ys < S:
                        shadow[0, ys, xs] = 1.0

        return {
            "objectRGB": obj.float(),
            "mask": mask.float(),
            "shadow": shadow.float(),
            "depth": depth.float(),
            "theta": theta.float(),
            "phi": phi.float(),
            "size": size.float()
        }