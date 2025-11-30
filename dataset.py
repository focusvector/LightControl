import random
import math
import os
import json
from pathlib import Path

# PyTorch imports - only needed for dataset loading
try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Dummy Dataset class for when torch isn't available
    class Dataset:
        pass

# Blender imports - only needed for shape generation
try:
    import bpy
    from mathutils import Vector
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False

# PIL for image loading
try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

class LightDataset(Dataset):
    """
    File-based dataset that loads rendered images from disk.
    Expected directory structure:
        dataset_dir/
            meta_00000.json
            render_00000.png
            mask_00000.png
            shadow_00000.png
            ...
    
    Each meta_*.json contains:
        {"theta": float, "phi": float, "size": float, "model": str, ...}
    """
    def __init__(self, dataset_dir, res=128, shadow_res=32):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available. This class requires torch.")
        if not PIL_AVAILABLE:
            raise RuntimeError("PIL not available. This class requires Pillow.")
        
        self.dataset_dir = Path(dataset_dir)
        self.res = res
        self.shadow_res = shadow_res
        
        # Find all meta files
        self.meta_files = sorted(self.dataset_dir.glob("meta_*.json"))
        if len(self.meta_files) == 0:
            raise RuntimeError(f"No meta_*.json files found in {dataset_dir}")
        
        print(f"LightDataset: Found {len(self.meta_files)} samples in {dataset_dir}")
    
    def __len__(self):
        return len(self.meta_files)
    
    def __getitem__(self, idx):
        meta_path = self.meta_files[idx]
        base_name = meta_path.stem.replace("meta_", "")
        
        # Load metadata
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        # Load images
        render_path = self.dataset_dir / f"render_{base_name}.png"
        mask_path = self.dataset_dir / f"mask_{base_name}.png"
        shadow_path = self.dataset_dir / f"shadow_{base_name}.png"
        
        # Load and resize RGB
        rgb = Image.open(render_path).convert("RGB").resize((self.res, self.res))
        rgb_tensor = torch.from_numpy(np.array(rgb)).float().permute(2, 0, 1) / 255.0
        
        # Load and resize mask
        mask = Image.open(mask_path).convert("L").resize((self.res, self.res))
        mask_tensor = torch.from_numpy(np.array(mask)).float().unsqueeze(0) / 255.0
        
        # Load shadow at full resolution and downsampled
        shadow_full = Image.open(shadow_path).convert("L").resize((self.res, self.res))
        shadow_full_tensor = torch.from_numpy(np.array(shadow_full)).float().unsqueeze(0) / 255.0
        
        shadow_down = Image.open(shadow_path).convert("L").resize((self.shadow_res, self.shadow_res))
        shadow_down_tensor = torch.from_numpy(np.array(shadow_down)).float().unsqueeze(0) / 255.0
        
        return {
            "RGB": rgb_tensor,
            "mask": mask_tensor,
            "shadow_full": shadow_full_tensor,
            "shadow": shadow_down_tensor,
            "theta": torch.tensor(meta.get("theta", 45.0)),
            "phi": torch.tensor(meta.get("phi", 90.0)),
            "size": torch.tensor(meta.get("size", 1.0)),
        }

class ToyShadowDataset(Dataset):
    """
    Synthetic dataset: simple squares with analytic shadow maps.
    Returns: object RGB, mask, shadow map, and light parameters (θ, φ, s)
    
    Note: Requires PyTorch. Not available when running in Blender.
    """
    def __init__(self,n=2000,size=128):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available. This class requires torch.")
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
    """
    Synthetic 3D dataset with procedural shapes (cube/sphere/pyramid).
    
    Note: Requires PyTorch. Not available when running in Blender.
    """
    def __init__(self,n=2000, size=128):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available. This class requires torch.")
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


# ============================================================================
# BLENDER-BASED 3D SHAPE GENERATION
# ============================================================================
# These functions generate diverse 3D shapes using Blender's bpy module.
# They are compatible with the dataset_generator.py rendering pipeline.
#
# Usage:
#   from dataset import generate_shape_library
#   generate_shape_library(output_dir="./models", count=50)
#
# Then use dataset_generator.py to render shadows:
#   blender --background --python dataset_generator.py -- \
#       --models ./models --output ./dataset --samples 128
# ============================================================================

def clear_blender_scene():
    """Remove all objects from the Blender scene."""
    if not BLENDER_AVAILABLE:
        raise RuntimeError("Blender (bpy) not available. Run this script inside Blender.")
    
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for coll in (bpy.data.meshes, bpy.data.materials):
        for datablock in list(coll):
            if datablock.users == 0:
                coll.remove(datablock)


def create_cube_variant(scale_x=1.0, scale_y=1.0, scale_z=1.0):
    """Create a cube with variable dimensions."""
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.scale = (scale_x, scale_y, scale_z)
    bpy.ops.object.transform_apply(scale=True)
    return obj


def create_sphere_variant(subdivisions=2):
    """Create a UV sphere with variable subdivisions."""
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0, 0, 0))
    obj = bpy.context.active_object
    if subdivisions > 0:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        for _ in range(subdivisions):
            bpy.ops.mesh.subdivide()
        bpy.ops.object.mode_set(mode='OBJECT')
    return obj


def create_cylinder_variant(radius=1.0, depth=2.0, vertices=32):
    """Create a cylinder with variable dimensions."""
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius, depth=depth, vertices=vertices, location=(0, 0, 0)
    )
    return bpy.context.active_object


def create_cone_variant(radius1=1.0, radius2=0.0, depth=2.0, vertices=32):
    """Create a cone or frustum."""
    bpy.ops.mesh.primitive_cone_add(
        radius1=radius1, radius2=radius2, depth=depth, 
        vertices=vertices, location=(0, 0, 0)
    )
    return bpy.context.active_object


def create_torus_variant(major_radius=1.0, minor_radius=0.25):
    """Create a torus."""
    bpy.ops.mesh.primitive_torus_add(
        major_radius=major_radius, minor_radius=minor_radius, location=(0, 0, 0)
    )
    return bpy.context.active_object


def create_ico_sphere_variant(subdivisions=2, radius=1.0):
    """Create an icosphere (more uniform than UV sphere)."""
    bpy.ops.mesh.primitive_ico_sphere_add(
        subdivisions=subdivisions, radius=radius, location=(0, 0, 0)
    )
    return bpy.context.active_object


def create_rounded_cube(radius=1.0, subdivisions=2):
    """Create a cube with subdivision surface for rounded edges."""
    bpy.ops.mesh.primitive_cube_add(size=2*radius, location=(0, 0, 0))
    obj = bpy.context.active_object
    mod = obj.modifiers.new(name="Subsurf", type='SUBSURF')
    mod.levels = subdivisions
    mod.render_levels = subdivisions
    bpy.ops.object.modifier_apply(modifier="Subsurf")
    return obj


def create_pyramid():
    """Create a 4-sided pyramid."""
    bpy.ops.mesh.primitive_cone_add(
        vertices=4, radius1=1.0, radius2=0.0, depth=2.0, location=(0, 0, 0)
    )
    obj = bpy.context.active_object
    obj.rotation_euler[2] = math.radians(45)
    bpy.ops.object.transform_apply(rotation=True)
    return obj


def create_prism(sides=6, radius=1.0, depth=2.0):
    """Create a prism with variable number of sides."""
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=sides, radius=radius, depth=depth, location=(0, 0, 0)
    )
    return bpy.context.active_object


def create_l_shape():
    """Create an L-shaped object by joining two boxes."""
    # Vertical part
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.75))
    vertical = bpy.context.active_object
    vertical.scale = (0.3, 0.3, 1.5)
    bpy.ops.object.transform_apply(scale=True)
    
    # Horizontal part
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0.6, 0, 0))
    horizontal = bpy.context.active_object
    horizontal.scale = (1.2, 0.3, 0.3)
    bpy.ops.object.transform_apply(scale=True)
    
    # Join
    bpy.context.view_layer.objects.active = vertical
    horizontal.select_set(True)
    bpy.ops.object.join()
    return vertical


def create_cross_shape():
    """Create a cross/plus shape."""
    # Vertical bar
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
    vertical = bpy.context.active_object
    vertical.scale = (0.3, 0.3, 1.5)
    bpy.ops.object.transform_apply(scale=True)
    
    # Horizontal bar
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.5))
    horizontal = bpy.context.active_object
    horizontal.scale = (1.2, 0.3, 0.3)
    bpy.ops.object.transform_apply(scale=True)
    
    # Join
    bpy.context.view_layer.objects.active = vertical
    horizontal.select_set(True)
    bpy.ops.object.join()
    return vertical


def create_organic_blob(subdivisions=3, displacement_strength=0.3):
    """Create an organic blob using ico sphere + displacement."""
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=subdivisions, radius=1.0, location=(0, 0, 0))
    obj = bpy.context.active_object
    
    # Add displacement modifier with random noise
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    
    # Subdivide more for organic feel
    for _ in range(random.randint(1, 2)):
        bpy.ops.mesh.subdivide()
    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Add displacement modifier
    mod = obj.modifiers.new(name="Displace", type='DISPLACE')
    
    # Create noise texture
    tex = bpy.data.textures.new(name="NoiseDisplace", type='VORONOI')
    tex.noise_scale = random.uniform(1.5, 4.0)
    mod.texture = tex
    mod.strength = displacement_strength
    
    bpy.ops.object.modifier_apply(modifier="Displace")
    
    # Smooth it out
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.faces_shade_smooth()
    bpy.ops.object.mode_set(mode='OBJECT')
    
    return obj


def create_deformed_shape(base_shape='cube'):
    """Create a deformed version of a base shape."""
    if base_shape == 'cube':
        bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
    elif base_shape == 'sphere':
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0, 0, 0))
    else:
        bpy.ops.mesh.primitive_cylinder_add(radius=1.0, depth=2.0, location=(0, 0, 0))
    
    obj = bpy.context.active_object
    
    # Add simple deform modifier
    mod = obj.modifiers.new(name="SimpleDeform", type='SIMPLE_DEFORM')
    mod.deform_method = random.choice(['TWIST', 'BEND', 'TAPER', 'STRETCH'])
    mod.angle = random.uniform(-math.pi/2, math.pi/2)
    mod.factor = random.uniform(0.3, 1.5)
    
    bpy.ops.object.modifier_apply(modifier="SimpleDeform")
    return obj


def create_metaball_blob():
    """Create an organic metaball blob."""
    # Create main metaball
    bpy.ops.object.metaball_add(type='BALL', radius=1.0, location=(0, 0, 0))
    mb_main = bpy.context.active_object
    mb_main.name = "MetaballMain"
    
    # Add random metaballs to same family
    for i in range(random.randint(2, 5)):
        offset = (random.uniform(-0.8, 0.8), 
                  random.uniform(-0.8, 0.8), 
                  random.uniform(-0.8, 0.8))
        bpy.ops.object.metaball_add(
            type=random.choice(['BALL', 'ELLIPSOID']),
            radius=random.uniform(0.3, 0.7),
            location=offset
        )
        # Metaballs in same family get auto-grouped
    
    # Select all metaballs and convert to mesh
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type == 'META':
            obj.select_set(True)
    
    bpy.context.view_layer.objects.active = mb_main
    bpy.ops.object.convert(target='MESH')
    
    # Get the converted mesh object
    mesh_obj = bpy.context.active_object
    
    # Delete any remaining metaballs
    for obj in list(bpy.context.scene.objects):
        if obj.type == 'META':
            bpy.data.objects.remove(obj, do_unlink=True)
    
    return mesh_obj


def create_random_material():
    """Create a random colored material with optional texture."""
    mat = bpy.data.materials.new(name="RandomMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear default nodes
    nodes.clear()
    
    # Add output node
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (400, 0)
    
    # Add principled BSDF
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    
    # Random base color
    color_type = random.choice(['solid', 'gradient', 'noise', 'checker'])
    
    if color_type == 'solid':
        # Solid random color
        bsdf.inputs['Base Color'].default_value = (
            random.uniform(0.2, 1.0),  # R
            random.uniform(0.2, 1.0),  # G
            random.uniform(0.2, 1.0),  # B
            1.0  # A
        )
    
    elif color_type == 'gradient':
        # Gradient texture
        coord = nodes.new(type='ShaderNodeTexCoord')
        coord.location = (-600, 0)
        
        gradient = nodes.new(type='ShaderNodeTexGradient')
        gradient.location = (-400, 0)
        
        color_ramp = nodes.new(type='ShaderNodeValToRGB')
        color_ramp.location = (-200, 0)
        
        # Random colors for gradient
        color_ramp.color_ramp.elements[0].color = (
            random.uniform(0.2, 1.0),
            random.uniform(0.2, 1.0),
            random.uniform(0.2, 1.0),
            1.0
        )
        color_ramp.color_ramp.elements[1].color = (
            random.uniform(0.2, 1.0),
            random.uniform(0.2, 1.0),
            random.uniform(0.2, 1.0),
            1.0
        )
        
        links.new(coord.outputs['Generated'], gradient.inputs['Vector'])
        links.new(gradient.outputs['Fac'], color_ramp.inputs['Fac'])
        links.new(color_ramp.outputs['Color'], bsdf.inputs['Base Color'])
    
    elif color_type == 'noise':
        # Noise texture
        coord = nodes.new(type='ShaderNodeTexCoord')
        coord.location = (-600, 0)
        
        noise = nodes.new(type='ShaderNodeTexNoise')
        noise.location = (-400, 0)
        noise.inputs['Scale'].default_value = random.uniform(2.0, 10.0)
        noise.inputs['Detail'].default_value = random.uniform(2.0, 8.0)
        
        color_ramp = nodes.new(type='ShaderNodeValToRGB')
        color_ramp.location = (-200, 0)
        
        color_ramp.color_ramp.elements[0].color = (
            random.uniform(0.2, 1.0),
            random.uniform(0.2, 1.0),
            random.uniform(0.2, 1.0),
            1.0
        )
        color_ramp.color_ramp.elements[1].color = (
            random.uniform(0.2, 1.0),
            random.uniform(0.2, 1.0),
            random.uniform(0.2, 1.0),
            1.0
        )
        
        links.new(coord.outputs['Generated'], noise.inputs['Vector'])
        links.new(noise.outputs['Fac'], color_ramp.inputs['Fac'])
        links.new(color_ramp.outputs['Color'], bsdf.inputs['Base Color'])
    
    elif color_type == 'checker':
        # Checker pattern
        coord = nodes.new(type='ShaderNodeTexCoord')
        coord.location = (-600, 0)
        
        checker = nodes.new(type='ShaderNodeTexChecker')
        checker.location = (-200, 0)
        checker.inputs['Scale'].default_value = random.uniform(2.0, 8.0)
        
        # Random colors for checker
        checker.inputs['Color1'].default_value = (
            random.uniform(0.2, 1.0),
            random.uniform(0.2, 1.0),
            random.uniform(0.2, 1.0),
            1.0
        )
        checker.inputs['Color2'].default_value = (
            random.uniform(0.2, 1.0),
            random.uniform(0.2, 1.0),
            random.uniform(0.2, 1.0),
            1.0
        )
        
        links.new(coord.outputs['Generated'], checker.inputs['Vector'])
        links.new(checker.outputs['Color'], bsdf.inputs['Base Color'])
    
    # Random material properties
    bsdf.inputs['Roughness'].default_value = random.uniform(0.2, 1.0)
    bsdf.inputs['Metallic'].default_value = random.choice([0.0, 0.0, 0.0, random.uniform(0.5, 1.0)])  # Mostly non-metallic
    
    # Link to output
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    return mat


def create_random_blender_shape():
    """
    Create a random 3D shape using Blender primitives and organic modifiers.
    Returns: (object, shape_type_string)
    """
    if not BLENDER_AVAILABLE:
        raise RuntimeError("Blender (bpy) not available")
    
    shape_type = random.choice([
        # Basic primitives (30%)
        'cube', 'stretched_cube', 'sphere', 'cylinder', 'cone',
        'torus', 'ico_sphere', 'rounded_cube', 'pyramid', 'prism',
        # Compound shapes (20%)  
        'l_shape', 'cross',
        # Organic shapes (50% - more variety!)
        'blob', 'blob', 'blob', 'blob', 'blob',
        'deformed_cube', 'deformed_sphere', 'deformed_cylinder',
        'metaball', 'metaball'
    ])
    
    if shape_type == 'cube':
        obj = create_cube_variant(
            scale_x=random.uniform(0.5, 1.5),
            scale_y=random.uniform(0.5, 1.5),
            scale_z=random.uniform(0.5, 1.5)
        )
    elif shape_type == 'stretched_cube':
        axis = random.choice(['x', 'y', 'z'])
        scales = {'x': 1.0, 'y': 1.0, 'z': 1.0}
        scales[axis] = random.uniform(1.5, 3.0)
        obj = create_cube_variant(
            scale_x=scales['x'], scale_y=scales['y'], scale_z=scales['z']
        )
        shape_type = f"cube_stretch_{axis}"
    elif shape_type == 'sphere':
        obj = create_sphere_variant(subdivisions=random.randint(1, 2))
    elif shape_type == 'cylinder':
        obj = create_cylinder_variant(
            radius=random.uniform(0.6, 1.2),
            depth=random.uniform(1.5, 2.5),
            vertices=random.choice([16, 32, 64])
        )
    elif shape_type == 'cone':
        obj = create_cone_variant(
            radius1=random.uniform(0.8, 1.5),
            radius2=random.uniform(0, 0.3),
            depth=random.uniform(1.5, 2.5)
        )
    elif shape_type == 'torus':
        obj = create_torus_variant(
            major_radius=random.uniform(0.8, 1.2),
            minor_radius=random.uniform(0.2, 0.4)
        )
    elif shape_type == 'ico_sphere':
        obj = create_ico_sphere_variant(
            subdivisions=random.randint(1, 3),
            radius=random.uniform(0.8, 1.2)
        )
    elif shape_type == 'rounded_cube':
        obj = create_rounded_cube(
            radius=random.uniform(0.7, 1.1),
            subdivisions=random.randint(1, 2)
        )
    elif shape_type == 'pyramid':
        obj = create_pyramid()
    elif shape_type == 'prism':
        obj = create_prism(
            sides=random.choice([3, 5, 6, 8]),
            radius=random.uniform(0.8, 1.2),
            depth=random.uniform(1.5, 2.5)
        )
    elif shape_type == 'l_shape':
        obj = create_l_shape()
    elif shape_type == 'cross':
        obj = create_cross_shape()
    elif shape_type == 'blob':
        obj = create_organic_blob(
            subdivisions=random.randint(2, 3),
            displacement_strength=random.uniform(0.2, 0.5)
        )
    elif shape_type == 'deformed_cube':
        obj = create_deformed_shape('cube')
    elif shape_type == 'deformed_sphere':
        obj = create_deformed_shape('sphere')
    elif shape_type == 'deformed_cylinder':
        obj = create_deformed_shape('cylinder')
    elif shape_type == 'metaball':
        obj = create_metaball_blob()
    else:
        obj = create_cube_variant()
        shape_type = 'cube'
    
    return obj, shape_type


def center_and_normalize_object(obj, target_size=1.0, add_material=True):
    """Center object at origin, normalize its scale, and optionally add random material."""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (0, 0, 0)
    bpy.context.view_layer.update()
    
    # Normalize scale
    dims = obj.dimensions
    max_dim = max(dims.x, dims.y, dims.z)
    if max_dim > 0:
        scale_factor = target_size / max_dim
        obj.scale = (obj.scale.x * scale_factor,
                     obj.scale.y * scale_factor,
                     obj.scale.z * scale_factor)
        bpy.ops.object.transform_apply(scale=True)
    
    # Random rotation around Z
    obj.rotation_euler[2] = random.uniform(0, 2 * math.pi)
    bpy.ops.object.transform_apply(rotation=True)
    
    # Add random material/color
    if add_material:
        mat = create_random_material()
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)


def generate_shape_library(output_dir="./models", count=50, seed=None):
    """
    Generate a library of diverse 3D shapes as .blend files.
    
    Args:
        output_dir: Directory to save .blend files
        count: Number of shapes to generate
        seed: Random seed for reproducibility
    
    Returns:
        List of dicts with shape info: [{'index': 0, 'type': 'cube', 'path': '...'}]
    
    Example:
        # Run inside Blender:
        import bpy
        import sys
        sys.path.append('/path/to/toyJasper')
        from dataset import generate_shape_library
        
        shapes = generate_shape_library(output_dir="./models", count=100)
        print(f"Generated {len(shapes)} shapes")
    """
    if not BLENDER_AVAILABLE:
        raise RuntimeError(
            "Blender (bpy) not available. This function must be run inside Blender.\n"
            "Usage: blender --background --python -c \"from dataset import generate_shape_library; "
            "generate_shape_library('./models', 100)\""
        )
    
    if seed is not None:
        random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    generated = []
    
    print("=" * 60)
    print(f"Generating {count} diverse 3D shapes...")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    for i in range(count):
        # Clear scene
        clear_blender_scene()
        
        # Create random shape
        obj, shape_type = create_random_blender_shape()
        obj.name = f"Shape_{i:04d}"
        
        # Process object
        center_and_normalize_object(obj, target_size=random.uniform(0.8, 1.2))
        
        # Save .blend file
        filepath = os.path.join(output_dir, f"shape_{i:04d}_{shape_type}.blend")
        bpy.ops.wm.save_as_mainfile(filepath=filepath)
        
        generated.append({
            'index': i,
            'type': shape_type,
            'path': filepath
        })
        
        print(f"[{i+1}/{count}] {shape_type:20s} -> {os.path.basename(filepath)}")
    
    # Save manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(generated, f, indent=2)
    
    print("=" * 60)
    print(f"✓ Generated {len(generated)} shapes")
    print(f"✓ Manifest: {manifest_path}")
    print("=" * 60)
    print("\nNext steps:")
    print(f"1. Render shadow dataset:")
    print(f"   blender --background --python blendify/dataset_generator.py -- \\")
    print(f"       --models {output_dir} \\")
    print(f"       --output ./dataset \\")
    print(f"       --samples 128")
    print(f"\n2. Train model:")
    print(f"   python train.py")
    
    return generated


# ============================================================================
# USAGE EXAMPLE FOR BLENDER
# ============================================================================
# To generate shapes, run this script inside Blender:
#
# Method 1 - Command line:
#   blender --background --python dataset.py
#
# Method 2 - Python console in Blender:
#   import sys
#   sys.path.append('/path/to/toyJasper')
#   from dataset import generate_shape_library
#   generate_shape_library(output_dir="./models", count=100)
#
# Method 3 - Standalone script:
#   Create a separate script that imports this module
# ============================================================================

if __name__ == "__main__" and BLENDER_AVAILABLE:
    # Default generation when run as: blender --background --python dataset.py
    import sys
    
    # Parse simple args
    output_dir = "./models"
    count = 50
    
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_dir = sys.argv[idx + 1]
    
    if "--count" in sys.argv:
        idx = sys.argv.index("--count")
        if idx + 1 < len(sys.argv):
            count = int(sys.argv[idx + 1])
    
    generate_shape_library(output_dir=output_dir, count=count)