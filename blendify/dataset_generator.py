"""
jasper_dataset_generator.py
------------------------------------------------------------

This script generates synthetic images (RGB, mask, shadow)
using Blender's Cycles renderer, conditioned on light
parameters (θ, φ, s) for training diffusion-based shadow
models.


------------------------------------------------------------
Usage:
    blender --background --python dataset_generator.py -- \
        --models /path/to/models \
        --output /path/to/output \
        --samples 128
------------------------------------------------------------
"""

import bpy
import os
import math
import json
import random
import sys
import argparse
from mathutils import Vector

# ------------------------------------------------------------
# Global parameters
# ------------------------------------------------------------
LIGHT_RADIUS = 8.0          # r from the paper (sphere radius)
IMG_RES = 1024              # output image resolution
MASK_MAT_NAME = "_MaskMat"  # special white material for binary mask
VARIATIONS_PER_MODEL = 2 # set to 1 for a single render per model


# ------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------
def get_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True, help="Path to folder with .obj/.fbx/.blend models")
    parser.add_argument("--output", required=True, help="Output folder for rendered images")
    parser.add_argument("--samples", type=int, default=128, help="Number of Cycles samples (render quality)")
    return parser.parse_args(argv)


# ------------------------------------------------------------
# Scene setup and cleanup
# ------------------------------------------------------------
def clear_scene():
    """Remove all objects, lights, and data blocks."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.images,
                 bpy.data.lights, bpy.data.cameras):
        for datablock in list(coll):
            if datablock.users == 0:
                coll.remove(datablock)


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def enable_gpu():
    """Enable CUDA/OptiX GPU rendering for Cycles."""
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type != 'CPU'
        if d.use:
            print(f"✅ Using GPU: {d.name} ({d.type})")
    bpy.context.scene.cycles.device = "GPU"


def setup_scene(samples):
    """Create camera and ground plane; configure Cycles."""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = samples
    scene.render.resolution_x = IMG_RES
    scene.render.resolution_y = IMG_RES
    scene.render.film_transparent = True
    if scene.world:
        scene.world.color = (0.0, 0.0, 0.0)
    enable_gpu()

    # Camera looking down from -Y direction
    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.data.lens = 45
    cam_obj.location = (
        random.uniform(-0.3, 0.3),
        -random.uniform(3.0, 4.5),
        random.uniform(2.0, 3.2),
    )
    cam_obj.rotation_mode = 'XYZ'
    look_at(cam_obj, (0, 0, 0.6))
    scene.camera = cam_obj

    # Ground plane (shadow catcher)
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name = "Ground"
    try:
        plane.cycles.is_shadow_catcher = True
    except Exception:
        pass
    return cam_obj, plane


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def normalize_object(obj):
    """Normalize object so its largest dimension = 1.0."""
    max_dim = max(obj.dimensions)
    if max_dim == 0:
        return
    scale = 1.0 / max_dim
    obj.scale = (scale, scale, scale)
    bpy.context.view_layer.update()


def import_model(path):
    """Import supported model formats (.obj, .fbx, .blend)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        bpy.ops.import_scene.obj(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == ".blend":
        with bpy.data.libraries.load(path) as (data_from, data_to):
            data_to.objects = data_from.objects
        for obj in data_to.objects:
            if obj:
                bpy.context.collection.objects.link(obj)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def first_mesh():
    """Return imported mesh object, joining parts if needed."""
    meshes = [obj for obj in bpy.context.scene.objects
              if obj.type == "MESH" and obj.name != "Ground"]
    if not meshes:
        return None

    if len(meshes) == 1:
        return meshes[0]

    # Join split mesh parts so downstream logic (scaling, masking) stays consistent.
    bpy.ops.object.select_all(action="DESELECT")
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='OBJECT')
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.join()
    return meshes[0]


def sph_to_cart(theta, phi, r=LIGHT_RADIUS):
    """Convert spherical (degrees) to Cartesian coordinates."""
    t = math.radians(theta)
    p = math.radians(phi)
    x = r * math.sin(t) * math.cos(p)
    y = r * math.cos(t)
    z = r * math.sin(t) * math.sin(p)
    return (x, y, z)


def look_at(obj, target):
    """Rotate object so its -Z axis points at target."""
    direction = Vector(target) - obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj.rotation_euler = rot_quat.to_euler()


def center_object(obj):
    """Move object so its bounding box center sits at the origin."""
    bbox = [Vector(corner) for corner in obj.bound_box]
    center = sum(bbox, Vector()) / 8.0
    world_center = obj.matrix_world @ center
    obj.location -= world_center
    bpy.context.view_layer.update()


def place_on_ground(obj):
    """Translate object upward so its lowest point rests on Z=0."""
    bbox_world = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_z = min(v.z for v in bbox_world)
    obj.location.z -= min_z
    bpy.context.view_layer.update()


def create_light(theta, phi, size):
    """Create an area light positioned on a sphere of radius r."""
    light_data = bpy.data.lights.new("Light", type='AREA')
    light_data.size = size
    light_data.energy = 1000
    light_obj = bpy.data.objects.new("Light", light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = sph_to_cart(theta, phi)
    light_obj.rotation_euler = (math.radians(90 - theta), 0, math.radians(phi))
    return light_obj


def make_mask_material():
    """Create a white emission material used for mask renders."""
    if MASK_MAT_NAME in bpy.data.materials:
        return bpy.data.materials[MASK_MAT_NAME]
    mat = bpy.data.materials.new(MASK_MAT_NAME)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1, 1, 1, 1)
    output = nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def render(scene, path):
    """Render current frame to a given path."""
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


# ------------------------------------------------------------
# Core: one sample render
# ------------------------------------------------------------
def render_sample(model_path, output_dir, samples, idx):
    """Render one sample with random light parameters (θ, φ, s)."""
    clear_scene()
    cam_obj, plane = setup_scene(samples)

    import_model(model_path)
    obj = first_mesh()
    if not obj:
        print(f"⚠️  No mesh found in {model_path}")
        return None

    normalize_object(obj)
    center_object(obj)
    place_on_ground(obj)
    obj.location.z += 0.1
    obj.rotation_euler = (0, 0, random.uniform(0, math.tau))

    # Sample light params according to Table 1
    theta = random.uniform(0, 45)
    phi = random.uniform(0, 360)
    size = random.uniform(2, 8)
    create_light(theta, phi, size)

    # Output paths
    rgb_path = os.path.join(output_dir, f"rgb_{idx:05d}.png")
    mask_path = os.path.join(output_dir, f"mask_{idx:05d}.png")
    shadow_path = os.path.join(output_dir, f"shadow_{idx:05d}.png")

    # RGB render
    render(bpy.context.scene, rgb_path)

    # Mask render (apply emission material)
    mask_mat = make_mask_material()
    orig_mats = list(obj.data.materials)
    obj.data.materials.clear()
    obj.data.materials.append(mask_mat)
    plane_prev_hide_render = plane.hide_render
    plane.hide_render = True
    bpy.context.scene.render.image_settings.color_mode = 'BW'
    render(bpy.context.scene, mask_path)
    bpy.context.scene.render.image_settings.color_mode = 'RGB'
    plane.hide_render = plane_prev_hide_render
    obj.data.materials.clear()
    for m in orig_mats:
        obj.data.materials.append(m)

    # ---------------- Shadow Render ----------------
    # Make sure the ground plane catches shadows
    plane.cycles.is_shadow_catcher = True
    plane.hide_render = False
    bpy.context.view_layer.cycles.use_pass_shadow_catcher = True

    # --- Store previous visibility settings ---
    prev_vis = {}
    for prop in [
        "visible_camera", "visible_diffuse",
        "visible_glossy", "visible_transmission", "visible_shadow"
    ]:
        if hasattr(obj, prop):
            prev_vis[prop] = getattr(obj, prop)

    # --- Hide object from camera but keep its shadow/reflection ---
    if hasattr(obj, "visible_camera"):
        obj.visible_camera = False
    if hasattr(obj, "visible_diffuse"):
        obj.visible_diffuse = True
    if hasattr(obj, "visible_glossy"):
        obj.visible_glossy = True
    if hasattr(obj, "visible_shadow"):
        obj.visible_shadow = True

    # Render the pure shadow pass (object invisible but shadow visible)
    render(bpy.context.scene, shadow_path)

    # --- Restore previous visibility ---
    for prop, value in prev_vis.items():
        if hasattr(obj, prop):
            setattr(obj, prop, value)


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
def main():
    args = get_args()
    ensure_dir(args.output)

    models = [os.path.join(args.models, f)
              for f in os.listdir(args.models)
              if f.lower().endswith((".obj", ".fbx", ".blend"))]

    if not models:
        print(f"No valid models found in {args.models}")
        return

    variations = max(1, int(VARIATIONS_PER_MODEL))
    metadata = []
    sample_idx = 0
    total_samples = len(models) * variations

    for model_path in models:
        model_name = os.path.basename(model_path)
        for variation in range(variations):
            print(f"🟢 Rendering {model_name} v{variation+1}/{variations} ({sample_idx+1}/{total_samples})")
            result = render_sample(model_path, args.output, args.samples, sample_idx)
            sample_idx += 1
            if result:
                result["variation"] = variation
                metadata.append(result)

    with open(os.path.join(args.output, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n🟢 Render complete. {len(metadata)} samples saved to {args.output}")


if __name__ == "__main__":
    main()
