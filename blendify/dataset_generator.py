"""
jasper_dataset_generator.py
------------------------------------------------------------
Synthetic dataset renderer for diffusion-based shadow models.

Now includes robust rigid-body settling:
works for OBJ/FBX/BLEND automatically.

Usage:
    blender --background --python jasper_dataset_generator.py -- \
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

LIGHT_RADIUS = 8.0
IMG_RES = 1024
MASK_MAT_NAME = "_MaskMat"
VARIATIONS_PER_MODEL = 8
HELPER_KEYWORDS = ("helper", "orient", "arrow", "axis", "pivot", "guide", "control")
RECENT_LIGHT_DIRS = []
MAX_RECENT_LIGHT_DIRS = 6
MAX_LIGHT_DIR_DOT = 0.55

# ---------------- Args ----------------
def get_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--models", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--samples", type=int, default=128)
    return p.parse_args(argv)

# ---------------- Scene Setup ----------------
def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.images,
                 bpy.data.lights, bpy.data.cameras):
        for datablock in list(coll):
            if datablock.users == 0:
                coll.remove(datablock)

def enable_gpu():
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type != 'CPU'
        bpy.context.scene.cycles.device = "GPU"
    except Exception:
        pass

def setup_scene(samples):
    s = bpy.context.scene
    s.render.engine = 'CYCLES'
    s.cycles.samples = samples
    s.render.resolution_x = IMG_RES
    s.render.resolution_y = IMG_RES
    s.render.film_transparent = True
    enable_gpu()
    if not s.world:
        s.world = bpy.data.worlds.new("World")
    s.world.use_nodes = True
    bg = s.world.node_tree.nodes.get("Background") or s.world.node_tree.nodes.new("ShaderNodeBackground")
    bg.inputs[0].default_value = (0.4, 0.4, 0.4, 1)
    bg.inputs[1].default_value = 0.15
    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    s.camera = cam_obj
    return cam_obj

# ---------------- Import ----------------
def import_model(p):
    ext = os.path.splitext(p)[1].lower()
    if ext == ".obj":
        bpy.ops.import_scene.obj(filepath=p)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=p)
    elif ext == ".blend":
        with bpy.data.libraries.load(p) as (f, t):
            t.objects = f.objects
        for o in t.objects:
            if o:
                bpy.context.collection.objects.link(o)
    else:
        raise ValueError("Unsupported type:", ext)

def cleanup_imported_objects():
    """Remove empties, lights, and helper geometry that shouldn't render."""
    scene = bpy.context.scene

    # Remove custom transform orientations left by the source file
    if hasattr(scene, "transform_orientations"):
        for orientation in list(scene.transform_orientations):
            if not getattr(orientation, "is_default", True):
                scene.transform_orientations.remove(orientation)

    protected = set()
    if scene.camera:
        protected.add(scene.camera)
        if scene.camera.original:
            protected.add(scene.camera.original)

    for obj in list(scene.objects):
        if obj.name == "Ground":
            continue
        if obj in protected:
            continue
        if obj.type != "MESH":
            bpy.data.objects.remove(obj, do_unlink=True)
            continue
        name = obj.name.lower()
        dims = obj.dimensions
        max_dim = max(dims)
        flat = dims.z < 0.01 and max(dims) > 10 * max(dims.z, 1e-4)
        tiny = max_dim < 0.05
        if any(k in name for k in HELPER_KEYWORDS) or flat or tiny:
            bpy.data.objects.remove(obj, do_unlink=True)

def first_mesh():
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH" and o.name != "Ground"]
    if not objs:
        return None
    prim, helpers = [], []
    for o in objs:
        n = o.name.lower()
        d = o.dimensions
        if any(k in n for k in HELPER_KEYWORDS) or max(d) < 0.05:
            helpers.append(o)
        else:
            prim.append(o)
    for h in helpers:
        bpy.data.objects.remove(h, do_unlink=True)
    if not prim:
        prim = [o for o in bpy.context.scene.objects if o.type == "MESH" and o.name != "Ground"]
    if len(prim) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for o in prim:
            o.select_set(True)
        bpy.context.view_layer.objects.active = prim[0]
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active

# ---------------- Transforms ----------------
def normalize_object(o):
    m = max(o.dimensions)
    if m > 0:
        s = 1.0 / m
        o.scale = (s, s, s)
    bpy.context.view_layer.update()

def center_object(o):
    bbox = [Vector(c) for c in o.bound_box]
    c = sum(bbox, Vector()) / 8.0
    wc = o.matrix_world @ c
    o.location -= wc
    bpy.context.view_layer.update()

def mesh_min_z_world(o):
    """Return the world-space minimum Z of the evaluated mesh."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = o.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    try:
        if mesh is None or len(mesh.vertices) == 0:
            return 0.0
        mw = eval_obj.matrix_world
        return min((mw @ v.co).z for v in mesh.vertices)
    finally:
        if hasattr(eval_obj, "to_mesh_clear"):
            eval_obj.to_mesh_clear()
        elif mesh:
            bpy.data.meshes.remove(mesh)

def place_on_ground(o):
    """Translate object so its lowest vertex rests exactly on Z=0."""
    min_z = mesh_min_z_world(o)
    if abs(min_z) > 1e-6:
        o.location.z -= min_z
        bpy.context.view_layer.update()
        # One more pass in case the shift changed evaluated mesh
        min_z = mesh_min_z_world(o)
        if min_z < 0:
            o.location.z -= min_z
    bpy.context.view_layer.update()
    final_min_z = mesh_min_z_world(o)
    if final_min_z > 5e-5:
        print(f"⚠️ Residual gap after grounding: {final_min_z:.6f}")

# ---------------- Physics Settle ----------------
def settle_object_with_physics(obj, drop_height=1.5, ground_size=6.0, steps=180):
    """
    Robust settle: create temp plane, drop with gravity, clean up.
    """
    sc = bpy.context.scene
    bpy.ops.mesh.primitive_plane_add(size=ground_size, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name = "PhysicsGround"
    bpy.context.view_layer.update()

    # add physics world if missing
    if sc.rigidbody_world is None:
        bpy.ops.rigidbody.world_add()
    sc.rigidbody_world.enabled = True
    sc.gravity = (0, 0, -9.81)
    sc.frame_start = 1
    sc.frame_end = steps

    # plane passive
    bpy.context.view_layer.objects.active = plane
    bpy.ops.rigidbody.object_add()
    plane.rigid_body.type = 'PASSIVE'
    plane.rigid_body.friction = 1.0
    plane.rigid_body.use_margin = False
    if hasattr(plane.rigid_body, "collision_margin"):
        plane.rigid_body.collision_margin = 0.0

    # object active
    bpy.context.view_layer.objects.active = obj
    bpy.ops.rigidbody.object_add()
    obj.rigid_body.type = 'ACTIVE'
    if hasattr(obj.rigid_body, "collision_shape"):
        obj.rigid_body.collision_shape = 'MESH'
    obj.rigid_body.mass = 1.0
    obj.rigid_body.use_margin = False
    if hasattr(obj.rigid_body, "collision_margin"):
        obj.rigid_body.collision_margin = 0.0
    obj.location.z += drop_height

    # simulate frames
    for f in range(1, steps + 1):
        sc.frame_set(f)
        bpy.context.view_layer.update()

    # cleanup
    bpy.context.view_layer.objects.active = obj
    bpy.ops.rigidbody.object_remove()
    bpy.context.view_layer.objects.active = plane
    bpy.ops.rigidbody.object_remove()
    bpy.ops.rigidbody.world_remove()
    bpy.data.objects.remove(plane, do_unlink=True)
    bpy.context.view_layer.update()
    place_on_ground(obj)
    print("💡 Settled with physics at Z =", obj.location.z)

# ---------------- Lighting ----------------
def sph_to_cart(theta, phi, r=LIGHT_RADIUS):
    t = math.radians(theta)
    p = math.radians(phi)
    return (r * math.sin(t) * math.cos(p),
            r * math.sin(t) * math.sin(p),
            r * math.cos(t))

def look_at(o, tgt):
    d = Vector(tgt) - o.location
    o.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()

def create_light(theta, phi, size, r, intensity=1.0):
    l = bpy.data.lights.new("Light", 'AREA')
    base_energy = 4500
    l.energy = base_energy * intensity
    l.size = size
    lo = bpy.data.objects.new("Light", l)
    bpy.context.collection.objects.link(lo)
    lo.location = sph_to_cart(theta, phi, r)
    look_at(lo, (0, 0, 0))
    return lo

def pick_light_setup(obj):
    """Return strongly varied light parameters while spacing directions."""
    angle_bands = [(12, 28), (32, 48), (56, 78)]
    phi_bands = [(20, 160), (200, 340)]
    attempts = 0
    selection = None
    while attempts < 20:
        theta = random.uniform(*random.choice(angle_bands))
        phi = random.uniform(*random.choice(phi_bands))
        if random.random() < 0.2:
            phi = random.uniform(0, 360)
        direction = Vector(sph_to_cart(theta, phi, 1.0)).normalized()
        if (not RECENT_LIGHT_DIRS or
                all(direction.dot(prev) <= MAX_LIGHT_DIR_DOT for prev in RECENT_LIGHT_DIRS) or
                attempts > 12):
            selection = (theta, phi, direction)
            break
        attempts += 1
    if selection is None:
        theta = random.uniform(18, 75)
        phi = random.uniform(0, 360)
        direction = Vector(sph_to_cart(theta, phi, 1.0)).normalized()
        selection = (theta, phi, direction)

    RECENT_LIGHT_DIRS.append(selection[2])
    if len(RECENT_LIGHT_DIRS) > MAX_RECENT_LIGHT_DIRS:
        RECENT_LIGHT_DIRS.pop(0)

    size_ranges = [(0.35, 1.0), (1.0, 3.2), (3.2, 8.8)]
    intensity_ranges = [(0.35, 0.7), (0.7, 1.4), (1.4, 2.6), (2.6, 3.8)]
    size = random.uniform(*random.choice(size_ranges))
    intensity = random.uniform(*random.choice(intensity_ranges))
    radius = max(3.0, max(obj.dimensions) * 3.0)
    return selection[0], selection[1], size, intensity, radius

# ---------------- Materials ----------------
def make_mask_material():
    if MASK_MAT_NAME in bpy.data.materials:
        return bpy.data.materials[MASK_MAT_NAME]
    m = bpy.data.materials.new(MASK_MAT_NAME)
    m.use_nodes = True
    n = m.node_tree.nodes
    n.clear()
    e = n.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = (1, 1, 1, 1)
    out = n.new("ShaderNodeOutputMaterial")
    m.node_tree.links.new(e.outputs["Emission"], out.inputs["Surface"])
    return m

def render_to(path):
    sc = bpy.context.scene
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)

# ---------------- Core Render ----------------
def render_sample(model, out_dir, samples, idx):
    clear_scene()
    cam = setup_scene(samples)
    import_model(model)
    cleanup_imported_objects()
    obj = first_mesh()
    if not obj:
        print("⚠️ Skipped:", model)
        return

    normalize_object(obj)
    center_object(obj)
    place_on_ground(obj)

    # ground plane
    bb = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    s = max(max(v.x for v in bb) - min(v.x for v in bb),
            max(v.y for v in bb) - min(v.y for v in bb)) * 2
    bpy.ops.mesh.primitive_plane_add(size=s, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name = "Ground"
    plane_mat = bpy.data.materials.new("GroundMat")
    plane_mat.use_nodes = True
    bsdf = plane_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.3, 0.3, 0.3, 1)
        bsdf.inputs["Roughness"].default_value = 1.0
    plane.data.materials.append(plane_mat)

    # settle using physics
    settle_object_with_physics(obj, drop_height=0.05, ground_size=3.0, steps=90)

    # random rotation and lighting
    obj.rotation_euler = (0, 0, random.uniform(0, math.tau))
    place_on_ground(obj)
    theta, phi, size, intensity, radius = pick_light_setup(obj)
    light = create_light(theta, phi, size, r=radius, intensity=intensity)

    # camera placement
    obj_size = max(obj.dimensions)
    cam.location = (0, -obj_size * 3.2, obj_size * 1.4)
    look_at(cam, (0, 0, obj_size * 0.4))

    rgb = os.path.join(out_dir, f"rgb_{idx:05d}.png")
    mask = os.path.join(out_dir, f"mask_{idx:05d}.png")
    shadow = os.path.join(out_dir, f"shadow_{idx:05d}.png")
    sc = bpy.context.scene

    # RGB
    sc.render.image_settings.color_mode = 'RGB'
    sc.render.film_transparent = False
    render_to(rgb)

    # Mask
    mask_mat = make_mask_material()
    orig_mats = list(obj.data.materials)
    obj.data.materials.clear()
    obj.data.materials.append(mask_mat)
    plane.hide_render = True
    sc.render.film_transparent = True
    sc.render.image_settings.color_mode = 'BW'
    render_to(mask)
    obj.data.materials.clear()
    for m in orig_mats:
        obj.data.materials.append(m)
    plane.hide_render = False

    # Shadow (object invisible to camera)
    if hasattr(obj, "visible_camera"):
        obj.visible_camera = False
    sc.render.film_transparent = False
    sc.render.image_settings.color_mode = 'RGB'
    render_to(shadow)
    if hasattr(obj, "visible_camera"):
        obj.visible_camera = True

# ---------------- Main ----------------
def main():
    a = get_args()
    os.makedirs(a.output, exist_ok=True)
    files = [os.path.join(a.models, f) for f in os.listdir(a.models)
             if f.lower().endswith((".obj", ".fbx", ".blend"))]
    idx = 0
    for m in files:
        for v in range(VARIATIONS_PER_MODEL):
            print(f"🟢 Rendering {os.path.basename(m)} v{v+1}")
            render_sample(m, a.output, a.samples, idx)
            idx += 1
    print("✅ Done:", idx, "samples")

if __name__ == "__main__":
    main()
