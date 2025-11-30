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
VARIATIONS_PER_MODEL = 32
HELPER_KEYWORDS = ("helper", "orient", "arrow", "axis", "pivot", "guide", "control")
RECENT_LIGHT_DIRS = []
MAX_RECENT_LIGHT_DIRS = 6
MAX_LIGHT_DIR_DOT = 0.55


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def quantize_angle(angle_degrees, step=5.0):
    return round(angle_degrees / step) * step

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
    p.add_argument("--resume", action="store_true", help="Resume from last checkpoint (default)")
    p.add_argument("--override", action="store_true", help="Start fresh, delete existing output")
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
    # Darker background to avoid blending with lit objects
    bg.inputs[0].default_value = (0.15, 0.15, 0.15, 1)
    bg.inputs[1].default_value = 0.08
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.type = 'ORTHO'
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

    mesh_objs = []
    for obj in list(scene.objects):
        if obj.name == "Ground":
            continue
        if obj in protected:
            continue
        if obj.type != "MESH":
            bpy.data.objects.remove(obj, do_unlink=True)
            continue
        mesh_objs.append(obj)

    if not mesh_objs:
        return

    keep, flagged = [], []
    for obj in mesh_objs:
        name = obj.name.lower()
        dims = obj.dimensions
        max_dim = max(dims)
        flat = dims.z < 0.01 and max(dims) > 10 * max(dims.z, 1e-4)
        tiny = max_dim < 0.05
        if any(k in name for k in HELPER_KEYWORDS) or flat or tiny:
            flagged.append(obj)
        else:
            keep.append(obj)

    if not keep and flagged:
        flagged.sort(key=lambda o: o.dimensions.x * o.dimensions.y * max(o.dimensions.z, 1e-6), reverse=True)
        salvage = flagged.pop(0)
        keep.append(salvage)
        print(f"ℹ️ Keeping potential helper mesh '{salvage.name}' to avoid empty scene")

    for obj in flagged:
        if obj not in keep:
            bpy.data.objects.remove(obj, do_unlink=True)

def first_mesh():
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH" and o.name != "Ground"]
    if not objs:
        print("⚠️ No mesh objects available after import/cleanup")
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
    elif len(prim) == 1:
        # Set the single object as active
        bpy.context.view_layer.objects.active = prim[0]
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
    
    # Reduce intensity for overhead angles (near 0°) to prevent overexposure
    # theta = 0° is directly overhead (12 o'clock)
    # Apply reduction for theta < 30°
    angle_factor = 1.0
    if theta < 30:
        # Smooth reduction: at 0° use 40%, at 30° use 100%
        angle_factor = 0.4 + (theta / 30.0) * 0.6
    
    l.energy = base_energy * intensity * angle_factor
    l.size = size
    lo = bpy.data.objects.new("Light", l)
    bpy.context.collection.objects.link(lo)
    lo.location = sph_to_cart(theta, phi, r)
    look_at(lo, (0, 0, 0))
    return lo

def pick_light_setup(scene_span):
    """Return strongly varied light parameters while spacing directions."""
    angle_bands = [(12, 28), (32, 48), (56, 78)]
    phi_bands = [(20, 160), (200, 340)]
    attempts = 0
    selection = None
    while attempts < 20:
        theta = quantize_angle(random.uniform(*random.choice(angle_bands)))
        phi = quantize_angle(random.uniform(*random.choice(phi_bands)))
        if random.random() < 0.2:
            phi = quantize_angle(random.uniform(0, 360))
        direction = Vector(sph_to_cart(theta, phi, 1.0)).normalized()
        if (not RECENT_LIGHT_DIRS or
                all(direction.dot(prev) <= MAX_LIGHT_DIR_DOT for prev in RECENT_LIGHT_DIRS) or
                attempts > 12):
            selection = (theta, phi, direction)
            break
        attempts += 1
    if selection is None:
        theta = quantize_angle(random.uniform(18, 75))
        phi = quantize_angle(random.uniform(0, 360))
        direction = Vector(sph_to_cart(theta, phi, 1.0)).normalized()
        selection = (theta, phi, direction)

    RECENT_LIGHT_DIRS.append(selection[2])
    if len(RECENT_LIGHT_DIRS) > MAX_RECENT_LIGHT_DIRS:
        RECENT_LIGHT_DIRS.pop(0)

    size_ranges = [(0.45, 1.1), (1.0, 2.6), (2.2, 5.5)]
    intensity_ranges = [(0.35, 0.9), (0.9, 1.6), (1.2, 2.1)]
    size = random.uniform(*random.choice(size_ranges))
    intensity = random.uniform(*random.choice(intensity_ranges))
    span_factor = clamp(scene_span, 0.5, 3.5)
    radius = clamp(span_factor * random.uniform(2.4, 3.6), 3.0, 18.0)
    return selection[0], selection[1], size, intensity, radius, selection[2]

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


def invert_shadow_mask(path):
    if not os.path.exists(path):
        return
    image = bpy.data.images.load(path)
    try:
        pixels = list(image.pixels)
        for i in range(0, len(pixels), 4):
            pixels[i] = 1.0 - pixels[i]
            pixels[i + 1] = 1.0 - pixels[i + 1]
            pixels[i + 2] = 1.0 - pixels[i + 2]
        image.pixels[:] = pixels
        image.filepath_raw = path
        image.file_format = 'PNG'
        image.save()
    finally:
        bpy.data.images.remove(image)

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

    # ground plane sized to cover camera view
    bb = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    span_x = max(v.x for v in bb) - min(v.x for v in bb)
    span_y = max(v.y for v in bb) - min(v.y for v in bb)
    ground_extent = max(span_x, span_y) * 6
    ground_size = max(ground_extent, 8.0)
    bpy.ops.mesh.primitive_plane_add(size=ground_size, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name = "Ground"
    plane_mat = bpy.data.materials.new("GroundMat")
    plane_mat.use_nodes = True
    bsdf = plane_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        # Darker ground to prevent blending with lit objects
        bsdf.inputs["Base Color"].default_value = (0.18, 0.18, 0.18, 1)
        bsdf.inputs["Roughness"].default_value = 1.0
    plane.data.materials.append(plane_mat)

    # settle using physics
    settle_object_with_physics(obj, drop_height=0.05, ground_size=3.0, steps=90)

    # random rotation and lighting
    obj.rotation_euler = (0, 0, random.uniform(0, math.tau))
    place_on_ground(obj)
    span_xy = max(obj.dimensions.x, obj.dimensions.y)
    span_xy = clamp(span_xy, 0.35, 3.5)

    cam_height = clamp(span_xy * 2.4, 2.2, 11.0)
    cam.location = (0, 0, cam_height)
    cam.data.ortho_scale = clamp(span_xy * 3.0, 3.5, ground_size * 0.9)
    look_at(cam, (0, 0, 0))
    cam_view_dir = (Vector((0.0, 0.0, 0.0)) - cam.location).normalized()

    theta, phi, size, intensity, radius, light_dir_scene = pick_light_setup(span_xy)
    light_to_origin = (-light_dir_scene).normalized()
    alignment = max(0.0, light_to_origin.dot(cam_view_dir))
    if alignment > 0.45:
        falloff = 1.0 - 0.7 * ((alignment - 0.45) / 0.55)
        intensity *= clamp(falloff, 0.25, 1.0)
    size *= clamp(span_xy / 1.2, 0.7, 1.35)
    intensity = clamp(intensity, 0.35, 2.4)
    radius = clamp(radius, 3.0, 18.0)
    light = create_light(theta, phi, size, r=radius, intensity=intensity)

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
    invert_shadow_mask(shadow)
    if hasattr(obj, "visible_camera"):
        obj.visible_camera = True

    meta = {
        "theta": float(theta),
        "phi": float(phi),
        "size": float(size),
        "light_intensity": float(intensity),
        "light_radius": float(radius),
        "camera_height": float(cam_height),
        "camera_ortho_scale": float(cam.data.ortho_scale),
        "span_xy": float(span_xy),
        "ground_size": float(ground_size)
    }
    meta_path = os.path.join(out_dir, f"meta_{idx:05d}.json")
    with open(meta_path, "w", encoding="utf-8") as meta_file:
        json.dump(meta, meta_file, indent=2)

# ---------------- Progress Tracking ----------------
def get_progress_file(output_dir):
    """Get path to progress tracking file."""
    return os.path.join(output_dir, ".generation_progress.json")

def load_progress(output_dir):
    """Load generation progress from file."""
    progress_file = get_progress_file(output_dir)
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                return json.load(f)
        except Exception:
            return {"completed_samples": 0, "last_index": -1}
    return {"completed_samples": 0, "last_index": -1}

def save_progress(output_dir, completed_samples, last_index):
    """Save generation progress to file."""
    progress_file = get_progress_file(output_dir)
    progress = {
        "completed_samples": completed_samples,
        "last_index": last_index
    }
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)

def find_next_available_index(output_dir):
    """Find the next available sample index by checking existing files."""
    idx = 0
    while True:
        meta_path = os.path.join(output_dir, f"meta_{idx:05d}.json")
        if not os.path.exists(meta_path):
            return idx
        idx += 1

# ---------------- Main ----------------
def main():
    a = get_args()
    
    # Handle override flag
    if a.override:
        import shutil
        if os.path.exists(a.output):
            print(f"🗑️  Overriding existing dataset at {a.output}")
            shutil.rmtree(a.output)
        os.makedirs(a.output, exist_ok=True)
        start_idx = 0
        progress = {"completed_samples": 0, "last_index": -1}
    else:
        # Resume mode (default)
        os.makedirs(a.output, exist_ok=True)
        progress = load_progress(a.output)
        start_idx = find_next_available_index(a.output)
        
        if start_idx > 0:
            print(f"📂 Resuming from existing dataset")
            print(f"✓ Found {start_idx} existing samples")
            print(f"▶️  Starting from index {start_idx}")
        else:
            print(f"📂 Starting new dataset generation")
    
    files = [os.path.join(a.models, f) for f in os.listdir(a.models)
             if f.lower().endswith((".obj", ".fbx", ".blend"))]
    
    total_samples = len(files) * a.samples
    print(f"📊 Total target: {len(files)} models × {a.samples} samples = {total_samples} samples")
    if start_idx > 0:
        remaining = total_samples - start_idx
        print(f"⏳ Remaining: {remaining} samples")
    print()
    
    idx = start_idx
    completed_count = 0
    
    for m in files:
        for v in range(a.samples):  # Use command-line --samples argument
            # Calculate what this sample's index should be
            expected_idx = (files.index(m) * a.samples) + v
            
            # Skip if already generated
            if expected_idx < start_idx:
                continue
            
            progress_pct = (idx / total_samples) * 100 if total_samples > 0 else 0
            print(f"🟢 [{idx+1}/{total_samples}] ({progress_pct:.1f}%) Rendering {os.path.basename(m)} v{v+1}/{a.samples}")
            
            try:
                render_sample(m, a.output, a.samples, idx)
                completed_count += 1
                
                # Save progress every 10 samples
                if completed_count % 10 == 0:
                    save_progress(a.output, start_idx + completed_count, idx)
                    
            except Exception as e:
                print(f"❌ Error rendering sample {idx}: {e}")
                # Continue to next sample on error
                
            idx += 1
    
    # Final progress save
    save_progress(a.output, idx, idx - 1)
    print()
    print("="*60)
    print(f"✅ Generation complete!")
    print(f"📊 Total samples generated: {idx}")
    if start_idx > 0:
        print(f"📈 New samples in this run: {completed_count}")
    print("="*60)

if __name__ == "__main__":
    main()