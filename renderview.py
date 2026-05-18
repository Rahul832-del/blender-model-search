"""
Blender Multi-View Batch Renderer v3
======================================
NO physical lights — uses environment lighting only (zero hotspots).
16 NAMED camera views: front, back, left, right, top, bottom, + 10 isometric.

USAGE:
  blender --background --python render_views.py -- --input ./models --output ./renders
"""

import bpy
import math
import os
import sys
import glob
import mathutils
import time


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PARSE ARGUMENTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []


def get_arg(flag, default):
    if flag in argv:
        return argv[argv.index(flag) + 1]
    return default


INPUT_DIR     = get_arg("--input",   "./models")
OUTPUT_DIR    = get_arg("--output",  "./renders")
RESOLUTION    = int(get_arg("--res", "512"))
ENGINE        = get_arg("--engine",  "cycles").lower()
SAMPLES       = int(get_arg("--samples", "128"))
SKIP_EXISTING = "--skip" in argv


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RENDER SETTINGS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def setup_render_settings():
    scene = bpy.context.scene

    if ENGINE == "cycles":
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'GPU'
        scene.cycles.samples = SAMPLES
        scene.cycles.use_denoising = True
        scene.cycles.max_bounces = 12
        scene.cycles.diffuse_bounces = 4
        scene.cycles.glossy_bounces = 4
        scene.cycles.transmission_bounces = 8
        scene.cycles.transparent_max_bounces = 8
        prefs = bpy.context.preferences.addons.get('cycles')
        if prefs:
            prefs.preferences.compute_device_type = 'CUDA'
            for device in prefs.preferences.devices:
                device.use = True
    else:
        try:
            scene.render.engine = 'BLENDER_EEVEE_NEXT'
        except:
            scene.render.engine = 'BLENDER_EEVEE'

    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.resolution_percentage = 100

    # Color management — Standard preserves original material colors accurately
    scene.view_settings.view_transform = 'Standard'
    try:
        scene.view_settings.look = 'None'
    except:
        pass
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    # ── FREESTYLE EDGE RENDERING ──
    # Draws visible edges/outlines on the 3D geometry
    # White objects get clear black outlines — CLIP can see shape even without color contrast
    scene.render.use_freestyle = True
    scene.render.line_thickness = 1.2  # Subtle but visible edge lines

    # Configure Freestyle on the active view layer
    view_layer = scene.view_layers[0]
    view_layer.use_freestyle = True

    # Clear existing line sets and create our own
    freestyle = view_layer.freestyle_settings
    # Remove existing linesets (Blender 5.0 compatible)
    while len(freestyle.linesets) > 0:
        freestyle.linesets.remove(freestyle.linesets[0])

    lineset = freestyle.linesets.new("EdgeLines")
    lineset.select_silhouette = True        # Outer silhouette
    lineset.select_border = True            # Mesh borders
    lineset.select_crease = True            # Sharp edges / creases
    lineset.select_edge_mark = False
    lineset.select_external_contour = True  # External outline
    lineset.select_material_boundary = True # Where materials change
    lineset.select_suggestive_contour = False
    lineset.select_ridge_valley = False

    # Edge line style — thin dark lines
    linestyle = lineset.linestyle
    linestyle.color = (0.15, 0.15, 0.18)   # Dark gray (not pure black — more natural)
    linestyle.thickness = 1.2
    linestyle.alpha = 0.7                   # Slightly transparent — not overpowering

    # Output format
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.compression = 15


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENVIRONMENT LIGHTING (NO PHYSICAL LIGHTS!)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def setup_environment():
    """
    Realistic studio lighting:
    - HDRI-style gradient environment (blue sky feel)
    - One physical sun light for sharp shadows + reflections
    - Environment provides fill light from all directions
    """
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    # --- Background gradient (visible behind model) ---
    tex_coord = nodes.new('ShaderNodeTexCoord')
    separate = nodes.new('ShaderNodeSeparateXYZ')
    gradient_ramp = nodes.new('ShaderNodeValToRGB')
    gradient_bg = nodes.new('ShaderNodeBackground')

    gradient_ramp.color_ramp.elements[0].position = 0.0
    gradient_ramp.color_ramp.elements[0].color = (0.50, 0.50, 0.52, 1)  # Bottom: medium gray
    gradient_ramp.color_ramp.elements[1].position = 1.0
    gradient_ramp.color_ramp.elements[1].color = (0.65, 0.65, 0.67, 1)  # Top: lighter gray
    gradient_bg.inputs[1].default_value = 0.7

    links.new(tex_coord.outputs['Generated'], separate.inputs[0])
    links.new(separate.outputs['Z'], gradient_ramp.inputs['Fac'])
    links.new(gradient_ramp.outputs['Color'], gradient_bg.inputs['Color'])

    # --- Environment light for fill (lower intensity to preserve dark materials) ---
    light_bg = nodes.new('ShaderNodeBackground')
    light_bg.inputs[0].default_value = (1.0, 0.98, 0.95, 1)  # Slightly warm white
    light_bg.inputs[1].default_value = 1.2                     # Lower fill — preserves dark colors

    # --- Mix: camera sees gradient, light rays see warm white ---
    mix = nodes.new('ShaderNodeMixShader')
    light_path = nodes.new('ShaderNodeLightPath')
    output = nodes.new('ShaderNodeOutputWorld')

    links.new(light_path.outputs['Is Camera Ray'], mix.inputs['Fac'])
    links.new(light_bg.outputs[0], mix.inputs[1])
    links.new(gradient_bg.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], output.inputs[0])

    # --- ADD a physical sun light for sharp shadows + reflections ---
    # This is what makes metallic surfaces look real
    sun_data = bpy.data.lights.new("Sun", type='SUN')
    sun_data.energy = 1.5
    sun_data.angle = math.radians(5)  # Sharp shadows
    sun_obj = bpy.data.objects.new("Sun", sun_data)
    sun_obj.location = (5, -5, 10)
    sun_obj.rotation_euler = (math.radians(40), math.radians(10), math.radians(30))
    bpy.context.scene.collection.objects.link(sun_obj)

    # --- ADD a soft area light from opposite side for fill ---
    fill_data = bpy.data.lights.new("Fill", type='AREA')
    fill_data.energy = 60
    fill_data.size = 8  # Large = soft
    fill_obj = bpy.data.objects.new("Fill", fill_data)
    fill_obj.location = (-5, 3, 6)
    fill_obj.rotation_euler = (math.radians(50), 0, math.radians(-40))
    bpy.context.scene.collection.objects.link(fill_obj)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SCENE HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def remove_cameras_and_lights():
    """Remove ALL cameras and ALL lights — we use environment lighting only."""
    for obj in list(bpy.context.scene.objects):
        if obj.type in ('CAMERA', 'LIGHT'):
            bpy.data.objects.remove(obj, do_unlink=True)


def fix_materials():
    """
    Simple and safe material handling:
      - ANY existing material → KEEP IT (never overwrite)
      - Empty material slot (None) → apply clay
      - No material at all → apply clay
    
    This preserves ALL original colors, textures, and node setups.
    Only truly empty objects get clay fallback.
    
    Returns: "original" if model had any materials, "clay" if all were empty
    """
    clay = bpy.data.materials.new("ClayFallback")
    clay.use_nodes = True
    bsdf = clay.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.35, 0.25, 0.18, 1)
        bsdf.inputs['Roughness'].default_value = 0.95
        try:
            bsdf.inputs['Specular IOR Level'].default_value = 0.0
        except:
            pass

    has_real_material = False

    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue

        if len(obj.data.materials) == 0:
            # No material at all → apply clay
            obj.data.materials.append(clay)
        else:
            for i, slot in enumerate(obj.material_slots):
                if slot.material is None:
                    # Empty slot → fill with clay
                    obj.data.materials[i] = clay
                else:
                    # Has ANY material → keep it, don't touch
                    has_real_material = True

    return "original" if has_real_material else "clay"


def get_scene_bounds():
    min_c = mathutils.Vector((float('inf'),) * 3)
    max_c = mathutils.Vector((float('-inf'),) * 3)
    found = False

    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            found = True
            for corner in obj.bound_box:
                wc = obj.matrix_world @ mathutils.Vector(corner)
                for i in range(3):
                    if wc[i] < min_c[i]: min_c[i] = wc[i]
                    if wc[i] > max_c[i]: max_c[i] = wc[i]

    if not found:
        return mathutils.Vector((0, 0, 0)), 2.0

    center = (min_c + max_c) / 2
    dims = max_c - min_c
    return center, max(dims.x, dims.y, dims.z)


def create_camera():
    cam_data = bpy.data.cameras.new("RenderCam")
    cam_data.lens = 50
    cam_obj = bpy.data.objects.new("RenderCam", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    return cam_obj


def look_at(camera, target):
    direction = target - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 16 NAMED CAMERA VIEWS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_named_views(center, dist):
    """
    16 explicit, named views that guarantee full coverage:

    6 ORTHOGRAPHIC FACES:
      front, back, left, right, top, bottom

    4 ISOMETRIC CORNERS (30° elevation, 4 quadrants):
      iso_front_right, iso_front_left, iso_back_right, iso_back_left

    4 HIGH ISOMETRIC (60° elevation, 4 quadrants offset 45°):
      high_front_right, high_front_left, high_back_right, high_back_left

    2 EXTRA (low angle):
      low_front, low_back
    """
    cx, cy, cz = center.x, center.y, center.z
    views = []

    # --- 6 ORTHOGRAPHIC FACE VIEWS (straight on) ---
    # Slight elevation (5°) to avoid perfectly flat angle
    d = dist
    views.append(("front",   mathutils.Vector((cx, cy - d, cz + d * 0.05))))
    views.append(("back",    mathutils.Vector((cx, cy + d, cz + d * 0.05))))
    views.append(("right",   mathutils.Vector((cx + d, cy, cz + d * 0.05))))
    views.append(("left",    mathutils.Vector((cx - d, cy, cz + d * 0.05))))
    views.append(("top",     mathutils.Vector((cx, cy - d * 0.01, cz + d))))
    views.append(("bottom",  mathutils.Vector((cx, cy - d * 0.01, cz - d * 0.7))))

    # --- 4 ISOMETRIC CORNERS at 30° elevation ---
    elev = math.radians(30)
    for i, name in enumerate(["iso_front_right", "iso_front_left", "iso_back_left", "iso_back_right"]):
        az = math.radians(45 + 90 * i)  # 45°, 135°, 225°, 315°
        x = cx + d * math.cos(elev) * math.cos(az)
        y = cy + d * math.cos(elev) * math.sin(az)
        z = cz + d * math.sin(elev)
        views.append((name, mathutils.Vector((x, y, z))))

    # --- 4 HIGH ISOMETRIC at 60° elevation ---
    elev = math.radians(60)
    for i, name in enumerate(["high_front_right", "high_front_left", "high_back_left", "high_back_right"]):
        az = math.radians(0 + 90 * i)  # 0°, 90°, 180°, 270°
        x = cx + d * math.cos(elev) * math.cos(az)
        y = cy + d * math.cos(elev) * math.sin(az)
        z = cz + d * math.sin(elev)
        views.append((name, mathutils.Vector((x, y, z))))

    # --- 2 LOW ANGLE views (-15° elevation) ---
    elev = math.radians(-15)
    for i, name in enumerate(["low_front", "low_back"]):
        az = math.radians(-60 + 180 * i)
        x = cx + d * math.cos(elev) * math.cos(az)
        y = cy + d * math.cos(elev) * math.sin(az)
        z = cz + d * math.sin(elev)
        views.append((name, mathutils.Vector((x, y, z))))

    return views


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RENDER ONE .blend FILE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_model(blend_path, output_dir):
    model_name = os.path.splitext(os.path.basename(blend_path))[0]
    model_output = os.path.join(output_dir, model_name)

    if SKIP_EXISTING and os.path.isdir(model_output):
        existing = [f for f in os.listdir(model_output) if f.endswith('.png')]
        if len(existing) >= 16:
            print(f"  SKIP: {model_name} ({len(existing)} views exist)")
            return 0

    os.makedirs(model_output, exist_ok=True)
    start = time.time()

    bpy.ops.wm.open_mainfile(filepath=blend_path)

    # Setup — NO physical lights!
    remove_cameras_and_lights()
    material_type = fix_materials()  # Returns "original" or "clay"
    setup_render_settings()
    setup_environment()
    camera = create_camera()

    # Camera distance
    center, max_dim = get_scene_bounds()
    cam_distance = max_dim * 2.5

    # Get 16 named views
    views = get_named_views(center, cam_distance)

    for idx, (view_name, cam_pos) in enumerate(views):
        camera.location = cam_pos
        look_at(camera, center)

        filename = f"{model_name}_{view_name}.png"
        bpy.context.scene.render.filepath = os.path.join(model_output, filename)
        bpy.ops.render.render(write_still=True)

    elapsed = time.time() - start
    mat_label = "TEXTURED" if material_type == "original" else "CLAY"
    print(f"  DONE: {model_name} [{mat_label}] — {len(views)} views in {elapsed:.1f}s")
    return len(views)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    blend_files = sorted(glob.glob(os.path.join(INPUT_DIR, "**", "*.blend"), recursive=True))

    if not blend_files:
        print(f"ERROR: No .blend files found in {INPUT_DIR}")
        sys.exit(1)

    print(f"\n  Models: {len(blend_files)} | Views: 16 | "
          f"Res: {RESOLUTION}x{RESOLUTION} | Engine: {ENGINE.upper()}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total = 0
    failed = []

    for i, path in enumerate(blend_files, 1):
        print(f"[{i}/{len(blend_files)}] {os.path.basename(path)}")
        try:
            total += render_model(path, OUTPUT_DIR)
        except Exception as e:
            print(f"  FAIL: {e}")
            failed.append(os.path.basename(path))

    print(f"\n  COMPLETE: {total} images rendered, {len(failed)} failed\n")
    if failed:
        print("  Failed:")
        for f in failed:
            print(f"    - {f}")


if __name__ == "__main__":
    main()