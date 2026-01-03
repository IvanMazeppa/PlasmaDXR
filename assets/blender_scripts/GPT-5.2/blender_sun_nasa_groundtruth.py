#!/usr/bin/env python3
"""
PlasmaDXR (GPT-5.2) — NASA Ground-Truth Sun Volume (Blender 5.0.1+)

Goal:
  Generate a *volumetric* sun-like asset (dense disk + filamentary prominences)
  suitable for OpenVDB → NanoVDB conversion, and render preview frames for
  evaluation against the NASA SDO reference dataset under:
    assets/reference_images/star/Eruptions_20241008_Activity_2048p30/

Design principles (based on what we've learned):
  - Prefer OpenVDB cache output with FULL precision when possible
    (improves downstream NanoVDB FLOAT-grid compatibility).
  - Avoid "surface-only" shader tricks; produce real volumetric density.
  - Produce limb activity by placing multiple small inflow emitters on/near
    the solar limb with outward initial velocity + vortex/turbulence fields.
  - Keep background black and color deep red-orange (SDO 304Å-like appearance).

Usage:
  blender -b -P assets/blender_scripts/GPT-5.2/blender_sun_nasa_groundtruth.py -- [options]

Common (recommended):
  assets/blender_scripts/GPT-5.2/run_blender_cli.sh \
    assets/blender_scripts/GPT-5.2/blender_sun_nasa_groundtruth.py -- \
    --output_dir build/vdb_output/sun_nasa_groundtruth_v1 \
    --resolution 96 --frame_end 60 --bake 1 --render 1 --render_frames mid
"""

import bpy
import sys
import math
import random
from pathlib import Path


class Config:
    # Output
    OUTPUT_DIR = "/home/maz3ppa/projects/PlasmaDXR/build/vdb_output/sun_nasa_groundtruth_v1"
    SEED = 1337

    # Animation
    FRAME_START = 1
    FRAME_END = 60

    # Simulation
    RESOLUTION = 96
    DOMAIN_SIZE = 8.0
    SUN_RADIUS = 2.2

    # Emitters
    CORE_DENSITY = 1.8
    CORE_TEMPERATURE = 2.0

    PROMINENCE_COUNT = 8
    PROM_DENSITY = 1.2
    PROM_TEMPERATURE = 3.0
    PROM_VEL_NORMAL = 1.6
    PROM_VEL_RANDOM = 0.55

    # Domain dynamics
    VORTICITY = 1.05
    USE_NOISE = True
    NOISE_SCALE = 2
    NOISE_STRENGTH = 1.25
    NOISE_TIME_ANIM = 0.45

    USE_DISSOLVE = True
    DISSOLVE_SPEED = 10

    # Render previews
    RENDER = True
    RENDER_FRAMES = "mid"  # "mid", "all", or "1,30,60"
    RENDER_X = 768
    RENDER_Y = 768
    RENDER_SAMPLES = 48
    FILM_TRANSPARENT = False

    # Material tuning (render-only knobs; do not require re-bake)
    MAT_COLOR = (0.95, 0.12, 0.03, 1.0)  # deep red-orange
    MAT_DENSITY = 1.6
    MAT_ANISOTROPY = 0.25
    MAT_EMISSION_STRENGTH = 2.5

    # Execution
    BAKE = True


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--output_dir" and i + 1 < len(argv):
            Config.OUTPUT_DIR = argv[i + 1]; i += 2
        elif a == "--seed" and i + 1 < len(argv):
            Config.SEED = int(argv[i + 1]); i += 2
        elif a == "--resolution" and i + 1 < len(argv):
            Config.RESOLUTION = int(argv[i + 1]); i += 2
        elif a == "--frame_start" and i + 1 < len(argv):
            Config.FRAME_START = int(argv[i + 1]); i += 2
        elif a == "--frame_end" and i + 1 < len(argv):
            Config.FRAME_END = int(argv[i + 1]); i += 2
        elif a == "--bake" and i + 1 < len(argv):
            Config.BAKE = argv[i + 1].lower() in ("1", "true", "yes"); i += 2
        elif a == "--render" and i + 1 < len(argv):
            Config.RENDER = argv[i + 1].lower() in ("1", "true", "yes"); i += 2
        elif a == "--render_frames" and i + 1 < len(argv):
            Config.RENDER_FRAMES = argv[i + 1]; i += 2
        elif a == "--mat_density" and i + 1 < len(argv):
            Config.MAT_DENSITY = float(argv[i + 1]); i += 2
        elif a == "--mat_emission" and i + 1 < len(argv):
            Config.MAT_EMISSION_STRENGTH = float(argv[i + 1]); i += 2
        else:
            i += 1


def _safe_enum_set(obj, prop: str, value: str) -> bool:
    """Set an enum if the property exists and supports the value."""
    if not hasattr(obj, prop):
        return False
    try:
        enum_items = getattr(obj.bl_rna.properties[prop], "enum_items", None)
        if enum_items is not None and value not in enum_items.keys():
            return False
        setattr(obj, prop, value)
        return True
    except Exception:
        return False


def _log_domain_enum_capabilities(settings):
    # Helps reconcile docs vs actual Blender build behavior.
    def enum_keys(name: str):
        if not hasattr(settings, name):
            return None
        try:
            return list(settings.bl_rna.properties[name].enum_items.keys())
        except Exception:
            return None

    print("[sun_nasa] Domain enum capabilities:", flush=True)
    for p in ("openvdb_cache_compress_type", "cache_precision", "cache_data_format", "cache_type"):
        keys = enum_keys(p)
        if keys is None:
            print(f"  - {p}: <not present>", flush=True)
        else:
            print(f"  - {p}: {keys}", flush=True)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.curves, bpy.data.lights):
        for block in list(coll):
            if getattr(block, "users", 0) == 0:
                try:
                    coll.remove(block)
                except Exception:
                    pass


def setup_scene():
    random.seed(Config.SEED)
    bpy.context.scene.frame_start = Config.FRAME_START
    bpy.context.scene.frame_end = Config.FRAME_END

    # Black world
    scene = bpy.context.scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)
        bg.inputs[1].default_value = 1.0


def ensure_active(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def create_domain():
    bpy.ops.mesh.primitive_cube_add(size=Config.DOMAIN_SIZE, location=(0.0, 0.0, 0.0))
    domain = bpy.context.active_object
    domain.name = "SunDomain"

    mod = domain.modifiers.new(name="Fluid", type="FLUID")
    mod.fluid_type = "DOMAIN"
    settings = mod.domain_settings
    settings.domain_type = "GAS"

    # Cache: OpenVDB
    #
    # IMPORTANT: keep Mantaflow cache in a dedicated subfolder so:
    # - the output dir stays tidy (blend + renders at root)
    # - conversion scripts can target one folder consistently
    cache_root = Path(Config.OUTPUT_DIR).resolve() / "vdb_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    # Cache type:
    # - ALL: bake + write caches (used when --bake 1)
    # - REPLAY: load existing caches without triggering bake (used when --bake 0)
    settings.cache_type = "ALL" if Config.BAKE else "REPLAY"
    settings.cache_directory = str(cache_root)
    settings.cache_data_format = "OPENVDB"

    # CRITICAL: Mantaflow uses its own cache frame range, independent of scene.frame_end.
    # If not set, it defaults to ~250 frames and will generate a huge sequence.
    if hasattr(settings, "cache_frame_start"):
        settings.cache_frame_start = int(Config.FRAME_START)
    if hasattr(settings, "cache_frame_end"):
        settings.cache_frame_end = int(Config.FRAME_END)
    if hasattr(settings, "cache_frame_offset"):
        settings.cache_frame_offset = 0

    # IMPORTANT:
    # Some Blender 5.0.1 builds show RNA warnings for openvdb_cache_compress_type and can crash during bake.
    # We therefore DO NOT touch compression here; we rely on Blender defaults for stability.

    # Prefer FULL precision if available (critical for float-grid workflows).
    if hasattr(settings, "cache_precision"):
        _safe_enum_set(settings, "cache_precision", "FULL")

    # Adaptive domain
    settings.resolution_max = int(Config.RESOLUTION)
    settings.use_adaptive_domain = True
    settings.adapt_threshold = 0.001
    settings.adapt_margin = 4

    # Dynamics: no buoyancy; turbulence from vorticity/noise/emitters.
    settings.gravity = (0.0, 0.0, 0.0)
    settings.alpha = 0.0
    settings.beta = 0.0
    settings.vorticity = float(Config.VORTICITY)

    # Noise up-res
    settings.use_noise = bool(Config.USE_NOISE)
    if Config.USE_NOISE:
        settings.noise_scale = int(Config.NOISE_SCALE)
        settings.noise_strength = float(Config.NOISE_STRENGTH)
        if hasattr(settings, "noise_time_anim"):
            settings.noise_time_anim = float(Config.NOISE_TIME_ANIM)

    # Dissolve (keeps structure but avoids infinite accumulation)
    settings.use_dissolve_smoke = bool(Config.USE_DISSOLVE)
    if Config.USE_DISSOLVE:
        settings.dissolve_speed = int(Config.DISSOLVE_SPEED)

    _log_domain_enum_capabilities(settings)

    assign_volume_material(domain)
    return domain


def assign_volume_material(domain):
    mat = bpy.data.materials.new("SunVolumeMaterial_NASA304")
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new(type="ShaderNodeOutputMaterial")
    out.location = (400, 0)

    pv = nodes.new(type="ShaderNodeVolumePrincipled")
    pv.location = (0, 0)

    # Deep red-orange (close to SDO 304Å feel)
    pv.inputs["Color"].default_value = Config.MAT_COLOR
    pv.inputs["Density"].default_value = float(Config.MAT_DENSITY)
    pv.inputs["Anisotropy"].default_value = float(Config.MAT_ANISOTROPY)
    # Keep strong but avoid clipping to white
    if "Emission Strength" in pv.inputs:
        pv.inputs["Emission Strength"].default_value = float(Config.MAT_EMISSION_STRENGTH)
    else:
        pv.inputs["Blackbody Intensity"].default_value = 4.0
        pv.inputs["Temperature"].default_value = 3800.0

    links.new(pv.outputs["Volume"], out.inputs["Volume"])

    if domain.data.materials:
        domain.data.materials[0] = mat
    else:
        domain.data.materials.append(mat)


def create_core_emitter(radius):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=(0.0, 0.0, 0.0), segments=64, ring_count=32)
    emitter = bpy.context.active_object
    emitter.name = "SunEmitter_Core"

    mod = emitter.modifiers.new(name="Fluid", type="FLUID")
    mod.fluid_type = "FLOW"
    flow = mod.flow_settings

    flow.flow_type = "SMOKE"
    flow.flow_behavior = "INFLOW"
    flow.use_inflow = True

    flow.density = float(Config.CORE_DENSITY)
    flow.temperature = float(Config.CORE_TEMPERATURE)
    flow.use_initial_velocity = True
    flow.velocity_random = 0.35
    flow.velocity_normal = 0.10

    emitter.hide_render = True
    return emitter


def create_prominence_emitters(star_radius, count):
    emitters = []
    for i in range(count):
        # Random point biased toward limb (|y| high in camera-forward view)
        # Camera will be on -Y, looking toward origin; limb activity should be around ±X/±Z.
        theta = random.random() * 2.0 * math.pi
        phi = (random.random() * 0.8 + 0.1) * math.pi  # avoid exact poles
        x = math.sin(phi) * math.cos(theta)
        y = math.cos(phi)
        z = math.sin(phi) * math.sin(theta)

        # Bias away from camera axis: reduce |y|
        y *= 0.35
        v = math.sqrt(max(1e-6, x*x + y*y + z*z))
        x, y, z = x / v, y / v, z / v

        pos = (x * star_radius * 0.95, y * star_radius * 0.95, z * star_radius * 0.95)
        r = star_radius * (0.12 + random.random() * 0.06)

        bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=pos, segments=32, ring_count=16)
        e = bpy.context.active_object
        e.name = f"SunEmitter_Prom_{i:02d}"

        mod = e.modifiers.new(name="Fluid", type="FLUID")
        mod.fluid_type = "FLOW"
        flow = mod.flow_settings
        flow.flow_type = "SMOKE"
        flow.flow_behavior = "INFLOW"
        flow.use_inflow = True

        flow.density = float(Config.PROM_DENSITY)
        flow.temperature = float(Config.PROM_TEMPERATURE)
        flow.use_initial_velocity = True
        flow.velocity_normal = float(Config.PROM_VEL_NORMAL)
        flow.velocity_random = float(Config.PROM_VEL_RANDOM)

        # Burst profile: ramp up then fade
        base = flow.density
        bpy.context.scene.frame_set(Config.FRAME_START)
        flow.density = base * 0.25
        flow.keyframe_insert(data_path="density")
        bpy.context.scene.frame_set((Config.FRAME_START + Config.FRAME_END) // 3)
        flow.density = base * (1.0 + random.random() * 0.7)
        flow.keyframe_insert(data_path="density")
        bpy.context.scene.frame_set(Config.FRAME_END)
        flow.density = base * 0.05
        flow.keyframe_insert(data_path="density")

        e.hide_render = True
        emitters.append(e)
    return emitters


def create_force_fields(star_radius):
    # Two vortices to encourage curved, looping prominences.
    # These are deliberately subtle; too strong tends to create "cat ears".
    fields = []
    for i, ang in enumerate((0.6, 2.4)):
        bpy.ops.object.effector_add(type="VORTEX", location=(math.cos(ang) * star_radius * 0.9, 0.0, math.sin(ang) * star_radius * 0.9))
        f = bpy.context.active_object
        f.name = f"SunField_Vortex_{i}"
        f.field.strength = 2.0
        f.field.flow = 1.0
        f.field.seed = Config.SEED + i * 13
        fields.append(f)

    # Turbulence field for filamentary structure
    bpy.ops.object.effector_add(type="TURBULENCE", location=(0.0, 0.0, 0.0))
    t = bpy.context.active_object
    t.name = "SunField_Turbulence"
    t.field.strength = 4.0
    t.field.size = star_radius * 1.5
    t.field.flow = 1.0
    t.field.seed = Config.SEED + 101
    fields.append(t)
    return fields


def setup_camera():
    scene = bpy.context.scene
    bpy.ops.object.camera_add(location=(0.0, -10.0, 0.0))
    cam = bpy.context.active_object
    cam.name = "Camera"
    cam.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    scene.camera = cam


def setup_render():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = int(Config.RENDER_SAMPLES)

    scene.render.resolution_x = int(Config.RENDER_X)
    scene.render.resolution_y = int(Config.RENDER_Y)
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = bool(Config.FILM_TRANSPARENT)

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.compression = 15


def frames_to_render():
    if Config.RENDER_FRAMES == "mid":
        return [(Config.FRAME_START + Config.FRAME_END) // 2]
    if Config.RENDER_FRAMES == "all":
        step = max(1, (Config.FRAME_END - Config.FRAME_START) // 5)
        return list(range(Config.FRAME_START, Config.FRAME_END + 1, step))
    try:
        return [int(x.strip()) for x in Config.RENDER_FRAMES.split(",") if x.strip()]
    except Exception:
        return [(Config.FRAME_START + Config.FRAME_END) // 2]


def bake(domain):
    print(f"[sun_nasa] Baking frames {Config.FRAME_START}-{Config.FRAME_END} @ res {Config.RESOLUTION}", flush=True)
    out_dir = Path(Config.OUTPUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vdb_cache").mkdir(parents=True, exist_ok=True)

    ensure_active(domain)
    # Avoid bpy.ops.fluid.free_all() in headless runs (observed to be crash-prone in some builds).
    bpy.ops.fluid.bake_all()
    print("[sun_nasa] Bake complete.", flush=True)


def render_previews():
    setup_render()
    out_dir = Path(Config.OUTPUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene

    rendered = []
    frames = frames_to_render()
    print(f"[sun_nasa] Rendering previews: {frames}", flush=True)
    for f in frames:
        f = max(Config.FRAME_START, min(Config.FRAME_END, f))
        scene.frame_set(f)
        path = out_dir / f"render_{f:04d}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        rendered.append(str(path))
        print(f"[sun_nasa] Saved: {path}", flush=True)
    return rendered


def main():
    parse_args()

    print("=" * 70, flush=True)
    print("NASA Ground-Truth Sun Volume (fresh approach)", flush=True)
    print("=" * 70, flush=True)
    print(f"[sun_nasa] output_dir: {Config.OUTPUT_DIR}", flush=True)
    print(f"[sun_nasa] seed: {Config.SEED}", flush=True)

    clear_scene()
    setup_scene()

    domain = create_domain()
    create_core_emitter(Config.SUN_RADIUS * 0.9)
    create_prominence_emitters(Config.SUN_RADIUS, Config.PROMINENCE_COUNT)
    create_force_fields(Config.SUN_RADIUS)
    setup_camera()

    if Config.BAKE:
        bake(domain)
    else:
        print("[sun_nasa] Skipping bake (--bake 0)", flush=True)

    if Config.RENDER:
        render_previews()
    else:
        print("[sun_nasa] Skipping render (--render 0)", flush=True)

    blend_path = Path(Config.OUTPUT_DIR).resolve() / "sun_nasa_groundtruth_v1.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    print(f"[sun_nasa] Saved: {blend_path}", flush=True)


if __name__ == "__main__":
    main()


