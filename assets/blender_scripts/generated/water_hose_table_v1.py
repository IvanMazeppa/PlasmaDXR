#!/usr/bin/env python3
"""
Water Hose Spray onto Table (Mesh-based Liquid) - Blender 5.0+

Goal:
- A hose spraying water onto a cluttered table until it saturates the surface/items,
  then water starts dripping off the table edges into a catch area.

Design notes:
- Reuses the proven mesh-based LIQUID domain settings from water_pouring_bowl_v3.
- Uses a small cluster of inflow nozzles to create a wider spray pattern (more "hose-like").
- Extends cache frame range beyond the old 250-frame default to a longer shot.

Usage:
    blender --background --python water_hose_table_v1.py -- [options]

Options:
    --resolution INT       Domain resolution (default: 96)
    --frame_start INT      Start frame (default: 1)
    --frame_end INT        End frame & cache end frame (default: 360)
    --spray_end INT        Frame to stop spraying (default: 300)
    --output_dir PATH      Output directory for cache/renders (default: build/vdb_output/water_hose_table_v1)
    --clean_output BOOL    Delete output_dir before run (default: 1)
    --bake BOOL            Run bake (1) or skip (0) (default: 1)
    --render BOOL          Render preview frames (1) or skip (0) (default: 1)
    --render_frames MODE   "key" | "all" | "mid" | "comma,separated,frames" (default: key)
"""

import bpy
import sys
import math
import shutil
from pathlib import Path

try:
    import mathutils
except Exception:
    mathutils = None


class Config:
    # Simulation
    RESOLUTION = 96
    FRAME_START = 1
    FRAME_END = 360
    SPRAY_END = 300

    # Output
    OUTPUT_DIR = "/home/maz3ppa/projects/PlasmaDXR/build/vdb_output/water_hose_table_v1"
    CLEAN_OUTPUT = True
    BAKE = True
    RENDER = True
    RENDER_FRAMES = "key"

    # Domain bounds (bigger than bowl scene; includes space under table for drips)
    DOMAIN_SCALE = (6.0, 4.0, 4.0)  # x,y,z scale of a unit cube
    DOMAIN_LOCATION = (0.0, 0.0, 1.7)  # center; z range ~= [-2.3, 5.7]

    # Table
    TABLE_TOP_SIZE = (3.4, 2.2, 0.12)  # x,y,z half-extents after scaling
    TABLE_TOP_Z = 0.95
    TABLE_LEG_THICKNESS = 0.12
    TABLE_LEG_HEIGHT = 0.9

    # Hose spray
    NOZZLE_CLUSTER_COUNT = 3
    NOZZLE_RADIUS = 0.07
    # IMPORTANT: keep emitter volume small so it doesn't look like it's "filling" from the source
    NOZZLE_DEPTH = 0.08
    NOZZLE_ORIGIN = (-2.4, -0.6, 1.9)
    NOZZLE_TARGET = (-0.2, 0.0, 1.0)  # aim point on table
    SPRAY_VELOCITY = 7.5
    SPRAY_SPREAD = 0.18  # spacing for cluster emitters

    # Collision robustness (copied from successful bowl scene)
    EFFECTOR_SURFACE_DISTANCE = 0.15
    EFFECTOR_SUBFRAMES = 5

    # Render
    RENDER_RESOLUTION_X = 768
    RENDER_RESOLUTION_Y = 432
    RENDER_SAMPLES = 128

    # Drain basin (prevents long-running sims from just filling the whole domain)
    USE_DRAIN_BASIN = True
    BASIN_SIZE = (4.6, 3.2, 0.45)  # x,y,z scale of outer cube
    BASIN_Z = 0.10
    BASIN_WALL_THICKNESS = 0.10
    DRAIN_SIZE = (0.45, 0.45, 0.35)  # small outflow volume inside basin
    DRAIN_LOCATION = (1.8, 1.1, 0.15)


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--resolution" and i + 1 < len(argv):
            Config.RESOLUTION = int(argv[i + 1]); i += 2
        elif arg == "--frame_start" and i + 1 < len(argv):
            Config.FRAME_START = int(argv[i + 1]); i += 2
        elif arg == "--frame_end" and i + 1 < len(argv):
            Config.FRAME_END = int(argv[i + 1]); i += 2
        elif arg == "--spray_end" and i + 1 < len(argv):
            Config.SPRAY_END = int(argv[i + 1]); i += 2
        elif arg == "--output_dir" and i + 1 < len(argv):
            Config.OUTPUT_DIR = argv[i + 1]; i += 2
        elif arg == "--clean_output" and i + 1 < len(argv):
            Config.CLEAN_OUTPUT = argv[i + 1].lower() in ("1", "true", "yes"); i += 2
        elif arg == "--bake" and i + 1 < len(argv):
            Config.BAKE = argv[i + 1].lower() in ("1", "true", "yes"); i += 2
        elif arg == "--render" and i + 1 < len(argv):
            Config.RENDER = argv[i + 1].lower() in ("1", "true", "yes"); i += 2
        elif arg == "--render_frames" and i + 1 < len(argv):
            Config.RENDER_FRAMES = argv[i + 1]; i += 2
        else:
            i += 1

    # Clamp spray end into frame range
    Config.SPRAY_END = max(Config.FRAME_START, min(Config.SPRAY_END, Config.FRAME_END))


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def setup_output_dir():
    out = Path(Config.OUTPUT_DIR)
    if Config.CLEAN_OUTPUT and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    return out


def setup_scene():
    scene = bpy.context.scene
    scene.frame_start = Config.FRAME_START
    scene.frame_end = Config.FRAME_END
    scene.render.engine = "CYCLES"

    # Keep render device selection simple for headless runs
    if hasattr(scene, "cycles"):
        try:
            scene.cycles.device = "GPU"
        except Exception:
            scene.cycles.device = "CPU"

        scene.cycles.caustics_reflective = True
        scene.cycles.caustics_refractive = True


def add_effector(obj, surface_distance=None, subframes=None):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_add(type="FLUID")
    obj.modifiers["Fluid"].fluid_type = "EFFECTOR"

    eff = obj.modifiers["Fluid"].effector_settings
    eff.effector_type = "COLLISION"
    eff.use_effector = True
    eff.surface_distance = surface_distance if surface_distance is not None else Config.EFFECTOR_SURFACE_DISTANCE
    eff.subframes = subframes if subframes is not None else Config.EFFECTOR_SUBFRAMES
    return eff


def create_domain():
    bpy.ops.mesh.primitive_cube_add(size=1, location=Config.DOMAIN_LOCATION)
    domain = bpy.context.active_object
    domain.name = "LiquidDomain"

    domain.scale = Config.DOMAIN_SCALE
    bpy.ops.object.transform_apply(scale=True)

    bpy.ops.object.modifier_add(type="FLUID")
    domain.modifiers["Fluid"].fluid_type = "DOMAIN"
    settings = domain.modifiers["Fluid"].domain_settings
    settings.domain_type = "LIQUID"

    # --- Settings lifted from extracted bowl v3 blend ---
    settings.resolution_max = Config.RESOLUTION
    settings.use_adaptive_domain = False
    settings.flip_ratio = 0.85
    settings.timesteps_min = 2
    settings.timesteps_max = 6
    settings.cfl_condition = 3.0
    settings.gravity = (0, 0, -9.81)

    # Cache: IMPORTANT to extend beyond 250 as requested
    settings.cache_type = "ALL"
    settings.cache_directory = Config.OUTPUT_DIR
    settings.cache_data_format = "OPENVDB"
    settings.cache_frame_start = Config.FRAME_START
    settings.cache_frame_end = Config.FRAME_END

    # Mesh generation (match v3 characteristics)
    settings.use_mesh = True
    settings.mesh_scale = 1
    settings.mesh_particle_radius = 2.0
    settings.mesh_concave_upper = 3.5
    settings.mesh_concave_lower = 0.4
    settings.mesh_smoothen_pos = 1
    settings.mesh_smoothen_neg = 1

    # Secondary particles
    settings.use_spray_particles = True
    settings.use_foam_particles = True
    settings.use_bubble_particles = True

    settings.sndparticle_sampling_wavecrest = 40
    settings.sndparticle_sampling_trappedair = 40
    settings.sndparticle_potential_min_wavecrest = 0.3
    settings.sndparticle_potential_max_wavecrest = 4.0
    settings.sndparticle_potential_min_energy = 0.1
    settings.sndparticle_potential_max_energy = 4.0
    settings.sndparticle_life_min = 15.0
    settings.sndparticle_life_max = 60.0

    domain.display_type = "WIRE"
    return domain


def _set_material_principled(obj, name, base_color=(0.5, 0.5, 0.5, 1.0), roughness=0.5, metallic=0.0):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = base_color
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic
    obj.data.materials.append(mat)


def add_water_material(domain):
    mat = bpy.data.materials.new(name="WaterMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    glass = nodes.new("ShaderNodeBsdfGlass")
    glass.location = (0, 0)
    glass.inputs["Color"].default_value = (0.85, 0.92, 1.0, 1.0)
    glass.inputs["Roughness"].default_value = 0.0
    glass.inputs["IOR"].default_value = 1.33

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (300, 0)
    links.new(glass.outputs["BSDF"], out.inputs["Surface"])

    domain.data.materials.append(mat)


def create_table():
    # Table top (solid block for robust collision)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, Config.TABLE_TOP_Z))
    top = bpy.context.active_object
    top.name = "TableTop"
    top.scale = (Config.TABLE_TOP_SIZE[0], Config.TABLE_TOP_SIZE[1], Config.TABLE_TOP_SIZE[2])
    bpy.ops.object.transform_apply(scale=True)
    _set_material_principled(top, "TableWood", base_color=(0.22, 0.16, 0.10, 1.0), roughness=0.65, metallic=0.0)
    add_effector(top)

    # Legs
    leg_x = Config.TABLE_TOP_SIZE[0] * 0.90
    leg_y = Config.TABLE_TOP_SIZE[1] * 0.90
    for sx in (-1, 1):
        for sy in (-1, 1):
            bpy.ops.mesh.primitive_cube_add(size=1, location=(sx * leg_x, sy * leg_y, Config.TABLE_TOP_Z - Config.TABLE_LEG_HEIGHT / 2))
            leg = bpy.context.active_object
            leg.name = f"TableLeg_{sx}_{sy}"
            leg.scale = (Config.TABLE_LEG_THICKNESS, Config.TABLE_LEG_THICKNESS, Config.TABLE_LEG_HEIGHT / 2)
            bpy.ops.object.transform_apply(scale=True)
            _set_material_principled(leg, "TableLegMat", base_color=(0.18, 0.12, 0.08, 1.0), roughness=0.7, metallic=0.0)
            add_effector(leg, surface_distance=0.12, subframes=3)

    return top


def create_clutter_items():
    items = []

    # Shallow tray (helps show pooling/saturation)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0.6, 0.4, Config.TABLE_TOP_Z + 0.08))
    tray_outer = bpy.context.active_object
    tray_outer.name = "TrayOuter"
    tray_outer.scale = (0.7, 0.5, 0.08)
    bpy.ops.object.transform_apply(scale=True)

    bpy.ops.mesh.primitive_cube_add(size=1, location=(0.6, 0.4, Config.TABLE_TOP_Z + 0.12))
    tray_inner = bpy.context.active_object
    tray_inner.name = "TrayInner"
    tray_inner.scale = (0.55, 0.38, 0.06)
    bpy.ops.object.transform_apply(scale=True)

    bpy.context.view_layer.objects.active = tray_outer
    mod = tray_outer.modifiers.new(name="TrayCut", type="BOOLEAN")
    mod.operation = "DIFFERENCE"
    mod.object = tray_inner
    bpy.ops.object.modifier_apply(modifier="TrayCut")
    bpy.data.objects.remove(tray_inner, do_unlink=True)

    tray = tray_outer
    tray.name = "Tray"
    _set_material_principled(tray, "TrayPlastic", base_color=(0.05, 0.05, 0.06, 1.0), roughness=0.35, metallic=0.0)
    add_effector(tray)
    items.append(tray)

    # Cup (solid for collision)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.16, depth=0.35, location=(-0.4, 0.35, Config.TABLE_TOP_Z + 0.18))
    cup = bpy.context.active_object
    cup.name = "Cup"
    _set_material_principled(cup, "CupCeramic", base_color=(0.85, 0.84, 0.82, 1.0), roughness=0.2, metallic=0.0)
    add_effector(cup, surface_distance=0.12, subframes=4)
    items.append(cup)

    # Plate-like disk (solid)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.35, depth=0.06, location=(0.1, -0.35, Config.TABLE_TOP_Z + 0.05))
    plate = bpy.context.active_object
    plate.name = "Plate"
    _set_material_principled(plate, "PlateMat", base_color=(0.75, 0.76, 0.78, 1.0), roughness=0.25, metallic=0.0)
    add_effector(plate, surface_distance=0.12, subframes=4)
    items.append(plate)

    # Sponge block (rough)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-0.8, -0.3, Config.TABLE_TOP_Z + 0.08))
    sponge = bpy.context.active_object
    sponge.name = "Sponge"
    sponge.scale = (0.22, 0.16, 0.08)
    bpy.ops.object.transform_apply(scale=True)
    _set_material_principled(sponge, "SpongeMat", base_color=(0.85, 0.78, 0.20, 1.0), roughness=0.9, metallic=0.0)
    add_effector(sponge, surface_distance=0.10, subframes=3)
    items.append(sponge)

    return items


def _aim_object_at(obj, target):
    if mathutils is None:
        return
    origin = mathutils.Vector(obj.location)
    tgt = mathutils.Vector(target)
    direction = (tgt - origin).normalized()
    # We want the cone's local +Z axis to point toward direction after rotation.
    quat = direction.to_track_quat("Z", "Y")
    obj.rotation_euler = quat.to_euler()

def _camera_look_at(cam, target):
    """Point camera at target. Uses -Z as forward (Blender camera forward)."""
    if mathutils is None:
        return
    origin = mathutils.Vector(cam.location)
    tgt = mathutils.Vector(target)
    forward = (tgt - origin).normalized()
    quat = forward.to_track_quat("-Z", "Y")
    cam.rotation_euler = quat.to_euler()


def create_hose_emitters():
    emitters = []

    # Visible hose (not a fluid object)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.06, depth=1.2, location=(-2.2, -0.6, 1.8))
    hose = bpy.context.active_object
    hose.name = "Hose"
    hose.rotation_euler = (0, math.radians(90), 0)
    bpy.ops.object.transform_apply(rotation=True)
    _set_material_principled(hose, "HoseRubber", base_color=(0.02, 0.02, 0.02, 1.0), roughness=0.8, metallic=0.0)

    # Clustered nozzles: multiple small inflows for a wider spray
    base_x, base_y, base_z = Config.NOZZLE_ORIGIN
    for i in range(Config.NOZZLE_CLUSTER_COUNT):
        offset = (i - (Config.NOZZLE_CLUSTER_COUNT - 1) / 2) * Config.SPRAY_SPREAD

        # Use a thin cylinder (small emitter volume) rather than a big cone.
        bpy.ops.mesh.primitive_cylinder_add(
            radius=Config.NOZZLE_RADIUS,
            depth=Config.NOZZLE_DEPTH,
            location=(base_x, base_y + offset, base_z + 0.02 * i),
        )
        nozzle = bpy.context.active_object
        nozzle.name = f"SprayNozzle_{i:02d}"

        # Aim each nozzle at a slightly different point on the table
        target = (
            Config.NOZZLE_TARGET[0],
            Config.NOZZLE_TARGET[1] + offset * 0.7,
            Config.NOZZLE_TARGET[2],
        )
        _aim_object_at(nozzle, target)
        bpy.ops.object.transform_apply(rotation=True)

        bpy.ops.object.modifier_add(type="FLUID")
        nozzle.modifiers["Fluid"].fluid_type = "FLOW"
        flow = nozzle.modifiers["Fluid"].flow_settings
        flow.flow_type = "LIQUID"
        flow.flow_behavior = "INFLOW"
        flow.use_inflow = True
        flow.use_initial_velocity = True
        flow.velocity_factor = 1.0

        # CRITICAL: Use velocity_normal (in the nozzle's local +Z direction).
        # This matches the proven pattern in water_pouring_bowl_v3 and avoids the
        # "fluid looks like it's filling the emitter volume" failure mode.
        flow.velocity_normal = Config.SPRAY_VELOCITY
        flow.velocity_coord = (0.0, 0.0, 0.0)

        nozzle.hide_render = True
        emitters.append(nozzle)

    return hose, emitters


def animate_spray(emitters):
    scene = bpy.context.scene
    for nozzle in emitters:
        flow = nozzle.modifiers["Fluid"].flow_settings

        scene.frame_set(Config.FRAME_START)
        flow.use_inflow = True
        flow.keyframe_insert(data_path="use_inflow")

        scene.frame_set(Config.SPRAY_END)
        flow.use_inflow = True
        flow.keyframe_insert(data_path="use_inflow")

        scene.frame_set(Config.SPRAY_END + 1)
        flow.use_inflow = False
        flow.keyframe_insert(data_path="use_inflow")

        scene.frame_set(Config.FRAME_END)
        flow.use_inflow = False
        flow.keyframe_insert(data_path="use_inflow")

    scene.frame_set(Config.FRAME_START)


def create_drain_basin():
    """Catch drips in a basin, then remove excess via an OUTFLOW drain volume.

    Without a drain, long sprays will simply accumulate and the domain will
    eventually look like it's "filling up" rather than showcasing dripping.
    """
    # Outer basin
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, Config.BASIN_Z))
    outer = bpy.context.active_object
    outer.name = "BasinOuter"
    outer.scale = (Config.BASIN_SIZE[0], Config.BASIN_SIZE[1], Config.BASIN_SIZE[2])
    bpy.ops.object.transform_apply(scale=True)

    # Inner cavity for boolean cut (open top)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, Config.BASIN_Z + Config.BASIN_WALL_THICKNESS))
    inner = bpy.context.active_object
    inner.name = "BasinInner"
    inner.scale = (
        max(0.1, Config.BASIN_SIZE[0] - Config.BASIN_WALL_THICKNESS),
        max(0.1, Config.BASIN_SIZE[1] - Config.BASIN_WALL_THICKNESS),
        max(0.1, Config.BASIN_SIZE[2] - Config.BASIN_WALL_THICKNESS),
    )
    bpy.ops.object.transform_apply(scale=True)

    bpy.context.view_layer.objects.active = outer
    mod = outer.modifiers.new(name="BasinCut", type="BOOLEAN")
    mod.operation = "DIFFERENCE"
    mod.object = inner
    bpy.ops.object.modifier_apply(modifier="BasinCut")
    bpy.data.objects.remove(inner, do_unlink=True)

    basin = outer
    basin.name = "DrainBasin"
    _set_material_principled(basin, "BasinMat", base_color=(0.08, 0.08, 0.09, 1.0), roughness=0.45, metallic=0.0)
    add_effector(basin, surface_distance=0.12, subframes=4)

    # Drain volume (OUTFLOW) inside basin
    bpy.ops.mesh.primitive_cube_add(size=1, location=Config.DRAIN_LOCATION)
    drain = bpy.context.active_object
    drain.name = "DrainOutflow"
    drain.scale = (Config.DRAIN_SIZE[0], Config.DRAIN_SIZE[1], Config.DRAIN_SIZE[2])
    bpy.ops.object.transform_apply(scale=True)

    bpy.ops.object.modifier_add(type="FLUID")
    drain.modifiers["Fluid"].fluid_type = "FLOW"
    flow = drain.modifiers["Fluid"].flow_settings
    flow.flow_type = "LIQUID"
    flow.flow_behavior = "OUTFLOW"
    drain.hide_render = True
    drain.display_type = "WIRE"

    return basin, drain


def setup_camera_and_lighting():
    # Camera: 3/4 view with table edges in frame
    bpy.ops.object.camera_add(location=(4.4, -4.0, 2.3))
    cam = bpy.context.active_object
    cam.name = "Camera"
    # Robustly point at the table center so we don't accidentally end up under the table.
    _camera_look_at(cam, (0.0, 0.0, Config.TABLE_TOP_Z + 0.05))
    cam.data.lens = 45
    bpy.context.scene.camera = cam

    # Key light
    bpy.ops.object.light_add(type="AREA", location=(2.8, -3.4, 4.0))
    key = bpy.context.active_object
    key.name = "KeyLight"
    key.data.energy = 500
    key.data.size = 2.8

    # Fill
    bpy.ops.object.light_add(type="AREA", location=(-3.2, -1.0, 2.2))
    fill = bpy.context.active_object
    fill.name = "FillLight"
    fill.data.energy = 180
    fill.data.size = 3.5

    # Rim/back for highlights on droplets
    bpy.ops.object.light_add(type="SPOT", location=(0.0, 3.2, 2.6))
    rim = bpy.context.active_object
    rim.name = "RimLight"
    rim.data.energy = 700
    rim.data.spot_size = math.radians(55)
    rim.rotation_euler = (math.radians(120), 0, math.radians(180))

    # World background
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    bg = nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.10, 0.12, 0.15, 1.0)
    bg.inputs["Strength"].default_value = 0.9
    out = nodes.new("ShaderNodeOutputWorld")
    out.location = (200, 0)
    world.node_tree.links.new(bg.outputs["Background"], out.inputs["Surface"])


def setup_render_settings():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    if hasattr(scene, "cycles"):
        try:
            scene.cycles.device = "GPU"
        except Exception:
            scene.cycles.device = "CPU"
        scene.cycles.samples = Config.RENDER_SAMPLES
        scene.cycles.use_denoising = True
        scene.cycles.caustics_reflective = True
        scene.cycles.caustics_refractive = True

    scene.render.resolution_x = Config.RENDER_RESOLUTION_X
    scene.render.resolution_y = Config.RENDER_RESOLUTION_Y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"


def get_frames_to_render():
    if Config.RENDER_FRAMES == "mid":
        return [(Config.FRAME_START + Config.FRAME_END) // 2]
    if Config.RENDER_FRAMES == "all":
        step = max(1, (Config.FRAME_END - Config.FRAME_START) // 8)
        return list(range(Config.FRAME_START, Config.FRAME_END + 1, step))
    if Config.RENDER_FRAMES == "key":
        return [
            20,
            60,
            120,
            180,
            240,
            min(Config.SPRAY_END, Config.FRAME_END),
            min(Config.SPRAY_END + 30, Config.FRAME_END),
            Config.FRAME_END,
        ]
    try:
        return [int(x.strip()) for x in Config.RENDER_FRAMES.split(",") if x.strip()]
    except Exception:
        return [(Config.FRAME_START + Config.FRAME_END) // 2]


def bake_simulation(domain):
    print("\n" + "=" * 70)
    print("BAKING LIQUID SIMULATION (HOSE → TABLE → DRIP)")
    print(f"Frames: {Config.FRAME_START}-{Config.FRAME_END} (spray until {Config.SPRAY_END})")
    print(f"Resolution: {Config.RESOLUTION}")
    print(f"Output: {Config.OUTPUT_DIR}")
    print("=" * 70 + "\n")

    bpy.ops.object.select_all(action="DESELECT")
    domain.select_set(True)
    bpy.context.view_layer.objects.active = domain
    bpy.ops.fluid.bake_all()
    print("[script] Bake complete!")


def render_previews(output_dir: Path):
    setup_render_settings()
    scene = bpy.context.scene

    frames = get_frames_to_render()
    print(f"[script] Rendering {len(frames)} frame(s): {frames}")

    for f in frames:
        f = max(Config.FRAME_START, min(f, Config.FRAME_END))
        scene.frame_set(f)
        out_path = output_dir / f"render_{f:04d}.png"
        scene.render.filepath = str(out_path)
        bpy.ops.render.render(write_still=True)
        print(f"[script] Saved: {out_path}")


def save_blend(output_dir: Path):
    path = output_dir / "water_hose_table.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(path))
    print(f"[script] Saved: {path}")


def main():
    parse_args()
    out_dir = setup_output_dir()

    clear_scene()
    setup_scene()

    domain = create_domain()
    create_table()
    create_clutter_items()
    if Config.USE_DRAIN_BASIN:
        create_drain_basin()
    hose, emitters = create_hose_emitters()
    animate_spray(emitters)

    add_water_material(domain)
    setup_camera_and_lighting()

    if Config.BAKE:
        bake_simulation(domain)
    else:
        print("[script] Skipping bake (--bake 0)")

    if Config.RENDER:
        render_previews(out_dir)
    else:
        print("[script] Skipping render (--render 0)")

    save_blend(out_dir)
    print(f"[script] Done. Output: {out_dir}")


if __name__ == "__main__":
    main()


