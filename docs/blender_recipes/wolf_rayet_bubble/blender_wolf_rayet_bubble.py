"""
GPT-5.2 — Wolf-Rayet Bubble Nebula (OpenVDB) - Blender 5.0+

Creates a multi-shell Wolf-Rayet (WR) bubble nebula using the Three Wind Model:
- Inner fast wind from the WR star (current epoch)
- Outer slow wind from previous red supergiant phase
- Shell interaction creates characteristic bubble morphology

Key visual characteristics (based on astronomical observations):
- Bubble/shell structure (like NGC 6888 Crescent Nebula, Sharpless 308)
- Strong OIII emission (blue-green color)
- Asymmetric "break-out" structures where wind pierces weaker regions
- Clumpy, filamentary structure

Pipeline target: Blender 5.0 Mantaflow -> OpenVDB (.vdb) -> NanoVDB -> PlasmaDX-Clean.

Run (headless):
  blender -b -P docs/blender_recipes/GPT-5-2_Scripts_Docs_Advice/blender_wolf_rayet_bubble.py -- \
    --output_dir "/abs/path/to/out/WolfRayetBubble" \
    --name "GPT-5-2_WolfRayetBubble" \
    --resolution 96 \
    --domain_size 8.0 \
    --bubble_radius 2.5 \
    --frame_end 120 \
    --bake 1 \
    --render_still 0 \
    --still_frame 80 \
    --render_anim 0 \
    --render_res 1920 1080

Notes (Blender 5.0 API verified via blender-manual MCP):
  - Fluid bake operators: bpy.ops.fluid.bake_all(), bake_data(), bake_noise(), etc.
  - FluidDomainSettings cache_data_format supports 'OPENVDB'
  - FluidDomainSettings openvdb_cache_compress_type enum is ['ZIP','NONE'] in the Python API
    (manual UI may mention BLOSC; scripts should fall back safely).

Astronomical Sources:
  - Wolf-Rayet nebula morphology: Wikipedia + A&A WISE study (2015)
  - Three Wind Model: WR ring nebula formation via shock fronts
  - Visual references: NGC 6888 (Crescent), Sharpless 308, WR 31a bubble
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy

ASSET_TAG = "GPT-5.2"


def _safe_enum_set(obj, prop_name: str, desired: str) -> bool:
    """Attempt to set an enum property to desired. Returns True if set."""
    try:
        prop = obj.bl_rna.properties[prop_name]
        valid = {it.identifier for it in prop.enum_items}
        if desired in valid:
            setattr(obj, prop_name, desired)
            return True
        return False
    except Exception:
        return False


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_rna_set(obj, prop_name: str, value) -> bool:
    """
    Best-effort setter for Blender RNA properties.
    Returns True when the property exists AND assignment succeeds.
    """
    try:
        if not hasattr(obj, "bl_rna"):
            return False
        if prop_name not in obj.bl_rna.properties:
            return False
        setattr(obj, prop_name, value)
        return True
    except Exception:
        return False


def _resolve_output_dir(output_dir: str, default_subdir: str) -> Path:
    """
    Robust output directory resolution for both:
    - headless: `blender -b -P script.py -- --output_dir ...`
    - interactive: run from Blender Text Editor with an unsaved/saved .blend
    """
    if output_dir:
        return Path(output_dir).expanduser().resolve()

    # Prefer .blend-relative paths when possible
    try:
        base = Path(bpy.path.abspath("//")).resolve()
        if str(base) and base.exists():
            return base / default_subdir
    except Exception:
        pass

    # Next best: the script file's directory (available when run via -P)
    try:
        if "__file__" in globals():
            return Path(__file__).resolve().parent / default_subdir  # type: ignore[name-defined]
    except Exception:
        pass

    # Fallback: current working directory
    return Path.cwd().resolve() / default_subdir


def _hide_emitter(obj: bpy.types.Object, *, show_in_viewport: bool) -> None:
    """
    Avoid the common "large white ball" problem in renders:
    emitter meshes render as opaque surfaces unless hidden.
    """
    try:
        obj.hide_render = True
    except Exception:
        pass

    # Keep the emitter visible as wire in the viewport unless explicitly hidden.
    if not show_in_viewport:
        try:
            obj.hide_viewport = True
        except Exception:
            pass
    else:
        try:
            obj.display_type = "WIRE"
        except Exception:
            pass


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for datablocks in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.textures,
        bpy.data.images,
        bpy.data.lights,
        bpy.data.cameras,
    ):
        for db in list(datablocks):
            try:
                datablocks.remove(db)
            except Exception:
                pass


def set_render_engine_prefer_cycles() -> str:
    engines = set()
    try:
        engines = {it.identifier for it in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    except Exception:
        engines = set()

    if "CYCLES" in engines:
        bpy.context.scene.render.engine = "CYCLES"
        return "CYCLES"
    if "BLENDER_EEVEE_NEXT" in engines:
        bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
        return "BLENDER_EEVEE_NEXT"
    if "BLENDER_EEVEE" in engines:
        bpy.context.scene.render.engine = "BLENDER_EEVEE"
        return "BLENDER_EEVEE"
    return bpy.context.scene.render.engine


def apply_tdr_safe_render_preset(*, enable: bool) -> None:
    """
    Apply a conservative render preset designed to avoid Windows GPU TDR timeouts.

    Motivation (Blender 5 manual):
    - GPU rendering on Windows can trigger a driver timeout on heavy scenes/volumes.
    - Mitigations include using CPU device, increasing volume step size, and limiting max steps.
      See: `render/cycles/render_settings/volumes.html` and `render/cycles/gpu_rendering.html`.
    """
    if not enable:
        return

    scene = bpy.context.scene

    # Prefer EEVEE for viewport/interaction by default (lighter than Cycles path tracing).
    try:
        engines = {it.identifier for it in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    except Exception:
        engines = set()

    if "BLENDER_EEVEE_NEXT" in engines:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    elif "BLENDER_EEVEE" in engines:
        scene.render.engine = "BLENDER_EEVEE"

    # If Cycles is available and selected, force CPU unless user disables this.
    try:
        cyc = scene.cycles

        _safe_rna_set(cyc, "samples", 16)
        _safe_enum_set(cyc, "device", "CPU")

        for step_rate_name in ("volume_step_rate", "volume_step_rate_render"):
            _safe_rna_set(cyc, step_rate_name, 2.0)
        for step_rate_vp_name in ("volume_step_rate_viewport",):
            _safe_rna_set(cyc, step_rate_vp_name, 4.0)
        for max_steps_name in ("volume_max_steps",):
            _safe_rna_set(cyc, max_steps_name, 128)

        _safe_rna_set(cyc, "volume_bounces", 0)
        _safe_rna_set(cyc, "max_bounces", 2)
        _safe_rna_set(cyc, "transparent_max_bounces", 2)
    except Exception:
        pass


def make_black_world() -> None:
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    for n in list(nt.nodes):
        if n.type in {"BACKGROUND", "OUTPUT_WORLD"}:
            continue
    bg = next((n for n in nt.nodes if n.type == "BACKGROUND"), None)
    if bg:
        bg.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)
        bg.inputs[1].default_value = 0.0


def create_camera(domain_size: float) -> bpy.types.Object:
    bpy.ops.object.camera_add(location=(0.0, -domain_size * 1.8, domain_size * 0.5))
    cam = bpy.context.active_object
    bpy.context.scene.camera = cam

    # Aim at center of domain
    target = (0.0, 0.0, domain_size * 0.5)
    direction = (target[0] - cam.location.x, target[1] - cam.location.y, target[2] - cam.location.z)
    rot_quat = direction_to_quat(direction)
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = rot_quat
    return cam


def direction_to_quat(direction) -> tuple[float, float, float, float]:
    import mathutils

    v = mathutils.Vector(direction)
    if v.length < 1e-8:
        return (1.0, 0.0, 0.0, 0.0)
    v.normalize()
    # Blender camera looks down -Z in local space
    forward = mathutils.Vector((0.0, 0.0, -1.0))
    q = forward.rotation_difference(v)
    return (q.w, q.x, q.y, q.z)


def create_lights(domain_size: float) -> None:
    # Key light (simulating hot WR star)
    bpy.ops.object.light_add(type="POINT", location=(0.0, 0.0, domain_size * 0.5))
    star_light = bpy.context.active_object
    star_light.name = "WR_StarLight"
    star_light.data.energy = 3000.0
    star_light.data.color = (0.7, 0.85, 1.0)  # Blue-white

    # Rim light
    bpy.ops.object.light_add(type="AREA", location=(domain_size * 0.8, -domain_size * 0.6, domain_size * 0.8))
    rim = bpy.context.active_object
    rim.data.energy = 1200.0
    rim.data.size = domain_size * 0.4


def create_domain(domain_size: float, center_z: float) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=domain_size, location=(0.0, 0.0, center_z))
    domain = bpy.context.active_object
    domain.name = "WR_BubbleDomain"

    mod = domain.modifiers.new(name="Fluid", type="FLUID")
    mod.fluid_type = "DOMAIN"
    settings = mod.domain_settings
    settings.domain_type = "GAS"
    return domain


def configure_domain_cache(settings, cache_dir: Path, frame_start: int, frame_end: int, resolution: int) -> None:
    settings.cache_type = "ALL"
    settings.cache_directory = str(cache_dir)
    settings.cache_frame_start = frame_start
    settings.cache_frame_end = frame_end

    settings.cache_data_format = "OPENVDB"
    # Blender 5 Python API: openvdb_cache_compress_type is ['ZIP','NONE'] (no BLOSC)
    if not _safe_enum_set(settings, "openvdb_cache_compress_type", "ZIP"):
        _safe_enum_set(settings, "openvdb_cache_compress_type", "NONE")

    _safe_enum_set(settings, "cache_precision", "HALF")

    settings.resolution_max = int(resolution)
    settings.use_adaptive_domain = True
    settings.adapt_threshold = 0.001
    settings.adapt_margin = 4

    # "Space" defaults: no gravity; turbulence from vorticity + noise
    settings.gravity = (0.0, 0.0, 0.0)
    settings.alpha = 0.0  # density buoyancy
    settings.beta = 0.0   # heat buoyancy

    # High vorticity for turbulent bubble structure
    settings.vorticity = 0.85
    settings.use_noise = True
    settings.noise_scale = 3
    settings.noise_strength = 1.5
    settings.noise_pos_scale = 2.0
    settings.noise_time_anim = 0.25

    # Slow dissolve to maintain bubble structure
    settings.use_dissolve_smoke = True
    settings.dissolve_speed = 180


def create_inner_wind_emitter(bubble_radius: float, domain_center_z: float) -> bpy.types.Object:
    """
    Create the inner fast wind emitter (current WR star wind).
    This drives the high-velocity expansion that creates the inner cavity.
    """
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=bubble_radius * 0.15,
        location=(0.0, 0.0, domain_center_z),
        segments=32,
        ring_count=16
    )
    emitter = bpy.context.active_object
    emitter.name = "WR_InnerWindEmitter"

    mod = emitter.modifiers.new(name="Fluid", type="FLUID")
    mod.fluid_type = "FLOW"
    flow = mod.flow_settings

    flow.flow_type = "SMOKE"
    flow.flow_behavior = "INFLOW"
    flow.use_inflow = True

    # High-velocity outward wind (1000-3000 km/s scaled to sim)
    flow.density = 0.6  # Lower density for hot wind
    flow.temperature = 5.0  # Very hot (WR stars are 30,000-200,000 K)
    flow.use_initial_velocity = True
    flow.velocity_normal = 1.8  # Strong outward push
    flow.velocity_random = 0.2  # Some variation

    return emitter


def create_outer_shell_emitter(bubble_radius: float, domain_center_z: float) -> bpy.types.Object:
    """
    Create the outer slow wind shell (previous red supergiant wind).
    This denser, slower material gets swept up by the fast wind.
    """
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=bubble_radius * 0.85,
        location=(0.0, 0.0, domain_center_z),
        segments=48,
        ring_count=24
    )
    emitter = bpy.context.active_object
    emitter.name = "WR_OuterShellEmitter"

    mod = emitter.modifiers.new(name="Fluid", type="FLUID")
    mod.fluid_type = "FLOW"
    flow = mod.flow_settings

    flow.flow_type = "SMOKE"
    flow.flow_behavior = "INFLOW"
    flow.use_inflow = True

    # Denser, slower wind from previous mass-loss epoch
    flow.density = 1.8  # Higher density
    flow.temperature = 1.5  # Cooler (red supergiant phase)
    flow.use_initial_velocity = True
    flow.velocity_normal = 0.4  # Slower outward expansion
    flow.velocity_random = 0.3

    return emitter


def create_breakout_emitter(bubble_radius: float, domain_center_z: float) -> bpy.types.Object:
    """
    Create an asymmetric "break-out" emitter.
    This simulates where the fast wind punches through weaker regions of the shell,
    creating the characteristic asymmetric bulges seen in WR nebulae like NGC 6888.
    """
    # Position offset from center to create asymmetry
    offset_x = bubble_radius * 0.6
    offset_z = bubble_radius * 0.4

    bpy.ops.mesh.primitive_cone_add(
        radius1=bubble_radius * 0.2,
        radius2=bubble_radius * 0.05,
        depth=bubble_radius * 0.4,
        location=(offset_x, 0.0, domain_center_z + offset_z),
        rotation=(0.0, math.radians(45.0), 0.0)  # Tilted outward
    )
    emitter = bpy.context.active_object
    emitter.name = "WR_BreakoutEmitter"

    mod = emitter.modifiers.new(name="Fluid", type="FLUID")
    mod.fluid_type = "FLOW"
    flow = mod.flow_settings

    flow.flow_type = "SMOKE"
    flow.flow_behavior = "INFLOW"
    flow.use_inflow = True

    # High-velocity breakout jet
    flow.density = 1.2
    flow.temperature = 4.0
    flow.use_initial_velocity = True
    flow.velocity_normal = 2.0  # Very fast
    flow.velocity_random = 0.15

    return emitter


def animate_emitters(
    inner: bpy.types.Object,
    outer: bpy.types.Object,
    breakout: bpy.types.Object,
    frame_start: int,
    frame_end: int,
    domain_center_z: float
) -> None:
    """
    Animate emitters to create evolving structure.
    - Inner wind: pulsates slightly (irregular mass loss)
    - Outer shell: slowly rotates (precession)
    - Breakout: moves direction over time (wind clumping)
    """
    def set_scale(obj, frame, scale):
        bpy.context.scene.frame_set(frame)
        obj.scale = scale
        obj.keyframe_insert(data_path="scale")

    def set_rotation(obj, frame, rot):
        bpy.context.scene.frame_set(frame)
        obj.rotation_euler = rot
        obj.keyframe_insert(data_path="rotation_euler")

    # Inner wind: slight pulsation
    mid = (frame_start + frame_end) // 2
    set_scale(inner, frame_start, (1.0, 1.0, 1.0))
    set_scale(inner, mid, (1.15, 1.15, 1.15))  # Pulse outward
    set_scale(inner, frame_end, (0.95, 0.95, 0.95))  # Contract slightly

    # Outer shell: slow precession
    set_rotation(outer, frame_start, (0.0, 0.0, 0.0))
    set_rotation(outer, frame_end, (math.radians(15.0), math.radians(10.0), math.radians(45.0)))

    # Breakout: sweeping motion
    set_rotation(breakout, frame_start, (0.0, math.radians(45.0), 0.0))
    set_rotation(breakout, mid, (math.radians(10.0), math.radians(60.0), math.radians(20.0)))
    set_rotation(breakout, frame_end, (math.radians(-10.0), math.radians(30.0), math.radians(40.0)))


def assign_domain_volume_material(domain: bpy.types.Object) -> None:
    """
    Create a volume material appropriate for Wolf-Rayet nebulae:
    - Blue-green dominant (strong OIII emission at 500.7nm)
    - Some pink/red (H-alpha at 656.3nm)
    - Self-emission (ionized gas glows)
    """
    mat = bpy.data.materials.new("WR_BubbleMaterial")
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links

    # Clear default nodes
    for n in list(nodes):
        nodes.remove(n)

    out = nodes.new(type="ShaderNodeOutputMaterial")
    out.location = (400, 0)

    pv = nodes.new(type="ShaderNodeVolumePrincipled")
    pv.location = (0, 0)

    # Blue-green OIII emission color (characteristic of WR nebulae)
    pv.inputs["Color"].default_value = (0.3, 0.75, 0.85, 1.0)  # Teal/cyan
    pv.inputs["Density"].default_value = 1.2
    pv.inputs["Anisotropy"].default_value = -0.2  # Slight backward scattering
    pv.inputs["Absorption Color"].default_value = (0.05, 0.1, 0.15, 1.0)
    pv.inputs["Emission Strength"].default_value = 8.0
    pv.inputs["Emission Color"].default_value = (0.4, 0.8, 0.9, 1.0)  # Blue-green emission

    links.new(pv.outputs["Volume"], out.inputs["Volume"])

    if domain.data.materials:
        domain.data.materials[0] = mat
    else:
        domain.data.materials.append(mat)


def add_turbulence_force_field(domain_size: float, domain_center_z: float) -> bpy.types.Object | None:
    """
    Add a turbulence force field to create clumpy, filamentary structure
    characteristic of WR nebulae.
    """
    if not hasattr(bpy.ops.object, "effector_add"):
        return None

    try:
        bpy.ops.object.effector_add(type="TURBULENCE", location=(0.0, 0.0, domain_center_z))
        turb = bpy.context.active_object
        turb.name = "WR_Turbulence"
        turb.field.strength = 25.0
        turb.field.size = domain_size * 0.4
        turb.field.flow = 0.8  # Add some flow to the turbulence
        return turb
    except Exception:
        return None


def ensure_domain_active(domain: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    domain.select_set(True)
    bpy.context.view_layer.objects.active = domain


def bake(domain: bpy.types.Object) -> bool:
    ensure_domain_active(domain)

    try:
        if bpy.context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass

    try:
        bpy.context.scene.frame_set(int(bpy.context.scene.frame_start))
    except Exception:
        pass

    try:
        bpy.ops.fluid.free_all()
    except Exception as e:
        print(f"[{ASSET_TAG} WolfRayetBubble] WARNING: bpy.ops.fluid.free_all() failed: {e}")

    try:
        bpy.ops.fluid.bake_all()
        return True
    except Exception as e:
        print(f"[{ASSET_TAG} WolfRayetBubble] ERROR: bpy.ops.fluid.bake_all() failed: {e}")
        print(f"[{ASSET_TAG} WolfRayetBubble] Tip: Try --resolution 64 and a shorter --frame_end to validate the pipeline.")
        return False


def save_blend(filepath: Path) -> None:
    bpy.ops.wm.save_as_mainfile(filepath=str(filepath))


def render_still(output_path: Path, frame: int) -> None:
    scene = bpy.context.scene
    _ensure_dir(output_path.parent)
    scene.render.filepath = str(output_path)
    scene.frame_set(frame)
    bpy.ops.render.render(write_still=True)


def render_animation(output_dir: Path, frame_start: int, frame_end: int) -> None:
    scene = bpy.context.scene
    _ensure_dir(output_dir)
    scene.render.filepath = str(output_dir / "frame_")
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    bpy.ops.render.render(animation=True, write_still=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--name", type=str, default="GPT-5-2_WolfRayetBubble")
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--domain_size", type=float, default=8.0)
    parser.add_argument("--bubble_radius", type=float, default=2.5)
    parser.add_argument("--frame_start", type=int, default=1)
    parser.add_argument("--frame_end", type=int, default=120)
    parser.add_argument("--bake", type=int, default=1)
    parser.add_argument("--render_still", type=int, default=0)
    parser.add_argument("--still_frame", type=int, default=80)
    parser.add_argument("--render_anim", type=int, default=0)
    parser.add_argument("--render_res", type=int, nargs=2, default=(1920, 1080))
    parser.add_argument("--show_emitters", type=int, default=1, help="1=keep emitter meshes visible in viewport")
    parser.add_argument("--tdr_safe", type=int, default=1, help="1=apply conservative render settings to avoid Windows GPU TDR")
    parser.add_argument("--render_engine", type=str, default="AUTO", help="AUTO, CYCLES, BLENDER_EEVEE_NEXT, BLENDER_EEVEE")
    parser.add_argument("--cycles_device", type=str, default="CPU", help="CPU (default) or GPU")
    parser.add_argument("--enable_breakout", type=int, default=1, help="1=include asymmetric breakout emitter")

    argv_src = list(sys.argv)
    try:
        if hasattr(bpy, "app") and hasattr(bpy.app, "argv"):
            argv_src = list(bpy.app.argv)
    except Exception:
        pass

    argv = []
    if "--" in argv_src:
        argv = argv_src[argv_src.index("--") + 1 :]
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    out_dir = _ensure_dir(_resolve_output_dir(str(args.output_dir), default_subdir="gpt52_wolf_rayet_bubble_out"))
    cache_dir = _ensure_dir(out_dir / "vdb_cache")
    renders_dir = _ensure_dir(out_dir / "renders")
    blend_path = out_dir / f"{args.name}.blend"

    clear_scene()

    scene = bpy.context.scene
    scene.frame_start = int(args.frame_start)
    scene.frame_end = int(args.frame_end)
    scene.render.resolution_x = int(args.render_res[0])
    scene.render.resolution_y = int(args.render_res[1])
    scene.render.film_transparent = True

    # Apply TDR-safe settings
    apply_tdr_safe_render_preset(enable=int(args.tdr_safe) != 0)

    # Handle render engine selection
    want_render = int(args.render_still) == 1 or int(args.render_anim) == 1

    if want_render:
        requested_engine = str(args.render_engine).upper()
        engines = set()
        try:
            engines = {it.identifier for it in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
        except Exception:
            pass

        if requested_engine != "AUTO" and (not engines or requested_engine in engines):
            try:
                bpy.context.scene.render.engine = requested_engine
            except Exception:
                pass
        else:
            for candidate in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
                if not engines or candidate in engines:
                    try:
                        bpy.context.scene.render.engine = candidate
                        break
                    except Exception:
                        continue

        try:
            if bpy.context.scene.render.engine == "CYCLES":
                cyc = bpy.context.scene.cycles
                desired = str(args.cycles_device).upper()
                if desired not in ("CPU", "GPU"):
                    desired = "CPU"
                _safe_enum_set(cyc, "device", desired)
        except Exception:
            pass

    engine = bpy.context.scene.render.engine
    make_black_world()

    # Calculate domain center
    domain_center_z = args.domain_size * 0.5

    # Create camera and lights
    create_camera(args.domain_size)
    create_lights(args.domain_size)

    # Create domain
    domain = create_domain(args.domain_size, domain_center_z)
    assign_domain_volume_material(domain)

    settings = domain.modifiers["Fluid"].domain_settings
    configure_domain_cache(
        settings,
        cache_dir=cache_dir,
        frame_start=int(args.frame_start),
        frame_end=int(args.frame_end),
        resolution=int(args.resolution),
    )

    # Create the Three Wind Model emitters
    inner_emitter = create_inner_wind_emitter(args.bubble_radius, domain_center_z)
    outer_emitter = create_outer_shell_emitter(args.bubble_radius, domain_center_z)

    breakout_emitter = None
    if int(args.enable_breakout) == 1:
        breakout_emitter = create_breakout_emitter(args.bubble_radius, domain_center_z)

    # Animate emitters for evolving structure
    animate_emitters(
        inner_emitter,
        outer_emitter,
        breakout_emitter if breakout_emitter else inner_emitter,  # Use inner if no breakout
        int(args.frame_start),
        int(args.frame_end),
        domain_center_z
    )

    # Add turbulence for clumpy structure
    turbulence = add_turbulence_force_field(args.domain_size, domain_center_z)

    # Hide emitters from renders
    show_emitters = int(args.show_emitters) != 0
    _hide_emitter(inner_emitter, show_in_viewport=show_emitters)
    _hide_emitter(outer_emitter, show_in_viewport=show_emitters)
    if breakout_emitter:
        _hide_emitter(breakout_emitter, show_in_viewport=show_emitters)

    # Save project before baking
    save_blend(blend_path)

    if int(args.bake) == 1:
        print(f"[{ASSET_TAG} WolfRayetBubble] Baking Wolf-Rayet bubble nebula (this can take minutes)...")
        baked_ok = bake(domain)
        print(f"[{ASSET_TAG} WolfRayetBubble] Bake done: {baked_ok}")
    else:
        baked_ok = False

    # Render outputs
    if int(args.render_still) == 1:
        still_path = renders_dir / f"{args.name}_still.png"
        render_still(still_path, int(args.still_frame))

    if int(args.render_anim) == 1:
        render_animation(renders_dir / f"{args.name}_anim", int(args.frame_start), int(args.frame_end))

    print(f"[{ASSET_TAG} WolfRayetBubble] Done. Output dir: {out_dir}")
    print(f"[{ASSET_TAG} WolfRayetBubble] Render engine: {engine}")
    print(f"[{ASSET_TAG} WolfRayetBubble] Cache dir: {cache_dir}")
    print(f"")
    print(f"[{ASSET_TAG} WolfRayetBubble] Wolf-Rayet Bubble Nebula Features:")
    print(f"  - Three Wind Model: Inner fast wind + outer slow shell + breakout")
    print(f"  - OIII emission color (blue-green characteristic of WR nebulae)")
    print(f"  - Asymmetric breakout structure: {bool(breakout_emitter)}")
    print(f"  - Turbulence force field: {bool(turbulence)}")


if __name__ == "__main__":
    main()
