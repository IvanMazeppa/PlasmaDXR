"""
GPT-5.2 — Supergiant Star (OpenVDB) - Blender 5.0+

Creates a single, self-contained volumetric **supergiant star** using Mantaflow GAS:
- A large spherical inflow emitter drives a turbulent "stellar interior"
- A smaller hotspot emitter near the surface adds evolving convection-like detail
- Domain caches to OpenVDB for NanoVDB conversion
- Saves a .blend and renders a still (optional animation)

Pipeline target: Blender 5.0 Mantaflow -> OpenVDB (.vdb) -> NanoVDB -> PlasmaDX-Clean.

Run (headless):
  blender -b -P assets/blender_scripts/GPT-5.2/blender_supergiant_star.py -- \
    --output_dir "/abs/path/to/out/SupergiantStarAsset" \
    --name "GPT-5-2_SupergiantStar" \
    --resolution 128 \
    --domain_size 10.0 \
    --star_radius 3.0 \
    --frame_end 96 \
    --bake 1 \
    --render_still 0 \
    --still_frame 60 \
    --render_anim 0 \
    --render_res 1920 1080

Notes (Blender 5.0 API verified via blender-manual MCP):
  - Fluid bake operators: bpy.ops.fluid.bake_all(), bake_data(), bake_noise(), etc.
  - FluidDomainSettings cache_data_format supports 'OPENVDB'
  - FluidDomainSettings openvdb_cache_compress_type enum is ['ZIP','NONE'] in the Python API
    (manual UI may mention BLOSC; scripts should fall back safely).
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
    Avoid the common “large white ball” problem in renders:
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
    # If a user explicitly wants Cycles, they can set --render_engine CYCLES.
    try:
        engines = {it.identifier for it in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    except Exception:
        engines = set()

    # Do not override if user already selected an engine explicitly later.
    if "BLENDER_EEVEE_NEXT" in engines:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    elif "BLENDER_EEVEE" in engines:
        scene.render.engine = "BLENDER_EEVEE"

    # If Cycles is available and selected, force CPU unless user disables this.
    try:
        cyc = scene.cycles

        # Lower samples by default to avoid long GPU kernels.
        _safe_rna_set(cyc, "samples", 16)

        # Prefer CPU rendering to avoid Windows GPU timeout.
        # NOTE: `Device` in the manual corresponds to CPU vs GPU Compute in Cycles render settings.
        _safe_enum_set(cyc, "device", "CPU")

        # Volume safety knobs (Biased ray marching): increase step size via step rate,
        # and limit max steps to guard against extreme render times.
        #
        # Blender exposes these as properties on Cycles render settings, but names can vary.
        # We try common RNA identifiers; if they don't exist, we simply skip.
        for step_rate_name in ("volume_step_rate", "volume_step_rate_render"):
            _safe_rna_set(cyc, step_rate_name, 2.0)
        for step_rate_vp_name in ("volume_step_rate_viewport",):
            _safe_rna_set(cyc, step_rate_vp_name, 4.0)
        for max_steps_name in ("volume_max_steps",):
            _safe_rna_set(cyc, max_steps_name, 128)

        # Keep path complexity low.
        _safe_rna_set(cyc, "volume_bounces", 0)
        _safe_rna_set(cyc, "max_bounces", 2)
        _safe_rna_set(cyc, "transparent_max_bounces", 2)
    except Exception:
        # Cycles may not be available or accessible; ignore.
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
    bpy.ops.object.camera_add(location=(0.0, -domain_size * 1.4, domain_size * 0.35))
    cam = bpy.context.active_object
    bpy.context.scene.camera = cam

    # Aim at origin
    target = (0.0, 0.0, 0.0)
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
    # Key light
    bpy.ops.object.light_add(type="AREA", location=(domain_size * 0.4, -domain_size * 0.5, domain_size * 0.5))
    key = bpy.context.active_object
    key.data.energy = 1500.0
    key.data.size = domain_size * 0.5

    # Rim light
    bpy.ops.object.light_add(type="POINT", location=(-domain_size * 0.6, domain_size * 0.3, domain_size * 0.4))
    rim = bpy.context.active_object
    rim.data.energy = 800.0


def create_domain(domain_size: float) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=domain_size, location=(0.0, 0.0, 0.0))
    domain = bpy.context.active_object
    domain.name = "StarDomain"

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

    # cache_precision exists in Blender 5 UI; guard in case API differs in a point release.
    _safe_enum_set(settings, "cache_precision", "HALF")

    settings.resolution_max = int(resolution)
    settings.use_adaptive_domain = True
    settings.adapt_threshold = 0.001
    settings.adapt_margin = 4

    # “Space” defaults: no gravity; turbulence comes from vorticity + noise + moving inflows.
    settings.gravity = (0.0, 0.0, 0.0)
    settings.alpha = 0.0  # density buoyancy
    settings.beta = 0.0   # heat buoyancy

    settings.vorticity = 0.75
    settings.use_noise = True
    settings.noise_scale = 2
    settings.noise_strength = 1.0
    settings.noise_pos_scale = 1.5
    settings.noise_time_anim = 0.35

    settings.use_dissolve_smoke = True
    settings.dissolve_speed = 6


def create_star_emitter(radius: float) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=(0.0, 0.0, 0.0), segments=64, ring_count=32)
    emitter = bpy.context.active_object
    emitter.name = "StarEmitter_Core"

    mod = emitter.modifiers.new(name="Fluid", type="FLUID")
    mod.fluid_type = "FLOW"
    flow = mod.flow_settings

    # Use smoke-only to keep density as the main scalar field.
    flow.flow_type = "SMOKE"
    flow.flow_behavior = "INFLOW"
    flow.use_inflow = True

    flow.density = 2.0
    flow.temperature = 2.5
    flow.use_initial_velocity = True
    flow.velocity_random = 0.35
    flow.velocity_normal = 0.10
    flow.velocity_coord = (0.0, 0.0, 0.0)

    return emitter


def create_hotspot_emitter(star_radius: float) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(radius=star_radius * 0.35, location=(star_radius * 0.55, 0.0, 0.0), segments=48, ring_count=24)
    emitter = bpy.context.active_object
    emitter.name = "StarEmitter_Hotspot"

    mod = emitter.modifiers.new(name="Fluid", type="FLUID")
    mod.fluid_type = "FLOW"
    flow = mod.flow_settings

    flow.flow_type = "SMOKE"
    flow.flow_behavior = "INFLOW"
    flow.use_inflow = True

    flow.density = 1.2
    flow.temperature = 3.5
    flow.use_initial_velocity = True
    flow.velocity_random = 0.6
    flow.velocity_normal = 0.25

    return emitter


def animate_emitters(core: bpy.types.Object, hotspot: bpy.types.Object, frame_start: int, frame_end: int) -> None:
    # Move core emitter in a small Lissajous-like loop to create evolving turbulence.
    def set_loc(obj, frame, x, y, z):
        bpy.context.scene.frame_set(frame)
        obj.location = (x, y, z)
        obj.keyframe_insert(data_path="location")

    r = 0.25
    zamp = 0.15
    mid = (frame_start + frame_end) // 2

    set_loc(core, frame_start, 0.0, 0.0, 0.0)
    set_loc(core, mid, r, -r * 0.7, zamp)
    set_loc(core, frame_end, -r * 0.8, r, -zamp)

    # Hotspot orbits around the star slowly.
    for f in (frame_start, mid, frame_end):
        t = (f - frame_start) / max(1, (frame_end - frame_start))
        ang = t * 2.0 * math.pi * 0.65
        x = math.cos(ang) * hotspot.location.x - math.sin(ang) * hotspot.location.y
        y = math.sin(ang) * hotspot.location.x + math.cos(ang) * hotspot.location.y
        z = 0.2 * math.sin(ang * 1.7)
        bpy.context.scene.frame_set(f)
        hotspot.location = (x, y, z)
        hotspot.keyframe_insert(data_path="location")


def assign_domain_volume_material(domain: bpy.types.Object) -> None:
    mat = bpy.data.materials.new("StarVolumeMaterial")
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

    # Warm supergiant emission (looks good in Cycles; in PlasmaDX emission is computed differently)
    pv.inputs["Color"].default_value = (1.0, 0.55, 0.25, 1.0)
    pv.inputs["Density"].default_value = 2.5
    pv.inputs["Anisotropy"].default_value = 0.25
    # High values can saturate to pure white in Cycles; keep it strong but not insane.
    pv.inputs["Emission Strength"].default_value = 14.0

    links.new(pv.outputs["Volume"], out.inputs["Volume"])

    if domain.data.materials:
        domain.data.materials[0] = mat
    else:
        domain.data.materials.append(mat)


def ensure_domain_active(domain: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    domain.select_set(True)
    bpy.context.view_layer.objects.active = domain


def bake(domain: bpy.types.Object) -> bool:
    ensure_domain_active(domain)

    # Make sure we're in object mode (common cause of operator failure when run interactively).
    try:
        if bpy.context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass

    # Bake starting from frame_start to avoid weird partial-cache states.
    try:
        bpy.context.scene.frame_set(int(bpy.context.scene.frame_start))
    except Exception:
        pass

    # Free existing cache if possible (but never crash if it fails).
    try:
        bpy.ops.fluid.free_all()
    except Exception as e:
        print(f"[{ASSET_TAG} SupergiantStar] WARNING: bpy.ops.fluid.free_all() failed: {e}")

    # Bake (this can take minutes; if it fails, report and return False).
    try:
        bpy.ops.fluid.bake_all()
        return True
    except Exception as e:
        print(f"[{ASSET_TAG} SupergiantStar] ERROR: bpy.ops.fluid.bake_all() failed: {e}")
        print(f"[{ASSET_TAG} SupergiantStar] Tip: Try --resolution 64 and a shorter --frame_end to validate the pipeline.")
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
    parser.add_argument("--name", type=str, default="GPT-5-2_SupergiantStar")
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--domain_size", type=float, default=10.0)
    parser.add_argument("--star_radius", type=float, default=3.0)
    parser.add_argument("--frame_start", type=int, default=1)
    parser.add_argument("--frame_end", type=int, default=96)
    parser.add_argument("--bake", type=int, default=1)
    parser.add_argument("--render_still", type=int, default=0)
    parser.add_argument("--still_frame", type=int, default=60)
    parser.add_argument("--render_anim", type=int, default=0)
    parser.add_argument("--render_res", type=int, nargs=2, default=(1280, 720))
    parser.add_argument("--show_emitters", type=int, default=1, help="1=keep emitter meshes visible in viewport (still hidden in renders)")
    parser.add_argument("--tdr_safe", type=int, default=1, help="1=apply conservative render settings to avoid Windows GPU TDR")
    parser.add_argument(
        "--render_engine",
        type=str,
        default="AUTO",
        help="AUTO (default), CYCLES, BLENDER_EEVEE_NEXT, BLENDER_EEVEE",
    )
    parser.add_argument("--cycles_device", type=str, default="CPU", help="CPU (default) or GPU (only used when render_engine=CYCLES)")

    # Blender passes args after "--".
    #
    # NOTE: Some Blender builds expose `bpy.app.argv`, others do not (observed in Blender 5).
    # `sys.argv` is always available, so prefer that and fall back to `bpy.app.argv` only when present.
    argv_src = list(sys.argv)
    try:
        if hasattr(bpy, "app") and hasattr(bpy.app, "argv"):
            argv_src = list(bpy.app.argv)  # type: ignore[attr-defined]
    except Exception:
        pass

    argv = []
    if "--" in argv_src:
        argv = argv_src[argv_src.index("--") + 1 :]
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    out_dir = _ensure_dir(_resolve_output_dir(str(args.output_dir), default_subdir="gpt52_supergiant_star_out"))
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

    # Apply conservative defaults first (helps in interactive sessions too).
    apply_tdr_safe_render_preset(enable=int(args.tdr_safe) != 0)

    # Only pick a potentially-expensive render engine if we're actually rendering.
    want_render = int(args.render_still) == 1 or int(args.render_anim) == 1

    if want_render:
        requested_engine = str(args.render_engine).upper()
        engines = set()
        try:
            engines = {it.identifier for it in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
        except Exception:
            engines = set()

        if requested_engine != "AUTO" and (not engines or requested_engine in engines):
            try:
                bpy.context.scene.render.engine = requested_engine
            except Exception:
                pass
        else:
            # Auto: prefer EEVEE Next first (more interactive / lower TDR risk),
            # then EEVEE, and only then Cycles.
            for candidate in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
                if not engines or candidate in engines:
                    try:
                        bpy.context.scene.render.engine = candidate
                        break
                    except Exception:
                        continue

        # If we ended up on Cycles, force device selection (CPU by default).
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

    # Light/camera
    create_camera(args.domain_size)
    create_lights(args.domain_size)

    # Domain + emitters
    domain = create_domain(args.domain_size)
    assign_domain_volume_material(domain)

    settings = domain.modifiers["Fluid"].domain_settings
    configure_domain_cache(
        settings,
        cache_dir=cache_dir,
        frame_start=int(args.frame_start),
        frame_end=int(args.frame_end),
        resolution=int(args.resolution),
    )

    core = create_star_emitter(args.star_radius)
    hotspot = create_hotspot_emitter(args.star_radius)
    animate_emitters(core, hotspot, int(args.frame_start), int(args.frame_end))

    # Prevent emitter meshes from showing up as a “big white ball” in renders.
    show_emitters = int(args.show_emitters) != 0
    _hide_emitter(core, show_in_viewport=show_emitters)
    _hide_emitter(hotspot, show_in_viewport=show_emitters)

    # Save project (helps debugging / reproducibility)
    save_blend(blend_path)

    if int(args.bake) == 1:
        print(f"[{ASSET_TAG} SupergiantStar] Baking (this can take minutes)...")
        baked_ok = bake(domain)
        print(f"[{ASSET_TAG} SupergiantStar] Bake done: {baked_ok}")
    else:
        baked_ok = False

    # Render outputs
    if int(args.render_still) == 1:
        still_path = renders_dir / f"{args.name}_still.png"
        render_still(still_path, int(args.still_frame))

    if int(args.render_anim) == 1:
        render_animation(renders_dir / f"{args.name}_anim", int(args.frame_start), int(args.frame_end))

    print(f"[{ASSET_TAG} SupergiantStar] Done. Output dir: {out_dir}")
    print(f"[{ASSET_TAG} SupergiantStar] Render engine: {engine}")
    print(f"[{ASSET_TAG} SupergiantStar] Cache dir: {cache_dir}")


if __name__ == "__main__":
    main()


