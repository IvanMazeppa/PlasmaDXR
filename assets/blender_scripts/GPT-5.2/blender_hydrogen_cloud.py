"""
GPT-5.2 — Hydrogen Cloud (OpenVDB) - Blender 5.0+

Purpose:
Create a simple emission nebula / hydrogen cloud using Mantaflow GAS and export OpenVDB caches.
Designed as a "clean, minimal" example for the Blender 5 → NanoVDB → PlasmaDX pipeline.

Run (headless):
  blender -b -P assets/blender_scripts/GPT-5.2/blender_hydrogen_cloud.py -- \
    --output_dir "/abs/path/to/out/HydrogenCloudAsset" \
    --name "GPT-5-2_HydrogenCloud" \
    --resolution 96 \
    --domain_size 6.0 \
    --frame_end 48 \
    --bake 1 \
    --render_still 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy

ASSET_TAG = "GPT-5.2"
DEFAULT_OUTPUT_SUBDIR = "build/blender_generated/GPT-5.2/hydrogen_cloud"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_enum_set(obj, prop_name: str, desired: str) -> bool:
    try:
        prop = obj.bl_rna.properties[prop_name]
        valid = {it.identifier for it in prop.enum_items}
        if desired in valid:
            setattr(obj, prop_name, desired)
            return True
        return False
    except Exception:
        return False


def _safe_rna_set(obj, prop_name: str, value) -> bool:
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
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    try:
        base = Path(bpy.path.abspath("//")).resolve()
        if str(base) and base.exists():
            return base / default_subdir
    except Exception:
        pass
    try:
        if "__file__" in globals():
            return Path(__file__).resolve().parent / default_subdir  # type: ignore[name-defined]
    except Exception:
        pass
    return Path.cwd().resolve() / default_subdir


def _hide_emitter(obj: bpy.types.Object, *, show_in_viewport: bool) -> None:
    try:
        obj.hide_render = True
    except Exception:
        pass
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


def apply_tdr_safe_render_preset(*, enable: bool) -> None:
    if not enable:
        return
    scene = bpy.context.scene
    try:
        engines = {it.identifier for it in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    except Exception:
        engines = set()
    if "BLENDER_EEVEE_NEXT" in engines:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    elif "BLENDER_EEVEE" in engines:
        scene.render.engine = "BLENDER_EEVEE"

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


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def make_black_world() -> None:
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)
        bg.inputs[1].default_value = 0.0


def create_domain(domain_size: float) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=domain_size, location=(0.0, 0.0, domain_size * 0.5))
    domain = bpy.context.active_object
    domain.name = "HydrogenCloud_Domain"
    bpy.ops.object.modifier_add(type="FLUID")
    domain.modifiers["Fluid"].fluid_type = "DOMAIN"
    dset = domain.modifiers["Fluid"].domain_settings
    dset.domain_type = "GAS"
    return domain


def configure_domain(dset, *, cache_dir: Path, frame_start: int, frame_end: int, resolution: int, cache_precision: str) -> None:
    dset.resolution_max = int(resolution)
    dset.use_adaptive_domain = True
    dset.adapt_threshold = 0.0015

    # Space-like motion
    dset.gravity = (0.0, 0.0, 0.0)
    dset.alpha = 0.0
    dset.beta = 0.0
    dset.vorticity = 0.6

    dset.use_noise = True
    dset.noise_scale = 2
    dset.noise_strength = 1.0
    dset.noise_time_anim = 0.2

    dset.use_dissolve_smoke = True
    dset.dissolve_speed = 80

    # Cache (OpenVDB)
    dset.cache_type = "ALL"
    dset.cache_data_format = "OPENVDB"
    # IMPORTANT: use forward slashes in cache paths to avoid Mantaflow path-escape issues on Windows.
    dset.cache_directory = str(cache_dir).replace("\\", "/")
    dset.cache_frame_start = int(frame_start)
    dset.cache_frame_end = int(frame_end)

    if not _safe_enum_set(dset, "openvdb_cache_compress_type", "ZIP"):
        _safe_enum_set(dset, "openvdb_cache_compress_type", "NONE")

    if hasattr(dset, "cache_precision"):
        _safe_enum_set(dset, "cache_precision", str(cache_precision).upper())


def create_emitter(domain_size: float) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(radius=domain_size * 0.12, location=(0.0, 0.0, domain_size * 0.5), segments=48, ring_count=24)
    emitter = bpy.context.active_object
    emitter.name = "HydrogenCloud_Emitter"
    bpy.ops.object.modifier_add(type="FLUID")
    emitter.modifiers["Fluid"].fluid_type = "FLOW"
    flow = emitter.modifiers["Fluid"].flow_settings
    flow.flow_type = "SMOKE"
    flow.flow_behavior = "INFLOW"
    flow.density = 1.0
    flow.temperature = 2.0
    flow.use_initial_velocity = True
    flow.velocity_normal = 0.6
    flow.velocity_random = 0.25
    return emitter


def assign_domain_material(domain: bpy.types.Object) -> None:
    mat = bpy.data.materials.new("HydrogenCloud_Volume")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    pv = nt.nodes.new(type="ShaderNodeVolumePrincipled")
    pv.location = (0, 0)
    pv.inputs["Color"].default_value = (0.9, 0.35, 0.5, 1.0)
    pv.inputs["Density"].default_value = 0.8
    pv.inputs["Anisotropy"].default_value = -0.2
    pv.inputs["Absorption Color"].default_value = (0.08, 0.06, 0.12, 1.0)
    pv.inputs["Emission Strength"].default_value = 2.0
    pv.inputs["Emission Color"].default_value = (0.95, 0.4, 0.55, 1.0)

    out = nt.nodes.new(type="ShaderNodeOutputMaterial")
    out.location = (320, 0)
    nt.links.new(pv.outputs["Volume"], out.inputs["Volume"])

    if domain.data.materials:
        domain.data.materials[0] = mat
    else:
        domain.data.materials.append(mat)


def ensure_active(obj: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def bake(domain: bpy.types.Object) -> bool:
    ensure_active(domain)
    try:
        if bpy.context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass
    try:
        bpy.ops.fluid.free_all()
    except Exception:
        pass
    try:
        bpy.ops.fluid.bake_all()
        return True
    except Exception as e:
        print(f"[{ASSET_TAG} HydrogenCloud] ERROR: bake_all failed: {e}")
        return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--name", type=str, default="GPT-5-2_HydrogenCloud")
    p.add_argument("--domain_size", type=float, default=6.0)
    p.add_argument("--resolution", type=int, default=96)
    p.add_argument("--frame_start", type=int, default=1)
    p.add_argument("--frame_end", type=int, default=48)
    p.add_argument("--bake", type=int, default=0)
    p.add_argument(
        "--allow_headless_bake",
        type=int,
        default=0,
        help="1=attempt Mantaflow bake in --background (Blender 5.0 Windows build may crash in Manta); default 0 skips headless bake",
    )
    p.add_argument("--render_still", type=int, default=0)
    p.add_argument("--still_frame", type=int, default=24)
    p.add_argument("--tdr_safe", type=int, default=1)
    p.add_argument("--cache_precision", type=str, default="FULL")
    argv = list(sys.argv)
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    out_dir = _ensure_dir(_resolve_output_dir(str(args.output_dir), default_subdir=DEFAULT_OUTPUT_SUBDIR))
    cache_dir = _ensure_dir(out_dir / "vdb_cache")
    renders_dir = _ensure_dir(out_dir / "renders")

    clear_scene()
    make_black_world()
    apply_tdr_safe_render_preset(enable=int(args.tdr_safe) != 0)

    scene = bpy.context.scene
    scene.frame_start = int(args.frame_start)
    scene.frame_end = int(args.frame_end)
    scene.render.film_transparent = True

    domain = create_domain(float(args.domain_size))
    assign_domain_material(domain)
    dset = domain.modifiers["Fluid"].domain_settings
    configure_domain(
        dset,
        cache_dir=cache_dir,
        frame_start=int(args.frame_start),
        frame_end=int(args.frame_end),
        resolution=int(args.resolution),
        cache_precision=str(args.cache_precision),
    )

    emitter = create_emitter(float(args.domain_size))
    _hide_emitter(emitter, show_in_viewport=True)

    # Save .blend for reproducibility (in output dir).
    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(out_dir / f"{args.name}.blend"))
    except Exception:
        pass

    if int(args.bake) == 1:
        if getattr(bpy.app, "background", False) and int(args.allow_headless_bake) == 0:
            print(f"[{ASSET_TAG} HydrogenCloud] WARNING: Mantaflow bake requested in headless mode.")
            print(f"[{ASSET_TAG} HydrogenCloud] Blender 5.0 Windows build is currently crashing in Manta when baking via CLI.")
            print(f"[{ASSET_TAG} HydrogenCloud] Skipping bake. If you want to *try anyway*, pass --allow_headless_bake 1 (may crash).")
        else:
            print(f"[{ASSET_TAG} HydrogenCloud] Baking...")
            ok = bake(domain)
            print(f"[{ASSET_TAG} HydrogenCloud] Bake done: {ok}")

    if int(args.render_still) == 1:
        scene.render.filepath = str(renders_dir / f"{args.name}_still.png")
        scene.frame_set(int(args.still_frame))
        bpy.ops.render.render(write_still=True)

    print(f"[{ASSET_TAG} HydrogenCloud] Done. Output dir: {out_dir}")
    print(f"[{ASSET_TAG} HydrogenCloud] Cache dir: {cache_dir}")


if __name__ == "__main__":
    main()


