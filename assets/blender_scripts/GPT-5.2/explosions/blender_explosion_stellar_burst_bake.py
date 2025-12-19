"""
GPT-5.2 — Stellar Burst Explosion (Bake Script) — Blender 5.0+

Mid-scale “starburst / nova” explosion:
- ring shock + core burst
- turbulence + vortex to create filamentary structure
- exports OpenVDB cache for NanoVDB conversion and/or rendering
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy

ASSET_TAG = "GPT-5.2"
DEFAULT_OUTPUT_SUBDIR = "build/blender_generated/GPT-5.2/explosions/stellar_burst_bake"

def _ctx_active_object() -> bpy.types.Object:
    """
    Blender context compatibility:
    Some builds/environments do not expose `bpy.context.active_object`.
    Prefer `context.object`, then `view_layer.objects.active`, then selected objects.
    """
    ctx = bpy.context
    obj = getattr(ctx, "active_object", None)
    if obj is None:
        obj = getattr(ctx, "object", None)
    if obj is None:
        try:
            obj = ctx.view_layer.objects.active
        except Exception:
            obj = None
    if obj is None:
        try:
            sel = list(getattr(ctx, "selected_objects", []) or [])
            if sel:
                obj = sel[-1]
        except Exception:
            obj = None
    if obj is None:
        raise RuntimeError("Failed to resolve active object from context after operator call.")
    return obj


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


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


def clear_scene(*, factory_empty: bool) -> None:
    if factory_empty:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        return
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def ensure_active(obj: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _hide_emitter(obj: bpy.types.Object) -> None:
    try:
        obj.hide_render = True
    except Exception:
        pass
    try:
        obj.display_type = "WIRE"
    except Exception:
        pass


def create_domain(*, size: float) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=size, location=(0.0, 0.0, 0.0))
    dom = _ctx_active_object()
    dom.name = "StellarBurst_Domain"
    fluid_mod = dom.modifiers.new(name="Fluid", type="FLUID")
    fluid_mod.fluid_type = "DOMAIN"
    dset = fluid_mod.domain_settings
    dset.domain_type = "GAS"
    return dom


def configure_domain(
    dset,
    *,
    resolution: int,
    frame_start: int,
    frame_end: int,
    cache_dir: Path,
    cache_precision: str,
) -> None:
    dset.resolution_max = int(resolution)
    dset.use_adaptive_domain = True
    dset.adapt_threshold = 0.0015
    dset.adapt_margin = 8

    dset.gravity = (0.0, 0.0, 0.0)
    dset.vorticity = 1.2

    dset.use_noise = True
    dset.noise_scale = 2
    dset.noise_strength = 1.2
    dset.noise_time_anim = 0.25

    dset.use_dissolve_smoke = True
    dset.dissolve_speed = 55

    dset.cache_type = "ALL"
    dset.cache_data_format = "OPENVDB"
    if not _safe_enum_set(dset, "openvdb_cache_compress_type", "ZIP"):
        _safe_enum_set(dset, "openvdb_cache_compress_type", "NONE")

    dset.cache_directory = str(cache_dir).replace("\\", "/")
    dset.cache_frame_start = int(frame_start)
    dset.cache_frame_end = int(frame_end)

    if hasattr(dset, "cache_precision"):
        _safe_enum_set(dset, "cache_precision", str(cache_precision).upper())


def create_ring_emitter(*, radius: float, z: float) -> bpy.types.Object:
    bpy.ops.mesh.primitive_torus_add(
        major_radius=radius,
        minor_radius=radius * 0.08,
        major_segments=72,
        minor_segments=16,
        location=(0.0, 0.0, z),
    )
    ring = _ctx_active_object()
    ring.name = "StellarBurst_RingEmitter"
    fmod = ring.modifiers.new(name="Fluid", type="FLUID")
    fmod.fluid_type = "FLOW"
    fset = fmod.flow_settings

    if not _safe_enum_set(fset, "flow_type", "BOTH"):
        _safe_enum_set(fset, "flow_type", "SMOKE")
    _safe_enum_set(fset, "flow_behavior", "INFLOW")

    fset.use_initial_velocity = True
    fset.velocity_normal = 10.0
    fset.velocity_random = 4.0

    # Burst window
    fset.density = 3.0
    try:
        fset.keyframe_insert(data_path="density", frame=1)
        fset.density = 0.0
        fset.keyframe_insert(data_path="density", frame=14)
    except Exception:
        pass

    if hasattr(fset, "temperature"):
        try:
            fset.temperature = 2.0
            fset.keyframe_insert(data_path="temperature", frame=1)
            fset.temperature = 0.0
            fset.keyframe_insert(data_path="temperature", frame=18)
        except Exception:
            pass

    return ring


def create_core_emitter(*, radius: float) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=(0.0, 0.0, 0.0), segments=48, ring_count=24)
    core = _ctx_active_object()
    core.name = "StellarBurst_CoreEmitter"
    fmod = core.modifiers.new(name="Fluid", type="FLUID")
    fmod.fluid_type = "FLOW"
    fset = fmod.flow_settings

    if not _safe_enum_set(fset, "flow_type", "BOTH"):
        _safe_enum_set(fset, "flow_type", "SMOKE")
    _safe_enum_set(fset, "flow_behavior", "INFLOW")

    fset.use_initial_velocity = True
    fset.velocity_normal = 14.0
    fset.velocity_random = 6.0

    fset.density = 7.0
    try:
        fset.keyframe_insert(data_path="density", frame=1)
        fset.density = 0.0
        fset.keyframe_insert(data_path="density", frame=8)
    except Exception:
        pass

    if hasattr(fset, "temperature"):
        try:
            fset.temperature = 3.0
            fset.keyframe_insert(data_path="temperature", frame=1)
            fset.temperature = 0.0
            fset.keyframe_insert(data_path="temperature", frame=10)
        except Exception:
            pass

    return core


def add_forces(*, turbulence: float, vortex: float) -> None:
    bpy.ops.object.effector_add(type="TURBULENCE", location=(0.0, 0.0, 0.0))
    t = _ctx_active_object()
    t.name = "StellarBurst_Turbulence"
    t.field.strength = turbulence
    t.field.noise = 1.0

    bpy.ops.object.effector_add(type="VORTEX", location=(0.0, 0.0, 0.0))
    v = _ctx_active_object()
    v.name = "StellarBurst_Vortex"
    v.field.strength = vortex
    v.field.noise = 1.0


def bake(domain: bpy.types.Object) -> None:
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
    bpy.ops.fluid.bake_all()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--name", type=str, default="GPT-5-2_StellarBurstExplosion")
    p.add_argument("--domain_size", type=float, default=10.0)
    p.add_argument("--resolution", type=int, default=160)
    p.add_argument("--frame_start", type=int, default=1)
    p.add_argument("--frame_end", type=int, default=96)
    p.add_argument("--cache_precision", type=str, default="FULL")
    p.add_argument("--bake", type=int, default=0)
    p.add_argument(
        "--allow_headless_bake",
        type=int,
        default=0,
        help="1=attempt Mantaflow bake in --background (Blender 5.0 Windows build may crash in Manta); default 0 skips headless bake",
    )
    p.add_argument("--factory_empty", type=int, default=1)
    p.add_argument("--turbulence", type=float, default=3.0)
    p.add_argument("--vortex", type=float, default=1.2)
    argv = list(sys.argv)
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    out_dir = _ensure_dir(_resolve_output_dir(str(args.output_dir), default_subdir=DEFAULT_OUTPUT_SUBDIR))
    cache_dir = _ensure_dir(out_dir / "vdb_cache")

    clear_scene(factory_empty=int(args.factory_empty) != 0)

    scene = bpy.context.scene
    scene.frame_start = int(args.frame_start)
    scene.frame_end = int(args.frame_end)

    domain = create_domain(size=float(args.domain_size))
    dset = domain.modifiers["Fluid"].domain_settings
    configure_domain(
        dset,
        resolution=int(args.resolution),
        frame_start=int(args.frame_start),
        frame_end=int(args.frame_end),
        cache_dir=cache_dir,
        cache_precision=str(args.cache_precision),
    )

    ring = create_ring_emitter(radius=float(args.domain_size) * 0.22, z=0.0)
    core = create_core_emitter(radius=float(args.domain_size) * 0.08)
    _hide_emitter(ring)
    _hide_emitter(core)
    add_forces(turbulence=float(args.turbulence), vortex=float(args.vortex))

    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(out_dir / f"{args.name}.blend"))
    except Exception:
        pass

    if int(args.bake) != 0:
        if getattr(bpy.app, "background", False) and int(args.allow_headless_bake) == 0:
            print(f"[{ASSET_TAG} StellarBurst] WARNING: Mantaflow bake requested in headless mode.")
            print(f"[{ASSET_TAG} StellarBurst] Blender 5.0 Windows build may crash in Manta when baking via CLI.")
            print(f"[{ASSET_TAG} StellarBurst] Skipping bake. If you want to *try anyway*, pass --allow_headless_bake 1 (may crash).")
        else:
            print(f"[{ASSET_TAG} StellarBurst] Baking to: {cache_dir}")
            bake(domain)
            print(f"[{ASSET_TAG} StellarBurst] Bake complete.")

    print(f"[{ASSET_TAG} StellarBurst] Output dir: {out_dir}")
    print(f"[{ASSET_TAG} StellarBurst] Cache dir:  {cache_dir}")


if __name__ == "__main__":
    main()

