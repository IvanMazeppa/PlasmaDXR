"""
GPT-5.2 — Explosion Pipeline Test (Bake Setup) — Blender 5.0+

Goal:
- Create a small Mantaflow GAS “explosion burst” suitable for validating the
  Blender → OpenVDB → NanoVDB → PlasmaDX-Clean pipeline.

Safety (important):
- Blender 5.0 Windows `blender.exe` may crash when baking Mantaflow in headless mode (`--background`).
  This script defaults to *not* baking in headless mode unless explicitly allowed.

Typical CLI usage (WSL → Windows blender.exe):
  assets/blender_scripts/GPT-5.2/run_blender_cli.sh \
    assets/blender_scripts/GPT-5.2/explosions/blender_explosion_pipeline_test_bake.py -- \
    --bake 0

Then (recommended):
1) Open the saved `.blend` in Blender UI and Bake All.
2) Convert OpenVDB frames to NanoVDB (codec NONE, density grid):
   cd <out_dir> && bash convert_density_to_nvdb_nozip.sh
3) Load `.nvdb` files from `<out_dir>/nvdb_nozip/` in PlasmaDX (NanoVDB ImGui panel).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy

ASSET_TAG = "GPT-5.2"
DEFAULT_OUTPUT_SUBDIR = "build/blender_generated/GPT-5.2/explosions/pipeline_test_bake"


def _ctx_active_object() -> bpy.types.Object:
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
    dom.name = "ExplosionTest_Domain"
    fluid_mod = dom.modifiers.new(name="Fluid", type="FLUID")
    fluid_mod.fluid_type = "DOMAIN"
    dset = fluid_mod.domain_settings
    dset.domain_type = "GAS"
    return dom


def create_emitter(*, radius: float, z: float) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=(0.0, 0.0, z), segments=48, ring_count=24)
    em = _ctx_active_object()
    em.name = "ExplosionTest_Emitter"
    fmod = em.modifiers.new(name="Fluid", type="FLUID")
    fmod.fluid_type = "FLOW"
    fset = fmod.flow_settings

    # Fire+smoke if supported; otherwise smoke.
    if not _safe_enum_set(fset, "flow_type", "BOTH"):
        _safe_enum_set(fset, "flow_type", "SMOKE")
    _safe_enum_set(fset, "flow_behavior", "INFLOW")

    fset.use_initial_velocity = True
    fset.velocity_normal = 10.0
    fset.velocity_random = 3.0

    # Burst window: heavy at start, then quickly shut off.
    fset.density = 6.0
    try:
        fset.keyframe_insert(data_path="density", frame=1)
        fset.density = 0.0
        fset.keyframe_insert(data_path="density", frame=10)
    except Exception:
        pass

    if hasattr(fset, "temperature"):
        try:
            fset.temperature = 2.5
            fset.keyframe_insert(data_path="temperature", frame=1)
            fset.temperature = 0.0
            fset.keyframe_insert(data_path="temperature", frame=12)
        except Exception:
            pass

    return em


def add_forces(*, turbulence: float, vortex: float) -> None:
    bpy.ops.object.effector_add(type="TURBULENCE", location=(0.0, 0.0, 0.0))
    t = _ctx_active_object()
    t.name = "ExplosionTest_Turbulence"
    t.field.strength = turbulence
    t.field.noise = 1.0

    bpy.ops.object.effector_add(type="VORTEX", location=(0.0, 0.0, 0.0))
    v = _ctx_active_object()
    v.name = "ExplosionTest_Vortex"
    v.field.strength = vortex
    v.field.noise = 1.0


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
    dset.adapt_threshold = 0.002
    dset.adapt_margin = 6

    dset.vorticity = 0.9
    dset.gravity = (0.0, 0.0, -1.5)

    dset.use_noise = True
    dset.noise_scale = 2
    dset.noise_strength = 1.0
    dset.noise_time_anim = 0.35

    dset.use_dissolve_smoke = True
    dset.dissolve_speed = 35

    dset.cache_type = "ALL"
    dset.cache_data_format = "OPENVDB"

    # IMPORTANT: forward slashes to avoid escape issues in Manta scripts on Windows paths.
    dset.cache_directory = str(cache_dir).replace("\\", "/")
    dset.cache_frame_start = int(frame_start)
    dset.cache_frame_end = int(frame_end)

    # Compression enum naming varies by Blender build. Try best-effort.
    if not _safe_enum_set(dset, "openvdb_cache_compress_type", "ZIP"):
        _safe_enum_set(dset, "openvdb_cache_compress_type", "NONE")
    _safe_enum_set(dset, "cache_compression", "ZIP")

    if hasattr(dset, "cache_precision"):
        _safe_enum_set(dset, "cache_precision", str(cache_precision).upper())


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


def _write_nvdb_convert_helper(*, out_dir: Path, vdb_cache_dir: Path, nvdb_dir: Path, grid: str) -> Path:
    script_path = out_dir / "convert_density_to_nvdb_nozip.sh"
    rel_cache = "./" + vdb_cache_dir.relative_to(out_dir).as_posix()
    rel_nvdb = "./" + nvdb_dir.relative_to(out_dir).as_posix()

    content = f"""#!/usr/bin/env bash
set -euo pipefail

GRID=\"{grid}\"
CACHE_DIR=\"$(cd \"$(dirname \"$0\")\" && pwd)/{rel_cache.lstrip('./')}\"
OUT_DIR=\"$(cd \"$(dirname \"$0\")\" && pwd)/{rel_nvdb.lstrip('./')}\"

mkdir -p \"$OUT_DIR\"

if ! command -v nanovdb_convert >/dev/null 2>&1; then
  echo \"ERROR: nanovdb_convert not found on PATH (expected /usr/bin/nanovdb_convert in WSL).\" >&2
  exit 127
fi

mapfile -t VDBS < <(find \"$CACHE_DIR\" -type f -name '*.vdb' | sort)
if [[ ${{#VDBS[@]}} -eq 0 ]]; then
  echo \"ERROR: No .vdb files found under: $CACHE_DIR\" >&2
  exit 1
fi

echo \"Converting ${{#VDBS[@]}} VDB file(s) → NanoVDB (grid=$GRID, codec NONE)...\"

for vdb in \"${{VDBS[@]}}\"; do
  base=\"$(basename \"$vdb\" .vdb)\"
  out=\"$OUT_DIR/${{base}}_${{GRID}}_nozip.nvdb\"
  nanovdb_convert -f -g \"$GRID\" \"$vdb\" \"$out\"
done

echo \"Done. Output: $OUT_DIR\"
"""

    script_path.write_text(content, encoding="utf-8", newline="\n")
    return script_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--name", type=str, default="GPT-5-2_ExplosionPipelineTest")
    p.add_argument("--domain_size", type=float, default=6.0)
    p.add_argument("--resolution", type=int, default=96)
    p.add_argument("--frame_start", type=int, default=1)
    p.add_argument("--frame_end", type=int, default=48)
    p.add_argument("--cache_precision", type=str, default="FULL")
    p.add_argument("--bake", type=int, default=0)
    p.add_argument(
        "--allow_headless_bake",
        type=int,
        default=0,
        help="1=attempt Mantaflow bake in --background (Blender 5.0 Windows build may crash in Manta); default 0 skips headless bake",
    )
    p.add_argument("--factory_empty", type=int, default=1)
    p.add_argument("--turbulence", type=float, default=2.5)
    p.add_argument("--vortex", type=float, default=0.9)
    p.add_argument("--write_convert_helper", type=int, default=1)
    p.add_argument("--nvdb_subdir", type=str, default="nvdb_nozip")
    p.add_argument("--grid", type=str, default="density")
    argv = list(sys.argv)
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()

    out_dir = _ensure_dir(_resolve_output_dir(str(args.output_dir), default_subdir=DEFAULT_OUTPUT_SUBDIR))
    cache_dir = _ensure_dir(out_dir / "vdb_cache")
    nvdb_dir = _ensure_dir(out_dir / str(args.nvdb_subdir))

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

    emitter = create_emitter(radius=float(args.domain_size) * 0.12, z=float(args.domain_size) * 0.1)
    _hide_emitter(emitter)
    add_forces(turbulence=float(args.turbulence), vortex=float(args.vortex))

    # Save a reproducible .blend next to outputs
    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(out_dir / f"{args.name}.blend"))
    except Exception:
        pass

    if int(args.write_convert_helper) != 0:
        helper = _write_nvdb_convert_helper(
            out_dir=out_dir,
            vdb_cache_dir=cache_dir,
            nvdb_dir=nvdb_dir,
            grid=str(args.grid),
        )
        print(f"[{ASSET_TAG} ExplosionTest] Wrote helper: {helper}")
        print(f"[{ASSET_TAG} ExplosionTest] Convert: cd \"{out_dir}\" && bash \"{helper.name}\"")

    if int(args.bake) != 0:
        if getattr(bpy.app, "background", False) and int(args.allow_headless_bake) == 0:
            print(f"[{ASSET_TAG} ExplosionTest] WARNING: Mantaflow bake requested in headless mode.")
            print(f"[{ASSET_TAG} ExplosionTest] Blender 5.0 Windows build may crash in Manta when baking via CLI.")
            print(f"[{ASSET_TAG} ExplosionTest] Skipping bake. If you want to *try anyway*, pass --allow_headless_bake 1 (may crash).")
        else:
            print(f"[{ASSET_TAG} ExplosionTest] Baking to: {cache_dir}")
            bake(domain)
            print(f"[{ASSET_TAG} ExplosionTest] Bake complete.")

    print(f"[{ASSET_TAG} ExplosionTest] Output dir: {out_dir}")
    print(f"[{ASSET_TAG} ExplosionTest] Cache dir:  {cache_dir}")
    print(f"[{ASSET_TAG} ExplosionTest] NVDB dir:   {nvdb_dir}")


if __name__ == "__main__":
    main()
