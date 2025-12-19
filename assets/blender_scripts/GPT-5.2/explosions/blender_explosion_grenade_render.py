"""
GPT-5.2 — Grenade Explosion (Render Script) — Blender 5.0+

Loads a VDB sequence as a Volume and renders an MP4.
Use this with VDB caches produced by `blender_explosion_grenade_bake.py` (or any similar VDB sequence).

Key API references (Blender 5):
- bpy.types.Volume: `filepath`, `is_sequence`, `frame_start`, `frame_duration`, `sequence_mode`
- bpy.types.FFmpegSettings: `codec`, `format`, `constant_rate_factor`
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import bpy

ASSET_TAG = "GPT-5.2"
DEFAULT_OUTPUT_SUBDIR = "build/blender_generated/GPT-5.2/explosions/grenade_render"

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


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


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


def create_camera_and_light() -> None:
    bpy.ops.object.camera_add(location=(0.0, -6.5, 2.6))
    cam = _ctx_active_object()
    cam.rotation_euler = (1.15, 0.0, 0.0)
    bpy.context.scene.camera = cam

    bpy.ops.object.light_add(type="POINT", location=(2.0, -3.0, 4.0))
    l = _ctx_active_object()
    l.data.energy = 700.0


def import_vdb_sequence(vdb_path: Path) -> bpy.types.Object:
    # Import one frame; Volume datablock will be set up to sequence if file pattern exists.
    bpy.ops.object.volume_import(filepath=str(vdb_path))
    vol_obj = _ctx_active_object()
    vol_obj.name = "GrenadeExplosion_Volume"
    return vol_obj


def set_volume_sequence(volume_data: bpy.types.Volume, *, frame_start: int, frame_end: int) -> None:
    # Best-effort; Blender determines sequence if filenames match.
    volume_data.frame_start = int(frame_start)
    volume_data.frame_duration = int(max(1, frame_end - frame_start + 1))
    volume_data.sequence_mode = "CLIP"


def assign_emissive_volume_material(vol_obj: bpy.types.Object) -> None:
    mat = bpy.data.materials.new("GrenadeExplosion_VolumeMat")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    pv = nt.nodes.new(type="ShaderNodeVolumePrincipled")
    pv.location = (0, 0)
    pv.inputs["Density"].default_value = 2.0
    pv.inputs["Anisotropy"].default_value = 0.2
    pv.inputs["Color"].default_value = (0.55, 0.55, 0.6, 1.0)
    pv.inputs["Absorption Color"].default_value = (0.05, 0.04, 0.06, 1.0)
    pv.inputs["Emission Strength"].default_value = 4.0
    pv.inputs["Emission Color"].default_value = (1.0, 0.45, 0.2, 1.0)

    out = nt.nodes.new(type="ShaderNodeOutputMaterial")
    out.location = (320, 0)
    nt.links.new(pv.outputs["Volume"], out.inputs["Volume"])

    if vol_obj.data.materials:
        vol_obj.data.materials[0] = mat
    else:
        vol_obj.data.materials.append(mat)


def configure_render_mp4(out_mp4: Path, *, res: tuple[int, int], fps: int) -> None:
    scene = bpy.context.scene
    scene.render.resolution_x = int(res[0])
    scene.render.resolution_y = int(res[1])
    scene.render.resolution_percentage = 100
    scene.render.fps = int(fps)

    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.filepath = str(out_mp4)
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.ffmpeg.audio_codec = "NONE"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--name", type=str, default="GPT-5-2_GrenadeExplosion")
    p.add_argument("--vdb_file", type=str, default="", help="Path to a VDB frame file (e.g. .../fluid_data_0001.vdb)")
    p.add_argument("--frame_start", type=int, default=1)
    p.add_argument("--frame_end", type=int, default=64)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--render_res", type=int, nargs=2, default=(1920, 1080))
    p.add_argument("--factory_empty", type=int, default=1)
    argv = list(sys.argv)
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    out_dir = _ensure_dir(_resolve_output_dir(str(args.output_dir), default_subdir=DEFAULT_OUTPUT_SUBDIR))
    renders_dir = _ensure_dir(out_dir / "renders")

    clear_scene(factory_empty=int(args.factory_empty) != 0)
    make_black_world()
    create_camera_and_light()

    scene = bpy.context.scene
    scene.frame_start = int(args.frame_start)
    scene.frame_end = int(args.frame_end)
    scene.render.film_transparent = True

    if args.vdb_file:
        vdb_file = Path(args.vdb_file).expanduser().resolve()
    else:
        # Default: assume paired bake output layout
        vdb_file = (out_dir / "vdb_cache" / "data" / "fluid_data_0001.vdb").resolve()

    vol_obj = import_vdb_sequence(vdb_file)
    assign_emissive_volume_material(vol_obj)
    set_volume_sequence(vol_obj.data, frame_start=int(args.frame_start), frame_end=int(args.frame_end))

    out_mp4 = renders_dir / f"{args.name}.mp4"
    configure_render_mp4(out_mp4, res=(int(args.render_res[0]), int(args.render_res[1])), fps=int(args.fps))

    # Save .blend next to outputs
    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(out_dir / f"{args.name}_render.blend"))
    except Exception:
        pass

    print(f"[{ASSET_TAG} GrenadeExplosion Render] Rendering animation to: {out_mp4}")
    bpy.ops.render.render(animation=True)
    print(f"[{ASSET_TAG} GrenadeExplosion Render] Done.")


if __name__ == "__main__":
    main()


