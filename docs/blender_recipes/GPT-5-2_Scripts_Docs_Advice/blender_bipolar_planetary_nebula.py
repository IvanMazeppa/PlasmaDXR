"""
GPT-5.2 — Bipolar Planetary Nebula (OpenVDB) - Blender 5.0+
Creates a Mantaflow GAS domain with a torus "ring" emitter + two polar jet emitters,
bakes to OpenVDB cache, saves a .blend, and renders a still (and optionally an animation).

Pipeline target: Blender 5.0 Mantaflow -> OpenVDB (.vdb) -> NanoVDB -> PlasmaDX-Clean.

Run (headless):
  blender -b -P assets/blender_scripts/GPT-5.2/blender_bipolar_planetary_nebula.py -- \
    --output_dir "/abs/path/to/out" \
    --name "GPT-5-2_BipolarNebula" \
    --resolution 96 \
    --domain_size 6.0 \
    --frame_end 120 \
    --bake 1 \
    --render_still 1 \
    --still_frame 80 \
    --render_anim 0 \
    --render_res 1920 1080

Notes (Blender 5.0 API verified via blender-manual MCP):
  - Fluid bake operators: bpy.ops.fluid.bake_all(), bake_data(), bake_noise(), etc.
  - FluidDomainSettings cache_data_format supports 'OPENVDB'
  - FluidDomainSettings openvdb_cache_compress_type enum is ['ZIP','NONE'] in the Python API
    (manual UI may mention BLOSC; this script will fall back safely).
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import bpy


def _safe_enum_set(obj, prop_name: str, desired: str) -> bool:
    """
    Attempt to set an enum property to desired. If desired isn't available, no-op.
    Returns True if set, False otherwise.
    """
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


def clear_scene() -> None:
    """Remove all objects and orphan data blocks in the current scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Remove leftover data (best-effort; safe in background and interactive)
    for datablocks in (bpy.data.meshes, bpy.data.materials, bpy.data.textures, bpy.data.images, bpy.data.lights, bpy.data.cameras):
        for db in list(datablocks):
            try:
                datablocks.remove(db)
            except Exception:
                pass


def set_render_engine_prefer_cycles() -> str:
    """Choose a render engine with a robust fallback."""
    engines = set()
    try:
        engines = {it.identifier for it in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    except Exception:
        pass

    # Prefer Cycles, then EEVEE Next / EEVEE.
    for candidate in ("CYCLES", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        if not engines or candidate in engines:
            try:
                bpy.context.scene.render.engine = candidate
                return candidate
            except Exception:
                continue

    # Fallback to whatever Blender already had set.
    return bpy.context.scene.render.engine


def create_principled_volume_material(name: str) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    # Principled Volume
    vol = nodes.new(type="ShaderNodeVolumePrincipled")
    vol.location = (0, 0)
    vol.inputs["Color"].default_value = (0.35, 0.65, 0.95, 1.0)  # cyan-blue
    vol.inputs["Density"].default_value = 0.55
    vol.inputs["Anisotropy"].default_value = -0.25  # slight backward scattering
    vol.inputs["Absorption Color"].default_value = (0.02, 0.03, 0.05, 1.0)
    vol.inputs["Emission Strength"].default_value = 3.0
    vol.inputs["Emission Color"].default_value = (0.55, 0.85, 1.0, 1.0)

    out = nodes.new(type="ShaderNodeOutputMaterial")
    out.location = (320, 0)
    links.new(vol.outputs["Volume"], out.inputs["Volume"])
    return mat


def _select_activate(obj: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def create_bipolar_nebula_scene(
    *,
    name: str,
    domain_size: float,
    resolution: int,
    frame_start: int,
    frame_end: int,
    cache_dir: Path,
) -> dict:
    """
    Returns dict with keys: domain, ring, jet_pos, jet_neg, camera
    """
    scene = bpy.context.scene
    scene.frame_start = frame_start
    scene.frame_end = frame_end

    # Domain cube
    bpy.ops.mesh.primitive_cube_add(size=domain_size, location=(0.0, 0.0, domain_size * 0.5))
    domain = bpy.context.active_object
    domain.name = f"{name}_Domain"

    bpy.ops.object.modifier_add(type="FLUID")
    domain.modifiers["Fluid"].fluid_type = "DOMAIN"
    dset = domain.modifiers["Fluid"].domain_settings

    dset.domain_type = "GAS"
    dset.resolution_max = int(resolution)
    dset.use_adaptive_domain = True
    dset.adapt_threshold = 0.0015

    # "Space-like": remove gravity-driven buoyancy; keep turbulence via vorticity/noise.
    dset.gravity = (0.0, 0.0, 0.0)
    dset.alpha = 0.0
    dset.beta = 0.0
    dset.vorticity = 0.9

    dset.use_dissolve_smoke = True
    dset.dissolve_speed = 140

    # Add higher-frequency detail
    dset.use_noise = True
    dset.noise_scale = 2
    dset.noise_strength = 1.2
    dset.noise_time_anim = 0.15

    # Cache settings (OpenVDB)
    dset.cache_type = "ALL"
    dset.cache_data_format = "OPENVDB"
    dset.cache_directory = str(cache_dir)
    dset.cache_frame_start = int(frame_start)
    dset.cache_frame_end = int(frame_end)

    # Compression enum in Blender 5.0 Python API: ['ZIP','NONE'].
    # If future builds expose BLOSC, we will prefer it automatically.
    if not _safe_enum_set(dset, "openvdb_cache_compress_type", "BLOSC"):
        _safe_enum_set(dset, "openvdb_cache_compress_type", "ZIP")

    # Some Blender builds may expose precision control; keep best-effort and non-fatal.
    if hasattr(dset, "cache_precision"):
        _safe_enum_set(dset, "cache_precision", "HALF")

    # Material for preview/render
    mat = create_principled_volume_material(f"{name}_Material")
    if domain.data.materials:
        domain.data.materials[0] = mat
    else:
        domain.data.materials.append(mat)

    # Ring emitter (torus)
    bpy.ops.mesh.primitive_torus_add(
        major_radius=domain_size * 0.22,
        minor_radius=domain_size * 0.035,
        location=(0.0, 0.0, domain_size * 0.5),
        rotation=(math.radians(90.0), 0.0, 0.0),
    )
    ring = bpy.context.active_object
    ring.name = f"{name}_RingEmitter"
    bpy.ops.object.modifier_add(type="FLUID")
    ring.modifiers["Fluid"].fluid_type = "FLOW"
    f_ring = ring.modifiers["Fluid"].flow_settings
    f_ring.flow_type = "SMOKE"
    f_ring.flow_behavior = "INFLOW"
    f_ring.density = 1.0
    f_ring.temperature = 2.5
    f_ring.smoke_color = (0.35, 0.65, 0.95)
    f_ring.subframes = 1
    f_ring.use_initial_velocity = True
    f_ring.velocity_normal = 0.7
    f_ring.velocity_random = 0.2

    # Jet emitters (cones)
    def _make_jet(suffix: str, z_sign: float) -> bpy.types.Object:
        bpy.ops.mesh.primitive_cone_add(
            radius1=domain_size * 0.03,
            radius2=0.0,
            depth=domain_size * 0.35,
            location=(0.0, 0.0, domain_size * 0.5 + z_sign * domain_size * 0.03),
            rotation=(0.0, 0.0, 0.0) if z_sign > 0 else (math.radians(180.0), 0.0, 0.0),
        )
        jet = bpy.context.active_object
        jet.name = f"{name}_JetEmitter_{suffix}"
        bpy.ops.object.modifier_add(type="FLUID")
        jet.modifiers["Fluid"].fluid_type = "FLOW"
        f = jet.modifiers["Fluid"].flow_settings
        f.flow_type = "SMOKE"
        f.flow_behavior = "INFLOW"
        f.density = 1.35
        f.temperature = 4.0
        f.smoke_color = (0.85, 0.45, 1.0)  # magenta accent
        f.subframes = 1
        f.use_initial_velocity = True
        f.velocity_normal = 1.4
        f.velocity_random = 0.15
        return jet

    jet_pos = _make_jet("POS", +1.0)
    jet_neg = _make_jet("NEG", -1.0)

    # Animate emitters (gentle precession / rotation)
    def _key(obj: bpy.types.Object, frame: int, rot: tuple[float, float, float]) -> None:
        obj.rotation_euler = rot
        obj.keyframe_insert(data_path="rotation_euler", frame=frame)

    _key(ring, frame_start, (ring.rotation_euler.x, ring.rotation_euler.y, 0.0))
    _key(ring, frame_end, (ring.rotation_euler.x, ring.rotation_euler.y, math.radians(360.0)))

    _key(jet_pos, frame_start, (0.0, 0.0, 0.0))
    _key(jet_pos, frame_end, (math.radians(20.0), math.radians(-10.0), math.radians(180.0)))

    _key(jet_neg, frame_start, (math.radians(180.0), 0.0, 0.0))
    _key(jet_neg, frame_end, (math.radians(200.0), math.radians(10.0), math.radians(-180.0)))

    # Optional turbulence force field (guarded)
    if hasattr(bpy.ops.object, "effector_add"):
        try:
            bpy.ops.object.effector_add(type="TURBULENCE", location=(0.0, 0.0, domain_size * 0.5))
            turb = bpy.context.active_object
            turb.name = f"{name}_Turbulence"
            turb.field.strength = 18.0
            turb.field.size = domain_size * 0.35
        except Exception:
            pass

    # Camera + light
    bpy.ops.object.camera_add(location=(domain_size * 1.25, -domain_size * 1.35, domain_size * 0.75))
    cam = bpy.context.active_object
    cam.name = f"{name}_Camera"
    bpy.context.scene.camera = cam

    # Point the camera at the center
    center = (0.0, 0.0, domain_size * 0.5)
    direction = (
        center[0] - cam.location.x,
        center[1] - cam.location.y,
        center[2] - cam.location.z,
    )
    rot_z = math.atan2(direction[1], direction[0])
    dist_xy = math.sqrt(direction[0] ** 2 + direction[1] ** 2)
    rot_x = math.atan2(direction[2], dist_xy)
    cam.rotation_euler = (math.radians(90.0) - rot_x, 0.0, rot_z + math.radians(90.0))

    bpy.ops.object.light_add(type="SUN", location=(domain_size * 2.0, domain_size * 1.5, domain_size * 2.0))
    sun = bpy.context.active_object
    sun.data.energy = 1.8

    # Dark world background
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)
        bg.inputs[1].default_value = 1.0

    return {"domain": domain, "ring": ring, "jet_pos": jet_pos, "jet_neg": jet_neg, "camera": cam}


def bake_fluid(domain_obj: bpy.types.Object) -> None:
    _select_activate(domain_obj)
    # Ensure we're in object mode (safe in background)
    if bpy.context.mode != "OBJECT":
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            pass
    bpy.ops.fluid.bake_all()


def render_outputs(
    *,
    output_dir: Path,
    name: str,
    render_res: tuple[int, int],
    still_frame: int,
    render_still: bool,
    render_anim: bool,
) -> None:
    scene = bpy.context.scene
    scene.render.resolution_x = int(render_res[0])
    scene.render.resolution_y = int(render_res[1])
    scene.render.resolution_percentage = 100

    engine = set_render_engine_prefer_cycles()

    # Best-effort quality knobs (may differ by engine; keep guarded)
    if engine == "CYCLES" and hasattr(scene, "cycles"):
        scene.cycles.samples = 64
        scene.cycles.volume_bounces = 2
        scene.cycles.transparent_max_bounces = 2

    _ensure_dir(output_dir / "renders")
    if render_still:
        scene.frame_set(int(still_frame))
        scene.render.filepath = str(output_dir / "renders" / f"{name}_still.png")
        bpy.ops.render.render(write_still=True)

    if render_anim:
        scene.render.filepath = str(output_dir / "renders" / f"{name}_anim_")
        bpy.ops.render.render(animation=True)


def save_blend(output_dir: Path, name: str) -> Path:
    _ensure_dir(output_dir)
    blend_path = output_dir / f"{name}.blend"
    # save_as_mainfile is stable across Blender versions; keep best-effort.
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    return blend_path


def parse_args() -> argparse.Namespace:
    argv = list(getattr(__import__("sys"), "argv", []))
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--name", type=str, default="GPT-5-2_BipolarNebula")
    p.add_argument("--domain_size", type=float, default=6.0)
    p.add_argument("--resolution", type=int, default=96)
    p.add_argument("--frame_start", type=int, default=1)
    p.add_argument("--frame_end", type=int, default=120)
    p.add_argument("--bake", type=int, default=1)
    p.add_argument("--render_still", type=int, default=1)
    p.add_argument("--still_frame", type=int, default=80)
    p.add_argument("--render_anim", type=int, default=0)
    p.add_argument("--render_res", type=int, nargs=2, default=(1920, 1080))
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()

    # Determine output directory (prefer explicit absolute path)
    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        # If the blend is saved, // resolves; otherwise use CWD.
        try:
            base = Path(bpy.path.abspath("//")).resolve()
        except Exception:
            base = Path(os.getcwd()).resolve()
        out_dir = base / "bipolar_nebula_out"

    cache_dir = _ensure_dir(out_dir / "vdb_cache")

    print(f"[BipolarNebula] Output dir: {out_dir}")
    print(f"[BipolarNebula] Cache dir:  {cache_dir}")

    clear_scene()
    objs = create_bipolar_nebula_scene(
        name=args.name,
        domain_size=float(args.domain_size),
        resolution=int(args.resolution),
        frame_start=int(args.frame_start),
        frame_end=int(args.frame_end),
        cache_dir=cache_dir,
    )

    blend_path = save_blend(out_dir, args.name)
    print(f"[BipolarNebula] Saved .blend: {blend_path}")

    if int(args.bake) != 0:
        print("[BipolarNebula] Baking fluid simulation (this can take a while)...")
        bake_fluid(objs["domain"])
        print("[BipolarNebula] Bake complete.")
        print("[BipolarNebula] Expect OpenVDB frames in cache directory (subfolders may exist).")

    if int(args.render_still) != 0 or int(args.render_anim) != 0:
        print("[BipolarNebula] Rendering...")
        render_outputs(
            output_dir=out_dir,
            name=args.name,
            render_res=(int(args.render_res[0]), int(args.render_res[1])),
            still_frame=int(args.still_frame),
            render_still=int(args.render_still) != 0,
            render_anim=int(args.render_anim) != 0,
        )
        print("[BipolarNebula] Render done.")

    print("\n[BipolarNebula] Next steps (PlasmaDX pipeline):")
    print(f"  - Convert a frame: python scripts/convert_vdb_to_nvdb.py '<frame>.vdb'")
    print("  - Load .nvdb in PlasmaDX NanoVDBSystem and adjust density/emission/scattering in ImGui.")


if __name__ == "__main__":
    main()


