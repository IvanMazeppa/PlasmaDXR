#!/usr/bin/env python3
"""
Convert liquid mesh sequence to VDB SDF volumes.

Liquid simulations produce mesh surfaces, not volumetric density grids.
This script converts each mesh frame to a VDB Signed Distance Field (SDF)
which can then be ray marched in real-time renderers.

Usage:
    blender --background --python mesh_to_vdb_sdf.py -- --blend_file /path/to/cache/simulation.blend --output_dir /path/to/output --frame_start 20 --frame_end 95 --step 5
"""

import bpy
import sys
import bmesh
import math
from pathlib import Path


def parse_args():
    """Parse arguments after '--'."""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    blend_file = None
    output_dir = None
    frame_start = 1
    frame_end = 100
    step = 1
    resolution = 0.05  # Voxel size in world units

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--blend_file" and i + 1 < len(argv):
            blend_file = argv[i + 1]
            i += 2
        elif arg == "--output_dir" and i + 1 < len(argv):
            output_dir = argv[i + 1]
            i += 2
        elif arg == "--frame_start" and i + 1 < len(argv):
            frame_start = int(argv[i + 1])
            i += 2
        elif arg == "--frame_end" and i + 1 < len(argv):
            frame_end = int(argv[i + 1])
            i += 2
        elif arg == "--step" and i + 1 < len(argv):
            step = int(argv[i + 1])
            i += 2
        elif arg == "--resolution" and i + 1 < len(argv):
            resolution = float(argv[i + 1])
            i += 2
        else:
            i += 1

    return blend_file, output_dir, frame_start, frame_end, step, resolution


def mesh_to_volume(obj, voxel_size=0.05, frame=1):
    """
    Convert a mesh object to a volume using Blender's Mesh to Volume modifier.

    Args:
        obj: The mesh object to convert
        voxel_size: Size of each voxel in world units
        frame: Current frame number

    Returns:
        Volume object or None if conversion failed
    """
    # Make sure we're working with evaluated mesh data
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)

    # Check if mesh has any faces
    if not eval_obj.data or len(eval_obj.data.polygons) == 0:
        return None

    # Create a new mesh object from evaluated data
    mesh_data = bpy.data.meshes.new_from_object(eval_obj)
    if not mesh_data or len(mesh_data.polygons) == 0:
        bpy.data.meshes.remove(mesh_data)
        return None

    temp_mesh_obj = bpy.data.objects.new(f"TempMesh_{frame:04d}", mesh_data)
    bpy.context.collection.objects.link(temp_mesh_obj)

    # Add Mesh to Volume modifier
    mod = temp_mesh_obj.modifiers.new(name="MeshToVolume", type='MESH_TO_VOLUME')
    mod.voxel_size = voxel_size
    mod.voxel_amount = 128  # Fallback resolution
    mod.resolution_mode = 'VOXEL_SIZE'
    mod.interior_band_width = 0.1  # Interior SDF distance

    # Create a volume object from the modifier
    bpy.context.view_layer.objects.active = temp_mesh_obj
    temp_mesh_obj.select_set(True)

    # Apply the modifier and convert to volume
    try:
        # Apply all modifiers
        bpy.ops.object.convert(target='VOLUME')
        volume_obj = bpy.context.active_object
        volume_obj.name = f"WaterVolume_{frame:04d}"

        return volume_obj
    except Exception as e:
        print(f"  Error converting mesh to volume: {e}")
        # Cleanup
        bpy.data.objects.remove(temp_mesh_obj)
        bpy.data.meshes.remove(mesh_data)
        return None


def export_volume_as_vdb(volume_obj, output_path):
    """
    Export a volume object as VDB file.
    """
    # Select only the volume
    bpy.ops.object.select_all(action='DESELECT')
    volume_obj.select_set(True)
    bpy.context.view_layer.objects.active = volume_obj

    # Use the File > Export > Volume menu
    # Note: Blender 4.0+ has a different export path
    try:
        # Try the new volume export
        bpy.ops.wm.vdb_export(filepath=str(output_path))
        return True
    except:
        pass

    try:
        # Try export_scene.vdb if available
        bpy.ops.export_scene.vdb(filepath=str(output_path))
        return True
    except:
        pass

    # Manual export via saving volume data
    # This is a workaround if standard exports don't work
    print(f"  Standard VDB export not available")
    return False


def find_liquid_domain(scene):
    """Find the liquid domain object in the scene."""
    for obj in scene.objects:
        if obj.type == 'MESH':
            for mod in obj.modifiers:
                if mod.type == 'FLUID':
                    if hasattr(mod, 'fluid_type') and mod.fluid_type == 'DOMAIN':
                        return obj
                    if hasattr(mod, 'domain_settings'):
                        if hasattr(mod.domain_settings, 'domain_type'):
                            if mod.domain_settings.domain_type == 'LIQUID':
                                return obj
    return None


def main():
    blend_file, output_dir, frame_start, frame_end, step, resolution = parse_args()

    print("=" * 60)
    print("Liquid Mesh to VDB SDF Converter")
    print("=" * 60)

    if not blend_file:
        print("ERROR: --blend_file required")
        print("Usage: blender --background --python mesh_to_vdb_sdf.py -- --blend_file sim.blend --output_dir ./vdb")
        return

    blend_path = Path(blend_file)
    if not blend_path.exists():
        print(f"ERROR: Blend file not found: {blend_file}")
        return

    if not output_dir:
        output_dir = str(blend_path.parent / "sdf_vdb")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Blend file: {blend_file}")
    print(f"Output:     {output_dir}")
    print(f"Frames:     {frame_start} - {frame_end} (step {step})")
    print(f"Resolution: {resolution} (voxel size)")
    print()

    # Open the blend file
    print("Loading blend file...")
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))

    scene = bpy.context.scene

    # Find liquid domain
    domain = find_liquid_domain(scene)
    if domain:
        print(f"Found liquid domain: {domain.name}")
    else:
        print("WARNING: No liquid domain found, will try to find mesh objects")

    # Process each frame
    converted = 0
    for frame in range(frame_start, frame_end + 1, step):
        print(f"\n[Frame {frame}]")

        # Set the frame
        scene.frame_set(frame)

        # Get the mesh at this frame
        if domain:
            # The domain IS the mesh (for liquid, domain displays the generated mesh)
            mesh_obj = domain
        else:
            # Try to find any mesh that might be the liquid
            mesh_obj = None
            for obj in scene.objects:
                if obj.type == 'MESH' and 'liquid' in obj.name.lower():
                    mesh_obj = obj
                    break

        if not mesh_obj:
            print(f"  No mesh object found for frame {frame}")
            continue

        # Check if mesh has any faces at this frame
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_mesh = mesh_obj.evaluated_get(depsgraph)

        if not eval_mesh.data or len(eval_mesh.data.polygons) == 0:
            print(f"  Frame {frame}: No mesh geometry (water hasn't appeared yet)")
            continue

        print(f"  Mesh: {mesh_obj.name} ({len(eval_mesh.data.polygons)} faces)")

        # Convert mesh to volume
        volume = mesh_to_volume(mesh_obj, resolution, frame)
        if not volume:
            print(f"  Failed to convert mesh to volume")
            continue

        # Export as VDB
        vdb_path = output_path / f"water_sdf_{frame:04d}.vdb"
        if export_volume_as_vdb(volume, vdb_path):
            print(f"  Saved: {vdb_path.name}")
            converted += 1
        else:
            print(f"  Could not export VDB")

        # Cleanup volume object
        bpy.data.objects.remove(volume, do_unlink=True)

    print(f"\n{'=' * 60}")
    print(f"Converted {converted} frames")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
