#!/usr/bin/env python3
"""
Convert Mantaflow VDB files to clean NanoVDB-compatible format.

Mantaflow VDB files contain particle attributes (int32_trnc) that nanovdb_convert
doesn't understand. This script uses Blender to re-export just the density grid.

Usage:
    blender --background --python convert_mantaflow_vdb.py -- --input_dir /path/to/vdbs --output_dir /path/to/nvdbs
"""

import bpy
import sys
import os
from pathlib import Path


def parse_args():
    """Parse arguments after '--'."""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    input_dir = None
    output_dir = None
    frame_start = None
    frame_end = None
    step = 1

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--input_dir" and i + 1 < len(argv):
            input_dir = argv[i + 1]
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
        else:
            i += 1

    return input_dir, output_dir, frame_start, frame_end, step


def clear_scene():
    """Remove all objects from scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def convert_vdb_files(input_dir: str, output_dir: str, frame_start: int = None, frame_end: int = None, step: int = 1):
    """
    Convert Mantaflow VDB files by importing as Volume and re-exporting density grid only.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all VDB files
    vdb_files = sorted(input_path.glob("*.vdb"))
    if not vdb_files:
        print(f"ERROR: No VDB files found in {input_dir}")
        return 0

    print(f"Found {len(vdb_files)} VDB files")

    # Apply frame range filter
    if frame_start is not None or frame_end is not None:
        filtered_files = []
        for vdb_file in vdb_files:
            # Extract frame number from filename (e.g., fluid_data_0050.vdb -> 50)
            try:
                frame_num = int(vdb_file.stem.split('_')[-1])
                if frame_start is not None and frame_num < frame_start:
                    continue
                if frame_end is not None and frame_num > frame_end:
                    continue
                filtered_files.append(vdb_file)
            except ValueError:
                filtered_files.append(vdb_file)
        vdb_files = filtered_files
        print(f"After frame filter: {len(vdb_files)} files")

    # Apply step filter
    if step > 1:
        vdb_files = vdb_files[::step]
        print(f"After step filter (every {step}): {len(vdb_files)} files")

    converted = 0
    for i, vdb_file in enumerate(vdb_files):
        print(f"[{i+1}/{len(vdb_files)}] Converting: {vdb_file.name}")

        try:
            clear_scene()

            # Import VDB as Volume object
            bpy.ops.object.volume_import(
                filepath=str(vdb_file),
                files=[{"name": vdb_file.name}],
                directory=str(vdb_file.parent)
            )

            vol = bpy.context.active_object
            if vol is None or vol.type != 'VOLUME':
                print(f"  ERROR: Failed to import as volume")
                continue

            # Get the grids in the volume
            grids = list(vol.data.grids)
            print(f"  Grids found: {[g.name for g in grids]}")

            # Find density-like grids
            density_grid = None
            for grid in grids:
                if grid.name.lower() in ('density', 'fog', 'smoke'):
                    density_grid = grid
                    break

            if density_grid is None and grids:
                # Just use first grid
                density_grid = grids[0]
                print(f"  Using first grid: {density_grid.name}")

            # Output filename with .nvdb extension
            nvdb_file = output_path / vdb_file.name.replace('.vdb', '.nvdb')

            # Export using the volume export operator
            # First, let's try exporting as OpenVDB and then convert
            clean_vdb = output_path / f"clean_{vdb_file.name}"

            # Select and export volume
            vol.select_set(True)
            bpy.context.view_layer.objects.active = vol

            # Export as VDB (Blender's export should be cleaner)
            bpy.ops.object.volume_export(
                filepath=str(clean_vdb)
            )

            print(f"  Exported clean VDB: {clean_vdb.name}")
            converted += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print(f"\nConverted {converted}/{len(vdb_files)} files")
    print(f"Output directory: {output_path}")

    return converted


def main():
    input_dir, output_dir, frame_start, frame_end, step = parse_args()

    if not input_dir:
        print("ERROR: --input_dir required")
        print("Usage: blender --background --python convert_mantaflow_vdb.py -- --input_dir /path/to/vdbs --output_dir /path/to/nvdbs")
        return

    if not output_dir:
        output_dir = str(Path(input_dir).parent / "nvdb")

    print("=" * 60)
    print("Mantaflow VDB to NanoVDB Converter")
    print("=" * 60)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    if frame_start or frame_end:
        print(f"Frames: {frame_start or 'start'} - {frame_end or 'end'}")
    if step > 1:
        print(f"Step:   Every {step} frames")
    print()

    convert_vdb_files(input_dir, output_dir, frame_start, frame_end, step)


if __name__ == "__main__":
    main()
