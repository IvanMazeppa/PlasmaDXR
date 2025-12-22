#!/usr/bin/env python3
"""
Blender CLI VDB to NanoVDB Converter

Run with Blender in headless mode to convert OpenVDB files to NanoVDB format.

Usage:
    blender --background --python scripts/convert_vdb_cli.py -- input.vdb output.nvdb
    blender --background --python scripts/convert_vdb_cli.py -- --batch input_dir/ output_dir/

Examples:
    # Single file conversion
    ~/apps/blender-5.0.1-linux-x64/blender --background --python scripts/convert_vdb_cli.py -- \
        assets/VDB_packs/CloudPackVDB/CloudPack/CloudPackVDB/cloud_01_variant_0000.vdb \
        assets/volumes/cloud_01.nvdb

    # Batch conversion
    ~/apps/blender-5.0.1-linux-x64/blender --background --python scripts/convert_vdb_cli.py -- \
        --batch assets/VDB_packs/CloudPackVDB/CloudPack/CloudPackVDB/ \
        assets/volumes/clouds/

Requirements:
    - Blender 5.0+ (has native NanoVDB export support)
"""

import bpy
import sys
import os
from pathlib import Path


def parse_args():
    """Parse command line arguments after Blender's '--' separator."""
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        argv = []

    import argparse
    parser = argparse.ArgumentParser(
        description='Convert OpenVDB to NanoVDB using Blender',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('input', nargs='?', help='Input .vdb file or directory')
    parser.add_argument('output', nargs='?', help='Output .nvdb file or directory')
    parser.add_argument('--batch', action='store_true', help='Batch convert directory')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress progress messages')

    return parser.parse_args(argv)


def log(msg, quiet=False):
    """Print message unless quiet mode."""
    if not quiet:
        print(f"[VDB→NanoVDB] {msg}")


def clear_volumes():
    """Remove all volume objects from scene."""
    for obj in list(bpy.data.objects):
        if obj.type == 'VOLUME':
            bpy.data.objects.remove(obj, do_unlink=True)

    # Also clean up orphan volume data
    for vol in list(bpy.data.volumes):
        bpy.data.volumes.remove(vol, do_unlink=True)


def convert_single_vdb(input_path: str, output_path: str, quiet: bool = False) -> bool:
    """
    Convert a single VDB file to NanoVDB format.

    Args:
        input_path: Path to input .vdb file
        output_path: Path to output .nvdb file
        quiet: Suppress progress messages

    Returns:
        bool: True if conversion succeeded
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        log(f"ERROR: Input file not found: {input_path}", quiet)
        return False

    log(f"Converting: {input_file.name}", quiet)

    try:
        # Clear existing volumes
        clear_volumes()

        # Import the VDB file
        bpy.ops.object.volume_import(filepath=str(input_file))

        # Get the imported volume
        vol_obj = bpy.context.active_object
        if not vol_obj or vol_obj.type != 'VOLUME':
            log(f"  ERROR: Failed to import as volume object", quiet)
            return False

        vol_data = vol_obj.data
        log(f"  Imported: {vol_obj.name}", quiet)

        # Print grid info
        if hasattr(vol_data, 'grids'):
            grids_info = [f"{g.name}:{g.data_type}" for g in vol_data.grids]
            log(f"  Grids: {', '.join(grids_info)}", quiet)

        # Select the volume for export
        bpy.ops.object.select_all(action='DESELECT')
        vol_obj.select_set(True)
        bpy.context.view_layer.objects.active = vol_obj

        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Export as NanoVDB
        # Blender 5.0 uses export_scene.openvdb with use_nanovdb option
        if hasattr(bpy.ops.export_scene, 'openvdb'):
            bpy.ops.export_scene.openvdb(
                filepath=str(output_file),
                use_nanovdb=True,  # Enable NanoVDB format
            )

            # Verify output
            if output_file.exists():
                size_mb = output_file.stat().st_size / (1024 * 1024)
                log(f"  Saved: {output_file.name} ({size_mb:.1f} MB)", quiet)
                return True
            else:
                log(f"  ERROR: Export failed - output file not created", quiet)
                return False
        else:
            log(f"  ERROR: export_scene.openvdb not available in this Blender version", quiet)
            return False

    except Exception as e:
        log(f"  ERROR: {e}", quiet)
        return False
    finally:
        # Clean up
        clear_volumes()


def batch_convert(input_dir: str, output_dir: str, quiet: bool = False):
    """
    Batch convert all VDB files in a directory.

    Args:
        input_dir: Directory containing .vdb files
        output_dir: Output directory for .nvdb files
        quiet: Suppress progress messages
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        log(f"ERROR: Input directory not found: {input_dir}")
        return

    # Find all VDB files
    vdb_files = sorted(input_path.glob("*.vdb"))

    if not vdb_files:
        log(f"No .vdb files found in {input_dir}")
        return

    log(f"Found {len(vdb_files)} VDB file(s)")
    log(f"Output directory: {output_path}")
    log("")

    output_path.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for vdb_file in vdb_files:
        # Create output filename
        nvdb_file = output_path / vdb_file.with_suffix('.nvdb').name

        if convert_single_vdb(str(vdb_file), str(nvdb_file), quiet):
            success_count += 1

    log("")
    log("=" * 50)
    log(f"Conversion complete: {success_count}/{len(vdb_files)} files")

    # Summary of output files
    if success_count > 0:
        log(f"\nOutput files in: {output_path}")
        total_size = 0
        for nvdb_file in sorted(output_path.glob("*.nvdb")):
            size_mb = nvdb_file.stat().st_size / (1024 * 1024)
            total_size += size_mb
            log(f"  {nvdb_file.name}: {size_mb:.1f} MB")
        log(f"\nTotal: {total_size:.1f} MB")


def main():
    args = parse_args()

    if not args.input:
        print(__doc__)
        return

    log("=" * 50)
    log("Blender VDB to NanoVDB Converter")
    log(f"Blender version: {bpy.app.version_string}")
    log("=" * 50)
    log("")

    if args.batch or os.path.isdir(args.input):
        if not args.output:
            args.output = str(Path(args.input) / "nvdb")
        batch_convert(args.input, args.output, args.quiet)
    else:
        if not args.output:
            args.output = str(Path(args.input).with_suffix('.nvdb'))

        success = convert_single_vdb(args.input, args.output, args.quiet)
        if success:
            log("\nConversion successful!")
        else:
            log("\nConversion failed!")
            sys.exit(1)


if __name__ == "__main__":
    main()
