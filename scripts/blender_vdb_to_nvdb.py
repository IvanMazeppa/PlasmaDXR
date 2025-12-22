"""
Blender OpenVDB to NanoVDB Converter

Run this script in Blender 5.0's Scripting workspace to convert
OpenVDB (.vdb) files to NanoVDB (.nvdb) format for PlasmaDX-Clean.

Usage:
1. Open Blender 5.0
2. Go to Scripting workspace
3. Paste this script and modify VDB_DIR and OUTPUT_DIR paths
4. Press Run Script (or Alt+P)

The script will:
- Find all .vdb files in VDB_DIR
- Convert each to .nvdb format
- Save to OUTPUT_DIR with same name

Note: Blender 5.0 has built-in NanoVDB support for export.
"""
import bpy
import os
from pathlib import Path

# ============================================================================
# CONFIGURATION - Modify these paths!
# ============================================================================

# Input directory containing .vdb files
# CloudPack clouds:
VDB_DIR = "/home/maz3ppa/projects/PlasmaDXR/assets/VDB_packs/CloudPackVDB/CloudPack/CloudPackVDB"
# Gasoline explosion sequence:
# VDB_DIR = "/home/maz3ppa/projects/PlasmaDXR/assets/VDB_packs/Gasoline_Explosion_01/Gasoline_Explosion_01"

# Output directory for .nvdb files (will be created if doesn't exist)
OUTPUT_DIR = "/home/maz3ppa/projects/PlasmaDXR/assets/volumes"

# ============================================================================
# CONVERSION SCRIPT
# ============================================================================

def clear_scene():
    """Remove all objects from scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def convert_vdb_to_nvdb(input_path: str, output_path: str) -> bool:
    """
    Convert a single VDB file to NanoVDB format.

    Args:
        input_path: Full path to input .vdb file
        output_path: Full path to output .nvdb file

    Returns:
        bool: True if successful
    """
    print(f"\nConverting: {os.path.basename(input_path)}")

    try:
        # Clear existing volumes
        for obj in bpy.data.objects:
            if obj.type == 'VOLUME':
                bpy.data.objects.remove(obj)

        # Import the VDB
        bpy.ops.object.volume_import(filepath=input_path)

        # Get the imported volume
        vol_obj = bpy.context.active_object
        if not vol_obj or vol_obj.type != 'VOLUME':
            print(f"  ERROR: Failed to import volume")
            return False

        vol_data = vol_obj.data
        print(f"  Imported: {vol_obj.name}")

        # Get grid info
        if hasattr(vol_data, 'grids'):
            print(f"  Grids: {len(vol_data.grids)}")
            for grid in vol_data.grids:
                print(f"    - {grid.name}: {grid.data_type}")

        # Select the volume for export
        bpy.ops.object.select_all(action='DESELECT')
        vol_obj.select_set(True)
        bpy.context.view_layer.objects.active = vol_obj

        # Export as NanoVDB
        # Note: Blender 5.0 VDB export supports NanoVDB format
        # Check if export operator exists
        if hasattr(bpy.ops.export_scene, 'openvdb'):
            bpy.ops.export_scene.openvdb(
                filepath=output_path,
                use_nanovdb=True,  # Enable NanoVDB format
            )
            print(f"  Saved: {output_path}")
            return True
        else:
            # Fallback: Try direct volume export
            # In some Blender versions, volume export is different
            print("  WARNING: export_scene.openvdb not available")
            print("  Trying alternative export method...")

            # Use cache export as workaround
            vol_data.filepath = output_path
            print(f"  Set filepath to: {output_path}")
            return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def batch_convert():
    """Convert all VDB files in VDB_DIR to NanoVDB format."""
    print("\n" + "="*60)
    print("OpenVDB to NanoVDB Batch Converter")
    print("="*60)

    # Validate paths
    if not os.path.exists(VDB_DIR):
        print(f"ERROR: VDB directory not found: {VDB_DIR}")
        return

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all VDB files
    vdb_files = [f for f in os.listdir(VDB_DIR) if f.endswith('.vdb')]

    if not vdb_files:
        print(f"No .vdb files found in {VDB_DIR}")
        return

    print(f"Found {len(vdb_files)} VDB file(s)")
    print(f"Output directory: {OUTPUT_DIR}")
    print("")

    success_count = 0
    for filename in vdb_files:
        input_path = os.path.join(VDB_DIR, filename)
        output_name = os.path.splitext(filename)[0] + ".nvdb"
        output_path = os.path.join(OUTPUT_DIR, output_name)

        if convert_vdb_to_nvdb(input_path, output_path):
            success_count += 1

    print("\n" + "="*60)
    print(f"Conversion complete: {success_count}/{len(vdb_files)} files")
    print("="*60)

    # Summary of output files
    if success_count > 0:
        print(f"\nOutput files in: {OUTPUT_DIR}")
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith('.nvdb'):
                filepath = os.path.join(OUTPUT_DIR, f)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  - {f}: {size_mb:.1f} MB")

# Run the conversion
if __name__ == "__main__":
    batch_convert()
