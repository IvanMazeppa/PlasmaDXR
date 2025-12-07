"""
VDB Inspector Script for Blender
Run this in Blender's Scripting workspace to inspect cloud VDB files.

Usage:
1. Open Blender 5.0
2. Go to Scripting workspace
3. Paste this script
4. Press Run Script (or Alt+P)
"""
import bpy
import os

# Path to VDB files (update this path!)
VDB_DIR = r"D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\VDBs\Clouds\CloudPackVDB"

def inspect_vdb(filepath):
    """Load a VDB file and print its properties."""
    print(f"\n{'='*60}")
    print(f"Inspecting: {os.path.basename(filepath)}")
    print(f"{'='*60}")

    # Clear existing volume objects
    for obj in bpy.data.objects:
        if obj.type == 'VOLUME':
            bpy.data.objects.remove(obj)

    # Import the VDB
    try:
        bpy.ops.object.volume_import(filepath=filepath)

        # Get the imported volume
        vol_obj = bpy.context.active_object
        if vol_obj and vol_obj.type == 'VOLUME':
            vol_data = vol_obj.data

            print(f"Object Name: {vol_obj.name}")
            print(f"Volume Data: {vol_data.name}")
            print(f"Location: {vol_obj.location}")
            print(f"Scale: {vol_obj.scale}")

            # Get grid info if available
            if hasattr(vol_data, 'grids'):
                print(f"\nGrids ({len(vol_data.grids)}):")
                for grid in vol_data.grids:
                    print(f"  - {grid.name}: {grid.data_type}")
            else:
                print("\nGrid info not accessible via Python API")

            # Get bounding box
            if vol_obj.bound_box:
                bb = vol_obj.bound_box
                min_co = [min(v[i] for v in bb) for i in range(3)]
                max_co = [max(v[i] for v in bb) for i in range(3)]
                print(f"\nBounding Box:")
                print(f"  Min: ({min_co[0]:.2f}, {min_co[1]:.2f}, {min_co[2]:.2f})")
                print(f"  Max: ({max_co[0]:.2f}, {max_co[1]:.2f}, {max_co[2]:.2f})")
                size = [max_co[i] - min_co[i] for i in range(3)]
                print(f"  Size: ({size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f})")

            return True
        else:
            print("ERROR: No volume object created")
            return False

    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("VDB Inspector for PlasmaDX-Clean")
    print("="*60)

    if not os.path.exists(VDB_DIR):
        print(f"ERROR: VDB directory not found: {VDB_DIR}")
        return

    vdb_files = [f for f in os.listdir(VDB_DIR) if f.endswith('.vdb')]
    print(f"Found {len(vdb_files)} VDB files\n")

    # Inspect first file as sample
    if vdb_files:
        first_vdb = os.path.join(VDB_DIR, vdb_files[0])
        inspect_vdb(first_vdb)

        print(f"\n\nAll VDB files in directory:")
        for f in vdb_files:
            filepath = os.path.join(VDB_DIR, f)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  - {f}: {size_mb:.1f} MB")

if __name__ == "__main__":
    main()
