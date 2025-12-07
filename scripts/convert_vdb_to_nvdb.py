"""
OpenVDB to NanoVDB Converter for PlasmaDX-Clean

This script converts OpenVDB (.vdb) files to NanoVDB (.nvdb) format
for use with PlasmaDX-Clean's volumetric rendering system.

Requirements:
    pip install pyopenvdb  (or conda install -c conda-forge openvdb)

Usage:
    python convert_vdb_to_nvdb.py input.vdb output.nvdb
    python convert_vdb_to_nvdb.py --batch VDBs/Clouds/  (converts all .vdb files)

Note: pyopenvdb includes NanoVDB conversion functionality.
      If pyopenvdb is not available, use Blender's VDB export with NanoVDB format.
"""

import sys
import os
import argparse
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are available."""
    try:
        import pyopenvdb as vdb
        return True, vdb
    except ImportError:
        return False, None

def convert_single_file(input_path: str, output_path: str, verbose: bool = True):
    """
    Convert a single OpenVDB file to NanoVDB format.

    Args:
        input_path: Path to input .vdb file
        output_path: Path to output .nvdb file
        verbose: Print progress messages

    Returns:
        bool: True if conversion succeeded
    """
    success, vdb = check_dependencies()
    if not success:
        print("ERROR: pyopenvdb not installed!")
        print("Install with: pip install pyopenvdb")
        print("Or use conda: conda install -c conda-forge openvdb")
        print("")
        print("Alternative: Use Blender to export VDB with NanoVDB format")
        return False

    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        return False

    try:
        if verbose:
            print(f"Reading: {input_path}")

        # Read the OpenVDB file
        grids = vdb.readAll(input_path)

        if not grids:
            print(f"ERROR: No grids found in {input_path}")
            return False

        if verbose:
            print(f"  Found {len(grids)} grid(s):")
            for grid in grids:
                print(f"    - {grid.name}: {type(grid).__name__}")

        # Check if pyopenvdb supports NanoVDB conversion
        # This depends on the pyopenvdb version and build
        if hasattr(vdb, 'nanovdb'):
            if verbose:
                print(f"Converting to NanoVDB format...")

            # Convert to NanoVDB and save
            for grid in grids:
                nano_grid = vdb.nanovdb.createNanoGrid(grid)
                # Save NanoVDB grid
                vdb.nanovdb.write(output_path, nano_grid)

            if verbose:
                print(f"Saved: {output_path}")
            return True
        else:
            print("ERROR: This version of pyopenvdb doesn't support NanoVDB conversion")
            print("")
            print("Alternative methods:")
            print("1. Use Blender 5.0:")
            print("   - Import VDB as Volume object")
            print("   - Export with File > Export > OpenVDB")
            print("   - Enable 'Use NanoVDB' option")
            print("")
            print("2. Build nanovdb_convert tool from external/nanovdb")
            print("")
            print("3. Use NVIDIA's nanovdb_convert standalone tool")
            return False

    except Exception as e:
        print(f"ERROR: Conversion failed: {e}")
        return False

def batch_convert(input_dir: str, output_dir: str = None, verbose: bool = True):
    """
    Convert all .vdb files in a directory to .nvdb format.

    Args:
        input_dir: Directory containing .vdb files
        output_dir: Output directory (default: same as input)
        verbose: Print progress messages
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"ERROR: Directory not found: {input_dir}")
        return

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = input_path

    vdb_files = list(input_path.glob("**/*.vdb"))
    if not vdb_files:
        print(f"No .vdb files found in {input_dir}")
        return

    print(f"Found {len(vdb_files)} VDB file(s)")
    print("")

    success_count = 0
    for vdb_file in vdb_files:
        # Create output path with .nvdb extension
        relative_path = vdb_file.relative_to(input_path)
        nvdb_file = output_path / relative_path.with_suffix('.nvdb')
        nvdb_file.parent.mkdir(parents=True, exist_ok=True)

        if convert_single_file(str(vdb_file), str(nvdb_file), verbose):
            success_count += 1

    print("")
    print(f"Converted {success_count}/{len(vdb_files)} file(s)")

def print_alternative_methods():
    """Print alternative conversion methods when pyopenvdb is not available."""
    print("""
=========================================
OpenVDB to NanoVDB Conversion Options
=========================================

Option 1: Blender 5.0 (Recommended)
-----------------------------------
1. Open Blender 5.0
2. Import your VDB file:
   - File > Import > Volume
   - Select your .vdb file
3. Export as NanoVDB:
   - File > Export > OpenVDB
   - Check "Use NanoVDB" checkbox
   - Save as .nvdb

Option 2: Build nanovdb_convert
-------------------------------
1. Navigate to external/nanovdb directory
2. Build with CMake:
   mkdir build && cd build
   cmake .. -DNANOVDB_BUILD_TOOLS=ON
   cmake --build . --target nanovdb_convert
3. Run:
   nanovdb_convert input.vdb output.nvdb

Option 3: NVIDIA nanovdb_convert
--------------------------------
Download from NVIDIA or build from OpenVDB source:
https://github.com/AcademySoftwareFoundation/openvdb/tree/master/nanovdb

Option 4: Python with OpenVDB
-----------------------------
Install: pip install pyopenvdb
(Requires OpenVDB library installed on system)

=========================================
""")

def main():
    parser = argparse.ArgumentParser(
        description='Convert OpenVDB (.vdb) to NanoVDB (.nvdb) format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.vdb output.nvdb          # Convert single file
  %(prog)s --batch VDBs/Clouds/           # Convert all .vdb in directory
  %(prog)s --info                         # Show conversion options
        """
    )

    parser.add_argument('input', nargs='?', help='Input .vdb file or directory')
    parser.add_argument('output', nargs='?', help='Output .nvdb file or directory')
    parser.add_argument('--batch', action='store_true', help='Batch convert directory')
    parser.add_argument('--info', action='store_true', help='Show conversion options')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress messages')

    args = parser.parse_args()

    if args.info:
        print_alternative_methods()
        return

    if not args.input:
        parser.print_help()
        print("")
        print_alternative_methods()
        return

    verbose = not args.quiet

    # Check for pyopenvdb
    success, _ = check_dependencies()
    if not success:
        print("pyopenvdb not available.")
        print_alternative_methods()
        return

    if args.batch or os.path.isdir(args.input):
        batch_convert(args.input, args.output, verbose)
    else:
        if not args.output:
            # Default output: same name with .nvdb extension
            args.output = Path(args.input).with_suffix('.nvdb')

        convert_single_file(args.input, str(args.output), verbose)

if __name__ == "__main__":
    main()
