"""
OpenVDB (.vdb) -> NanoVDB (.nvdb) converter for PlasmaDX-Clean.

Why this exists:
- PlasmaDX-Clean's `nanovdb_raymarch.hlsl` currently only supports **FLOAT** grids.
- Blender/Mantaflow VDBs often contain multiple grids (e.g. density=float, velocity=vec3).
- If the wrong grid gets converted, the volume may render as "invisible" in-game.

This script therefore defaults to converting a **single** grid, preferring:
1) A grid explicitly selected via `--grid`
2) A grid named "density" (common Mantaflow convention)
3) The first float grid found

Requirements:
    pip install pyopenvdb  (or conda install -c conda-forge openvdb)

Usage:
    python scripts/convert_vdb_to_nvdb.py input.vdb output.nvdb
    python scripts/convert_vdb_to_nvdb.py input.vdb --grid density
    python scripts/convert_vdb_to_nvdb.py --list input.vdb
    python scripts/convert_vdb_to_nvdb.py --all-grids input.vdb out_dir/
    python scripts/convert_vdb_to_nvdb.py --batch VDBs/Clouds/  (converts all .vdb files)
"""

import sys
import os
import argparse
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are available."""
    # First try nanovdb_convert CLI tool (preferred)
    import shutil
    if shutil.which('nanovdb_convert'):
        return True, 'nanovdb_convert'

    # Fallback to pyopenvdb
    try:
        import pyopenvdb as vdb
        return True, vdb
    except ImportError:
        return False, None


def _grid_name(grid) -> str:
    return getattr(grid, "name", "") or getattr(grid, "gridName", "") or ""


def _grid_type_name(grid) -> str:
    # pyopenvdb types vary; fall back to Python class name.
    try:
        if hasattr(grid, "valueTypeName"):
            return str(grid.valueTypeName())
    except Exception:
        pass
    return type(grid).__name__


def _is_probably_float_grid(grid) -> bool:
    t = _grid_type_name(grid).lower()
    # Heuristic: FloatGrid / float / float32 / float16
    return ("float" in t) and ("vec" not in t) and ("vector" not in t)


def _select_grid(grids, requested_name: str | None):
    if not grids:
        return None

    # Build lookup
    by_name = {}
    for g in grids:
        n = _grid_name(g)
        if n:
            by_name[n] = g

    if requested_name:
        if requested_name in by_name:
            return by_name[requested_name]
        # Try case-insensitive match
        for k, v in by_name.items():
            if k.lower() == requested_name.lower():
                return v
        return None

    # Prefer density
    for key in ("density", "Density", "fog", "Fog", "smoke", "Smoke"):
        if key in by_name and _is_probably_float_grid(by_name[key]):
            return by_name[key]

    # Otherwise first float-ish grid
    for g in grids:
        if _is_probably_float_grid(g):
            return g

    # Fallback: first grid (caller will likely fail in PlasmaDX)
    return grids[0]


def _print_grid_summary(grids) -> None:
    print(f"  Found {len(grids)} grid(s):")
    for grid in grids:
        print(f"    - {_grid_name(grid) or '<unnamed>'}: {_grid_type_name(grid)}")


def _convert_with_cli(input_path: str, output_path: str, verbose: bool = True, grid_name: str | None = None) -> bool:
    """
    Convert VDB to NanoVDB using the nanovdb_convert CLI tool.

    Args:
        input_path: Path to input .vdb file
        output_path: Path to output .nvdb file
        verbose: Print progress messages
        grid_name: Optional grid name to convert (default: density)

    Returns:
        bool: True if conversion succeeded
    """
    import subprocess

    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        return False

    # Ensure output directory exists
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = ['nanovdb_convert', '-f']  # -f to overwrite existing
    if grid_name:
        cmd.extend(['-g', grid_name])
    if verbose:
        cmd.append('-v')
    cmd.extend([input_path, output_path])

    if verbose:
        print(f"Converting: {os.path.basename(input_path)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr}")
            return False

        if verbose and result.stdout:
            # Parse output for grid info
            for line in result.stdout.strip().split('\n'):
                if 'grid named' in line.lower() or 'allocated' in line.lower():
                    print(f"  {line}")

        if out_path.exists():
            if verbose:
                size_mb = out_path.stat().st_size / (1024 * 1024)
                print(f"  Saved: {out_path.name} ({size_mb:.1f} MB)")
            return True
        else:
            print(f"  ERROR: Output file not created")
            return False

    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def convert_single_file(
    input_path: str,
    output_path: str,
    verbose: bool = True,
    grid_name: str | None = None,
    all_grids: bool = False,
):
    """
    Convert a single OpenVDB file to NanoVDB format.

    Args:
        input_path: Path to input .vdb file
        output_path: Path to output .nvdb file
        verbose: Print progress messages

    Returns:
        bool: True if conversion succeeded
    """
    success, backend = check_dependencies()
    if not success:
        print("ERROR: No conversion backend available!")
        print("Install nanovdb_convert: sudo apt install libnanovdb-dev")
        print("Or install pyopenvdb: pip install pyopenvdb")
        print("")
        print("Alternative: Use Blender to export VDB with NanoVDB format")
        return False

    # Use nanovdb_convert CLI if available (faster and simpler)
    if backend == 'nanovdb_convert':
        return _convert_with_cli(input_path, output_path, verbose, grid_name)

    # Fallback to pyopenvdb
    vdb = backend

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
            _print_grid_summary(grids)

        # Check if pyopenvdb supports NanoVDB conversion
        # This depends on the pyopenvdb version and build
        if hasattr(vdb, 'nanovdb'):
            if verbose:
                print(f"Converting to NanoVDB format...")

            out_path = Path(output_path)

            if all_grids:
                # Write one .nvdb per grid (prevents accidental overwrites and preserves names)
                out_dir = out_path if out_path.suffix == "" else out_path.parent
                out_dir.mkdir(parents=True, exist_ok=True)

                wrote = 0
                for grid in grids:
                    name = _grid_name(grid) or "grid"
                    per_file = out_dir / f"{Path(input_path).stem}_{name}.nvdb"
                    nano_grid = vdb.nanovdb.createNanoGrid(grid)
                    vdb.nanovdb.write(str(per_file), nano_grid)
                    wrote += 1
                    if verbose:
                        print(f"  Saved: {per_file}")

                return wrote > 0

            # Default: select ONE grid to avoid output overwrite ambiguity
            chosen = _select_grid(grids, grid_name)
            if chosen is None:
                print("ERROR: Failed to select a grid to convert")
                return False

            chosen_name = _grid_name(chosen) or "<unnamed>"
            chosen_type = _grid_type_name(chosen)
            if verbose:
                print(f"  Selected grid: {chosen_name} ({chosen_type})")
                if not _is_probably_float_grid(chosen):
                    print("  WARNING: Selected grid does not look like a float grid.")
                    print("           PlasmaDX shader may render it as 'invisible'. Prefer --grid density.")

            nano_grid = vdb.nanovdb.createNanoGrid(chosen)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            vdb.nanovdb.write(str(out_path), nano_grid)

            if verbose:
                print(f"Saved: {out_path}")
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
    parser.add_argument('--grid', default=None, help='Grid name to convert (e.g. density)')
    parser.add_argument('--all-grids', action='store_true', help='Write one .nvdb per grid to output directory')
    parser.add_argument('--list', action='store_true', help='List grids in a .vdb (no conversion)')
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

    if args.list:
        success, vdb = check_dependencies()
        if not success:
            print("ERROR: pyopenvdb not installed!")
            return
        grids = vdb.readAll(args.input)
        if not grids:
            print("No grids found.")
            return
        _print_grid_summary(grids)
        return

    if args.batch or os.path.isdir(args.input):
        batch_convert(args.input, args.output, verbose)
    else:
        if not args.output:
            # Default output: same name with .nvdb extension
            args.output = Path(args.input).with_suffix('.nvdb')

        convert_single_file(
            args.input,
            str(args.output),
            verbose,
            grid_name=args.grid,
            all_grids=args.all_grids,
        )

if __name__ == "__main__":
    main()
