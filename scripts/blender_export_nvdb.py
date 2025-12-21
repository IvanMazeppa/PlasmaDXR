#!/usr/bin/env python3
"""
Blender CLI NanoVDB Export Script

Exports fluid simulations from .blend files to NanoVDB format suitable for PlasmaDX.
Runs in Blender's headless mode for batch processing and automation.

Usage:
    blender --background --python scripts/blender_export_nvdb.py -- [OPTIONS]

Examples:
    # Export single frame at resolution 300
    blender --background scene.blend --python scripts/blender_export_nvdb.py -- \\
        --output ~/volumes/my_smoke \\
        --resolution 300

    # Export frame range 1-100
    blender --background scene.blend --python scripts/blender_export_nvdb.py -- \\
        --output ~/volumes/explosion \\
        --frames 1-100 \\
        --resolution 256

    # Just increase resolution and re-bake (no export)
    blender --background scene.blend --python scripts/blender_export_nvdb.py -- \\
        --resolution 384 \\
        --bake-only

Requirements:
    - Blender 4.0+ (5.0 recommended for best NanoVDB support)
    - Scene must contain a Fluid Domain object

Output:
    - Creates .nvdb files in output directory
    - Naming: volume_XXXX.nvdb (frame padded to 4 digits)
    - Grid: 'density' (FLOAT type, compatible with PlasmaDX shader)

Notes:
    - Always uses FULL precision to ensure FLOAT grids (not HALF/FP16)
    - Disables adaptive domain for consistent bounds
    - See docs/NanoVDB/BLENDER_HIGH_RES_EXPORT_GUIDE.md for settings guide
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
        description='Export Blender fluid simulation to NanoVDB format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  blender --background scene.blend --python %(prog)s -- --output ./volumes
  blender --background scene.blend --python %(prog)s -- --frames 1-50 --resolution 300
        """
    )

    parser.add_argument('--output', '-o', default='./nvdb_export',
                        help='Output directory for .nvdb files (default: ./nvdb_export)')
    parser.add_argument('--frames', '-f', default='current',
                        help='Frame range: "current", single number, or "start-end" (default: current)')
    parser.add_argument('--resolution', '-r', type=int, default=None,
                        help='Domain resolution (overrides scene setting, 256-512 recommended)')
    parser.add_argument('--grid', '-g', default='density',
                        help='Grid name to export (default: density)')
    parser.add_argument('--bake-only', action='store_true',
                        help='Only update settings and bake, do not export')
    parser.add_argument('--no-bake', action='store_true',
                        help='Skip baking, export existing cache only')
    parser.add_argument('--domain', '-d', default=None,
                        help='Name of domain object (auto-detected if not specified)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress messages')

    return parser.parse_args(argv)


def log(msg, quiet=False):
    """Print message unless quiet mode."""
    if not quiet:
        print(f"[NanoVDB Export] {msg}")


def find_fluid_domain(domain_name=None):
    """Find the fluid domain object in the scene."""
    for obj in bpy.data.objects:
        if domain_name and obj.name != domain_name:
            continue

        if obj.type != 'MESH':
            continue

        for mod in obj.modifiers:
            if mod.type == 'FLUID' and mod.fluid_type == 'DOMAIN':
                return obj, mod

    return None, None


def configure_domain_for_export(domain, mod, resolution=None, quiet=False):
    """Configure domain settings for high-quality NanoVDB export."""
    settings = mod.domain_settings

    # Store original values for reporting
    original_res = settings.resolution_max

    # Set resolution if specified
    if resolution:
        settings.resolution_max = resolution
        log(f"Resolution: {original_res} -> {resolution}", quiet)

    # Critical: Use OpenVDB format with FULL precision
    # This ensures FLOAT grids that PlasmaDX shader can read
    settings.cache_data_format = 'OPENVDB'

    # Disable adaptive domain for consistent bounds
    if settings.use_adaptive_domain:
        settings.use_adaptive_domain = False
        log("Disabled adaptive domain (required for consistent bounds)", quiet)

    # Cache type should be Modular for separate grid export
    if hasattr(settings, 'cache_type'):
        settings.cache_type = 'MODULAR'

    # Note: Precision setting may vary by Blender version
    # In Blender 4.0+, check domain_settings.openvdb_data_depth
    if hasattr(settings, 'openvdb_data_depth'):
        settings.openvdb_data_depth = '32'  # Full precision
        log("Set OpenVDB precision to FULL (32-bit)", quiet)

    return settings


def bake_simulation(domain, settings, quiet=False):
    """Bake the fluid simulation."""
    log("Baking simulation (this may take a while)...", quiet)

    # Free existing cache
    bpy.ops.fluid.free_all()

    # Bake data
    bpy.ops.fluid.bake_data()

    log("Bake complete!", quiet)


def export_frames(domain, settings, output_dir, frames, grid_name, quiet=False):
    """Export specified frames to NanoVDB format."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Parse frame range
    if frames == 'current':
        frame_list = [bpy.context.scene.frame_current]
    elif '-' in frames:
        start, end = map(int, frames.split('-'))
        frame_list = list(range(start, end + 1))
    else:
        frame_list = [int(frames)]

    log(f"Exporting {len(frame_list)} frame(s) to {out_path}", quiet)

    exported = 0
    for frame in frame_list:
        bpy.context.scene.frame_set(frame)

        output_file = out_path / f"volume_{frame:04d}.nvdb"

        try:
            # Export using Blender's OpenVDB exporter with NanoVDB option
            if hasattr(bpy.ops.export_scene, 'openvdb'):
                bpy.ops.export_scene.openvdb(
                    filepath=str(output_file),
                    use_nanovdb=True,  # Enable NanoVDB format
                )
                log(f"  Frame {frame}: {output_file.name}", quiet)
                exported += 1
            else:
                log(f"  Frame {frame}: ERROR - OpenVDB export not available", quiet)

        except Exception as e:
            log(f"  Frame {frame}: ERROR - {e}", quiet)

    return exported, len(frame_list)


def main():
    args = parse_args()

    log("=" * 60, args.quiet)
    log("Blender NanoVDB Export for PlasmaDX", args.quiet)
    log("=" * 60, args.quiet)

    # Find fluid domain
    domain, mod = find_fluid_domain(args.domain)
    if not domain:
        log("ERROR: No fluid domain found in scene!")
        log("Make sure your scene contains a Fluid Domain object.")
        sys.exit(1)

    log(f"Found domain: {domain.name}", args.quiet)

    # Configure domain settings
    settings = configure_domain_for_export(
        domain, mod,
        resolution=args.resolution,
        quiet=args.quiet
    )

    # Bake if needed
    if not args.no_bake:
        bake_simulation(domain, settings, args.quiet)

    if args.bake_only:
        log("Bake-only mode, skipping export.", args.quiet)
        return

    # Export frames
    exported, total = export_frames(
        domain, settings,
        args.output, args.frames, args.grid,
        args.quiet
    )

    log("=" * 60, args.quiet)
    log(f"Export complete: {exported}/{total} frames", args.quiet)
    log(f"Output directory: {args.output}", args.quiet)
    log("=" * 60, args.quiet)

    if exported > 0:
        log("", args.quiet)
        log("Next steps:", args.quiet)
        log("1. Copy .nvdb files to PlasmaDX assets/volumes/", args.quiet)
        log("2. In PlasmaDX ImGui: NanoVDB > Load Asset", args.quiet)
        log("3. Adjust scale (typically 100-200x for Blender units)", args.quiet)


if __name__ == "__main__":
    main()
