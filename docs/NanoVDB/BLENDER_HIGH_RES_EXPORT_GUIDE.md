# Blender 5 High-Resolution NanoVDB Export Guide

## Why Your VDBs Look Blocky

**The Problem**: Blender's Mantaflow fluid domain defaults to 64-128 voxels per axis. Professional VDB assets (like cloud_pack) use 256-512+ voxels, resulting in 10-100x more detail.

**File Size Reference** (from working assets):
| Asset | Size | Resolution | Quality |
|-------|------|------------|---------|
| cloud_pack | 32 MB | ~300-400 voxels | Smooth, detailed |
| chimney_smoke | 20 MB | ~250 voxels | Good |
| bipolar_nebula | 3.4 MB | ~100 voxels | Blocky |

**Rule of Thumb**: Target 15-50 MB for hero volumes, 5-15 MB for background elements.

---

## Quick Fix: Increase Domain Resolution

### In Blender UI (Physics Properties > Fluid > Settings):

```
Domain Type: Gas
Resolution Divisions: 256 (minimum for smooth results)
                      384 (recommended for hero assets)
                      512 (maximum quality, slow simulation)
```

### Critical Cache Settings (Physics > Fluid > Cache):

```
Type: OpenVDB
Precision: Full (32-bit)  <-- CRITICAL: Half/Mini causes invisible volumes!
Compression: None or ZIP
```

**Warning**: "Half" or "Mini" precision creates FP16 grids that PlasmaDX shader may not render correctly.

---

## Step-by-Step: Creating High-Quality Smoke/Cloud

### 1. Domain Setup
```python
# Blender Python - set via Scripting workspace
import bpy

domain = bpy.context.object
settings = domain.modifiers["Fluid"].domain_settings

# Resolution: Higher = smoother but slower simulation
settings.resolution_max = 300  # Minimum for smooth results

# Cache settings - CRITICAL for PlasmaDX compatibility
settings.cache_type = 'MODULAR'  # Separate data types
settings.cache_data_format = 'OPENVDB'
settings.openvdb_cache_compress_type = 'NONE'  # Best quality

# Use FULL precision to ensure FLOAT grids (not HALF/FP16)
# Note: This setting may vary by Blender version
```

### 2. Simulation Quality
```python
# Higher CFL improves temporal stability
settings.cfl_condition = 4.0  # Default is 4, increase for fast motion

# Adaptive domain saves VRAM but can cause edge artifacts
settings.use_adaptive_domain = False  # Disable for consistent bounds

# Noise adds detail without increasing resolution cost as much
settings.noise_scale = 2  # Multiply effective resolution
settings.noise_strength = 1.0
```

### 3. Bake and Export
```
1. Bake Data (takes time with high resolution)
2. File > Export > OpenVDB
3. Options:
   - Select frame range
   - Grid: density (most important)
   - Use NanoVDB: ENABLED  <-- Creates .nvdb directly
```

---

## CLI Batch Export (Headless Blender)

Save this as `scripts/blender_export_nvdb.py`:

```python
#!/usr/bin/env python3
"""
Blender CLI NanoVDB Export Script

Usage:
    blender --background --python scripts/blender_export_nvdb.py -- \
        --input scene.blend \
        --output /path/to/output \
        --frames 1-100 \
        --resolution 300

From WSL with Linux Blender:
    /path/to/blender --background --python scripts/blender_export_nvdb.py -- ...
"""
import bpy
import sys
import argparse
from pathlib import Path

def parse_args():
    # Find args after '--'
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input .blend file')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--frames', default='1-1', help='Frame range (e.g., 1-100)')
    parser.add_argument('--resolution', type=int, default=256, help='Domain resolution')
    parser.add_argument('--grid', default='density', help='Grid to export')
    return parser.parse_args(argv)

def main():
    args = parse_args()

    # Parse frame range
    if '-' in args.frames:
        start, end = map(int, args.frames.split('-'))
    else:
        start = end = int(args.frames)

    # Open blend file
    bpy.ops.wm.open_mainfile(filepath=args.input)

    # Find fluid domain
    domain = None
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            for mod in obj.modifiers:
                if mod.type == 'FLUID' and mod.fluid_type == 'DOMAIN':
                    domain = obj
                    break

    if not domain:
        print("ERROR: No fluid domain found!")
        sys.exit(1)

    settings = domain.modifiers["Fluid"].domain_settings

    # Set high resolution
    settings.resolution_max = args.resolution
    settings.cache_data_format = 'OPENVDB'

    # Create output directory
    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)

    # Export frames
    for frame in range(start, end + 1):
        bpy.context.scene.frame_set(frame)

        output_file = out_path / f"volume_{frame:04d}.nvdb"

        # Export with NanoVDB format
        bpy.ops.export_scene.openvdb(
            filepath=str(output_file),
            use_nanovdb=True,
        )
        print(f"Exported: {output_file}")

    print(f"\nDone! Exported {end - start + 1} frames to {out_path}")

if __name__ == "__main__":
    main()
```

### Running from WSL:

```bash
# If Blender is installed in WSL
blender --background --python scripts/blender_export_nvdb.py -- \
    --input /path/to/scene.blend \
    --output ~/projects/PlasmaDXR/assets/volumes/my_effect \
    --frames 1-50 \
    --resolution 300
```

---

## Troubleshooting

### Volume is Invisible
1. Check grid type: Must be FLOAT, not HALF/FP16
2. Check grid name: Loader prefers "density"
3. Enable debug mode in ImGui to see AABB intersection

### Volume is Blocky
1. Increase `resolution_max` (256 minimum, 384+ recommended)
2. Enable noise modifier (multiplies effective resolution)
3. Check voxel size in Blender's Volume Properties panel

### Volume is Too Small/Large
1. Use ImGui scale controls (50x-200x typical)
2. Check Blender units vs PlasmaDX world scale
3. Blender is Z-up, some engines are Y-up

### Performance Issues
1. Lower step size increases quality but costs FPS
2. High-res grids (500+ voxels) may need LOD system
3. Consider baking at multiple resolutions

---

## Recommended Settings by Use Case

### Hero Volumetric (center of frame)
- Resolution: 384-512
- Noise: Enabled (scale 2)
- Precision: Full
- Target file size: 30-60 MB

### Background Element
- Resolution: 128-192
- Noise: Optional
- Precision: Full
- Target file size: 5-15 MB

### Animated Sequence
- Resolution: 256 (balance speed/quality)
- Noise: Optional
- Precision: Full
- Per-frame size: 10-30 MB

---

## Integration with PlasmaDX

### Loading in Engine
1. Copy .nvdb to `assets/volumes/`
2. In ImGui: NanoVDB panel > Load Asset
3. Adjust scale (typically 100-200x for Blender units)
4. Set material type (SMOKE, FIRE, PLASMA, NEBULA)

### Shadow Ray Integration (TODO)
Shadow rays require the TLAS to include the NanoVDB AABB. Currently shadows only work with Gaussian particles. See `SHADOW_RAYS_FOR_NANOVDB.md` for implementation plan.

---

## References
- Blender Manual: physics/fluid/type/domain/cache.html
- NanoVDB Paper: Museth, SIGGRAPH 2021
- PlasmaDX NanoVDB System: src/rendering/NanoVDBSystem.cpp
