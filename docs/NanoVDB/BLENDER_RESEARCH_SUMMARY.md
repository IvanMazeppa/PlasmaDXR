# Blender 5.x OpenVDB Research Summary

**Research Date**: 2025-12-17
**Researcher**: Framework Documentation Researcher (Claude)
**Target Project**: PlasmaDX-Clean NanoVDB Pipeline

---

## Overview

This document summarizes comprehensive research into Blender 5.x OpenVDB export and volumetric workflows, conducted using official Blender documentation and community resources. The research focuses on enabling a robust Blender → OpenVDB → NanoVDB → PlasmaDX pipeline.

---

## Documentation Produced

### 1. BLENDER_5_OPENVDB_API_REFERENCE.md

**Location**: `/docs/NanoVDB/BLENDER_5_OPENVDB_API_REFERENCE.md`

**Comprehensive reference covering**:
- Blender 5.0 breaking changes (NanoVDB rendering, ZSTD compression)
- Fluid simulation cache settings (format, compression, precision)
- Volume object properties (grid types, transforms, rendering)
- Python API for VDB inspection (bpy.types.Volume, VolumeGrid, VolumeGrids)
- Command-line rendering and export automation
- Pipeline integration notes and troubleshooting

**Use this document when**:
- You need detailed API reference for Python scripting
- You're debugging VDB export issues
- You want to understand Blender's OpenVDB implementation
- You need to verify compatibility with PlasmaDX shader requirements

---

### 2. BLENDER_CLI_QUICK_REFERENCE.md

**Location**: `/docs/NanoVDB/BLENDER_CLI_QUICK_REFERENCE.md`

**Quick reference guide with**:
- Ready-to-use command-line examples
- Python scripts for baking, rendering, and VDB inspection
- Batch processing utilities (Bash and PowerShell)
- Frame-by-frame operation workflows
- Common pitfalls and solutions
- End-to-end pipeline test commands

**Use this document when**:
- You need a quick copy-paste command for Blender CLI
- You're automating VDB export workflows
- You want batch conversion scripts
- You need debugging commands

---

## Key Findings

### Critical for PlasmaDX Compatibility

#### 1. Precision Settings MUST Be "Full"

**Blender Cache Setting**: `cache_precision = 'FULL'` (32-bit float)

**Why This Matters**:
- PlasmaDX shader `nanovdb_raymarch.hlsl` currently supports **only** `PNANOVDB_GRID_TYPE_FLOAT (1)`
- Blender's "Half" (16-bit) or "Mini" (8-bit) precision creates incompatible grid types
- Incompatible grids render as **invisible** (density = 0 everywhere)

**Python API**:
```python
fluid.cache_precision = 'FULL'  # REQUIRED - NOT 'HALF' or 'MINI'
```

**Evidence**: See `BLENDER_5_TO_NANOVDB_FINDINGS_2025-12-17.md` Section B and Hypothesis H1

---

#### 2. Grid Selection by Index 0 Can Fail

**Current PlasmaDX Behavior**: `NanoVDBSystem::LoadFromFile()` reads grid index 0

**Problem**: Multi-grid VDB files may have `temperature` or `velocity` at index 0, not `density`

**Solution Options**:
1. **Short-term**: Use `--grid density` flag in conversion script to select by name
2. **Long-term**: Update `NanoVDBSystem::LoadFromFile()` to search for `density` grid by name

**Evidence**: See `BLENDER_5_TO_NANOVDB_FINDINGS_2025-12-17.md` Section C and Hypothesis H2

---

#### 3. Blender 5.0 Uses NanoVDB for Rendering

**Breaking Change**: Blender 5.0 switched from OpenVDB to NanoVDB for internal rendering

**Benefits**:
- Reduced memory usage
- GPU backend agnostic (Metal, CUDA, OptiX, HIP)
- Aligns with PlasmaDX's NanoVDB approach

**Caveat**: Export still produces OpenVDB `.vdb` files (conversion to `.nvdb` still required)

**Reference**: [Blender 5.0 Release Notes - Physics](https://developer.blender.org/docs/release_notes/5.0/physics/)

---

#### 4. Compression Method Standardized to ZSTD

**Breaking Change**: LZO and LZMA compression removed in favor of ZSTD

**Impact**:
- Older `.blend` files using LZO/LZMA will warn on load
- Must rebake simulations with ZSTD (automatic)
- No UI option for compression method (always enabled)

**Benefits**:
- Smaller cache files than LZO/LZMA
- Faster read/write performance

---

### Python API Essentials

#### Fluid Domain Settings

```python
import bpy

domain_obj = bpy.data.objects['FluidDomain']
fluid = domain_obj.modifiers['Fluid'].domain_settings

# Cache format
fluid.cache_data_format = 'OPENVDB'  # or 'RAW'

# Compression
fluid.cache_compression = 'BLOSC'  # or 'ZIP', 'NONE'

# Precision (CRITICAL)
fluid.cache_precision = 'FULL'  # REQUIRED for PlasmaDX

# Frame range
fluid.cache_frame_start = 1
fluid.cache_frame_end = 120

# Cache directory
fluid.cache_directory = "//cache_fluid"

# Bake all
bpy.ops.fluid.bake_all()
```

**API Reference**: [FluidDomainSettings](https://docs.blender.org/api/current/bpy.types.FluidDomainSettings.html)

---

#### Volume Grid Inspection

```python
import bpy

# Import VDB file
bpy.ops.object.volume_import(filepath="/path/to/file.vdb")
vol_obj = bpy.context.active_object
vol_data = vol_obj.data  # bpy.types.Volume

# List all grids
for i, grid in enumerate(vol_data.grids):
    print(f"Grid {i}: {grid.name}")
    print(f"  Type: {grid.data_type}")  # 'FLOAT', 'HALF', 'VECTOR_FLOAT', etc.
    print(f"  Channels: {grid.channels}")  # 1 for scalar, 3 for vector
    print(f"  Loaded: {grid.is_loaded}")
    print(f"  Transform:\n{grid.matrix_object}")
```

**API References**:
- [VolumeGrids](https://docs.blender.org/api/current/bpy.types.VolumeGrids.html)
- [VolumeGrid](https://docs.blender.org/api/current/bpy.types.VolumeGrid.html)

---

### Command-Line Workflows

#### Basic VDB Export Workflow

```bash
# 1. Bake simulation with Python script
blender --background scene.blend --python bake_vdb_sequence.py

# 2. VDB files are written to cache_fluid/data/*.vdb

# 3. Convert to NanoVDB
for vdb in cache_fluid/data/*.vdb; do
    nvdb="assets/volumes/$(basename ${vdb%.vdb}.nvdb)"
    python scripts/convert_vdb_to_nvdb.py --input "$vdb" --output "$nvdb" --grid density
done

# 4. Load in PlasmaDX
./PlasmaDX-Clean.exe --nvdb assets/volumes/fluid_0001.nvdb
```

---

#### VDB Inspection from Command Line

```bash
# Inspect VDB grids without opening GUI
blender --background --python inspect_vdb.py -- /path/to/file.vdb
```

**Script**: See `BLENDER_CLI_QUICK_REFERENCE.md` for full `inspect_vdb.py` script

---

### Known Issues and Workarounds

#### Issue: Command-Line Baking Fails on Windows

**Symptom**: `bpy.ops.fluid.bake_all()` creates empty cache directory on Windows CLI

**Root Cause**: Known Blender bug ([T41865](https://developer.blender.org/T41865))

**Workarounds**:
1. Use Linux/WSL for command-line baking
2. Bake in GUI, export via command-line script
3. Use frame-by-frame manual baking loop (see CLI reference)

**Status**: Works reliably on Linux/macOS

---

#### Issue: Invisible Volumes After Loading

**Symptoms**:
- Volume AABB visible in debug mode
- "Inside AABB but no density" colors (cyan)
- No rendering despite correct bounds

**Root Causes** (in order of likelihood):
1. Grid type mismatch (Half/Mini precision instead of Full)
2. Wrong grid selected (index 0 is not `density`)
3. Grid not named "density" (conversion defaults to first float grid)

**Solutions**:
1. Always use `cache_precision = 'FULL'` in Blender
2. Use `--grid density` flag when converting
3. Verify grid names in Blender Volume Properties before exporting
4. Add logging to `NanoVDBSystem::LoadFromFile()` to surface grid name/type

**Evidence**: See `BLENDER_5_TO_NANOVDB_FINDINGS_2025-12-17.md` Sections A, B, C

---

#### Issue: Scale/Transform Mismatch

**Symptoms**:
- Volume appears tiny or off-center
- Camera appears to miss AABB entirely

**Root Causes**:
- Blender units may be small (1-10 units) vs PlasmaDX scene scale (100s-1000s)
- Axis convention mismatch (Blender Z-up vs PlasmaDX Y-up)

**Solutions**:
- Use PlasmaDX runtime controls: `ScaleGridBounds(50.0)`, `SetGridCenter(position)`
- Apply rotation if coordinate system differs
- Enable NanoVDB debug mode to visualize bounds and diagnose hit/miss

**Evidence**: See `BLENDER_5_TO_NANOVDB_FINDINGS_2025-12-17.md` Hypothesis H3

---

## Recommended Pipeline Configuration

### Blender Fluid Domain Settings (Python)

```python
import bpy

domain = bpy.data.objects['FluidDomain'].modifiers['Fluid'].domain_settings

# REQUIRED for PlasmaDX compatibility
domain.cache_data_format = 'OPENVDB'
domain.cache_precision = 'FULL'  # CRITICAL - must be 32-bit float

# RECOMMENDED for performance
domain.cache_compression = 'BLOSC'  # Multithreaded, fast

# Frame range (adjust as needed)
domain.cache_frame_start = 1
domain.cache_frame_end = 120

# Cache directory (relative to .blend file)
domain.cache_directory = "//cache_fluid"
```

---

### Conversion Script Arguments

```bash
python scripts/convert_vdb_to_nvdb.py \
    --input cache_fluid/data/fluid_0001.vdb \
    --output assets/volumes/fluid_0001.nvdb \
    --grid density  # Select density grid by name (critical)
```

---

### PlasmaDX Runtime Configuration

**ImGui Controls** (recommended initial values):
- **Scale**: 50.0 - 200.0 (adjust based on Blender domain size)
- **Center**: Match camera position or origin
- **Debug Mode**: Enable to verify AABB hit/miss and density sampling

**Configuration File** (optional):
```json
{
  "nanovdb": {
    "file": "assets/volumes/fluid_0001.nvdb",
    "scale": 100.0,
    "center": [0.0, 0.0, 0.0],
    "debug_mode": true
  }
}
```

---

## Testing and Validation Checklist

### Pre-Export Validation (Blender)

- [ ] Fluid domain cache format is **OPENVDB**
- [ ] Cache precision is **FULL** (32-bit float)
- [ ] Frame range is correct (start, end)
- [ ] Cache directory is accessible
- [ ] Bake completes without errors
- [ ] VDB files exist in `cache_directory/data/*.vdb`

### Post-Export Validation (Blender)

- [ ] Import one VDB file as Volume object
- [ ] Open Volume Properties → Grids panel
- [ ] Verify `density` grid exists
- [ ] Verify `density` data type is **FLOAT** (not HALF/MINI)
- [ ] Check other grids (temperature, velocity, color) as needed

### Conversion Validation (CLI)

- [ ] Conversion script runs without errors
- [ ] Output `.nvdb` file is non-zero size
- [ ] Conversion log shows grid name: `density`
- [ ] Conversion log shows grid type: `FLOAT (1)`

### Loading Validation (PlasmaDX)

- [ ] `NanoVDBSystem::LoadFromFile()` succeeds
- [ ] ImGui shows grid bounds (worldMin/Max)
- [ ] ImGui shows active voxel count > 0
- [ ] Debug mode shows AABB hit (not miss)
- [ ] Debug mode shows density found (green, not cyan)

### Rendering Validation (PlasmaDX)

- [ ] Volume is visible (not invisible)
- [ ] Density scale produces reasonable opacity
- [ ] Lighting interacts with volume
- [ ] Ray marching produces smooth falloff (not hard edges)
- [ ] No artifacts (flickering, banding, noise beyond expected)

---

## Future Enhancements

### Shader Support for Half/FP16 Grids (Planned)

**Goal**: Support Blender's Half/Mini precision exports to reduce memory usage

**Implementation** (in `shaders/volumetric/nanovdb_raymarch.hlsl`):
- Accept `PNANOVDB_GRID_TYPE_HALF (9)` in addition to FLOAT
- Accept `PNANOVDB_GRID_TYPE_FP16 (15)` in addition to FLOAT
- Read values via `pnanovdb_read_half()` helper
- Convert to float for accumulation

**Benefits**:
- 50% memory reduction (Half vs Full precision)
- Better performance for large assets
- Direct compatibility with Blender's Half/Mini exports

**Risks**:
- Requires careful sampling correctness
- May need different normalization/scaling

**Reference**: See `BLENDER_5_TO_NANOVDB_FINDINGS_2025-12-17.md` Option A

---

### Multi-Grid Support (Planned)

**Goal**: Load and render multiple grids from single VDB file (density + color + temperature)

**Implementation**:
- Extend `NanoVDBSystem::LoadFromFile()` to load all grids
- Expose grid selection via ImGui dropdown
- Support temperature-based emission (blackbody colors)
- Support color grids for heterogeneous volumes

**Benefits**:
- Richer volumetric rendering (colored smoke, fire)
- Better match to Blender's Principled Volume shader
- More expressive artistic control

---

### Sequence Animation Support (Planned)

**Goal**: Load and animate VDB frame sequences automatically

**Implementation**:
- VDB sequence metadata (frame range, path template)
- Frame interpolation/blending for smooth motion blur
- Background streaming for large sequences
- Frame caching strategy

**Benefits**:
- Animated smoke, explosions, clouds
- Precomputed physics simulations from Blender
- No runtime physics cost (playback only)

**Reference**: See `docs/NanoVDB/NANOVDB_UNIFIED_ROADMAP_V1.md` for multi-volume system design

---

## References and Sources

### Official Blender Documentation

1. [Cache - Blender 5.0 Manual](https://docs.blender.org/manual/en/latest/physics/fluid/type/domain/cache.html)
2. [Volume Properties - Blender 5.0 Manual](https://docs.blender.org/manual/en/latest/modeling/volumes/properties.html)
3. [Volume Introduction - Blender 5.0 Manual](https://docs.blender.org/manual/en/latest/modeling/volumes/introduction.html)
4. [Blender 5.0 Release Notes - Physics](https://developer.blender.org/docs/release_notes/5.0/physics/)
5. [FluidDomainSettings - Python API](https://docs.blender.org/api/current/bpy.types.FluidDomainSettings.html)
6. [VolumeGrids - Python API](https://docs.blender.org/api/current/bpy.types.VolumeGrids.html)
7. [VolumeGrid - Python API](https://docs.blender.org/api/current/bpy.types.VolumeGrid.html)

### Community Resources

8. [Visualizing volumetric data through OpenVDB](https://surf-visualization.github.io/blender-course/advanced/python_scripting/4_volumetric_data/)
9. [Command-line rendering Tutorial](https://surf-visualization.github.io/blender-course/basics/blender_fundamentals/command_line/)
10. [Volume Grids in Geometry Nodes - Blender Blog](https://code.blender.org/2025/10/volume-grids-in-geometry-nodes/)
11. [Blenderless - GitHub](https://github.com/oqton/blenderless)
12. [Baking from Command Line - FLIP Fluids Wiki](https://github.com/rlguy/Blender-FLIP-Fluids/wiki/Baking-from-the-Command-Line)

### PlasmaDX Documentation

13. `BLENDER_5_TO_NANOVDB_FINDINGS_2025-12-17.md` - Root cause analysis
14. `NANOVDB_UNIFIED_ROADMAP_V1.md` - Multi-volume system design
15. `TROUBLESHOOTING_ASSET_VISIBILITY_AND_ANIMATION.md` - Debug workflows

---

## Research Methodology

**Tools Used**:
- WebSearch (Brave Search API)
- Blender 5.0 Official Documentation (manual + API)
- Community forums and GitHub repositories
- Developer bug trackers

**Sources Verified**:
- All code examples cross-referenced with official Python API
- Precision/compression settings verified against Blender 5.0 manual
- Breaking changes confirmed via release notes (Nov 18, 2025)
- Known issues validated against developer bug tracker

**Limitations**:
- MCP `blender-manual` tools were not accessible (attempted but not configured)
- Used web search as primary research method (authoritative sources prioritized)
- Some advanced features (Geometry Nodes volume grids) documented but not deeply tested
- Windows command-line baking issue confirmed but not personally reproduced

---

## Conclusion

This research provides a comprehensive foundation for the Blender → OpenVDB → NanoVDB → PlasmaDX pipeline. The two key blockers (precision mismatch and grid selection) have clear solutions:

1. **Always use FULL precision** when exporting from Blender
2. **Select grids by name** (`density`) rather than index during conversion

With these guidelines, the pipeline should produce reliable, visible volumetric assets for PlasmaDX-Clean.

Future work (Half/FP16 support, multi-grid, sequences) will expand capabilities but is not required for basic functionality.

---

**Last Updated**: 2025-12-17
**Research Status**: Complete
**Next Steps**: Validate findings with end-to-end test (Blender → VDB → NanoVDB → PlasmaDX)
