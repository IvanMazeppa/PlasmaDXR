# Blender 5.x OpenVDB and Volumetric Export - API Reference

**Document Purpose**: Comprehensive reference for Blender 5.x OpenVDB fluid simulation, volume object properties, Python API for VDB inspection, and command-line rendering automation.

**Last Updated**: 2025-12-17
**Blender Version**: 5.0 (released November 18, 2025)
**Target**: PlasmaDX-Clean NanoVDB pipeline integration

---

## Table of Contents

1. [Blender 5.0 Breaking Changes](#blender-50-breaking-changes)
2. [Fluid Simulation Cache Settings](#fluid-simulation-cache-settings)
3. [Volume Object Properties](#volume-object-properties)
4. [Python API for VDB Inspection](#python-api-for-vdb-inspection)
5. [Command-Line Rendering and Export](#command-line-rendering-and-export)
6. [Pipeline Integration Notes](#pipeline-integration-notes)

---

## Blender 5.0 Breaking Changes

### NanoVDB Rendering (NEW in 5.0)

**Critical Change**: Smoke and fire simulations are now rendered using **NanoVDB** rather than standard OpenVDB.

- **Memory Impact**: Reduces memory usage significantly
- **GPU Support**: Works on all GPU rendering backends (Cycles Metal, CUDA, OptiX, HIP)
- **Originally Developed by**: NVIDIA, but now supported across all modern graphics APIs

**Implications for PlasmaDX-Clean**:
- Blender 5.0 internally uses NanoVDB for rendering (consistent with our pipeline)
- Export still produces OpenVDB `.vdb` files (requires conversion to `.nvdb`)

### Compression Changes (BREAKING)

**Removed in 5.0**:
- LZO (lite) compression
- LZMA (heavy) compression

**New Default**: ZSTD compression (always enabled, no UI option)

**Migration Impact**:
- Older `.blend` files using LZO/LZMA will trigger warnings
- Users must delete old caches and rebake with ZSTD

**Benefits**:
- Smaller cache files than LZO/LZMA
- Faster read/write performance

**Reference**: [Blender 5.0 Release Notes - Physics](https://developer.blender.org/docs/release_notes/5.0/physics/)

---

## Fluid Simulation Cache Settings

### Cache Format Options

**Location in UI**: Domain Settings → Cache → Cache Type

**Available Formats**:

1. **OpenVDB** (`.vdb`)
   - Description: "Advanced and efficient storage format"
   - Structure: All simulation objects (grids + particles) stored in **one `.vdb` file per frame**
   - Particle Data: Stored using OpenVDB's PointGrid data structures
   - **Recommended for**: PlasmaDX-Clean pipeline

2. **Raw Cache** (`.raw`)
   - Description: Legacy format
   - Structure: Separate files per grid type
   - **Not recommended** for our pipeline

**Reference**: [Cache - Blender 5.0 Manual](https://docs.blender.org/manual/en/latest/physics/fluid/type/domain/cache.html)

---

### OpenVDB Compression Methods

**Property**: `cache_compression` (FluidDomainSettings)

| Compression | Description | Performance | File Size |
|-------------|-------------|-------------|-----------|
| **Zip** | Effective single-threaded compression | Slower | Good |
| **Blosc** | Multithreaded compression | Faster | Similar to Zip |
| **None** | No compression | Fastest | Largest |

**Recommendation**: Use **Blosc** for best balance of speed and size (important for large simulations).

**ZSTD Note**: In Blender 5.0, the underlying disk cache uses ZSTD regardless of this setting (UI option affects OpenVDB-specific compression).

---

### OpenVDB Precision Levels

**Property**: `cache_precision` (FluidDomainSettings)

**CRITICAL FOR PLASMADX-CLEAN**: This setting determines the grid value type in the `.vdb` file, which **directly affects** whether PlasmaDX-Clean's shader can read it.

| Precision | Bit Depth | Grid Type | PlasmaDX Support |
|-----------|-----------|-----------|------------------|
| **Full** | 32-bit float | `PNANOVDB_GRID_TYPE_FLOAT (1)` | ✅ Fully Supported |
| **Half** | 16-bit float | `PNANOVDB_GRID_TYPE_HALF (9)` | ⚠️ Not Yet Supported |
| **Mini** | 8-bit (fallback 16-bit) | Mixed types | ⚠️ Not Yet Supported |

**Current PlasmaDX Limitation** (as documented in `BLENDER_5_TO_NANOVDB_FINDINGS_2025-12-17.md`):
- Shader `nanovdb_raymarch.hlsl` currently accepts **only** `PNANOVDB_GRID_TYPE_FLOAT (1)`
- Half/Mini precision grids will render as **invisible** (density = 0 everywhere)

**Pipeline Requirement**: **Always use "Full" precision** when exporting for PlasmaDX-Clean until Half/FP16 shader support is implemented.

**Trade-offs**:
- **Full**: Larger files, higher bandwidth, but guaranteed compatibility
- **Half/Mini**: 50% memory reduction, but requires shader extension

---

### Cache Bake Types

**Property**: `cache_type`

| Type | Description | Use Case |
|------|-------------|----------|
| **REPLAY** | Use timeline to bake | Interactive preview |
| **MODULAR** | Bake each stage separately | Granular control (data, mesh, particles, noise) |
| **ALL** | Bake all settings at once | Production (fastest, but cannot pause/resume) |

**Command-Line Baking**: Use `MODULAR` or `ALL` for automation.

**Reference**: [FluidDomainSettings - Blender Python API](https://docs.blender.org/api/current/bpy.types.FluidDomainSettings.html)

---

## Volume Object Properties

### Grid Listing and Data Types

**Location in UI**: Object Data Properties → Grids (when a Volume object is selected)

**What You See**:
- List of grids in the OpenVDB file
- Grid names (e.g., `density`, `temperature`, `velocity`, `color`)
- Grid data types (float, vec3f, etc.)

**Grid Data Types Available in Blender**:
- **Float Grid** (scalar values, e.g., density, temperature)
- **Vec3f Grid** (vector values, e.g., velocity, color)
- **Bool Grid** (boolean masks)
- **Int Grid** (integer values)

**Standard Grid Names for Rendering**:

The **Principled Volume** shader expects these grid names by default:
- `density` - Volume density
- `color` - Volume color (RGB)
- `temperature` - Blackbody emission temperature

**Reference**: [Volume Properties - Blender 5.0 Manual](https://docs.blender.org/manual/en/latest/modeling/volumes/properties.html)

---

### Grid Transform and World-Space Mapping

**Key Concept**: Each volume grid has a **transform** (location, rotation, scale) that maps **index-space** to **object-space**.

**Example**:
- If grid scale = 0.1, each voxel is 0.1 units in each direction
- Voxel at index `(5, 0, 0)` maps to object space position `(0.5, 0, 0)`

**Geometry Nodes Access** (NEW in Blender 5.0):
- **Set Grid Transform** node - Modify grid transforms
- **Grid Info** node - Query transform properties

**Implication for PlasmaDX**:
- Blender grid transforms must be correctly interpreted when loading `.nvdb` files
- PlasmaDX has runtime controls: `ScaleGridBounds(scale)`, `SetGridCenter(center)`
- Axis conventions: Blender is **Z-up**, PlasmaDX may be **Y-up** (requires rotation)

**Reference**: [Volume Grids in Geometry Nodes](https://code.blender.org/2025/10/volume-grids-in-geometry-nodes/)

---

### Volume Rendering Properties

**Density and Step Size Computation**:
- **Uniform** - Keeps volume Density and Detail the same regardless of object scale
- **Object Space** - Specify Step Size and Density in world space

**Precision** (runtime, not cache export):
- Value under which voxels are considered empty space (for render optimization)
- Lower values reduce memory at the cost of detail

**Velocity Grids**:
- **Frame** - Velocity grid values are distances per frame
- **Second** - Velocity grid values are distances per second
- Custom velocity multiplier available

**Reference**: [Volume Properties - Blender 5.0 Manual](https://docs.blender.org/manual/en/latest/modeling/volumes/properties.html)

---

## Python API for VDB Inspection

### Volume Object and Grids

**Key Classes**:
- `bpy.types.Volume` - Volume datablock (container for grids)
- `bpy.types.VolumeGrids` - Collection of grids
- `bpy.types.VolumeGrid` - Single 3D volume grid

**Copy-on-Write System**:
- When a volume datablock is copied, both original and new reference **the same grid in memory**
- Deep copy only happens when calling `BKE_volume_grid_openvdb_for_write()`

---

### VolumeGrids API (`bpy.types.VolumeGrids`)

**Properties**:

```python
volume.grids.active_index  # int, index of active grid (0-based)
volume.grids.error_message  # str, error details if loading failed (readonly)
volume.grids.frame  # int, current frame number for loading (readonly)
volume.grids.frame_filepath  # str, VDB file path for current frame (readonly)
```

**Usage Example**:

```python
import bpy

# Get volume object
vol_obj = bpy.data.objects['Volume']
vol_data = vol_obj.data  # bpy.types.Volume

# List all grids
for i, grid in enumerate(vol_data.grids):
    print(f"Grid {i}: {grid.name} (type: {grid.data_type}, channels: {grid.channels})")
```

**Reference**: [VolumeGrids - Blender Python API](https://docs.blender.org/api/current/bpy.types.VolumeGrids.html)

---

### VolumeGrid API (`bpy.types.VolumeGrid`)

**Properties**:

```python
grid.name            # str, grid name (e.g., "density")
grid.data_type       # enum, voxel data type (readonly)
grid.channels        # int, number of dimensions (1 for scalar, 3 for vector)
grid.is_loaded       # bool, whether grid tree is loaded in memory (readonly)
grid.matrix_object   # mathutils.Matrix (4x4), transform from voxel index to object space (readonly)
```

**Data Types** (enum `data_type`):
- `'FLOAT'` - 32-bit float
- `'HALF'` - 16-bit float
- `'DOUBLE'` - 64-bit float
- `'INT'` - 32-bit integer
- `'BOOLEAN'` - Boolean
- `'VECTOR_FLOAT'` - 3D vector of floats
- `'VECTOR_INT'` - 3D vector of integers
- `'UNKNOWN'` - Unknown type

**Example: Inspect Grid Types**:

```python
import bpy

vol = bpy.data.volumes['MyVolume']

for grid in vol.grids:
    print(f"Grid: {grid.name}")
    print(f"  Type: {grid.data_type}")
    print(f"  Channels: {grid.channels}")
    print(f"  Loaded: {grid.is_loaded}")
    print(f"  Transform:\n{grid.matrix_object}")
```

**Reference**: [VolumeGrid - Blender Python API](https://docs.blender.org/api/current/bpy.types.VolumeGrid.html)

---

### Importing VDB Files via Python

**Method 1: Operator**

```python
import bpy

# Import OpenVDB file (creates Volume object)
bpy.ops.object.volume_import(filepath="/path/to/file.vdb")

# Access the volume object (assumes it's active)
vol_obj = bpy.context.active_object
vol_data = vol_obj.data
```

**Method 2: Direct Data Block Creation**

```python
import bpy

# Create a new volume datablock
vol = bpy.data.volumes.new("MyVolume")

# Set filepath for frame sequence
vol.filepath = "/path/to/sequence_####.vdb"

# Set frame range
vol.frame_start = 1
vol.frame_end = 120
vol.frame_offset = 0

# Create volume object
vol_obj = bpy.data.objects.new("VolumeObject", vol)
bpy.context.collection.objects.link(vol_obj)
```

**Frame Sequences**: Use `####` for zero-padded frame numbers (e.g., `smoke_0001.vdb`, `smoke_0002.vdb`)

---

### Inspecting Fluid Domain Settings

**Access Fluid Domain**:

```python
import bpy

# Get domain object
domain_obj = bpy.data.objects['FluidDomain']
fluid_settings = domain_obj.modifiers['Fluid'].domain_settings

# Cache format and compression
print(f"Cache Format: {fluid_settings.cache_data_format}")  # 'OPENVDB' or 'RAW'
print(f"Compression: {fluid_settings.cache_compression}")   # 'ZIP', 'BLOSC', 'NONE'
print(f"Precision: {fluid_settings.cache_precision}")       # 'FULL', 'HALF', 'MINI'

# Cache frame range
print(f"Start: {fluid_settings.cache_frame_start}")
print(f"End: {fluid_settings.cache_frame_end}")

# Cache directory
print(f"Cache Dir: {fluid_settings.cache_directory}")
```

**Key Properties**:
- `cache_data_format` - `'OPENVDB'` or `'RAW'`
- `cache_compression` - `'ZIP'`, `'BLOSC'`, `'NONE'`
- `cache_precision` - `'FULL'`, `'HALF'`, `'MINI'`
- `cache_directory` - Path to cache folder
- `cache_frame_start` / `cache_frame_end` - Frame range

**Reference**: [FluidDomainSettings - Blender Python API](https://docs.blender.org/api/current/bpy.types.FluidDomainSettings.html)

---

## Command-Line Rendering and Export

### Basic Command-Line Rendering

**Render Single Frame**:

```bash
blender --background scene.blend --render-frame 1
```

**Render Animation**:

```bash
blender --background scene.blend --render-anim
```

**Render Specific Frame Range**:

```bash
blender --background scene.blend --frame-start 1 --frame-end 120 --render-anim
```

**Reference**: [Command-line rendering](https://surf-visualization.github.io/blender-course/basics/blender_fundamentals/command_line/)

---

### Running Python Scripts from Command Line

**Execute Script Before Opening Blend File**:

```bash
blender --python setup.py scene.blend
```

**Execute Script After Opening Blend File**:

```bash
blender scene.blend --python process_scene.py
```

**Execute Script in Background Mode**:

```bash
blender --background scene.blend --python export_vdb.py
```

**Reference**: [Quickstart - Blender Python API](https://docs.blender.org/api/current/info_quickstart.html)

---

### Baking Fluid Simulations from Command Line

**Python Operators Available** (from `bpy.ops.fluid`):

```python
bpy.ops.fluid.bake_all()        # Bake entire simulation
bpy.ops.fluid.bake_data()       # Bake fluid data only
bpy.ops.fluid.bake_mesh()       # Bake fluid mesh only
bpy.ops.fluid.bake_noise()      # Bake noise only
bpy.ops.fluid.bake_particles()  # Bake particles only
bpy.ops.fluid.free_all()        # Free entire simulation
bpy.ops.fluid.pause_bake()      # Pause baking
```

**Example Script: Automated Bake**

```python
# bake_fluid.py
import bpy

# Open blend file (already loaded if using --background file.blend --python)
# bpy.ops.wm.open_mainfile(filepath="/path/to/scene.blend")

# Set domain settings (optional, if not already configured)
domain_obj = bpy.data.objects['FluidDomain']
fluid = domain_obj.modifiers['Fluid'].domain_settings

# Configure cache settings
fluid.cache_data_format = 'OPENVDB'
fluid.cache_compression = 'BLOSC'
fluid.cache_precision = 'FULL'  # CRITICAL for PlasmaDX compatibility
fluid.cache_frame_start = 1
fluid.cache_frame_end = 120

# Bake all
bpy.ops.fluid.bake_all()

print("Bake complete!")
```

**Run Script**:

```bash
blender --background scene.blend --python bake_fluid.py
```

**Known Issue (Windows)**: Baking from command line can fail on Windows (creates empty cache directory). Works reliably on Linux. See [T41865](https://developer.blender.org/T41865).

**Workaround**: Use `bpy.app.handlers` or manual frame-by-frame baking.

**Reference**: [Fluid Operators - Blender Python API](https://docs.blender.org/api/current/bpy.ops.fluid.html)

---

### Exporting VDB Sequences Programmatically

**Approach**: Blender does **not** have a built-in "export VDB" operator. Instead, you bake fluid simulations with OpenVDB cache format, which **automatically** writes `.vdb` files.

**Steps**:

1. Configure domain with OpenVDB cache format
2. Bake simulation using `bpy.ops.fluid.bake_all()`
3. VDB files are written to `cache_directory`
4. Copy/convert VDB files to NanoVDB using PlasmaDX conversion script

**Example: Automated Workflow**

```python
# export_vdb_sequence.py
import bpy
import os
import shutil
import subprocess

# Configure fluid domain
domain_obj = bpy.data.objects['FluidDomain']
fluid = domain_obj.modifiers['Fluid'].domain_settings

# Set cache to OpenVDB with FULL precision
fluid.cache_data_format = 'OPENVDB'
fluid.cache_precision = 'FULL'  # REQUIRED for PlasmaDX
fluid.cache_compression = 'BLOSC'
fluid.cache_frame_start = 1
fluid.cache_frame_end = 120

# Set cache directory
cache_dir = "/tmp/blender_cache"
fluid.cache_directory = cache_dir

# Bake
print("Starting bake...")
bpy.ops.fluid.bake_all()
print("Bake complete!")

# VDB files are now in: cache_dir/data/*.vdb
vdb_dir = os.path.join(cache_dir, "data")
vdb_files = sorted([f for f in os.listdir(vdb_dir) if f.endswith('.vdb')])

print(f"Generated {len(vdb_files)} VDB files:")
for vdb in vdb_files:
    print(f"  {vdb}")

# Optional: Convert to NanoVDB using PlasmaDX script
# for vdb_file in vdb_files:
#     vdb_path = os.path.join(vdb_dir, vdb_file)
#     nvdb_path = vdb_path.replace('.vdb', '.nvdb')
#     subprocess.run(['python', 'scripts/convert_vdb_to_nvdb.py', vdb_path, nvdb_path])
```

**Run**:

```bash
blender --background scene.blend --python export_vdb_sequence.py
```

---

### Headless Rendering with Blenderless (Advanced)

**Blenderless Package**: Python library for headless Blender rendering without GUI.

**Installation**:

```bash
pip install blenderless
```

**Why Use Blenderless**:
- `bpy` can only be imported once per Python process
- Blenderless handles virtual framebuffer for headless servers
- Defines scene in advance, interacts with `bpy` only at render time

**Example Usage**:

```python
from blenderless import Blender

# Create blender instance
blender = Blender(resolution=(1920, 1080), samples=128)

# Define scene (without bpy imported yet)
blender.add_mesh('cube', location=(0, 0, 0))
blender.add_light('sun', energy=5.0)

# Render (this spawns subprocess with virtual framebuffer)
blender.render(output_path="output.png")
```

**Reference**: [Blenderless - GitHub](https://github.com/oqton/blenderless)

---

### Frame-by-Frame Rendering Loop

**Use Case**: Render specific frames, process each frame, or implement custom rendering logic.

**Example Script**:

```python
# render_frames.py
import bpy

scene = bpy.context.scene

# Set render settings
scene.render.filepath = "/tmp/render/frame_"
scene.render.image_settings.file_format = 'PNG'

# Render frames 1-120
for frame in range(1, 121):
    scene.frame_set(frame)
    scene.render.filepath = f"/tmp/render/frame_{frame:04d}.png"
    bpy.ops.render.render(write_still=True)
    print(f"Rendered frame {frame}")
```

**Run**:

```bash
blender --background scene.blend --python render_frames.py
```

**Reference**: [How to run script on every frame](https://blenderartists.org/t/how-to-run-script-on-every-frame-in-blender-render/699404)

---

## Pipeline Integration Notes

### Critical Settings for PlasmaDX-Clean Compatibility

**REQUIRED**:
1. **Cache Format**: OpenVDB (not Raw)
2. **Precision**: **FULL** (32-bit float) - Half/Mini are not yet supported by shader
3. **Compression**: Blosc recommended (good balance of speed/size)

**RECOMMENDED**:
1. **Grid Naming**: Use standard names (`density`, `temperature`, `color`)
2. **Frame Sequences**: Use zero-padded numeric suffixes (`####`)
3. **Coordinate System**: Z-up (standard Blender), rotate to Y-up in PlasmaDX if needed

---

### Conversion Workflow

**Blender → PlasmaDX Pipeline**:

1. **Bake Simulation** in Blender with OpenVDB cache (FULL precision)
2. **Locate VDB Files** in cache directory (e.g., `//cache_fluid/data/*.vdb`)
3. **Convert to NanoVDB** using `scripts/convert_vdb_to_nvdb.py`
4. **Load in PlasmaDX** using `NanoVDBSystem::LoadFromFile()`

**Example Conversion**:

```bash
# Convert single VDB file
python scripts/convert_vdb_to_nvdb.py \
    --input /path/to/cache/data/fluid_0001.vdb \
    --output assets/volumes/fluid_0001.nvdb \
    --grid density

# Batch convert sequence
for vdb in /path/to/cache/data/*.vdb; do
    nvdb="${vdb%.vdb}.nvdb"
    python scripts/convert_vdb_to_nvdb.py --input "$vdb" --output "$nvdb" --grid density
done
```

---

### Common Issues and Solutions

**Issue 1: Invisible Volume After Loading**

**Symptoms**:
- Volume AABB is visible in debug mode
- "Inside AABB but no density" debug colors (cyan)
- No rendering despite correct bounds

**Root Causes** (from `BLENDER_5_TO_NANOVDB_FINDINGS_2025-12-17.md`):
1. **Grid Type Mismatch**: Blender exported with Half/Mini precision → shader only supports FLOAT
2. **Wrong Grid Selected**: Loader reads grid index 0, but `density` is at a different index
3. **Grid Not Named "density"**: Conversion script defaults to first float grid if `density` not found

**Solutions**:
- Set Blender cache precision to **FULL**
- Verify grid name is `density` in Blender Volume Properties
- Use `--grid density` flag when converting
- Log grid type/name in `NanoVDBSystem::LoadFromFile()` for debugging

**Issue 2: Scale/Transform Mismatch**

**Symptoms**:
- Volume appears tiny or off-center
- Camera appears to miss AABB entirely

**Root Causes**:
- Blender units may be small (1-10 units) vs PlasmaDX scene scale (100s-1000s)
- Axis convention mismatch (Z-up vs Y-up)

**Solutions**:
- Use PlasmaDX runtime controls: `ScaleGridBounds(50.0)`, `SetGridCenter(position)`
- Apply rotation if coordinate system differs
- Enable NanoVDB debug mode to visualize bounds

**Issue 3: Baking Fails on Windows Command Line**

**Symptoms**:
- `bpy.ops.fluid.bake_all()` creates empty cache directory on Windows
- Works fine in GUI, fails in `--background` mode

**Root Cause**: Known Blender bug ([T41865](https://developer.blender.org/T41865))

**Solutions**:
- Bake in GUI, export from command line
- Use Linux/WSL for command-line baking
- Implement frame-by-frame manual baking loop

---

### Debugging Checklist

**Step 1: Verify Blender Export**

In Blender UI:
1. Select Volume object
2. Go to Object Data Properties → Grids
3. Verify grid names (should include `density`)
4. Verify data types (should be `FLOAT` if precision was FULL)

**Step 2: Verify Conversion**

In PlasmaDX conversion script:
1. Log grid count, names, types during conversion
2. Ensure `--grid density` selects correct grid
3. Verify output `.nvdb` file is non-zero size

**Step 3: Verify Loading**

In PlasmaDX engine:
1. Add logging to `NanoVDBSystem::LoadFromFile()`:
   - Grid name
   - Grid value type (should be `PNANOVDB_GRID_TYPE_FLOAT = 1`)
   - Bounds (worldMin/Max)
   - Active voxel count
2. Check ImGui for grid info display

**Step 4: Verify Rendering**

In PlasmaDX shader:
1. Enable NanoVDB debug visualization
2. Check for "density found" colors (green) vs "no density" (cyan)
3. If no density despite correct type, check density scale/coefficients

---

## Sources and References

### Official Blender 5.0 Documentation

- [Cache - Blender 5.0 Manual](https://docs.blender.org/manual/en/latest/physics/fluid/type/domain/cache.html)
- [Volume Properties - Blender 5.0 Manual](https://docs.blender.org/manual/en/latest/modeling/volumes/properties.html)
- [Introduction to Volumes - Blender 5.0 Manual](https://docs.blender.org/manual/en/latest/modeling/volumes/introduction.html)
- [Blender 5.0 Release Notes - Physics](https://developer.blender.org/docs/release_notes/5.0/physics/)
- [Blender 5.0 Release Notes - Python API](https://developer.blender.org/docs/release_notes/5.0/python_api/)

### Python API Documentation

- [Blender Python API](https://docs.blender.org/api/current/index.html)
- [FluidDomainSettings - Blender Python API](https://docs.blender.org/api/current/bpy.types.FluidDomainSettings.html)
- [VolumeGrids - Blender Python API](https://docs.blender.org/api/current/bpy.types.VolumeGrids.html)
- [VolumeGrid - Blender Python API](https://docs.blender.org/api/current/bpy.types.VolumeGrid.html)
- [Fluid Operators - Blender Python API](https://docs.blender.org/api/current/bpy.ops.fluid.html)

### Community Resources

- [Visualizing volumetric data through OpenVDB](https://surf-visualization.github.io/blender-course/advanced/python_scripting/4_volumetric_data/)
- [Command-line rendering Tutorial](https://surf-visualization.github.io/blender-course/basics/blender_fundamentals/command_line/)
- [Volume Grids in Geometry Nodes - Blender Blog](https://code.blender.org/2025/10/volume-grids-in-geometry-nodes/)
- [Blenderless - GitHub](https://github.com/oqton/blenderless)
- [Baking from Command Line - FLIP Fluids Wiki](https://github.com/rlguy/Blender-FLIP-Fluids/wiki/Baking-from-the-Command-Line)

### Bug Reports and Issues

- [T41865 - Fluid bake not possible from command line on Windows](https://developer.blender.org/T41865)
- [Blender openvdb liquid simulation format specifications](https://devtalk.blender.org/t/blender-openvdb-liquid-simulation-format-specifications/25596)
- [Getting OpenVDB grids in render engine 2.83](https://devtalk.blender.org/t/getting-openvdb-grids-in-render-engine-2-83/12270)

---

## Version History

**2025-12-17**: Initial documentation created based on Blender 5.0 release (Nov 18, 2025)
**Target Blender Version**: 5.0
**Compatible with**: PlasmaDX-Clean NanoVDB pipeline (branch 0.24.1)
