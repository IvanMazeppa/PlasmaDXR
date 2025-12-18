# Blender 5.x Command-Line Quick Reference

**Purpose**: Fast reference for common Blender command-line operations for VDB export and automation.

**Last Updated**: 2025-12-17

---

## Essential Commands

### Render Operations

```bash
# Render single frame
blender --background scene.blend --render-frame 1

# Render animation (all frames)
blender --background scene.blend --render-anim

# Render specific frame range
blender --background scene.blend --frame-start 1 --frame-end 120 --render-anim

# Render with custom output path
blender --background scene.blend --render-output /tmp/render/frame_#### --render-anim
```

---

### Python Script Execution

```bash
# Run script before opening blend file
blender --python setup.py scene.blend

# Run script after opening blend file
blender scene.blend --python process_scene.py

# Run script in background mode
blender --background scene.blend --python export_vdb.py

# Run script and exit (no rendering)
blender --background scene.blend --python script.py --python-exit
```

---

## VDB Export Workflows

### Method 1: Bake Fluid with OpenVDB Cache (Recommended)

**Step 1: Create Baking Script**

File: `bake_vdb_sequence.py`

```python
import bpy
import os

# Get fluid domain
domain_obj = bpy.data.objects['FluidDomain']  # Adjust name
fluid = domain_obj.modifiers['Fluid'].domain_settings

# Configure cache for PlasmaDX compatibility
fluid.cache_data_format = 'OPENVDB'
fluid.cache_compression = 'BLOSC'
fluid.cache_precision = 'FULL'  # CRITICAL: Must be FULL for PlasmaDX
fluid.cache_frame_start = 1
fluid.cache_frame_end = 120

# Set cache directory (relative to .blend file)
fluid.cache_directory = "//cache_fluid"

# Bake all
print("Starting fluid bake...")
bpy.ops.fluid.bake_all()
print("Bake complete!")

# List generated VDB files
cache_dir = bpy.path.abspath(fluid.cache_directory)
vdb_dir = os.path.join(cache_dir, "data")
vdb_files = sorted([f for f in os.listdir(vdb_dir) if f.endswith('.vdb')])
print(f"\nGenerated {len(vdb_files)} VDB files in: {vdb_dir}")
for vdb in vdb_files:
    print(f"  {vdb}")
```

**Step 2: Run Baking Script**

```bash
# Linux/macOS
blender --background scene.blend --python bake_vdb_sequence.py

# Windows (may fail - use Linux/WSL or GUI baking)
blender --background scene.blend --python bake_vdb_sequence.py
```

**Step 3: Convert VDB to NanoVDB**

```bash
# Single file conversion
python scripts/convert_vdb_to_nvdb.py \
    --input cache_fluid/data/fluid_0001.vdb \
    --output assets/volumes/fluid_0001.nvdb \
    --grid density

# Batch conversion (Linux/macOS)
for vdb in cache_fluid/data/*.vdb; do
    nvdb="assets/volumes/$(basename ${vdb%.vdb}.nvdb)"
    python scripts/convert_vdb_to_nvdb.py --input "$vdb" --output "$nvdb" --grid density
done

# Batch conversion (Windows PowerShell)
Get-ChildItem cache_fluid/data/*.vdb | ForEach-Object {
    $nvdb = "assets/volumes/$($_.BaseName).nvdb"
    python scripts/convert_vdb_to_nvdb.py --input $_.FullName --output $nvdb --grid density
}
```

---

### Method 2: Inspect Existing VDB Files

**Script: Inspect VDB Grid Properties**

File: `inspect_vdb.py`

```python
import bpy
import sys

# Get VDB file path from command line
vdb_path = sys.argv[sys.argv.index("--") + 1]

# Import VDB
bpy.ops.object.volume_import(filepath=vdb_path)
vol_obj = bpy.context.active_object
vol_data = vol_obj.data

print(f"\n=== VDB File: {vdb_path} ===")
print(f"Grids: {len(vol_data.grids)}")

for i, grid in enumerate(vol_data.grids):
    print(f"\nGrid {i}:")
    print(f"  Name: {grid.name}")
    print(f"  Type: {grid.data_type}")
    print(f"  Channels: {grid.channels}")
    print(f"  Loaded: {grid.is_loaded}")
    print(f"  Transform:\n{grid.matrix_object}")
```

**Run Inspection**:

```bash
blender --background --python inspect_vdb.py -- /path/to/file.vdb
```

---

### Method 3: Automated Render + Export Pipeline

**Complete Workflow Script**

File: `render_and_export_vdb.py`

```python
import bpy
import os
import subprocess

# Configure scene
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 120

# Get fluid domain
domain_obj = bpy.data.objects['FluidDomain']
fluid = domain_obj.modifiers['Fluid'].domain_settings

# Set OpenVDB cache (FULL precision required)
fluid.cache_data_format = 'OPENVDB'
fluid.cache_precision = 'FULL'
fluid.cache_compression = 'BLOSC'
fluid.cache_directory = "//cache_fluid"

# Bake fluid simulation
print("Baking fluid simulation...")
bpy.ops.fluid.bake_all()

# Render animation
print("Rendering animation...")
scene.render.filepath = "//renders/frame_"
scene.render.image_settings.file_format = 'PNG'
bpy.ops.render.render(animation=True)

# Convert VDB to NanoVDB (external script)
cache_dir = bpy.path.abspath(fluid.cache_directory)
vdb_dir = os.path.join(cache_dir, "data")
output_dir = bpy.path.abspath("//volumes")
os.makedirs(output_dir, exist_ok=True)

for vdb_file in sorted(os.listdir(vdb_dir)):
    if vdb_file.endswith('.vdb'):
        vdb_path = os.path.join(vdb_dir, vdb_file)
        nvdb_path = os.path.join(output_dir, vdb_file.replace('.vdb', '.nvdb'))

        print(f"Converting {vdb_file} to NanoVDB...")
        subprocess.run([
            'python', 'scripts/convert_vdb_to_nvdb.py',
            '--input', vdb_path,
            '--output', nvdb_path,
            '--grid', 'density'
        ])

print("Pipeline complete!")
```

**Run Complete Pipeline**:

```bash
blender --background scene.blend --python render_and_export_vdb.py
```

---

## Frame-by-Frame Operations

### Render Specific Frames

```python
# render_specific_frames.py
import bpy

frames_to_render = [1, 30, 60, 90, 120]  # Custom frame list
scene = bpy.context.scene

for frame in frames_to_render:
    scene.frame_set(frame)
    scene.render.filepath = f"//renders/frame_{frame:04d}.png"
    bpy.ops.render.render(write_still=True)
    print(f"Rendered frame {frame}")
```

---

### Bake Specific Frame Range (Manual Loop)

**Use Case**: Workaround for Windows command-line baking issues

```python
# bake_frame_range.py
import bpy

domain_obj = bpy.data.objects['FluidDomain']
fluid = domain_obj.modifiers['Fluid'].domain_settings

# Configure cache
fluid.cache_data_format = 'OPENVDB'
fluid.cache_precision = 'FULL'
fluid.cache_directory = "//cache_fluid"

# Manual frame-by-frame bake
scene = bpy.context.scene
for frame in range(1, 121):
    scene.frame_set(frame)
    bpy.ops.fluid.bake_data()  # Bake current frame only
    print(f"Baked frame {frame}")
```

---

## VDB Sequence Inspection

### List All Grids in Sequence

```python
# list_sequence_grids.py
import bpy
import os
import sys

# Get directory from command line
vdb_dir = sys.argv[sys.argv.index("--") + 1]

vdb_files = sorted([f for f in os.listdir(vdb_dir) if f.endswith('.vdb')])

print(f"\n=== Inspecting {len(vdb_files)} VDB files in {vdb_dir} ===\n")

for vdb_file in vdb_files[:5]:  # Inspect first 5 files
    vdb_path = os.path.join(vdb_dir, vdb_file)

    # Import VDB
    bpy.ops.object.volume_import(filepath=vdb_path)
    vol_obj = bpy.context.active_object
    vol_data = vol_obj.data

    print(f"File: {vdb_file}")
    print(f"  Grids: {len(vol_data.grids)}")

    for grid in vol_data.grids:
        print(f"    {grid.name} ({grid.data_type}, {grid.channels} channels)")

    # Clean up
    bpy.data.objects.remove(vol_obj)
    bpy.data.volumes.remove(vol_data)

    print()
```

**Run**:

```bash
blender --background --python list_sequence_grids.py -- /path/to/vdb/directory
```

---

## Batch Processing Utilities

### Bash Script: Convert All VDB Files in Directory

File: `batch_convert_vdb.sh`

```bash
#!/bin/bash

# Usage: ./batch_convert_vdb.sh <input_dir> <output_dir>

INPUT_DIR="$1"
OUTPUT_DIR="$2"
GRID_NAME="${3:-density}"  # Default to "density"

mkdir -p "$OUTPUT_DIR"

for vdb in "$INPUT_DIR"/*.vdb; do
    filename=$(basename "$vdb" .vdb)
    nvdb="$OUTPUT_DIR/${filename}.nvdb"

    echo "Converting $filename..."
    python scripts/convert_vdb_to_nvdb.py \
        --input "$vdb" \
        --output "$nvdb" \
        --grid "$GRID_NAME"
done

echo "Conversion complete! Generated $(ls -1 $OUTPUT_DIR/*.nvdb | wc -l) files."
```

**Run**:

```bash
chmod +x batch_convert_vdb.sh
./batch_convert_vdb.sh cache_fluid/data assets/volumes density
```

---

### PowerShell Script: Convert All VDB Files in Directory

File: `BatchConvertVDB.ps1`

```powershell
# Usage: .\BatchConvertVDB.ps1 -InputDir "cache_fluid\data" -OutputDir "assets\volumes" -GridName "density"

param(
    [Parameter(Mandatory=$true)]
    [string]$InputDir,

    [Parameter(Mandatory=$true)]
    [string]$OutputDir,

    [Parameter(Mandatory=$false)]
    [string]$GridName = "density"
)

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$vdbFiles = Get-ChildItem -Path $InputDir -Filter "*.vdb"

foreach ($vdb in $vdbFiles) {
    $nvdbFile = Join-Path $OutputDir "$($vdb.BaseName).nvdb"

    Write-Host "Converting $($vdb.Name)..."
    python scripts/convert_vdb_to_nvdb.py `
        --input $vdb.FullName `
        --output $nvdbFile `
        --grid $GridName
}

$nvdbCount = (Get-ChildItem -Path $OutputDir -Filter "*.nvdb").Count
Write-Host "Conversion complete! Generated $nvdbCount files."
```

**Run**:

```powershell
.\BatchConvertVDB.ps1 -InputDir "cache_fluid\data" -OutputDir "assets\volumes"
```

---

## Common Pitfalls and Solutions

### Issue: Baking Fails on Windows Command Line

**Symptom**: Empty cache directory, no VDB files generated

**Solution**: Use Linux/WSL for command-line baking, or bake in GUI and export via script

```bash
# WSL example
wsl blender --background scene.blend --python bake_vdb_sequence.py
```

---

### Issue: VDB Files are HALF/MINI Precision (Incompatible with PlasmaDX)

**Symptom**: Volume loads but renders as invisible

**Solution**: Always set precision to FULL in baking script

```python
fluid.cache_precision = 'FULL'  # NOT 'HALF' or 'MINI'
```

---

### Issue: Wrong Grid Selected During Conversion

**Symptom**: Conversion succeeds but volume is empty

**Solution**: Inspect VDB file first to identify correct grid name

```bash
# Inspect grids
blender --background --python inspect_vdb.py -- file.vdb

# Convert with correct grid name
python scripts/convert_vdb_to_nvdb.py --input file.vdb --output file.nvdb --grid density
```

---

### Issue: Script Can't Find Fluid Domain

**Symptom**: `KeyError: 'FluidDomain'` in Python script

**Solution**: Use correct object name from your blend file

```python
# Check object names in Blender Console:
# >>> bpy.data.objects.keys()
# ['Camera', 'Light', 'Domain', 'Emitter']

domain_obj = bpy.data.objects['Domain']  # Use actual name
```

---

## Environment Variables

### BLENDER_USER_SCRIPTS

Set custom scripts directory:

```bash
export BLENDER_USER_SCRIPTS=/path/to/scripts
blender --background scene.blend --python $BLENDER_USER_SCRIPTS/export_vdb.py
```

---

### BLENDER_USER_CONFIG

Set custom config directory:

```bash
export BLENDER_USER_CONFIG=/path/to/config
blender --background scene.blend
```

---

## Debugging Tips

### Enable Verbose Logging

```bash
blender --background --debug-all scene.blend --python script.py 2>&1 | tee blender.log
```

---

### Check Python API Version

```bash
blender --background --python-expr "import bpy; print(bpy.app.version_string)"
```

Output example:
```
5.0.0
```

---

### List All Fluid Domain Properties

```python
# list_fluid_properties.py
import bpy

domain_obj = bpy.data.objects['FluidDomain']
fluid = domain_obj.modifiers['Fluid'].domain_settings

print("\n=== Fluid Domain Settings ===")
for prop in dir(fluid):
    if not prop.startswith('_') and not callable(getattr(fluid, prop)):
        try:
            value = getattr(fluid, prop)
            print(f"{prop}: {value}")
        except:
            pass
```

---

## Quick Test: End-to-End VDB Pipeline

**Single Command Test** (Linux/macOS):

```bash
# 1. Bake simulation
blender --background scene.blend --python bake_vdb_sequence.py

# 2. Convert first frame to NanoVDB
python scripts/convert_vdb_to_nvdb.py \
    --input cache_fluid/data/fluid_0001.vdb \
    --output test_volume.nvdb \
    --grid density

# 3. Load in PlasmaDX
./build/bin/Debug/PlasmaDX-Clean.exe --nvdb test_volume.nvdb
```

---

## References

- Full API documentation: See `BLENDER_5_OPENVDB_API_REFERENCE.md`
- Blender Manual: https://docs.blender.org/manual/en/latest/
- Python API: https://docs.blender.org/api/current/
- Troubleshooting: See `BLENDER_5_TO_NANOVDB_FINDINGS_2025-12-17.md`
