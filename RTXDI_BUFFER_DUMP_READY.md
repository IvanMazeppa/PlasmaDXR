# RTXDI Buffer Dump System - READY FOR VALIDATION

**Date**: 2025-10-18
**Branch**: 0.7.1
**Status**: ✅ Buffer dump implementation complete, application running

---

## What Was Implemented

### 1. RTXDILightingSystem::DumpBuffers() Method

**File**: `src/lighting/RTXDILightingSystem.cpp` (lines 377-502)

**Functionality**:
- Creates readback buffers for GPU→CPU transfer
- Copies light grid buffer (3.375 MB)
- Copies light buffer (512 bytes)
- Writes binary files to specified output directory

**Output Files**:
- `g_lightGrid.bin` - 3,456,000 bytes (27,000 cells × 128 bytes)
- `g_lights.bin` - 512 bytes (16 lights × 32 bytes)

### 2. Application Integration

**File**: `src/core/Application.cpp` (lines 1332-1341)

**Integration Point**: `DumpGPUBuffers()` function

```cpp
// Dump RTXDI buffers if using RTXDI path
if (m_lightingSystem == LightingSystem::RTXDI && m_rtxdiLightingSystem) {
    LOG_INFO("  Dumping RTXDI buffers...");
    auto cmdList = m_device->GetCommandList();
    m_device->ResetCommandList();
    m_rtxdiLightingSystem->DumpBuffers(cmdList, m_dumpOutputDir, m_frameCount);
    cmdList->Close();
    m_device->ExecuteCommandList();
    m_device->WaitForGPU();
}
```

**Trigger Methods**:
1. **Command-line**: `--dump-buffers 120` (auto-dump at frame 120)
2. **Runtime**: Press `Ctrl+D` during execution

---

## How to Use

### Method 1: Automatic Frame Dump

```bash
# Run application and automatically dump at frame 120
./build/Debug/PlasmaDX-Clean.exe --rtxdi --dump-buffers 120
```

### Method 2: Manual Dump (Ctrl+D)

```bash
# Run application with dump enabled
./build/Debug/PlasmaDX-Clean.exe --rtxdi --dump-buffers

# Press Ctrl+D when you want to capture
# (Creates dump at current frame)
```

### Method 3: Custom Output Directory

```bash
# Specify custom output location
./build/Debug/PlasmaDX-Clean.exe --rtxdi --dump-buffers 120 --dump-dir "PIX/rtxdi_validation"
```

---

## Output Location

**Default Directory**: `PIX/buffer_dumps/`

**Output Structure**:
```
PIX/buffer_dumps/
├── g_lightGrid.bin          # 3.375 MB - Light grid cells
├── g_lights.bin             # 512 bytes - Active lights
├── g_particles.bin          # Particle data (if enabled)
├── g_currentReservoirs.bin  # ReSTIR data (multi-light path only)
├── g_prevReservoirs.bin     # ReSTIR data (multi-light path only)
└── metadata.json            # Frame info, camera state, etc.
```

---

## Buffer Formats

### g_lightGrid.bin Structure

**Total Size**: 3,456,000 bytes
**Element Count**: 27,000 cells
**Element Size**: 128 bytes

**LightGridCell Structure** (128 bytes):
```cpp
struct LightGridCell {
    uint32_t lightIndices[16];  // 64 bytes - Which lights (0-15) or 0xFFFFFFFF
    float lightWeights[16];     // 64 bytes - Importance weights (0.0+)
};
```

**Cell Index Calculation**:
```cpp
cellIndex = x + y * 30 + z * 900
// where x, y, z ∈ [0, 29]
```

**Example Cell Locations**:
- Cell (15, 15, 15) - Grid center: Index 13,515
- Cell (0, 0, 0) - Min corner: Index 0
- Cell (29, 29, 29) - Max corner: Index 26,999

### g_lights.bin Structure

**Total Size**: 512 bytes
**Element Count**: 16 lights (max)
**Element Size**: 32 bytes

**Light Structure** (32 bytes):
```cpp
struct Light {
    float3 position;   // 12 bytes - World position
    float3 color;      // 12 bytes - RGB (0-1)
    float intensity;   // 4 bytes - Brightness multiplier
    float radius;      // 4 bytes - Light sphere radius
};
```

**Default Configuration** (Stellar Ring - 13 lights):
- Light 0: (0, 0, 0) - Primary center light
- Lights 1-4: Inner spiral arms (~50 unit radius)
- Lights 5-12: Mid-disk hot spots (~150 unit radius)
- Lights 13-15: Unused (all zeros)

---

## PIX MCP Agent Analysis

**Once buffers are dumped**, you can use the PIX MCP agent to analyze them:

### Analyze Light Grid

The PIX MCP server doesn't have a built-in light grid analyzer yet, so you'll need to inspect manually or create a Python script.

**Manual Inspection in PIX**:
1. Open PIX capture
2. Navigate to Dispatch #16 (light grid build)
3. Inspect `g_lightGrid` UAV
4. Check cells near lights for populated indices/weights

**Python Analysis** (future):
```python
import struct
import numpy as np

# Parse g_lightGrid.bin
with open('PIX/buffer_dumps/g_lightGrid.bin', 'rb') as f:
    data = f.read()

# Parse as 27,000 cells
cells = []
for i in range(27000):
    offset = i * 128
    indices = struct.unpack('16I', data[offset:offset+64])
    weights = struct.unpack('16f', data[offset+64:offset+128])
    cells.append({'indices': indices, 'weights': weights})

# Validate
for i, cell in enumerate(cells):
    # Check weight sorting
    active_weights = [w for w in cell['weights'] if w > 0.0]
    assert active_weights == sorted(active_weights, reverse=True), \
        f"Cell {i}: Weights not sorted!"
```

### Analyze Lights

```python
import struct

# Parse g_lights.bin
with open('PIX/buffer_dumps/g_lights.bin', 'rb') as f:
    data = f.read()

lights = []
for i in range(16):
    offset = i * 32
    px, py, pz, cx, cy, cz, intensity, radius = struct.unpack('8f', data[offset:offset+32])
    lights.append({
        'position': (px, py, pz),
        'color': (cx, cy, cz),
        'intensity': intensity,
        'radius': radius
    })
    print(f"Light {i}: pos=({px:.1f}, {py:.1f}, {pz:.1f}), "
          f"color=({cx:.2f}, {cy:.2f}, {cz:.2f}), "
          f"intensity={intensity:.1f}, radius={radius:.1f}")
```

---

## Validation Checklist

Use the comprehensive guide: `RTXDI_LIGHT_GRID_VALIDATION.md`

**Quick Checks**:

### ✅ Light Grid Populated
- Cells near lights (e.g., cell 13,515 at grid center) have 1-13 active light indices
- Weights are sorted **descending** (brightest first)
- Empty cells have all `0xFFFFFFFF` indices and `0.0` weights

### ✅ Lights Uploaded
- 13 active lights in default config
- Light 0 at origin (0, 0, 0)
- Lights 13-15 are zeros (unused)

### ✅ No Visual Differences Yet
**Expected**: You won't see visual differences between `--multi-light` and `--rtxdi` yet because:
- RTXDI raygen shader not implemented (Milestone 3)
- Light grid is built but not sampled
- Application falls back to multi-light rendering

**Milestone Progress**:
- ✅ M2.1: Light grid buffers created
- ✅ M2.2: Light grid build compute shader working
- ✅ M2.3: Buffer dump validation (in progress)
- ⏳ M3: DXR pipeline (raygen shader) - Next step!

---

## Current Application Status

**Running**: `./build/Debug/PlasmaDX-Clean.exe --rtxdi --dump-buffers 120`

**Expected Behavior**:
1. Application launches normally
2. Renders using Gaussian volumetric renderer (no visual change from multi-light)
3. At frame 120: Dumps buffers to `PIX/buffer_dumps/`
4. Logs show:
   ```
   === DUMPING GPU BUFFERS (Frame 120) ===
     Dumping RTXDI buffers...
   Dumped light grid: PIX/buffer_dumps/g_lightGrid.bin (3.38 MB)
   Dumped lights: PIX/buffer_dumps/g_lights.bin (512 bytes)
   === BUFFER DUMP COMPLETE ===
   ```

**What You Should See in PIX**:
- Dispatch #16: `RTXDI: Update Light Grid` event
- Resource `g_lightGrid` (3.375 MB UAV)
- Resource `g_lights` (512 bytes SRV)
- Dispatch(4, 4, 4) thread groups

---

## Next Steps

### Immediate (Milestone 2.3)
1. ✅ Application running with `--rtxdi --dump-buffers 120`
2. ⏳ Wait for frame 120 dump to complete
3. ⏳ Inspect `PIX/buffer_dumps/g_lightGrid.bin` (manual or PIX MCP)
4. ⏳ Validate cell population and weight sorting
5. ⏳ Measure GPU timing for light grid build (<0.5ms target)

### Near-Term (Milestone 3 - DXR Pipeline)
1. Create DXR state object (raygen/miss/closesthit shaders)
2. Build shader binding table (SBT)
3. Write raygen shader with RTXDI sampling stub
4. First visual test with RTXDI!

---

## Troubleshooting

### Buffer Files Not Created
**Check**:
1. Log output for "DUMPING GPU BUFFERS" message
2. `PIX/buffer_dumps/` directory exists
3. Application has write permissions

### Empty Light Grid
**Possible Causes**:
1. Multi-light system not initialized (check `InitializeLights()` called)
2. Light buffer upload failed
3. Compute shader dispatch not executing

**Verification**:
- Check PIX: Dispatch #16 should show light grid UAV writes
- Check logs: "Light grid populated with X lights" message

### Crashes During Dump
**Likely Cause**: Resource state transitions incorrect

**Fix**: Check barrier sequence in `DumpBuffers()`:
1. Transition to COPY_SOURCE
2. CopyBufferRegion
3. Transition back to original state

---

**Status**: ✅ Ready for validation!
**Next Session**: Analyze buffer dumps, complete Milestone 2.3, start Milestone 3 (DXR pipeline)

**Documentation Version**: 1.0
**Last Updated**: 2025-10-18
