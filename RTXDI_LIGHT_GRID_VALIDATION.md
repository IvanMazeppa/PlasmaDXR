# RTXDI Light Grid Validation Guide

**Branch**: 0.7.1
**Milestone**: 2.2 COMPLETE → 2.3 Validation
**Date**: 2025-10-18

---

## Overview

This document provides a comprehensive checklist for validating the RTXDI light grid implementation in PIX. Use this when inspecting buffer dumps or PIX captures to ensure the light grid build compute shader is working correctly.

---

## Quick Start: What to Look For in PIX

### 1. Light Grid Buffer (`g_lightGrid` - 3.375 MB)

**Buffer Specifications:**
- **Size**: 3,456,000 bytes (3.375 MB)
- **Element Count**: 27,000 cells
- **Element Size**: 128 bytes
- **Structure**: `LightGridCell` (16 uint32 light indices + 16 float weights)

**What to Inspect:**

#### A. Cells Near Lights (Expected: 1-13 active lights per cell)

Look for cells in the world bounds where lights are positioned. With the default "Stellar Ring" configuration (13 lights):

**Expected Light Positions** (from multi-light system):
- Primary light: (0, 0, 0)
- 4 inner spiral arms: radius ~50 units
- 8 mid-disk hot spots: radius ~150 units

**Cell Examples to Check:**

1. **Cell at origin** (15, 15, 15) - grid center:
   ```
   Cell Index: 13,515 (15 + 15*30 + 15*900)
   Expected: 1-4 lights (primary + nearby spiral arms)

   lightIndices[0-3]: 0, 1, 2, 3 (or similar)
   lightWeights[0-3]: Should be sorted descending
   lightIndices[4-15]: 0xFFFFFFFF (empty slots)
   ```

2. **Cell near inner spiral** (17, 15, 15):
   ```
   Cell Index: 13,517
   Expected: 2-6 lights (spiral arm + primary + neighbors)

   Weights should reflect distance/intensity:
   - Closest light: highest weight
   - Far lights: lower weights
   ```

3. **Cell in outer disk** (25, 15, 20):
   ```
   Cell Index: 18,025
   Expected: 0-3 lights (far from most lights)

   May have only distant lights with low weights
   ```

#### B. Empty Cells (Expected: lightIndices = 0xFFFFFFFF)

Cells far from all lights should be completely empty:

**Test Cells at Extreme Edges:**
- Cell (0, 0, 0): Index 0 - far corner
- Cell (29, 29, 29): Index 26,999 - opposite corner

**Expected:**
```
lightIndices[0-15]: 0xFFFFFFFF, 0xFFFFFFFF, ... (all empty)
lightWeights[0-15]: 0.0, 0.0, ... (all zero)
```

#### C. Weight Sorting Validation

**Critical Check**: Weights must be sorted in descending order (brightest light first).

For any cell with multiple lights:
```
lightWeights[0] >= lightWeights[1] >= lightWeights[2] >= ... >= lightWeights[15]
```

**Why This Matters**: RTXDI sampling prioritizes high-weight lights. Incorrect sorting = poor sampling quality.

---

## 2. Light Buffer (`g_lights` - 512 bytes)

**Buffer Specifications:**
- **Size**: 512 bytes
- **Element Count**: 16 lights (max)
- **Element Size**: 32 bytes
- **Structure**: `Light { float3 position, float3 color, float intensity, float radius }`

**What to Inspect:**

### Active Light Count

Check how many lights are actually populated (based on multi-light system):

**Default Config**: 13 lights (Stellar Ring preset)

### Light 0 (Primary - Center)
```
position: (0.0, 0.0, 0.0)
color: (1.0, 0.9, 0.8) - blue-white
intensity: 10.0
radius: 5.0
```

### Lights 1-4 (Inner Spiral Arms)
```
position: Radius ~50 units, 90° intervals
color: (1.0, 0.8, 0.6) - orange
intensity: 5.0
radius: 10.0
```

### Lights 5-12 (Mid-Disk Hot Spots)
```
position: Radius ~150 units, 45° intervals
color: (1.0, 0.7, 0.4) - yellow-orange
intensity: 2.0
radius: 15.0
```

### Lights 13-15 (Unused)
```
All fields should be 0.0 (inactive slots)
```

---

## 3. Grid Constants Buffer (32 bytes)

**Structure:**
```cpp
struct GridConstants {
    uint gridCellsX;        // 30
    uint gridCellsY;        // 30
    uint gridCellsZ;        // 30
    uint lightCount;        // 13 (default config)

    float worldMin;         // -300.0
    float worldMax;         // 300.0
    float cellSize;         // 20.0
    uint maxLightsPerCell;  // 16
};
```

**Validation:**
- Ensure constants match the specification above
- `lightCount` should match the number of active lights in `g_lights`

---

## 4. PIX Event Markers

**Expected Timeline Events:**

```
Frame 120
├─ RTXDI: Update Light Grid
│  ├─ Upload Lights (CPU→GPU)
│  ├─ Transition: Light Buffer (COPY_DEST → NON_PIXEL_SHADER_RESOURCE)
│  ├─ Transition: Light Grid (COMMON → UNORDERED_ACCESS)
│  ├─ Set Compute Root Signature
│  ├─ Set Compute PSO (light_grid_build_cs)
│  ├─ Set Grid Constants (b0)
│  ├─ Set Light Buffer SRV (t0)
│  ├─ Set Light Grid UAV (u0)
│  ├─ Dispatch(4, 4, 4) - 27,000 threads
│  ├─ UAV Barrier (Light Grid)
│  └─ Transition: Light Grid (UNORDERED_ACCESS → COMMON)
└─ [Continue with Gaussian Rendering...]
```

**What to Check:**
- ✅ All resource transitions present
- ✅ Dispatch has correct thread group count (4, 4, 4)
- ✅ UAV barrier after dispatch
- ✅ No D3D12 errors or warnings

---

## 5. Performance Metrics

**Target Performance** (from Milestone 2.2 estimate):
- Light grid build time: **<0.5ms**
- Memory bandwidth: ~7 GB/s (3.375 MB write @ 0.5ms)
- Compute occupancy: Should be high (512 threads/group)

**PIX Timing Counters to Check:**

### GPU Duration
```
RTXDI: Update Light Grid event
Expected: 0.2-0.5ms (may vary by GPU)
```

### Dispatch Statistics
```
Thread Groups Dispatched: 64 (4×4×4)
Threads per Group: 512 (8×8×8)
Total Threads: 32,768 (27,000 active + 5,768 idle)
Wave Size: 32 (NVIDIA) or 64 (AMD)
```

### Memory Access Patterns
```
UAV Writes: 27,000 cells × 128 bytes = 3.375 MB
SRV Reads: 13 lights × 32 bytes × 27,000 cells = ~10.6 MB total
(But early exit means not all cells read all lights)
```

---

## 6. Known Issues & Edge Cases

### Issue 1: Light Radius Too Small
**Symptom**: Most cells have 0 lights even though lights exist
**Cause**: Sphere-AABB intersection test fails (light radius < cell diagonal)
**Check**: Ensure `light.radius >= sqrt(3) * cellSize / 2 ≈ 17.3 units` for guaranteed intersection

### Issue 2: Weight Calculation Overflow
**Symptom**: NaN or Inf weights
**Cause**: Very high luminance × intensity values
**Check**: All weights should be finite positive values (0.0 to ~1000.0)

### Issue 3: Incorrect Cell Index Calculation
**Symptom**: Random pattern of populated cells
**Cause**: Thread ID → cell coordinate math error
**Expected Formula**:
```hlsl
uint3 cellCoords = dispatchThreadID.xyz;
uint cellIndex = cellCoords.x + cellCoords.y * 30 + cellCoords.z * 900;
```

### Issue 4: Empty Light Buffer
**Symptom**: All cells have 0 lights
**Cause**: Multi-light system not initialized or light upload failed
**Check**: Verify `Application.cpp` calls `InitializeLights()` and `UpdateLightGrid(m_lights)`

---

## 7. Validation Test Cases

### Test 1: Single Light at Origin
**Setup**: Disable all lights except primary (0, 0, 0)
**Expected**:
- Cell (15, 15, 15): 1 light, high weight
- Cells within ~3-5 cell radius: 1 light, decreasing weights
- Cells beyond radius: 0 lights

### Test 2: 13-Light Stellar Ring (Default)
**Setup**: Default configuration
**Expected**:
- Inner cells (near origin): 1-4 lights
- Mid-range cells (50-150 units): 2-8 lights
- Outer cells (>200 units): 0-3 lights
- No cell should have all 16 slots filled (unless you add more lights!)

### Test 3: Light Movement
**Setup**: Move a light 100 units in X direction
**Expected**:
- Old cells lose that light (index removed)
- New cells gain that light (index added)
- Weights recalculated based on new distance

### Test 4: Extreme Light Count (16 lights max)
**Setup**: Add 3 more lights (total 16)
**Expected**:
- Some cells may fill all 16 slots
- Sorting still correct (brightest first)
- No crashes or buffer overruns

---

## 8. Automated Validation (Future: PIX MCP Agent)

**When the PIX MCP agent is working**, you can automate these checks:

```python
# Pseudo-code for automated validation
def validate_light_grid(buffer_path):
    grid = parse_buffer(buffer_path, LightGridCell, 27000)

    # Check 1: Empty cells are truly empty
    for cell in grid:
        if all(idx == 0xFFFFFFFF for idx in cell.lightIndices):
            assert all(w == 0.0 for w in cell.lightWeights)

    # Check 2: Weights are sorted
    for cell in grid:
        active = [w for w in cell.lightWeights if w > 0.0]
        assert active == sorted(active, reverse=True)

    # Check 3: Light indices are valid
    for cell in grid:
        for idx in cell.lightIndices:
            if idx != 0xFFFFFFFF:
                assert 0 <= idx < 16

    return "PASS"
```

---

## 9. Quick Reference: Buffer Dump Commands

### Capture Buffers at Frame 120
```bash
./build/Debug/PlasmaDX-Clean.exe --rtxdi --dump-buffers 120
```

**Output Location**:
```
PIX/buffer_dumps/frame_120_<timestamp>/
├── metadata.json
├── g_particles.bin (if using particles)
├── g_lights.bin (512 bytes)
└── g_lightGrid.bin (3.375 MB) ← **THIS IS KEY**
```

### PIX Capture (Full GPU Timeline)
```bash
./build/DebugPIX/PlasmaDX-Clean-PIX.exe --rtxdi
# Use PIX GUI to capture frame 120
```

---

## 10. Success Criteria Summary

**Milestone 2.2 is VALIDATED when:**

- ✅ Light buffer contains 13 active lights (default config)
- ✅ Light grid cells near lights have 1-13 populated indices
- ✅ Light grid cells far from lights are empty (0xFFFFFFFF)
- ✅ Weights are sorted descending in every cell
- ✅ No NaN/Inf values in weights
- ✅ GPU timing <0.5ms for grid build
- ✅ No D3D12 errors or validation warnings
- ✅ Application runs without crashes with `--rtxdi` flag

**Milestone 2.3 (Next):**
- Performance profiling (measure exact GPU time)
- Buffer dump analysis (Python script validation)
- Compare with expected distributions (statistics)

---

## 11. PIX Screenshot Checklist

**When inspecting in PIX, capture screenshots of:**

1. **Resource Inspector** - Light grid buffer view
   - Show hex dump of cells near (0,0,0)
   - Show hex dump of cells at grid edges (empty)

2. **Timeline** - "RTXDI: Update Light Grid" event
   - Show dispatch statistics
   - Show GPU duration
   - Show resource transitions

3. **Pipeline State** - Light grid build compute shader
   - Root signature
   - Bound resources (SRV/UAV)
   - Thread group dimensions

4. **Memory View** - Light buffer contents
   - Show all 13 active lights
   - Verify positions/colors/intensities

---

**Document Version**: 1.0
**Created**: 2025-10-18
**Status**: Ready for validation
**Next**: Milestone 2.3 - Automated analysis with PIX MCP agent
