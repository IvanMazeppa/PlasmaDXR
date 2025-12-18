# fix: NanoVDB Volumetric Asset Integration Issues

**Created:** 2025-12-18
**Type:** Bug Fix / Enhancement
**Priority:** High
**Estimated Effort:** 4-6 hours across 4 phases

---

## Overview

Fix four interconnected issues with NanoVDB volumetric asset integration that prevent reliable loading, display, and animation of gas clouds, nebulae, and smoke effects in the PlasmaDX-Clean real-time RT engine.

---

## Problem Statement

The current NanoVDB implementation has the following critical issues:

| Issue | Symptom | Severity | Root Cause |
|-------|---------|----------|------------|
| **Crash on load** | Some .nvdb files crash the program | CRITICAL | Null pointer dereference when loading non-FLOAT grids in animation sequences |
| **Green color shift** | Smoke renders green instead of neutral grey | HIGH | Temperature-based emission applied to ALL volumes regardless of material type |
| **Scaling broken** | Grids appear tiny, scale controls ineffective | MEDIUM | UX issue - slider doesn't auto-apply; Blender exports are inherently small |
| **Animation stuck** | Animation frames don't advance | MEDIUM | Auto-play disabled by default; user must manually click Play button |

---

## Technical Analysis

### Issue 1: Crash on Load (CRITICAL)

**Location:** `src/rendering/NanoVDBSystem.cpp:964-978`

**Root Cause:** In `LoadAnimationSequence()`, the code assumes the first frame is always a FLOAT grid:

```cpp
const nanovdb::NanoGrid<float>* floatGrid = handle.grid<float>();
if (floatGrid) {
    auto worldBBox = floatGrid->worldBBox();  // CRASH if floatGrid is nullptr for HALF/FP16 grids
```

When loading HALF (9) or FP16 (15) grids (common in Blender exports), `handle.grid<float>()` returns nullptr, but the code still tries to access `worldBBox()`.

**Also affects:** Single-file loading can crash if grid type validation is bypassed due to exception handling gaps.

---

### Issue 2: Green Color Shift (HIGH)

**Location:** `shaders/volumetric/nanovdb_raymarch.hlsl:551`

**Root Cause:**
```hlsl
float3 emission = TemperatureToColor(density * 10000.0) * emissionStrength * density;
```

The `TemperatureToColor()` function (lines 363-368) maps all densities to an orange-to-blue blackbody curve:
- Low density: `float3(1.0, 0.3, 0.1)` - orange/red
- High density: `float3(0.8, 0.9, 1.0)` - blue/white

**Problem:** Smoke is NOT hot plasma. It's cold particulate matter that should scatter light (grey/white albedo) WITHOUT emitting colored light. The current shader treats ALL volumetrics as hot gas.

**Screenshot evidence:** The screenshot metadata shows green scene lighting `[0.109, 0.910, 0.136]`. Orange emission + additive blend with green lighting = green-shifted result.

---

### Issue 3: Scaling Tools Don't Work (MEDIUM)

**Location:** `src/core/Application.cpp:5392-5393`

**Root Cause:** UX friction - the scale slider requires clicking "Apply Scale" button:

```cpp
if (ImGui::Button("Apply Scale")) {
    m_nanoVDBSystem->ScaleGridBounds(gridScale);
    gridScale = 1.0f;  // Reset slider
}
```

Users adjust the slider but forget to click the button. Additionally, Blender exports are inherently small (1-6 units) while PlasmaDX uses 300+ unit world scale.

---

### Issue 4: Animation Not Advancing (MEDIUM)

**Location:** `src/rendering/NanoVDBSystem.cpp:1051`

**Root Cause:** Animation defaults to paused (`m_animPlaying = false`):

```cpp
void NanoVDBSystem::UpdateAnimation(float deltaTime) {
    if (!m_animPlaying || m_animFrames.empty() || m_animFPS <= 0.0f) {
        return;  // Early out - animation never advances
    }
```

User loads animation, sees "Frame 0/6" in ImGui, waits, nothing happens. Must manually click "Play" button.

---

## Proposed Solution

### Phase 1: Fix Critical Crash (30 min)

**File:** `src/rendering/NanoVDBSystem.cpp`

**Changes:**

1. **Add safe grid bounds extraction** (around line 966):
```cpp
// After loading handle, extract bounds safely for ANY grid type
const nanovdb::GridData* gridData = reinterpret_cast<const nanovdb::GridData*>(handle.data());
if (gridData) {
    auto worldBBox = gridData->mWorldBBox;  // Works for ALL grid types
    m_gridWorldMin = {
        static_cast<float>(worldBBox.mCoord[0][0]),
        static_cast<float>(worldBBox.mCoord[0][1]),
        static_cast<float>(worldBBox.mCoord[0][2])
    };
    m_gridWorldMax = {
        static_cast<float>(worldBBox.mCoord[1][0]),
        static_cast<float>(worldBBox.mCoord[1][1]),
        static_cast<float>(worldBBox.mCoord[1][2])
    };
}
```

2. **Add validation before grid access** in both `LoadFromFile()` and `LoadAnimationSequence()`:
```cpp
// Validate grid bounds are sensible (not zero, not inverted)
float sizeX = m_gridWorldMax.x - m_gridWorldMin.x;
float sizeY = m_gridWorldMax.y - m_gridWorldMin.y;
float sizeZ = m_gridWorldMax.z - m_gridWorldMin.z;

if (sizeX <= 0.0f || sizeY <= 0.0f || sizeZ <= 0.0f) {
    LOG_ERROR("[NanoVDB] Invalid grid bounds (zero or negative size)");
    return false;
}
```

**Files to modify:**
- `src/rendering/NanoVDBSystem.cpp:964-978` (animation sequence bounds extraction)
- `src/rendering/NanoVDBSystem.cpp:326-354` (single file bounds extraction - already safe, verify)

---

### Phase 2: Fix Green Color Shift (45 min)

**File:** `shaders/volumetric/nanovdb_raymarch.hlsl`

**Changes:**

1. **Add material type to constant buffer** (line 46):
```hlsl
// After existing constants:
uint materialType;  // 0=SMOKE (neutral), 1=FIRE (orange), 2=PLASMA (hot), 3=NEBULA (custom)
float3 albedo;      // Base color for scattering (smoke=white, nebula=purple)
float _pad2;        // Alignment padding
```

2. **Update emission calculation** (line 551):
```hlsl
// Replace:
// float3 emission = TemperatureToColor(density * 10000.0) * emissionStrength * density;

// With:
float3 emission;
if (materialType == 0) {  // SMOKE - no emission, pure scattering
    emission = float3(0.0, 0.0, 0.0);
} else if (materialType == 1) {  // FIRE - temperature-based
    emission = TemperatureToColor(density * 8000.0) * emissionStrength * density;
} else if (materialType == 2) {  // PLASMA - hot blue-white
    emission = TemperatureToColor(density * 20000.0) * emissionStrength * density;
} else {  // NEBULA - use custom albedo as emission tint
    emission = albedo * emissionStrength * density;
}
```

3. **Update scattering to use albedo** (line 508):
```hlsl
// Replace:
// float3 scattering = light.color * attenuation * phase * scatteringCoeff;

// With:
float3 scatterAlbedo = (materialType == 0) ? float3(0.9, 0.9, 0.9) : albedo;  // Smoke is grey-white
float3 scattering = light.color * scatterAlbedo * attenuation * phase * scatteringCoeff;
```

**Files to modify:**
- `shaders/volumetric/nanovdb_raymarch.hlsl:19-46` (constant buffer)
- `shaders/volumetric/nanovdb_raymarch.hlsl:497-513` (lighting)
- `shaders/volumetric/nanovdb_raymarch.hlsl:547-555` (emission)
- `src/rendering/NanoVDBSystem.h:48-80` (NanoVDBConstants struct - add materialType, albedo)
- `src/rendering/NanoVDBSystem.cpp:713-740` (Render() - populate new constants)
- `src/core/Application.cpp:5248-5290` (ImGui - add material type dropdown)

---

### Phase 3: Fix Scaling UX (20 min)

**File:** `src/core/Application.cpp`

**Changes:**

1. **Make scale slider live** (around line 5392):
```cpp
// Replace manual Apply button with live slider
static float gridScale = 1.0f;
static float lastAppliedScale = 1.0f;

if (ImGui::SliderFloat("Grid Scale", &gridScale, 1.0f, 500.0f, "%.1fx")) {
    // Only apply when slider actually changes
    float scaleFactor = gridScale / lastAppliedScale;
    m_nanoVDBSystem->ScaleGridBounds(scaleFactor);
    lastAppliedScale = gridScale;
}

ImGui::SameLine();
if (ImGui::Button("Reset")) {
    float resetFactor = 1.0f / lastAppliedScale;
    m_nanoVDBSystem->ScaleGridBounds(resetFactor);
    gridScale = 1.0f;
    lastAppliedScale = 1.0f;
}
```

2. **Auto-scale small Blender grids** in `NanoVDBSystem::LoadFromFile()` (around line 350):
```cpp
// After bounds size warning (line 353)
if (sizeX < 10.0f && sizeY < 10.0f && sizeZ < 10.0f) {
    LOG_WARN("[NanoVDB]   WARNING: Bounds are very small (<10 units)!");
    LOG_INFO("[NanoVDB]   Auto-scaling 100x for visibility...");
    ScaleGridBounds(100.0f);  // Auto-scale to ~270-522 units (reasonable for PlasmaDX)
}
```

**Files to modify:**
- `src/core/Application.cpp:5388-5410` (ImGui scale controls)
- `src/rendering/NanoVDBSystem.cpp:349-354` (auto-scale on load)

---

### Phase 4: Fix Animation Playback (15 min)

**File:** `src/rendering/NanoVDBSystem.cpp`

**Changes:**

1. **Auto-play animation on load** (around line 946):
```cpp
// In LoadAnimationSequence(), after successful frame loading (line 946):
if (loadedCount > 0) {
    LOG_INFO("[NanoVDB] Animation loaded: {} frames ({:.1f} MB total)", ...);

    // AUTO-START playback
    m_animPlaying = true;
    m_animCurrentFrame = 0;
    m_animAccumulator = 0.0f;
    LOG_INFO("[NanoVDB] Animation auto-started (FPS: {:.1f}, Loop: {})",
             m_animFPS, m_animLoop ? "ON" : "OFF");

    // Use first frame as current grid...
```

2. **Add visual indicator for pause state** in ImGui (Application.cpp ~5425):
```cpp
// After frame counter display
if (!m_nanoVDBSystem->IsAnimationPlaying() && m_nanoVDBSystem->GetAnimationFrameCount() > 1) {
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "[PAUSED]");
}
```

**Files to modify:**
- `src/rendering/NanoVDBSystem.cpp:946-996` (auto-play on load)
- `src/core/Application.cpp:5420-5440` (pause indicator)

---

## Acceptance Criteria

### Functional Requirements

- [ ] All .nvdb files from `assets/volumes/nanovdb_examples/` load without crashing
- [ ] HALF and FP16 grids (Blender exports) load and render correctly
- [ ] Smoke volumes render neutral grey/white (no color tint)
- [ ] Fire/plasma volumes render with appropriate orange/blue blackbody colors
- [ ] Grid scale slider applies changes in real-time
- [ ] Small Blender grids auto-scale to visible size on load
- [ ] Animation sequences auto-play on load
- [ ] Pause indicator visible when animation is paused

### Non-Functional Requirements

- [ ] No performance regression (maintain 90+ FPS @ 1080p with NanoVDB enabled)
- [ ] Existing procedural fog sphere still works
- [ ] ImGui controls remain responsive

### Quality Gates

- [ ] Build succeeds (MSBuild Debug configuration)
- [ ] Shader compiles (dxc nanovdb_raymarch.hlsl)
- [ ] No D3D12 validation errors in debug layer output
- [ ] Screenshot comparison: smoke should be grey, not green

---

## Test Plan

### Manual Testing

1. **Crash Test:**
   - Load `chimney_smoke` animation sequence (6 frames, HALF grids)
   - Load `bipolar_nebula` single frame (should not crash)
   - Load `cloud_pack` single frame (FLOAT grid baseline)

2. **Color Test:**
   - Load smoke volume, verify neutral grey appearance
   - Load fire volume (if available), verify orange emission
   - Take screenshot, compare with baseline

3. **Scale Test:**
   - Load small Blender grid, verify auto-scale message in log
   - Adjust scale slider, verify immediate visual change
   - Click Reset, verify grid returns to original scale

4. **Animation Test:**
   - Load animation sequence, verify auto-play starts
   - Click Pause, verify [PAUSED] indicator appears
   - Click Play, verify animation resumes

### Test Files

| File | Type | Expected Behavior |
|------|------|-------------------|
| `cloud_01_variant_0000_density_nozip.nvdb` | FLOAT | Load, display grey/white |
| `industrial_chimney_smoke_0050_density_nozip.nvdb` | FLOAT | Load, display neutral smoke |
| `fluid_data_0024_density_nozip.nvdb` | FLOAT | Load, display nebula |
| `chimney_smoke/*.nvdb` | Animation | Load 6 frames, auto-play |

---

## Dependencies & Risks

### Dependencies
- NanoVDB header (`external/nanovdb/NanoVDB.h`)
- PNanoVDB HLSL header (`shaders/volumetric/PNanoVDB.h`)
- DXC shader compiler for HLSL recompilation

### Risks

| Risk | Mitigation |
|------|------------|
| Shader constant buffer size increase causes root signature mismatch | Verify 256-byte alignment, test incrementally |
| Auto-scale factor (100x) may be wrong for some grids | Make configurable via ImGui, log applied scale |
| Material type enum may conflict with future additions | Reserve values 0-15, document in header |

---

## Implementation Order

1. **Phase 1 (Crash fix)** - Must be first, blocking for other testing
2. **Phase 2 (Color fix)** - Visual quality, can be tested independently
3. **Phase 3 (Scale UX)** - Quality of life, low risk
4. **Phase 4 (Animation)** - Quality of life, low risk

---

## References

### Internal
- `src/rendering/NanoVDBSystem.h:1-400` - System header with all method signatures
- `src/rendering/NanoVDBSystem.cpp:1-1075` - Full implementation
- `shaders/volumetric/nanovdb_raymarch.hlsl:1-649` - Ray marching shader
- `docs/claude_compound_prompts/NANOVDB_REFRACTOR_PROMPT_GPT52.md` - Detailed analysis document

### External
- [NanoVDB GitHub](https://github.com/AcademySoftwareFoundation/openvdb/tree/master/nanovdb) - Official NanoVDB documentation
- [PNanoVDB.h](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/PNanoVDB.h) - Portable NanoVDB header for HLSL

---

## Related Issues

- Log files analyzed:
  - `build/bin/Debug/logs/PlasmaDX-Clean_20251218_035405.log`
  - `build/bin/Debug/logs/PlasmaDX-Clean_20251218_040358.log`
  - `build/bin/Debug/logs/PlasmaDX-Clean_20251218_041243.log`
- Screenshot: `build/bin/Debug/screenshots/screenshot_2025-12-18_04-14-16.png`
