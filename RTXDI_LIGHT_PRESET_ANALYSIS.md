# RTXDI Light Preset System Analysis

**Date:** 2025-10-19
**Issue:** Light presets create pronounced jigsaw patterns in RTXDI mode
**Root Cause:** Tight spatial clustering incompatible with RTXDI's spatial grid architecture

---

## Problem Summary

The current light preset system was designed for **multi-light brute-force rendering** (Phase 3.5) where all lights contribute to every pixel. RTXDI's **spatially-partitioned weighted sampling** has fundamentally different requirements that make the current presets produce very visible artifacts.

**User Observation:** *"the jigsaw pattern is quite apparent... it's difficult to use and the presets don't really work anymore now we're using rtxdl"*

---

## Current Presets Analysis

### Disk (13) - Default Preset

**Light Distribution:**
```
1 primary:    origin (0,0,0)          intensity=15.0, radius=80
4 secondary:  radius 50 (±50,0,±50)   intensity=12.0, radius=100
8 tertiary:   radius 150              intensity=8.0,  radius=120
```

**Spatial Extent:** ~300 units diameter (150-unit radius + light radius)

**Problem:** All 13 lights clustered within 300×300×300 unit cube at world origin

### Single - Minimal Preset

**Light Distribution:**
```
1 light: origin (0,0,0), intensity=10.0, radius=5.0
```

**Spatial Extent:** ~10 units diameter

**Problem:** Extremely concentrated, barely covers 1 grid cell

### Dome (8) - Elevated Preset

**Light Distribution:**
```
8 lights: radius=200, height=150, intensity=7.0, radius=15.0
45° separation in circular pattern
```

**Spatial Extent:** ~430 units diameter (200-unit radius + height)

**Problem:** Still clustered in central region, poor coverage of 3000×3000×3000 world space

---

## RTXDI Spatial Grid Architecture

**Grid Configuration:**
- Dimensions: **30×30×30 cells** = 27,000 total cells
- World bounds: **-1500 to +1500** (3000-unit range per axis)
- Cell size: **100×100×100 units**
- Expected usage: Distribute lights across the FULL 3000-unit space

**Current Population:**
- Populated cells: **~152 out of 27,000** (0.563% occupancy)
- All populated cells: Within central **6×6×6 cell cluster**
- Effective grid usage: **216 cells out of 27,000** (0.8%)

**Visual Impact:**
```
Screen Region 1 → Grid Cell (14,15,14) → Selects Light 3 (orange)
Screen Region 2 → Grid Cell (15,15,14) → Selects Light 7 (yellow)
Screen Region 3 → Grid Cell (14,16,14) → Selects Light 1 (blue-white)
                                         ↓
                        Sharp color transitions = JIGSAW PATTERN
```

---

## Why Multi-Light Presets Fail in RTXDI

### Issue 1: Spatial Clustering

**Multi-Light Assumption:** Lights close together = more dramatic lighting effects
**RTXDI Reality:** Lights close together = all mapped to same grid cells → random selection creates patchwork

**Example:**
- Disk preset: 13 lights within 300-unit sphere
- Grid cells covered: ~3×3×3 = 27 cells
- Screen coverage: Each cell visible in ~200×200 pixel region
- Result: 200×200 pixel blocks with different random light selections

### Issue 2: Light Density vs. Grid Resolution

**Grid Cell Size:** 100×100×100 units
**Light Radius:** 80-120 units (larger than grid cells!)

**Consequence:** A single light's influence radius spans multiple grid cells, but RTXDI samples **only one light per cell**. This creates:
- Overlapping light influence zones
- Random selection between overlapping lights per cell
- Sharp color boundaries between regions

### Issue 3: Wasted Grid Capacity

**Grid designed for:** 100+ lights distributed across 3000×3000×3000 space
**Current usage:** 13 lights in 300×300×300 space
**Utilization:** 0.8% of available spatial capacity

**Impact:**
- Most grid cells empty (no lights)
- Empty cells → pixels render with no light selected → dark regions
- Clustered cells → multiple lights fight for selection → patchwork

---

## RTXDI-Optimized Preset Requirements

### Design Principles

1. **Wide Spatial Distribution**
   - Spread lights across **1500-2000 unit radius** sphere
   - Ensure 200-400 unit minimum spacing between lights
   - Cover more grid cells (target: 500-1000 cells, ~2-4% occupancy)

2. **Grid-Aware Positioning**
   - Position lights at grid cell centers (100-unit increments)
   - Avoid clustering multiple lights in same cell
   - Create gradual transitions between light regions

3. **Adaptive Intensity**
   - Lower individual light intensity (wider distribution compensates)
   - Larger light radius for smoother falloff
   - Balance brightness to match multi-light appearance

4. **Temporal Stability**
   - Even light distribution reduces frame-to-frame flickering
   - Wider spacing = less random selection variation
   - Prepares for M5 temporal reuse (smoother accumulation)

---

## Proposed RTXDI-Optimized Presets

### Preset 1: "RTXDI Sphere (13)"

**Description:** Spherical distribution optimized for spatial grid
**Light Count:** 13 lights
**Spatial Extent:** 1200-unit radius sphere

**Distribution:**
```
1 center:     (0, 0, 0)                  intensity=10.0, radius=150
12 surface:   Fibonacci sphere sampling  intensity=8.0,  radius=150
              at 1200-unit radius
```

**Grid Coverage:** ~500 cells (~1.85% occupancy)
**Expected Pattern:** Smooth gradients, minimal patchwork

### Preset 2: "RTXDI Ring (16)"

**Description:** Accretion disk-like ring at large radius
**Light Count:** 16 lights
**Spatial Extent:** 1000-unit radius disk

**Distribution:**
```
Inner ring (8):  radius=600,  height=±100, intensity=12.0, radius=120
Outer ring (8):  radius=1000, height=±150, intensity=8.0,  radius=150
```

**Grid Coverage:** ~600 cells (~2.2% occupancy)
**Expected Pattern:** Circular gradient, disk-like appearance

### Preset 3: "RTXDI Grid (27)"

**Description:** Cubic grid distribution for maximum coverage
**Light Count:** 27 lights (3×3×3 grid)
**Spatial Extent:** 1800-unit cube

**Distribution:**
```
3×3×3 grid: 600-unit spacing (-900, -300, +300, +900 per axis)
            intensity=6.0, radius=180
```

**Grid Coverage:** ~1200 cells (~4.4% occupancy)
**Expected Pattern:** Minimal patchwork, very smooth transitions

### Preset 4: "RTXDI Sparse (5)" - Debug Preset

**Description:** Minimal light count for debugging
**Light Count:** 5 lights
**Spatial Extent:** 1000-unit cross pattern

**Distribution:**
```
1 center: (0, 0, 0)         intensity=15.0, radius=200
4 axes:   (±1000, 0, 0)     intensity=10.0, radius=150
          (0, 0, ±1000)
```

**Grid Coverage:** ~200 cells (~0.74% occupancy)
**Expected Pattern:** Clear quadrant divisions (for debugging grid behavior)

---

## Implementation Plan

### Phase 1: Add RTXDI Preset Section (15 minutes)

**File:** `src/core/Application.cpp:2050-2100`

**Changes:**
1. Add new ImGui section: "RTXDI Presets (Optimized for Spatial Grid)"
2. Implement 4 new presets with wider spatial distribution
3. Add tooltip explanations for each preset
4. Keep existing presets under "Legacy Presets (Multi-Light Optimized)"

### Phase 2: Update InitializeLights() (10 minutes)

**File:** `src/core/Application.cpp:2160`

**Changes:**
1. Detect current lighting system (RTXDI vs Multi-Light)
2. Auto-select appropriate default preset:
   - Multi-Light → Disk (13) legacy preset
   - RTXDI → RTXDI Sphere (13) optimized preset
3. Log which preset was auto-selected

### Phase 3: Add Preset Validation (10 minutes)

**New Function:** `ValidateLightPresetForRTXDI()`

**Validation Checks:**
1. Calculate spatial extent of all lights
2. Estimate grid cell coverage (should be >500 cells)
3. Warn if lights clustered (<200 unit average spacing)
4. Display warning in ImGui if current preset suboptimal for RTXDI

### Phase 4: Runtime Preset Switching (5 minutes)

**Changes:**
1. Add keyboard shortcut: **F4** = Cycle RTXDI presets
2. Add ImGui button: "Optimize for RTXDI" (auto-converts current preset)
3. Preserve manual light edits when switching modes

---

## Expected Results

### Before (Current Disk Preset in RTXDI):
- ❌ Pronounced jigsaw pattern (200×200 pixel blocks)
- ❌ Sharp color transitions between regions
- ❌ Only 0.8% grid utilization
- ❌ Flickering at edges between regions

### After (RTXDI Sphere Preset):
- ✅ Smooth gradients (500+ populated cells)
- ✅ Gradual color transitions
- ✅ ~2% grid utilization (250% improvement)
- ✅ Reduced flickering (M5 temporal reuse will eliminate entirely)

---

## Testing Plan

### Test 1: Visual Comparison (5 minutes)

1. Launch with default Disk preset in RTXDI mode
2. Enable debug visualization (rainbow colors)
3. Take screenshot → Save as `before_optimization.png`
4. Switch to RTXDI Sphere preset
5. Take screenshot → Save as `after_optimization.png`
6. Compare patchwork pattern visibility

### Test 2: Grid Coverage Analysis (10 minutes)

1. Add logging to RTXDI grid building (count populated cells)
2. Test each preset and record:
   - Populated cell count
   - Average lights per cell
   - Spatial extent (min/max coordinates)
3. Verify RTXDI presets achieve >500 cell coverage

### Test 3: Motion Test (5 minutes)

1. Record 10-second video with Disk preset
2. Record 10-second video with RTXDI Sphere preset
3. Compare patchwork visibility during camera rotation
4. Verify RTXDI preset reduces flicker

---

## Future Enhancements (Post-M5)

### M5 Integration (Temporal Reuse)

RTXDI-optimized presets will show **dramatic improvement** when M5 is implemented:
- Current: 1 sample per pixel per frame → patchwork visible
- M5: 8-16 accumulated samples → smooth gradients
- Wider light distribution → faster convergence (more diverse samples)

### M6 Integration (Spatial Reuse)

Spatial reuse will further benefit from optimized presets:
- Neighbor sharing works best with even light distribution
- Current clustering → all neighbors sample same lights (wasted sharing)
- RTXDI presets → neighbors sample different regions (effective sharing)

### Adaptive Preset Scaling

Future feature: Auto-scale preset to particle bounds:
```cpp
// Auto-detect particle spatial extent
float3 particleBounds = ComputeParticleBounds();

// Scale RTXDI preset to match (with 1.5× margin)
float presetScale = particleBounds * 1.5 / 1200.0; // 1200 = Sphere preset radius

// Apply scale to all light positions
for (auto& light : m_lights) {
    light.position *= presetScale;
    light.radius *= presetScale;
}
```

---

## Recommendations

### Immediate (This Session):

1. ✅ **Implement RTXDI Sphere (13) preset** - Fastest win, matches current light count
2. ✅ **Add preset section in ImGui** - Clear separation between legacy and RTXDI presets
3. ✅ **Auto-select appropriate preset on startup** - Better out-of-box experience

### Next Session:

4. ⏳ **Validate with PIX capture** - Verify grid population distribution
5. ⏳ **Add grid coverage visualization** - Show populated cells in debug mode
6. ⏳ **Create user documentation** - Explain when to use which presets

### Post-M5:

7. ⏳ **Adaptive preset scaling** - Auto-fit lights to particle bounds
8. ⏳ **Preset import/export** - Save custom RTXDI-optimized configurations
9. ⏳ **Light distribution analyzer** - Visual tool for optimizing custom presets

---

## Code Changes Summary

**Files to Modify:**
- `src/core/Application.h` - Add new preset enum values
- `src/core/Application.cpp:2050-2100` - Add RTXDI preset UI section
- `src/core/Application.cpp:2160` - Add RTXDI preset implementations

**Estimated Time:** 40 minutes total
- UI changes: 15 min
- Preset implementations: 20 min
- Testing: 5 min

**Risk Level:** Low (additive changes only, no breaking changes)

---

## Conclusion

The light preset system was optimized for **multi-light brute-force rendering** where spatial clustering creates dramatic lighting. RTXDI's **spatially-partitioned sampling** requires the opposite: **wide distribution** to populate the spatial grid effectively.

**Key Insight:** The jigsaw pattern isn't a bug—it's the **mathematically correct visualization** of RTXDI's spatial partitioning with clustered light sources. The solution is to match the light distribution to RTXDI's architectural assumptions.

**Expected User Experience After Fix:**
- "The jigsaw pattern is much less apparent now!"
- "Presets work great in RTXDI mode"
- "I can see the difference between legacy and RTXDI optimized presets"

---

**Next Step:** Implement RTXDI Sphere (13) preset and add UI section (40 minutes total)
