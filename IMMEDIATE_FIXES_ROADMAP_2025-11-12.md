# Immediate Fixes Roadmap - Material System
**Date:** 2025-11-12 22:30
**Priority Order:** Temporal Flashing → Material Diversity → Temperature Gradient
**Estimated Total Time:** 4-6 hours

---

## PRIORITY 1: FIX TEMPORAL FLASHING (CRITICAL)
**Time:** 2-3 hours | **Impact:** Eliminates strobing, restores smooth volumetric appearance

### Problem
Particles rapidly flicker between bright (fully lit) and dark (fully shadowed) states. RayQuery inline RT lighting samples new random light/shadow each frame without temporal history.

### Solution: Add Temporal Accumulation

**File:** `shaders/particles/particle_gaussian_raytrace.hlsl`
**Location:** ~line 1400-1500 (RT lighting calculation)

**Current Code (BROKEN):**
```hlsl
// Overwrites buffer every frame - causes flashing
float3 rtLighting = ComputeRTLighting(particlePos, ...);
g_rtLighting[particleID] = float4(rtLighting, 1.0);
```

**Fixed Code (ADD THIS):**
```hlsl
// Compute current frame lighting
float3 currentLight = ComputeRTLighting(particlePos, lightPos, ...);

// Read previous frame lighting
float3 prevLight = g_rtLighting[particleID].rgb;

// Exponential moving average: 10% new, 90% history
float blendFactor = 0.1;
float3 blendedLight = lerp(prevLight, currentLight, blendFactor);

// Write blended result
g_rtLighting[particleID] = float4(blendedLight, 1.0);
```

### Build & Test
```bash
# Recompile shaders
"/mnt/c/Program Files/Microsoft Visual Studio/2022/Community/MSBuild/Current/Bin/MSBuild.exe" \
  build/CompileShaders.vcxproj /p:Configuration=Debug /p:Platform=x64 /t:Build

# Test
cd build/bin/Debug
./PlasmaDX-Clean.exe  # Press F2 to capture

# Compare with agent
/mcp dxr-image-quality-analyst compare_screenshots_ml \
  --before_path "screenshot_before_temporal.bmp" \
  --after_path "screenshot_after_temporal.bmp" \
  --save_heatmap true
```

### Success Criteria
- ✅ No visible flashing/strobing
- ✅ Smooth illumination transitions
- ✅ Particles maintain consistent brightness frame-to-frame
- ✅ Blocky appearance resolves (likely secondary effect)
- ✅ Agent quality score improves: Temporal Stability 10/100 → 80+/100

### Expected Convergence Time
- ~100ms to stabilize (12 frames @ 120 FPS)
- ~67ms with temporal blend 0.1 (8 frames @ 120 FPS)

---

## PRIORITY 2: DEBUG MATERIAL DIVERSITY (HIGH)
**Time:** 1-2 hours | **Impact:** Makes 5 material types visually distinct

### Problem
All particles uniform brown color. No brightness variation, no wispy gas clouds, no bright stars.

### Diagnostic Steps

**Step 1: Check Material Assignment Distribution**

**File:** `src/particles/ParticleSystem.cpp` (particle initialization)

**Add Logging:**
```cpp
// After particle initialization loop
std::unordered_map<uint32_t, int> materialCounts;
for (const auto& particle : m_particles) {
    materialCounts[particle.materialType]++;
}

LOG_INFO("=== Material Distribution ===");
for (const auto& [type, count] : materialCounts) {
    const char* names[] = {"PLASMA", "STAR", "GAS_CLOUD", "ROCKY", "ICY"};
    LOG_INFO("  Type {} ({}): {} particles", type, names[type], count);
}
```

**Expected Output:**
```
Type 0 (PLASMA): 2000 particles
Type 1 (STAR): 2000 particles
Type 2 (GAS_CLOUD): 2000 particles
Type 3 (ROCKY): 2000 particles
Type 4 (ICY): 2000 particles
```

**If all Type 0:** Need to implement heterogeneous material assignment

---

**Step 2: Add Debug Visualization**

**File:** `shaders/particles/particle_gaussian_raytrace.hlsl` (~line 1162)

**Add Color Override:**
```hlsl
// After material lookup
MaterialTypeProperties mat = g_materials[p.materialType];

// DEBUG: Force color by material type (REMOVE AFTER TESTING)
#define DEBUG_MATERIAL_COLORS 1
#if DEBUG_MATERIAL_COLORS
    if (p.materialType == 0) {      // PLASMA
        emission = float3(1.0, 0.4, 0.1);  // Orange
    } else if (p.materialType == 1) {  // STAR
        emission = float3(1.0, 1.0, 0.0);  // Yellow
    } else if (p.materialType == 2) {  // GAS_CLOUD
        emission = float3(0.0, 0.5, 1.0);  // Blue
    } else if (p.materialType == 3) {  // ROCKY
        emission = float3(0.3, 0.3, 0.3);  // Grey
    } else if (p.materialType == 4) {  // ICY
        emission = float3(0.9, 0.9, 1.0);  // White-blue
    }
#else
    // Normal material-aware emission
    emission = ComputeMaterialEmission(p, mat, ...);
#endif
```

**Build, Test, Capture:** Should see 5 distinct color bands if materials assigned correctly

---

**Step 3: Verify Material Buffer Binding**

```bash
# Check logs for null buffer warning
grep -i "material.*null" build/bin/Debug/logs/*.log

# Expected: No matches (buffer should be valid)
# If found: Material buffer creation failed, check ParticleSystem::Initialize()
```

---

**Step 4: PIX GPU Capture (If Still Broken)**

```bash
cd build/bin/Debug
./PlasmaDX-Clean.exe --config=configs/agents/pix_agent.json
```

**In PIX:**
1. Find "particle_gaussian_raytrace" compute shader dispatch
2. Check root parameter 12 (b1) - should show 320-byte material buffer
3. Inspect buffer contents - should see 5 material entries with distinct values
4. If buffer is null or wrong size: Fix binding in `ParticleRenderer_Gaussian.cpp:839-845`

### Success Criteria
- ✅ Logs show heterogeneous material distribution
- ✅ Debug visualization shows 5 color bands
- ✅ PIX confirms material buffer bound at b1
- ✅ Each material type visually distinct
- ✅ Agent quality score improves: Temperature Gradient 15/100 → 70+/100

---

## PRIORITY 3: RESTORE TEMPERATURE GRADIENT (MEDIUM)
**Time:** 1 hour | **Impact:** Hot white/blue core → cool orange/red edges

### Problem
No color variation across disk. Should have hot inner core (white/blue) and cool outer edges (orange/red).

### Diagnostic Steps

**Step 1: Verify Temperature Initialization**

**File:** `src/particles/ParticleSystem.cpp` (InitializeParticles or equivalent)

**Check Temperature Calculation:**
```cpp
// Temperature should vary with radius
float radius = length(position);
float temperature = CalculateTemperatureFromRadius(radius);

// Should range from ~800K (outer) to ~26000K (inner)
// Add logging:
static float minTemp = FLT_MAX, maxTemp = 0.0f;
minTemp = std::min(minTemp, temperature);
maxTemp = std::max(maxTemp, temperature);

LOG_INFO("Temperature range: {:.0f} K to {:.0f} K", minTemp, maxTemp);
```

**Expected:** 800K to 26000K range

---

**Step 2: Reduce Material Albedo Blend**

**File:** `shaders/particles/particle_gaussian_raytrace.hlsl` (~line 1260)

**Current (Too Strong):**
```hlsl
// 50% material albedo blend - overpowers temperature colors
emission = lerp(temperatureColor, mat.albedo, 0.5);
```

**Fixed (Preserve Temperature):**
```hlsl
// 20% material albedo blend - tints but preserves temperature gradient
emission = lerp(temperatureColor, mat.albedo, 0.2);
```

---

**Step 3: Enable Physical Emission (If Disabled)**

**Check Metadata:** `screenshot_*.bmp.json`
```json
{
  "physical_effects": {
    "physical_emission": {
      "enabled": false,  // ← If this is false, need to enable
      "strength": 0.00
    }
  }
}
```

**If Disabled:**
- Enable via ImGui: Check "Use Physical Emission"
- Or config: Set `physicalEmissionEnabled: true` in config JSON
- Set strength: `physicalEmissionStrength: 1.0`

### Success Criteria
- ✅ Hot white/blue particles in inner disk (r < 50 units)
- ✅ Cool orange/red particles in outer disk (r > 200 units)
- ✅ Smooth gradient transition
- ✅ Material colors tint but don't overwhelm temperature
- ✅ Agent quality score improves: Temperature Gradient 15/100 → 85+/100

---

## TESTING WORKFLOW (AFTER EACH FIX)

### 1. Build
```bash
"/mnt/c/Program Files/Microsoft Visual Studio/2022/Community/MSBuild/Current/Bin/MSBuild.exe" \
  build/PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64 /t:Build /nologo /v:minimal
```

### 2. Test
```bash
cd build/bin/Debug
./PlasmaDX-Clean.exe  # Press F2 to capture screenshot
```

### 3. Agent Visual Assessment
```bash
/mcp dxr-image-quality-analyst assess_visual_quality \
  --screenshot_path "build/bin/Debug/screenshots/screenshot_latest.bmp"
```

### 4. ML Before/After Comparison
```bash
/mcp dxr-image-quality-analyst compare_screenshots_ml \
  --before_path "screenshot_before.bmp" \
  --after_path "screenshot_after.bmp" \
  --save_heatmap true
```

**LPIPS Score Interpretation:**
- 0.00-0.05: Imperceptible changes
- 0.05-0.15: Subtle improvements
- 0.15-0.30: Noticeable differences (expected after temporal fix)
- 0.30+: Major visual changes

### 5. Review Agent Feedback
- Check 7 quality dimension scores
- Read "ACTIONABLE RECOMMENDATIONS" section
- Compare scores before/after fix
- Review difference heatmap for hotspots

---

## QUICK REFERENCE: KEY FILE LOCATIONS

### Shader Files
- **Main Renderer:** `shaders/particles/particle_gaussian_raytrace.hlsl`
  - Line ~1162: Material lookup
  - Line ~1230: Physical emission blending
  - Line ~1260: Artistic emission blending
  - Line ~1400-1500: RT lighting calculation (ADD TEMPORAL ACCUMULATION HERE)
  - Line ~1515: Scattering color

### C++ Files
- **Particle System:** `src/particles/ParticleSystem.cpp`
  - Particle initialization (check material assignment)
  - Temperature calculation
  - Material properties initialization (line 623)

- **Gaussian Renderer:** `src/particles/ParticleRenderer_Gaussian.cpp`
  - Root signature (line 526-542)
  - Material buffer binding (line 839-845)

### Logs & Screenshots
- **Logs:** `build/bin/Debug/logs/PlasmaDX-Clean_*.log`
- **Screenshots:** `build/bin/Debug/screenshots/screenshot_*.bmp`
- **Metadata:** `build/bin/Debug/screenshots/screenshot_*.bmp.json`

---

## EXPECTED FINAL STATE

**Visual Quality:**
- ✅ Smooth volumetric particles (no flashing)
- ✅ 5 distinct material types:
  - Orange plasma (default)
  - Brilliant yellow stars (8× brighter)
  - Wispy blue gas clouds (backward scatter)
  - Dark grey rocky bodies
  - Bright white icy particles
- ✅ Temperature gradient: hot white/blue core → cool orange/red edges
- ✅ Rim lighting from 13 lights
- ✅ Atmospheric depth and scattering

**Performance:**
- ✅ 90-120 FPS @ 10K particles
- ✅ < 10ms frame time
- ✅ Smooth temporal convergence (<100ms)

**Agent Scores (Target):**
- Volumetric Depth: 60/100 → 85+/100
- Lighting Quality: 40/100 → 80+/100
- Temperature Gradient: 15/100 → 85+/100
- Temporal Stability: 10/100 → 90+/100
- **Overall Grade: C+ (70/100) → A- (85+/100)**

---

**Document Status:** Complete
**Next Action:** Start with Priority 1 (Temporal Flashing)
**Total Estimated Time:** 4-6 hours for all 3 priorities
