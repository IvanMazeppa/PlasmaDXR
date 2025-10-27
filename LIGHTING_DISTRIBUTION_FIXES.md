# Lighting Distribution & Physical Emission Fixes

**Project:** PlasmaDX-Clean
**Issue:** Grey dead zone particles and physical emission beam artifacts
**Date Started:** 2025-10-27
**Current Branch:** 0.10.9

---

## Problem Analysis

### Visual Issues Identified

**Screenshot Evidence:** `screenshots/screenshot_2025-10-27_20-02-20.bmp` and `20-02-54.bmp`

1. **Central Orange Sphere (~100-200 unit radius)**
   - Bright yellow/orange core where physical emission + particle-to-particle RT lighting is active
   - **Hard spherical boundary** - particles abruptly transition from lit to unlit
   - Physical emission creates beam-like directional artifacts (non-physical)

2. **Grey Dead Zone (200-1000 units)**
   - Particles outside the orange sphere receive **zero illumination**
   - Appear grey/dark because:
     - Physical emission doesn't reach them (local effect only)
     - Particle-to-particle RT lighting is out of range (`m_rtMaxDistance = 100.0f`)
     - Multi-lights are too far away (~1200 units at Fibonacci sphere preset)
   - No smooth falloff - binary lit/unlit state

3. **Physical Emission Artifacts**
   - Creates **directional beam structures** instead of omnidirectional glow
   - **Overpowering brightness** masks subtle RT lighting contributions
   - **Quantized stepping** as particles enter/leave emission range
   - Boosting physical emission (strength 5.0+) exacerbates beam artifacts

### Root Causes

#### 1. RT Lighting Range Too Small ⚠️ CRITICAL
```cpp
float m_rtMaxDistance = 100.0f;  // Application.h:124
```
- Particles only illuminate others within 100 units
- Accretion disk outer radius: 300 units
- **Most particles are outside each other's lighting range**
- Creates hard boundary at ~100-150 unit radius

#### 2. Physical Emission Overpowering ⚠️ HIGH
- Physical emission so strong in center that RT lighting is masked
- RT lighting only visible when emission "drastically lowered" (user quote)
- No distance-based falloff mixing

#### 3. Multi-Light Dead Zone ⚠️ HIGH
- Multi-lights positioned at 1200-unit radius (Sphere preset)
- Accretion disk is ~300 units
- **900-unit gap** with zero light coverage

#### 4. No Ambient/Fallback Illumination ⚠️ MEDIUM
- Particles go completely grey/black when outside all lighting ranges
- Physically unrealistic (space has scattered light, cosmic background)

---

## Phase 1: Quick Wins ✅ COMPLETED

**Status:** Implemented and merged (branch 0.10.9)

### Implementation

**1. Global Ambient Parameter**
```cpp
// Application.h:126
float m_rtMinAmbient = 0.05f;  // Global ambient term (0.0-0.2)
```
- Prevents particles from going completely black
- Default: 0.05 (subtle ambient glow)
- Range: 0.0-0.2 (adjustable via ImGui)
- Applied in shader: `particle_gaussian_raytrace.hlsl:623`

**2. RT Max Distance Slider**
```cpp
// Application.cpp:2500-2513
ImGui::SliderFloat("RT Max Distance", &m_rtMaxDistance, 50.0f, 400.0f, "%.0f units");
```
- Default: 100 units (original)
- Max: 400 units (user finds GPU sweet spot)
- Real-time adjustment without rebuild
- Automatically updates RTLightingSystem

**3. Struct Alignment Fixed**
- Added fields at **END** of constant buffer struct (not middle)
- Proper padding: `DirectX::XMFLOAT3 lightingPadding`
- Prevents TDR crashes from struct mismatch

### Results

✅ **Successes:**
- No more TDR crashes
- RT Max Distance adjustable up to 400 units
- Global ambient prevents pure black particles

⚠️ **Limitations:**
- Increasing RT distance + ambient enough to eliminate grey zone → scene becomes too dim
- Physical emission artifacts remain
- Hard sphere boundary still visible (just pushed outward)
- **User quote:** "i can find a combination of settings that eliminates the grey particles but then becomes too dim"

---

## Phase 2: Advanced Lighting Solutions 📋 PLANNED

### Solution 1: Distance-Based Emission Scaling ⭐ HIGH IMPACT

**Problem:** Physical emission is uniform across all particles, overpowering RT lighting.

**Solution:** Fade emission with distance from center.

```hlsl
// In particle_gaussian_raytrace.hlsl
float distFromCenter = length(particlePos);
float emissionScale = saturate(1.0 - distFromCenter / 500.0f);  // Fade out at 500 units
physicalEmission *= emissionScale;
```

**Parameters to Add:**
```cpp
// Application.h
float m_emissionFalloffRadius = 500.0f;  // Distance where emission fades to zero
float m_emissionFalloffPower = 1.0f;     // 1.0 = linear, 2.0 = quadratic, etc.
```

**Expected Result:**
- Center particles (0-200 units): Full emission
- Middle particles (200-400 units): Reduced emission, RT lighting visible
- Outer particles (400+ units): No emission, pure RT/multi-light

**Implementation Time:** 2-3 hours

---

### Solution 2: Adaptive Emission/RT Blending ⭐ HIGH IMPACT

**Problem:** Emission and RT lighting don't smoothly transition.

**Solution:** Blend between emission-dominant (center) and RT-dominant (edges).

```hlsl
// In particle_gaussian_raytrace.hlsl
float distFromCenter = length(particlePos);
float emissionToRTRatio = saturate(distFromCenter / 300.0f);  // 0 at center, 1 at edge

finalColor = lerp(
    physicalEmission * particleColor,           // Center: emission dominant
    rtLightingContribution * particleColor,     // Edge: RT lighting dominant
    emissionToRTRatio
);
```

**Parameters to Add:**
```cpp
// Application.h
float m_emissionRTBlendRadius = 300.0f;  // Distance where blend transitions
float m_emissionRTBlendSharpness = 1.0f; // 0.5 = soft, 2.0 = sharp
```

**Expected Result:**
- Smooth transition from emission-lit center to RT-lit edges
- No hard sphere boundary
- RT lighting visible throughout disk

**Implementation Time:** 3-4 hours

---

### Solution 3: Reposition Multi-Lights Closer ⭐ HIGH IMPACT, EASY

**Problem:** Multi-lights at 1200-unit radius create 900-unit dead zone.

**Solution:** Create new preset with lights at 400-600 unit radius.

```cpp
// Application.cpp
void Application::InitializeRTXDIMediumRingLights() {
    m_lights.clear();

    // 13 lights in Fibonacci sphere at 500-unit radius (vs 1200)
    const float radius = 500.0f;
    const int numLights = 13;
    const float phi = (1.0f + sqrtf(5.0f)) / 2.0f;  // Golden ratio

    for (int i = 0; i < numLights; i++) {
        float y = 1.0f - (2.0f * i) / (numLights - 1.0f);
        float radiusAtY = sqrtf(1.0f - y * y);
        float theta = 2.0f * 3.14159f * i / phi;

        ParticleRenderer_Gaussian::Light light;
        light.position = DirectX::XMFLOAT3(
            radiusAtY * cosf(theta) * radius,
            y * radius,
            radiusAtY * sinf(theta) * radius
        );
        light.color = DirectX::XMFLOAT3(1.0f, 0.95f, 0.9f);  // Warm white
        light.intensity = 10.0f;
        light.radius = 150.0f;  // Large radius for soft falloff
        m_lights.push_back(light);
    }

    LOG_INFO("Applied RTXDI Medium Ring preset (13 lights, 500-unit radius)");
}
```

**Expected Result:**
- Multi-lights fill the 200-600 unit gap
- Combined with RT lighting (0-200 units), full disk coverage
- No performance cost (same light count)

**Implementation Time:** 1-2 hours

---

### Solution 4: Increase Multi-Light Count ⭐ MEDIUM IMPACT

**Problem:** 13 lights may not provide enough coverage.

**Solution:** Increase to 20-24 lights (still well under 32-light hardware limit).

**Considerations:**
- Performance: +7-11 lights = ~20-30% cost increase
- Coverage: Much better spatial distribution
- RTXDI: Weighted sampling handles 20+ lights efficiently

**Expected Result:**
- Smoother lighting gradients
- Fewer "light cells" in RTXDI mode
- Better coverage of 400-800 unit range

**Implementation Time:** 1 hour (just modify light count in presets)

---

### Solution 5: Temporal Smoothing for RT Lighting ⭐ MEDIUM IMPACT, HARD

**Problem:** RT lighting has "stepping/jumping" artifacts (user quote).

**Solution:** Implement temporal accumulation (like RTXDI M5).

```cpp
// Add to ParticleRenderer_Gaussian.h
Microsoft::WRL::ComPtr<ID3D12Resource> m_rtLightingAccumulationBuffer;  // Ping
Microsoft::WRL::ComPtr<ID3D12Resource> m_rtLightingPreviousBuffer;       // Pong
```

```hlsl
// In particle_gaussian_raytrace.hlsl
float3 currentRTLight = /* current frame calculation */;
float3 previousRTLight = g_rtLightingPrevious[pixelPos].rgb;

// Temporal blend (0.1 = smooth, 0.5 = responsive)
float3 accumulatedRTLight = lerp(previousRTLight, currentRTLight, 0.1);
g_rtLightingCurrent[pixelPos] = float4(accumulatedRTLight, 1.0);
```

**Expected Result:**
- Smooth RT lighting without stepping artifacts
- Converges in ~8-16 frames (67-133ms @ 120 FPS)
- Similar to RTXDI M5 temporal smoothing

**Implementation Time:** 6-8 hours (complex, requires ping-pong buffers)

---

### Solution 6: Global Illumination Approximation ⭐ LOW IMPACT, EASY

**Problem:** No scattered/indirect light from bright regions.

**Solution:** Add simple ambient cube map or spherical harmonics.

```hlsl
// Sample ambient light from direction
float3 GetAmbientLight(float3 normal) {
    // Simple hemisphere approximation
    float up = saturate(normal.y);
    return lerp(
        float3(0.02, 0.02, 0.03),  // Bottom: cool blue
        float3(0.08, 0.08, 0.06),  // Top: warm yellow
        up
    );
}
```

**Expected Result:**
- Particles outside all lighting ranges get directional ambient
- More physically plausible than uniform ambient
- Very cheap to compute

**Implementation Time:** 2-3 hours

---

## Phase 3: Optimization & Polish 📋 FUTURE

### Solution 7: Adaptive RT Quality Based on Distance

**Problem:** RT lighting expensive at high distances.

**Solution:** Reduce ray count for distant particles.

```cpp
// In RT lighting system
uint raysPerParticle = (distFromCamera < 500.0f) ? 16 : 4;
```

**Expected Result:**
- Close particles: Full 16-ray quality
- Distant particles: 4-ray quality (75% faster)
- Imperceptible quality difference at distance

---

### Solution 8: Light Importance Sampling (RTXDI-Style)

**Problem:** Multi-lights uniformly weighted even when far away.

**Solution:** Weight lights by distance and occlusion.

```hlsl
float lightImportance = (lightIntensity * lightRadius) / (lightDist * lightDist);
// Use importance for weighted random selection
```

**Expected Result:**
- Closer, brighter lights contribute more
- Smoother lighting gradients
- Better use of limited light budget

---

## Recommended Implementation Order

### Week 1: Core Lighting Coverage
1. ✅ **Phase 1: Global Ambient + RT Distance Slider** (COMPLETE)
2. 🔄 **Solution 3: Reposition Multi-Lights Closer** (1-2 hours)
   - Immediate impact, low complexity
   - Creates "Medium Ring" preset at 500-unit radius
3. 🔄 **Solution 1: Distance-Based Emission Scaling** (2-3 hours)
   - Fixes physical emission overpowering RT lighting
   - Allows RT lighting to show through

### Week 2: Smooth Transitions
4. 🔄 **Solution 2: Adaptive Emission/RT Blending** (3-4 hours)
   - Eliminates hard sphere boundary
   - Smooth center-to-edge transition
5. 🔄 **Solution 4: Increase Multi-Light Count** (1 hour)
   - Try 20 lights instead of 13
   - Test performance impact

### Week 3: Advanced (Optional)
6. 🔄 **Solution 5: Temporal Smoothing** (6-8 hours)
   - Fixes stepping/jumping artifacts
   - Only if previous solutions insufficient
7. 🔄 **Solution 6: Global Illumination Approximation** (2-3 hours)
   - Polish pass for ambient lighting
   - Nice-to-have, not critical

---

## Testing Procedure

### Before Each Solution
1. Capture baseline screenshot (F2 key)
2. Note current FPS (display in window title)
3. Record RT Max Distance and Global Ambient values

### After Each Solution
1. Test at multiple RT distances: 100, 200, 300, 400 units
2. Test at multiple particle counts: 2K, 5K, 10K, 20K
3. Capture comparison screenshot
4. Verify no TDR crashes
5. Check for visual improvements:
   - Grey particles eliminated? ✅/❌
   - Hard sphere boundary gone? ✅/❌
   - Physical emission artifacts reduced? ✅/❌
   - RT lighting visible throughout disk? ✅/❌

### Performance Targets
- 10K particles: 120 FPS (maintain current)
- 20K particles: 80+ FPS (acceptable)
- 50K particles: 40+ FPS (stretch goal)

---

## Known Constraints

### GPU Limits (RTX 4060 Ti)
- **TDR Timeout:** ~2 seconds per frame max
- **RT Distance:** 400 units seems to be upper limit at 10K particles
- **BLAS Rebuild:** 2.1ms @ 10K particles (acceptable)

### Design Trade-offs
- **Emission vs RT:** Cannot have both at full strength everywhere
  - Center: Emission dominant (physically accurate)
  - Middle: Blend zone (smooth transition)
  - Edge: RT/multi-light dominant (external illumination)

### User Preferences (from conversation)
- Prefers **visual quality** over strict physical accuracy
- Wants **smooth gradients** not hard boundaries
- Needs **dynamic shadows** from physics-driven lights
- Values **performance** (120 FPS target)

---

## Implementation Notes

### Struct Alignment Rules (CRITICAL!)
When adding new parameters to `GaussianConstants` struct:

1. ✅ **ADD AT END** of struct (safest)
2. ✅ **ADD PADDING** to maintain 16-byte alignment
3. ❌ **NEVER INSERT IN MIDDLE** (breaks existing fields)
4. ✅ **MATCH C++ AND HLSL EXACTLY** (same order, same types)

**Example (correct):**
```cpp
// C++ (ParticleRenderer_Gaussian.h)
float godRayDensity;
float godRayStepMultiplier;
DirectX::XMFLOAT2 godRayPadding;

float rtMinAmbient;              // NEW - at end
DirectX::XMFLOAT3 lightingPadding;  // NEW - padding
```

```hlsl
// HLSL (particle_gaussian_raytrace.hlsl)
float godRayDensity;
float godRayStepMultiplier;
float2 godRayPadding;

float rtMinAmbient;              // NEW - matches C++
float3 lightingPadding;          // NEW - matches C++
```

### ImGui Best Practices
- Always add tooltips for technical parameters
- Use descriptive labels (not just variable names)
- Provide recommended values in tooltips
- Log changes to help debugging

---

## References

### Key Files
- `src/core/Application.h/cpp` - Main lighting parameters
- `src/particles/ParticleRenderer_Gaussian.h/cpp` - Gaussian renderer
- `shaders/particles/particle_gaussian_raytrace.hlsl` - Main volumetric shader
- `src/lighting/RTLightingSystem_RayQuery.h/cpp` - RT lighting system

### Related Issues
- Physical emission artifacts: See `screenshots/screenshot_2025-10-27_20-02-20.bmp`
- Grey dead zone: See `screenshots/screenshot_2025-10-27_20-02-54.bmp`
- TDR crashes: Fixed by proper struct alignment (Phase 1)

### Discussion History
- Initial analysis: 2025-10-27 conversation
- Phase 1 implementation: 2025-10-27 (this session)
- User feedback: "can find a combination that eliminates grey particles but becomes too dim"

---

**Last Updated:** 2025-10-27
**Status:** Phase 1 Complete, Phase 2 Planned
**Next Step:** Implement Solution 3 (Reposition Multi-Lights)
