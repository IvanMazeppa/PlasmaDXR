# Advanced Self-Shadowing System Design

**Date:** 2025-11-07
**Goal:** Replace PCSS with a more advanced shadow system that:
- Works with inline RayQuery + probe grid (no Spatial Interpolation needed)
- Provides visible quality improvement (unlike current PCSS)
- Maintains high performance
- Adds self-shadowing for volumetric particles

---

## Current PCSS Problems

### Issues:
1. **No visible difference** between 1 and 16 shadow rays
2. **Massive performance cost** (~50% fps drop)
3. **Doesn't work with inline RQ** (needs Spatial Interpolation, which is broken)
4. **Only works with external lights** (multi-light system), not particle-to-particle

### Why Current PCSS Fails for Inline RQ:

Inline RayQuery uses pre-computed `g_rtLighting[]` buffer:
```cpp
directRTLight = g_rtLighting[hit.particleIdx].rgb;  // Just a color
```

This is a **flattened RGB value** - no information about which neighbor particles contributed. PCSS needs to know:
- Which particles are illuminating this particle
- Direction/distance to each neighbor
- Ability to cast shadow rays to neighbors

**Result:** PCSS is architecturally incompatible with inline RQ in legacy mode.

---

## Proposed Solution: Hybrid Shadow System

Combine **three complementary techniques** for maximum quality and flexibility:

### 1. Screen-Space Contact Shadows (Primary)

**What it is:**
- Ray march through depth buffer in screen space
- Test for occlusion between pixel and light sources
- Extremely cheap (~0.3-0.5ms @ 1440p)

**Why it works:**
- **Lighting-agnostic**: Works with probe grid, inline RQ, multi-light
- **Self-shadowing**: Automatically captures particle-to-particle occlusion
- **No RT acceleration structure needed**: Uses depth buffer only
- **Contact hardening**: Shadows get sharper near contact points (like PCSS)

**Implementation:**
```hlsl
// Screen-Space Contact Shadow Ray March
float ScreenSpaceShadow(float3 worldPos, float3 lightDir, float maxDist) {
    // 1. Transform world position to screen space
    float4 screenPos = mul(float4(worldPos, 1.0), viewProj);
    screenPos.xyz /= screenPos.w;

    // 2. Transform light direction to screen space
    float4 screenLightEnd = mul(float4(worldPos + lightDir * maxDist, 1.0), viewProj);
    screenLightEnd.xyz /= screenLightEnd.w;

    // 3. Ray march in screen space
    float2 rayDir = normalize(screenLightEnd.xy - screenPos.xy);
    float rayLength = length(screenLightEnd.xy - screenPos.xy);

    const int NUM_STEPS = 16;  // Adaptive based on distance
    float stepSize = rayLength / NUM_STEPS;

    float occlusion = 0.0;
    for (int i = 0; i < NUM_STEPS; i++) {
        float2 sampleUV = screenPos.xy + rayDir * stepSize * i;

        // Sample depth buffer
        float sceneDepth = g_depthBuffer.SampleLevel(sampler, sampleUV, 0).r;
        float rayDepth = screenPos.z + (screenLightEnd.z - screenPos.z) * (i / NUM_STEPS);

        // If ray is behind scene geometry, accumulate occlusion
        if (rayDepth > sceneDepth + 0.001) {
            occlusion += 1.0;
        }
    }

    // Contact hardening: occlusion strength based on distance
    float contactFactor = 1.0 - saturate(rayLength / 0.1);  // Harder near contact
    return 1.0 - saturate(occlusion / NUM_STEPS) * contactFactor;
}
```

**Performance:**
- 16 steps × full screen = ~142M samples @ 1440p
- RTX 4060 Ti: **~0.3ms** (texture cache friendly)
- **10× cheaper than current PCSS**

### 2. Depth-Aware Volumetric Shadows

**What it is:**
- Use particle density along view ray for self-shadowing
- Accumulate opacity from camera to each sample point
- Beer-Lambert law for volumetric attenuation

**Why it works:**
- **Physically accurate** for volumetric media
- **Very cheap** (already doing volume ray marching)
- **Density-based**: Denser particle regions cast stronger shadows
- **Works with all lighting**: Probe grid, inline RQ, multi-light

**Implementation:**
```hlsl
// Already in your ray marching loop!
float accumulatedDensity = 0.0;

for (each volume sample) {
    // Accumulate density along ray
    accumulatedDensity += density * stepSize;

    // Self-shadowing factor (Beer-Lambert)
    float selfShadow = exp(-accumulatedDensity * extinctionCoeff);

    // Apply to lighting
    float3 litColor = emission * selfShadow;

    // Also apply to external lighting
    float3 externalLight = (probeGridLight + directRTLight) * selfShadow;
}
```

**Performance:**
- **Zero cost** (already ray marching)
- Just one exp() per sample

### 3. Enhanced Temporal Accumulation

**What it is:**
- Accumulate shadows over 8-16 frames instead of 2
- Motion-vector aware reprojection
- Variance-based adaptive sampling

**Why it works:**
- **Smooth 1-ray quality** → looks like 16+ rays
- **Much cheaper** than increasing rays per frame
- **Reduces flashing** (your main complaint about probe grid)

**Implementation:**
```hlsl
// Temporal shadow accumulation
float4 currentShadow = ComputeShadow(worldPos, lightDir);  // 1 ray
float4 prevShadow = g_shadowHistory.Sample(sampler, reprojectedUV);

// Compute reprojection confidence
float2 motionVector = worldPos.xy - prevWorldPos.xy;
float confidence = saturate(1.0 - length(motionVector) * 10.0);

// Adaptive blend based on motion
float blendFactor = lerp(0.05, 0.3, 1.0 - confidence);  // Slow blend when static
float4 accumShadow = lerp(prevShadow, currentShadow, blendFactor);

// Variance estimation for quality assessment
float variance = abs(currentShadow - prevShadow);
if (variance > THRESHOLD) {
    // High variance = need more samples this frame
    blendFactor = 0.5;  // Blend faster
}

g_shadowHistoryCurrent[pixelPos] = accumShadow;
```

**Performance:**
- 1 shadow ray per light instead of 8-16
- **8-16× cheaper** than current PCSS
- Converges to smooth result in ~60-120ms

---

## Combined System Architecture

### Integration Points:

1. **Render depth buffer** (pre-pass before Gaussian rendering)
   - Render particle AABBs to depth buffer
   - ~0.1-0.2ms cost
   - Enables screen-space shadows

2. **Volumetric self-shadowing** (in Gaussian ray marching loop)
   - Accumulate density along view ray
   - Apply Beer-Lambert attenuation
   - Zero additional cost

3. **Screen-space contact shadows** (per light, in Gaussian renderer)
   - Cast 1 screen-space ray per light per pixel
   - Accumulate occlusion
   - ~0.3ms for multi-light (13 lights)

4. **Temporal accumulation** (post-shadow, per frame)
   - Reproject previous frame shadows
   - Blend with current frame
   - ~0.2ms

**Total shadow cost: ~0.6-0.9ms** (vs ~3-5ms for current PCSS!)

---

## Quality Improvements

### Visible Differences:

1. **Contact shadows**: Dark halos around particles where they overlap
2. **Self-shadowing**: Dense particle clouds cast shadows on themselves
3. **Smooth temporal convergence**: 1-ray → 16-ray quality in 60ms
4. **Distance-based hardening**: Sharp shadows near contact, soft shadows far away

### Comparison:

| Feature | Current PCSS | New System |
|---------|--------------|------------|
| Cost | ~3-5ms | ~0.6-0.9ms |
| Works with inline RQ | ❌ No | ✅ Yes |
| Works with probe grid | ❌ No | ✅ Yes |
| Self-shadowing | ❌ No | ✅ Yes |
| Visible quality | ⚠️ Minimal | ✅ Significant |
| Temporal smoothness | ⚠️ 2 frames | ✅ 8-16 frames |

---

## Implementation Plan

### Phase 1: Depth Pre-Pass (1 hour)
- Render particle AABBs to depth buffer
- Use existing BLAS for fast rasterization
- Create depth texture resource

### Phase 2: Screen-Space Shadows (2-3 hours)
- Implement screen-space ray marching
- Add contact hardening
- Integrate with multi-light system

### Phase 3: Volumetric Self-Shadowing (1 hour)
- Add density accumulation to ray marching loop
- Apply Beer-Lambert attenuation
- Tune extinction coefficient

### Phase 4: Enhanced Temporal Accumulation (2 hours)
- Expand shadow history buffer (2 frames → 16 frames)
- Add motion-vector reprojection
- Implement variance-based adaptive blending

### Phase 5: Integration with Inline RQ + Probe Grid (1 hour)
- Apply screen-space shadows to inline RQ lighting
- Apply to probe grid sampling
- Add ImGui controls

**Total implementation time: ~7-9 hours**

---

## ImGui Controls

```cpp
ImGui::Checkbox("Screen-Space Shadows", &m_useScreenSpaceShadows);
ImGui::SliderInt("SS Shadow Steps", &m_ssSteps, 8, 32);  // Quality vs performance
ImGui::SliderFloat("Contact Hardening", &m_contactHardening, 0.0f, 1.0f);
ImGui::SliderFloat("Self-Shadow Strength", &m_selfShadowStrength, 0.0f, 2.0f);
ImGui::SliderInt("Temporal History Frames", &m_shadowHistoryFrames, 2, 16);
ImGui::SliderFloat("Temporal Blend Speed", &m_shadowBlendSpeed, 0.05f, 0.5f);
```

---

## Performance Budget

**Target: 100 FPS @ 10K particles @ 1440p**

| System | Current | With New Shadows | Budget |
|--------|---------|------------------|---------|
| Probe grid (128 rays) | 2.0ms | 2.0ms | ✅ |
| Inline RQ lighting | 1.5ms | 1.5ms | ✅ |
| PCSS shadows | 3-5ms | 0.6-0.9ms | ✅ **Saves 2-4ms!** |
| **Total** | **~7ms** | **~4.5ms** | ✅ **40% faster** |

**Expected FPS improvement: 100 → 140 FPS** (40% gain from shadow optimization!)

---

## Why This Works Without Spatial Interpolation

### Key Insight:

Screen-space shadows and volumetric self-shadowing **don't care about the lighting source**. They work on the **final image** and **density field**, not the lighting computation itself.

**Legacy inline RQ path:**
```hlsl
// 1. Get pre-computed lighting (no neighbor info)
directRTLight = g_rtLighting[hit.particleIdx].rgb;

// 2. Apply screen-space shadow (lighting-agnostic)
float ssShadow = ScreenSpaceShadow(pos, toLight, lightDist);
directRTLight *= ssShadow;

// 3. Apply volumetric self-shadowing (density-based)
float volShadow = exp(-accumulatedDensity * extinction);
directRTLight *= volShadow;

// 4. Combine with probe grid
rtLight = probeGridLight + directRTLight;
```

**No need for Spatial Interpolation!** Shadows are computed independently of the lighting source.

---

## Conclusion

This hybrid shadow system:
- ✅ **2-4ms performance saving** vs current PCSS
- ✅ **Works with inline RQ + probe grid** (no Spatial Interpolation needed)
- ✅ **Visible quality improvement** (contact shadows, self-shadowing)
- ✅ **Smooth temporal convergence** (eliminates flashing)
- ✅ **Flexible and extensible** (easy to tune and enhance)

**Recommendation:** Implement this to replace current PCSS. Expected result: better quality + better performance + works with all lighting systems.
