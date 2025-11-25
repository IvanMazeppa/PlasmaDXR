# RTXDI & Celestial Effects Roadmap for PlasmaDX-Clean

**Document Version:** 3.1
**Created:** November 24, 2025
**Updated:** November 25, 2025
**Author:** Claude Opus 4.5 + Ben
**Current Version:** 0.19.6

## Overview
~~Fix RTXDI temporal reprojection (biggest win)~~ ✅ **COMPLETE**, then pivot to celestial variety and pyro effects. M6 spatial reuse deferred until celestial work complete.

**User Priorities:**
- ~~Fix M5 temporal first (90% patchwork reduction)~~ ✅ **ACHIEVED - 95%+ reduction!**
- Pyro/explosions are NEAR-TERM priority
- Quality > FPS (75-85 FPS acceptable)

**Performance Target:** 75+ FPS @ 1080p, 10K particles, RTX 4060 Ti

---

## Execution Order

```
Phase 1: Fix M5 Temporal Reprojection     [✅ COMPLETE - v0.19.6]
    ↓
Phase 2: Celestial Variety & Pyro Effects [NEXT - Ready to Start]
    ↓
Phase 3: RTXDI Refinements (Jitter + M6)  [LATER - After Celestial]
```

---

## Phase 1: Fix M5 Temporal Reprojection ✅ COMPLETE
**Completed:** November 25, 2025 | **Version:** 0.19.6

### Root Cause (Identified & Fixed)
M5 used planar Z=0 assumption for world position reconstruction. Additionally, RTXDI selected only ONE light per pixel, causing hard patchwork boundaries regardless of temporal stability.

### Solution Implemented
**Two-Part Fix:**
1. **Depth-Based Reprojection** - Proper 3D world position via RT depth buffer + inverse view-projection
2. **Soft Spatial Blending** - 70% RTXDI-selected light + 30% distance-weighted blend from nearby lights
3. **HG Phase Optimization** - Phase function only on primary light (prevents compound darkening)

**Result:** 95%+ patchwork reduction. Boundaries essentially invisible once temporally stabilized.

### 1.1 Add Depth Buffer Output
Gaussian renderer already computes hit distances. Output to depth texture.

**Files:**
- `src/particles/ParticleRenderer_Gaussian.h` - Add `ComPtr<ID3D12Resource> m_rtDepthBuffer`
- `src/particles/ParticleRenderer_Gaussian.cpp` - Create R32_FLOAT depth texture, bind as UAV (u4)
- `shaders/particles/particle_gaussian_raytrace.hlsl` - Add `RWTexture2D<float> g_rtDepth : register(u4)` and output `hits[0].tNear`

### 1.2 Fix Temporal Reprojection
Replace planar Z=0 with proper depth-based unprojection.

**File:** `shaders/rtxdi/rtxdi_temporal_accumulate.hlsl`

```hlsl
// Add to cbuffer
row_major float4x4 g_invViewProj;

// Add texture binding
Texture2D<float> g_depth : register(t2);

// Replace PixelToWorldPosition with:
float3 PixelToWorldPosition_WithDepth(uint2 pixelCoord) {
    float depth = g_depth[pixelCoord];
    float2 uv = (float2(pixelCoord) + 0.5) / float2(g_screenWidth, g_screenHeight);
    float2 ndc = uv * 2.0 - 1.0;
    ndc.y = -ndc.y;
    float4 clipPos = float4(ndc.x, ndc.y, depth, 1.0);
    float4 worldPos = mul(clipPos, g_invViewProj);
    return worldPos.xyz / worldPos.w;
}
```

### 1.3 Update C++ Bindings
**Files:**
- `src/lighting/RTXDILightingSystem.cpp`:
  - Compute `invViewProj = XMMatrixInverse(viewProj)` (~line 1227)
  - Bind depth buffer as SRV (t2) to temporal accumulation shader (~line 1240)
  - Update constant buffer to include invViewProj (add 64 bytes)
- `src/lighting/RTXDILightingSystem.h` - Add depth texture handle member

### Success Criteria ✅ ALL MET
- [x] Temporal history persists during camera movement
- [x] Sample count accumulates to maxSamples (8)
- [x] Patchwork smooths over 8-16 frames (actually ~2-4 frames!)
- [x] No visual regression from multi-light mode
- [x] **BONUS:** HG phase function works without excessive darkening

**Actual Result:** 95%+ patchwork reduction. RTXDI is now production-ready!

---

## Phase 2: Celestial Variety & Pyro Effects (NEAR-TERM)
**Estimated Time: 12-20 hours | Next 2-3 Sessions**

### 2A: Particle Lifetime System (Foundation for Pyro)
**Estimated Time: 4-6 hours**

Add lifetime tracking to enable temporal effects (explosions, fades).

**Files:**
- `src/particles/ParticleSystem.h`:
  - Extend particle struct from 48 → 64 bytes
  - Add: `float lifetime;` (current age in seconds)
  - Add: `float maxLifetime;` (total duration, 0 = infinite)
  - Add: `float spawnTime;` (frame time when spawned)
  - Add: `uint32_t flags;` (EXPLOSION, FADING, etc.)

- `shaders/particles/particle_physics.hlsl`:
  - Update lifetime each frame: `lifetime += deltaTime`
  - Kill particles when `lifetime >= maxLifetime && maxLifetime > 0`

- `shaders/particles/particle_gaussian_raytrace.hlsl`:
  - Fade opacity based on `lifetime / maxLifetime`
  - Apply velocity expansion curves for explosions

### 2B: Supernova/Explosion Material Type
**Estimated Time: 3-4 hours**

Add SUPERNOVA material type with explosion dynamics.

**Files:**
- `src/particles/ParticleSystem.h`:
  - Add to enum: `SUPERNOVA = 5, STELLAR_FLARE = 6`

- `src/particles/ParticleSystem.cpp` (InitializeMaterialProperties):
  ```cpp
  // SUPERNOVA - extremely hot, expanding, self-luminous
  m_materialProperties[5] = {
      .albedo = {1.0f, 0.9f, 0.7f},      // Brilliant white-yellow
      .opacity = 0.95f,
      .emissionMultiplier = 15.0f,       // VERY bright
      .scatteringCoefficient = 1.5f,
      .phaseG = 0.8f                     // Strong forward scattering
  };
  ```

- `shaders/particles/gaussian_common.hlsl`:
  - Add expansion velocity curve: `radius *= 1.0 + lifetime * expansionRate`
  - Add temperature decay: `temp *= exp(-lifetime * coolingRate)`

### 2C: Explosion Spawning System
**Estimated Time: 4-6 hours**

Create explosion event system to spawn particle bursts.

**Files:**
- `src/particles/ParticleSystem.h`:
  - Add `void SpawnExplosion(XMFLOAT3 center, float energy, uint32_t particleCount)`
  - Add explosion queue for GPU dispatch

- `src/particles/ParticleSystem.cpp`:
  - Implement explosion spawning with:
    - Radial velocity distribution (Fibonacci sphere)
    - Temperature based on energy (50,000K+ for supernovae)
    - Lifetime based on energy (2-10 seconds)

- `shaders/particles/explosion_spawn.hlsl` (NEW):
  - GPU compute shader for parallel explosion particle initialization
  - Randomized velocities, temperatures, lifetimes

### 2D: Dynamic Explosion Lights
**Estimated Time: 2-3 hours**

Register explosion particles as temporary point lights.

**Files:**
- `src/lighting/RTXDILightingSystem.cpp`:
  - `void RegisterExplosionLight(XMFLOAT3 pos, float intensity, float duration)`
  - Auto-remove lights when duration expires
  - Integrate with existing 13-light multi-light system

### Success Criteria
- [ ] Particles can have finite lifetimes
- [ ] SUPERNOVA material type visible with correct emission
- [ ] Can spawn explosion at any world position
- [ ] Explosion particles expand, cool, and fade over time
- [ ] Explosion creates temporary light source

---

## Phase 3: RTXDI Refinements (DEFERRED)
**Estimated Time: 12-16 hours | After Celestial Complete**

### 3A: Multi-Cell Jittering
Add per-pixel cell jitter to soften cell boundaries.

**File:** `shaders/rtxdi/rtxdi_raygen.hlsl`
```hlsl
float2 jitter = float2(Random(pixelCoord, g_frameIndex),
                       Random(pixelCoord, g_frameIndex + 1000));
worldPos.xy += (jitter - 0.5) * g_cellSize;
```

### 3B: Spatial Reuse (M6)
Share light samples between neighboring pixels.

**Files to Create:**
- `shaders/rtxdi/rtxdi_spatial_reuse.hlsl`

**Files to Modify:**
- `src/lighting/RTXDILightingSystem.h/cpp` - Add M6 dispatch

**Algorithm:**
- Read current reservoir
- Sample 5 neighbors (Poisson disk, 30-pixel radius)
- Combine using weighted reservoir sampling

---

## Performance Budget

| Component | Cost | Cumulative FPS |
|-----------|------|----------------|
| Baseline (Multi-Light) | - | 115-120 |
| Phase 1 (M5 Fix) | +0.2ms | 105-110 |
| Phase 2 (Lifetime) | +0.1ms | 100-105 |
| Phase 2 (Explosions) | +0.5ms | 90-100 |
| Phase 3 (M6) | +0.8ms | 75-85 |
| **With Full RTXDI + Pyro** | **+1.6ms** | **75-85** | **Acceptable per user** |

---

## Critical Files Summary

### Phase 1 (This Session)
| File | Changes | Priority |
|------|---------|----------|
| `shaders/rtxdi/rtxdi_temporal_accumulate.hlsl` | Depth-based reprojection | CRITICAL |
| `src/lighting/RTXDILightingSystem.cpp` | Depth SRV, invViewProj | CRITICAL |
| `shaders/particles/particle_gaussian_raytrace.hlsl` | Depth output (u4) | HIGH |
| `src/particles/ParticleRenderer_Gaussian.cpp` | Depth texture creation | HIGH |

### Phase 2 (Near-Term)
| File | Changes | Priority |
|------|---------|----------|
| `src/particles/ParticleSystem.h` | Lifetime fields, SUPERNOVA enum | HIGH |
| `src/particles/ParticleSystem.cpp` | Material presets, SpawnExplosion | HIGH |
| `shaders/particles/particle_physics.hlsl` | Lifetime update | HIGH |
| `shaders/particles/gaussian_common.hlsl` | Expansion/cooling curves | MEDIUM |
| `shaders/particles/explosion_spawn.hlsl` | NEW - explosion initialization | MEDIUM |

---

## Risk Mitigation

| Risk | Mitigation | Fallback |
|------|------------|----------|
| Depth integration issues | Gaussian already computes tNear | Separate depth pre-pass |
| Particle struct size change | 64 bytes maintains alignment | Keep 48 bytes, pack tighter |
| Explosion performance | Limit spawn count | Cap at 1000 explosion particles |
| Complete RTXDI failure | Multi-light works beautifully | Keep as runtime toggle |

**Critical Fallback:** Multi-light mode (13 lights, 115-120 FPS) remains available regardless.

---

## Implementation Progress (This Session)

### Completed
- [x] Created m_rtDepthBuffer in ParticleRenderer_Gaussian.h (outside ENABLE_DLSS)
- [x] Added GetRTDepthBuffer() and GetRTDepthSRV() public getters
- [x] Added depth buffer creation code in ParticleRenderer_Gaussian.cpp

### All Phase 1 Tasks Complete ✅
- [x] Add depth UAV binding to Gaussian shader (u4)
- [x] Add depth output in particle_gaussian_raytrace.hlsl
- [x] Update RTXDI temporal shader with depth-based reprojection
- [x] Add invViewProj to RTXDI constant buffer
- [x] Bind depth SRV to temporal accumulation dispatch

### Additional Fixes Implemented
- [x] Soft spatial light blending (70% RTXDI + 30% distance-weighted)
- [x] HG phase function optimization (primary light only)

---

## Research Sources

- [NVIDIA RTXDI Integration Guide](https://github.com/NVIDIA-RTX/RTXDI/blob/main/Doc/Integration.md)
- [ReSTIR Temporal Reprojection Fix](https://computergraphics.stackexchange.com/questions/119/how-does-temporal-reprojection-work)
- [GPU Pro 360 - Volumetric Explosions](https://www.taylorfrancis.com/chapters/edit/10.1201/b22483-17/realistic-volumetric-explosions-games-alex-dunn)
- [EmberGen Real-Time Pyro](https://jangafx.com/software/embergen)
