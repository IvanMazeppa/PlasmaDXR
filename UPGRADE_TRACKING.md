# PlasmaDX Upgrade Tracking Document

**Project:** PlasmaDX - 3D Gaussian Volumetric Particle Renderer with DXR 1.1
**Hardware:** NVIDIA RTX 4060 Ti (Ada Lovelace, 16GB VRAM)
**Last Updated:** 2025-10-11

---

## 🎯 Project Vision

Real-time volumetric accretion disk simulator using 3D Gaussian splatting and hardware ray tracing for physically accurate particle-to-particle lighting at 60 FPS with 100,000+ particles.

---

## 📊 Current Status Overview

| Metric | Current | Target |
|--------|---------|--------|
| **Particle Count** | 20,000 | 100,000+ |
| **Performance** | 15-30 FPS | 60 FPS |
| **Rendering** | 3D Gaussian Splatting | ✓ Working |
| **Ray Tracing** | DXR 1.1 Inline (RayQuery) | ✓ Working |
| **BLAS/TLAS** | Per-particle AABBs | ✓ Working |
| **Volumetric Lighting** | Partially working | In progress |

---

# ✅ COMPLETED UPGRADES

## Phase 0: Foundation (Completed September-October 2025)

### 1. ✅ 3D Gaussian Splatting Renderer
**Status:** Fully working
**Completion Date:** October 9, 2025
**Impact:** Proper volumetric particle representation (not billboards)

**What Was Built:**
- Anisotropic 3D Gaussian ellipsoids with velocity-aligned stretching
- Analytic ray-Gaussian intersection using RayQuery API
- Volumetric ray marching through sorted Gaussians (16 steps per particle)
- Beer-Lambert absorption model for optical depth
- Per-particle procedural AABBs as ray tracing primitives
- Runtime toggle: `--gaussian` vs `--billboard`

**Key Files:**
- `shaders/particles/gaussian_common.hlsl` - Gaussian math library
- `shaders/particles/particle_gaussian_raytrace.hlsl` - Main compute shader
- `src/particles/ParticleRenderer_Gaussian.h/cpp` - Renderer implementation

**Performance:**
- 127 FPS baseline @ 20K particles (no RT features)
- 10-20 FPS with all RT features enabled

**Documentation:** `GAUSSIAN_INTEGRATION_STATUS.md`

---

### 2. ✅ DXR 1.1 Infrastructure
**Status:** Fully working
**Completion Date:** September 2025
**Impact:** Hardware-accelerated ray tracing foundation

**What Was Built:**
- Bottom-Level Acceleration Structure (BLAS) with 20,000 procedural AABBs
- Top-Level Acceleration Structure (TLAS) with single instance
- Inline ray tracing using RayQuery API (not TraceRay())
- AABB bounds generated from 3σ Gaussian extent
- Dynamic rebuild every frame (particles in motion)

**Performance:**
- BLAS/TLAS build: ~2-3ms per frame
- Ray traversal: 12.5M primary rays @ baseline
- Up to 206M hardware rays with RT features (PIX verified)

**Key Technique:** Procedural primitives require manual intersection commitment via `query.CommitProceduralPrimitiveHit()`

---

### 3. ✅ Keplerian Orbital Physics
**Status:** Fully working
**Completion Date:** October 2025
**Impact:** Realistic accretion disk dynamics

**What Was Built:**
- N-body gravitational simulation with inverse square law
- Angular momentum conservation for stable orbits
- Turbulence system for particle interactions
- Velocity damping for visual effect control
- Runtime controls (V/N/B/M keys)

**Physics Constants:**
- Gravity: 100-1000 (Ctrl+V/Shift+V)
- Angular Momentum: 0.5-2.0 (Ctrl+N/Shift+N)
- Turbulence: 5-50 (Ctrl+B/Shift+B)
- Damping: 0.9-1.0 (Ctrl+M/Shift+M)

---

### 4. ✅ Physical Emission System
**Status:** Working in billboard mode, needs fixing for Gaussian mode
**Completion Date:** October 2025
**Impact:** Temperature-based blackbody emission

**What Was Built:**
- Blackbody temperature range: 1,000K - 40,000K
- Temperature-to-color conversion (Planck's law approximation)
- Emission intensity calculation with exponential falloff
- Runtime controls (E/R/G keys) with adjustable strength
- Status bar display: `[E:1.0] [RT]`

**Issue:** Not properly integrated into Gaussian volumetric rendering
**TODO:** Fix emission in `particle_gaussian_raytrace.hlsl`

---

### 5. ✅ ACES Tone Mapping
**Status:** Fully working
**Completion Date:** October 2025
**Impact:** Professional HDR color handling

**What Was Built:**
- Industry-standard ACES filmic tone mapping curve
- Gamma correction (2.2)
- Black background (no blue overexposure)
- Proper color gamut preservation

**Visual Improvement:** Changed from brown/monotone to vibrant plasma colors

**Code:**
```hlsl
float3 ACES_tonemap(float3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

finalColor = ACES_tonemap(finalColor);
finalColor = pow(finalColor, 1.0 / 2.2); // Gamma correction
```

---

### 6. ✅ Runtime Toggle System
**Status:** Working for physics, working for RT features (with CBV fix)
**Completion Date:** October 9-11, 2025
**Impact:** Performance vs quality trade-offs

**Implemented Toggles:**

| Key | Feature | Performance Impact | Status |
|-----|---------|-------------------|--------|
| **V/N/B/M** | Physics parameters | Minimal | ✅ Working |
| **E/R/G** | Emission parameters | Minimal | ✅ Working |
| **F5** | Shadow Rays | ~50M rays, -80% FPS | ✅ Working |
| **F6** | In-Scattering | ~20M rays, -40% FPS | ✅ Working |
| **F7** | ReSTIR Temporal | NEW (Phase 1) | 🟡 Testing |
| **F8** | Phase Function | Free (math only) | ✅ Working |
| **F11** | Anisotropic Gaussians | Minimal | ✅ Working |

**Status Bar Display:** `G:500 A:1.0 T:15 [E:1.0] [RT] [F5:Shadow] [F6:InScat:2.0] [F7:ReSTIR:0.9] [F8:Phase:5.0] [F11:Aniso:1.0]`

**Critical Fix Applied:** Changed from root constants (48 DWORD limit) to Constant Buffer View (no limit)

---

### 7. ✅ Volumetric Shadow Rays (Phase 1)
**Status:** Working (toggled via F5)
**Completion Date:** October 2025
**Impact:** Particle-to-particle occlusion for depth perception

**What Was Built:**
- Shadow ray casting from each particle to light source
- Beer-Lambert optical depth accumulation through occluding particles
- Variable transmittance based on density (not fixed 0.3)
- Minimum ambient lighting (5%) to prevent pure black

**Implementation:**
```hlsl
float CastShadowRay(float3 pos, float3 toLightDir, float lightDist) {
    RayDesc shadowRay;
    shadowRay.Origin = pos + toLightDir * 0.01;
    shadowRay.Direction = toLightDir;
    shadowRay.TMin = 0.001;
    shadowRay.TMax = lightDist - 0.01;

    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(g_particleBVH, RAY_FLAG_NONE, 0xFF, shadowRay);

    float accumOpticalDepth = 0.0;
    while (query.Proceed()) {
        if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
            uint hitIdx = query.CandidatePrimitiveIndex();
            Particle p = g_particles[hitIdx];
            float density = p.density;
            accumOpticalDepth += density * 2.0;
            query.CommitProceduralPrimitiveHit(query.CandidateTriangleRayT());
        }
    }

    float transmittance = exp(-accumOpticalDepth * 3.0);
    return max(0.05, transmittance); // 5% minimum ambient
}
```

**Performance:** ~50M additional rays/frame (~2-3ms)

**Documentation:** `VOLUMETRIC_RT_FIXES_SUMMARY.md`

---

### 8. ✅ Volumetric In-Scattering (Phase 1)
**Status:** Working (toggled via F6)
**Completion Date:** October 2025
**Impact:** Glowing halos around particles

**What Was Built:**
- Monte Carlo sampling of nearby particles (12 samples per march step)
- Light-biased hemisphere sampling (not uniform random)
- Shadow-aware scattering (shadows affect in-scattering)
- Phase function integration with Henyey-Greenstein
- 150 unit sampling range (increased from 50)

**Key Parameters:**
- Samples: 12 (was 4)
- Range: 150 units (was 50)
- Amplification: 3× multiplier
- Shadow integration: `scattering *= shadowTerm`

**Visual Effect:** Bright halos around particles, forward scattering glow

**Performance:** ~20M additional rays/frame (~1-2ms)

**Control:** Ctrl+F6/Shift+F6 adjusts strength (0.0-5.0)

---

### 9. ✅ Henyey-Greenstein Phase Function
**Status:** Fully working (toggled via F8)
**Completion Date:** October 2025
**Impact:** Anisotropic forward scattering for realistic particle glow

**What Was Built:**
- Henyey-Greenstein phase function with g=0.7 (strong forward scatter)
- 5× phase boost multiplier
- Rim lighting addition for edge glow
- Runtime strength adjustment (0-20)

**Implementation:**
```hlsl
float HenyeyGreenstein(float cosTheta, float g) {
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cosTheta;
    return (1.0 - g2) / (4.0 * 3.14159265 * pow(abs(denom), 1.5));
}

// In rendering loop
float3 viewDir = normalize(pos - ray.Origin);
float3 lightDir = normalize(lightPos - pos);
float cosTheta = dot(viewDir, lightDir);

float phase = HenyeyGreenstein(cosTheta, 0.7);
float phaseBoost = 1.0 + phase * phaseStrength * 5.0;
float rimLight = pow(1.0 - abs(cosTheta), 2.0) * 0.5;

totalEmission *= (phaseBoost + rimLight);
```

**Visual Effect:** Dramatic forward-scattering halos, edge glow on particles

**Performance:** Negligible (pure math, no rays)

**Control:** Ctrl+F8/Shift+F8 adjusts strength (0.0-20.0)

---

### 10. ✅ Anisotropic Gaussian Rendering
**Status:** Fully working (toggled via F11)
**Completion Date:** October 11, 2025
**Impact:** Velocity-aligned particle stretching for motion blur effect

**What Was Built:**
- Velocity-aligned ellipsoid stretching (not spherical)
- Orthonormal basis construction from velocity vector
- Stretch factor based on speed (0-100 units/sec mapped to 1-5× stretch)
- Anisotropic AABB bounds for ray tracing
- Runtime strength control (0.0-3.0)

**Implementation:**
```hlsl
float3 ComputeGaussianScale(Particle p, float baseRadius,
                            bool useAnisotropic, float anisotropyStrength) {
    float3 scale = float3(baseRadius, baseRadius, baseRadius);

    if (useAnisotropic) {
        float speed = length(p.velocity);
        float speedFactor = saturate(speed / 100.0);
        float stretch = 1.0 + speedFactor * anisotropyStrength;

        // Scale gets multiplied by rotation matrix later
        scale.z *= stretch; // Stretch in Z (velocity direction)
    }

    return scale;
}

float3x3 ComputeGaussianRotation(float3 velocity) {
    float3 forward = normalize(velocity);
    float3 right = normalize(cross(float3(0, 1, 0), forward));
    float3 up = cross(forward, right);
    return float3x3(right, up, forward);
}
```

**Visual Effect:** Particles elongate along velocity direction, creating streaking motion trails

**Performance:** Minimal (<1% overhead)

**Control:** Ctrl+F11/Shift+F11 adjusts strength (0.0-3.0)

---

# 🟡 IN PROGRESS UPGRADES

## Phase 1: ReSTIR Temporal Reuse (Current Work)

### ReSTIR Phase 1: Temporal Resampling
**Status:** Infrastructure complete, debugging data flow
**Started:** October 11, 2025
**Expected Completion:** October 11-12, 2025
**Impact:** 10-60× faster lighting convergence

**What's Been Built:**

#### C++ Infrastructure ✅
- Reservoir buffers (2× 63MB ping-pong @ 1080p)
- Root signature extension (5 → 7 parameters)
- Descriptor tables for StructuredBuffer binding
- Automatic ping-pong buffer swapping
- F7 toggle + Ctrl/Shift controls (temporal weight 0-1)

**Structure:**
```cpp
struct Reservoir {
    float3 lightPos;    // 12 bytes - selected light position
    float weightSum;    // 4 bytes  - sum of weights
    uint M;             // 4 bytes  - samples accumulated
    float W;            // 4 bytes  - final weight
    uint particleIdx;   // 4 bytes  - which particle is light
    float pad;          // 4 bytes  - alignment
};
// Total: 32 bytes per pixel
```

**Memory:** 126MB total (2 × 2,073,600 pixels × 32 bytes)

#### HLSL Algorithm ✅
- `Hash()` - Pseudo-random number generation
- `UpdateReservoir()` - Weighted reservoir sampling
- `ValidateReservoir()` - Temporal visibility checking with shadow rays
- `SampleLightParticles()` - Initial candidate generation (16 samples)
- Temporal decay (M × 0.9) to prevent infinite accumulation
- Integration with existing RT lighting

#### Current Issue 🔴
`SampleLightParticles()` not finding particles - returns empty reservoirs (M=0)

**Symptoms:**
- PIX shows M=12345 when F7 OFF (test write works) → buffer binding OK ✓
- PIX shows M=1 when F7 ON (only temporal sample) → validation works ✓
- No new samples being added → `SampleLightParticles()` broken ✗
- Colors become muted/brown → low reservoir weight causes dim lighting
- Yellow debug indicator dims over time → reservoir quality = M/16 = 6%

**Root Cause:** Procedural primitive ray queries require looping through candidates and manually committing them

**Latest Fix Applied (Awaiting Test):**
```hlsl
// Process candidates - for procedural primitives we need to loop!
while (query.Proceed()) {
    if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
        // Found a candidate AABB - commit it and use first hit
        query.CommitProceduralPrimitiveHit(query.CandidateTriangleRayT());
        break;  // Take first hit only
    }
}

// Check if we got a hit
if (query.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT) {
    uint hitParticleIdx = query.CommittedPrimitiveIndex();
    // ... process hit
}
```

Also changed:
- Hemisphere sampling → Full sphere sampling (cosTheta from 0-1 to -1 to +1)
- Range: 200 → 500 units
- Added weight threshold (skip if < 0.001)

**Next Steps:**
1. Test latest build - check PIX for M values 1-16 (not stuck at 1)
2. Verify colors don't become muted
3. Verify yellow bar stays bright or brightens over time
4. If still broken: Add debug counters, verify BVH binding, check particle range

**Expected Performance:**
- Initial candidates: ~3-4ms (16 rays/pixel = 33M rays)
- Temporal validation: ~0.5ms (1 shadow ray/pixel = 2M rays)
- Total overhead: ~4-5ms

**Expected Quality:**
- 10-60× faster convergence than Monte Carlo
- Smooth temporal stability
- Smart light selection (finds brightest particles automatically)

**Documentation:** `SESSION_SUMMARY_20251011_ReSTIR.md`

**Key Files:**
- `src/particles/ParticleRenderer_Gaussian.h/cpp` - Reservoir buffers
- `shaders/particles/particle_gaussian_raytrace.hlsl` - ReSTIR algorithm
- `src/core/Application.h/cpp` - F7 controls

---

# 📋 PLANNED UPGRADES

## Phase 2: ReSTIR Spatial Reuse (High Priority)

**Status:** Not started
**Dependencies:** ReSTIR Phase 1 working
**Estimated Time:** 16-24 hours
**Expected Impact:** Additional 3-5× quality improvement

**What Needs Building:**
- Share reservoirs with 3-5 neighboring pixels
- Spatial validation (verify neighbors see same light)
- Biased sampling (prefer neighbors with high M)
- Spatial weight computation
- Multiple spatial iterations (1-2)

**Algorithm:**
```hlsl
// After temporal reuse
for (uint neighbor = 0; neighbor < 5; neighbor++) {
    int2 offset = spatialOffsets[neighbor]; // Predefined pattern
    uint neighborIdx = PixelToIndex(pixelPos + offset);

    Reservoir neighborReservoir = g_currentReservoirs[neighborIdx];

    // Validate neighbor's light is visible from our position
    if (ValidateReservoir(neighborReservoir, ourPosition)) {
        // Merge neighbor's sample using weighted probability
        float combinedWeight = currentReservoir.weightSum + neighborReservoir.weightSum;
        float neighborProbability = neighborReservoir.weightSum / combinedWeight;

        if (Hash(seed) < neighborProbability) {
            currentReservoir.lightPos = neighborReservoir.lightPos;
            currentReservoir.particleIdx = neighborReservoir.particleIdx;
        }

        currentReservoir.M += neighborReservoir.M;
        currentReservoir.weightSum = combinedWeight;
    }
}
```

**Expected M Values:** 16 (initial) + 14 (temporal) + 50 (5 neighbors) = 80 effective samples

**Performance Cost:** ~1-2ms (5 additional shadow rays/pixel)

**Reference:** `agent/AdvancedTechniqueWebSearches/efficiency_optimizations/ReSTIR_Particle_Integration.md`

---

## Phase 3: Shader Execution Reordering (Medium Priority)

**Status:** Not started
**Dependencies:** None (can start anytime)
**Estimated Time:** 8-12 hours
**Expected Impact:** 1.5-2× ray tracing speedup

**What Needs Building:**
- Use HitObject API (Shader Model 6.8)
- Call `ReorderThread()` before shading
- Group coherent rays together for better cache utilization

**Hardware Requirement:** ✓ Ada Lovelace (RTX 4060 Ti has this!)

**Implementation:**
```hlsl
HitObject hit = HitObject::TraceRay(/* ... */);
ReorderThread(hit); // Hardware sorts threads by hit material/geometry
hit.Invoke(); // Now execute with better coherence
```

**Performance:** Expected to reduce ray tracing cost from 12ms to 6-8ms

**Reference:** `agent/AdvancedTechniqueWebSearches/efficiency_optimizations/ADA_LOVELACE_DXR12_FEATURES.md`

---

## Phase 4: Clustered BLAS Architecture (High Priority - Scale to 100K)

**Status:** Not started
**Dependencies:** Current system working well at 20K
**Estimated Time:** 24-32 hours
**Expected Impact:** Enable 100,000+ particles with similar performance

**Current Problem:**
- 100K individual AABBs in 1 giant BLAS → poor BVH quality
- 100K individual BLAS → infeasible memory/build time

**Solution: Clustered Approach**
- Divide particles into ~1000 spatial clusters (100 particles each)
- Build 1 BLAS per cluster (1000 BLAS total)
- Build 1 TLAS with 1000 instances
- Use memory pooling (single 500MB buffer)

**Implementation Steps:**
1. Spatial clustering (grid-based, compute shader, ~0.5ms)
2. Per-cluster BLAS build (8-10ms total for 1000 clusters)
3. TLAS build with cluster instances (~1ms)
4. Temporal BLAS caching (rebuild 70% per frame, cache stable 30%)

**Memory Budget:**
- BLAS pool: ~500MB
- TLAS: ~2MB
- Geometry buffers: ~100MB
- **Total:** ~650MB (acceptable)

**Performance Projection:**
| Component | Time @ 100K |
|-----------|-------------|
| Clustering | 0.5ms |
| BLAS Build | 8-10ms |
| TLAS Build | 1ms |
| Ray Tracing | 6-8ms (with SER) |
| **Total** | **15-20ms (50-60 FPS)** |

**Reference:** `agent/AdvancedTechniqueWebSearches/efficiency_optimizations/BLAS_PERFORMANCE_GUIDE.md`

---

## Phase 5: Multi-Bounce Lighting (Medium Priority)

**Status:** Not started
**Dependencies:** ReSTIR Phase 1+2 working well
**Estimated Time:** 40-48 hours
**Expected Impact:** Global illumination between particles

**What Needs Building:**
- Secondary ray tracing from particle to particle
- Radiance accumulation over multiple bounces
- Russian roulette for bounce termination
- Energy conservation per bounce

**Algorithm:**
```hlsl
float3 ComputeRadiance(float3 pos, float3 viewDir, uint bounceCount) {
    if (bounceCount >= maxBounces) return 0;

    // Sample light using ReSTIR
    Reservoir reservoir = SampleLightParticles(pos, viewDir, ...);

    // Direct lighting
    float3 directLight = EvaluateLight(reservoir);

    // Secondary bounce (Russian roulette)
    float continueProbability = 0.7 * (1.0 - bounceCount / maxBounces);
    if (Hash(seed) < continueProbability) {
        float3 randomDir = SampleHemisphere(normal);
        float3 indirectLight = ComputeRadiance(pos, randomDir, bounceCount + 1);
        return directLight + indirectLight / continueProbability;
    }

    return directLight;
}
```

**Performance Cost:** Significant (~2-3× ray count)
- 1 bounce: ~70M rays (current)
- 2 bounces: ~140M rays
- 3 bounces: ~210M rays

**Expected Quality:** Particles glow from light bouncing off other particles (realistic GI)

**Reference:** `agent/AdvancedTechniqueWebSearches/ai_lighting/Secondary_Ray_Inter_Particle_Bouncing.md`

---

## Phase 6: NRD Denoising Integration (Low Priority - Polish)

**Status:** Not started
**Dependencies:** ReSTIR working, stable temporal accumulation
**Estimated Time:** 16-24 hours
**Expected Impact:** Reduce noise, allow fewer samples

**What Needs Building:**
- Integrate NVIDIA Real-Time Denoiser (NRD) library
- Generate motion vectors for temporal reprojection
- Provide depth, normal, roughness buffers
- Handle disocclusion and ghosting

**Benefits:**
- Reduce ReSTIR candidates from 16 → 8 (2× speedup)
- Smoother temporal stability
- Production-quality noise-free output

**Reference:** `agent/AdvancedTechniqueWebSearches/DXR_Denoising_NRD_Integration_Guide.md`

---

## Phase 7: Temporal Accumulation (Low Priority)

**Status:** Not started
**Dependencies:** Motion vectors, stable camera
**Estimated Time:** 8-16 hours
**Expected Impact:** Accumulate quality over time for still/slow camera

**What Needs Building:**
- Frame-to-frame accumulation with exponential decay
- Motion vector reprojection for moving camera
- Disocclusion detection and handling
- Adaptive sample count based on camera motion

**Algorithm:**
```hlsl
// In compute shader
float3 currentFrame = Render(currentRay);
float3 historyFrame = g_history[pixelPos + motionVector];

// Disocclusion check
if (IsValidHistory(depth, prevDepth)) {
    float blend = 0.05; // 95% history, 5% current
    finalColor = lerp(historyFrame, currentFrame, blend);
} else {
    finalColor = currentFrame; // Disoccluded, use current only
}

g_history[pixelPos] = finalColor;
```

**Expected Quality:** Converges to near-offline quality after 20-30 frames (~1 second)

**Performance:** Minimal overhead (~0.5ms)

---

## Phase 8: Adaptive Quality System (Medium Priority)

**Status:** Not started
**Dependencies:** Profiling infrastructure
**Estimated Time:** 16-24 hours
**Expected Impact:** Maintain target FPS dynamically

**What Needs Building:**
- GPU timestamp queries for per-pass profiling
- Quality presets (Low, Medium, High, Ultra)
- Automatic quality adjustment based on FPS
- Per-feature cost tracking

**Quality Levels:**

| Setting | Particles | ReSTIR Candidates | Shadow Rays | FPS Target |
|---------|-----------|------------------|-------------|------------|
| **Low** | 50K | 8 | Off | 60 |
| **Medium** | 100K | 12 | On (50%) | 45 |
| **High** | 150K | 16 | On (100%) | 30 |
| **Ultra** | 200K | 32 | On + Multi-bounce | 24 |

**Auto-Adjust Logic:**
```cpp
if (frameDeltaTime > targetFrameTime * 1.2f) {
    // Too slow - reduce quality
    if (restirCandidates > 8) restirCandidates -= 4;
    else if (useShadowRays) useShadowRays = false;
    else particleCount *= 0.8f;
} else if (frameDeltaTime < targetFrameTime * 0.8f) {
    // Too fast - increase quality
    if (particleCount < maxParticles) particleCount *= 1.1f;
    else if (!useShadowRays) useShadowRays = true;
    else if (restirCandidates < 32) restirCandidates += 4;
}
```

---

## Phase 9: Resolution Scaling (Medium Priority)

**Status:** Not started
**Dependencies:** Working upscaling algorithm
**Estimated Time:** 8-16 hours
**Expected Impact:** 3-4ms savings for 0.75× resolution

**What Needs Building:**
- Render particle lighting at lower resolution (e.g., 1440×810 instead of 1920×1080)
- Bilateral upscale to full resolution (preserve edges)
- Separate high-res pass for particle positions (prevent aliasing)

**Upscaling Options:**
1. **Simple Bilateral** (cheapest, ~0.5ms)
2. **TAA Upscale** (better quality, ~1-2ms)
3. **FSR 2.0** (best quality, ~2-3ms) - if AMD library integrated

**Expected Savings:**
- Pixel count: 2,073,600 → 1,166,400 (56%)
- ReSTIR cost: 3-4ms → 2ms (-40%)
- Shadow ray cost: 6-8ms → 3-4ms (-45%)
- **Total savings:** 4-6ms

**Trade-off:** Slight quality loss (edges may blur), but generally imperceptible

---

## Phase 10: Billboard Shader Bug Fix (Low Priority)

**Status:** Not started (known issue)
**Issue:** Billboard mode runs at 0 FPS after time handling changes
**Estimated Time:** 2-4 hours
**Root Cause:** Physics timestep separation broke billboard shader time parameter

**Fix Needed:**
- Investigate how billboard shader uses time
- Ensure fixed physics timestep doesn't affect billboard rendering
- Likely a trivial constant buffer update

**Documentation:** Noted in `SESSION_SUMMARY_2025-10-09.md`

---

# 📚 DOCUMENTATION & ROADMAP REFERENCES

## Core Documentation Files

### Session Summaries (Chronological)
1. `SESSION_SUMMARY_2025-10-09.md` - Volumetric RT debugging session
2. `SESSION_SUMMARY_20251011_ReSTIR.md` - ReSTIR Phase 1 implementation

### Implementation Guides
1. `IMPLEMENTATION_QUICKSTART.md` - Phase-by-phase roadmap (Agility_SDI_DXR_MCP/agent/)
2. `EXECUTIVE_SUMMARY_PARTICLE_RT.md` - 100K particle RT feasibility analysis
3. `GPT5_RT_CONSULTATION_PROMPT.md` - RT enhancement questions for GPT-5
4. `GPT5_NON_RT_CONSULTATION_PROMPT.md` - Non-RT enhancement ideas

### Status Documents
1. `GAUSSIAN_INTEGRATION_STATUS.md` - 3D Gaussian splatting complete status
2. `VOLUMETRIC_RT_FIXES_SUMMARY.md` - Shadow rays and in-scattering fixes
3. `VOLUMETRIC_QUICK_REFERENCE.md` - Quick reference for RT features
4. `PARTICLE_RT_LIGHTING_FIXES.md` - Lighting integration fixes

### Technical References
- `agent/AdvancedTechniqueWebSearches/efficiency_optimizations/ReSTIR_Particle_Integration.md`
- `agent/AdvancedTechniqueWebSearches/efficiency_optimizations/BLAS_PERFORMANCE_GUIDE.md`
- `agent/AdvancedTechniqueWebSearches/DXR_ReSTIR_DI_GI_Guide.md`
- `agent/AdvancedTechniqueWebSearches/DXR_Inline_Ray_Tracing_RayQuery_Guide.md`

---

# 🎯 IMMEDIATE NEXT STEPS (Priority Order)

## 1. **Test ReSTIR Phase 1 Fix** (1-2 hours)
- Compile latest shader with procedural query loop fix
- Run with F7 enabled
- Check PIX: M should be 1-16, not stuck at 1
- Verify colors don't become muted
- Verify yellow indicator stays bright

**If Working:** Remove test write (M=12345), tune parameters
**If Broken:** Add debug counters, verify BVH binding, check particle range

---

## 2. **Implement ReSTIR Phase 2: Spatial Reuse** (16-24 hours)
- Once Phase 1 working and tested
- Expected 3-5× additional quality improvement
- Share reservoirs with 3-5 neighbors
- Spatial validation via shadow rays
- Expected M values: 50-100 (vs 16 current)

---

## 3. **Fix Billboard Shader** (2-4 hours)
- Quick win to restore billboard mode functionality
- Investigate time parameter handling
- Should be simple constant buffer fix

---

## 4. **Start Clustered BLAS Research** (4-8 hours)
- Begin designing 1000-cluster architecture
- Prototype spatial clustering algorithm
- Plan memory pooling strategy
- Target: Scale to 100K particles

---

## 5. **Implement Shader Execution Reordering** (8-12 hours)
- Can be done in parallel with ReSTIR Phase 2
- Expected 1.5-2× ray tracing speedup
- Ada Lovelace hardware feature (we have it!)
- Good ROI for moderate effort

---

# ⚠️ KNOWN ISSUES & TECHNICAL DEBT

## Critical Issues
1. **ReSTIR Phase 1:** Procedural primitive query not returning hits (fix applied, awaiting test)
2. **Billboard Mode:** 0 FPS after physics timestep changes
3. **Physical Emission:** Not properly integrated in Gaussian mode

## Minor Issues
1. **Status Bar:** Can get cluttered with all F-key indicators
2. **Debug Visualization:** Corner squares should be removed after testing
3. **Parameter Tuning:** Many magic numbers need proper tuning (shadow opacity, phase strength, etc.)

## Technical Debt
1. **Constant Buffer Size:** Had to switch from root constants to CBV due to 48 DWORD limit
2. **Memory Usage:** 126MB for ReSTIR buffers is high (can optimize later)
3. **Shader Complexity:** `particle_gaussian_raytrace.hlsl` is getting large (consider splitting)

---

# 📊 PERFORMANCE BUDGET ANALYSIS

## Current Performance @ 20K Particles (1920×1080)

| Pass | Time (ms) | % of 16.67ms | Rays |
|------|-----------|--------------|------|
| **Particle Physics** | 2-3 | 15% | 0 |
| **BLAS Build** | 2-3 | 15% | 0 |
| **TLAS Build** | 0.5-1 | 4% | 0 |
| **Gaussian Raytrace (baseline)** | 8-10 | 52% | 12.5M |
| **+ Shadow Rays (F5)** | +6-8 | +45% | +50M |
| **+ In-Scattering (F6)** | +3-4 | +20% | +20M |
| **+ ReSTIR Phase 1 (F7)** | +4-5 | +25% | +35M |
| **TOTAL (All ON)** | **26-34** | **170%** | **~118M** |

**Conclusion:** Over budget by ~10-17ms. Need optimizations to hit 60 FPS.

---

## Target Performance @ 100K Particles (1920×1080, 60 FPS)

With optimizations (Clustered BLAS, SER, ReSTIR Phase 2, 0.75× resolution):

| Pass | Time (ms) | % of 16.67ms | Optimization |
|------|-----------|--------------|--------------|
| **Particle Physics** | 3-4 | 20% | GPU-driven |
| **Clustered BLAS** | 8-10 | 55% | Memory pooling |
| **TLAS Build** | 1-2 | 8% | 1000 instances |
| **ReSTIR (0.75× res)** | 2-3 | 15% | Lower resolution |
| **Shadow Rays (SER)** | 3-4 | 20% | Hardware reordering |
| **Final Composite** | 1-2 | 8% | Bilateral upscale |
| **TOTAL** | **18-25** | **126%** | |

**Conclusion:** Still over budget by 1-8ms. Need aggressive quality/resolution trade-offs OR 30-45 FPS target.

**Acceptable Outcome:** 30-45 FPS with 100K particles and full quality is still a massive win.

---

# 🏆 SUCCESS METRICS

## MVP (Minimum Viable Product) - Phase 1
- [x] 20,000 particles visible with 3D Gaussian splatting
- [x] DXR 1.1 inline ray tracing working
- [x] Shadow rays for particle occlusion
- [x] Volumetric in-scattering
- [x] Phase function for forward scattering
- [x] Runtime toggles for all features
- [🟡] ReSTIR Phase 1 working (testing)

## V1 (Production Ready) - Phase 2-3
- [ ] ReSTIR Phase 2 (spatial reuse) working
- [ ] Shader Execution Reordering enabled
- [ ] Stable 30 FPS @ 20K particles with all features ON
- [ ] No visual artifacts (flickering, ghosting)

## V2 (Scaled) - Phase 4-5
- [ ] Clustered BLAS architecture for 100K particles
- [ ] Multi-bounce lighting (2 bounces)
- [ ] 30-45 FPS @ 100K particles
- [ ] Adaptive quality system

## V3 (Polish) - Phase 6-9
- [ ] NRD denoising integration
- [ ] Temporal accumulation for convergence
- [ ] Resolution scaling with upscaling
- [ ] 60 FPS @ 100K particles (with quality trade-offs)

---

**Document maintained by:** Claude (Graphics Engineering Agent)
**Last updated:** 2025-10-11
**Status:** Living document - update after each major milestone
