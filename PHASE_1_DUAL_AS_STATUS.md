# Phase 1 Dual AS Architecture - Status Report

**Date:** 2025-11-05
**Branch:** 0.13.9
**Status:** PARTIAL SUCCESS - All particles visible, but runtime controls bug identified

---

## Problem Statement

**Ada Lovelace 2045 Particle Crash Bug:**
- RTX 4060 Ti GPU crashes at exactly 2045 particles with pure procedural BLAS
- Root cause: NVIDIA hardware bug with power-of-2 BVH leaf boundaries
- Previous workarounds (mixed geometry, batching) all failed

---

## Phase 1 Solution: Dual Acceleration Structure Architecture

Split particles into TWO separate BLAS (each under 2044 limit), then combine into single TLAS for rendering.

### Architecture

**For >2044 particles:**
1. **Probe Grid BLAS** - Particles 0-2043 (2044 total)
2. **Direct RT BLAS** - Particles 2044+ (overflow)
3. **Combined TLAS** - Single TLAS with 2 instances pointing to both BLAS

**Gaussian renderer traces the combined TLAS** → sees all particles via automatic multi-instance traversal.

---

## What's Working ✅

1. **No crash at any particle count** - Each BLAS stays under 2044 limit
2. **All particles visible** - Combined TLAS correctly references both BLAS instances
3. **Correct framerate** - ~100 FPS @ 10K particles (no longer suspiciously high at 300 FPS)
4. **Particle-to-particle RT lighting working** - Likely because overflow particles now rendered

### Key Implementation Files

**src/lighting/RTLightingSystem_RayQuery.h:**
- Added `AccelerationStructureSet` struct (lines 101-113)
- Added `PROBE_GRID_PARTICLE_LIMIT = 2044` constant (line 115)
- Members: `m_probeGridAS`, `m_directRTAS` (lines 183-184)
- `GetTLAS()` returns combined TLAS for >2044 particles (lines 68-75)

**src/lighting/RTLightingSystem_RayQuery.cpp:**
- `CreateAccelerationStructureSet()` - Creates resources for one AS set (lines 207-365)
- `GenerateAABBs_Dual()` - Generates AABBs for both sets (lines 605-713)
- `BuildBLAS_ForSet()` - Builds BLAS with particle offset (lines 715-759)
- `BuildTLAS_ForSet()` - Builds single-instance TLAS (lines 761-807)
- `BuildCombinedTLAS()` - **KEY FUNCTION** - Builds 2-instance TLAS (lines 809-861)
- `ComputeLighting()` - Main pipeline (lines 912-969)

---

## Current Bug ❌ - Runtime Controls Don't Affect All Particles

**Symptom:** Changing particle size, adaptive radius, etc. affects some particles but not others inconsistently.

**Root Cause Identified:**

The Direct RT AABB generation is reading the **WRONG particles** from the buffer!

**Current behavior:**
```cpp
// Direct RT dispatch at line 692
directRTConstants.particleCount = totalParticleCount;  // 10,000
// Shader reads: particles[0], particles[1], ... particles[9999]
// Writes to: DirectRTAS AABB buffer indices 0-9999
// BLAS reads from: AABB buffer offset 2044 (particles 2044-9999 of the AABB buffer)
```

**The bug:**
- Direct RT generates AABBs for particles **0-9999** (duplicates!)
- But BLAS reads from offset 2044 in the AABB buffer
- So Direct RT instance shows **duplicates of particles 0-2043** (plus extras)
- NOT the actual overflow particles 2044-9999

**Why runtime controls fail:**
- Probe Grid AABBs: Generated from particles 0-2043 ✅ Correct
- Direct RT AABBs: Generated from particles 0-9999 ❌ Should be 2044-9999
- When you adjust particle size, Probe Grid updates correctly, Direct RT shows stale/wrong particles

---

## The Fix (Phase 1.5) - Add Particle Offset to Shader

### Step 1: Update Shader Constants

**shaders/dxr/generate_particle_aabbs.hlsl (lines 7-21):**

```hlsl
cbuffer AABBConstants : register(b0)
{
    uint particleCount;
    float particleRadius;

    // Phase 1.5 - CRITICAL FIX
    uint particleOffset;           // NEW: Start reading from this particle index
    uint padding1;                 // Alignment

    // Phase 1.5 Adaptive Particle Radius
    uint enableAdaptiveRadius;
    float adaptiveInnerZone;
    float adaptiveOuterZone;
    float adaptiveInnerScale;
    float adaptiveOuterScale;
    float densityScaleMin;
    float densityScaleMax;
    float padding2;
};
```

### Step 2: Update Shader Logic

**shaders/dxr/generate_particle_aabbs.hlsl (line 44):**

```hlsl
// OLD:
Particle p = particles[particleIndex];

// NEW:
Particle p = particles[particleIndex + particleOffset];
```

### Step 3: Update CPU-side Constants

**src/lighting/RTLightingSystem_RayQuery.cpp (lines 570-593 - Probe Grid):**

```cpp
struct AABBConstants {
    uint32_t particleCount;
    float particleRadius;
    uint32_t particleOffset;        // NEW: 0 for probe grid
    uint32_t padding1;
    uint32_t enableAdaptiveRadius;
    float adaptiveInnerZone;
    float adaptiveOuterZone;
    float adaptiveInnerScale;
    float adaptiveOuterScale;
    float densityScaleMin;
    float densityScaleMax;
    float padding2;
} probeGridConstants = {
    probeGridCount,                 // 2044
    m_particleRadius,
    0,                              // Offset = 0 (start at particle 0)
    0,
    // ... rest of fields
};
```

**src/lighting/RTLightingSystem_RayQuery.cpp (lines 680-703 - Direct RT):**

```cpp
struct AABBConstants {
    uint32_t particleCount;
    float particleRadius;
    uint32_t particleOffset;        // NEW: 2044 for Direct RT
    uint32_t padding1;
    uint32_t enableAdaptiveRadius;
    float adaptiveInnerZone;
    float adaptiveOuterZone;
    float adaptiveInnerScale;
    float adaptiveOuterScale;
    float densityScaleMin;
    float densityScaleMax;
    float padding2;
} directRTConstants = {
    directRTCount,                  // 7956 (not totalParticleCount!)
    m_particleRadius,
    PROBE_GRID_PARTICLE_LIMIT,      // Offset = 2044 (skip first 2044 particles)
    0,
    // ... rest of fields
};
```

**CRITICAL:** Change `particleCount` from `totalParticleCount` to `directRTCount` (actual overflow count, not total).

### Step 4: Recompile Shader

```bash
dxc.exe -T cs_6_5 -E main shaders/dxr/generate_particle_aabbs.hlsl \
    -Fo build/bin/Debug/shaders/dxr/generate_particle_aabbs.dxil
```

---

## Memory Optimization (Phase 1.5 Bonus)

Once particle offset is working, we can reduce Direct RT AABB buffer size:

**Current (wasteful):**
```cpp
// Allocates for ALL particles
CreateAccelerationStructureSet(m_directRTAS, m_particleCount, "DirectRTAS");
```

**Optimized:**
```cpp
// Allocates only for overflow particles
CreateAccelerationStructureSet(m_directRTAS, directRTCount, "DirectRTAS");
```

This saves ~49 KB @ 10K particles (eliminates duplicate 0-2043 AABBs).

---

## Testing Checklist

After implementing the fix:

1. **2044 particles** - Should see full spiral (single BLAS, probe grid only)
2. **3922 particles** - Should see full spiral (combined TLAS, 1878 overflow)
3. **10000 particles** - Should see full spiral (combined TLAS, 7956 overflow)
4. **Runtime controls** - Adjust particle size → all particles resize uniformly
5. **Adaptive radius** - Toggle on/off → all particles respond
6. **Camera zoom** - Particles at all distances resize correctly

---

## Phase 2 Roadmap (After Fix)

1. **Re-enable probe grid system** - Volumetric light scattering for first 2044 particles
2. **Fix probe grid bounds** - Expand from cubic volume at origin to full world coverage
3. **Fix probe grid flashing** - Temporal accumulation with ping-pong buffers
4. **Tune probe grid intensity** - Balance with direct RT lighting
5. **Integrate dual lighting** - Probe grid GI for 0-2043, direct RT for 2044+

---

## Performance Data (10K particles @ 1440p, RTX 4060 Ti)

**Before dual AS:** 300+ FPS (suspiciously high, only 2044 particles rendered)
**After dual AS:** ~100 FPS (correct for 10K particles with RT lighting)

**Memory usage:**
- Probe Grid AS: 52 KB BLAS, 2 KB TLAS, 49 KB AABBs, 32 KB lighting = **135 KB**
- Direct RT AS: 252 KB BLAS, 2 KB TLAS, 240 KB AABBs, 160 KB lighting = **654 KB**
- Combined TLAS: 2 KB (2 instances)
- **Total:** ~791 KB for dual AS (acceptable overhead)

---

## Key Learnings

1. **Instance traversal is free** - Combined TLAS with 2 instances has negligible overhead
2. **BLAS offset works** - Reading from AABB buffer offset 2044 successfully splits particles
3. **Shader needs particle offset** - Can't just offset BLAS read, must offset particle read too
4. **Buffer overrun was critical** - Allocating DirectRTAS for overflow count caused crash at 3922

---

## Next Session TODO

1. ✅ Read this document
2. Add `particleOffset` to shader constant buffer
3. Update shader to read `particles[particleIndex + particleOffset]`
4. Update CPU constants for both probe grid (offset=0) and direct RT (offset=2044)
5. Change Direct RT `particleCount` from `totalParticleCount` to `directRTCount`
6. Recompile shader
7. Test runtime controls (particle size, adaptive radius, etc.)
8. If working, optimize Direct RT AABB buffer size

---

**End of Phase 1 Status Report**
