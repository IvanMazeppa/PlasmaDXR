# Probe Grid 2045 Particle Crash - Investigation Summary

**Date:** 2025-11-04
**Status:** UNRESOLVED - Crashes at 2045+ particles with probe grid enabled
**Branch:** 0.13.2

---

## The Problem

**Symptom:** Application crashes at 2045+ particles (and other particle counts) when probe grid is enabled
**Crash Pattern:**
- ~5 second pause after frame 0
- Navy blue screen flash
- TDR timeout (GPU hang/driver reset)
- Crashes even with BVH power-of-2 padding applied

**Key Observation:** Disabling probe grid completely → NO CRASH at any particle count

---

## What We've Tried (Chronological)

### Attempt 1: BVH Power-of-2 Leaf Boundary Fix (FAILED)

**Hypothesis:** NVIDIA driver bug when BVH leaf count is exactly power of 2

**Analysis:**
```
2045 particles with 4 prims/leaf:
- 2045 AABBs → ceil(2045/4) = 512 leaves (2^9) ← POWER OF 2!
```

**Fix Applied:**
- Added +1 AABB padding (WRONG - still 512 leaves)
- Corrected to +4 AABB padding (2049 AABBs → 513 leaves)

**Result:** STILL CRASHES despite 513 leaves (not power of 2)

**Files Modified:**
- `src/lighting/RTLightingSystem_RayQuery.cpp:206-208` - Allocate +4 AABBs
- `src/lighting/RTLightingSystem_RayQuery.cpp:385-402` - Padding logic in BuildBLAS

**Log Evidence:**
```
[WARN] BVH leaf count 512 is power-of-2 (particles=2045), adding 4 padding AABBs → 513 leaves
```
- Padding is working
- Still crashes

**Conclusion:** Either the padding doesn't avoid the driver bug, or the crash isn't from the BVH structure

---

### Attempt 2: Disable Probe Grid (SUCCESS)

**Test:** Comment out probe grid update call entirely

**Result:** NO CRASH at 2045+ particles

**Files Modified:**
- `src/core/Application.cpp:659-686` - Commented out UpdateProbes call

**Conclusion:** The probe grid shader/system is definitely the culprit, NOT the BVH structure alone

---

### Attempt 3: Reduce Rays Per Probe (FAILED)

**Hypothesis:** Too many ray queries causing timeout

**Original:** 64 rays per probe × 8,192 active probes = 524K rays/frame
**Reduced:** 16 rays per probe × 8,192 active probes = 131K rays/frame (4× reduction)

**Result:** STILL CRASHES (4× reduction not enough)

**Files Modified:**
- `src/lighting/ProbeGridSystem.h:168` - Changed m_raysPerProbe from 64 → 16

**Conclusion:** Ray count reduction alone doesn't solve the issue

---

## Current Understanding

### What We Know:

1. **Crash is specific to probe grid:**
   - RT lighting alone at 2045+ particles: ✅ WORKS
   - Probe grid enabled at 2045+ particles: ❌ CRASHES

2. **Crash pattern indicates GPU hang:**
   - 5-second pause = Windows TDR timeout
   - Navy blue screen = GPU device removed
   - Happens after completing frame 0

3. **Not purely a power-of-2 bug:**
   - Crashes at multiple particle counts (not just 2045)
   - Crashes even with 513 BVH leaves (not power of 2)

4. **Probe buffer shows all zeros:**
   - Irradiance data never computed
   - Suggests shader never completes successfully

### What We Don't Know:

1. **Why the probe shader hangs:**
   - Infinite loop in `while(q.Proceed())`?
   - Invalid TLAS resource state?
   - Resource binding mismatch?
   - Driver bug with specific RayQuery pattern?

2. **Why it works at 2044 but crashes at 2045:**
   - If not BVH power-of-2, what's the threshold?
   - Thread group boundary (33 groups)?
   - Dispatch size issue?

---

## Potential Root Causes

### Theory 1: RayQuery Infinite Loop

**Evidence:**
- `while(q.Proceed())` in shader can loop indefinitely
- No timeout protection in shader
- TDR timeout suggests infinite loop

**How to Test:**
- Add iteration counter with max limit (e.g., 1000 iterations)
- Early break if counter exceeded
- Log warning if limit hit

### Theory 2: TLAS Resource State Mismatch

**Evidence:**
- TLAS built with UAV barrier in RT lighting
- Immediately used by probe grid shader
- No explicit state transition

**How to Test:**
- Add explicit UAV barrier before probe grid dispatch
- Verify TLAS is in RAYTRACING_ACCELERATION_STRUCTURE state

### Theory 3: Descriptor Heap Exhaustion

**Evidence:**
- Probe grid allocates many descriptors
- May exceed heap limits at high particle counts

**How to Test:**
- Log descriptor heap usage
- Check for allocation failures

### Theory 4: Shader Compilation/Validation Issue

**Evidence:**
- Shader compiled at build time, not runtime
- May have invalid bytecode for certain inputs

**How to Test:**
- Recompile shader with debug info
- Use PIX shader debugger to step through execution

---

## Proposed Solutions

### Option 1: Particle Batching (RECOMMENDED)

**Concept:** Split particles into groups of ~2000, each with separate BLAS/TLAS

**Advantages:**
- Avoids 2045 threshold entirely
- Each batch stays well below problematic size
- Scalable to 100K+ particles (50 batches × 2000 particles)

**Implementation:**
```cpp
struct ParticleBatch {
    ID3D12Resource* aabbBuffer;
    ID3D12Resource* blas;
    ID3D12Resource* tlas;
    uint32_t particleStart;
    uint32_t particleCount;
};

std::vector<ParticleBatch> batches;
const uint32_t BATCH_SIZE = 2000;
```

**Probe Grid Changes:**
- Loop over all batches when tracing rays
- Each probe traces against multiple TLAS structures
- Accumulate lighting from all batches

**Memory Impact:**
- 50 batches × ~50 KB per BLAS = 2.5 MB (negligible with 8GB VRAM)
- No significant performance impact (TLAS traversal is fast)

**Files to Modify:**
1. `src/lighting/RTLightingSystem_RayQuery.h` - Add batching structures
2. `src/lighting/RTLightingSystem_RayQuery.cpp` - CreateAccelerationStructures, BuildBLAS, BuildTLAS
3. `shaders/probe_grid/update_probes.hlsl` - Add multi-TLAS support (or bind via descriptor table)

**Estimated Effort:** 4-6 hours

---

### Option 2: Limit Probe Updates Per Frame

**Concept:** Only update N probes per frame (not 8,192)

**Advantages:**
- Reduces total ray queries per frame
- May stay under TDR threshold

**Implementation:**
```cpp
// Update only 1024 probes per frame (instead of 8192)
const uint32_t PROBES_PER_FRAME = 1024;
uint32_t probeOffset = (frameIndex * PROBES_PER_FRAME) % TOTAL_PROBES;
```

**Disadvantages:**
- Slower convergence (32 frames for full update instead of 4)
- May still crash if single probe trace times out

**Estimated Effort:** 1-2 hours

---

### Option 3: Use TraceRay Instead of RayQuery

**Concept:** Replace inline RayQuery with full DXR pipeline (raygen/hit/miss shaders)

**Advantages:**
- Hardware BVH traversal with automatic culling
- Better performance for dense scenes
- May avoid RayQuery-specific driver bugs

**Disadvantages:**
- Requires shader binding table (SBT) setup
- More complex resource management
- Hit shaders need to return lighting data

**Estimated Effort:** 8-12 hours (significant refactor)

---

### Option 4: Aggressive Ray Culling

**Concept:** Only trace rays for probes near particles

**Implementation:**
```hlsl
// Skip probes far from all particles
float minDistanceToParticles = GetMinDistanceToParticles(probePos);
if (minDistanceToParticles > 500.0) {
    // Probe too far, use ambient or skip
    g_probes[probeLinearIdx].irradiance[0] = float3(0.1, 0.1, 0.1);
    return;
}
```

**Advantages:**
- Reduces ray count dramatically
- May avoid timeout

**Disadvantages:**
- Requires particle spatial structure (grid/octree)
- Added complexity

**Estimated Effort:** 6-8 hours

---

### Option 5: Switch to Voxel Grid

**Concept:** Abandon ray tracing, use compute shader voxelization

**Advantages:**
- Predictable performance
- No ray tracing bugs

**Disadvantages:**
- Loses zero-atomic-contention benefit
- Reintroduces atomic issues from Volumetric ReSTIR

**Estimated Effort:** 12-16 hours (major redesign)

---

## Recommended Next Steps

### Immediate (1-2 hours):

1. **Add RayQuery Iteration Limit:**
   ```hlsl
   uint iterationCount = 0;
   const uint MAX_ITERATIONS = 1000;
   while (q.Proceed() && iterationCount < MAX_ITERATIONS) {
       iterationCount++;
       // ... existing code
   }
   if (iterationCount >= MAX_ITERATIONS) {
       // Log error somehow (can't printf in shader)
       totalIrradiance = float3(1.0, 0.0, 0.0); // Red = timeout
   }
   ```

2. **Add Explicit TLAS Barrier:**
   ```cpp
   // After RT lighting, before probe grid
   D3D12_RESOURCE_BARRIER tlasBarrier = {};
   tlasBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
   tlasBarrier.UAV.pResource = m_rtLighting->GetTLAS();
   cmdList->ResourceBarrier(1, &tlasBarrier);
   ```

3. **Test with 1 Ray Per Probe:**
   - Change `m_raysPerProbe = 1` (extreme test)
   - If still crashes, not a ray count issue

### Short-Term (4-8 hours):

4. **Implement Particle Batching (Option 1):**
   - Most robust solution
   - Scalable to 100K+ particles
   - Avoids all power-of-2 thresholds

5. **Add Diagnostic Logging:**
   - Log TLAS GPU address before probe dispatch
   - Log resource states
   - Verify all bindings are non-null

### Long-Term (1-2 weeks):

6. **Consider TraceRay Pipeline (Option 3):**
   - If batching doesn't work
   - More performant for large particle counts
   - May avoid driver bugs entirely

---

## Files Reference

### Core Systems:
- `src/lighting/ProbeGridSystem.h/cpp` - Probe grid implementation
- `src/lighting/RTLightingSystem_RayQuery.h/cpp` - BVH and TLAS management
- `src/core/Application.cpp:659-686` - Probe grid dispatch call

### Shaders:
- `shaders/probe_grid/update_probes.hlsl` - Main probe update compute shader
- `shaders/particles/particle_gaussian_raytrace.hlsl:608-670` - SampleProbeGrid function

### Documentation:
- `BVH_POWER_OF_2_BUG_FIX.md` - Initial BVH padding attempt
- `PROBE_GRID_FAILURE_ANALYSIS.md` - Earlier analysis (pre-BVH fix)

---

## Particle Batching Implementation Sketch

```cpp
// RTLightingSystem_RayQuery.h
struct AccelerationBatch {
    ComPtr<ID3D12Resource> aabbBuffer;
    ComPtr<ID3D12Resource> blas;
    ComPtr<ID3D12Resource> tlas;
    ComPtr<ID3D12Resource> blasScratch;
    ComPtr<ID3D12Resource> tlasScratch;
    ComPtr<ID3D12Resource> instanceDesc;
    uint32_t startIndex;
    uint32_t count;
};

std::vector<AccelerationBatch> m_batches;
const uint32_t BATCH_SIZE = 2000;

// Create batches
void CreateBatchedAccelerationStructures() {
    uint32_t numBatches = (m_particleCount + BATCH_SIZE - 1) / BATCH_SIZE;
    m_batches.resize(numBatches);

    for (uint32_t i = 0; i < numBatches; i++) {
        uint32_t start = i * BATCH_SIZE;
        uint32_t count = min(BATCH_SIZE, m_particleCount - start);

        // Create AABB buffer for this batch
        size_t aabbSize = (count + 4) * 24; // +4 for padding
        m_batches[i].aabbBuffer = CreateBuffer(aabbSize);
        m_batches[i].startIndex = start;
        m_batches[i].count = count;

        // Create BLAS/TLAS for this batch
        // ... (similar to existing code, but per-batch)
    }
}

// Build all batches
void BuildBatchedBLAS(ID3D12GraphicsCommandList4* cmdList, ID3D12Resource* particleBuffer) {
    for (auto& batch : m_batches) {
        GenerateAABBsForBatch(cmdList, particleBuffer, batch);
        BuildBLASForBatch(cmdList, batch);
        BuildTLASForBatch(cmdList, batch);
    }
}
```

```hlsl
// Shader changes: Loop over batches (or use descriptor array)
RaytracingAccelerationStructure g_particleTLAS_batch0 : register(t2);
RaytracingAccelerationStructure g_particleTLAS_batch1 : register(t3);
// ... (or use descriptor table with unbounded array)

// Trace against all batches
for (uint batchIdx = 0; batchIdx < g_numBatches; batchIdx++) {
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(GetBatchTLAS(batchIdx), RAY_FLAG_NONE, 0xFF, ray);

    while (q.Proceed()) {
        // ... existing intersection logic
    }

    if (q.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT) {
        // Accumulate lighting from this batch
        totalIrradiance += ComputeParticleLighting(...);
    }
}
```

---

## Test Plan

### Phase 1: Verify Batching Works
1. Implement batching for 2045 particles (2 batches: 2000 + 45)
2. Test at 2045 particles with probe grid enabled
3. Expected: NO CRASH

### Phase 2: Scale Testing
1. Test at 4000 particles (2 batches: 2000 + 2000)
2. Test at 10,000 particles (5 batches)
3. Test at 100,000 particles (50 batches)
4. Expected: Stable at all counts, <5% performance impact

### Phase 3: Quality Verification
1. Compare probe lighting output vs non-batched (at 2044 particles)
2. Verify no visual artifacts at batch boundaries
3. Check memory usage (should be <10 MB for 50 batches)

---

## Conclusion

**The probe grid crashes at 2045+ particles due to GPU hang/TDR timeout.**

**Root cause remains unclear despite BVH padding fix.**

**Particle batching (~2000 per batch) is the most promising solution:**
- Avoids all problematic thresholds
- Scalable to 100K+ particles
- Minimal memory/performance overhead
- 4-6 hour implementation

**Alternative: Aggressively reduce probe updates per frame (1024 instead of 8192) as quick test.**

---

**Last Updated:** 2025-11-04 19:30
**Status:** Investigation ongoing, batching recommended
