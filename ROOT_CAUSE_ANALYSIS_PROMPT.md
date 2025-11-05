# Root Cause Analysis Prompt: 2045 Particle Count GPU Crash

## Executive Summary

We are experiencing a **highly reproducible GPU crash (TDR timeout)** at exactly **2045 particles** in a DirectX 12 volumetric particle renderer using DXR 1.1 inline ray tracing. The crash occurs **identically** in two completely independent systems: a custom ReSTIR implementation (deprecated) and a probe grid volumetric lighting system (current). This suggests a **deep architectural issue** rather than application-level bugs.

**Key Characteristics:**
- **Instant crash** at 2045 particles (no gradual performance degradation)
- **High FPS (~150)** at 2044 particles immediately before crash
- **Identical behavior** across two different ray tracing implementations
- **Unaffected by complexity reductions** (grid size, ray count, iteration limits)
- **Specific to DXR acceleration structure traversal** (GPU device removed error)

## Hardware & Software Environment

**GPU:** NVIDIA GeForce RTX 4060 Ti (8GB VRAM, Ada Lovelace architecture)
**Driver:** 566.14 (latest)
**OS:** Windows 11 23H2 (WSL2 Ubuntu 22.04 build environment)
**API:** DirectX 12 with Agility SDK 1.614.1
**DXR Tier:** 1.1 (RayQuery inline ray tracing)
**Shader Model:** 6.5

**Critical Detail:** RTX 40-series (Ada Lovelace) has known mesh shader descriptor access bugs (workaround implemented). This GPU generation may have undocumented ray tracing quirks.

## System Architecture

### Particle Representation
- **128 bytes per particle** (position, velocity, temperature, radius, elongation)
- **Procedural primitives** (AABBs, 24 bytes each: min XYZ, max XYZ)
- **Anisotropic ellipsoid volumes** (not spheres - elongated along velocity)
- **Dynamic radius** (adaptive sizing 0.3×-3.0× based on camera distance)

### Acceleration Structure Pipeline
```
GPU Physics (128B particles)
    ↓
AABB Generation Compute Shader (24B AABBs)
    ↓
BLAS Build (Procedural Primitive Geometry, PREFER_FAST_BUILD)
    ↓
TLAS Build (Single instance, identity transform)
    ↓
RayQuery Traversal (while(q.Proceed()) loops in compute shader)
```

**Resource States:**
- AABB buffer: UNORDERED_ACCESS → UAV barrier → NON_PIXEL_SHADER_RESOURCE
- BLAS/TLAS: RAYTRACING_ACCELERATION_STRUCTURE state (no transitions)

### Two Crash Scenarios (Identical Behavior)

#### Scenario A: Custom ReSTIR Implementation (Deprecated)
- **Purpose:** Weighted reservoir sampling for light selection
- **Shader:** DXR raygen shader with TraceRay() calls
- **Reservoirs:** 2× R32G32B32A32_FLOAT buffers (63 MB each @ 2045 particles)
- **Spatial Grid:** 30³ cells covering 3000×3000×3000 units
- **Crash Point:** TraceRay() in raygen shader during reservoir update
- **Result:** Navy blue screen → GPU device removed (0x887A0006)

#### Scenario B: Probe Grid System (Current)
- **Purpose:** 32³ volumetric light probes for indirect illumination
- **Shader:** Compute shader with RayQuery inline ray tracing
- **Rays:** 16 rays per probe (Fibonacci sphere distribution)
- **Tracing:** `while(q.Proceed())` loop with procedural primitive intersection
- **Crash Point:** RayQuery traversal during probe update
- **Result:** Navy blue screen → GPU device removed (0x887A0006)

**CRITICAL OBSERVATION:** Both systems crash at **exactly 2045 particles** despite:
- Different ray tracing APIs (TraceRay vs RayQuery)
- Different shader stages (raygen vs compute)
- Different data structures (reservoirs vs probes)
- Different traversal patterns (light sampling vs volumetric gather)

This strongly suggests the issue is in the **shared acceleration structure traversal code** (NVIDIA driver or hardware scheduler).

## Suspected Root Cause: BVH Leaf Count Power-of-2 Bug

### Theory
At 2045 particles with 4 primitives/leaf (typical for PREFER_FAST_BUILD):
```
Leaf Count = (2045 + 3) / 4 = 512 leaves = 2^9
```

**512 leaves is a perfect power-of-2**, which may trigger:
1. **Hardware traversal stack overflow** (Ada Lovelace BVH traversal units)
2. **Driver BVH compression bug** at specific boundaries
3. **Memory alignment issue** in acceleration structure layout
4. **Undocumented Ada Lovelace silicon bug** similar to mesh shader descriptor bug

### Evidence Supporting This Theory

**Test 1: Power-of-2 Padding Workaround (Failed)**
```cpp
// In BuildBLAS():
uint32_t leafCount = (m_particleCount + 3) / 4;
if (isPowerOf2(leafCount) && leafCount >= 512) {
    aabbCount += 4;  // Push to 513 leaves
}
```
**Result:** Still crashes at 2045 particles. This suggests either:
- Leaf count calculation is wrong (driver uses different formula)
- Bug occurs at BVH *depth* not leaf count
- Bug is in TLAS traversal not BLAS

**Test 2: Complexity Reduction (Failed)**
- Reduced probe grid: 32³ → 16³ (87.5% reduction)
- Reduced rays per probe: 16 → 1 (93.75% reduction)
- Added RayQuery iteration limit: 1000 max iterations
- Added explicit TLAS UAV barrier before probe dispatch

**Result:** 2044/2045 boundary unchanged. This rules out:
- Computational bottleneck (would see gradual slowdown)
- Memory pressure (VRAM usage low at 2045 particles)
- Infinite loop timeout (iteration limit didn't help)

**Test 3: High FPS at 2044 Particles**
- Achieved ~150 FPS at 2044 particles with full complexity
- Instant crash at 2045 particles (no warning signs)

**Result:** This rules out performance-related TDR. The 5-second TDR watchdog isn't being triggered by slow execution - something is **instantly breaking** the GPU.

## Attempted Mitigations (All Failed Before Batching)

### 1. Power-of-2 Leaf Padding
**Implementation:** Add 4 dummy AABBs when leaf count is power-of-2
**Result:** No effect (still crashes at 2045)
**File:** `RTLightingSystem_RayQuery.cpp:557-574`

### 2. RayQuery Iteration Limiting
**Implementation:** `while(q.Proceed() && iterationCount < 1000)`
**Result:** No effect (probe effects became fainter, still crashes)
**File:** `shaders/probe_grid/update_probes.hlsl:247-281`

### 3. Explicit TLAS Barrier
**Implementation:** UAV barrier on TLAS before probe dispatch
**Result:** No effect
**File:** `Application.cpp:659-666`

### 4. Drastic Complexity Reduction
**Implementation:**
- Probe grid: 32³ → 16³
- Rays per probe: 16 → 1
- Shadow rays: Disabled

**Result:** No effect (2044/2045 boundary unchanged)

### 5. Spatial Interpolation Disable
**Implementation:** Disabled volumetric spatial blending
**Result:** No effect

## Current Mitigation: Particle Batching (In Progress)

Since we cannot fix the root cause, we're **mitigating** the bug by **avoiding the 2045 threshold entirely**.

### Batching Strategy
- Split particles into groups of **2000 particles per batch**
- Each batch gets **separate AABB buffer, BLAS, TLAS**
- Ray tracing systems trace against **all batch TLAS structures sequentially**

**For 2045 particles:**
- Batch 0: Particles 0-1999 (2000 particles) → 500 BVH leaves
- Batch 1: Particles 2000-2044 (45 particles) → 12 BVH leaves

**Neither batch reaches 512 leaves**, bypassing the suspected power-of-2 bug.

### Implementation Status (Phase 0.13.3)

**Completed:**
1. ✅ `AccelerationBatch` struct with per-batch resources (header)
2. ✅ `CreateBatchedAccelerationStructures()` - creates all batch resources at init
3. ✅ `GenerateAABBsForBatch()` - AABB generation with particle buffer offset
4. ✅ `BuildBatchBLAS()` - BLAS build using batch resources
5. ✅ `BuildBatchTLAS()` - TLAS build using batch resources
6. ✅ `ComputeLighting()` modified to loop over batches
7. ✅ `GetTLAS()` returns batch[0] for backward compatibility

**Remaining:**
8. ⏳ Update probe shader for multi-TLAS tracing
9. ⏳ Expand probe root signature (5 → 12 params for t2-t9 TLAS array)
10. ⏳ Update `ProbeGridSystem::UpdateProbes()` binding code
11. ⏳ Test at 2045, 4000, 10K particles

### Code References

**Batch Structure (`RTLightingSystem_RayQuery.h:92-104`):**
```cpp
struct AccelerationBatch {
    Microsoft::WRL::ComPtr<ID3D12Resource> aabbBuffer;
    Microsoft::WRL::ComPtr<ID3D12Resource> blas;
    Microsoft::WRL::ComPtr<ID3D12Resource> tlas;
    Microsoft::WRL::ComPtr<ID3D12Resource> blasScratch;
    Microsoft::WRL::ComPtr<ID3D12Resource> tlasScratch;
    Microsoft::WRL::ComPtr<ID3D12Resource> instanceDesc;
    uint32_t startIndex;
    uint32_t count;
    size_t blasSize;
    size_t tlasSize;
};
```

**Batch Build Loop (`RTLightingSystem_RayQuery.cpp:827-836`):**
```cpp
if (m_useBatching && !m_batches.empty()) {
    // BATCHED PATH: Build all batches
    for (auto& batch : m_batches) {
        GenerateAABBsForBatch(cmdList, particleBuffer, batch);
        BuildBatchBLAS(cmdList, batch);
        BuildBatchTLAS(cmdList, batch);
    }
}
```

**AABB Generation with Offset (`RTLightingSystem_RayQuery.cpp:584-588`):**
```cpp
// Bind particle buffer WITH OFFSET for this batch
// Each particle is 128 bytes (see Particle struct)
D3D12_GPU_VIRTUAL_ADDRESS particleAddr = particleBuffer->GetGPUVirtualAddress();
particleAddr += batch.startIndex * 128;
cmdList->SetComputeRootShaderResourceView(1, particleAddr);
```

**Multi-TLAS Probe Shader (Pending):**
```hlsl
// Replace single TLAS:
// OLD: RaytracingAccelerationStructure g_particleTLAS : register(t2);
// NEW:
RaytracingAccelerationStructure g_particleTLAS[8] : register(t2);

// Trace against all batches:
for (uint batchIdx = 0; batchIdx < g_numBatches; batchIdx++) {
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(g_particleTLAS[batchIdx], RAY_FLAG_NONE, 0xFF, ray);

    while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
            uint localIdx = q.CandidatePrimitiveIndex();
            uint globalIdx = (batchIdx * 2000) + localIdx;  // Batch offset
            // ... intersection test ...
        }
    }
}
```

## Questions for Expert Analysis

We need help identifying the **true root cause** of this issue:

### Question 1: BVH Leaf Count Calculation
Is our assumption of **4 primitives per leaf** for PREFER_FAST_BUILD correct? NVIDIA's actual implementation may use:
- Variable leaf sizes (2-8 primitives depending on spatial coherence)
- Different thresholds for procedural primitives vs triangles
- Cache line alignment (64-byte boundaries)

**How can we query the actual BVH leaf count from the driver?**

### Question 2: Ada Lovelace Hardware Quirks
The RTX 4060 Ti uses Ada Lovelace architecture (AD104 GPU). Known issues:
- Mesh shader descriptor table access bug (documented, workaround applied)
- **Are there undocumented ray tracing bugs at specific BVH configurations?**

NVIDIA's Ada whitepaper mentions **third-generation RT cores** with "enhanced BVH traversal". Could this have introduced new edge cases?

### Question 3: TLAS vs BLAS Traversal
Our padding workaround targeted BLAS leaf count, but the bug might be in:
- **TLAS traversal** (single instance, but maybe instance count matters?)
- **Combined TLAS+BLAS depth** (total tree depth limit?)
- **Memory layout alignment** (TLAS pointer alignment in Ada architecture?)

Should we try:
- Multiple dummy instances in TLAS?
- Different BLAS/TLAS build flags (PREFER_FAST_TRACE vs PREFER_FAST_BUILD)?
- BLAS update instead of rebuild (though risky)?

### Question 4: Procedural Primitive Intersection
We use **custom ray-ellipsoid intersection** in compute shaders:
```hlsl
if (q.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
    uint particleIdx = q.CandidatePrimitiveIndex();
    // Custom intersection test
    float3 oc = ray.Origin - particle.position;
    float b = dot(oc, ray.Direction);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - c;

    if (discriminant >= 0.0) {
        q.CommitProceduralPrimitiveHit(t);
    }
}
```

Could **CommitProceduralPrimitiveHit()** have driver bugs at specific primitive counts? Should we:
- Add validation (check t > ray.TMin && t < ray.TMax before committing)?
- Limit max hits per ray?
- Use different AABB generation (tighter bounds, padding)?

### Question 5: Memory Alignment and Resource States
DirectX 12 acceleration structures have strict alignment requirements:
- BLAS/TLAS: **64KB alignment** (D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT)
- Scratch buffers: **256-byte alignment** (D3D12_RAYTRACING_ACCELERATION_STRUCTURE_SCRATCH_BUFFER_BYTE_ALIGNMENT)

At 2045 particles:
- BLAS size: ~130KB
- TLAS size: ~1KB
- Scratch: ~150KB

**Could 2045 particles produce a specific size that hits a driver alignment bug?**

### Question 6: ReSTIR vs Probe Grid Commonality
Both systems crash identically despite different implementations. The **only shared code** is:
1. Particle AABB generation shader
2. BLAS/TLAS build in RTLightingSystem_RayQuery
3. NVIDIA driver BVH traversal

**What could cause identical failures in TraceRay() (raygen) and RayQuery (compute)?**
- Both use the same TLAS
- Both traverse procedural primitive AABBs
- Both use the same driver traversal code

But:
- ReSTIR uses DXR pipeline (state object, SBT)
- Probe grid uses inline ray tracing (compute shader)

This points to **low-level traversal bug**, not pipeline setup issue.

### Question 7: Batching as Root Cause Test
Our batching mitigation is also a **diagnostic tool**. If batching works:
- **Bug is count-specific** (2045 particles triggers it)
- **Bug is BLAS-specific** (keeping each BLAS under 2000 particles avoids it)
- **Bug is NOT memory size** (total GPU memory usage same with batching)

If batching still crashes:
- **Bug is elsewhere** (TLAS instance count? Total primitive count?)
- **Bug is race condition** (batching changes timing?)
- **Bug is in our application code** (we missed something)

## Diagnostic Data Collection

To help diagnose this, we can provide:

1. **PIX GPU Captures** (.wpix files)
   - Capture at 2044 particles (working)
   - Capture attempt at 2045 particles (crashes during capture)
   - Available in `PIX/Captures/` directory

2. **Buffer Dumps**
   - Particle data (positions, velocities, temperatures)
   - AABB data (min/max bounds)
   - Available via `--dump-buffers` command-line flag

3. **Build Logs**
   - DXC shader compilation output
   - BLAS/TLAS size queries
   - Power-of-2 padding logs

4. **GPU Diagnostics**
   - NVIDIA Nsight aftermath crash dumps (if configured)
   - Windows Event Viewer TDR logs
   - Driver internal state (if NVIDIA provides tools)

## Expected Batching Behavior

If batching successfully mitigates the issue:
- **2045 particles** (2 batches) → NO CRASH
- **4000 particles** (2 batches) → NO CRASH
- **10000 particles** (5 batches) → NO CRASH

This would confirm:
1. Issue is BLAS-size or primitive-count related
2. Keeping individual BLAS under 2000 particles is sufficient
3. No issues with multiple TLAS traversal

If batching STILL crashes:
- Need to investigate TLAS instance count
- Need to investigate total scene primitive count limits
- May need more aggressive batching (1000 particles per batch?)

## Request for Expert Input

**We need help with:**

1. **NVIDIA Driver Internals:**
   - Known issues with Ada Lovelace RT cores?
   - BVH leaf count calculation for procedural primitives?
   - Debug tools to inspect BVH structure?

2. **DirectX 12 Best Practices:**
   - Are we building acceleration structures correctly?
   - Should we use BLAS updates instead of rebuilds?
   - Any known issues with PREFER_FAST_BUILD at specific sizes?

3. **Alternative Debugging Approaches:**
   - Binary search on particle count (2030, 2038, 2042, etc.)?
   - Test with different AABB sizes (tight vs loose bounds)?
   - Test with simpler intersection (spheres instead of ellipsoids)?

4. **Root Cause Theories:**
   - **What specific hardware/driver condition would cause identical crashes in two different ray tracing implementations?**
   - **Why would high FPS at 2044 particles instantly become a crash at 2045?**
   - **What could make a count threshold (2045) more important than computational complexity?**

## Additional Context

**Project:** PlasmaDX-Clean (volumetric particle renderer)
**Purpose:** Black hole accretion disk simulation with physically-based rendering
**Scale:** Targets 10K-100K particles with real-time ray tracing
**Performance:** Currently limited to 2044 particles due to this bug

**Documentation:**
- `BATCHING_STATUS.md` - Current implementation status
- `BATCHING_NEXT_STEPS.md` - Remaining work for batching mitigation
- `PROBE_GRID_2045_CRASH_INVESTIGATION.md` - Full investigation history

**Repository:** Available on request for detailed code review

---

**Thank you for any insights you can provide!** This issue has blocked development for several sessions and we're eager to understand the true root cause, whether it's our code, the driver, or the hardware.
