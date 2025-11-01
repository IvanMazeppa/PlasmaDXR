# Volumetric ReSTIR - 2045 Particle Bug Investigation Summary

**Date**: 2025-11-01 01:45
**Branch**: 0.12.6
**Status**: CRITICAL BUG - Unresolved after multiple fix attempts
**Symptom**: Exactly 2045+ particles causes 3-second GPU stall on frame 1, then complete freeze/crash on frame 2+

---

## Executive Summary

A **100% reproducible bug** occurs at exactly **2045 particles** (2044 works, 2045+ fails). The bug manifests as:
1. **Frame 1**: 3-second GPU stall during `WaitForGPU()` (visible in logs)
2. **Frame 2+**: Complete freeze (frozen image) OR TDR crash depending on particle count
3. **Exact threshold**: 2044 works perfectly, 2045 fails, consistent across all tests

**Root Cause**: Still unknown after multiple fix attempts. Bug is NOT race conditions, NOT loop bounds, NOT atomics.

---

## Critical Evidence

### The 3-Second GPU Stall (Smoking Gun)

**Logs show consistent pattern:**
```
[01:43:57] VolumetricReSTIR: About to WaitForGPU
[01:44:00] VolumetricReSTIR: WaitForGPU completed - frame finished!
```

**3-second delay on frame 1** at 2045 particles, but **no delay at 2044 particles**. This proves:
- GPU is executing something extremely expensive
- Not a hard crash (frame completes eventually)
- The expensive operation scales with particle count in a non-linear way
- 2048 = 2^11 is likely a GPU scheduling boundary

### Dispatch Calculations

| Particles | Thread Groups | Total Threads | Active | Wasted |
|-----------|---------------|---------------|--------|--------|
| 2044      | 32            | 2048          | 2044   | 4      |
| 2045      | 32            | 2048          | 2045   | 3      |

**Both dispatch identical thread group counts** - rules out dispatch size as the issue.

### Shader Execution Flow

```
PopulateVolumeMip2 (32 thread groups, 2048 threads)
  ↓
Triple nested loop per thread (max 8×8×8 = 512 voxels per particle)
  ↓
InterlockedMax atomic writes to 64³ volume texture
  ↓
3-second GPU stall during WaitForGPU
```

---

## What We Know (Facts)

### Confirmed Working
✅ **2044 particles** - No stall, no freeze, runs perfectly
✅ **Frame 1 at 2045** - Completes (after 3-second stall)
✅ **All infrastructure** - Shaders compile, PSOs bind, resources created
✅ **DLSS integration** - Writes to correct textures
✅ **Descriptor management** - Pre-allocated, no heap exhaustion

### Confirmed Failing
❌ **2045+ particles** - 3-second stall on frame 1, freeze/crash on frame 2+
❌ **Frame 2+ at 2045** - Complete freeze (frozen image) or TDR crash
❌ **Exact threshold** - 2044 vs 2045, no variance across tests

### Key Observations
1. **Power-of-2 boundary**: 2048 = 2^11, crash at 2045 (3 below boundary)
2. **GPU scheduler behavior**: Different at exactly this thread count
3. **No D3D12 errors**: Debug layer shows no validation errors
4. **No shader compilation errors**: All shaders compile successfully
5. **Stall is compute-bound**: Happens during PopulateVolumeMip2 compute dispatch

---

## Fix Attempts (Chronological)

### Attempt 1: Resolution Reduction (❌ Failed)
**Hypothesis**: Too many shader invocations at native resolution
**Action**: Reduced ReSTIR from 2560×1440 to 640×360 (1/4 resolution)
**Result**: Still crashed at exactly 2045 particles
**Conclusion**: Not a total invocation count issue

### Attempt 2: Voxel Write Limit (❌ Failed)
**Hypothesis**: Triple nested loop writes too many voxels
**Action**: Clamped AABB to 8×8×8 voxels per particle (512 max)
**Result**: Still crashed at exactly 2045 particles
**Conclusion**: Not a voxel write count issue

### Attempt 3: Atomic Operations (❌ Failed)
**Hypothesis**: Race conditions from non-atomic UAV writes
**Action**: Changed to R32_UINT with InterlockedMax for race-free writes
**Result**: Still crashed at exactly 2045 particles (3-second stall persists)
**Conclusion**: Not a race condition issue (though atomics are still good to have)

### Attempt 4: Loop Bounds Validation (❌ Failed)
**Hypothesis**: Invalid loop bounds from particles outside volume
**Action**: Added `if (any(voxelMin > voxelMax)) return;` guard
**Result**: Still crashed at exactly 2045 particles (3-second stall persists)
**Conclusion**: Not invalid loop bounds

### Attempt 5: Isolation Test (✅ Confirmed Bug Location)
**Action**: Disabled PopulateVolumeMip2 entirely (commented out call)
**Result**: No crash at 10K particles, but output texture bug remains
**Conclusion**: PopulateVolumeMip2 is definitely the source of the crash

---

## Technical Details

### PopulateVolumeMip2 Shader

**Purpose**: Splat particle density into 64³ voxel grid for transmittance estimation

**Algorithm**:
```hlsl
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint particleIdx = dispatchThreadID.x;
    if (particleIdx >= g_particleCount) return;

    Particle particle = g_particles[particleIdx];

    // Compute AABB in voxel space
    int3 voxelMin = WorldToVoxel(particle.position - particle.radius);
    int3 voxelMax = WorldToVoxel(particle.position + particle.radius);

    // Clamp to [0, 63]
    voxelMin = max(voxelMin, int3(0, 0, 0));
    voxelMax = min(voxelMax, int3(63, 63, 63));

    // Guard against invalid bounds
    if (any(voxelMin > voxelMax)) return;

    // Limit to 8×8×8 voxels per particle
    // ... AABB size limiting logic ...

    // Triple nested loop
    for (int z = voxelMin.z; z <= voxelMax.z; z++) {
        for (int y = voxelMin.y; y <= voxelMax.y; y++) {
            for (int x = voxelMin.x; x <= voxelMax.x; x++) {
                // Compute density
                float density = ComputeDensityContribution(particle, voxelCenter);

                // Atomic write (race-free)
                if (density > 0.0001) {
                    uint densityAsUint = asuint(density);
                    uint originalValue;
                    InterlockedMax(g_volumeTexture[int3(x,y,z)], densityAsUint, originalValue);
                }
            }
        }
    }
}
```

**What's Happening**:
- 32 thread groups × 64 threads = 2048 threads total
- At 2045 particles: threads 0-2044 process particles, threads 2045-2047 early return
- Each thread writes up to 512 voxels (8×8×8 max)
- Total worst case: 2045 × 512 = 1,047,040 atomic writes

### Volume Texture Specifications

**Format**: R32_UINT (changed from R16_FLOAT for atomic operations)
**Resolution**: 64³ voxels = 262,144 voxels total
**Size**: 1 MB (4 bytes per voxel)
**World Bounds**: [-1500, -1500, -1500] to [+1500, +1500, +1500]
**Voxel Size**: 46.875 units per voxel (3000 / 64)

### Resource State Flow

**Frame 1**:
```
Created: D3D12_RESOURCE_STATE_UNORDERED_ACCESS
Skip SRV→UAV transition (m_volumeFirstFrame = true)
Clear: ClearUnorderedAccessViewFloat (zeros)
Dispatch: PopulateVolumeMip2 (32 thread groups) → 3-SECOND STALL HERE
UAV Barrier: Sync writes
Transition: UAV→SRV for reading
```

**Frame 2** (at 2045 particles):
```
Current: D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
Transition: SRV→UAV
Clear: ClearUnorderedAccessViewFloat
Dispatch: PopulateVolumeMip2 → FREEZE/CRASH HERE
```

---

## Hypotheses (Ranked by Probability)

### 1. GPU Cache/Memory Conflict (HIGH)
**Theory**: At exactly 2045 particles, memory access pattern creates L2 cache thrashing or memory bank conflicts

**Evidence**:
- 3-second stall suggests memory system bottleneck
- 2048 boundary aligns with GPU cache line/bank size
- Atomic operations to same voxels from different threads serialize

**How to test**:
- PIX GPU capture and analyze memory bandwidth/cache misses
- Check if changing volume resolution from 64³ changes the threshold
- Profile with NVIDIA Nsight Compute for cache analysis

### 2. Atomic Contention Hotspot (MEDIUM-HIGH)
**Theory**: Specific voxels at 2045 particles have extreme atomic contention (100+ threads writing to same voxel)

**Evidence**:
- InterlockedMax serializes all writes to same memory location
- Particle distribution at 2045 creates worst-case overlap
- GPU hang could be from atomic operation deadlock

**How to test**:
- Add atomicAdd counter per voxel to measure contention
- Histogram of writes per voxel
- Try InterlockedAdd instead of InterlockedMax (different GPU path)

### 3. Shader Compiler Edge Case (MEDIUM)
**Theory**: DXC compiler generates inefficient code at this specific thread count

**Evidence**:
- Exact threshold suggests compiler heuristic boundary
- Triple nested loop with atomics is complex
- GPU scheduler behavior changes at power-of-2 boundaries

**How to test**:
- Disassemble DXIL to check for suspicious patterns
- Try different optimization flags (/O0, /O3)
- Manually unroll loops to see if behavior changes

### 4. Descriptor/UAV Indexing Issue (LOW-MEDIUM)
**Theory**: UAV access pattern at 2045 particles triggers descriptor pipeline stall

**Evidence**:
- Descriptor table bindings at root signature parameter 2
- Could be related to descriptor heap memory layout

**How to test**:
- Bind volume UAV via root descriptor instead of descriptor table
- Create volume texture at different size (32³ or 128³)

### 5. Resource Barrier Deadlock (LOW)
**Theory**: UAV barrier creates circular dependency at this thread count

**Evidence**:
- UAV barrier after PopulateVolumeMip2 dispatch
- Could be related to GPU scheduling at 2048 thread boundary

**How to test**:
- Remove UAV barrier (may cause corruption but test if stall remains)
- Split dispatch into 2× smaller dispatches

---

## Next Steps (Priority Order)

### IMMEDIATE: PIX GPU Capture Analysis (60-90 min)

**Why**: Only way to see what GPU is actually doing during the 3-second stall

**How**:
```bash
# 1. Run with PIX attached
cd build/bin/Debug
PlasmaDX-Clean.exe --restir --particles 2045

# 2. Capture frame 1 manually (during the 3-second stall, press Ctrl+F12)

# 3. Analyze in PIX:
#    - Timeline view: Find the 3-second gap
#    - Which shader is running during the stall?
#    - GPU utilization: Compute units busy? Memory stalls?
#    - Wavefront occupancy: Are threads blocked?
```

**What to look for**:
- **Memory bandwidth saturation** → cache thrashing
- **Low wavefront occupancy** → atomic contention
- **Long-running wavefronts** → infinite loop (despite our guards)
- **Barrier stalls** → synchronization deadlock

### SHORT-TERM: Diagnostic Logging (30-45 min)

**Add instrumentation to shader**:
```hlsl
// Global atomic counter for diagnostics
RWStructuredBuffer<uint> g_debugCounters : register(u1);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint particleIdx = dispatchThreadID.x;

    // Count total particles processed
    InterlockedAdd(g_debugCounters[0], 1);

    if (particleIdx >= g_particleCount) {
        // Count early returns
        InterlockedAdd(g_debugCounters[1], 1);
        return;
    }

    // ... rest of shader ...

    // Count voxels written
    for (...) {
        if (density > 0.0001) {
            InterlockedAdd(g_debugCounters[2], 1);  // Total voxel writes
            InterlockedMax(g_volumeTexture[voxelCoords], densityAsUint, originalValue);
        }
    }
}
```

**CPU readback after frame 1**:
```cpp
// Read debug counters
uint32_t counters[3];
ReadbackBuffer(debugCounters, counters, sizeof(counters));

LOG_INFO("PopulateVolumeMip2 debug counters:");
LOG_INFO("  Total threads executed: {}", counters[0]);
LOG_INFO("  Early returns: {}", counters[1]);
LOG_INFO("  Voxel writes: {}", counters[2]);
```

**Expected at 2044**: counters = {2048, 4, ~500000}
**Expected at 2045**: counters = {2048, 3, ~510000}

If counters don't match expectations → shader logic issue
If counters match but still stalls → GPU hardware issue

### MEDIUM-TERM: Alternative Approaches (2-4 hours)

**If PIX reveals atomic contention**:
1. **Two-pass splatting**:
   - Pass 1: Each thread writes to private buffer
   - Pass 2: Combine private buffers (no contention)

2. **Reduce voxel resolution**:
   - Try 32³ or 48³ instead of 64³
   - Less contention, fewer writes

3. **Particle binning**:
   - Sort particles by voxel grid cell
   - Process one cell at a time (no overlap)

**If PIX reveals memory bandwidth saturation**:
1. **Coarsen writes**:
   - Write to every 2nd voxel, interpolate later
   - Reduces bandwidth by 8×

2. **Use shared memory**:
   - Accumulate in group-shared memory
   - Flush to global memory once per thread group

---

## Files Modified This Session

### Core Changes
- **populate_volume_mip2.hlsl**: Added atomic operations, loop bounds validation, AABB clamping
- **VolumetricReSTIRSystem.cpp**: Changed texture format to R32_UINT for atomics
- **volumetric_restir_common.hlsl**: Updated texture sampling to use `.Load()` with `asfloat()`
- **path_generation.hlsl**: Changed texture type to `Texture3D<uint>`
- **Application.cpp**: Re-enabled PopulateVolumeMip2 call

### Current State
- ✅ All shaders compile without errors
- ✅ Pipeline binds correctly
- ✅ Frame 1 completes (with 3-second stall)
- ❌ Frame 2+ freeze/crash at 2045+ particles
- ❌ 3-second GPU stall unresolved

---

## Performance Impact

### With PopulateVolumeMip2 Disabled
- **2044 particles**: ~190 FPS
- **2045 particles**: ~190 FPS
- **10K particles**: ~80 FPS
- **No crashes**, but output texture bug remains

### With PopulateVolumeMip2 Enabled
- **2044 particles**: Works perfectly, no stall
- **2045 particles**: 3-second stall frame 1, freeze/crash frame 2+
- **10K particles**: Immediate crash

---

## Key Insights

### What We've Learned
1. **Bug is deterministic** - Exact same behavior every test
2. **Threshold is precise** - 2044 vs 2045, not approximate
3. **Bug is in PopulateVolumeMip2** - Isolation test proves this
4. **Bug is GPU-side** - No D3D12 API errors, happens during GPU execution
5. **Bug is compute-bound** - Memory/cache/atomic contention likely

### What Doesn't Help
1. **Reducing resolution** - Problem scales with particle count, not resolution
2. **Limiting voxel writes** - 8×8×8 limit doesn't prevent stall
3. **Atomic operations** - Adding atomics didn't fix it (though necessary for correctness)
4. **Loop bounds guards** - Already validated, not the issue
5. **Descriptor management** - Pre-allocated, no heap exhaustion

### Critical Questions
1. **Why exactly 2045?** - What's special about 3 below 2048?
2. **What takes 3 seconds?** - Is it waiting, computing, or deadlocked?
3. **Why does frame 1 complete but frame 2 freezes?** - State corruption?
4. **Is it memory, cache, or atomics?** - Need PIX to distinguish

---

## Recommended Debug Workflow for Next Session

### Step 1: PIX GPU Capture (CRITICAL)
```bash
# Launch with PIX attached, 2045 particles
PlasmaDX-Clean.exe --restir --particles 2045

# During 3-second stall, capture frame 1 (Ctrl+F12)
# Analyze:
#   - What operation is taking 3 seconds?
#   - Memory bandwidth utilization?
#   - Wavefront occupancy?
#   - Atomic contention?
```

### Step 2: Add Diagnostic Counters
```cpp
// Shader: Count voxel writes per thread
// CPU: Readback and log after frame 1
// Compare 2044 vs 2045 distributions
```

### Step 3: Test Alternative Implementations
```hlsl
// A: Remove atomics (test if contention is the issue)
// B: Reduce volume resolution (32³ vs 64³)
// C: Split into 2× dispatches (16 thread groups each)
```

### Step 4: Use Updated PIX Agent
```bash
# If PIX fails, use autonomous debugging agent
# Agent located at: pix-debugging-agent-v4/
# AUTONOMOUS_DEBUGGING_COMPLETE.md has latest instructions
```

---

## References

**Previous Debug Sessions**:
- `VOLUMETRIC_RESTIR_TDR_DEBUG_SESSION.md` - Initial isolation test
- `VOLUMETRIC_RESTIR_STATUS_2025-10-31.md` - Earlier session status

**Implementation Plan**:
- `VOLUMETRIC_RESTIR_IMPLEMENTATION_PLAN.md` - Phase 1 roadmap

**PIX Debugging Agent**:
- `pix-debugging-agent-v4/README.md` - Agent capabilities
- `pix-debugging-agent-v4/AUTONOMOUS_DEBUGGING_COMPLETE.md` - Latest setup
- `pix-debugging-agent-v4/OPERATING_INSTRUCTIONS.md` - Usage guide

**Logs**:
- `build/bin/Debug/logs/PlasmaDX-Clean_20251101_005939.log` - Atomic fix test (failed)
- `build/bin/Debug/logs/PlasmaDX-Clean_20251101_014355.log` - Loop bounds fix test (failed)

---

## Branch & Version Info

**Current Branch**: 0.12.6
**Parent Branch**: 0.12.5 (DLSS integration fix complete)
**Build**: Debug (no optimizations)
**GPU**: NVIDIA RTX 4060 Ti (8GB)
**Driver**: Latest
**API**: DirectX 12 with Agility SDK

**Recent Commits**:
- 7ed336f: feat: Implement Volume Mip 2 population
- 4f0aab7: feat: Enhance Volumetric ReSTIR integration
- f9f538f: feat: Refactor VolumetricReSTIR initialization

**Files with Uncommitted Changes**:
- `populate_volume_mip2.hlsl` - Multiple fix attempts
- `VolumetricReSTIRSystem.cpp` - R32_UINT format change
- `volumetric_restir_common.hlsl` - asfloat() conversion
- `path_generation.hlsl` - Texture3D<uint> declaration
- `Application.cpp` - PopulateVolumeMip2 re-enabled

---

**Last Updated**: 2025-11-01 01:45
**Next Session Focus**: PIX GPU capture analysis of the 3-second stall
**Confidence**: MEDIUM - Need GPU profiling data to proceed
**Estimated Time to Fix**: 2-6 hours with PIX analysis
