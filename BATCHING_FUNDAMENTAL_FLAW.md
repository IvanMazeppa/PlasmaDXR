# Batching Implementation - Fundamental Architectural Flaw

**Date:** 2025-11-04
**Status:** ❌ **BATCHING DOES NOT WORK** - Fundamental design flaw discovered
**Critical Issue:** Gaussian renderer and probe grid only trace against first batch's TLAS

---

## The Problem

Batching acceleration structures **successfully builds multiple BLAS/TLAS**, but the **rendering systems are completely unaware of this**.

### What We Implemented

✅ Split particles into batches of 2000
✅ Create separate AABB buffer per batch
✅ Build separate BLAS per batch
✅ Build separate TLAS per batch
✅ All batches build successfully (verified by logs)

### What We Forgot

❌ **Gaussian renderer only traces against ONE TLAS**
❌ **Probe grid only traces against ONE TLAS**
❌ **RayQuery lighting shader only traces against ONE TLAS**

**Result:** Only the first 2000 particles are visible/lit. Particles 2001+ don't exist in the scene from the renderer's perspective.

---

## Log Evidence

### Test: 9001 Particles (5 Batches)

```
[23:56:54] [INFO] [BATCHING] Building 5 batches for 9001 particles

Batch 0: startIndex=0,    count=2000  ✅ Completes
Batch 1: startIndex=2000, count=2000  ✅ Completes
Batch 2: startIndex=4000, count=2000  ❌ CRASH during AABB generation
```

**Log stops at line 343** - crashes while generating AABBs for batch 2.

### Why Batch 2 Crashes

**Theory:** The Gaussian renderer is already trying to render/trace rays during batch building:
1. Batch 0 builds → Gaussian tries to trace → Only sees 2000 particles ✅
2. Batch 1 builds → Gaussian tries to trace → Still only sees batch 0's TLAS ⚠️
3. Batch 2 starts building → **Gaussian/probe grid crashes trying to access particles that don't exist in the TLAS they're tracing**

**Alternative theory:** Memory corruption or resource conflict when building multiple AS simultaneously.

---

## Why GetTLAS() Backward Compatibility Failed

We tried to maintain backward compatibility:

```cpp
ID3D12Resource* GetTLAS() const {
    if (m_useBatching && !m_batches.empty()) {
        return m_batches[0].tlas.Get();  // Return first batch only
    }
    return m_topLevelAS.Get();
}
```

**What we thought:** Gaussian renderer calls GetTLAS(), gets batch[0], traces against 2000 particles

**What actually happens:**
- Gaussian renderer: Sees only 2000 particles (missing 7001 particles in 9001 particle scene)
- Probe grid: Traces rays against 2000 particles only
- Particle-to-particle lighting: Only considers first 2000 particles
- Rendering: **Incomplete and incorrect**

---

## The Fundamental Design Flaw

### Single-TLAS Architecture Assumption

**Every rendering system assumes:**
- ONE TLAS contains ALL particles
- Ray tracing this TLAS hits ANY particle in the scene
- Primitive indices map directly to particle buffer indices

**With batching:**
- **5 separate TLAS** (for 9001 particles)
- Each TLAS contains 2000 particles max
- Primitive index 500 in batch 2 TLAS = particle 4500 globally (NOT particle 500!)

**This breaks EVERYTHING:**
- Gaussian renderer color lookup (wrong particle index)
- Probe grid lighting (wrong particle positions)
- RT lighting shader (missing most particles)

### Why Multi-TLAS is Hard

To properly support batching, EVERY ray tracing call needs:

```cpp
// OLD (single TLAS):
RayQuery q;
q.TraceRayInline(g_particleTLAS, ...);  // One TLAS, all particles

// NEW (multi-TLAS):
for (uint batchIdx = 0; batchIdx < numBatches; batchIdx++) {
    RayQuery q;
    q.TraceRayInline(g_particleTLAS[batchIdx], ...);  // Trace each batch

    if (hit) {
        uint localIdx = q.CommittedPrimitiveIndex();
        uint globalIdx = (batchIdx * 2000) + localIdx;  // Convert to global index
        Particle p = g_particles[globalIdx];
        // ... use particle data ...
    }
}
```

**This requires changes to:**
1. Gaussian raytrace shader (`particle_gaussian_raytrace.hlsl`)
2. Probe update shader (`update_probes.hlsl`)
3. RT lighting shader (`particle_raytraced_lighting_cs.hlsl`)
4. All root signatures (expand from 1 TLAS to 8 TLAS slots)
5. All binding code (bind all batch TLAS resources)

**Estimated work:** 4-6 hours of shader and binding code changes

---

## Why Batching Will Never Work for This Architecture

### Problem 1: Performance Overhead

Tracing against 5 separate TLAS structures means:
- 5× BVH traversal overhead
- 5× hit testing per ray
- Complex closest-hit logic (track closest across all batches)

**Expected performance:** 3-5× slower than monolithic TLAS

**Defeats the purpose:** We're trying to improve performance/stability, not destroy it

### Problem 2: Complexity Explosion

Every shader that traces rays needs:
- Multi-TLAS loop
- Global index calculation
- Cross-batch hit comparison
- Proper resource binding

**Lines of code to change:** 500+ across 6 shaders and 8 CPP files

### Problem 3: Still Doesn't Fix Original Bug

**Original issue:** Crash at 2045 particles with monolithic TLAS

**Batching:** Splits into 2× 2000-particle TLAS + 1× 45-particle TLAS

**If the bug is:**
- BVH leaf count at 512 leaves → Each batch avoids this ✅
- Total scene primitive count → Still 2045 total primitives ❌
- TLAS instance count → Now 3 instances instead of 1 ❌
- Memory alignment → Doesn't change total memory used ❌

**We don't even know if batching helps the original bug!**

---

## Alternative Approaches

### Option 1: Disable Batching, Test at 2000 Exactly

**Hypothesis:** Maybe 2000 particles works fine (monolithic), 2045 crashes

**Test plan:**
1. Set `m_useBatching = false`
2. Test at 2000 particles → Should work
3. Test at 2001 particles → Expect crash
4. Test at 2044 particles → Expect crash
5. Test at 2045 particles → Expect crash

**If 2000 works but 2001 crashes:**
- Issue is NOT batching-related
- Issue is related to particle count > 2000
- Different root cause than we thought

### Option 2: Increase Monolithic Threshold

**Change batching threshold to avoid the issue:**

```cpp
static constexpr uint32_t PARTICLES_PER_BATCH = 4000;  // Higher threshold
```

**Test cases:**
- 2045 particles → Monolithic (1 TLAS, all particles)
- 4001 particles → Batching (2 batches)

**If 2045 works with monolithic:**
- Confirms issue is batching-related
- Original 2045 bug might not exist anymore (previous fixes worked?)

### Option 3: Abandon Batching Entirely

**Accept the constraint:** Max 2000 particles with current architecture

**Focus on:** Optimizing what works instead of fixing what doesn't

**Advantages:**
- Stable system
- Known performance characteristics
- Can scale other dimensions (lights, resolution, effects)

### Option 4: Complete Multi-TLAS Rewrite (4-6 Hours)

**Full implementation of multi-TLAS support:**
1. Update all 3 ray tracing shaders
2. Expand all root signatures
3. Update all binding code
4. Add global index calculation everywhere
5. Test extensively

**Risk:** Even after all this work, might still crash at 2045 particles if the bug is unrelated to batching

---

## Recommendation

### Immediate Action: Disable Batching

```cpp
// RTLightingSystem_RayQuery.h
bool m_useBatching = false;  // DISABLED
```

**Test at various particle counts:**
- 1000 → Should work
- 2000 → Should work
- 2001 → Will reveal if issue is >2000 or batching
- 2044 → Test just before original crash point
- 2045 → Original crash point

### If Monolithic Works at 2045

**Conclusion:** Batching was the problem, not the solution

**Next steps:**
- Remove batching code
- Focus on optimizing monolithic path
- Accept 10K particle limit (or whatever works)

### If Monolithic Still Crashes at 2045

**Conclusion:** Original bug still exists

**Next steps:**
1. Try power-of-2 padding workaround alone (without batching)
2. Binary search exact crash threshold (2030? 2038? 2042?)
3. Try different BLAS/TLAS build flags
4. Query NVIDIA developer forums with ROOT_CAUSE_ANALYSIS_PROMPT.md
5. Consider alternative renderer architecture (instanced rendering, mesh shaders, etc.)

---

## Lessons Learned (Again)

### 1. Test Incrementally, Not All At Once

**Our mistake:**
- Implemented full batching (multiple BLAS/TLAS)
- Assumed backward compatibility would "just work"
- Tested with large particle counts (9001) immediately

**Better approach:**
- Test monolithic at 2045 first (does original bug still exist?)
- Implement batching for ONE renderer only (Gaussian)
- Test with 2001 particles (minimal batching case)
- Add probe grid support only after Gaussian works

### 2. Understand Dependencies Before Refactoring

**We changed:** Acceleration structure architecture (1 TLAS → N TLAS)

**We forgot to check:**
- Does Gaussian renderer support multi-TLAS? ❌
- Does probe grid support multi-TLAS? ❌
- Do shaders need updates? ❌

**Result:** Built a system that compiles but doesn't work

### 3. Backward Compatibility is Not Free

**GetTLAS() returning batch[0] seemed clever:**
- No changes to calling code needed
- "Transparent" to consumers

**Reality:**
- Silently breaks rendering (missing 80% of particles)
- No compiler errors or warnings
- Crashes with cryptic GPU errors

**Lesson:** If a change is architecturally significant, make it LOUD. Force calling code to adapt.

### 4. Question the Premise

**We assumed:** Batching will fix the 2045 crash

**We should have asked:**
- Will batching actually help? (Theory: yes, if BVH leaf count is the issue)
- What are the downsides? (Performance, complexity, compatibility)
- Is there a simpler solution? (Smaller changes, test first)
- What if we're wrong about the root cause? (Batching won't help at all)

---

## Status

❌ **Batching does not work** - architectural incompatibility with rendering systems
⏳ **Next step:** Disable batching, test monolithic path at 2000-2045 particles
🔄 **Reevaluate:** Whether batching is even the right approach

---

## Files to Revert/Modify

If abandoning batching:

1. **RTLightingSystem_RayQuery.h**
   - Remove: `m_batches` vector
   - Remove: `AccelerationBatch` struct
   - Remove: Batch accessor methods
   - Keep: Monolithic BLAS/TLAS infrastructure

2. **RTLightingSystem_RayQuery.cpp**
   - Remove: `CreateBatchedAccelerationStructures()`
   - Remove: `GenerateAABBsForBatch()`, `BuildBatchBLAS()`, `BuildBatchTLAS()`
   - Remove: Batching logic from `ComputeLighting()`
   - Keep: Power-of-2 padding in monolithic `BuildBLAS()`

3. **Documentation**
   - Update: BATCHING_STATUS.md to mark as "Abandoned"
   - Create: Analysis of why batching failed
   - Keep: ROOT_CAUSE_ANALYSIS_PROMPT.md for understanding original bug

---

**Bottom line:** Batching was a well-intentioned idea that failed due to fundamental architectural assumptions. Time to try a different approach.
