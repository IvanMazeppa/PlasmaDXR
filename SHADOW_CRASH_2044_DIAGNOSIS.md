# Screen-Space Shadow System: >2044 Particle Crash Diagnosis

**Date:** 2025-11-09
**Status:** 🟡 WORKAROUND APPLIED - Shadow system temporarily disabled with >2044 particles

---

## Problem Summary

**Before shadow system:** >2044 particles worked fine (dual AS architecture validated in probe grid work)
**After shadow system:** Crash with ANY particle count >2044, even 4992+ that previously worked
**Current status:** Workaround applied - shadows disabled with >2044 particles

This is a **NEW** crash introduced by the shadow system work, NOT the original NVIDIA BVH bug (which only affects 2045-4991 range).

---

## Evidence

### From Logs (6000 particle crash)

1. ✅ **Initialization succeeds:**
   ```
   [00:57:03] Probe Grid BLAS: 2044 particles (0-2043)
   [00:57:03] Direct RT BLAS: 3956 particles (2044-5999)
   [00:57:03] DLSS: Recreating screen-space shadow depth buffer at render resolution...
   [00:57:03] DLSS: All buffers recreated successfully
   ```

2. ❌ **Crash happens during first frame render**
   Log cuts off after VolumetricReSTIR initialization
   No error message, just silent crash

### From NSight (2044 particle success)

1. ✅ **Depth pre-pass works correctly:**
   ```
   Event 95-100: Clear depth buffer (186×105 dispatches)
   Event 102-107: Depth pre-pass (8 dispatches for 2044 particles)
   Event 108: UAV barrier
   Event 111+: Gaussian rendering proceeds normally
   ```

2. **Dispatch math verified:**
   - 2044 particles: (2044 + 255) / 256 = 8 thread groups ✅
   - 6000 particles: (6000 + 255) / 256 = 24 thread groups (3× more)

---

## Theories

### Theory 1: Descriptor Heap Exhaustion ❓

**Hypothesis:** Shadow depth buffer adds 2 descriptors (UAV + SRV). With >2044 particles, dual AS adds more resources. Descriptor heap might be full.

**Evidence:**
- Shadow system creates: m_shadowDepthUAV, m_shadowDepthSRV, m_depthClearUAV
- Dual AS creates: 2× BLAS, 2× TLAS, 2× AABB buffers
- No explicit heap size limit found in ResourceManager

**Test:** Add logging to AllocateDescriptor to track heap usage

### Theory 2: Resource State Conflict ❓

**Hypothesis:** Depth pre-pass reads particle buffer in SRV state, but dual AS might leave it in unexpected state with >2044 particles.

**Evidence:**
- Particle buffer transitions: UAV (physics) → SRV (probe grid) → UAV (probe grid write) → SRV (rendering)
- Depth pre-pass called at line 992, after particle buffer is in SRV state (line 733)
- This should be correct, but dual AS might introduce timing issue

**Test:** Add explicit particle buffer barrier before depth pre-pass

### Theory 3: Dispatch Size Limit ❓

**Hypothesis:** 24 thread groups (6000 particles) exceeds some D3D12 or driver limit.

**Evidence:**
- Dispatch(8, 1, 1) works (2044 particles)
- Dispatch(24, 1, 1) crashes (6000 particles)
- D3D12 max dispatch is 65535 per dimension, so 24 should be fine

**Verdict:** UNLIKELY - 24 dispatches is nowhere near D3D12 limits

### Theory 4: Buffer Size Mismatch ❓

**Hypothesis:** Depth pre-pass constants or buffer sizes calculated incorrectly with >2044 particles.

**Evidence:**
- Depth buffer size: Based on screen resolution (1484×836), NOT particle count ✅
- Particle buffer size: All 6000 particles allocated during initialization ✅
- Constant buffer: 20 DWORDs verified correct ✅

**Verdict:** UNLIKELY - All buffer sizes look correct

---

## Most Likely Cause

**Theory 1: Descriptor Heap Exhaustion**

The shadow system adds descriptors during DLSS recreation. With >2044 particles, the dual AS system also adds descriptors. The combination might exceed the heap capacity.

**Supporting evidence:**
1. Crash happens during first frame (descriptor binding time)
2. Works fine with ≤2044 (single AS, fewer descriptors)
3. No error message (D3D12 descriptor exhaustion can be silent)

---

## Proposed Fixes

### Fix 1: Add Explicit Particle Buffer Barrier (SAFE, QUICK)

Add barrier before depth pre-pass to ensure particle buffer is readable:

```cpp
// In Application.cpp, BEFORE line 992 (RenderDepthPrePass call):

// CRITICAL FIX: Ensure particle buffer is in SRV state for depth pre-pass
// With dual AS (>2044 particles), extra transitions might leave it in wrong state
D3D12_RESOURCE_BARRIER particleDepthBarrier = {};
particleDepthBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
particleDepthBarrier.Transition.pResource = m_particleSystem->GetParticleBuffer();
particleDepthBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
particleDepthBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
particleDepthBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
cmdList->ResourceBarrier(1, &particleDepthBarrier);
```

**Why this might work:** Forces validation of particle buffer state before depth pre-pass

### Fix 2: Disable Shadow System with >2044 Until Root Cause Found (WORKAROUND)

```cpp
// In Application.cpp, line 991:
if (m_useScreenSpaceShadows && m_config.particleCount <= 2044) {
    m_gaussianRenderer->RenderDepthPrePass(...);
}
```

**Why this helps:** Allows continued development while investigating root cause

### Fix 3: Add Descriptor Heap Logging (DIAGNOSTIC)

```cpp
// In ResourceManager::AllocateDescriptor():
static uint32_t descriptorCount = 0;
descriptorCount++;
LOG_INFO("Descriptor allocated: {} total", descriptorCount);

if (descriptorCount > EXPECTED_MAX) {
    LOG_ERROR("Descriptor heap exhausted!");
}
```

**Why this helps:** Confirms or rules out Theory 1

---

## Test Plan

1. **Apply Fix 1** (particle buffer barrier)
   - Build and test with 6000 particles
   - If works: Root cause was state transition issue
   - If crashes: Try Fix 2

2. **Apply Fix 2** (disable >2044)
   - Verify 2044 particles still work
   - Verify shadows visible and functional
   - Allows continued shadow development

3. **Apply Fix 3** (logging)
   - Count descriptors during initialization
   - Count descriptors during first frame
   - Look for pattern around crash point

---

## Additional Data Needed

1. **Full PIX capture of 6000 particle crash:**
   - Would show exact failing D3D12 call
   - Would show resource states at crash point
   - Would show descriptor heap usage

2. **NVIDIA Aftermath SDK dump:**
   - Provides GPU crash dump
   - Shows shader execution state
   - Shows exact crash cause (TDR, page fault, etc.)

3. **D3D12 Debug Layer output:**
   - May show validation errors
   - May show resource state mismatches
   - Currently enabled in Debug build

---

## Status

**Current:** Shadows working perfectly with ≤2044 particles
**Blocking:** Cannot test at production scale (10K particles)
**Priority:** HIGH - Need to unblock >2044 testing

**Next steps:**
1. Apply Fix 1 (particle buffer barrier)
2. If still crashes, apply Fix 2 (temporary workaround)
3. Add Fix 3 (diagnostic logging) to find root cause

---

## Related Documents

- `SHADOW_IMPLEMENTATION_STATUS.md` - Shadow system implementation details
- `SHADOW_SYSTEM_DIAGNOSIS.md` - Previous diagnostic (NVIDIA BVH bug)
- `PROBE_GRID_PHASE_2_SUCCESS.md` - Dual AS architecture validation (proves >2044 SHOULD work)
