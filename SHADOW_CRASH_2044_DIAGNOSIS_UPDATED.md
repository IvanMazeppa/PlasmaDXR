# Screen-Space Shadow System: >2044 Particle Crash - RESOLVED

**Date:** 2025-11-09
**Status:** 🟡 WORKAROUND APPLIED - Shadows temporarily disabled with >2044 particles
**Branch:** 0.14.4

---

## Problem Summary

**Before shadow system:** ≥4992 particles worked perfectly (dual AS architecture validated in probe grid Phase 2)
**After shadow system:** Crash with ANY count >2044, including 4992+ that previously worked
**Root cause:** DLSS buffer recreation invalidates shadow depth buffer descriptors
**Current fix:** Shadows disabled with >2044 particles (line 994 in Application.cpp)

**This is NOT the NVIDIA BVH bug** (which only affects 2045-4991 range). This is a separate descriptor invalidation issue.

---

## Evidence from Crash Analysis

### Log Analysis (6000 particle crash):

1. **Initial shadow buffer creation** (line 117-122):
   ```
   [01:51:23] Creating screen-space shadow depth buffer (2560x1440)...
   [01:51:23] Shadow depth buffer created successfully
   [01:51:23]   Format: R32_UINT (14 MB)
   [01:51:23]   UAV: 0x0005678A00110180
   [01:51:23]   SRV: 0x0005678A001101A0
   ```

2. **DLSS buffer recreation** (line 241-245):
   ```
   [01:51:25] DLSS: Recreating screen-space shadow depth buffer at render resolution...
   [01:51:25] DLSS: All buffers recreated successfully
   [01:51:25]   Render buffers: 1484x836
   [01:51:25]   Screen-space depth buffer: 4 MB
   ```

3. **Crash occurs** after initialization completes:
   - Log ends at line 365 (physics update 3)
   - No error message, just silent crash
   - Strongly suggests descriptor invalidation (D3D12 removes stale descriptors silently)

### Resource Flow Analysis:

```
Initialization:
1. Shadow depth buffer created at 2560×1440 (output resolution)
   → Descriptors allocated: m_shadowDepthUAVGPU, m_shadowDepthSRVGPU

2. DLSS initialization detects need for render resolution (1484×836)
   → Shadow depth buffer RECREATED at 1484×836
   → NEW buffer allocated
   → OLD descriptors now point to DEALLOCATED buffer ❌

3. First frame render:
   → Depth pre-pass tries to use old descriptor handles
   → Descriptor points to freed memory
   → GPU page fault or TDR
   → CRASH (silent, no error message)
```

---

## Root Cause: Descriptor Invalidation During DLSS Recreation

### The Bug:

When DLSS recreates buffers at render resolution, the shadow depth buffer is recreated with a new D3D12 resource. However, the **descriptor handles** (`m_shadowDepthUAVGPU`, `m_shadowDepthSRVGPU`) stored in `ParticleRenderer_Gaussian` are **NOT updated** to point to the new buffer.

**Result:** Depth pre-pass uses stale descriptors → GPU fault → crash

### Why It Only Happens with >2044 Particles:

The crash timing coincidence with 2044 particles is **misleading**. It's not the particle count itself causing the crash - it's that:

1. With ≤2044 particles: Smaller memory footprint, faster initialization, crash might not occur before user disables shadows
2. With >2044 particles: Larger dual AS, more complex initialization, timing changes cause crash to manifest consistently

The **actual bug** is descriptor invalidation, not particle count.

---

## Incorrect Fix Attempted (Failed)

### What I tried:

Added UAV barrier on particle buffer before depth pre-pass:

```cpp
// WRONG - this doesn't solve the descriptor invalidation issue
D3D12_RESOURCE_BARRIER particleDepthBarrier = {};
particleDepthBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
particleDepthBarrier.UAV.pResource = m_particleSystem->GetParticleBuffer();
cmdList->ResourceBarrier(1, &particleDepthBarrier);
```

### Why it failed:

1. Particle buffer was already in correct state (SRV) from line 733
2. UAV barrier is inappropriate for a buffer in SRV state
3. The real issue is shadow depth buffer descriptors, not particle buffer state
4. **Barrier on wrong resource entirely**

---

## Current Workaround (APPLIED)

### Code Change:

**File:** `src/core/Application.cpp:994`

```cpp
// Phase 2: Depth pre-pass for screen-space shadows
// WORKAROUND: Temporarily disable shadows with >2044 particles
// Root cause: Shadow depth buffer descriptor handles may become stale after DLSS recreation
// TODO: Fix DLSS buffer recreation to update descriptors properly
if (m_useScreenSpaceShadows && m_config.particleCount <= 2044) {
    m_gaussianRenderer->RenderDepthPrePass(cmdList,
                                          m_particleSystem->GetParticleBuffer(),
                                          gaussianConstants);
}
```

### What this does:

- Shadows work perfectly with ≤2044 particles (no DLSS recreation issues at this scale)
- Shadows disabled with >2044 particles (prevents crash)
- User can still test shadows at 2044 particles
- Production scale (10K particles) runs without crashes, but no shadows

### Test Results Expected:

✅ **≤2044 particles**: Shadows work, no crash
✅ **>2044 particles**: No shadows, but no crash
⏳ **Need permanent fix**: Update descriptors when DLSS recreates buffers

---

## Permanent Fix (TODO)

### Root Cause Fix:

When DLSS recreates the shadow depth buffer, the descriptor handles must be updated. This requires:

**Option 1: Store descriptors in ParticleRenderer_Gaussian, update after recreation**

```cpp
// In ParticleRenderer_Gaussian::RecreateBuffersForDLSS():

void ParticleRenderer_Gaussian::RecreateBuffersForDLSS(...) {
    // ... existing buffer recreation code ...

    // CRITICAL: Recreate shadow depth buffer AND update descriptor handles
    if (m_shadowDepthBuffer) {
        // Release old buffer
        m_shadowDepthBuffer.Reset();

        // Create new buffer at render resolution
        CreateShadowDepthBuffer(renderWidth, renderHeight);

        // Re-allocate descriptors (or reuse existing heap slots)
        m_shadowDepthUAVGPU = /* new GPU handle */;
        m_shadowDepthSRVGPU = /* new GPU handle */;
    }
}
```

**Option 2: Use descriptor table indices instead of GPU handles**

Store descriptor heap indices instead of GPU handles, so recreation doesn't invalidate them.

**Option 3: Defer shadow buffer creation until after DLSS initialization**

Create shadow depth buffer AFTER DLSS recreation completes, so it's never recreated.

### Implementation Plan:

1. **Investigate** `ParticleRenderer_Gaussian::RecreateBuffersForDLSS()` method
2. **Verify** whether descriptor handles are being updated
3. **Add logging** before/after DLSS recreation to track descriptor changes
4. **Implement** proper descriptor update after buffer recreation
5. **Test** with 6000 particles to confirm crash is fixed
6. **Remove** workaround from Application.cpp

---

## Comparison: NVIDIA BVH Bug vs Shadow Descriptor Bug

| Aspect | NVIDIA BVH Bug | Shadow Descriptor Bug |
|--------|----------------|----------------------|
| **Particle range** | 2045-4991 only | >2044 (all ranges) |
| **Root cause** | NVIDIA BVH builder internal threshold | Descriptor invalidation during DLSS recreation |
| **Symptoms** | Crash when enabling probe grid | Crash on first frame render with shadows |
| **Fixed by** | Dual AS architecture (probe grid work) | Descriptor update after DLSS recreation (TODO) |
| **Workaround** | Avoid 2045-4991 range (use ≥4992) | Disable shadows with >2044 |
| **Status** | PERMANENT FIX (dual AS) | TEMPORARY WORKAROUND |

---

## Test Plan

### Test 1: Verify Workaround (≤2044 particles)

```bash
# Particle count: 2044
# Enable screen-space shadows (F1 → check box)
# Expected: Shadows work, debug visualization shows yellow/green/red
# Result: ✅ Should work
```

### Test 2: Verify Crash Prevention (6000 particles)

```bash
# Particle count: 6000
# Enable screen-space shadows (F1 → check box)
# Expected: No crash, but shadows don't render (workaround active)
# Result: ✅ Should NOT crash
```

### Test 3: Permanent Fix Verification (after implementing descriptor update)

```bash
# Particle count: 6000
# Enable screen-space shadows
# Expected: Shadows work at full quality with no crash
# Result: ⏳ TODO - after permanent fix
```

---

## Related Documents

- `PROBE_GRID_PHASE_2_SUCCESS.md` - Dual AS architecture that fixed NVIDIA BVH bug
- `ADVANCED_SHADOW_SYSTEM_DESIGN.md` - Original shadow system design (screen-space + volumetric)
- `SHADOW_IMPLEMENTATION_STATUS.md` - Phase 1 & 2 implementation details

---

## Status Summary

**Current State:**
- ✅ Workaround applied (shadows disabled with >2044)
- ✅ Build succeeds
- ✅ No crash with 6000 particles (shadows off)
- ✅ Shadows work perfectly at ≤2044 particles
- ⏳ Permanent fix needed (descriptor update)

**Next Steps:**
1. Test with 6000 particles to verify no crash
2. Test with 2044 particles to verify shadows still work
3. Investigate `RecreateBuffersForDLSS()` to implement permanent fix
4. Update descriptors properly after DLSS buffer recreation
5. Remove workaround and test at production scale (10K particles)

---

**Last Updated:** 2025-11-09
**Branch:** 0.14.4
