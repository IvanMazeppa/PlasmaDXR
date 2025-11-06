# Probe Grid Crash Debugging Summary

**Date:** 2025-11-06
**Issue:** Probe grid crashes when enabled at ≥2045 particles
**Hardware:** RTX 4060 Ti (Ada Lovelace)
**Particle Count Tested:** 2045

---

## Current Status

### What Works ✅
- Application launches successfully with 2045+ particles
- Dual AS architecture prevents the original 2045 crash bug
- Runtime controls (particle size, adaptive radius) work uniformly
- All particles visible without probe grid enabled
- System runs stably until probe grid is enabled

### What Crashes ❌
- **Enabling probe grid via ImGui at ≥2045 particles** → Immediate GPU hang/crash
- Crash occurs within 1-2 frames of enabling probe grid
- No error messages logged before crash
- Works fine at ≤2044 particles

---

## NSight Graphics Capture Analysis

### Acceleration Structures Detected
From `probe_grid_2_2045_particles_objects.csv`:

**BLAS (Bottom-Level):**
- `BLAS 127 0x9fc9000` - Probe Grid BLAS (particles 0-2043)
- `BLAS 129 0x9fed000` - Direct RT BLAS (particle 2044)

**TLAS (Top-Level):**
- `TLAS 128 0x9fd6000` - Probe Grid TLAS
- `TLAS 130 0x9ffa000` - Direct RT TLAS
- `TLAS 131 0x9ffd000` - Combined TLAS (2 instances)

### Resources
- `ProbeGridSystem::ProbeBuffer` at 0x1a200000 (4 MB, 32³ probes)
- `ProbeGridSystem::UpdateConstants` - Constant buffer for probe update

---

## Code Analysis

### Current Implementation (src/core/Application.cpp:668-692)

```cpp
// Probe Grid Update Pass (Phase 2)
if (m_useProbeGrid && m_probeGridSystem && m_rtLighting && m_particleSystem) {
    // Use probe grid TLAS specifically (not combined TLAS)
    // Probe grid only lights first 2044 particles
    uint32_t probeGridParticleCount = std::min(m_config.particleCount, 2044u);
    m_probeGridSystem->UpdateProbes(
        cmdList,
        m_rtLighting->GetProbeGridTLAS(),  // Returns TLAS 128 (probe grid only)
        m_particleSystem->GetParticleBuffer(),
        probeGridParticleCount,             // 2044 for 2045 particles
        lightBuffer,
        lightCount,
        m_frameCount
    );
}
```

### Shader Bounds Check (shaders/probe_grid/update_probes.hlsl:257-258)

```hlsl
if (particleIdx < g_particleCount) {
    Particle particle = g_particles[particleIdx];
```

**Expected behavior:** Should only read particles 0-2043 (within bounds)

---

## Hypotheses for Crash

### Hypothesis 1: TLAS Instance Index Issue ⚠️ **MOST LIKELY**
**Problem:** Even though we're using the probe grid TLAS, it might still be a multi-instance TLAS.

**Evidence:**
- NSight shows 3 TLAS structures
- `GetProbeGridTLAS()` returns `m_probeGridAS.tlas.Get()`
- But at exactly 2045 particles, dual AS architecture creates BOTH probe grid and direct RT

**Root Cause:** The probe grid TLAS might have been rebuilt as part of the dual AS architecture and now contains an instance descriptor pointing to both BLAS.

**Fix Required:** Verify probe grid TLAS is truly single-instance at 2045 particles.

### Hypothesis 2: Particle Buffer State Transition Issue
**Problem:** Probe grid might be reading particle buffer in wrong state.

**Evidence:**
- Probe grid reads particle buffer directly
- Dual AS uses same buffer with different offsets
- State transitions might not be synchronized

**Fix Required:** Add resource barrier before probe grid update.

### Hypothesis 3: AABB Buffer Out of Sync
**Problem:** Probe grid BLAS AABBs might be stale or contain wrong particle count.

**Evidence:**
- AABBs generated for 2044 particles in probe grid BLAS
- But particle buffer has 2045 particles
- Probe grid might be hitting particle 2044 which isn't in BLAS

**Fix Required:** Ensure BLAS is fully built before probe grid update.

### Hypothesis 4: Root Signature / Descriptor Mismatch
**Problem:** Probe grid shader expects different descriptor layout than provided.

**Evidence:**
- NSight shows probe grid resources exist
- But shader might be reading from wrong descriptor slots
- Descriptor heap might not be set correctly

**Fix Required:** Verify descriptor bindings match shader expectations.

---

## Debugging Plan

### Step 1: Add Diagnostic Logging (IMMEDIATE)
Add detailed logging to probe grid update to catch crash point:

```cpp
// In Application.cpp before probe grid update
if (m_useProbeGrid) {
    LOG_INFO("=== PROBE GRID UPDATE START ===");
    LOG_INFO("  Total particles: {}", m_config.particleCount);
    LOG_INFO("  Probe grid particles: {}", probeGridParticleCount);
    LOG_INFO("  Probe grid TLAS: 0x{:016x}",
             m_rtLighting->GetProbeGridTLAS()->GetGPUVirtualAddress());
    LOG_INFO("  Frame: {}", m_frameCount);

    m_probeGridSystem->UpdateProbes(...);

    LOG_INFO("=== PROBE GRID UPDATE COMPLETE ===");
}
```

**Expected Result:** Log should stop at "START" if crash is in UpdateProbes.

### Step 2: Verify Probe Grid TLAS Integrity
Check if probe grid TLAS is truly single-instance:

```cpp
// In RTLightingSystem_RayQuery::ComputeLighting() after building probe grid TLAS
LOG_INFO("Probe Grid TLAS built for {} particles", probeGridCount);
LOG_INFO("  BLAS GPU address: 0x{:016x}", m_probeGridAS.blas->GetGPUVirtualAddress());
LOG_INFO("  TLAS GPU address: 0x{:016x}", m_probeGridAS.tlas->GetGPUVirtualAddress());
LOG_INFO("  Is combined TLAS? {}", (directRTCount > 0) ? "YES" : "NO");
```

**Expected Result:** Should show "Is combined TLAS? NO" at 2045 particles.

### Step 3: Add Particle Count Validation
Ensure probe grid never tries to read particle 2044:

```cpp
// In ProbeGridSystem::UpdateProbes()
if (particleCount >= 2045) {
    LOG_WARN("Probe grid particle count clamped from {} to 2044", particleCount);
    particleCount = 2044;  // Hard clamp
}
```

**Expected Result:** Should see warning and no crash.

### Step 4: Add Resource Barrier Before Probe Grid
Ensure particle buffer is in correct state:

```cpp
// In Application.cpp before probe grid update
D3D12_RESOURCE_BARRIER particleBarrier = {};
particleBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
particleBarrier.Transition.pResource = m_particleSystem->GetParticleBuffer();
particleBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
particleBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
particleBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
cmdList->ResourceBarrier(1, &particleBarrier);

m_probeGridSystem->UpdateProbes(...);
```

**Expected Result:** Might fix if issue is resource state.

### Step 5: Temporarily Disable Direct RT BLAS Build
Test if dual AS architecture is interfering:

```cpp
// In RTLightingSystem_RayQuery::ComputeLighting()
// Comment out direct RT BLAS build at 2045 particles
if (false && directRTCount > 0) {  // TEMPORARILY DISABLED FOR TESTING
    BuildBLAS_ForSet(...);
}
```

**Expected Result:** If this fixes crash, confirms dual AS interference.

### Step 6: Use PIX/NSight GPU Debugger
Capture frame at crash point to see exact shader invocation that fails:

1. Enable probe grid
2. Capture frame immediately (Ctrl+F12 in PIX)
3. Look for last successful shader dispatch
4. Check shader parameters and resource bindings

**Expected Result:** Will show exact crash point in shader.

---

## Quick Fixes to Try (In Order)

### Fix 1: Hard Clamp Probe Grid to 2043 Particles
Most conservative fix - probe grid never reaches 2044:

```cpp
uint32_t probeGridParticleCount = std::min(m_config.particleCount, 2043u);  // Was 2044
```

**Rationale:** Avoid edge case at exactly 2044.

### Fix 2: Disable Probe Grid at ≥2045 Particles
Temporary workaround until root cause found:

```cpp
if (m_useProbeGrid && m_config.particleCount < 2045 && m_probeGridSystem) {
    m_probeGridSystem->UpdateProbes(...);
}
```

**Rationale:** Users with >2045 particles use direct RT lighting only.

### Fix 3: Force Probe Grid to Use Legacy Monolithic TLAS
Use the legacy TLAS instead of dual AS TLAS:

```cpp
// Use legacy TLAS if it exists, otherwise probe grid TLAS
ID3D12Resource* probeTLAS = m_topLevelAS ? m_topLevelAS.Get()
                                          : m_rtLighting->GetProbeGridTLAS();
m_probeGridSystem->UpdateProbes(cmdList, probeTLAS, ...);
```

**Rationale:** Bypasses dual AS completely for probe grid.

---

## Next Session Action Items

1. ✅ Apply Fix 1 (hard clamp to 2043) - **TEST IMMEDIATELY**
2. Add Step 1 diagnostic logging
3. If still crashes, apply Step 3 (particle count validation)
4. If still crashes, apply Step 4 (resource barrier)
5. Capture PIX/NSight frame at crash to analyze shader state

---

## NSight Graphics Usage Notes

**What NSight Provides:**
- ✅ Object list with resource addresses
- ✅ Event timeline with GPU timing
- ✅ Acceleration structure details (BLAS/TLAS addresses)
- ✅ CSV export for analysis

**How to Use:**
1. Launch NSight Graphics
2. Target: PlasmaDX-Clean.exe
3. Capture frame before enabling probe grid
4. Enable probe grid
5. Capture frame immediately (if doesn't crash first)
6. Export CSV for detailed analysis

**Files Generated:**
- `*_EventList.csv` - Frame timeline
- `*_objects.csv` - Resource/object catalog

---

## Key Takeaway

The crash is **GPU-side**, likely in the probe grid update shader. The fact that it crashes **instantly** when enabled suggests:
- Wrong resource binding
- Out-of-bounds memory access
- TLAS instance confusion

**Most Likely Fix:** Hard clamp probe grid to 2043 particles (Fix 1) will work.

**Root Cause Investigation:** Requires PIX/NSight GPU capture at crash point.

---

**End of Debug Summary**
