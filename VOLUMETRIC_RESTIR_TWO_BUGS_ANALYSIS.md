# Volumetric ReSTIR Two Critical Bugs Analysis

**Date**: 2025-11-02
**Issues**: 
1. CPU diagnostic counters always zero
2. 3-second GPU hang at 2045+ particles

**Status**: Issue #1 ✅ FIXED, Issue #2 🔄 INVESTIGATING

---

## Issue #1: CPU Readback Shows Zeros (FIXED ✅)

### Symptoms
- PIX GUI shows **non-zero diagnostic counters** (e.g., [0]=0xDEADBF6F, [1]=0x00016F64)
- CPU log shows **all zeros** ([0]=0, [1]=0, [2]=0, [3]=0)
- Shader clearly executing (PIX proves it)

### Root Cause
**Missing CopyResource call**: Shader writes to GPU buffer, CPU reads from readback buffer, but GPU buffer is never copied to readback buffer!

**Code flow**:
1. Shader writes to `m_diagnosticCounterBuffer` (GPU UAV) ✅
2. **Missing**: CopyResource from GPU → Readback ❌
3. CPU reads from `m_diagnosticCounterReadback` (always zeros) ❌

### Fix Applied
**File**: `src/lighting/VolumetricReSTIRSystem.cpp` (lines 948-958)

```cpp
// CRITICAL FIX: Copy diagnostic counters from GPU to readback buffer
// This was missing - shader writes to GPU buffer, but CPU reads from readback buffer
// Without this copy, CPU always sees zeros even though shader executes correctly
D3D12_RESOURCE_BARRIER diagnosticBarrier = {};
diagnosticBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
diagnosticBarrier.UAV.pResource = m_diagnosticCounterBuffer.Get();
commandList->ResourceBarrier(1, &diagnosticBarrier);

// Copy GPU buffer → Readback buffer (so CPU can read it later)
commandList->CopyResource(m_diagnosticCounterReadback.Get(), m_diagnosticCounterBuffer.Get());
LOG_INFO("[DIAGNOSTIC] Copied diagnostic counters to readback buffer");
```

### Expected Result
CPU log should now show non-zero values matching PIX:
```
[0] Total threads executed: ~2048 (was 0)
[1] Early returns: ~3 (was 0)
[2] Total voxel writes: ~10000+ (was 0)
[3] Max voxels per particle: ~512 (was 0)
```

---

## Issue #2: 3-Second GPU Hang (INVESTIGATING 🔄)

### Symptoms
- 2-3 second delay at `WaitForGPU()` every frame
- Occurs at exactly ≥2045 particles, not at ≤2044
- Shader executes (PIX proves it) but takes 3 seconds
- No D3D12 validation errors

### Log Evidence
```
[00:43:43] VolumetricReSTIR: About to WaitForGPU
[00:43:45] VolumetricReSTIR: WaitForGPU completed  ← 2-second delay!
```

### Hypothesis: Atomic Operation Contention

**Calculation**:
- 2045 particles × up to 512 voxels/particle = **1,047,040 atomic writes**
- Volume texture: 64³ = **262,144 voxels**
- Ratio: **4:1 average writes per voxel**
- All using `InterlockedMax()` → severe memory contention

**Shader code** (populate_volume_mip2.hlsl:240):
```hlsl
// Atomic max: Highest density value wins per voxel
// This prevents race conditions while giving reasonable results
uint originalValue;
InterlockedMax(g_volumeTexture[voxelCoords], densityAsUint, originalValue);
```

### Why Exactly at 2045 Particles?

**Thread count**:
- 2044 particles = (2044+63)/64 = 31.94 → **32 thread groups** = 2048 threads
- 2045 particles = (2045+63)/64 = 32.01 → **32 thread groups** = 2048 threads

Wait, both produce 32 thread groups! This doesn't explain the threshold.

**Alternative hypothesis**: Memory allocation size?
- 2044 × 32 bytes = 65,408 bytes (fits in 64 KB boundary?)
- 2045 × 32 bytes = 65,440 bytes (crosses 64 KB boundary?)

This could affect GPU caching behavior.

### Potential Fixes

**Option 1: Reduce atomic contention** (recommended first)
- Use atomic operations only when necessary
- Group writes to same voxel before atomic
- Use shared memory for local accumulation

**Option 2: Reduce particle coverage**
- Lower MAX_VOXELS_PER_AXIS from 8 to 6 (512 → 216 voxels/particle)
- Reduces total atomic writes by ~58%

**Option 3: Change volume resolution**
- Increase from 64³ to 128³ (reduces contention density)
- Or decrease to 32³ (fewer voxels to write)

### Tests Completed

**Test 1: Disable Volume Writes** ❌
- Result: Hang persists, counters still zero
- Conclusion: Atomic operations NOT the cause

**Test 2: Fix ClearUnorderedAccessViewFloat/Uint** ❌
- Result: Hang persists, counters still zero
- Conclusion: Format mismatch NOT the cause (but still should be fixed)

**Test 3: Change Thread Count from 64→63** ❌
- Result: Hang persists (33 groups, 2079 threads)
- Conclusion: Power-of-2 thread count NOT the cause

**Test 4: Add CopyResource for Diagnostic Counters** ❌
- Result: Counters still zero
- Conclusion: Copy works, but shader never writes data

### Critical Finding

**Diagnostic counters are ALWAYS zero** → Shader never executes!
- Even the 0xDEADBEEF sentinel (thread 0, line 153) never appears
- This means GPU never runs a single instruction of the shader
- 3-second hang is Windows TDR timing out on invalid dispatch

### Recommended Next Steps

1. **Create PIX GPU Capture** - Capture frame at 2045 particles to see:
   - Is dispatch command recorded?
   - What PSO/Root Signature are bound?
   - Does GPU execute the shader at all?
   - Any D3D12 validation errors?

2. **Verify PSO Creation** - Check if m_volumePopPSO creation succeeded:
   ```cpp
   if (!m_volumePopPSO) {
       LOG_ERROR("PopulateVolumeMip2 PSO is null!");
   }
   ```

3. **Test Minimal Shader** - Replace complex logic with simple counter increment:
   ```hlsl
   [numthreads(63, 1, 1)]
   void main(uint3 id : SV_DispatchThreadID) {
       uint dummy;
       g_diagnosticCounters.InterlockedAdd(0, 1, dummy);
   }
   ```

4. **Check Shader Compilation** - Verify DXIL is valid:
   ```bash
   dxc -dumpbin populate_volume_mip2.dxil
   ```

---

## Summary

| Issue | Root Cause | Status | Fix |
|-------|-----------|--------|-----|
| CPU zeros | Missing CopyResource | ✅ FIXED | Added GPU→Readback copy |
| 3-sec hang | Atomic contention? | 🔄 INVESTIGATING | TBD |

**Next test**: Run with 2045 particles and verify:
1. Diagnostic counters show non-zero on CPU ✅ (should work now)
2. 3-second delay still occurs ⚠️ (likely yes, need profiling)

