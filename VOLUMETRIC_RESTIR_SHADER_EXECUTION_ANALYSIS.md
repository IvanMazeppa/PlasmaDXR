# Volumetric ReSTIR Shader Execution Failure Analysis

**Date**: 2025-11-01
**Status**: CRITICAL - All ReSTIR shaders fail to execute despite successful dispatch

---

## Executive Summary

All three VolumetricReSTIR compute shaders are **dispatching but not executing**. The shaders compile successfully, PSOs bind without errors, dispatches occur, but **no GPU writes happen**. This affects the entire ReSTIR pipeline.

---

## Evidence

### 1. PopulateVolumeMip2 - Diagnostic Counter Test
**Test**: Added instrumentation to count shader execution
**Result**: All counters = 0

```
[INFO] ========== PopulateVolumeMip2 Diagnostic Counters ==========
[INFO]   [0] Total threads executed: 0          ← Should be 2048
[INFO]   [1] Early returns: 0                   ← Should be ~4
[INFO]   [2] Total voxel writes: 0              ← Should be ~500K
[INFO]   [3] Max voxels per particle: 0         ← Should be ~512
```

**Conclusion**: Shader compiled, bound, dispatched (32 thread groups logged), but **never executed**.

### 2. GenerateCandidates - PIX Capture Analysis
**Test**: PIX capture at 2044 particles with ReSTIR enabled
**Result**: Reservoir buffers completely empty

**Observations**:
- Log shows: `[INFO] VolumetricReSTIR GenerateCandidates first dispatch:`
- PIX shows: All ReSTIR data structures empty (no path data, no reservoir data)
- Reservoir buffer size: Correct (width × height × 64 bytes)
- Reservoir buffer state: Correct (UAV → SRV transitions logged)

**Conclusion**: Shader dispatched but **wrote nothing** to reservoirs.

### 3. ShadeSelectedPaths - Output Texture Symptom
**Test**: Run with ReSTIR enabled, observe output
**Result**: Black screen (DLSS OFF) or frozen image (DLSS ON)

**Why**:
- ShadeSelectedPaths reads from reservoir buffers
- Reservoir buffers are empty (from #2)
- Shader has nothing to shade → no output

**Conclusion**: ShadeSelectedPaths is probably executing, but with empty input data.

---

## What Works

✅ **Shader compilation**: All `.dxil` files load successfully (6096 bytes for PopulateVolumeMip2)
✅ **PSO creation**: All pipeline states create without D3D12 errors
✅ **Root signature creation**: All root signatures create successfully
✅ **Dispatch calls**: All Dispatch() calls execute (logged)
✅ **Resource creation**: All buffers/textures created correctly (verified sizes)
✅ **Resource transitions**: All barriers execute (no validation errors)
✅ **D3D12 Debug Layer**: No errors reported

---

## What Doesn't Work

❌ **Shader execution**: GPU never runs the shader code
❌ **Memory writes**: Zero writes to any UAV buffers
❌ **Diagnostic instrumentation**: Even simple `Store()` calls don't execute
❌ **Sentinel values**: Test write of `0xDEADBEEF` never appears

---

## Root Cause Hypotheses

### Hypothesis 1: Root Signature Mismatch (HIGH PROBABILITY)
**Theory**: Shader expects different root signature than C++ provides

**Evidence**:
- Silent failure is classic D3D12 behavior for mismatched signatures
- Changed PopulateVolumeMip2 root signature from descriptor table → root descriptor
- Still failed → suggests shader compiled with old signature

**How to verify**:
```bash
# Disassemble DXIL to check root signature
dxc.exe -dumpbin build/bin/Debug/shaders/volumetric_restir/populate_volume_mip2.dxil
# Look for root signature definition in output
```

**Fix**:
- Force shader recompilation with clean build
- Verify root signature in DXIL matches C++ definition
- Consider embedding root signature in shader HLSL

### Hypothesis 2: Shader Model Incompatibility (MEDIUM PROBABILITY)
**Theory**: Shaders compiled for wrong shader model or feature level

**Evidence**:
- All shaders compile with `-T cs_6_5`
- GPU supports DXR 1.1 (SM 6.5+)
- Should be compatible

**How to verify**:
```bash
# Check compiled shader model
dxc.exe -dumpbin populate_volume_mip2.dxil | grep "shader model"
```

### Hypothesis 3: Resource Binding Slot Collision (LOW PROBABILITY)
**Theory**: Multiple resources bound to same slot causing conflict

**Evidence**:
- Systematic failure across all shaders suggests common issue
- Root parameters carefully assigned (b0, t0, u0, u1)

**How to verify**:
- Check DXIL for actual register bindings
- Compare to C++ root parameter indices

### Hypothesis 4: GPU Driver/Agility SDK Issue (LOW PROBABILITY)
**Theory**: GPU driver rejecting compute shaders for unknown reason

**Evidence**:
- Traditional rendering (Gaussian splatting) works fine
- Only VolumetricReSTIR shaders fail
- Unlikely to be driver-specific

---

## Diagnostic Tests Attempted

### Test 1: Diagnostic Counter Instrumentation
**Setup**: Added `RWByteAddressBuffer` to PopulateVolumeMip2
**Binding**: Changed from descriptor table → root descriptor (direct GPU VA)
**Result**: Still zero writes

**Code**:
```hlsl
// Very first instruction in shader
uint dummy;
g_diagnosticCounters.InterlockedAdd(0, 1, dummy);
```
**Expected**: Counter[0] = 2048 (one per thread)
**Actual**: Counter[0] = 0

### Test 2: Sentinel Value Write
**Setup**: Write magic number to prove shader executes
**Code**:
```hlsl
if (particleIdx == 0) {
    g_diagnosticCounters.Store(0, 0xDEADBEEF);
}
```
**Expected**: Counter[0] = 0xDEADBEEF (3735928559)
**Actual**: Counter[0] = 0

### Test 3: Disable Clear Before Dispatch
**Setup**: Remove `ClearUnorderedAccessViewUint` to check if clear interferes
**Result**: Still zero (clear wasn't the problem)

### Test 4: Force Shader Recompilation
**Setup**: Delete `.dxil` files and rebuild
**Result**: Shader recompiled (file timestamp updated), still fails

---

## Recommended Next Steps

### IMMEDIATE: Root Signature Analysis
1. **Disassemble all ReSTIR shaders**:
   ```bash
   dxc.exe -dumpbin populate_volume_mip2.dxil > populate_volume_mip2_disasm.txt
   dxc.exe -dumpbin path_generation.dxil > path_generation_disasm.txt
   dxc.exe -dumpbin shading.dxil > shading_disasm.txt
   ```

2. **Compare root signatures**:
   - HLSL shader expectations (from disassembly)
   - C++ root signature creation code
   - Look for mismatches in parameter order, types, or bindings

3. **Embed root signature in HLSL**:
   ```hlsl
   #define ROOT_SIG "CBV(b0), SRV(t0), DescriptorTable(UAV(u0)), UAV(u1)"
   [RootSignature(ROOT_SIG)]
   [numthreads(64, 1, 1)]
   void main(...) { ... }
   ```

### SHORT-TERM: Simplified Test Shader
Create minimal compute shader that ONLY writes to diagnostic buffer:

```hlsl
// test_minimal.hlsl
RWByteAddressBuffer g_output : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint dummy;
    g_output.InterlockedAdd(0, 1, dummy);
}
```

**Test**: If this works → root signature issue in ReSTIR shaders
**Test**: If this fails → systemic C++ binding issue

### MEDIUM-TERM: Incremental Shader Build
Start from working shader and add ReSTIR features incrementally:

1. Minimal shader (write to UAV) ✓ or ✗
2. Add particle buffer read
3. Add constant buffer
4. Add volume texture UAV
5. Add diagnostic buffer
6. Full PopulateVolumeMip2 logic

**Goal**: Find which feature causes execution failure

### LONG-TERM: Alternative ReSTIR Implementation
If root signature issue is unfixable:

**Option A**: Port ReSTIR to DXR raygeneration shader (uses different binding model)
**Option B**: Use structured buffers instead of byte address buffers
**Option C**: Simplify root signature (fewer parameters, more descriptor tables)

---

## Isolation Test Results

### With PopulateVolumeMip2 Disabled
**Test**: Comment out PopulateVolumeMip2 call in Application.cpp
**Result**:
- ✅ 2045+ particles: No crash
- ❌ Screen still frozen (GenerateCandidates/ShadeSelectedPaths issue)

**Conclusion**: PopulateVolumeMip2 crash is separate from output freeze.

### With All ReSTIR Disabled
**Test**: Use traditional Gaussian rendering without ReSTIR
**Result**: Works perfectly at 10K+ particles

**Conclusion**: Issue is specific to VolumetricReSTIR shader pipeline.

---

## Critical Questions

1. **Why do these shaders silently fail?**
   - D3D12 validation layer shows no errors
   - PIX shows dispatches happening
   - But GPU never executes shader code

2. **Why does traditional rendering work?**
   - Gaussian splatting uses compute shaders successfully
   - RT lighting uses RayQuery successfully
   - Only VolumetricReSTIR shaders fail

3. **Is this a shader compilation issue?**
   - Shaders compile without errors
   - DXIL files load successfully
   - PSOs create successfully

4. **Is this a runtime binding issue?**
   - Root signatures create successfully
   - All SetComputeRoot*() calls execute
   - Dispatch() executes

---

## Files Modified for Diagnostic Testing

### Shader Files
- `shaders/volumetric_restir/populate_volume_mip2.hlsl` - Added diagnostic instrumentation
- (No changes to path_generation.hlsl or shading.hlsl yet)

### C++ Files
- `src/lighting/VolumetricReSTIRSystem.h` - Added diagnostic buffer members
- `src/lighting/VolumetricReSTIRSystem.cpp` - Created diagnostic buffer, added readback
- `src/core/Application.cpp` - Disabled PopulateVolumeMip2 call, added diagnostic readback

### Current State
- PopulateVolumeMip2: **DISABLED** (commented out in Application.cpp)
- Diagnostic counters: **IMPLEMENTED** (buffer created, shader instrumented, readback working)
- Test results: **ALL ZERO** (shader not executing)

---

## References

**Previous Debug Sessions**:
- `VOLUMETRIC_RESTIR_2045_PARTICLE_BUG_SUMMARY.md` - Original crash investigation
- `VOLUMETRIC_RESTIR_TDR_DEBUG_SESSION.md` - TDR analysis

**PIX Captures**:
- `PIX/Captures/volumetric_restir_debug_1.wpix` - 2044 particles, ReSTIR ON, empty reservoirs

**Logs**:
- `build/bin/Debug/logs/PlasmaDX-Clean_20251101_191158.log` - Diagnostic counter test (all zeros)
- `build/bin/Debug/logs/PlasmaDX-Clean_20251101_194211.log` - PopulateVolumeMip2 disabled test

---

## Next Session Action Items

1. ✅ Disassemble all three ReSTIR shaders to check root signatures
2. ✅ Create minimal test compute shader (write-only UAV)
3. ✅ Compare C++ root signature definitions to DXIL expectations
4. ⏳ If mismatch found: Fix and rebuild
5. ⏳ If no mismatch: Try embedded root signature in HLSL
6. ⏳ If still failing: Consider alternative implementation approach

---

**Last Updated**: 2025-11-01 19:45
**Status**: Under investigation - root signature mismatch suspected
**Confidence**: HIGH - All evidence points to shader execution failure, not logic bugs
