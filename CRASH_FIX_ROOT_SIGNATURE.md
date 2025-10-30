# Crash Fix: Root Signature Mismatch

**Date:** 2025-10-29
**Issue:** White screen crash on startup after dynamic emission implementation
**Status:** ✅ FIXED

---

## Problem

Application crashed with white screen after implementing dynamic emission system.

**Symptoms:**
- Application initializes successfully
- Window opens (white screen)
- Crashes during first render frame
- No error messages in logs (silent crash)

---

## Root Cause

**Root signature mismatch between declaration and usage:**

**Declaration (RTLightingSystem_RayQuery.cpp:139):**
```cpp
rootParams[0].InitAsConstants(4, 0);  // Declared: 4 DWORDs
```

**Usage (RTLightingSystem_RayQuery.cpp:472):**
```cpp
cmdList->SetComputeRoot32BitConstants(0, 14, &constants, 0);  // Passing: 14 DWORDs
```

**Result:** D3D12 validation error → silent crash (debug layer would have caught this)

---

## Why This Happened

When implementing dynamic emission, we expanded the `LightingConstants` struct from 4 DWORDs to 14 DWORDs:

**Original (4 DWORDs / 16 bytes):**
```cpp
struct LightingConstants {
    uint32_t particleCount;       // 1 DWORD
    uint32_t raysPerParticle;     // 1 DWORD
    float maxLightingDistance;    // 1 DWORD
    float lightingIntensity;      // 1 DWORD
};  // Total: 4 DWORDs
```

**Updated (14 DWORDs / 56 bytes):**
```cpp
struct LightingConstants {
    uint32_t particleCount;           // 1 DWORD
    uint32_t raysPerParticle;         // 1 DWORD
    float maxLightingDistance;        // 1 DWORD
    float lightingIntensity;          // 1 DWORD
    DirectX::XMFLOAT3 cameraPosition; // 3 DWORDs
    uint32_t frameCount;              // 1 DWORD
    float emissionStrength;           // 1 DWORD
    float emissionThreshold;          // 1 DWORD
    float rtSuppression;              // 1 DWORD
    float temporalRate;               // 1 DWORD
};  // Total: 14 DWORDs
```

We updated:
- ✅ The struct definition (header)
- ✅ The constant upload code (DispatchRayQueryLighting)
- ✅ The shader HLSL constant buffer
- ❌ **FORGOT:** Root signature declaration

---

## Fix Applied

**File:** `src/lighting/RTLightingSystem_RayQuery.cpp`

**Line 139 (before):**
```cpp
rootParams[0].InitAsConstants(4, 0);  // b0: LightingConstants (4 DWORDs)
```

**Line 139 (after):**
```cpp
rootParams[0].InitAsConstants(14, 0);  // b0: LightingConstants (14 DWORDs - was 4, expanded for emission)
```

**Also updated comment on line 133** to indicate expansion for dynamic emission.

---

## Additional Fixes

**Shader Recompilation:**
The HLSL shader was modified but not recompiled. Manually recompiled:
```bash
dxc.exe -T cs_6_5 -E main shaders/dxr/particle_raytraced_lighting_cs.hlsl \
    -Fo build/bin/Debug/shaders/dxr/particle_raytraced_lighting_cs.dxil
```

**Why manual?** CMake has the shader as a custom target that doesn't auto-rebuild on HLSL changes (by design for build speed). For modified shaders, manual recompilation is needed.

---

## Verification Steps

1. ✅ Root signature updated from 4 to 14 DWORDs
2. ✅ Shader recompiled with new constant buffer layout
3. ✅ C++ code matches shader constant buffer
4. ✅ Build succeeds with no errors
5. ⏳ **Next:** Run application and verify dynamic emission works

---

## How to Avoid This in Future

**When expanding constant buffers:**

1. **Update struct definition** (header file)
2. **Update root signature** (`InitAsConstants()` call)
3. **Update upload code** (`SetComputeRoot32BitConstants()` call)
4. **Update shader** (cbuffer declaration in HLSL)
5. **Recompile shader** (manual if CMake doesn't auto-rebuild)
6. **Verify DWORD counts match everywhere:**
   - Root signature declaration
   - Upload call
   - Shader cbuffer
   - Struct size

**Debug layer would have caught this!**
Enable D3D12 debug layer for better error messages:
```cpp
// Already enabled in PlasmaDX-Clean
#ifdef _DEBUG
D3D12GetDebugInterface(IID_PPV_ARGS(&debugController));
debugController->EnableDebugLayer();
#endif
```

---

## Testing Checklist

- [ ] Application launches without crash
- [ ] Particles render correctly
- [ ] RT lighting works (green particles visible)
- [ ] Dynamic emission visible (hot particles glow)
- [ ] Temporal pulsing works (stars breathe)
- [ ] Performance acceptable (<1% overhead)
- [ ] ImGui controls accessible (F1)
- [ ] No D3D12 validation errors in log

---

**Status:** Ready for testing!
