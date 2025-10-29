# DLSS 4.0 Phase 2 Integration Summary (v0.11.1)

**Date:** 2025-10-29
**Branch:** v0.11.1
**Status:** ✅ Phase 2 COMPLETE - NGX SDK Initialization Successful

---

## What We Accomplished

Successfully integrated NVIDIA DLSS 4.0 NGX SDK initialization into PlasmaDX-Clean. The SDK now initializes correctly on RTX 4060 Ti with driver 581.57 and loads the Ray Reconstruction denoiser (DLSS-RR v310.3.0).

---

## Critical Discoveries & Fixes

### 1. **Missing Main SDK DLL**
**Problem:** Only copied `nvngx_dlssd.dll` (denoiser), but NGX SDK requires the main `nvngx_dlss.dll` to initialize.

**Fix:** Updated CMakeLists.txt to copy BOTH DLLs:
```cmake
# Main DLSS SDK DLL (REQUIRED for NGX initialization)
COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${DLSS_SDK_DIR}/lib/Windows_x86_64/$<IF:$<CONFIG:Debug>,dev,rel>/nvngx_dlss.dll"
    "$<TARGET_FILE_DIR:${PROJECT_NAME}>/"
# DLSS Denoiser / Ray Reconstruction DLL
COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${DLSS_SDK_DIR}/lib/Windows_x86_64/$<IF:$<CONFIG:Debug>,dev,rel>/nvngx_dlssd.dll"
    "$<TARGET_FILE_DIR:${PROJECT_NAME}>/"
```

**Location:** `CMakeLists.txt:308-321`

---

### 2. **ProjectID Format Validation**
**Problem:** NGX SDK rejected project IDs with ANY alphabetic characters.

**Error Logs:**
```
[NGXValidateIdentifier] Error: projectID [PlasmaDX-Clean] contains invalid character [P]
[NGXValidateIdentifier] Error: projectID [plasmadx-clean] contains invalid character [p]
```

**Solution:** ProjectID **MUST** be in UUID/GUID format (hexadecimal only).

**Fix:**
```cpp
// BEFORE (FAILED):
#define DLSS_PROJECT_ID "PlasmaDX-Clean"

// AFTER (SUCCESS):
#define DLSS_PROJECT_ID "a0b1c2d3-4e5f-6a7b-8c9d-0e1f2a3b4c5d"
```

**Location:** `src/dlss/DLSSSystem.cpp:11`

---

### 3. **API Signature Corrections**

#### 3.1 Init Function Parameters
**Problem:** Incorrect parameter types and missing parameters.

**Fix:**
```cpp
// 7 parameters required (was passing incorrectly)
NVSDK_NGX_Result result = NVSDK_NGX_D3D12_Init_with_ProjectID(
    DLSS_PROJECT_ID,                   // const char* (narrow string!)
    NVSDK_NGX_ENGINE_TYPE_CUSTOM,      // Custom engine
    DLSS_ENGINE_VERSION,                // const char* "1.0.0"
    appDataPath,                        // const wchar_t* (absolute path)
    device,                             // ID3D12Device*
    &featureInfo,                       // NVSDK_NGX_FeatureCommonInfo* (NOT nullptr!)
    NVSDK_NGX_Version_API               // SDK version enum
);
```

**Key Changes:**
- Project ID: narrow string (`const char*`), not wide string
- Engine version: string literal `"1.0.0"`, not macro
- FeatureCommonInfo: required structure, not `nullptr`
- SDK version: `NVSDK_NGX_Version_API` enum value

**Location:** `src/dlss/DLSSSystem.cpp:41-49`

#### 3.2 Shutdown Function
**Problem:** Using deprecated `NVSDK_NGX_D3D12_Shutdown()` (no parameters).

**Fix:** Changed to `NVSDK_NGX_D3D12_Shutdown1(device)` in 4 locations:
- Lines 53, 57, 71, 83, 103

#### 3.3 Parameter Names
**Problem:** Wrong parameter name for denoiser strength.

**Fix:**
```cpp
// BEFORE:
evalParams->Set(NVSDK_NGX_Parameter_Denoise_Strength, m_denoiserStrength);

// AFTER:
evalParams->Set(NVSDK_NGX_Parameter_Denoise, m_denoiserStrength);
```

**Location:** `src/dlss/DLSSSystem.cpp:214`

---

### 4. **FeatureCommonInfo Configuration**

**Problem:** Passing `nullptr` for FeatureCommonInfo caused 0xBAD00005 (INVALID_PARAMETER).

**Solution:** Create proper structure with logging enabled.

**Fix:**
```cpp
// Configure NGX common features (logging, DLL paths)
const wchar_t* dllPaths[] = { L"." };  // Search local directory first
NVSDK_NGX_FeatureCommonInfo featureInfo = {};
featureInfo.LoggingInfo.LoggingCallback = nullptr;  // Use default file logging
featureInfo.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_ON;
featureInfo.LoggingInfo.DisableOtherLoggingSinks = false;
featureInfo.PathListInfo.Path = dllPaths;
featureInfo.PathListInfo.Length = 1;
```

**Benefits:**
- Enabled NGX logging to `build/bin/Debug/ngx/` directory
- Logs revealed the ProjectID validation error
- Proper DLL search path configuration

**Location:** `src/dlss/DLSSSystem.cpp:31-38`

---

### 5. **NGX Data Path**

**Problem:** Passing relative path `L"./"` caused initialization issues.

**Solution:** Create dedicated directory and pass absolute path.

**Fix (Application.cpp:265-269):**
```cpp
// Create NGX directory for DLSS logs and cache
CreateDirectoryW(L"ngx", nullptr); // Ignore error if exists
wchar_t ngxPath[MAX_PATH];
GetFullPathNameW(L"ngx", MAX_PATH, ngxPath, nullptr);
LOG_INFO("NGX data path: {}", std::filesystem::path(ngxPath).string());

m_dlssSystem = std::make_unique<DLSSSystem>();
if (m_dlssSystem->Initialize(m_device->GetDevice(), ngxPath)) {
```

**Result:** NGX now writes logs and cache to `build/bin/Debug/ngx/`

---

### 6. **Error Logging Format**

**Problem:** Logger uses `{}` placeholders, not printf-style format specifiers.

**Fix:** Convert error codes to hex strings before logging:
```cpp
// BEFORE (displayed literally as "0x{:X}"):
LOG_ERROR("DLSS: Failed to initialize NGX SDK: 0x{:X}", static_cast<uint32_t>(result));

// AFTER (displays actual hex code like "0xBAD00005"):
char hexStr[16];
sprintf_s(hexStr, "%08X", static_cast<uint32_t>(result));
LOG_ERROR("DLSS: Failed to initialize NGX SDK: 0x{}", hexStr);
```

**Location:** Applied to 6 error logging sites in `DLSSSystem.cpp`

---

### 7. **CMake Source Ordering**

**Problem:** DLSS sources added to build before `ENABLE_DLSS` flag was set.

**Fix:** Moved DLSS SDK detection from line 98-106 to line 24-32 (BEFORE SOURCES list).

**Location:** `CMakeLists.txt:24-32`

---

## Files Modified

### 1. **CMakeLists.txt**
- Moved DLSS SDK detection before SOURCES list (line 24-32)
- Added both `nvngx_dlss.dll` and `nvngx_dlssd.dll` copying (lines 308-321)
- Removed duplicate DLSS configuration section

### 2. **src/dlss/DLSSSystem.cpp**
- Fixed ProjectID to UUID format
- Fixed all API function calls (Init, Shutdown, parameter names)
- Added FeatureCommonInfo configuration
- Fixed error logging format (6 locations)

### 3. **src/core/Application.cpp**
- Added `#include <filesystem>` (line 21)
- Added NGX directory creation with absolute path (lines 265-269)
- Integrated DLSSSystem initialization (lines 262-286)

### 4. **src/core/Application.h**
- Added DLSSSystem forward declaration with `#ifdef ENABLE_DLSS`
- Added member variable `std::unique_ptr<DLSSSystem> m_dlssSystem;`
- Added runtime controls: `m_enableDLSS`, `m_dlssDenoiserStrength`

---

## Final Success

**Application Log (build/bin/Debug/logs/PlasmaDX-Clean_20251029_005603.log):**
```
[00:56:04] [INFO] Initializing DLSS Ray Reconstruction...
[00:56:04] [INFO] NGX data path: D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\build\bin\Debug\ngx
[00:56:06] [INFO] DLSS: NGX SDK initialized successfully
[00:56:06] [INFO] DLSS: Ray Reconstruction supported
```

**NGX Logs (build/bin/Debug/ngx/):**
```
[NGXSecureLoadFeature] app 876232C feature dlssd snippet:
    C:\ProgramData/NVIDIA/NGX/models/dlssd\versions\20316928\files/160_E658700.bin
    version: 310.3.0
```

DLSS denoiser v310.3.0 loaded successfully from driver-provided NGX models!

---

## Known Issue: CreateFeature Crash

**Problem:** Application crashes after successful Init when calling `CreateRayReconstructionFeature()`.

**Root Cause:** `NVSDK_NGX_D3D12_CreateFeature()` requires a **valid ID3D12GraphicsCommandList**, currently passing `nullptr`.

**Function Signature:**
```cpp
NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_CreateFeature(
    ID3D12GraphicsCommandList *InCmdList,  // ← CANNOT be nullptr!
    NVSDK_NGX_Feature InFeatureID,
    const NVSDK_NGX_Parameter *InParameters,
    NVSDK_NGX_Handle **OutHandle
);
```

**Current Code (Application.cpp:272-275):**
```cpp
if (m_dlssSystem->CreateRayReconstructionFeature(m_width, m_height)) {
    // CreateFeature passes nullptr for command list → CRASH
}
```

**Solution:** Defer `CreateRayReconstructionFeature()` call until after command lists are created during rendering initialization (likely in Gaussian renderer setup).

**Status:** Phase 2 complete (Init success), feature creation deferred to Phase 2.5/3.

---

## Test Hardware

- **GPU:** NVIDIA GeForce RTX 4060 Ti (Ada Lovelace)
- **Driver:** 581.57 (requirement: 531.00+)
- **OS:** Windows 11 (WSL2 build environment)
- **Resolution:** 1920×1080

---

## Next Steps (Phase 2.5)

1. **Move CreateFeature Call:**
   - Remove from Application.cpp initialization
   - Add to Gaussian renderer initialization (after command list creation)
   - Pass valid command list from rendering context

2. **Add Lazy Feature Creation:**
   - Create feature on first render (when command list is available)
   - Store feature handle for reuse across frames

3. **Add ImGui Controls:**
   - F4 toggle for DLSS on/off
   - Denoiser strength slider (0.0-2.0)
   - Status indicator showing feature readiness

4. **Testing:**
   - Verify no crashes with valid command list
   - Check feature creates successfully
   - Confirm feature handle is valid for evaluation

---

## Key Learnings

1. **NVIDIA NGX ProjectID Validation is Strict:**
   - Only UUID/GUID format accepted
   - No alphabetic characters allowed
   - Hyphens required in standard UUID format

2. **FeatureCommonInfo is NOT Optional:**
   - Logging configuration required for debugging
   - DLL search paths improve load reliability
   - Enables diagnostic log files

3. **NGX Logging is Essential:**
   - Revealed ProjectID validation error
   - Shows DLL loading sequence and fallbacks
   - Confirms denoiser version and source

4. **Command Lists are Required:**
   - CreateFeature is NOT a standalone call
   - Must be called within active GPU command recording
   - Deferred initialization pattern needed

---

**Last Updated:** 2025-10-29
**Branch:** v0.11.1
**Next Phase:** Phase 2.5 - Feature Creation with Command List
