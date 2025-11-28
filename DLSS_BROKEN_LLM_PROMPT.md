# DLSS Super Resolution Initialization Failure - Expert Consultation

## Problem Statement

NVIDIA DLSS Super Resolution was **working perfectly** in my DirectX 12 ray tracing engine (PlasmaDX-Clean), but suddenly stopped functioning after recent codebase updates to volumetric ReSTIR and frustum voxel grid rendering systems (both systems have since been deprecated/removed). Despite having the correct hardware, driver, and SDK, DLSS-SR initialization now fails with NGX error code `0xBAD00014` (-1160773628).

**CRITICAL**: I need DLSS working for upcoming PINN (Physics-Informed Neural Network) integration to achieve target performance of 280+ FPS with 100K particles. I **absolutely do not want** to downgrade to older DLSS SDK versions - it was working perfectly at this version before the rendering updates broke it!

## System Configuration

### Hardware & Driver
- **GPU**: NVIDIA GeForce RTX 4060 Ti (Ada Lovelace architecture)
- **VRAM**: 7949 MB dedicated
- **Driver Version**: Latest (one of the newest available as of November 2025)
- **Driver Path**: `C:\WINDOWS\System32\DriverStore\FileRepository\nv_dispig.inf_amd64_8ca39bf4ab8eccb6`
- **DirectX**: DirectX 12 with Agility SDK
- **DXR Support**: Tier 1.2 (full ray tracing capabilities)

### DLSS SDK Configuration
- **NGX SDK API Version**: 1.5.0 (0x0000015)
- **DLSS DLL Version**: 310.4.0 (DLSS 3.1.4)
- **SDK Location**: `dlss/` directory (NVIDIA DLSS SDK, always used official SDK)
- **DLL Variants**: Both `dev` (development) and `rel` (release) available
- **Current Status**: Was hoping to upgrade to 310.4.0, possibly was using DLSS 3.7 before (uncertain of previous exact version)
- **Integration Method**: Always used official NVIDIA DLSS SDK with `NVSDK_NGX_D3D12_Init_with_ProjectID()`

### Application Details
- **Project ID**: `a0b1c2d3-4e5f-6a7b-8c9d-0e1f2a3b4c5d` (custom UUID for NGX)
- **CMS ID Mapping**: `876232c` (confirmed by NGX logs)
- **Engine Type**: `NVSDK_NGX_ENGINE_TYPE_CUSTOM`
- **Engine Version**: "1.0.0"
- **Executable**: `PlasmaDX-Clean.exe`

## Technical History

### What Changed
1. **Volumetric ReSTIR System** implemented (path generation, reservoir sampling, shading passes)
2. **Frustum Voxel Grid (Froxel) System** implemented (160×90×128 grid, density injection, lighting)
3. Both systems have since been **deprecated/removed** from active rendering
4. **DLSS stopped working** sometime during or after these updates

### Previous Working State
- DLSS Super Resolution was **fully operational** before rendering updates
- No hardware, driver, or SDK changes between working and broken states
- ImGui DLSS-SR toggle was functional and responsive
- F4 hotkey successfully toggled DLSS at runtime

### Current Broken State
- NGX SDK initializes successfully: `DLSS: NGX SDK initialized successfully`
- **Capability check fails**: `NVSDK_NGX_Parameter_SuperSampling_Available` returns 0 (false)
- Warning logged: `DLSS: Super Resolution not supported on this GPU` (FALSE - RTX 4060 Ti fully supports DLSS)
- ImGui DLSS-SR checkbox is **greyed out** (disabled)
- F4 hotkey does nothing (DLSS system never initializes)

## Detailed Diagnostics

### NGX Log Analysis (nvngx.log)

**Successful Operations:**
```
[MapProjectId:1894] Found cms id 876232c for engine: custom engineVersion 1.0.0 projectID a0b1c2d3-4e5f-6a7b-8c9d-0e1f2a3b4c5d
[NGXInitContext:245] called from module PlasmaDX-Clean.exe at D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\build\bin\Debug
```

**Critical Failure:**
```
[`anonymous-namespace'::SnippetLocationInfo::load:677] NGXLoadFromPath failed: -1160773628
```
(This error repeats ~30+ times)

**Root Cause Discovery:**
- **DLSS-G (Frame Generation)**: Driver provides `nvngx_dlssg.dll` in DriverStore ✅
- **DLSS-SR (Super Resolution)**: `nvngx_dlss.dll` **NOT FOUND** in DriverStore ❌
- **DLSS-D (Denoiser)**: `nvngx_dlssd.dll` **NOT FOUND** in DriverStore ❌

**DriverStore Contents:**
```
C:\WINDOWS\System32\DriverStore\FileRepository\nv_dispig.inf_amd64_8ca39bf4ab8eccb6\
  ├── _nvngx.dll           (1.4 MB, NGX core)
  ├── nvngx.dll            (478 KB, NGX wrapper)
  ├── nvngx_dlssg.dll      (8.9 MB, DLSS Frame Generation) ✅
  ├── nvngx_dlss.dll       ❌ MISSING
  └── nvngx_dlssd.dll      ❌ MISSING
```

**NGX Cache Status:**
- Cache location: `C:\ProgramData\NVIDIA\NGX\models\dlss\versions\`
- Cached model files exist for OTHER CMS IDs (B9B0430, B9D3EF0, B9E26B0, F7361DC)
- **No cached models for our CMS ID `876232c`** ❌
- NGX attempts to load: `160_876232C.bin` (not found)

### DLL Bundling Attempts

**Attempt 1: Bundle DEV DLLs (from SDK `dev/` directory)**
- Result: Signature validation failure `0xBAD00014`
- NGX Error: `nvLoadSignedLibraryW() failed on snippet './nvngx_dlss.dll' missing or corrupted - last error One or more arguments are not correct.`
- Reason: DLL version (310.4.0) doesn't match driver's expected signature/version

**Attempt 2: Remove All Bundled DLLs (driver-only approach)**
- Result: NGX initialization succeeds, but `SuperSampling_Available` returns 0
- Reason: Driver doesn't provide fallback DLSS-SR DLLs (only DLSS-G)

**Attempt 3: Bundle REL DLLs (from SDK `rel/` directory)**
- Status: Currently testing (same expected outcome as DEV DLLs)

## Code Implementation (Confirmed Correct)

### DLSSSystem.cpp Initialization
```cpp
// Initialize NGX SDK with Project ID
NVSDK_NGX_Result result = NVSDK_NGX_D3D12_Init_with_ProjectID(
    DLSS_PROJECT_ID,                   // "a0b1c2d3-4e5f-6a7b-8c9d-0e1f2a3b4c5d"
    NVSDK_NGX_ENGINE_TYPE_CUSTOM,      // Custom engine
    DLSS_ENGINE_VERSION,                // "1.0.0"
    appDataPath,                        // NGX cache path
    device,                             // ID3D12Device*
    &featureInfo,                       // FeatureCommonInfo (logging enabled)
    NVSDK_NGX_Version_API               // SDK version (1.5.0)
);

// Check DLSS Super Resolution availability
int dlssAvailable = 0;
result = m_params->Get(NVSDK_NGX_Parameter_SuperSampling_Available, &dlssAvailable);
m_dlssSupported = (dlssAvailable != 0);  // Returns FALSE ❌

if (!m_dlssSupported) {
    LOG_WARN("DLSS: Super Resolution not supported on this GPU");
    NVSDK_NGX_D3D12_Shutdown1(device);
    return false;
}
```

### CMakeLists.txt DLL Copying (Currently Disabled)
```cmake
# DLSS DLL Handling: Let NGX discover DLLs from driver (DO NOT bundle)
#
# CRITICAL: Modern DLSS should NOT bundle DLLs with the application!
# NGX will automatically discover and load DLSS DLLs from:
#   1. Driver's DriverStore directory
#   2. System-wide NGX cache (C:\ProgramData\NVIDIA\NGX\models\dlss\)
#   3. NVIDIA's runtime download service
#
# Bundling SDK DLLs causes signature validation failures (error 0xBAD00014)
# because the DLL version (310.4.0) doesn't match what the driver expects.
```

## Questions for Expert Analysis

### Primary Question
**What is the correct approach to restore DLSS Super Resolution functionality without downgrading SDK versions?**

### Specific Sub-Questions

1. **Driver Fallback Missing**: Why doesn't the driver provide `nvngx_dlss.dll` and `nvngx_dlssd.dll` in DriverStore when it provides `nvngx_dlssg.dll`? Is this expected behavior for certain driver versions?

2. **Signature Validation**: Why do SDK-provided DLSS DLLs (version 310.4.0) fail signature validation with error `0xBAD00014`? Is there a signing process or metadata missing from bundled DLLs?

3. **CMS ID Cache**: How can we force NGX to download and cache DLSS models for our CMS ID (`876232c`)? Is there a manual cache population method?

4. **Version Compatibility**: The NGX SDK API is 1.5.0, but DLSS DLL is 310.4.0. Is there a version mismatch? Should I be using a different DLSS SDK version?

5. **Working → Broken Transition**: Since DLSS was working before and no hardware/driver/SDK changed, what could the rendering code updates (volumetric ReSTIR, froxel grid) have done to break DLSS initialization? Are there:
   - Resource state conflicts?
   - Descriptor heap exhaustion?
   - DirectX 12 validation errors interfering with NGX?
   - Timing issues in initialization order?

6. **Streamline vs NGX Direct**: Should I migrate from direct NGX SDK integration to NVIDIA Streamline SDK for automatic DLL management? Would this solve the problem?

7. **Project ID Regeneration**: Could regenerating the UUID-based Project ID help? Perhaps the CMS ID mapping became corrupted?

8. **Alternative Initialization**: Are there alternative NGX initialization methods (e.g., `NVSDK_NGX_D3D12_Init()` instead of `Init_with_ProjectID()`) that might work better?

## Expected Solution Characteristics

### Requirements
- ✅ **Must work with current DLSS SDK 3.1.4 (310.4.0)** - no downgrades
- ✅ **Must support RTX 4060 Ti with latest drivers** - hardware is confirmed compatible
- ✅ **Must integrate with existing DirectX 12 DXR 1.1 pipeline** - volumetric ray tracing, 3D Gaussian splatting
- ✅ **Must support lazy feature creation** - DLSS feature created on first render frame, not at startup
- ✅ **Must be redistributable** - application should work on other systems without manual DLL installation

### Acceptable Solutions
1. Correct DLL bundling approach with proper signing/validation
2. Method to force NGX to download missing DLLs at runtime
3. Configuration change to make driver provide missing DLSS-SR DLLs
4. Alternative SDK integration method (e.g., Streamline)
5. Any fix that restores DLSS-SR to working state **without downgrading**

## Additional Context

### Performance Requirements
- **Current**: 120 FPS @ 10K particles, 1080p, multi-light RT
- **Target with PINN**: 280+ FPS @ 100K particles with ML physics
- **DLSS Critical For**: Achieving target performance with PINN implementation
- **Quality Mode**: Performance mode (50% render scale) for maximum FPS gain

### Rendering Pipeline
- **Primary Renderer**: 3D Gaussian Splatting (particle_gaussian_raytrace.hlsl)
- **RT Lighting**: DXR 1.1 RayQuery API inline ray tracing
- **Shadow System**: PCSS (Percentage-Closer Soft Shadows) with temporal accumulation
- **Upscaling Need**: Render at lower resolution (e.g., 720p → 1440p) for performance

### What Has NOT Changed
- ✅ GPU hardware (RTX 4060 Ti)
- ✅ NVIDIA driver version (latest maintained throughout)
- ✅ DLSS SDK files (same DLLs in `dlss/` directory)
- ✅ NGX initialization code (unchanged in DLSSSystem.cpp)
- ✅ Windows OS version

### What HAS Changed
- ❌ Volumetric ReSTIR system added (then deprecated)
- ❌ Froxel volumetric fog system added (then deprecated)
- ❌ DLSS stopped working sometime during these updates
- ❌ Possibly CMake build configuration for DLL copying (uncertain)

## Files Available for Analysis

If you need to see specific code, I can provide:
- `src/dlss/DLSSSystem.h` (DLSS wrapper interface)
- `src/dlss/DLSSSystem.cpp` (NGX initialization, feature creation, evaluation)
- `src/particles/ParticleRenderer_Gaussian.cpp` (DLSS integration in rendering loop)
- `CMakeLists.txt` (build configuration, DLL copying logic)
- `build/bin/Debug/ngx/nvngx.log` (full NGX diagnostic log)
- `build/bin/Debug/logs/PlasmaDX-Clean_*.log` (application logs)

## Request for Gemini 3 Pro / OPUS 4.5

Please provide:
1. **Root cause analysis** of why DLSS-SR stopped working
2. **Step-by-step solution** to restore functionality without downgrading
3. **Explanation of NGX DLL discovery** and why driver fallback is missing
4. **Best practices** for DLSS SDK integration in custom DirectX 12 engines
5. **Preventive measures** to avoid this issue in future codebase updates

Thank you for your expertise!

---

**Generated**: 2025-11-28 01:23 UTC
**Project**: PlasmaDX-Clean (DirectX 12 Volumetric Ray Tracing Engine)
**Version**: 0.20.1
**DLSS SDK**: NVIDIA NGX 1.5.0 with DLSS 3.1.4 (310.4.0)
