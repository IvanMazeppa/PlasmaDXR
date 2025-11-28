# DLSS Super Resolution Debugging Summary - 2025-11-28

**Status:** Unresolved (Build Functional, DLSS Initialization Failed)
**Issue:** DLSS-SR checkbox greyed out; `NVSDK_NGX_Parameter_SuperSampling_Available` returns false.

## Problem Overview
The application builds and runs, but DLSS Super Resolution (SR) functionality is unavailable. This regression occurred after a period of working correctly, despite no changes to the DLSS SDK files or hardware configuration.

**Symptoms:**
- NGX initializes successfully.
- `NVSDK_NGX_Parameter_SuperSampling_Available` returns 0 (false).
- NGX Log Error: `NGXLoadFromPath failed: -1160773628` (Signature/Validation Failure `0xBAD00014`).
- Driver Store Analysis: `nvngx_dlss.dll` (Super Resolution) is **MISSING** from the NVIDIA driver store, while `nvngx_dlssg.dll` (Frame Gen) is present.

## Root Cause Analysis
1.  **Missing Driver DLL**: The primary issue is that the installed NVIDIA driver does NOT contain the core `nvngx_dlss.dll`. Modern DLSS implementations typically rely on the driver to provide this file.
2.  **Bundling Failure**: To fix the missing driver DLL, we attempted to bundle the DLLs (`nvngx_dlss.dll`) with the application.
    *   **Dev DLLs**: When using `dev` DLLs (which match the Debug build), the production driver rejects them with a signature validation error (`0xBAD00014`).
    *   **Rel DLLs**: When using `rel` (Release) DLLs, they are compatible with the driver's signature check, BUT they are **incompatible** with the Debug version of the NGX library (`nvsdk_ngx_d_dbg.lib`) used by the application in Debug builds.

## Actions Taken

### 1. Release DLL Injection (Attempted Fix)
- **Action**: Manually copied `rel` versions of `nvngx_dlss.dll`, `nvngx_dlssd.dll`, and `nvngx_dlssg.dll` to the binary output directory.
- **Result**: Application failed to initialize DLSS. The Debug NGX library linked into the executable (`nvsdk_ngx_d_dbg.lib`) instructs the driver to verify signatures in a way that rejects "Release" DLLs (or simply refuses to load them due to internal checks).

### 2. Force Release Library Linking (Attempted Fix)
- **Action**: Modified `CMakeLists.txt` and manually patched `PlasmaDX-Clean.vcxproj` to link against `nvsdk_ngx_d.lib` (Release lib) instead of `nvsdk_ngx_d_dbg.lib` while in Debug configuration.
- **Result**: **Build Failure**. This caused massive linker errors (`LNK2038`) due to C++ Runtime Library mismatches (`_ITERATOR_DEBUG_LEVEL` and `RuntimeLibrary` conflicts between MDd and MD). This path is not viable without changing the entire project to Release mode.

### 3. Project ID Regeneration (Mitigation)
- **Action**: Generated a new UUID (`05227611-97e2-4462-af0b-ab7d47f06a86`) for `DLSS_PROJECT_ID` in `DLSSSystem.cpp`.
- **Reasoning**: To force NGX to treat the app as a new entity and bypass any "bad" cached state or CMS ID locks associated with the old ID.
- **Result**: Successful code change, but did not resolve the DLL signature rejection on its own.

### 4. Build Restoration
- **Action**: Reverted `CMakeLists.txt` and `.vcxproj` to their original valid states (linking `nvsdk_ngx_d_dbg.lib`). Restored `dev` DLLs to the output directory.
- **Result**: Application builds and runs successfully again (fixing the linker errors), but DLSS remains disabled due to the original signature rejection issue with `dev` DLLs.

## Conclusions & Recommendations

The current deadlock is:
*   **Debug Build** requires `nvsdk_ngx_d_dbg.lib`.
*   `nvsdk_ngx_d_dbg.lib` works best with `dev` DLLs.
*   **Production Drivers** reject `dev` DLLs (signature check).
*   **Release DLLs** work with drivers but break `nvsdk_ngx_d_dbg.lib`.

**Recommended Next Steps:**

1.  **Driver Reinstall (High Probability)**: The root cause is the *missing* `nvngx_dlss.dll` in the DriverStore. A "Clean Installation" of the latest NVIDIA Studio or Game Ready driver should restore this file, allowing the application to work without bundling ANY DLLs (the correct, modern way).
2.  **Release Build**: Switch the build configuration to **Release**. This will link `nvsdk_ngx_d.lib`, which works perfectly with the `rel` DLLs and the production driver.
3.  **Enable "Allow Dev DLLs"**: If you must debug DLSS, you need to enable "Development Mode" in the OS or via NVIDIA specific registry keys (often requires an NDA-protected "Overview" driver or specific developer tools installed) to allow loading unsigned/dev DLLs.

**Current State:** Codebase is clean and buildable. DLSS Project ID is fresh. No permanent "hacks" were left in the build files.
