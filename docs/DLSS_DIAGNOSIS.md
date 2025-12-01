# DLSS Super Resolution Diagnosis Report

**Date:** 2025-12-01
**Status:** BROKEN - Signature Validation Failure
**Impact:** Performance degradation without AI upscaling

---

## Root Cause Analysis

### What's Happening

1. **NGX SDK initializes successfully** ✅
   ```
   [21:40:59] DLSS: NGX SDK initialized successfully
   ```

2. **DLL signature validation FAILS** ❌
   ```
   nvLoadSignedLibraryW() failed on snippet './nvngx_dlss.dll' missing or corrupted
   Last error: One or more arguments are not correct (ERROR_INVALID_PARAMETER = 87)
   ```

3. **Feature marked unavailable** ❌
   ```
   DLSS: Super Resolution not supported on this GPU  (FALSE - RTX 4060 Ti supports DLSS!)
   ```

### Error Code Breakdown

| Error | Hex | Meaning |
|-------|-----|---------|
| -1160773628 | 0xBAD00004 | `NVSDK_NGX_Result_FAIL` - Generic failure |
| 87 (Windows) | N/A | `ERROR_INVALID_PARAMETER` - Invalid argument to LoadLibrary |

### Evidence

1. **DLLs are NOT corrupted** - MD5 checksums match between SDK and build directory
2. **SDK version**: 310.4.0 (DLSS 3.1.4)
3. **NGX API version**: 1.5.0 (0x0000015)
4. **Driver provides**: Only `nvngx_dlssg.dll` (Frame Gen), NOT `nvngx_dlss.dll` (Super Resolution)

---

## Diagnosis: SDK/Driver Version Mismatch

The bundled SDK DLLs (version 310.4.0 from October 2025) are being **rejected** by the driver's NGX loader. This can happen when:

1. **Driver is newer than SDK** - Driver expects newer DLL signature/format
2. **SDK is too old** - NVIDIA retired support for older DLL versions
3. **NGX security policy changed** - Driver won't load unsigned/older DLLs

### Timeline

```
Working:   [Unknown date] DLSS-SR functional
           ↓
Changed:   [Around November 15-28] Probe Grid + Volumetric ReSTIR implemented
           ↓
Broken:    [November 28+] DLSS-SR fails signature validation
```

---

## Solutions (Ranked by Likelihood of Success)

### Solution 1: Update DLSS SDK to Latest Version ⭐ RECOMMENDED

**Current SDK**: 310.4.0 (DLSS 3.1.4)
**Latest SDK**: ~310.6.0+ (DLSS 3.7+)

**Steps:**

1. Download latest DLSS SDK from [NVIDIA Developer Portal](https://developer.nvidia.com/rtx/dlss/get-started)
   - Or from GitHub: https://github.com/NVIDIA/DLSS

2. Replace SDK files:
   ```bash
   # Backup current SDK
   mv dlss dlss_old_310.4.0
   
   # Extract new SDK
   unzip DLSS_SDK_*.zip -d dlss
   ```

3. Rebuild project:
   ```bash
   MSBuild.exe build/PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64
   ```

4. Verify DLLs are copied to output:
   ```
   build/bin/Debug/nvngx_dlss.dll   (should be newer version)
   build/bin/Debug/nvngx_dlssd.dll
   build/bin/Debug/nvngx_dlssg.dll
   ```

---

### Solution 2: Clear NGX Cache

The NGX cache might have stale/conflicting data.

**Steps (PowerShell as Admin):**

```powershell
# Stop NVIDIA services
Stop-Service -Name "NVDisplay.ContainerLocalSystem" -Force -ErrorAction SilentlyContinue

# Clear NGX cache
Remove-Item -Recurse -Force "C:\ProgramData\NVIDIA\NGX\*" -ErrorAction SilentlyContinue

# Restart driver
Start-Service -Name "NVDisplay.ContainerLocalSystem"
```

Then restart the application.

---

### Solution 3: Update NVIDIA Driver

Even if you haven't manually updated, Windows Update might have:

1. Check current driver version:
   ```
   nvidia-smi --query-gpu=driver_version --format=csv,noheader
   ```

2. Download latest from [NVIDIA Drivers](https://www.nvidia.com/Download/index.aspx)

3. Clean install with "Perform clean installation" checked

---

### Solution 4: Migrate to Streamline SDK

NVIDIA Streamline SDK manages DLSS DLLs automatically, solving version conflicts.

**Pros:**
- Automatic DLL discovery and loading
- No bundling required
- Future-proof

**Cons:**
- Significant code changes required
- Different API than NGX direct

**Documentation:** https://github.com/NVIDIAGameWorks/Streamline

---

### Solution 5: Use Driver-Provided DLLs Only (Workaround)

Remove bundled DLLs and let NGX use driver-provided ones:

1. Remove from CMakeLists.txt:
   ```cmake
   # Comment out DLSS DLL copying
   # if(ENABLE_DLSS)
   #     add_custom_command(...)
   # endif()
   ```

2. Delete bundled DLLs from build:
   ```bash
   rm build/bin/Debug/nvngx_dlss*.dll
   ```

3. **Problem:** Driver doesn't provide `nvngx_dlss.dll` (only DLSS-G).
   This won't work unless NVIDIA adds driver fallback for DLSS-SR.

---

## Verification Steps

After applying a solution, verify:

1. **Check nvngx.log** - Should NOT show signature validation failures
2. **Check application log** - Should show:
   ```
   DLSS: NGX SDK initialized successfully
   DLSS: Super Resolution supported  (NOT "not supported")
   DLSS: Super Resolution feature created successfully
   ```
3. **Test F4 key** - Should toggle DLSS on/off
4. **Check ImGui** - DLSS checkbox should be enabled (not greyed out)

---

## Files to Update

| File | Change |
|------|--------|
| `dlss/` | Replace with latest SDK |
| `CMakeLists.txt` | Update if SDK structure changed |
| `src/dlss/DLSSSystem.cpp` | May need API updates for new SDK |

---

## Quick Test Command

After updating SDK:

```bash
# Build
MSBuild.exe build/PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64

# Run with DLSS enabled
./build/bin/Debug/PlasmaDX-Clean.exe

# Check logs for DLSS status
grep -i dlss build/bin/Debug/logs/PlasmaDX-Clean_*.log | tail -20
```

---

## Summary

| Issue | Root Cause | Fix |
|-------|------------|-----|
| Signature validation fails | SDK 310.4.0 too old for current driver | Update to latest DLSS SDK |
| SuperSampling_Available = 0 | DLL can't load → feature unavailable | Update SDK or driver |
| ERROR_INVALID_PARAMETER (87) | DLL format/signature mismatch | Update SDK |

**Primary Recommendation:** Download and install the latest DLSS SDK (3.7+). The current SDK (310.4.0) is from October 2025 and may be incompatible with recent driver updates.

---

## References

- [NVIDIA DLSS SDK Documentation](https://developer.nvidia.com/rtx/dlss/get-started)
- [NVIDIA NGX Programming Guide](https://docs.nvidia.com/ngx/)
- [DLSS GitHub Repository](https://github.com/NVIDIA/DLSS)
- [Streamline SDK](https://github.com/NVIDIAGameWorks/Streamline)

---

**Next Steps:**
1. Download latest DLSS SDK
2. Replace `dlss/` folder contents
3. Rebuild and test
4. If still failing, try clearing NGX cache
5. If still failing, update NVIDIA driver

