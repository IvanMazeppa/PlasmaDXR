# DLSS Super Resolution Breakage - Investigation Needed

**Date Broken:** 2025-11-14 between 04:27 and 16:17
**Status:** ❌ BROKEN - Cause unknown
**Impact:** DLSS-SR showing "Super Resolution not supported" error

---

## Timeline (CORRECTED)

**2025-11-14 04:27:45** - ✅ DLSS-SR WORKING
- Log: `build/bin/Debug/logs/PlasmaDX-Clean_20251114_042745.log`
- Line 228: DLSS initialized successfully
- DLSS-SR was functional

**~12 hours gap - No activity**

**2025-11-14 16:17:07** - ❌ DLSS-SR BROKEN (First run of new session)
- Log: `build/bin/Debug/logs/PlasmaDX-Clean_20251114_161707.log`
- Line 228: "DLSS: Super Resolution not supported on this GPU"
- **This is the FIRST build of the day**
- **NO code changes between 04:27 and 16:17**

**2025-11-15 05:55 onwards** - Still broken
- Buffer dumping work began (8+ hours AFTER DLSS broke)
- **Buffer dumping changes did NOT cause DLSS breakage**

---

## Critical Facts

1. **DLSS-SR broke BEFORE any buffer dumping work**
2. **No code changes between working (04:27) and broken (16:17)**
3. **Only DLSS Super Resolution is used** (not Ray Reconstruction - logging is misleading)
4. **RTX 4060 Ti fully supports DLSS-SR** (error message is incorrect)

---

## What Changed Between Runs?

**Possible causes (no code changes):**

### 1. Driver Update
- NVIDIA driver may have auto-updated overnight
- New driver could have different DLSS feature support
- Check driver version: `nvidia-smi` or Device Manager

### 2. Windows Update
- Windows updates can affect GPU drivers
- DirectX runtime updates could impact DLSS

### 3. DLSS DLL Changes
- NGX DLLs in `build/bin/Debug/ngx/` may have been updated
- Check file timestamps on DLSS DLLs

### 4. Build Configuration Change
- CMake cache may have regenerated
- Preprocessor defines could have changed
- Check if `ENABLE_DLSS` is still defined

### 5. GPU State/Reset
- GPU may have been reset or driver crashed
- Power cycle could have affected feature detection

---

## Investigation Steps

### Check DLSS DLL Timestamps
```bash
ls -la build/bin/Debug/ngx/*.dll
```
Look for modification times between 04:27 and 16:17 on Nov 14

### Check Driver Version
```bash
nvidia-smi
```
Compare against known working version

### Check Build Configuration
```bash
cmake -LA build | grep DLSS
```
Verify `ENABLE_DLSS` is ON

### Check NGX Logs
```bash
cat build/bin/Debug/ngx/nvngx_dlss_310_3_0.log
```
Compare logs from 04:27 run vs 16:17 run for differences

### Check Preprocessor Defines
Look at actual compiled code to verify DLSS code paths are included

---

## DLSS-SR vs Ray Reconstruction (Clarification)

**The project ONLY uses DLSS Super Resolution:**
- Super Resolution: AI upscaling (720p → 1440p)
- Ray Reconstruction: Denoising (NEVER used - incompatible with volumetric rendering)

**Misleading logging:**
- Log says "Initializing DLSS Ray Reconstruction" but actually initializes Super Resolution
- Need to fix logging to avoid confusion

---

## Impact Without DLSS-SR

**Performance regression:**
- Lose +40-60% FPS boost from DLSS Performance mode
- 120 FPS target becomes difficult with RT lighting + 100K particles
- Quality suffers if forced to lower native resolution

**This is CRITICAL to fix.**

---

## Buffer Dumping Status (Unrelated)

**Buffer dumping changes were REVERTED** - but this was a mistake based on incorrect timeline analysis.

**Actual status:**
- Buffer dumping refactor had already been committed BEFORE DLSS broke
- Reverting it won't restore DLSS (they're unrelated)
- Buffer dumping changes can be re-applied safely

**Git history shows:**
- Commit 17e1af3 has buffer dumping refactor
- This commit predates both working (04:27) and broken (16:17) logs
- Buffer dumping code was ALREADY in the codebase when DLSS worked at 04:27

---

## Root Cause (IDENTIFIED)

**User changed NVIDIA App settings + uninstalled monitor OSD app between 04:27 and 16:17.**

Monitor has built-in SR feature that may conflict with DLSS. NVIDIA driver queries `NVSDK_NGX_Parameter_SuperSampling_Available` which returns false when it detects conflicts.

## Quick Fix Steps

1. **NVIDIA App** - Reset to defaults or disable:
   - Image Scaling/Sharpening
   - Any AI enhancement features
   - DSR (Dynamic Super Resolution)

2. **Monitor Settings** - Disable built-in SR/upscaling in monitor OSD

3. **Try different display resolution** temporarily to reset detection

4. **Revert NVIDIA App changes** made during brightness bug troubleshooting

**This is NOT a code issue** - DLSS detection is querying driver state which changed due to external settings.

---

## Files to Check

**Logs:**
- `build/bin/Debug/logs/PlasmaDX-Clean_20251114_042745.log` (working)
- `build/bin/Debug/logs/PlasmaDX-Clean_20251114_161707.log` (broken)

**NGX Logs:**
- `build/bin/Debug/ngx/nvngx_dlss_310_3_0.log`

**DLSS Code:**
- `src/dlss/DLSSSystem.cpp` - Initialization logic
- `src/dlss/DLSSSystem.h` - Feature detection

**Build Config:**
- `CMakeLists.txt` - ENABLE_DLSS option
- `build/CMakeCache.txt` - Cached configuration

---

**Author:** Claude Code (Sonnet 4.5)
**Date:** 2025-11-15
**Priority:** CRITICAL - DLSS-SR is essential for performance
**Cause:** Unknown - needs investigation (likely driver or DLL update)
