# DLSS Integration Quick Reference (v0.11.1)

**Branch:** v0.11.1
**Status:** Phase 2.5 Complete ✅ - Feature Creation Fixed

---

## What's Working Now

✅ **NGX SDK Initialization:** Successfully initializes on RTX 4060 Ti
✅ **Ray Reconstruction Support:** Detected and available
✅ **DLSS Denoiser Loaded:** v310.3.0 from driver
✅ **Logging Enabled:** Writes to `build/bin/Debug/ngx/`
✅ **Feature Creation:** Lazy creation with valid command list (no crash!)

---

## Critical Discoveries (Don't Forget!)

### 1. ProjectID Format
**MUST** be UUID format (hexadecimal only):
```cpp
#define DLSS_PROJECT_ID "a0b1c2d3-4e5f-6a7b-8c9d-0e1f2a3b4c5d"
```
❌ `"PlasmaDX-Clean"` → rejected
❌ `"plasmadx-clean"` → rejected
✅ UUID format → works

### 2. Both DLLs Required
- `nvngx_dlss.dll` (30MB) - Main SDK ← **REQUIRED FOR INIT**
- `nvngx_dlssd.dll` (39MB) - Ray Reconstruction denoiser

### 3. FeatureCommonInfo Not Optional
```cpp
NVSDK_NGX_FeatureCommonInfo featureInfo = {};
featureInfo.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_ON;
// Pass &featureInfo, NOT nullptr!
```

### 4. CreateFeature Needs Command List
**CANNOT** pass `nullptr` for command list:
```cpp
// WRONG - causes crash:
NVSDK_NGX_D3D12_CreateFeature(nullptr, ...);

// RIGHT - pass valid command list from rendering context:
NVSDK_NGX_D3D12_CreateFeature(cmdList, ...);
```

---

## Phase 2.5 Complete ✅

**Goal:** Fix feature creation crash - **DONE!**

**What was done:**
1. ✅ Removed `CreateRayReconstructionFeature()` call from `Application.cpp` initialization
2. ✅ Added DLSS system reference to `ParticleRenderer_Gaussian`
3. ✅ Implemented lazy feature creation in `Render()` function (first frame)
4. ✅ Pass valid `ID3D12GraphicsCommandList*` from rendering context

**Result:** Application no longer crashes during DLSS initialization!

---

## File Locations

**Core DLSS Code:**
- `src/dlss/DLSSSystem.h` - Interface
- `src/dlss/DLSSSystem.cpp` - Implementation (ProjectID line 11)

**Integration:**
- `src/core/Application.h` - DLSSSystem member
- `src/core/Application.cpp` - Initialization (lines 262-286)

**Build:**
- `CMakeLists.txt` - SDK detection (line 24), DLL copying (line 308)

**Logs:**
- `build/bin/Debug/ngx/*.log` - NGX diagnostic logs
- `build/bin/Debug/logs/*.log` - Application logs

**Docs:**
- `DLSS_PHASE2_SUMMARY.md` - Complete implementation details
- `DLSS_INTEGRATION_STATUS.md` - Current status and roadmap

---

## Useful Commands

**Build:**
```bash
"/mnt/c/Program Files/Microsoft Visual Studio/2022/Community/MSBuild/Current/Bin/MSBuild.exe" \
    build/PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64 /t:Build /nologo /v:minimal
```

**Check Logs:**
```bash
tail -50 build/bin/Debug/logs/PlasmaDX-Clean_*.log | grep DLSS
cat build/bin/Debug/ngx/nvngx.log
```

**Verify DLLs:**
```bash
ls -lh build/bin/Debug/nvngx*.dll
```

---

## Success Indicators

**Init Success:**
```
[INFO] DLSS: NGX SDK initialized successfully
[INFO] DLSS: Ray Reconstruction supported
```

**Feature Creation Success (not yet working):**
```
[INFO] DLSS Ray Reconstruction feature created (1920x1080)
[INFO] DLSS Ray Reconstruction ready
```

---

## Known Issues

### CreateFeature Crash (Current)
**Symptom:** App initializes DLSS successfully, then crashes
**Cause:** Passing `nullptr` for command list
**Fix:** Defer creation to rendering initialization
**Status:** Next task (Phase 2.5)

---

## Performance Targets (Phase 4)

| Configuration | Target FPS | Expected Quality |
|---------------|------------|------------------|
| 8-ray traditional | ~120 FPS | Baseline |
| 1-ray + DLSS | 240-360 FPS | Good |
| 2-ray + DLSS | 180-240 FPS | Excellent (≥ 8-ray) |

---

**Last Updated:** 2025-10-29
**Next Session:** Start with Phase 2.5 - Feature Creation Fix
