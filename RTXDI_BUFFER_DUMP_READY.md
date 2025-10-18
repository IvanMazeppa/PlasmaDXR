# RTXDI Buffer Validation - Ready for Testing

**Status**: Debug logging added, ready to identify root cause
**Date**: 2025-10-18 21:45
**Time Required**: 10 minutes

---

## Changes Made

### Added Comprehensive Debug Logging

**File**: `src/lighting/RTXDILightingSystem.cpp`

**Validation Points Added**:

1. **Source Data Validation** (line 257-260)
   - Logs first light from source `m_lights` array
   - Verifies Application.cpp is passing non-zero data

2. **Upload Heap Validation** (line 263-265)
   - Logs first light after memcpy to upload heap
   - Verifies upload heap allocation is writable

3. **Upload Heap Details** (line 267-269)
   - Logs upload heap resource pointer
   - Logs offset into upload heap
   - Logs CPU address (memory mapping)

4. **Compute Constants Validation** (line 336-339)
   - Logs grid dimensions (30×30×30)
   - Logs light count (should be 13)
   - Logs cell size (20.0)

5. **GPU Address Validation** (line 355-357)
   - Logs light buffer GPU virtual address
   - Logs light grid GPU virtual address
   - Verifies buffers are created and accessible

---

## How to Test

### Step 1: Build (2 min)

Open **x64 Native Tools Command Prompt for VS 2022**:

```cmd
cd D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean
MSBuild PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64
```

**Expected**: Successful build (RTXDILightingSystem.cpp recompiled)

### Step 2: Run with RTXDI (3 min)

```cmd
cd D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean
build\Debug\PlasmaDX-Clean.exe --rtxdi
```

**Let run for 10 frames**, then close.

### Step 3: Check Log for Validation Output (1 min)

Open latest log file in `logs/` directory.

**Search for `[VALIDATION]` entries**:

```
Ctrl+F → "[VALIDATION]"
```

**Expected Output** (if working correctly):
```
[INFO]   [VALIDATION] Source light 0: pos=(150.00,0.00,0.00), intensity=8.00
[INFO]   [VALIDATION] Uploaded light 0: pos=(150.00,0.00,0.00), intensity=8.00
[INFO]   [VALIDATION] Upload heap: resource=0x000002653F1A0000, offset=0, cpuAddr=0x000002653F1A0000
[INFO]   [VALIDATION] Compute constants: gridCells=(30,30,30), lightCount=13, cellSize=20.0
[INFO]   [VALIDATION] Light buffer GPU address: 0x00000265AB3C0000
[INFO]   [VALIDATION] Light grid GPU address: 0x00000265AB800000
```

### Step 4: Dump Buffers (2 min)

Run again and press **Ctrl+D** after 10 frames:

```cmd
build\Debug\PlasmaDX-Clean.exe --rtxdi
```

Wait 10 frames → Press **Ctrl+D** → Close app

**Check for buffer files**:
```
PIX/buffer_dumps/frame_XXX/g_lights.bin
PIX/buffer_dumps/frame_XXX/g_lightGrid.bin
```

### Step 5: Validate Buffers (2 min)

Open **WSL terminal** (Ubuntu on Windows):

```bash
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean
python PIX/scripts/validate_rtxdi_buffers.py PIX/buffer_dumps/frame_XXX
```

Replace `frame_XXX` with actual frame number from dump directory.

---

## Expected Diagnostic Outcomes

### Scenario A: Source Data is Zeros

**Log shows**:
```
[INFO]   [VALIDATION] Source light 0: pos=(0.00,0.00,0.00), intensity=0.00
```

**Root Cause**: Multi-light system not populating `m_lights` vector correctly

**Fix Location**: `src/core/Application.cpp:2149` (InitializeLights function)

### Scenario B: Upload Data is Zeros (Source is Valid)

**Log shows**:
```
[INFO]   [VALIDATION] Source light 0: pos=(150.00,0.00,0.00), intensity=8.00
[INFO]   [VALIDATION] Uploaded light 0: pos=(0.00,0.00,0.00), intensity=0.00
```

**Root Cause**: Upload heap allocation failed or memory not writable

**Fix**: Replace ResourceManager upload heap with explicit upload buffer creation

### Scenario C: Upload Works, GPU Addresses are Null

**Log shows**:
```
[INFO]   [VALIDATION] Light buffer GPU address: 0x0000000000000000
```

**Root Cause**: Buffers not created correctly during initialization

### Scenario D: Everything Logs Correctly, Buffer Dump Still Zeros

**Root Cause**: Command list synchronization issue (copy not flushed before dump)

**Fix**: Add GPU fence/wait after UpdateLightGrid()

### Scenario E: Success!

**Buffer validation shows**:
```
=== Summary: 13 valid lights out of 16 ===
✅ SUCCESS: All 13 lights present

✅ SUCCESS: Light grid populated with data
✅ BOTH BUFFERS VALID - RTXDI ready for Milestone 2.4
```

---

## Quick Command Summary

```cmd
REM Build
cd D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean
MSBuild PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64

REM Run with RTXDI
build\Debug\PlasmaDX-Clean.exe --rtxdi

REM (Wait 10 frames, press Ctrl+D, close app)

REM Switch to WSL
wsl

REM Validate buffers
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean
python PIX/scripts/validate_rtxdi_buffers.py PIX/buffer_dumps/frame_XXX
```

---

**Begin testing now. Report validation log output to determine next steps.**
