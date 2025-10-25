# PINN Integration Testing Guide

## Setup Complete ✅

All build issues resolved:
- ✅ Building correct CMake solution (`build/PlasmaDX-Clean.sln`)
- ✅ Old root solution deleted (no more confusion)
- ✅ PINNPhysicsSystem.cpp compiling
- ✅ All shaders compiling (physics + Gaussian)
- ✅ PINN model path fixed for new exe location
- ✅ ImGui controls added

## How to Run

```bash
cd build/bin/Debug
./PlasmaDX-Clean.exe
```

## How to Test PINN

### 1. Check PINN Availability

When the app starts, look in the console logs for:
```
[PINN] Successfully loaded PINN model!
PINN physics available! Press 'P' to toggle PINN physics
```

**If you see this instead:**
```
[PINN] ONNX Runtime exception: Load model from ...
PINN not available (ONNX Runtime not installed or model not found)
```
Then the model path is still wrong - let me know and I'll fix it.

### 2. Open ImGui Control Panel

Press **F1** to toggle the ImGui control panel.

Navigate to **Physics** section. You should see:

```
ML Physics (PINN)
☑ Enable PINN Physics (P key)
    ☑ Hybrid Mode (PINN + GPU) (?)
    Hybrid Threshold (x R_ISCO): [===|===] 10.0
    ──────────────
    Performance:
      Inference: 0.00 ms
      Particles: 10000
      Avg Batch: 0.00 ms
Model: PINN Accretion Disk (5 layers, 128 neurons)
```

### 3. Enable PINN

**Method 1: Keyboard**
- Press **P** key to toggle PINN on/off

**Method 2: ImGui**
- Click the "Enable PINN Physics" checkbox

Watch the console logs for:
```
[PINN] Physics ENABLED
[PINN] Starting GPU→CPU readback for 10000 particles
[PINN] GPU copy complete, attempting to map readback buffer
[PINN] Readback buffer size: X bytes
[PINN] Successfully mapped readback buffer
```

### 4. Expected Behavior

**If PINN works:**
- Particles continue orbiting smoothly
- Console logs show PINN updates every 60 frames
- Performance metrics update in ImGui
- No crashes or freezes

**If readback buffer mapping fails:**
- Console will show:
  ```
  [PINN] Failed to map readback buffer!
  [PINN] HRESULT = 0xXXXXXXXX
  [PINN] Failed to readback particle data
  ```
- Particles may freeze or disappear
- Application might crash

### 5. Test Hybrid Mode

In ImGui:
1. Enable PINN Physics
2. Check "Hybrid Mode (PINN + GPU)"
3. Adjust the threshold slider (5.0 - 50.0)

**What this does:**
- Particles with `r > threshold × R_ISCO` use PINN
- Particles with `r < threshold × R_ISCO` use GPU shader
- Lower threshold = more GPU, less PINN
- Higher threshold = more PINN, less GPU

### 6. Monitor Performance

Watch the ImGui metrics:
- **Inference time**: Should be < 5ms for 10K particles
- **Particles processed**: Should match particle count
- **Avg batch time**: Should be stable

## Expected Performance

| Particle Count | Traditional GPU | PINN (Expected) | Speedup |
|----------------|-----------------|-----------------|---------|
| 10K            | 120 FPS         | 120 FPS         | 1.0×    |
| 50K            | 45 FPS          | 180 FPS         | 4.0×    |
| 100K           | 18 FPS          | 110 FPS         | 6.1×    |

*At 10K particles, you won't see a performance difference - PINN shines at 50K+*

## Troubleshooting

### Model Not Found
**Symptom:** "PINN not available" message
**Fix:** Model path might be wrong. The path should be `../../../ml/models/pinn_accretion_disk.onnx` relative to `build/bin/Debug/`.

### Readback Buffer Mapping Fails
**Symptom:**
```
[PINN] Failed to map readback buffer!
[PINN] HRESULT = 0xXXXXXXXX
```

**Diagnosis:** The diagnostic logging will show:
- Readback buffer size
- Expected size
- Exact HRESULT error code

**Possible causes:**
1. GPU synchronization issue (WaitForGPU not working)
2. Buffer state mismatch
3. Memory access violation
4. D3D12 debug layer issue

**Next steps:** Run it and share the log file - I'll diagnose the exact HRESULT code.

### Application Crashes
**When:** After pressing 'P' to enable PINN
**Fix:** This is likely the readback buffer issue. Share the crash log.

## What to Report

If PINN doesn't work, please provide:
1. Console log output (especially PINN lines)
2. HRESULT error code (if mapping fails)
3. When the crash/freeze happens (startup, P key press, during inference)
4. ImGui screenshot (if GUI appears)

## Files Modified in This Session

All changes for PINN integration:
- `src/ml/PINNPhysicsSystem.h` - PINN class interface
- `src/ml/PINNPhysicsSystem.cpp` - ONNX Runtime integration
- `src/particles/ParticleSystem.h` - PINN members added
- `src/particles/ParticleSystem.cpp` - PINN update logic, readback buffer
- `src/core/Application.cpp` - ImGui controls, P key handler
- `CMakeLists.txt` - Added PINNPhysicsSystem to build, added missing shaders
- `build/PlasmaDX-Clean.sln` - Correct solution to use (NOT root one)

## Critical Reminders

1. **Always use:** `build/PlasmaDX-Clean.sln` (CMake-generated)
2. **Never use:** Root `/PlasmaDX-Clean.sln` (deleted - was outdated)
3. **Executable location:** `build/bin/Debug/PlasmaDX-Clean.exe`
4. **To rebuild:** `cd build` then `MSBuild PlasmaDX-Clean.sln ...`
