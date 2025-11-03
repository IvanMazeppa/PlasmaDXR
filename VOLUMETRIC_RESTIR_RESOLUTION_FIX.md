# Volumetric ReSTIR Resolution Fix + Logger Format Specifier Fix

**Date:** 2025-11-03 20:05
**Build:** Debug (branch 0.12.10)

---

## Issues Fixed

### 1. Logger Format Specifier Support (CRITICAL)

**Problem:** Custom Logger only handled `{}` placeholders, not format specifiers like `{:08X}`, `{:016X}`, `{:.2f}`. This caused 20+ log statements per run to show "0x{:08X}" instead of actual hex values, losing critical debugging information.

**Root Cause:** `Logger::LogFormat()` used `ReplaceFirst(msg, "{}", ToString(value))` which only replaced empty braces. The `ToString()` method called `std::to_string()` which outputs decimal only.

**Fix:** Enhanced Logger.h with format specifier parsing:

**Files Modified:**
- `src/utils/Logger.h` (lines 32-132)

**Supported Format Specifiers:**
- `{:08X}` - 8-digit uppercase hex with zero padding (e.g., HRESULT codes)
- `{:016X}` - 16-digit uppercase hex (e.g., GPU handles, pointers)
- `{:04X}` - 4-digit uppercase hex (e.g., vendor IDs)
- `{:.2f}` - 2 decimal places for floats
- `{}` - Default decimal (unchanged from before)

**Expected Result:** All 20+ format errors per log should now display correctly:
```
Before: [INFO] Vendor ID: 0x{:04X}
After:  [INFO] Vendor ID: 0x10DE
```

---

### 2. Shader Staleness Issue (CRITICAL)

**Problem:** Volume resolution changed from 64³ to 32³ in C++ (VolumetricReSTIRSystem.cpp), but shader still produced 175,204 voxel writes (indicating 64³ volume).

**Root Cause:** CMake incremental builds only recompile files that changed. When only C++ changes but shader constants change, the `.dxil` doesn't get regenerated.

**Evidence:**
- Shader compiled: 2025-11-03 01:09 AM
- Test executed: 2025-11-03 19:55 PM (7:55 PM)
- 18+ hour gap with no recompilation

**Fix:** Forced shader recompilation:
```bash
rm -f build/bin/Debug/shaders/volumetric_restir/populate_volume_mip2.dxil
MSBuild build/CompileShaders.vcxproj /t:Rebuild
```

**Shader timestamp after fix:**
```
-rwxrwxrwx 1 maz3ppa maz3ppa 6.1K Nov  3 20:01 populate_volume_mip2.dxil
```

**Expected Result:** Next test should show ~43,000 voxel writes (75% reduction from 175k):
- 64³ volume = 262,144 voxels → 175,204 writes (67% fill rate)
- 32³ volume = 32,768 voxels → **~43,000 writes expected** (67% fill rate)

---

### 3. Volume Resolution Reduction (PERFORMANCE FIX)

**Problem:** GPU TDR (timeout) at ≥2045 particles with 64³ volume. 3-second hang, then crash.

**Solution:** Reduced volume resolution from 64³ to 32³ (8× fewer voxels).

**Files Modified:**
- `src/lighting/VolumetricReSTIRSystem.cpp` (lines 172, 896)
- `src/lighting/VolumetricReSTIRSystem.h` (line 209 comment - needs update)

**Performance Impact:**
- 8× fewer voxels (262,144 → 32,768)
- 75% reduction in voxel writes (175k → ~43k expected)
- Reduced memory bandwidth (64 KB → 8 KB volume texture)
- Lower atomic contention (InterlockedAdd operations)

**Expected Result:** GPU hang at ≥2045 particles may be resolved. If 32³ is still too expensive, we'll switch to the hybrid probe grid approach.

---

## Testing Instructions

### What to Test

1. **Logger Format Specifiers** - Check for hex values in logs:
   ```bash
   tail -n 50 build/bin/Debug/logs/PlasmaDX-Clean_*.log | grep "0x"
   ```
   - Should see `0x10DE` (NVIDIA vendor ID), not `0x{:04X}`
   - Should see proper hex GPU handles, not `0x{:016X}`

2. **Voxel Write Reduction** - Check diagnostic counter[2]:
   ```bash
   grep "Total voxel writes" build/bin/Debug/logs/PlasmaDX-Clean_*.log
   ```
   - **Expected:** ~43,000 writes (frames 1-4)
   - **Before:** 175,204 writes (64³ volume)

3. **GPU Hang Test** - Gradually increase particle count:
   - Start: 2000 particles
   - Increment: +10 particles
   - Watch for: 2045+ particle threshold
   - **Expected:** Should work beyond 2045 with 32³ volume

### Expected Diagnostic Output

```
[TIME] [INFO] Frame 1: Diagnostic counters:
[TIME] [INFO]   [0] Total threads executed: 2076
[TIME] [INFO]   [1] Early returns (out of bounds): 0
[TIME] [INFO]   [2] Total voxel writes: 43758     // ✅ REDUCED from 175k
[TIME] [INFO]   [3] Max voxels written per particle: 512
```

### If Issues Persist

If GPU hang still occurs at ≥2045 particles with 32³ volume:

**Option A:** Further reduce resolution to 16³ (4,096 voxels)
- Edit `VolumetricReSTIRSystem.cpp:172` → `const uint32_t volumeSize = 16;`
- Delete shader and rebuild

**Option B (RECOMMENDED):** Switch to Hybrid Probe Grid approach
- See `PROBE_GRID_IMPLEMENTATION_OUTLINE.md` for 7-day implementation plan
- Expected: 90+ FPS at 10K particles with smooth volumetric scattering

---

## Build Information

**Built:** 2025-11-03 20:05
**Configuration:** Debug
**Shader timestamp:** 2025-11-03 20:01 (fresh compilation)
**Branch:** 0.12.10

---

## Files Changed This Session

1. **`src/utils/Logger.h`** - Added format specifier support
2. **`src/lighting/VolumetricReSTIRSystem.cpp`** - Volume resolution 64³→32³
3. **`shaders/volumetric_restir/populate_volume_mip2.dxil`** - Forced recompilation

---

## Next Steps

1. **Test this build** - Check voxel write reduction and particle count threshold
2. **If GPU hang persists** - Try 16³ volume or switch to probe grid
3. **Verify all logging** - Confirm hex values display correctly

**User's preference:** If last ReSTIR fix doesn't work, implement probe grid approach (see `PROBE_GRID_IMPLEMENTATION_OUTLINE.md`).
