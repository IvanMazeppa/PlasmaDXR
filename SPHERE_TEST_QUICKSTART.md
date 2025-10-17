# Sphere Boundary Test - Quick Start Guide

**Test Suite:** Isolate the ~300 unit multi-light boundary issue
**Time Required:** 5 minutes to run, 15-30 minutes to analyze
**Status:** Ready to execute

---

## TL;DR - Run This Now

```bash
# 1. Build DebugPIX configuration
MSBuild PlasmaDX-Clean.sln /p:Configuration=DebugPIX /p:Platform=x64

# 2. Run automated test suite
./test_sphere_boundary.sh

# 3. Review results
explorer test_results
```

---

## What This Tests

Tests a single light at 7 different distances:

| Distance | Color | Expected |
|----------|-------|----------|
| 50u | RED | Works |
| 100u | ORANGE | Works |
| 200u | YELLOW | Works |
| **300u** | **GREEN** | **BOUNDARY** |
| 400u | CYAN | Fails |
| 500u | BLUE | Fails |
| 1000u | MAGENTA | Fails |

**Goal:** Find the exact cutoff point and identify root cause.

---

## Files Created

### Test Configs (7 files)
```
configs/scenarios/
├── sphere_test_050u.json   (RED light at 50 units)
├── sphere_test_100u.json   (ORANGE light at 100 units)
├── sphere_test_200u.json   (YELLOW light at 200 units)
├── sphere_test_300u.json   (GREEN light at 300 units - CRITICAL)
├── sphere_test_400u.json   (CYAN light at 400 units)
├── sphere_test_500u.json   (BLUE light at 500 units)
└── sphere_test_1000u.json  (MAGENTA light at 1000 units)
```

### Test Execution
- `test_sphere_boundary.sh` - Automated test runner
- `SPHERE_BOUNDARY_TEST_PLAN.md` - Detailed test plan (17KB)
- `SPHERE_TEST_QUICKSTART.md` - This file

---

## Test Execution Steps

### Step 1: Build DebugPIX

```bash
MSBuild PlasmaDX-Clean.sln /p:Configuration=DebugPIX /p:Platform=x64
```

**Expected output:**
```
Build succeeded.
    0 Warning(s)
    0 Error(s)
```

### Step 2: Run Test Suite

```bash
./test_sphere_boundary.sh
```

**What happens:**
- Runs 7 tests (each ~10 seconds)
- Captures PIX GPU traces at frame 120
- Dumps GPU buffers to disk
- Generates summary report

**Expected duration:** ~90 seconds

### Step 3: Review Results

Results are saved to timestamped directory:
```
test_results/sphere_boundary_YYYYMMDD_HHMMSS/
├── captures/           # PIX GPU captures
├── buffer_dumps/       # Raw GPU buffer data
├── logs/               # Application logs
└── test_summary.txt    # Overall results
```

---

## Analysis Checklist

### Quick Visual Check

Open PIX captures and look for:

1. **50-100 unit tests (RED/ORANGE):**
   - [ ] Particles are brightly lit
   - [ ] Light color visible on particles
   - [ ] Shadow rays working (realistic occlusion)

2. **200 unit test (YELLOW):**
   - [ ] Particles still lit (may be dimmer)
   - [ ] Light color still visible

3. **300 unit test (GREEN) - CRITICAL:**
   - [ ] Are particles lit at all?
   - [ ] Is lighting weaker than 200u?
   - [ ] Hard cutoff or gradual falloff?

4. **400-1000 unit tests (CYAN/BLUE/MAGENTA):**
   - [ ] Are particles completely dark?
   - [ ] No light contribution at all?

### Log Analysis

Check `test_summary.txt`:

```
TEST RESULTS:
================================================================

  50 units - RED     light - COMPLETED
 100 units - ORANGE  light - COMPLETED
 200 units - YELLOW  light - COMPLETED
 300 units - GREEN   light - COMPLETED  <-- KEY TEST
 400 units - CYAN    light - COMPLETED
 500 units - BLUE    light - COMPLETED
1000 units - MAGENTA light - COMPLETED
```

All should show "COMPLETED" (not TIMEOUT or CRASHED).

---

## Interpreting Results

### Scenario A: Hard Cutoff at 300u

**Visual:**
- 50-200u: Bright lighting ✅
- 300u: Partial lighting or complete darkness ⚠️
- 400-1000u: Complete darkness ❌

**Root Cause:** Shader code or config using `physics.outerRadius = 300.0` as hard limit

**Next Steps:**
```bash
# Search for hardcoded 300.0 or outerRadius checks
grep -r "300.0" shaders/particles/particle_gaussian_raytrace.hlsl
grep -r "outerRadius" shaders/
grep -r "if.*distance.*>" shaders/particles/particle_gaussian_raytrace.hlsl
```

### Scenario B: Gradual Falloff

**Visual:**
- 50u: Very bright ✅
- 100u: Bright ✅
- 200u: Dimmer ⚠️
- 300u: Very dim ⚠️
- 400-1000u: Nearly black ❌

**Root Cause:** Attenuation formula too aggressive

**Next Steps:**
Check attenuation calculation in shader:
```hlsl
// At 300 units:
float attenuation = 1.0 / (lightDist * lightDist);
// = 1.0 / 90000 = 0.000011 (effectively zero!)
```

### Scenario C: Acceleration Structure Bounds

**Visual:**
- 50-300u: Normal lighting ✅
- 400-1000u: No raytracing at all (PIX shows no traversal) ❌

**Root Cause:** TLAS/BLAS built with 300 unit bounds

**Next Steps:**
Check `RTLightingSystem_RayQuery.cpp`:
```cpp
// Bug suspect:
D3D12_RAYTRACING_AABB aabb;
aabb.MinX = -outerRadius;  // Should be scene bounds!
aabb.MaxX = +outerRadius;
```

### Scenario D: All Tests Pass

**Visual:** All 7 distances show correct lighting ✅

**Possible reasons:**
1. Bug was already fixed
2. Bug requires specific conditions (multiple lights, ReSTIR enabled, etc.)
3. Test conditions don't reproduce the issue

**Next Steps:**
- Test with 13 lights simultaneously
- Test with ReSTIR enabled
- Test with physics enabled

---

## Manual Testing (If Script Fails)

### Quick Manual Test

```bash
# Build
MSBuild PlasmaDX-Clean.sln /p:Configuration=DebugPIX /p:Platform=x64

# Run critical test (300 units)
./build/DebugPIX/PlasmaDX-Clean-PIX.exe --config=configs/scenarios/sphere_test_300u.json

# Observe:
# - Are particles lit by GREEN light?
# - Can you see green color on particles?
# - Are shadow rays working?
# - Any hard cutoff?
```

### Compare Two Distances

```bash
# Test 100 units (should work)
./build/DebugPIX/PlasmaDX-Clean-PIX.exe --config=configs/scenarios/sphere_test_100u.json
# -> Take screenshot, note brightness

# Test 400 units (should fail)
./build/DebugPIX/PlasmaDX-Clean-PIX.exe --config=configs/scenarios/sphere_test_400u.json
# -> Take screenshot, compare

# Difference obvious? -> Confirms boundary issue
```

---

## PIX Analysis (Deep Dive)

### Open Critical Test Capture

```bash
# Open PIX
explorer PIX/Captures

# Find: sphere_300u_*.wpix
# Double-click to open in PIX for Windows
```

### Shader Debugging Steps

1. **Navigate to frame 120** (capture frame)

2. **Find compute dispatch:** `particle_gaussian_raytrace`

3. **Inspect shader constants:**
   - lightCount = 1
   - light[0].position = (300, 0, 0)
   - light[0].intensity = 10.0
   - light[0].radius = 300.0
   - light[0].color = (0, 1, 0) = GREEN

4. **Set breakpoint at line 715** (light loop):
   ```hlsl
   for (uint lightIdx = 0; lightIdx < lightCount; ++lightIdx)
   ```

5. **Step through light calculation:**
   - Watch `lightDir` (direction to light)
   - Watch `lightDist` (distance to light)
   - Watch `attenuation` (should be non-zero!)
   - Watch `totalLight` (accumulated lighting)

6. **Look for culling:**
   - Does shader skip light due to distance check?
   - Does shader return 0 attenuation?
   - Does shader cull particle?

---

## Expected Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| Setup | 5 min | Build DebugPIX, verify configs |
| Execution | 2 min | Run automated test suite |
| Quick Analysis | 5 min | Review summary, open PIX captures |
| Deep Analysis | 15-30 min | PIX shader debugging, buffer dumps |
| Root Cause ID | Variable | Code search, hypothesis testing |

**Total:** 30-60 minutes from zero to root cause identified

---

## Success Criteria

This test is successful if you can answer:

1. ✅ **What is the exact boundary distance?**
   - [ ] 300 units (exactly at `physics.outerRadius`)
   - [ ] Other: ______

2. ✅ **Is it a hard cutoff or gradual falloff?**
   - [ ] Hard cutoff (works at 300u, fails at 301u)
   - [ ] Gradual falloff (works at 100u, dims by 300u)

3. ✅ **Where is the root cause?**
   - [ ] Shader distance check
   - [ ] Attenuation formula
   - [ ] Acceleration structure bounds
   - [ ] Configuration constant
   - [ ] Other: ______

4. ✅ **What is the fix?**
   - Specific line of code to change
   - Specific constant to adjust
   - Specific logic to remove

---

## Common Issues

### Issue: Script doesn't run

**Symptom:** `./test_sphere_boundary.sh: Permission denied`

**Fix:**
```bash
chmod +x test_sphere_boundary.sh
```

### Issue: Build fails

**Symptom:** MSBuild errors

**Fix:**
```bash
# Clean and rebuild
MSBuild PlasmaDX-Clean.sln /t:Clean /p:Configuration=DebugPIX
MSBuild PlasmaDX-Clean.sln /p:Configuration=DebugPIX /p:Platform=x64
```

### Issue: No PIX captures generated

**Symptom:** `WARNING: No PIX capture found`

**Fix:**
- Ensure PIX for Windows is installed
- Check `debug.enablePIX = true` in config
- Check `debug.pixAutoCapture = true` in config
- Manually capture with F12 key during test

### Issue: Application crashes

**Symptom:** Exit code 1, "CRASHED" in summary

**Fix:**
- Check logs in `test_results/.../logs/`
- Look for D3D12 errors
- Verify shaders compiled successfully
- Try Debug build first (better error messages)

### Issue: All tests show darkness

**Symptom:** All 7 tests fail (even 50u)

**Fix:**
- Verify light system is working at all
- Check ImGui shows "Active Lights: 1 / 16"
- Verify light buffer is non-null in PIX
- Test with default config to confirm lights work in general

---

## Quick Command Reference

```bash
# Build
MSBuild PlasmaDX-Clean.sln /p:Configuration=DebugPIX /p:Platform=x64

# Run full suite
./test_sphere_boundary.sh

# Run single test
./build/DebugPIX/PlasmaDX-Clean-PIX.exe --config=configs/scenarios/sphere_test_300u.json

# View results
explorer test_results

# Open PIX
explorer PIX/Captures

# Search suspect code
grep -r "300.0" shaders/
grep -r "outerRadius" shaders/
grep -r "attenuation" shaders/particles/particle_gaussian_raytrace.hlsl
```

---

## Next Steps After Results

1. **Document findings** in test summary
2. **Identify root cause** based on observations
3. **Search codebase** for suspect code sections
4. **Implement fix** (separate task)
5. **Re-run test suite** to validate fix
6. **Performance regression test** (FPS check)

---

## Support

**Detailed Documentation:** See `SPHERE_BOUNDARY_TEST_PLAN.md` (17KB)

**Key Sections:**
- Test Matrix (page 1)
- Hypothesis (page 2)
- Expected Outcomes (page 4)
- Debugging Workflow (page 5)
- Root Cause Suspects (page 6)

**Questions?**
- Check logs in `test_results/.../logs/`
- Review PIX captures
- Search CLAUDE.md for related documentation

---

**Last Updated:** 2025-10-17
**Status:** READY TO EXECUTE
**Estimated Time:** 30-60 minutes total
