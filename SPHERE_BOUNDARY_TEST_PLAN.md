# Sphere Boundary Test Plan

**Created:** 2025-10-17
**Test Suite Version:** 1.0
**Target Issue:** Multi-light system fails beyond ~300 units

---

## Executive Summary

The multi-light system exhibits a hard cutoff at approximately 300 units from the origin. This test suite systematically tests lights at 7 distances (50, 100, 200, 300, 400, 500, 1000 units) to:

1. Confirm the exact boundary distance
2. Determine if it's a hard cutoff or gradual falloff
3. Isolate the root cause (shader code, acceleration structure, or configuration)
4. Validate fixes after implementation

---

## Test Matrix

### Distance Points

| Distance | Light Color | Expected Result | Rationale |
|----------|-------------|-----------------|-----------|
| 50 units | RED | PASS | Well within working range |
| 100 units | ORANGE | PASS | Should work reliably |
| 200 units | YELLOW | PASS (possible degradation) | Approaching boundary |
| 300 units | GREEN | **BOUNDARY TEST** | Exactly at `physics.outerRadius` |
| 400 units | CYAN | FAIL | Just beyond boundary (+100) |
| 500 units | BLUE | FAIL | Well beyond boundary (+200) |
| 1000 units | MAGENTA | FAIL | Far beyond boundary (+700) |

### Common Configuration

All tests share these settings:

- **Particle Count:** 10,000
- **Physics:** DISABLED (allows particles to drift freely)
- **ReSTIR:** DISABLED (eliminates ReSTIR as a variable)
- **Shadow Rays:** ENABLED (test if they work at all distances)
- **Phase Function:** ENABLED (visibility of lighting effects)
- **Physical Emission:** DISABLED (0.1 strength for minimal self-glow)
- **Light Intensity:** 10.0 (very bright for easy identification)
- **Light Radius:** 300.0 (large enough to reach particles at origin)

### Per-Test Variables

Each test varies:
- **Light position:** (distance, 0, 0)
- **Light color:** Unique RGB value for visual identification
- **Camera position:** Adjusted to frame both light and particles
- **PIX capture prefix:** `sphere_XXXu_` for easy identification

---

## Hypothesis

The boundary condition is related to `physics.outerRadius = 300.0` being used incorrectly somewhere in:

1. **Shader culling logic** - Particles/lights beyond 300 units culled
2. **Acceleration structure bounds** - TLAS/BLAS built with 300 unit hard limit
3. **Configuration constant** - Some shader constant using outerRadius as max distance
4. **Attenuation formula** - Light attenuation zeroing out at 300 units

---

## Test Execution

### Prerequisites

1. Build DebugPIX configuration:
   ```bash
   MSBuild PlasmaDX-Clean.sln /p:Configuration=DebugPIX /p:Platform=x64
   ```

2. Ensure PIX for Windows is installed (for capture analysis)

3. Verify test configs exist:
   ```bash
   ls configs/scenarios/sphere_test_*.json
   ```

### Running the Test Suite

**Automated execution:**
```bash
./test_sphere_boundary.sh
```

**Manual execution (single test):**
```bash
./build/DebugPIX/PlasmaDX-Clean-PIX.exe --config=configs/scenarios/sphere_test_300u.json
```

### Test Duration

- Per test: ~10 seconds (3 seconds runtime + capture + cooldown)
- Full suite: ~90 seconds (7 tests)

### Output Locations

```
test_results/sphere_boundary_YYYYMMDD_HHMMSS/
├── captures/           # PIX GPU captures (.wpix files)
├── buffer_dumps/       # GPU buffer dumps per test
│   ├── sphere_050u/
│   ├── sphere_100u/
│   └── ...
├── logs/               # Application logs per test
│   ├── sphere_050u.log
│   └── ...
└── test_summary.txt    # Overall test results
```

---

## Success Criteria

### Visual Inspection (Primary)

Open each PIX capture and verify:

**✅ PASS Criteria:**
- Particles are illuminated by the light
- Light color is visible on particle surfaces
- Shadow rays create realistic occlusion
- Phase function creates rim lighting halos
- No hard cutoff at arbitrary distance

**❌ FAIL Criteria:**
- Particles are completely black (no lighting)
- Light color not visible anywhere
- Hard cutoff at 300 units
- Shadow rays not working
- GPU errors or crashes

### Quantitative Metrics

Analyze buffer dumps to check:

1. **Light Buffer (`g_lights`):**
   - Light count = 1
   - Light position matches config
   - Light intensity = 10.0
   - Light radius = 300.0

2. **Particle Buffer (`g_particles`):**
   - Particles distributed within ~10-300 unit radius
   - Positions valid (no NaN/Inf)
   - Velocities reasonable

3. **Output Buffer (`g_output`):**
   - Non-zero RGB values in lit regions
   - Light color contribution visible
   - Not all pixels black (indicates rendering)

### Log File Analysis

Check logs for:
- ✅ "Rendering with 1 light(s)" message
- ✅ No D3D12 errors or warnings
- ✅ Stable FPS (>100 FPS @ 10K particles)
- ❌ Shader compilation errors
- ❌ Resource creation failures
- ❌ TLAS/BLAS build errors

---

## Expected Outcomes

### Scenario A: Hard Cutoff at 300 Units

**Observation:**
- Tests 1-3 (50-200u): PASS
- Test 4 (300u): Boundary behavior (partial lighting?)
- Tests 5-7 (400-1000u): FAIL (complete darkness)

**Implication:** Configuration constant or shader logic using `outerRadius` as hard limit

**Next Step:** Search shader code for:
```hlsl
if (distance > outerRadius) { discard/cull }
if (distance > 300.0) { return 0; }
```

### Scenario B: Gradual Falloff

**Observation:**
- Tests 1-2 (50-100u): Full brightness
- Test 3 (200u): Dimmer but visible
- Test 4 (300u): Very dim
- Tests 5-7 (400-1000u): Nearly black

**Implication:** Attenuation formula too aggressive or incorrect

**Next Step:** Review attenuation calculation in `particle_gaussian_raytrace.hlsl`:
```hlsl
float attenuation = 1.0 / (distance * distance);  // Too steep?
```

### Scenario C: Acceleration Structure Bounds

**Observation:**
- Tests 1-4 (50-300u): PASS
- Tests 5-7 (400-1000u): FAIL (but PIX shows no raytracing at all)

**Implication:** TLAS/BLAS built with 300 unit bounds, rays beyond bounds don't traverse

**Next Step:** Check `RTLightingSystem_RayQuery.cpp` TLAS/BLAS build:
```cpp
D3D12_RAYTRACING_AABB aabb;
aabb.MinX = -outerRadius;  // BUG: Should be scene bounds, not physics bounds!
aabb.MaxX = +outerRadius;
```

### Scenario D: All Tests PASS

**Observation:** All 7 tests show correct lighting

**Implication:** Bug was fixed, or test conditions don't reproduce the issue

**Next Step:**
1. Test with multiple lights simultaneously (13 lights at various distances)
2. Test with ReSTIR enabled
3. Test with physics enabled (particles constrained to disk)

---

## Debugging Workflow

### Phase 1: Identify Boundary Type (This Test Suite)

Run automated test suite and determine which scenario (A, B, C, or D) matches observations.

### Phase 2: Shader Code Search

Based on Phase 1 results, search for:

**Hard cutoff (Scenario A):**
```bash
grep -r "outerRadius" shaders/
grep -r "300.0" shaders/
grep -r "distance.*>" shaders/particles/particle_gaussian_raytrace.hlsl
```

**Gradual falloff (Scenario B):**
```bash
grep -r "attenuation" shaders/particles/particle_gaussian_raytrace.hlsl
grep -r "/ (distance" shaders/
```

**Acceleration structure (Scenario C):**
```bash
grep -r "RAYTRACING_AABB" src/lighting/RTLightingSystem_RayQuery.cpp
grep -r "outerRadius" src/lighting/RTLightingSystem_RayQuery.cpp
```

### Phase 3: PIX GPU Capture Analysis

1. Open `sphere_300u_*.wpix` capture (critical boundary test)
2. Navigate to frame 120 (capture frame)
3. Inspect `particle_gaussian_raytrace` compute shader dispatch
4. Check shader constants:
   - `lightCount = 1`
   - `light[0].position = (300, 0, 0)`
   - `light[0].radius = 300.0`
5. Set breakpoint at light loop: `for (uint lightIdx = 0; lightIdx < lightCount; ++lightIdx)`
6. Step through light contribution calculation
7. Watch variables:
   - `lightDir` - Direction from particle to light
   - `lightDist` - Distance to light
   - `attenuation` - Attenuation factor
   - `totalLight` - Accumulated lighting

### Phase 4: Buffer Dump Analysis

Use Python scripts to analyze buffer dumps:

```bash
cd PIX/scripts/analysis
python analyze_light_buffer.py ../../test_results/sphere_boundary_XXX/buffer_dumps/sphere_300u/
```

Expected output:
```
Light 0:
  Position: (300.0, 0.0, 0.0)
  Intensity: 10.0
  Color: (0.0, 1.0, 0.0)  # GREEN
  Radius: 300.0
  Distance to origin: 300.0 units
  Distance to nearest particle: ~290-310 units (should be in range!)
```

---

## Root Cause Suspects (Priority Order)

### 1. TLAS/BLAS Bounds (HIGH PROBABILITY)

**Location:** `src/lighting/RTLightingSystem_RayQuery.cpp`

**Suspect Code:**
```cpp
// Building BLAS for procedural primitives
D3D12_RAYTRACING_AABB aabb;
aabb.MinX = particle.position.x - particleRadius;
aabb.MaxX = particle.position.x + particleRadius;
// ... but what if particles are culled based on outerRadius?
```

**Fix:** Ensure BLAS includes ALL particles, not just those within `outerRadius`.

### 2. Shader Distance Culling (MEDIUM PROBABILITY)

**Location:** `shaders/particles/particle_gaussian_raytrace.hlsl`

**Suspect Code:**
```hlsl
// Early exit if light too far?
float lightDist = length(lightPos - particlePos);
if (lightDist > 300.0) continue;  // BUG!
```

**Fix:** Remove arbitrary distance checks, or use light.radius instead.

### 3. Attenuation Formula (LOW PROBABILITY)

**Location:** `shaders/particles/particle_gaussian_raytrace.hlsl`

**Suspect Code:**
```hlsl
// Inverse square law too aggressive?
float attenuation = 1.0 / (lightDist * lightDist + 1.0);
// At 300 units: 1.0 / (90000 + 1) = 0.000011 (effectively zero!)
```

**Fix:** Adjust attenuation formula or use light.radius for clamping:
```hlsl
float attenuation = 1.0 / (lightDist * lightDist + 1.0);
attenuation *= smoothstep(light.radius, 0.0, lightDist);  // Smooth falloff to radius
```

### 4. Configuration Load Error (VERY LOW)

**Location:** `src/core/Application.cpp`

**Suspect Code:**
```cpp
// Is "lights" array being loaded from JSON correctly?
if (config.contains("lights")) {
    // ...
}
```

**Fix:** Add logging to verify light positions at startup.

---

## Validation After Fix

Once root cause is identified and fixed:

1. **Re-run full test suite:**
   ```bash
   ./test_sphere_boundary.sh
   ```

2. **Verify all tests PASS:**
   - All 7 distances show correct lighting
   - No hard cutoff at 300 units
   - Shadow rays work at all distances
   - No visual artifacts

3. **Stress test:**
   ```bash
   # 13 lights at various distances (50-1000 units)
   ./build/DebugPIX/PlasmaDX-Clean-PIX.exe --config=configs/user/default.json
   ```

4. **Performance regression check:**
   - FPS should be unchanged (multi-light already expensive)
   - No new GPU errors or warnings

---

## Manual Testing Procedure (Fallback)

If automated script fails, run tests manually:

### Step 1: Build

```bash
MSBuild PlasmaDX-Clean.sln /p:Configuration=DebugPIX /p:Platform=x64
```

### Step 2: Test Each Distance

For each config (sphere_test_050u.json through sphere_test_1000u.json):

```bash
# Run application
./build/DebugPIX/PlasmaDX-Clean-PIX.exe --config=configs/scenarios/sphere_test_XXXu.json

# Let it run for 3-5 seconds, then close

# Note observations:
# - Are particles lit?
# - Can you see the light color?
# - Are shadow rays working?
# - Any visual artifacts?
```

### Step 3: Capture Analysis

```bash
# Open latest capture in PIX
explorer PIX/Captures

# In PIX:
# 1. Open sphere_XXXu_*.wpix file
# 2. Navigate to frame 120
# 3. Find particle_gaussian_raytrace dispatch
# 4. Inspect shader constants and buffers
# 5. Set breakpoint in light loop
# 6. Check light position, intensity, radius
# 7. Check distance calculation
# 8. Check attenuation calculation
```

### Step 4: Document Findings

Create `test_results/manual_test_YYYYMMDD.txt`:

```
Distance | Light Color | Particles Lit? | Shadow Rays? | Notes
---------|-------------|----------------|--------------|-------
50       | RED         | YES            | YES          | Perfect
100      | ORANGE      | YES            | YES          | Perfect
200      | YELLOW      | YES            | YES          | Slightly dimmer?
300      | GREEN       | PARTIAL?       | PARTIAL?     | BOUNDARY!
400      | CYAN        | NO             | NO           | Complete failure
500      | BLUE        | NO             | NO           | Complete failure
1000     | MAGENTA     | NO             | NO           | Complete failure
```

---

## FAQ

### Q: Why disable physics?

**A:** Physics constraints particles within 10-300 unit radius. Disabling physics lets particles drift freely, eliminating "no particles near distant light" as a confounding variable.

### Q: Why disable ReSTIR?

**A:** ReSTIR is a complex system with its own distance-based logic. Disabling it isolates the bug to the direct lighting path.

### Q: Why use different colors per distance?

**A:** Visual confirmation. If you see GREEN light on particles, you know the 300-unit light is working. If you see only RED/ORANGE, you know 50-100 unit lights work but 300+ don't.

### Q: What if ALL tests fail (complete darkness)?

**A:** Likely a config loading issue or broken light system unrelated to distance. Check:
1. Logs for "Rendering with X light(s)" message
2. ImGui shows "Active Lights: 1 / 16"
3. Light buffer is non-null in PIX

### Q: What if ALL tests pass?

**A:** Bug may require specific conditions:
1. Multiple lights simultaneously (test with 13 lights)
2. ReSTIR enabled (temporal/spatial reuse bug)
3. Physics enabled (particles constrained to disk)
4. Specific camera angle or distance

Try reproducing the original issue conditions.

### Q: Can I run tests on Debug build instead of DebugPIX?

**A:** Yes, but you won't get PIX captures. Change script to use `build/Debug/PlasmaDX-Clean.exe`.

### Q: How do I analyze buffer dumps?

**A:** Use Python scripts in `PIX/scripts/analysis/`:
```bash
cd PIX/scripts/analysis
python analyze_buffers.py ../../test_results/sphere_boundary_XXX/buffer_dumps/sphere_300u/
```

Or manually with hex editor:
- `g_lights.bin` - 32 bytes per light (float3 pos, float intensity, float3 color, float radius)
- `g_particles.bin` - 64 bytes per particle (float3 pos, float3 vel, float temp, etc.)

---

## Success Metrics

This test suite is considered successful if:

1. ✅ All 7 tests execute without crashes
2. ✅ PIX captures generated for all tests
3. ✅ Exact boundary distance identified (50-300u work, 400-1000u fail)
4. ✅ Root cause hypothesis formed based on observations
5. ✅ Next debugging steps clearly defined

After fix implementation:

6. ✅ Re-run shows all 7 tests PASS
7. ✅ No performance regression (<5% FPS drop)
8. ✅ No new visual artifacts introduced

---

## Appendix A: Quick Reference Commands

```bash
# Build DebugPIX
MSBuild PlasmaDX-Clean.sln /p:Configuration=DebugPIX /p:Platform=x64

# Run full test suite
./test_sphere_boundary.sh

# Run single test
./build/DebugPIX/PlasmaDX-Clean-PIX.exe --config=configs/scenarios/sphere_test_300u.json

# View latest results
explorer test_results

# Open PIX captures
explorer PIX/Captures

# Analyze buffer dumps
cd PIX/scripts/analysis && python analyze_buffers.py ../../test_results/latest/

# Search for suspect code
grep -r "outerRadius" shaders/
grep -r "300.0" shaders/
grep -r "attenuation" shaders/particles/particle_gaussian_raytrace.hlsl
```

---

## Appendix B: Config JSON Schema

Each test config includes:

```json
{
  "lights": [
    {
      "position": [X, Y, Z],     // Float3 - Light position in world space
      "intensity": 10.0,          // Float - Brightness multiplier
      "color": [R, G, B],         // Float3 - RGB color (0-1 range)
      "radius": 300.0             // Float - Effective range of light
    }
  ],
  "physics": {
    "physicsEnabled": false       // Disable to let particles drift
  },
  "features": {
    "enableReSTIR": false,        // Disable to isolate direct lighting
    "enableShadowRays": true,     // Keep shadow rays for visibility test
    "usePhysicalEmission": false, // Minimal self-glow (0.1 strength)
    "emissionStrength": 0.1
  }
}
```

---

## Appendix C: Light Structure (HLSL)

From `particle_gaussian_raytrace.hlsl`:

```hlsl
struct Light {
    float3 position;   // World space position
    float intensity;   // Brightness multiplier
    float3 color;      // RGB color (0-1)
    float radius;      // Effective range (used for attenuation)
};

StructuredBuffer<Light> g_lights : register(t4);
```

Ensure shader reads `light.radius` and doesn't hardcode 300.0!

---

**Last Updated:** 2025-10-17
**Author:** stress-tester-v3 (autonomous testing agent)
**Status:** READY FOR EXECUTION
