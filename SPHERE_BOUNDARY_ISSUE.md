# Multi-Light System Sphere Boundary Issue

**Date:** 2025-10-17
**Status:** CRITICAL BUG - Blocking Phase 3.5 completion
**Severity:** HIGH - Renders multi-light system unusable beyond 300 units

---

## Executive Summary

The Phase 3.5 Multi-Light System exhibits a hard spherical boundary at approximately 300 units from world origin (0,0,0). Lights positioned beyond this boundary completely fail to illuminate particles, despite:

1. Particles being clearly visible at 400-600+ units (proven by screenshots)
2. Physics constraints disabled (`m_constraintShape = 0`)
3. All 10,000 particles included in BLAS with no distance filtering
4. No hard-coded limits found in shadow ray or lighting shader code
5. TLAS bounds appearing correct (includes all particles)

**Visual Evidence:**
- Light at (20, 0, 0): ✅ Works perfectly
- Light at (500, 0, 0): ❌ Complete failure (zero illumination)
- Light at (2000, 0, 0): ❌ Complete failure (zero illumination)

**The Mystery:** The boundary precisely matches `OUTER_DISK_RADIUS = 300.0f`, but changing this constant to 1000.0 had **zero effect** on the boundary.

---

## Technical Background

### System Architecture

**Multi-Light Pipeline (Phase 3.5):**
```
CPU: m_lights[] (std::vector<Light>) → GPU: g_lights (StructuredBuffer)
     ↓
Gaussian Shader: particle_gaussian_raytrace.hlsl
     ↓
For each light in g_lights:
    1. Compute light direction and distance
    2. Calculate attenuation based on distance
    3. Cast shadow ray via RayQuery (inline ray tracing)
    4. Traverse TLAS to check occlusion
    5. Apply visibility and accumulate lighting
```

**Acceleration Structure (Reused from RTLightingSystem):**
```
RTLightingSystem_RayQuery.cpp:
    → Generates AABBs for all particles (generate_particle_aabbs.hlsl)
    → Builds BLAS (procedural primitives, 10,000 AABBs)
    → Builds TLAS (single instance, identity transform)

ParticleRenderer_Gaussian.cpp:
    → Reuses TLAS via GetTLAS() (no duplicate infrastructure)
```

---

## The Problem: Detailed Analysis

### Symptom Timeline

**Initial Discovery (Session: SPHERE_BOUNDARY_MYSTERY):**
1. Multi-light system implemented and working at close range
2. Moved light from (20,0,0) to (500,0,0) → Complete lighting failure
3. Particles still visible at 500+ units (orbital drift confirmed)
4. Shadow rays failing silently (no errors, just zero illumination)

**Hypothesis 1: Physics Constraints (REJECTED):**
- Suspected: `m_constraintShape` limiting particles to 300-unit radius
- **Evidence Against:**
  - Set `m_constraintShape = 0` (disabled)
  - Particles visibly drifting to 400-600+ units
  - Rasterization rendering particles correctly at distance
  - **Conclusion:** Particles exist beyond boundary, just not lit

**Hypothesis 2: AABB Filtering (REJECTED):**
- Suspected: AABB generation shader filtering particles by distance
- **Evidence Against:**
  - Reviewed `shaders/dxr/generate_particle_aabbs.hlsl` lines 30-70
  - No distance checks, no radius filtering
  - All 10,000 particles get AABBs unconditionally
  - **Conclusion:** BLAS includes all particles

**Hypothesis 3: Shadow Ray TMax (REJECTED):**
- Suspected: Shadow rays using fixed max distance
- **Evidence Against:**
  - Reviewed `shaders/particles/particle_gaussian_raytrace.hlsl` lines 710-760
  - TMax set to `lightDist` (actual distance to light)
  - No hardcoded 300.0 limit found
  - **Conclusion:** Ray traversal not limited by TMax

**Hypothesis 4: TLAS Transform Scaling (CATASTROPHICALLY WRONG):**
- **PIX Agent's Failed Fix:**
  - Scaled TLAS instance transform 3x: `instanceDesc.Transform[0][0] = 3.0f`
  - **Result:** Rendering completely broken (weird concentric shells)
  - FPS: 120+ → 2-5 FPS (frame time 317ms)
  - **Why it failed:** Scaling transform scales ALL geometry (AABBs become 3x larger), breaks ray-ellipsoid intersections
  - **Reverted:** `git checkout src/lighting/RTLightingSystem_RayQuery.cpp`

---

## Code Analysis

### Key Files and Suspicious Constants

#### 1. `src/particles/ParticleSystem.h` (Line 19)
```cpp
static constexpr float OUTER_DISK_RADIUS = 300.0f;  // Schwarzschild radii
```

**Usage:**
- Physics initialization (spawn radius)
- Temperature gradient calculation
- **NOT used in:** AABB generation, shadow rays, TLAS bounds

**Test Result:**
- Changed to 1000.0 in `config_dev.json`
- Logs still showed: `outerRadius: 300.000000`
- **Reason:** Static constexpr overrides config value
- **Impact:** Changing this had ZERO effect on sphere boundary

---

#### 2. `shaders/particles/particle_gaussian_raytrace.hlsl` (Lines 710-760)

**Multi-Light Loop:**
```hlsl
// Loop over all lights (up to 16)
for (uint lightIdx = 0; lightIdx < lightCount; lightIdx++) {
    Light light = g_lights[lightIdx];

    // Direction and distance to this light
    float3 lightDir = normalize(light.position - pos);
    float lightDist = length(light.position - pos);

    // Use light.radius for soft falloff (FIX #1 - JUST APPLIED)
    float normalizedDist = lightDist / max(light.radius, 1.0);
    float attenuation = 1.0 / (1.0 + normalizedDist * normalizedDist);  // Quadratic

    // Cast shadow ray to check visibility
    if (useShadowRays) {
        RayDesc shadowRay;
        shadowRay.Origin = pos + lightDir * 0.01;  // Bias to avoid self-intersection
        shadowRay.Direction = lightDir;
        shadowRay.TMin = 0.0;
        shadowRay.TMax = lightDist;  // ← IMPORTANT: Uses actual distance, no 300 limit!

        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> shadowQuery;
        shadowQuery.TraceRayInline(
            g_tlas,                     // ← Uses RTLightingSystem's TLAS
            RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES,
            0xFF,
            shadowRay
        );

        shadowQuery.Proceed();

        // Check if ray hit anything
        if (shadowQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT ||
            shadowQuery.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT) {
            visibility = 0.0;  // In shadow
        } else {
            visibility = 1.0;  // Lit
        }
    }

    // Accumulate lighting
    float3 lightContribution = light.color * light.intensity * attenuation * visibility;
    multiLightColor += lightContribution;
}
```

**Analysis:**
- `TMax = lightDist` → No distance limit
- `RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES` → **WAIT, THIS IS SUSPICIOUS!**
- Uses `g_tlas` from RTLightingSystem (shares acceleration structure)
- No conditional based on distance or radius

**CRITICAL FINDING:**
```hlsl
RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES
```

**This flag tells the ray to skip procedural primitive intersection testing!**

But our particles ARE procedural primitives (AABB-based, no triangles). This means shadow rays are looking for TRIANGLE_HIT, but our BLAS only contains procedural primitives!

**Expected behavior with this flag:**
- Shadow rays skip AABB intersection tests
- Only look for triangle geometry
- If BLAS contains ONLY procedural primitives → rays always miss
- **Result:** `shadowQuery.CommittedStatus() == COMMITTED_NOTHING` → visibility = 1.0 (fully lit, no shadows)

**But wait...** This doesn't explain the 300-unit boundary. If this flag was wrong, it would fail at ALL distances, not just beyond 300 units.

---

#### 3. `src/lighting/RTLightingSystem_RayQuery.cpp` (Lines 350-450)

**BLAS Build Configuration:**
```cpp
// Configure geometry for procedural primitives
D3D12_RAYTRACING_GEOMETRY_DESC geometryDesc = {};
geometryDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
geometryDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;  // ← Important!
geometryDesc.AABBs.AABBCount = m_particleCount;  // All 10,000 particles
geometryDesc.AABBs.AABBs.StartAddress = m_aabbBuffer->GetGPUVirtualAddress();
geometryDesc.AABBs.AABBs.StrideInBytes = sizeof(D3D12_RAYTRACING_AABB);  // 24 bytes

// BLAS build inputs
D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {};
inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
inputs.NumDescs = 1;
inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
inputs.pGeometryDescs = &geometryDesc;

// Build BLAS (no distance filtering, no culling)
commandList->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);
```

**TLAS Build Configuration:**
```cpp
// Single instance (identity transform)
D3D12_RAYTRACING_INSTANCE_DESC instanceDesc = {};
instanceDesc.Transform[0][0] = 1.0f;  // Identity
instanceDesc.Transform[1][1] = 1.0f;
instanceDesc.Transform[2][2] = 1.0f;
instanceDesc.InstanceID = 0;
instanceDesc.InstanceMask = 0xFF;  // ← All rays can hit
instanceDesc.InstanceContributionToHitGroupIndex = 0;
instanceDesc.Flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
instanceDesc.AccelerationStructure = m_blas->GetGPUVirtualAddress();

// Build TLAS
commandList->BuildRaytracingAccelerationStructure(&tlasBuildDesc, 0, nullptr);
```

**Analysis:**
- BLAS includes all 10,000 particles (no distance filtering)
- TLAS uses identity transform (no scaling, no translation)
- `InstanceMask = 0xFF` → All rays can hit this instance
- No conditional logic based on distance

---

#### 4. `shaders/dxr/generate_particle_aabbs.hlsl` (Lines 30-70)

**AABB Generation (runs every frame):**
```hlsl
[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint particleIdx = DTid.x;
    if (particleIdx >= g_particleCount) return;

    // Load particle
    Particle particle = g_particles[particleIdx];
    float3 center = particle.position;

    // Compute AABB based on particle radius
    float3 extent = float3(g_particleRadius, g_particleRadius, g_particleRadius);

    // Write AABB (NO distance filtering!)
    g_aabbs[particleIdx].MinX = center.x - extent.x;
    g_aabbs[particleIdx].MinY = center.y - extent.y;
    g_aabbs[particleIdx].MinZ = center.z - extent.z;
    g_aabbs[particleIdx].MaxX = center.x + extent.x;
    g_aabbs[particleIdx].MaxY = center.y + extent.y;
    g_aabbs[particleIdx].MaxZ = center.z + extent.z;
}
```

**Analysis:**
- Unconditionally generates AABBs for ALL particles
- No `if (length(center) < 300.0)` checks
- No radius-based culling
- **Conclusion:** All particles at 400-600+ units DO have AABBs in BLAS

---

## Debugging Strategies

### Strategy 1: Buffer Dumps (IN-APP APPROACH)

**Goal:** Verify AABB contents at runtime

**Implementation:**
```cpp
// Add to Application.cpp (around line 1500)
void Application::DumpBuffers(uint32_t frameNum) {
    // Read back AABB buffer
    std::vector<D3D12_RAYTRACING_AABB> aabbs(m_particleCount);
    m_rtLightingSystem->ReadbackAABBs(aabbs.data(), m_particleCount);

    // Analyze spatial distribution
    uint32_t countNear = 0;   // < 300 units
    uint32_t countFar = 0;    // >= 300 units
    float maxRadius = 0.0f;

    for (uint32_t i = 0; i < m_particleCount; i++) {
        float centerX = (aabbs[i].MinX + aabbs[i].MaxX) * 0.5f;
        float centerY = (aabbs[i].MinY + aabbs[i].MaxY) * 0.5f;
        float centerZ = (aabbs[i].MinZ + aabbs[i].MaxZ) * 0.5f;
        float radius = sqrt(centerX*centerX + centerY*centerY + centerZ*centerZ);

        if (radius < 300.0f) countNear++;
        else countFar++;

        maxRadius = max(maxRadius, radius);
    }

    LOG_INFO("=== AABB Spatial Distribution (Frame {}) ===", frameNum);
    LOG_INFO("  Particles < 300 units: {}", countNear);
    LOG_INFO("  Particles >= 300 units: {}", countFar);
    LOG_INFO("  Max radius: {:.2f} units", maxRadius);
    LOG_INFO("============================================");

    // Dump to file
    std::string filename = std::format("PIX/buffer_dumps/aabbs_frame_{}.bin", frameNum);
    FILE* f = fopen(filename.c_str(), "wb");
    fwrite(aabbs.data(), sizeof(D3D12_RAYTRACING_AABB), m_particleCount, f);
    fclose(f);
}

// Trigger with keyboard
if (GetAsyncKeyState('B') & 0x8000) {
    DumpBuffers(m_frameCount);
    LOG_INFO("Dumped buffers for frame {}", m_frameCount);
}
```

**Expected Results:**
- If `countFar > 0`: Particles beyond 300 units ARE in BLAS
- If `maxRadius > 300`: BLAS bounds extend beyond 300 units
- **If both true:** Problem is NOT in BLAS, must be in shader logic

---

### Strategy 2: Shadow Ray Debug Visualization

**Goal:** Visualize which shadow rays succeed vs fail

**Implementation:**
```hlsl
// Add to particle_gaussian_raytrace.hlsl (around line 750)

// Debug: Color-code by shadow ray success
if (lightIdx == 0) {  // Only first light for clarity
    if (visibility > 0.5) {
        // Shadow ray succeeded (hit nothing)
        debugColor = float3(0, 1, 0);  // Green = lit
    } else {
        // Shadow ray hit something (occluded)
        debugColor = float3(1, 0, 0);  // Red = shadowed
    }

    // Distance-based debug (check for 300-unit boundary)
    float distFromOrigin = length(pos);
    if (distFromOrigin > 300.0) {
        debugColor.b = 1.0;  // Add blue channel for far particles
    }
}

// Output debug visualization
g_output[DTid.xy] = float4(debugColor, 1.0);
```

**Expected Results:**
- Particles < 300 units: Green (lit) or Red (shadowed) ← Normal
- Particles >= 300 units: Should be Green/Red+Blue
- **If all far particles are black:** Shadow rays failing (not even running?)
- **If all far particles are green (no occlusion):** Shadow rays running but finding nothing (TLAS miss?)

---

### Strategy 3: TLAS Bounds Verification

**Goal:** Verify TLAS world-space bounds include particles beyond 300 units

**Implementation:**
```cpp
// Add to RTLightingSystem_RayQuery.cpp (after TLAS build)
void RTLightingSystem_RayQuery::VerifyTLASBounds() {
    // Query TLAS postbuild info
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC postbuildDesc = {};
    postbuildDesc.InfoType = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE;
    postbuildDesc.DestBuffer = /* readback buffer */;

    // Log BLAS/TLAS sizes
    LOG_INFO("=== Acceleration Structure Info ===");
    LOG_INFO("  BLAS size: {} bytes", blasSize);
    LOG_INFO("  TLAS size: {} bytes", tlasSize);

    // Manually compute world-space bounds
    DirectX::XMFLOAT3 worldMin(FLT_MAX, FLT_MAX, FLT_MAX);
    DirectX::XMFLOAT3 worldMax(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    // Read back particle positions
    for (uint32_t i = 0; i < m_particleCount; i++) {
        DirectX::XMFLOAT3 pos = m_particles[i].position;
        worldMin.x = min(worldMin.x, pos.x - m_particleRadius);
        worldMin.y = min(worldMin.y, pos.y - m_particleRadius);
        worldMin.z = min(worldMin.z, pos.z - m_particleRadius);
        worldMax.x = max(worldMax.x, pos.x + m_particleRadius);
        worldMax.y = max(worldMax.y, pos.y + m_particleRadius);
        worldMax.z = max(worldMax.z, pos.z + m_particleRadius);
    }

    float boundsRadius = sqrt(worldMax.x*worldMax.x + worldMax.y*worldMax.y + worldMax.z*worldMax.z);

    LOG_INFO("  World bounds: ({:.2f}, {:.2f}, {:.2f}) to ({:.2f}, {:.2f}, {:.2f})",
             worldMin.x, worldMin.y, worldMin.z,
             worldMax.x, worldMax.y, worldMax.z);
    LOG_INFO("  Bounds radius: {:.2f} units", boundsRadius);
    LOG_INFO("===================================");
}
```

**Expected Results:**
- If `boundsRadius > 300`: TLAS should include far particles
- Compare with actual TLAS bounds from D3D12 debug layer
- **If bounds stop at 300:** TLAS build bug (but where?)

---

### Strategy 4: Step-Through Debugging with RenderDoc/PIX

**RenderDoc Capture:**
1. Launch RenderDoc
2. Attach to PlasmaDX-Clean.exe
3. Capture frame with light at (500, 0, 0)
4. Inspect:
   - TLAS resource view (check bounds visualization)
   - Shadow ray dispatch (verify rays launched for far particles)
   - AABB buffer contents (verify far AABBs exist)

**PIX Capture (Already supported):**
```bash
./build/DebugPIX/PlasmaDX-Clean-PIX.exe --config=configs/agents/pix_agent.json
# Auto-captures at frame 120
```

Then analyze:
- Timeline view: Find "Multi-Light Shadow Rays" event
- Resource viewer: Inspect `m_aabbBuffer` and `m_tlas`
- Shader debugger: Step through shadow ray loop for pixel at far particle

---

## Hypotheses to Test

### Hypothesis A: Ray Query Ray Mask Mismatch

**Theory:** Shadow rays use wrong ray mask, fail to intersect TLAS instance

**Test:**
```hlsl
// Current:
shadowQuery.TraceRayInline(g_tlas, RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES, 0xFF, shadowRay);

// Try:
shadowQuery.TraceRayInline(g_tlas, RAY_FLAG_NONE, 0xFF, shadowRay);
// AND verify TLAS instance mask: instanceDesc.InstanceMask = 0xFF;
```

**Expected:** If this is the issue, removing `RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES` should fix it at ALL distances, not just beyond 300 units.

---

### Hypothesis B: TLAS Rebuild Frequency Issue

**Theory:** TLAS not rebuilt every frame, caches old particle positions (< 300 units)

**Test:**
```cpp
// Add logging to RTLightingSystem_RayQuery::Update()
LOG_INFO("Frame {}: Rebuilding BLAS/TLAS with {} particles", frameNum, m_particleCount);

// Verify rebuild called every frame
static uint64_t lastRebuildFrame = 0;
if (frameNum - lastRebuildFrame > 1) {
    LOG_WARN("TLAS rebuild skipped! Last rebuild: frame {}", lastRebuildFrame);
}
lastRebuildFrame = frameNum;
```

**Expected:** If rebuilds are skipped, TLAS would contain stale particle positions (all < 300 units from initial spawn).

---

### Hypothesis C: Shader Constant Buffer Corruption

**Theory:** `g_tlas` descriptor pointing to stale/wrong acceleration structure beyond certain distance

**Test:**
```hlsl
// Add to particle_gaussian_raytrace.hlsl (top of multi-light loop)
if (lightIdx == 0) {
    // Verify TLAS is valid by tracing test ray
    RayDesc testRay;
    testRay.Origin = float3(0, 0, 0);
    testRay.Direction = float3(1, 0, 0);  // +X axis
    testRay.TMin = 0.0;
    testRay.TMax = 1000.0;  // Well beyond 300 units

    RayQuery<RAY_FLAG_NONE> testQuery;
    testQuery.TraceRayInline(g_tlas, RAY_FLAG_NONE, 0xFF, testRay);
    testQuery.Proceed();

    // Log hit distance
    if (testQuery.CommittedStatus() != COMMITTED_NOTHING) {
        float hitDist = testQuery.CommittedRayT();
        // If hitDist always < 300, TLAS bounds stop at 300!
    }
}
```

---

### Hypothesis D: Light Buffer Upload Issue

**Theory:** Lights beyond 300 units not properly uploaded to GPU

**Test:**
```cpp
// Add to Application::Update() (after UpdateLights call)
LOG_INFO("=== Light Buffer Contents ===");
for (size_t i = 0; i < m_lights.size(); i++) {
    const Light& light = m_lights[i];
    float distFromOrigin = sqrt(light.position.x * light.position.x +
                                light.position.y * light.position.y +
                                light.position.z * light.position.z);
    LOG_INFO("  Light {}: pos=({:.2f}, {:.2f}, {:.2f}), dist={:.2f}, color=({:.2f}, {:.2f}, {:.2f})",
             i, light.position.x, light.position.y, light.position.z, distFromOrigin,
             light.color.x, light.color.y, light.color.z);
}
LOG_INFO("=============================");
```

**Expected:** If light at (500,0,0) not in buffer, it's a CPU→GPU upload issue.

---

## Smoking Gun: RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES

**MOST LIKELY CULPRIT:**

```hlsl
// Line ~730 in particle_gaussian_raytrace.hlsl
shadowQuery.TraceRayInline(
    g_tlas,
    RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES,  // ← THIS FLAG IS WRONG!
    0xFF,
    shadowRay
);
```

**What this flag does:**
- Tells ray traversal to SKIP intersection shaders for procedural primitives
- Only tests against triangle geometry
- Our particles are 100% procedural primitives (AABBs, no triangles)

**Expected behavior with this flag:**
- ALL shadow rays should miss (no geometry to hit)
- Result: All particles fully lit (no shadows)

**But why does it work < 300 units?**

**Possible explanation:**
There might be TRIANGLE geometry in the BLAS for particles < 300 units (from initial disk spawn), but NO triangle geometry generated for particles that drift beyond 300 units.

**How to test:**
```hlsl
// Change flag from:
RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES

// To:
RAY_FLAG_NONE  // Test all geometry
```

If this fixes the issue at ALL distances, we found the bug!

---

## Recommended Debugging Sequence

1. **Quick Test (5 minutes):**
   - Change `RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES` → `RAY_FLAG_NONE`
   - Recompile shader
   - Test with light at (500, 0, 0)
   - **If fixed:** Problem solved!
   - **If not fixed:** Continue to step 2

2. **Buffer Dump (15 minutes):**
   - Implement `DumpBuffers()` function
   - Trigger at frame 120
   - Analyze AABB spatial distribution
   - Verify particles beyond 300 units have AABBs

3. **TLAS Bounds Verification (15 minutes):**
   - Implement `VerifyTLASBounds()`
   - Check if world bounds extend beyond 300 units
   - Compare with D3D12 debug layer TLAS info

4. **PIX/RenderDoc Capture (30 minutes):**
   - Capture frame with light at (500, 0, 0)
   - Inspect TLAS resource (bounds visualization)
   - Step through shadow ray shader for far particle pixel
   - Check committed ray status

5. **Shadow Ray Debug Visualization (20 minutes):**
   - Implement color-coded debug output
   - Green = lit, Red = shadowed, Blue = far (>300 units)
   - Identify spatial pattern of failures

---

## Success Criteria

**Bug is fixed when:**
- [ ] Light at (500, 0, 0) illuminates visible particles
- [ ] Light at (2000, 0, 0) illuminates visible particles
- [ ] No hard boundary at 300 units
- [ ] Shadow rays work correctly at all distances
- [ ] No performance regression (still 120+ FPS)
- [ ] No artifacts introduced

---

## Files Requiring Investigation

**High Priority:**
1. `shaders/particles/particle_gaussian_raytrace.hlsl` (Lines 710-760) - Shadow ray loop
2. `src/lighting/RTLightingSystem_RayQuery.cpp` (Lines 350-450) - BLAS/TLAS build
3. `shaders/dxr/generate_particle_aabbs.hlsl` (Lines 30-70) - AABB generation

**Medium Priority:**
4. `src/particles/ParticleRenderer_Gaussian.cpp` (Lines 200-250) - TLAS reuse logic
5. `src/core/Application.cpp` (Lines 480-520) - Light buffer upload
6. `shaders/dxr/particle_intersection.hlsl` - Procedural primitive intersection shader

**Low Priority:**
7. `src/particles/ParticleSystem.h` (Line 19) - OUTER_DISK_RADIUS constant
8. `config_dev.json` - Runtime configuration

---

## Previous Failed Attempts

**DO NOT REPEAT THESE:**

1. ❌ **Scaling TLAS Transform:**
   - Attempted by PIX agent
   - Scaled instance transform 3x
   - Result: Complete rendering breakdown (317ms frame time)
   - Reason: Scales geometry, not bounds; breaks ray intersections

2. ❌ **Changing OUTER_DISK_RADIUS:**
   - Changed config value to 1000.0
   - Had zero effect on boundary
   - Reason: Static constexpr in code overrides config

3. ❌ **Disabling Physics Constraints:**
   - Set `m_constraintShape = 0`
   - Particles drifted beyond 300 units
   - But still not lit by multi-lights
   - Reason: Confirms particles exist, problem is in lighting/shadowing

---

## Next Steps for Agent

**If you are an autonomous debugging agent, follow this sequence:**

1. Read this entire document carefully
2. Implement the "Quick Test" (change ray flag) first
3. If that doesn't work, implement buffer dumps
4. Analyze buffer dumps and log TLAS bounds
5. If still not resolved, request PIX capture and analyze
6. Report findings with specific line numbers and code snippets
7. Propose fix with detailed explanation

**Do NOT:**
- Scale TLAS transforms (proven to break rendering)
- Change physics constants (won't affect lighting)
- Guess without evidence (test hypotheses systematically)

---

**Document Version:** 1.0
**Last Updated:** 2025-10-17
**Ready for Agent Deployment**
