# BUG: Water Mesh RT Hit Detection Failure

**Date:** 2025-12-30
**Status:** OPEN - Requires Multi-Agent Debugging
**Severity:** HIGH - Blocks Blender-to-RT Pipeline Integration
**Feature Branch:** `feature/water-mesh-rt`

---

## Executive Summary

Water mesh assets load successfully and GPU resources are created, but rays fail to intersect the water geometry in the TLAS. Debug visualization confirms `enableWaterMesh` shader constant is set (cyan background tint appears), but `waterMeshHit` is never true.

**Key Symptom:** Screen shows cyan tint (debug mode active) but no visible water geometry.

---

## Related Plan Documents

- **Primary Plan:** `plans/feat-water-shader-minimum-viable.md`
- **Pipeline Context:** `plans/feat-realtime-fluid-simulation.md`
- **Blender Integration:** Multi-agent Blender asset generator pipeline

---

## System Architecture

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    BLENDER ASSET GENERATION PIPELINE                     │
├─────────────────────────────────────────────────────────────────────────┤
│  1. script-generator MCP  →  Generate Blender Python scripts            │
│  2. blender-executor MCP  →  Run Blender CLI with fluid simulation      │
│  3. asset-evaluator MCP   →  LPIPS/CLIP quality evaluation              │
│  4. Export water_mesh.bin →  Raw binary: header + vertices + indices    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PLASMADX RT RENDERING PIPELINE                        │
├─────────────────────────────────────────────────────────────────────────┤
│  1. LoadWaterMesh()       →  Parse binary, create vertex/index buffers  │
│  2. SetWaterMesh()        →  Build triangle BLAS for water geometry     │
│  3. BuildCombinedTLAS()   →  Add water BLAS as Instance ID 3            │
│  4. RayQuery traversal    →  Trace rays, detect water hits              │
│  5. ShadeWater()          →  Fresnel + Snell + Beer-Lambert shading     │
└─────────────────────────────────────────────────────────────────────────┘
```

### TLAS Instance Structure

| Instance ID | Geometry Type | Purpose |
|-------------|--------------|---------|
| 0 | Procedural AABBs | Probe Grid particles (0-2044) |
| 1 | Procedural AABBs | Direct RT particles (overflow) |
| 2 | Triangles | Ground Plane (reflective surface) |
| 3 | Triangles | **Water Mesh (RT liquid)** ← PROBLEM HERE |

### Key Files

**C++ Implementation:**
- `src/core/Application.cpp` - LoadWaterMesh(), ImGui controls, constant buffer setup
- `src/lighting/RTLightingSystem_RayQuery.cpp` - SetWaterMesh(), BLAS build, TLAS integration
- `src/lighting/RTLightingSystem_RayQuery.h` - WaterMeshGeometry struct
- `src/particles/ParticleRenderer_Gaussian.h` - RenderConstants struct (enableWaterMesh field)

**Shaders:**
- `shaders/particles/particle_gaussian_raytrace.hlsl` - Main volumetric renderer, water hit detection
- `shaders/materials/water_material.hlsl` - Fresnel/refraction/absorption functions

**Asset Generation:**
- `scripts/generate_test_water_mesh.py` - Test mesh generator (bowl, sphere, plane, wavy)
- `scripts/export_water_mesh.py` - Blender export script for liquid simulations

---

## Current Behavior

### What Works
1. ✅ Water mesh files load successfully (`water_bowl.bin`, `water_sphere.bin`)
2. ✅ Vertex/index buffers created on GPU (logged: "Water GPU buffers created successfully")
3. ✅ Water BLAS created (logged: "Water mesh BLAS: size=75392 bytes")
4. ✅ `enableWaterMesh` constant reaches shader (cyan debug tint visible)
5. ✅ `SetWaterMeshEnabled(true)` called on RT system

### What Fails
1. ❌ `waterMeshHit` is never true in shader
2. ❌ No log message "Water mesh added to TLAS" (debug logging added but not seen)
3. ❌ Water geometry not visible despite debug tint showing constant is set

---

## Log Evidence

**Successful mesh loading:**
```
[20:51:45] [INFO] Loading water mesh from: assets/water/water_bowl.bin
[20:51:45] [INFO] Water mesh header: 544 vertices, 3072 indices (1024 triangles)
[20:51:45] [INFO] Water mesh Y range: [-50.00, -0.00]
[20:51:45] [INFO] Water vertex buffer created: 12 KB
[20:51:45] [INFO] Water index buffer created: 12 KB
[20:51:45] [INFO] Water GPU buffers created successfully
[20:51:45] [INFO] Setting water mesh: 544 vertices, 1024 triangles
[20:51:45] [INFO] Water mesh BLAS: size=75392 bytes, scratch=24320 bytes
[20:51:45] [INFO] Water mesh BLAS buffers created successfully
[20:51:45] [INFO] Water mesh connected to RT lighting system
[20:51:45] [INFO] Water mesh loaded successfully: 24 KB
[20:51:45] [INFO] Loaded water bowl mesh
```

**Missing log (should appear if TLAS includes water):**
```
[EXPECTED] Water mesh added to TLAS (InstanceID=3, vertices=544, triangles=1024)
```

---

## Hypothesis: Root Cause Analysis

### Hypothesis 1: TLAS Not Including Water Mesh (MOST LIKELY)

The `BuildCombinedTLAS()` function checks `m_waterMesh.enabled && m_waterMesh.blas` before adding water to TLAS. If this condition fails, water won't be in the TLAS even though BLAS exists.

**Verification needed:**
- Confirm `m_waterMesh.enabled` is `true` when `BuildCombinedTLAS()` is called
- The debug log "Water mesh added to TLAS" should appear but doesn't

**Code path:**
```cpp
// RTLightingSystem_RayQuery.cpp:1024
if (m_waterMesh.enabled && m_waterMesh.blas) {
    SetupInstance(instances[instanceCount], 3, m_waterMesh.blas.Get());
    instanceCount++;
    // Debug log added here - NOT appearing in logs
}
```

### Hypothesis 2: Water BLAS Not Built Before TLAS

The water BLAS is built in `SetWaterMesh()`, but if `BuildCombinedTLAS()` is called before `SetWaterMesh()` completes, the BLAS might not be ready.

**Verification needed:**
- Check call order: LoadWaterMesh → SetWaterMesh → next frame's BuildCombinedTLAS

### Hypothesis 3: Instance ID Mismatch

Shader expects `INSTANCE_ID_WATER = 3`, but if the instance array ordering changes dynamically (ground plane optional, direct RT AS optional), the water might get a different ID.

**Current instance setup logic:**
```cpp
if (m_probeGridAS.blas)        instanceCount++;  // ID 0
if (m_directRTAS.blas)         instanceCount++;  // ID 1 (if exists)
if (m_groundPlane.enabled)     instanceCount++;  // ID 2 (if enabled)
if (m_waterMesh.enabled)       instanceCount++;  // ID 3 (if enabled)
```

**Problem:** IDs are assigned sequentially based on `instanceCount`, NOT fixed values. If ground plane is disabled, water gets ID 2 instead of 3.

### Hypothesis 4: Mesh Geometry Position

Water mesh Y range is [-50, 0]. Camera starts at Y=1200, looking at origin. The water should be visible looking down, but:
- Particles might occlude the water
- Ray TMax might not reach the water

---

## Debug Visualization Added

**Shader debug code (particle_gaussian_raytrace.hlsl:1793-1798):**
```hlsl
// DEBUG: Show cyan tint if water is enabled (helps verify constant is set)
if (enableWaterMesh != 0 && !waterMeshHit) {
    backgroundColor += float3(0.0, 0.02, 0.03);
}
```

**Result:** Cyan tint IS visible → `enableWaterMesh` constant IS reaching shader
**Conclusion:** Problem is in TLAS or ray intersection, not constant buffer

---

## Recommended Debugging Steps

### Step 1: Verify TLAS Contents
Add logging to `BuildCombinedTLAS()` to show all instances being added:
```cpp
LOG_INFO("BuildCombinedTLAS: {} instances", instanceCount);
for (uint32_t i = 0; i < instanceCount; i++) {
    LOG_INFO("  Instance {}: ID={}", i, instances[i].InstanceID);
}
```

### Step 2: Verify Instance ID Assignment
The `SetupInstance` lambda uses the passed `id` parameter, but confirm it's actually being set:
```cpp
auto SetupInstance = [](D3D12_RAYTRACING_INSTANCE_DESC& inst, uint32_t id, ID3D12Resource* blas) {
    inst.InstanceID = id;  // <-- Verify this is 3 for water
    LOG_INFO("SetupInstance: InstanceID={}", id);
    ...
};
```

### Step 3: Add Shader Debug Output
Modify shader to output instance ID when ANY triangle is hit:
```hlsl
if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
    uint hitInstanceID = query.CommittedInstanceID();
    // Output to debug buffer or color-code by instance ID
}
```

### Step 4: PIX GPU Capture
1. Build with DebugPIX configuration
2. Capture frame after water mesh loads
3. Inspect TLAS in PIX:
   - Verify water BLAS is present
   - Check instance transform (should be identity)
   - Confirm instance mask is 0xFF

### Step 5: Test Ground Plane Behavior
Ground plane uses the same pattern (triangle geometry, ID 2). If ground plane works but water doesn't:
- Compare BLAS build parameters
- Check vertex format differences

---

## Compatibility Requirements

### Blender Asset Generator Integration

The water mesh system MUST be compatible with assets created by the multi-agent Blender pipeline:

**Asset format (raw binary):**
```
Header:     uint32 vertexCount, uint32 indexCount
Vertices:   float[vertexCount * 6]  // pos.xyz + normal.xyz interleaved
Indices:    uint32[indexCount]      // triangle indices
```

**MCP tools involved:**
- `mcp__script-generator__generate_script` - Creates Blender fluid simulation scripts
- `mcp__blender-executor__execute_blender_script` - Runs simulation and exports mesh
- `mcp__asset-evaluator__evaluate_render` - Quality validation

**Expected workflow:**
1. Agent generates fluid simulation script via `script-generator`
2. Agent executes simulation via `blender-executor`
3. Agent exports mesh frames to `assets/water_meshes/<scene>/<frame>.bin`
4. PlasmaDX loads mesh and renders with RT water shader
5. Agent evaluates render quality via `asset-evaluator`

---

## Files to Investigate

| File | Line Range | Purpose |
|------|------------|---------|
| `RTLightingSystem_RayQuery.cpp` | 1024-1033 | Water mesh TLAS addition |
| `RTLightingSystem_RayQuery.cpp` | 994-1003 | SetupInstance lambda |
| `RTLightingSystem_RayQuery.cpp` | 1578-1609 | BuildWaterBLAS function |
| `particle_gaussian_raytrace.hlsl` | 1293-1320 | Water hit detection |
| `particle_gaussian_raytrace.hlsl` | 1787-1798 | Water color output |
| `Application.cpp` | 7296-7350 | LoadWaterMesh function |
| `Application.cpp` | 1260-1272 | Constant buffer setup |

---

## Test Assets

**Available test meshes (in `build/bin/Debug/assets/water/`):**
- `water_bowl.bin` - 544 vertices, 1024 triangles, bowl shape
- `water_sphere.bin` - 544 vertices, 960 triangles, sphere shape
- `test_plane.bin` - 1089 vertices, 2048 triangles, flat grid
- `test_wavy.bin` - 4225 vertices, 8192 triangles, animated-style

**Generation command:**
```bash
python3 scripts/generate_test_water_mesh.py --type all
```

---

## Success Criteria

1. Water mesh visible in rendered output (not just cyan debug tint)
2. Log message "Water mesh added to TLAS" appears after loading
3. Fresnel reflection/refraction visible on water surface
4. Compatible with animated mesh sequences from Blender pipeline
5. Performance: <1ms overhead for water rendering at 1080p

---

## Related Issues

- Ground plane rendering (Instance ID 2) - WORKING, can use as reference
- NanoVDB volumetric system - Separate codepath, not affected
- Particle BLAS (procedural primitives) - Different geometry type, not comparable
