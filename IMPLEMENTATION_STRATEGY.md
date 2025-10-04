# PlasmaDX-Clean Implementation Strategy

## The Core Problem & Solution

### Problem
NVIDIA driver + Agility SDK has a bug where mesh shaders cannot read from descriptor tables (specifically SRV tables). This breaks the ability to read RT lighting data in mesh shaders.

### Solution: Pre-merge RT Lighting
Instead of having mesh shaders read from two buffers:
- ❌ `t0: ParticleBuffer` + `t1: RTLightingBuffer` (BROKEN)

We pre-merge the data:
- ✅ Compute shader merges RT lighting → Single buffer
- ✅ Mesh shader reads from single pre-merged buffer (WORKS)

## Implementation Pipeline

```
Frame Pipeline:
1. Physics Update (compute shader)
   ↓
2. RT Lighting Calculation (DXR)
   ↓
3. **MERGE STEP** (compute shader) ← THE FIX
   - Reads: Particles + RT Lighting
   - Writes: ParticlesWithLighting buffer
   ↓
4. Mesh Shader Rendering
   - Reads: ParticlesWithLighting (single buffer)
   - Works on buggy drivers!
```

## Automatic Fallback System

```cpp
if (FeatureDetector::CanUseMeshShaders()) {
    // Path A: Mesh shaders with pre-merge
    MergeLighting();      // Compute shader
    DispatchMesh();       // Mesh shader (reads single buffer)
} else {
    // Path B: Pure compute fallback
    BuildVertices();      // Compute shader
    DrawIndexedInstanced(); // Traditional VS/PS
}
```

## Key Files to Implement

### 1. Core Framework (Minimal)
- ✅ `main.cpp` - Entry point
- ✅ `Application.h/cpp` - Window only
- ✅ `FeatureDetector.h/cpp` - Test features

### 2. Shaders (Already Have)
- ✅ `particle_physics.hlsl` - Physics simulation
- ✅ `particle_mesh_fixed.hlsl` - Fixed mesh shader
- ✅ `merge_rt_lighting.hlsl` - The workaround

### 3. Next Steps (Priority Order)
1. `Device.cpp` - D3D12 initialization
2. `ResourceManager.cpp` - Buffer management
3. `ParticleRenderer.cpp` - Render paths
4. `RTLightingSystem.cpp` - RT setup

## Testing Strategy

### Test 1: Mesh Shader Path
1. Initialize with mesh shaders
2. Run merge compute shader
3. Render with `particle_mesh_fixed.hlsl`
4. Look for GREEN particles (RT lighting working)

### Test 2: Compute Fallback
1. Force compute path
2. Build vertices in compute shader
3. Render with traditional VS/PS
4. Verify particles render

### Test 3: Performance
- Target: 100K particles @ 60 FPS
- Mesh path: ~5ms frame time
- Compute fallback: ~8ms frame time

## Constants to Port

From original `ParticleSystem.h`:
```cpp
const float BLACK_HOLE_MASS = 4.3e6f;
const float INNER_STABLE_ORBIT = 10.0f;
const float OUTER_DISK_RADIUS = 1000.0f;
const float DISK_THICKNESS = 50.0f;
const float INITIAL_ANGULAR_MOMENTUM = 100.0f;
```

## Success Criteria

✅ **Working:** Particles render with RT lighting (green test)
✅ **Robust:** Automatic fallback on incompatible hardware
✅ **Clean:** No file >500 lines
✅ **Fast:** 60+ FPS with 100K particles

## The Bottom Line

We're working around a driver bug by pre-merging RT lighting data in a compute shader before mesh shaders access it. This adds one compute dispatch but ensures compatibility across all configurations.