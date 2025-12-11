# GPU Frustum Culling Implementation

**Date:** 2025-12-11
**Version:** 0.23.1
**Status:** ✅ COMPLETE

## Overview

GPU-side frustum culling for particle AABB generation. Particles outside the camera's view frustum are assigned degenerate AABBs (min > max) which DXR treats as zero-volume geometry, reducing BLAS build time and ray traversal cost.

## Technical Implementation

### Approach: Degenerate AABB Method

Instead of resizing buffers or using instance masking, culled particles output inverted AABBs:
```hlsl
// Degenerate AABB - DXR handles gracefully
AABBOutput degenerateAABB;
degenerateAABB.minX = 1.0; degenerateAABB.maxX = 0.0;  // min > max
degenerateAABB.minY = 1.0; degenerateAABB.maxY = 0.0;
degenerateAABB.minZ = 1.0; degenerateAABB.maxZ = 0.0;
```

**Why this approach:**
- No buffer resizing required
- No CPU-GPU synchronization
- Compatible with existing dual AS architecture
- Zero architectural changes to BLAS/TLAS pipeline

### Files Modified

| File | Changes |
|------|---------|
| `src/lighting/RTLightingSystem_RayQuery.h` | Added frustum culling settings, member variables, public API |
| `src/lighting/RTLightingSystem_RayQuery.cpp` | Frustum plane extraction function, expanded root signature (12→38 DWORDs), updated `ComputeLighting` and `GenerateAABBs_Dual` |
| `shaders/dxr/generate_particle_aabbs.hlsl` | Frustum culling logic with sphere-frustum test |
| `src/core/Application.cpp` | ImGui controls for runtime toggle, pass viewProjMatrix to RT system |

### Frustum Plane Extraction

Uses the Gribb/Hartmann method for DirectX left-handed, row-major matrices:

```cpp
// Reference: https://www.braynzarsoft.net/viewtutorial/q16390-34-aabb-cpu-side-frustum-culling
// XMFLOAT4X4 uses _RC naming: _14 = row 1, column 4

// Left plane: col4 + col1
planes[0] = XMFLOAT4(m._14 + m._11, m._24 + m._21, m._34 + m._31, m._44 + m._41);

// Right plane: col4 - col1
planes[1] = XMFLOAT4(m._14 - m._11, m._24 - m._21, m._34 - m._31, m._44 - m._41);

// Bottom plane: col4 + col2
planes[2] = XMFLOAT4(m._14 + m._12, m._24 + m._22, m._34 + m._32, m._44 + m._42);

// Top plane: col4 - col2
planes[3] = XMFLOAT4(m._14 - m._12, m._24 - m._22, m._34 - m._32, m._44 - m._42);

// Near plane: col3
planes[4] = XMFLOAT4(m._13, m._23, m._33, m._43);

// Far plane: col4 - col3
planes[5] = XMFLOAT4(m._14 - m._13, m._24 - m._23, m._34 - m._33, m._44 - m._43);
```

### Shader Constant Buffer

Expanded from 12 to 38 DWORDs:

```hlsl
cbuffer AABBConstants : register(b0)
{
    // Original constants (12 DWORDs)
    uint particleCount;
    float particleRadius;
    uint particleOffset;
    uint padding1;
    uint enableAdaptiveRadius;
    float adaptiveInnerZone;
    float adaptiveOuterZone;
    float adaptiveInnerScale;
    float adaptiveOuterScale;
    float densityScaleMin;
    float densityScaleMax;
    float padding2;

    // Frustum Culling (26 DWORDs)
    float4 frustumPlanes[6];    // 24 DWORDs - Left, Right, Bottom, Top, Near, Far
    uint enableFrustumCulling;  // 1 DWORD
    float frustumExpansion;     // 1 DWORD
};
```

### Sphere-Frustum Test

Conservative test using particle's bounding sphere:

```hlsl
bool SphereOutsidePlane(float3 center, float radius, float4 plane)
{
    float dist = dot(center, plane.xyz) + plane.w;
    return dist < -radius;  // Completely behind plane
}

bool SphereOutsideFrustum(float3 center, float radius)
{
    float expandedRadius = radius * frustumExpansion;  // 1.5x default

    [unroll]
    for (uint i = 0; i < 6; i++)
    {
        if (SphereOutsidePlane(center, expandedRadius, frustumPlanes[i]))
            return true;  // Outside this plane = outside frustum
    }
    return false;  // Inside all planes = visible
}
```

## Configuration

### Runtime Controls (ImGui)

Located in **"GPU Frustum Culling (Optimization)"** section:

| Control | Default | Description |
|---------|---------|-------------|
| Enable Frustum Culling | ✅ ON | Toggle culling on/off |
| Frustum Expansion | 1.5x | Expand frustum to prevent pop-in |

### Programmatic API

```cpp
// Enable/disable
m_rtLighting->SetFrustumCullingEnabled(true);
m_rtLighting->SetFrustumCullingEnabled(false);

// Query state
bool enabled = m_rtLighting->IsFrustumCullingEnabled();

// Adjust expansion factor (1.0 = exact, 1.5 = conservative, 2.0+ = very conservative)
m_rtLighting->SetFrustumExpansion(1.5f);
```

## Performance Characteristics

### Expected Benefits

| Scenario | Benefit |
|----------|---------|
| Camera zoomed in (close to disk) | **HIGH** - Many particles outside frustum |
| Camera at default distance | **MODERATE** - Some edge particles culled |
| Camera zoomed out (full disk visible) | **LOW** - Most particles in frustum anyway |

### Overhead

- **GPU:** ~0.1ms additional compute in AABB generation shader
- **CPU:** Zero (frustum planes extracted once per frame, passed via constants)
- **Memory:** 104 bytes additional constant buffer data (26 DWORDs × 4 bytes)

### Benchmarking

1. Launch application from `build/bin/Debug/`
2. Note FPS with culling **ON** (default)
3. Press F2 to capture screenshot with metadata
4. Toggle **OFF** via ImGui checkbox
5. Note FPS with culling **OFF**
6. Press F2 again for comparison screenshot
7. Run: `python3 optimization/frustum_culling_benchmark.py`

## Debugging

### Visual Verification

If particles appear to be incorrectly culled (sharp edges in particle cloud):

1. **Check frustum expansion** - Increase to 2.0x or higher
2. **Disable culling temporarily** - Use ImGui toggle to verify it's the cause
3. **Check camera matrices** - Ensure `viewProjMatrix` is passed correctly

### Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Half of disk missing | Wrong plane extraction formula | Fixed in commit - ensure using column extraction pattern |
| Particles pop in at edges | Expansion factor too low | Increase frustum expansion to 2.0x |
| No performance improvement | Camera showing all particles | Zoom in to see benefit |
| Black screen | All particles culled | Check frustum plane signs, disable culling to verify |

## References

- [Braynzar Soft: AABB CPU Side Frustum Culling](https://www.braynzarsoft.net/viewtutorial/q16390-34-aabb-cpu-side-frustum-culling)
- [Gribb/Hartmann: Fast Extraction of Viewing Frustum Planes](https://www.gamedevs.org/uploads/fast-extraction-viewing-frustum-planes-from-world-view-projection-matrix.pdf)
- [GameDev.net: Frustum Planes Extraction](https://gamedev.net/forums/topic/268212-frustum-planes-extraction-i-just-dont-get-it/)

## Commit Information

- **Branch:** 0.23.1
- **Files changed:** 4
- **Lines added:** ~150
- **Lines modified:** ~50
