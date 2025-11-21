# Froxel System Critical Bug Fix Summary

## Overview
This document summarizes the critical fixes applied to the **PlasmaDX-Clean** volumetric renderer to resolve the "black froxel grid" and "useFroxelFog = 0" issues.

## 1. Resource Binding Fix (`src/rendering/FroxelSystem.cpp`)

**Issue:**
The `InjectDensity` compute shader was producing an empty density grid because `particleCount` was not being uploaded to the GPU constant buffer. The shader read `particleCount` as 0 (from the `lightCount` field which defaults to 0 in this context) and exited immediately.

**Fix:**
Updated `FroxelSystem::InjectDensity` to explicitly set the particle count in the shared constant buffer parameter slot before dispatch.

```cpp
// src/rendering/FroxelSystem.cpp

void FroxelSystem::InjectDensity(...)
{
    // ...
    
    // CRITICAL FIX: Reuse lightCount field for particleCount since they share the same offset (44)
    // The shader expects 'particleCount' at this offset.
    m_gridParams.lightCount = particleCount;
    memcpy(m_constantBufferMapped, &m_gridParams, sizeof(GridParams));

    // ...
}
```

## 2. Struct Alignment Fix (`shaders/particles/particle_gaussian_raytrace.hlsl`)

**Issue:**
The `GaussianConstants` constant buffer definition in HLSL did not match the C++ `RenderConstants` struct layout. Specifically, the `useFroxelFog` (uint) and `froxelGridMin` (float3) fields were swapped. This caused the shader to read part of the float vector as the boolean flag (resulting in `useFroxelFog` usually being read as 0) and the boolean flag as part of the grid bounds.

**Fix:**
Reordered the fields in the HLSL `GaussianConstants` cbuffer to strictly match the C++ alignment.

```hlsl
// shaders/particles/particle_gaussian_raytrace.hlsl

cbuffer GaussianConstants : register(b0)
{
    // ... (previous fields)

    // Froxel volumetric fog system (Phase 5 - replaces god rays)
    // MATCH C++ ORDER: useFroxelFog, gridMin, gridMax, padding0, dims, density, voxelSize, padding1
    
    uint useFroxelFog;             // Toggle: 0=disabled, 1=enabled (Offset 0 relative to block)
    float3 froxelGridMin;          // Grid world-space minimum (Offset 4)
    float3 froxelGridMax;          // Grid world-space maximum (Offset 16)
    float froxelPadding0;          // Padding for alignment (Offset 28)
    uint3 froxelGridDimensions;    // Voxel count (Offset 32)
    float froxelDensityMultiplier; // Fog density multiplier (Offset 44)
    float3 froxelVoxelSize;        // Computed voxel size (Offset 48)
    float froxelPadding1;          // Padding for alignment (Offset 60)

    // ... (subsequent fields)
};
```

## Impact
*   **Black Grid Resolved:** `InjectDensity` now correctly iterates over particles, populating the density grid.
*   **Flag Detection Resolved:** The main renderer now correctly reads `useFroxelFog` as true/1 when enabled.
*   **Visuals:** Volumetric fog should now render correctly based on the populated density and lighting grids.

