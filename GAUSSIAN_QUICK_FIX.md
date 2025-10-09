# Gaussian Renderer Quick Fix (10 minutes)

## Problem
PSO creation fails because root signature/constants don't match shader.

**Shader Expects** (`particle_gaussian_raytrace.hlsl`):
```hlsl
cbuffer CameraConstants : register(b0) {
    row_major float4x4 viewProj;
    row_major float4x4 invViewProj;
    float3 cameraPos;
    float padding0;
    float3 cameraRight;
    float padding1;
    float3 cameraUp;
    float padding2;
    float2 resolution;
    float2 invResolution;
}

cbuffer GaussianConstants : register(b1) {
    float baseParticleRadius;
    uint maxIntersectionsPerRay;
    float volumeStepSize;
    float densityMultiplier;
}

StructuredBuffer<Particle> g_particles : register(t0);
StructuredBuffer<float4> g_rtLighting : register(t1);
RaytracingAccelerationStructure g_particleBVH : register(t2);
RWTexture2D<float4> g_output : register(u0);
```

**Current Root Signature** (Wrong):
```cpp
rootParams[0].InitAsConstants(48, 0);              // Too big, wrong layout
rootParams[1].InitAsShaderResourceView(0);         // Correct
rootParams[2].InitAsShaderResourceView(1);         // Correct
rootParams[3].InitAsShaderResourceView(2);         // Correct
rootParams[4].InitAsUnorderedAccessView(0);        // Correct
```

## Fix

### Option 1: Simplify Shader (EASIEST - 5 minutes)

Change shader to match our C++ structure:

```hlsl
cbuffer GaussianConstants : register(b0) {
    row_major float4x4 viewProj;
    row_major float4x4 invViewProj;
    float3 cameraPos;
    float particleRadius;
    float3 cameraRight;
    float time;
    float3 cameraUp;
    uint screenWidth;
    float3 cameraForward;
    uint screenHeight;
    float fovY;
    float aspectRatio;
    uint particleCount;
    float padding;

    uint usePhysicalEmission;
    float emissionStrength;
    uint useDopplerShift;
    float dopplerStrength;
    uint useGravitationalRedshift;
    float redshiftStrength;
    float2 padding2;
}

// Remove second cbuffer, use constants from above
```

Then recompile:
```bash
dxc -T cs_6_5 -E main particle_gaussian_raytrace.hlsl -Fo particle_gaussian_raytrace.dxil
```

### Option 2: Fix C++ to Match Shader (HARDER - 15 minutes)

Update `ParticleRenderer_Gaussian.cpp` CreatePipeline():

```cpp
// Root signature with 2 cbuffers
rootParams[0].InitAsConstants(sizeof(CameraConstants)/4, 0);      // b0: 22 DWORDs
rootParams[1].InitAsConstants(sizeof(GaussianConstants)/4, 1);    // b1: 4 DWORDs
rootParams[2].InitAsShaderResourceView(0);                        // t0
rootParams[3].InitAsShaderResourceView(1);                        // t1
rootParams[4].InitAsShaderResourceView(2);                        // t2
rootParams[5].InitAsUnorderedAccessView(0);                       // u0

// 6 root params now!
```

And update Render() to set both cbuffers separately.

## Recommendation

**Do Option 1** - Simplify the shader to match what we're already passing. The current RenderConstants structure has everything we need. Just remove the split cbuffer design from the shader.

## After Fix

Once shader matches C++ structure:
1. Recompile shader
2. Rebuild C++
3. Launch with `--gaussian --particles 20000`
4. Should initialize successfully
5. Add UAV→backbuffer copy (10 lines from SESSION_COMPLETE.md)
6. See beautiful volumetric rendering!

## Current Status

✅ Billboard renderer: Perfect, stable, beautiful physics
❌ Gaussian renderer: PSO creation fails (cbuffer mismatch)

**Estimated fix time**: 5-10 minutes
