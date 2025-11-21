# Deep Analysis Request: Froxel Volumetric Fog System Critical Bug

## Context: DirectX 12 Volumetric Particle Renderer

You are analyzing a DirectX 12 real-time volumetric particle renderer called **PlasmaDX-Clean**. The engine uses:
- **DXR 1.1 inline ray tracing** (RayQuery API) for volumetric lighting
- **3D Gaussian splatting** for volumetric rendering (not 2D billboard splatting)
- **Multi-light system** with 13 lights positioned in a dome around particles
- **Compute shaders** for physics and lighting
- **HLSL Shader Model 6.5+** with DXR capabilities

The user (Ben) is implementing a **Froxel Volumetric Fog System** to replace an older "God Rays" system that was too slow. Froxels (frustum-aligned voxels) pre-compute lighting in a 3D texture grid (160×90×64 = 921,600 voxels) to decouple expensive lighting calculations from per-pixel rendering.

---

## Froxel System Architecture (3-Pass Pipeline)

### Pass 1: ClearGrid
- Clears density and lighting grids to zero (UAV clear operation)

### Pass 2: InjectDensity (compute shader)
- **Input:** Particle buffer (10,000 particles with positions, temperatures, velocities)
- **Output:** 3D density grid (R16_FLOAT texture, 1.8 MB)
- **Algorithm:** Trilinear splatting - each particle contributes density to 8 surrounding voxels
- **Dispatch:** 40 thread groups × 256 threads = 10,240 threads (one per particle)

### Pass 3: LightVoxels (compute shader) **← THIS IS WHERE THE BUG IS**
- **Input:**
  - Density grid (SRV from Pass 2)
  - Light buffer (13 lights, each 64 bytes)
  - Particle BVH (acceleration structure for shadow rays)
- **Output:** 3D lighting grid (R16G16B16A16_FLOAT texture, 14.4 MB)
- **Algorithm:**
  - For each voxel with density > 0.001:
    - Loop over all 13 lights
    - Calculate distance attenuation (quadratic falloff)
    - **Shadow rays disabled** (performance optimization - see note below)
    - Accumulate `light.color × light.intensity × attenuation`
  - Store accumulated lighting in RGB, density in Alpha
- **Dispatch:** 20×12×8 thread groups × 512 threads (8×8×8 per group) = 983,040 threads

### Pass 4: Sample Grid (in main Gaussian renderer)
- **Input:** Lighting grid (SRV), grid parameters (min, max, dimensions, voxel size)
- **Algorithm:**
  - During volumetric ray marching, sample froxel grid with trilinear interpolation
  - Apply fog lighting to particle rendering
- **Expected:** Multi-colored volumetric fog from all 13 lights

---

## THE CRITICAL BUG

### Symptoms:
1. **Visual:** Bright red cloud with single yellow ellipsoid (Light 0 position)
2. **Performance:** Under 10 FPS (expected 100+ FPS without shadow rays, 30-40 FPS with shadows)
3. **PIX Analysis:**
   - `g_froxelLightingGrid` texture is **completely black** (all zeros)
   - `useFroxelFog = 0` in captured `GaussianConstants` buffer (despite being enabled in UI)
   - 450M hardware rays traced (not directly related to froxel system)
4. **Only Light 0 works:** Moving Light 0 in runtime controls moves the yellow ellipsoid, other 12 lights have zero contribution
5. **BVH bounds enlarged:** After recent changes, particle bounding volumes became huge (possibly related?)

### What We've Tried:
1. ✅ **Fixed GridParams struct mismatch** - Changed `uint32_t particleCount` → `uint32_t lightCount`
2. ✅ **Upload lightCount to GPU** - Added `m_gridParams.lightCount = lightCount` before dispatch
3. ✅ **Recompiled shaders** - Forced recompilation of all froxel HLSL shaders
4. ✅ **Disabled shadow rays** - Commented out RayQuery shadow tests (was killing performance with 12M rays/frame)
5. ✅ **Verified CPU logs** - Log shows "FROXEL: Lighting 921600 voxels with 13 lights" every frame
6. ❌ **No visual change** - Red cloud persists, froxel grid stays black, only Light 0 visible

### Root Cause Hypothesis:
The shader is **either not executing** or **not writing to the output texture**. Possible causes:
- Resource binding mismatch (UAV not bound correctly)
- Resource state transition missing (UAV barrier or state transition)
- Shader early-out condition (all voxels have density < 0.001?)
- Output texture format mismatch or write failure
- Thread group size calculation error (dispatching zero groups?)

---

## Key Code Files to Analyze

### 1. C++ Host Code

#### `src/rendering/FroxelSystem.h` (GridParams struct)
```cpp
struct GridParams {
    DirectX::XMFLOAT3 gridMin;      // World-space minimum [-1500, -1500, -1500]
    float padding0;
    DirectX::XMFLOAT3 gridMax;      // World-space maximum [1500, 1500, 1500]
    float padding1;
    DirectX::XMUINT3 gridDimensions; // Voxel count [160, 90, 64]
    uint32_t lightCount;            // Number of lights (CRITICAL - was missing!)
    DirectX::XMFLOAT3 voxelSize;    // Computed size of each voxel
    float lightingMultiplier;       // Global lighting scale (replaces densityMultiplier)
};
```

#### `src/rendering/FroxelSystem.cpp` (LightVoxels dispatch)
```cpp
void FroxelSystem::LightVoxels(
    ID3D12GraphicsCommandList* commandList,
    ID3D12Resource* particleBuffer,
    uint32_t particleCount,
    ID3D12Resource* lightBuffer,
    uint32_t lightCount,
    ID3D12Resource* particleBVH)
{
    // CRITICAL FIX: Update lightCount in constant buffer BEFORE dispatch!
    m_gridParams.lightCount = lightCount;
    memcpy(m_constantBufferMapped, &m_gridParams, sizeof(GridParams));

    // Set pipeline state and root signature
    commandList->SetPipelineState(m_lightVoxelsPSO.Get());
    commandList->SetComputeRootSignature(m_lightVoxelsRootSig.Get());

    // Root param 0: Constant buffer (b0)
    commandList->SetComputeRootConstantBufferView(0, m_constantBuffer->GetGPUVirtualAddress());

    // Root param 1: Density grid SRV (t0)
    commandList->SetComputeRootDescriptorTable(1, m_densityGridSRVGPU);

    // Root param 2: Light buffer SRV (t1) - use root descriptor (no heap allocation!)
    commandList->SetComputeRootShaderResourceView(2, lightBuffer->GetGPUVirtualAddress());

    // Root param 3: Particle BVH (t2) - acceleration structure (root descriptor)
    commandList->SetComputeRootShaderResourceView(3, particleBVH->GetGPUVirtualAddress());

    // Root param 4: Lighting grid UAV (u0)
    commandList->SetComputeRootDescriptorTable(4, m_lightingGridUAVGPU);

    // Thread groups: Dispatch 8×8×8 thread groups to cover entire grid
    uint32_t groupsX = (m_gridParams.gridDimensions.x + 7) / 8;  // 160 → 20
    uint32_t groupsY = (m_gridParams.gridDimensions.y + 7) / 8;  // 90 → 12
    uint32_t groupsZ = (m_gridParams.gridDimensions.z + 7) / 8;  // 64 → 8

    LOG_INFO("FROXEL: Lighting {} voxels with {} lights ({}×{}×{} thread groups)",
              m_gridParams.gridDimensions.x * m_gridParams.gridDimensions.y * m_gridParams.gridDimensions.z,
              lightCount,
              groupsX, groupsY, groupsZ);

    commandList->Dispatch(groupsX, groupsY, groupsZ);  // 20×12×8 = 1920 thread groups

    LOG_INFO("FROXEL: Voxel lighting dispatch complete");
}
```

#### Root Signature (5 root parameters)
```cpp
// Root param 0: b0 (constant buffer - CBV)
// Root param 1: t0 (density grid - descriptor table with 1 SRV)
// Root param 2: t1 (light buffer - root descriptor SRV)
// Root param 3: t2 (particle BVH - root descriptor SRV for acceleration structure)
// Root param 4: u0 (lighting grid UAV - descriptor table)
```

### 2. HLSL Shader Code

#### `shaders/froxel/light_voxels.hlsl` (Compute Shader)
```hlsl
// Light structure (64 bytes, matches C++)
struct Light {
    float3 position;
    float intensity;
    float3 color;
    float radius;
    float enableGodRays;
    float godRayIntensity;
    float godRayLength;
    float godRayFalloff;
    float3 godRayDirection;
    float godRayConeAngle;
    float godRayRotationSpeed;
    float _padding;
};

// Froxel grid parameters
cbuffer FroxelParams : register(b0)
{
    float3 gridMin;             // [-1500, -1500, -1500]
    float padding0;
    float3 gridMax;             // [1500, 1500, 1500]
    float padding1;
    uint3 gridDimensions;       // [160, 90, 64]
    uint lightCount;            // 13 (CRITICAL - this was zero before fix!)
    float3 voxelSize;           // ~18.75 units per voxel
    float lightingMultiplier;   // 1.0 (global scale)
};

// Input: Density grid from Pass 1
Texture3D<float> g_densityGrid : register(t0);

// Input: Light buffer
StructuredBuffer<Light> g_lights : register(t1);

// Input: Particle BVH for shadow rays (currently disabled)
RaytracingAccelerationStructure g_particleBVH : register(t2);

// Output: 3D lighting grid (R16G16B16A16_FLOAT)
// RGB = accumulated light color, A = density (for sampling)
RWTexture3D<float4> g_lightingGrid : register(u0);

//------------------------------------------------------------------------------
// Voxel Lighting Compute Shader
// Thread group: 8×8×8 = 512 voxels per group
//------------------------------------------------------------------------------
[numthreads(8, 8, 8)]
void main(uint3 voxelIdx : SV_DispatchThreadID)
{
    // Bounds check
    if (any(voxelIdx >= gridDimensions))
        return;

    // Sample density at this voxel (from Pass 1 density injection)
    float density = g_densityGrid[voxelIdx];

    // Early out if empty voxel (no particles here)
    if (density < 0.001) {
        g_lightingGrid[voxelIdx] = float4(0, 0, 0, 0);
        return;
    }

    // Convert voxel index to world position (center of voxel)
    float3 worldPos = gridMin + (float3(voxelIdx) + 0.5) * voxelSize;

    // === ACCUMULATE LIGHTING FROM ALL LIGHTS ===
    float3 totalLight = float3(0, 0, 0);

    for (uint lightIdx = 0; lightIdx < lightCount; lightIdx++) {
        Light light = g_lights[lightIdx];

        // Calculate light direction and distance
        float3 toLight = light.position - worldPos;
        float lightDist = length(toLight);
        float3 lightDir = toLight / max(lightDist, 0.001);

        // Distance attenuation (quadratic falloff with light radius)
        float normalizedDist = lightDist / max(light.radius, 1.0);
        float attenuation = 1.0 / (1.0 + normalizedDist * normalizedDist);

        // Shadow rays DISABLED (performance optimization)
        float shadowTerm = 1.0;  // Fully lit (no shadows)

        // Accumulate this light's contribution
        float3 lightContribution = light.color * light.intensity * attenuation * shadowTerm;
        totalLight += lightContribution;
    }

    // Apply global lighting multiplier
    totalLight *= lightingMultiplier;

    // Store in lighting grid
    // RGB = accumulated light color (pre-multiplied by density)
    // A = density (for sampling in Pass 3 - determines fog visibility)
    g_lightingGrid[voxelIdx] = float4(totalLight * density, density);
}
```

### 3. Resource Transitions and Barriers

#### `src/core/Application.cpp` (Render loop integration)
```cpp
// === FROXEL VOLUMETRIC FOG SYSTEM (Phase 5) ===
if (m_enableFroxelFog && m_froxelSystem) {
    // Pass 1: Clear grids
    m_froxelSystem->ClearGrid(cmdList);

    // Pass 2: Inject density (particles → density grid)
    m_froxelSystem->InjectDensity(
        cmdList,
        m_particleSystem->GetParticleBuffer(),
        m_particleSystem->GetParticleCount()
    );

    // Barrier: UAV → SRV (density grid must finish writing before lighting reads)
    CD3DX12_RESOURCE_BARRIER densityBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        m_froxelSystem->GetDensityGrid(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );
    cmdList->ResourceBarrier(1, &densityBarrier);

    // Pass 3: Light voxels (density grid + lights → lighting grid)
    ID3D12Resource* particleBVH = m_rtLighting ? m_rtLighting->GetBVH() : nullptr;
    m_froxelSystem->LightVoxels(
        cmdList,
        m_particleSystem->GetParticleBuffer(),
        m_particleSystem->GetParticleCount(),
        m_lightSystem->GetLightBuffer(),
        m_lightSystem->GetActiveLightCount(),
        particleBVH
    );

    // Barrier: UAV → SRV (lighting grid must finish before sampling)
    CD3DX12_RESOURCE_BARRIER lightingBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        m_froxelSystem->GetLightingGrid(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );
    cmdList->ResourceBarrier(1, &lightingBarrier);

    LOG_INFO("Froxel volumetric fog computed (frame {}, {} particles, {} voxels, {} lights)",
             m_frameCount,
             m_particleSystem->GetParticleCount(),
             160 * 90 * 64,
             m_lightSystem->GetActiveLightCount());
}

// === GAUSSIAN VOLUMETRIC RENDERER ===
// ... later in render loop ...
m_gaussianRenderer->Render(
    cmdList,
    m_particleSystem->GetParticleBuffer(),
    rtLightingBuffer,
    m_rtLighting->GetTLAS(),
    gaussianConstants,
    rtxdiOutput,
    m_probeGridSystem.get(),
    m_particleSystem->GetMaterialPropertiesBuffer(),
    m_froxelSystem.get()  // ← Froxel system passed here
);
```

### 4. Gaussian Renderer Integration

#### `src/particles/ParticleRenderer_Gaussian.cpp` (Froxel grid binding)
```cpp
// Root param 15: Bind froxel lighting grid (SRV descriptor table - t10, Phase 5)
if (froxelSystem) {
    D3D12_GPU_DESCRIPTOR_HANDLE froxelSRVHandle = froxelSystem->GetLightingGridSRV();
    if (froxelSRVHandle.ptr != 0) {
        cmdList->SetComputeRootDescriptorTable(15, froxelSRVHandle);
    } else {
        LOG_WARN("Froxel system provided but lighting grid SRV is ZERO!");
    }
}
```

#### `shaders/particles/particle_gaussian_raytrace.hlsl` (Sampling code)
```hlsl
// Froxel resources
Texture3D<float4> g_froxelLightingGrid : register(t10);
SamplerState g_linearClampSampler : register(s0);

// Froxel fog parameters (in constant buffer)
float3 froxelGridMin;          // Grid world-space minimum
uint useFroxelFog;             // Toggle: 0=disabled, 1=enabled
float3 froxelGridMax;          // Grid world-space maximum
float froxelDensityMultiplier; // Fog density multiplier (0.1-5.0)
uint3 froxelGridDimensions;    // Voxel count [160, 90, 64]
float froxelPadding0;
float3 froxelVoxelSize;        // Computed voxel size
float froxelPadding1;

// Sampling function (included from sample_froxel_grid.hlsl)
float3 SampleFroxelGrid(float3 worldPos)
{
    if (useFroxelFog == 0)
        return float3(0, 0, 0);

    // Convert world position to UVW coordinates [0, 1]
    float3 uvw = (worldPos - froxelGridMin) / (froxelGridMax - froxelGridMin);

    // Clamp to valid range
    uvw = saturate(uvw);

    // Sample lighting grid with trilinear interpolation
    float4 froxelData = g_froxelLightingGrid.SampleLevel(g_linearClampSampler, uvw, 0);

    // RGB = lighting, A = density
    return froxelData.rgb * froxelDensityMultiplier;
}
```

---

## Captured Data for Analysis

### PIX Capture: `PIX/Captures/froxel_fog_1.wpix`
- Full GPU capture with froxel fog enabled
- Shows all dispatches, resource bindings, and transitions
- **Key observation:** `g_froxelLightingGrid` is completely black after `LightVoxels()` dispatch

### Constant Buffer: `PIX/buffer_dumps/GaussianConstants.bin`
- 768-byte constant buffer captured from PIX
- **Key observation:** `useFroxelFog = 0` despite being enabled in UI
- Hexdump provided above (offsets 0x000-0x280)

### Lighting Grid: `PIX/buffer_dumps/g_froxelLightingGrid.hdr`
- 14.4 MB HDR texture (R16G16B16A16_FLOAT, 160×90×64 voxels)
- **Expected:** Multi-colored lighting data from 13 lights
- **Actual:** Completely black (all zeros)

### Log File: `build/bin/Debug/logs/PlasmaDX-Clean_20251121_175502.log`
- Shows "FROXEL: Lighting 921600 voxels with 13 lights" every frame
- No errors or warnings related to froxel system
- Performance: Under 10 FPS

---

## Specific Questions for Analysis

1. **Resource Binding:** Is the UAV for `g_lightingGrid` (u0) bound correctly? Check root signature, descriptor table setup, and descriptor heap allocation.

2. **Struct Alignment:** Does the HLSL `FroxelParams` cbuffer **exactly** match the C++ `GridParams` struct byte-for-byte? Check padding, field order, and alignment.

3. **Resource States:** Are the resource state transitions correct? Should there be a UAV barrier between `InjectDensity()` and `LightVoxels()` since both write to UAVs?

4. **Thread Group Calculation:** Is `Dispatch(20, 12, 8)` correct for a 160×90×64 grid with 8×8×8 threads per group? Are we dispatching enough threads?

5. **Early-Out Condition:** Is the density grid actually populated? Could all voxels have `density < 0.001`, causing the shader to early-out and write zeros?

6. **Light Buffer Binding:** Is the light buffer at t1 bound correctly as a root descriptor? Does the shader see valid light data (positions, colors, intensities)?

7. **Constant Buffer Upload:** Is `memcpy()` uploading the full 48-byte struct correctly? Is the constant buffer persistently mapped with correct CPU/GPU sync?

8. **Write Mask:** Could there be a write mask or blend state preventing writes to the UAV? (Unlikely for compute, but check PSO creation)

9. **BVH Issue:** You mentioned BVH bounds became huge - could this be corrupting memory or causing the froxel system to fail? Is there memory overlap?

10. **Performance Cliff:** Why did performance drop to under 10 FPS? Shadow rays are disabled (no longer tracing 12M rays). What's the new bottleneck?

11. **`useFroxelFog = 0` in PIX:** Why is the Gaussian renderer seeing `useFroxelFog = 0` when it's enabled in the UI? Is the constant buffer upload path broken?

---

## Expected Behavior (When Fixed)

- **Performance:** 100+ FPS without shadows, 30-40 FPS with optimized shadows
- **Visual:** Multi-colored volumetric fog sphere with all 13 lights contributing (yellows, oranges, reds, blues blending)
- **Froxel Grid:** `g_froxelLightingGrid` should show bright colors in PIX, not all black
- **Gaussian Constants:** `useFroxelFog = 1` should be visible in captured constant buffer

---

## Request

Please perform a **deep, systematic analysis** of this implementation. Focus on:
1. **Root cause** of the all-black lighting grid
2. **Root cause** of `useFroxelFog = 0` in captured constant buffer
3. **Any struct misalignment** between C++ and HLSL
4. **Any resource binding errors** or missing barriers
5. **Performance bottleneck** causing <10 FPS

Provide specific file locations, line numbers, and code snippets for any issues found. If you need additional code files or PIX capture details, please specify exactly what you need.

Thank you for your thorough analysis! 🙏
