# Balanced GI Implementation Plan (99 FPS Target)

**Project**: PlasmaDX-Clean
**Feature**: Single-Bounce Indirect Global Illumination
**Author**: Implementation plan generated 2025-10-25
**Status**: Ready for implementation

---

## Overview

**Goal**: Add single-bounce indirect lighting to existing RT system
**Timeline**: 1-2 days implementation + 1 day testing
**Performance Target**: 99 FPS @ 1080p, 10K particles
**Performance Cost**: -21 FPS from current 120 FPS baseline
**Risk Level**: LOW - Non-invasive addition to existing system

---

## What This Adds

### Visual Improvements
- **Color bleeding** - Hot particles cast red/orange tint on neighbors
- **Soft ambient lighting** - Shadowed regions receive indirect bounce light
- **Volumetric coherence** - More realistic scattering throughout disk
- **Depth perception** - Better sense of 3D structure from indirect cues

### Technical Architecture
- **Reuses existing TLAS** - No duplicate acceleration structures
- **Separate compute pass** - Runs after direct lighting
- **Diffuse BRDF** - Physically correct Lambertian reflectance
- **8 rays/particle** - Half the count of direct lighting (16 rays)

---

## Implementation Steps

### Step 1: Create Indirect Lighting Shader (2-3 hours)

**New file**: `shaders/dxr/particle_indirect_lighting_cs.hlsl`

```hlsl
// Particle-to-Particle Indirect Lighting (First Bounce GI)
// Traces hemisphere rays and samples DIRECT lighting from hit particles
// Applies diffuse BRDF for physically correct indirect reflection

cbuffer IndirectLightingConstants : register(b0)
{
    uint particleCount;
    uint raysPerParticle;      // 8 for balanced quality
    float maxLightingDistance;  // Same as direct (100.0)
    float indirectIntensity;    // Global indirect multiplier (0.3-0.5)
};

struct Particle
{
    float3 position;
    float temperature;
    float3 velocity;
    float density;
};

StructuredBuffer<Particle> g_particles : register(t0);
RaytracingAccelerationStructure g_particleBVH : register(t1);

// INPUT: Direct lighting from previous pass
StructuredBuffer<float4> g_directLighting : register(t2);

// OUTPUT: Indirect lighting contribution
RWStructuredBuffer<float4> g_indirectLighting : register(u0);

// Fibonacci hemisphere sampling (same as direct lighting)
float3 FibonacciHemisphere(uint sampleIndex, uint numSamples, float3 normal)
{
    const float PHI = 1.618033988749895;
    float theta = 2.0 * 3.14159265359 * sampleIndex / PHI;
    float phi = acos(1.0 - 2.0 * (sampleIndex + 0.5) / numSamples);

    float sinPhi = sin(phi);
    float x = cos(theta) * sinPhi;
    float y = sin(theta) * sinPhi;
    float z = cos(phi);

    return normalize(float3(x, y, z));
}

// Diffuse BRDF: Lambertian reflectance
// Returns: albedo * (N·L) / π
float3 DiffuseBRDF(float3 normal, float3 lightDir, float3 albedo)
{
    float NdotL = max(0.0, dot(normal, lightDir));
    return albedo * NdotL / 3.14159265359;
}

// Calculate particle albedo from temperature (color it reflects)
float3 ParticleAlbedo(float temperature)
{
    // Hot particles reflect less (emit more)
    // Cool particles reflect more (emit less)
    float t = saturate((temperature - 800.0) / 25200.0);

    // Inverse relationship: hotter = darker albedo
    float reflectivity = 1.0 - (t * 0.7); // 30% min reflectivity

    // Slightly warm tint for reflected light
    return float3(reflectivity * 0.9, reflectivity * 0.8, reflectivity * 0.7);
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint particleIdx = dispatchThreadID.x;

    if (particleIdx >= particleCount)
        return;

    Particle receiver = g_particles[particleIdx];
    float3 receiverPos = receiver.position;
    float3 receiverNormal = float3(0, 1, 0); // Simplified (could use velocity for anisotropy)
    float3 receiverAlbedo = ParticleAlbedo(receiver.temperature);

    float3 accumulatedIndirect = float3(0, 0, 0);

    // Cast indirect rays (HALF the count of direct for performance)
    for (uint rayIdx = 0; rayIdx < raysPerParticle; rayIdx++)
    {
        float3 rayDir = FibonacciHemisphere(rayIdx, raysPerParticle, receiverNormal);

        RayDesc ray;
        ray.Origin = receiverPos + rayDir * 0.01;
        ray.Direction = rayDir;
        ray.TMin = 0.001;
        ray.TMax = maxLightingDistance;

        RayQuery<RAY_FLAG_NONE> query;
        query.TraceRayInline(g_particleBVH, RAY_FLAG_NONE, 0xFF, ray);

        while (query.Proceed())
        {
            if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
            {
                uint candidateIdx = query.CandidatePrimitiveIndex();

                if (candidateIdx == particleIdx)
                    continue;

                Particle candidate = g_particles[candidateIdx];

                // Sphere-ray intersection (same as direct)
                const float rtLightingRadius = 5.0;
                float3 oc = ray.Origin - candidate.position;
                float b = dot(oc, ray.Direction);
                float c = dot(oc, oc) - (rtLightingRadius * rtLightingRadius);
                float discriminant = b * b - c;

                if (discriminant >= 0.0)
                {
                    float t = -b - sqrt(discriminant);
                    if (t >= ray.TMin && t <= ray.TMax)
                    {
                        query.CommitProceduralPrimitiveHit(t);
                    }
                }
            }
        }

        if (query.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT)
        {
            uint hitParticleIdx = query.CommittedPrimitiveIndex();

            if (hitParticleIdx == particleIdx)
                continue;

            // KEY DIFFERENCE: Read DIRECT lighting from the hit particle
            float3 hitDirectLight = g_directLighting[hitParticleIdx].rgb;

            // Calculate indirect contribution using diffuse BRDF
            float3 lightDir = normalize(g_particles[hitParticleIdx].position - receiverPos);
            float3 brdf = DiffuseBRDF(receiverNormal, lightDir, receiverAlbedo);

            // Distance attenuation (same as direct)
            float distance = query.CommittedRayT();
            float attenuation = 1.0 / (1.0 + distance * 0.01);

            // Accumulate: hitDirectLight * BRDF * attenuation
            accumulatedIndirect += hitDirectLight * brdf * attenuation;
        }
    }

    // Average and apply global intensity
    float3 finalIndirect = (accumulatedIndirect / float(raysPerParticle)) * indirectIntensity;

    g_indirectLighting[particleIdx] = float4(finalIndirect, 0.0);
}
```

**Add to CMakeLists.txt** (line ~240, after rtxdi_temporal_accumulate):

```cmake
# particle_indirect_lighting (compute shader for multi-bounce GI)
add_custom_command(
    OUTPUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/dxr/particle_indirect_lighting_cs.dxil"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/dxr"
    COMMAND dxc.exe -T cs_6_5 -E main "${CMAKE_CURRENT_SOURCE_DIR}/shaders/dxr/particle_indirect_lighting_cs.hlsl" -Fo "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/dxr/particle_indirect_lighting_cs.dxil"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/dxr/particle_indirect_lighting_cs.hlsl"
    COMMENT "Compiling particle_indirect_lighting_cs.hlsl"
)
list(APPEND SHADER_OUTPUTS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/dxr/particle_indirect_lighting_cs.dxil")
```

---

### Step 2: Extend RTLightingSystem_RayQuery (2-3 hours)

**Modify**: `src/lighting/RTLightingSystem_RayQuery.h`

**Add after line 21** (LightingConstants struct):
```cpp
struct IndirectLightingConstants {
    uint32_t particleCount;
    uint32_t raysPerParticle;      // 8 for balanced
    float maxLightingDistance;
    float indirectIntensity;       // 0.3-0.5
};
```

**Add to private members** (after line 90):
```cpp
// Indirect lighting resources
Microsoft::WRL::ComPtr<ID3DBlob> m_indirectLightingShader;
Microsoft::WRL::ComPtr<ID3D12RootSignature> m_indirectLightingRootSig;
Microsoft::WRL::ComPtr<ID3D12PipelineState> m_indirectLightingPSO;
Microsoft::WRL::ComPtr<ID3D12Resource> m_indirectLightingBuffer;

// Settings
uint32_t m_indirectRaysPerParticle = 8;     // Half of direct (16)
float m_indirectIntensity = 0.4f;           // 40% strength
```

**Add helper method declarations** (after line 62):
```cpp
void DispatchIndirectLighting(ID3D12GraphicsCommandList4* cmdList, ID3D12Resource* particleBuffer);
ID3D12Resource* GetIndirectLightingBuffer() const { return m_indirectLightingBuffer.Get(); }
void SetIndirectRaysPerParticle(uint32_t rays) { m_indirectRaysPerParticle = rays; }
void SetIndirectIntensity(float intensity) { m_indirectIntensity = intensity; }
```

**Modify**: `src/lighting/RTLightingSystem_RayQuery.cpp`

**Add to LoadShaders()** (after line 96):

```cpp
// Load indirect lighting shader
std::ifstream indirectFile("shaders/dxr/particle_indirect_lighting_cs.dxil", std::ios::binary);
if (!indirectFile) {
    LOG_ERROR("Failed to open particle_indirect_lighting_cs.dxil");
    return false;
}

std::vector<char> indirectData((std::istreambuf_iterator<char>(indirectFile)), std::istreambuf_iterator<char>());
hr = D3DCreateBlob(indirectData.size(), &m_indirectLightingShader);
if (FAILED(hr)) {
    LOG_ERROR("Failed to create blob for indirect lighting shader");
    return false;
}
memcpy(m_indirectLightingShader->GetBufferPointer(), indirectData.data(), indirectData.size());
```

**Add to CreateRootSignatures()** (after line 162):

```cpp
// Indirect Lighting Root Signature
// b0: IndirectLightingConstants
// t0: StructuredBuffer<Particle> g_particles
// t1: RaytracingAccelerationStructure g_particleBVH
// t2: StructuredBuffer<float4> g_directLighting (INPUT)
// u0: RWStructuredBuffer<float4> g_indirectLighting (OUTPUT)
{
    CD3DX12_ROOT_PARAMETER1 rootParams[5];
    rootParams[0].InitAsConstants(4, 0);  // b0: IndirectLightingConstants
    rootParams[1].InitAsShaderResourceView(0);  // t0: particles
    rootParams[2].InitAsShaderResourceView(1);  // t1: TLAS
    rootParams[3].InitAsShaderResourceView(2);  // t2: direct lighting (INPUT)
    rootParams[4].InitAsUnorderedAccessView(0);  // u0: indirect lighting (OUTPUT)

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSigDesc;
    rootSigDesc.Init_1_1(5, rootParams, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);

    Microsoft::WRL::ComPtr<ID3DBlob> signature, error;
    hr = D3DX12SerializeVersionedRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1_1, &signature, &error);
    if (FAILED(hr)) {
        if (error) {
            LOG_ERROR("Indirect lighting root signature serialization failed: {}", (char*)error->GetBufferPointer());
        }
        return false;
    }

    hr = m_device->GetDevice()->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(),
                                                     IID_PPV_ARGS(&m_indirectLightingRootSig));
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create indirect lighting root signature");
        return false;
    }
}
```

**Add to CreatePipelineStates()** (after line 197):

```cpp
// Indirect Lighting PSO
{
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = m_indirectLightingRootSig.Get();
    psoDesc.CS = CD3DX12_SHADER_BYTECODE(m_indirectLightingShader.Get());

    hr = m_device->GetDevice()->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_indirectLightingPSO));
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create indirect lighting PSO");
        return false;
    }
}
```

**Add to CreateAccelerationStructures()** (after line 236):

```cpp
// Create indirect lighting output buffer
{
    size_t indirectBufferSize = m_particleCount * 16;  // float4 = 16 bytes

    ResourceManager::BufferDesc desc = {};
    desc.size = indirectBufferSize;
    desc.heapType = D3D12_HEAP_TYPE_DEFAULT;
    desc.flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    desc.initialState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

    m_indirectLightingBuffer = m_resources->CreateBuffer("IndirectLightingOutput", desc);
    if (!m_indirectLightingBuffer) {
        LOG_ERROR("Failed to create indirect lighting output buffer");
        return false;
    }
}
```

**Add new method DispatchIndirectLighting()** (after line 463):

```cpp
void RTLightingSystem_RayQuery::DispatchIndirectLighting(ID3D12GraphicsCommandList4* cmdList, ID3D12Resource* particleBuffer) {
    cmdList->SetPipelineState(m_indirectLightingPSO.Get());
    cmdList->SetComputeRootSignature(m_indirectLightingRootSig.Get());

    // Set root parameters
    IndirectLightingConstants constants = {
        m_particleCount,
        m_indirectRaysPerParticle,
        m_maxLightingDistance,
        m_indirectIntensity
    };
    cmdList->SetComputeRoot32BitConstants(0, 4, &constants, 0);
    cmdList->SetComputeRootShaderResourceView(1, particleBuffer->GetGPUVirtualAddress());
    cmdList->SetComputeRootShaderResourceView(2, m_topLevelAS->GetGPUVirtualAddress());
    cmdList->SetComputeRootShaderResourceView(3, m_lightingBuffer->GetGPUVirtualAddress());  // Read direct
    cmdList->SetComputeRootUnorderedAccessView(4, m_indirectLightingBuffer->GetGPUVirtualAddress());  // Write indirect

    // Dispatch
    uint32_t threadGroups = (m_particleCount + 63) / 64;
    cmdList->Dispatch(threadGroups, 1, 1);

    // UAV barrier
    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_indirectLightingBuffer.Get());
    cmdList->ResourceBarrier(1, &barrier);
}
```

**Modify ComputeLighting()** (replace line 465-482):

```cpp
void RTLightingSystem_RayQuery::ComputeLighting(ID3D12GraphicsCommandList4* cmdList,
                                                ID3D12Resource* particleBuffer,
                                                uint32_t particleCount) {
    m_particleCount = particleCount;

    // Pipeline:
    // 1. Generate AABBs from particle positions
    GenerateAABBs(cmdList, particleBuffer);

    // 2. Build BLAS from AABBs
    BuildBLAS(cmdList);

    // 3. Build TLAS from BLAS
    BuildTLAS(cmdList);

    // 4. Dispatch direct lighting (1st bounce)
    DispatchRayQueryLighting(cmdList, particleBuffer);

    // 5. Dispatch indirect lighting (2nd bounce) - reads direct, writes indirect
    DispatchIndirectLighting(cmdList, particleBuffer);
}
```

**Add to Shutdown()** (after line 62):

```cpp
m_indirectLightingShader.Reset();
m_indirectLightingRootSig.Reset();
m_indirectLightingPSO.Reset();
m_indirectLightingBuffer.Reset();
```

---

### Step 3: Integrate with Gaussian Renderer (30 min)

**Modify**: `src/particles/ParticleRenderer_Gaussian.cpp`

In the `Render()` method where you bind RT lighting resources, add indirect buffer binding.

Find where you set root parameters for the Gaussian shader (around line ~300-400 where you bind `m_rtLighting->GetLightingBuffer()`).

**Add**:
```cpp
// Bind indirect lighting buffer (t7 - next available slot)
cmdList->SetComputeRootShaderResourceView(7, m_rtLighting->GetIndirectLightingBuffer()->GetGPUVirtualAddress());
```

**Modify**: `shaders/particles/particle_gaussian_raytrace.hlsl`

**Add after line 34** (after `StructuredBuffer<float4> g_particleLighting : register(t5);`):

```hlsl
// Indirect lighting (2nd bounce GI)
StructuredBuffer<float4> g_particleIndirectLighting : register(t7);
```

Find the final color output section (around line 750-800) and modify:

```hlsl
// OLD:
float3 rtLightingContrib = g_particleLighting[sortedIdx].rgb;
float3 finalColor = baseColor + rtLightingContrib;

// NEW:
float3 directLighting = g_particleLighting[sortedIdx].rgb;
float3 indirectLighting = g_particleIndirectLighting[sortedIdx].rgb;
float3 finalColor = baseColor + directLighting + indirectLighting;
```

---

### Step 4: Add ImGui Controls (30 min)

**Modify**: `src/core/Application.cpp`

In the ImGui RT Lighting section (around line 800-900), add:

```cpp
ImGui::Text("Multi-Bounce Global Illumination");
ImGui::SliderInt("Indirect Rays/Particle", (int*)&m_indirectRaysPerParticle, 2, 16);
ImGui::SliderFloat("Indirect Intensity", &m_indirectIntensity, 0.0f, 1.0f);

if (ImGui::Button("Reset Indirect Settings")) {
    m_indirectRaysPerParticle = 8;
    m_indirectIntensity = 0.4f;
}

// Apply settings
m_rtLighting->SetIndirectRaysPerParticle(m_indirectRaysPerParticle);
m_rtLighting->SetIndirectIntensity(m_indirectIntensity);
```

**Add member variables to Application.h**:

```cpp
uint32_t m_indirectRaysPerParticle = 8;
float m_indirectIntensity = 0.4f;
```

---

## Testing Checklist

- [ ] Build succeeds with no errors
- [ ] Indirect shader compiles to .dxil
- [ ] Application launches without crashes
- [ ] Particles show subtle color bleeding
- [ ] Shadowed regions have soft ambient glow
- [ ] FPS reads 95-105 (target: 99)
- [ ] ImGui sliders affect visual result
- [ ] Setting indirect intensity to 0 disables GI (matches current look)
- [ ] Setting to 1.0 shows strong color bleeding

---

## Expected Visual Results

**Before (Direct Only)**:
- Hard shadows (black)
- No color interaction between particles
- High contrast

**After (Direct + Indirect)**:
- Soft ambient lighting in shadows
- Warm glow from nearby hot particles reflects onto cooler ones
- Color bleeding (red-hot particles cast red tint on neighbors)
- More cohesive, volumetric look
- Subtle but noticeable improvement

---

## Troubleshooting Guide

### Build fails with shader compilation error
**Symptom**: `dxc.exe` fails to compile `particle_indirect_lighting_cs.hlsl`
**Fix**: Ensure shader is in correct directory, check HLSL syntax

### Application crashes at startup
**Symptom**: Crash in `RTLightingSystem_RayQuery::Initialize()`
**Fix**: Check that shader file exists at runtime, verify buffer creation succeeds

### Black screen or missing particles
**Symptom**: Rendering broken after adding indirect lighting
**Fix**: Ensure resource state transitions are correct, check UAV barriers

### FPS lower than expected (<90 FPS)
**Symptom**: Performance below 95 FPS target
**Fix**: Reduce `m_indirectRaysPerParticle` to 4-6, verify BLAS/TLAS aren't rebuilding twice

### No visual difference
**Symptom**: Scene looks identical with indirect lighting enabled
**Fix**: Increase `m_indirectIntensity` to 1.0 for debugging, verify indirect buffer is bound in Gaussian renderer

---

## Rollback Plan

If implementation causes issues, rollback is simple:

1. Comment out indirect lighting dispatch in `RTLightingSystem_RayQuery::ComputeLighting()`
2. Remove indirect buffer binding from Gaussian renderer
3. Application returns to exact previous state

**Critical**: All changes are additive - no modifications to existing direct lighting code.

---

## Performance Metrics

**Expected frame breakdown @ 1080p, 10K particles**:

| Pass | Time | Details |
|------|------|---------|
| AABB Generation | 0.2ms | Unchanged |
| BLAS Build | 1.5ms | Unchanged |
| TLAS Build | 0.6ms | Unchanged |
| Direct Lighting | 4.0ms | 16 rays/particle |
| **Indirect Lighting** | **3.5ms** | **8 rays/particle (NEW)** |
| Gaussian Render | 0.5ms | Unchanged |
| **Total** | **10.3ms** | **97 FPS** |

**Variance**: ±2 FPS depending on particle distribution, camera angle

---

## Next Steps After Completion

1. **Capture before/after screenshots** for documentation
2. **Run performance profiling** with PIX to verify timing
3. **Test with different particle counts** (5K, 10K, 20K)
4. **Proceed to Phase 2** - Temporal accumulation (restore to 115 FPS)

---

**Document Version**: 1.0
**Last Updated**: 2025-10-25
**Implementation Status**: Ready for development
