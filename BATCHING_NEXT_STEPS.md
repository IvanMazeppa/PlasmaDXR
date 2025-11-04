# Particle Batching - Next Steps

**Current Status:** Core infrastructure complete (60%), need implementation completion
**Estimated Remaining Time:** 2 hours
**Goal:** Fix 2045+ particle crash via batching mitigation

---

## ✅ What's Complete

1. **Header structure** - AccelerationBatch, batch accessors, helper declarations
2. **CreateBatchedAccelerationStructures()** - Full implementation creates all batch resources
3. **Initialize() routing** - Chooses batching vs monolithic based on particle count
4. **Documentation** - Comprehensive guides in BATCHING_STATUS.md and PARTICLE_BATCHING_IMPLEMENTATION.md

---

## 🔥 Critical Remaining Work (In Order)

### 1. Add 3 Batch Build Functions (~30 min)

Insert these after line 555 in `RTLightingSystem_RayQuery.cpp`:

```cpp
void RTLightingSystem_RayQuery::GenerateAABBsForBatch(
    ID3D12GraphicsCommandList4* cmdList,
    ID3D12Resource* particleBuffer,
    AccelerationBatch& batch) {

    cmdList->SetPipelineState(m_aabbGenPSO.Get());
    cmdList->SetComputeRootSignature(m_aabbGenRootSig.Get());

    AABBConstants constants = {
        batch.count,  // THIS BATCH particle count
        m_particleRadius,
        m_enableAdaptiveRadius ? 1u : 0u,
        m_adaptiveInnerZone,
        m_adaptiveOuterZone,
        m_adaptiveInnerScale,
        m_adaptiveOuterScale,
        m_densityScaleMin,
        m_densityScaleMax,
        0.0f
    };
    cmdList->SetComputeRoot32BitConstants(0, 10, &constants, 0);

    // Particle buffer WITH BATCH OFFSET
    D3D12_GPU_VIRTUAL_ADDRESS particleAddr = particleBuffer->GetGPUVirtualAddress();
    particleAddr += batch.startIndex * 128;  // 128 bytes per particle
    cmdList->SetComputeRootShaderResourceView(1, particleAddr);

    // Batch AABB buffer
    cmdList->SetComputeRootUnorderedAccessView(2, batch.aabbBuffer->GetGPUVirtualAddress());

    uint32_t threadGroups = (batch.count + 255) / 256;
    cmdList->Dispatch(threadGroups, 1, 1);

    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(batch.aabbBuffer.Get());
    cmdList->ResourceBarrier(1, &barrier);
}

void RTLightingSystem_RayQuery::BuildBatchBLAS(
    ID3D12GraphicsCommandList4* cmdList,
    AccelerationBatch& batch) {

    uint32_t aabbCount = batch.count;
    uint32_t leafCount = (aabbCount + 3) / 4;
    bool isPowerOf2 = (leafCount & (leafCount - 1)) == 0 && leafCount > 0;

    if (isPowerOf2 && leafCount >= 512) {
        uint32_t paddingNeeded = 4;
        aabbCount += paddingNeeded;
        // (logging omitted for brevity)
    }

    D3D12_RAYTRACING_GEOMETRY_DESC geomDesc = {};
    geomDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
    geomDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    geomDesc.AABBs.AABBCount = aabbCount;
    geomDesc.AABBs.AABBs.StartAddress = batch.aabbBuffer->GetGPUVirtualAddress();
    geomDesc.AABBs.AABBs.StrideInBytes = 24;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blasInputs = {};
    blasInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    blasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
    blasInputs.NumDescs = 1;
    blasInputs.pGeometryDescs = &geomDesc;
    blasInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC blasDesc = {};
    blasDesc.Inputs = blasInputs;
    blasDesc.DestAccelerationStructureData = batch.blas->GetGPUVirtualAddress();
    blasDesc.ScratchAccelerationStructureData = batch.blasScratch->GetGPUVirtualAddress();

    cmdList->BuildRaytracingAccelerationStructure(&blasDesc, 0, nullptr);

    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(batch.blas.Get());
    cmdList->ResourceBarrier(1, &barrier);
}

void RTLightingSystem_RayQuery::BuildBatchTLAS(
    ID3D12GraphicsCommandList4* cmdList,
    AccelerationBatch& batch) {

    D3D12_RAYTRACING_INSTANCE_DESC instanceDesc = {};
    instanceDesc.InstanceID = 0;
    instanceDesc.InstanceMask = 0xFF;
    instanceDesc.InstanceContributionToHitGroupIndex = 0;
    instanceDesc.Flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
    instanceDesc.AccelerationStructure = batch.blas->GetGPUVirtualAddress();

    instanceDesc.Transform[0][0] = 1.0f;
    instanceDesc.Transform[1][1] = 1.0f;
    instanceDesc.Transform[2][2] = 1.0f;

    void* mapped = nullptr;
    batch.instanceDesc->Map(0, nullptr, &mapped);
    memcpy(mapped, &instanceDesc, sizeof(instanceDesc));
    batch.instanceDesc->Unmap(0, nullptr);

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlasInputs = {};
    tlasInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    tlasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
    tlasInputs.NumDescs = 1;
    tlasInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    tlasInputs.InstanceDescs = batch.instanceDesc->GetGPUVirtualAddress();

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC tlasDesc = {};
    tlasDesc.Inputs = tlasInputs;
    tlasDesc.DestAccelerationStructureData = batch.tlas->GetGPUVirtualAddress();
    tlasDesc.ScratchAccelerationStructureData = batch.tlasScratch->GetGPUVirtualAddress();

    cmdList->BuildRaytracingAccelerationStructure(&tlasDesc, 0, nullptr);

    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(batch.tlas.Get());
    cmdList->ResourceBarrier(1, &barrier);
}
```

### 2. Modify ComputeLighting() (~10 min)

Find `ComputeLighting()` and replace AABB/BLAS/TLAS build calls:

```cpp
void RTLightingSystem_RayQuery::ComputeLighting(...) {
    m_frameCount++;

    if (m_useBatching && !m_batches.empty()) {
        // BATCHED PATH
        for (auto& batch : m_batches) {
            GenerateAABBsForBatch(cmdList, particleBuffer, batch);
            BuildBatchBLAS(cmdList, batch);
            BuildBatchTLAS(cmdList, batch);
        }
    } else {
        // MONOLITHIC PATH
        GenerateAABBs(cmdList, particleBuffer);
        BuildBLAS(cmdList);
        BuildTLAS(cmdList);
    }

    // Dispatch lighting (unchanged)
    DispatchRayQueryLighting(cmdList, particleBuffer, cameraPosition);
}
```

### 3. Update GetTLAS() for Backward Compatibility (~2 min)

In `RTLightingSystem_RayQuery.h`, modify GetTLAS():

```cpp
ID3D12Resource* GetTLAS() const {
    if (m_useBatching && !m_batches.empty()) {
        return m_batches[0].tlas.Get();  // First batch for backward compat
    }
    return m_topLevelAS.Get();
}
```

### 4. Test RT Lighting Alone (~5 min)

Build and test at 2045 particles with probe grid DISABLED:

```bash
MSBuild.exe build/PlasmaDX-Clean.sln /p:Configuration=Debug /t:Build
# Test at 2045 particles
# Expected: NO CRASH if batching works for RT lighting
```

---

##  🔴 IF RT LIGHTING WORKS, Proceed to Probe Grid Updates

### 5. Update Probe Constants (~5 min)

In `ProbeGridSystem.h`, update struct:

```cpp
struct ProbeUpdateConstants {
    DirectX::XMFLOAT3 gridMin;
    float gridSpacing;

    uint32_t gridSize;
    uint32_t raysPerProbe;
    uint32_t particleCount;
    uint32_t lightCount;

    uint32_t frameIndex;
    uint32_t updateInterval;
    uint32_t numBatches;     // NEW
    uint32_t padding1;
};
```

### 6. Update Probe Shader (~20 min)

**update_probes.hlsl changes:**

```hlsl
// At top, replace single TLAS:
// OLD: RaytracingAccelerationStructure g_particleTLAS : register(t2);
// NEW:
RaytracingAccelerationStructure g_particleTLAS[8] : register(t2);  // Up to 8 batches

cbuffer ProbeUpdateConstants : register(b0) {
    float3 g_gridMin;
    float g_gridSpacing;
    uint g_gridSize;
    uint g_raysPerProbe;
    uint g_particleCount;
    uint g_lightCount;
    uint g_frameIndex;
    uint g_updateInterval;
    uint g_numBatches;  // NEW
    uint g_padding1;
};

// In ray tracing loop (around line 232-290), replace:
// OLD: Single TLAS trace
// NEW: Multi-batch trace

for (uint rayIdx = 0; rayIdx < g_raysPerProbe; rayIdx++) {
    float3 direction = FibonacciSphere(rayIdx, g_raysPerProbe);

    RayDesc ray;
    ray.Origin = probePos;
    ray.Direction = direction;
    ray.TMin = 0.01;
    ray.TMax = 200.0;

    // TRACE AGAINST ALL BATCHES
    for (uint batchIdx = 0; batchIdx < g_numBatches; batchIdx++) {
        RayQuery<RAY_FLAG_NONE> q;
        q.TraceRayInline(g_particleTLAS[batchIdx], RAY_FLAG_NONE, 0xFF, ray);

        uint iterationCount = 0;
        const uint MAX_ITERATIONS = 1000;

        while (q.Proceed() && iterationCount < MAX_ITERATIONS) {
            iterationCount++;

            if (q.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
                uint localIdx = q.CandidatePrimitiveIndex();
                uint globalIdx = (batchIdx * 2000) + localIdx;  // Batch offset

                if (globalIdx < g_particleCount) {
                    Particle particle = g_particles[globalIdx];

                    // ... existing intersection test ...
                    float3 oc = ray.Origin - particle.position;
                    float radius = particle.radius;
                    float b = dot(oc, ray.Direction);
                    float c = dot(oc, oc) - radius * radius;
                    float discriminant = b * b - c;

                    if (discriminant >= 0.0) {
                        float t = -b - sqrt(discriminant);
                        if (t > ray.TMin && t < ray.TMax) {
                            q.CommitProceduralPrimitiveHit(t);
                        }
                    }
                }
            }
        }

        // Check for hit in THIS BATCH
        if (q.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT) {
            uint localIdx = q.CommittedPrimitiveIndex();
            uint globalIdx = (batchIdx * 2000) + localIdx;

            if (globalIdx < g_particleCount) {
                Particle particle = g_particles[globalIdx];
                totalIrradiance += ComputeParticleLighting(
                    probePos,
                    particle.position,
                    particle.radius,
                    particle.temperature
                );
            }
        }

        // Timeout diagnostic
        if (iterationCount >= MAX_ITERATIONS) {
            totalIrradiance = float3(10.0, 0.0, 0.0);  // Red = timeout
        }
    }
}
```

### 7. Update Probe Root Signature (~15 min)

In `ProbeGridSystem.cpp`, update `CreatePipelines()`:

```cpp
// OLD: 5 root parameters
// NEW: 12 root parameters (t2-t9 for 8 batch TLAS slots)

CD3DX12_ROOT_PARAMETER1 rootParams[12];  // Was 5

rootParams[0].InitAsConstantBufferView(0, 0, ...);  // b0
rootParams[1].InitAsShaderResourceView(0, 0, ...);  // t0 particles
rootParams[2].InitAsShaderResourceView(1, 0, ...);  // t1 lights

// t2-t9: Batch TLAS array (8 slots)
for (uint32_t i = 0; i < 8; i++) {
    rootParams[3 + i].InitAsShaderResourceView(2 + i, 0, ...);
}

rootParams[11].InitAsUnorderedAccessView(0, 0, ...);  // u0 probe buffer

CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSigDesc;
rootSigDesc.Init_1_1(12, rootParams, 0, nullptr, ...);  // Was 5
```

### 8. Update Probe Binding Code (~15 min)

In `ProbeGridSystem.cpp`, update `UpdateProbes()`:

```cpp
void ProbeGridSystem::UpdateProbes(...) {
    // ... existing setup ...

    // Get batch count (needs RTLightingSystem pointer passed in)
    uint32_t numBatches = 1;  // Default to monolithic
    if (rtLightingSystem) {
        numBatches = rtLightingSystem->GetBatchCount();
        if (numBatches == 0) numBatches = 1;
    }

    constants.numBatches = numBatches;
    // ... upload constants ...

    // Bind root parameters
    commandList->SetComputeRootConstantBufferView(0, ...);
    commandList->SetComputeRootShaderResourceView(1, ...);  // Particles
    commandList->SetComputeRootShaderResourceView(2, ...);  // Lights

    // Bind batch TLAS array (t2-t9)
    if (numBatches > 1 && rtLightingSystem) {
        for (uint32_t i = 0; i < numBatches && i < 8; i++) {
            ID3D12Resource* batchTLAS = rtLightingSystem->GetBatchTLAS(i);
            if (batchTLAS) {
                commandList->SetComputeRootShaderResourceView(3 + i, batchTLAS->GetGPUVirtualAddress());
            }
        }
    } else {
        // Monolithic: bind single TLAS to first slot
        commandList->SetComputeRootShaderResourceView(3, particleTLAS->GetGPUVirtualAddress());
    }

    commandList->SetComputeRootUnorderedAccessView(11, ...);  // Probe buffer (was 4, now 11)

    // Dispatch
    commandList->Dispatch(4, 4, 4);

    // Barrier
    CD3DX12_RESOURCE_BARRIER uavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_probeBuffer.Get());
    commandList->ResourceBarrier(1, &uavBarrier);
}
```

**CRITICAL:** `UpdateProbes()` signature needs RTLightingSystem pointer to get batch count/TLAS. Update the call site in `Application.cpp` to pass `m_rtLighting`.

### 9. Update Application.cpp Probe Call (~5 min)

In `Application.cpp` where probe grid is updated:

```cpp
m_probeGridSystem->UpdateProbes(
    cmdList,
    m_rtLighting,  // ADD THIS - pass RTLightingSystem pointer
    m_particleSystem->GetParticleBuffer(),
    m_config.particleCount,
    lightBuffer,
    lightCount,
    m_frameCount
);
```

Update `ProbeGridSystem.h` signature:
```cpp
void UpdateProbes(
    ID3D12GraphicsCommandList* commandList,
    RTLightingSystem_RayQuery* rtLightingSystem,  // ADD THIS
    ID3D12Resource* particleBuffer,
    uint32_t particleCount,
    ID3D12Resource* lightBuffer,
    uint32_t lightCount,
    uint32_t frameIndex);
```

### 10. Rebuild and Test (~10 min)

```bash
MSBuild.exe build/PlasmaDX-Clean.sln /p:Configuration=Debug /t:Build
# Test at 2045 particles with probe grid ENABLED
# Expected: NO CRASH!!!
```

---

## Success Criteria

✅ **Complete Success:** No crash at 2045, 4000, 10K particles
⚠️ **Partial Success:** Crash threshold moves higher
❌ **Failure:** Still crashes at 2045

---

## If It Doesn't Work

1. Reduce PARTICLES_PER_BATCH from 2000 → 1000
2. Test with 1 batch (2045 particles in single batch - rules out batching logic bugs)
3. Add extensive logging to track which batch/ray/probe causes hang
4. Consider alternative: Disable probe grid entirely, accept RT lighting only

---

**Total Estimated Time:** ~2 hours
**Key Files:** RTLightingSystem_RayQuery.cpp, update_probes.hlsl, ProbeGridSystem.cpp, Application.cpp
