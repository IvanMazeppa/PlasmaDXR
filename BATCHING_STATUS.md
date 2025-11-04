# Particle Batching Implementation Status

**Date:** 2025-11-04
**Goal:** Implement batching to avoid 2045+ particle crash

---

## ✅ Completed So Far

### 1. Header Modifications (RTLightingSystem_RayQuery.h)
- ✅ Added `AccelerationBatch` struct with all per-batch resources
- ✅ Added `PARTICLES_PER_BATCH = 2000` constant
- ✅ Added `m_useBatching` flag (enabled by default)
- ✅ Added `m_batches` vector
- ✅ Added batch accessor methods (`GetBatchCount()`, `GetBatchTLAS()`)
- ✅ Added batch helper function declarations

### 2. Initialization Logic (RTLightingSystem_RayQuery.cpp)
- ✅ Modified `Initialize()` to choose batching vs monolithic
- ✅ Implemented complete `CreateBatchedAccelerationStructures()`
  - Creates all AABB buffers per batch
  - Creates all BLAS/TLAS per batch
  - Creates scratch buffers
  - Creates instance descriptors
  - Logs batching info

---

## 🔄 Remaining Implementation

### Phase 1: Batch Build Functions (HIGH PRIORITY)

Need to implement 3 batch-specific build functions. These are simpler than the full CreateBatchedAccelerationStructures because resources are already created:

#### A. GenerateAABBsForBatch()
```cpp
void RTLightingSystem_RayQuery::GenerateAABBsForBatch(
    ID3D12GraphicsCommandList4* cmdList,
    ID3D12Resource* particleBuffer,
    AccelerationBatch& batch) {

    cmdList->SetPipelineState(m_aabbGenPSO.Get());
    cmdList->SetComputeRootSignature(m_aabbGenRootSig.Get());

    // Constants for THIS BATCH ONLY
    AABBConstants constants = {
        batch.count,  // NOT m_particleCount!
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

    // Bind particle buffer WITH OFFSET for this batch
    D3D12_GPU_VIRTUAL_ADDRESS particleAddr = particleBuffer->GetGPUVirtualAddress();
    particleAddr += batch.startIndex * 128;  // 128 bytes per particle
    cmdList->SetComputeRootShaderResourceView(1, particleAddr);

    // Bind batch AABB buffer
    cmdList->SetComputeRootUnorderedAccessView(2, batch.aabbBuffer->GetGPUVirtualAddress());

    // Dispatch for THIS BATCH ONLY
    uint32_t threadGroups = (batch.count + 255) / 256;
    cmdList->Dispatch(threadGroups, 1, 1);

    // UAV barrier
    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(batch.aabbBuffer.Get());
    cmdList->ResourceBarrier(1, &barrier);
}
```

#### B. BuildBatchBLAS()
Copy existing `BuildBLAS()` but use batch resources:
```cpp
void RTLightingSystem_RayQuery::BuildBatchBLAS(
    ID3D12GraphicsCommandList4* cmdList,
    AccelerationBatch& batch) {

    // Power-of-2 padding check (same as monolithic)
    uint32_t aabbCount = batch.count;
    uint32_t leafCount = (aabbCount + 3) / 4;
    bool isPowerOf2 = (leafCount & (leafCount - 1)) == 0 && leafCount > 0;

    if (isPowerOf2 && leafCount >= 512) {
        uint32_t paddingNeeded = 4;
        aabbCount += paddingNeeded;
        uint32_t newLeafCount = (aabbCount + 3) / 4;
        LOG_WARN("BVH leaf count {} is power-of-2 (batch particles={}), adding {} padding AABBs → {} leaves",
                 leafCount, batch.count, paddingNeeded, newLeafCount);
    }

    // Setup geometry desc
    D3D12_RAYTRACING_GEOMETRY_DESC geomDesc = {};
    geomDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
    geomDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    geomDesc.AABBs.AABBCount = aabbCount;
    geomDesc.AABBs.AABBs.StartAddress = batch.aabbBuffer->GetGPUVirtualAddress();  // Batch buffer!
    geomDesc.AABBs.AABBs.StrideInBytes = 24;

    // Build BLAS desc
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blasInputs = {};
    blasInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    blasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
    blasInputs.NumDescs = 1;
    blasInputs.pGeometryDescs = &geomDesc;
    blasInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC blasDesc = {};
    blasDesc.Inputs = blasInputs;
    blasDesc.DestAccelerationStructureData = batch.blas->GetGPUVirtualAddress();  // Batch BLAS!
    blasDesc.ScratchAccelerationStructureData = batch.blasScratch->GetGPUVirtualAddress();  // Batch scratch!

    cmdList->BuildRaytracingAccelerationStructure(&blasDesc, 0, nullptr);

    // UAV barrier
    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(batch.blas.Get());
    cmdList->ResourceBarrier(1, &barrier);
}
```

#### C. BuildBatchTLAS()
Copy existing `BuildTLAS()` but use batch resources:
```cpp
void RTLightingSystem_RayQuery::BuildBatchTLAS(
    ID3D12GraphicsCommandList4* cmdList,
    AccelerationBatch& batch) {

    // Map instance desc buffer
    D3D12_RAYTRACING_INSTANCE_DESC* instanceDesc = nullptr;
    batch.instanceDesc->Map(0, nullptr, reinterpret_cast<void**>(&instanceDesc));

    // Setup identity transform
    instanceDesc->Transform[0][0] = 1.0f;
    instanceDesc->Transform[1][1] = 1.0f;
    instanceDesc->Transform[2][2] = 1.0f;
    instanceDesc->InstanceID = 0;
    instanceDesc->InstanceMask = 0xFF;
    instanceDesc->InstanceContributionToHitGroupIndex = 0;
    instanceDesc->Flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
    instanceDesc->AccelerationStructure = batch.blas->GetGPUVirtualAddress();  // Batch BLAS!

    batch.instanceDesc->Unmap(0, nullptr);

    // Build TLAS desc
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlasInputs = {};
    tlasInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    tlasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
    tlasInputs.NumDescs = 1;
    tlasInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    tlasInputs.InstanceDescs = batch.instanceDesc->GetGPUVirtualAddress();

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC tlasDesc = {};
    tlasDesc.Inputs = tlasInputs;
    tlasDesc.DestAccelerationStructureData = batch.tlas->GetGPUVirtualAddress();  // Batch TLAS!
    tlasDesc.ScratchAccelerationStructureData = batch.tlasScratch->GetGPUVirtualAddress();  // Batch scratch!

    cmdList->BuildRaytracingAccelerationStructure(&tlasDesc, 0, nullptr);

    // UAV barrier
    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(batch.tlas.Get());
    cmdList->ResourceBarrier(1, &barrier);
}
```

### Phase 2: Modify ComputeLighting()

Find the `ComputeLighting()` function and add batch dispatch logic:

```cpp
void RTLightingSystem_RayQuery::ComputeLighting(...) {
    m_frameCount++;

    if (m_useBatching && !m_batches.empty()) {
        // BATCHED PATH: Build all batches
        for (auto& batch : m_batches) {
            GenerateAABBsForBatch(cmdList, particleBuffer, batch);
            BuildBatchBLAS(cmdList, batch);
            BuildBatchTLAS(cmdList, batch);
        }

        // Note: Gaussian renderer will use GetBatchTLAS(0) for backward compat
        // Probe grid will use all batches
    } else {
        // MONOLITHIC PATH: Existing code
        GenerateAABBs(cmdList, particleBuffer);
        BuildBLAS(cmdList);
        BuildTLAS(cmdList);
    }

    // Dispatch lighting (unchanged - writes to shared buffer)
    DispatchRayQueryLighting(cmdList, particleBuffer, cameraPosition);
}
```

### Phase 3: Update GetTLAS() for Backward Compatibility

The Gaussian renderer calls `GetTLAS()` expecting a single TLAS. Return batch 0's TLAS:

```cpp
// In header:
ID3D12Resource* GetTLAS() const {
    if (m_useBatching && !m_batches.empty()) {
        return m_batches[0].tlas.Get();  // Return first batch for backward compat
    }
    return m_topLevelAS.Get();
}
```

---

## Phase 4: Probe Grid Multi-TLAS Support (CRITICAL!)

This is the key part that makes probe grid work with batching.

### A. Update Probe Constants (ProbeGridSystem.h)

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
    uint32_t numBatches;        // NEW: Number of particle batches
    uint32_t padding1;
};
```

### B. Update Probe Root Signature (ProbeGridSystem.cpp)

**Current:**
```cpp
// Root parameter 3: t2 - Particle TLAS (single)
rootParams[3].InitAsShaderResourceView(2, 0, ...);
```

**New (support up to 8 batches):**
```cpp
// Root parameter 3-10: t2-t9 - Particle TLAS array (up to 8 batches = 16K particles)
for (uint32_t i = 0; i < 8; i++) {
    rootParams[3 + i].InitAsShaderResourceView(2 + i, 0, ...);
}
```

### C. Update Probe Shader (update_probes.hlsl)

**Current:**
```hlsl
RaytracingAccelerationStructure g_particleTLAS : register(t2);
```

**New:**
```hlsl
// Support up to 8 batches (16K particles)
RaytracingAccelerationStructure g_particleTLAS[8] : register(t2);

cbuffer ProbeUpdateConstants : register(b0) {
    // ... existing ...
    uint g_numBatches;  // NEW
    uint g_padding1;
};
```

**Update ray tracing loop:**
```hlsl
// Cast rays in Fibonacci sphere distribution
for (uint rayIdx = 0; rayIdx < g_raysPerProbe; rayIdx++) {
    float3 direction = FibonacciSphere(rayIdx, g_raysPerProbe);

    RayDesc ray;
    ray.Origin = probePos;
    ray.Direction = direction;
    ray.TMin = 0.01;
    ray.TMax = 200.0;

    // CRITICAL: Trace against ALL batches
    for (uint batchIdx = 0; batchIdx < g_numBatches; batchIdx++) {
        RayQuery<RAY_FLAG_NONE> q;
        q.TraceRayInline(g_particleTLAS[batchIdx], RAY_FLAG_NONE, 0xFF, ray);

        uint iterationCount = 0;
        const uint MAX_ITERATIONS = 1000;

        while (q.Proceed() && iterationCount < MAX_ITERATIONS) {
            iterationCount++;

            if (q.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
                uint localParticleIdx = q.CandidatePrimitiveIndex();
                uint globalParticleIdx = (batchIdx * 2000) + localParticleIdx;  // Adjust for batch offset

                if (globalParticleIdx < g_particleCount) {
                    Particle particle = g_particles[globalParticleIdx];

                    // ... existing intersection test ...
                }
            }
        }

        // Check for hit in THIS BATCH
        if (q.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT) {
            uint localIdx = q.CommittedPrimitiveIndex();
            uint globalIdx = (batchIdx * 2000) + localIdx;

            if (globalIdx < g_particleCount) {
                Particle particle = g_particles[globalIdx];
                totalIrradiance += ComputeParticleLighting(...);
            }
        }
    }
}
```

### D. Update ProbeGridSystem::UpdateProbes()

Bind all batch TLAS resources:

```cpp
void ProbeGridSystem::UpdateProbes(...) {
    // ... existing setup ...

    // Get batch count from RT lighting system
    uint32_t numBatches = rtLightingSystem->GetBatchCount();
    if (numBatches == 0) numBatches = 1;  // Fallback to monolithic

    constants.numBatches = numBatches;
    // ... upload constants ...

    // Bind root parameters
    commandList->SetComputeRootConstantBufferView(0, ...);  // Constants
    commandList->SetComputeRootShaderResourceView(1, ...);  // Particles
    commandList->SetComputeRootShaderResourceView(2, ...);  // Lights

    // Bind ALL batch TLAS resources (t2-t9)
    for (uint32_t i = 0; i < numBatches && i < 8; i++) {
        ID3D12Resource* batchTLAS = rtLightingSystem->GetBatchTLAS(i);
        if (batchTLAS) {
            commandList->SetComputeRootShaderResourceView(3 + i, batchTLAS->GetGPUVirtualAddress());
        }
    }

    // If using monolithic, bind to first slot
    if (numBatches == 1) {
        commandList->SetComputeRootShaderResourceView(3, rtLightingSystem->GetTLAS()->GetGPUVirtualAddress());
    }

    commandList->SetComputeRootUnorderedAccessView(11, ...);  // Probe buffer (shifted from 4 to 11)

    // Dispatch
    commandList->Dispatch(4, 4, 4);

    // UAV barrier
    CD3DX12_RESOURCE_BARRIER uavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_probeBuffer.Get());
    commandList->ResourceBarrier(1, &uavBarrier);
}
```

---

## Implementation Order (Priority)

1. **Add batch build functions** (GenerateAABBsForBatch, BuildBatchBLAS, BuildBatchTLAS) - 30 min
2. **Modify ComputeLighting()** to use batching - 10 min
3. **Test RT lighting alone at 2045** (without probe grid) - 5 min
4. **Update probe root signature** (expand from 5 to 12 params) - 15 min
5. **Update probe shader** for multi-TLAS - 20 min
6. **Update ProbeGridSystem binding code** - 15 min
7. **Recompile shaders** - 5 min
8. **Test at 2045 particles with probe grid** - 10 min
9. **Test at 4000, 10000 particles** - 10 min

**Total:** ~2 hours remaining work

---

## Expected Results

**Success Criteria:**
- ✅ Application launches at 2045 particles without crash
- ✅ Probe grid updates successfully (no TDR timeout)
- ✅ RT lighting works correctly
- ✅ Visual quality matches 2044 particles (no artifacts at batch boundaries)

**If Still Crashes:**
- Try reducing PARTICLES_PER_BATCH from 2000 → 1000
- Try disabling probe grid entirely (test if RT lighting alone works)
- Consider alternative mitigations (reduce probe count, different traversal pattern)

---

**Last Updated:** 2025-11-04
**Status:** ~60% complete, core infrastructure done, need batch build functions + probe shader updates
