# Particle Batching Implementation - Phase 0.13.3

**Date:** 2025-11-04
**Status:** Implementation in progress
**Goal:** Mitigate probe grid 2045+ particle crash by batching particles into groups of ~2000

---

## Rationale

**Problem:** Probe grid crashes at 2045+ particles with GPU hang/TDR timeout (5-second pause → navy blue screen). High FPS (~150) until instant crash indicates hard threshold, not computational bottleneck.

**Failed Attempts:**
1. BVH power-of-2 padding (513 leaves) - Still crashes
2. RayQuery iteration limit (1000) - Still crashes
3. Explicit TLAS barrier - Still crashes
4. 1 ray per probe (extreme reduction) - Still crashes

**Root Cause Theory:** Driver bug or hardware limit at specific particle counts (2045, and possibly others). NOT a performance issue.

**Solution:** Particle batching - split particles into groups of ~2000, each with separate BLAS/TLAS. Probe grid traces against all batches sequentially.

---

## Architecture Design

### Batch Structure

```cpp
struct AccelerationBatch {
    ComPtr<ID3D12Resource> aabbBuffer;      // Per-batch AABB buffer
    ComPtr<ID3D12Resource> blas;            // Bottom-level AS
    ComPtr<ID3D12Resource> tlas;            // Top-level AS
    ComPtr<ID3D12Resource> blasScratch;     // Build scratch memory
    ComPtr<ID3D12Resource> tlasScratch;     // Build scratch memory
    ComPtr<ID3D12Resource> instanceDesc;    // TLAS instance descriptor
    uint32_t startIndex;                    // First particle in batch
    uint32_t count;                         // Particles in this batch
    size_t blasSize;                        // BLAS memory size
    size_t tlasSize;                        // TLAS memory size
};
```

### Batch Calculation

```cpp
static constexpr uint32_t PARTICLES_PER_BATCH = 2000;

// Examples:
// 2045 particles → 2 batches (2000 + 45)
// 4000 particles → 2 batches (2000 + 2000)
// 10000 particles → 5 batches (2000 × 5)
// 100000 particles → 50 batches (2000 × 50)

uint32_t numBatches = (particleCount + PARTICLES_PER_BATCH - 1) / PARTICLES_PER_BATCH;
```

### Memory Impact

**Per-batch overhead:**
- AABB buffer: ~48 KB (2000 × 24 bytes)
- BLAS: ~50 KB (typical for 2000 procedural primitives)
- TLAS: ~2 KB (1 instance)
- Scratch: Transient (reused across batches)
- **Total per batch: ~100 KB**

**50 batches (100K particles):**
- 50 × 100 KB = 5 MB total overhead
- Negligible with 8GB VRAM

---

## Implementation Plan

### Phase 1: RTLightingSystem Modifications

#### 1.1 Header Changes (RTLightingSystem_RayQuery.h)

**Add AccelerationBatch struct** (lines 92-104):
```cpp
struct AccelerationBatch {
    Microsoft::WRL::ComPtr<ID3D12Resource> aabbBuffer;
    Microsoft::WRL::ComPtr<ID3D12Resource> blas;
    Microsoft::WRL::ComPtr<ID3D12Resource> tlas;
    Microsoft::WRL::ComPtr<ID3D12Resource> blasScratch;
    Microsoft::WRL::ComPtr<ID3D12Resource> tlasScratch;
    Microsoft::WRL::ComPtr<ID3D12Resource> instanceDesc;
    uint32_t startIndex;
    uint32_t count;
    size_t blasSize;
    size_t tlasSize;
};
```

**Add batch helpers** (lines 112-115):
```cpp
bool CreateBatchedAccelerationStructures();
void GenerateAABBsForBatch(ID3D12GraphicsCommandList4* cmdList, ID3D12Resource* particleBuffer, AccelerationBatch& batch);
void BuildBatchBLAS(ID3D12GraphicsCommandList4* cmdList, AccelerationBatch& batch);
void BuildBatchTLAS(ID3D12GraphicsCommandList4* cmdList, AccelerationBatch& batch);
```

**Add batch members** (lines 175-179):
```cpp
static constexpr uint32_t PARTICLES_PER_BATCH = 2000;
bool m_useBatching = true;
std::vector<AccelerationBatch> m_batches;
```

**Add batch accessors** (lines 65-69):
```cpp
uint32_t GetBatchCount() const { return static_cast<uint32_t>(m_batches.size()); }
ID3D12Resource* GetBatchTLAS(uint32_t batchIdx) const {
    return (batchIdx < m_batches.size()) ? m_batches[batchIdx].tlas.Get() : nullptr;
}
```

#### 1.2 Implementation Changes (RTLightingSystem_RayQuery.cpp)

**Modify Initialize():**
```cpp
bool RTLightingSystem_RayQuery::Initialize(...) {
    // ... existing code ...

    // Choose batching vs monolithic based on particle count
    if (m_useBatching && m_particleCount > PARTICLES_PER_BATCH) {
        if (!CreateBatchedAccelerationStructures()) {
            LOG_ERROR("Failed to create batched acceleration structures");
            return false;
        }
    } else {
        if (!CreateAccelerationStructures()) {
            LOG_ERROR("Failed to create acceleration structures");
            return false;
        }
    }

    return true;
}
```

**Implement CreateBatchedAccelerationStructures():**
```cpp
bool RTLightingSystem_RayQuery::CreateBatchedAccelerationStructures() {
    uint32_t numBatches = (m_particleCount + PARTICLES_PER_BATCH - 1) / PARTICLES_PER_BATCH;
    m_batches.resize(numBatches);

    LOG_INFO("Creating batched acceleration structures:");
    LOG_INFO("  Total particles: {}", m_particleCount);
    LOG_INFO("  Particles per batch: {}", PARTICLES_PER_BATCH);
    LOG_INFO("  Number of batches: {}", numBatches);

    for (uint32_t i = 0; i < numBatches; i++) {
        AccelerationBatch& batch = m_batches[i];
        batch.startIndex = i * PARTICLES_PER_BATCH;
        batch.count = std::min(PARTICLES_PER_BATCH, m_particleCount - batch.startIndex);

        LOG_INFO("  Batch {}: particles [{}, {})", i, batch.startIndex, batch.startIndex + batch.count);

        // Create AABB buffer for this batch
        size_t aabbBufferSize = (batch.count + 4) * 24;  // +4 for power-of-2 padding
        ResourceManager::BufferDesc aabbDesc = {};
        aabbDesc.size = aabbBufferSize;
        aabbDesc.heapType = D3D12_HEAP_TYPE_DEFAULT;
        aabbDesc.flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        aabbDesc.initialState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

        std::string batchName = "ParticleAABBs_Batch" + std::to_string(i);
        batch.aabbBuffer = m_resources->CreateBuffer(batchName.c_str(), aabbDesc);
        if (!batch.aabbBuffer) {
            LOG_ERROR("Failed to create AABB buffer for batch {}", i);
            return false;
        }

        // Get BLAS/TLAS size requirements for this batch
        D3D12_RAYTRACING_GEOMETRY_DESC geomDesc = {};
        geomDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
        geomDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
        geomDesc.AABBs.AABBCount = batch.count;
        geomDesc.AABBs.AABBs.StrideInBytes = 24;

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blasInputs = {};
        blasInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        blasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
        blasInputs.NumDescs = 1;
        blasInputs.pGeometryDescs = &geomDesc;
        blasInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO blasPrebuild = {};
        m_device->GetDevice()->GetRaytracingAccelerationStructurePrebuildInfo(&blasInputs, &blasPrebuild);

        batch.blasSize = blasPrebuild.ResultDataMaxSizeInBytes;

        // Create BLAS buffer
        ResourceManager::BufferDesc blasDesc = {};
        blasDesc.size = batch.blasSize;
        blasDesc.heapType = D3D12_HEAP_TYPE_DEFAULT;
        blasDesc.flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        blasDesc.initialState = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;

        batchName = "BLAS_Batch" + std::to_string(i);
        batch.blas = m_resources->CreateBuffer(batchName.c_str(), blasDesc);
        if (!batch.blas) {
            LOG_ERROR("Failed to create BLAS for batch {}", i);
            return false;
        }

        // Create BLAS scratch buffer
        ResourceManager::BufferDesc scratchDesc = {};
        scratchDesc.size = blasPrebuild.ScratchDataSizeInBytes;
        scratchDesc.heapType = D3D12_HEAP_TYPE_DEFAULT;
        scratchDesc.flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        scratchDesc.initialState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

        batchName = "BLAS_Scratch_Batch" + std::to_string(i);
        batch.blasScratch = m_resources->CreateBuffer(batchName.c_str(), scratchDesc);
        if (!batch.blasScratch) {
            LOG_ERROR("Failed to create BLAS scratch for batch {}", i);
            return false;
        }

        // Create TLAS for this batch (similar to above)
        // ... TLAS creation code ...
    }

    // Create shared lighting output buffer (not batched)
    size_t lightingBufferSize = m_particleCount * 16;
    ResourceManager::BufferDesc lightingDesc = {};
    lightingDesc.size = lightingBufferSize;
    lightingDesc.heapType = D3D12_HEAP_TYPE_DEFAULT;
    lightingDesc.flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    lightingDesc.initialState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

    m_lightingBuffer = m_resources->CreateBuffer("RTLightingOutput", lightingDesc);
    if (!m_lightingBuffer) {
        LOG_ERROR("Failed to create lighting output buffer");
        return false;
    }

    LOG_INFO("Batched acceleration structures created successfully");
    LOG_INFO("  Total memory: {:.2f} MB", (numBatches * 100) / 1024.0f);
    return true;
}
```

**Modify ComputeLighting() to build all batches:**
```cpp
void RTLightingSystem_RayQuery::ComputeLighting(...) {
    if (m_useBatching && !m_batches.empty()) {
        // Build all batches
        for (auto& batch : m_batches) {
            GenerateAABBsForBatch(cmdList, particleBuffer, batch);
            BuildBatchBLAS(cmdList, batch);
            BuildBatchTLAS(cmdList, batch);
        }
    } else {
        // Monolithic path (existing code)
        GenerateAABBs(cmdList, particleBuffer);
        BuildBLAS(cmdList);
        BuildTLAS(cmdList);
    }

    // Dispatch lighting (unchanged - writes to shared output buffer)
    DispatchRayQueryLighting(cmdList, particleBuffer, cameraPosition);
}
```

**Implement GenerateAABBsForBatch():**
```cpp
void RTLightingSystem_RayQuery::GenerateAABBsForBatch(
    ID3D12GraphicsCommandList4* cmdList,
    ID3D12Resource* particleBuffer,
    AccelerationBatch& batch) {

    cmdList->SetPipelineState(m_aabbGenPSO.Get());
    cmdList->SetComputeRootSignature(m_aabbGenRootSig.Get());

    // Constants for this batch
    AABBConstants constants = {};
    constants.particleCount = batch.count;  // Only this batch's particles
    constants.particleRadius = m_particleRadius;
    constants.enableAdaptiveRadius = m_enableAdaptiveRadius ? 1 : 0;
    // ... other constants ...

    cmdList->SetComputeRoot32BitConstants(0, sizeof(AABBConstants) / 4, &constants, 0);

    // Bind particle buffer at batch start offset
    D3D12_GPU_VIRTUAL_ADDRESS particleAddr = particleBuffer->GetGPUVirtualAddress();
    particleAddr += batch.startIndex * 128;  // 128 bytes per particle
    cmdList->SetComputeRootShaderResourceView(1, particleAddr);

    // Bind batch AABB buffer
    cmdList->SetComputeRootUnorderedAccessView(2, batch.aabbBuffer->GetGPUVirtualAddress());

    // Dispatch
    uint32_t numGroups = (batch.count + 255) / 256;
    cmdList->Dispatch(numGroups, 1, 1);

    // UAV barrier
    CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(batch.aabbBuffer.Get());
    cmdList->ResourceBarrier(1, &barrier);
}
```

---

### Phase 2: Probe Shader Modifications

#### 2.1 Update Probe Root Signature

**Current (update_probes.hlsl):**
```hlsl
// t2: Particle TLAS (SRV, raytracing acceleration structure)
RaytracingAccelerationStructure g_particleTLAS : register(t2);
```

**New (multi-TLAS support):**
```hlsl
// Option A: Fixed array (simpler, but limited)
RaytracingAccelerationStructure g_particleTLAS[8] : register(t2);  // Max 8 batches (16K particles)
uint g_numBatches : register(b0);  // In constants

// Option B: Descriptor table (more flexible, but complex)
// Use descriptor heap with unbounded array
```

#### 2.2 Update Probe Update Loop

**Current:**
```hlsl
RayQuery<RAY_FLAG_NONE> q;
q.TraceRayInline(g_particleTLAS, RAY_FLAG_NONE, 0xFF, ray);

while (q.Proceed() && iterationCount < MAX_ITERATIONS) {
    // ... intersection testing
}
```

**New (multi-batch tracing):**
```hlsl
// Trace against all batches
for (uint batchIdx = 0; batchIdx < g_numBatches; batchIdx++) {
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(g_particleTLAS[batchIdx], RAY_FLAG_NONE, 0xFF, ray);

    uint iterationCount = 0;
    const uint MAX_ITERATIONS = 1000;

    while (q.Proceed() && iterationCount < MAX_ITERATIONS) {
        iterationCount++;

        if (q.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
            uint particleIdx = q.CandidatePrimitiveIndex();

            // Adjust index for batch offset
            uint globalParticleIdx = (batchIdx * PARTICLES_PER_BATCH) + particleIdx;

            if (globalParticleIdx < g_particleCount) {
                Particle particle = g_particles[globalParticleIdx];

                // ... existing intersection test ...
            }
        }
    }

    // Check for hit in this batch
    if (q.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT) {
        uint localIdx = q.CommittedPrimitiveIndex();
        uint globalIdx = (batchIdx * PARTICLES_PER_BATCH) + localIdx;

        if (globalIdx < g_particleCount) {
            Particle particle = g_particles[globalIdx];
            totalIrradiance += ComputeParticleLighting(...);
        }
    }
}
```

#### 2.3 Update ProbeGridSystem.cpp Binding

**ProbeGridSystem::UpdateProbes():**
```cpp
void ProbeGridSystem::UpdateProbes(...) {
    // ... existing setup ...

    // Root parameter 0: b0 - ProbeUpdateConstants
    constants.numBatches = rtLightingSystem->GetBatchCount();
    // ... upload constants ...

    // Root parameter 3: t2-t9 - Particle TLAS array (up to 8 batches)
    for (uint32_t i = 0; i < constants.numBatches; i++) {
        ID3D12Resource* batchTLAS = rtLightingSystem->GetBatchTLAS(i);
        D3D12_GPU_VIRTUAL_ADDRESS tlasAddr = batchTLAS->GetGPUVirtualAddress();
        commandList->SetComputeRootShaderResourceView(3 + i, tlasAddr);
    }

    // ... dispatch ...
}
```

---

## Testing Plan

### Phase 1: Single Batch Verification
1. Test at 1000 particles (1 batch) - should work as before
2. Verify BLAS/TLAS builds correctly
3. Verify RT lighting output matches non-batched version

### Phase 2: Multi-Batch Verification
1. Test at 2045 particles (2 batches: 2000 + 45)
2. **Expected: NO CRASH** (avoids driver bug threshold)
3. Verify probe grid updates without TDR timeout
4. Check for visual artifacts at batch boundaries

### Phase 3: Scaling Test
1. Test at 4000 particles (2 batches: 2000 + 2000)
2. Test at 10,000 particles (5 batches)
3. Test at 100,000 particles (50 batches)
4. Monitor FPS and memory usage

### Phase 4: Quality Verification
1. Compare screenshots: batched vs non-batched (at 2044 particles)
2. Verify probe lighting is consistent
3. Check for no visual discontinuities

---

## Performance Expectations

| Particle Count | Batches | Expected FPS | Memory Overhead |
|----------------|---------|--------------|-----------------|
| 2044 (current limit) | 2 | 150 FPS | ~200 KB |
| 10,000 | 5 | 140 FPS | ~500 KB |
| 50,000 | 25 | 90 FPS | ~2.5 MB |
| 100,000 | 50 | 60 FPS | ~5 MB |

**Key Insight:** Batching overhead is minimal (<1% FPS impact). Main benefit is avoiding hard threshold crashes.

---

## Rollback Plan

If batching doesn't solve the crash:

1. Keep batching implementation (useful for 100K+ particles anyway)
2. Try alternative mitigations:
   - Reduce probe count (32³ → 16³)
   - Reduce probe update frequency (4 frames → 8 frames)
   - Use TraceRay() instead of RayQuery (full DXR pipeline)

---

## Success Criteria

✅ **Full Success:** No crash at 2045+ particles with batching enabled
⚠️ **Partial Success:** Crash threshold moves higher (e.g., from 2045 → 4000)
❌ **Failure:** Still crashes at 2045 particles with batching

---

**Last Updated:** 2025-11-04
**Status:** Implementation in progress
**Estimated Time:** 4-6 hours
