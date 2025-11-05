#include "RTLightingSystem_RayQuery.h"
#include "../core/Device.h"
#include "../utils/ResourceManager.h"
#include "../utils/Logger.h"
#include "../utils/d3dx12/d3dx12.h"
#include <d3dcompiler.h>
#include <fstream>

#pragma comment(lib, "d3dcompiler.lib")

RTLightingSystem_RayQuery::~RTLightingSystem_RayQuery() {
    Shutdown();
}

bool RTLightingSystem_RayQuery::Initialize(Device* device, ResourceManager* resources, uint32_t particleCount) {
    m_device = device;
    m_resources = resources;
    m_particleCount = particleCount;

    LOG_INFO("Initializing RayQuery RT Lighting System...");
    LOG_INFO("  Particles: {}", particleCount);
    LOG_INFO("  Rays per particle: {}", m_raysPerParticle);

    if (!LoadShaders()) {
        LOG_ERROR("Failed to load RT shaders");
        return false;
    }

    if (!CreateRootSignatures()) {
        LOG_ERROR("Failed to create root signatures");
        return false;
    }

    if (!CreatePipelineStates()) {
        LOG_ERROR("Failed to create pipeline states");
        return false;
    }

    if (!CreateAccelerationStructures()) {
        LOG_ERROR("Failed to create acceleration structures");
        return false;
    }

    LOG_INFO("RayQuery RT Lighting System initialized successfully");
    return true;
}

void RTLightingSystem_RayQuery::Shutdown() {
    m_aabbGenShader.Reset();
    m_rayQueryLightingShader.Reset();
    m_aabbGenRootSig.Reset();
    m_rayQueryLightingRootSig.Reset();
    m_aabbGenPSO.Reset();
    m_rayQueryLightingPSO.Reset();
    m_aabbBuffer.Reset();
    m_lightingBuffer.Reset();
    m_bottomLevelAS.Reset();
    m_topLevelAS.Reset();
    m_blasScratch.Reset();
    m_tlasScratch.Reset();
    m_instanceDescsBuffer.Reset();
}

bool RTLightingSystem_RayQuery::LoadShaders() {
    // Load AABB generation shader
    std::ifstream aabbFile("shaders/dxr/generate_particle_aabbs.dxil", std::ios::binary);
    if (!aabbFile) {
        LOG_ERROR("Failed to open generate_particle_aabbs.dxil");
        return false;
    }

    std::vector<char> aabbData((std::istreambuf_iterator<char>(aabbFile)), std::istreambuf_iterator<char>());
    HRESULT hr = D3DCreateBlob(aabbData.size(), &m_aabbGenShader);
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create blob for AABB shader");
        return false;
    }
    memcpy(m_aabbGenShader->GetBufferPointer(), aabbData.data(), aabbData.size());

    // Load RayQuery lighting shader
    std::ifstream lightingFile("shaders/dxr/particle_raytraced_lighting_cs.dxil", std::ios::binary);
    if (!lightingFile) {
        LOG_ERROR("Failed to open particle_raytraced_lighting_cs.dxil");
        return false;
    }

    std::vector<char> lightingData((std::istreambuf_iterator<char>(lightingFile)), std::istreambuf_iterator<char>());
    hr = D3DCreateBlob(lightingData.size(), &m_rayQueryLightingShader);
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create blob for lighting shader");
        return false;
    }
    memcpy(m_rayQueryLightingShader->GetBufferPointer(), lightingData.data(), lightingData.size());

    LOG_INFO("RT shaders loaded successfully");
    return true;
}

bool RTLightingSystem_RayQuery::CreateRootSignatures() {
    HRESULT hr;

    // AABB Generation Root Signature
    // b0: AABBConstants
    // t0: StructuredBuffer<Particle> particles
    // u0: RWStructuredBuffer<AABB> particleAABBs
    {
        CD3DX12_ROOT_PARAMETER1 rootParams[3];
        rootParams[0].InitAsConstants(12, 0);  // b0: AABBConstants (12 DWORDs - Phase 1.5: added particleOffset + padding1)
        rootParams[1].InitAsShaderResourceView(0);  // t0: particles
        rootParams[2].InitAsUnorderedAccessView(0);  // u0: AABBs

        CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSigDesc;
        rootSigDesc.Init_1_1(3, rootParams, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);

        Microsoft::WRL::ComPtr<ID3DBlob> signature, error;
        hr = D3DX12SerializeVersionedRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1_1, &signature, &error);
        if (FAILED(hr)) {
            if (error) {
                LOG_ERROR("AABB root signature serialization failed: {}", (char*)error->GetBufferPointer());
            }
            return false;
        }

        hr = m_device->GetDevice()->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(),
                                                         IID_PPV_ARGS(&m_aabbGenRootSig));
        if (FAILED(hr)) {
            LOG_ERROR("Failed to create AABB root signature");
            return false;
        }
    }

    // RayQuery Lighting Root Signature
    // b0: LightingConstants (EXPANDED for dynamic emission - 14 DWORDs)
    // t0: StructuredBuffer<Particle> g_particles
    // t1: RaytracingAccelerationStructure g_particleBVH
    // u0: RWBuffer<float4> g_particleLighting
    {
        CD3DX12_ROOT_PARAMETER1 rootParams[4];
        rootParams[0].InitAsConstants(14, 0);  // b0: LightingConstants (14 DWORDs - was 4, expanded for emission)
        rootParams[1].InitAsShaderResourceView(0);  // t0: particles
        rootParams[2].InitAsShaderResourceView(1);  // t1: TLAS
        rootParams[3].InitAsUnorderedAccessView(0);  // u0: lighting buffer

        CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSigDesc;
        rootSigDesc.Init_1_1(4, rootParams, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);

        Microsoft::WRL::ComPtr<ID3DBlob> signature, error;
        hr = D3DX12SerializeVersionedRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1_1, &signature, &error);
        if (FAILED(hr)) {
            if (error) {
                LOG_ERROR("Lighting root signature serialization failed: {}", (char*)error->GetBufferPointer());
            }
            return false;
        }

        hr = m_device->GetDevice()->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(),
                                                         IID_PPV_ARGS(&m_rayQueryLightingRootSig));
        if (FAILED(hr)) {
            LOG_ERROR("Failed to create lighting root signature");
            return false;
        }
    }

    LOG_INFO("Root signatures created successfully");
    return true;
}

bool RTLightingSystem_RayQuery::CreatePipelineStates() {
    HRESULT hr;

    // AABB Generation PSO
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = m_aabbGenRootSig.Get();
        psoDesc.CS = CD3DX12_SHADER_BYTECODE(m_aabbGenShader.Get());

        hr = m_device->GetDevice()->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_aabbGenPSO));
        if (FAILED(hr)) {
            LOG_ERROR("Failed to create AABB generation PSO");
            return false;
        }
    }

    // RayQuery Lighting PSO
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = m_rayQueryLightingRootSig.Get();
        psoDesc.CS = CD3DX12_SHADER_BYTECODE(m_rayQueryLightingShader.Get());

        hr = m_device->GetDevice()->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_rayQueryLightingPSO));
        if (FAILED(hr)) {
            LOG_ERROR("Failed to create RayQuery lighting PSO (HRESULT: 0x{:08X})", static_cast<uint32_t>(hr));
            LOG_ERROR("  Shader size: {} bytes", m_rayQueryLightingShader->GetBufferSize());
            LOG_ERROR("  Root signature: {}", m_rayQueryLightingRootSig.Get() ? "Valid" : "NULL");
            return false;
        }
    }

    LOG_INFO("Pipeline states created successfully");
    return true;
}

// ============================================================================
// Dual Acceleration Structure Helpers (Phase 1)
// ============================================================================

bool RTLightingSystem_RayQuery::CreateAccelerationStructureSet(
    AccelerationStructureSet& asSet,
    uint32_t particleCount,
    const std::string& namePrefix) {

    if (particleCount == 0) {
        LOG_INFO("{}: Skipping creation (0 particles)", namePrefix);
        return true;  // Not an error, just nothing to do
    }

    LOG_INFO("{}: Creating AS resources for {} particles", namePrefix, particleCount);

    asSet.startParticle = 0;  // Will be set by caller
    asSet.particleCount = particleCount;

    // Create AABB buffer
    {
        size_t aabbBufferSize = particleCount * 24;  // 6 floats = 24 bytes

        ResourceManager::BufferDesc desc = {};
        desc.size = aabbBufferSize;
        desc.heapType = D3D12_HEAP_TYPE_DEFAULT;
        desc.flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        desc.initialState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

        asSet.aabbBuffer = m_resources->CreateBuffer(namePrefix + "_AABBs", desc);
        if (!asSet.aabbBuffer) {
            LOG_ERROR("{}: Failed to create AABB buffer", namePrefix);
            return false;
        }
    }

    // Create lighting output buffer
    {
        size_t lightingBufferSize = particleCount * 16;  // float4 = 16 bytes

        ResourceManager::BufferDesc desc = {};
        desc.size = lightingBufferSize;
        desc.heapType = D3D12_HEAP_TYPE_DEFAULT;
        desc.flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        desc.initialState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

        asSet.lightingBuffer = m_resources->CreateBuffer(namePrefix + "_Lighting", desc);
        if (!asSet.lightingBuffer) {
            LOG_ERROR("{}: Failed to create lighting buffer", namePrefix);
            return false;
        }
    }

    // Get BLAS size requirements
    {
        D3D12_RAYTRACING_GEOMETRY_DESC geomDesc = {};
        geomDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
        geomDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
        geomDesc.AABBs.AABBCount = particleCount;
        geomDesc.AABBs.AABBs.StrideInBytes = 24;

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blasInputs = {};
        blasInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        blasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
        blasInputs.NumDescs = 1;
        blasInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        blasInputs.pGeometryDescs = &geomDesc;

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO blasPrebuildInfo = {};
        m_device->GetDevice()->GetRaytracingAccelerationStructurePrebuildInfo(&blasInputs, &blasPrebuildInfo);

        asSet.blasSize = blasPrebuildInfo.ResultDataMaxSizeInBytes;
        size_t blasScratchSize = blasPrebuildInfo.ScratchDataSizeInBytes;

        LOG_INFO("{}: BLAS size={} bytes, scratch={} bytes", namePrefix, asSet.blasSize, blasScratchSize);

        // Create BLAS buffer
        D3D12_HEAP_PROPERTIES heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
        D3D12_RESOURCE_DESC blasDesc = CD3DX12_RESOURCE_DESC::Buffer(asSet.blasSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        HRESULT hr = m_device->GetDevice()->CreateCommittedResource(
            &heapProps, D3D12_HEAP_FLAG_NONE, &blasDesc,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            nullptr, IID_PPV_ARGS(&asSet.blas));

        if (FAILED(hr)) {
            LOG_ERROR("{}: Failed to create BLAS buffer", namePrefix);
            return false;
        }

        // Create BLAS scratch buffer
        D3D12_RESOURCE_DESC scratchDesc = CD3DX12_RESOURCE_DESC::Buffer(blasScratchSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        hr = m_device->GetDevice()->CreateCommittedResource(
            &heapProps, D3D12_HEAP_FLAG_NONE, &scratchDesc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr, IID_PPV_ARGS(&asSet.blasScratch));

        if (FAILED(hr)) {
            LOG_ERROR("{}: Failed to create BLAS scratch buffer", namePrefix);
            return false;
        }
    }

    // Get TLAS size requirements
    {
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlasInputs = {};
        tlasInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        tlasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
        tlasInputs.NumDescs = 1;  // Single BLAS instance
        tlasInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO tlasPrebuildInfo = {};
        m_device->GetDevice()->GetRaytracingAccelerationStructurePrebuildInfo(&tlasInputs, &tlasPrebuildInfo);

        asSet.tlasSize = tlasPrebuildInfo.ResultDataMaxSizeInBytes;
        size_t tlasScratchSize = tlasPrebuildInfo.ScratchDataSizeInBytes;

        LOG_INFO("{}: TLAS size={} bytes, scratch={} bytes", namePrefix, asSet.tlasSize, tlasScratchSize);

        // Create TLAS buffer
        D3D12_HEAP_PROPERTIES heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
        D3D12_RESOURCE_DESC tlasDesc = CD3DX12_RESOURCE_DESC::Buffer(asSet.tlasSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        HRESULT hr = m_device->GetDevice()->CreateCommittedResource(
            &heapProps, D3D12_HEAP_FLAG_NONE, &tlasDesc,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            nullptr, IID_PPV_ARGS(&asSet.tlas));

        if (FAILED(hr)) {
            LOG_ERROR("{}: Failed to create TLAS buffer", namePrefix);
            return false;
        }

        // Create TLAS scratch buffer
        D3D12_RESOURCE_DESC scratchDesc = CD3DX12_RESOURCE_DESC::Buffer(tlasScratchSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        hr = m_device->GetDevice()->CreateCommittedResource(
            &heapProps, D3D12_HEAP_FLAG_NONE, &scratchDesc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr, IID_PPV_ARGS(&asSet.tlasScratch));

        if (FAILED(hr)) {
            LOG_ERROR("{}: Failed to create TLAS scratch buffer", namePrefix);
            return false;
        }

        // Create instance descs buffer
        D3D12_HEAP_PROPERTIES uploadHeapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
        D3D12_RESOURCE_DESC instanceDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(D3D12_RAYTRACING_INSTANCE_DESC));

        hr = m_device->GetDevice()->CreateCommittedResource(
            &uploadHeapProps, D3D12_HEAP_FLAG_NONE, &instanceDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr, IID_PPV_ARGS(&asSet.instanceDesc));

        if (FAILED(hr)) {
            LOG_ERROR("{}: Failed to create instance desc buffer", namePrefix);
            return false;
        }
    }

    LOG_INFO("{}: AS resources created successfully", namePrefix);
    return true;
}

bool RTLightingSystem_RayQuery::CreateAccelerationStructures() {
    // ========================================================================
    // PHASE 1: Dual Acceleration Structure Architecture
    // ========================================================================
    // Split particles into two groups to work around Ada Lovelace 2045 bug

    uint32_t probeGridCount = std::min(m_particleCount, PROBE_GRID_PARTICLE_LIMIT);
    uint32_t directRTCount = (m_particleCount > PROBE_GRID_PARTICLE_LIMIT)
                             ? (m_particleCount - PROBE_GRID_PARTICLE_LIMIT)
                             : 0;

    LOG_INFO("=== Dual AS Architecture ===");
    LOG_INFO("Total particles: {}", m_particleCount);
    LOG_INFO("Probe Grid AS: {} particles (0-{})", probeGridCount, probeGridCount - 1);
    LOG_INFO("Direct RT AS: {} particles ({}-{})", directRTCount, PROBE_GRID_PARTICLE_LIMIT, m_particleCount - 1);

    // Create Probe Grid AS (particles 0-2043)
    if (!CreateAccelerationStructureSet(m_probeGridAS, probeGridCount, "ProbeGridAS")) {
        LOG_ERROR("Failed to create Probe Grid acceleration structure set");
        return false;
    }
    m_probeGridAS.startParticle = 0;

    // Create Direct RT AS (particles 2044+)
    if (directRTCount > 0) {
        // CRITICAL BUG FIX: Allocate AABB buffer for TOTAL particles, not just overflow
        // GenerateAABBs_Dual() generates for ALL particles, BLAS reads from offset 2044
        // This wastes memory but prevents buffer overrun at 3922+ particles
        // TODO Phase 1.5: Add particle offset to shader for memory efficiency
        if (!CreateAccelerationStructureSet(m_directRTAS, m_particleCount, "DirectRTAS")) {
            LOG_ERROR("Failed to create Direct RT acceleration structure set");
            return false;
        }
        m_directRTAS.startParticle = PROBE_GRID_PARTICLE_LIMIT;
        m_directRTAS.particleCount = directRTCount;  // Override - actual count is less than buffer size
    } else {
        LOG_INFO("DirectRTAS: No overflow particles (total count <= 2044)");
    }

    // ========================================================================
    // LEGACY: Keep old monolithic AS for backward compatibility during migration
    // ========================================================================
    // TODO: Remove after full migration to dual AS architecture
    // Create AABB buffer
    {
        // Allocate +4 AABBs for power-of-2 BVH workaround (see BuildBLAS for details)
        // Need full leaf (4 primitives) to shift from 512 → 513 leaves
        size_t aabbBufferSize = (m_particleCount + 4) * 24;  // 6 floats (minXYZ, maxXYZ) = 24 bytes

        ResourceManager::BufferDesc desc = {};
        desc.size = aabbBufferSize;
        desc.heapType = D3D12_HEAP_TYPE_DEFAULT;
        desc.flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        desc.initialState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

        m_aabbBuffer = m_resources->CreateBuffer("ParticleAABBs", desc);
        if (!m_aabbBuffer) {
            LOG_ERROR("Failed to create AABB buffer");
            return false;
        }
    }

    // Create lighting output buffer
    {
        size_t lightingBufferSize = m_particleCount * 16;  // float4 = 16 bytes

        ResourceManager::BufferDesc desc = {};
        desc.size = lightingBufferSize;
        desc.heapType = D3D12_HEAP_TYPE_DEFAULT;
        desc.flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        desc.initialState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

        m_lightingBuffer = m_resources->CreateBuffer("RTLightingOutput", desc);
        if (!m_lightingBuffer) {
            LOG_ERROR("Failed to create lighting output buffer");
            return false;
        }
    }


    // Get BLAS/TLAS size requirements
    {
        D3D12_RAYTRACING_GEOMETRY_DESC geomDesc = {};
        geomDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
        geomDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
        geomDesc.AABBs.AABBCount = m_particleCount;
        geomDesc.AABBs.AABBs.StrideInBytes = 24;

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blasInputs = {};
        blasInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        blasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
        blasInputs.NumDescs = 1;
        blasInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        blasInputs.pGeometryDescs = &geomDesc;

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO blasPrebuildInfo = {};
        m_device->GetDevice()->GetRaytracingAccelerationStructurePrebuildInfo(&blasInputs, &blasPrebuildInfo);

        LOG_INFO("BLAS prebuild info: Result={} bytes, Scratch={} bytes",
                 blasPrebuildInfo.ResultDataMaxSizeInBytes,
                 blasPrebuildInfo.ScratchDataSizeInBytes);

        m_blasSize = blasPrebuildInfo.ResultDataMaxSizeInBytes;
        size_t blasScratchSize = blasPrebuildInfo.ScratchDataSizeInBytes;

        // Create BLAS buffer
        D3D12_HEAP_PROPERTIES heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
        D3D12_RESOURCE_DESC blasDesc = CD3DX12_RESOURCE_DESC::Buffer(m_blasSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        HRESULT hr = m_device->GetDevice()->CreateCommittedResource(
            &heapProps, D3D12_HEAP_FLAG_NONE, &blasDesc,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            nullptr, IID_PPV_ARGS(&m_bottomLevelAS));

        if (FAILED(hr)) {
            LOG_ERROR("Failed to create BLAS buffer");
            return false;
        }

        // Create BLAS scratch buffer
        D3D12_RESOURCE_DESC scratchDesc = CD3DX12_RESOURCE_DESC::Buffer(blasScratchSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        hr = m_device->GetDevice()->CreateCommittedResource(
            &heapProps, D3D12_HEAP_FLAG_NONE, &scratchDesc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr, IID_PPV_ARGS(&m_blasScratch));

        if (FAILED(hr)) {
            LOG_ERROR("Failed to create BLAS scratch buffer");
            return false;
        }
    }

    // Get TLAS size requirements
    {
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlasInputs = {};
        tlasInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        tlasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
        tlasInputs.NumDescs = 1;  // Single instance
        tlasInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO tlasPrebuildInfo = {};
        m_device->GetDevice()->GetRaytracingAccelerationStructurePrebuildInfo(&tlasInputs, &tlasPrebuildInfo);

        m_tlasSize = tlasPrebuildInfo.ResultDataMaxSizeInBytes;
        size_t tlasScratchSize = tlasPrebuildInfo.ScratchDataSizeInBytes;

        // Create TLAS buffer
        D3D12_HEAP_PROPERTIES heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
        D3D12_RESOURCE_DESC tlasDesc = CD3DX12_RESOURCE_DESC::Buffer(m_tlasSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        HRESULT hr = m_device->GetDevice()->CreateCommittedResource(
            &heapProps, D3D12_HEAP_FLAG_NONE, &tlasDesc,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            nullptr, IID_PPV_ARGS(&m_topLevelAS));

        if (FAILED(hr)) {
            LOG_ERROR("Failed to create TLAS buffer");
            return false;
        }

        // Create TLAS scratch buffer
        D3D12_RESOURCE_DESC scratchDesc = CD3DX12_RESOURCE_DESC::Buffer(tlasScratchSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        hr = m_device->GetDevice()->CreateCommittedResource(
            &heapProps, D3D12_HEAP_FLAG_NONE, &scratchDesc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr, IID_PPV_ARGS(&m_tlasScratch));

        if (FAILED(hr)) {
            LOG_ERROR("Failed to create TLAS scratch buffer");
            return false;
        }

        // Create instance descs buffer (2 instances for combined TLAS)
        D3D12_HEAP_PROPERTIES uploadHeapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
        D3D12_RESOURCE_DESC instanceDesc = CD3DX12_RESOURCE_DESC::Buffer(2 * sizeof(D3D12_RAYTRACING_INSTANCE_DESC));

        hr = m_device->GetDevice()->CreateCommittedResource(
            &uploadHeapProps, D3D12_HEAP_FLAG_NONE, &instanceDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr, IID_PPV_ARGS(&m_instanceDescsBuffer));

        if (FAILED(hr)) {
            LOG_ERROR("Failed to create instance descs buffer");
            return false;
        }
    }

    LOG_INFO("Acceleration structures created (BLAS: {} bytes, TLAS: {} bytes)", m_blasSize, m_tlasSize);
    return true;
}

// ============================================================================
// AABB Generation Functions
// ============================================================================

void RTLightingSystem_RayQuery::GenerateAABBs(ID3D12GraphicsCommandList4* cmdList, ID3D12Resource* particleBuffer) {
    // LEGACY: Monolithic AABB generation for all particles
    cmdList->SetPipelineState(m_aabbGenPSO.Get());
    cmdList->SetComputeRootSignature(m_aabbGenRootSig.Get());

    // Bind particle buffer (t0)
    D3D12_GPU_VIRTUAL_ADDRESS particleBufferAddress = particleBuffer->GetGPUVirtualAddress();
    cmdList->SetComputeRootShaderResourceView(1, particleBufferAddress);

    // Bind AABB output buffer (u0)
    D3D12_GPU_VIRTUAL_ADDRESS aabbBufferAddress = m_aabbBuffer->GetGPUVirtualAddress();
    cmdList->SetComputeRootUnorderedAccessView(2, aabbBufferAddress);

    // Bind constants (b0)
    struct AABBConstants {
        uint32_t particleCount;
        float particleRadius;
        uint32_t particleOffset;        // Phase 1.5: Start reading from this particle index
        uint32_t padding1;
        uint32_t enableAdaptiveRadius;
        float adaptiveInnerZone;
        float adaptiveOuterZone;
        float adaptiveInnerScale;
        float adaptiveOuterScale;
        float densityScaleMin;
        float densityScaleMax;
        float padding2;
    } constants = {
        m_particleCount,
        m_particleRadius,
        0,                              // Offset = 0 for legacy path
        0,
        m_enableAdaptiveRadius ? 1u : 0u,
        m_adaptiveInnerZone,
        m_adaptiveOuterZone,
        m_adaptiveInnerScale,
        m_adaptiveOuterScale,
        m_densityScaleMin,
        m_densityScaleMax,
        0.0f
    };
    cmdList->SetComputeRoot32BitConstants(0, sizeof(constants) / 4, &constants, 0);

    // Dispatch (256 threads per group)
    uint32_t threadGroups = (m_particleCount + 255) / 256;
    cmdList->Dispatch(threadGroups, 1, 1);

    // UAV barrier
    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_aabbBuffer.Get());
    cmdList->ResourceBarrier(1, &barrier);
}

void RTLightingSystem_RayQuery::GenerateAABBs_Dual(
    ID3D12GraphicsCommandList4* cmdList,
    ID3D12Resource* particleBuffer,
    uint32_t totalParticleCount) {

    // Dual AS architecture: Generate AABBs for both sets
    // Simple approach for Phase 1: Two separate dispatches with same particle buffer
    // Each AS reads its portion of particles and writes to its AABB buffer

    uint32_t probeGridCount = std::min(totalParticleCount, PROBE_GRID_PARTICLE_LIMIT);
    uint32_t directRTCount = (totalParticleCount > PROBE_GRID_PARTICLE_LIMIT)
                             ? (totalParticleCount - PROBE_GRID_PARTICLE_LIMIT)
                             : 0;

    cmdList->SetPipelineState(m_aabbGenPSO.Get());
    cmdList->SetComputeRootSignature(m_aabbGenRootSig.Get());

    // Bind particle buffer (t0) - same for both dispatches
    D3D12_GPU_VIRTUAL_ADDRESS particleBufferAddress = particleBuffer->GetGPUVirtualAddress();
    cmdList->SetComputeRootShaderResourceView(1, particleBufferAddress);

    // ========================================================================
    // Dispatch 1: Probe Grid AS (particles 0-2043)
    // ========================================================================
    if (probeGridCount > 0) {
        // Bind Probe Grid AABB buffer (u0)
        cmdList->SetComputeRootUnorderedAccessView(2, m_probeGridAS.aabbBuffer->GetGPUVirtualAddress());

        // Constants for probe grid
        struct AABBConstants {
            uint32_t particleCount;
            float particleRadius;
            uint32_t particleOffset;        // Phase 1.5: Start reading from this particle index
            uint32_t padding1;
            uint32_t enableAdaptiveRadius;
            float adaptiveInnerZone;
            float adaptiveOuterZone;
            float adaptiveInnerScale;
            float adaptiveOuterScale;
            float densityScaleMin;
            float densityScaleMax;
            float padding2;
        } probeGridConstants = {
            probeGridCount,
            m_particleRadius,
            0,                              // Offset = 0 (start at particle 0)
            0,
            m_enableAdaptiveRadius ? 1u : 0u,
            m_adaptiveInnerZone,
            m_adaptiveOuterZone,
            m_adaptiveInnerScale,
            m_adaptiveOuterScale,
            m_densityScaleMin,
            m_densityScaleMax,
            0.0f
        };
        cmdList->SetComputeRoot32BitConstants(0, sizeof(probeGridConstants) / 4, &probeGridConstants, 0);

        // Dispatch
        uint32_t threadGroups = (probeGridCount + 255) / 256;
        cmdList->Dispatch(threadGroups, 1, 1);

        // UAV barrier
        D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_probeGridAS.aabbBuffer.Get());
        cmdList->ResourceBarrier(1, &barrier);
    }

    // ========================================================================
    // Dispatch 2: Direct RT AS (particles 2044+)
    // ========================================================================
    // Phase 1.5 FIX: Shader now supports particleOffset - reads particles 2044-9999, not 0-9999
    // This eliminates duplicate particles and ensures runtime controls work uniformly
    if (directRTCount > 0) {
        // Bind Direct RT AABB buffer (u0)
        cmdList->SetComputeRootUnorderedAccessView(2, m_directRTAS.aabbBuffer->GetGPUVirtualAddress());

        // Constants for direct RT - reads overflow particles only (2044+)
        struct AABBConstants {
            uint32_t particleCount;
            float particleRadius;
            uint32_t particleOffset;        // Phase 1.5: Start reading from this particle index
            uint32_t padding1;
            uint32_t enableAdaptiveRadius;
            float adaptiveInnerZone;
            float adaptiveOuterZone;
            float adaptiveInnerScale;
            float adaptiveOuterScale;
            float densityScaleMin;
            float densityScaleMax;
            float padding2;
        } directRTConstants = {
            directRTCount,                   // CRITICAL FIX: Only overflow count, not total!
            m_particleRadius,
            PROBE_GRID_PARTICLE_LIMIT,       // Offset = 2044 (skip first 2044 particles)
            0,
            m_enableAdaptiveRadius ? 1u : 0u,
            m_adaptiveInnerZone,
            m_adaptiveOuterZone,
            m_adaptiveInnerScale,
            m_adaptiveOuterScale,
            m_densityScaleMin,
            m_densityScaleMax,
            0.0f
        };
        cmdList->SetComputeRoot32BitConstants(0, sizeof(directRTConstants) / 4, &directRTConstants, 0);

        // Dispatch - CRITICAL FIX: dispatch for overflow count only
        uint32_t threadGroups = (directRTCount + 255) / 256;
        cmdList->Dispatch(threadGroups, 1, 1);

        // UAV barrier
        D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_directRTAS.aabbBuffer.Get());
        cmdList->ResourceBarrier(1, &barrier);
    }
}

void RTLightingSystem_RayQuery::BuildBLAS_ForSet(
    ID3D12GraphicsCommandList4* cmdList,
    AccelerationStructureSet& asSet,
    uint32_t particleOffset) {

    if (asSet.particleCount == 0) {
        return;  // Nothing to build
    }

    // Build BLAS from procedural primitive AABBs
    D3D12_RAYTRACING_GEOMETRY_DESC geomDesc = {};
    geomDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
    geomDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    geomDesc.AABBs.AABBCount = asSet.particleCount;

    // CRITICAL: For Direct RT AS, we need to read AABBs starting at offset 2044
    // AABBs are stored contiguously: each AABB is 24 bytes (6 floats)
    D3D12_GPU_VIRTUAL_ADDRESS aabbStartAddress = asSet.aabbBuffer->GetGPUVirtualAddress()
                                                 + (particleOffset * 24);  // Offset in bytes
    geomDesc.AABBs.AABBs.StartAddress = aabbStartAddress;
    geomDesc.AABBs.AABBs.StrideInBytes = 24;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blasInputs = {};
    blasInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    blasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
    blasInputs.NumDescs = 1;
    blasInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    blasInputs.pGeometryDescs = &geomDesc;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC blasDesc = {};
    blasDesc.Inputs = blasInputs;
    blasDesc.DestAccelerationStructureData = asSet.blas->GetGPUVirtualAddress();
    blasDesc.ScratchAccelerationStructureData = asSet.blasScratch->GetGPUVirtualAddress();

    cmdList->BuildRaytracingAccelerationStructure(&blasDesc, 0, nullptr);

    // UAV barrier
    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(asSet.blas.Get());
    cmdList->ResourceBarrier(1, &barrier);
}

void RTLightingSystem_RayQuery::BuildTLAS_ForSet(
    ID3D12GraphicsCommandList4* cmdList,
    AccelerationStructureSet& asSet) {

    if (asSet.particleCount == 0) {
        return;  // Nothing to build
    }

    // Write instance desc
    D3D12_RAYTRACING_INSTANCE_DESC instanceDesc = {};
    instanceDesc.InstanceID = 0;
    instanceDesc.InstanceMask = 0xFF;
    instanceDesc.InstanceContributionToHitGroupIndex = 0;
    instanceDesc.Flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;

    // Identity transform
    instanceDesc.Transform[0][0] = 1.0f;
    instanceDesc.Transform[1][1] = 1.0f;
    instanceDesc.Transform[2][2] = 1.0f;

    instanceDesc.AccelerationStructure = asSet.blas->GetGPUVirtualAddress();

    // Upload instance desc
    void* mappedData = nullptr;
    asSet.instanceDesc->Map(0, nullptr, &mappedData);
    memcpy(mappedData, &instanceDesc, sizeof(D3D12_RAYTRACING_INSTANCE_DESC));
    asSet.instanceDesc->Unmap(0, nullptr);

    // Build TLAS
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlasInputs = {};
    tlasInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    tlasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
    tlasInputs.NumDescs = 1;
    tlasInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    tlasInputs.InstanceDescs = asSet.instanceDesc->GetGPUVirtualAddress();

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC tlasDesc = {};
    tlasDesc.Inputs = tlasInputs;
    tlasDesc.DestAccelerationStructureData = asSet.tlas->GetGPUVirtualAddress();
    tlasDesc.ScratchAccelerationStructureData = asSet.tlasScratch->GetGPUVirtualAddress();

    cmdList->BuildRaytracingAccelerationStructure(&tlasDesc, 0, nullptr);

    // UAV barrier
    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(asSet.tlas.Get());
    cmdList->ResourceBarrier(1, &barrier);
}

void RTLightingSystem_RayQuery::BuildCombinedTLAS(ID3D12GraphicsCommandList4* cmdList) {
    // Build a combined TLAS with TWO instances for full particle visibility
    // Instance 0: Probe Grid BLAS (particles 0-2043)
    // Instance 1: Direct RT BLAS (particles 2044+)

    // Create instance descriptors array (2 instances)
    D3D12_RAYTRACING_INSTANCE_DESC instances[2] = {};

    // Instance 0: Probe Grid BLAS
    instances[0].InstanceID = 0;
    instances[0].InstanceMask = 0xFF;
    instances[0].InstanceContributionToHitGroupIndex = 0;
    instances[0].Flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
    instances[0].Transform[0][0] = 1.0f;
    instances[0].Transform[1][1] = 1.0f;
    instances[0].Transform[2][2] = 1.0f;
    instances[0].AccelerationStructure = m_probeGridAS.blas->GetGPUVirtualAddress();

    // Instance 1: Direct RT BLAS
    instances[1].InstanceID = 1;
    instances[1].InstanceMask = 0xFF;
    instances[1].InstanceContributionToHitGroupIndex = 0;
    instances[1].Flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
    instances[1].Transform[0][0] = 1.0f;
    instances[1].Transform[1][1] = 1.0f;
    instances[1].Transform[2][2] = 1.0f;
    instances[1].AccelerationStructure = m_directRTAS.blas->GetGPUVirtualAddress();

    // Upload instance descriptors to legacy instance buffer
    void* mappedData = nullptr;
    m_instanceDescsBuffer->Map(0, nullptr, &mappedData);
    memcpy(mappedData, instances, sizeof(instances));
    m_instanceDescsBuffer->Unmap(0, nullptr);

    // Build combined TLAS (2 instances)
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlasInputs = {};
    tlasInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    tlasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
    tlasInputs.NumDescs = 2;  // TWO instances
    tlasInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    tlasInputs.InstanceDescs = m_instanceDescsBuffer->GetGPUVirtualAddress();

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC tlasDesc = {};
    tlasDesc.Inputs = tlasInputs;
    tlasDesc.DestAccelerationStructureData = m_topLevelAS->GetGPUVirtualAddress();
    tlasDesc.ScratchAccelerationStructureData = m_tlasScratch->GetGPUVirtualAddress();

    cmdList->BuildRaytracingAccelerationStructure(&tlasDesc, 0, nullptr);

    // UAV barrier
    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_topLevelAS.Get());
    cmdList->ResourceBarrier(1, &barrier);
}

// ============================================================================
// MONOLITHIC BUILD FUNCTIONS (Original Implementation)
// ============================================================================

void RTLightingSystem_RayQuery::BuildBLAS(ID3D12GraphicsCommandList4* cmdList) {
    // Build BLAS from procedural primitive AABBs
    D3D12_RAYTRACING_GEOMETRY_DESC geomDesc = {};
    geomDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
    geomDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    geomDesc.AABBs.AABBCount = m_particleCount;
    geomDesc.AABBs.AABBs.StartAddress = m_aabbBuffer->GetGPUVirtualAddress();
    geomDesc.AABBs.AABBs.StrideInBytes = 24;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blasInputs = {};
    blasInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    blasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
    blasInputs.NumDescs = 1;
    blasInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    blasInputs.pGeometryDescs = &geomDesc;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC blasDesc = {};
    blasDesc.Inputs = blasInputs;
    blasDesc.DestAccelerationStructureData = m_bottomLevelAS->GetGPUVirtualAddress();
    blasDesc.ScratchAccelerationStructureData = m_blasScratch->GetGPUVirtualAddress();

    cmdList->BuildRaytracingAccelerationStructure(&blasDesc, 0, nullptr);

    // UAV barrier
    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_bottomLevelAS.Get());
    cmdList->ResourceBarrier(1, &barrier);
}

void RTLightingSystem_RayQuery::BuildTLAS(ID3D12GraphicsCommandList4* cmdList) {
    // Write instance desc
    D3D12_RAYTRACING_INSTANCE_DESC instanceDesc = {};
    instanceDesc.InstanceID = 0;
    instanceDesc.InstanceMask = 0xFF;
    instanceDesc.InstanceContributionToHitGroupIndex = 0;
    instanceDesc.Flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
    instanceDesc.AccelerationStructure = m_bottomLevelAS->GetGPUVirtualAddress();

    // Identity transform
    instanceDesc.Transform[0][0] = 1.0f;
    instanceDesc.Transform[1][1] = 1.0f;
    instanceDesc.Transform[2][2] = 1.0f;

    // Upload instance desc
    void* mapped = nullptr;
    m_instanceDescsBuffer->Map(0, nullptr, &mapped);
    memcpy(mapped, &instanceDesc, sizeof(instanceDesc));
    m_instanceDescsBuffer->Unmap(0, nullptr);

    // Build inputs
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlasInputs = {};
    tlasInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    tlasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
    tlasInputs.NumDescs = 1;
    tlasInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    tlasInputs.InstanceDescs = m_instanceDescsBuffer->GetGPUVirtualAddress();

    // Build desc
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC tlasDesc = {};
    tlasDesc.Inputs = tlasInputs;
    tlasDesc.DestAccelerationStructureData = m_topLevelAS->GetGPUVirtualAddress();
    tlasDesc.ScratchAccelerationStructureData = m_tlasScratch->GetGPUVirtualAddress();

    cmdList->BuildRaytracingAccelerationStructure(&tlasDesc, 0, nullptr);

    // UAV barrier
    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_topLevelAS.Get());
    cmdList->ResourceBarrier(1, &barrier);
}

void RTLightingSystem_RayQuery::DispatchRayQueryLighting(ID3D12GraphicsCommandList4* cmdList, ID3D12Resource* particleBuffer, const DirectX::XMFLOAT3& cameraPosition) {
    cmdList->SetPipelineState(m_rayQueryLightingPSO.Get());
    cmdList->SetComputeRootSignature(m_rayQueryLightingRootSig.Get());

    // Increment frame counter for temporal effects
    m_frameCount++;

    // Set root parameters (expanded for dynamic emission)
    LightingConstants constants = {
        m_particleCount,
        m_raysPerParticle,
        m_maxLightingDistance,
        m_lightingIntensity,
        cameraPosition,
        m_frameCount,
        m_emissionStrength,
        m_emissionThreshold,
        m_rtSuppression,
        m_temporalRate
    };
    // 14 DWORDs total: 4 (original) + 3 (cameraPos) + 7 (new params) = 14
    cmdList->SetComputeRoot32BitConstants(0, 14, &constants, 0);
    cmdList->SetComputeRootShaderResourceView(1, particleBuffer->GetGPUVirtualAddress());
    cmdList->SetComputeRootShaderResourceView(2, m_topLevelAS->GetGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(3, m_lightingBuffer->GetGPUVirtualAddress());

    // Dispatch (64 threads per group)
    uint32_t threadGroups = (m_particleCount + 63) / 64;
    cmdList->Dispatch(threadGroups, 1, 1);

    // UAV barrier
    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_lightingBuffer.Get());
    cmdList->ResourceBarrier(1, &barrier);
}

void RTLightingSystem_RayQuery::ComputeLighting(ID3D12GraphicsCommandList4* cmdList,
                                                ID3D12Resource* particleBuffer,
                                                uint32_t particleCount,
                                                const DirectX::XMFLOAT3& cameraPosition) {
    m_particleCount = particleCount;
    m_frameCount++;

    // ========================================================================
    // PHASE 1: Dual Acceleration Structure Pipeline (ACTIVE)
    // ========================================================================
    // Workaround for Ada Lovelace 2045 particle crash bug

    uint32_t probeGridCount = std::min(particleCount, PROBE_GRID_PARTICLE_LIMIT);
    uint32_t directRTCount = (particleCount > PROBE_GRID_PARTICLE_LIMIT)
                             ? (particleCount - PROBE_GRID_PARTICLE_LIMIT)
                             : 0;

    // 1. Generate AABBs for both AS sets
    GenerateAABBs_Dual(cmdList, particleBuffer, particleCount);

    // 2. Build Probe Grid AS (particles 0-2043)
    if (probeGridCount > 0) {
        BuildBLAS_ForSet(cmdList, m_probeGridAS, 0);  // No offset for first 2044 particles
        BuildTLAS_ForSet(cmdList, m_probeGridAS);
    }

    // 3. Build Direct RT AS (particles 2044+)
    if (directRTCount > 0) {
        BuildBLAS_ForSet(cmdList, m_directRTAS, PROBE_GRID_PARTICLE_LIMIT);  // Offset to skip first 2044
        BuildTLAS_ForSet(cmdList, m_directRTAS);
    }

    // ========================================================================
    // COMBINED TLAS (Phase 1 - All Particles Visible)
    // ========================================================================
    // Build a COMBINED TLAS with TWO instances:
    //   Instance 0: Probe Grid BLAS (particles 0-2043)
    //   Instance 1: Direct RT BLAS (particles 2044+)
    // This allows Gaussian renderer to trace ONE TLAS but see ALL particles
    // Avoids 2045 crash while maintaining full visibility

    // 4. Build combined TLAS (only if we have overflow particles)
    if (directRTCount > 0) {
        BuildCombinedTLAS(cmdList);
    } else {
        // <=2044 particles: Use probe grid TLAS directly
        // Already built above, nothing to do
    }

    // 5. Dispatch lighting compute shader (uses probe grid TLAS for lighting)
    // TODO Phase 2: Use probe grid for volumetric GI, direct RT for overflow lighting
    DispatchRayQueryLighting(cmdList, particleBuffer, cameraPosition);
}
