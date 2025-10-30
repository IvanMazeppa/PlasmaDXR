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
        rootParams[0].InitAsConstants(10, 0);  // b0: AABBConstants (10 DWORDs - Phase 1.5 adaptive radius)
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

bool RTLightingSystem_RayQuery::CreateAccelerationStructures() {
    // Create AABB buffer
    {
        size_t aabbBufferSize = m_particleCount * 24;  // 6 floats (minXYZ, maxXYZ) = 24 bytes

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
        // Setup dummy geometry descriptor for size calculation
        D3D12_RAYTRACING_GEOMETRY_DESC geomDesc = {};
        geomDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
        geomDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
        geomDesc.AABBs.AABBCount = m_particleCount;
        geomDesc.AABBs.AABBs.StrideInBytes = 24;  // 6 floats

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blasInputs = {};
        blasInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        blasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
        blasInputs.NumDescs = 1;  // One geometry desc
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

        // Create instance descs buffer
        D3D12_HEAP_PROPERTIES uploadHeapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
        D3D12_RESOURCE_DESC instanceDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(D3D12_RAYTRACING_INSTANCE_DESC));

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

void RTLightingSystem_RayQuery::GenerateAABBs(ID3D12GraphicsCommandList4* cmdList, ID3D12Resource* particleBuffer) {
    cmdList->SetPipelineState(m_aabbGenPSO.Get());
    cmdList->SetComputeRootSignature(m_aabbGenRootSig.Get());

    // Set root parameters (Phase 1.5: Added adaptive radius parameters)
    AABBConstants constants = {
        m_particleCount,
        m_particleRadius,
        m_enableAdaptiveRadius ? 1u : 0u,
        m_adaptiveInnerZone,
        m_adaptiveOuterZone,
        m_adaptiveInnerScale,
        m_adaptiveOuterScale,
        m_densityScaleMin,
        m_densityScaleMax,
        0.0f  // padding
    };
    cmdList->SetComputeRoot32BitConstants(0, 10, &constants, 0);
    cmdList->SetComputeRootShaderResourceView(1, particleBuffer->GetGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(2, m_aabbBuffer->GetGPUVirtualAddress());

    // Dispatch (256 threads per group)
    uint32_t threadGroups = (m_particleCount + 255) / 256;
    cmdList->Dispatch(threadGroups, 1, 1);

    // UAV barrier
    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_aabbBuffer.Get());
    cmdList->ResourceBarrier(1, &barrier);
}

void RTLightingSystem_RayQuery::BuildBLAS(ID3D12GraphicsCommandList4* cmdList) {
    // Setup geometry desc
    D3D12_RAYTRACING_GEOMETRY_DESC geomDesc = {};
    geomDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
    geomDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    geomDesc.AABBs.AABBCount = m_particleCount;
    geomDesc.AABBs.AABBs.StartAddress = m_aabbBuffer->GetGPUVirtualAddress();
    geomDesc.AABBs.AABBs.StrideInBytes = 24;  // 6 floats

    // Build inputs
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blasInputs = {};
    blasInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    blasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
    blasInputs.NumDescs = 1;
    blasInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    blasInputs.pGeometryDescs = &geomDesc;

    // Build desc
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

    // Full pipeline:
    // 1. Generate AABBs from particle positions
    GenerateAABBs(cmdList, particleBuffer);

    // 2. Build BLAS from AABBs
    BuildBLAS(cmdList);

    // 3. Build TLAS from BLAS
    BuildTLAS(cmdList);

    // 4. Dispatch RayQuery lighting compute shader (with camera position for dynamic emission)
    DispatchRayQueryLighting(cmdList, particleBuffer, cameraPosition);
}
