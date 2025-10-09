#include "ParticleRenderer_Gaussian.h"
#include "../core/Device.h"
#include "../utils/ResourceManager.h"
#include "../utils/Logger.h"
#include "d3dx12.h"
#include <fstream>

ParticleRenderer_Gaussian::~ParticleRenderer_Gaussian() {
    // Resources cleaned up automatically by ComPtr
}

bool ParticleRenderer_Gaussian::Initialize(Device* device,
                                           ResourceManager* resources,
                                           uint32_t particleCount) {
    m_device = device;
    m_resources = resources;
    m_particleCount = particleCount;

    LOG_INFO("Initializing 3D Gaussian Splatting renderer...");
    LOG_INFO("  Particle count: {}", particleCount);

    if (!CreateRayTracingPipeline()) {
        LOG_ERROR("Failed to create ray tracing pipeline");
        return false;
    }

    if (!CreateAccelerationStructures()) {
        LOG_ERROR("Failed to create acceleration structures");
        return false;
    }

    LOG_INFO("3D Gaussian Splatting renderer initialized successfully");
    return true;
}

bool ParticleRenderer_Gaussian::CreateRayTracingPipeline() {
    // Load Gaussian ray tracing compute shader
    std::ifstream shaderFile("shaders/particles/particle_gaussian_raytrace.dxil", std::ios::binary);
    if (!shaderFile.is_open()) {
        LOG_ERROR("Failed to load Gaussian raytrace shader: shaders/particles/particle_gaussian_raytrace.dxil");
        return false;
    }

    std::vector<char> shaderData((std::istreambuf_iterator<char>(shaderFile)), std::istreambuf_iterator<char>());
    Microsoft::WRL::ComPtr<ID3DBlob> computeShader;
    HRESULT hr = D3DCreateBlob(shaderData.size(), &computeShader);
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create blob for Gaussian shader");
        return false;
    }
    memcpy(computeShader->GetBufferPointer(), shaderData.data(), shaderData.size());

    // Create root signature
    // b0: Camera/render constants
    // t0: Particle buffer (SRV)
    // t1: RT lighting buffer (SRV)
    // t2: BVH acceleration structure (SRV)
    // u0: Output texture (UAV)
    CD3DX12_DESCRIPTOR_RANGE1 ranges[3];
    ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 3, 0);  // t0-t2
    ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // u0
    ranges[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 3);  // t3: BLAS (acceleration structure)

    CD3DX12_ROOT_PARAMETER1 rootParams[4];
    rootParams[0].InitAsConstants(48, 0);  // b0: Render constants
    rootParams[1].InitAsDescriptorTable(1, &ranges[0]);  // t0-t2
    rootParams[2].InitAsDescriptorTable(1, &ranges[1]);  // u0
    rootParams[3].InitAsDescriptorTable(1, &ranges[2]);  // t3: BLAS

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSigDesc;
    rootSigDesc.Init_1_1(4, rootParams, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);

    Microsoft::WRL::ComPtr<ID3DBlob> signature, error;
    hr = D3DX12SerializeVersionedRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1_1, &signature, &error);
    if (FAILED(hr)) {
        if (error) {
            LOG_ERROR("Root signature serialization failed: {}", (char*)error->GetBufferPointer());
        }
        return false;
    }

    hr = m_device->GetDevice()->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(),
                                                     IID_PPV_ARGS(&m_rootSignature));
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create Gaussian root signature");
        return false;
    }

    // Create compute PSO
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = m_rootSignature.Get();
    psoDesc.CS = CD3DX12_SHADER_BYTECODE(computeShader.Get());

    hr = m_device->GetDevice()->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_rayTracingPSO));
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create Gaussian compute PSO");
        return false;
    }

    LOG_INFO("Gaussian ray tracing pipeline created");
    return true;
}

bool ParticleRenderer_Gaussian::CreateAccelerationStructures() {
    // Create AABB buffer for particles (will be filled by compute shader)
    size_t aabbBufferSize = m_particleCount * sizeof(D3D12_RAYTRACING_AABB);

    D3D12_RESOURCE_DESC aabbDesc = CD3DX12_RESOURCE_DESC::Buffer(
        aabbBufferSize,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
    );

    D3D12_HEAP_PROPERTIES defaultHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    HRESULT hr = m_device->GetDevice()->CreateCommittedResource(
        &defaultHeap,
        D3D12_HEAP_FLAG_NONE,
        &aabbDesc,
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
        nullptr,
        IID_PPV_ARGS(&m_aabbBuffer)
    );

    if (FAILED(hr)) {
        LOG_ERROR("Failed to create AABB buffer");
        return false;
    }

    // Get BLAS prebuild info
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blasInputs = {};
    blasInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    blasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    blasInputs.NumDescs = 1;
    blasInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;

    D3D12_RAYTRACING_GEOMETRY_DESC geometryDesc = {};
    geometryDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
    geometryDesc.AABBs.AABBCount = m_particleCount;
    geometryDesc.AABBs.AABBs.StartAddress = m_aabbBuffer->GetGPUVirtualAddress();
    geometryDesc.AABBs.AABBs.StrideInBytes = sizeof(D3D12_RAYTRACING_AABB);
    geometryDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;

    blasInputs.pGeometryDescs = &geometryDesc;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO blasPrebuildInfo = {};
    m_device->GetDevice()->GetRaytracingAccelerationStructurePrebuildInfo(&blasInputs, &blasPrebuildInfo);

    LOG_INFO("BLAS prebuild: Result={} bytes, Scratch={} bytes",
             blasPrebuildInfo.ResultDataMaxSizeInBytes,
             blasPrebuildInfo.ScratchDataSizeInBytes);

    // Create BLAS buffer
    D3D12_RESOURCE_DESC blasDesc = CD3DX12_RESOURCE_DESC::Buffer(
        blasPrebuildInfo.ResultDataMaxSizeInBytes,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
    );

    hr = m_device->GetDevice()->CreateCommittedResource(
        &defaultHeap,
        D3D12_HEAP_FLAG_NONE,
        &blasDesc,
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
        nullptr,
        IID_PPV_ARGS(&m_blasBuffer)
    );

    if (FAILED(hr)) {
        LOG_ERROR("Failed to create BLAS buffer");
        return false;
    }

    // Create BLAS scratch buffer
    D3D12_RESOURCE_DESC scratchDesc = CD3DX12_RESOURCE_DESC::Buffer(
        blasPrebuildInfo.ScratchDataSizeInBytes,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
    );

    hr = m_device->GetDevice()->CreateCommittedResource(
        &defaultHeap,
        D3D12_HEAP_FLAG_NONE,
        &scratchDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,
        IID_PPV_ARGS(&m_blasScratch)
    );

    if (FAILED(hr)) {
        LOG_ERROR("Failed to create BLAS scratch buffer");
        return false;
    }

    LOG_INFO("Acceleration structures created");
    return true;
}

void ParticleRenderer_Gaussian::RebuildAccelerationStructure(ID3D12GraphicsCommandList* cmdList,
                                                              ID3D12Resource* particleBuffer) {
    // TODO: Implement BLAS rebuild
    // This will be called when particles move significantly
    // For now, we'll build it once and assume static particles
}

void ParticleRenderer_Gaussian::Render(ID3D12GraphicsCommandList* cmdList,
                                       ID3D12Resource* particleBuffer,
                                       ID3D12Resource* rtLightingBuffer,
                                       const RenderConstants& constants) {
    // Set pipeline
    cmdList->SetPipelineState(m_rayTracingPSO.Get());
    cmdList->SetComputeRootSignature(m_rootSignature.Get());

    // Set constants
    cmdList->SetComputeRoot32BitConstants(0, sizeof(constants) / 4, &constants, 0);

    // Set resources (descriptors)
    // TODO: Create descriptor tables for SRVs and UAVs

    // Dispatch rays (8x8 thread groups for screen tiles)
    uint32_t dispatchX = (constants.screenWidth + 7) / 8;
    uint32_t dispatchY = (constants.screenHeight + 7) / 8;
    cmdList->Dispatch(dispatchX, dispatchY, 1);
}
