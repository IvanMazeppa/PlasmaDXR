#include "ParticleRenderer_Gaussian.h"
#include "../core/Device.h"
#include "../utils/ResourceManager.h"
#include "../utils/Logger.h"
#include "d3dx12.h"
#include <fstream>

ParticleRenderer_Gaussian::~ParticleRenderer_Gaussian() {
    // ComPtr cleanup automatic
}

bool ParticleRenderer_Gaussian::Initialize(Device* device,
                                           ResourceManager* resources,
                                           uint32_t particleCount,
                                           uint32_t screenWidth,
                                           uint32_t screenHeight) {
    m_device = device;
    m_resources = resources;
    m_particleCount = particleCount;

    LOG_INFO("Initializing 3D Gaussian Splatting renderer...");
    LOG_INFO("  Particles: {}", particleCount);
    LOG_INFO("  Resolution: {}x{}", screenWidth, screenHeight);
    LOG_INFO("  Reusing existing RTLightingSystem BLAS/TLAS");

    if (!CreateOutputTexture(screenWidth, screenHeight)) {
        return false;
    }

    if (!CreatePipeline()) {
        return false;
    }

    LOG_INFO("Gaussian Splatting renderer initialized successfully");
    return true;
}

bool ParticleRenderer_Gaussian::CreateOutputTexture(uint32_t width, uint32_t height) {
    // Create UAV texture for Gaussian rendering output
    D3D12_RESOURCE_DESC texDesc = {};
    texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texDesc.Width = width;
    texDesc.Height = height;
    texDesc.DepthOrArraySize = 1;
    texDesc.MipLevels = 1;
    texDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    texDesc.SampleDesc.Count = 1;
    texDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    D3D12_HEAP_PROPERTIES defaultHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    HRESULT hr = m_device->GetDevice()->CreateCommittedResource(
        &defaultHeap,
        D3D12_HEAP_FLAG_NONE,
        &texDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,
        IID_PPV_ARGS(&m_outputTexture)
    );

    if (FAILED(hr)) {
        LOG_ERROR("Failed to create Gaussian output texture");
        return false;
    }

    // Create UAV
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    uavDesc.Texture2D.MipSlice = 0;

    m_outputUAV = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, m_outputUAVGPU);
    m_device->GetDevice()->CreateUnorderedAccessView(
        m_outputTexture.Get(),
        nullptr,
        &uavDesc,
        m_outputUAV
    );

    LOG_INFO("Created Gaussian output texture: {}x{}", width, height);
    return true;
}

bool ParticleRenderer_Gaussian::CreatePipeline() {
    // Load Gaussian raytrace compute shader
    std::ifstream shaderFile("shaders/particles/particle_gaussian_raytrace.dxil", std::ios::binary);
    if (!shaderFile.is_open()) {
        LOG_ERROR("Failed to load particle_gaussian_raytrace.dxil");
        LOG_ERROR("  Make sure shader is compiled!");
        return false;
    }

    std::vector<char> shaderData((std::istreambuf_iterator<char>(shaderFile)), std::istreambuf_iterator<char>());
    Microsoft::WRL::ComPtr<ID3DBlob> computeShader;
    HRESULT hr = D3DCreateBlob(shaderData.size(), &computeShader);
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create blob");
        return false;
    }
    memcpy(computeShader->GetBufferPointer(), shaderData.data(), shaderData.size());
    LOG_INFO("Loaded Gaussian shader: {} bytes", shaderData.size());

    // Root signature matches particle_gaussian_raytrace.hlsl
    // b0: GaussianConstants (48 DWORDs)
    // t0: StructuredBuffer<Particle> g_particles
    // t1: Buffer<float4> g_rtLighting
    // t2: RaytracingAccelerationStructure g_particleBVH (TLAS from RTLightingSystem!)
    // u0: RWTexture2D<float4> g_output
    CD3DX12_ROOT_PARAMETER1 rootParams[5];
    rootParams[0].InitAsConstants(48, 0);                  // b0
    rootParams[1].InitAsShaderResourceView(0);             // t0
    rootParams[2].InitAsShaderResourceView(1);             // t1
    rootParams[3].InitAsShaderResourceView(2);             // t2 (TLAS)
    rootParams[4].InitAsUnorderedAccessView(0);            // u0

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSigDesc;
    rootSigDesc.Init_1_1(5, rootParams, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);

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
        LOG_ERROR("Failed to create root signature");
        return false;
    }

    // Create compute PSO
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = m_rootSignature.Get();
    psoDesc.CS = CD3DX12_SHADER_BYTECODE(computeShader.Get());

    hr = m_device->GetDevice()->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_pso));
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create Gaussian PSO");
        return false;
    }

    LOG_INFO("Gaussian pipeline created");
    return true;
}

void ParticleRenderer_Gaussian::Render(ID3D12GraphicsCommandList4* cmdList,
                                       ID3D12Resource* particleBuffer,
                                       ID3D12Resource* rtLightingBuffer,
                                       ID3D12Resource* tlas,
                                       const RenderConstants& constants) {
    // Set Gaussian raytrace pipeline
    cmdList->SetPipelineState(m_pso.Get());
    cmdList->SetComputeRootSignature(m_rootSignature.Get());

    // Set constants
    cmdList->SetComputeRoot32BitConstants(0, sizeof(constants) / 4, &constants, 0);

    // Set resources (direct SRV/UAV binding)
    cmdList->SetComputeRootShaderResourceView(1, particleBuffer->GetGPUVirtualAddress());
    cmdList->SetComputeRootShaderResourceView(2, rtLightingBuffer->GetGPUVirtualAddress());
    cmdList->SetComputeRootShaderResourceView(3, tlas->GetGPUVirtualAddress());  // Reuse RT lighting TLAS!
    cmdList->SetComputeRootUnorderedAccessView(4, m_outputTexture->GetGPUVirtualAddress());

    // Dispatch (8x8 thread groups)
    uint32_t dispatchX = (constants.screenWidth + 7) / 8;
    uint32_t dispatchY = (constants.screenHeight + 7) / 8;
    cmdList->Dispatch(dispatchX, dispatchY, 1);

    LOG_INFO("Gaussian render dispatched: {}x{} thread groups", dispatchX, dispatchY);
}
