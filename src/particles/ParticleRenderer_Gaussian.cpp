#include "ParticleRenderer_Gaussian.h"
#include "../core/Device.h"
#include "../utils/ResourceManager.h"
#include "../utils/Logger.h"
#include "../utils/d3dx12/d3dx12.h"
#include <d3dcompiler.h>
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

    // Create UAV descriptor (typed UAVs require descriptor table binding)
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    uavDesc.Texture2D.MipSlice = 0;

    m_outputUAV = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    m_device->GetDevice()->CreateUnorderedAccessView(
        m_outputTexture.Get(),
        nullptr,
        &uavDesc,
        m_outputUAV
    );

    // Get GPU handle for binding in Render()
    m_outputUAVGPU = m_resources->GetGPUHandle(m_outputUAV);

    LOG_INFO("Created Gaussian output texture: {}x{}", width, height);
    LOG_INFO("  UAV CPU handle: 0x{:016X}", m_outputUAV.ptr);
    LOG_INFO("  UAV GPU handle: 0x{:016X}", m_outputUAVGPU.ptr);

    if (m_outputUAVGPU.ptr == 0) {
        LOG_ERROR("CRITICAL: GPU handle is null!");
        return false;
    }

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
    // b0: GaussianConstants (48 DWORDs - core constants only, no emission for test)
    // t0: StructuredBuffer<Particle> g_particles
    // t1: Buffer<float4> g_rtLighting
    // t2: RaytracingAccelerationStructure g_particleBVH (TLAS from RTLightingSystem!)
    // u0: RWTexture2D<float4> g_output (descriptor table - typed UAV requirement)
    CD3DX12_DESCRIPTOR_RANGE1 uavRange;
    uavRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // u0: RWTexture2D

    CD3DX12_ROOT_PARAMETER1 rootParams[5];
    rootParams[0].InitAsConstants(64, 0);                  // b0 - 64 DWORDs (increased for RT toggles)
    rootParams[1].InitAsShaderResourceView(0);             // t0
    rootParams[2].InitAsShaderResourceView(1);             // t1
    rootParams[3].InitAsShaderResourceView(2);             // t2 (TLAS)
    rootParams[4].InitAsDescriptorTable(1, &uavRange);     // u0 via descriptor table

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
        LOG_ERROR("  HRESULT: 0x{:08X}", static_cast<unsigned int>(hr));

        // Try to get debug messages
        Microsoft::WRL::ComPtr<ID3D12InfoQueue> infoQueue;
        if (SUCCEEDED(m_device->GetDevice()->QueryInterface(IID_PPV_ARGS(&infoQueue)))) {
            UINT64 numMessages = infoQueue->GetNumStoredMessages();
            LOG_ERROR("  D3D12 Debug Messages ({} messages):", numMessages);
            for (UINT64 i = 0; i < numMessages && i < 10; i++) {
                SIZE_T messageLength = 0;
                infoQueue->GetMessage(i, nullptr, &messageLength);
                if (messageLength > 0) {
                    std::vector<char> messageData(messageLength);
                    D3D12_MESSAGE* message = (D3D12_MESSAGE*)messageData.data();
                    if (SUCCEEDED(infoQueue->GetMessage(i, message, &messageLength))) {
                        LOG_ERROR("    [{}]: {}", i, message->pDescription);
                    }
                }
            }
        }
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
    if (!cmdList || !particleBuffer || !rtLightingBuffer || !m_resources) {
        LOG_ERROR("Gaussian Render: null resource!");
        return;
    }

    // Set descriptor heap (required for descriptor tables!)
    ID3D12DescriptorHeap* heap = m_resources->GetDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    if (heap) {
        // Cast to base ID3D12GraphicsCommandList for SetDescriptorHeaps
        ID3D12GraphicsCommandList* baseList = static_cast<ID3D12GraphicsCommandList*>(cmdList);
        baseList->SetDescriptorHeaps(1, &heap);
    }

    // Set Gaussian raytrace pipeline
    cmdList->SetPipelineState(m_pso.Get());
    cmdList->SetComputeRootSignature(m_rootSignature.Get());

    // Set constants (only 48 DWORDs to match root signature)
    const uint32_t constantSize = 48;  // Limited to 48 DWORDs for root signature
    cmdList->SetComputeRoot32BitConstants(0, constantSize, &constants, 0);

    // Set resources
    cmdList->SetComputeRootShaderResourceView(1, particleBuffer->GetGPUVirtualAddress());
    cmdList->SetComputeRootShaderResourceView(2, rtLightingBuffer->GetGPUVirtualAddress());

    // Set TLAS for RayQuery operations
    if (!tlas) {
        LOG_ERROR("TLAS is null! RayQuery will fail");
        return;
    }
    cmdList->SetComputeRootShaderResourceView(3, tlas->GetGPUVirtualAddress());

    // Get GPU handle for UAV descriptor table
    D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle = m_resources->GetGPUHandle(m_outputUAV);
    if (gpuHandle.ptr == 0) {
        LOG_ERROR("GPU handle is ZERO!");
        return;
    }
    cmdList->SetComputeRootDescriptorTable(4, gpuHandle);

    // Dispatch (8x8 thread groups)
    uint32_t dispatchX = (constants.screenWidth + 7) / 8;
    uint32_t dispatchY = (constants.screenHeight + 7) / 8;
    cmdList->Dispatch(dispatchX, dispatchY, 1);

    // Add UAV barrier to ensure compute shader completes before we use the output
    D3D12_RESOURCE_BARRIER uavBarrier = {};
    uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uavBarrier.UAV.pResource = m_outputTexture.Get();
    cmdList->ResourceBarrier(1, &uavBarrier);
}
