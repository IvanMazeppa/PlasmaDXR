#include "ParticleRenderer_Gaussian.h"
#include "../core/Device.h"
#include "../utils/ResourceManager.h"
#include "../utils/Logger.h"
#include "../utils/d3dx12/d3dx12.h"
#include "../debug/pix3.h"
#include <d3dcompiler.h>
#include <fstream>

#ifdef ENABLE_DLSS
#include "../dlss/DLSSSystem.h"
#include "nvsdk_ngx_helpers.h"  // For NGX_DLSS_GET_OPTIMAL_SETTINGS
#endif

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
    m_screenWidth = screenWidth;
    m_screenHeight = screenHeight;

    LOG_INFO("Initializing 3D Gaussian Splatting renderer...");
    LOG_INFO("  Particles: {}", particleCount);
    LOG_INFO("  Resolution: {}x{}", screenWidth, screenHeight);
    LOG_INFO("  Reusing existing RTLightingSystem BLAS/TLAS");

    // Create constant buffer for RenderConstants
    // Use upload heap so we can map/write every frame
    const UINT constantBufferSize = (sizeof(RenderConstants) + 255) & ~255;  // Align to 256 bytes
    D3D12_HEAP_PROPERTIES uploadHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    D3D12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(constantBufferSize);

    HRESULT hr = m_device->GetDevice()->CreateCommittedResource(
        &uploadHeap,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&m_constantBuffer)
    );

    if (FAILED(hr)) {
        LOG_ERROR("Failed to create constant buffer");
        return false;
    }

    // Map the constant buffer (keep it mapped for the lifetime of the buffer)
    hr = m_constantBuffer->Map(0, nullptr, &m_constantBufferMapped);
    if (FAILED(hr)) {
        LOG_ERROR("Failed to map constant buffer");
        return false;
    }

    LOG_INFO("Created constant buffer: {} bytes (aligned from {} bytes)",
             constantBufferSize, sizeof(RenderConstants));

    if (!CreateOutputTexture(screenWidth, screenHeight)) {
        return false;
    }

    // Create light buffer (structured buffer for multi-light system)
    // MAX_LIGHTS = 16, Light struct = 64 bytes (32 base + 32 god ray parameters)
    const uint32_t MAX_LIGHTS = 16;
    const uint32_t lightStructSize = 64;  // Must match HLSL Light struct (extended for god rays)
    const uint32_t lightBufferSize = MAX_LIGHTS * lightStructSize;

    LOG_INFO("Creating light buffer...");
    LOG_INFO("  Max lights: {}", MAX_LIGHTS);
    LOG_INFO("  Light struct size: {} bytes", lightStructSize);
    LOG_INFO("  Buffer size: {} bytes", lightBufferSize);

    // Use UPLOAD heap so we can update lights from CPU each frame
    D3D12_HEAP_PROPERTIES uploadHeapForLights = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    D3D12_RESOURCE_DESC lightBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(lightBufferSize);

    hr = m_device->GetDevice()->CreateCommittedResource(
        &uploadHeapForLights,
        D3D12_HEAP_FLAG_NONE,
        &lightBufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&m_lightBuffer)
    );

    if (FAILED(hr)) {
        LOG_ERROR("Failed to create light buffer");
        return false;
    }

    // Map light buffer for CPU writes
    hr = m_lightBuffer->Map(0, nullptr, &m_lightBufferMapped);
    if (FAILED(hr)) {
        LOG_ERROR("Failed to map light buffer");
        return false;
    }

    // Create SRV for shader access (t4)
    D3D12_SHADER_RESOURCE_VIEW_DESC lightSrvDesc = {};
    lightSrvDesc.Format = DXGI_FORMAT_UNKNOWN;
    lightSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    lightSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    lightSrvDesc.Buffer.FirstElement = 0;
    lightSrvDesc.Buffer.NumElements = MAX_LIGHTS;
    lightSrvDesc.Buffer.StructureByteStride = lightStructSize;

    m_lightSRV = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    m_device->GetDevice()->CreateShaderResourceView(
        m_lightBuffer.Get(),
        &lightSrvDesc,
        m_lightSRV
    );
    m_lightSRVGPU = m_resources->GetGPUHandle(m_lightSRV);

    LOG_INFO("Created light buffer: SRV=0x{:016X}", m_lightSRVGPU.ptr);

    // Initialize with empty lights (caller will call UpdateLights())
    std::vector<Light> emptyLights;
    UpdateLights(emptyLights);

    // Create PCSS temporal shadow buffers (ping-pong for temporal filtering)
    // R16_FLOAT format: 16-bit float per pixel (2MB @ 1080p per buffer)
    LOG_INFO("Creating PCSS temporal shadow buffers...");
    LOG_INFO("  Resolution: {}x{} pixels", screenWidth, screenHeight);
    LOG_INFO("  Format: R16_FLOAT (16-bit per pixel)");
    LOG_INFO("  Buffer size: {} MB per buffer", (screenWidth * screenHeight * 2) / (1024 * 1024));
    LOG_INFO("  Total memory: {} MB (2x buffers)", (screenWidth * screenHeight * 4) / (1024 * 1024));

    D3D12_RESOURCE_DESC shadowTexDesc = {};
    shadowTexDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    shadowTexDesc.Width = screenWidth;
    shadowTexDesc.Height = screenHeight;
    shadowTexDesc.DepthOrArraySize = 1;
    shadowTexDesc.MipLevels = 1;
    shadowTexDesc.Format = DXGI_FORMAT_R16_FLOAT;
    shadowTexDesc.SampleDesc.Count = 1;
    shadowTexDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    D3D12_HEAP_PROPERTIES defaultHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    for (int i = 0; i < 2; i++) {
        hr = m_device->GetDevice()->CreateCommittedResource(
            &defaultHeap,
            D3D12_HEAP_FLAG_NONE,
            &shadowTexDesc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_PPV_ARGS(&m_shadowBuffer[i])
        );

        if (FAILED(hr)) {
            LOG_ERROR("Failed to create shadow buffer {}", i);
            return false;
        }

        // Create SRV (for reading previous frame shadow)
        D3D12_SHADER_RESOURCE_VIEW_DESC shadowSrvDesc = {};
        shadowSrvDesc.Format = DXGI_FORMAT_R16_FLOAT;
        shadowSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        shadowSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        shadowSrvDesc.Texture2D.MipLevels = 1;

        m_shadowSRV[i] = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        m_device->GetDevice()->CreateShaderResourceView(
            m_shadowBuffer[i].Get(),
            &shadowSrvDesc,
            m_shadowSRV[i]
        );
        m_shadowSRVGPU[i] = m_resources->GetGPUHandle(m_shadowSRV[i]);

        // Create UAV (for writing current frame shadow)
        D3D12_UNORDERED_ACCESS_VIEW_DESC shadowUavDesc = {};
        shadowUavDesc.Format = DXGI_FORMAT_R16_FLOAT;
        shadowUavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        shadowUavDesc.Texture2D.MipSlice = 0;

        m_shadowUAV[i] = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        m_device->GetDevice()->CreateUnorderedAccessView(
            m_shadowBuffer[i].Get(),
            nullptr,
            &shadowUavDesc,
            m_shadowUAV[i]
        );
        m_shadowUAVGPU[i] = m_resources->GetGPUHandle(m_shadowUAV[i]);

        LOG_INFO("Created shadow buffer {}: SRV=0x{:016X}, UAV=0x{:016X}",
                 i, m_shadowSRVGPU[i].ptr, m_shadowUAVGPU[i].ptr);
    }

#ifdef ENABLE_DLSS
    // Create motion vector buffer for DLSS Ray Reconstruction (RG16_FLOAT)
    LOG_INFO("Creating motion vector buffer for DLSS...");
    LOG_INFO("  Resolution: {}x{} pixels", screenWidth, screenHeight);
    LOG_INFO("  Format: RG16_FLOAT (32-bit per pixel, 2 components)");
    LOG_INFO("  Buffer size: {} MB", (screenWidth * screenHeight * 4) / (1024 * 1024));

    D3D12_RESOURCE_DESC mvTexDesc = {};
    mvTexDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    mvTexDesc.Width = screenWidth;
    mvTexDesc.Height = screenHeight;
    mvTexDesc.DepthOrArraySize = 1;
    mvTexDesc.MipLevels = 1;
    mvTexDesc.Format = DXGI_FORMAT_R16G16_FLOAT;  // 2-component motion vector
    mvTexDesc.SampleDesc.Count = 1;
    mvTexDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    hr = m_device->GetDevice()->CreateCommittedResource(
        &defaultHeap,
        D3D12_HEAP_FLAG_NONE,
        &mvTexDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,
        IID_PPV_ARGS(&m_motionVectorBuffer)
    );

    if (FAILED(hr)) {
        LOG_ERROR("Failed to create motion vector buffer");
        return false;
    }

    // Create SRV for reading motion vectors
    D3D12_SHADER_RESOURCE_VIEW_DESC mvSrvDesc = {};
    mvSrvDesc.Format = DXGI_FORMAT_R16G16_FLOAT;
    mvSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    mvSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    mvSrvDesc.Texture2D.MipLevels = 1;

    m_motionVectorSRV = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    m_device->GetDevice()->CreateShaderResourceView(
        m_motionVectorBuffer.Get(),
        &mvSrvDesc,
        m_motionVectorSRV
    );
    m_motionVectorSRVGPU = m_resources->GetGPUHandle(m_motionVectorSRV);

    // Create UAV for writing motion vectors
    D3D12_UNORDERED_ACCESS_VIEW_DESC mvUavDesc = {};
    mvUavDesc.Format = DXGI_FORMAT_R16G16_FLOAT;
    mvUavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    mvUavDesc.Texture2D.MipSlice = 0;

    m_motionVectorUAV = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    m_device->GetDevice()->CreateUnorderedAccessView(
        m_motionVectorBuffer.Get(),
        nullptr,
        &mvUavDesc,
        m_motionVectorUAV
    );
    m_motionVectorUAVGPU = m_resources->GetGPUHandle(m_motionVectorUAV);

    LOG_INFO("Created motion vector buffer: SRV=0x{:016X}, UAV=0x{:016X}",
             m_motionVectorSRVGPU.ptr, m_motionVectorUAVGPU.ptr);

    // Create denoised output texture (DLSS target)
    LOG_INFO("Creating denoised output texture for DLSS...");
    D3D12_RESOURCE_DESC denoisedTexDesc = {};
    denoisedTexDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    denoisedTexDesc.Width = screenWidth;
    denoisedTexDesc.Height = screenHeight;
    denoisedTexDesc.DepthOrArraySize = 1;
    denoisedTexDesc.MipLevels = 1;
    denoisedTexDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;  // Match main output
    denoisedTexDesc.SampleDesc.Count = 1;
    denoisedTexDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    hr = m_device->GetDevice()->CreateCommittedResource(
        &defaultHeap,
        D3D12_HEAP_FLAG_NONE,
        &denoisedTexDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,
        IID_PPV_ARGS(&m_upscaledOutputTexture)
    );

    if (FAILED(hr)) {
        LOG_ERROR("Failed to create denoised output texture");
        return false;
    }

    // Create SRV for reading denoised output (for blit pass)
    D3D12_SHADER_RESOURCE_VIEW_DESC denoisedSrvDesc = {};
    denoisedSrvDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    denoisedSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    denoisedSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    denoisedSrvDesc.Texture2D.MipLevels = 1;

    m_upscaledOutputSRV = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    m_device->GetDevice()->CreateShaderResourceView(
        m_upscaledOutputTexture.Get(),
        &denoisedSrvDesc,
        m_upscaledOutputSRV
    );
    m_upscaledOutputSRVGPU = m_resources->GetGPUHandle(m_upscaledOutputSRV);

    LOG_INFO("Created denoised output texture: SRV=0x{:016X}",
             m_upscaledOutputSRVGPU.ptr);

    // Create depth buffer for DLSS (REQUIRED input)
    LOG_INFO("Creating depth buffer for DLSS...");
    D3D12_RESOURCE_DESC depthTexDesc = {};
    depthTexDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    depthTexDesc.Width = screenWidth;
    depthTexDesc.Height = screenHeight;
    depthTexDesc.DepthOrArraySize = 1;
    depthTexDesc.MipLevels = 1;
    depthTexDesc.Format = DXGI_FORMAT_R32_FLOAT;  // Single-channel depth
    depthTexDesc.SampleDesc.Count = 1;
    depthTexDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    hr = m_device->GetDevice()->CreateCommittedResource(
        &defaultHeap,
        D3D12_HEAP_FLAG_NONE,
        &depthTexDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,
        IID_PPV_ARGS(&m_depthBuffer)
    );

    if (FAILED(hr)) {
        LOG_ERROR("Failed to create depth buffer");
        return false;
    }

    // Create UAV for shader to write depth
    D3D12_UNORDERED_ACCESS_VIEW_DESC depthUavDesc = {};
    depthUavDesc.Format = DXGI_FORMAT_R32_FLOAT;
    depthUavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;

    m_depthUAV = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    m_device->GetDevice()->CreateUnorderedAccessView(
        m_depthBuffer.Get(),
        nullptr,
        &depthUavDesc,
        m_depthUAV
    );
    m_depthUAVGPU = m_resources->GetGPUHandle(m_depthUAV);

    // Create SRV for DLSS to read depth
    D3D12_SHADER_RESOURCE_VIEW_DESC depthSrvDesc = {};
    depthSrvDesc.Format = DXGI_FORMAT_R32_FLOAT;
    depthSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    depthSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    depthSrvDesc.Texture2D.MipLevels = 1;

    m_depthSRV = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    m_device->GetDevice()->CreateShaderResourceView(
        m_depthBuffer.Get(),
        &depthSrvDesc,
        m_depthSRV
    );
    m_depthSRVGPU = m_resources->GetGPUHandle(m_depthSRV);

    LOG_INFO("Created depth buffer: UAV=0x{:016X}, SRV=0x{:016X}",
             m_depthUAVGPU.ptr, m_depthSRVGPU.ptr);
#endif

    if (!CreatePipeline()) {
        return false;
    }

#ifdef ENABLE_DLSS
    if (!CreateMotionVectorPipeline()) {
        LOG_ERROR("Failed to create motion vector compute pipeline");
        return false;
    }
#endif

    LOG_INFO("Gaussian Splatting renderer initialized successfully");
    return true;
}

bool ParticleRenderer_Gaussian::CreateOutputTexture(uint32_t width, uint32_t height) {
    // Create UAV texture for Gaussian rendering output
    // TODO: Use R16G16B16A16_FLOAT to prevent color banding (needs format check)
    D3D12_RESOURCE_DESC texDesc = {};
    texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texDesc.Width = width;
    texDesc.Height = height;
    texDesc.DepthOrArraySize = 1;
    texDesc.MipLevels = 1;
    texDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT; // 16-bit HDR (256x better precision, needs blit to SDR swap chain)
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
    uavDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT; // MUST match resource format!
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

    // Create SRV for blit pass (read HDR texture in pixel shader)
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Texture2D.MipLevels = 1;

    m_outputSRV = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    m_device->GetDevice()->CreateShaderResourceView(
        m_outputTexture.Get(),
        &srvDesc,
        m_outputSRV
    );
    m_outputSRVGPU = m_resources->GetGPUHandle(m_outputSRV);

    LOG_INFO("Created Gaussian output texture: {}x{} (R16G16B16A16_FLOAT - 16-bit HDR)", width, height);
    LOG_INFO("  UAV CPU handle: 0x{:016X}", m_outputUAV.ptr);
    LOG_INFO("  UAV GPU handle: 0x{:016X}", m_outputUAVGPU.ptr);
    LOG_INFO("  SRV GPU handle: 0x{:016X}", m_outputSRVGPU.ptr);

    if (m_outputUAVGPU.ptr == 0 || m_outputSRVGPU.ptr == 0) {
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
    // b0: GaussianConstants (CBV - no size limit!)
    // t0: StructuredBuffer<Particle> g_particles
    // t1: Buffer<float4> g_rtLighting
    // t2: RaytracingAccelerationStructure g_particleBVH (TLAS from RTLightingSystem!)
    // t4: StructuredBuffer<Light> g_lights (multi-light system)
    // t5: Texture2D<float> g_prevShadow (PCSS temporal shadow - previous frame, descriptor table)
    // t6: Texture2D<float4> g_rtxdiOutput (RTXDI selected lights - optional, descriptor table)
    // u0: RWTexture2D<float4> g_output (descriptor table - typed UAV requirement)
    // u2: RWTexture2D<float> g_currShadow (PCSS temporal shadow - current frame, descriptor table)
    CD3DX12_DESCRIPTOR_RANGE1 srvRanges[2];
    srvRanges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 5);  // t5: Texture2D (prev shadow)
    srvRanges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 6);  // t6: Texture2D (RTXDI output)

    CD3DX12_DESCRIPTOR_RANGE1 uavRanges[2];
    uavRanges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // u0: RWTexture2D (output)
    uavRanges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 2);  // u2: RWTexture2D (current shadow)

    CD3DX12_ROOT_PARAMETER1 rootParams[9];  // +1 for RTXDI output
    rootParams[0].InitAsConstantBufferView(0);              // b0 - CBV (no DWORD limit!)
    rootParams[1].InitAsShaderResourceView(0);              // t0 - particles (raw buffer is OK)
    rootParams[2].InitAsShaderResourceView(1);              // t1 - rtLighting (raw buffer is OK)
    rootParams[3].InitAsShaderResourceView(2);              // t2 - TLAS (raw is OK)
    rootParams[4].InitAsShaderResourceView(4);              // t4 - lights (raw buffer is OK)
    rootParams[5].InitAsDescriptorTable(1, &uavRanges[0]);  // u0 - output texture
    rootParams[6].InitAsDescriptorTable(1, &srvRanges[0]);  // t5 - previous shadow (SRV)
    rootParams[7].InitAsDescriptorTable(1, &uavRanges[1]);  // u2 - current shadow (UAV)
    rootParams[8].InitAsDescriptorTable(1, &srvRanges[1]);  // t6 - RTXDI output (SRV)

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSigDesc;
    rootSigDesc.Init_1_1(9, rootParams, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);

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

#ifdef ENABLE_DLSS
bool ParticleRenderer_Gaussian::CreateMotionVectorPipeline() {
    // Load motion vector compute shader
    std::ifstream shaderFile("shaders/particles/compute_motion_vectors.dxil", std::ios::binary);
    if (!shaderFile.is_open()) {
        LOG_ERROR("Failed to load compute_motion_vectors.dxil");
        LOG_ERROR("  Make sure shader is compiled!");
        return false;
    }

    std::vector<char> shaderData((std::istreambuf_iterator<char>(shaderFile)), std::istreambuf_iterator<char>());
    Microsoft::WRL::ComPtr<ID3DBlob> computeShader;
    HRESULT hr = D3DCreateBlob(shaderData.size(), &computeShader);
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create blob for motion vector shader");
        return false;
    }
    memcpy(computeShader->GetBufferPointer(), shaderData.data(), shaderData.size());
    LOG_INFO("Loaded motion vector shader: {} bytes", shaderData.size());

    // Root signature for compute_motion_vectors.hlsl
    // b0: MotionVectorConstants (CBV)
    // t0: StructuredBuffer<Particle> g_particles
    // u0: RWTexture2D<float2> g_motionVectors (descriptor table - typed UAV)

    CD3DX12_DESCRIPTOR_RANGE1 uavRange;
    uavRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // u0: RWTexture2D

    CD3DX12_ROOT_PARAMETER1 rootParams[3];
    rootParams[0].InitAsConstantBufferView(0);              // b0 - constants
    rootParams[1].InitAsShaderResourceView(0);              // t0 - particles
    rootParams[2].InitAsDescriptorTable(1, &uavRange);      // u0 - motion vectors (UAV)

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSigDesc;
    rootSigDesc.Init_1_1(3, rootParams, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);

    Microsoft::WRL::ComPtr<ID3DBlob> signature, error;
    hr = D3DX12SerializeVersionedRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1_1, &signature, &error);
    if (FAILED(hr)) {
        if (error) {
            LOG_ERROR("Motion vector root signature serialization failed: {}", (char*)error->GetBufferPointer());
        }
        return false;
    }

    hr = m_device->GetDevice()->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(),
                                                     IID_PPV_ARGS(&m_motionVectorRootSig));
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create motion vector root signature");
        return false;
    }

    // Create compute PSO
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = m_motionVectorRootSig.Get();
    psoDesc.CS = CD3DX12_SHADER_BYTECODE(computeShader.Get());

    hr = m_device->GetDevice()->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_motionVectorPSO));
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create motion vector PSO");
        LOG_ERROR("  HRESULT: 0x{:08X}", static_cast<unsigned int>(hr));
        return false;
    }

    LOG_INFO("Motion vector pipeline created");
    return true;
}
#endif

void ParticleRenderer_Gaussian::Render(ID3D12GraphicsCommandList4* cmdList,
                                       ID3D12Resource* particleBuffer,
                                       ID3D12Resource* rtLightingBuffer,
                                       ID3D12Resource* tlas,
                                       const RenderConstants& constants,
                                       ID3D12Resource* rtxdiOutputBuffer) {
    if (!cmdList || !particleBuffer || !rtLightingBuffer || !m_resources) {
        LOG_ERROR("Gaussian Render: null resource!");
        return;
    }

#ifdef ENABLE_DLSS
    // Lazy DLSS Super Resolution feature creation (first frame only)
    if (m_dlssSystem && !m_dlssFeatureCreated) {
        LOG_INFO("DLSS: Creating Super Resolution feature ({}x{} → {}x{})...",
                 m_renderWidth, m_renderHeight, m_outputWidth, m_outputHeight);

        // Cast to ID3D12GraphicsCommandList for feature creation
        ID3D12GraphicsCommandList* baseList = static_cast<ID3D12GraphicsCommandList*>(cmdList);

        // Use calculated render/output resolutions from SetDLSSSystem
        if (m_dlssSystem->CreateSuperResolutionFeature(baseList,
                                                        m_renderWidth, m_renderHeight,
                                                        m_outputWidth, m_outputHeight,
                                                        m_dlssQualityMode)) {
            m_dlssFeatureCreated = true;
            LOG_INFO("DLSS: Super Resolution feature created successfully!");
            LOG_INFO("  Render: {}x{}, Output: {}x{}", m_renderWidth, m_renderHeight, m_outputWidth, m_outputHeight);
        } else {
            LOG_ERROR("DLSS: Feature creation failed");
            LOG_WARN("  DLSS will be disabled for this session");
            m_dlssSystem = nullptr;  // Don't try again
        }
    }
#endif

#ifdef ENABLE_DLSS
    // TODO: Compute motion vectors for DLSS temporal denoising
    // TEMPORARILY DISABLED - needs separate constant buffer with prevViewProj
    // For now, DLSS will use zero motion vectors (static scene assumption)
    //
    // if (m_dlssSystem && m_dlssFeatureCreated && m_motionVectorPSO) {
    //     // Set motion vector compute pipeline
    //     cmdList->SetPipelineState(m_motionVectorPSO.Get());
    //     cmdList->SetComputeRootSignature(m_motionVectorRootSig.Get());
    //     ...
    // }
#endif

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

    // Upload constants to GPU (copy to mapped constant buffer)
    memcpy(m_constantBufferMapped, &constants, sizeof(RenderConstants));

    // Bind constant buffer view (no size limit!)
    cmdList->SetComputeRootConstantBufferView(0, m_constantBuffer->GetGPUVirtualAddress());

    // Set resources
    cmdList->SetComputeRootShaderResourceView(1, particleBuffer->GetGPUVirtualAddress());
    cmdList->SetComputeRootShaderResourceView(2, rtLightingBuffer->GetGPUVirtualAddress());

    // Set TLAS for RayQuery operations
    if (!tlas) {
        LOG_ERROR("TLAS is null! RayQuery will fail");
        return;
    }
    cmdList->SetComputeRootShaderResourceView(3, tlas->GetGPUVirtualAddress());

    // Bind light buffer (t4 - multi-light system)
    if (m_lightBuffer) {
        cmdList->SetComputeRootShaderResourceView(4, m_lightBuffer->GetGPUVirtualAddress());
    }

    // Bind output texture (UAV descriptor table) - root param 5
    D3D12_GPU_DESCRIPTOR_HANDLE outputUAVHandle = m_resources->GetGPUHandle(m_outputUAV);
    if (outputUAVHandle.ptr == 0) {
        LOG_ERROR("GPU handle is ZERO!");
        return;
    }
    cmdList->SetComputeRootDescriptorTable(5, outputUAVHandle);

    // PCSS: Bind previous frame's shadow buffer (SRV descriptor table) - root param 6
    uint32_t prevShadowIndex = 1 - m_currentShadowIndex;  // Ping-pong
    D3D12_GPU_DESCRIPTOR_HANDLE prevShadowSRVHandle = m_shadowSRVGPU[prevShadowIndex];
    if (prevShadowSRVHandle.ptr == 0) {
        LOG_ERROR("Previous shadow SRV handle is ZERO!");
        return;
    }
    cmdList->SetComputeRootDescriptorTable(6, prevShadowSRVHandle);

    // PCSS: Bind current frame's shadow buffer (UAV descriptor table) - root param 7
    D3D12_GPU_DESCRIPTOR_HANDLE currentShadowUAVHandle = m_shadowUAVGPU[m_currentShadowIndex];
    if (currentShadowUAVHandle.ptr == 0) {
        LOG_ERROR("Current shadow UAV handle is ZERO!");
        return;
    }
    cmdList->SetComputeRootDescriptorTable(7, currentShadowUAVHandle);

    // RTXDI: Bind RTXDI output buffer (SRV descriptor table) - root param 8 (optional)
    if (rtxdiOutputBuffer && constants.useRTXDI != 0) {
        // Create SRV for RTXDI output texture (R32G32B32A32_FLOAT) - CACHED to prevent descriptor leak
        if (m_rtxdiSRVGPU.ptr == 0) {
            // First time: Allocate descriptor and cache it
            D3D12_SHADER_RESOURCE_VIEW_DESC rtxdiSrvDesc = {};
            rtxdiSrvDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
            rtxdiSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            rtxdiSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            rtxdiSrvDesc.Texture2D.MipLevels = 1;

            m_rtxdiSRV = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            m_device->GetDevice()->CreateShaderResourceView(
                rtxdiOutputBuffer,
                &rtxdiSrvDesc,
                m_rtxdiSRV
            );
            m_rtxdiSRVGPU = m_resources->GetGPUHandle(m_rtxdiSRV);

            LOG_INFO("Created RTXDI output SRV (cached): GPU=0x{:016X}", m_rtxdiSRVGPU.ptr);
        }

        if (m_rtxdiSRVGPU.ptr == 0) {
            LOG_ERROR("RTXDI output SRV handle is ZERO!");
            return;
        }
        cmdList->SetComputeRootDescriptorTable(8, m_rtxdiSRVGPU);
    } else {
        // Bind dummy descriptor (use previous shadow buffer as placeholder)
        // This is safe since shader won't read t6 when useRTXDI=0
        cmdList->SetComputeRootDescriptorTable(8, prevShadowSRVHandle);
    }

    // Dispatch (8x8 thread groups)
    uint32_t dispatchX = (constants.screenWidth + 7) / 8;
    uint32_t dispatchY = (constants.screenHeight + 7) / 8;
    cmdList->Dispatch(dispatchX, dispatchY, 1);

    // Add UAV barriers to ensure compute shader completes before next frame
    D3D12_RESOURCE_BARRIER uavBarriers[2] = {};
    uavBarriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uavBarriers[0].UAV.pResource = m_outputTexture.Get();
    uavBarriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uavBarriers[1].UAV.pResource = m_shadowBuffer[m_currentShadowIndex].Get();
    cmdList->ResourceBarrier(2, uavBarriers);

#ifdef ENABLE_DLSS
    // DLSS Super Resolution: AI upscaling from render resolution to output resolution
    if (m_dlssSystem && m_dlssFeatureCreated) {
        // Transition noisy output: UAV → SRV (DLSS input)
        CD3DX12_RESOURCE_BARRIER toSRV = CD3DX12_RESOURCE_BARRIER::Transition(
            m_outputTexture.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
        );
        cmdList->ResourceBarrier(1, &toSRV);

        // Transition depth buffer: UAV → SRV (DLSS input)
        CD3DX12_RESOURCE_BARRIER depthToSRV = CD3DX12_RESOURCE_BARRIER::Transition(
            m_depthBuffer.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
        );
        cmdList->ResourceBarrier(1, &depthToSRV);

        // Setup DLSS Super Resolution parameters
        DLSSSystem::SuperResolutionParams dlssParams = {};
        dlssParams.inputColor = m_outputTexture.Get();              // Gaussian render output (low-res)
        dlssParams.outputUpscaled = m_upscaledOutputTexture.Get();  // AI upscaled result (high-res)
        dlssParams.inputMotionVectors = m_motionVectorBuffer.Get(); // Zero MVs (static scene assumption)
        dlssParams.inputDepth = m_depthBuffer.Get();                // Optional (improves quality)

        // Use calculated resolutions from SetDLSSSystem()
        dlssParams.renderWidth = m_screenWidth;    // m_screenWidth is set to render resolution
        dlssParams.renderHeight = m_screenHeight;  // m_screenHeight is set to render resolution
        dlssParams.outputWidth = m_outputWidth;    // Target native resolution
        dlssParams.outputHeight = m_outputHeight;  // Target native resolution

        dlssParams.jitterOffsetX = 0.0f;  // No TAA
        dlssParams.jitterOffsetY = 0.0f;
        dlssParams.sharpness = m_dlssSharpness;  // User-configurable sharpness
        dlssParams.reset = m_dlssFirstFrame ? 1 : 0;  // Clear history on first frame

        // Reset flag for subsequent frames
        if (m_dlssFirstFrame) {
            m_dlssFirstFrame = false;
        }

        // Call DLSS Super Resolution
        bool dlssSuccess = m_dlssSystem->EvaluateSuperResolution(
            static_cast<ID3D12GraphicsCommandList*>(cmdList),
            dlssParams
        );

        if (dlssSuccess) {
            // DLSS succeeded! Upscaled output is ready for blit pass
            // (No per-frame logging - success is expected behavior)

            // Transition upscaled output: UAV → SRV (for blit pass)
            CD3DX12_RESOURCE_BARRIER upscaledToSRV = CD3DX12_RESOURCE_BARRIER::Transition(
                m_upscaledOutputTexture.Get(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
            );
            cmdList->ResourceBarrier(1, &upscaledToSRV);
        } else {
            // DLSS failed: Transition non-upscaled output back to SRV for blit pass
            // (Blit will use native-resolution output as fallback)
            LOG_WARN("DLSS: Super Resolution failed, using native-resolution output");
        }

        // Transition depth buffer back to UAV for next frame
        CD3DX12_RESOURCE_BARRIER depthBackToUAV = CD3DX12_RESOURCE_BARRIER::Transition(
            m_depthBuffer.Get(),
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        );
        cmdList->ResourceBarrier(1, &depthBackToUAV);

        // Note: m_outputTexture stays in SRV state for potential fallback
    } else {
        // DLSS not available: Transition noisy output to SRV for blit pass
        CD3DX12_RESOURCE_BARRIER toSRV = CD3DX12_RESOURCE_BARRIER::Transition(
            m_outputTexture.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
        );
        cmdList->ResourceBarrier(1, &toSRV);
    }
#else
    // No DLSS: Transition noisy output to SRV for blit pass
    CD3DX12_RESOURCE_BARRIER toSRV = CD3DX12_RESOURCE_BARRIER::Transition(
        m_outputTexture.Get(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );
    cmdList->ResourceBarrier(1, &toSRV);
#endif

    // PCSS: Swap shadow buffers for next frame (ping-pong temporal filtering)
    m_currentShadowIndex = 1 - m_currentShadowIndex;
}

void ParticleRenderer_Gaussian::UpdateLights(const std::vector<Light>& lights) {
    if (!m_lightBufferMapped) {
        LOG_ERROR("Light buffer not mapped!");
        return;
    }

    const uint32_t MAX_LIGHTS = 16;
    uint32_t lightCount = static_cast<uint32_t>(lights.size());
    if (lightCount > MAX_LIGHTS) {
        LOG_WARN("Too many lights ({} provided, max is {}), truncating", lightCount, MAX_LIGHTS);
        lightCount = MAX_LIGHTS;
    }

    // Copy lights to GPU buffer
    if (lightCount > 0) {
        memcpy(m_lightBufferMapped, lights.data(), lightCount * sizeof(Light));
    }

    // Clear remaining slots (optional, but good practice)
    if (lightCount < MAX_LIGHTS) {
        Light emptyLight = {};
        char* bufferPtr = static_cast<char*>(m_lightBufferMapped);
        for (uint32_t i = lightCount; i < MAX_LIGHTS; i++) {
            memcpy(bufferPtr + i * sizeof(Light), &emptyLight, sizeof(Light));
        }
    }

    // Only log when light count changes (prevent log spam at 120 FPS)
    static uint32_t s_lastLightCount = UINT32_MAX;
    if (lightCount != s_lastLightCount) {
        LOG_INFO("Updated light buffer: {} lights", lightCount);
        s_lastLightCount = lightCount;
    }
}

bool ParticleRenderer_Gaussian::Resize(uint32_t newWidth, uint32_t newHeight) {
    if (newWidth == m_screenWidth && newHeight == m_screenHeight) {
        return true;  // No resize needed
    }

    LOG_INFO("Resizing Gaussian renderer: {}x{} -> {}x{}", m_screenWidth, m_screenHeight, newWidth, newHeight);

    m_screenWidth = newWidth;
    m_screenHeight = newHeight;

    // Release old resources
    m_outputTexture.Reset();

    // Recreate output texture at new resolution
    if (!CreateOutputTexture(newWidth, newHeight)) {
        LOG_ERROR("Failed to recreate output texture during resize");
        return false;
    }

    LOG_INFO("Gaussian renderer resized successfully to {}x{}", newWidth, newHeight);
    return true;
}

#ifdef ENABLE_DLSS
void ParticleRenderer_Gaussian::SetDLSSSystem(DLSSSystem* dlss, uint32_t width, uint32_t height, int qualityMode) {
    if (!dlss) {
        LOG_ERROR("DLSS: Null DLSS system pointer");
        return;
    }

    m_dlssSystem = dlss;
    m_outputWidth = width;   // Native window resolution (target for upscaling)
    m_outputHeight = height;

    // Map int quality mode to DLSSQualityMode enum
    switch (qualityMode) {
        case 0: m_dlssQualityMode = DLSSSystem::DLSSQualityMode::Quality; break;
        case 1: m_dlssQualityMode = DLSSSystem::DLSSQualityMode::Balanced; break;
        case 2: m_dlssQualityMode = DLSSSystem::DLSSQualityMode::Performance; break;
        case 3: m_dlssQualityMode = DLSSSystem::DLSSQualityMode::UltraPerf; break;
        default: m_dlssQualityMode = DLSSSystem::DLSSQualityMode::Balanced; break;
    }

    // Store old resolution for logging
    uint32_t oldWidth = m_screenWidth;
    uint32_t oldHeight = m_screenHeight;

    // Get NGX parameters for resolution calculation
    NVSDK_NGX_Parameter* ngxParams = m_dlssSystem->GetParameters();
    if (!ngxParams) {
        LOG_ERROR("DLSS: Failed to get NGX parameters for resolution calculation");
        m_dlssSystem = nullptr;
        return;
    }

    // Calculate optimal render resolution using NGX_DLSS_GET_OPTIMAL_SETTINGS
    // This gives us the ideal resolution to render at for the chosen quality mode
    uint32_t optimalRenderWidth = 0;
    uint32_t optimalRenderHeight = 0;
    uint32_t minRenderWidth = 0;
    uint32_t minRenderHeight = 0;
    uint32_t maxRenderWidth = 0;
    uint32_t maxRenderHeight = 0;
    float sharpness = 0.0f;

    // Convert quality mode to NGX enum
    NVSDK_NGX_PerfQuality_Value perfQuality = NVSDK_NGX_PerfQuality_Value_Balanced;
    switch (m_dlssQualityMode) {
        case DLSSSystem::DLSSQualityMode::Quality:
            perfQuality = NVSDK_NGX_PerfQuality_Value_MaxQuality;
            break;
        case DLSSSystem::DLSSQualityMode::Balanced:
            perfQuality = NVSDK_NGX_PerfQuality_Value_Balanced;
            break;
        case DLSSSystem::DLSSQualityMode::Performance:
            perfQuality = NVSDK_NGX_PerfQuality_Value_MaxPerf;
            break;
        case DLSSSystem::DLSSQualityMode::UltraPerf:
            perfQuality = NVSDK_NGX_PerfQuality_Value_UltraPerformance;
            break;
    }

    // Manual resolution calculation based on DLSS quality mode percentages
    // This is more reliable than NGX_DLSS_GET_OPTIMAL_SETTINGS which often returns native res
    float renderScale = 1.0f;
    switch (m_dlssQualityMode) {
        case DLSSSystem::DLSSQualityMode::Quality:
            renderScale = 0.667f;  // 66.7% - 1.5× upscale
            break;
        case DLSSSystem::DLSSQualityMode::Balanced:
            renderScale = 0.58f;   // 58% - 1.72× upscale (RECOMMENDED)
            break;
        case DLSSSystem::DLSSQualityMode::Performance:
            renderScale = 0.5f;    // 50% - 2× upscale
            break;
        case DLSSSystem::DLSSQualityMode::UltraPerf:
            renderScale = 0.33f;   // 33% - 3× upscale
            break;
    }

    // Calculate render resolution
    m_renderWidth = static_cast<uint32_t>(width * renderScale);
    m_renderHeight = static_cast<uint32_t>(height * renderScale);

    // Align to 2 pixels (DLSS requirement)
    m_renderWidth = (m_renderWidth + 1) & ~1;
    m_renderHeight = (m_renderHeight + 1) & ~1;

    // Set recommended sharpness
    m_dlssSharpness = 0.0f;  // DLSS 3.x auto-tunes sharpness

    // Query NGX_DLSS_GET_OPTIMAL_SETTINGS for validation (optional)
    NVSDK_NGX_Parameter_SetUI(ngxParams, NVSDK_NGX_Parameter_OutWidth, width);
    NVSDK_NGX_Parameter_SetUI(ngxParams, NVSDK_NGX_Parameter_OutHeight, height);
    NVSDK_NGX_Parameter_SetI(ngxParams, NVSDK_NGX_Parameter_PerfQualityValue, perfQuality);

    NVSDK_NGX_Result result = NGX_DLSS_GET_OPTIMAL_SETTINGS(
        ngxParams,
        width, height,
        perfQuality,
        &optimalRenderWidth,
        &optimalRenderHeight,
        &maxRenderWidth,
        &maxRenderHeight,
        &minRenderWidth,
        &minRenderHeight,
        &sharpness
    );

    if (!NVSDK_NGX_FAILED(result)) {
        LOG_INFO("DLSS: NGX_DLSS_GET_OPTIMAL_SETTINGS validation:");
        LOG_INFO("  NGX recommended: {}x{}", optimalRenderWidth, optimalRenderHeight);
        LOG_INFO("  Manual calculation: {}x{}", m_renderWidth, m_renderHeight);
        LOG_INFO("  Using manual calculation (more reliable)");
    }

    LOG_INFO("DLSS: Resolution calculated successfully");
    LOG_INFO("  Old resolution: {}x{}", oldWidth, oldHeight);
    LOG_INFO("  Output resolution: {}x{}", m_outputWidth, m_outputHeight);
    LOG_INFO("  Render resolution: {}x{} ({:.1f}% scaling)",
             m_renderWidth, m_renderHeight,
             renderScale * 100.0f);
    LOG_INFO("  Quality mode: {} ({}% render scale)",
             m_dlssQualityMode == DLSSSystem::DLSSQualityMode::Quality ? "Quality" :
             m_dlssQualityMode == DLSSSystem::DLSSQualityMode::Balanced ? "Balanced" :
             m_dlssQualityMode == DLSSSystem::DLSSQualityMode::Performance ? "Performance" : "Ultra Performance",
             static_cast<int>(renderScale * 100));
    LOG_INFO("  Expected FPS boost: {:.2f}×",
             (float)(m_outputWidth * m_outputHeight) / (float)(m_renderWidth * m_renderHeight));

    // Update m_screenWidth/m_screenHeight to use render resolution
    // This ensures all rendering happens at the lower resolution
    m_screenWidth = m_renderWidth;
    m_screenHeight = m_renderHeight;

    // Recreate output texture at render resolution (lower)
    LOG_INFO("DLSS: Recreating output texture at render resolution...");
    m_outputTexture.Reset();
    if (!CreateOutputTexture(m_renderWidth, m_renderHeight)) {
        LOG_ERROR("DLSS: Failed to recreate output texture at render resolution");
        m_dlssSystem = nullptr;
        return;
    }

    // Create upscaled output texture at output resolution (native)
    LOG_INFO("DLSS: Creating upscaled output texture at native resolution...");
    D3D12_RESOURCE_DESC texDesc = {};
    texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texDesc.Width = m_outputWidth;
    texDesc.Height = m_outputHeight;
    texDesc.DepthOrArraySize = 1;
    texDesc.MipLevels = 1;
    texDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    texDesc.SampleDesc.Count = 1;
    texDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    D3D12_HEAP_PROPERTIES defaultHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    HRESULT hr = m_device->GetDevice()->CreateCommittedResource(
        &defaultHeap,
        D3D12_HEAP_FLAG_NONE,
        &texDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,
        IID_PPV_ARGS(&m_upscaledOutputTexture)
    );

    if (FAILED(hr)) {
        LOG_ERROR("DLSS: Failed to create upscaled output texture");
        m_dlssSystem = nullptr;
        return;
    }

    // Create SRV for blit pass (DLSS output → swap chain)
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Texture2D.MipLevels = 1;

    m_upscaledOutputSRV = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    m_device->GetDevice()->CreateShaderResourceView(
        m_upscaledOutputTexture.Get(),
        &srvDesc,
        m_upscaledOutputSRV
    );
    m_upscaledOutputSRVGPU = m_resources->GetGPUHandle(m_upscaledOutputSRV);

    LOG_INFO("DLSS: Upscaled output texture created successfully");
    LOG_INFO("  Resolution: {}x{}", m_outputWidth, m_outputHeight);
    LOG_INFO("  SRV GPU handle: 0x{:016X}", m_upscaledOutputSRVGPU.ptr);

    // Recreate motion vector buffer at render resolution
    LOG_INFO("DLSS: Recreating motion vector buffer at render resolution...");
    m_motionVectorBuffer.Reset();

    D3D12_RESOURCE_DESC mvTexDesc = {};
    mvTexDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    mvTexDesc.Width = m_renderWidth;
    mvTexDesc.Height = m_renderHeight;
    mvTexDesc.DepthOrArraySize = 1;
    mvTexDesc.MipLevels = 1;
    mvTexDesc.Format = DXGI_FORMAT_R16G16_FLOAT;
    mvTexDesc.SampleDesc.Count = 1;
    mvTexDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    hr = m_device->GetDevice()->CreateCommittedResource(
        &defaultHeap,
        D3D12_HEAP_FLAG_NONE,
        &mvTexDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,
        IID_PPV_ARGS(&m_motionVectorBuffer)
    );

    if (FAILED(hr)) {
        LOG_ERROR("DLSS: Failed to recreate motion vector buffer");
        m_dlssSystem = nullptr;
        return;
    }

    // Recreate depth buffer at render resolution
    LOG_INFO("DLSS: Recreating depth buffer at render resolution...");
    m_depthBuffer.Reset();

    D3D12_RESOURCE_DESC depthTexDesc = {};
    depthTexDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    depthTexDesc.Width = m_renderWidth;
    depthTexDesc.Height = m_renderHeight;
    depthTexDesc.DepthOrArraySize = 1;
    depthTexDesc.MipLevels = 1;
    depthTexDesc.Format = DXGI_FORMAT_R32_FLOAT;
    depthTexDesc.SampleDesc.Count = 1;
    depthTexDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    hr = m_device->GetDevice()->CreateCommittedResource(
        &defaultHeap,
        D3D12_HEAP_FLAG_NONE,
        &depthTexDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,
        IID_PPV_ARGS(&m_depthBuffer)
    );

    if (FAILED(hr)) {
        LOG_ERROR("DLSS: Failed to recreate depth buffer");
        m_dlssSystem = nullptr;
        return;
    }

    // Recreate PCSS shadow buffers at render resolution
    LOG_INFO("DLSS: Recreating PCSS shadow buffers at render resolution...");
    for (int i = 0; i < 2; i++) {
        m_shadowBuffer[i].Reset();

        D3D12_RESOURCE_DESC shadowTexDesc = {};
        shadowTexDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        shadowTexDesc.Width = m_renderWidth;
        shadowTexDesc.Height = m_renderHeight;
        shadowTexDesc.DepthOrArraySize = 1;
        shadowTexDesc.MipLevels = 1;
        shadowTexDesc.Format = DXGI_FORMAT_R16_FLOAT;
        shadowTexDesc.SampleDesc.Count = 1;
        shadowTexDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        hr = m_device->GetDevice()->CreateCommittedResource(
            &defaultHeap,
            D3D12_HEAP_FLAG_NONE,
            &shadowTexDesc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_PPV_ARGS(&m_shadowBuffer[i])
        );

        if (FAILED(hr)) {
            LOG_ERROR("DLSS: Failed to recreate shadow buffer {}", i);
            m_dlssSystem = nullptr;
            return;
        }

        // Recreate SRV
        D3D12_SHADER_RESOURCE_VIEW_DESC shadowSrvDesc = {};
        shadowSrvDesc.Format = DXGI_FORMAT_R16_FLOAT;
        shadowSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        shadowSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        shadowSrvDesc.Texture2D.MipLevels = 1;

        m_shadowSRV[i] = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        m_device->GetDevice()->CreateShaderResourceView(
            m_shadowBuffer[i].Get(),
            &shadowSrvDesc,
            m_shadowSRV[i]
        );
        m_shadowSRVGPU[i] = m_resources->GetGPUHandle(m_shadowSRV[i]);

        // Recreate UAV
        D3D12_UNORDERED_ACCESS_VIEW_DESC shadowUavDesc = {};
        shadowUavDesc.Format = DXGI_FORMAT_R16_FLOAT;
        shadowUavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        shadowUavDesc.Texture2D.MipSlice = 0;

        m_shadowUAV[i] = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        m_device->GetDevice()->CreateUnorderedAccessView(
            m_shadowBuffer[i].Get(),
            nullptr,
            &shadowUavDesc,
            m_shadowUAV[i]
        );
        m_shadowUAVGPU[i] = m_resources->GetGPUHandle(m_shadowUAV[i]);
    }

    LOG_INFO("DLSS: All buffers recreated successfully at optimal resolutions");
    LOG_INFO("  Render buffers: {}x{} (motion vectors, depth, color, shadows)", m_renderWidth, m_renderHeight);
    LOG_INFO("  Shadow buffer size: {} MB per buffer", (m_renderWidth * m_renderHeight * 2) / (1024 * 1024));
    LOG_INFO("  Output buffer: {}x{} (upscaled result)", m_outputWidth, m_outputHeight);

    // Mark feature for recreation with new parameters
    m_dlssFeatureCreated = false;
    m_dlssFirstFrame = true;  // Reset history when changing settings
}
#endif // ENABLE_DLSS
