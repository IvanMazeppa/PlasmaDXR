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
    // MAX_LIGHTS = 16, Light struct = 32 bytes (position=12, intensity=4, color=12, radius=4)
    const uint32_t MAX_LIGHTS = 16;
    const uint32_t lightStructSize = 32;  // Must match HLSL Light struct
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

    if (!CreatePipeline()) {
        return false;
    }

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
    // u0: RWTexture2D<float4> g_output (descriptor table - typed UAV requirement)
    // u2: RWTexture2D<float> g_currShadow (PCSS temporal shadow - current frame, descriptor table)
    CD3DX12_DESCRIPTOR_RANGE1 srvRanges[1];
    srvRanges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 5);  // t5: Texture2D (prev shadow)

    CD3DX12_DESCRIPTOR_RANGE1 uavRanges[2];
    uavRanges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // u0: RWTexture2D (output)
    uavRanges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 2);  // u2: RWTexture2D (current shadow)

    CD3DX12_ROOT_PARAMETER1 rootParams[8];
    rootParams[0].InitAsConstantBufferView(0);              // b0 - CBV (no DWORD limit!)
    rootParams[1].InitAsShaderResourceView(0);              // t0 - particles (raw buffer is OK)
    rootParams[2].InitAsShaderResourceView(1);              // t1 - rtLighting (raw buffer is OK)
    rootParams[3].InitAsShaderResourceView(2);              // t2 - TLAS (raw is OK)
    rootParams[4].InitAsShaderResourceView(4);              // t4 - lights (raw buffer is OK)
    rootParams[5].InitAsDescriptorTable(1, &uavRanges[0]);  // u0 - output texture
    rootParams[6].InitAsDescriptorTable(1, &srvRanges[0]);  // t5 - previous shadow (SRV)
    rootParams[7].InitAsDescriptorTable(1, &uavRanges[1]);  // u2 - current shadow (UAV)

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSigDesc;
    rootSigDesc.Init_1_1(8, rootParams, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);

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

    // PCSS: Swap shadow buffers for next frame (ping-pong)
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
