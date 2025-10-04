#include "RTLightingSystem.h"
#include "../core/Device.h"
#include "../utils/ResourceManager.h"
#include "../utils/Logger.h"

RTLightingSystem::~RTLightingSystem() {
    Shutdown();
}

bool RTLightingSystem::Initialize(Device* device, ResourceManager* resources, uint32_t particleCount) {
    m_device = device;
    m_resources = resources;
    m_particleCount = particleCount;

    LOG_INFO("Initializing RT Lighting System for {} particles...", particleCount);

    // Check DXR support
    if (!m_device->SupportsDXR()) {
        LOG_ERROR("DXR not supported on this device!");
        return false;
    }

    LOG_INFO("  DXR Tier: {}", static_cast<int>(m_device->GetRaytracingTier()));

    // Create output lighting buffer
    if (!CreateOutputBuffers()) {
        LOG_ERROR("Failed to create output buffers");
        return false;
    }

    // For minimal version: Skip full RT pipeline creation
    // Just create the output buffer that will be filled with GREEN
    LOG_INFO("RT Lighting System initialized (minimal GREEN test mode)");
    LOG_INFO("  Note: Full DXR pipeline not yet implemented");
    LOG_INFO("  Lighting buffer will be filled with test pattern");

    return true;
}

void RTLightingSystem::Shutdown() {
    m_rtGlobalRootSignature.Reset();
    m_rtPipelineState.Reset();
    m_sbtBuffer.Reset();
    m_bottomLevelAS.Reset();
    m_topLevelAS.Reset();
    m_scratchBuffer.Reset();
    m_instanceDescsBuffer.Reset();
    m_lightingBuffer.Reset();

    LOG_INFO("RT Lighting System shut down");
}

bool RTLightingSystem::CreateOutputBuffers() {
    // Create lighting output buffer (RGB float3 per particle)
    ResourceManager::BufferDesc desc = {};
    desc.size = m_particleCount * sizeof(DirectX::XMFLOAT3);
    desc.heapType = D3D12_HEAP_TYPE_DEFAULT;
    desc.flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    desc.initialState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

    m_lightingBuffer = m_resources->CreateBuffer("RTLightingBuffer", desc);
    if (!m_lightingBuffer) {
        LOG_ERROR("Failed to create RT lighting buffer");
        return false;
    }

    LOG_INFO("RT lighting buffer created ({} MB)",
             (desc.size / (1024 * 1024)));

    return true;
}

bool RTLightingSystem::CreateRootSignatures() {
    // To be implemented when we add full RT pipeline
    return true;
}

bool RTLightingSystem::CreateRTPipeline() {
    // To be implemented when we add full RT pipeline
    return true;
}

bool RTLightingSystem::CreateShaderBindingTable() {
    // To be implemented when we add full RT pipeline
    return true;
}

void RTLightingSystem::BuildBLAS(ID3D12GraphicsCommandList4* cmdList) {
    // Build Bottom Level Acceleration Structure
    // To be implemented when we add full RT pipeline
}

void RTLightingSystem::BuildTLAS(ID3D12GraphicsCommandList4* cmdList,
                                 ID3D12Resource* particleBuffer,
                                 uint32_t particleCount) {
    // Build Top Level Acceleration Structure
    // To be implemented when we add full RT pipeline
}

void RTLightingSystem::ComputeLighting(ID3D12GraphicsCommandList4* cmdList,
                                      const RTConstants& constants) {
    // For minimal version: Just clear lighting buffer to GREEN
    // This proves the buffer exists and can be accessed

    // GREEN test pattern (0, 100, 0) will be filled by a simple compute shader
    // For now, just log that we're "computing" lighting

    static uint32_t frameCount = 0;
    if (frameCount % 60 == 0) {
        LOG_INFO("RT Lighting compute called (frame {}) - GREEN test mode", frameCount);
    }
    frameCount++;

    // TODO: Fill lighting buffer with GREEN via compute shader
    // For full implementation:
    // 1. Build/update acceleration structures
    // 2. DispatchRays() with DXR
    // 3. Output particle-to-particle lighting
}