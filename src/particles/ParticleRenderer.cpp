#include "ParticleRenderer.h"
#include "../core/Device.h"
#include "../core/FeatureDetector.h"
#include "../utils/ResourceManager.h"
#include "../utils/Logger.h"

ParticleRenderer::~ParticleRenderer() {
    m_meshRootSignature.Reset();
    m_meshPSO.Reset();
    m_computeRootSignature.Reset();
    m_computePSO.Reset();
    m_rasterRootSignature.Reset();
    m_rasterPSO.Reset();
    m_vertexBuffer.Reset();
    m_indexBuffer.Reset();
    m_constantBuffer.Reset();
}

bool ParticleRenderer::Initialize(Device* device, ResourceManager* resources,
                                 const FeatureDetector* features, uint32_t particleCount) {
    m_device = device;
    m_resources = resources;
    m_particleCount = particleCount;

    LOG_INFO("Initializing ParticleRenderer for {} particles...", particleCount);

    // Determine rendering path based on feature detection
    if (features->CanUseMeshShaders()) {
        LOG_INFO("Mesh shaders available - attempting mesh shader path");
        m_activePath = RenderPath::MeshShaders;

        if (!InitializeMeshShaderPath()) {
            LOG_WARN("Mesh shader path failed, falling back to compute");
            m_activePath = RenderPath::ComputeFallback;
        }
    }

    // If mesh shaders aren't available or failed, use compute fallback
    if (m_activePath == RenderPath::ComputeFallback || !features->CanUseMeshShaders()) {
        LOG_INFO("Using compute shader fallback path");
        m_activePath = RenderPath::ComputeFallback;

        if (!InitializeComputeFallbackPath()) {
            LOG_ERROR("Compute fallback path also failed!");
            return false;
        }
    }

    LOG_INFO("ParticleRenderer initialized successfully");
    LOG_INFO("  Active path: {}", GetActivePathName());
    LOG_INFO("  Particle count: {}", particleCount);

    return true;
}

bool ParticleRenderer::InitializeMeshShaderPath() {
    // Mesh shader path - will implement later
    // For now, return false to force compute fallback
    LOG_WARN("Mesh shader path not yet implemented - use compute fallback");
    return false;
}

// Implementation in ParticleRenderer_Billboard.cpp

void ParticleRenderer::Render(ID3D12GraphicsCommandList* cmdList,
                              ID3D12Resource* particleBuffer,
                              ID3D12Resource* rtLightingBuffer,
                              const RenderConstants& constants) {
    if (!cmdList || !particleBuffer) {
        LOG_ERROR("Invalid parameters for particle rendering");
        return;
    }

    // For minimal version: Just validate we can get here
    // Full rendering will be implemented after we verify the pipeline works

    if (m_activePath == RenderPath::MeshShaders) {
        RenderWithMeshShaders(cmdList, particleBuffer, rtLightingBuffer, constants);
    } else {
        RenderWithComputeFallback(cmdList, particleBuffer, rtLightingBuffer, constants);
    }
}

void ParticleRenderer::RenderWithMeshShaders(ID3D12GraphicsCommandList* cmdList,
                                            ID3D12Resource* particleBuffer,
                                            ID3D12Resource* rtLightingBuffer,
                                            const RenderConstants& constants) {
    // Mesh shader rendering - to be implemented
    LOG_WARN("Mesh shader rendering not yet implemented");
}

// Implementation in ParticleRenderer_Billboard.cpp