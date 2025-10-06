// ParticleRenderer - Debug Utilities Implementation
// PIX markers, debug logging, and test modes for particle rendering

#include "ParticleRenderer.h"
#include "../core/Device.h"
#include "../utils/Logger.h"
#include <pix3.h>
#include <DirectXMath.h>

// Debug mode control
enum class ParticleDebugMode {
    Normal,           // Regular particle rendering
    HardcodedQuad,   // Single hardcoded quad in clip space
    SingleParticle,  // Only render particle 0
    GridTest,        // 2x2 grid of particles
    VertexColors     // Each vertex gets different color
};

static ParticleDebugMode g_debugMode = ParticleDebugMode::HardcodedQuad;
static uint32_t g_debugFrameCounter = 0;

void ParticleRenderer::SetDebugMode(int mode) {
    g_debugMode = static_cast<ParticleDebugMode>(mode);
    LOG_INFO("Particle debug mode set to: {}", mode);
}

void ParticleRenderer::AddPIXMarkers(ID3D12GraphicsCommandList* cmdList,
                                    uint32_t particleCount,
                                    bool rtLightingEnabled) {
    // Main render event
    PIXBeginEvent(cmdList, PIX_COLOR_INDEX(0), "ParticleBillboardRender");

    // Mark render configuration
    PIXSetMarker(cmdList, PIX_COLOR_INDEX(1),
                "Config: Particles=%d RTLighting=%s DebugMode=%d",
                particleCount,
                rtLightingEnabled ? "ON" : "OFF",
                static_cast<int>(g_debugMode));

    // Mark specific debug modes
    switch (g_debugMode) {
        case ParticleDebugMode::HardcodedQuad:
            PIXBeginEvent(cmdList, PIX_COLOR_INDEX(2), "DEBUG_HardcodedQuad");
            break;
        case ParticleDebugMode::SingleParticle:
            PIXBeginEvent(cmdList, PIX_COLOR_INDEX(2), "DEBUG_SingleParticle");
            break;
        case ParticleDebugMode::GridTest:
            PIXBeginEvent(cmdList, PIX_COLOR_INDEX(2), "DEBUG_GridTest");
            break;
        case ParticleDebugMode::VertexColors:
            PIXBeginEvent(cmdList, PIX_COLOR_INDEX(2), "DEBUG_VertexColors");
            break;
        default:
            PIXBeginEvent(cmdList, PIX_COLOR_INDEX(2), "NormalRender");
            break;
    }
}

void ParticleRenderer::EndPIXMarkers(ID3D12GraphicsCommandList* cmdList) {
    PIXEndEvent(cmdList);  // End debug mode event
    PIXEndEvent(cmdList);  // End main render event
}

void ParticleRenderer::LogVertexDebugInfo(uint32_t frameNum, uint32_t particleCount) {
    // Only log on specific frames to avoid spam
    if (frameNum == 0 || frameNum == 60 || frameNum == 120 || frameNum == 300) {
        LOG_INFO("=== Particle Vertex Debug Frame {} ===", frameNum);
        LOG_INFO("Debug Mode: {}", static_cast<int>(g_debugMode));
        LOG_INFO("Draw call: DrawInstanced(6, {}, 0, 0)", particleCount);
        LOG_INFO("Expected triangle formation:");
        LOG_INFO("  Triangle 0: V0(BL) -> V1(BR) -> V2(TR) [CCW]");
        LOG_INFO("  Triangle 1: V3(BL) -> V4(TR) -> V5(TL) [CCW]");

        if (g_debugMode == ParticleDebugMode::HardcodedQuad) {
            LOG_INFO("Hardcoded positions (clip space):");
            LOG_INFO("  V0: (-0.25, -0.25) [Red]");
            LOG_INFO("  V1: ( 0.25, -0.25) [Green]");
            LOG_INFO("  V2: ( 0.25,  0.25) [Blue]");
            LOG_INFO("  V3: (-0.25, -0.25) [Red]");
            LOG_INFO("  V4: ( 0.25,  0.25) [Blue]");
            LOG_INFO("  V5: (-0.25,  0.25) [Yellow]");
        }
    }
}

// Enhanced render function with debug support
void ParticleRenderer::RenderWithDebug(ID3D12GraphicsCommandList* cmdList,
                                       ID3D12Resource* particleBuffer,
                                       ID3D12Resource* rtLightingBuffer,
                                       const RenderConstants& constants) {
    // Add PIX markers
    AddPIXMarkers(cmdList, m_particleCount, rtLightingBuffer != nullptr);

    // Log debug info
    LogVertexDebugInfo(g_debugFrameCounter, m_particleCount);

    // Choose shader based on debug mode
    ID3D12PipelineState* pso = m_rasterPSO.Get();
    if (g_debugMode == ParticleDebugMode::HardcodedQuad ||
        g_debugMode == ParticleDebugMode::VertexColors) {
        // Use debug PSO if available
        if (m_debugPSO) {
            pso = m_debugPSO.Get();
        }
    }

    // Set pipeline state
    cmdList->SetPipelineState(pso);
    cmdList->SetGraphicsRootSignature(m_rasterRootSignature.Get());
    cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    // Calculate camera vectors for billboarding
    DirectX::XMVECTOR cameraPos = DirectX::XMLoadFloat3(&constants.cameraPos);
    DirectX::XMVECTOR lookAt = DirectX::XMVectorSet(0, 0, 0, 0);
    DirectX::XMVECTOR worldUp = DirectX::XMLoadFloat3(&constants.cameraUp);

    DirectX::XMVECTOR forward = DirectX::XMVector3Normalize(DirectX::XMVectorSubtract(lookAt, cameraPos));
    DirectX::XMVECTOR cameraRight = DirectX::XMVector3Normalize(DirectX::XMVector3Cross(forward, worldUp));
    DirectX::XMVECTOR cameraUpVec = DirectX::XMVector3Normalize(DirectX::XMVector3Cross(forward, cameraRight));

    // Set camera constants
    struct CameraConstants {
        DirectX::XMFLOAT4X4 viewProj;
        DirectX::XMFLOAT3 cameraPos;
        float padding0;
        DirectX::XMFLOAT3 cameraRight;
        float padding1;
        DirectX::XMFLOAT3 cameraUp;
        float padding2;
    } cameraConsts;

    cameraConsts.viewProj = constants.viewProj;
    DirectX::XMStoreFloat3(&cameraConsts.cameraPos, cameraPos);
    DirectX::XMStoreFloat3(&cameraConsts.cameraRight, cameraRight);
    DirectX::XMStoreFloat3(&cameraConsts.cameraUp, cameraUpVec);

    cmdList->SetGraphicsRoot32BitConstants(0, 28, &cameraConsts, 0);

    // Set particle constants
    struct ParticleConstants {
        float particleRadius;
        float debugMode;
        float frameNumber;
        float padding;
    } particleConsts = {
        constants.particleSize,
        static_cast<float>(g_debugMode),
        static_cast<float>(g_debugFrameCounter),
        0
    };

    cmdList->SetGraphicsRoot32BitConstants(1, 4, &particleConsts, 0);
    cmdList->SetGraphicsRootShaderResourceView(2, particleBuffer->GetGPUVirtualAddress());

    if (rtLightingBuffer) {
        cmdList->SetGraphicsRootShaderResourceView(3, rtLightingBuffer->GetGPUVirtualAddress());
    }

    // Adjust instance count based on debug mode
    uint32_t instanceCount = m_particleCount;
    if (g_debugMode == ParticleDebugMode::HardcodedQuad ||
        g_debugMode == ParticleDebugMode::SingleParticle) {
        instanceCount = 1;  // Only render first instance
    } else if (g_debugMode == ParticleDebugMode::GridTest) {
        instanceCount = min(4, m_particleCount);  // Render up to 4 particles
    }

    // Mark the actual draw call
    PIXBeginEvent(cmdList, PIX_COLOR_INDEX(3), "DrawInstanced_%d", instanceCount);
    cmdList->DrawInstanced(6, instanceCount, 0, 0);
    PIXEndEvent(cmdList);

    // End PIX markers
    EndPIXMarkers(cmdList);

    g_debugFrameCounter++;
}

// Test utility functions
void ParticleRenderer::RunVertexOrderTest() {
    LOG_INFO("=== Running Vertex Order Test ===");

    // Test the corrected vertex ordering
    uint32_t vertexIndices[6] = {0, 1, 2, 3, 4, 5};
    uint32_t cornerIndices[6];

    // Apply the fixed mapping
    for (uint32_t i = 0; i < 6; i++) {
        uint32_t vertIdx = vertexIndices[i];
        uint32_t cornerIdx;

        if (vertIdx == 0) cornerIdx = 0;      // BL
        else if (vertIdx == 1) cornerIdx = 1; // BR
        else if (vertIdx == 2) cornerIdx = 3; // TR
        else if (vertIdx == 3) cornerIdx = 0; // BL
        else if (vertIdx == 4) cornerIdx = 3; // TR
        else cornerIdx = 2;                    // TL

        cornerIndices[i] = cornerIdx;
    }

    LOG_INFO("Vertex -> Corner mapping:");
    const char* cornerNames[4] = {"BL", "BR", "TL", "TR"};
    for (uint32_t i = 0; i < 6; i++) {
        LOG_INFO("  V{} -> Corner {} ({})", i, cornerIndices[i], cornerNames[cornerIndices[i]]);
    }

    LOG_INFO("Triangle 0: {} -> {} -> {}",
             cornerNames[cornerIndices[0]],
             cornerNames[cornerIndices[1]],
             cornerNames[cornerIndices[2]]);

    LOG_INFO("Triangle 1: {} -> {} -> {}",
             cornerNames[cornerIndices[3]],
             cornerNames[cornerIndices[4]],
             cornerNames[cornerIndices[5]]);

    // Verify triangles cover the quad
    bool triangle0_valid = (cornerIndices[0] != cornerIndices[1]) &&
                          (cornerIndices[1] != cornerIndices[2]) &&
                          (cornerIndices[0] != cornerIndices[2]);

    bool triangle1_valid = (cornerIndices[3] != cornerIndices[4]) &&
                          (cornerIndices[4] != cornerIndices[5]) &&
                          (cornerIndices[3] != cornerIndices[5]);

    LOG_INFO("Triangle 0 valid (no duplicate corners): {}", triangle0_valid ? "YES" : "NO");
    LOG_INFO("Triangle 1 valid (no duplicate corners): {}", triangle1_valid ? "YES" : "NO");

    // Check if all corners are covered
    bool corners_used[4] = {false, false, false, false};
    for (uint32_t i = 0; i < 6; i++) {
        corners_used[cornerIndices[i]] = true;
    }

    LOG_INFO("All corners used: BL={} BR={} TL={} TR={}",
             corners_used[0], corners_used[1], corners_used[2], corners_used[3]);

    bool all_corners_used = corners_used[0] && corners_used[1] && corners_used[2] && corners_used[3];
    LOG_INFO("RESULT: {}", all_corners_used ? "PASS - Quad fully covered" : "FAIL - Missing corners");
}