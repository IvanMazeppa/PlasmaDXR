// ParticleRenderer - Billboard Implementation
// Uses traditional VS/PS pipeline with instanced rendering
// Each particle instance generates a camera-facing billboard

#include "ParticleRenderer.h"
#include "../core/Device.h"
#include "../utils/ResourceManager.h"
#include "../utils/Logger.h"
#include "../utils/d3dx12/d3dx12.h"
#include <fstream>
#include <d3dcompiler.h>

#pragma comment(lib, "d3dcompiler.lib")

bool ParticleRenderer::InitializeComputeFallbackPath() {
    LOG_INFO("Initializing billboard particle renderer...");

    // Load vertex shader - try multiple paths for different working directories
    std::vector<std::string> vsShaderPaths = {
        "shaders/particles/particle_billboard_vs.dxil",           // From project root
        "../shaders/particles/particle_billboard_vs.dxil",        // From build/
        "../../shaders/particles/particle_billboard_vs.dxil"      // From build/Debug/
    };

    std::ifstream vsFile;
    std::string foundVsPath;
    for (const auto& path : vsShaderPaths) {
        vsFile.open(path, std::ios::binary);
        if (vsFile) {
            foundVsPath = path;
            LOG_INFO("Found particle_billboard_vs.dxil at: {}", path);
            break;
        }
        vsFile.clear();
    }

    if (!vsFile) {
        LOG_ERROR("Failed to open particle_billboard_vs.dxil - tried {} paths", vsShaderPaths.size());
        for (const auto& path : vsShaderPaths) {
            LOG_ERROR("  - {}", path);
        }
        return false;
    }

    std::vector<char> vsData((std::istreambuf_iterator<char>(vsFile)), std::istreambuf_iterator<char>());
    Microsoft::WRL::ComPtr<ID3DBlob> vsBlob;
    HRESULT hr = D3DCreateBlob(vsData.size(), &vsBlob);
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create VS blob");
        return false;
    }
    memcpy(vsBlob->GetBufferPointer(), vsData.data(), vsData.size());

    // Load pixel shader - try multiple paths for different working directories
    std::vector<std::string> psShaderPaths = {
        "shaders/particles/particle_billboard_ps.dxil",           // From project root
        "../shaders/particles/particle_billboard_ps.dxil",        // From build/
        "../../shaders/particles/particle_billboard_ps.dxil"      // From build/Debug/
    };

    std::ifstream psFile;
    std::string foundPsPath;
    for (const auto& path : psShaderPaths) {
        psFile.open(path, std::ios::binary);
        if (psFile) {
            foundPsPath = path;
            LOG_INFO("Found particle_billboard_ps.dxil at: {}", path);
            break;
        }
        psFile.clear();
    }

    if (!psFile) {
        LOG_ERROR("Failed to open particle_billboard_ps.dxil - tried {} paths", psShaderPaths.size());
        for (const auto& path : psShaderPaths) {
            LOG_ERROR("  - {}", path);
        }
        return false;
    }

    std::vector<char> psData((std::istreambuf_iterator<char>(psFile)), std::istreambuf_iterator<char>());
    Microsoft::WRL::ComPtr<ID3DBlob> psBlob;
    hr = D3DCreateBlob(psData.size(), &psBlob);
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create PS blob");
        return false;
    }
    memcpy(psBlob->GetBufferPointer(), psData.data(), psData.size());

    // Create root signature
    // b0: Camera constants (viewProj, cameraPos, cameraRight, cameraUp)
    // b1: Particle constants (radius, etc.)
    // t0: StructuredBuffer<Particle> particles
    // t1: Buffer<float4> rtLighting
    {
        CD3DX12_ROOT_PARAMETER1 rootParams[4];
        rootParams[0].InitAsConstants(28, 0);  // b0: Camera (4x4 matrix=16 + 3 vec3=9 + 3 padding=3 = 28 DWORDs)
        rootParams[1].InitAsConstants(4, 1);   // b1: Particle constants
        rootParams[2].InitAsShaderResourceView(0);  // t0: particles
        rootParams[3].InitAsShaderResourceView(1);  // t1: RT lighting

        CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSigDesc;
        rootSigDesc.Init_1_1(4, rootParams, 0, nullptr,
            D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

        Microsoft::WRL::ComPtr<ID3DBlob> signature, error;
        hr = D3DX12SerializeVersionedRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1_1, &signature, &error);
        if (FAILED(hr)) {
            if (error) {
                LOG_ERROR("Root signature serialization failed: {}", (char*)error->GetBufferPointer());
            }
            return false;
        }

        hr = m_device->GetDevice()->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(),
                                                         IID_PPV_ARGS(&m_rasterRootSignature));
        if (FAILED(hr)) {
            LOG_ERROR("Failed to create billboard root signature");
            return false;
        }
    }

    // Create graphics PSO for billboards
    {
        LOG_INFO("Creating billboard PSO...");
        LOG_INFO("  VS shader: {} bytes", vsBlob->GetBufferSize());
        LOG_INFO("  PS shader: {} bytes", psBlob->GetBufferSize());

        D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = m_rasterRootSignature.Get();
        psoDesc.IBStripCutValue = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED;
        psoDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
        psoDesc.NodeMask = 0;
        psoDesc.VS = CD3DX12_SHADER_BYTECODE(vsBlob.Get());
        psoDesc.PS = CD3DX12_SHADER_BYTECODE(psBlob.Get());

        // Blend state for additive blending
        psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
        psoDesc.BlendState.RenderTarget[0].BlendEnable = TRUE;
        psoDesc.BlendState.RenderTarget[0].SrcBlend = D3D12_BLEND_SRC_ALPHA;
        psoDesc.BlendState.RenderTarget[0].DestBlend = D3D12_BLEND_ONE;
        psoDesc.BlendState.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
        psoDesc.BlendState.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ONE;
        psoDesc.BlendState.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_ZERO;
        psoDesc.BlendState.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
        psoDesc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;

        // Rasterizer state
        psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
        psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;  // Billboard faces camera

        // Depth state - no depth writes for particles
        psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
        psoDesc.DepthStencilState.DepthEnable = FALSE;
        psoDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;

        // Input layout - none needed (using SV_VertexID)
        psoDesc.InputLayout = { nullptr, 0 };

        // Sample mask and topology
        psoDesc.SampleMask = UINT_MAX;
        psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;

        // Render target format
        psoDesc.NumRenderTargets = 1;
        psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        psoDesc.DSVFormat = DXGI_FORMAT_UNKNOWN;  // No depth buffer
        psoDesc.SampleDesc.Count = 1;
        psoDesc.SampleDesc.Quality = 0;

        hr = m_device->GetDevice()->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_rasterPSO));
        if (FAILED(hr)) {
            char hrStr[32];
            sprintf_s(hrStr, "0x%08X", static_cast<unsigned int>(hr));
            LOG_ERROR("Failed to create billboard graphics PSO");
            LOG_ERROR("  HRESULT: {}", hrStr);

            // Try to get debug messages from info queue
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

            // Common errors:
            if (hr == E_INVALIDARG) LOG_ERROR("  Cause: E_INVALIDARG - Invalid PSO descriptor");
            if (hr == E_OUTOFMEMORY) LOG_ERROR("  Cause: E_OUTOFMEMORY");

            return false;
        }
    }

    LOG_INFO("Billboard particle renderer initialized successfully");
    return true;
}

void ParticleRenderer::RenderWithComputeFallback(ID3D12GraphicsCommandList* cmdList,
                                                  ID3D12Resource* particleBuffer,
                                                  ID3D12Resource* rtLightingBuffer,
                                                  const RenderConstants& constants) {
    if (!m_rasterPSO || !m_rasterRootSignature) {
        LOG_ERROR("Billboard render: PSO or root signature not initialized!");
        return;  // Not initialized
    }

    static int frameCount = 0;
    bool isFirstFrame = (frameCount == 0);
    if (isFirstFrame) {
        LOG_INFO("=== First Billboard Render ===");
        LOG_INFO("  Particle count: {}", m_particleCount);
        LOG_INFO("  Particle buffer: {}", particleBuffer ? "Valid" : "NULL");
        LOG_INFO("  RT lighting buffer: {}", rtLightingBuffer ? "Valid" : "NULL");
        LOG_INFO("  Camera: ({}, {}, {})", constants.cameraPos.x, constants.cameraPos.y, constants.cameraPos.z);
        LOG_INFO("  Particle size: {}", constants.particleSize);
    }
    frameCount++;

    // Set pipeline state
    cmdList->SetPipelineState(m_rasterPSO.Get());
    cmdList->SetGraphicsRootSignature(m_rasterRootSignature.Get());
    cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    // Calculate billboard-facing vectors
    // Use Y-axis aligned billboards (up is always world Y)
    DirectX::XMVECTOR cameraPos = DirectX::XMLoadFloat3(&constants.cameraPos);
    DirectX::XMVECTOR lookAt = DirectX::XMVectorSet(0, 0, 0, 0);

    // Forward = direction from camera to lookat (view direction)
    DirectX::XMVECTOR forward = DirectX::XMVector3Normalize(DirectX::XMVectorSubtract(lookAt, cameraPos));

    // For Y-axis billboards: up is ALWAYS (0, 1, 0)
    DirectX::XMVECTOR cameraUpVec = DirectX::XMVectorSet(0, 1, 0, 0);

    // Right = up × forward (perpendicular to both, pointing right)
    DirectX::XMVECTOR cameraRight = DirectX::XMVector3Normalize(DirectX::XMVector3Cross(cameraUpVec, forward));

    // Set root parameters
    struct CameraConstants {
        DirectX::XMFLOAT4X4 viewProj;
        DirectX::XMFLOAT3 cameraPos;
        float padding0;
        DirectX::XMFLOAT3 cameraRight;
        float padding1;
        DirectX::XMFLOAT3 cameraUp;
        float padding2;
    } cameraConsts;

    // ViewProj matrix is already transposed from Application.cpp, just copy it
    cameraConsts.viewProj = constants.viewProj;
    DirectX::XMStoreFloat3(&cameraConsts.cameraPos, cameraPos);
    DirectX::XMStoreFloat3(&cameraConsts.cameraRight, cameraRight);
    DirectX::XMStoreFloat3(&cameraConsts.cameraUp, cameraUpVec);

    // Log matrix and camera vectors on first frame
    if (isFirstFrame) {
        LOG_INFO("  ViewProj row 0: ({}, {}, {}, {})", cameraConsts.viewProj._11, cameraConsts.viewProj._12, cameraConsts.viewProj._13, cameraConsts.viewProj._14);
        LOG_INFO("  ViewProj row 1: ({}, {}, {}, {})", cameraConsts.viewProj._21, cameraConsts.viewProj._22, cameraConsts.viewProj._23, cameraConsts.viewProj._24);
        LOG_INFO("  ViewProj row 2: ({}, {}, {}, {})", cameraConsts.viewProj._31, cameraConsts.viewProj._32, cameraConsts.viewProj._33, cameraConsts.viewProj._34);
        LOG_INFO("  ViewProj row 3: ({}, {}, {}, {})", cameraConsts.viewProj._41, cameraConsts.viewProj._42, cameraConsts.viewProj._43, cameraConsts.viewProj._44);
        LOG_INFO("  Camera Right: ({}, {}, {})", cameraConsts.cameraRight.x, cameraConsts.cameraRight.y, cameraConsts.cameraRight.z);
        LOG_INFO("  Camera Up: ({}, {}, {})", cameraConsts.cameraUp.x, cameraConsts.cameraUp.y, cameraConsts.cameraUp.z);
    }

    cmdList->SetGraphicsRoot32BitConstants(0, 28, &cameraConsts, 0);

    struct ParticleConstants {
        float particleRadius;
        float padding[3];
    } particleConsts = { constants.particleSize, {0, 0, 0} };

    cmdList->SetGraphicsRoot32BitConstants(1, 4, &particleConsts, 0);
    cmdList->SetGraphicsRootShaderResourceView(2, particleBuffer->GetGPUVirtualAddress());

    if (rtLightingBuffer) {
        cmdList->SetGraphicsRootShaderResourceView(3, rtLightingBuffer->GetGPUVirtualAddress());
    }

    // Draw instanced billboards (6 vertices per instance = 2 triangles)
    cmdList->DrawInstanced(6, m_particleCount, 0, 0);
}
