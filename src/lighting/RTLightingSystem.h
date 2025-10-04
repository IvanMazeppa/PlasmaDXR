#pragma once

#include <d3d12.h>
#include <wrl/client.h>
#include <DirectXMath.h>
#include <vector>

// RTLightingSystem - Handles DXR 1.1 ray-traced lighting for particles
// Primary goal: Get GREEN test pattern working to prove RT lighting works

class Device;
class ResourceManager;

class RTLightingSystem {
public:
    struct RTConstants {
        DirectX::XMFLOAT4X4 viewProj;
        DirectX::XMFLOAT3 cameraPos;
        float time;
        DirectX::XMFLOAT3 lightDir;
        float lightIntensity;
        uint32_t particleCount;
        uint32_t frameIndex;
        uint32_t enableShadows;
        float pad;
    };

    // Particle data for BLAS building
    struct ParticleInstance {
        DirectX::XMFLOAT3 position;
        float radius;
        DirectX::XMFLOAT4 color;
    };

public:
    RTLightingSystem() = default;
    ~RTLightingSystem();

    // Initialize RT pipeline for RTX 4060 Ti
    bool Initialize(Device* device, ResourceManager* resources, uint32_t particleCount);
    void Shutdown();

    // Build acceleration structures
    void BuildBLAS(ID3D12GraphicsCommandList4* cmdList);
    void BuildTLAS(ID3D12GraphicsCommandList4* cmdList,
                   ID3D12Resource* particleBuffer,
                   uint32_t particleCount);

    // Compute RT lighting (fills lighting buffer with green test pattern)
    void ComputeLighting(ID3D12GraphicsCommandList4* cmdList,
                        const RTConstants& constants);

    // Get output lighting buffer
    ID3D12Resource* GetLightingBuffer() const { return m_lightingBuffer.Get(); }

    // Update settings
    void SetEnableShadows(bool enable) { m_enableShadows = enable; }
    void SetLightIntensity(float intensity) { m_lightIntensity = intensity; }

private:
    bool CreateRootSignatures();
    bool CreateRTPipeline();
    bool CreateShaderBindingTable();
    bool CreateOutputBuffers();

private:
    Device* m_device = nullptr;
    ResourceManager* m_resources = nullptr;

    // Configuration
    uint32_t m_particleCount = 0;
    bool m_enableShadows = true;
    float m_lightIntensity = 1.0f;

    // RT Pipeline State
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rtGlobalRootSignature;
    Microsoft::WRL::ComPtr<ID3D12StateObject> m_rtPipelineState;

    // Shader Binding Table
    Microsoft::WRL::ComPtr<ID3D12Resource> m_sbtBuffer;
    struct {
        D3D12_GPU_VIRTUAL_ADDRESS raygenRecord = 0;
        uint32_t raygenRecordSize = 0;
        D3D12_GPU_VIRTUAL_ADDRESS missRecord = 0;
        uint32_t missRecordSize = 0;
        uint32_t missRecordStride = 0;
        D3D12_GPU_VIRTUAL_ADDRESS hitGroupRecord = 0;
        uint32_t hitGroupRecordSize = 0;
        uint32_t hitGroupRecordStride = 0;
    } m_sbtInfo;

    // Acceleration Structures
    Microsoft::WRL::ComPtr<ID3D12Resource> m_bottomLevelAS;  // BLAS for particles
    Microsoft::WRL::ComPtr<ID3D12Resource> m_topLevelAS;     // TLAS
    Microsoft::WRL::ComPtr<ID3D12Resource> m_scratchBuffer;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_instanceDescsBuffer;

    // Output
    Microsoft::WRL::ComPtr<ID3D12Resource> m_lightingBuffer;  // Per-particle RT lighting

    // Shader identifiers
    static constexpr UINT SHADER_IDENTIFIER_SIZE = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
};