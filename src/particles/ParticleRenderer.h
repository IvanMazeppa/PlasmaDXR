#pragma once

#include <d3d12.h>
#include <wrl/client.h>
#include <memory>
#include <DirectXMath.h>

// Forward declarations
class Device;
class ResourceManager;
class FeatureDetector;

// ParticleRenderer - Handles rendering with automatic fallback
// PRIMARY PATH: Mesh shaders (if working)
// FALLBACK PATH: Compute shader builds vertices → traditional rendering

class ParticleRenderer {
public:
    enum class RenderPath {
        MeshShaders,      // Optimal path
        ComputeFallback   // Workaround for driver bugs
    };

    struct RenderConstants {
        DirectX::XMFLOAT4X4 viewProj;
        DirectX::XMFLOAT3 cameraPos;
        float time;
        DirectX::XMFLOAT3 cameraUp;
        float particleSize;
        uint32_t screenWidth;
        uint32_t screenHeight;

        // Physical emission toggles and strengths
        bool usePhysicalEmission = false;
        float emissionStrength = 1.0f;
        bool useDopplerShift = false;
        float dopplerStrength = 1.0f;
        bool useGravitationalRedshift = false;
        float redshiftStrength = 1.0f;
    };

public:
    ParticleRenderer() = default;
    ~ParticleRenderer();

    // Initialize with automatic path selection
    bool Initialize(Device* device,
                   ResourceManager* resources,
                   const FeatureDetector* features,
                   uint32_t particleCount);

    // Render particles (automatically uses best available path)
    void Render(ID3D12GraphicsCommandList* cmdList,
               ID3D12Resource* particleBuffer,
               ID3D12Resource* rtLightingBuffer,
               const RenderConstants& constants);

    // Query current path
    RenderPath GetActivePath() const { return m_activePath; }
    const char* GetActivePathName() const {
        return m_activePath == RenderPath::MeshShaders ?
               "Mesh Shaders" : "Compute Fallback";
    }

private:
    // Path-specific initialization
    bool InitializeMeshShaderPath();
    bool InitializeComputeFallbackPath();

    // Path-specific rendering
    void RenderWithMeshShaders(ID3D12GraphicsCommandList* cmdList,
                               ID3D12Resource* particleBuffer,
                               ID3D12Resource* rtLightingBuffer,
                               const RenderConstants& constants);

    void RenderWithComputeFallback(ID3D12GraphicsCommandList* cmdList,
                                   ID3D12Resource* particleBuffer,
                                   ID3D12Resource* rtLightingBuffer,
                                   const RenderConstants& constants);

    // Shader loading
    bool LoadShaders();

private:
    // Core references
    Device* m_device = nullptr;
    ResourceManager* m_resources = nullptr;

    // Configuration
    uint32_t m_particleCount = 0;
    RenderPath m_activePath = RenderPath::ComputeFallback;

    // Mesh shader path resources
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_meshRootSignature;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_meshPSO;
    Microsoft::WRL::ComPtr<ID3DBlob> m_meshShader;
    Microsoft::WRL::ComPtr<ID3DBlob> m_pixelShader;

    // Compute fallback path resources
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_computeRootSignature;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_computePSO;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rasterRootSignature;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_rasterPSO;
    Microsoft::WRL::ComPtr<ID3DBlob> m_computeShader;
    Microsoft::WRL::ComPtr<ID3DBlob> m_vertexShader;
    Microsoft::WRL::ComPtr<ID3DBlob> m_pixelShaderFallback;

    // Vertex buffer for compute fallback
    Microsoft::WRL::ComPtr<ID3D12Resource> m_vertexBuffer;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_indexBuffer;

    // Constants buffer
    Microsoft::WRL::ComPtr<ID3D12Resource> m_constantBuffer;
};