#pragma once

#include <d3d12.h>
#include <wrl/client.h>
#include <DirectXMath.h>

class Device;
class ResourceManager;

// ParticleSystem - Manages particle data and physics
// Focused on NASA-quality accretion disk simulation

class ParticleSystem {
public:
    // Physical constants for accretion disk
    static constexpr float BLACK_HOLE_MASS = 4.3e6f;        // Solar masses (Sgr A*)
    static constexpr float INNER_STABLE_ORBIT = 10.0f;      // Schwarzschild radii
    static constexpr float OUTER_DISK_RADIUS = 300.0f;      // Reduced for denser, more visible disk
    static constexpr float DISK_THICKNESS = 50.0f;
    static constexpr float INITIAL_ANGULAR_MOMENTUM = 100.0f;

    struct Particle {
        DirectX::XMFLOAT3 position;
        float temperature;
        DirectX::XMFLOAT3 velocity;
        float density;
    };

public:
    ParticleSystem() = default;
    ~ParticleSystem();

    bool Initialize(Device* device, ResourceManager* resources, uint32_t particleCount);
    void Shutdown();

    // Update particle physics
    void Update(float deltaTime, float totalTime);

    // Get particle buffer for rendering
    ID3D12Resource* GetParticleBuffer() const { return m_particleBuffer.Get(); }
    uint32_t GetParticleCount() const { return m_particleCount; }

private:
    void InitializeAccretionDisk();
    bool CreateComputePipeline();

private:
    Device* m_device = nullptr;
    ResourceManager* m_resources = nullptr;

    // Particle data
    uint32_t m_particleCount = 0;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_particleBuffer;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_particleUploadBuffer;

    // Physics compute pipeline
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_computeRootSignature;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_computePSO;

    // Simulation parameters
    float m_totalTime = 0.0f;

    // Physics parameters (runtime adjustable)
    float m_gravityStrength = 500.0f;
    float m_turbulenceStrength = 15.0f;
    float m_dampingFactor = 0.99f;
    float m_angularMomentumBoost = 1.0f;
    float m_viscosity = 0.01f;
    uint32_t m_constraintShape = 0;  // 0=NONE, 1=SPHERE, 2=DISC, 3=TORUS, 4=ACCRETION_DISK

    DirectX::XMFLOAT3 m_blackHolePosition = DirectX::XMFLOAT3(0.0f, 0.0f, 0.0f);
    DirectX::XMFLOAT3 m_diskAxis = DirectX::XMFLOAT3(0.0f, 1.0f, 0.0f);
};