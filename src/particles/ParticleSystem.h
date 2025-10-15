#pragma once

#include <d3d12.h>
#include <wrl/client.h>
#include <DirectXMath.h>
#include <algorithm>

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

    // Physics parameter accessors for runtime control
    float GetGravityStrength() const { return m_gravityStrength; }
    void SetGravityStrength(float value) { m_gravityStrength = value; }
    void AdjustGravityStrength(float delta) { m_gravityStrength = (std::max)(0.0f, m_gravityStrength + delta); }

    float GetAngularMomentum() const { return m_angularMomentumBoost; }
    void SetAngularMomentum(float value) { m_angularMomentumBoost = value; }
    void AdjustAngularMomentum(float delta) { m_angularMomentumBoost = (std::max)(0.0f, m_angularMomentumBoost + delta); }

    float GetTurbulence() const { return m_turbulenceStrength; }
    void SetTurbulence(float value) { m_turbulenceStrength = value; }
    void AdjustTurbulence(float delta) { m_turbulenceStrength = (std::max)(0.0f, m_turbulenceStrength + delta); }

    float GetDamping() const { return m_dampingFactor; }
    void SetDamping(float value) { m_dampingFactor = (std::max)(0.0f, (std::min)(1.0f, value)); }
    void AdjustDamping(float delta) { m_dampingFactor = (std::max)(0.0f, (std::min)(1.0f, m_dampingFactor + delta)); }

    // NEW: Black hole mass parameter (affects orbital velocity)
    float GetBlackHoleMass() const { return m_blackHoleMass; }
    void SetBlackHoleMass(float value) { m_blackHoleMass = (std::max)(1.0f, value); } // Min 1 solar mass
    void AdjustBlackHoleMass(float delta) { m_blackHoleMass = (std::max)(1.0f, m_blackHoleMass + delta); }

    // NEW: Alpha viscosity parameter (Shakura-Sunyaev accretion)
    float GetAlphaViscosity() const { return m_alphaViscosity; }
    void SetAlphaViscosity(float value) { m_alphaViscosity = (std::max)(0.0f, (std::min)(1.0f, value)); }
    void AdjustAlphaViscosity(float delta) { m_alphaViscosity = (std::max)(0.0f, (std::min)(1.0f, m_alphaViscosity + delta)); }

    // Debug: Readback particle data from GPU
    void DebugReadbackParticles(int count = 5);

private:
    void InitializeAccretionDisk();
    bool CreateComputePipeline();

private:
    Device* m_device = nullptr;
    ResourceManager* m_resources = nullptr;

    // Particle data
    uint32_t m_particleCount = 0;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_particleBuffer;  // GPU-initialized by physics shader

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
    float m_viscosity = 0.01f;  // Legacy - replaced by m_alphaViscosity
    uint32_t m_constraintShape = 0;  // 0=NONE, 1=SPHERE, 2=DISC, 3=TORUS, 4=ACCRETION_DISK

    // NEW: Enhanced physics parameters
    float m_blackHoleMass = BLACK_HOLE_MASS;  // Solar masses (default: Sgr A*)
    float m_alphaViscosity = 0.1f;            // Shakura-Sunyaev α parameter (0.0-1.0)

    DirectX::XMFLOAT3 m_blackHolePosition = DirectX::XMFLOAT3(0.0f, 0.0f, 0.0f);
    DirectX::XMFLOAT3 m_diskAxis = DirectX::XMFLOAT3(0.0f, 1.0f, 0.0f);
};