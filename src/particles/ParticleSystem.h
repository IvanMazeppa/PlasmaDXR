#pragma once

#include <d3d12.h>
#include <wrl/client.h>
#include <DirectXMath.h>
#include <algorithm>
#include <memory>
#include <vector>
#include <string>

class Device;
class ResourceManager;
class PINNPhysicsSystem;

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

    // Material type enumeration for diverse particle rendering
    // Sprint 1: MVP with 5 material types (PLASMA must be index 0 for backward compatibility)
    enum class ParticleMaterialType : uint32_t {
        PLASMA = 0,              // Legacy - accretion disk plasma (hot orange/red)
        STAR_MAIN_SEQUENCE = 1,  // G-type stars (Sun-like) - 5800K, high emission
        GAS_CLOUD = 2,           // Nebulae - wispy, low density, colorful
        ROCKY_BODY = 3,          // Asteroids, rocky particles - grey, low emission
        ICY_BODY = 4,            // Comets, icy particles - white/blue, reflective
        // Future: STAR_GIANT, STAR_NEUTRON, DUST_CLOUD (Phase 2)
    };

    // Extended particle structure (48 bytes, 16-byte aligned)
    // CRITICAL: First 32 bytes MUST match legacy layout for backward compatibility
    struct Particle {
        // === LEGACY FIELDS (32 bytes) - DO NOT REORDER ===
        DirectX::XMFLOAT3 position;    // 12 bytes (offset 0)
        float temperature;             // 4 bytes  (offset 12)
        DirectX::XMFLOAT3 velocity;    // 12 bytes (offset 16)
        float density;                 // 4 bytes  (offset 28)

        // === NEW FIELDS (16 bytes) ===
        DirectX::XMFLOAT3 albedo;      // 12 bytes (offset 32) - Surface/volume color
        uint32_t materialType;         // 4 bytes  (offset 44) - ParticleMaterialType enum
    };  // Total: 48 bytes (16-byte aligned ✓)

    // Material properties for each material type
    // Sprint 1: 5 material types, each with distinct visual properties
    // GPU constant buffer: 320 bytes (5 materials × 64 bytes)
    struct MaterialTypeProperties {
        DirectX::XMFLOAT3 albedo;             // 12 bytes - Base surface/volume color (RGB)
        float opacity;                        // 4 bytes  - Opacity multiplier (0-1)
        float emissionMultiplier;             // 4 bytes  - Emission strength multiplier
        float scatteringCoefficient;          // 4 bytes  - Volumetric scattering (higher = more scattering)
        float phaseG;                         // 4 bytes  - Henyey-Greenstein phase function (-1 to 1)
        float padding[9];                     // 36 bytes - Padding to 64 bytes for alignment
    };  // Total: 64 bytes per material

    struct MaterialPropertiesConstants {
        MaterialTypeProperties materials[5];  // 5 types × 64 bytes = 320 bytes
    };  // Total: 320 bytes

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

    // Sprint 1: Material System accessors
    ID3D12Resource* GetMaterialPropertiesBuffer() const { return m_materialPropertiesBuffer.Get(); }

    // Runtime active particle count control
    uint32_t GetActiveParticleCount() const { return m_activeParticleCount; }
    void SetActiveParticleCount(uint32_t count) {
        m_activeParticleCount = (std::min)(count, m_particleCount);
    }

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

    // NEW: Timescale parameter (simulation speed multiplier)
    float GetTimeScale() const { return m_timeScale; }
    void SetTimeScale(float value) { m_timeScale = (std::max)(0.0f, (std::min)(10.0f, value)); }
    void AdjustTimeScale(float delta) { m_timeScale = (std::max)(0.0f, (std::min)(10.0f, m_timeScale + delta)); }

    // Debug: Readback particle data from GPU
    void DebugReadbackParticles(int count = 5);

    // PINN Physics Controls
    bool IsPINNAvailable() const;
    bool IsPINNEnabled() const;
    void SetPINNEnabled(bool enabled);
    void TogglePINNPhysics();

    bool IsPINNHybridMode() const;
    void SetPINNHybridMode(bool hybrid);

    float GetPINNHybridThreshold() const;
    void SetPINNHybridThreshold(float radiusMultiplier);

    std::string GetPINNModelInfo() const;

    struct PINNMetrics {
        float inferenceTimeMs = 0.0f;
        uint32_t particlesProcessed = 0;
        float avgBatchTimeMs = 0.0f;
    };
    PINNMetrics GetPINNMetrics() const;

private:
    void InitializeAccretionDisk();
    void InitializeAccretionDisk_CPU();  // CPU-based initialization for PINN mode
    bool CreateComputePipeline();

    // Sprint 1: Material System Helper Methods
    void InitializeMaterialProperties();    // Set up 5 material presets with vibrant colors
    bool CreateMaterialPropertiesBuffer();  // Create and upload GPU constant buffer

    // PINN Physics Helper Methods
    void UpdatePhysics_GPU(float deltaTime, float totalTime);
    void UpdatePhysics_PINN(float deltaTime, float totalTime);
    void UploadParticleData(const std::vector<DirectX::XMFLOAT3>& positions, const std::vector<DirectX::XMFLOAT3>& velocities);
    void IntegrateForces(const std::vector<DirectX::XMFLOAT3>& forces, float deltaTime);

private:
    Device* m_device = nullptr;
    ResourceManager* m_resources = nullptr;

    // Particle data
    uint32_t m_particleCount = 0;           // Maximum particle count (buffer size)
    uint32_t m_activeParticleCount = 0;     // Active particle count (runtime adjustable)
    Microsoft::WRL::ComPtr<ID3D12Resource> m_particleBuffer;  // GPU-initialized by physics shader

    // Sprint 1: Material System
    Microsoft::WRL::ComPtr<ID3D12Resource> m_materialPropertiesBuffer;  // Material properties constant buffer (320 bytes)
    MaterialPropertiesConstants m_materialProperties;                    // CPU-side material properties

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
    float m_timeScale = 1.0f;                 // Simulation speed multiplier (0.0-10.0, 1.0 = normal speed)

    DirectX::XMFLOAT3 m_blackHolePosition = DirectX::XMFLOAT3(0.0f, 0.0f, 0.0f);
    DirectX::XMFLOAT3 m_diskAxis = DirectX::XMFLOAT3(0.0f, 1.0f, 0.0f);

    // PINN ML Physics System
    PINNPhysicsSystem* m_pinnPhysics = nullptr;
    bool m_usePINN = false;
    bool m_particlesOnCPU = false;  // When true, particles are stored on CPU (PINN mode)

    // CPU-side particle data buffers (for PINN inference)
    std::vector<DirectX::XMFLOAT3> m_cpuPositions;
    std::vector<DirectX::XMFLOAT3> m_cpuVelocities;
    std::vector<DirectX::XMFLOAT3> m_cpuForces;
};