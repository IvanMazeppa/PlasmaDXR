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
class SIRENVortexField;

// ParticleSystem - Manages particle data and physics
// Focused on NASA-quality accretion disk simulation

class ParticleSystem {
public:
    // Physical constants for accretion disk
    // Phase 5 UPDATED: Realistic scale for volumetric particles (radius ~20 units)
    static constexpr float BLACK_HOLE_MASS = 4.3e6f;        // Solar masses (Sgr A*)
    static constexpr float INNER_STABLE_ORBIT = 50.0f;      // Schwarzschild radii (was 10, now 2.5× particle diameter)
    static constexpr float OUTER_DISK_RADIUS = 1000.0f;     // Disk edge (was 300, now 50× particle diameter)
    static constexpr float DISK_THICKNESS = 100.0f;         // Vertical extent (was 50, now 5× particle diameter)

    // PINN Normalized Unit System (G*M = 100) - UPDATED FOR v3 MODEL
    // Training data uses: r=10-300, v=sqrt(100/r), F=-100/r^2
    static constexpr float PINN_GM = 100.0f;                // Gravitational parameter (MUST match training script!)
    static constexpr float PINN_R_ISCO = 6.0f;              // Innermost stable circular orbit
    static constexpr float INITIAL_ANGULAR_MOMENTUM = 100.0f;

    // Material type enumeration for diverse particle rendering
    // Phase 2: Extended with pyro/explosion types
    enum class ParticleMaterialType : uint32_t {
        PLASMA = 0,              // Legacy - accretion disk plasma (hot orange/red)
        STAR_MAIN_SEQUENCE = 1,  // G-type stars (Sun-like) - 5800K, high emission
        GAS_CLOUD = 2,           // Nebulae - wispy, low density, colorful
        ROCKY_BODY = 3,          // Asteroids, rocky particles - grey, low emission
        ICY_BODY = 4,            // Comets, icy particles - white/blue, reflective
        SUPERNOVA = 5,           // Phase 2: Explosive stellar death - extreme emission, expanding
        STELLAR_FLARE = 6,       // Phase 2: Solar flare ejection - hot plasma burst
        SHOCKWAVE = 7,           // Phase 2: Expanding shockwave ring - fast, fading
        COUNT = 8                // Total material types
    };

    // Particle flags for special behaviors
    enum class ParticleFlags : uint32_t {
        NONE = 0,
        EXPLOSION = 1 << 0,      // Part of explosion event
        FADING = 1 << 1,         // Currently fading out
        IMMORTAL = 1 << 2,       // Never expires (lifetime ignored)
        EMISSIVE_ONLY = 1 << 3,  // Pure emission, no scattering
    };

    // Extended particle structure (64 bytes, 16-byte aligned)
    // CRITICAL: First 32 bytes MUST match legacy layout for backward compatibility
    struct Particle {
        // === LEGACY FIELDS (32 bytes) - DO NOT REORDER ===
        DirectX::XMFLOAT3 position;    // 12 bytes (offset 0)
        float temperature;             // 4 bytes  (offset 12)
        DirectX::XMFLOAT3 velocity;    // 12 bytes (offset 16)
        float density;                 // 4 bytes  (offset 28)

        // === MATERIAL FIELDS (16 bytes) ===
        DirectX::XMFLOAT3 albedo;      // 12 bytes (offset 32) - Surface/volume color
        uint32_t materialType;         // 4 bytes  (offset 44) - ParticleMaterialType enum

        // === LIFETIME FIELDS (16 bytes) - Phase 2 Pyro/Explosions ===
        float lifetime;                // 4 bytes  (offset 48) - Current age in seconds
        float maxLifetime;             // 4 bytes  (offset 52) - Total duration (0 = infinite/immortal)
        float spawnTime;               // 4 bytes  (offset 56) - Time when spawned (for effects sync)
        uint32_t flags;                // 4 bytes  (offset 60) - ParticleFlags bitmask
    };  // Total: 64 bytes (16-byte aligned ✓)

    // Material properties for each material type
    // Phase 2: Extended to 8 material types for pyro/explosion effects
    // GPU constant buffer: 512 bytes (8 materials × 64 bytes)
    struct MaterialTypeProperties {
        DirectX::XMFLOAT3 albedo;             // 12 bytes - Base surface/volume color (RGB)
        float opacity;                        // 4 bytes  - Opacity multiplier (0-1)
        float emissionMultiplier;             // 4 bytes  - Emission strength multiplier
        float scatteringCoefficient;          // 4 bytes  - Volumetric scattering (higher = more scattering)
        float phaseG;                         // 4 bytes  - Henyey-Greenstein phase function (-1 to 1)
        float expansionRate;                  // 4 bytes  - Phase 2: Radius expansion per second (for explosions)
        float coolingRate;                    // 4 bytes  - Phase 2: Temperature decay rate (K/second)
        float fadeStartRatio;                 // 4 bytes  - Phase 2: Lifetime ratio when fade begins (0.7 = 70%)
        float padding[6];                     // 24 bytes - Padding to 64 bytes for alignment
    };  // Total: 64 bytes per material

    struct MaterialPropertiesConstants {
        MaterialTypeProperties materials[8];  // 8 types × 64 bytes = 512 bytes
    };  // Total: 512 bytes

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
    uint32_t GetMaxActiveParticleCount() const { return m_explosionPoolStart > 0 ? m_explosionPoolStart : m_particleCount; }
    void SetActiveParticleCount(uint32_t count) {
        // Clamp to explosion pool boundary (don't let physics overwrite explosion particles)
        uint32_t maxActive = m_explosionPoolStart > 0 ? m_explosionPoolStart : m_particleCount;
        m_activeParticleCount = (std::min)(count, maxActive);
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
    void SetTimeScale(float value) { m_timeScale = (std::max)(0.0f, (std::min)(50.0f, value)); }
    void AdjustTimeScale(float delta) { m_timeScale = (std::max)(0.0f, (std::min)(50.0f, m_timeScale + delta)); }

    // Benchmark Physics Parameters (Phase 1: Runtime Controls)
    float GetGM() const { return m_gm; }
    void SetGM(float value) { m_gm = (std::max)(1.0f, value); }

    float GetDiskThickness() const { return m_diskThickness; }
    void SetDiskThickness(float value) { m_diskThickness = (std::max)(0.01f, (std::min)(0.5f, value)); }

    float GetDensityScale() const { return m_densityScale; }
    void SetDensityScale(float value) { m_densityScale = (std::max)(0.01f, (std::min)(10.0f, value)); }

    float GetInnerRadius() const { return m_innerRadius; }
    void SetInnerRadius(float value) { m_innerRadius = (std::max)(1.0f, value); }

    float GetOuterRadius() const { return m_outerRadius; }
    void SetOuterRadius(float value) { m_outerRadius = (std::max)(m_innerRadius + 10.0f, value); }

    float GetForceClamp() const { return m_forceClamp; }
    void SetForceClamp(float value) { m_forceClamp = (std::max)(0.1f, value); }

    float GetVelocityClamp() const { return m_velocityClamp; }
    void SetVelocityClamp(float value) { m_velocityClamp = (std::max)(0.1f, value); }

    int GetBoundaryMode() const { return m_boundaryMode; }
    void SetBoundaryMode(int mode) { m_boundaryMode = (std::max)(0, (std::min)(3, mode)); }

    bool GetEnforceBoundaries() const { return m_enforceBoundaries; }
    void SetEnforceBoundaries(bool enforce) { m_enforceBoundaries = enforce; }

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

    // PINN v2 Model: Physics Parameters (runtime adjustable)
    bool IsPINNParameterConditioned() const;
    int GetPINNModelVersion() const;

    // Black hole mass multiplier (0.5 - 2.0, 1.0 = default 10 solar masses)
    float GetPINNBlackHoleMass() const;
    void SetPINNBlackHoleMass(float normalized);

    // Shakura-Sunyaev alpha viscosity (0.01 - 0.3)
    float GetPINNAlphaViscosity() const;
    void SetPINNAlphaViscosity(float alpha);

    // Disk thickness H/R ratio (0.05 - 0.2)
    float GetPINNDiskThickness() const;
    void SetPINNDiskThickness(float hrRatio);

    // PINN Visualization Parameters (post-processing controls)
    // These adjust appearance without retraining the model
    float GetPINNVelocityMultiplier() const { return m_pinnVelocityMultiplier; }
    void SetPINNVelocityMultiplier(float mult) { m_pinnVelocityMultiplier = std::clamp(mult, 0.1f, 50.0f); }

    float GetPINNTurbulence() const { return m_pinnTurbulence; }
    void SetPINNTurbulence(float turb) { m_pinnTurbulence = std::clamp(turb, 0.0f, 1.0f); }

    float GetPINNDamping() const { return m_pinnDamping; }
    void SetPINNDamping(float damp) { m_pinnDamping = std::clamp(damp, 0.9f, 1.0f); }

    float GetPINNRadialSpread() const { return m_pinnRadialSpread; }
    void SetPINNRadialSpread(float spread) { m_pinnRadialSpread = std::clamp(spread, 0.1f, 3.0f); }

    bool GetPINNEnforceBoundaries() const { return m_pinnEnforceBoundaries; }
    void SetPINNEnforceBoundaries(bool enforce) { m_pinnEnforceBoundaries = enforce; }

    // Reinitialize particles with current PINN settings
    void ReinitializePINNParticles();
    
    // PINN Model Selection (runtime switching)
    std::string GetPINNModelName() const;
    std::string GetPINNModelPath() const;
    std::vector<std::pair<std::string, std::string>> GetAvailablePINNModels() const;
    bool LoadPINNModel(const std::string& modelPath);
    
    // SIREN Vortex Field (ML-based turbulence)
    bool IsSIRENAvailable() const;
    bool IsSIRENEnabled() const;
    void SetSIRENEnabled(bool enabled);
    
    float GetSIRENIntensity() const;
    void SetSIRENIntensity(float intensity);
    
    float GetSIRENSeed() const;
    void SetSIRENSeed(float seed);
    
    std::string GetSIRENModelInfo() const;
    
    struct SIRENMetrics {
        float inferenceTimeMs = 0.0f;
        uint32_t particlesProcessed = 0;
        float avgBatchTimeMs = 0.0f;
    };
    SIRENMetrics GetSIRENMetrics() const;

    // ========== Benchmark Support ==========
    
    // Physics snapshot for benchmark metrics (no rendering required)
    struct PhysicsSnapshot {
        float totalKineticEnergy = 0.0f;
        float totalPotentialEnergy = 0.0f;
        float totalEnergy = 0.0f;
        float totalAngularMomentum = 0.0f;
        
        uint32_t particlesInBounds = 0;
        uint32_t particlesEscaped = 0;
        uint32_t particlesCollapsed = 0;
        
        float velocityMean = 0.0f;
        float velocityMax = 0.0f;
        float velocityStdDev = 0.0f;
        
        float avgForceMagnitude = 0.0f;
        float avgRadialForce = 0.0f;
        uint32_t correctRadialForceCount = 0;
        
        float keplerianVelocityError = 0.0f;
        
        // Coherent motion detection
        float velocityCovarianceXZ = 0.0f;
        
        // Disk shape
        float diskThicknessRatio = 0.0f;  // H/R
    };
    
    /**
     * Capture current physics state for benchmark metrics.
     * Can be called without GPU rendering active.
     * @return PhysicsSnapshot with stability, accuracy, and visual quality metrics
     */
    PhysicsSnapshot CapturePhysicsSnapshot() const;

    // ========== Phase 2C: Explosion Spawning System ==========

    // Configuration for spawning explosion effects
    struct ExplosionConfig {
        DirectX::XMFLOAT3 position = { 0.0f, 0.0f, 0.0f };  // World position
        ParticleMaterialType type = ParticleMaterialType::SUPERNOVA;  // Material type
        uint32_t particleCount = 100;        // Number of particles (clamped to pool size)
        float initialRadius = 10.0f;         // Starting spawn radius
        float expansionSpeed = 100.0f;       // Outward velocity (units/sec)
        float temperature = 50000.0f;        // Initial temperature (Kelvin)
        float lifetime = 3.0f;               // Particle lifetime (seconds)
        float density = 0.8f;                // Initial density (affects opacity)
    };

    /**
     * Queue an explosion effect to be spawned
     *
     * Uses a reserved pool of particles (last 1000 in buffer) for explosions.
     * Old explosion particles are recycled when pool is exhausted.
     *
     * NOTE: This queues the explosion for processing. Call ProcessPendingExplosions()
     * AFTER the command list is reset but BEFORE rendering to actually upload particles.
     *
     * @param config Explosion configuration
     */
    void SpawnExplosion(const ExplosionConfig& config);

    /**
     * Queue explosion at random position in disk (convenience method)
     */
    void SpawnRandomExplosion(ParticleMaterialType type = ParticleMaterialType::SUPERNOVA);

    /**
     * Process all pending explosion requests
     *
     * MUST be called AFTER command list is reset but BEFORE particle buffer is used.
     * This uploads explosion particle data to the GPU.
     *
     * @return Total number of particles spawned this frame
     */
    uint32_t ProcessPendingExplosions();

    /**
     * Get explosion pool statistics
     */
    uint32_t GetExplosionPoolSize() const { return EXPLOSION_POOL_SIZE; }
    uint32_t GetExplosionPoolUsed() const { return m_explosionPoolUsed; }

    // Explosion pool constants
    static constexpr uint32_t EXPLOSION_POOL_SIZE = 1000;  // Reserved particles for explosions

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

    // Phase 2C: Explosion System Helper Methods
    void UploadExplosionParticles(const std::vector<Particle>& particles, uint32_t startIndex);

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
    float m_timeScale = 1.0f;                 // Simulation speed multiplier (0.0-50.0, 1.0 = normal speed)

    // Benchmark physics parameters (Phase 1: Runtime Controls)
    // Phase 5 UPDATED: Realistic scale for volumetric particles (radius ~20 units)
    float m_gm = 100.0f;                      // Gravitational parameter (G*M)
    float m_diskThickness = 0.1f;             // H/R ratio (disk height/radius, 0.01-0.5)
    float m_densityScale = 1.0f;              // Global density multiplier
    float m_innerRadius = 50.0f;              // Inner disk radius (was 6, now 2.5× particle diameter)
    float m_outerRadius = 1000.0f;            // Outer disk radius (was 300, now 50× particle diameter)
    float m_forceClamp = 10.0f;               // Maximum force magnitude (safety limit)
    float m_velocityClamp = 20.0f;            // Maximum velocity magnitude (safety limit)
    int m_boundaryMode = 0;                   // Boundary handling (0=none, 1=reflect, 2=wrap, 3=respawn)
    bool m_enforceBoundaries = false;         // Whether to apply boundary constraints

    // PINN Visualization Parameters (post-processing after PINN inference)
    // These control appearance without retraining the model
    float m_pinnVelocityMultiplier = 1.0f;    // Scale orbital velocities (1.0=physical, 5-20=visible rotation)
    float m_pinnTurbulence = 0.0f;            // Random velocity perturbation (0.0-1.0)
    float m_pinnDamping = 1.0f;               // Velocity damping per frame (1.0=none, PINN includes learned viscosity)
    float m_pinnRadialSpread = 1.0f;          // Initial radius distribution width (0.5=narrow, 2.0=wide)
    bool m_pinnEnforceBoundaries = false;     // Whether to clamp particles to disk region (DISABLED by default for realistic physics)

    DirectX::XMFLOAT3 m_blackHolePosition = DirectX::XMFLOAT3(0.0f, 0.0f, 0.0f);
    DirectX::XMFLOAT3 m_diskAxis = DirectX::XMFLOAT3(0.0f, 1.0f, 0.0f);

    // PINN ML Physics System
    PINNPhysicsSystem* m_pinnPhysics = nullptr;
    bool m_usePINN = false;
    bool m_particlesOnCPU = false;  // When true, particles are stored on CPU (PINN mode)
    
    // SIREN Vortex Field (ML-based turbulence)
    SIRENVortexField* m_sirenVortex = nullptr;
    std::vector<DirectX::XMFLOAT3> m_cpuTurbulenceForces;  // Buffer for SIREN output

    // CPU-side particle data buffers (for PINN inference)
    std::vector<DirectX::XMFLOAT3> m_cpuPositions;
    std::vector<DirectX::XMFLOAT3> m_cpuVelocities;
    std::vector<DirectX::XMFLOAT3> m_cpuForces;

    // Phase 2C: Explosion pool tracking
    uint32_t m_explosionPoolStart = 0;      // First index of explosion pool (particleCount - EXPLOSION_POOL_SIZE)
    uint32_t m_nextExplosionIndex = 0;      // Next index to use in explosion pool (circular)
    uint32_t m_explosionPoolUsed = 0;       // Number of explosion particles currently active

    // Phase 2C: Persistent upload buffer for explosions (avoids mid-frame command list disruption)
    Microsoft::WRL::ComPtr<ID3D12Resource> m_explosionUploadBuffer;
    static constexpr size_t EXPLOSION_UPLOAD_BUFFER_SIZE = EXPLOSION_POOL_SIZE * sizeof(Particle);  // 64KB

    // Phase 2C: Pending explosion queue (processed after command list reset)
    std::vector<ExplosionConfig> m_pendingExplosions;

    // PINN: Persistent upload buffer for particle data (CRITICAL: must outlive GPU command execution)
    Microsoft::WRL::ComPtr<ID3D12Resource> m_pinnUploadBuffer;
};