#pragma once

#ifdef ENABLE_ML_FEATURES
#include <onnxruntime_cxx_api.h>
#endif

#include <d3d12.h>
#include <DirectXMath.h>
#include <vector>
#include <string>
#include <memory>

// Physics-Informed Neural Network (PINN) for accretion disk particle dynamics
//
// Version 1 (Legacy): State-only input [r, θ, φ, v_r, v_θ, v_φ, t] → [F_r, F_θ, F_φ]
// Version 2 (New): Parameter-conditioned input:
//   - particle_state [batch, 7]: [r, θ, φ, v_r, v_θ, v_φ, t]
//   - physics_params [batch, 3]: [M_bh_normalized, α_viscosity, disk_thickness]
//   → forces [batch, 3]: [F_r, F_θ, F_φ]
//
// Version 2 allows runtime control of physics parameters without retraining.

class PINNPhysicsSystem {
public:
    // Particle state in spherical coordinates (used by PINN)
    struct ParticleStateSpherical {
        float r;         // Radial distance from black hole
        float theta;     // Polar angle (0 = north pole, π = south pole)
        float phi;       // Azimuthal angle (0-2π)
        float v_r;       // Radial velocity
        float v_theta;   // Polar velocity
        float v_phi;     // Azimuthal velocity
        float time;      // Simulation time
    };

    // Predicted forces in spherical coordinates
    struct PredictedForces {
        float F_r;       // Radial force
        float F_theta;   // Polar force
        float F_phi;     // Azimuthal force
    };

    // Physics parameters for v2 model (runtime adjustable)
    struct PhysicsParams {
        float blackHoleMassNormalized = 1.0f;  // Multiplier (0.5-2.0, 1.0 = default 10 solar masses)
        float alphaViscosity = 0.1f;           // Shakura-Sunyaev α (0.01-0.3)
        float diskThickness = 0.1f;            // H/R ratio (0.05-0.2)
    };

    // Performance metrics
    struct PerformanceMetrics {
        float inferenceTimeMs;
        uint32_t particlesProcessed;
        uint32_t batchCount;
        float avgBatchTimeMs;
    };

public:
    PINNPhysicsSystem();
    ~PINNPhysicsSystem();

    // Initialize PINN system with trained ONNX model
    bool Initialize(const std::string& modelPath = "ml/models/pinn_accretion_disk.onnx");

    // Check if ML features are available (compiled with ONNX Runtime)
    bool IsAvailable() const;

    // Enable/disable PINN physics
    void SetEnabled(bool enabled) { m_enabled = enabled; }
    bool IsEnabled() const { return m_enabled; }

    // Hybrid mode: PINN for far particles, GPU shader for near ISCO
    void SetHybridMode(bool hybrid) { m_hybridMode = hybrid; }
    bool IsHybridMode() const { return m_hybridMode; }

    // Set ISCO radius threshold for hybrid mode (default: 10× R_ISCO)
    void SetHybridThreshold(float radiusMultiplier);
    float GetHybridThreshold() const { return m_hybridThresholdRadius; }

    // ========== Physics Parameters (v2 model) ==========
    // These parameters can be adjusted at runtime when using a v2 (parameter-conditioned) model.
    // The v1 model ignores these parameters.

    // Set/get all physics parameters at once
    void SetPhysicsParams(const PhysicsParams& params) { m_physicsParams = params; }
    const PhysicsParams& GetPhysicsParams() const { return m_physicsParams; }

    // Individual parameter controls with clamping
    void SetBlackHoleMass(float normalized);  // 0.5 - 2.0 (multiplier of default mass)
    float GetBlackHoleMass() const { return m_physicsParams.blackHoleMassNormalized; }

    void SetAlphaViscosity(float alpha);      // 0.01 - 0.3 (Shakura-Sunyaev α)
    float GetAlphaViscosity() const { return m_physicsParams.alphaViscosity; }

    void SetDiskThickness(float hrRatio);     // 0.05 - 0.2 (H/R ratio)
    float GetDiskThickness() const { return m_physicsParams.diskThickness; }

    // Check if using v2 (parameter-conditioned) model
    bool IsParameterConditioned() const { return m_isV2Model; }
    int GetModelVersion() const { 
        if (m_isV4Model) return 4;
        if (m_isV3Model) return 3;
        if (m_isV2Model) return 2;
        return 1;
    }
    
    // Model selection at runtime
    std::string GetCurrentModelName() const { return m_currentModelName; }
    std::string GetCurrentModelPath() const { return m_currentModelPath; }
    
    // Get list of available models in ml/models/ directory
    static std::vector<std::pair<std::string, std::string>> GetAvailableModels();
    
    // Reload with a different model (returns true if successful)
    bool LoadModel(const std::string& modelPath);

    // Batch inference: predict forces for multiple particles
    // Input: particle positions/velocities in Cartesian coordinates
    // Output: forces in Cartesian coordinates (ready for integration)
    // Note: v2 model uses current physics parameters (SetPhysicsParams/SetBlackHoleMass/etc.)
    bool PredictForcesBatch(
        const DirectX::XMFLOAT3* positions,    // [particleCount]
        const DirectX::XMFLOAT3* velocities,   // [particleCount]
        DirectX::XMFLOAT3* outForces,          // [particleCount]
        uint32_t particleCount,
        float currentTime);

    // Single particle inference (convenience wrapper)
    bool PredictForces(
        const DirectX::XMFLOAT3& position,
        const DirectX::XMFLOAT3& velocity,
        DirectX::XMFLOAT3& outForce,
        float currentTime);

    // Performance monitoring
    const PerformanceMetrics& GetPerformanceMetrics() const { return m_metrics; }
    void ResetMetrics();

    // Get model information
    std::string GetModelInfo() const;

private:
    // Coordinate transformations
    ParticleStateSpherical CartesianToSpherical(
        const DirectX::XMFLOAT3& position,
        const DirectX::XMFLOAT3& velocity) const;

    DirectX::XMFLOAT3 SphericalForcesToCartesian(
        const PredictedForces& forces,
        const ParticleStateSpherical& state) const;

    // ONNX Runtime inference
    bool RunInference(
        const std::vector<float>& inputData,
        std::vector<float>& outputData);

    // Helper: Determine if particle should use PINN (hybrid mode)
    bool ShouldUsePINN(float radius) const;

private:
#ifdef ENABLE_ML_FEATURES
    // ONNX Runtime session
    std::unique_ptr<Ort::Env> m_ortEnv;
    std::unique_ptr<Ort::Session> m_ortSession;
    std::unique_ptr<Ort::SessionOptions> m_sessionOptions;
    Ort::MemoryInfo m_memoryInfo;

    // Model metadata
    std::vector<std::string> m_inputNames;
    std::vector<std::string> m_outputNames;
    std::vector<int64_t> m_inputShape;   // Expected: [batch_size, 7]
    std::vector<int64_t> m_outputShape;  // Expected: [batch_size, 3]
#endif

    // System state
    bool m_enabled = false;
    bool m_modelLoaded = false;
    bool m_hybridMode = true;
    float m_hybridThresholdRadius = 60.0f;  // 10× R_ISCO (R_ISCO = 6.0 in normalized units)

    // V2 model support (parameter-conditioned)
    PhysicsParams m_physicsParams;
    bool m_isV2Model = false;  // v2: 2 inputs (state + params)
    bool m_isV3Model = false;  // v3: 1 input, 10D Cartesian (total forces output)
    bool m_isV4Model = false;  // v4: 1 input, 10D Cartesian (turbulence-robust)
    
    // Model identification
    std::string m_currentModelName = "none";
    std::string m_currentModelPath;
#ifdef ENABLE_ML_FEATURES
    std::vector<int64_t> m_paramsInputShape;  // Expected: [batch_size, 3] for v2 model
#endif

    // Performance tracking
    PerformanceMetrics m_metrics = {};

    // Physics constants (normalized units)
    const float R_ISCO = 6.0f;  // Innermost Stable Circular Orbit (3× Schwarzschild radius)
};
