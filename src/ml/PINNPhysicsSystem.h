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
// Predicts forces on particles while respecting:
// - General Relativity (Schwarzschild metric)
// - Keplerian motion
// - Angular momentum conservation
// - Shakura-Sunyaev α-disk viscosity
// - Energy conservation

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

    // Batch inference: predict forces for multiple particles
    // Input: particle positions/velocities in Cartesian coordinates
    // Output: forces in Cartesian coordinates (ready for integration)
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

    // Performance tracking
    PerformanceMetrics m_metrics = {};

    // Physics constants (normalized units)
    const float R_ISCO = 6.0f;  // Innermost Stable Circular Orbit (3× Schwarzschild radius)
};
