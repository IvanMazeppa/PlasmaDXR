#pragma once

#ifdef ENABLE_ML_FEATURES
#include <onnxruntime_cxx_api.h>
#endif

#include <d3d12.h>
#include <DirectXMath.h>
#include <vector>
#include <string>
#include <memory>

// SIREN (Sinusoidal Representation Network) for vortex field evaluation
//
// Input: [x, y, z, t, seed] → 5D position/time/seed
// Output: [ω_x, ω_y, ω_z] → 3D vorticity vector
//
// The vorticity is converted to turbulent forces via:
//   F_turb = cross(velocity, vorticity) * intensity
//
// This is additive with PINN orbital forces for physically-based turbulence.

class SIRENVortexField {
public:
    // Performance metrics
    struct PerformanceMetrics {
        float inferenceTimeMs;
        uint32_t particlesProcessed;
        float avgBatchTimeMs;
        uint32_t batchCount;
    };

public:
    SIRENVortexField();
    ~SIRENVortexField();

    // Initialize SIREN with trained ONNX model
    bool Initialize(const std::string& modelPath = "ml/models/vortex_siren.onnx");

    // Check if available
    bool IsAvailable() const;

    // Enable/disable turbulence
    void SetEnabled(bool enabled) { m_enabled = enabled; }
    bool IsEnabled() const { return m_enabled; }

    // Turbulence intensity (scales the output vorticity)
    void SetIntensity(float intensity) { m_intensity = std::clamp(intensity, 0.0f, 10.0f); }
    float GetIntensity() const { return m_intensity; }

    // Seed for deterministic turbulence patterns
    void SetSeed(float seed) { m_seed = seed; }
    float GetSeed() const { return m_seed; }

    // Batch inference: compute vorticity for multiple particles
    // Output can be converted to forces via: F = cross(v, vorticity) * intensity
    bool PredictVorticityBatch(
        const DirectX::XMFLOAT3* positions,
        DirectX::XMFLOAT3* outVorticity,
        uint32_t particleCount,
        float currentTime);

    // Convenience: Compute turbulent FORCES directly
    // F_turb = cross(velocity, vorticity) * intensity
    bool ComputeTurbulentForcesBatch(
        const DirectX::XMFLOAT3* positions,
        const DirectX::XMFLOAT3* velocities,
        DirectX::XMFLOAT3* outForces,
        uint32_t particleCount,
        float currentTime);

    // Single particle (convenience wrapper)
    bool PredictVorticity(
        const DirectX::XMFLOAT3& position,
        DirectX::XMFLOAT3& outVorticity,
        float currentTime);

    // Performance monitoring
    const PerformanceMetrics& GetPerformanceMetrics() const { return m_metrics; }
    void ResetMetrics();

    // Model info
    std::string GetModelInfo() const;
    std::string GetModelName() const { return m_modelName; }

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
    std::vector<int64_t> m_inputShape;   // Expected: [batch, 5]
    std::vector<int64_t> m_outputShape;  // Expected: [batch, 3]
#endif

    // System state
    bool m_enabled = false;
    bool m_modelLoaded = false;
    
    // Turbulence parameters
    float m_intensity = 0.5f;   // Scale factor for turbulence (0.0 - 10.0)
    float m_seed = 0.0f;        // Seed for deterministic patterns
    
    // Model info
    std::string m_modelName = "none";
    std::string m_modelPath;

    // Performance tracking
    PerformanceMetrics m_metrics = {};
};

