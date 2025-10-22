#pragma once

#include <d3d12.h>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>

// ML-based adaptive quality system for PlasmaDX-Clean
// Predicts frame time based on scene complexity and quality settings
// Automatically adjusts quality to maintain target FPS

class AdaptiveQualitySystem {
public:
    // Quality levels (in order of increasing performance)
    enum class QualityLevel {
        Ultra = 0,   // 16 shadow rays, all features max
        High = 1,    // 8 shadow rays, all features enabled
        Medium = 2,  // 4 shadow rays, selective features
        Low = 3,     // 1 shadow ray + temporal, essential only
        Minimal = 4  // 1 shadow ray, no extras
    };

    // Scene complexity features
    struct SceneFeatures {
        // Scene parameters
        float particleCount;           // 10000-100000
        float lightCount;              // 1-16
        float cameraDistance;          // 100-2000

        // Quality settings
        float shadowRaysPerLight;      // 1-16
        float useShadowRays;           // 0 or 1
        float useInScattering;         // 0 or 1
        float usePhaseFunction;        // 0 or 1
        float useAnisotropicGaussians; // 0 or 1
        float enableTemporalFiltering; // 0 or 1
        float useRTXDI;                // 0 or 1
        float enableRTLighting;        // 0 or 1
        float godRayDensity;           // 0.0-1.0

        // Runtime performance
        float actualFrameTime;         // Measured frame time (ms) - for training
    };

    // Quality preset configuration
    struct QualityPreset {
        uint32_t shadowRaysPerLight;
        bool useShadowRays;
        bool useInScattering;
        bool usePhaseFunction;
        bool useAnisotropicGaussians;
        bool enableTemporalFiltering;
        bool enableRTLighting;
        float godRayDensity;
        float rtLightingStrength;
    };

public:
    AdaptiveQualitySystem();
    ~AdaptiveQualitySystem();

    // Initialize system
    bool Initialize(const std::string& modelPath = "ml/models/adaptive_quality.model");

    // Enable/disable adaptive quality
    void SetEnabled(bool enabled) { m_enabled = enabled; }
    bool IsEnabled() const { return m_enabled; }

    // Set target FPS (60, 120, 144, etc.)
    void SetTargetFPS(float targetFPS);
    float GetTargetFPS() const { return m_targetFPS; }

    // Performance data collection (for training)
    void StartDataCollection(const std::string& outputPath = "ml/training_data/performance_data.csv");
    void StopDataCollection();
    void RecordSample(const SceneFeatures& features);

    // ML inference and quality adjustment
    float PredictFrameTime(const SceneFeatures& features);
    QualityLevel RecommendQualityLevel(const SceneFeatures& currentScene);
    QualityPreset GetQualityPreset(QualityLevel level) const;

    // Apply quality settings to application
    void ApplyQualityLevel(QualityLevel level);

    // Smooth transitions (prevents flickering)
    void Update(float deltaTime, const SceneFeatures& currentFeatures);

    // Get current quality level
    QualityLevel GetCurrentQuality() const { return m_currentQuality; }

    // Get stats
    float GetPredictedFrameTime() const { return m_predictedFrameTime; }
    float GetConfidence() const { return m_confidence; }

private:
    // Model management
    bool LoadModel(const std::string& path);
    void InitializeDefaultModel(); // Fallback if no trained model

    // Simple decision tree inference (lightweight, no external ML libs)
    float InferFrameTime(const SceneFeatures& features);

    // Quality adjustment logic
    QualityLevel SelectOptimalQuality(const SceneFeatures& features, float targetFrameTime);

    // Hysteresis to prevent oscillation
    bool ShouldChangeQuality(QualityLevel newLevel);

private:
    // System state
    bool m_enabled = false;
    QualityLevel m_currentQuality = QualityLevel::High;
    float m_targetFPS = 120.0f;
    float m_targetFrameTime = 8.333f; // 1000ms / 120fps

    // Prediction stats
    float m_predictedFrameTime = 0.0f;
    float m_confidence = 0.0f;

    // Hysteresis (prevent rapid quality changes)
    std::chrono::steady_clock::time_point m_lastQualityChange;
    float m_qualityChangeDelay = 2.0f; // Minimum 2 seconds between changes
    int m_consecutiveFramesBelowTarget = 0;
    int m_consecutiveFramesAboveTarget = 0;
    const int m_hysteresisThreshold = 30; // 30 frames @ 60fps = 0.5s

    // Data collection for training
    bool m_collectingData = false;
    std::ofstream m_dataFile;
    uint32_t m_sampleCount = 0;

    // Simple decision tree model (trained offline)
    struct DecisionNode {
        int featureIndex = -1;  // Which feature to split on (-1 = leaf node)
        float threshold = 0.0f; // Split threshold
        int leftChild = -1;     // Index of left child (-1 = none)
        int rightChild = -1;    // Index of right child (-1 = none)
        float prediction = 0.0f; // Leaf node prediction (frame time in ms)
    };
    std::vector<DecisionNode> m_decisionTree;
    bool m_modelLoaded = false;

    // Feature scaling parameters (learned during training)
    struct FeatureScaling {
        float mean = 0.0f;
        float stdDev = 1.0f;
    };
    std::vector<FeatureScaling> m_featureScaling;
};
