#include "AdaptiveQualitySystem.h"
#include "../utils/Logger.h"
#include <algorithm>
#include <cmath>
#include <sstream>

AdaptiveQualitySystem::AdaptiveQualitySystem() {
    m_lastQualityChange = std::chrono::steady_clock::now();

    // Initialize feature scaling (12 features)
    m_featureScaling.resize(12);
}

AdaptiveQualitySystem::~AdaptiveQualitySystem() {
    if (m_collectingData) {
        StopDataCollection();
    }
}

bool AdaptiveQualitySystem::Initialize(const std::string& modelPath) {
    LOG_INFO("[AdaptiveQuality] Initializing ML-based adaptive quality system...");

    // Try to load trained model
    if (LoadModel(modelPath)) {
        LOG_INFO("[AdaptiveQuality] Loaded trained model from: {}", modelPath);
        m_modelLoaded = true;
    } else {
        LOG_WARN("[AdaptiveQuality] No trained model found, using default heuristics");
        InitializeDefaultModel();
        m_modelLoaded = false;
    }

    // Set default target
    SetTargetFPS(120.0f);

    return true;
}

void AdaptiveQualitySystem::SetTargetFPS(float targetFPS) {
    m_targetFPS = targetFPS;
    m_targetFrameTime = 1000.0f / targetFPS; // Convert to ms
    LOG_INFO("[AdaptiveQuality] Target FPS set to {:.1f} ({:.2f}ms)", targetFPS, m_targetFrameTime);
}

void AdaptiveQualitySystem::StartDataCollection(const std::string& outputPath) {
    if (m_collectingData) {
        LOG_WARN("[AdaptiveQuality] Data collection already active");
        return;
    }

    m_dataFile.open(outputPath);
    if (!m_dataFile.is_open()) {
        LOG_ERROR("[AdaptiveQuality] Failed to open data file: {}", outputPath);
        return;
    }

    // Write CSV header
    m_dataFile << "particleCount,lightCount,cameraDistance,shadowRaysPerLight,"
               << "useShadowRays,useInScattering,usePhaseFunction,useAnisotropicGaussians,"
               << "enableTemporalFiltering,useRTXDI,enableRTLighting,godRayDensity,"
               << "frameTime\n";

    m_collectingData = true;
    m_sampleCount = 0;
    LOG_INFO("[AdaptiveQuality] Started data collection: {}", outputPath);
}

void AdaptiveQualitySystem::StopDataCollection() {
    if (!m_collectingData) return;

    m_dataFile.close();
    m_collectingData = false;
    LOG_INFO("[AdaptiveQuality] Stopped data collection. Samples recorded: {}", m_sampleCount);
}

void AdaptiveQualitySystem::RecordSample(const SceneFeatures& features) {
    if (!m_collectingData) return;

    // Write CSV row
    m_dataFile << features.particleCount << ","
               << features.lightCount << ","
               << features.cameraDistance << ","
               << features.shadowRaysPerLight << ","
               << features.useShadowRays << ","
               << features.useInScattering << ","
               << features.usePhaseFunction << ","
               << features.useAnisotropicGaussians << ","
               << features.enableTemporalFiltering << ","
               << features.useRTXDI << ","
               << features.enableRTLighting << ","
               << features.godRayDensity << ","
               << features.actualFrameTime << "\n";

    m_sampleCount++;

    // Flush every 100 samples
    if (m_sampleCount % 100 == 0) {
        m_dataFile.flush();
    }
}

float AdaptiveQualitySystem::PredictFrameTime(const SceneFeatures& features) {
    if (m_modelLoaded) {
        return InferFrameTime(features);
    } else {
        // Fallback heuristic (simple cost model)
        float baseCost = 2.0f; // Base frame time

        // Particle cost
        float particleCost = features.particleCount / 10000.0f;

        // Shadow ray cost (most expensive)
        float shadowCost = features.useShadowRays * features.shadowRaysPerLight * features.lightCount * 0.5f;

        // Feature costs
        float featureCost = 0.0f;
        featureCost += features.useInScattering * 2.0f;        // Very expensive
        featureCost += features.usePhaseFunction * 0.5f;
        featureCost += features.useAnisotropicGaussians * 0.3f;
        featureCost += features.enableRTLighting * 1.5f;
        featureCost += features.godRayDensity * 3.0f;          // Proportional to density

        // Distance affects overdraw
        float distanceFactor = 1.0f;
        if (features.cameraDistance < 500.0f) {
            distanceFactor = 1.5f; // Close-up more expensive
        }

        return (baseCost + particleCost + shadowCost + featureCost) * distanceFactor;
    }
}

AdaptiveQualitySystem::QualityLevel AdaptiveQualitySystem::RecommendQualityLevel(
    const SceneFeatures& currentScene) {

    if (!m_enabled) {
        return m_currentQuality; // No change if disabled
    }

    return SelectOptimalQuality(currentScene, m_targetFrameTime);
}

AdaptiveQualitySystem::QualityPreset AdaptiveQualitySystem::GetQualityPreset(QualityLevel level) const {
    QualityPreset preset;

    switch (level) {
        case QualityLevel::Ultra:
            preset.shadowRaysPerLight = 16;
            preset.useShadowRays = true;
            preset.useInScattering = true;
            preset.usePhaseFunction = true;
            preset.useAnisotropicGaussians = true;
            preset.enableTemporalFiltering = true;
            preset.enableRTLighting = true;
            preset.godRayDensity = 1.0f;
            preset.rtLightingStrength = 2.0f;
            break;

        case QualityLevel::High:
            preset.shadowRaysPerLight = 8;
            preset.useShadowRays = true;
            preset.useInScattering = false; // Too expensive
            preset.usePhaseFunction = true;
            preset.useAnisotropicGaussians = true;
            preset.enableTemporalFiltering = true;
            preset.enableRTLighting = true;
            preset.godRayDensity = 0.5f;
            preset.rtLightingStrength = 2.0f;
            break;

        case QualityLevel::Medium:
            preset.shadowRaysPerLight = 4;
            preset.useShadowRays = true;
            preset.useInScattering = false;
            preset.usePhaseFunction = true;
            preset.useAnisotropicGaussians = true;
            preset.enableTemporalFiltering = true;
            preset.enableRTLighting = true;
            preset.godRayDensity = 0.2f;
            preset.rtLightingStrength = 1.5f;
            break;

        case QualityLevel::Low:
            preset.shadowRaysPerLight = 1;
            preset.useShadowRays = true;
            preset.useInScattering = false;
            preset.usePhaseFunction = true;
            preset.useAnisotropicGaussians = false;
            preset.enableTemporalFiltering = true; // Keep temporal for soft shadows
            preset.enableRTLighting = true;
            preset.godRayDensity = 0.0f;
            preset.rtLightingStrength = 1.0f;
            break;

        case QualityLevel::Minimal:
            preset.shadowRaysPerLight = 1;
            preset.useShadowRays = true;
            preset.useInScattering = false;
            preset.usePhaseFunction = false;
            preset.useAnisotropicGaussians = false;
            preset.enableTemporalFiltering = true;
            preset.enableRTLighting = false; // Disable RT lighting
            preset.godRayDensity = 0.0f;
            preset.rtLightingStrength = 0.0f;
            break;
    }

    return preset;
}

void AdaptiveQualitySystem::Update(float deltaTime, const SceneFeatures& currentFeatures) {
    if (!m_enabled) return;

    // Predict frame time
    m_predictedFrameTime = PredictFrameTime(currentFeatures);

    // Determine if quality adjustment needed
    if (currentFeatures.actualFrameTime > m_targetFrameTime * 1.1f) {
        // Running slow - consider reducing quality
        m_consecutiveFramesBelowTarget++;
        m_consecutiveFramesAboveTarget = 0;
    } else if (currentFeatures.actualFrameTime < m_targetFrameTime * 0.8f) {
        // Running fast - consider increasing quality
        m_consecutiveFramesAboveTarget++;
        m_consecutiveFramesBelowTarget = 0;
    } else {
        // Within target range - reset counters
        m_consecutiveFramesBelowTarget = 0;
        m_consecutiveFramesAboveTarget = 0;
    }

    // Apply hysteresis - only change after sustained deviation
    if (m_consecutiveFramesBelowTarget > m_hysteresisThreshold) {
        // Reduce quality
        QualityLevel newLevel = static_cast<QualityLevel>(
            std::min(static_cast<int>(m_currentQuality) + 1,
                     static_cast<int>(QualityLevel::Minimal)));

        if (ShouldChangeQuality(newLevel)) {
            LOG_INFO("[AdaptiveQuality] Performance below target, reducing quality: {} -> {}",
                     static_cast<int>(m_currentQuality), static_cast<int>(newLevel));
            ApplyQualityLevel(newLevel);
            m_consecutiveFramesBelowTarget = 0;
        }
    } else if (m_consecutiveFramesAboveTarget > m_hysteresisThreshold * 2) {
        // Increase quality (require longer period before upgrading)
        QualityLevel newLevel = static_cast<QualityLevel>(
            std::max(static_cast<int>(m_currentQuality) - 1,
                     static_cast<int>(QualityLevel::Ultra)));

        if (ShouldChangeQuality(newLevel)) {
            LOG_INFO("[AdaptiveQuality] Performance above target, increasing quality: {} -> {}",
                     static_cast<int>(m_currentQuality), static_cast<int>(newLevel));
            ApplyQualityLevel(newLevel);
            m_consecutiveFramesAboveTarget = 0;
        }
    }

    // Record training data if collecting
    if (m_collectingData) {
        RecordSample(currentFeatures);
    }
}

void AdaptiveQualitySystem::ApplyQualityLevel(QualityLevel level) {
    m_currentQuality = level;
    m_lastQualityChange = std::chrono::steady_clock::now();

    // Note: Actual application of settings happens in Application class
    // This just stores the recommended level
}

bool AdaptiveQualitySystem::LoadModel(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Simple binary format:
    // - uint32_t: number of nodes
    // - DecisionNode[]: nodes
    // - FeatureScaling[12]: scaling parameters

    uint32_t nodeCount = 0;
    file.read(reinterpret_cast<char*>(&nodeCount), sizeof(nodeCount));

    if (nodeCount == 0 || nodeCount > 10000) {
        LOG_ERROR("[AdaptiveQuality] Invalid model file (node count: {})", nodeCount);
        return false;
    }

    m_decisionTree.resize(nodeCount);
    file.read(reinterpret_cast<char*>(m_decisionTree.data()),
              nodeCount * sizeof(DecisionNode));

    // Load feature scaling
    file.read(reinterpret_cast<char*>(m_featureScaling.data()),
              12 * sizeof(FeatureScaling));

    file.close();
    return true;
}

void AdaptiveQualitySystem::InitializeDefaultModel() {
    // Simple heuristic-based model (no ML)
    // Predictions are calculated in PredictFrameTime using cost model
    m_decisionTree.clear();

    // Initialize default feature scaling (identity)
    for (auto& scaling : m_featureScaling) {
        scaling.mean = 0.0f;
        scaling.stdDev = 1.0f;
    }
}

float AdaptiveQualitySystem::InferFrameTime(const SceneFeatures& features) {
    if (m_decisionTree.empty()) {
        LOG_WARN("[AdaptiveQuality] No model loaded, using heuristics");
        return PredictFrameTime(features); // Fall back to heuristic
    }

    // Convert features to array
    float featureArray[12] = {
        features.particleCount,
        features.lightCount,
        features.cameraDistance,
        features.shadowRaysPerLight,
        features.useShadowRays,
        features.useInScattering,
        features.usePhaseFunction,
        features.useAnisotropicGaussians,
        features.enableTemporalFiltering,
        features.useRTXDI,
        features.enableRTLighting,
        features.godRayDensity
    };

    // Normalize features
    float normalizedFeatures[12];
    for (int i = 0; i < 12; i++) {
        normalizedFeatures[i] = (featureArray[i] - m_featureScaling[i].mean) /
                                m_featureScaling[i].stdDev;
    }

    // Traverse decision tree
    int nodeIndex = 0;
    while (nodeIndex >= 0 && nodeIndex < static_cast<int>(m_decisionTree.size())) {
        const DecisionNode& node = m_decisionTree[nodeIndex];

        if (node.featureIndex == -1) {
            // Leaf node - return prediction
            return node.prediction;
        }

        // Internal node - traverse based on feature value
        if (normalizedFeatures[node.featureIndex] <= node.threshold) {
            nodeIndex = node.leftChild;
        } else {
            nodeIndex = node.rightChild;
        }
    }

    LOG_ERROR("[AdaptiveQuality] Decision tree traversal error");
    return m_targetFrameTime; // Fallback
}

AdaptiveQualitySystem::QualityLevel AdaptiveQualitySystem::SelectOptimalQuality(
    const SceneFeatures& features, float targetFrameTime) {

    // Try each quality level and pick the highest that meets target
    for (int level = static_cast<int>(QualityLevel::Ultra);
         level <= static_cast<int>(QualityLevel::Minimal);
         level++) {

        // Create test features with this quality level
        SceneFeatures testFeatures = features;
        QualityPreset preset = GetQualityPreset(static_cast<QualityLevel>(level));

        testFeatures.shadowRaysPerLight = preset.shadowRaysPerLight;
        testFeatures.useShadowRays = preset.useShadowRays ? 1.0f : 0.0f;
        testFeatures.useInScattering = preset.useInScattering ? 1.0f : 0.0f;
        testFeatures.usePhaseFunction = preset.usePhaseFunction ? 1.0f : 0.0f;
        testFeatures.useAnisotropicGaussians = preset.useAnisotropicGaussians ? 1.0f : 0.0f;
        testFeatures.enableTemporalFiltering = preset.enableTemporalFiltering ? 1.0f : 0.0f;
        testFeatures.enableRTLighting = preset.enableRTLighting ? 1.0f : 0.0f;
        testFeatures.godRayDensity = preset.godRayDensity;

        float predictedTime = PredictFrameTime(testFeatures);

        // Add 10% safety margin
        if (predictedTime <= targetFrameTime * 1.1f) {
            return static_cast<QualityLevel>(level);
        }
    }

    // If nothing works, return Minimal
    return QualityLevel::Minimal;
}

bool AdaptiveQualitySystem::ShouldChangeQuality(QualityLevel newLevel) {
    if (newLevel == m_currentQuality) {
        return false; // No change
    }

    // Enforce minimum delay between changes
    auto now = std::chrono::steady_clock::now();
    float elapsed = std::chrono::duration<float>(now - m_lastQualityChange).count();

    if (elapsed < m_qualityChangeDelay) {
        return false; // Too soon
    }

    return true;
}
