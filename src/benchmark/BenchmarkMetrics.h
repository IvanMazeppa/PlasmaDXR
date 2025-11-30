#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace Benchmark {

// Aggregated statistics for a single metric
struct MetricStats {
    float mean = 0.0f;
    float median = 0.0f;
    float stdDev = 0.0f;
    float min = 0.0f;
    float max = 0.0f;
    float percentile95 = 0.0f;
    float percentile99 = 0.0f;
    std::vector<float> samples;
    
    void Compute() {
        if (samples.empty()) return;
        
        // Sort for percentiles
        std::vector<float> sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        
        size_t n = sorted.size();
        min = sorted.front();
        max = sorted.back();
        median = sorted[n / 2];
        percentile95 = sorted[static_cast<size_t>(n * 0.95)];
        percentile99 = sorted[static_cast<size_t>(n * 0.99)];
        
        // Mean
        mean = std::accumulate(samples.begin(), samples.end(), 0.0f) / n;
        
        // StdDev
        float sumSq = 0.0f;
        for (float v : samples) {
            float diff = v - mean;
            sumSq += diff * diff;
        }
        stdDev = std::sqrt(sumSq / n);
    }
    
    void AddSample(float value) {
        samples.push_back(value);
    }
};

// Stability metrics - how well particles maintain orbits
struct StabilityMetrics {
    // Orbit containment
    MetricStats escapeRate;        // Particles leaving outer boundary / total / time
    MetricStats collapseRate;      // Particles crossing inner boundary / total / time
    uint32_t totalEscaped = 0;
    uint32_t totalCollapsed = 0;
    
    // Energy conservation
    float totalEnergyStart = 0.0f;
    float totalEnergyEnd = 0.0f;
    float energyDriftPercent = 0.0f;
    MetricStats energyValues;
    
    // Angular momentum conservation
    float angularMomentumStart = 0.0f;
    float angularMomentumEnd = 0.0f;
    float angularMomentumDriftPercent = 0.0f;
    MetricStats angularMomentumValues;
    
    // Velocity distribution
    MetricStats velocityMean;
    MetricStats velocityMax;
    float velocityStdDev = 0.0f;
    
    void Finalize() {
        escapeRate.Compute();
        collapseRate.Compute();
        energyValues.Compute();
        angularMomentumValues.Compute();
        velocityMean.Compute();
        velocityMax.Compute();
        
        if (totalEnergyStart != 0.0f) {
            energyDriftPercent = ((totalEnergyEnd - totalEnergyStart) / std::abs(totalEnergyStart)) * 100.0f;
        }
        if (angularMomentumStart != 0.0f) {
            angularMomentumDriftPercent = ((angularMomentumEnd - angularMomentumStart) / std::abs(angularMomentumStart)) * 100.0f;
        }
    }
};

// Performance metrics - computational efficiency
struct PerformanceMetrics {
    MetricStats pinnInferenceMs;
    MetricStats sirenInferenceMs;
    MetricStats integrationMs;
    MetricStats totalPhysicsMs;
    
    float particlesPerSecond = 0.0f;
    float framesPerSecond = 0.0f;
    float totalSimulationTimeMs = 0.0f;
    
    size_t peakMemoryMB = 0;
    
    void Finalize() {
        pinnInferenceMs.Compute();
        sirenInferenceMs.Compute();
        integrationMs.Compute();
        totalPhysicsMs.Compute();
    }
};

// Physical accuracy metrics - comparison to analytical solutions
struct PhysicalAccuracyMetrics {
    // Keplerian velocity comparison: v = sqrt(GM/r)
    MetricStats keplerianVelocityError;  // Percentage deviation
    float keplerianVelocityRMSE = 0.0f;
    
    // Force magnitude: F = GM/r²
    MetricStats gravityForceError;
    MetricStats avgForceMagnitude;
    MetricStats avgRadialForce;
    float radialForceSignCorrectPercent = 0.0f;  // Should be negative (attractive)
    
    // Orbit shape
    MetricStats eccentricity;
    float meanEccentricity = 0.0f;
    
    void Finalize() {
        keplerianVelocityError.Compute();
        gravityForceError.Compute();
        avgForceMagnitude.Compute();
        avgRadialForce.Compute();
        eccentricity.Compute();
        meanEccentricity = eccentricity.mean;
    }
};

// Visual quality metrics - proxy for appearance without rendering
struct VisualQualityMetrics {
    // Coherent motion (bad - all particles moving together)
    MetricStats coherentMotionIndex;
    
    // Disk structure
    MetricStats diskThicknessActual;  // H/R ratio
    float diskThicknessTarget = 0.1f;
    MetricStats diskSymmetry;  // 1.0 = perfect
    
    // Turbulence quality (if SIREN enabled)
    MetricStats turbulenceStrength;
    float vortexDetectionCount = 0.0f;
    
    // Temporal smoothness
    MetricStats velocityJerk;  // Rate of velocity change
    
    void Finalize() {
        coherentMotionIndex.Compute();
        diskThicknessActual.Compute();
        diskSymmetry.Compute();
        turbulenceStrength.Compute();
        velocityJerk.Compute();
    }
};

// Complete benchmark results
struct BenchmarkResults {
    // Configuration used
    std::string pinnModel;
    int pinnVersion = 0;
    bool sirenEnabled = false;
    float sirenIntensity = 0.0f;
    uint32_t particleCount = 0;
    uint32_t framesSimulated = 0;
    float timestep = 0.0f;
    float timescale = 0.0f;
    
    // Timestamp
    std::string timestamp;
    float durationSeconds = 0.0f;
    
    // Metrics
    StabilityMetrics stability;
    PerformanceMetrics performance;
    PhysicalAccuracyMetrics accuracy;
    VisualQualityMetrics visual;
    
    // Scores (0-100)
    float stabilityScore = 0.0f;
    float performanceScore = 0.0f;
    float accuracyScore = 0.0f;
    float visualScore = 0.0f;
    float overallScore = 0.0f;
    
    // Recommendation
    std::string recommendation;
    
    void Finalize() {
        stability.Finalize();
        performance.Finalize();
        accuracy.Finalize();
        visual.Finalize();
        ComputeScores();
    }
    
    void ComputeScores() {
        // Stability score (0-100)
        stabilityScore = 100.0f;
        stabilityScore -= stability.escapeRate.mean * 10000.0f;  // -1 per 0.01% escape
        stabilityScore -= stability.collapseRate.mean * 10000.0f;
        stabilityScore -= std::abs(stability.energyDriftPercent) * 2.0f;  // -2 per 1% drift
        stabilityScore = std::clamp(stabilityScore, 0.0f, 100.0f);
        
        // Compute radial force sign correctness (negative = attractive = correct)
        // Must be done here AFTER accuracy.Finalize() computes avgRadialForce.mean
        accuracy.radialForceSignCorrectPercent = 
            (accuracy.avgRadialForce.mean < 0.0f) ? 100.0f : 0.0f;
        
        // Accuracy score (0-100)
        accuracyScore = 100.0f;
        accuracyScore -= accuracy.keplerianVelocityError.mean * 5.0f;  // -5 per 1% error
        accuracyScore -= (100.0f - accuracy.radialForceSignCorrectPercent);
        accuracyScore = std::clamp(accuracyScore, 0.0f, 100.0f);
        
        // Performance score (0-100)
        float targetFPS = 120.0f;
        performanceScore = std::min(100.0f, (performance.framesPerSecond / targetFPS) * 100.0f);
        
        // Visual score (0-100)
        visualScore = 100.0f;
        visualScore -= visual.coherentMotionIndex.mean * 100.0f;
        visualScore -= (1.0f - visual.diskSymmetry.mean) * 50.0f;
        visualScore = std::clamp(visualScore, 0.0f, 100.0f);
        
        // Overall score (weighted average)
        const float W_STABILITY = 0.35f;
        const float W_ACCURACY = 0.30f;
        const float W_PERFORMANCE = 0.20f;
        const float W_VISUAL = 0.15f;
        
        overallScore = W_STABILITY * stabilityScore +
                       W_ACCURACY * accuracyScore +
                       W_PERFORMANCE * performanceScore +
                       W_VISUAL * visualScore;
        
        // Generate recommendation
        if (overallScore >= 90.0f) {
            recommendation = "EXCELLENT - Production ready, ideal for this simulation type";
        } else if (overallScore >= 80.0f) {
            recommendation = "GOOD - Suitable for production with minor caveats";
        } else if (overallScore >= 70.0f) {
            recommendation = "ACCEPTABLE - Works but has noticeable issues";
        } else if (overallScore >= 50.0f) {
            recommendation = "POOR - Significant issues, consider different model";
        } else {
            recommendation = "UNSUITABLE - Do not use for this simulation type";
        }
    }
};

// Physics snapshot captured each sample interval
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
    
    // For coherent motion detection
    float velocityCovarianceXZ = 0.0f;
    
    // Disk shape
    float diskThicknessRatio = 0.0f;  // H/R
};

} // namespace Benchmark

