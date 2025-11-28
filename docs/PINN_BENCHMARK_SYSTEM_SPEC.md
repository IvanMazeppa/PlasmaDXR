# PINN Model Benchmarking & Preset Generation System

## Technical Specification v1.0

**Status**: Design Complete
**Author**: Claude (AI Assistant)
**Date**: 2025-11-28

---

## 1. Executive Summary

This system adds a **headless benchmark mode** to PlasmaDX-Clean for automated evaluation of PINN models. It enables:

- Rapid comparison of PINN v1/v2/v3/v4 under various conditions
- Automated preset generation with optimal parameters
- Comprehensive metrics: stability, performance, physical accuracy, visual quality
- JSON output for further analysis and preset import

**Zero Overhead Guarantee**: When `--benchmark` is not specified, the application runs identically to before with no additional code paths executed.

---

## 2. Architecture Overview

### 2.1 High-Level Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PlasmaDX-Clean Application                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   if (--benchmark) {                                                 │
│       ┌─────────────────────────────────────────────────────────┐   │
│       │              BENCHMARK MODE (Headless)                   │   │
│       │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │   │
│       │  │   Particle   │  │     PINN     │  │    SIREN     │   │   │
│       │  │    System    │  │   Physics    │  │   Vortex     │   │   │
│       │  └──────────────┘  └──────────────┘  └──────────────┘   │   │
│       │                          │                               │   │
│       │              ┌───────────▼────────────┐                  │   │
│       │              │   BenchmarkRunner       │                  │   │
│       │              │   - Run simulations    │                  │   │
│       │              │   - Collect metrics    │                  │   │
│       │              │   - Generate presets   │                  │   │
│       │              └───────────┬────────────┘                  │   │
│       │                          │                               │   │
│       │              ┌───────────▼────────────┐                  │   │
│       │              │   JSON/CSV Output       │                  │   │
│       └──────────────┴────────────────────────┴──────────────────┘   │
│                                                                       │
│   } else {                                                           │
│       // Normal rendering path (unchanged)                           │
│   }                                                                   │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Zero Overhead Design

The benchmark system is isolated behind a single boolean flag:

```cpp
// In Application.h
bool m_benchmarkMode = false;  // Set ONLY by --benchmark flag

// In Application::Initialize()
if (m_benchmarkMode) {
    return InitializeBenchmarkMode(argc, argv);
}
// ... normal initialization continues unchanged
```

When `m_benchmarkMode == false`:
- The `InitializeBenchmarkMode()` function is never called
- All benchmark-related code is dead code (optimized out by compiler)
- BenchmarkRunner class is never instantiated
- **Result: Zero overhead**

---

## 3. Command-Line Interface

### 3.1 Basic Usage

```bash
# Run single benchmark
PlasmaDX-Clean.exe --benchmark --pinn v4 --frames 1000

# Run with SIREN turbulence
PlasmaDX-Clean.exe --benchmark --pinn v4 --siren --siren-intensity 0.5

# Compare all models
PlasmaDX-Clean.exe --benchmark --compare-all --frames 500

# Run from config file
PlasmaDX-Clean.exe --benchmark --config benchmarks/orbital_stability.json

# Parallel sweep (runs multiple configs)
PlasmaDX-Clean.exe --benchmark --sweep benchmarks/param_sweep.json --parallel 4
```

### 3.2 Full Argument Reference

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--benchmark` | flag | - | Enable benchmark mode (required) |
| `--pinn <model>` | string | v4 | PINN model: v1, v2, v3, v4, or path |
| `--siren` | flag | off | Enable SIREN turbulence |
| `--siren-intensity` | float | 0.5 | SIREN turbulence intensity (0-5) |
| `--siren-seed` | float | 0.0 | SIREN random seed |
| `--particles` | int | 10000 | Number of particles |
| `--frames` | int | 1000 | Simulation frames to run |
| `--timestep` | float | 0.016 | Fixed timestep (seconds) |
| `--timescale` | float | 1.0 | Time multiplier (1-50) |
| `--output` | string | benchmark_results.json | Output file path |
| `--output-format` | string | json | Output format: json, csv, both |
| `--compare-all` | flag | - | Test all available PINN models |
| `--config` | string | - | Load benchmark config from JSON |
| `--sweep` | string | - | Run parameter sweep from config |
| `--parallel` | int | 1 | Parallel sweep instances |
| `--warmup` | int | 100 | Warmup frames (excluded from metrics) |
| `--sample-interval` | int | 10 | Frames between metric samples |
| `--verbose` | flag | - | Detailed logging |

---

## 4. Metrics System

### 4.1 Metric Categories

#### 4.1.1 Stability Metrics

Measure how well particles maintain stable orbits:

```cpp
struct StabilityMetrics {
    // Orbit Stability
    float escapeRate;           // Particles leaving outer boundary / total / time
    float collapseRate;         // Particles crossing inner boundary / total / time
    float boundaryViolations;   // Total boundary violations per frame
    
    // Energy Conservation
    float totalEnergyStart;     // Total kinetic + potential energy at start
    float totalEnergyEnd;       // Total energy at end
    float energyDrift;          // (end - start) / start as percentage
    float energyDriftPerSecond; // Energy drift rate
    
    // Angular Momentum
    float angularMomentumStart;
    float angularMomentumEnd;
    float angularMomentumDrift;
    
    // Velocity Distribution
    float velocityMean;
    float velocityStdDev;
    float velocityMin;
    float velocityMax;
    float velocityKurtosis;     // Detects velocity explosions
};
```

**Calculation Example:**
```cpp
float ComputeEscapeRate() {
    int escaped = 0;
    for (uint32_t i = 0; i < particleCount; i++) {
        float r = sqrt(pos[i].x*pos[i].x + pos[i].z*pos[i].z);
        if (r > OUTER_DISK_RADIUS) escaped++;
    }
    return (float)escaped / particleCount / elapsedTime;
}
```

#### 4.1.2 Performance Metrics

Measure computational efficiency:

```cpp
struct PerformanceMetrics {
    // Inference Time
    float pinnInferenceMs;       // PINN ONNX inference time
    float sirenInferenceMs;      // SIREN ONNX inference time (if enabled)
    float integrationMs;         // Force integration time
    float totalPhysicsMs;        // Total physics update time
    
    // Throughput
    float particlesPerSecond;    // Particles processed per second
    float framesPerSecond;       // Physics frames per second (uncapped)
    
    // Memory
    size_t peakMemoryMB;         // Peak memory usage
    size_t cpuBuffersMB;         // CPU particle buffer sizes
};
```

#### 4.1.3 Physical Accuracy Metrics

Compare simulation to analytical solutions:

```cpp
struct PhysicalAccuracyMetrics {
    // Keplerian Velocity Comparison
    float keplerianVelocityError;  // Average deviation from v=sqrt(GM/r)
    float keplerianVelocityRMSE;   // RMS error
    
    // Orbital Period
    float orbitalPeriodError;      // Deviation from T=2π*sqrt(r³/GM)
    
    // Force Magnitude
    float gravityForceError;       // Deviation from F=GM/r²
    float avgForceMagnitude;
    float avgRadialForce;          // Should be negative (attractive)
    float radialForceSign;         // Percentage with correct sign
    
    // Orbit Shape
    float eccentricityMean;        // Mean orbital eccentricity (0=circular)
    float eccentricityStdDev;
    float semiMajorAxisDrift;      // Change in semi-major axis over time
};
```

**Keplerian Velocity Error Calculation:**
```cpp
float ComputeKeplerianError() {
    float totalError = 0.0f;
    for (uint32_t i = 0; i < particleCount; i++) {
        float r = sqrt(pos[i].x*pos[i].x + pos[i].z*pos[i].z);
        float v_actual = sqrt(vel[i].x*vel[i].x + vel[i].z*vel[i].z);
        float v_kepler = sqrt(GM / r);
        totalError += abs(v_actual - v_kepler) / v_kepler;
    }
    return totalError / particleCount;
}
```

#### 4.1.4 Visual Quality Metrics

Proxy metrics for visual appearance (without rendering):

```cpp
struct VisualQualityMetrics {
    // Coherent Motion Detection
    float coherentMotionIndex;   // How much particles move as a group (bad)
    float velocityCovariance;    // Covariance of velocity vectors
    
    // Disk Structure
    float diskThicknessActual;   // Actual H/R ratio
    float diskThicknessTarget;   // Target H/R ratio
    float diskSymmetry;          // Azimuthal symmetry (1.0 = perfect)
    
    // Turbulence Quality (if SIREN enabled)
    float turbulenceStrength;    // Measured turbulence intensity
    float turbulenceCoherence;   // Spatial correlation of turbulence
    float vortexDetection;       // Number of detected vortex structures
    
    // Temporal Smoothness
    float velocityJerk;          // Rate of velocity change (lower = smoother)
    float positionFlicker;       // High-frequency position changes
};
```

### 4.2 Metric Aggregation

Over the simulation run, metrics are sampled at intervals and aggregated:

```cpp
struct AggregatedMetrics {
    // Per-metric statistics
    float mean;
    float median;
    float stdDev;
    float min;
    float max;
    float percentile95;
    float percentile99;
    
    // Time series (for plotting)
    std::vector<float> samples;
    std::vector<float> timestamps;
};
```

---

## 5. Output Formats

### 5.1 JSON Output Structure

```json
{
    "benchmark": {
        "version": "1.0",
        "timestamp": "2025-11-28T12:30:45Z",
        "duration_seconds": 16.5,
        "config": {
            "pinn_model": "pinn_v4_turbulence_robust",
            "pinn_version": 4,
            "siren_enabled": true,
            "siren_intensity": 0.5,
            "siren_seed": 0.0,
            "particle_count": 10000,
            "frames": 1000,
            "timestep": 0.016,
            "timescale": 1.0,
            "warmup_frames": 100
        }
    },
    "stability": {
        "escape_rate": { "mean": 0.0001, "max": 0.0005 },
        "collapse_rate": { "mean": 0.0, "max": 0.0 },
        "energy_drift_percent": -0.15,
        "angular_momentum_drift_percent": 0.02,
        "velocity_distribution": {
            "mean": 2.45,
            "stddev": 0.82,
            "min": 0.12,
            "max": 4.21
        }
    },
    "performance": {
        "pinn_inference_ms": { "mean": 3.2, "max": 4.1 },
        "siren_inference_ms": { "mean": 1.1, "max": 1.5 },
        "total_physics_ms": { "mean": 5.8, "max": 7.2 },
        "particles_per_second": 1720000,
        "frames_per_second": 172
    },
    "physical_accuracy": {
        "keplerian_velocity_error_percent": 2.3,
        "gravity_force_error_percent": 1.8,
        "radial_force_sign_correct_percent": 99.8,
        "mean_eccentricity": 0.05
    },
    "visual_quality": {
        "coherent_motion_index": 0.02,
        "disk_symmetry": 0.95,
        "turbulence_coherence": 0.45,
        "velocity_jerk": 0.12
    },
    "summary": {
        "overall_score": 87.5,
        "stability_score": 92.0,
        "performance_score": 85.0,
        "accuracy_score": 88.0,
        "visual_score": 85.0,
        "recommendation": "EXCELLENT - Suitable for production"
    },
    "generated_preset": {
        "name": "pinn_v4_balanced",
        "pinn_model": "pinn_v4_turbulence_robust",
        "timescale": 1.0,
        "siren_enabled": true,
        "siren_intensity": 0.5,
        "particle_count": 10000
    }
}
```

### 5.2 CSV Output (for Data Analysis)

```csv
timestamp,frame,escape_rate,collapse_rate,energy,angular_momentum,velocity_mean,velocity_max,pinn_ms,siren_ms,keplerian_error
0.000,0,0.0000,0.0000,1.0000,1.0000,2.45,4.21,3.2,1.1,0.023
0.160,10,0.0001,0.0000,0.9998,1.0001,2.46,4.19,3.1,1.2,0.024
...
```

### 5.3 Preset File Format

Generated presets can be directly loaded by the main application:

```json
{
    "preset_name": "pinn_v4_turbulent_fast",
    "description": "V4 model with moderate SIREN turbulence, 10x timescale",
    "generated_by": "benchmark_system",
    "benchmark_score": 85.2,
    
    "physics": {
        "pinn_model": "ml/models/pinn_v4_turbulence_robust.onnx",
        "timescale": 10.0,
        "enforce_boundaries": true
    },
    "turbulence": {
        "siren_enabled": true,
        "siren_intensity": 0.3,
        "siren_seed": 42.0
    },
    "particles": {
        "count": 10000,
        "radius_min": 10.0,
        "radius_max": 250.0
    }
}
```

---

## 6. Implementation Plan

### Phase 1: Core Infrastructure (2-3 hours)

#### 1.1 BenchmarkRunner Class

Create `src/benchmark/BenchmarkRunner.h`:

```cpp
#pragma once

#include <string>
#include <vector>
#include <memory>
#include "../particles/ParticleSystem.h"

struct BenchmarkConfig {
    std::string pinnModel = "ml/models/pinn_v4_turbulence_robust.onnx";
    bool sirenEnabled = false;
    float sirenIntensity = 0.5f;
    float sirenSeed = 0.0f;
    uint32_t particleCount = 10000;
    uint32_t frames = 1000;
    float timestep = 0.016f;
    float timescale = 1.0f;
    uint32_t warmupFrames = 100;
    uint32_t sampleInterval = 10;
    std::string outputPath = "benchmark_results.json";
    std::string outputFormat = "json"; // json, csv, both
};

struct StabilityMetrics { /* ... as defined above ... */ };
struct PerformanceMetrics { /* ... */ };
struct PhysicalAccuracyMetrics { /* ... */ };
struct VisualQualityMetrics { /* ... */ };

struct BenchmarkResults {
    BenchmarkConfig config;
    StabilityMetrics stability;
    PerformanceMetrics performance;
    PhysicalAccuracyMetrics accuracy;
    VisualQualityMetrics visual;
    float overallScore;
    std::string recommendation;
};

class BenchmarkRunner {
public:
    BenchmarkRunner();
    ~BenchmarkRunner();
    
    // Initialize with config (no GPU/rendering)
    bool Initialize(const BenchmarkConfig& config);
    
    // Run benchmark simulation
    BenchmarkResults Run();
    
    // Save results
    bool SaveResults(const BenchmarkResults& results, const std::string& path);
    
    // Generate preset from results
    bool GeneratePreset(const BenchmarkResults& results, const std::string& path);
    
private:
    void ComputeStabilityMetrics();
    void ComputePerformanceMetrics();
    void ComputePhysicalAccuracyMetrics();
    void ComputeVisualQualityMetrics();
    float ComputeOverallScore();
    
    std::unique_ptr<ParticleSystem> m_particleSystem;
    BenchmarkConfig m_config;
    
    // Metric accumulators
    std::vector<float> m_escapeRates;
    std::vector<float> m_energyValues;
    // ... etc
};
```

#### 1.2 Application Integration

Modify `src/core/Application.h`:

```cpp
// Add at top
class BenchmarkRunner;

// Add member
bool m_benchmarkMode = false;
std::unique_ptr<BenchmarkRunner> m_benchmarkRunner;

// Add methods
bool InitializeBenchmarkMode(int argc, char** argv);
int RunBenchmark();
```

Modify `src/core/Application.cpp`:

```cpp
bool Application::Initialize(HINSTANCE hInstance, int nCmdShow, int argc, char** argv) {
    // Check for benchmark mode FIRST
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--benchmark") {
            m_benchmarkMode = true;
            break;
        }
    }
    
    // If benchmark mode, skip all GPU/window initialization
    if (m_benchmarkMode) {
        return InitializeBenchmarkMode(argc, argv);
    }
    
    // ... rest of normal initialization unchanged ...
}

int Application::Run() {
    if (m_benchmarkMode) {
        return RunBenchmark();
    }
    
    // ... normal render loop unchanged ...
}
```

### Phase 2: Metric Collection (2-3 hours)

#### 2.1 Add Metric Collection to ParticleSystem

Add methods to `ParticleSystem.h`:

```cpp
// Benchmark metric collection
struct PhysicsSnapshot {
    float totalKineticEnergy;
    float totalPotentialEnergy;
    float totalAngularMomentum;
    uint32_t particlesInBounds;
    uint32_t particlesEscaped;
    uint32_t particlesCollapsed;
    float velocityMean;
    float velocityMax;
    float avgRadialForce;
};

PhysicsSnapshot CapturePhysicsSnapshot() const;
```

Implement in `ParticleSystem.cpp`:

```cpp
ParticleSystem::PhysicsSnapshot ParticleSystem::CapturePhysicsSnapshot() const {
    PhysicsSnapshot snap = {};
    
    float totalKE = 0.0f, totalPE = 0.0f, totalL = 0.0f;
    float velSum = 0.0f, velMax = 0.0f;
    
    for (uint32_t i = 0; i < m_activeParticleCount; i++) {
        float r = sqrtf(m_cpuPositions[i].x * m_cpuPositions[i].x +
                       m_cpuPositions[i].z * m_cpuPositions[i].z);
        float v2 = m_cpuVelocities[i].x * m_cpuVelocities[i].x +
                   m_cpuVelocities[i].y * m_cpuVelocities[i].y +
                   m_cpuVelocities[i].z * m_cpuVelocities[i].z;
        float v = sqrtf(v2);
        
        // Kinetic energy: 0.5 * m * v^2 (m=1)
        totalKE += 0.5f * v2;
        
        // Potential energy: -GM/r
        totalPE -= PINN_GM / (r + 1e-6f);
        
        // Angular momentum: r × v (z-component for 2D disk)
        totalL += m_cpuPositions[i].x * m_cpuVelocities[i].z -
                  m_cpuPositions[i].z * m_cpuVelocities[i].x;
        
        velSum += v;
        velMax = std::max(velMax, v);
        
        // Count boundary violations
        if (r < PINN_R_ISCO) snap.particlesCollapsed++;
        else if (r > OUTER_DISK_RADIUS) snap.particlesEscaped++;
        else snap.particlesInBounds++;
    }
    
    snap.totalKineticEnergy = totalKE;
    snap.totalPotentialEnergy = totalPE;
    snap.totalAngularMomentum = totalL;
    snap.velocityMean = velSum / m_activeParticleCount;
    snap.velocityMax = velMax;
    
    return snap;
}
```

### Phase 3: Output Generation (1-2 hours)

#### 3.1 JSON Output

Use the existing pattern in the codebase with `nlohmann/json` or a simple hand-written serializer:

```cpp
bool BenchmarkRunner::SaveResults(const BenchmarkResults& results, const std::string& path) {
    std::ofstream file(path);
    if (!file.is_open()) return false;
    
    file << "{\n";
    file << "  \"benchmark\": {\n";
    file << "    \"version\": \"1.0\",\n";
    file << "    \"timestamp\": \"" << GetTimestamp() << "\",\n";
    // ... serialize all metrics ...
    file << "  }\n";
    file << "}\n";
    
    return true;
}
```

### Phase 4: Parameter Sweeps (2-3 hours)

#### 4.1 Sweep Configuration

```json
{
    "sweep_name": "timescale_comparison",
    "base_config": {
        "pinn_model": "v4",
        "particles": 10000,
        "frames": 500
    },
    "parameters": [
        {
            "name": "timescale",
            "values": [1.0, 5.0, 10.0, 20.0, 50.0]
        },
        {
            "name": "siren_intensity",
            "values": [0.0, 0.25, 0.5, 1.0]
        }
    ],
    "output_dir": "benchmark_results/timescale_sweep/"
}
```

#### 4.2 Parallel Execution

```cpp
// Use std::async for parallel sweeps
std::vector<std::future<BenchmarkResults>> futures;

for (const auto& config : sweepConfigs) {
    futures.push_back(std::async(std::launch::async, [config]() {
        BenchmarkRunner runner;
        runner.Initialize(config);
        return runner.Run();
    }));
}

// Collect results
for (auto& future : futures) {
    results.push_back(future.get());
}
```

---

## 7. Files to Create/Modify

### New Files

| File | Description |
|------|-------------|
| `src/benchmark/BenchmarkRunner.h` | Core benchmark runner class |
| `src/benchmark/BenchmarkRunner.cpp` | Implementation |
| `src/benchmark/BenchmarkMetrics.h` | Metric structures |
| `src/benchmark/BenchmarkConfig.h` | Config structures |
| `src/benchmark/BenchmarkOutput.h` | JSON/CSV output utilities |
| `src/benchmark/BenchmarkOutput.cpp` | Output implementation |
| `benchmarks/default.json` | Default benchmark config |
| `benchmarks/compare_all.json` | Compare all models config |
| `benchmarks/param_sweep.json` | Parameter sweep template |

### Modified Files

| File | Changes |
|------|---------|
| `src/core/Application.h` | Add `m_benchmarkMode`, `BenchmarkRunner` pointer |
| `src/core/Application.cpp` | Add `--benchmark` parsing, `InitializeBenchmarkMode()` |
| `src/particles/ParticleSystem.h` | Add `CapturePhysicsSnapshot()` |
| `src/particles/ParticleSystem.cpp` | Implement snapshot method |
| `CMakeLists.txt` | Add benchmark source files |

---

## 8. Scoring Algorithm

The overall score is computed as a weighted average:

```cpp
float ComputeOverallScore() {
    // Weights (sum to 1.0)
    const float W_STABILITY = 0.35f;
    const float W_ACCURACY = 0.30f;
    const float W_PERFORMANCE = 0.20f;
    const float W_VISUAL = 0.15f;
    
    // Stability score (0-100)
    // - Penalize escape rate, collapse rate, energy drift
    float stabilityScore = 100.0f;
    stabilityScore -= m_stability.escapeRate * 1000.0f;      // -1 per 0.1% escape
    stabilityScore -= m_stability.collapseRate * 1000.0f;
    stabilityScore -= abs(m_stability.energyDrift) * 10.0f;  // -1 per 10% drift
    stabilityScore = std::clamp(stabilityScore, 0.0f, 100.0f);
    
    // Accuracy score (0-100)
    float accuracyScore = 100.0f;
    accuracyScore -= m_accuracy.keplerianVelocityError * 200.0f;  // -2 per 1% error
    accuracyScore -= (100.0f - m_accuracy.radialForceSignCorrect);
    accuracyScore = std::clamp(accuracyScore, 0.0f, 100.0f);
    
    // Performance score (0-100)
    // Based on target of 120 FPS physics
    float targetFPS = 120.0f;
    float performanceScore = std::min(100.0f, (m_performance.framesPerSecond / targetFPS) * 100.0f);
    
    // Visual score (0-100)
    float visualScore = 100.0f;
    visualScore -= m_visual.coherentMotionIndex * 100.0f;
    visualScore -= (1.0f - m_visual.diskSymmetry) * 50.0f;
    visualScore = std::clamp(visualScore, 0.0f, 100.0f);
    
    return W_STABILITY * stabilityScore +
           W_ACCURACY * accuracyScore +
           W_PERFORMANCE * performanceScore +
           W_VISUAL * visualScore;
}
```

---

## 9. Example Workflow

### 9.1 Quick Model Comparison

```bash
# Compare all PINN versions with default settings
PlasmaDX-Clean.exe --benchmark --compare-all --frames 500 --output results/model_comparison.json

# View results
cat results/model_comparison.json | jq '.summary'
```

### 9.2 Parameter Tuning

```bash
# Find optimal SIREN intensity for v4
PlasmaDX-Clean.exe --benchmark --pinn v4 --siren \
    --sweep-param siren_intensity:0.1:0.5:0.1 \
    --frames 300 --output results/siren_sweep.json

# Find best timescale
PlasmaDX-Clean.exe --benchmark --pinn v4 \
    --sweep-param timescale:1:50:5 \
    --frames 200 --output results/timescale_sweep.json
```

### 9.3 Generate Production Presets

```bash
# Run comprehensive benchmark and generate preset
PlasmaDX-Clean.exe --benchmark --pinn v4 --siren --frames 1000 \
    --generate-preset presets/v4_turbulent_balanced.json
```

---

## 10. Future Extensions

### 10.1 Web Dashboard
- Real-time benchmark progress visualization
- Historical comparison charts
- Automated regression testing

### 10.2 ML-Based Preset Generation
- Use benchmark results to train a model that predicts optimal parameters
- Genetic algorithm for parameter optimization

### 10.3 CI/CD Integration
- Automated benchmark runs on PINN model changes
- Performance regression alerts

---

## 11. Estimated Implementation Time

| Phase | Description | Time |
|-------|-------------|------|
| Phase 1 | Core Infrastructure | 2-3 hours |
| Phase 2 | Metric Collection | 2-3 hours |
| Phase 3 | Output Generation | 1-2 hours |
| Phase 4 | Parameter Sweeps | 2-3 hours |
| **Total** | | **7-11 hours** |

---

## 12. Quick Start Implementation

To begin implementation, start with these files in order:

1. **`src/benchmark/BenchmarkConfig.h`** - Define all config structures
2. **`src/benchmark/BenchmarkMetrics.h`** - Define all metric structures
3. **`src/benchmark/BenchmarkRunner.h/cpp`** - Core runner implementation
4. **`src/core/Application.cpp`** - Add `--benchmark` parsing
5. **`src/particles/ParticleSystem.cpp`** - Add `CapturePhysicsSnapshot()`
6. **`src/benchmark/BenchmarkOutput.cpp`** - JSON/CSV serialization

The implementation is modular - you can start with basic benchmarking (Phase 1-2) and add sweeps later (Phase 4).

