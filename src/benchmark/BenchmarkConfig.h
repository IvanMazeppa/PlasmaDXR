#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace Benchmark {

// Physics parameter configuration
struct PhysicsConfig {
    // Gravitational
    float gm = 100.0f;                    // Gravitational parameter (G*M)
    float blackHoleMass = 1.0f;           // Mass multiplier (0.1-10.0)

    // Viscosity
    float alphaViscosity = 0.1f;          // Shakura-Sunyaev alpha (0.001-1.0)
    float damping = 1.0f;                 // Velocity damping (0.9-1.0)

    // Disk geometry
    float diskThickness = 0.1f;           // H/R ratio (0.01-0.5)
    float innerRadius = 6.0f;             // ISCO
    float outerRadius = 300.0f;           // Disk edge

    // Density
    float densityScale = 1.0f;            // Global density multiplier

    // Dynamics
    float angularMomentumBoost = 1.0f;    // Initial velocity boost
};

// Simulation configuration
struct SimulationConfig {
    float forceClamp = 10.0f;             // Max force magnitude
    float velocityClamp = 20.0f;          // Max velocity magnitude
    int boundaryMode = 1;                 // Boundary handling (0=none, 1=reflect, 2=wrap, 3=respawn)
};

// Turbulence configuration
struct TurbulenceConfig {
    bool sirenEnabled = false;
    float sirenIntensity = 0.5f;
    float sirenSeed = 0.0f;

    // Physics constraints
    bool conserveAngularMomentum = true;  // Project out net torque
    float vortexScale = 1.0f;             // Eddy size multiplier
    float vortexDecay = 0.1f;             // Temporal decay rate
};

// Configuration for a single benchmark run
struct BenchmarkConfig {
    // PINN Model
    std::string pinnModel = "ml/models/pinn_v4_turbulence_robust.onnx";

    // Physics parameters
    PhysicsConfig physics;

    // Simulation parameters
    SimulationConfig simulation;

    // Turbulence parameters
    TurbulenceConfig turbulence;

    // Simulation Parameters
    uint32_t particleCount = 10000;
    uint32_t frames = 1000;
    float timestep = 0.016f;       // 60 FPS equivalent
    float timescale = 1.0f;        // Time multiplier
    bool enforceBoundaries = false; // Containment volume (disabled by default)
    bool hybridMode = false;        // PINN + GPU hybrid (disabled by default)
    
    // Benchmark Settings
    uint32_t warmupFrames = 100;   // Excluded from metrics
    uint32_t sampleInterval = 10;  // Frames between metric samples
    
    // Output
    std::string outputPath = "benchmark_results.json";
    std::string outputFormat = "json";  // json, csv, both
    bool generatePreset = false;
    std::string presetPath = "";
    
    // Verbosity
    bool verbose = false;
};

// Configuration for parameter sweep
struct SweepParameter {
    std::string name;
    std::vector<float> values;
};

struct SweepConfig {
    std::string sweepName;
    BenchmarkConfig baseConfig;
    std::vector<SweepParameter> parameters;
    std::string outputDir = "benchmark_results/";
    uint32_t parallelInstances = 1;
};

// Available PINN models
struct PINNModelInfo {
    std::string name;
    std::string path;
    int version;
    std::string description;
};

inline std::vector<PINNModelInfo> GetAvailablePINNModels() {
    return {
        {"v4", "ml/models/pinn_v4_turbulence_robust.onnx", 4, "Turbulence-Robust (best for dynamic scenes)"},
        {"v3", "ml/models/pinn_v3_total_forces.onnx", 3, "Total Forces (orbital physics)"},
        {"v2", "ml/models/pinn_v2_turbulent.onnx", 2, "Legacy Turbulent"},
        {"v1", "ml/models/pinn_accretion_disk.onnx", 1, "Legacy Original"}
    };
}

// Resolve shorthand model names (v1, v2, v3, v4) to full paths
inline std::string ResolvePINNModelPath(const std::string& modelArg) {
    if (modelArg == "v4") return "ml/models/pinn_v4_turbulence_robust.onnx";
    if (modelArg == "v3") return "ml/models/pinn_v3_total_forces.onnx";
    if (modelArg == "v2") return "ml/models/pinn_v2_turbulent.onnx";
    if (modelArg == "v1") return "ml/models/pinn_accretion_disk.onnx";
    return modelArg;  // Assume full path
}

} // namespace Benchmark

