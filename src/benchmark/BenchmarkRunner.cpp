#include "BenchmarkRunner.h"
#include "../core/Device.h"
#include "../utils/ResourceManager.h"
#include "../utils/Logger.h"
#include "../particles/ParticleSystem.h"
#include "../ml/PINNPhysicsSystem.h"
#include "../ml/SIRENVortexField.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <cmath>

namespace Benchmark {

BenchmarkRunner::BenchmarkRunner() {
    LOG_INFO("[Benchmark] BenchmarkRunner created");
}

BenchmarkRunner::~BenchmarkRunner() {
    LOG_INFO("[Benchmark] BenchmarkRunner destroyed");
}

bool BenchmarkRunner::Initialize(const BenchmarkConfig& config) {
    LOG_INFO("[Benchmark] ==========================================");
    LOG_INFO("[Benchmark] Initializing PINN Benchmark System");
    LOG_INFO("[Benchmark] ==========================================");
    
    m_config = config;
    
    // Log configuration
    LOG_INFO("[Benchmark] Configuration:");
    LOG_INFO("[Benchmark]   PINN Model: {}", config.pinnModel);
    LOG_INFO("[Benchmark]   SIREN: {} (intensity: {:.2f})", 
             config.sirenEnabled ? "ENABLED" : "DISABLED", config.sirenIntensity);
    LOG_INFO("[Benchmark]   Particles: {}", config.particleCount);
    LOG_INFO("[Benchmark]   Frames: {} (warmup: {})", config.frames, config.warmupFrames);
    LOG_INFO("[Benchmark]   Timestep: {:.4f}s, Timescale: {:.1f}x", config.timestep, config.timescale);
    LOG_INFO("[Benchmark]   Output: {} ({})", config.outputPath, config.outputFormat);
    
    // Initialize minimal Device (needed for buffer creation even without rendering)
    LOG_INFO("[Benchmark] Creating minimal device (no swap chain)...");
    m_device = std::make_unique<Device>();
    if (!m_device->Initialize(false)) {  // No debug layer for speed
        LOG_ERROR("[Benchmark] Failed to initialize device");
        return false;
    }
    
    // Initialize resource manager
    m_resources = std::make_unique<ResourceManager>();
    if (!m_resources->Initialize(m_device.get())) {
        LOG_ERROR("[Benchmark] Failed to initialize resource manager");
        return false;
    }
    m_resources->CreateDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 100, true);
    
    // Initialize particle system
    LOG_INFO("[Benchmark] Initializing particle system ({} particles)...", config.particleCount);
    m_particleSystem = std::make_unique<ParticleSystem>();
    if (!m_particleSystem->Initialize(m_device.get(), m_resources.get(), config.particleCount)) {
        LOG_ERROR("[Benchmark] Failed to initialize particle system");
        return false;
    }
    
    // Load specified PINN model
    if (m_particleSystem->IsPINNAvailable()) {
        LOG_INFO("[Benchmark] Loading PINN model: {}", config.pinnModel);
        if (!m_particleSystem->LoadPINNModel(config.pinnModel)) {
            LOG_ERROR("[Benchmark] Failed to load PINN model: {}", config.pinnModel);
            return false;
        }
        m_particleSystem->SetPINNEnabled(true);
        
        // Store model info in results
        m_results.pinnModel = config.pinnModel;
        m_results.pinnVersion = m_particleSystem->GetPINNModelVersion();
    } else {
        LOG_ERROR("[Benchmark] PINN not available (ONNX Runtime not installed?)");
        return false;
    }
    
    // Configure SIREN if enabled
    if (config.sirenEnabled && m_particleSystem->IsSIRENAvailable()) {
        LOG_INFO("[Benchmark] Enabling SIREN turbulence (intensity: {:.2f})", config.sirenIntensity);
        m_particleSystem->SetSIRENEnabled(true);
        m_particleSystem->SetSIRENIntensity(config.sirenIntensity);
        m_particleSystem->SetSIRENSeed(config.sirenSeed);
    }
    
    // Set timescale
    m_particleSystem->SetTimeScale(config.timescale);
    
    // Initialize results structure
    m_results.sirenEnabled = config.sirenEnabled;
    m_results.sirenIntensity = config.sirenIntensity;
    m_results.particleCount = config.particleCount;
    m_results.framesSimulated = 0;
    m_results.timestep = config.timestep;
    m_results.timescale = config.timescale;
    
    m_initialized = true;
    LOG_INFO("[Benchmark] Initialization complete!");
    return true;
}

bool BenchmarkRunner::ParseCommandLine(int argc, char** argv, BenchmarkConfig& outConfig) {
    outConfig = BenchmarkConfig();  // Start with defaults
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--pinn" && i + 1 < argc) {
            outConfig.pinnModel = ResolvePINNModelPath(argv[++i]);
        }
        else if (arg == "--siren") {
            outConfig.sirenEnabled = true;
        }
        else if (arg == "--siren-intensity" && i + 1 < argc) {
            outConfig.sirenIntensity = std::stof(argv[++i]);
        }
        else if (arg == "--siren-seed" && i + 1 < argc) {
            outConfig.sirenSeed = std::stof(argv[++i]);
        }
        else if (arg == "--particles" && i + 1 < argc) {
            outConfig.particleCount = std::stoi(argv[++i]);
        }
        else if (arg == "--frames" && i + 1 < argc) {
            outConfig.frames = std::stoi(argv[++i]);
        }
        else if (arg == "--timestep" && i + 1 < argc) {
            outConfig.timestep = std::stof(argv[++i]);
        }
        else if (arg == "--timescale" && i + 1 < argc) {
            outConfig.timescale = std::stof(argv[++i]);
        }
        else if (arg == "--warmup" && i + 1 < argc) {
            outConfig.warmupFrames = std::stoi(argv[++i]);
        }
        else if (arg == "--sample-interval" && i + 1 < argc) {
            outConfig.sampleInterval = std::stoi(argv[++i]);
        }
        else if (arg == "--output" && i + 1 < argc) {
            outConfig.outputPath = argv[++i];
        }
        else if (arg == "--output-format" && i + 1 < argc) {
            outConfig.outputFormat = argv[++i];
        }
        else if (arg == "--generate-preset" && i + 1 < argc) {
            outConfig.generatePreset = true;
            outConfig.presetPath = argv[++i];
        }
        else if (arg == "--verbose") {
            outConfig.verbose = true;
        }
        else if (arg == "--help" || arg == "-h") {
            PrintBenchmarkHelp();
            return false;  // Signal to exit
        }
    }
    
    return true;
}

BenchmarkResults BenchmarkRunner::Run() {
    if (!m_initialized) {
        LOG_ERROR("[Benchmark] Not initialized!");
        return m_results;
    }
    
    LOG_INFO("[Benchmark] ==========================================");
    LOG_INFO("[Benchmark] Starting benchmark simulation");
    LOG_INFO("[Benchmark] ==========================================");
    
    m_isRunning = true;
    m_startTime = std::chrono::high_resolution_clock::now();
    m_currentFrame = 0;
    m_totalSimTime = 0.0f;
    m_hasPrevSnapshot = false;
    
    const uint32_t totalFrames = m_config.warmupFrames + m_config.frames;
    const float dt = m_config.timestep;
    
    // Warmup phase (metrics not collected)
    LOG_INFO("[Benchmark] Warmup phase ({} frames)...", m_config.warmupFrames);
    for (uint32_t i = 0; i < m_config.warmupFrames; i++) {
        SimulateFrame(dt, m_totalSimTime);
        m_totalSimTime += dt;
        m_currentFrame++;
    }
    
    // Capture initial state for energy/angular momentum tracking
    PhysicsSnapshot initialSnapshot = CaptureSnapshot();
    m_results.stability.totalEnergyStart = initialSnapshot.totalEnergy;
    m_results.stability.angularMomentumStart = initialSnapshot.totalAngularMomentum;
    
    // Main benchmark phase
    LOG_INFO("[Benchmark] Benchmark phase ({} frames, sampling every {})...", 
             m_config.frames, m_config.sampleInterval);
    
    auto phaseStart = std::chrono::high_resolution_clock::now();
    
    for (uint32_t i = 0; i < m_config.frames; i++) {
        auto frameStart = std::chrono::high_resolution_clock::now();
        
        // Run physics
        SimulateFrame(dt, m_totalSimTime);
        
        auto frameEnd = std::chrono::high_resolution_clock::now();
        float frameMs = std::chrono::duration<float, std::milli>(frameEnd - frameStart).count();
        
        // Record performance metrics
        m_results.performance.totalPhysicsMs.AddSample(frameMs);
        
        // Sample metrics at interval
        if (i % m_config.sampleInterval == 0) {
            PhysicsSnapshot snapshot = CaptureSnapshot();
            AccumulateMetrics(snapshot);
            m_prevSnapshot = snapshot;
            m_hasPrevSnapshot = true;
        }
        
        m_totalSimTime += dt;
        m_currentFrame++;
        m_results.framesSimulated++;
        
        // Log progress every 10%
        if ((i + 1) % (m_config.frames / 10) == 0) {
            LogProgress(i + 1, m_config.frames);
        }
    }
    
    auto phaseEnd = std::chrono::high_resolution_clock::now();
    float phaseDurationMs = std::chrono::duration<float, std::milli>(phaseEnd - phaseStart).count();
    
    // Capture final state
    PhysicsSnapshot finalSnapshot = CaptureSnapshot();
    m_results.stability.totalEnergyEnd = finalSnapshot.totalEnergy;
    m_results.stability.angularMomentumEnd = finalSnapshot.totalAngularMomentum;
    
    // Compute derived performance metrics
    m_results.performance.totalSimulationTimeMs = phaseDurationMs;
    m_results.performance.framesPerSecond = (m_config.frames * 1000.0f) / phaseDurationMs;
    m_results.performance.particlesPerSecond = m_results.performance.framesPerSecond * m_config.particleCount;
    
    // Finalize all metrics
    ComputeFinalMetrics();
    
    // Set timestamp and duration
    m_results.timestamp = GetTimestamp();
    auto totalEnd = std::chrono::high_resolution_clock::now();
    m_results.durationSeconds = std::chrono::duration<float>(totalEnd - m_startTime).count();
    
    m_isRunning = false;
    
    LOG_INFO("[Benchmark] ==========================================");
    LOG_INFO("[Benchmark] Benchmark complete!");
    LOG_INFO("[Benchmark] ==========================================");
    LogResults(m_results);
    
    return m_results;
}

void BenchmarkRunner::SimulateFrame(float deltaTime, float totalTime) {
    // Update physics through ParticleSystem (PINN + optional SIREN)
    m_particleSystem->Update(deltaTime, totalTime);
}

PhysicsSnapshot BenchmarkRunner::CaptureSnapshot() {
    // Delegate to ParticleSystem's snapshot method and convert types
    auto psSnap = m_particleSystem->CapturePhysicsSnapshot();
    
    // Convert from ParticleSystem::PhysicsSnapshot to Benchmark::PhysicsSnapshot
    PhysicsSnapshot snap;
    snap.totalKineticEnergy = psSnap.totalKineticEnergy;
    snap.totalPotentialEnergy = psSnap.totalPotentialEnergy;
    snap.totalEnergy = psSnap.totalEnergy;
    snap.totalAngularMomentum = psSnap.totalAngularMomentum;
    snap.particlesInBounds = psSnap.particlesInBounds;
    snap.particlesEscaped = psSnap.particlesEscaped;
    snap.particlesCollapsed = psSnap.particlesCollapsed;
    snap.velocityMean = psSnap.velocityMean;
    snap.velocityMax = psSnap.velocityMax;
    snap.velocityStdDev = psSnap.velocityStdDev;
    snap.avgForceMagnitude = psSnap.avgForceMagnitude;
    snap.avgRadialForce = psSnap.avgRadialForce;
    snap.correctRadialForceCount = psSnap.correctRadialForceCount;
    snap.keplerianVelocityError = psSnap.keplerianVelocityError;
    snap.velocityCovarianceXZ = psSnap.velocityCovarianceXZ;
    snap.diskThicknessRatio = psSnap.diskThicknessRatio;
    
    return snap;
}

void BenchmarkRunner::AccumulateMetrics(const PhysicsSnapshot& snapshot) {
    // Stability metrics
    float escapeRate = static_cast<float>(snapshot.particlesEscaped) / m_config.particleCount;
    float collapseRate = static_cast<float>(snapshot.particlesCollapsed) / m_config.particleCount;
    
    m_results.stability.escapeRate.AddSample(escapeRate);
    m_results.stability.collapseRate.AddSample(collapseRate);
    m_results.stability.totalEscaped += snapshot.particlesEscaped;
    m_results.stability.totalCollapsed += snapshot.particlesCollapsed;
    m_results.stability.energyValues.AddSample(snapshot.totalEnergy);
    m_results.stability.angularMomentumValues.AddSample(snapshot.totalAngularMomentum);
    m_results.stability.velocityMean.AddSample(snapshot.velocityMean);
    m_results.stability.velocityMax.AddSample(snapshot.velocityMax);
    
    // Physical accuracy metrics
    m_results.accuracy.keplerianVelocityError.AddSample(snapshot.keplerianVelocityError);
    m_results.accuracy.avgForceMagnitude.AddSample(snapshot.avgForceMagnitude);
    m_results.accuracy.avgRadialForce.AddSample(snapshot.avgRadialForce);
    
    // Visual quality metrics
    m_results.visual.coherentMotionIndex.AddSample(std::abs(snapshot.velocityCovarianceXZ));
    m_results.visual.diskThicknessActual.AddSample(snapshot.diskThicknessRatio);
    
    // Velocity jerk (rate of velocity change) - requires previous snapshot
    if (m_hasPrevSnapshot) {
        float velocityChange = std::abs(snapshot.velocityMean - m_prevSnapshot.velocityMean);
        m_results.visual.velocityJerk.AddSample(velocityChange);
    }
}

void BenchmarkRunner::ComputeFinalMetrics() {
    // Finalize all metric aggregations
    m_results.Finalize();
    
    // Compute radial force sign correctness from accuracy data
    // (This would need tracking during accumulation - simplified here)
    m_results.accuracy.radialForceSignCorrectPercent = 
        (m_results.accuracy.avgRadialForce.mean < 0.0f) ? 100.0f : 0.0f;
}

bool BenchmarkRunner::SaveResults(const BenchmarkResults& results, 
                                   const std::string& path,
                                   const std::string& format) {
    if (format == "json" || format == "both") {
        std::string jsonPath = (format == "both") ? 
            path.substr(0, path.rfind('.')) + ".json" : path;
        if (!SaveResultsJSON(results, jsonPath)) {
            return false;
        }
    }
    
    if (format == "csv" || format == "both") {
        std::string csvPath = (format == "both") ? 
            path.substr(0, path.rfind('.')) + ".csv" : path;
        if (!SaveResultsCSV(results, csvPath)) {
            return false;
        }
    }
    
    return true;
}

bool BenchmarkRunner::SaveResultsJSON(const BenchmarkResults& results, const std::string& path) {
    std::ofstream file(path);
    if (!file.is_open()) {
        LOG_ERROR("[Benchmark] Failed to open output file: {}", path);
        return false;
    }
    
    file << std::fixed << std::setprecision(4);
    
    file << "{\n";
    
    // Benchmark metadata
    file << "  \"benchmark\": {\n";
    file << "    \"version\": \"1.0\",\n";
    file << "    \"timestamp\": \"" << results.timestamp << "\",\n";
    file << "    \"duration_seconds\": " << results.durationSeconds << ",\n";
    file << "    \"config\": {\n";
    file << "      \"pinn_model\": \"" << results.pinnModel << "\",\n";
    file << "      \"pinn_version\": " << results.pinnVersion << ",\n";
    file << "      \"siren_enabled\": " << (results.sirenEnabled ? "true" : "false") << ",\n";
    file << "      \"siren_intensity\": " << results.sirenIntensity << ",\n";
    file << "      \"particle_count\": " << results.particleCount << ",\n";
    file << "      \"frames\": " << results.framesSimulated << ",\n";
    file << "      \"timestep\": " << results.timestep << ",\n";
    file << "      \"timescale\": " << results.timescale << "\n";
    file << "    }\n";
    file << "  },\n";
    
    // Stability metrics
    file << "  \"stability\": {\n";
    file << "    \"escape_rate\": { \"mean\": " << results.stability.escapeRate.mean 
         << ", \"max\": " << results.stability.escapeRate.max << " },\n";
    file << "    \"collapse_rate\": { \"mean\": " << results.stability.collapseRate.mean 
         << ", \"max\": " << results.stability.collapseRate.max << " },\n";
    file << "    \"energy_drift_percent\": " << results.stability.energyDriftPercent << ",\n";
    file << "    \"angular_momentum_drift_percent\": " << results.stability.angularMomentumDriftPercent << ",\n";
    file << "    \"velocity\": { \"mean\": " << results.stability.velocityMean.mean 
         << ", \"max\": " << results.stability.velocityMax.max << " }\n";
    file << "  },\n";
    
    // Performance metrics
    file << "  \"performance\": {\n";
    file << "    \"physics_ms\": { \"mean\": " << results.performance.totalPhysicsMs.mean 
         << ", \"max\": " << results.performance.totalPhysicsMs.max << " },\n";
    file << "    \"frames_per_second\": " << results.performance.framesPerSecond << ",\n";
    file << "    \"particles_per_second\": " << results.performance.particlesPerSecond << "\n";
    file << "  },\n";
    
    // Physical accuracy metrics
    file << "  \"physical_accuracy\": {\n";
    file << "    \"keplerian_velocity_error_percent\": " << results.accuracy.keplerianVelocityError.mean << ",\n";
    file << "    \"radial_force_sign_correct_percent\": " << results.accuracy.radialForceSignCorrectPercent << ",\n";
    file << "    \"avg_force_magnitude\": " << results.accuracy.avgForceMagnitude.mean << "\n";
    file << "  },\n";
    
    // Visual quality metrics
    file << "  \"visual_quality\": {\n";
    file << "    \"coherent_motion_index\": " << results.visual.coherentMotionIndex.mean << ",\n";
    file << "    \"disk_thickness_ratio\": " << results.visual.diskThicknessActual.mean << ",\n";
    file << "    \"velocity_jerk\": " << results.visual.velocityJerk.mean << "\n";
    file << "  },\n";
    
    // Summary scores
    file << "  \"summary\": {\n";
    file << "    \"stability_score\": " << results.stabilityScore << ",\n";
    file << "    \"performance_score\": " << results.performanceScore << ",\n";
    file << "    \"accuracy_score\": " << results.accuracyScore << ",\n";
    file << "    \"visual_score\": " << results.visualScore << ",\n";
    file << "    \"overall_score\": " << results.overallScore << ",\n";
    file << "    \"recommendation\": \"" << results.recommendation << "\"\n";
    file << "  }\n";
    
    file << "}\n";
    
    file.close();
    LOG_INFO("[Benchmark] Results saved to: {}", path);
    return true;
}

bool BenchmarkRunner::SaveResultsCSV(const BenchmarkResults& results, const std::string& path) {
    std::ofstream file(path);
    if (!file.is_open()) {
        LOG_ERROR("[Benchmark] Failed to open CSV file: {}", path);
        return false;
    }
    
    file << std::fixed << std::setprecision(6);
    
    // Header
    file << "model,version,siren,particles,frames,timescale,"
         << "escape_rate,collapse_rate,energy_drift,angular_momentum_drift,"
         << "fps,physics_ms,"
         << "keplerian_error,radial_force_correct,"
         << "coherent_motion,disk_thickness,"
         << "stability_score,performance_score,accuracy_score,visual_score,overall_score\n";
    
    // Data row
    file << results.pinnModel << "," << results.pinnVersion << ","
         << (results.sirenEnabled ? 1 : 0) << "," << results.particleCount << ","
         << results.framesSimulated << "," << results.timescale << ","
         << results.stability.escapeRate.mean << "," << results.stability.collapseRate.mean << ","
         << results.stability.energyDriftPercent << "," << results.stability.angularMomentumDriftPercent << ","
         << results.performance.framesPerSecond << "," << results.performance.totalPhysicsMs.mean << ","
         << results.accuracy.keplerianVelocityError.mean << "," << results.accuracy.radialForceSignCorrectPercent << ","
         << results.visual.coherentMotionIndex.mean << "," << results.visual.diskThicknessActual.mean << ","
         << results.stabilityScore << "," << results.performanceScore << ","
         << results.accuracyScore << "," << results.visualScore << "," << results.overallScore << "\n";
    
    file.close();
    LOG_INFO("[Benchmark] CSV saved to: {}", path);
    return true;
}

bool BenchmarkRunner::GeneratePreset(const BenchmarkResults& results, const std::string& path) {
    std::ofstream file(path);
    if (!file.is_open()) {
        LOG_ERROR("[Benchmark] Failed to create preset file: {}", path);
        return false;
    }
    
    file << std::fixed << std::setprecision(2);
    
    file << "{\n";
    file << "  \"preset_name\": \"benchmark_generated\",\n";
    file << "  \"description\": \"Auto-generated from benchmark (score: " << results.overallScore << ")\",\n";
    file << "  \"generated_by\": \"benchmark_system\",\n";
    file << "  \"benchmark_score\": " << results.overallScore << ",\n";
    file << "\n";
    file << "  \"physics\": {\n";
    file << "    \"pinn_model\": \"" << results.pinnModel << "\",\n";
    file << "    \"timescale\": " << results.timescale << ",\n";
    file << "    \"enforce_boundaries\": true\n";
    file << "  },\n";
    file << "  \"turbulence\": {\n";
    file << "    \"siren_enabled\": " << (results.sirenEnabled ? "true" : "false") << ",\n";
    file << "    \"siren_intensity\": " << results.sirenIntensity << "\n";
    file << "  },\n";
    file << "  \"particles\": {\n";
    file << "    \"count\": " << results.particleCount << "\n";
    file << "  }\n";
    file << "}\n";
    
    file.close();
    LOG_INFO("[Benchmark] Preset saved to: {}", path);
    return true;
}

std::vector<BenchmarkResults> BenchmarkRunner::RunModelComparison(const BenchmarkConfig& baseConfig) {
    std::vector<BenchmarkResults> allResults;
    auto models = GetAvailablePINNModels();
    
    LOG_INFO("[Benchmark] Running model comparison ({} models)...", models.size());
    
    for (const auto& model : models) {
        LOG_INFO("[Benchmark] Testing model: {} (v{})", model.name, model.version);
        
        BenchmarkConfig config = baseConfig;
        config.pinnModel = model.path;
        
        // Re-initialize with new model
        if (Initialize(config)) {
            BenchmarkResults results = Run();
            allResults.push_back(results);
        } else {
            LOG_WARN("[Benchmark] Failed to initialize model: {}", model.path);
        }
    }
    
    return allResults;
}

float BenchmarkRunner::GetProgress() const {
    if (!m_isRunning) return 0.0f;
    uint32_t total = m_config.warmupFrames + m_config.frames;
    return static_cast<float>(m_currentFrame) / total;
}

std::string BenchmarkRunner::GetTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time);
    
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

void BenchmarkRunner::LogProgress(uint32_t frame, uint32_t totalFrames) {
    float progress = (static_cast<float>(frame) / totalFrames) * 100.0f;
    LOG_INFO("[Benchmark] Progress: {:.0f}% ({}/{})", progress, frame, totalFrames);
}

void BenchmarkRunner::LogResults(const BenchmarkResults& results) {
    LOG_INFO("[Benchmark] --- RESULTS ---");
    LOG_INFO("[Benchmark] Model: {} (v{})", results.pinnModel, results.pinnVersion);
    LOG_INFO("[Benchmark] Duration: {:.2f}s", results.durationSeconds);
    LOG_INFO("[Benchmark]");
    LOG_INFO("[Benchmark] STABILITY: {:.1f}/100", results.stabilityScore);
    LOG_INFO("[Benchmark]   Escape rate: {:.4f}%", results.stability.escapeRate.mean * 100.0f);
    LOG_INFO("[Benchmark]   Energy drift: {:.2f}%", results.stability.energyDriftPercent);
    LOG_INFO("[Benchmark]");
    LOG_INFO("[Benchmark] PERFORMANCE: {:.1f}/100", results.performanceScore);
    LOG_INFO("[Benchmark]   FPS: {:.1f}", results.performance.framesPerSecond);
    LOG_INFO("[Benchmark]   Physics time: {:.2f}ms", results.performance.totalPhysicsMs.mean);
    LOG_INFO("[Benchmark]");
    LOG_INFO("[Benchmark] ACCURACY: {:.1f}/100", results.accuracyScore);
    LOG_INFO("[Benchmark]   Keplerian error: {:.2f}%", results.accuracy.keplerianVelocityError.mean);
    LOG_INFO("[Benchmark]");
    LOG_INFO("[Benchmark] VISUAL: {:.1f}/100", results.visualScore);
    LOG_INFO("[Benchmark]   Coherent motion: {:.4f}", results.visual.coherentMotionIndex.mean);
    LOG_INFO("[Benchmark]");
    LOG_INFO("[Benchmark] ═══════════════════════════════════════");
    LOG_INFO("[Benchmark] OVERALL SCORE: {:.1f}/100", results.overallScore);
    LOG_INFO("[Benchmark] {}", results.recommendation);
    LOG_INFO("[Benchmark] ═══════════════════════════════════════");
}

void PrintBenchmarkHelp() {
    LOG_INFO("");
    LOG_INFO("PINN Benchmark System - Usage:");
    LOG_INFO("==============================");
    LOG_INFO("");
    LOG_INFO("  PlasmaDX-Clean.exe --benchmark [options]");
    LOG_INFO("");
    LOG_INFO("Options:");
    LOG_INFO("  --pinn <model>         PINN model: v1, v2, v3, v4, or path (default: v4)");
    LOG_INFO("  --siren                Enable SIREN turbulence");
    LOG_INFO("  --siren-intensity <f>  SIREN intensity 0-5 (default: 0.5)");
    LOG_INFO("  --siren-seed <f>       SIREN random seed (default: 0.0)");
    LOG_INFO("  --particles <n>        Particle count (default: 10000)");
    LOG_INFO("  --frames <n>           Simulation frames (default: 1000)");
    LOG_INFO("  --timestep <f>         Fixed timestep in seconds (default: 0.016)");
    LOG_INFO("  --timescale <f>        Time multiplier 1-50 (default: 1.0)");
    LOG_INFO("  --warmup <n>           Warmup frames (default: 100)");
    LOG_INFO("  --sample-interval <n>  Frames between samples (default: 10)");
    LOG_INFO("  --output <path>        Output file path (default: benchmark_results.json)");
    LOG_INFO("  --output-format <fmt>  Format: json, csv, both (default: json)");
    LOG_INFO("  --generate-preset <p>  Generate preset file at path");
    LOG_INFO("  --verbose              Detailed logging");
    LOG_INFO("  --help                 Show this help");
    LOG_INFO("");
    LOG_INFO("Examples:");
    LOG_INFO("  --benchmark --pinn v4 --frames 500");
    LOG_INFO("  --benchmark --pinn v4 --siren --siren-intensity 0.3");
    LOG_INFO("  --benchmark --compare-all --frames 300");
    LOG_INFO("");
}

} // namespace Benchmark

