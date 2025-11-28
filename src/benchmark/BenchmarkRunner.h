#pragma once

#include "BenchmarkConfig.h"
#include "BenchmarkMetrics.h"
#include <memory>
#include <string>
#include <chrono>

// Forward declarations
class ParticleSystem;
class Device;
class ResourceManager;

namespace Benchmark {

/**
 * BenchmarkRunner - Headless PINN model evaluation system
 * 
 * Runs physics simulations without GPU rendering to measure:
 * - Stability (orbit maintenance, energy conservation)
 * - Performance (inference times, throughput)
 * - Physical accuracy (Keplerian velocity, force magnitudes)
 * - Visual quality proxies (coherent motion, disk structure)
 * 
 * Usage:
 *   BenchmarkRunner runner;
 *   runner.Initialize(config);
 *   BenchmarkResults results = runner.Run();
 *   runner.SaveResults(results, "output.json");
 */
class BenchmarkRunner {
public:
    BenchmarkRunner();
    ~BenchmarkRunner();
    
    // Prevent copying (owns resources)
    BenchmarkRunner(const BenchmarkRunner&) = delete;
    BenchmarkRunner& operator=(const BenchmarkRunner&) = delete;
    
    /**
     * Initialize the benchmark runner with configuration.
     * Creates minimal resources needed (ParticleSystem only, no GPU rendering).
     * 
     * @param config Benchmark configuration
     * @return true if initialization successful
     */
    bool Initialize(const BenchmarkConfig& config);
    
    /**
     * Parse command-line arguments into config.
     * 
     * @param argc Argument count
     * @param argv Argument values
     * @param outConfig Output configuration
     * @return true if parsing successful
     */
    static bool ParseCommandLine(int argc, char** argv, BenchmarkConfig& outConfig);
    
    /**
     * Run the benchmark simulation.
     * Executes physics simulation for configured number of frames,
     * collecting metrics at each sample interval.
     * 
     * @return Complete benchmark results
     */
    BenchmarkResults Run();
    
    /**
     * Save results to file (JSON or CSV).
     * 
     * @param results Benchmark results
     * @param path Output file path
     * @param format Output format ("json" or "csv")
     * @return true if save successful
     */
    bool SaveResults(const BenchmarkResults& results, 
                     const std::string& path,
                     const std::string& format = "json");
    
    /**
     * Generate a preset file from results.
     * Creates a JSON preset that can be loaded by the main application.
     * 
     * @param results Benchmark results
     * @param path Output preset file path
     * @return true if generation successful
     */
    bool GeneratePreset(const BenchmarkResults& results, const std::string& path);
    
    /**
     * Run comparison benchmark across all available PINN models.
     * 
     * @param baseConfig Base configuration (model will be overridden)
     * @return Vector of results, one per model
     */
    std::vector<BenchmarkResults> RunModelComparison(const BenchmarkConfig& baseConfig);
    
    /**
     * Get current progress (for logging).
     * @return Progress as fraction (0.0 - 1.0)
     */
    float GetProgress() const;
    
    /**
     * Check if benchmark is currently running.
     */
    bool IsRunning() const { return m_isRunning; }
    
private:
    // Simulation step (one physics frame)
    void SimulateFrame(float deltaTime, float totalTime);
    
    // Metric collection
    PhysicsSnapshot CaptureSnapshot();
    void AccumulateMetrics(const PhysicsSnapshot& snapshot);
    
    // Score computation
    void ComputeFinalMetrics();
    
    // Output helpers
    bool SaveResultsJSON(const BenchmarkResults& results, const std::string& path);
    bool SaveResultsCSV(const BenchmarkResults& results, const std::string& path);
    std::string GetTimestamp() const;
    
    // Logging
    void LogProgress(uint32_t frame, uint32_t totalFrames);
    void LogResults(const BenchmarkResults& results);
    
private:
    // Configuration
    BenchmarkConfig m_config;
    bool m_initialized = false;
    bool m_isRunning = false;
    
    // Minimal system (no GPU)
    std::unique_ptr<Device> m_device;
    std::unique_ptr<ResourceManager> m_resources;
    std::unique_ptr<ParticleSystem> m_particleSystem;
    
    // Timing
    std::chrono::high_resolution_clock::time_point m_startTime;
    float m_totalSimTime = 0.0f;
    
    // Metrics accumulator
    BenchmarkResults m_results;
    uint32_t m_currentFrame = 0;
    
    // Previous snapshot for derivative metrics
    PhysicsSnapshot m_prevSnapshot;
    bool m_hasPrevSnapshot = false;
};

/**
 * Print benchmark help to console.
 */
void PrintBenchmarkHelp();

} // namespace Benchmark

