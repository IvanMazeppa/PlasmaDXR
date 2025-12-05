#pragma once

#include <string>
#include <vector>
#include <optional>

// Configuration system for PlasmaDX-Clean
// Supports multiple profiles: dev, user, pix_analysis
//
// Priority order:
// 1. Command line: --config=config_user.json
// 2. Environment variable: PLASMADX_CONFIG=config_pix_analysis.json
// 3. Default: config_dev.json (if exists)
// 4. Fallback: Hardcoded defaults

namespace Config {

    enum class Profile {
        Dev,           // Developer debug mode - fast iteration
        User,          // User experience mode - visual quality
        PIXAnalysis,   // PIX analysis mode - maximum diagnostics
        Custom         // Custom JSON file
    };

    enum class RendererType {
        Billboard,
        Gaussian
    };

    enum class LogLevel {
        Info,
        Debug,
        Trace
    };

    struct RenderingConfig {
        uint32_t particleCount = 10000;
        RendererType rendererType = RendererType::Gaussian;
        uint32_t resolutionWidth = 1920;
        uint32_t resolutionHeight = 1080;
    };

    struct FeaturesConfig {
        bool enableReSTIR = false;
        uint32_t restirCandidates = 16;
        bool restirTemporalReuse = true;
        bool restirSpatialReuse = true;
        float restirTemporalWeight = 0.9f;

        bool enableInScattering = false;
        float inScatterStrength = 1.0f;

        bool enableShadowRays = true;
        bool enablePhaseFunction = true;
        float phaseStrength = 5.0f;

        bool useAnisotropicGaussians = true;
        float anisotropyStrength = 1.0f;

        float rtLightingStrength = 2.0f;

        bool usePhysicalEmission = false;
        float emissionStrength = 1.0f;

        bool useDopplerShift = false;
        float dopplerStrength = 1.0f;

        bool useGravitationalRedshift = false;
        float redshiftStrength = 1.0f;
    };

    struct LightingConfig {
        // Primary lighting system (mutually exclusive)
        std::string system = "MultiLight";  // "MultiLight", "RTXDI", or "VolumetricReSTIR"

        // Multi-light settings
        bool multiLightEnabled = true;
        uint32_t lightCount = 16;
        std::string multiLightPreset = "stellar_ring";
        float multiLightIntensity = 1.0f;

        // Probe grid settings (additive/complementary system)
        bool probeGridEnabled = false;
        uint32_t probeGridSize = 32;
        uint32_t raysPerProbe = 16;
        float probeGridIntensity = 800.0f;
        uint32_t probeUpdateInterval = 4;

        // RTXDI settings
        bool rtxdiEnabled = false;
        std::string rtxdiMode = "M5";
        float rtxdiTemporalWeight = 0.9f;
    };

    struct PhysicsConfig {
        // Basic geometry
        float innerRadius = 50.0f;
        float outerRadius = 1000.0f;
        float diskThickness = 100.0f;
        float timeStep = 1.0f / 120.0f;  // 120Hz physics
        bool physicsEnabled = true;

        // GA-optimized physics parameters (Phase 5)
        float gm = 100.0f;                   // Gravitational parameter (50-200)
        float bh_mass = 5.0f;                // Black hole mass in solar masses (0.1-10)
        float alpha = 0.1f;                  // Shakura-Sunyaev alpha viscosity (0.01-0.5)
        float damping = 0.98f;               // Velocity damping factor (0.95-1.0)
        float angular_boost = 1.5f;          // Angular momentum boost (0.8-2.0)
        float density_scale = 2.0f;          // Particle density scaling (0.5-3.0)
        float force_clamp = 500.0f;          // Maximum force magnitude (100-1000)
        float velocity_clamp = 200.0f;       // Maximum velocity magnitude (100-500)
        int boundary_mode = 0;               // Boundary mode: 0=none, 1=reflect
        float time_multiplier = 1.0f;        // Physics time multiplier (1.0-5.0)
    };

    struct SIRENConfig {
        bool enabled = false;                // Enable SIREN vortex turbulence
        float intensity = 0.0f;              // Turbulence intensity (0-1)
        float vortex_scale = 1.0f;           // Vortex eddy size (0.5-3)
        float vortex_decay = 0.1f;           // Vortex temporal decay (0.01-0.5)
    };

    struct PINNConfig {
        bool enabled = false;                // Enable PINN physics
        std::string model = "pinn_v3_total_forces";  // PINN model to use (v3 or v4)
        bool enforce_boundaries = false;     // Enforce particle boundaries
    };

    struct CameraConfig {
        float startDistance = 800.0f;
        float startHeight = 1200.0f;
        float startAngle = 0.0f;
        float startPitch = 0.0f;
        float moveSpeed = 100.0f;
        float rotateSpeed = 0.5f;
        float particleSize = 50.0f;
    };

    struct DebugConfig {
        bool enableDebugLayer = false;
        LogLevel logLevel = LogLevel::Info;
        bool enablePIX = false;
        bool pixAutoCapture = false;
        uint32_t pixCaptureFrame = 120;
        bool showFPS = true;
        bool showParticleStats = true;
    };

    struct PIXAnalysisConfig {
        std::vector<uint32_t> captureFrames = {1, 60, 120, 300};
        std::string capturePrefix = "analysis_";
        bool enableReservoirLogging = false;
        bool enablePerformanceCounters = false;
        bool trackResourceUsage = false;
    };

    // Main configuration structure
    struct AppConfig {
        Profile profile = Profile::Dev;
        std::string profileName = "dev";

        RenderingConfig rendering;
        FeaturesConfig features;
        LightingConfig lighting;
        PhysicsConfig physics;
        SIRENConfig siren;
        PINNConfig pinn;
        CameraConfig camera;
        DebugConfig debug;
        PIXAnalysisConfig pixAnalysis;

        // Metadata
        std::string configFilePath;
        bool loadedFromFile = false;
    };

    // Configuration loader/manager
    class ConfigManager {
    public:
        ConfigManager();
        ~ConfigManager();

        // Load configuration from file
        bool LoadFromFile(const std::string& filepath);

        // Load configuration by profile name
        bool LoadProfile(Profile profile);

        // Parse command-line arguments and load config
        bool Initialize(int argc, char** argv);

        // Get current configuration
        const AppConfig& GetConfig() const { return m_config; }

        // Get mutable config (for runtime changes)
        AppConfig& GetMutableConfig() { return m_config; }

        // Save current config to file
        bool SaveToFile(const std::string& filepath) const;

        // Generate default config files
        static bool GenerateDefaultConfigs();

    private:
        // Parse JSON from file
        bool ParseJSON(const std::string& filepath);

        // Set defaults based on profile
        void SetProfileDefaults(Profile profile);

        // Get default config file path for profile
        static std::string GetProfilePath(Profile profile);

        AppConfig m_config;
    };

} // namespace Config