#pragma once

#include <windows.h>
#include <wrl/client.h>
#include <d3d12.h>
#include <memory>
#include <chrono>
#include <string>
#include <vector>

// Forward declarations
struct ID3D12Resource;
struct ID3D12DescriptorHeap;
class Device;
class SwapChain;
class FeatureDetector;
class ParticleSystem;
class ParticleRenderer;
class RTLightingSystem_RayQuery;
    class RTXDILightingSystem;
    class ProbeGridSystem;
    class ResourceManager;
class AdaptiveQualitySystem;
#ifdef ENABLE_DLSS
class DLSSSystem;
#endif

// Benchmark system
namespace Benchmark {
    class BenchmarkRunner;
    struct BenchmarkConfig;
}

// Need full include for ParticleRenderer_Gaussian::Light nested type
#include "../particles/ParticleRenderer_Gaussian.h"

class Application {
public:
    Application();
    ~Application();

    // Lifecycle
    bool Initialize(HINSTANCE hInstance, int nCmdShow, int argc = 0, char** argv = nullptr);
    int Run();
    void Shutdown();
    
    // Benchmark mode (headless)
    bool IsBenchmarkMode() const { return m_benchmarkMode; }

private:
    // Window management
    bool CreateAppWindow(HINSTANCE hInstance, int nCmdShow);
    void ToggleFullscreen();
    static LRESULT CALLBACK WindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
    
    // Benchmark mode (headless simulation)
    bool InitializeBenchmarkMode(int argc, char** argv);
    int RunBenchmark();

    // Frame timing
    void UpdateFrameStats(float actualFrameTime);

    // Core update/render
    void Update(float deltaTime);
    void Render();

    // Input handling
    void OnKeyPress(UINT8 key);
    void OnMouseMove(int dx, int dy);

private:
    // Window
    HWND m_hwnd = nullptr;
    bool m_isRunning = false;
    int m_width = 1920;
    int m_height = 1080;
    bool m_isFullscreen = false;
    RECT m_windowedRect = {};  // Store windowed position/size for fullscreen toggle
    
    // Benchmark mode (headless - no window/rendering)
    bool m_benchmarkMode = false;
    std::unique_ptr<Benchmark::BenchmarkRunner> m_benchmarkRunner;

    // Core systems (dependency injection - no god object!)
    std::unique_ptr<Device> m_device;
    std::unique_ptr<SwapChain> m_swapChain;
    std::unique_ptr<FeatureDetector> m_features;
    std::unique_ptr<ResourceManager> m_resources;

    // Subsystems
    std::unique_ptr<ParticleSystem> m_particleSystem;
    std::unique_ptr<ParticleRenderer> m_particleRenderer;           // Billboard renderer (stable)
    std::unique_ptr<ParticleRenderer_Gaussian> m_gaussianRenderer;  // Gaussian Splatting (optional)
    std::unique_ptr<RTLightingSystem_RayQuery> m_rtLighting;
    std::unique_ptr<RTXDILightingSystem> m_rtxdiLightingSystem;     // RTXDI parallel lighting path
    std::unique_ptr<ProbeGridSystem> m_probeGridSystem;             // Probe Grid (Phase 0.13.1 - replaces ReSTIR)
    std::unique_ptr<AdaptiveQualitySystem> m_adaptiveQuality;       // ML-based adaptive quality
#ifdef ENABLE_DLSS
    std::unique_ptr<DLSSSystem> m_dlssSystem;                       // DLSS 4.0 Ray Reconstruction (AI denoising)
#endif

    // Timing
    std::chrono::high_resolution_clock::time_point m_lastFrameTime;
    float m_deltaTime = 0.0f;
    float m_totalTime = 0.0f;

    // Stats
    uint32_t m_frameCount = 0;
    float m_fps = 0.0f;

    // Configuration
    enum class RendererType {
        Billboard,      // Traditional billboard particles (current/stable)
        Gaussian        // 3D Gaussian Splatting (volumetric)
    };

    enum class LightingSystem {
        MultiLight,        // Multi-light brute force (Phase 3.5, good for <20 lights)
        RTXDI              // NVIDIA RTXDI with ReSTIR (Phase 4, scales to 100+ lights)
    };

    struct Config {
        uint32_t particleCount = 10000;        // Default: 10K particles (was 100K)
        bool enableRT = true;
        bool preferMeshShaders = true;
        bool enableDebugLayer = false;
        RendererType rendererType = RendererType::Gaussian;  // Default: Gaussian (was Billboard)
    } m_config;

    // Runtime camera controls - TOP-DOWN VIEW of accretion disk
    float m_cameraDistance = 800.0f;   // Distance from center (horizontal)
    float m_cameraHeight = 1200.0f;    // HIGH above disk to see ring structure from above
    float m_cameraAngle = 0.0f;         // Orbit angle
    float m_cameraPitch = 0.0f;         // Vertical rotation
    float m_particleSize = 50.0f;       // Larger for initial visibility
    float m_cameraMoveSpeed = 100.0f;   // Camera movement speed
    float m_cameraRotateSpeed = 0.5f;   // Camera rotation speed
    bool m_physicsEnabled = true;       // ENABLED - physics shader initializes particles on GPU
    float m_physicsTimeMultiplier = 1.0f; // Physics deltaTime multiplier (1-200) for accelerated orbit settling

    // RTXDI temporal accumulation state
    DirectX::XMFLOAT4X4 m_prevViewProj; // For planar reprojection
    bool m_firstFrame = true;           // To initialize prevViewProj on first run

    // Physics system parameters (readonly for now - would require ParticleSystem API changes)
    float m_innerRadius = 10.0f;        // Inner accretion disk radius
    float m_outerRadius = 300.0f;       // Outer accretion disk radius
    float m_diskThickness = 50.0f;      // Disk thickness
    float m_physicsTimeStep = 0.008333f; // Physics timestep (fixed at 120Hz)

    // RT Lighting runtime controls
    bool m_enableRTLighting = true;  // Toggle for particle-to-particle RT lighting
    float m_rtLightingIntensity = 1.0f;
    float m_rtMaxDistance = 100.0f;  // Updated to match shader (max 400 via ImGui)
    float m_rtParticleRadius = 5.0f;   // Updated to match shader
    float m_rtMinAmbient = 0.0f;     // Global ambient term (DISABLED for probe grid focus)

    // === Adaptive Particle Radius System (Phase 1.5) ===
    bool m_enableAdaptiveRadius = true;      // Toggle for density/distance-based radius scaling
    float m_adaptiveInnerZone = 100.0f;      // Distance threshold for inner shrinking (0-200 units)
    float m_adaptiveOuterZone = 300.0f;      // Distance threshold for outer expansion (200-600 units)
    float m_adaptiveInnerScale = 0.5f;       // Min scale for inner dense regions (0.1-1.0)
    float m_adaptiveOuterScale = 2.0f;       // Max scale for outer sparse regions (1.0-3.0)
    float m_densityScaleMin = 0.3f;          // Min density scale clamp (0.1-1.0)
    float m_densityScaleMax = 3.0f;          // Max density scale clamp (1.0-5.0)

    // === Spatial RT Interpolation (Phase 3.9) ===
    bool m_useVolumetricRT = false;          // Enable spatial interpolation of RT lighting (DISABLED - interferes with probe grid)
    uint32_t m_volumetricRTSamples = 8;      // Number of neighbor particles to sample (4-32)
    float m_volumetricRTDistance = 200.0f;   // Smoothness radius for interpolation (100-400)
    float m_volumetricRTAttenuation = 0.0001f; // Unused (kept for compatibility)
    float m_volumetricRTIntensity = 200.0f;  // Unused (kept for compatibility)

    // === Hybrid Probe Grid System (Phase 0.13.1) ===
    uint32_t m_useProbeGrid = 1u;            // Toggle probe grid lighting (1=enabled by default for diagnostics)

    // === Dynamic Emission (RT-Driven Star Radiance) - DISABLED ===
    float m_rtEmissionStrength = 0.0f;      // Global emission multiplier (DISABLED - not working as intended)
    float m_rtEmissionThreshold = 22000.0f;  // Temperature cutoff for emission (K)
    float m_rtEmissionSuppression = 0.7f;    // How much RT lighting suppresses emission (0.0-1.0)
    float m_rtEmissionTemporalRate = 0.03f;  // Temporal modulation frequency (0.0-0.1)

    // === God Ray System (Phase 5 Milestone 5.3c) ===
    // DEPRECATED: Removed in favor of Gaussian Volume
    float m_godRayDensity = 0.0f;          // Global god ray density (0.0-1.0, 0=disabled)
    float m_godRayStepMultiplier = 1.0f;   // Ray march step multiplier (0.5-2.0, quality vs speed)

    // Enhancement toggles and strengths (DISABLED - focusing on RT lighting)
    bool m_usePhysicalEmission = false;
    float m_emissionStrength = 0.0f;       // 0.0-5.0 (DISABLED)
    float m_emissionBlendFactor = 0.0f;    // 0.0-1.0 (0=artistic, 1=physical) (DISABLED)

    // Multi-light system (Phase 3.5)
    std::vector<ParticleRenderer_Gaussian::Light> m_lights;  // Active lights (max 16)
    void InitializeLights();  // Create default 13-light configuration (legacy, multi-light optimized)

    // RTXDI-optimized light presets (wide spatial distribution for grid-based sampling)
    void InitializeRTXDISphereLights();  // Fibonacci sphere (13 lights, 1200-unit radius)
    void InitializeRTXDIRingLights();    // Dual-ring disk (16 lights, 600-1000 unit radii)
    void InitializeRTXDIGridLights();    // 3×3×3 cubic grid (27 lights, 600-unit spacing)
    void InitializeRTXDISparseLights();  // Minimal debug preset (5 lights, cross pattern)

    int m_selectedLightIndex = -1;  // For ImGui light selection

    // Physics-driven lights system (celestial body lights)
    bool m_physicsDrivenLights = false;        // Toggle: lights move with physics like celestial bodies
    std::vector<uint32_t> m_lightParticleIndices;  // Particle index for each light (empty = random selection)

    // Stellar temperature color system (Phase 2)
    bool m_useStellarTemperatureColors = false;  // Toggle: auto-apply stellar temperature colors based on intensity
    DirectX::XMFLOAT3 GetStellarColorFromTemperature(float temperature);
    float GetStellarTemperatureFromIntensity(float intensity);  // Maps intensity (0.1-20.0) → temperature (3000K-30000K)

    // === Bulk Light Color Control System (Phase 5 Milestone 5.3b) ===
    enum class ColorPreset {
        Custom,
        CoolBlue,
        White,
        WarmWhite,
        WarmSunset,
        DeepRed,
        Rainbow,
        Complementary,
        MonochromeBlue,
        MonochromeRed,
        MonochromeGreen,
        Neon,
        Pastel,
        StellarNursery,
        RedGiant,
        AccretionDisk,
        BinarySystem,
        DustTorus
    };

    enum class GradientType {
        Radial,       // Distance from center
        LinearX,      // Position along X
        LinearY,      // Position along Y
        LinearZ,      // Position along Z
        Circular      // Angle around Y-axis
    };

    enum class LightSelection {
        All,
        InnerRing,
        OuterRing,
        TopHalf,
        BottomHalf,
        EvenIndices,
        OddIndices,
        CustomRange
    };

    // Bulk color control state
    ColorPreset m_currentColorPreset = ColorPreset::Custom;
    LightSelection m_lightSelection = LightSelection::All;
    int m_customRangeStart = 0;
    int m_customRangeEnd = 15;
    float m_radialThreshold = 800.0f;  // Distance threshold for inner/outer ring

    // Gradient application state
    GradientType m_gradientType = GradientType::Radial;
    DirectX::XMFLOAT3 m_gradientColorStart = {1.0f, 1.0f, 1.0f};
    DirectX::XMFLOAT3 m_gradientColorEnd = {1.0f, 0.0f, 0.0f};

    // Global color operations
    float m_hueShift = 0.0f;            // -180 to +180 degrees
    float m_saturationAdjust = 1.0f;    // 0.0 to 2.0 (multiplier)
    float m_valueAdjust = 1.0f;         // 0.0 to 2.0 (multiplier)
    float m_temperatureShift = 0.0f;    // -1.0 (cooler) to +1.0 (warmer)

    // Helper functions for bulk color control
    void ApplyColorPreset(ColorPreset preset);
    void ApplyGradient(GradientType type, DirectX::XMFLOAT3 startColor, DirectX::XMFLOAT3 endColor);
    void ApplyGlobalHueShift(float degrees);
    void ApplyGlobalSaturationAdjust(float multiplier);
    void ApplyGlobalValueAdjust(float multiplier);
    void ApplyTemperatureShift(float amount);
    std::vector<int> GetSelectedLightIndices();
    DirectX::XMFLOAT3 RGBtoHSV(DirectX::XMFLOAT3 rgb);
    DirectX::XMFLOAT3 HSVtoRGB(DirectX::XMFLOAT3 hsv);
    DirectX::XMFLOAT3 BlackbodyColor(float temperature);

    // Particle count control
    uint32_t m_activeParticleCount = 10000;  // Runtime-adjustable particle count

    bool m_useDopplerShift = false;
    float m_dopplerStrength = 1.0f;        // 0.0-5.0 (multiplier)

    bool m_useGravitationalRedshift = false;
    float m_redshiftStrength = 1.0f;       // 0.0-5.0 (multiplier)

    // Gaussian RT system toggles
    bool m_useShadowRays = false;          // F5 to toggle (PCSS - disabled by default, replaced by screen-space)
    bool m_useScreenSpaceShadows = true;   // Phase 2: Screen-space contact shadows (NEW!)
    uint32_t m_ssSteps = 16;               // Screen-space shadow quality: 8=fast, 16=balanced, 32=quality
    bool m_debugScreenSpaceShadows = false; // Debug visualization: red=shadow, green=lit
    bool m_useInScattering = false;        // F6 to toggle (OFF by default - very expensive!)
    LightingSystem m_lightingSystem = LightingSystem::MultiLight;  // --multi-light (default) or --rtxdi
    bool m_debugRTXDISelection = false;    // DEBUG: Visualize RTXDI light selection (rainbow colors)
    bool m_usePhaseFunction = true;        // F8 to toggle
    float m_phaseStrength = 5.0f;          // Ctrl+F8/Shift+F8 to adjust (0.0-20.0)
    float m_inScatterStrength = 1.0f;      // F9/Shift+F9 to adjust (0.0-10.0)
    float m_rtLightingStrength = 2.0f;     // F10/Shift+F10 to adjust (0.0-10.0)
    bool m_useAnisotropicGaussians = true; // F11 to toggle (anisotropic particle shapes)
    float m_anisotropyStrength = 1.0f;     // F12/Shift+F12 to adjust (0.0-3.0, how stretched)

    // PCSS soft shadow system
    enum class ShadowPreset {
        Performance,  // 1-ray + temporal filtering (default)
        Balanced,     // 4-ray PCSS
        Quality,      // 8-ray PCSS
        Custom        // User-defined settings
    };
    ShadowPreset m_shadowPreset = ShadowPreset::Performance;
    uint32_t m_shadowRaysPerLight = 1;           // Shadow rays per light (1-16)
    bool m_enableTemporalFiltering = true;       // Temporal shadow accumulation
    float m_temporalBlend = 0.1f;                // Temporal blend factor (0.0-1.0)

    // DLSS 4.0 Ray Reconstruction system (AI denoising for shadow rays)
#ifdef ENABLE_DLSS
    bool m_enableDLSS = false;                   // Toggle DLSS Ray Reconstruction
    float m_dlssDenoiserStrength = 1.0f;         // Denoiser strength (0.0-2.0)
    int m_dlssQualityMode = 1;                   // 0=Quality(67%), 1=Balanced(58%), 2=Performance(50%), 3=Ultra(33%)
    bool m_dlssQualityModeChanged = false;       // Flag to defer quality mode change until safe
#endif

    int m_rtQualityMode = 0;  // 0=normal, 1=ReSTIR, 2=adaptive

    // Mouse look state
    bool m_mouseLookActive = false;
    int m_lastMouseX = 0;
    int m_lastMouseY = 0;

    // === Adaptive Quality System (ML-based) ===
    bool m_enableAdaptiveQuality = false;        // F12 to toggle
    float m_adaptiveTargetFPS = 120.0f;          // Target FPS (60/120/144)
    bool m_collectPerformanceData = false;       // Collect training data

    // Buffer dump feature (zero overhead when disabled)
    bool m_enableBufferDump = false;
    bool m_dumpBuffersNextFrame = false;
    int m_dumpTargetFrame = -1;
    std::string m_dumpOutputDir = "pix/buffer_dumps/";

    // === Ground Plane Feature (Reflective Surface Experiment) ===
    bool m_enableGroundPlane = false;           // --ground-plane flag
    float m_groundPlaneHeight = -500.0f;        // Y position (below accretion disk)
    float m_groundPlaneSize = 3000.0f;          // Width/depth extent
    float m_groundPlaneAlbedo[3] = {0.3f, 0.3f, 0.35f};  // Surface reflectance (gray)
    
    // === PINN Model Selection (--pinn flag) ===
    std::string m_pinnModelPath;                // Model path from --pinn flag (empty = auto-detect)
    int m_pinnModelIndex = 0;                   // Currently selected model in dropdown
    std::vector<std::pair<std::string, std::string>> m_availablePINNModels;  // Cached model list

    // Helper functions for buffer dumping
    void DumpGPUBuffers();
    void DumpBufferToFile(ID3D12Resource* buffer, const char* name);
    void WriteMetadataJSON();

    // Screenshot capture (F2 to capture)
    bool m_captureScreenshotNextFrame = false;
    std::string m_screenshotOutputDir = "screenshots/";

    // Screenshot metadata structure v3.0 (Enhanced Configuration Capture)
    struct ScreenshotMetadata {
        // Schema versioning
        std::string schemaVersion = "3.0";

        // === RENDERING CONFIGURATION ===

        // Active systems
        std::string activeLightingSystem;  // "MultiLight" or "RTXDI"
        std::string rendererType;          // "Billboard" or "Gaussian"

        // RTXDI configuration (only relevant if RTXDI active)
        struct RTXDIConfig {
            bool enabled = false;
            bool m4Enabled = false;
            bool m5Enabled = false;
            float temporalBlendFactor = 0.0f;
        } rtxdi;

        // Light configuration
        struct LightConfig {
            int count = 0;
            struct LightInfo {
                float posX, posY, posZ;
                float colorR, colorG, colorB;
                float intensity;
                float radius;
            };
            std::vector<LightInfo> lights;
        } lightConfig;

        // Shadow configuration
        struct ShadowConfig {
            std::string preset;  // "Performance", "Balanced", "Quality", "Custom"
            int raysPerLight = 0;
            bool temporalFilteringEnabled = false;
            float temporalBlendFactor = 0.0f;
        } shadows;

        // === QUALITY PRESET ===

        std::string qualityPreset;  // "Maximum", "Ultra", "High", "Medium", "Low"
        float targetFPS = 0.0f;     // 0 (Maximum), 30 (Ultra), 60 (High), 120 (Medium), 165 (Low)

        // === PHYSICAL EFFECTS ===

        struct PhysicalEffects {
            // Emission
            bool usePhysicalEmission = false;
            float emissionStrength = 0.0f;
            float emissionBlendFactor = 0.0f;

            // Relativistic effects
            bool useDopplerShift = false;
            float dopplerStrength = 0.0f;
            bool useGravitationalRedshift = false;
            float redshiftStrength = 0.0f;

            // Scattering
            bool usePhaseFunction = false;
            float phaseStrength = 0.0f;

            // Anisotropic Gaussians
            bool useAnisotropicGaussians = false;
            float anisotropyStrength = 0.0f;
        } physicalEffects;

        // === FEATURE STATUS FLAGS ===

        struct FeatureStatus {
            // Working features
            bool multiLightWorking = true;
            bool shadowRaysWorking = true;
            bool phaseFunctionWorking = true;
            bool physicalEmissionWorking = true;
            bool anisotropicGaussiansWorking = true;

            // WIP features (visible but not fully functional)
            bool dopplerShiftWorking = false;      // No visible effect currently
            bool redshiftWorking = false;          // No visible effect currently
            bool rtxdiM5Working = false;           // Temporal accumulation in progress

            // Deprecated/non-functional
            bool inScatteringDeprecated = true;
            bool godRaysDeprecated = true;
        } featureStatus;

        // === PARTICLES ===

        struct ParticleConfig {
            int count = 0;
            float radius = 0.0f;
            float gravityStrength = 0.0f;
            bool physicsEnabled = false;

            // Physics system details
            float innerRadius = 0.0f;
            float outerRadius = 0.0f;
            float diskThickness = 0.0f;
        } particles;

        // === PERFORMANCE ===

        struct Performance {
            float fps = 0.0f;
            float frameTime = 0.0f;
            float targetFPS = 0.0f;
            float fpsRatio = 0.0f;  // current / target (1.0 = on target)
        } performance;

        // === CAMERA ===

        struct CameraState {
            float x, y, z;       // Camera position
            float lookAtX, lookAtY, lookAtZ;  // Look-at point
            float distance;      // Distance from center
            float height;        // Height above disk
            float angle;         // Orbit angle
            float pitch;         // Vertical rotation
        } camera;

        // === ML/QUALITY ===

        struct MLQuality {
            bool pinnEnabled = false;
            std::string modelPath;
            bool adaptiveQualityEnabled = false;
            float adaptiveTargetFPS = 0.0f;

            // PINN hybrid mode details (Phase 5)
            bool hybridModeEnabled = false;
            float hybridThresholdRISCO = 10.0f;  // × R_ISCO
        } mlQuality;

        // === MATERIAL SYSTEM (Phase 5 / Sprint 1) ===

        struct MaterialSystem {
            bool enabled = false;
            int particleStructSizeBytes = 32;  // 32=legacy, 48=material system
            int materialTypesCount = 1;         // 1=legacy, 5+=material system

            struct MaterialTypeDistribution {
                int plasmaCount = 0;
                int starCount = 0;
                int gasCount = 0;
                int rockyCount = 0;
                int icyCount = 0;
            } distribution;
        } materialSystem;

        // === ADAPTIVE PARTICLE RADIUS (Phase 1.5 - COMPLETE) ===

        struct AdaptiveRadius {
            bool enabled = false;
            float innerZoneDistance = 150.0f;
            float outerZoneDistance = 800.0f;
            float innerScaleMultiplier = 0.3f;
            float outerScaleMultiplier = 3.0f;
            float densityScaleMin = 0.5f;
            float densityScaleMax = 1.5f;
        } adaptiveRadius;

        // === DLSS INTEGRATION (Phase 7 - PARTIAL) ===

        struct DLSSConfig {
            bool enabled = false;
            std::string qualityMode;  // "Performance", "Balanced", "Quality", "UltraPerformance"
            int internalResolutionWidth = 0;
            int internalResolutionHeight = 0;
            int outputResolutionWidth = 0;
            int outputResolutionHeight = 0;
            bool motionVectorsEnabled = false;
        } dlss;

        // === DYNAMIC EMISSION (Phase 3.8 - COMPLETE) ===

        struct DynamicEmission {
            float emissionStrength = 0.25f;
            float temperatureThreshold = 22000.0f;  // Kelvin
            float rtSuppressionFactor = 0.7f;
            float temporalModulationRate = 0.03f;
        } dynamicEmission;

        // === PERFORMANCE FEATURES ===

        bool variableRefreshRateEnabled = false;  // Tearing mode (Phase 3)

        // === METADATA ===

        std::string timestamp;
        std::string configFile;
    };

    void CaptureScreenshot();
    void SaveBackBufferToFile(ID3D12Resource* backBuffer, const std::string& filename);
    void SaveScreenshotMetadata(const std::string& screenshotPath, const ScreenshotMetadata& metadata);
    ScreenshotMetadata GatherScreenshotMetadata();
    void DetectQualityPreset(ScreenshotMetadata& meta);  // Helper to determine quality preset from settings

    // ImGui
    void InitializeImGui();
    void ShutdownImGui();
    void RenderImGui();
    bool m_showImGui = true;  // F1 to toggle
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_imguiDescriptorHeap;

    // HDR→SDR blit pipeline
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_blitRootSignature;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_blitPSO;
    bool CreateBlitPipeline();
};