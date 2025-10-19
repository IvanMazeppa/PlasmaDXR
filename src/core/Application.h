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
class ResourceManager;

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

private:
    // Window management
    bool CreateAppWindow(HINSTANCE hInstance, int nCmdShow);
    void ToggleFullscreen();
    static LRESULT CALLBACK WindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

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
    std::unique_ptr<RTXDILightingSystem> m_rtxdiLightingSystem;  // RTXDI parallel lighting path

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
        MultiLight,     // Multi-light brute force (Phase 3.5, good for <20 lights)
        RTXDI           // NVIDIA RTXDI with ReSTIR (Phase 4, scales to 100+ lights)
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

    // Physics system parameters (readonly for now - would require ParticleSystem API changes)
    float m_innerRadius = 10.0f;        // Inner accretion disk radius
    float m_outerRadius = 300.0f;       // Outer accretion disk radius
    float m_diskThickness = 50.0f;      // Disk thickness
    float m_physicsTimeStep = 0.008333f; // Physics timestep (fixed at 120Hz)

    // RT Lighting runtime controls
    bool m_enableRTLighting = true;  // Toggle for particle-to-particle RT lighting
    float m_rtLightingIntensity = 1.0f;
    float m_rtMaxDistance = 100.0f;  // Updated to match shader
    float m_rtParticleRadius = 5.0f;   // Updated to match shader

    // Enhancement toggles and strengths
    bool m_usePhysicalEmission = false;
    float m_emissionStrength = 1.0f;       // 0.0-5.0
    float m_emissionBlendFactor = 1.0f;    // 0.0-1.0 (0=artistic, 1=physical)

    // Multi-light system (Phase 3.5)
    std::vector<ParticleRenderer_Gaussian::Light> m_lights;  // Active lights (max 16)
    void InitializeLights();  // Create default 13-light configuration
    int m_selectedLightIndex = -1;  // For ImGui light selection

    // Particle count control
    uint32_t m_activeParticleCount = 10000;  // Runtime-adjustable particle count

    bool m_useDopplerShift = false;
    float m_dopplerStrength = 1.0f;        // 0.0-5.0 (multiplier)

    bool m_useGravitationalRedshift = false;
    float m_redshiftStrength = 1.0f;       // 0.0-5.0 (multiplier)

    // Gaussian RT system toggles
    bool m_useShadowRays = true;           // F5 to toggle
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

    int m_rtQualityMode = 0;  // 0=normal, 1=ReSTIR, 2=adaptive

    // Mouse look state
    bool m_mouseLookActive = false;
    int m_lastMouseX = 0;
    int m_lastMouseY = 0;

    // Buffer dump feature (zero overhead when disabled)
    bool m_enableBufferDump = false;
    bool m_dumpBuffersNextFrame = false;
    int m_dumpTargetFrame = -1;
    std::string m_dumpOutputDir = "pix/buffer_dumps/";

    // Helper functions for buffer dumping
    void DumpGPUBuffers();
    void DumpBufferToFile(ID3D12Resource* buffer, const char* name);
    void WriteMetadataJSON();

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