#pragma once

#include <windows.h>
#include <memory>
#include <chrono>

// Forward declarations
class Device;
class SwapChain;
class FeatureDetector;
class ParticleSystem;
class ParticleRenderer;
class ParticleRenderer_Gaussian;
class RTLightingSystem_RayQuery;
class ResourceManager;

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

    struct Config {
        uint32_t particleCount = 100000;
        bool enableRT = true;
        bool preferMeshShaders = true;
        bool enableDebugLayer = false;
        RendererType rendererType = RendererType::Billboard;
    } m_config;

    // Runtime camera controls - TOP-DOWN VIEW of accretion disk
    float m_cameraDistance = 800.0f;   // Distance from center (horizontal)
    float m_cameraHeight = 1200.0f;    // HIGH above disk to see ring structure from above
    float m_cameraAngle = 0.0f;         // Orbit angle
    float m_cameraPitch = 0.0f;         // Vertical rotation
    float m_particleSize = 50.0f;       // Larger for initial visibility
    bool m_physicsEnabled = true;       // ENABLED - physics shader initializes particles on GPU

    // RT Lighting runtime controls
    float m_rtLightingIntensity = 1.0f;
    float m_rtMaxDistance = 100.0f;  // Updated to match shader
    float m_rtParticleRadius = 5.0f;   // Updated to match shader

    // Enhancement toggles and strengths
    bool m_usePhysicalEmission = false;
    float m_emissionStrength = 1.0f;       // 0.0-5.0

    bool m_useDopplerShift = false;
    float m_dopplerStrength = 1.0f;        // 0.0-5.0 (multiplier)

    bool m_useGravitationalRedshift = false;
    float m_redshiftStrength = 1.0f;       // 0.0-5.0 (multiplier)

    // Gaussian RT system toggles
    bool m_useShadowRays = true;           // F5 to toggle
    bool m_useInScattering = false;        // F6 to toggle (OFF by default - very expensive!)
    bool m_useReSTIR = false;              // F7 to toggle (ReSTIR temporal resampling)
    float m_restirTemporalWeight = 0.9f;   // Ctrl+F7/Shift+F7 to adjust (0.0-1.0, temporal trust)
    bool m_usePhaseFunction = true;        // F8 to toggle
    float m_phaseStrength = 5.0f;          // Ctrl+F8/Shift+F8 to adjust (0.0-20.0)
    float m_inScatterStrength = 1.0f;      // F9/Shift+F9 to adjust (0.0-10.0)
    float m_rtLightingStrength = 2.0f;     // F10/Shift+F10 to adjust (0.0-10.0)
    bool m_useAnisotropicGaussians = true; // F11 to toggle (anisotropic particle shapes)
    float m_anisotropyStrength = 1.0f;     // F12/Shift+F12 to adjust (0.0-3.0, how stretched)
    uint32_t m_restirInitialCandidates = 16; // Number of light candidates to test (16-32)

    int m_rtQualityMode = 0;  // 0=normal, 1=ReSTIR, 2=adaptive

    // Mouse look state
    bool m_mouseLookActive = false;
    int m_lastMouseX = 0;
    int m_lastMouseY = 0;
};