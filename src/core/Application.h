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
class RTLightingSystem_RayQuery;
class ResourceManager;

class Application {
public:
    Application();
    ~Application();

    // Lifecycle
    bool Initialize(HINSTANCE hInstance, int nCmdShow);
    int Run();
    void Shutdown();

private:
    // Window management
    bool CreateAppWindow(HINSTANCE hInstance, int nCmdShow);
    static LRESULT CALLBACK WindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

    // Frame timing
    void UpdateFrameStats();

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
    std::unique_ptr<ParticleRenderer> m_particleRenderer;
    std::unique_ptr<RTLightingSystem_RayQuery> m_rtLighting;

    // Timing
    std::chrono::high_resolution_clock::time_point m_lastFrameTime;
    float m_deltaTime = 0.0f;
    float m_totalTime = 0.0f;

    // Stats
    uint32_t m_frameCount = 0;
    float m_fps = 0.0f;

    // Configuration
    struct Config {
        uint32_t particleCount = 100000;
        bool enableRT = true;
        bool preferMeshShaders = true;
        bool enableDebugLayer = false;
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
    float m_rtMaxDistance = 500.0f;
    float m_rtParticleRadius = 25.0f;

    // Mouse look state
    bool m_mouseLookActive = false;
    int m_lastMouseX = 0;
    int m_lastMouseY = 0;
};