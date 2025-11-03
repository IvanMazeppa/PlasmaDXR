#include "Application.h"
#include "Device.h"
#include "SwapChain.h"
#include "FeatureDetector.h"
#include "../config/Config.h"
#include "../particles/ParticleSystem.h"
#include "../particles/ParticleRenderer.h"
#include "../particles/ParticleRenderer_Gaussian.h"
#include "../lighting/RTLightingSystem_RayQuery.h"
#include "../lighting/RTXDILightingSystem.h"
#include "../lighting/VolumetricReSTIRSystem.h"
#include "../lighting/ProbeGridSystem.h"
#include "../utils/ResourceManager.h"
#include "../utils/Logger.h"
#include "../ml/AdaptiveQualitySystem.h"
#ifdef ENABLE_DLSS
#include "../dlss/DLSSSystem.h"
#endif
#ifdef USE_PIX
#include "../debug/PIXCaptureHelper.h"
#endif
#include <algorithm>
#include <filesystem>

// ImGui
#include "../../external/imgui/imgui.h"
#include "../../external/imgui/backends/imgui_impl_win32.h"
#include "../../external/imgui/backends/imgui_impl_dx12.h"

// Forward declare ImGui WndProc handler
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// Window procedure forward declaration
LRESULT CALLBACK WindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

Application::Application() {
    m_lastFrameTime = std::chrono::high_resolution_clock::now();
}

Application::~Application() {
    Shutdown();
}

bool Application::Initialize(HINSTANCE hInstance, int nCmdShow, int argc, char** argv) {
    LOG_INFO("Initializing Application...");

    // Load configuration from JSON file
    ::Config::ConfigManager configMgr;
    configMgr.Initialize(argc, argv);
    const ::Config::AppConfig& appConfig = configMgr.GetConfig();

    // Initialize PIX with config parameters (must be after config load, before window creation)
#ifdef USE_PIX
    Debug::PIXCaptureHelper::InitializeWithConfig(appConfig.debug.pixAutoCapture,
                                                    appConfig.debug.pixCaptureFrame);
#else
    LOG_INFO("[PIX] PIX support disabled (USE_PIX not defined)");
#endif

    // Apply configuration to Application settings
    m_config.particleCount = appConfig.rendering.particleCount;
    m_config.rendererType = (appConfig.rendering.rendererType == ::Config::RendererType::Gaussian) ?
                            RendererType::Gaussian : RendererType::Billboard;
    m_config.enableRT = true; // Always enabled
    m_config.preferMeshShaders = true;
    m_config.enableDebugLayer = appConfig.debug.enableDebugLayer;

    m_width = appConfig.rendering.resolutionWidth;
    m_height = appConfig.rendering.resolutionHeight;

    // Initialize active particle count (runtime adjustable)
    m_activeParticleCount = m_config.particleCount;

    // Apply camera config
    m_cameraDistance = appConfig.camera.startDistance;
    m_cameraHeight = appConfig.camera.startHeight;
    m_cameraAngle = appConfig.camera.startAngle;
    m_cameraPitch = appConfig.camera.startPitch;
    m_particleSize = appConfig.camera.particleSize;

    // Apply feature toggles
    m_useShadowRays = appConfig.features.enableShadowRays;
    m_useInScattering = appConfig.features.enableInScattering;
    m_inScatterStrength = appConfig.features.inScatterStrength;
    m_usePhaseFunction = appConfig.features.enablePhaseFunction;
    m_phaseStrength = appConfig.features.phaseStrength;
    m_useAnisotropicGaussians = appConfig.features.useAnisotropicGaussians;
    m_anisotropyStrength = appConfig.features.anisotropyStrength;
    m_rtLightingStrength = appConfig.features.rtLightingStrength;
    m_usePhysicalEmission = appConfig.features.usePhysicalEmission;
    m_emissionStrength = appConfig.features.emissionStrength;
    m_useDopplerShift = appConfig.features.useDopplerShift;
    m_dopplerStrength = appConfig.features.dopplerStrength;
    m_useGravitationalRedshift = appConfig.features.useGravitationalRedshift;
    m_redshiftStrength = appConfig.features.redshiftStrength;

    // Parse command-line argument overrides (these override config file)
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--gaussian" || arg == "-g") {
            m_config.rendererType = RendererType::Gaussian;
        } else if (arg == "--billboard" || arg == "-b") {
            m_config.rendererType = RendererType::Billboard;
        } else if (arg == "--particles" && i + 1 < argc) {
            m_config.particleCount = std::atoi(argv[++i]);
        } else if (arg == "--rtxdi") {
            m_lightingSystem = LightingSystem::RTXDI;
            LOG_INFO("Lighting system: RTXDI (NVIDIA RTX Direct Illumination)");
        } else if (arg == "--multi-light") {
            m_lightingSystem = LightingSystem::MultiLight;
            LOG_INFO("Lighting system: Multi-Light (brute force)");
        } else if (arg == "--restir") {
            m_lightingSystem = LightingSystem::VolumetricReSTIR;
            LOG_INFO("Lighting system: Volumetric ReSTIR (autonomous testing mode)");
        } else if (arg == "--dump-buffers") {
            m_enableBufferDump = true;
            // Check if next arg is a frame number (optional)
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                m_dumpTargetFrame = std::atoi(argv[i + 1]);
                i++;
                LOG_INFO("Buffer dump enabled (frame {})", m_dumpTargetFrame);
            } else {
                LOG_INFO("Buffer dump enabled (manual trigger with Ctrl+D)");
            }
        } else if (arg == "--dump-dir" && i + 1 < argc) {
            m_dumpOutputDir = argv[i + 1];
            i++;
            LOG_INFO("Dump directory: {}", m_dumpOutputDir);
        } else if (arg == "--help" || arg == "-h") {
            LOG_INFO("Usage: PlasmaDX-Clean.exe [options]");
            LOG_INFO("  --config=<file>      : Load configuration from JSON file");
            LOG_INFO("  --gaussian, -g       : Use 3D Gaussian Splatting renderer");
            LOG_INFO("  --billboard, -b      : Use Billboard renderer");
            LOG_INFO("  --particles <count>  : Set particle count");
            LOG_INFO("  --rtxdi              : Use NVIDIA RTXDI lighting (Phase 4)");
            LOG_INFO("  --multi-light        : Use multi-light system (default, Phase 3.5)");
            LOG_INFO("  --restir             : Use Volumetric ReSTIR (Phase 1 - experimental)");
            LOG_INFO("  --dump-buffers [frame] : Enable buffer dumps (optional: auto-dump at frame)");
            LOG_INFO("  --dump-dir <path>    : Set buffer dump output directory");
        }
    }

    // Log configuration
    LOG_INFO("Particle count: {}", m_config.particleCount);
    LOG_INFO("Renderer: {}", m_config.rendererType == RendererType::Gaussian ?
             "3D Gaussian Splatting (volumetric)" : "Billboard (stable)");

    // Create window
    if (!CreateAppWindow(hInstance, nCmdShow)) {
        LOG_ERROR("Failed to create window");
        return false;
    }

    // Initialize core systems
    m_device = std::make_unique<Device>();
    if (!m_device->Initialize(m_config.enableDebugLayer)) {
        LOG_ERROR("Failed to initialize device");
        return false;
    }

    // Feature detection
    m_features = std::make_unique<FeatureDetector>();
    if (!m_features->Initialize(m_device->GetDevice())) {
        LOG_ERROR("Failed to detect features");
        return false;
    }

    // Resource manager
    m_resources = std::make_unique<ResourceManager>();
    if (!m_resources->Initialize(m_device.get())) {
        LOG_ERROR("Failed to initialize resource manager");
        return false;
    }

    // Create descriptor heaps
    m_resources->CreateDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1000, true);
    m_resources->CreateDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 10, false);

    // Swap chain
    m_swapChain = std::make_unique<SwapChain>();
    if (!m_swapChain->Initialize(m_device.get(), m_hwnd, m_width, m_height)) {
        LOG_ERROR("Failed to initialize swap chain");
        return false;
    }

    // Initialize particle system
    m_particleSystem = std::make_unique<ParticleSystem>();
    if (!m_particleSystem->Initialize(m_device.get(), m_resources.get(), m_config.particleCount)) {
        LOG_ERROR("Failed to initialize particle system");
        return false;
    }

    // Initialize particle renderer based on command-line selection
    if (m_config.rendererType == RendererType::Gaussian) {
        LOG_INFO("Initializing 3D Gaussian Splatting renderer...");
        m_gaussianRenderer = std::make_unique<ParticleRenderer_Gaussian>();
        if (!m_gaussianRenderer->Initialize(m_device.get(), m_resources.get(),
                                            m_config.particleCount, m_width, m_height)) {
            LOG_ERROR("Failed to initialize Gaussian renderer");
            return false;
        }

        // Initialize multi-light system (Phase 3.5) with default 13-light configuration
        // Users can switch presets: "Disk (13)", "Single", "Dome (8)", or add manually with ] key
        InitializeLights();  // Start with default 13-light disk configuration

        LOG_INFO("Render Path: 3D Gaussian Splatting");
    } else {
        LOG_INFO("Initializing Billboard renderer...");
        m_particleRenderer = std::make_unique<ParticleRenderer>();
        if (!m_particleRenderer->Initialize(m_device.get(), m_resources.get(),
                                            m_features.get(), m_config.particleCount)) {
            LOG_ERROR("Failed to initialize particle renderer");
            return false;
        }
        LOG_INFO("Render Path: {}", m_particleRenderer->GetActivePathName());
    }

    // Initialize RT lighting if supported
    if (m_features->CanUseDXR() && m_config.enableRT) {
        m_rtLighting = std::make_unique<RTLightingSystem_RayQuery>();
        if (!m_rtLighting->Initialize(m_device.get(), m_resources.get(), m_config.particleCount)) {
            LOG_ERROR("Failed to initialize RT lighting system");
            // Continue without RT
            m_rtLighting.reset();
        } else {
            LOG_INFO("RT Lighting (RayQuery) initialized - looking for GREEN particles!");
        }
    }

    // Initialize RTXDI lighting (always, since user can switch via ImGui/F3)
    LOG_INFO("Initializing RTXDI Lighting System...");
    m_rtxdiLightingSystem = std::make_unique<RTXDILightingSystem>();
    if (!m_rtxdiLightingSystem->Initialize(m_device.get(), m_resources.get(), m_width, m_height)) {
        LOG_ERROR("Failed to initialize RTXDI lighting system");
        LOG_ERROR("  RTXDI will not be available (F3 toggle disabled)");
        m_rtxdiLightingSystem.reset();
        // Don't force fallback - user can still use multi-light
        if (m_lightingSystem == LightingSystem::RTXDI) {
            LOG_ERROR("  Startup mode was RTXDI - falling back to Multi-Light");
            m_lightingSystem = LightingSystem::MultiLight;
        }
    } else {
        LOG_INFO("RTXDI Lighting System initialized successfully!");
        LOG_INFO("  Light grid: 30x30x30 cells (27,000 total)");
        LOG_INFO("  Ready for runtime switching (F3 key)");
    }

    // Initialize blit pipeline (HDR→SDR conversion for Gaussian renderer)
    if (m_gaussianRenderer) {
        if (!CreateBlitPipeline()) {
            LOG_ERROR("Failed to create blit pipeline");
            return false;
        }
    }

    // Initialize Adaptive Quality System (ML-based performance prediction)
    LOG_INFO("Initializing Adaptive Quality System...");
    m_adaptiveQuality = std::make_unique<AdaptiveQualitySystem>();
    if (!m_adaptiveQuality->Initialize("ml/models/adaptive_quality.model")) {
        LOG_WARN("Adaptive Quality System initialized with default heuristics (no trained model)");
    } else {
        LOG_INFO("Adaptive Quality System ready (ML model loaded)");
    }
    m_adaptiveQuality->SetTargetFPS(m_adaptiveTargetFPS);

#ifdef ENABLE_DLSS
    // Initialize DLSS 4.0 Ray Reconstruction (AI denoising for shadow rays)
    LOG_INFO("Initializing DLSS Ray Reconstruction...");

    // Create NGX directory for DLSS logs and cache
    CreateDirectoryW(L"ngx", nullptr); // Ignore error if exists
    wchar_t ngxPath[MAX_PATH];
    GetFullPathNameW(L"ngx", MAX_PATH, ngxPath, nullptr);
    LOG_INFO("NGX data path: {}", std::filesystem::path(ngxPath).string());

    m_dlssSystem = std::make_unique<DLSSSystem>();
    if (m_dlssSystem->Initialize(m_device->GetDevice(), ngxPath)) {
        LOG_INFO("DLSS: NGX SDK initialized successfully");
        LOG_INFO("  Press F4 to toggle DLSS denoising");
        LOG_INFO("  Feature creation deferred to first render (requires command list)");
        LOG_INFO("  Denoiser strength: {:.1f} (adjustable in ImGui)", m_dlssDenoiserStrength);

        // Pass DLSS system to Gaussian renderer for lazy feature creation
        if (m_gaussianRenderer) {
            m_gaussianRenderer->SetDLSSSystem(m_dlssSystem.get(), m_width, m_height, m_dlssQualityMode);
            LOG_INFO("  DLSS system reference passed to Gaussian renderer");
        }
    } else {
        LOG_WARN("DLSS initialization failed");
        LOG_WARN("  Possible causes: Driver too old (need 531.00+), non-RTX GPU, missing DLL");
        LOG_WARN("  Shadow rays will use traditional denoising");
        m_dlssSystem.reset();
    }
#else
    LOG_INFO("DLSS support not enabled (ENABLE_DLSS not defined)");
#endif

    // Initialize Volumetric ReSTIR system (Phase 1 - experimental)
    // IMPORTANT: Initialize AFTER DLSS to get correct render resolution
    LOG_INFO("Initializing Volumetric ReSTIR System (Phase 1)...");

    // IMPORTANT: VolumetricReSTIR uses REDUCED resolution for Phase 1 to avoid GPU timeout
    // Phase 1 testing: 1/4 native resolution (640×360 @ 2560×1440)
    // This reduces shader invocations from 3.6M to 230K, preventing TDR crashes
    // Will be upscaled in later phases for production quality
    uint32_t restirWidth = m_width / 4;
    uint32_t restirHeight = m_height / 4;
    LOG_INFO("Initializing VolumetricReSTIR at 1/4 resolution (Phase 1 testing): {}x{}", restirWidth, restirHeight);
    LOG_INFO("  Native resolution: {}x{}", m_width, m_height);
    LOG_INFO("  Shader invocations reduced from {:.1f}M to {:.1f}K",
            (m_width * m_height) / 1000000.0f,
            (restirWidth * restirHeight) / 1000.0f);

    m_volumetricReSTIR = std::make_unique<VolumetricReSTIRSystem>();
    if (!m_volumetricReSTIR->Initialize(m_device.get(), m_resources.get(), restirWidth, restirHeight)) {
        LOG_ERROR("Failed to initialize Volumetric ReSTIR system");
        LOG_ERROR("  Volumetric ReSTIR will not be available");
        m_volumetricReSTIR.reset();
        // Don't force fallback - user can still use other lighting systems
        if (m_lightingSystem == LightingSystem::VolumetricReSTIR) {
            LOG_ERROR("  Startup mode was VolumetricReSTIR - falling back to Multi-Light");
            m_lightingSystem = LightingSystem::MultiLight;
        }
    } else {
        LOG_INFO("Volumetric ReSTIR System initialized successfully!");
        LOG_INFO("  Reservoir buffers: {:.1f} MB @ {}x{}",
                (restirWidth * restirHeight * 64 * 2) / (1024.0f * 1024.0f),
                restirWidth, restirHeight);
        LOG_INFO("  Phase 1: RIS candidate generation (no spatial/temporal reuse yet)");
        LOG_INFO("  Ready for testing (experimental)");

        // Pre-allocate descriptor for clear operation (prevents descriptor heap exhaustion)
        // Use final output texture (DLSS upscaled if enabled, otherwise render-res)
        m_volumetricReSTIRClearUAV = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        uavDesc.Texture2D.MipSlice = 0;
        m_device->GetDevice()->CreateUnorderedAccessView(
            m_gaussianRenderer->GetFinalOutputTexture(),  // DLSS upscaled or render-res
            nullptr,
            &uavDesc,
            m_volumetricReSTIRClearUAV
        );
        LOG_INFO("  Pre-allocated clear UAV descriptor (final output texture)");
    }

    // Initialize Probe Grid System (Phase 0.13.1)
    // Replaces Volumetric ReSTIR (which suffered from atomic contention at ≥2045 particles)
    m_probeGridSystem = std::make_unique<ProbeGridSystem>();
    if (!m_probeGridSystem->Initialize(m_device.get(), m_resources.get())) {
        LOG_ERROR("Failed to initialize Probe Grid System");
        LOG_ERROR("  Probe Grid will not be available");
        m_probeGridSystem.reset();
    } else {
        LOG_INFO("Probe Grid System initialized successfully!");
        LOG_INFO("  Grid: 32³ = 32,768 probes @ 93.75-unit spacing");
        LOG_INFO("  Memory: 4.06 MB probe buffer");
        LOG_INFO("  Zero atomic operations = zero contention!");
        LOG_INFO("  Target: 10K particles @ 90-110 FPS");
    }

    // Initialize ImGui
    InitializeImGui();

    m_isRunning = true;
    LOG_INFO("Application initialized successfully");

    return true;
}

void Application::Shutdown() {
    if (m_device) {
        m_device->WaitForGPU();
    }

    // Shutdown ImGui
    ShutdownImGui();

#ifdef ENABLE_DLSS
    // Shutdown DLSS (releases NGX SDK resources)
    m_dlssSystem.reset();
#endif

    m_rtLighting.reset();
    m_particleRenderer.reset();
    m_particleSystem.reset();
    m_swapChain.reset();
    m_resources.reset();
    m_features.reset();
    m_device.reset();

    if (m_hwnd) {
        DestroyWindow(m_hwnd);
        m_hwnd = nullptr;
    }
}

int Application::Run() {
    MSG msg = {};

    // Reset frame timer so first deltaTime isn't huge (initialization time)
    m_lastFrameTime = std::chrono::high_resolution_clock::now();

    while (m_isRunning) {
        // Process Windows messages
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) {
                m_isRunning = false;
                break;
            }
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        if (m_isRunning) {
            // Calculate ACTUAL frame time for FPS measurement
            auto currentTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> deltaTimeDuration = currentTime - m_lastFrameTime;
            float actualFrameTime = deltaTimeDuration.count();
            m_lastFrameTime = currentTime;

            // Use fixed timestep for consistent physics regardless of framerate
            // This ensures physics runs at same speed whether you're at 10 FPS or 200 FPS
            const float fixedTimeStep = 1.0f / 120.0f; // 120 Hz physics
            m_deltaTime = fixedTimeStep;

            // Update and render (with fixed physics timestep)
            Update(m_deltaTime);
            Render();

#ifdef USE_PIX
            // Check for PIX auto-capture (may exit app if capture triggered)
            if (Debug::PIXCaptureHelper::CheckAutomaticCapture(m_frameCount)) {
                // Capture was triggered - app will exit
                m_isRunning = false;
            }
#endif

            // Update stats (with ACTUAL frame time for correct FPS)
            UpdateFrameStats(actualFrameTime);
        }
    }

    return static_cast<int>(msg.wParam);
}

void Application::Update(float deltaTime) {
    m_totalTime += deltaTime;

    // === Adaptive Quality System Update ===
    if (m_adaptiveQuality && m_enableAdaptiveQuality) {
        // Build scene features for prediction
        AdaptiveQualitySystem::SceneFeatures features;
        features.particleCount = m_activeParticleCount;
        features.lightCount = static_cast<float>(m_lights.size());
        features.cameraDistance = m_cameraDistance;
        features.shadowRaysPerLight = m_shadowRaysPerLight;
        features.useShadowRays = m_useShadowRays ? 1.0f : 0.0f;
        features.useInScattering = m_useInScattering ? 1.0f : 0.0f;
        features.usePhaseFunction = m_usePhaseFunction ? 1.0f : 0.0f;
        features.useAnisotropicGaussians = m_useAnisotropicGaussians ? 1.0f : 0.0f;
        features.enableTemporalFiltering = m_enableTemporalFiltering ? 1.0f : 0.0f;
        features.useRTXDI = (m_lightingSystem == LightingSystem::RTXDI) ? 1.0f : 0.0f;
        features.enableRTLighting = m_enableRTLighting ? 1.0f : 0.0f;
        features.godRayDensity = m_godRayDensity;
        features.actualFrameTime = m_deltaTime * 1000.0f; // Convert to ms

        // Update adaptive quality (may adjust settings)
        m_adaptiveQuality->Update(deltaTime, features);

        // Apply recommended quality level
        auto recommendedQuality = m_adaptiveQuality->GetCurrentQuality();
        auto preset = m_adaptiveQuality->GetQualityPreset(recommendedQuality);

        // Apply settings
        m_shadowRaysPerLight = preset.shadowRaysPerLight;
        m_useShadowRays = preset.useShadowRays;
        m_useInScattering = preset.useInScattering;
        m_usePhaseFunction = preset.usePhaseFunction;
        m_useAnisotropicGaussians = preset.useAnisotropicGaussians;
        m_enableTemporalFiltering = preset.enableTemporalFiltering;
        m_enableRTLighting = preset.enableRTLighting;
        m_godRayDensity = preset.godRayDensity;
        m_rtLightingStrength = preset.rtLightingStrength;
    }

    // Physics update moved to Render() to ensure proper command list ordering
    // (physics commands must be recorded after ResetCommandList())

    // DEBUG: Disabled for now - readback needs to be called after frame completes
    // TODO: Add readback in a separate debug command or after Present()
    /*
    static int s_debugFrameCount = 0;
    s_debugFrameCount++;
    if (s_debugFrameCount == 10 && m_particleSystem) {
        LOG_INFO("=== Frame 10: Checking GPU particle data ===");
        m_particleSystem->DebugReadbackParticles(5);
    }
    */
}

void Application::Render() {
#ifdef ENABLE_DLSS
    // Apply deferred DLSS quality mode change (safe point: before GPU work begins)
    if (m_dlssQualityModeChanged && m_gaussianRenderer) {
        LOG_INFO("DLSS: Applying quality mode change now (safe point)...");
        m_gaussianRenderer->SetDLSSSystem(m_dlssSystem.get(), m_width, m_height, m_dlssQualityMode);
        m_dlssQualityModeChanged = false;
    }
#endif

    // Check if we should schedule buffer dump (zero overhead: 2 int comparisons + 1 bool check)
    if (m_enableBufferDump && m_dumpTargetFrame > 0 && m_frameCount == static_cast<uint32_t>(m_dumpTargetFrame)) {
        m_dumpBuffersNextFrame = true;
        LOG_INFO("Buffer dump scheduled for next frame (frame {})", m_frameCount);
    }

    // Reset command list
    m_device->ResetCommandList();
    auto cmdList = m_device->GetCommandList();

    // CRITICAL: Run physics update AFTER reset, so commands are recorded properly!
    if (m_physicsEnabled && m_particleSystem) {
        // Sync runtime particle count control
        m_particleSystem->SetActiveParticleCount(m_activeParticleCount);
        m_particleSystem->Update(m_deltaTime, m_totalTime);

        // PHYSICS-DRIVEN LIGHTS: Update light positions to simulate orbital motion
        // Lights move like celestial bodies (stars/clusters) orbiting the black hole
        if (m_physicsDrivenLights && !m_lights.empty()) {
            // Keplerian orbit: angular velocity ∝ 1/√r
            float orbitSpeedScale = 5.0f;  // Adjustable speed multiplier

            for (size_t i = 0; i < m_lights.size(); i++) {
                // Current position
                float x = m_lights[i].position.x;
                float z = m_lights[i].position.z;
                float radius = sqrtf(x * x + z * z);

                if (radius > 1.0f) {  // Skip lights at origin
                    float angle = atan2f(z, x);

                    // Keplerian angular velocity: ω = √(GM/r³) ≈ 1/√r
                    float angularVelocity = orbitSpeedScale / sqrtf(radius);
                    angle -= angularVelocity * m_deltaTime;  // Negative for clockwise rotation (matches particles)

                    // Update XZ position (orbital plane)
                    m_lights[i].position.x = radius * cosf(angle);
                    m_lights[i].position.z = radius * sinf(angle);

                    // Small vertical oscillation for disk thickness variation
                    m_lights[i].position.y += sinf(m_totalTime * 0.3f + i * 2.0f) * 10.0f * m_deltaTime;
                    m_lights[i].position.y = fmaxf(-50.0f, fminf(50.0f, m_lights[i].position.y));  // Clamp to disk thickness
                }
            }
        }

        // STELLAR TEMPERATURE COLORS: Auto-apply colors based on intensity
        // Maps intensity → temperature → realistic stellar colors
        if (m_useStellarTemperatureColors && !m_lights.empty()) {
            for (size_t i = 0; i < m_lights.size(); i++) {
                float temperature = GetStellarTemperatureFromIntensity(m_lights[i].intensity);
                m_lights[i].color = GetStellarColorFromTemperature(temperature);
            }
        }
    }

    // Get current back buffer
    auto backBuffer = m_swapChain->GetCurrentBackBuffer();
    auto rtvHandle = m_swapChain->GetCurrentRTV();

    // Transition to render target
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = backBuffer;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    cmdList->ResourceBarrier(1, &barrier);

    // Clear render target (dark blue background for space)
    float clearColor[] = { 0.0f, 0.0f, 0.1f, 1.0f };
    cmdList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);

    // Set render target
    cmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);

    // Set viewport and scissor
    D3D12_VIEWPORT viewport = { 0, 0, static_cast<float>(m_width), static_cast<float>(m_height), 0.0f, 1.0f };
    D3D12_RECT scissor = { 0, 0, static_cast<LONG>(m_width), static_cast<LONG>(m_height) };
    cmdList->RSSetViewports(1, &viewport);
    cmdList->RSSetScissorRects(1, &scissor);

    // RTXDI Light Grid Update (if RTXDI lighting system is active)
    if (m_lightingSystem == LightingSystem::RTXDI && m_rtxdiLightingSystem && !m_lights.empty()) {
        m_rtxdiLightingSystem->UpdateLightGrid(m_lights.data(), static_cast<uint32_t>(m_lights.size()), cmdList);

        // Log first few frames for verification
        static int gridUpdateCount = 0;
        gridUpdateCount++;
        if (gridUpdateCount <= 5) {
            LOG_INFO("RTXDI Light Grid updated (frame {}, {} lights)", m_frameCount, m_lights.size());
        }

        // === Milestone 4: RTXDI Reservoir Sampling ===
        // Dispatch rays to sample light grid and select optimal light per pixel
        ComPtr<ID3D12GraphicsCommandList4> cmdList4;
        if (SUCCEEDED(cmdList->QueryInterface(IID_PPV_ARGS(&cmdList4)))) {
            m_rtxdiLightingSystem->DispatchRays(cmdList4.Get(), m_width, m_height, static_cast<uint32_t>(m_frameCount));

            if (gridUpdateCount <= 5) {
                LOG_INFO("RTXDI DispatchRays executed ({}x{}, frame {})", m_width, m_height, m_frameCount);
            }

            // === Milestone 5: Temporal Accumulation ===
            // Smooth patchwork pattern by accumulating samples over time
            DirectX::XMFLOAT3 cameraPos = {
                m_cameraDistance * cosf(m_cameraAngle),
                m_cameraHeight,
                m_cameraDistance * sinf(m_cameraAngle)
            };
            m_rtxdiLightingSystem->DispatchTemporalAccumulation(cmdList, cameraPos, static_cast<uint32_t>(m_frameCount));

            if (gridUpdateCount <= 5) {
                LOG_INFO("RTXDI M5 Temporal Accumulation dispatched (frame {})", m_frameCount);
            }
        } else {
            if (gridUpdateCount == 1) {
                LOG_ERROR("Failed to query ID3D12GraphicsCommandList4 for RTXDI DispatchRays");
            }
        }
    }

    // Compute camera position (needed for both RT lighting and rendering)
    float camX = m_cameraDistance * cosf(m_cameraPitch) * sinf(m_cameraAngle);
    float camY = m_cameraDistance * sinf(m_cameraPitch);
    float camZ = m_cameraDistance * cosf(m_cameraPitch) * cosf(m_cameraAngle);
    DirectX::XMFLOAT3 cameraPosition = DirectX::XMFLOAT3(camX, camY + m_cameraHeight, camZ);

    // RT Lighting Pass (if enabled) - now with dynamic emission!
    ID3D12Resource* rtLightingBuffer = nullptr;
    if (m_rtLighting && m_particleSystem) {
        // Full DXR 1.1 RayQuery pipeline: AABB → BLAS → TLAS → RT Lighting
        // RT lighting uses particle buffer as UAV (already in correct state from initialization)
        m_rtLighting->ComputeLighting(cmdList,
                                     m_particleSystem->GetParticleBuffer(),
                                     m_config.particleCount,
                                     cameraPosition); // NEW: Pass camera for dynamic emission!

        rtLightingBuffer = m_rtLighting->GetLightingBuffer();

        // Log every 60 frames
        if ((m_frameCount % 60) == 0) {
            LOG_INFO("RT Lighting computed with dynamic emission (frame {})", m_frameCount);
        }
    }

    // Probe Grid Update Pass (Phase 0.13.1)
    // Reuses TLAS from RT lighting system (zero duplication!)
    if (m_probeGridSystem && m_rtLighting && m_particleSystem) {
        // Get light buffer from Gaussian renderer (already populated with 13 lights)
        ID3D12Resource* lightBuffer = nullptr;
        uint32_t lightCount = 0;

        if (m_gaussianRenderer) {
            lightBuffer = m_gaussianRenderer->GetLightBuffer();
            lightCount = static_cast<uint32_t>(m_lights.size());  // 13 lights from Application.h:118
        }

        m_probeGridSystem->UpdateProbes(
            cmdList,
            m_rtLighting->GetTLAS(),
            m_particleSystem->GetParticleBuffer(),
            m_config.particleCount,
            lightBuffer,
            lightCount,
            m_frameCount
        );

        // Log every 60 frames
        if ((m_frameCount % 60) == 0) {
            LOG_INFO("Probe Grid updated (frame {})", m_frameCount);
        }
    }

    // Transition particle buffer from UAV to SRV for rendering
    if (m_particleSystem) {
        D3D12_RESOURCE_BARRIER particleBarrier = {};
        particleBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        particleBarrier.Transition.pResource = m_particleSystem->GetParticleBuffer();
        particleBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        particleBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        particleBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        cmdList->ResourceBarrier(1, &particleBarrier);
    }

    // Render particles (Billboard or Gaussian path)
    if (m_particleSystem) {
        ParticleRenderer::RenderConstants renderConstants = {};

        // Use camera position computed earlier (for consistency with RT lighting)
        DirectX::XMVECTOR cameraPos = DirectX::XMLoadFloat3(&cameraPosition);
        DirectX::XMVECTOR lookAt = DirectX::XMVectorSet(0, 0, 0, 1.0f);
        DirectX::XMVECTOR up = DirectX::XMVectorSet(0, 1, 0, 0);
        DirectX::XMMATRIX view = DirectX::XMMatrixLookAtLH(cameraPos, lookAt, up);
        DirectX::XMMATRIX proj = DirectX::XMMatrixPerspectiveFovLH(DirectX::XM_PIDIV4,
                                                                    static_cast<float>(m_width) / static_cast<float>(m_height),
                                                                    0.1f, 10000.0f);
        // DON'T transpose - HLSL uses row-major by default, DirectXMath is row-major
        DirectX::XMStoreFloat4x4(&renderConstants.viewProj, view * proj);

        // Use runtime camera controls (precomputed for RT lighting consistency)
        renderConstants.cameraPos = cameraPosition;
        renderConstants.cameraUp = DirectX::XMFLOAT3(0, 1, 0);
        renderConstants.time = m_totalTime;
        renderConstants.particleSize = m_particleSize;
        renderConstants.screenWidth = m_width;
        renderConstants.screenHeight = m_height;

        // Physical emission toggles and strengths
        renderConstants.usePhysicalEmission = m_usePhysicalEmission;
        renderConstants.emissionStrength = m_emissionStrength;
        renderConstants.useDopplerShift = m_useDopplerShift;
        renderConstants.dopplerStrength = m_dopplerStrength;
        renderConstants.useGravitationalRedshift = m_useGravitationalRedshift;
        renderConstants.redshiftStrength = m_redshiftStrength;

        // Log camera view on first frame
        static bool loggedCamera = false;
        if (!loggedCamera) {
            LOG_INFO("=== Camera Configuration ===");
            LOG_INFO("  Position: ({}, {}, {})", camX, camY + m_cameraHeight, camZ);
            LOG_INFO("  Looking at: (0, 0, 0)");
            LOG_INFO("  Disk: inner r={}, outer r={}", 10.0f, 300.0f);
            LOG_INFO("  Mouse Look: CTRL+LMB drag");
            LOG_INFO("  Particle size: {}", m_particleSize);
            LOG_INFO("=== Gaussian RT Controls ===");
            LOG_INFO("  F5: Shadow Rays [{}]", m_useShadowRays ? "ON" : "OFF");
            LOG_INFO("  F6: In-Scattering [{}]", m_useInScattering ? "ON" : "OFF");
            LOG_INFO("  F7: Phase Function [{}]", m_usePhaseFunction ? "ON" : "OFF");
            LOG_INFO("  F8/Shift+F8: Phase Strength [{:.1f}]", m_phaseStrength);
            LOG_INFO("============================");
            loggedCamera = true;
        }

        // Choose rendering path
        if (m_gaussianRenderer) {
            // 3D Gaussian Splatting path
            ParticleRenderer_Gaussian::RenderConstants gaussianConstants = {};

            // Get actual render resolution (may differ from window if DLSS is enabled)
            uint32_t renderWidth = m_gaussianRenderer->GetRenderWidth();
            uint32_t renderHeight = m_gaussianRenderer->GetRenderHeight();
            float renderAspect = static_cast<float>(renderWidth) / static_cast<float>(renderHeight);

            // Recalculate view-projection with render resolution aspect ratio (CRITICAL for DLSS!)
            DirectX::XMMATRIX view = DirectX::XMMatrixLookAtLH(
                DirectX::XMLoadFloat3(&renderConstants.cameraPos),
                DirectX::XMVectorSet(0, 0, 0, 1.0f),
                DirectX::XMVectorSet(0, 1, 0, 0)
            );
            DirectX::XMMATRIX proj = DirectX::XMMatrixPerspectiveFovLH(
                DirectX::XM_PIDIV4,  // 45 degrees FOV
                renderAspect,        // Render aspect ratio (not window!)
                1.0f,                // Near plane
                10000.0f             // Far plane
            );
            DirectX::XMMATRIX viewProj = DirectX::XMMatrixMultiply(view, proj);
            DirectX::XMStoreFloat4x4(&gaussianConstants.viewProj, viewProj);

            // Calculate inverse view-projection for ray generation
            DirectX::XMVECTOR det;
            DirectX::XMMATRIX invViewProjMat = DirectX::XMMatrixInverse(&det, viewProj);
            DirectX::XMStoreFloat4x4(&gaussianConstants.invViewProj, invViewProjMat);

            // Camera vectors
            gaussianConstants.cameraPos = renderConstants.cameraPos;
            DirectX::XMVECTOR camPos = DirectX::XMLoadFloat3(&renderConstants.cameraPos);
            DirectX::XMVECTOR target = DirectX::XMVectorSet(0, 0, 0, 1.0f);
            DirectX::XMVECTOR upVec = DirectX::XMVectorSet(0, 1, 0, 0);
            DirectX::XMVECTOR forward = DirectX::XMVector3Normalize(DirectX::XMVectorSubtract(target, camPos));
            DirectX::XMVECTOR right = DirectX::XMVector3Normalize(DirectX::XMVector3Cross(upVec, forward));
            DirectX::XMVECTOR up = DirectX::XMVector3Cross(forward, right);
            DirectX::XMStoreFloat3(&gaussianConstants.cameraRight, right);
            DirectX::XMStoreFloat3(&gaussianConstants.cameraUp, up);
            DirectX::XMStoreFloat3(&gaussianConstants.cameraForward, forward);

            // Other parameters
            gaussianConstants.particleRadius = renderConstants.particleSize;
            gaussianConstants.time = renderConstants.time;
            gaussianConstants.screenWidth = renderWidth;   // Use render resolution (not window!)
            gaussianConstants.screenHeight = renderHeight; // Use render resolution (not window!)
            gaussianConstants.fovY = DirectX::XM_PIDIV4; // 45 degrees
            gaussianConstants.aspectRatio = renderAspect;   // Use render aspect ratio (not window!)
            gaussianConstants.particleCount = m_config.particleCount;
            gaussianConstants.usePhysicalEmission = renderConstants.usePhysicalEmission ? 1u : 0u;
            gaussianConstants.emissionStrength = renderConstants.emissionStrength;
            gaussianConstants.useDopplerShift = renderConstants.useDopplerShift ? 1u : 0u;
            gaussianConstants.dopplerStrength = renderConstants.dopplerStrength;
            gaussianConstants.useGravitationalRedshift = renderConstants.useGravitationalRedshift ? 1u : 0u;
            gaussianConstants.redshiftStrength = renderConstants.redshiftStrength;
            gaussianConstants.emissionBlendFactor = m_emissionBlendFactor;
            gaussianConstants.padding2 = 0.0f;

            // RT system toggles
            gaussianConstants.useShadowRays = m_useShadowRays ? 1u : 0u;
            gaussianConstants.useInScattering = m_useInScattering ? 1u : 0u;
            gaussianConstants.usePhaseFunction = m_usePhaseFunction ? 1u : 0u;
            gaussianConstants.phaseStrength = m_phaseStrength;
            gaussianConstants.inScatterStrength = m_inScatterStrength;

            // CRITICAL FIX: Disable RT particle-to-particle lighting when RTXDI is active
            // RTXDI provides external lighting, so RT lighting would be redundant and incorrect
            bool useRTLighting = m_enableRTLighting && (m_lightingSystem != LightingSystem::RTXDI);
            gaussianConstants.rtLightingStrength = useRTLighting ? m_rtLightingStrength : 0.0f;
            gaussianConstants.useAnisotropicGaussians = m_useAnisotropicGaussians ? 1u : 0u;
            gaussianConstants.anisotropyStrength = m_anisotropyStrength;

            // Multi-light system (Phase 3.5)
            gaussianConstants.lightCount = static_cast<uint32_t>(m_lights.size());
            m_gaussianRenderer->UpdateLights(m_lights);

            // PCSS soft shadow system
            gaussianConstants.shadowRaysPerLight = m_shadowRaysPerLight;
            gaussianConstants.enableTemporalFiltering = m_enableTemporalFiltering ? 1u : 0u;
            gaussianConstants.temporalBlend = m_temporalBlend;

            // RTXDI lighting system (Phase 4)
            gaussianConstants.useRTXDI = (m_lightingSystem == LightingSystem::RTXDI) ? 1u : 0u;
            gaussianConstants.debugRTXDISelection = m_debugRTXDISelection ? 1u : 0u;
            gaussianConstants.debugPadding = DirectX::XMFLOAT3(0, 0, 0);

            // God ray system (Phase 5 Milestone 5.3c)
            gaussianConstants.godRayDensity = m_godRayDensity;
            gaussianConstants.godRayStepMultiplier = m_godRayStepMultiplier;
            gaussianConstants.godRayPadding = DirectX::XMFLOAT2(0, 0);

            // Phase 1 Lighting Fix: Global ambient to prevent completely black particles
            gaussianConstants.rtMinAmbient = m_rtMinAmbient;
            gaussianConstants.lightingPadding = DirectX::XMFLOAT3(0, 0, 0);

            // Phase 1.5 Adaptive Particle Radius: Fix overlap artifacts and sparse coverage
            gaussianConstants.enableAdaptiveRadius = m_enableAdaptiveRadius ? 1u : 0u;
            gaussianConstants.adaptiveInnerZone = m_adaptiveInnerZone;
            gaussianConstants.adaptiveOuterZone = m_adaptiveOuterZone;
            gaussianConstants.adaptiveInnerScale = m_adaptiveInnerScale;
            gaussianConstants.adaptiveOuterScale = m_adaptiveOuterScale;
            gaussianConstants.densityScaleMin = m_densityScaleMin;
            gaussianConstants.densityScaleMax = m_densityScaleMax;
            gaussianConstants.adaptivePadding = 0.0f;

            // Phase 3.9 Volumetric RT Lighting: Per-sample-point evaluation for smooth volumetric glow
            gaussianConstants.volumetricRTSamples = m_volumetricRTSamples;
            gaussianConstants.volumetricRTDistance = m_volumetricRTDistance;
            gaussianConstants.volumetricRTAttenuation = m_volumetricRTAttenuation;

            // Phase 0.13.1 Probe Grid System: Zero-atomic-contention volumetric lighting
            gaussianConstants.useProbeGrid = m_useProbeGrid;  // ImGui toggle in "Probe Grid (Phase 0.13.1)" section
            gaussianConstants.probeGridPadding2 = DirectX::XMFLOAT3(0.0f, 0.0f, 0.0f);
            gaussianConstants.useVolumetricRT = m_useVolumetricRT ? 1u : 0u;
            gaussianConstants.volumetricRTIntensity = m_volumetricRTIntensity;
            gaussianConstants.volumetricRTPadding = DirectX::XMFLOAT3(0.0f, 0.0f, 0.0f);

            // Debug: Log RT toggle values once
            static bool loggedToggles = false;
            if (!loggedToggles) {
                LOG_INFO("=== DEBUG: Gaussian Constants ===");
                LOG_INFO("  useShadowRays: {}", gaussianConstants.useShadowRays);
                LOG_INFO("  useInScattering: {}", gaussianConstants.useInScattering);
                LOG_INFO("  usePhaseFunction: {}", gaussianConstants.usePhaseFunction);
                LOG_INFO("  phaseStrength: {}", gaussianConstants.phaseStrength);
                LOG_INFO("  useRTXDI: {}", gaussianConstants.useRTXDI);
                LOG_INFO("================================");
                loggedToggles = true;
            }

            // Get RTXDI output buffer (if RTXDI mode is enabled)
            // Use M5 accumulated buffer (temporally smoothed) instead of raw M4 debug output
            ID3D12Resource* rtxdiOutput = nullptr;
            if (m_lightingSystem == LightingSystem::RTXDI && m_rtxdiLightingSystem) {
                rtxdiOutput = m_rtxdiLightingSystem->GetAccumulatedBuffer();  // M5 temporal accumulation output
            }

            // VolumetricReSTIR: Replace entire rendering with path tracing
            if (m_lightingSystem == LightingSystem::VolumetricReSTIR && m_volumetricReSTIR) {
                // Extract camera matrices
                DirectX::XMMATRIX view = DirectX::XMMatrixLookAtLH(
                    DirectX::XMLoadFloat3(&renderConstants.cameraPos),
                    DirectX::XMVectorSet(0, 0, 0, 1.0f),  // Look at origin
                    DirectX::XMVectorSet(0, 1, 0, 0)      // Up vector
                );
                DirectX::XMMATRIX proj = DirectX::XMMatrixPerspectiveFovLH(
                    DirectX::XM_PIDIV4,  // 45° FOV
                    renderAspect,
                    0.1f,   // Near plane
                    10000.0f  // Far plane
                );

                // Populate Volume Mip 2 with particle density (T* transmittance)
                // RE-ENABLED: Root signature verified correct (4/4 resources match DXIL)
                // Testing with diagnostic tools to find shader execution issue
                m_volumetricReSTIR->PopulateVolumeMip2(
                    reinterpret_cast<ID3D12GraphicsCommandList4*>(cmdList),
                    m_particleSystem->GetParticleBuffer(),
                    m_config.particleCount
                );

                // Generate path candidates (Phase 1: RIS only)
                m_volumetricReSTIR->GenerateCandidates(
                    reinterpret_cast<ID3D12GraphicsCommandList4*>(cmdList),
                    m_rtLighting ? m_rtLighting->GetTLAS() : nullptr,  // Particle TLAS
                    m_particleSystem->GetParticleBuffer(),
                    m_config.particleCount,
                    renderConstants.cameraPos,
                    view,
                    proj,
                    m_frameCount  // Frame index for RNG seed
                );

                // Get final output texture (DLSS upscaled if enabled, otherwise render-res)
                ID3D12Resource* finalOutputTexture = m_gaussianRenderer->GetFinalOutputTexture();
                D3D12_GPU_DESCRIPTOR_HANDLE finalOutputUAV = m_gaussianRenderer->GetFinalOutputUAV();

                // Shade selected paths to output texture
                // Now writes to correct texture (DLSS upscaled if enabled)
                m_volumetricReSTIR->ShadeSelectedPaths(
                    reinterpret_cast<ID3D12GraphicsCommandList4*>(cmdList),
                    finalOutputTexture,  // Write to DLSS upscaled texture or render-res
                    finalOutputUAV,      // GPU descriptor handle for UAV
                    m_rtLighting ? m_rtLighting->GetTLAS() : nullptr,  // Particle TLAS
                    m_particleSystem->GetParticleBuffer(),
                    m_config.particleCount,
                    renderConstants.cameraPos,
                    view,
                    proj
                );
            } else {
                // Standard Gaussian volumetric rendering
                // Render to UAV texture
                m_gaussianRenderer->Render(reinterpret_cast<ID3D12GraphicsCommandList4*>(cmdList),
                                      m_particleSystem->GetParticleBuffer(),
                                      rtLightingBuffer,
                                      m_rtLighting ? m_rtLighting->GetTLAS() : nullptr,
                                      gaussianConstants,
                                      rtxdiOutput,  // RTXDI output buffer
                                      m_probeGridSystem.get());  // Probe Grid System (Phase 0.13.1)
            }  // End else (standard Gaussian rendering)

            // HDR→SDR blit pass (replaces CopyTextureRegion) - shared by Gaussian and VolumetricReSTIR
            D3D12_RESOURCE_BARRIER blitBarriers[2] = {};

            // Transition output texture (HDR) from UAV to SRV for sampling
            // For VolumetricReSTIR: use final output (DLSS upscaled if enabled)
            // For Gaussian: use GetOutputTexture() (DLSS handles its own transitions)
            ID3D12Resource* blitSourceTexture = (m_lightingSystem == LightingSystem::VolumetricReSTIR)
                ? m_gaussianRenderer->GetFinalOutputTexture()
                : m_gaussianRenderer->GetOutputTexture();

            blitBarriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            blitBarriers[0].Transition.pResource = blitSourceTexture;
            blitBarriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            blitBarriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            blitBarriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

            // Backbuffer already in RENDER_TARGET state from earlier in Render()
            cmdList->ResourceBarrier(1, &blitBarriers[0]);

            // Set blit pipeline
            cmdList->SetPipelineState(m_blitPSO.Get());
            cmdList->SetGraphicsRootSignature(m_blitRootSignature.Get());

            // Set render target (backbuffer already set earlier)
            cmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);

            // Set viewport and scissor
            D3D12_VIEWPORT blitViewport = { 0, 0, static_cast<float>(m_width), static_cast<float>(m_height), 0.0f, 1.0f };
            D3D12_RECT blitScissor = { 0, 0, static_cast<LONG>(m_width), static_cast<LONG>(m_height) };
            cmdList->RSSetViewports(1, &blitViewport);
            cmdList->RSSetScissorRects(1, &blitScissor);

            // Set descriptor heap for SRV
            ID3D12DescriptorHeap* descriptorHeaps[] = { m_resources->GetDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) };
            cmdList->SetDescriptorHeaps(1, descriptorHeaps);

            // Bind HDR texture SRV (t0)
            D3D12_GPU_DESCRIPTOR_HANDLE srvHandle = m_gaussianRenderer->GetOutputSRV();

            // DEBUG: Log SRV binding
            static bool loggedSRV = false;
            if (m_lightingSystem == LightingSystem::VolumetricReSTIR && !loggedSRV) {
                LOG_INFO("VolumetricReSTIR: Binding output SRV 0x{:016X} to blit shader", srvHandle.ptr);
                loggedSRV = true;
            }

            cmdList->SetGraphicsRootDescriptorTable(0, srvHandle);

            // Draw fullscreen triangle
            cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

            // DEBUG: Log blit draw call
            static bool loggedDraw = false;
            if (m_lightingSystem == LightingSystem::VolumetricReSTIR && !loggedDraw) {
                LOG_INFO("VolumetricReSTIR: About to DrawInstanced(3, 1, 0, 0) for blit");
                loggedDraw = true;
            }

            cmdList->DrawInstanced(3, 1, 0, 0);

            static bool loggedDrawDone = false;
            if (m_lightingSystem == LightingSystem::VolumetricReSTIR && !loggedDrawDone) {
                LOG_INFO("VolumetricReSTIR: DrawInstanced completed");
                loggedDrawDone = true;
            }

            // Transition Gaussian output back to UAV for next frame
            blitBarriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            blitBarriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            cmdList->ResourceBarrier(1, &blitBarriers[0]);

            // DEBUG: Log completion of blit pass
            static bool loggedBlitComplete = false;
            if (m_lightingSystem == LightingSystem::VolumetricReSTIR && !loggedBlitComplete) {
                LOG_INFO("VolumetricReSTIR: HDR->SDR blit pass completed successfully");
                loggedBlitComplete = true;
            }

        } else if (m_particleRenderer) {
            // Billboard path (current/stable)
            m_particleRenderer->Render(cmdList,
                                      m_particleSystem->GetParticleBuffer(),
                                      rtLightingBuffer,
                                      renderConstants);
        }
    }

    // Transition particle buffer back to UAV for next frame's physics/RT lighting
    if (m_particleSystem) {
        D3D12_RESOURCE_BARRIER particleBackToUAV = {};
        particleBackToUAV.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        particleBackToUAV.Transition.pResource = m_particleSystem->GetParticleBuffer();
        particleBackToUAV.Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        particleBackToUAV.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        particleBackToUAV.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        cmdList->ResourceBarrier(1, &particleBackToUAV);
    }

    // Render ImGui overlay
    if (m_showImGui) {
        // Set ImGui descriptor heap
        ID3D12DescriptorHeap* descriptorHeaps[] = { m_imguiDescriptorHeap.Get() };
        cmdList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);

        // Render ImGui commands
        RenderImGui();
        ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), cmdList);
    }

    // Transition to present
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    cmdList->ResourceBarrier(1, &barrier);

    // Close and execute
    cmdList->Close();

    // DEBUG: Log before ExecuteCommandList
    static bool loggedExecute = false;
    if (m_lightingSystem == LightingSystem::VolumetricReSTIR && !loggedExecute) {
        LOG_INFO("VolumetricReSTIR: About to ExecuteCommandList");
        loggedExecute = true;
    }

    m_device->ExecuteCommandList();

    // DEBUG: Log after ExecuteCommandList
    static bool loggedExecuteDone = false;
    if (m_lightingSystem == LightingSystem::VolumetricReSTIR && !loggedExecuteDone) {
        LOG_INFO("VolumetricReSTIR: ExecuteCommandList completed");
        loggedExecuteDone = true;
    }

    // Present
    static bool loggedPresent = false;
    if (m_lightingSystem == LightingSystem::VolumetricReSTIR && !loggedPresent) {
        LOG_INFO("VolumetricReSTIR: About to Present");
        loggedPresent = true;
    }

    m_swapChain->Present(0);

    static bool loggedPresentDone = false;
    if (m_lightingSystem == LightingSystem::VolumetricReSTIR && !loggedPresentDone) {
        LOG_INFO("VolumetricReSTIR: Present completed");
        loggedPresentDone = true;
    }

    // Reset upload heap for next frame (MUST be called before WaitForGPU)
    m_resources->ResetUploadHeap();

    // Wait for frame completion (simple sync for now)
    static bool loggedWait = false;
    if (m_lightingSystem == LightingSystem::VolumetricReSTIR && !loggedWait) {
        LOG_INFO("VolumetricReSTIR: About to WaitForGPU");
        loggedWait = true;
    }

    m_device->WaitForGPU();

    static bool loggedWaitDone = false;
    if (m_lightingSystem == LightingSystem::VolumetricReSTIR && !loggedWaitDone) {
        LOG_INFO("VolumetricReSTIR: WaitForGPU completed - frame finished!");
        loggedWaitDone = true;
    }

    // Read diagnostic counters for first 5 frames for debugging (GPU work is guaranteed complete)
    if (m_lightingSystem == LightingSystem::VolumetricReSTIR && m_frameCount < 5) {
        LOG_INFO("=== Reading diagnostic counters for frame {} ===", m_frameCount);
        m_volumetricReSTIR->ReadDiagnosticCounters();
    }

    m_frameCount++;

    // Dump GPU buffers if requested (only executes when flag is set)
    if (m_dumpBuffersNextFrame) {
        DumpGPUBuffers();

        // Exit if this was an auto-dump (one-shot capture)
        if (m_dumpTargetFrame > 0) {
            LOG_INFO("Auto-dump complete, exiting...");
            m_isRunning = false;
        }
    }

    // Capture screenshot if requested (F2 key)
    if (m_captureScreenshotNextFrame) {
        CaptureScreenshot();
        m_captureScreenshotNextFrame = false;
    }
}

bool Application::CreateAppWindow(HINSTANCE hInstance, int nCmdShow) {
    // Make the application DPI-aware so Windows doesn't scale our window
    // This prevents the 1920x1080 window from becoming 2880x1620 on 150% scaled displays
    SetProcessDPIAware();

    // Register window class
    WNDCLASSEX wc = {};
    wc.cbSize = sizeof(WNDCLASSEX);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName = L"PlasmaDXClean";

    if (!RegisterClassEx(&wc)) {
        return false;
    }

    // Get monitor resolution to ensure window fits on screen
    MONITORINFO monitorInfo = { sizeof(MONITORINFO) };
    HMONITOR hMonitor = MonitorFromWindow(nullptr, MONITOR_DEFAULTTOPRIMARY);
    GetMonitorInfo(hMonitor, &monitorInfo);

    int monitorWidth = monitorInfo.rcWork.right - monitorInfo.rcWork.left;
    int monitorHeight = monitorInfo.rcWork.bottom - monitorInfo.rcWork.top;

    LOG_INFO("Monitor resolution: {}x{}", monitorWidth, monitorHeight);
    LOG_INFO("Requested window: {}x{}", m_width, m_height);

    // If requested resolution is larger than monitor, scale down to 90% of monitor size
    if (m_width > monitorWidth || m_height > monitorHeight) {
        float scaleW = static_cast<float>(monitorWidth) / m_width * 0.9f;
        float scaleH = static_cast<float>(monitorHeight) / m_height * 0.9f;
        float scale = (scaleW < scaleH) ? scaleW : scaleH;
        m_width = static_cast<int>(m_width * scale);
        m_height = static_cast<int>(m_height * scale);
        LOG_INFO("Scaled window to fit monitor: {}x{}", m_width, m_height);
    }

    // Create window
    RECT rc = { 0, 0, static_cast<LONG>(m_width), static_cast<LONG>(m_height) };
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);

    m_hwnd = CreateWindow(
        L"PlasmaDXClean",
        L"PlasmaDX-Clean - RT Lighting Test",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        rc.right - rc.left, rc.bottom - rc.top,
        nullptr, nullptr, hInstance, this
    );

    if (!m_hwnd) {
        return false;
    }

    ShowWindow(m_hwnd, nCmdShow);
    UpdateWindow(m_hwnd);

    // Start in fullscreen mode (Alt+Enter to toggle)
    ToggleFullscreen();
    LOG_INFO("Starting in fullscreen mode (Alt+Enter to toggle)");

    return true;
}

void Application::ToggleFullscreen() {
    m_isFullscreen = !m_isFullscreen;

    if (m_isFullscreen) {
        // Save windowed position/size
        GetWindowRect(m_hwnd, &m_windowedRect);

        // Get monitor info for fullscreen resolution
        HMONITOR hMonitor = MonitorFromWindow(m_hwnd, MONITOR_DEFAULTTOPRIMARY);
        MONITORINFO monitorInfo = { sizeof(MONITORINFO) };
        GetMonitorInfo(hMonitor, &monitorInfo);

        int monitorWidth = monitorInfo.rcMonitor.right - monitorInfo.rcMonitor.left;
        int monitorHeight = monitorInfo.rcMonitor.bottom - monitorInfo.rcMonitor.top;

        // Set borderless window style
        SetWindowLong(m_hwnd, GWL_STYLE, WS_POPUP | WS_VISIBLE);

        // Move window to cover entire monitor
        SetWindowPos(m_hwnd, HWND_TOP,
                     monitorInfo.rcMonitor.left, monitorInfo.rcMonitor.top,
                     monitorWidth, monitorHeight,
                     SWP_FRAMECHANGED | SWP_SHOWWINDOW);

        LOG_INFO("Fullscreen enabled ({}x{})", monitorWidth, monitorHeight);
    } else {
        // Restore windowed style
        SetWindowLong(m_hwnd, GWL_STYLE, WS_OVERLAPPEDWINDOW | WS_VISIBLE);

        // Restore windowed position/size
        int width = m_windowedRect.right - m_windowedRect.left;
        int height = m_windowedRect.bottom - m_windowedRect.top;

        SetWindowPos(m_hwnd, HWND_NOTOPMOST,
                     m_windowedRect.left, m_windowedRect.top,
                     width, height,
                     SWP_FRAMECHANGED | SWP_SHOWWINDOW);

        LOG_INFO("Fullscreen disabled ({}x{})", m_width, m_height);
    }
}

LRESULT CALLBACK Application::WindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    Application* app = reinterpret_cast<Application*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));

    // Handle Alt+Enter BEFORE ImGui (fullscreen toggle should always work)
    if (msg == WM_SYSKEYDOWN && wParam == VK_RETURN && app) {
        app->ToggleFullscreen();
        return 0;
    }

    // Let ImGui handle input first (but only if it wants it)
    ImGui_ImplWin32_WndProcHandler(hwnd, msg, wParam, lParam);

    switch (msg) {
    case WM_CREATE:
        {
            LPCREATESTRUCT pCreateStruct = reinterpret_cast<LPCREATESTRUCT>(lParam);
            SetWindowLongPtr(hwnd, GWLP_USERDATA,
                           reinterpret_cast<LONG_PTR>(pCreateStruct->lpCreateParams));
        }
        return 0;

    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;

    case WM_SIZE:
        if (app && wParam != SIZE_MINIMIZED) {
            UINT width = LOWORD(lParam);
            UINT height = HIWORD(lParam);

            // Only resize if dimensions actually changed
            if (width != app->m_width || height != app->m_height) {
                app->m_width = width;
                app->m_height = height;

                // Resize swap chain and recreate render targets
                if (app->m_swapChain) {
                    app->m_device->WaitForGPU();  // Wait for GPU before resizing
                    app->m_swapChain->Resize(width, height);

                    // Resize Gaussian renderer output texture and reservoirs
                    if (app->m_gaussianRenderer) {
                        app->m_gaussianRenderer->Resize(width, height);
                    }

                    // Update ImGui display size
                    ImGuiIO& io = ImGui::GetIO();
                    io.DisplaySize = ImVec2(static_cast<float>(width), static_cast<float>(height));

                    LOG_INFO("Window resized: {}x{}", width, height);
                }
            }
        }
        return 0;

    case WM_KEYDOWN:
        if (app) {
            // Always allow F-keys and special keys even if ImGui wants keyboard
            UINT8 key = static_cast<UINT8>(wParam);
            bool isFunctionKey = (key >= VK_F1 && key <= VK_F12);
            bool isSpecialKey = (key == VK_ESCAPE);

            // Handle function keys and special keys regardless of ImGui state
            if (isFunctionKey || isSpecialKey) {
                app->OnKeyPress(key);
            }
            // For other keys, only handle if ImGui doesn't want keyboard input
            else if (!ImGui::GetIO().WantCaptureKeyboard) {
                app->OnKeyPress(key);
            }
        }
        return 0;

    case WM_LBUTTONDOWN:
        if (app && (GetKeyState(VK_CONTROL) & 0x8000)) {
            app->m_mouseLookActive = true;
            app->m_lastMouseX = LOWORD(lParam);
            app->m_lastMouseY = HIWORD(lParam);
            SetCapture(hwnd);
        }
        return 0;

    case WM_LBUTTONUP:
        if (app && app->m_mouseLookActive) {
            app->m_mouseLookActive = false;
            ReleaseCapture();
        }
        return 0;

    case WM_MOUSEMOVE:
        if (app) {
            int x = LOWORD(lParam);
            int y = HIWORD(lParam);

            if (app->m_mouseLookActive) {
                int dx = x - app->m_lastMouseX;
                int dy = y - app->m_lastMouseY;
                app->OnMouseMove(dx, dy);
                app->m_lastMouseX = x;
                app->m_lastMouseY = y;
            }
        }
        return 0;
    }

    return DefWindowProc(hwnd, msg, wParam, lParam);
}

void Application::OnKeyPress(UINT8 key) {
    switch (key) {
    case VK_ESCAPE:
        m_isRunning = false;
        break;

    case VK_F1:
        m_showImGui = !m_showImGui;
        LOG_INFO("ImGui: {}", m_showImGui ? "ON" : "OFF");
        break;

    case VK_F2:
        m_captureScreenshotNextFrame = true;
        LOG_INFO("Screenshot will be captured next frame (F2)");
        break;

    case 'S':
        // Cycle through ray counts: 2 → 4 → 8 → 16 → 2 ...
        if (m_rtLighting) {
            static const uint32_t rayCounts[] = {2, 4, 8, 16};
            static int rayIndex = 2; // Start at 8 (balanced quality/performance)

            rayIndex = (rayIndex + 1) % 4;
            uint32_t newRayCount = rayCounts[rayIndex];

            m_rtLighting->SetRaysPerParticle(newRayCount);
            LOG_INFO("Rays per particle: {} (S key cycles 2/4/8/16)", newRayCount);
            LOG_INFO("  Quality: {} | Variance: {:.1f}% | Expected FPS: {}",
                     newRayCount >= 16 ? "Ultra" : newRayCount >= 8 ? "High" : newRayCount >= 4 ? "Medium" : "Low",
                     (100.0f / newRayCount) * (100.0f / newRayCount),
                     newRayCount == 2 ? "300+" : newRayCount == 4 ? "250+" : newRayCount == 8 ? "180+" : "140+");
        } else {
            LOG_INFO("RT lighting not initialized");
        }
        break;

    case VK_SPACE:
        LOG_INFO("Frame: {}, FPS: {:.1f}, Render Path: {}",
                m_frameCount, m_fps, m_particleRenderer->GetActivePathName());
        break;

    // Camera controls
    case VK_UP: m_cameraHeight += 50.0f; break;
    case VK_DOWN: m_cameraHeight -= 50.0f; break;
    case VK_LEFT: m_cameraAngle -= 0.1f; break;
    case VK_RIGHT: m_cameraAngle += 0.1f; break;
    case 'W': m_cameraDistance -= 50.0f; break;
    case 'A': m_cameraDistance += 50.0f; break;

    // Particle size / Particle count (Ctrl+Alt modifier for count)
    case VK_OEM_PLUS: case VK_ADD:
        if ((GetAsyncKeyState(VK_CONTROL) & 0x8000) && (GetAsyncKeyState(VK_MENU) & 0x8000)) {
            // Ctrl+Alt+= : Increase active particle count
            m_activeParticleCount = (std::min)(m_config.particleCount, m_activeParticleCount + 1000);
            LOG_INFO("Active Particles: {} / {}", m_activeParticleCount, m_config.particleCount);
        } else {
            // Default: Increase particle size
            m_particleSize += 2.0f;
            LOG_INFO("Particle size: {}", m_particleSize);
        }
        break;
    case VK_OEM_MINUS: case VK_SUBTRACT:
        if ((GetAsyncKeyState(VK_CONTROL) & 0x8000) && (GetAsyncKeyState(VK_MENU) & 0x8000)) {
            // Ctrl+Alt+- : Decrease active particle count
            m_activeParticleCount = (std::max)(100u, m_activeParticleCount - 1000);
            LOG_INFO("Active Particles: {} / {}", m_activeParticleCount, m_config.particleCount);
        } else {
            // Default: Decrease particle size
            m_particleSize = (std::max)(1.0f, m_particleSize - 2.0f);
            LOG_INFO("Particle size: {}", m_particleSize);
        }
        break;

    // Toggle physics (PINN if available, otherwise standard physics)
    case 'P':
        if (m_particleSystem && m_particleSystem->IsPINNAvailable()) {
            // Toggle PINN physics
            m_particleSystem->TogglePINNPhysics();
            LOG_INFO("PINN Physics: {} (Press 'P' to toggle)",
                     m_particleSystem->IsPINNEnabled() ? "ENABLED" : "DISABLED");
            if (m_particleSystem->IsPINNEnabled()) {
                LOG_INFO("  Hybrid Mode: {}", m_particleSystem->IsPINNHybridMode() ? "ON" : "OFF");
                if (m_particleSystem->IsPINNHybridMode()) {
                    LOG_INFO("  Threshold: {:.1f}× R_ISCO", m_particleSystem->GetPINNHybridThreshold());
                }
            }
        } else {
            // Fall back to standard physics toggle
            m_physicsEnabled = !m_physicsEnabled;
            LOG_INFO("Physics: {}", m_physicsEnabled ? "ENABLED" : "DISABLED");
            if (!m_particleSystem || !m_particleSystem->IsPINNAvailable()) {
                LOG_INFO("  (PINN not available - using GPU physics only)");
            }
        }
        break;

    // Debug: Readback particle data (Ctrl+D = buffer dump if enabled)
    case 'D':
        if (m_enableBufferDump && (GetAsyncKeyState(VK_CONTROL) & 0x8000)) {
            // Ctrl+D: Schedule buffer dump for next frame
            m_dumpBuffersNextFrame = true;
            LOG_INFO("=== Buffer dump scheduled for frame {} ===", m_frameCount + 1);
        } else {
            // D alone: Original particle readback behavior
            LOG_INFO("=== DEBUG: Reading back first 10 particles from GPU ===");
            if (m_particleSystem) {
                m_particleSystem->DebugReadbackParticles(10);
            }
        }
        break;

    // Log camera state
    case 'C':
        LOG_INFO("Camera - Distance: {}, Height: {}, Angle: {}, ParticleSize: {}",
                 m_cameraDistance, m_cameraHeight, m_cameraAngle, m_particleSize);
        break;

    // RT Lighting Intensity controls
    case 'I':  // Increase intensity
        m_rtLightingIntensity *= 2.0f;
        if (m_rtLighting) {
            m_rtLighting->SetLightingIntensity(m_rtLightingIntensity);
        }
        LOG_INFO("RT Lighting Intensity: {}", m_rtLightingIntensity);
        break;

    case 'K':  // Decrease intensity
        m_rtLightingIntensity *= 0.5f;
        if (m_rtLighting) {
            m_rtLighting->SetLightingIntensity(m_rtLightingIntensity);
        }
        LOG_INFO("RT Lighting Intensity: {}", m_rtLightingIntensity);
        break;

    // RT Max Distance controls
    case 'O':  // Increase distance
        m_rtMaxDistance += 100.0f;
        if (m_rtLighting) {
            m_rtLighting->SetMaxLightingDistance(m_rtMaxDistance);
        }
        LOG_INFO("RT Max Distance: {}", m_rtMaxDistance);
        break;

    case 'L':  // Decrease distance
        m_rtMaxDistance = (std::max)(50.0f, m_rtMaxDistance - 100.0f);
        if (m_rtLighting) {
            m_rtLighting->SetMaxLightingDistance(m_rtMaxDistance);
        }
        LOG_INFO("RT Max Distance: {}", m_rtMaxDistance);
        break;

    // Physical emission strength (E = toggle, Shift+E = decrease, Ctrl+E = increase)
    case 'E':
        if (GetAsyncKeyState(VK_SHIFT) & 0x8000) {
            // Shift+E: Decrease strength
            m_emissionStrength = (std::max)(0.0f, m_emissionStrength - 0.25f);
            LOG_INFO("Emission Strength: {:.2f}", m_emissionStrength);
        } else if (GetAsyncKeyState(VK_CONTROL) & 0x8000) {
            // Ctrl+E: Increase strength
            m_emissionStrength = (std::min)(5.0f, m_emissionStrength + 0.25f);
            LOG_INFO("Emission Strength: {:.2f}", m_emissionStrength);
        } else {
            // E: Toggle on/off
            m_usePhysicalEmission = !m_usePhysicalEmission;
            LOG_INFO("Physical Emission: {} (strength: {:.2f})",
                    m_usePhysicalEmission ? "ON" : "OFF", m_emissionStrength);
        }
        break;

    // Doppler shift strength (R = toggle, Shift+R = decrease, Ctrl+R = increase)
    case 'R':
        if (GetAsyncKeyState(VK_SHIFT) & 0x8000) {
            m_dopplerStrength = (std::max)(0.0f, m_dopplerStrength - 0.25f);
            LOG_INFO("Doppler Strength: {:.2f}", m_dopplerStrength);
        } else if (GetAsyncKeyState(VK_CONTROL) & 0x8000) {
            m_dopplerStrength = (std::min)(5.0f, m_dopplerStrength + 0.25f);
            LOG_INFO("Doppler Strength: {:.2f}", m_dopplerStrength);
        } else {
            m_useDopplerShift = !m_useDopplerShift;
            LOG_INFO("Doppler Shift: {} (strength: {:.2f})",
                    m_useDopplerShift ? "ON" : "OFF", m_dopplerStrength);
        }
        break;

    // Gravitational redshift strength (G = toggle, Shift+G = decrease, Ctrl+G = increase)
    case 'G':
        if (GetAsyncKeyState(VK_SHIFT) & 0x8000) {
            m_redshiftStrength = (std::max)(0.0f, m_redshiftStrength - 0.25f);
            LOG_INFO("Redshift Strength: {:.2f}", m_redshiftStrength);
        } else if (GetAsyncKeyState(VK_CONTROL) & 0x8000) {
            m_redshiftStrength = (std::min)(5.0f, m_redshiftStrength + 0.25f);
            LOG_INFO("Redshift Strength: {:.2f}", m_redshiftStrength);
        } else {
            m_useGravitationalRedshift = !m_useGravitationalRedshift;
            LOG_INFO("Gravitational Redshift: {} (strength: {:.2f})",
                    m_useGravitationalRedshift ? "ON" : "OFF", m_redshiftStrength);
        }
        break;

    // Cycle RT quality modes
    case 'Q': {
        m_rtQualityMode = (m_rtQualityMode + 1) % 3;
        const char* modes[] = {"Normal", "ReSTIR", "Adaptive"};
        LOG_INFO("RT Quality Mode: {}", modes[m_rtQualityMode]);
        break;
    }

    // F3: Toggle RTXDI vs Multi-Light (Phase 4 M4 Phase 2)
    case VK_F3:
        m_lightingSystem = (m_lightingSystem == LightingSystem::RTXDI)
            ? LightingSystem::MultiLight
            : LightingSystem::RTXDI;
        LOG_INFO("Lighting System: {}", m_lightingSystem == LightingSystem::RTXDI ? "RTXDI" : "Multi-Light");
        break;

#ifdef ENABLE_DLSS
    // F4: Toggle DLSS Ray Reconstruction (AI denoising for shadow rays)
    case VK_F4:
        m_enableDLSS = !m_enableDLSS;
        if (m_enableDLSS && !m_dlssSystem) {
            LOG_WARN("DLSS: Cannot enable - DLSS system not initialized");
            LOG_WARN("  Check: Driver 531.00+, RTX GPU, nvngx_dlssd.dll in exe directory");
            m_enableDLSS = false;
        } else {
            LOG_INFO("DLSS Ray Reconstruction: {}", m_enableDLSS ? "ENABLED" : "DISABLED");
            if (m_enableDLSS) {
                LOG_INFO("  AI denoising for shadow rays");
                LOG_INFO("  Denoiser strength: {:.1f} (adjust in ImGui)", m_dlssDenoiserStrength);
            }
        }
        break;
#endif

    // F5: Toggle shadow rays
    case VK_F5:
        m_useShadowRays = !m_useShadowRays;
        LOG_INFO("Shadow Rays: {}", m_useShadowRays ? "ON" : "OFF");
        break;

    // F6: Toggle in-scattering
    case VK_F6:
        m_useInScattering = !m_useInScattering;
        LOG_INFO("In-Scattering: {}", m_useInScattering ? "ON" : "OFF");
        break;

    // F7: Toggle Volumetric ReSTIR (for autonomous testing)
    case VK_F7:
        if (m_volumetricReSTIR) {
            // Toggle between VolumetricReSTIR and MultiLight
            if (m_lightingSystem == LightingSystem::VolumetricReSTIR) {
                m_lightingSystem = LightingSystem::MultiLight;
                LOG_INFO("Volumetric ReSTIR: OFF (switched to MultiLight)");
            } else {
                m_lightingSystem = LightingSystem::VolumetricReSTIR;
                LOG_INFO("Volumetric ReSTIR: ON");
            }
        } else {
            LOG_INFO("Volumetric ReSTIR system not initialized");
        }
        break;

    // F8: Toggle phase function (Ctrl+F8/Shift+F8 adjust strength)
    case VK_F8:
        if (GetAsyncKeyState(VK_CONTROL) & 0x8000) {
            m_phaseStrength = (std::min)(20.0f, m_phaseStrength + 1.0f);
            LOG_INFO("Phase Strength: {:.1f}", m_phaseStrength);
        } else if (GetAsyncKeyState(VK_SHIFT) & 0x8000) {
            m_phaseStrength = (std::max)(0.0f, m_phaseStrength - 1.0f);
            LOG_INFO("Phase Strength: {:.1f}", m_phaseStrength);
        } else {
            m_usePhaseFunction = !m_usePhaseFunction;
            LOG_INFO("Phase Function: {}", m_usePhaseFunction ? "ON" : "OFF");
        }
        break;

    // F9: Adjust in-scattering strength (Shift+F9 decrease, F9 increase)
    case VK_F9:
        if (GetAsyncKeyState(VK_SHIFT) & 0x8000) {
            m_inScatterStrength = (std::max)(0.0f, m_inScatterStrength - 0.5f);
        } else {
            m_inScatterStrength = (std::min)(10.0f, m_inScatterStrength + 0.5f);
        }
        LOG_INFO("In-Scattering Strength: {:.1f}", m_inScatterStrength);
        break;

    // F10: Adjust RT lighting strength (Shift+F10 decrease, F10 increase)
    case VK_F10:
        if (GetAsyncKeyState(VK_SHIFT) & 0x8000) {
            m_rtLightingStrength = (std::max)(0.0f, m_rtLightingStrength - 0.25f);
        } else {
            m_rtLightingStrength = (std::min)(10.0f, m_rtLightingStrength + 0.25f);
        }
        LOG_INFO("RT Lighting Strength: {:.2f}", m_rtLightingStrength);
        break;

    // F11: Toggle anisotropic Gaussians (velocity-stretched particles)
    case VK_F11:
        m_useAnisotropicGaussians = !m_useAnisotropicGaussians;
        LOG_INFO("Anisotropic Gaussians: {} ({})",
                 m_useAnisotropicGaussians ? "ON" : "OFF",
                 m_useAnisotropicGaussians ? "particles stretch with velocity" : "spherical particles");
        break;

    // F12: Adjust anisotropy strength (Shift+F12 decrease, F12 increase)
    case VK_F12:
        if (GetAsyncKeyState(VK_SHIFT) & 0x8000) {
            m_anisotropyStrength = (std::max)(0.0f, m_anisotropyStrength - 0.2f);
        } else {
            m_anisotropyStrength = (std::min)(3.0f, m_anisotropyStrength + 0.2f);
        }
        LOG_INFO("Anisotropy Strength: {:.1f} (0=spherical, 3=max stretch)", m_anisotropyStrength);
        break;

    // Physics controls: Gravity (V = velocity/gravity)
    case 'V':
        if (GetAsyncKeyState(VK_SHIFT) & 0x8000) {
            m_particleSystem->AdjustGravityStrength(-50.0f);
            LOG_INFO("Gravity Strength: {:.1f}", m_particleSystem->GetGravityStrength());
        } else if (GetAsyncKeyState(VK_CONTROL) & 0x8000) {
            m_particleSystem->AdjustGravityStrength(50.0f);
            LOG_INFO("Gravity Strength: {:.1f}", m_particleSystem->GetGravityStrength());
        }
        break;

    // aNgular momentum / orbital speed (N)
    case 'N':
        if (GetAsyncKeyState(VK_SHIFT) & 0x8000) {
            m_particleSystem->AdjustAngularMomentum(-0.1f);
            LOG_INFO("Angular Momentum: {:.2f}", m_particleSystem->GetAngularMomentum());
        } else if (GetAsyncKeyState(VK_CONTROL) & 0x8000) {
            m_particleSystem->AdjustAngularMomentum(0.1f);
            LOG_INFO("Angular Momentum: {:.2f}", m_particleSystem->GetAngularMomentum());
        }
        break;

    // Turbulence (B = brownian motion)
    case 'B':
        if (GetAsyncKeyState(VK_SHIFT) & 0x8000) {
            m_particleSystem->AdjustTurbulence(-2.0f);
            LOG_INFO("Turbulence: {:.1f}", m_particleSystem->GetTurbulence());
        } else if (GetAsyncKeyState(VK_CONTROL) & 0x8000) {
            m_particleSystem->AdjustTurbulence(2.0f);
            LOG_INFO("Turbulence: {:.1f}", m_particleSystem->GetTurbulence());
        }
        break;

    // Damping (M = momentum damping)
    case 'M':
        if (GetAsyncKeyState(VK_SHIFT) & 0x8000) {
            m_particleSystem->AdjustDamping(-0.01f);
            LOG_INFO("Damping: {:.3f}", m_particleSystem->GetDamping());
        } else if (GetAsyncKeyState(VK_CONTROL) & 0x8000) {
            m_particleSystem->AdjustDamping(0.01f);
            LOG_INFO("Damping: {:.3f}", m_particleSystem->GetDamping());
        }
        break;

    // Black Hole Mass (H = black hole mass)
    case 'H':
        if (GetAsyncKeyState(VK_SHIFT) & 0x8000) {
            // Decrease mass (logarithmic: divide by 10)
            float currentMass = m_particleSystem->GetBlackHoleMass();
            float newMass = currentMass / 10.0f;
            m_particleSystem->SetBlackHoleMass(newMass);
            if (newMass < 1e3f) {
                LOG_INFO("Black Hole Mass: {:.1f} M☉", newMass);
            } else if (newMass < 1e6f) {
                LOG_INFO("Black Hole Mass: {:.1f} thousand M☉", newMass / 1e3f);
            } else if (newMass < 1e9f) {
                LOG_INFO("Black Hole Mass: {:.2f} million M☉", newMass / 1e6f);
            } else {
                LOG_INFO("Black Hole Mass: {:.2f} billion M☉", newMass / 1e9f);
            }
        } else if (GetAsyncKeyState(VK_CONTROL) & 0x8000) {
            // Increase mass (logarithmic: multiply by 10)
            float currentMass = m_particleSystem->GetBlackHoleMass();
            float newMass = currentMass * 10.0f;
            if (newMass > 1e10f) newMass = 1e10f;  // Cap at 10 billion solar masses
            m_particleSystem->SetBlackHoleMass(newMass);
            if (newMass < 1e3f) {
                LOG_INFO("Black Hole Mass: {:.1f} M☉", newMass);
            } else if (newMass < 1e6f) {
                LOG_INFO("Black Hole Mass: {:.1f} thousand M☉", newMass / 1e3f);
            } else if (newMass < 1e9f) {
                LOG_INFO("Black Hole Mass: {:.2f} million M☉", newMass / 1e6f);
            } else {
                LOG_INFO("Black Hole Mass: {:.2f} billion M☉", newMass / 1e9f);
            }
        }
        break;

    // Alpha Viscosity (X = viscosity)
    case 'X':
        if (GetAsyncKeyState(VK_SHIFT) & 0x8000) {
            m_particleSystem->AdjustAlphaViscosity(-0.01f);
            LOG_INFO("Alpha Viscosity: {:.3f}", m_particleSystem->GetAlphaViscosity());
        } else if (GetAsyncKeyState(VK_CONTROL) & 0x8000) {
            m_particleSystem->AdjustAlphaViscosity(0.01f);
            LOG_INFO("Alpha Viscosity: {:.3f}", m_particleSystem->GetAlphaViscosity());
        }
        break;

    // Time Scale (T = time)
    case 'T':
        if (GetAsyncKeyState(VK_SHIFT) & 0x8000) {
            m_particleSystem->AdjustTimeScale(-0.1f);
            LOG_INFO("Time Scale: {:.2f}x", m_particleSystem->GetTimeScale());
        } else if (GetAsyncKeyState(VK_CONTROL) & 0x8000) {
            m_particleSystem->AdjustTimeScale(0.1f);
            LOG_INFO("Time Scale: {:.2f}x", m_particleSystem->GetTimeScale());
        }
        break;

    // Multi-Light System Controls
    case VK_OEM_6:  // ] key - Add light at camera position
        if (m_lights.size() < 16) {
            ParticleRenderer_Gaussian::Light newLight;
            newLight.position = DirectX::XMFLOAT3(
                m_cameraDistance * sinf(m_cameraAngle),
                m_cameraHeight,
                m_cameraDistance * cosf(m_cameraAngle)
            );
            newLight.color = DirectX::XMFLOAT3(1.0f, 1.0f, 1.0f);
            newLight.intensity = 5.0f;
            newLight.radius = 10.0f;
            m_lights.push_back(newLight);
            m_selectedLightIndex = static_cast<int>(m_lights.size()) - 1;
            LOG_INFO("Added light {} at camera position ({:.1f}, {:.1f}, {:.1f})",
                     m_lights.size() - 1, newLight.position.x, newLight.position.y, newLight.position.z);
        } else {
            LOG_INFO("Maximum light count (16) reached");
        }
        break;

    case VK_OEM_4:  // [ key - Remove selected light
        if (!m_lights.empty()) {
            int indexToRemove = m_selectedLightIndex >= 0 ? m_selectedLightIndex : static_cast<int>(m_lights.size()) - 1;
            if (indexToRemove >= 0 && indexToRemove < static_cast<int>(m_lights.size())) {
                m_lights.erase(m_lights.begin() + indexToRemove);
                m_selectedLightIndex = -1;
                LOG_INFO("Removed light {}, {} lights remaining", indexToRemove, m_lights.size());
            }
        } else {
            LOG_INFO("No lights to remove");
        }
        break;

    }
}

void Application::OnMouseMove(int dx, int dy) {
    // CTRL+LMB drag = mouse look
    const float sensitivity = 0.005f;
    m_cameraAngle += dx * sensitivity;
    m_cameraPitch += dy * sensitivity;

    // Clamp pitch to avoid gimbal lock
    m_cameraPitch = (std::max)(-1.5f, (std::min)(1.5f, m_cameraPitch));
}

void Application::DumpGPUBuffers() {
    LOG_INFO("\n=== DUMPING GPU BUFFERS (Frame {}) ===", m_frameCount);

    // Create output directory
    std::filesystem::create_directories(m_dumpOutputDir);

    // Dump buffers (only if they exist)
    if (m_particleSystem) {
        DumpBufferToFile(m_particleSystem->GetParticleBuffer(), "g_particles");
    }


    if (m_rtLighting) {
        auto rtBuffer = m_rtLighting->GetLightingBuffer();
        if (rtBuffer) {
            DumpBufferToFile(rtBuffer, "g_rtLighting");
        }
    }

    // Dump RTXDI buffers if using RTXDI path
    if (m_lightingSystem == LightingSystem::RTXDI && m_rtxdiLightingSystem) {
        LOG_INFO("  Dumping RTXDI buffers...");
        auto cmdList = m_device->GetCommandList();
        m_device->ResetCommandList();
        m_rtxdiLightingSystem->DumpBuffers(cmdList, m_dumpOutputDir, m_frameCount);
        cmdList->Close();
        m_device->ExecuteCommandList();
        m_device->WaitForGPU();
    }

    // Write metadata JSON
    WriteMetadataJSON();

    LOG_INFO("=== BUFFER DUMP COMPLETE ===\n");

    // Reset flag
    m_dumpBuffersNextFrame = false;
}

void Application::DumpBufferToFile(ID3D12Resource* buffer, const char* name) {
    if (!buffer) {
        LOG_WARN("Buffer {} is null, skipping", name);
        return;
    }

    D3D12_RESOURCE_DESC desc = buffer->GetDesc();
    UINT64 bufferSize = desc.Width;

    LOG_INFO("  Dumping {} ({} bytes)...", name, bufferSize);

    // Create readback buffer
    D3D12_HEAP_PROPERTIES readbackHeapProps = {};
    readbackHeapProps.Type = D3D12_HEAP_TYPE_READBACK;
    readbackHeapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    readbackHeapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

    D3D12_RESOURCE_DESC readbackDesc = {};
    readbackDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    readbackDesc.Alignment = 0;
    readbackDesc.Width = bufferSize;
    readbackDesc.Height = 1;
    readbackDesc.DepthOrArraySize = 1;
    readbackDesc.MipLevels = 1;
    readbackDesc.Format = DXGI_FORMAT_UNKNOWN;
    readbackDesc.SampleDesc.Count = 1;
    readbackDesc.SampleDesc.Quality = 0;
    readbackDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    readbackDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    ID3D12Resource* readbackBuffer = nullptr;
    HRESULT hr = m_device->GetDevice()->CreateCommittedResource(
        &readbackHeapProps,
        D3D12_HEAP_FLAG_NONE,
        &readbackDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&readbackBuffer)
    );

    if (FAILED(hr)) {
        LOG_ERROR("    FAILED to create readback buffer");
        return;
    }

    // Reset command list for copy operation
    m_device->ResetCommandList();
    auto cmdList = m_device->GetCommandList();

    // Transition source buffer to COPY_SOURCE
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = buffer;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;  // Most buffers are UAV
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmdList->ResourceBarrier(1, &barrier);

    // Copy GPU buffer → Readback buffer
    cmdList->CopyResource(readbackBuffer, buffer);

    // Transition back to UAV
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    cmdList->ResourceBarrier(1, &barrier);

    // Execute and wait
    cmdList->Close();
    m_device->ExecuteCommandList();
    m_device->WaitForGPU();

    // Map readback buffer and write to file
    void* mappedData = nullptr;
    D3D12_RANGE readRange = { 0, bufferSize };
    hr = readbackBuffer->Map(0, &readRange, &mappedData);

    if (SUCCEEDED(hr)) {
        std::string filepath = m_dumpOutputDir + "/" + name + ".bin";
        FILE* file = fopen(filepath.c_str(), "wb");

        if (file) {
            size_t written = fwrite(mappedData, 1, bufferSize, file);
            fclose(file);

            if (written == bufferSize) {
                LOG_INFO("    ✓ {} ({} bytes)", filepath, bufferSize);
            } else {
                LOG_ERROR("    FAILED (wrote {}/{} bytes)", written, bufferSize);
            }
        } else {
            LOG_ERROR("    FAILED to open file: {}", filepath);
        }

        D3D12_RANGE writeRange = { 0, 0 };  // No writes from CPU
        readbackBuffer->Unmap(0, &writeRange);
    } else {
        LOG_ERROR("    FAILED to map readback buffer");
    }

    // Cleanup
    readbackBuffer->Release();
}

void Application::WriteMetadataJSON() {
    std::string filepath = m_dumpOutputDir + "/metadata.json";
    FILE* file = fopen(filepath.c_str(), "w");

    if (file) {
        // Get timestamp
        auto now = std::chrono::system_clock::now();
        auto timeT = std::chrono::system_clock::to_time_t(now);
        char timestamp[100];
        strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&timeT));

        // Calculate camera position
        float camX = m_cameraDistance * cosf(m_cameraPitch) * sinf(m_cameraAngle);
        float camY = m_cameraDistance * sinf(m_cameraPitch);
        float camZ = m_cameraDistance * cosf(m_cameraPitch) * cosf(m_cameraAngle);
        float camDistance = sqrtf(camX * camX + (camY + m_cameraHeight) * (camY + m_cameraHeight) + camZ * camZ);

        fprintf(file, "{\n");
        fprintf(file, "  \"frame\": %u,\n", m_frameCount);
        fprintf(file, "  \"timestamp\": \"%s\",\n", timestamp);
        fprintf(file, "  \"camera_position\": [%.2f, %.2f, %.2f],\n", camX, camY + m_cameraHeight, camZ);
        fprintf(file, "  \"camera_distance\": %.2f,\n", camDistance);
        fprintf(file, "  \"particle_count\": %u,\n", m_config.particleCount);
        fprintf(file, "  \"particle_size\": %.2f,\n", m_particleSize);
        fprintf(file, "  \"render_mode\": \"%s\",\n",
                m_config.rendererType == RendererType::Gaussian ? "Gaussian" : "Billboard");
        fprintf(file, "  \"use_shadow_rays\": %s,\n", m_useShadowRays ? "true" : "false");
        fprintf(file, "  \"use_in_scattering\": %s,\n", m_useInScattering ? "true" : "false");
        fprintf(file, "  \"use_phase_function\": %s\n", m_usePhaseFunction ? "true" : "false");
        fprintf(file, "}\n");
        fclose(file);

        LOG_INFO("  Wrote metadata: {}", filepath);
    }
}

void Application::DetectQualityPreset(ScreenshotMetadata& meta) {
    // Detect quality preset based on settings (Phase 2 v2.0)
    // Quality tiers from FEATURE_STATUS.md:
    //   Maximum: Any FPS (video/screenshots, not realtime)
    //   Ultra: 30 FPS target
    //   High: 60 FPS target
    //   Medium: 120 FPS target
    //   Low: 165 FPS target

    // Heuristic based on shadow quality and particle count
    bool highQualityShadows = (m_shadowPreset == ShadowPreset::Quality || m_shadowRaysPerLight >= 8);
    bool mediumShadows = (m_shadowPreset == ShadowPreset::Balanced || m_shadowRaysPerLight >= 4);
    bool lowQualityShadows = (m_shadowPreset == ShadowPreset::Performance || m_shadowRaysPerLight <= 1);

    uint32_t particleCount = m_config.particleCount;

    // Ultra quality: High shadows + high particle count
    if (highQualityShadows && particleCount >= 50000) {
        meta.qualityPreset = "Ultra";
        meta.targetFPS = 30.0f;
    }
    // High quality: Medium shadows or medium particle count
    else if (mediumShadows || particleCount >= 25000) {
        meta.qualityPreset = "High";
        meta.targetFPS = 60.0f;
    }
    // Low quality: Low shadows + low particle count
    else if (lowQualityShadows && particleCount <= 5000) {
        meta.qualityPreset = "Low";
        meta.targetFPS = 165.0f;
    }
    // Medium quality (default/most common)
    else {
        meta.qualityPreset = "Medium";
        meta.targetFPS = 120.0f;
    }

    // Override: If adaptive quality is targeting specific FPS, use that
    if (m_enableAdaptiveQuality && m_adaptiveTargetFPS > 0.0f) {
        meta.targetFPS = m_adaptiveTargetFPS;

        // Update preset name based on adaptive target
        if (m_adaptiveTargetFPS >= 150.0f) {
            meta.qualityPreset = "Low";
        } else if (m_adaptiveTargetFPS >= 100.0f) {
            meta.qualityPreset = "Medium";
        } else if (m_adaptiveTargetFPS >= 50.0f) {
            meta.qualityPreset = "High";
        } else {
            meta.qualityPreset = "Ultra";
        }
    }
}

Application::ScreenshotMetadata Application::GatherScreenshotMetadata() {
    ScreenshotMetadata meta;

    // Schema version
    meta.schemaVersion = "2.0";

    // === RENDERING CONFIGURATION ===

    // Active systems
    meta.activeLightingSystem = (m_lightingSystem == LightingSystem::RTXDI) ? "RTXDI" : "MultiLight";
    meta.rendererType = (m_config.rendererType == RendererType::Gaussian) ? "Gaussian" : "Billboard";

    // RTXDI configuration
    meta.rtxdi.enabled = (m_lightingSystem == LightingSystem::RTXDI);
    meta.rtxdi.m4Enabled = meta.rtxdi.enabled;  // M4 is always on if RTXDI is enabled
    meta.rtxdi.m5Enabled = m_enableTemporalFiltering;
    meta.rtxdi.temporalBlendFactor = m_temporalBlend;

    // Light configuration (capture all light positions, colors, intensities)
    meta.lightConfig.count = static_cast<int>(m_lights.size());
    for (const auto& light : m_lights) {
        ScreenshotMetadata::LightConfig::LightInfo info;
        info.posX = light.position.x;
        info.posY = light.position.y;
        info.posZ = light.position.z;
        info.colorR = light.color.x;
        info.colorG = light.color.y;
        info.colorB = light.color.z;
        info.intensity = light.intensity;
        info.radius = light.radius;
        meta.lightConfig.lights.push_back(info);
    }

    // Shadow configuration
    switch (m_shadowPreset) {
        case ShadowPreset::Performance:
            meta.shadows.preset = "Performance";
            break;
        case ShadowPreset::Balanced:
            meta.shadows.preset = "Balanced";
            break;
        case ShadowPreset::Quality:
            meta.shadows.preset = "Quality";
            break;
        case ShadowPreset::Custom:
            meta.shadows.preset = "Custom";
            break;
    }
    meta.shadows.raysPerLight = static_cast<int>(m_shadowRaysPerLight);
    meta.shadows.temporalFilteringEnabled = m_enableTemporalFiltering;
    meta.shadows.temporalBlendFactor = m_temporalBlend;

    // Quality preset detection (call helper)
    DetectQualityPreset(meta);

    // === PHYSICAL EFFECTS ===

    meta.physicalEffects.usePhysicalEmission = m_usePhysicalEmission;
    meta.physicalEffects.emissionStrength = m_emissionStrength;
    meta.physicalEffects.emissionBlendFactor = m_emissionBlendFactor;

    meta.physicalEffects.useDopplerShift = m_useDopplerShift;
    meta.physicalEffects.dopplerStrength = m_dopplerStrength;

    meta.physicalEffects.useGravitationalRedshift = m_useGravitationalRedshift;
    meta.physicalEffects.redshiftStrength = m_redshiftStrength;

    meta.physicalEffects.usePhaseFunction = m_usePhaseFunction;
    meta.physicalEffects.phaseStrength = m_phaseStrength;

    meta.physicalEffects.useAnisotropicGaussians = m_useAnisotropicGaussians;
    meta.physicalEffects.anisotropyStrength = m_anisotropyStrength;

    // === FEATURE STATUS FLAGS ===
    // (These are hardcoded based on FEATURE_STATUS.md audit)

    meta.featureStatus.multiLightWorking = true;
    meta.featureStatus.shadowRaysWorking = true;
    meta.featureStatus.phaseFunctionWorking = true;
    meta.featureStatus.physicalEmissionWorking = true;
    meta.featureStatus.anisotropicGaussiansWorking = true;

    meta.featureStatus.dopplerShiftWorking = false;      // No visible effect (needs debugging)
    meta.featureStatus.redshiftWorking = false;          // No visible effect (needs debugging)
    meta.featureStatus.rtxdiM5Working = false;           // Temporal accumulation WIP

    meta.featureStatus.inScatteringDeprecated = true;
    meta.featureStatus.godRaysDeprecated = true;

    // === PARTICLES ===

    meta.particles.count = static_cast<int>(m_config.particleCount);
    meta.particles.radius = m_particleSize;
    meta.particles.gravityStrength = 1.0f;  // Would need ParticleSystem API
    meta.particles.physicsEnabled = m_physicsEnabled;
    meta.particles.innerRadius = m_innerRadius;
    meta.particles.outerRadius = m_outerRadius;
    meta.particles.diskThickness = m_diskThickness;

    // === PERFORMANCE ===

    meta.performance.fps = m_fps;
    meta.performance.frameTime = (m_fps > 0.0f) ? (1000.0f / m_fps) : 0.0f;
    meta.performance.targetFPS = meta.targetFPS;  // Set by DetectQualityPreset()
    meta.performance.fpsRatio = (meta.targetFPS > 0.0f) ? (m_fps / meta.targetFPS) : 0.0f;

    // === CAMERA ===

    float camX = m_cameraDistance * cos(m_cameraAngle);
    float camZ = m_cameraDistance * sin(m_cameraAngle);

    meta.camera.x = camX;
    meta.camera.y = m_cameraHeight;
    meta.camera.z = camZ;
    meta.camera.lookAtX = 0.0f;
    meta.camera.lookAtY = 0.0f;
    meta.camera.lookAtZ = 0.0f;
    meta.camera.distance = m_cameraDistance;
    meta.camera.height = m_cameraHeight;
    meta.camera.angle = m_cameraAngle;
    meta.camera.pitch = m_cameraPitch;

    // === ML/QUALITY ===

    meta.mlQuality.pinnEnabled = false;  // Would need to check m_adaptiveQuality system
    meta.mlQuality.modelPath = "";
    meta.mlQuality.adaptiveQualityEnabled = m_enableAdaptiveQuality;
    meta.mlQuality.adaptiveTargetFPS = m_adaptiveTargetFPS;

    // === METADATA ===

    auto now = std::chrono::system_clock::now();
    auto timeT = std::chrono::system_clock::to_time_t(now);
    char timestamp[100];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", gmtime(&timeT));
    meta.timestamp = std::string(timestamp);
    meta.configFile = "";  // Would need to track loaded config file

    return meta;
}

void Application::SaveScreenshotMetadata(const std::string& screenshotPath, const ScreenshotMetadata& meta) {
    // Create metadata filename (same as screenshot + .json)
    std::string metaPath = screenshotPath + ".json";

    FILE* file = fopen(metaPath.c_str(), "w");
    if (!file) {
        LOG_ERROR("Failed to create metadata file: {}", metaPath);
        return;
    }

    // Write JSON v2.0 (simple manual serialization - no library dependency)
    fprintf(file, "{\n");
    fprintf(file, "  \"schema_version\": \"%s\",\n", meta.schemaVersion.c_str());
    fprintf(file, "  \"timestamp\": \"%s\",\n", meta.timestamp.c_str());
    fprintf(file, "  \"config_file\": \"%s\",\n", meta.configFile.c_str());

    // === RENDERING CONFIGURATION ===
    fprintf(file, "  \"rendering\": {\n");
    fprintf(file, "    \"active_lighting_system\": \"%s\",\n", meta.activeLightingSystem.c_str());
    fprintf(file, "    \"renderer_type\": \"%s\",\n", meta.rendererType.c_str());

    // RTXDI
    fprintf(file, "    \"rtxdi\": {\n");
    fprintf(file, "      \"enabled\": %s,\n", meta.rtxdi.enabled ? "true" : "false");
    fprintf(file, "      \"m4_enabled\": %s,\n", meta.rtxdi.m4Enabled ? "true" : "false");
    fprintf(file, "      \"m5_enabled\": %s,\n", meta.rtxdi.m5Enabled ? "true" : "false");
    fprintf(file, "      \"temporal_blend_factor\": %.3f\n", meta.rtxdi.temporalBlendFactor);
    fprintf(file, "    },\n");

    // Lights
    fprintf(file, "    \"lights\": {\n");
    fprintf(file, "      \"count\": %d,\n", meta.lightConfig.count);
    fprintf(file, "      \"light_list\": [\n");
    for (size_t i = 0; i < meta.lightConfig.lights.size(); ++i) {
        const auto& light = meta.lightConfig.lights[i];
        fprintf(file, "        {\n");
        fprintf(file, "          \"position\": [%.1f, %.1f, %.1f],\n", light.posX, light.posY, light.posZ);
        fprintf(file, "          \"color\": [%.3f, %.3f, %.3f],\n", light.colorR, light.colorG, light.colorB);
        fprintf(file, "          \"intensity\": %.2f,\n", light.intensity);
        fprintf(file, "          \"radius\": %.1f\n", light.radius);
        fprintf(file, "        }%s\n", (i < meta.lightConfig.lights.size() - 1) ? "," : "");
    }
    fprintf(file, "      ]\n");
    fprintf(file, "    },\n");

    // Shadows
    fprintf(file, "    \"shadows\": {\n");
    fprintf(file, "      \"preset\": \"%s\",\n", meta.shadows.preset.c_str());
    fprintf(file, "      \"rays_per_light\": %d,\n", meta.shadows.raysPerLight);
    fprintf(file, "      \"temporal_filtering\": %s,\n", meta.shadows.temporalFilteringEnabled ? "true" : "false");
    fprintf(file, "      \"temporal_blend\": %.3f\n", meta.shadows.temporalBlendFactor);
    fprintf(file, "    }\n");
    fprintf(file, "  },\n");

    // === QUALITY PRESET ===
    fprintf(file, "  \"quality\": {\n");
    fprintf(file, "    \"preset\": \"%s\",\n", meta.qualityPreset.c_str());
    fprintf(file, "    \"target_fps\": %.1f\n", meta.targetFPS);
    fprintf(file, "  },\n");

    // === PHYSICAL EFFECTS ===
    fprintf(file, "  \"physical_effects\": {\n");
    fprintf(file, "    \"physical_emission\": {\n");
    fprintf(file, "      \"enabled\": %s,\n", meta.physicalEffects.usePhysicalEmission ? "true" : "false");
    fprintf(file, "      \"strength\": %.2f,\n", meta.physicalEffects.emissionStrength);
    fprintf(file, "      \"blend_factor\": %.2f\n", meta.physicalEffects.emissionBlendFactor);
    fprintf(file, "    },\n");
    fprintf(file, "    \"doppler_shift\": {\n");
    fprintf(file, "      \"enabled\": %s,\n", meta.physicalEffects.useDopplerShift ? "true" : "false");
    fprintf(file, "      \"strength\": %.2f\n", meta.physicalEffects.dopplerStrength);
    fprintf(file, "    },\n");
    fprintf(file, "    \"gravitational_redshift\": {\n");
    fprintf(file, "      \"enabled\": %s,\n", meta.physicalEffects.useGravitationalRedshift ? "true" : "false");
    fprintf(file, "      \"strength\": %.2f\n", meta.physicalEffects.redshiftStrength);
    fprintf(file, "    },\n");
    fprintf(file, "    \"phase_function\": {\n");
    fprintf(file, "      \"enabled\": %s,\n", meta.physicalEffects.usePhaseFunction ? "true" : "false");
    fprintf(file, "      \"strength\": %.2f\n", meta.physicalEffects.phaseStrength);
    fprintf(file, "    },\n");
    fprintf(file, "    \"anisotropic_gaussians\": {\n");
    fprintf(file, "      \"enabled\": %s,\n", meta.physicalEffects.useAnisotropicGaussians ? "true" : "false");
    fprintf(file, "      \"strength\": %.2f\n", meta.physicalEffects.anisotropyStrength);
    fprintf(file, "    }\n");
    fprintf(file, "  },\n");

    // === FEATURE STATUS ===
    fprintf(file, "  \"feature_status\": {\n");
    fprintf(file, "    \"working\": {\n");
    fprintf(file, "      \"multi_light\": %s,\n", meta.featureStatus.multiLightWorking ? "true" : "false");
    fprintf(file, "      \"shadow_rays\": %s,\n", meta.featureStatus.shadowRaysWorking ? "true" : "false");
    fprintf(file, "      \"phase_function\": %s,\n", meta.featureStatus.phaseFunctionWorking ? "true" : "false");
    fprintf(file, "      \"physical_emission\": %s,\n", meta.featureStatus.physicalEmissionWorking ? "true" : "false");
    fprintf(file, "      \"anisotropic_gaussians\": %s\n", meta.featureStatus.anisotropicGaussiansWorking ? "true" : "false");
    fprintf(file, "    },\n");
    fprintf(file, "    \"wip\": {\n");
    fprintf(file, "      \"doppler_shift\": %s,\n", !meta.featureStatus.dopplerShiftWorking ? "true" : "false");
    fprintf(file, "      \"gravitational_redshift\": %s,\n", !meta.featureStatus.redshiftWorking ? "true" : "false");
    fprintf(file, "      \"rtxdi_m5\": %s\n", !meta.featureStatus.rtxdiM5Working ? "true" : "false");
    fprintf(file, "    },\n");
    fprintf(file, "    \"deprecated\": {\n");
    fprintf(file, "      \"in_scattering\": %s,\n", meta.featureStatus.inScatteringDeprecated ? "true" : "false");
    fprintf(file, "      \"god_rays\": %s\n", meta.featureStatus.godRaysDeprecated ? "true" : "false");
    fprintf(file, "    }\n");
    fprintf(file, "  },\n");

    // === PARTICLES ===
    fprintf(file, "  \"particles\": {\n");
    fprintf(file, "    \"count\": %d,\n", meta.particles.count);
    fprintf(file, "    \"radius\": %.1f,\n", meta.particles.radius);
    fprintf(file, "    \"gravity_strength\": %.2f,\n", meta.particles.gravityStrength);
    fprintf(file, "    \"physics_enabled\": %s,\n", meta.particles.physicsEnabled ? "true" : "false");
    fprintf(file, "    \"inner_radius\": %.1f,\n", meta.particles.innerRadius);
    fprintf(file, "    \"outer_radius\": %.1f,\n", meta.particles.outerRadius);
    fprintf(file, "    \"disk_thickness\": %.1f\n", meta.particles.diskThickness);
    fprintf(file, "  },\n");

    // === PERFORMANCE ===
    fprintf(file, "  \"performance\": {\n");
    fprintf(file, "    \"fps\": %.1f,\n", meta.performance.fps);
    fprintf(file, "    \"frame_time_ms\": %.2f,\n", meta.performance.frameTime);
    fprintf(file, "    \"target_fps\": %.1f,\n", meta.performance.targetFPS);
    fprintf(file, "    \"fps_ratio\": %.3f\n", meta.performance.fpsRatio);
    fprintf(file, "  },\n");

    // === CAMERA ===
    fprintf(file, "  \"camera\": {\n");
    fprintf(file, "    \"position\": [%.1f, %.1f, %.1f],\n", meta.camera.x, meta.camera.y, meta.camera.z);
    fprintf(file, "    \"look_at\": [%.1f, %.1f, %.1f],\n",
            meta.camera.lookAtX, meta.camera.lookAtY, meta.camera.lookAtZ);
    fprintf(file, "    \"distance\": %.1f,\n", meta.camera.distance);
    fprintf(file, "    \"height\": %.1f,\n", meta.camera.height);
    fprintf(file, "    \"angle\": %.3f,\n", meta.camera.angle);
    fprintf(file, "    \"pitch\": %.3f\n", meta.camera.pitch);
    fprintf(file, "  },\n");

    // === ML/QUALITY ===
    fprintf(file, "  \"ml_quality\": {\n");
    fprintf(file, "    \"pinn_enabled\": %s,\n", meta.mlQuality.pinnEnabled ? "true" : "false");
    fprintf(file, "    \"model_path\": \"%s\",\n", meta.mlQuality.modelPath.c_str());
    fprintf(file, "    \"adaptive_quality_enabled\": %s,\n", meta.mlQuality.adaptiveQualityEnabled ? "true" : "false");
    fprintf(file, "    \"adaptive_target_fps\": %.1f\n", meta.mlQuality.adaptiveTargetFPS);
    fprintf(file, "  }\n");
    fprintf(file, "}\n");

    fclose(file);
    LOG_INFO("Metadata v2.0 saved: {}", metaPath);
}

void Application::CaptureScreenshot() {
    LOG_INFO("\n=== CAPTURING SCREENSHOT (Frame {}) ===", m_frameCount);

    // Create screenshots directory
    std::filesystem::create_directories(m_screenshotOutputDir);

    // Get timestamp for filename
    auto now = std::chrono::system_clock::now();
    auto timeT = std::chrono::system_clock::to_time_t(now);
    char timestamp[100];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d_%H-%M-%S", localtime(&timeT));

    std::string filename = m_screenshotOutputDir + "screenshot_" + std::string(timestamp) + ".bmp";

    // Get the current back buffer
    auto backBuffer = m_swapChain->GetCurrentBackBuffer();

    // Save the back buffer to file
    SaveBackBufferToFile(backBuffer, filename);

    // Phase 1: Capture and save metadata alongside screenshot
    ScreenshotMetadata metadata = GatherScreenshotMetadata();
    SaveScreenshotMetadata(filename, metadata);

    LOG_INFO("Screenshot saved: {}", filename);
}

void Application::SaveBackBufferToFile(ID3D12Resource* backBuffer, const std::string& filename) {
    // Get back buffer description
    D3D12_RESOURCE_DESC desc = backBuffer->GetDesc();

    // Create readback buffer
    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_READBACK;

    D3D12_RESOURCE_DESC readbackDesc = {};
    readbackDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    readbackDesc.Width = desc.Width * desc.Height * 4; // RGBA 8-bit
    readbackDesc.Height = 1;
    readbackDesc.DepthOrArraySize = 1;
    readbackDesc.MipLevels = 1;
    readbackDesc.Format = DXGI_FORMAT_UNKNOWN;
    readbackDesc.SampleDesc.Count = 1;
    readbackDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    Microsoft::WRL::ComPtr<ID3D12Resource> readbackBuffer;
    HRESULT hr = m_device->GetDevice()->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &readbackDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&readbackBuffer));

    if (FAILED(hr)) {
        LOG_ERROR("Failed to create readback buffer for screenshot");
        return;
    }

    // Reset command list for copy operation
    m_device->ResetCommandList();
    auto cmdList = m_device->GetCommandList();

    // Transition back buffer to copy source
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = backBuffer;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    cmdList->ResourceBarrier(1, &barrier);

    // Setup copy locations
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint = {};
    footprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    footprint.Footprint.Width = static_cast<UINT>(desc.Width);
    footprint.Footprint.Height = desc.Height;
    footprint.Footprint.Depth = 1;
    footprint.Footprint.RowPitch = static_cast<UINT>(desc.Width) * 4;

    D3D12_TEXTURE_COPY_LOCATION dst = {};
    dst.pResource = readbackBuffer.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = footprint;

    D3D12_TEXTURE_COPY_LOCATION src = {};
    src.pResource = backBuffer;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.SubresourceIndex = 0;

    // Copy back buffer to readback buffer
    cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

    // Transition back to present state
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    cmdList->ResourceBarrier(1, &barrier);

    // Execute and wait
    cmdList->Close();
    m_device->ExecuteCommandList();
    m_device->WaitForGPU();

    // Map readback buffer
    void* mappedData = nullptr;
    D3D12_RANGE readRange = { 0, static_cast<SIZE_T>(readbackDesc.Width) };
    hr = readbackBuffer->Map(0, &readRange, &mappedData);

    if (SUCCEEDED(hr)) {
        // Convert BGRA to RGB and flip vertically for BMP
        std::vector<uint8_t> rgbData(desc.Width * desc.Height * 3);
        uint8_t* src = reinterpret_cast<uint8_t*>(mappedData);

        for (UINT y = 0; y < desc.Height; ++y) {
            for (UINT x = 0; x < desc.Width; ++x) {
                UINT srcIdx = (y * footprint.Footprint.RowPitch) + (x * 4);
                UINT dstIdx = ((desc.Height - 1 - y) * desc.Width + x) * 3;

                rgbData[dstIdx + 0] = src[srcIdx + 2]; // R
                rgbData[dstIdx + 1] = src[srcIdx + 1]; // G
                rgbData[dstIdx + 2] = src[srcIdx + 0]; // B
            }
        }

        // Write BMP file
        FILE* file = fopen(filename.c_str(), "wb");
        if (file) {
            uint32_t fileSize = 54 + (desc.Width * desc.Height * 3);
            uint32_t imageSize = desc.Width * desc.Height * 3;

            uint8_t header[54] = {
                'B', 'M',
                0, 0, 0, 0,
                0, 0, 0, 0,
                54, 0, 0, 0,
                40, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0,
                1, 0,
                24, 0,
                0, 0, 0, 0,
                0, 0, 0, 0,
                0x13, 0x0B, 0, 0,
                0x13, 0x0B, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0
            };

            memcpy(header + 2, &fileSize, 4);
            memcpy(header + 18, &desc.Width, 4);
            memcpy(header + 22, &desc.Height, 4);
            memcpy(header + 34, &imageSize, 4);

            fwrite(header, 1, 54, file);
            fwrite(rgbData.data(), 1, imageSize, file);
            fclose(file);
        }

        readbackBuffer->Unmap(0, nullptr);
    }
}

void Application::UpdateFrameStats(float actualFrameTime) {
    static float elapsedTime = 0.0f;
    static uint32_t frameCounter = 0;

    elapsedTime += actualFrameTime;  // Use ACTUAL frame time, not physics timestep
    frameCounter++;

    if (elapsedTime >= 1.0f) {
        m_fps = static_cast<float>(frameCounter) / elapsedTime;

        // Build status string with runtime controls
        wchar_t statusBar[512];
        std::wstring features = L"";

        if (m_usePhysicalEmission) {
            wchar_t buf[32];
            swprintf_s(buf, L"[E:%.1f] ", m_emissionStrength);
            features += buf;
        }
        if (m_useDopplerShift) {
            wchar_t buf[32];
            swprintf_s(buf, L"[D:%.1f] ", m_dopplerStrength);
            features += buf;
        }
        if (m_useGravitationalRedshift) {
            wchar_t buf[32];
            swprintf_s(buf, L"[R:%.1f] ", m_redshiftStrength);
            features += buf;
        }
        if (m_rtLighting) features += L"[RT] ";

        // F-key RT feature indicators
        if (m_useShadowRays) features += L"[F5:Shadow] ";
        if (m_useInScattering) {
            wchar_t buf[32];
            swprintf_s(buf, L"[F6:InScat:%.1f] ", m_inScatterStrength);
            features += buf;
        }
        if (m_usePhaseFunction) {
            wchar_t buf[32];
            swprintf_s(buf, L"[F8:Phase:%.1f] ", m_phaseStrength);
            features += buf;
        }
        if (m_useAnisotropicGaussians) {
            wchar_t buf[32];
            swprintf_s(buf, L"[F11:Aniso:%.1f] ", m_anisotropyStrength);
            features += buf;
        }

        // Add physics parameters to status bar
        wchar_t physBuf[128];
        swprintf_s(physBuf, L"G:%.0f A:%.1f T:%.0f ",
                  m_particleSystem->GetGravityStrength(),
                  m_particleSystem->GetAngularMomentum(),
                  m_particleSystem->GetTurbulence());
        features += physBuf;

        // Update window title with FPS, renderer, and active features
        swprintf_s(statusBar, L"PlasmaDX-Clean - FPS: %.1f - Size: %.0f - %s%s",
                  m_fps,
                  m_particleSize,
                  features.c_str(),
                  m_particleRenderer ?
                  (m_particleRenderer->GetActivePath() == ParticleRenderer::RenderPath::MeshShaders ?
                   L"Mesh" : L"Billboard") : L"NoRender");
        SetWindowText(m_hwnd, statusBar);

        elapsedTime = 0.0f;
        frameCounter = 0;
    }
}

void Application::InitializeImGui() {
    LOG_INFO("Initializing ImGui...");

    // Setup Dear ImGui context FIRST
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable keyboard navigation

    // Setup Dear ImGui style (dark theme)
    ImGui::StyleColorsDark();

    // Build font atlas BEFORE initializing backends
    // This is critical - the DX12 backend needs the atlas built
    io.Fonts->Build();

    // Setup Win32 backend (must be before DX12 backend)
    ImGui_ImplWin32_Init(m_hwnd);

    // Create descriptor heap for ImGui (1 descriptor for font texture)
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.NumDescriptors = 1;
    heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

    HRESULT hr = m_device->GetDevice()->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&m_imguiDescriptorHeap));
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create ImGui descriptor heap");
        ImGui_ImplWin32_Shutdown();
        ImGui::DestroyContext();
        return;
    }

    // Setup DX12 backend - this will build the font atlas automatically
    ImGui_ImplDX12_Init(m_device->GetDevice(),
                        3,  // Number of frames in flight (swap chain buffer count)
                        DXGI_FORMAT_R8G8B8A8_UNORM,
                        m_imguiDescriptorHeap.Get(),
                        m_imguiDescriptorHeap->GetCPUDescriptorHandleForHeapStart(),
                        m_imguiDescriptorHeap->GetGPUDescriptorHandleForHeapStart());

    // Upload fonts to GPU
    // We need to do this manually by executing a command list
    m_device->ResetCommandList();
    auto cmdList = m_device->GetCommandList();
    ImGui_ImplDX12_CreateDeviceObjects();
    cmdList->Close();
    m_device->ExecuteCommandList();
    m_device->WaitForGPU();

    LOG_INFO("ImGui initialized successfully (Press F1 to toggle)");
}

bool Application::CreateBlitPipeline() {
    LOG_INFO("Creating HDR→SDR blit pipeline...");

    // Load precompiled shaders
    std::vector<uint8_t> vsCode;
    std::vector<uint8_t> psCode;

    // Read vertex shader
    FILE* vsFile = fopen("shaders/util/blit_hdr_to_sdr_vs.dxil", "rb");
    if (!vsFile) {
        LOG_ERROR("Failed to open blit_hdr_to_sdr_vs.dxil");
        return false;
    }
    fseek(vsFile, 0, SEEK_END);
    size_t vsSize = ftell(vsFile);
    fseek(vsFile, 0, SEEK_SET);
    vsCode.resize(vsSize);
    fread(vsCode.data(), 1, vsSize, vsFile);
    fclose(vsFile);

    // Read pixel shader
    FILE* psFile = fopen("shaders/util/blit_hdr_to_sdr_ps.dxil", "rb");
    if (!psFile) {
        LOG_ERROR("Failed to open blit_hdr_to_sdr_ps.dxil");
        return false;
    }
    fseek(psFile, 0, SEEK_END);
    size_t psSize = ftell(psFile);
    fseek(psFile, 0, SEEK_SET);
    psCode.resize(psSize);
    fread(psCode.data(), 1, psSize, psFile);
    fclose(psFile);

    LOG_INFO("  Loaded VS: {} bytes, PS: {} bytes", vsSize, psSize);

    // Create root signature
    // 1 descriptor table (1 SRV: t0), 1 static sampler
    D3D12_DESCRIPTOR_RANGE srvRange = {};
    srvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srvRange.NumDescriptors = 1;
    srvRange.BaseShaderRegister = 0;  // t0
    srvRange.RegisterSpace = 0;
    srvRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    D3D12_ROOT_PARAMETER rootParam = {};
    rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParam.DescriptorTable.NumDescriptorRanges = 1;
    rootParam.DescriptorTable.pDescriptorRanges = &srvRange;
    rootParam.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_STATIC_SAMPLER_DESC sampler = {};
    sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.MipLODBias = 0.0f;
    sampler.MaxAnisotropy = 1;
    sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
    sampler.BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK;
    sampler.MinLOD = 0.0f;
    sampler.MaxLOD = D3D12_FLOAT32_MAX;
    sampler.ShaderRegister = 0;  // s0
    sampler.RegisterSpace = 0;
    sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_ROOT_SIGNATURE_DESC rootSigDesc = {};
    rootSigDesc.NumParameters = 1;
    rootSigDesc.pParameters = &rootParam;
    rootSigDesc.NumStaticSamplers = 1;
    rootSigDesc.pStaticSamplers = &sampler;
    rootSigDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

    Microsoft::WRL::ComPtr<ID3DBlob> signature;
    Microsoft::WRL::ComPtr<ID3DBlob> error;
    HRESULT hr = D3D12SerializeRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error);
    if (FAILED(hr)) {
        if (error) {
            LOG_ERROR("Failed to serialize blit root signature: {}", (const char*)error->GetBufferPointer());
        }
        return false;
    }

    hr = m_device->GetDevice()->CreateRootSignature(0, signature->GetBufferPointer(),
                                                     signature->GetBufferSize(),
                                                     IID_PPV_ARGS(&m_blitRootSignature));
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create blit root signature");
        return false;
    }

    // Create graphics PSO
    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = m_blitRootSignature.Get();
    psoDesc.VS = { vsCode.data(), vsCode.size() };
    psoDesc.PS = { psCode.data(), psCode.size() };
    psoDesc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    psoDesc.RasterizerState.DepthClipEnable = FALSE;
    psoDesc.DepthStencilState.DepthEnable = FALSE;
    psoDesc.DepthStencilState.StencilEnable = FALSE;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    psoDesc.SampleDesc.Count = 1;

    hr = m_device->GetDevice()->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_blitPSO));
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create blit PSO");
        return false;
    }

    LOG_INFO("HDR→SDR blit pipeline created successfully");
    return true;
}

void Application::ShutdownImGui() {
    if (m_imguiDescriptorHeap) {
        ImGui_ImplDX12_Shutdown();
        ImGui_ImplWin32_Shutdown();
        ImGui::DestroyContext();
        m_imguiDescriptorHeap.Reset();
    }
}

void Application::RenderImGui() {
    if (!m_showImGui) return;

    // Start the Dear ImGui frame
    ImGui_ImplDX12_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();

    // Create control panel window
    ImGui::Begin("PlasmaDX Control Panel", &m_showImGui);

    ImGui::Text("FPS: %.1f", m_fps);
    ImGui::Text("Fullscreen: %s (Alt+Enter)", m_isFullscreen ? "ON" : "OFF");
    ImGui::Separator();

    // Camera controls
    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Distance (W/A)", &m_cameraDistance, 100.0f, 2000.0f);
        ImGui::SliderFloat("Height (Up/Down)", &m_cameraHeight, 0.0f, 2000.0f);
        ImGui::SliderFloat("Angle (Left/Right)", &m_cameraAngle, -3.14f, 3.14f);
        ImGui::SliderFloat("Pitch (Mouse Y)", &m_cameraPitch, -1.5f, 1.5f);
        ImGui::SliderFloat("Particle Size (+/-)", &m_particleSize, 1.0f, 100.0f);
        ImGui::Separator();
        ImGui::Text("Camera Speed Settings");
        ImGui::SliderFloat("Move Speed", &m_cameraMoveSpeed, 10.0f, 500.0f);
        ImGui::SliderFloat("Rotate Speed", &m_cameraRotateSpeed, 0.1f, 2.0f);
    }

    // Rendering features
    if (ImGui::CollapsingHeader("Rendering Features", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Shadow Rays (F5)", &m_useShadowRays);

        // PCSS shadow quality controls (indented under Shadow Rays)
        if (m_useShadowRays) {
            ImGui::Indent();
            ImGui::Text("Shadow Quality");

            // Preset dropdown
            const char* presetNames[] = { "Performance", "Balanced", "Quality", "Custom" };
            int currentPreset = static_cast<int>(m_shadowPreset);
            if (ImGui::Combo("Preset", &currentPreset, presetNames, 4)) {
                m_shadowPreset = static_cast<ShadowPreset>(currentPreset);

                // Apply preset settings
                switch (m_shadowPreset) {
                    case ShadowPreset::Performance:
                        m_shadowRaysPerLight = 1;
                        m_enableTemporalFiltering = true;
                        m_temporalBlend = 0.1f;
                        break;
                    case ShadowPreset::Balanced:
                        m_shadowRaysPerLight = 4;
                        m_enableTemporalFiltering = false;
                        m_temporalBlend = 0.1f;
                        break;
                    case ShadowPreset::Quality:
                        m_shadowRaysPerLight = 8;
                        m_enableTemporalFiltering = false;
                        m_temporalBlend = 0.1f;
                        break;
                    case ShadowPreset::Custom:
                        // Keep current settings
                        break;
                }
            }

            // Show info for current preset
            if (m_shadowPreset == ShadowPreset::Performance) {
                ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "1-ray + temporal (120 FPS target)");
            } else if (m_shadowPreset == ShadowPreset::Balanced) {
                ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.5f, 1.0f), "4-ray PCSS (90-100 FPS target)");
            } else if (m_shadowPreset == ShadowPreset::Quality) {
                ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "8-ray PCSS (60-75 FPS target)");
            }

            // Custom controls (only show if Custom preset selected)
            if (m_shadowPreset == ShadowPreset::Custom) {
                int raysPerLight = static_cast<int>(m_shadowRaysPerLight);
                if (ImGui::SliderInt("Rays Per Light", &raysPerLight, 1, 16)) {
                    m_shadowRaysPerLight = static_cast<uint32_t>(raysPerLight);
                }

                ImGui::Checkbox("Temporal Filtering", &m_enableTemporalFiltering);
                if (m_enableTemporalFiltering) {
                    ImGui::SliderFloat("Temporal Blend", &m_temporalBlend, 0.0f, 1.0f);
                    ImGui::SameLine();
                    ImGui::TextDisabled("(?)");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Lower = smoother (slower convergence)\n"
                                         "Higher = faster convergence (more noise)\n"
                                         "Default: 0.1 (~67ms convergence)");
                    }
                }
            }

            ImGui::Unindent();
        }

#ifdef ENABLE_DLSS
        // DLSS 4.0 Ray Reconstruction controls
        ImGui::Separator();
        bool dlssAvailable = (m_dlssSystem && m_dlssSystem->IsSuperResolutionSupported());

        if (!dlssAvailable) {
            ImGui::BeginDisabled();
        }

        if (ImGui::Checkbox("DLSS Ray Reconstruction (F4)", &m_enableDLSS)) {
            if (m_enableDLSS && !dlssAvailable) {
                LOG_WARN("DLSS: Cannot enable - system not available");
                m_enableDLSS = false;
            }
        }

        if (!dlssAvailable) {
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "(Not Available)");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("DLSS requires:\n"
                                 "- NVIDIA RTX GPU\n"
                                 "- Driver 531.00+\n"
                                 "- nvngx_dlssd.dll in exe directory");
            }
        } else {
            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("AI-powered denoising for shadow rays\n"
                                 "Reduces shadow noise at 1-2 rays/light\n"
                                 "Expected: 2-4× raytracing performance");
            }

            // Denoiser strength slider (only show when DLSS enabled)
            if (m_enableDLSS) {
                ImGui::Indent();
                if (ImGui::SliderFloat("Denoiser Strength", &m_dlssDenoiserStrength, 0.0f, 2.0f)) {
                    if (m_dlssSystem) {
                        m_dlssSystem->SetSharpness(m_dlssDenoiserStrength);  // TODO: Rename variable in Phase 3
                    }
                }
                ImGui::SameLine();
                ImGui::TextDisabled("(?)");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("0.0 = No denoising (full noise)\n"
                                     "1.0 = Default (balanced)\n"
                                     "2.0 = Maximum (smoothest)");
                }

                ImGui::Spacing();
                ImGui::Text("DLSS Quality Mode:");
                bool qualityChanged = false;
                qualityChanged |= ImGui::RadioButton("Quality (67% res, 1.5x)", &m_dlssQualityMode, 0);
                ImGui::SameLine();
                ImGui::TextDisabled("(?)");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Best quality, moderate performance gain\n"
                                     "2560x1440 → 1714x964");
                }

                qualityChanged |= ImGui::RadioButton("Balanced (58% res, 1.7x)", &m_dlssQualityMode, 1);
                ImGui::SameLine();
                ImGui::TextDisabled("(?)");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Recommended balance\n"
                                     "2560x1440 → 1484x836 (CURRENT)");
                }

                qualityChanged |= ImGui::RadioButton("Performance (50% res, 2x)", &m_dlssQualityMode, 2);
                ImGui::SameLine();
                ImGui::TextDisabled("(?)");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Higher performance, slight quality loss\n"
                                     "2560x1440 → 1280x720");
                }

                qualityChanged |= ImGui::RadioButton("Ultra Perf (33% res, 3x)", &m_dlssQualityMode, 3);
                ImGui::SameLine();
                ImGui::TextDisabled("(?)");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Maximum performance, quality depends on scene\n"
                                     "2560x1440 → 845x476\n"
                                     "Great for 4K displays!");
                }

                // If quality mode changed, defer recreation until safe (start of next frame)
                if (qualityChanged) {
                    m_dlssQualityModeChanged = true;
                    LOG_INFO("DLSS: Quality mode change requested (will apply at start of next frame)");
                }

                // Status indicator
                ImGui::Spacing();
                ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "DLSS Super Resolution Active");
                ImGui::Unindent();
            }
        }
        ImGui::Separator();
#endif

        ImGui::Checkbox("In-Scattering (F6)", &m_useInScattering);
        if (m_useInScattering) {
            ImGui::SliderFloat("In-Scatter Strength (F9)", &m_inScatterStrength, 0.0f, 10.0f);
        }
        ImGui::Checkbox("Phase Function (F8)", &m_usePhaseFunction);
        if (m_usePhaseFunction) {
            ImGui::SliderFloat("Phase Strength (Ctrl/Shift+F8)", &m_phaseStrength, 0.0f, 20.0f);
        }
        ImGui::Checkbox("RT Particle-Particle Lighting", &m_enableRTLighting);
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Toggle the particle-to-particle RT lighting system.\n"
                             "When OFF, only multi-light system illuminates particles.");
        }
        if (m_enableRTLighting) {
            ImGui::SliderFloat("RT Lighting Strength (F10)", &m_rtLightingStrength, 0.0f, 10.0f);
            ImGui::SliderFloat("RT Max Distance", &m_rtMaxDistance, 50.0f, 400.0f, "%.0f units");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Maximum distance for particle-to-particle RT lighting\n"
                                  "100 units = innermost particles only (default)\n"
                                  "200 units = middle disk coverage\n"
                                  "300 units = full disk coverage\n"
                                  "400 units = extended range (may impact performance)\n"
                                  "\n"
                                  "Find your GPU's sweet spot by gradually increasing!");
            }
            // Update RTLightingSystem with new distance
            if (m_rtLighting) {
                m_rtLighting->SetMaxLightingDistance(m_rtMaxDistance);
            }
        }
        ImGui::SliderFloat("Global Ambient", &m_rtMinAmbient, 0.0f, 0.2f, "%.3f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Minimum ambient light for all particles\n"
                              "Prevents particles from being completely black\n"
                              "\n"
                              "0.000 = Pure black when unlit (original)\n"
                              "0.050 = Subtle ambient glow (recommended)\n"
                              "0.100 = Brighter ambient\n"
                              "0.200 = Maximum ambient");
        }

        // === Dynamic Emission (RT-Driven Star Radiance) ===
        ImGui::Separator();
        ImGui::Text("Dynamic Emission (RT-Driven)");
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Emission responds to RT lighting dynamically:\n"
                              "- Well-lit particles: RT dominates (dynamic)\n"
                              "- Shadow particles: Emission fills in\n"
                              "- Hot particles (>threshold): Self-luminous\n"
                              "- Cool particles: Purely RT-driven\n"
                              "\n"
                              "Real-time tuning - no rebuild required!");
        }

        // Track if any emission value changed this frame
        bool emissionNeedsUpdate = false;

        if (ImGui::SliderFloat("Emission Strength", &m_rtEmissionStrength, 0.0f, 1.0f, "%.2f")) {
            emissionNeedsUpdate = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Global emission multiplier\n"
                              "0.15 = Maximum RT dynamicism (minimal emission)\n"
                              "0.25 = Balanced (default)\n"
                              "0.40 = Star-like (more glow)");
        }

        if (ImGui::SliderFloat("Temp Threshold (K)", &m_rtEmissionThreshold, 15000.0f, 28000.0f, "%.0f")) {
            emissionNeedsUpdate = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Temperature cutoff for emission\n"
                              "Only particles hotter than threshold emit\n"
                              "\n"
                              "18000K = More stars glow\n"
                              "22000K = Hot stars only (default)\n"
                              "25000K = Only hottest stars");
        }

        if (ImGui::SliderFloat("RT Suppression", &m_rtEmissionSuppression, 0.0f, 1.0f, "%.2f")) {
            emissionNeedsUpdate = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("How much RT lighting suppresses emission\n"
                              "Well-lit particles reduce emission strength\n"
                              "\n"
                              "0.5 = Emission visible everywhere (less dynamic)\n"
                              "0.7 = Good suppression (default)\n"
                              "0.9 = Strong suppression (maximum dynamicism)");
        }

        if (ImGui::SliderFloat("Temporal Rate", &m_rtEmissionTemporalRate, 0.0f, 0.1f, "%.3f")) {
            emissionNeedsUpdate = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Star twinkling/pulsing speed\n"
                              "Each particle pulses at slightly different rate\n"
                              "\n"
                              "0.02 = Very subtle breathing\n"
                              "0.03 = Subtle pulse (default)\n"
                              "0.05 = Noticeable twinkling\n"
                              "0.10 = Fast scintillation");
        }

        // Update RTLightingSystem only when values change (event-driven)
        if (emissionNeedsUpdate && m_rtLighting) {
            m_rtLighting->SetEmissionStrength(m_rtEmissionStrength);
            m_rtLighting->SetEmissionThreshold(m_rtEmissionThreshold);
            m_rtLighting->SetRTSuppression(m_rtEmissionSuppression);
            m_rtLighting->SetTemporalRate(m_rtEmissionTemporalRate);
        }

        // Preset buttons for quick testing
        ImGui::Spacing();
        ImGui::Text("Presets:");
        if (ImGui::Button("Max Dynamicism")) {
            m_rtEmissionStrength = 0.15f;
            m_rtEmissionThreshold = 25000.0f;
            m_rtEmissionSuppression = 0.9f;
            m_rtEmissionTemporalRate = 0.02f;
            if (m_rtLighting) {
                m_rtLighting->SetEmissionStrength(m_rtEmissionStrength);
                m_rtLighting->SetEmissionThreshold(m_rtEmissionThreshold);
                m_rtLighting->SetRTSuppression(m_rtEmissionSuppression);
                m_rtLighting->SetTemporalRate(m_rtEmissionTemporalRate);
            }
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("RT lighting drives visual, minimal emission");
        }

        ImGui::SameLine();
        if (ImGui::Button("Balanced")) {
            m_rtEmissionStrength = 0.25f;
            m_rtEmissionThreshold = 22000.0f;
            m_rtEmissionSuppression = 0.7f;
            m_rtEmissionTemporalRate = 0.03f;
            if (m_rtLighting) {
                m_rtLighting->SetEmissionStrength(m_rtEmissionStrength);
                m_rtLighting->SetEmissionThreshold(m_rtEmissionThreshold);
                m_rtLighting->SetRTSuppression(m_rtEmissionSuppression);
                m_rtLighting->SetTemporalRate(m_rtEmissionTemporalRate);
            }
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Default settings, good shadow fill");
        }

        ImGui::SameLine();
        if (ImGui::Button("Star-Like")) {
            m_rtEmissionStrength = 0.4f;
            m_rtEmissionThreshold = 18000.0f;
            m_rtEmissionSuppression = 0.5f;
            m_rtEmissionTemporalRate = 0.05f;
            if (m_rtLighting) {
                m_rtLighting->SetEmissionStrength(m_rtEmissionStrength);
                m_rtLighting->SetEmissionThreshold(m_rtEmissionThreshold);
                m_rtLighting->SetRTSuppression(m_rtEmissionSuppression);
                m_rtLighting->SetTemporalRate(m_rtEmissionTemporalRate);
            }
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("More glow, aesthetic starfield look");
        }

        // === Phase 1.5 Adaptive Particle Radius (Fix overlap artifacts) ===
        ImGui::Separator();
        ImGui::Text("Adaptive Particle Radius (Phase 1.5)");
        bool adaptiveToggled = ImGui::Checkbox("Enable Adaptive Radius", &m_enableAdaptiveRadius);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Fixes overlap artifacts in dense regions and improves visibility in sparse regions\n"
                              "- Dense inner particles shrink to reduce overlap\n"
                              "- Sparse outer particles grow to improve ray intersection");
        }

        // Update RT system if toggle changed
        if (adaptiveToggled && m_rtLighting) {
            m_rtLighting->SetAdaptiveRadiusEnabled(m_enableAdaptiveRadius);
        }

        if (m_enableAdaptiveRadius) {
            // Track if any value changed this frame
            bool needsUpdate = false;

            if (ImGui::SliderFloat("Inner Zone Threshold", &m_adaptiveInnerZone, 0.0f, 200.0f, "%.0f units")) {
                needsUpdate = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Distance below which particles shrink (dense region)");
            }

            if (ImGui::SliderFloat("Outer Zone Threshold", &m_adaptiveOuterZone, 200.0f, 600.0f, "%.0f units")) {
                needsUpdate = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Distance above which particles grow (sparse region)");
            }

            if (ImGui::SliderFloat("Inner Scale (Shrink)", &m_adaptiveInnerScale, 0.1f, 1.0f, "%.2f")) {
                needsUpdate = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Minimum radius scale for dense inner particles\n"
                                  "0.1 = 10%% size (maximum shrink)\n"
                                  "0.5 = 50%% size (default)\n"
                                  "1.0 = 100%% size (no shrink)");
            }

            if (ImGui::SliderFloat("Outer Scale (Grow)", &m_adaptiveOuterScale, 1.0f, 3.0f, "%.2f")) {
                needsUpdate = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Maximum radius scale for sparse outer particles\n"
                                  "1.0 = 100%% size (no grow)\n"
                                  "2.0 = 200%% size (default)\n"
                                  "3.0 = 300%% size (maximum grow)");
            }

            if (ImGui::SliderFloat("Density Scale Min", &m_densityScaleMin, 0.1f, 1.0f, "%.2f")) {
                needsUpdate = true;
            }
            if (ImGui::SliderFloat("Density Scale Max", &m_densityScaleMax, 1.0f, 5.0f, "%.2f")) {
                needsUpdate = true;
            }

            // FIXED: Only update RTLightingSystem when values actually change
            // Previously this ran EVERY FRAME which caused freezing on hover
            if (needsUpdate && m_rtLighting) {
                m_rtLighting->SetAdaptiveRadiusEnabled(m_enableAdaptiveRadius);
                m_rtLighting->SetAdaptiveInnerZone(m_adaptiveInnerZone);
                m_rtLighting->SetAdaptiveOuterZone(m_adaptiveOuterZone);
                m_rtLighting->SetAdaptiveInnerScale(m_adaptiveInnerScale);
                m_rtLighting->SetAdaptiveOuterScale(m_adaptiveOuterScale);
                m_rtLighting->SetDensityScaleMin(m_densityScaleMin);
                m_rtLighting->SetDensityScaleMax(m_densityScaleMax);
            }
        }

        // === Phase 3.9 Spatial RT Interpolation ===
        ImGui::Separator();
        ImGui::Text("Spatial RT Interpolation (Phase 3.9)");
        ImGui::Checkbox("Enable Spatial Interpolation", &m_useVolumetricRT);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("SPATIAL INTERPOLATION: Blends RT lighting from nearby particles\n"
                              "  - Reads g_rtLighting[] from 3-8 neighbors at sample point\n"
                              "  - Distance-weighted blending creates smooth gradients\n"
                              "  - Eliminates discrete jumps without recomputing lighting!\n\n"
                              "Enabled: Multi-light quality smoothness\n"
                              "Disabled: Legacy per-particle lookup (faster but jumpy)");
        }

        if (m_useVolumetricRT) {
            int samples = static_cast<int>(m_volumetricRTSamples);
            if (ImGui::SliderInt("Neighbor Samples", &samples, 4, 32)) {
                m_volumetricRTSamples = static_cast<uint32_t>(samples);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Number of neighbor particles to sample for interpolation\n"
                                  "4 = Tetrahedral (fast but angular)\n"
                                  "8 = Cubic (balanced, default)\n"
                                  "16 = High quality (smooth)\n"
                                  "32 = Maximum smoothness (expensive)");
            }

            if (ImGui::SliderFloat("Smoothness Radius", &m_volumetricRTDistance, 100.0f, 400.0f, "%.0f units")) {
                // Value updated automatically
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Maximum distance to search for neighbor particles\n"
                                  "Larger = smoother gradients but blurrier\n"
                                  "Smaller = sharper transitions but more discrete\n"
                                  "Recommended: 200 units (matches avg particle spacing ~139)");
            }
        }
        ImGui::Separator();

        ImGui::Checkbox("Anisotropic Gaussians (F11)", &m_useAnisotropicGaussians);
        if (m_useAnisotropicGaussians) {
            ImGui::SliderFloat("Anisotropy Strength (F12)", &m_anisotropyStrength, 0.0f, 3.0f);
        }

        // RTXDI Lighting System (Phase 4 - M4 Phase 2 Integration)
        ImGui::Separator();
        ImGui::Text("Lighting System (F3 to toggle)");
        int currentSystem = static_cast<int>(m_lightingSystem);
        if (ImGui::RadioButton("Multi-Light (13 lights, brute force)", currentSystem == 0)) {
            m_lightingSystem = LightingSystem::MultiLight;
            LOG_INFO("Switched to Multi-Light system");
        }
        if (ImGui::RadioButton("RTXDI (1 sampled light per pixel)", currentSystem == 1)) {
            m_lightingSystem = LightingSystem::RTXDI;
            LOG_INFO("Switched to RTXDI system");
        }
        if (ImGui::RadioButton("Volumetric ReSTIR (Experimental)", currentSystem == 2)) {
            if (m_volumetricReSTIR) {
                m_lightingSystem = LightingSystem::VolumetricReSTIR;
                LOG_INFO("Switched to Volumetric ReSTIR system");
            } else {
                LOG_WARN("Volumetric ReSTIR system not available");
            }
        }

        // Display current mode info
        ImGui::Indent();
        if (m_lightingSystem == LightingSystem::RTXDI) {
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "RTXDI: Weighted reservoir sampling");
            ImGui::Text("Expected FPS: Similar or better than multi-light");
            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("RTXDI selects 1 optimal light per pixel using\n"
                                 "importance-weighted random sampling from light grid.\n"
                                 "Phase 4 Milestone 4 - First Visual Test!");
            }

            // DEBUG: Visualize selected light index
            ImGui::Separator();
            if (ImGui::Checkbox("DEBUG: Visualize Light Selection", &m_debugRTXDISelection)) {
                LOG_INFO("RTXDI Debug Visualization: {}", m_debugRTXDISelection ? "ON" : "OFF");
            }
            if (m_debugRTXDISelection) {
                ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.5f, 1.0f), "Rainbow colors = different lights selected");
                ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), "Black = no lights in grid cell");
            }
        } else if (m_lightingSystem == LightingSystem::VolumetricReSTIR) {
            ImGui::TextColored(ImVec4(0.5f, 0.7f, 1.0f, 1.0f), "Volumetric ReSTIR: Path tracing (Phase 1)");
            ImGui::Text("Expected FPS: ~200 FPS @ 1080p (M=8, K=3)");
            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Phase 1: RIS-only candidate generation\n"
                                 "No spatial/temporal reuse yet (noisy output)\n"
                                 "Uses weighted reservoir sampling for volumetric light transport");
            }
        } else {
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.5f, 1.0f), "Multi-Light: All 13 lights evaluated");
            ImGui::Text("Expected FPS: Baseline (20 FPS @ 10K particles)");
        }
        ImGui::Unindent();

        // === Milestone 5: Temporal Accumulation Controls ===
        if (m_lightingSystem == LightingSystem::RTXDI && m_rtxdiLightingSystem) {
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "M5: Temporal Accumulation");

            // Max samples slider
            int maxSamples = static_cast<int>(m_rtxdiLightingSystem->GetMaxSamples());
            if (ImGui::SliderInt("Max Samples", &maxSamples, 1, 32)) {
                m_rtxdiLightingSystem->SetMaxSamples(static_cast<uint32_t>(maxSamples));
                LOG_INFO("M5 Max Samples: {}", maxSamples);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Temporal sample accumulation target\n"
                                  "8 = Performance (67ms convergence @ 120 FPS)\n"
                                  "16 = Quality (133ms convergence @ 120 FPS)\n"
                                  "Higher = smoother but slower convergence");
            }

            // Reset threshold slider
            float resetThreshold = m_rtxdiLightingSystem->GetResetThreshold();
            if (ImGui::SliderFloat("Reset Threshold", &resetThreshold, 1.0f, 100.0f, "%.1f units")) {
                m_rtxdiLightingSystem->SetResetThreshold(resetThreshold);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Camera movement distance to reset accumulation\n"
                                  "Lower = more responsive to movement\n"
                                  "Higher = smoother during slow camera pans");
            }

            // Manual reset button
            if (ImGui::Button("Reset Accumulation")) {
                m_rtxdiLightingSystem->ForceReset();
                LOG_INFO("M5 Temporal accumulation reset");
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Manually reset accumulated samples\n"
                                  "Use after changing lights or presets");
            }

            // Display convergence info
            int convergenceMs = maxSamples * 8;  // ~8ms per sample @ 120 FPS
            ImGui::Text("Convergence: ~%d ms @ 120 FPS", convergenceMs);
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Patchwork smooths over time");
        }

        // === Volumetric ReSTIR Controls (Phase 1) ===
        if (m_lightingSystem == LightingSystem::VolumetricReSTIR && m_volumetricReSTIR) {
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.4f, 0.7f, 1.0f, 1.0f), "Volumetric ReSTIR Parameters");

            // Random walks per pixel (M)
            int randomWalks = static_cast<int>(m_volumetricReSTIR->GetRandomWalksPerPixel());
            if (ImGui::SliderInt("Random Walks (M)", &randomWalks, 1, 32)) {
                m_volumetricReSTIR->SetRandomWalksPerPixel(static_cast<uint32_t>(randomWalks));
                LOG_INFO("Volumetric ReSTIR: Random walks per pixel = {}", randomWalks);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Number of random walk candidates per pixel\n"
                                  "4 = Fast but noisy\n"
                                  "8 = Balanced (recommended for Phase 1)\n"
                                  "16+ = Smoother but slower\n"
                                  "Higher M = better convergence");
            }

            // Max bounces (K)
            int maxBounces = static_cast<int>(m_volumetricReSTIR->GetMaxBounces());
            if (ImGui::SliderInt("Max Bounces (K)", &maxBounces, 1, 8)) {
                m_volumetricReSTIR->SetMaxBounces(static_cast<uint32_t>(maxBounces));
                LOG_INFO("Volumetric ReSTIR: Max bounces = {}", maxBounces);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Maximum scattering events per path\n"
                                  "1 = Single scattering (fast)\n"
                                  "3 = Multiple scattering (recommended)\n"
                                  "8 = High-order scattering (slow)\n"
                                  "Higher K = more accurate volume caustics");
            }

            // Performance estimate
            ImGui::Separator();
            float estimatedMs = randomWalks * maxBounces * 0.05f;  // Rough estimate
            ImGui::Text("Estimated cost: %.1f ms/frame", estimatedMs);
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                              "Phase 1: No spatial/temporal reuse (noisy)");
        }
    }

    // Hybrid Probe Grid System (Phase 0.13.1)
    if (ImGui::CollapsingHeader("Probe Grid (Phase 0.13.1)")) {
        if (m_probeGridSystem) {
            bool useProbeGrid = (m_useProbeGrid != 0);
            if (ImGui::Checkbox("Enable Probe Grid", &useProbeGrid)) {
                m_useProbeGrid = useProbeGrid ? 1u : 0u;
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Zero-atomic-contention volumetric lighting\n"
                                 "Pre-computes lighting at 32\u00b3 sparse grid\n"
                                 "Particles interpolate via trilinear sampling\n"
                                 "\n"
                                 "Designed to avoid atomic contention issues\n"
                                 "that plague Volumetric ReSTIR at 2045+ particles");
            }

            ImGui::Separator();
            ImGui::Text("Grid Architecture:");
            ImGui::BulletText("32\u00b3 = 32,768 probes");
            ImGui::BulletText("Spacing: 93.75 units");
            ImGui::BulletText("Coverage: -1500 to +1500 per axis");
            ImGui::BulletText("Memory: 4.06 MB probe buffer");

            ImGui::Separator();
            ImGui::Text("Performance Characteristics:");
            ImGui::BulletText("Zero atomic operations");
            ImGui::BulletText("Temporal amortization: 1/4 probes/frame");
            ImGui::BulletText("Update cost: ~0.5-1.0ms");
            ImGui::BulletText("Query cost: ~0.2-0.3ms");

            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f),
                              "SUCCESS METRIC:");
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f),
                              "2045+ particles - NO CRASH");
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                              "(vs Volumetric ReSTIR which crashes)");
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                              "Probe Grid System not initialized");
        }
    }

    // Physical effects
    if (ImGui::CollapsingHeader("Physical Effects")) {
        ImGui::Checkbox("Physical Emission (E)", &m_usePhysicalEmission);
        if (m_usePhysicalEmission) {
            ImGui::SliderFloat("Emission Strength (Ctrl/Shift+E)", &m_emissionStrength, 0.0f, 5.0f);
            ImGui::SliderFloat("Artistic ↔ Physical Blend", &m_emissionBlendFactor, 0.0f, 1.0f);
            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("0.0 = Pure artistic (warm colors)\n"
                                 "1.0 = Pure physical (accurate blackbody)\n"
                                 "Auto-blends based on temperature:\n"
                                 "  Cool (<8000K): Stays artistic\n"
                                 "  Hot (>18000K): Goes physical");
            }
        }
        ImGui::Checkbox("Doppler Shift (R)", &m_useDopplerShift);
        if (m_useDopplerShift) {
            ImGui::SliderFloat("Doppler Strength (Ctrl/Shift+R)", &m_dopplerStrength, 0.0f, 5.0f);
        }
        ImGui::Checkbox("Gravitational Redshift (G)", &m_useGravitationalRedshift);
        if (m_useGravitationalRedshift) {
            ImGui::SliderFloat("Redshift Strength (Ctrl/Shift+G)", &m_redshiftStrength, 0.0f, 5.0f);
        }
    }

    // === Adaptive Quality System (ML-Based) ===
    if (ImGui::CollapsingHeader("Adaptive Quality (ML)", ImGuiTreeNodeFlags_None)) {
        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "ML-Based Performance Prediction");

        // Enable/Disable toggle
        if (ImGui::Checkbox("Enable Adaptive Quality", &m_enableAdaptiveQuality)) {
            if (m_adaptiveQuality) {
                m_adaptiveQuality->SetEnabled(m_enableAdaptiveQuality);
                LOG_INFO("Adaptive Quality: {}", m_enableAdaptiveQuality ? "ENABLED" : "DISABLED");
            }
        }
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Automatically adjusts quality settings to maintain target FPS\n"
                             "Uses ML model to predict frame time based on scene complexity");
        }

        if (m_enableAdaptiveQuality) {
            ImGui::Separator();

            // Target FPS
            const char* fpsTargets[] = { "60 FPS", "90 FPS", "120 FPS", "144 FPS", "165 FPS" };
            float fpsValues[] = { 60.0f, 90.0f, 120.0f, 144.0f, 165.0f };
            int currentFPS = 2; // Default to 120 FPS
            for (int i = 0; i < 5; i++) {
                if (std::abs(m_adaptiveTargetFPS - fpsValues[i]) < 0.1f) {
                    currentFPS = i;
                    break;
                }
            }

            if (ImGui::Combo("Target FPS", &currentFPS, fpsTargets, 5)) {
                m_adaptiveTargetFPS = fpsValues[currentFPS];
                if (m_adaptiveQuality) {
                    m_adaptiveQuality->SetTargetFPS(m_adaptiveTargetFPS);
                    LOG_INFO("Adaptive Quality Target FPS: {:.0f}", m_adaptiveTargetFPS);
                }
            }

            // Display current quality level
            if (m_adaptiveQuality) {
                ImGui::Separator();
                ImGui::Text("Current Quality Level:");

                auto currentQuality = m_adaptiveQuality->GetCurrentQuality();
                const char* qualityNames[] = { "Ultra", "High", "Medium", "Low", "Minimal" };
                ImVec4 qualityColors[] = {
                    ImVec4(1.0f, 0.0f, 1.0f, 1.0f),  // Ultra: Magenta
                    ImVec4(0.0f, 1.0f, 0.0f, 1.0f),  // High: Green
                    ImVec4(1.0f, 1.0f, 0.0f, 1.0f),  // Medium: Yellow
                    ImVec4(1.0f, 0.5f, 0.0f, 1.0f),  // Low: Orange
                    ImVec4(1.0f, 0.0f, 0.0f, 1.0f)   // Minimal: Red
                };

                int qualityIndex = static_cast<int>(currentQuality);
                ImGui::TextColored(qualityColors[qualityIndex], "  %s", qualityNames[qualityIndex]);

                // Show predicted frame time
                float predictedFrameTime = m_adaptiveQuality->GetPredictedFrameTime();
                float targetFrameTime = 1000.0f / m_adaptiveTargetFPS;
                ImGui::Text("Predicted Frame Time: %.2f ms", predictedFrameTime);
                ImGui::Text("Target Frame Time: %.2f ms", targetFrameTime);

                // Show performance bar
                float performanceRatio = predictedFrameTime / targetFrameTime;
                ImVec4 barColor = (performanceRatio < 1.1f) ? ImVec4(0.0f, 1.0f, 0.0f, 1.0f) :  // Green
                                  (performanceRatio < 1.5f) ? ImVec4(1.0f, 1.0f, 0.0f, 1.0f) :  // Yellow
                                                              ImVec4(1.0f, 0.0f, 0.0f, 1.0f);   // Red

                ImGui::ProgressBar((std::min)(performanceRatio, 2.0f) / 2.0f, ImVec2(-1, 0));
            }

            ImGui::Separator();

            // Data collection for training
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 1.0f, 1.0f), "Training Data Collection");
            if (ImGui::Checkbox("Collect Performance Data", &m_collectPerformanceData)) {
                if (m_adaptiveQuality) {
                    if (m_collectPerformanceData) {
                        m_adaptiveQuality->StartDataCollection("ml/training_data/performance_data.csv");
                        LOG_INFO("Started collecting performance data");
                    } else {
                        m_adaptiveQuality->StopDataCollection();
                        LOG_INFO("Stopped collecting performance data");
                    }
                }
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Collect training data for ML model\n"
                                 "Data saved to ml/training_data/performance_data.csv\n"
                                 "Use different scenarios (particle counts, camera distances, features)\n"
                                 "Then run: python ml/train_adaptive_quality.py");
            }

            if (m_collectPerformanceData) {
                ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "  Recording samples...");
            }
        }
    }

    // Physics controls
    if (ImGui::CollapsingHeader("Physics", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Physics Enabled (P)", &m_physicsEnabled);

        // PINN ML Physics Controls
        if (m_particleSystem) {
            ImGui::Separator();
            ImGui::Text("ML Physics (PINN)");

            if (m_particleSystem->IsPINNAvailable()) {
                bool pinnEnabled = m_particleSystem->IsPINNEnabled();
                if (ImGui::Checkbox("Enable PINN Physics (P key)", &pinnEnabled)) {
                    m_particleSystem->SetPINNEnabled(pinnEnabled);
                }

                if (pinnEnabled) {
                    ImGui::Indent();

                    // Hybrid mode toggle
                    bool hybridMode = m_particleSystem->IsPINNHybridMode();
                    if (ImGui::Checkbox("Hybrid Mode (PINN + GPU)", &hybridMode)) {
                        m_particleSystem->SetPINNHybridMode(hybridMode);
                    }
                    ImGui::SameLine();
                    ImGui::TextDisabled("(?)");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("PINN for far particles (r > threshold),\nGPU shader for near particles (r < threshold)");
                    }

                    // Hybrid threshold slider
                    if (hybridMode) {
                        float threshold = m_particleSystem->GetPINNHybridThreshold();
                        if (ImGui::SliderFloat("Hybrid Threshold (x R_ISCO)", &threshold, 5.0f, 50.0f, "%.1f")) {
                            m_particleSystem->SetPINNHybridThreshold(threshold);
                        }
                    }

                    // Performance metrics
                    auto metrics = m_particleSystem->GetPINNMetrics();
                    ImGui::Separator();
                    ImGui::Text("Performance:");
                    ImGui::Text("  Inference: %.2f ms", metrics.inferenceTimeMs);
                    ImGui::Text("  Particles: %u", metrics.particlesProcessed);
                    ImGui::Text("  Avg Batch: %.2f ms", metrics.avgBatchTimeMs);

                    ImGui::Unindent();
                }

                // Model info
                ImGui::TextDisabled("Model: PINN Accretion Disk (5 layers, 128 neurons)");
            } else {
                ImGui::TextDisabled("PINN: Not Available");
                ImGui::TextDisabled("(ONNX Runtime or model not found)");
            }

            ImGui::Separator();
            ImGui::Text("Dynamic Physics Parameters");
            float gravity = m_particleSystem->GetGravityStrength();
            if (ImGui::SliderFloat("Gravity (V)", &gravity, 0.0f, 2000.0f)) {
                m_particleSystem->SetGravityStrength(gravity);
            }
            float angularMomentum = m_particleSystem->GetAngularMomentum();
            if (ImGui::SliderFloat("Angular Momentum (N)", &angularMomentum, 0.0f, 5.0f)) {
                m_particleSystem->SetAngularMomentum(angularMomentum);
            }
            float turbulence = m_particleSystem->GetTurbulence();
            if (ImGui::SliderFloat("Turbulence (B)", &turbulence, 0.0f, 100.0f)) {
                m_particleSystem->SetTurbulence(turbulence);
            }
            float damping = m_particleSystem->GetDamping();
            if (ImGui::SliderFloat("Damping (M)", &damping, 0.0f, 1.0f)) {
                m_particleSystem->SetDamping(damping);
            }

            // NEW: Black hole mass (logarithmic slider)
            ImGui::Separator();
            ImGui::Text("Black Hole Physics");
            float blackHoleMass = m_particleSystem->GetBlackHoleMass();
            // Logarithmic slider: 1-1e10 solar masses (10^0 to 10^10)
            float logMass = log10f(blackHoleMass);
            if (ImGui::SliderFloat("Black Hole Mass (Ctrl/Shift+H)", &logMass, 0.0f, 10.0f, "10^%.1f")) {
                float newMass = powf(10.0f, logMass);
                m_particleSystem->SetBlackHoleMass(newMass);
            }
            // Display actual mass value
            if (blackHoleMass < 1e3f) {
                ImGui::Text("  = %.1f solar masses", blackHoleMass);
            } else if (blackHoleMass < 1e6f) {
                ImGui::Text("  = %.1f thousand M☉", blackHoleMass / 1e3f);
            } else if (blackHoleMass < 1e9f) {
                ImGui::Text("  = %.2f million M☉", blackHoleMass / 1e6f);
            } else {
                ImGui::Text("  = %.2f billion M☉", blackHoleMass / 1e9f);
            }

            // Quick presets
            ImGui::SameLine();
            if (ImGui::Button("Stellar")) { m_particleSystem->SetBlackHoleMass(10.0f); }
            ImGui::SameLine();
            if (ImGui::Button("Sgr A*")) { m_particleSystem->SetBlackHoleMass(4.3e6f); }
            ImGui::SameLine();
            if (ImGui::Button("Quasar")) { m_particleSystem->SetBlackHoleMass(1e9f); }

            // NEW: Alpha viscosity (Shakura-Sunyaev accretion parameter)
            float alphaViscosity = m_particleSystem->GetAlphaViscosity();
            if (ImGui::SliderFloat("Alpha Viscosity (Ctrl/Shift+X)", &alphaViscosity, 0.0f, 1.0f, "%.3f")) {
                m_particleSystem->SetAlphaViscosity(alphaViscosity);
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Shakura-Sunyaev α parameter\n"
                                "Controls inward spiral (accretion)\n"
                                "0.0 = no accretion, 0.1 = realistic, 1.0 = fast");
            }

            // NEW: Timescale (simulation speed multiplier)
            ImGui::Separator();
            float timeScale = m_particleSystem->GetTimeScale();
            if (ImGui::SliderFloat("Time Scale (Ctrl/Shift+T)", &timeScale, 0.0f, 10.0f, "%.2fx")) {
                m_particleSystem->SetTimeScale(timeScale);
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Simulation speed multiplier\n"
                                "0.0 = paused, 0.5 = half speed\n"
                                "1.0 = normal, 2.0 = double speed");
            }
            // Quick presets
            ImGui::SameLine();
            if (ImGui::Button("Pause")) { m_particleSystem->SetTimeScale(0.0f); }
            ImGui::SameLine();
            if (ImGui::Button("0.5x")) { m_particleSystem->SetTimeScale(0.5f); }
            ImGui::SameLine();
            if (ImGui::Button("1x")) { m_particleSystem->SetTimeScale(1.0f); }
            ImGui::SameLine();
            if (ImGui::Button("2x")) { m_particleSystem->SetTimeScale(2.0f); }
        }
        ImGui::Separator();
        ImGui::Text("Simulation Info");

        // Runtime particle count control
        int activeParticles = static_cast<int>(m_activeParticleCount);
        if (ImGui::SliderInt("Active Particles (=/−)", &activeParticles, 100, static_cast<int>(m_config.particleCount), "%d")) {
            m_activeParticleCount = static_cast<uint32_t>(activeParticles);
        }
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Runtime particle count control\nPress = to increase, - to decrease\nMax: %u (set at startup)", m_config.particleCount);
        }

        // Quick presets
        ImGui::SameLine();
        if (ImGui::Button("1K")) { m_activeParticleCount = 1000; }
        ImGui::SameLine();
        if (ImGui::Button("5K")) { m_activeParticleCount = 5000; }
        ImGui::SameLine();
        if (ImGui::Button("10K")) { m_activeParticleCount = 10000; }
        ImGui::SameLine();
        if (ImGui::Button("Max")) { m_activeParticleCount = m_config.particleCount; }

        ImGui::Separator();
        ImGui::Text("Accretion Disk Parameters (Read-only)");
        ImGui::Text("Inner Radius: %.1f", m_innerRadius);
        ImGui::Text("Outer Radius: %.1f", m_outerRadius);
        ImGui::Text("Disk Thickness: %.1f", m_diskThickness);
        ImGui::Text("Physics Timestep: %.6f (120Hz)", m_physicsTimeStep);
    }

    // Multi-Light System (Phase 3.5)
    if (ImGui::CollapsingHeader("Multi-Light System (L)")) {
        ImGui::Text("Active Lights: %d / 16", static_cast<int>(m_lights.size()));
        ImGui::SameLine();
        if (ImGui::Button("Add Light (+)") && m_lights.size() < 16) {
            // Add new light at camera position
            ParticleRenderer_Gaussian::Light newLight;
            newLight.position = DirectX::XMFLOAT3(m_cameraDistance * sinf(m_cameraAngle), m_cameraHeight, m_cameraDistance * cosf(m_cameraAngle));
            newLight.color = DirectX::XMFLOAT3(1.0f, 1.0f, 1.0f);
            newLight.intensity = 5.0f;
            newLight.radius = 10.0f;
            m_lights.push_back(newLight);
            m_selectedLightIndex = static_cast<int>(m_lights.size()) - 1;
            LOG_INFO("Added light {} at ({:.1f}, {:.1f}, {:.1f})", m_lights.size() - 1, newLight.position.x, newLight.position.y, newLight.position.z);
        }

        ImGui::Separator();

        // Physics-Driven Lights Toggle
        if (ImGui::Checkbox("Physics-Driven Lights (Celestial Bodies)", &m_physicsDrivenLights)) {
            if (m_physicsDrivenLights) {
                LOG_INFO("Physics-driven lights ENABLED - lights will orbit like celestial bodies");
            } else {
                LOG_INFO("Physics-driven lights DISABLED - lights remain static");
            }
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Enable: Lights move with Keplerian orbits like bright stars/clusters\n"
                              "Disable: Lights remain in fixed positions\n"
                              "\n"
                              "Creates dynamic shadows and illumination as lights orbit!\n"
                              "Very effective for visualizing celestial body movement.");
        }

        // Stellar Temperature Colors Toggle
        if (ImGui::Checkbox("Stellar Temperature Colors (Auto-Apply)", &m_useStellarTemperatureColors)) {
            if (m_useStellarTemperatureColors) {
                LOG_INFO("Stellar temperature colors ENABLED - light colors based on intensity");
                // Apply colors immediately
                for (size_t i = 0; i < m_lights.size(); i++) {
                    float temperature = GetStellarTemperatureFromIntensity(m_lights[i].intensity);
                    m_lights[i].color = GetStellarColorFromTemperature(temperature);
                    LOG_INFO("Light {}: Intensity={:.1f} → Temp={:.0f}K → Color=({:.2f}, {:.2f}, {:.2f})",
                             i, m_lights[i].intensity, temperature,
                             m_lights[i].color.x, m_lights[i].color.y, m_lights[i].color.z);
                }
            } else {
                LOG_INFO("Stellar temperature colors DISABLED - manual color control restored");
            }
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Enable: Light colors automatically match stellar temperature\n"
                              "Disable: Manual color control\n"
                              "\n"
                              "Intensity → Temperature → Color mapping:\n"
                              "  • Low intensity (0.1-5):   Red (M-type, 3000K)\n"
                              "  • Medium (5-10):           Yellow (G-type, 5500K) - Sun-like\n"
                              "  • High (10-15):            White (A-type, 8000K)\n"
                              "  • Very high (15-20):       Blue (O-type, 25000K)\n"
                              "\n"
                              "Based on Morgan-Keenan stellar classification system!");
        }

        ImGui::Separator();

        // Bulk Light Controls
        if (ImGui::TreeNode("Bulk Light Controls (Apply to All)")) {
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Apply settings to all lights simultaneously\n"
                                  "Useful for creating uniform stellar properties");
            }

            static DirectX::XMFLOAT3 bulkColor = DirectX::XMFLOAT3(1.0f, 1.0f, 1.0f);
            static float bulkIntensity = 10.0f;
            static float bulkRadius = 10.0f;

            ImGui::ColorEdit3("Bulk Color", &bulkColor.x);
            ImGui::SliderFloat("Bulk Intensity", &bulkIntensity, 0.1f, 20.0f, "%.1f");
            ImGui::SliderFloat("Bulk Radius", &bulkRadius, 1.0f, 200.0f, "%.1f");

            ImGui::Spacing();

            if (ImGui::Button("Apply Color to All")) {
                for (auto& light : m_lights) {
                    light.color = bulkColor;
                }
                LOG_INFO("Applied bulk color ({:.2f}, {:.2f}, {:.2f}) to {} lights",
                         bulkColor.x, bulkColor.y, bulkColor.z, m_lights.size());
            }
            ImGui::SameLine();

            if (ImGui::Button("Apply Intensity to All")) {
                for (auto& light : m_lights) {
                    light.intensity = bulkIntensity;
                }
                LOG_INFO("Applied bulk intensity {:.1f} to {} lights", bulkIntensity, m_lights.size());
            }

            if (ImGui::Button("Apply Radius to All")) {
                for (auto& light : m_lights) {
                    light.radius = bulkRadius;
                }
                LOG_INFO("Applied bulk radius {:.1f} to {} lights", bulkRadius, m_lights.size());
            }
            ImGui::SameLine();

            if (ImGui::Button("Apply All Properties")) {
                for (auto& light : m_lights) {
                    light.color = bulkColor;
                    light.intensity = bulkIntensity;
                    light.radius = bulkRadius;
                }
                LOG_INFO("Applied all bulk properties to {} lights: Color({:.2f}, {:.2f}, {:.2f}), Intensity={:.1f}, Radius={:.1f}",
                         m_lights.size(), bulkColor.x, bulkColor.y, bulkColor.z, bulkIntensity, bulkRadius);
            }

            ImGui::TreePop();
        }

        ImGui::Separator();

        // RTXDI-Optimized Presets (wide spatial distribution for grid-based sampling)
        if (m_lightingSystem == LightingSystem::RTXDI) {
            ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "RTXDI Presets (Grid-Optimized):");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Optimized for RTXDI spatial grid (30x30x30 cells)\n"
                                  "Wide distribution reduces jigsaw pattern");
            }

            if (ImGui::Button("Sphere (13)")) {
                InitializeRTXDISphereLights();
                m_selectedLightIndex = -1;
                LOG_INFO("Applied RTXDI Sphere preset (13 lights, 1200-unit radius)");
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Fibonacci sphere distribution\n"
                                  "Coverage: ~500 grid cells\n"
                                  "Best for: Smooth gradients");
            }
            ImGui::SameLine();

            if (ImGui::Button("Ring (16)")) {
                InitializeRTXDIRingLights();
                m_selectedLightIndex = -1;
                LOG_INFO("Applied RTXDI Ring preset (16 lights, disk formation)");
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Dual-ring accretion disk\n"
                                  "Coverage: ~600 grid cells\n"
                                  "Best for: Disk-like appearance");
            }
            ImGui::SameLine();

            // Grid (27) preset removed - exceeds 16-light hardware limit
            // Will be re-added when max lights increased to 32

            if (ImGui::Button("Sparse (5)")) {
                InitializeRTXDISparseLights();
                m_selectedLightIndex = -1;
                LOG_INFO("Applied RTXDI Sparse preset (5 lights, cross pattern)");
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Minimal light debug preset\n"
                                  "Coverage: ~200 grid cells\n"
                                  "Best for: Debugging grid behavior");
            }

            ImGui::Separator();
        }

        // Legacy Presets (optimized for multi-light brute-force)
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.4f, 1.0f), "Legacy Presets (Multi-Light):");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Original presets designed for multi-light mode\n"
                              "May create jigsaw patterns in RTXDI mode due to tight clustering");
        }

        if (ImGui::Button("Disk (13)")) {
            InitializeLights();
            m_selectedLightIndex = -1;
            LOG_INFO("Reset to default 13-light disk configuration");
        }
        ImGui::SameLine();
        if (ImGui::Button("Single")) {
            m_lights.clear();
            ParticleRenderer_Gaussian::Light single;
            single.position = DirectX::XMFLOAT3(0.0f, 0.0f, 0.0f);
            single.color = DirectX::XMFLOAT3(1.0f, 1.0f, 1.0f);
            single.intensity = 10.0f;
            single.radius = 5.0f;
            m_lights.push_back(single);
            m_selectedLightIndex = 0;
            LOG_INFO("Reset to single light at origin");
        }
        ImGui::SameLine();
        if (ImGui::Button("Dome (8)")) {
            m_lights.clear();
            const float PI = 3.14159265f;
            // 8 lights in hemisphere above disk
            for (int i = 0; i < 8; i++) {
                float angle = (i * 45.0f) * (PI / 180.0f);
                float radius = 200.0f;
                float height = 150.0f;
                ParticleRenderer_Gaussian::Light light;
                light.position = DirectX::XMFLOAT3(cos(angle) * radius, height, sin(angle) * radius);
                light.color = DirectX::XMFLOAT3(1.0f, 0.95f, 0.9f);
                light.intensity = 7.0f;
                light.radius = 15.0f;
                m_lights.push_back(light);
            }
            m_selectedLightIndex = 0;
            LOG_INFO("Created dome configuration with 8 elevated lights");
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear")) {
            m_lights.clear();
            m_selectedLightIndex = -1;
            LOG_INFO("Cleared all lights");
        }

        ImGui::Separator();

        // Individual light controls
        for (int i = 0; i < static_cast<int>(m_lights.size()); i++) {
            ImGui::PushID(i);

            bool isSelected = (i == m_selectedLightIndex);
            if (isSelected) {
                ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.4f, 0.6f, 0.8f, 0.8f));
            }

            char label[64];
            snprintf(label, sizeof(label), "Light %d###Light%d", i, i);
            if (ImGui::TreeNode(label)) {
                m_selectedLightIndex = i;

                ImGui::DragFloat3("Position", &m_lights[i].position.x, 5.0f, -2000.0f, 2000.0f, "%.1f");
                ImGui::ColorEdit3("Color", &m_lights[i].color.x);
                ImGui::SliderFloat("Intensity", &m_lights[i].intensity, 0.0f, 20.0f, "%.1f");
                ImGui::SliderFloat("Radius", &m_lights[i].radius, 1.0f, 50.0f, "%.1f");

                ImGui::Separator();
                ImGui::Text("Position: (%.1f, %.1f, %.1f)", m_lights[i].position.x, m_lights[i].position.y, m_lights[i].position.z);

                if (ImGui::Button("Move to Camera")) {
                    m_lights[i].position.x = m_cameraDistance * sinf(m_cameraAngle);
                    m_lights[i].position.y = m_cameraHeight;
                    m_lights[i].position.z = m_cameraDistance * cosf(m_cameraAngle);
                    LOG_INFO("Moved light {} to camera position", i);
                }
                ImGui::SameLine();
                if (ImGui::Button("Delete (−)")) {
                    m_lights.erase(m_lights.begin() + i);
                    m_selectedLightIndex = -1;
                    LOG_INFO("Deleted light {}", i);
                    ImGui::TreePop();
                    ImGui::PopID();
                    if (isSelected) ImGui::PopStyleColor();
                    break;  // Exit loop after deletion
                }

                ImGui::TreePop();
            }

            if (isSelected) {
                ImGui::PopStyleColor();
            }

            ImGui::PopID();
        }

        // === GOD RAY SYSTEM (Phase 5 Milestone 5.3c) ===
        ImGui::Separator();
        ImGui::Separator();

        if (ImGui::TreeNode("God Ray System")) {
            ImGui::TextColored(ImVec4(0.8f, 1.0f, 0.4f, 1.0f), "Volumetric light beams (static in world space)");

            // Global controls
            ImGui::Separator();
            ImGui::Text("Global God Ray Settings:");
            ImGui::SliderFloat("God Ray Density", &m_godRayDensity, 0.0f, 1.0f, "%.2f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Global multiplier for all god ray brightness\n0.0 = disabled, 1.0 = full intensity");
            }

            ImGui::SliderFloat("Step Multiplier", &m_godRayStepMultiplier, 0.5f, 2.0f, "%.2f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Ray marching step size for god rays\n<1.0 = higher quality (slower)\n>1.0 = lower quality (faster)");
            }

            // Presets
            ImGui::Separator();
            ImGui::Text("God Ray Presets:");

            if (ImGui::Button("Static Downward Beams")) {
                m_godRayDensity = 0.5f;
                for (auto& light : m_lights) {
                    light.enableGodRays = 1.0f;
                    light.godRayIntensity = 2.0f;
                    light.godRayLength = 1500.0f;
                    light.godRayConeAngle = 0.3f;  // ~17 degrees
                    light.godRayDirection = DirectX::XMFLOAT3(0.0f, -1.0f, 0.0f);
                    light.godRayFalloff = 2.0f;
                    light.godRayRotationSpeed = 0.0f;
                }
                LOG_INFO("Applied god ray preset: Static Downward Beams");
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("All lights point straight down with moderate spread");
            }

            ImGui::SameLine();

            if (ImGui::Button("Rotating Searchlights")) {
                m_godRayDensity = 0.7f;
                for (auto& light : m_lights) {
                    light.enableGodRays = 1.0f;
                    light.godRayIntensity = 4.0f;
                    light.godRayLength = 2000.0f;
                    light.godRayConeAngle = 0.2f;  // ~11 degrees (narrow)
                    light.godRayDirection = DirectX::XMFLOAT3(0.0f, -1.0f, 0.0f);
                    light.godRayFalloff = 5.0f;  // Sharp edges
                    light.godRayRotationSpeed = 0.5f;  // Slow rotation
                }
                LOG_INFO("Applied god ray preset: Rotating Searchlights");
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Narrow rotating beams (cinematic searchlight effect)");
            }

            ImGui::SameLine();

            if (ImGui::Button("Radial Burst")) {
                m_godRayDensity = 0.3f;
                for (size_t i = 0; i < m_lights.size(); i++) {
                    auto& light = m_lights[i];
                    light.enableGodRays = 1.0f;
                    light.godRayIntensity = 3.0f;
                    light.godRayLength = 3000.0f;
                    light.godRayConeAngle = 0.5f;  // ~28 degrees (wide)

                    // Radial outward from light position
                    DirectX::XMVECTOR pos = DirectX::XMLoadFloat3(&light.position);
                    DirectX::XMVECTOR dir = DirectX::XMVector3Normalize(pos);
                    DirectX::XMStoreFloat3(&light.godRayDirection, dir);

                    light.godRayFalloff = 1.5f;  // Soft edges
                    light.godRayRotationSpeed = 0.0f;
                }
                LOG_INFO("Applied god ray preset: Radial Burst");
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Beams radiate outward from each light (stellar nursery effect)");
            }

            ImGui::SameLine();

            if (ImGui::Button("Disable All")) {
                m_godRayDensity = 0.0f;
                for (auto& light : m_lights) {
                    light.enableGodRays = 0.0f;
                }
                LOG_INFO("Disabled all god rays");
            }

            // Per-light controls
            ImGui::Separator();
            ImGui::Text("Per-Light God Ray Controls:");

            for (size_t i = 0; i < m_lights.size(); i++) {
                ImGui::PushID(static_cast<int>(i) + 1000);  // Unique ID for god ray controls

                char label[64];
                snprintf(label, sizeof(label), "Light %zu God Rays###GodRay%zu", i, i);

                if (ImGui::TreeNode(label)) {
                    bool enabled = (m_lights[i].enableGodRays > 0.5f);
                    if (ImGui::Checkbox("Enable God Rays", &enabled)) {
                        m_lights[i].enableGodRays = enabled ? 1.0f : 0.0f;
                    }

                    if (enabled) {
                        ImGui::SliderFloat("Intensity##godray", &m_lights[i].godRayIntensity, 0.0f, 10.0f, "%.1f");
                        ImGui::SliderFloat("Length##godray", &m_lights[i].godRayLength, 100.0f, 5000.0f, "%.0f");

                        // Cone angle in degrees for user-friendliness
                        float coneAngleDegrees = m_lights[i].godRayConeAngle * 180.0f / 3.14159f;
                        if (ImGui::SliderFloat("Cone Angle (degrees)", &coneAngleDegrees, 1.0f, 90.0f, "%.1f")) {
                            m_lights[i].godRayConeAngle = coneAngleDegrees * 3.14159f / 180.0f;
                        }

                        ImGui::SliderFloat("Falloff##godray", &m_lights[i].godRayFalloff, 0.1f, 10.0f, "%.1f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Higher = sharper beam edges\nLower = softer, wider beam");
                        }

                        ImGui::Text("Direction:");
                        ImGui::DragFloat3("Dir##godray", &m_lights[i].godRayDirection.x, 0.01f, -1.0f, 1.0f, "%.2f");

                        if (ImGui::Button("Normalize Direction")) {
                            DirectX::XMVECTOR dir = DirectX::XMLoadFloat3(&m_lights[i].godRayDirection);
                            dir = DirectX::XMVector3Normalize(dir);
                            DirectX::XMStoreFloat3(&m_lights[i].godRayDirection, dir);
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("Downward")) {
                            m_lights[i].godRayDirection = DirectX::XMFLOAT3(0.0f, -1.0f, 0.0f);
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("Radial Out")) {
                            DirectX::XMVECTOR pos = DirectX::XMLoadFloat3(&m_lights[i].position);
                            DirectX::XMVECTOR dir = DirectX::XMVector3Normalize(pos);
                            DirectX::XMStoreFloat3(&m_lights[i].godRayDirection, dir);
                        }

                        ImGui::SliderFloat("Rotation Speed (rad/s)", &m_lights[i].godRayRotationSpeed, -3.14f, 3.14f, "%.2f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("0.0 = static beam\n>0.0 = rotate counterclockwise\n<0.0 = rotate clockwise");
                        }
                    }

                    ImGui::TreePop();
                }

                ImGui::PopID();
            }

            ImGui::TreePop();
        }

        // === Bulk Light Color Controls (Phase 5 Milestone 5.3b) ===
        ImGui::Separator();
        ImGui::Separator();

        if (ImGui::TreeNode("Bulk Light Color Controls")) {
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.6f, 1.0f), "Quick color changes for all lights!");

            // === SECTION 1: Selection ===
            if (ImGui::TreeNode("Light Selection")) {
                const char* selectionModes[] = {
                    "All Lights", "Inner Ring", "Outer Ring", "Top Half",
                    "Bottom Half", "Even Indices", "Odd Indices", "Custom Range"
                };

                int currentSelection = (int)m_lightSelection;
                if (ImGui::Combo("Select Lights", &currentSelection, selectionModes, 8)) {
                    m_lightSelection = (LightSelection)currentSelection;
                }

                // Custom range controls
                if (m_lightSelection == LightSelection::CustomRange) {
                    ImGui::SliderInt("Range Start", &m_customRangeStart, 0, (int)m_lights.size() - 1);
                    ImGui::SliderInt("Range End", &m_customRangeEnd, 0, (int)m_lights.size() - 1);
                }

                // Radial threshold
                if (m_lightSelection == LightSelection::InnerRing || m_lightSelection == LightSelection::OuterRing) {
                    ImGui::SliderFloat("Radial Threshold", &m_radialThreshold, 100.0f, 2000.0f);
                }

                // Show selected count
                auto selectedIndices = GetSelectedLightIndices();
                ImGui::Text("Selected: %d lights", (int)selectedIndices.size());

                ImGui::TreePop();
            }

            ImGui::Separator();

            // === SECTION 2: Color Presets ===
            if (ImGui::TreeNode("Color Presets")) {

                ImGui::Text("Temperature Presets:");
                if (ImGui::Button("Cool Blue (10000K)")) ApplyColorPreset(ColorPreset::CoolBlue);
                ImGui::SameLine();
                if (ImGui::Button("White (6500K)")) ApplyColorPreset(ColorPreset::White);

                if (ImGui::Button("Warm White (4000K)")) ApplyColorPreset(ColorPreset::WarmWhite);
                ImGui::SameLine();
                if (ImGui::Button("Warm Sunset (2500K)")) ApplyColorPreset(ColorPreset::WarmSunset);

                if (ImGui::Button("Deep Red (1800K)")) ApplyColorPreset(ColorPreset::DeepRed);

                ImGui::Separator();

                ImGui::Text("Artistic Presets:");
                if (ImGui::Button("Rainbow")) ApplyColorPreset(ColorPreset::Rainbow);
                ImGui::SameLine();
                if (ImGui::Button("Complementary")) ApplyColorPreset(ColorPreset::Complementary);

                if (ImGui::Button("Monochrome Blue")) ApplyColorPreset(ColorPreset::MonochromeBlue);
                ImGui::SameLine();
                if (ImGui::Button("Monochrome Red")) ApplyColorPreset(ColorPreset::MonochromeRed);

                if (ImGui::Button("Monochrome Green")) ApplyColorPreset(ColorPreset::MonochromeGreen);
                ImGui::SameLine();
                if (ImGui::Button("Neon")) ApplyColorPreset(ColorPreset::Neon);

                if (ImGui::Button("Pastel")) ApplyColorPreset(ColorPreset::Pastel);

                ImGui::Separator();

                ImGui::Text("Scenario Presets:");
                if (ImGui::Button("Stellar Nursery")) ApplyColorPreset(ColorPreset::StellarNursery);
                ImGui::SameLine();
                if (ImGui::Button("Red Giant")) ApplyColorPreset(ColorPreset::RedGiant);

                if (ImGui::Button("Accretion Disk")) ApplyColorPreset(ColorPreset::AccretionDisk);
                ImGui::SameLine();
                if (ImGui::Button("Binary System")) ApplyColorPreset(ColorPreset::BinarySystem);

                if (ImGui::Button("Dust Torus")) ApplyColorPreset(ColorPreset::DustTorus);

                ImGui::TreePop();
            }

            ImGui::Separator();

            // === SECTION 3: Gradient Application ===
            if (ImGui::TreeNode("Gradient Application")) {

                const char* gradientTypes[] = {
                    "Radial (Distance)", "Linear X", "Linear Y", "Linear Z", "Circular (Angle)"
                };

                int currentGradient = (int)m_gradientType;
                if (ImGui::Combo("Gradient Type", &currentGradient, gradientTypes, 5)) {
                    m_gradientType = (GradientType)currentGradient;
                }

                ImGui::ColorEdit3("Start Color", &m_gradientColorStart.x);
                ImGui::ColorEdit3("End Color", &m_gradientColorEnd.x);

                if (ImGui::Button("Apply Gradient")) {
                    ApplyGradient(m_gradientType, m_gradientColorStart, m_gradientColorEnd);
                    m_currentColorPreset = ColorPreset::Custom;
                }

                ImGui::TreePop();
            }

            ImGui::Separator();

            // === SECTION 4: Global Color Operations ===
            if (ImGui::TreeNode("Global Color Operations")) {

                // Hue shift
                ImGui::SliderFloat("Hue Shift (degrees)", &m_hueShift, -180.0f, 180.0f);
                if (ImGui::Button("Apply Hue Shift")) {
                    ApplyGlobalHueShift(m_hueShift);
                    m_currentColorPreset = ColorPreset::Custom;
                }
                ImGui::SameLine();
                if (ImGui::Button("Reset##hue")) {
                    m_hueShift = 0.0f;
                }

                // Saturation adjust
                ImGui::SliderFloat("Saturation Multiplier", &m_saturationAdjust, 0.0f, 2.0f);
                if (ImGui::Button("Apply Saturation")) {
                    ApplyGlobalSaturationAdjust(m_saturationAdjust);
                    m_currentColorPreset = ColorPreset::Custom;
                }
                ImGui::SameLine();
                if (ImGui::Button("Reset##sat")) {
                    m_saturationAdjust = 1.0f;
                }

                // Value adjust
                ImGui::SliderFloat("Brightness Multiplier", &m_valueAdjust, 0.0f, 2.0f);
                if (ImGui::Button("Apply Brightness")) {
                    ApplyGlobalValueAdjust(m_valueAdjust);
                    m_currentColorPreset = ColorPreset::Custom;
                }
                ImGui::SameLine();
                if (ImGui::Button("Reset##val")) {
                    m_valueAdjust = 1.0f;
                }

                // Temperature shift
                ImGui::SliderFloat("Temperature Shift", &m_temperatureShift, -1.0f, 1.0f);
                if (ImGui::Button("Apply Temperature Shift")) {
                    ApplyTemperatureShift(m_temperatureShift);
                    m_currentColorPreset = ColorPreset::Custom;
                }
                ImGui::SameLine();
                if (ImGui::Button("Reset##temp")) {
                    m_temperatureShift = 0.0f;
                }

                ImGui::TreePop();
            }

            ImGui::Separator();

            // === SECTION 5: Quick Actions ===
            if (ImGui::Button("Copy Light 0 Color to All")) {
                if (m_lights.size() > 0) {
                    DirectX::XMFLOAT3 color = m_lights[0].color;
                    auto indices = GetSelectedLightIndices();
                    for (int idx : indices) {
                        m_lights[idx].color = color;
                    }
                    m_currentColorPreset = ColorPreset::Custom;
                }
            }

            if (ImGui::Button("Randomize Colors")) {
                auto indices = GetSelectedLightIndices();
                for (int idx : indices) {
                    float h = (float)(rand() % 360);
                    float s = 0.8f + (float)(rand() % 20) / 100.0f;  // 0.8-1.0
                    float v = 0.9f + (float)(rand() % 10) / 100.0f;  // 0.9-1.0
                    m_lights[idx].color = HSVtoRGB(DirectX::XMFLOAT3(h, s, v));
                }
                m_currentColorPreset = ColorPreset::Custom;
            }

            ImGui::TreePop();
        }
    }

    ImGui::End();

    // Rendering
    ImGui::Render();
}

// Initialize 13-light configuration for accretion disk
void Application::InitializeLights() {
    using DirectX::XMFLOAT3;
    const float PI = 3.14159265f;

    m_lights.clear();

    // Primary: Hot inner edge at origin (blue-white 20000K)
    ParticleRenderer_Gaussian::Light primaryLight;
    primaryLight.position = XMFLOAT3(0.0f, 0.0f, 0.0f);
    primaryLight.color = XMFLOAT3(1.0f, 0.9f, 0.8f);  // Blue-white
    primaryLight.intensity = 15.0f;  // Boosted from 10.0 for visibility
    primaryLight.radius = 80.0f;     // Boosted from 5.0 for wider coverage
    m_lights.push_back(primaryLight);

    // Secondary: 4 spiral arms at 50 unit radius (orange 12000K)
    for (int i = 0; i < 4; i++) {
        float angle = (i * 90.0f) * (PI / 180.0f);
        float radius = 50.0f;

        ParticleRenderer_Gaussian::Light armLight;
        armLight.position = XMFLOAT3(cos(angle) * radius, 0.0f, sin(angle) * radius);
        armLight.color = XMFLOAT3(1.0f, 0.8f, 0.6f);  // Orange
        armLight.intensity = 12.0f;  // Boosted from 5.0 for visibility
        armLight.radius = 100.0f;    // Boosted from 10.0 for wider coverage
        m_lights.push_back(armLight);
    }

    // Tertiary: 8 mid-disk hot spots at 150 unit radius (yellow-orange 8000K)
    for (int i = 0; i < 8; i++) {
        float angle = (i * 45.0f) * (PI / 180.0f);
        float radius = 150.0f;

        ParticleRenderer_Gaussian::Light hotSpot;
        hotSpot.position = XMFLOAT3(cos(angle) * radius, 0.0f, sin(angle) * radius);
        hotSpot.color = XMFLOAT3(1.0f, 0.7f, 0.4f);  // Yellow-orange
        hotSpot.intensity = 8.0f;   // Boosted from 2.0 for visibility
        hotSpot.radius = 120.0f;    // Boosted from 15.0 for wider coverage
        m_lights.push_back(hotSpot);
    }

    // Initialize god ray parameters (all disabled by default)
    for (auto& light : m_lights) {
        light.enableGodRays = 0.0f;                                  // Disabled
        light.godRayIntensity = 2.0f;                                // Moderate brightness
        light.godRayLength = 1500.0f;                                // Medium-range beam
        light.godRayFalloff = 2.0f;                                  // Moderate sharpness
        light.godRayDirection = DirectX::XMFLOAT3(0.0f, -1.0f, 0.0f); // Downward
        light.godRayConeAngle = 0.3f;                                // ~17 degrees
        light.godRayRotationSpeed = 0.0f;                            // Static
        light._padding = 0.0f;                                       // Zero padding
    }

    LOG_INFO("Initialized multi-light system: {} lights", m_lights.size());
    LOG_INFO("  1 primary (origin, blue-white, 20000K equiv)");
    LOG_INFO("  4 secondary (spiral arms @ 50 units, orange, 12000K equiv)");
    LOG_INFO("  8 tertiary (hot spots @ 150 units, yellow-orange, 8000K equiv)");
    LOG_INFO("  God ray parameters initialized (all disabled by default)");
}

// RTXDI Preset 1: Sphere (13) - Fibonacci sphere distribution
void Application::InitializeRTXDISphereLights() {
    using DirectX::XMFLOAT3;
    const float PI = 3.14159265f;
    const float PHI = 1.618033988749f;  // Golden ratio
    const float sphereRadius = 300.0f;  // 300-unit radius sphere (matches particle outer radius)

    m_lights.clear();

    // Generate 13 lights using Fibonacci sphere algorithm
    // This creates evenly-distributed points on a sphere surface
    for (int i = 0; i < 13; i++) {
        // Fibonacci sphere mapping
        float y = 1.0f - (i / (13.0f - 1.0f)) * 2.0f;  // -1 to 1
        float radiusAtY = sqrtf(1.0f - y * y);
        float theta = PHI * i * 2.0f * PI;

        // Convert to Cartesian coordinates
        float x = cosf(theta) * radiusAtY;
        float z = sinf(theta) * radiusAtY;

        // Scale to sphere radius
        ParticleRenderer_Gaussian::Light light;
        light.position = XMFLOAT3(x * sphereRadius, y * sphereRadius, z * sphereRadius);

        // Color based on height (blue-white at top, orange-red at bottom)
        float heightFactor = (y + 1.0f) * 0.5f;  // 0 to 1
        light.color = XMFLOAT3(
            1.0f,
            0.7f + heightFactor * 0.3f,  // 0.7-1.0
            0.5f + heightFactor * 0.5f   // 0.5-1.0
        );

        light.intensity = 8.0f;   // Lower intensity, wider distribution compensates
        light.radius = 150.0f;    // Large radius for smooth falloff
        m_lights.push_back(light);
    }

    // Initialize god ray parameters
    for (auto& light : m_lights) {
        light.enableGodRays = 0.0f;
        light.godRayIntensity = 2.0f;
        light.godRayLength = 1500.0f;
        light.godRayFalloff = 2.0f;
        light.godRayDirection = DirectX::XMFLOAT3(0.0f, -1.0f, 0.0f);
        light.godRayConeAngle = 0.3f;
        light.godRayRotationSpeed = 0.0f;
        light._padding = 0.0f;
    }

    LOG_INFO("Initialized RTXDI Sphere preset: {} lights", m_lights.size());
    LOG_INFO("  Fibonacci sphere distribution @ 300-unit radius");
    LOG_INFO("  Expected grid coverage: ~100 cells (~0.37% occupancy)");
}

// RTXDI Preset 2: Ring (16) - Dual-ring accretion disk formation
void Application::InitializeRTXDIRingLights() {
    using DirectX::XMFLOAT3;
    const float PI = 3.14159265f;

    m_lights.clear();

    // Inner ring: 8 lights at 150-unit radius
    for (int i = 0; i < 8; i++) {
        float angle = (i * 45.0f) * (PI / 180.0f);
        float radius = 150.0f;
        float height = (i % 2 == 0) ? 100.0f : -100.0f;  // Alternating height

        ParticleRenderer_Gaussian::Light light;
        light.position = XMFLOAT3(cosf(angle) * radius, height, sinf(angle) * radius);
        light.color = XMFLOAT3(1.0f, 0.8f, 0.5f);  // Warm orange
        light.intensity = 12.0f;
        light.radius = 120.0f;
        m_lights.push_back(light);
    }

    // Outer ring: 8 lights at 250-unit radius
    for (int i = 0; i < 8; i++) {
        float angle = ((i * 45.0f) + 22.5f) * (PI / 180.0f);  // Offset by 22.5° from inner ring
        float radius = 250.0f;
        float height = (i % 2 == 0) ? 150.0f : -150.0f;  // Alternating height

        ParticleRenderer_Gaussian::Light light;
        light.position = XMFLOAT3(cosf(angle) * radius, height, sinf(angle) * radius);
        light.color = XMFLOAT3(1.0f, 0.7f, 0.4f);  // Yellow-orange
        light.intensity = 8.0f;
        light.radius = 150.0f;
        m_lights.push_back(light);
    }

    // Initialize god ray parameters
    for (auto& light : m_lights) {
        light.enableGodRays = 0.0f;
        light.godRayIntensity = 2.0f;
        light.godRayLength = 1500.0f;
        light.godRayFalloff = 2.0f;
        light.godRayDirection = DirectX::XMFLOAT3(0.0f, -1.0f, 0.0f);
        light.godRayConeAngle = 0.3f;
        light.godRayRotationSpeed = 0.0f;
        light._padding = 0.0f;
    }

    LOG_INFO("Initialized RTXDI Ring preset: {} lights", m_lights.size());
    LOG_INFO("  8 inner ring @ 150 units, 8 outer ring @ 250 units");
    LOG_INFO("  Expected grid coverage: ~50 cells (~0.19% occupancy)");
}

// RTXDI Preset 3: Grid (27) - 3×3×3 cubic grid
void Application::InitializeRTXDIGridLights() {
    using DirectX::XMFLOAT3;

    m_lights.clear();

    // 3×3×3 grid with 600-unit spacing (-900, -300, +300, +900 per axis)
    const float positions[3] = { -900.0f, 0.0f, 900.0f };

    for (int x = 0; x < 3; x++) {
        for (int y = 0; y < 3; y++) {
            for (int z = 0; z < 3; z++) {
                ParticleRenderer_Gaussian::Light light;
                light.position = XMFLOAT3(positions[x], positions[y], positions[z]);

                // Color varies by position (creates gradient effect)
                light.color = XMFLOAT3(
                    0.5f + (x / 2.0f) * 0.5f,  // 0.5-1.0 based on X
                    0.5f + (y / 2.0f) * 0.5f,  // 0.5-1.0 based on Y
                    0.5f + (z / 2.0f) * 0.5f   // 0.5-1.0 based on Z
                );

                light.intensity = 6.0f;   // Lower intensity for many lights
                light.radius = 180.0f;    // Large radius for maximum smoothness
                m_lights.push_back(light);
            }
        }
    }

    // Initialize god ray parameters
    for (auto& light : m_lights) {
        light.enableGodRays = 0.0f;
        light.godRayIntensity = 2.0f;
        light.godRayLength = 1500.0f;
        light.godRayFalloff = 2.0f;
        light.godRayDirection = DirectX::XMFLOAT3(0.0f, -1.0f, 0.0f);
        light.godRayConeAngle = 0.3f;
        light.godRayRotationSpeed = 0.0f;
        light._padding = 0.0f;
    }

    LOG_INFO("Initialized RTXDI Grid preset: {} lights", m_lights.size());
    LOG_INFO("  3×3×3 cubic grid with 600-unit spacing");
    LOG_INFO("  Expected grid coverage: ~1200 cells (~4.4% occupancy)");
}

// RTXDI Preset 4: Sparse (5) - Minimal debug preset
void Application::InitializeRTXDISparseLights() {
    using DirectX::XMFLOAT3;

    m_lights.clear();

    // Center light
    ParticleRenderer_Gaussian::Light center;
    center.position = XMFLOAT3(0.0f, 0.0f, 0.0f);
    center.color = XMFLOAT3(1.0f, 1.0f, 1.0f);  // White
    center.intensity = 15.0f;
    center.radius = 200.0f;
    m_lights.push_back(center);

    // 4 axis lights (X and Z only, no Y variation for disk appearance)
    const XMFLOAT3 axisPositions[4] = {
        XMFLOAT3(250.0f, 0.0f, 0.0f),   // +X (Red tint)
        XMFLOAT3(-250.0f, 0.0f, 0.0f),  // -X (Cyan tint)
        XMFLOAT3(0.0f, 0.0f, 250.0f),   // +Z (Blue tint)
        XMFLOAT3(0.0f, 0.0f, -250.0f)   // -Z (Yellow tint)
    };

    const XMFLOAT3 axisColors[4] = {
        XMFLOAT3(1.0f, 0.6f, 0.6f),  // Red tint
        XMFLOAT3(0.6f, 1.0f, 1.0f),  // Cyan tint
        XMFLOAT3(0.6f, 0.6f, 1.0f),  // Blue tint
        XMFLOAT3(1.0f, 1.0f, 0.6f)   // Yellow tint
    };

    for (int i = 0; i < 4; i++) {
        ParticleRenderer_Gaussian::Light light;
        light.position = axisPositions[i];
        light.color = axisColors[i];
        light.intensity = 10.0f;
        light.radius = 150.0f;
        m_lights.push_back(light);
    }

    // Initialize god ray parameters
    for (auto& light : m_lights) {
        light.enableGodRays = 0.0f;
        light.godRayIntensity = 2.0f;
        light.godRayLength = 1500.0f;
        light.godRayFalloff = 2.0f;
        light.godRayDirection = DirectX::XMFLOAT3(0.0f, -1.0f, 0.0f);
        light.godRayConeAngle = 0.3f;
        light.godRayRotationSpeed = 0.0f;
        light._padding = 0.0f;
    }

    LOG_INFO("Initialized RTXDI Sparse preset: {} lights", m_lights.size());
    LOG_INFO("  1 center + 4 axis lights @ 250 units (cross pattern)");
    LOG_INFO("  Expected grid coverage: ~40 cells (~0.15% occupancy)");
}

// ============================================================================
// Bulk Light Color Control System (Phase 5 Milestone 5.3b)
// ============================================================================

DirectX::XMFLOAT3 Application::RGBtoHSV(DirectX::XMFLOAT3 rgb) {
    float r = rgb.x, g = rgb.y, b = rgb.z;
    float max = (std::max)({r, g, b});
    float min = (std::min)({r, g, b});
    float delta = max - min;

    // Hue
    float h = 0.0f;
    if (delta > 0.001f) {
        if (max == r) {
            h = 60.0f * fmod((g - b) / delta, 6.0f);
        } else if (max == g) {
            h = 60.0f * ((b - r) / delta + 2.0f);
        } else {
            h = 60.0f * ((r - g) / delta + 4.0f);
        }
    }
    if (h < 0.0f) h += 360.0f;

    // Saturation
    float s = (max < 0.001f) ? 0.0f : (delta / max);

    // Value
    float v = max;

    return DirectX::XMFLOAT3(h, s, v);
}

DirectX::XMFLOAT3 Application::HSVtoRGB(DirectX::XMFLOAT3 hsv) {
    float h = hsv.x, s = hsv.y, v = hsv.z;

    float c = v * s;
    float x = c * (1.0f - fabs(fmod(h / 60.0f, 2.0f) - 1.0f));
    float m = v - c;

    float r, g, b;
    if (h < 60.0f) {
        r = c; g = x; b = 0.0f;
    } else if (h < 120.0f) {
        r = x; g = c; b = 0.0f;
    } else if (h < 180.0f) {
        r = 0.0f; g = c; b = x;
    } else if (h < 240.0f) {
        r = 0.0f; g = x; b = c;
    } else if (h < 300.0f) {
        r = x; g = 0.0f; b = c;
    } else {
        r = c; g = 0.0f; b = x;
    }

    return DirectX::XMFLOAT3(r + m, g + m, b + m);
}

DirectX::XMFLOAT3 Application::BlackbodyColor(float temperature) {
    // Simplified Planck blackbody approximation
    // Temperature in Kelvin → RGB color

    float t = temperature / 100.0f;
    float r, g, b;

    // Red
    if (t <= 66.0f) {
        r = 1.0f;
    } else {
        r = 1.292936186f * pow(t - 60.0f, -0.1332047592f);
        r = (std::min)(1.0f, (std::max)(0.0f, r));
    }

    // Green
    if (t <= 66.0f) {
        g = 0.39008157876f * log(t) - 0.631841444f;
    } else {
        g = 1.129890861f * pow(t - 60.0f, -0.0755148492f);
    }
    g = (std::min)(1.0f, (std::max)(0.0f, g));

    // Blue
    if (t >= 66.0f) {
        b = 1.0f;
    } else if (t <= 19.0f) {
        b = 0.0f;
    } else {
        b = 0.543206789f * log(t - 10.0f) - 1.196254089f;
        b = (std::min)(1.0f, (std::max)(0.0f, b));
    }

    return DirectX::XMFLOAT3(r, g, b);
}

// ====================================================================
// Stellar Temperature Color System (Phase 2)
// ====================================================================

float Application::GetStellarTemperatureFromIntensity(float intensity) {
    // Map intensity (0.1 - 20.0) to stellar temperature (3000K - 30000K)
    // Low intensity = cool red stars (M-type, 3000K)
    // High intensity = hot blue stars (O-type, 30000K)

    const float minTemp = 3000.0f;   // M-type red dwarf/supergiant
    const float maxTemp = 30000.0f;  // O-type blue supergiant
    const float minIntensity = 0.1f;
    const float maxIntensity = 20.0f;

    // Linear mapping
    float t = (intensity - minIntensity) / (maxIntensity - minIntensity);
    t = (std::max)(0.0f, (std::min)(1.0f, t));  // Clamp to [0,1]

    return minTemp + t * (maxTemp - minTemp);
}

DirectX::XMFLOAT3 Application::GetStellarColorFromTemperature(float temperature) {
    // Map temperature to stellar classification colors
    // Based on Morgan-Keenan spectral classification system

    if (temperature >= 30000.0f) {
        // O-type: Blue (30,000K+)
        return DirectX::XMFLOAT3(0.6f, 0.7f, 1.0f);
    } else if (temperature >= 10000.0f) {
        // B-type: Blue-white (10,000K - 30,000K)
        float t = (temperature - 10000.0f) / 20000.0f;
        return DirectX::XMFLOAT3(
            0.6f + t * 0.2f,   // 0.6 → 0.8
            0.7f + t * 0.15f,  // 0.7 → 0.85
            1.0f
        );
    } else if (temperature >= 7500.0f) {
        // A-type: White (7,500K - 10,000K)
        float t = (temperature - 7500.0f) / 2500.0f;
        return DirectX::XMFLOAT3(
            0.8f + t * 0.2f,   // 0.8 → 1.0
            0.85f + t * 0.15f, // 0.85 → 1.0
            1.0f
        );
    } else if (temperature >= 6000.0f) {
        // F-type: Yellow-white (6,000K - 7,500K)
        float t = (temperature - 6000.0f) / 1500.0f;
        return DirectX::XMFLOAT3(
            1.0f,
            0.9f + t * 0.1f,   // 0.9 → 1.0
            0.7f + t * 0.3f    // 0.7 → 1.0 (moving from yellow to white)
        );
    } else if (temperature >= 5200.0f) {
        // G-type: Yellow - Sun-like! (5,200K - 6,000K)
        float t = (temperature - 5200.0f) / 800.0f;
        return DirectX::XMFLOAT3(
            1.0f,
            0.85f + t * 0.05f, // 0.85 → 0.9
            0.6f + t * 0.1f    // 0.6 → 0.7
        );
    } else if (temperature >= 3700.0f) {
        // K-type: Orange (3,700K - 5,200K)
        float t = (temperature - 3700.0f) / 1500.0f;
        return DirectX::XMFLOAT3(
            1.0f,
            0.6f + t * 0.25f,  // 0.6 → 0.85
            0.3f + t * 0.3f    // 0.3 → 0.6
        );
    } else {
        // M-type: Red dwarf/supergiant (2,400K - 3,700K)
        float t = (temperature - 2400.0f) / 1300.0f;
        t = (std::max)(0.0f, (std::min)(1.0f, t));
        return DirectX::XMFLOAT3(
            1.0f,
            0.4f + t * 0.2f,   // 0.4 → 0.6
            0.2f + t * 0.1f    // 0.2 → 0.3
        );
    }
}

std::vector<int> Application::GetSelectedLightIndices() {
    std::vector<int> indices;

    switch (m_lightSelection) {
    case LightSelection::All:
        for (int i = 0; i < (int)m_lights.size(); i++) {
            indices.push_back(i);
        }
        break;

    case LightSelection::InnerRing:
        for (int i = 0; i < (int)m_lights.size(); i++) {
            float dist = sqrt(m_lights[i].position.x * m_lights[i].position.x +
                            m_lights[i].position.y * m_lights[i].position.y +
                            m_lights[i].position.z * m_lights[i].position.z);
            if (dist < m_radialThreshold) {
                indices.push_back(i);
            }
        }
        break;

    case LightSelection::OuterRing:
        for (int i = 0; i < (int)m_lights.size(); i++) {
            float dist = sqrt(m_lights[i].position.x * m_lights[i].position.x +
                            m_lights[i].position.y * m_lights[i].position.y +
                            m_lights[i].position.z * m_lights[i].position.z);
            if (dist >= m_radialThreshold) {
                indices.push_back(i);
            }
        }
        break;

    case LightSelection::TopHalf:
        for (int i = 0; i < (int)m_lights.size(); i++) {
            if (m_lights[i].position.y > 0.0f) {
                indices.push_back(i);
            }
        }
        break;

    case LightSelection::BottomHalf:
        for (int i = 0; i < (int)m_lights.size(); i++) {
            if (m_lights[i].position.y <= 0.0f) {
                indices.push_back(i);
            }
        }
        break;

    case LightSelection::EvenIndices:
        for (int i = 0; i < (int)m_lights.size(); i += 2) {
            indices.push_back(i);
        }
        break;

    case LightSelection::OddIndices:
        for (int i = 1; i < (int)m_lights.size(); i += 2) {
            indices.push_back(i);
        }
        break;

    case LightSelection::CustomRange:
        for (int i = m_customRangeStart; i <= m_customRangeEnd && i < (int)m_lights.size(); i++) {
            indices.push_back(i);
        }
        break;
    }

    return indices;
}

void Application::ApplyGradient(GradientType type, DirectX::XMFLOAT3 startColor, DirectX::XMFLOAT3 endColor) {
    auto indices = GetSelectedLightIndices();

    // Calculate gradient parameter for each light
    std::vector<float> gradientParams;
    float minParam = FLT_MAX, maxParam = -FLT_MAX;

    for (int idx : indices) {
        float param = 0.0f;

        switch (type) {
        case GradientType::Radial:
            // Distance from origin
            param = sqrt(m_lights[idx].position.x * m_lights[idx].position.x +
                        m_lights[idx].position.y * m_lights[idx].position.y +
                        m_lights[idx].position.z * m_lights[idx].position.z);
            break;

        case GradientType::LinearX:
            param = m_lights[idx].position.x;
            break;

        case GradientType::LinearY:
            param = m_lights[idx].position.y;
            break;

        case GradientType::LinearZ:
            param = m_lights[idx].position.z;
            break;

        case GradientType::Circular:
            // Angle around Y-axis
            param = atan2(m_lights[idx].position.z, m_lights[idx].position.x);
            param = (param + 3.14159f) / (2.0f * 3.14159f);  // Normalize to 0-1
            break;
        }

        gradientParams.push_back(param);
        minParam = (std::min)(minParam, param);
        maxParam = (std::max)(maxParam, param);
    }

    // Apply gradient
    for (size_t i = 0; i < indices.size(); i++) {
        int idx = indices[i];

        // Normalize parameter to 0-1 range
        float t = (maxParam - minParam < 0.001f) ? 0.5f : (gradientParams[i] - minParam) / (maxParam - minParam);

        // Lerp between start and end color
        m_lights[idx].color.x = startColor.x * (1.0f - t) + endColor.x * t;
        m_lights[idx].color.y = startColor.y * (1.0f - t) + endColor.y * t;
        m_lights[idx].color.z = startColor.z * (1.0f - t) + endColor.z * t;
    }
}

void Application::ApplyGlobalHueShift(float degrees) {
    auto indices = GetSelectedLightIndices();

    for (int idx : indices) {
        // Convert RGB → HSV
        DirectX::XMFLOAT3 hsv = RGBtoHSV(m_lights[idx].color);

        // Shift hue
        hsv.x += degrees;
        while (hsv.x < 0.0f) hsv.x += 360.0f;
        while (hsv.x >= 360.0f) hsv.x -= 360.0f;

        // Convert back to RGB
        m_lights[idx].color = HSVtoRGB(hsv);
    }
}

void Application::ApplyGlobalSaturationAdjust(float multiplier) {
    auto indices = GetSelectedLightIndices();

    for (int idx : indices) {
        DirectX::XMFLOAT3 hsv = RGBtoHSV(m_lights[idx].color);
        hsv.y *= multiplier;
        hsv.y = (std::min)(1.0f, (std::max)(0.0f, hsv.y));
        m_lights[idx].color = HSVtoRGB(hsv);
    }
}

void Application::ApplyGlobalValueAdjust(float multiplier) {
    auto indices = GetSelectedLightIndices();

    for (int idx : indices) {
        DirectX::XMFLOAT3 hsv = RGBtoHSV(m_lights[idx].color);
        hsv.z *= multiplier;
        hsv.z = (std::min)(1.0f, (std::max)(0.0f, hsv.z));
        m_lights[idx].color = HSVtoRGB(hsv);
    }
}

void Application::ApplyTemperatureShift(float amount) {
    auto indices = GetSelectedLightIndices();

    for (int idx : indices) {
        // Temperature shift: interpolate toward warmer (red) or cooler (blue)
        DirectX::XMFLOAT3 warm = DirectX::XMFLOAT3(1.0f, 0.7f, 0.4f);
        DirectX::XMFLOAT3 cool = DirectX::XMFLOAT3(0.4f, 0.7f, 1.0f);

        float t = (amount + 1.0f) / 2.0f;  // Map -1..+1 to 0..1
        DirectX::XMFLOAT3 shift = DirectX::XMFLOAT3(
            cool.x * (1.0f - t) + warm.x * t,
            cool.y * (1.0f - t) + warm.y * t,
            cool.z * (1.0f - t) + warm.z * t
        );

        // Blend current color with shift color
        float blendAmount = abs(amount) * 0.3f;  // 30% max influence
        m_lights[idx].color.x = m_lights[idx].color.x * (1.0f - blendAmount) + shift.x * blendAmount;
        m_lights[idx].color.y = m_lights[idx].color.y * (1.0f - blendAmount) + shift.y * blendAmount;
        m_lights[idx].color.z = m_lights[idx].color.z * (1.0f - blendAmount) + shift.z * blendAmount;
    }
}

void Application::ApplyColorPreset(ColorPreset preset) {
    m_currentColorPreset = preset;
    auto indices = GetSelectedLightIndices();

    switch (preset) {
    case ColorPreset::CoolBlue:
        for (int idx : indices) {
            m_lights[idx].color = BlackbodyColor(10000.0f);  // Cool blue
        }
        break;

    case ColorPreset::White:
        for (int idx : indices) {
            m_lights[idx].color = BlackbodyColor(6500.0f);  // Daylight white
        }
        break;

    case ColorPreset::WarmWhite:
        for (int idx : indices) {
            m_lights[idx].color = BlackbodyColor(4000.0f);  // Warm white
        }
        break;

    case ColorPreset::WarmSunset:
        for (int idx : indices) {
            m_lights[idx].color = BlackbodyColor(2500.0f);  // Orange sunset
        }
        break;

    case ColorPreset::DeepRed:
        for (int idx : indices) {
            m_lights[idx].color = BlackbodyColor(1800.0f);  // Deep red
        }
        break;

    case ColorPreset::Rainbow:
        for (size_t i = 0; i < indices.size(); i++) {
            int idx = indices[i];
            float hue = (i * 360.0f) / indices.size();  // Distribute evenly
            DirectX::XMFLOAT3 hsv(hue, 1.0f, 1.0f);
            m_lights[idx].color = HSVtoRGB(hsv);
        }
        break;

    case ColorPreset::Complementary:
        for (size_t i = 0; i < indices.size(); i++) {
            int idx = indices[i];
            float hue = (i % 2 == 0) ? 30.0f : 210.0f;  // Orange vs Blue
            DirectX::XMFLOAT3 hsv(hue, 1.0f, 1.0f);
            m_lights[idx].color = HSVtoRGB(hsv);
        }
        break;

    case ColorPreset::MonochromeBlue:
        for (int idx : indices) {
            m_lights[idx].color = DirectX::XMFLOAT3(0.2f, 0.4f, 1.0f);
        }
        break;

    case ColorPreset::MonochromeRed:
        for (int idx : indices) {
            m_lights[idx].color = DirectX::XMFLOAT3(1.0f, 0.2f, 0.2f);
        }
        break;

    case ColorPreset::MonochromeGreen:
        for (int idx : indices) {
            m_lights[idx].color = DirectX::XMFLOAT3(0.2f, 1.0f, 0.3f);
        }
        break;

    case ColorPreset::Neon:
        for (size_t i = 0; i < indices.size(); i++) {
            int idx = indices[i];
            float hue = (i * 360.0f) / indices.size();
            DirectX::XMFLOAT3 hsv(hue, 1.0f, 1.0f);  // Fully saturated
            m_lights[idx].color = HSVtoRGB(hsv);
        }
        break;

    case ColorPreset::Pastel:
        for (size_t i = 0; i < indices.size(); i++) {
            int idx = indices[i];
            float hue = (i * 360.0f) / indices.size();
            DirectX::XMFLOAT3 hsv(hue, 0.3f, 1.0f);  // Low saturation
            m_lights[idx].color = HSVtoRGB(hsv);
        }
        break;

    case ColorPreset::StellarNursery:
        // Blue/white for hot young stars
        for (int idx : indices) {
            float temp = 15000.0f + (rand() % 10000);  // 15000-25000K variation
            m_lights[idx].color = BlackbodyColor(temp);
        }
        break;

    case ColorPreset::RedGiant:
        // Red/orange for cool giant stars
        for (int idx : indices) {
            float temp = 2800.0f + (rand() % 1000);  // 2800-3800K variation
            m_lights[idx].color = BlackbodyColor(temp);
        }
        break;

    case ColorPreset::AccretionDisk:
        // Radial gradient: blue (inner) → red (outer)
        ApplyGradient(GradientType::Radial,
                     BlackbodyColor(25000.0f),  // Blue/white
                     BlackbodyColor(2000.0f));   // Red
        break;

    case ColorPreset::BinarySystem:
        // Two-tone: first half blue, second half red
        for (size_t i = 0; i < indices.size(); i++) {
            int idx = indices[i];
            if (i < indices.size() / 2) {
                m_lights[idx].color = BlackbodyColor(30000.0f);  // Blue star
            } else {
                m_lights[idx].color = BlackbodyColor(3000.0f);   // Red star
            }
        }
        break;

    case ColorPreset::DustTorus:
        // Earth tones: brown/orange
        for (int idx : indices) {
            m_lights[idx].color = DirectX::XMFLOAT3(0.6f, 0.4f, 0.2f);
        }
        break;

    case ColorPreset::Custom:
        // No-op, leave colors as-is
        break;
    }

    LOG_INFO("Applied color preset: {} to {} lights", (int)preset, (int)indices.size());
}