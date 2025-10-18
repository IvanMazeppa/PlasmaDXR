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
#include "../utils/ResourceManager.h"
#include "../utils/Logger.h"
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
    m_useReSTIR = appConfig.features.enableReSTIR;
    m_restirInitialCandidates = appConfig.features.restirCandidates;
    m_restirTemporalWeight = appConfig.features.restirTemporalWeight;

    // Parse command-line argument overrides (these override config file)
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--gaussian" || arg == "-g") {
            m_config.rendererType = RendererType::Gaussian;
        } else if (arg == "--billboard" || arg == "-b") {
            m_config.rendererType = RendererType::Billboard;
        } else if (arg == "--particles" && i + 1 < argc) {
            m_config.particleCount = std::atoi(argv[++i]);
        } else if (arg == "--no-restir") {
            m_useReSTIR = false;
        } else if (arg == "--restir") {
            m_useReSTIR = true;
        } else if (arg == "--rtxdi") {
            m_lightingSystem = LightingSystem::RTXDI;
            LOG_INFO("Lighting system: RTXDI (NVIDIA RTX Direct Illumination)");
        } else if (arg == "--multi-light") {
            m_lightingSystem = LightingSystem::MultiLight;
            LOG_INFO("Lighting system: Multi-Light (brute force)");
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
            LOG_INFO("  --restir             : Enable ReSTIR");
            LOG_INFO("  --no-restir          : Disable ReSTIR");
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

    // Initialize RTXDI lighting (if --rtxdi flag is active)
    if (m_lightingSystem == LightingSystem::RTXDI) {
        m_rtxdiLightingSystem = std::make_unique<RTXDILightingSystem>();
        if (!m_rtxdiLightingSystem->Initialize(m_device.get(), m_resources.get(), m_width, m_height)) {
            LOG_ERROR("Failed to initialize RTXDI lighting system");
            LOG_ERROR("  Falling back to multi-light system");
            m_rtxdiLightingSystem.reset();
            m_lightingSystem = LightingSystem::MultiLight;  // Fallback
        } else {
            LOG_INFO("RTXDI Lighting System initialized successfully!");
            LOG_INFO("  Light grid: 30x30x30 cells (27,000 total)");
            LOG_INFO("  Ready for 100+ light scaling");
        }
    }

    // Initialize blit pipeline (HDR→SDR conversion for Gaussian renderer)
    if (m_gaussianRenderer) {
        if (!CreateBlitPipeline()) {
            LOG_ERROR("Failed to create blit pipeline");
            return false;
        }
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
    }

    // RT Lighting Pass (if enabled)
    ID3D12Resource* rtLightingBuffer = nullptr;
    if (m_rtLighting && m_particleSystem) {
        // Full DXR 1.1 RayQuery pipeline: AABB → BLAS → TLAS → RT Lighting
        // RT lighting uses particle buffer as UAV (already in correct state from initialization)
        m_rtLighting->ComputeLighting(cmdList,
                                     m_particleSystem->GetParticleBuffer(),
                                     m_config.particleCount);

        rtLightingBuffer = m_rtLighting->GetLightingBuffer();

        // Log every 60 frames
        if ((m_frameCount % 60) == 0) {
            LOG_INFO("RT Lighting computed (frame {})", m_frameCount);
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

        // Build proper view-projection matrix with mouse look support
        float camX = m_cameraDistance * cosf(m_cameraPitch) * sinf(m_cameraAngle);
        float camY = m_cameraDistance * sinf(m_cameraPitch);
        float camZ = m_cameraDistance * cosf(m_cameraPitch) * cosf(m_cameraAngle);
        DirectX::XMVECTOR cameraPos = DirectX::XMVectorSet(camX, camY + m_cameraHeight, camZ, 1.0f);
        DirectX::XMVECTOR lookAt = DirectX::XMVectorSet(0, 0, 0, 1.0f);
        DirectX::XMVECTOR up = DirectX::XMVectorSet(0, 1, 0, 0);
        DirectX::XMMATRIX view = DirectX::XMMatrixLookAtLH(cameraPos, lookAt, up);
        DirectX::XMMATRIX proj = DirectX::XMMatrixPerspectiveFovLH(DirectX::XM_PIDIV4,
                                                                    static_cast<float>(m_width) / static_cast<float>(m_height),
                                                                    0.1f, 10000.0f);
        // DON'T transpose - HLSL uses row-major by default, DirectXMath is row-major
        DirectX::XMStoreFloat4x4(&renderConstants.viewProj, view * proj);

        // Use runtime camera controls
        renderConstants.cameraPos = DirectX::XMFLOAT3(camX, camY + m_cameraHeight, camZ);
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
            gaussianConstants.viewProj = renderConstants.viewProj;

            // Calculate inverse view-projection for ray generation
            DirectX::XMMATRIX viewProjMat = DirectX::XMLoadFloat4x4(&renderConstants.viewProj);
            DirectX::XMVECTOR det;
            DirectX::XMMATRIX invViewProjMat = DirectX::XMMatrixInverse(&det, viewProjMat);
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
            gaussianConstants.screenWidth = renderConstants.screenWidth;
            gaussianConstants.screenHeight = renderConstants.screenHeight;
            gaussianConstants.fovY = DirectX::XM_PIDIV4; // 45 degrees
            gaussianConstants.aspectRatio = static_cast<float>(m_width) / static_cast<float>(m_height);
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
            gaussianConstants.rtLightingStrength = m_enableRTLighting ? m_rtLightingStrength : 0.0f;
            gaussianConstants.useAnisotropicGaussians = m_useAnisotropicGaussians ? 1u : 0u;
            gaussianConstants.anisotropyStrength = m_anisotropyStrength;

            // ReSTIR parameters
            gaussianConstants.useReSTIR = m_useReSTIR ? 1u : 0u;
            gaussianConstants.restirInitialCandidates = m_restirInitialCandidates;
            gaussianConstants.frameIndex = static_cast<uint32_t>(m_frameCount);
            gaussianConstants.restirTemporalWeight = m_restirTemporalWeight;

            // Multi-light system (Phase 3.5)
            gaussianConstants.lightCount = static_cast<uint32_t>(m_lights.size());
            m_gaussianRenderer->UpdateLights(m_lights);

            // PCSS soft shadow system
            gaussianConstants.shadowRaysPerLight = m_shadowRaysPerLight;
            gaussianConstants.enableTemporalFiltering = m_enableTemporalFiltering ? 1u : 0u;
            gaussianConstants.temporalBlend = m_temporalBlend;
            gaussianConstants.padding4 = 0.0f;

            // Debug: Log RT toggle values once
            static bool loggedToggles = false;
            static bool lastReSTIRState = false;
            if (!loggedToggles) {
                LOG_INFO("=== DEBUG: Gaussian Constants ===");
                LOG_INFO("  useShadowRays: {}", gaussianConstants.useShadowRays);
                LOG_INFO("  useInScattering: {}", gaussianConstants.useInScattering);
                LOG_INFO("  usePhaseFunction: {}", gaussianConstants.usePhaseFunction);
                LOG_INFO("  phaseStrength: {}", gaussianConstants.phaseStrength);
                LOG_INFO("  useReSTIR: {}", gaussianConstants.useReSTIR);
                LOG_INFO("  restirInitialCandidates: {}", gaussianConstants.restirInitialCandidates);
                LOG_INFO("================================");
                loggedToggles = true;
                lastReSTIRState = m_useReSTIR;
            }

            // Log when ReSTIR state changes
            if (m_useReSTIR != lastReSTIRState) {
                LOG_INFO("ReSTIR state changed: {} -> {}", lastReSTIRState, m_useReSTIR);
                LOG_INFO("  gaussianConstants.useReSTIR = {}", gaussianConstants.useReSTIR);
                LOG_INFO("  restirInitialCandidates = {}", gaussianConstants.restirInitialCandidates);
                LOG_INFO("  frameIndex = {}", gaussianConstants.frameIndex);
                lastReSTIRState = m_useReSTIR;
            }

            // Render to UAV texture
            m_gaussianRenderer->Render(reinterpret_cast<ID3D12GraphicsCommandList4*>(cmdList),
                                      m_particleSystem->GetParticleBuffer(),
                                      rtLightingBuffer,
                                      m_rtLighting ? m_rtLighting->GetTLAS() : nullptr,
                                      gaussianConstants);

            // HDR→SDR blit pass (replaces CopyTextureRegion)
            D3D12_RESOURCE_BARRIER blitBarriers[2] = {};

            // Transition Gaussian output (HDR) from UAV to SRV for sampling
            blitBarriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            blitBarriers[0].Transition.pResource = m_gaussianRenderer->GetOutputTexture();
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
            cmdList->SetGraphicsRootDescriptorTable(0, m_gaussianRenderer->GetOutputSRV());

            // Draw fullscreen triangle
            cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            cmdList->DrawInstanced(3, 1, 0, 0);

            // Transition Gaussian output back to UAV for next frame
            blitBarriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            blitBarriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            cmdList->ResourceBarrier(1, &blitBarriers[0]);

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
    m_device->ExecuteCommandList();

    // Present
    m_swapChain->Present(0);

    // Wait for frame completion (simple sync for now)
    m_device->WaitForGPU();

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

    // Let ImGui handle input first
    if (ImGui_ImplWin32_WndProcHandler(hwnd, msg, wParam, lParam))
        return true;

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
            app->OnKeyPress(static_cast<UINT8>(wParam));
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

    // Toggle physics
    case 'P':
        m_physicsEnabled = !m_physicsEnabled;
        LOG_INFO("Physics: {}", m_physicsEnabled ? "ENABLED" : "DISABLED");
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

    // F7: Toggle ReSTIR (Ctrl+F7/Shift+F7 adjust temporal weight)
    case VK_F7:
        if (GetAsyncKeyState(VK_CONTROL) & 0x8000) {
            m_restirTemporalWeight = (std::min)(1.0f, m_restirTemporalWeight + 0.1f);
            LOG_INFO("ReSTIR Temporal Weight: {:.1f}", m_restirTemporalWeight);
        } else if (GetAsyncKeyState(VK_SHIFT) & 0x8000) {
            m_restirTemporalWeight = (std::max)(0.0f, m_restirTemporalWeight - 0.1f);
            LOG_INFO("ReSTIR Temporal Weight: {:.1f}", m_restirTemporalWeight);
        } else {
            m_useReSTIR = !m_useReSTIR;
            LOG_INFO("ReSTIR: {} (temporal resampling for {} faster convergence)",
                     m_useReSTIR ? "ON" : "OFF",
                     m_useReSTIR ? "10-60x" : "");
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

    if (m_gaussianRenderer) {
        // Gaussian renderer has ReSTIR reservoirs
        auto reservoirs = m_gaussianRenderer->GetCurrentReservoirs();
        if (reservoirs) {
            DumpBufferToFile(reservoirs, "g_currentReservoirs");
        }

        auto prevReservoirs = m_gaussianRenderer->GetPrevReservoirs();
        if (prevReservoirs) {
            DumpBufferToFile(prevReservoirs, "g_prevReservoirs");
        }
    }

    if (m_rtLighting) {
        auto rtBuffer = m_rtLighting->GetLightingBuffer();
        if (rtBuffer) {
            DumpBufferToFile(rtBuffer, "g_rtLighting");
        }
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
        fprintf(file, "  \"restir_enabled\": %s,\n", m_useReSTIR ? "true" : "false");
        fprintf(file, "  \"particle_count\": %u,\n", m_config.particleCount);
        fprintf(file, "  \"particle_size\": %.2f,\n", m_particleSize);
        fprintf(file, "  \"render_mode\": \"%s\",\n",
                m_config.rendererType == RendererType::Gaussian ? "Gaussian" : "Billboard");
        fprintf(file, "  \"restir_temporal_weight\": %.2f,\n", m_restirTemporalWeight);
        fprintf(file, "  \"restir_initial_candidates\": %u,\n", m_restirInitialCandidates);
        fprintf(file, "  \"use_shadow_rays\": %s,\n", m_useShadowRays ? "true" : "false");
        fprintf(file, "  \"use_in_scattering\": %s,\n", m_useInScattering ? "true" : "false");
        fprintf(file, "  \"use_phase_function\": %s\n", m_usePhaseFunction ? "true" : "false");
        fprintf(file, "}\n");
        fclose(file);

        LOG_INFO("  Wrote metadata: {}", filepath);
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
        if (m_useReSTIR) {
            wchar_t buf[32];
            swprintf_s(buf, L"[F7:ReSTIR:%.1f] ", m_restirTemporalWeight);
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

        ImGui::Checkbox("In-Scattering (F6)", &m_useInScattering);
        if (m_useInScattering) {
            ImGui::SliderFloat("In-Scatter Strength (F9)", &m_inScatterStrength, 0.0f, 10.0f);
        }
        ImGui::Checkbox("ReSTIR (F7)", &m_useReSTIR);
        if (m_useReSTIR) {
            ImGui::SliderFloat("Temporal Weight (Ctrl/Shift+F7)", &m_restirTemporalWeight, 0.0f, 1.0f);
            ImGui::SliderInt("Initial Candidates", reinterpret_cast<int*>(&m_restirInitialCandidates), 8, 32);
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
        }
        ImGui::Checkbox("Anisotropic Gaussians (F11)", &m_useAnisotropicGaussians);
        if (m_useAnisotropicGaussians) {
            ImGui::SliderFloat("Anisotropy Strength (F12)", &m_anisotropyStrength, 0.0f, 3.0f);
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

    // Physics controls
    if (ImGui::CollapsingHeader("Physics", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Physics Enabled (P)", &m_physicsEnabled);
        if (m_particleSystem) {
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

        // Preset configurations
        ImGui::Text("Presets:");
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
    primaryLight.intensity = 10.0f;
    primaryLight.radius = 5.0f;
    m_lights.push_back(primaryLight);

    // Secondary: 4 spiral arms at 50 unit radius (orange 12000K)
    for (int i = 0; i < 4; i++) {
        float angle = (i * 90.0f) * (PI / 180.0f);
        float radius = 50.0f;

        ParticleRenderer_Gaussian::Light armLight;
        armLight.position = XMFLOAT3(cos(angle) * radius, 0.0f, sin(angle) * radius);
        armLight.color = XMFLOAT3(1.0f, 0.8f, 0.6f);  // Orange
        armLight.intensity = 5.0f;
        armLight.radius = 10.0f;
        m_lights.push_back(armLight);
    }

    // Tertiary: 8 mid-disk hot spots at 150 unit radius (yellow-orange 8000K)
    for (int i = 0; i < 8; i++) {
        float angle = (i * 45.0f) * (PI / 180.0f);
        float radius = 150.0f;

        ParticleRenderer_Gaussian::Light hotSpot;
        hotSpot.position = XMFLOAT3(cos(angle) * radius, 0.0f, sin(angle) * radius);
        hotSpot.color = XMFLOAT3(1.0f, 0.7f, 0.4f);  // Yellow-orange
        hotSpot.intensity = 2.0f;
        hotSpot.radius = 15.0f;
        m_lights.push_back(hotSpot);
    }

    LOG_INFO("Initialized multi-light system: {} lights", m_lights.size());
    LOG_INFO("  1 primary (origin, blue-white, 20000K equiv)");
    LOG_INFO("  4 secondary (spiral arms @ 50 units, orange, 12000K equiv)");
    LOG_INFO("  8 tertiary (hot spots @ 150 units, yellow-orange, 8000K equiv)");
}