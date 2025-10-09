#include "Application.h"
#include "Device.h"
#include "SwapChain.h"
#include "FeatureDetector.h"
#include "../particles/ParticleSystem.h"
#include "../particles/ParticleRenderer.h"
#include "../lighting/RTLightingSystem_RayQuery.h"
#include "../utils/ResourceManager.h"
#include "../utils/Logger.h"

// Window procedure forward declaration
LRESULT CALLBACK WindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

Application::Application() {
    m_lastFrameTime = std::chrono::high_resolution_clock::now();
}

Application::~Application() {
    Shutdown();
}

bool Application::Initialize(HINSTANCE hInstance, int nCmdShow) {
    LOG_INFO("Initializing Application...");

    // Load configuration (could be from file)
    m_config.particleCount = 100000;
    m_config.enableRT = true;
    m_config.preferMeshShaders = true;
#ifdef _DEBUG
    m_config.enableDebugLayer = true;
#else
    m_config.enableDebugLayer = false;
#endif

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

    // Initialize particle renderer with automatic path selection
    m_particleRenderer = std::make_unique<ParticleRenderer>();
    if (!m_particleRenderer->Initialize(m_device.get(), m_resources.get(),
                                        m_features.get(), m_config.particleCount)) {
        LOG_ERROR("Failed to initialize particle renderer");
        return false;
    }

    LOG_INFO("Render Path: {}", m_particleRenderer->GetActivePathName());

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

    m_isRunning = true;
    LOG_INFO("Application initialized successfully");

    return true;
}

void Application::Shutdown() {
    if (m_device) {
        m_device->WaitForGPU();
    }

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
            // Calculate frame time
            auto currentTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> deltaTimeDuration = currentTime - m_lastFrameTime;
            m_deltaTime = deltaTimeDuration.count();
            m_lastFrameTime = currentTime;

            // Update and render
            Update(m_deltaTime);
            Render();

            // Update stats
            UpdateFrameStats();
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
    // Reset command list
    m_device->ResetCommandList();
    auto cmdList = m_device->GetCommandList();

    // CRITICAL: Run physics update AFTER reset, so commands are recorded properly!
    if (m_physicsEnabled && m_particleSystem) {
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

    // Render particles
    if (m_particleRenderer && m_particleSystem) {
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
            LOG_INFO("============================");
            loggedCamera = true;
        }

        m_particleRenderer->Render(cmdList,
                                  m_particleSystem->GetParticleBuffer(),
                                  rtLightingBuffer,
                                  renderConstants);
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
}

bool Application::CreateAppWindow(HINSTANCE hInstance, int nCmdShow) {
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

LRESULT CALLBACK Application::WindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    Application* app = reinterpret_cast<Application*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));

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
        LOG_INFO("F1: Testing RT Lighting - particles should turn GREEN");
        break;

    case 'S':
        if (m_rtLighting) {
            static uint32_t raysPerParticle = 4;
            raysPerParticle = (raysPerParticle == 2) ? 4 : (raysPerParticle == 4) ? 8 : 2;
            m_rtLighting->SetRaysPerParticle(raysPerParticle);
            LOG_INFO("Rays per particle: {} (quality: {})", raysPerParticle,
                    raysPerParticle == 8 ? "HIGH" : raysPerParticle == 4 ? "MEDIUM" : "LOW");
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

    // Particle size
    case VK_OEM_PLUS: case VK_ADD:
        m_particleSize += 2.0f;
        LOG_INFO("Particle size: {}", m_particleSize);
        break;
    case VK_OEM_MINUS: case VK_SUBTRACT:
        m_particleSize = (std::max)(1.0f, m_particleSize - 2.0f);
        LOG_INFO("Particle size: {}", m_particleSize);
        break;

    // Toggle physics
    case 'P':
        m_physicsEnabled = !m_physicsEnabled;
        LOG_INFO("Physics: {}", m_physicsEnabled ? "ENABLED" : "DISABLED");
        break;

    // Debug: Readback particle data
    case 'D':
        LOG_INFO("=== DEBUG: Reading back first 10 particles from GPU ===");
        if (m_particleSystem) {
            m_particleSystem->DebugReadbackParticles(10);
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

    // Temperature variation test
    case 'T':
        LOG_INFO("=== Temperature Variation Test ===");
        if (m_particleSystem) {
            m_particleSystem->DebugReadbackParticles(20);
        }
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
    case 'Q':
        m_rtQualityMode = (m_rtQualityMode + 1) % 3;
        const char* modes[] = {"Normal", "ReSTIR", "Adaptive"};
        LOG_INFO("RT Quality Mode: {}", modes[m_rtQualityMode]);
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

void Application::UpdateFrameStats() {
    static float elapsedTime = 0.0f;
    static uint32_t frameCounter = 0;

    elapsedTime += m_deltaTime;
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