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

    // Update particle physics
    if (m_particleSystem) {
        m_particleSystem->Update(deltaTime, m_totalTime);
    }
}

void Application::Render() {
    // Reset command list
    m_device->ResetCommandList();
    auto cmdList = m_device->GetCommandList();

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
        m_rtLighting->ComputeLighting(cmdList,
                                     m_particleSystem->GetParticleBuffer(),
                                     m_config.particleCount);

        rtLightingBuffer = m_rtLighting->GetLightingBuffer();

        // Log every 60 frames
        if ((m_frameCount % 60) == 0) {
            LOG_INFO("RT Lighting computed (frame {})", m_frameCount);
        }
    }

    // Render particles
    if (m_particleRenderer && m_particleSystem) {
        ParticleRenderer::RenderConstants renderConstants = {};
        DirectX::XMStoreFloat4x4(&renderConstants.viewProj, DirectX::XMMatrixIdentity());
        renderConstants.cameraPos = DirectX::XMFLOAT3(0, 100, 200);
        renderConstants.cameraUp = DirectX::XMFLOAT3(0, 1, 0);
        renderConstants.time = m_totalTime;
        renderConstants.particleSize = 1.0f;

        m_particleRenderer->Render(cmdList,
                                  m_particleSystem->GetParticleBuffer(),
                                  rtLightingBuffer,
                                  renderConstants);
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

    case WM_MOUSEMOVE:
        if (app) {
            int x = LOWORD(lParam);
            int y = HIWORD(lParam);
            static int lastX = x, lastY = y;
            app->OnMouseMove(x - lastX, y - lastY);
            lastX = x;
            lastY = y;
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
    }
}

void Application::OnMouseMove(int dx, int dy) {
    // Camera control could go here
}

void Application::UpdateFrameStats() {
    static float elapsedTime = 0.0f;
    static uint32_t frameCounter = 0;

    elapsedTime += m_deltaTime;
    frameCounter++;

    if (elapsedTime >= 1.0f) {
        m_fps = static_cast<float>(frameCounter) / elapsedTime;

        // Update window title
        wchar_t title[256];
        swprintf_s(title, L"PlasmaDX-Clean - FPS: %.1f - %s",
                  m_fps, m_particleRenderer ?
                  (m_particleRenderer->GetActivePath() == ParticleRenderer::RenderPath::MeshShaders ?
                   L"Mesh Shaders" : L"Compute Fallback") : L"No Renderer");
        SetWindowText(m_hwnd, title);

        elapsedTime = 0.0f;
        frameCounter = 0;
    }
}