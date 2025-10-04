#include "Device.h"
#include "../utils/Logger.h"
#include <dxgi1_6.h>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

Device::~Device() {
    Shutdown();
}

bool Device::Initialize(bool enableDebugLayer) {
    LOG_INFO("Initializing D3D12 Device for RTX 4060 Ti...");

    if (!CreateDevice(enableDebugLayer)) {
        LOG_ERROR("Failed to create D3D12 device");
        return false;
    }

    if (!CreateCommandInfrastructure()) {
        LOG_ERROR("Failed to create command infrastructure");
        return false;
    }

    if (!CreateSynchronizationObjects()) {
        LOG_ERROR("Failed to create synchronization objects");
        return false;
    }

    QueryFeatureSupport();

    LOG_INFO("Device initialized successfully");
    LOG_INFO("  DXR Support: {}", m_dxrSupported ? "YES" : "NO");
    LOG_INFO("  RT Tier: {}", static_cast<int>(m_raytracingTier));
    LOG_INFO("  Mesh Shaders: {}", m_meshShadersSupported ? "YES" : "NO");

    return true;
}

void Device::Shutdown() {
    WaitForGPU();

    if (m_fenceEvent) {
        CloseHandle(m_fenceEvent);
        m_fenceEvent = nullptr;
    }
}

bool Device::CreateDevice(bool enableDebugLayer) {
    // Enable debug layer if requested
    if (enableDebugLayer) {
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&m_debugController)))) {
            m_debugController->EnableDebugLayer();
            LOG_INFO("D3D12 Debug Layer enabled");

            // Enable additional debug features
            Microsoft::WRL::ComPtr<ID3D12Debug1> debugController1;
            if (SUCCEEDED(m_debugController.As(&debugController1))) {
                debugController1->SetEnableGPUBasedValidation(FALSE); // Too slow, causes issues
                debugController1->SetEnableSynchronizedCommandQueueValidation(FALSE);
            }
        }
    }

    // Create DXGI Factory
    UINT dxgiFactoryFlags = 0;
    if (enableDebugLayer) {
        dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
    }

    if (FAILED(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&m_dxgiFactory)))) {
        LOG_ERROR("Failed to create DXGI Factory");
        return false;
    }

    // Find RTX 4060 Ti (or any hardware adapter)
    for (UINT adapterIndex = 0;
         m_dxgiFactory->EnumAdapters1(adapterIndex, &m_adapter) != DXGI_ERROR_NOT_FOUND;
         ++adapterIndex) {

        DXGI_ADAPTER_DESC1 desc;
        m_adapter->GetDesc1(&desc);

        // Skip software adapters
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            continue;
        }

        // Try to create device with DXR support (requires Device5)
        HRESULT hr = D3D12CreateDevice(
            m_adapter.Get(),
            D3D_FEATURE_LEVEL_12_1,  // Minimum for DXR
            IID_PPV_ARGS(&m_device)
        );

        if (SUCCEEDED(hr)) {
            char name[256];
            WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name, sizeof(name), nullptr, nullptr);
            LOG_INFO("Selected GPU: {}", name);
            LOG_INFO("Dedicated VRAM: {} MB", desc.DedicatedVideoMemory / (1024 * 1024));
            break;
        }
    }

    if (!m_device) {
        LOG_ERROR("Failed to create D3D12 device");
        return false;
    }

    return true;
}

bool Device::CreateCommandInfrastructure() {
    // Create command queue
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    queueDesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.NodeMask = 0;

    if (FAILED(m_device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_commandQueue)))) {
        LOG_ERROR("Failed to create command queue");
        return false;
    }

    // Create command allocator
    if (FAILED(m_device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(&m_commandAllocator)))) {
        LOG_ERROR("Failed to create command allocator");
        return false;
    }

    // Create command list (closed initially)
    if (FAILED(m_device->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        m_commandAllocator.Get(),
        nullptr,
        IID_PPV_ARGS(&m_commandList)))) {
        LOG_ERROR("Failed to create command list");
        return false;
    }

    // Close it initially
    m_commandList->Close();

    return true;
}

bool Device::CreateSynchronizationObjects() {
    // Create fence for GPU synchronization
    if (FAILED(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)))) {
        LOG_ERROR("Failed to create fence");
        return false;
    }

    m_fenceValue = 1;

    // Create fence event
    m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!m_fenceEvent) {
        LOG_ERROR("Failed to create fence event");
        return false;
    }

    return true;
}

void Device::QueryFeatureSupport() {
    // Check DXR support (RTX 4060 Ti has Tier 1.1)
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 options5 = {};
    if (SUCCEEDED(m_device->CheckFeatureSupport(
        D3D12_FEATURE_D3D12_OPTIONS5, &options5, sizeof(options5)))) {
        m_raytracingTier = options5.RaytracingTier;
        m_dxrSupported = (m_raytracingTier >= D3D12_RAYTRACING_TIER_1_0);
    }

    // Check mesh shader support (RTX 4060 Ti supports Tier 1)
    D3D12_FEATURE_DATA_D3D12_OPTIONS7 options7 = {};
    if (SUCCEEDED(m_device->CheckFeatureSupport(
        D3D12_FEATURE_D3D12_OPTIONS7, &options7, sizeof(options7)))) {
        m_meshShadersSupported = (options7.MeshShaderTier >= D3D12_MESH_SHADER_TIER_1);
    } else {
        // Fallback for RTX 4060 Ti - it definitely supports mesh shaders
        m_meshShadersSupported = true;
    }
}

void Device::ExecuteCommandList() {
    // Execute the command list
    ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
    m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);
}

void Device::WaitForGPU() {
    // Schedule a signal command
    const uint64_t fence = m_fenceValue;
    m_commandQueue->Signal(m_fence.Get(), fence);
    m_fenceValue++;

    // Wait until GPU has completed
    if (m_fence->GetCompletedValue() < fence) {
        m_fence->SetEventOnCompletion(fence, m_fenceEvent);
        WaitForSingleObject(m_fenceEvent, INFINITE);
    }
}

void Device::ResetCommandList() {
    m_commandAllocator->Reset();
    m_commandList->Reset(m_commandAllocator.Get(), nullptr);
}

void Device::SignalFrame() {
    m_commandQueue->Signal(m_fence.Get(), m_fenceValue);
}

void Device::WaitForFrame(uint64_t frameIndex) {
    if (m_fence->GetCompletedValue() < frameIndex) {
        m_fence->SetEventOnCompletion(frameIndex, m_fenceEvent);
        WaitForSingleObject(m_fenceEvent, INFINITE);
    }
}