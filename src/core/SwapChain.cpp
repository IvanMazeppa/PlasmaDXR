#include "SwapChain.h"
#include "Device.h"
#include "../utils/Logger.h"

SwapChain::~SwapChain() {
    Shutdown();
}

bool SwapChain::Initialize(Device* device, HWND hwnd, UINT width, UINT height) {
    m_device = device;
    m_width = width;
    m_height = height;

    if (!CreateSwapChain(hwnd, width, height)) {
        return false;
    }

    if (!CreateRenderTargets()) {
        return false;
    }

    LOG_INFO("SwapChain initialized ({}x{})", width, height);
    return true;
}

void SwapChain::Shutdown() {
    for (UINT i = 0; i < BUFFER_COUNT; i++) {
        m_renderTargets[i].Reset();
    }
    m_rtvHeap.Reset();
    m_swapChain.Reset();
}

bool SwapChain::CreateSwapChain(HWND hwnd, UINT width, UINT height) {
    // Describe and create the swap chain
    DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
    swapChainDesc.BufferCount = BUFFER_COUNT;
    swapChainDesc.Width = width;
    swapChainDesc.Height = height;
    swapChainDesc.Format = DXGI_FORMAT_R10G10B10A2_UNORM; // 10-bit color (4x better than 8-bit)
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapChainDesc.SampleDesc.Count = 1;

    Microsoft::WRL::ComPtr<IDXGISwapChain1> swapChain1;
    if (FAILED(m_device->GetDXGIFactory()->CreateSwapChainForHwnd(
        m_device->GetCommandQueue(),
        hwnd,
        &swapChainDesc,
        nullptr,
        nullptr,
        &swapChain1))) {
        LOG_ERROR("Failed to create swap chain");
        return false;
    }

    // Upgrade to IDXGISwapChain3
    if (FAILED(swapChain1.As(&m_swapChain))) {
        LOG_ERROR("Failed to upgrade swap chain to IDXGISwapChain3");
        return false;
    }

    m_currentFrame = m_swapChain->GetCurrentBackBufferIndex();

    // Disable Alt+Enter fullscreen toggle (we want manual control)
    m_device->GetDXGIFactory()->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);

    return true;
}

bool SwapChain::CreateRenderTargets() {
    // Create RTV descriptor heap
    D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
    rtvHeapDesc.NumDescriptors = BUFFER_COUNT;
    rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

    if (FAILED(m_device->GetDevice()->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&m_rtvHeap)))) {
        LOG_ERROR("Failed to create RTV descriptor heap");
        return false;
    }

    m_rtvDescriptorSize = m_device->GetDevice()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    // Create render target views
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = m_rtvHeap->GetCPUDescriptorHandleForHeapStart();

    for (UINT i = 0; i < BUFFER_COUNT; i++) {
        if (FAILED(m_swapChain->GetBuffer(i, IID_PPV_ARGS(&m_renderTargets[i])))) {
            LOG_ERROR("Failed to get swap chain buffer {}", i);
            return false;
        }

        m_device->GetDevice()->CreateRenderTargetView(m_renderTargets[i].Get(), nullptr, rtvHandle);
        rtvHandle.ptr += m_rtvDescriptorSize;
    }

    return true;
}

void SwapChain::Present(UINT syncInterval) {
    m_swapChain->Present(syncInterval, 0);
    m_currentFrame = m_swapChain->GetCurrentBackBufferIndex();
}

ID3D12Resource* SwapChain::GetCurrentBackBuffer() {
    return m_renderTargets[m_currentFrame].Get();
}

D3D12_CPU_DESCRIPTOR_HANDLE SwapChain::GetCurrentRTV() {
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = m_rtvHeap->GetCPUDescriptorHandleForHeapStart();
    rtvHandle.ptr += m_currentFrame * m_rtvDescriptorSize;
    return rtvHandle;
}

bool SwapChain::Resize(UINT width, UINT height) {
    if (width == 0 || height == 0) {
        return false;
    }

    // Wait for GPU to finish
    // (Device would need to expose a WaitForGPU method)

    // Release render targets
    for (UINT i = 0; i < BUFFER_COUNT; i++) {
        m_renderTargets[i].Reset();
    }

    // Resize swap chain
    if (FAILED(m_swapChain->ResizeBuffers(BUFFER_COUNT, width, height,
                                          DXGI_FORMAT_R10G10B10A2_UNORM, 0))) {
        LOG_ERROR("Failed to resize swap chain");
        return false;
    }

    m_width = width;
    m_height = height;
    m_currentFrame = m_swapChain->GetCurrentBackBufferIndex();

    // Recreate render targets
    return CreateRenderTargets();
}