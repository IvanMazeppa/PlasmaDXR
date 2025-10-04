#pragma once

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

class Device;

class SwapChain {
public:
    static constexpr UINT BUFFER_COUNT = 2;

    SwapChain() = default;
    ~SwapChain();

    bool Initialize(Device* device, HWND hwnd, UINT width, UINT height);
    void Shutdown();

    // Present the current frame
    void Present(UINT syncInterval);

    // Get current back buffer resources
    ID3D12Resource* GetCurrentBackBuffer();
    D3D12_CPU_DESCRIPTOR_HANDLE GetCurrentRTV();
    UINT GetCurrentFrameIndex() const { return m_currentFrame; }

    // Resize swap chain
    bool Resize(UINT width, UINT height);

private:
    bool CreateSwapChain(HWND hwnd, UINT width, UINT height);
    bool CreateRenderTargets();

private:
    Device* m_device = nullptr;
    Microsoft::WRL::ComPtr<IDXGISwapChain3> m_swapChain;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_renderTargets[BUFFER_COUNT];
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_rtvHeap;

    UINT m_width = 0;
    UINT m_height = 0;
    UINT m_currentFrame = 0;
    UINT m_rtvDescriptorSize = 0;
};