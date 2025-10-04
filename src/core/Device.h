#pragma once

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <memory>

// Device - Manages D3D12 device creation and command infrastructure
// Focused on RTX 4060 Ti capabilities

class Device {
public:
    Device() = default;
    ~Device();

    // Initialize device with optimal settings for RTX 4060 Ti
    bool Initialize(bool enableDebugLayer = false);
    void Shutdown();

    // Getters
    ID3D12Device5* GetDevice() const { return m_device.Get(); }
    ID3D12CommandQueue* GetCommandQueue() const { return m_commandQueue.Get(); }
    ID3D12CommandAllocator* GetCommandAllocator() const { return m_commandAllocator.Get(); }
    ID3D12GraphicsCommandList4* GetCommandList() const { return m_commandList.Get(); }
    IDXGIFactory4* GetDXGIFactory() const { return m_dxgiFactory.Get(); }

    // Command execution
    void ExecuteCommandList();
    void WaitForGPU();
    void ResetCommandList();

    // Frame synchronization
    void SignalFrame();
    void WaitForFrame(uint64_t frameIndex);
    uint64_t GetCurrentFenceValue() const { return m_fenceValue; }

    // Query capabilities
    bool SupportsDXR() const { return m_dxrSupported; }
    bool SupportsMeshShaders() const { return m_meshShadersSupported; }
    D3D12_RAYTRACING_TIER GetRaytracingTier() const { return m_raytracingTier; }

private:
    bool CreateDevice(bool enableDebugLayer);
    bool CreateCommandInfrastructure();
    bool CreateSynchronizationObjects();
    void QueryFeatureSupport();

private:
    // Core D3D12 objects
    Microsoft::WRL::ComPtr<ID3D12Device5> m_device;
    Microsoft::WRL::ComPtr<IDXGIFactory4> m_dxgiFactory;
    Microsoft::WRL::ComPtr<IDXGIAdapter1> m_adapter;

    // Command infrastructure
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_commandQueue;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> m_commandAllocator;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> m_commandList;

    // Synchronization
    Microsoft::WRL::ComPtr<ID3D12Fence> m_fence;
    uint64_t m_fenceValue = 0;
    HANDLE m_fenceEvent = nullptr;

    // Capabilities (RTX 4060 Ti specific)
    bool m_dxrSupported = false;
    bool m_meshShadersSupported = false;
    D3D12_RAYTRACING_TIER m_raytracingTier = D3D12_RAYTRACING_TIER_NOT_SUPPORTED;

    // Debug layer
    Microsoft::WRL::ComPtr<ID3D12Debug> m_debugController;
};