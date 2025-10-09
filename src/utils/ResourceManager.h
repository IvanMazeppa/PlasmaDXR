#pragma once

#include <d3d12.h>
#include <wrl/client.h>
#include <unordered_map>
#include <string>
#include <memory>

// ResourceManager - Centralized resource management
// No more scattered descriptors across a 4,842-line file!

class Device;

class ResourceManager {
public:
    struct BufferDesc {
        size_t size = 0;
        D3D12_HEAP_TYPE heapType = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE;
        D3D12_RESOURCE_STATES initialState = D3D12_RESOURCE_STATE_COMMON;
    };

    struct TextureDesc {
        uint32_t width = 1;
        uint32_t height = 1;
        DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM;
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE;
        D3D12_RESOURCE_STATES initialState = D3D12_RESOURCE_STATE_COMMON;
    };

public:
    ResourceManager() = default;
    ~ResourceManager();

    bool Initialize(Device* device);
    void Shutdown();

    // Buffer creation
    ID3D12Resource* CreateBuffer(const std::string& name, const BufferDesc& desc);
    ID3D12Resource* GetBuffer(const std::string& name);

    // Texture creation
    ID3D12Resource* CreateTexture(const std::string& name, const TextureDesc& desc);
    ID3D12Resource* GetTexture(const std::string& name);

    // Descriptor heap management
    bool CreateDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t numDescriptors, bool shaderVisible);
    ID3D12DescriptorHeap* GetDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE type);
    D3D12_CPU_DESCRIPTOR_HANDLE AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE type);
    D3D12_GPU_DESCRIPTOR_HANDLE GetGPUDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t index);
    D3D12_GPU_DESCRIPTOR_HANDLE GetGPUHandle(D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle);

    // Upload helpers
    void UploadBufferData(ID3D12Resource* buffer, const void* data, size_t size);

    // Resource barriers
    void TransitionResource(
        ID3D12GraphicsCommandList* cmdList,
        ID3D12Resource* resource,
        D3D12_RESOURCE_STATES from,
        D3D12_RESOURCE_STATES to);

private:
    Device* m_device = nullptr;

    // Named resources for easy access
    std::unordered_map<std::string, Microsoft::WRL::ComPtr<ID3D12Resource>> m_buffers;
    std::unordered_map<std::string, Microsoft::WRL::ComPtr<ID3D12Resource>> m_textures;

    // Descriptor heaps
    struct DescriptorHeapInfo {
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> heap;
        uint32_t descriptorSize = 0;
        uint32_t numDescriptors = 0;
        uint32_t currentIndex = 0;
    };

    std::unordered_map<D3D12_DESCRIPTOR_HEAP_TYPE, DescriptorHeapInfo> m_descriptorHeaps;

    // Upload buffer for staging
    Microsoft::WRL::ComPtr<ID3D12Resource> m_uploadBuffer;
    size_t m_uploadBufferSize = 64 * 1024 * 1024; // 64MB upload buffer
};