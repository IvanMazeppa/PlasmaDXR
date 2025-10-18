#include "ResourceManager.h"
#include "../core/Device.h"
#include "Logger.h"
#include "d3dx12/d3dx12.h"

ResourceManager::~ResourceManager() {
    Shutdown();
}

bool ResourceManager::Initialize(Device* device) {
    m_device = device;

    // Create upload buffer for staging data
    D3D12_HEAP_PROPERTIES uploadHeapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    D3D12_RESOURCE_DESC uploadBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(m_uploadBufferSize);

    HRESULT hr = m_device->GetDevice()->CreateCommittedResource(
        &uploadHeapProps,
        D3D12_HEAP_FLAG_NONE,
        &uploadBufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&m_uploadBuffer));

    if (FAILED(hr)) {
        LOG_ERROR("Failed to create upload buffer");
        return false;
    }

    LOG_INFO("ResourceManager initialized (64MB upload buffer)");
    return true;
}

void ResourceManager::Shutdown() {
    m_buffers.clear();
    m_textures.clear();
    m_descriptorHeaps.clear();
    m_uploadBuffer.Reset();
    LOG_INFO("ResourceManager shut down");
}

ID3D12Resource* ResourceManager::CreateBuffer(const std::string& name, const BufferDesc& desc) {
    // Check if buffer already exists
    auto it = m_buffers.find(name);
    if (it != m_buffers.end()) {
        LOG_WARN("Buffer '{}' already exists, returning existing", name);
        return it->second.Get();
    }

    // Create buffer
    D3D12_HEAP_PROPERTIES heapProps = CD3DX12_HEAP_PROPERTIES(desc.heapType);
    D3D12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(desc.size, desc.flags);

    Microsoft::WRL::ComPtr<ID3D12Resource> buffer;
    HRESULT hr = m_device->GetDevice()->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        desc.initialState,
        nullptr,
        IID_PPV_ARGS(&buffer));

    if (FAILED(hr)) {
        LOG_ERROR("Failed to create buffer '{}' (HRESULT: 0x{:08X})", name, static_cast<uint32_t>(hr));
        return nullptr;
    }

    m_buffers[name] = buffer;
    LOG_INFO("Created buffer '{}' ({} bytes, heap type: {})",
             name, desc.size, desc.heapType == D3D12_HEAP_TYPE_DEFAULT ? "DEFAULT" :
             desc.heapType == D3D12_HEAP_TYPE_UPLOAD ? "UPLOAD" : "READBACK");

    return buffer.Get();
}

ID3D12Resource* ResourceManager::GetBuffer(const std::string& name) {
    auto it = m_buffers.find(name);
    if (it == m_buffers.end()) {
        LOG_WARN("Buffer '{}' not found", name);
        return nullptr;
    }
    return it->second.Get();
}

ID3D12Resource* ResourceManager::CreateTexture(const std::string& name, const TextureDesc& desc) {
    auto it = m_textures.find(name);
    if (it != m_textures.end()) {
        LOG_WARN("Texture '{}' already exists, returning existing", name);
        return it->second.Get();
    }

    // Create texture
    D3D12_HEAP_PROPERTIES heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    D3D12_RESOURCE_DESC textureDesc = CD3DX12_RESOURCE_DESC::Tex2D(
        desc.format,
        desc.width,
        desc.height,
        1, 1, 1, 0,
        desc.flags);

    Microsoft::WRL::ComPtr<ID3D12Resource> texture;
    HRESULT hr = m_device->GetDevice()->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &textureDesc,
        desc.initialState,
        nullptr,
        IID_PPV_ARGS(&texture));

    if (FAILED(hr)) {
        LOG_ERROR("Failed to create texture '{}' (HRESULT: 0x{:08X})", name, static_cast<uint32_t>(hr));
        return nullptr;
    }

    m_textures[name] = texture;
    LOG_INFO("Created texture '{}' ({}x{}, format: {})", name, desc.width, desc.height, desc.format);

    return texture.Get();
}

ID3D12Resource* ResourceManager::GetTexture(const std::string& name) {
    auto it = m_textures.find(name);
    if (it == m_textures.end()) {
        LOG_WARN("Texture '{}' not found", name);
        return nullptr;
    }
    return it->second.Get();
}

bool ResourceManager::CreateDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE type,
                                          uint32_t numDescriptors,
                                          bool shaderVisible) {
    // Check if heap already exists
    if (m_descriptorHeaps.find(type) != m_descriptorHeaps.end()) {
        LOG_WARN("Descriptor heap of type {} already exists", static_cast<int>(type));
        return true;
    }

    D3D12_DESCRIPTOR_HEAP_DESC desc = {};
    desc.NumDescriptors = numDescriptors;
    desc.Type = type;
    desc.Flags = shaderVisible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE
                               : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

    DescriptorHeapInfo info;
    HRESULT hr = m_device->GetDevice()->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&info.heap));
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create descriptor heap type {} (HRESULT: 0x{:08X})",
                  static_cast<int>(type), static_cast<uint32_t>(hr));
        return false;
    }

    info.numDescriptors = numDescriptors;
    info.descriptorSize = m_device->GetDevice()->GetDescriptorHandleIncrementSize(type);
    info.currentIndex = 0;

    m_descriptorHeaps[type] = info;

    const char* typeName = (type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) ? "CBV_SRV_UAV" :
                          (type == D3D12_DESCRIPTOR_HEAP_TYPE_RTV) ? "RTV" :
                          (type == D3D12_DESCRIPTOR_HEAP_TYPE_DSV) ? "DSV" : "SAMPLER";

    LOG_INFO("Created descriptor heap {} ({} descriptors, shader visible: {})",
             typeName, numDescriptors, shaderVisible);

    return true;
}

ID3D12DescriptorHeap* ResourceManager::GetDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE type) {
    auto it = m_descriptorHeaps.find(type);
    if (it == m_descriptorHeaps.end()) {
        LOG_ERROR("Descriptor heap type {} not found", static_cast<int>(type));
        return nullptr;
    }
    return it->second.heap.Get();
}

D3D12_CPU_DESCRIPTOR_HANDLE ResourceManager::AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE type) {
    auto it = m_descriptorHeaps.find(type);
    if (it == m_descriptorHeaps.end()) {
        LOG_ERROR("Descriptor heap type {} not found", static_cast<int>(type));
        return { 0 };
    }

    auto& heapInfo = it->second;

    if (heapInfo.currentIndex >= heapInfo.numDescriptors) {
        LOG_ERROR("Descriptor heap type {} is full! ({}/{})",
                  static_cast<int>(type), heapInfo.currentIndex, heapInfo.numDescriptors);
        return { 0 };
    }

    D3D12_CPU_DESCRIPTOR_HANDLE handle = heapInfo.heap->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += heapInfo.currentIndex * heapInfo.descriptorSize;
    heapInfo.currentIndex++;

    return handle;
}

D3D12_GPU_DESCRIPTOR_HANDLE ResourceManager::GetGPUDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t index) {
    auto it = m_descriptorHeaps.find(type);
    if (it == m_descriptorHeaps.end()) {
        LOG_ERROR("Descriptor heap type {} not found", static_cast<int>(type));
        return { 0 };
    }

    auto& heapInfo = it->second;

    if (index >= heapInfo.numDescriptors) {
        LOG_ERROR("Descriptor index {} out of range (max: {})", index, heapInfo.numDescriptors);
        return { 0 };
    }

    D3D12_GPU_DESCRIPTOR_HANDLE handle = heapInfo.heap->GetGPUDescriptorHandleForHeapStart();
    handle.ptr += index * heapInfo.descriptorSize;

    return handle;
}

D3D12_GPU_DESCRIPTOR_HANDLE ResourceManager::GetGPUHandle(D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle) {
    // Find which heap this CPU handle belongs to
    for (auto& pair : m_descriptorHeaps) {
        auto& heapInfo = pair.second;
        D3D12_CPU_DESCRIPTOR_HANDLE heapStart = heapInfo.heap->GetCPUDescriptorHandleForHeapStart();

        // Check if CPU handle is within this heap's range
        SIZE_T offset = cpuHandle.ptr - heapStart.ptr;
        if (offset < heapInfo.numDescriptors * heapInfo.descriptorSize) {
            // Found the heap - calculate corresponding GPU handle
            D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle = heapInfo.heap->GetGPUDescriptorHandleForHeapStart();
            gpuHandle.ptr += offset;
            return gpuHandle;
        }
    }

    LOG_ERROR("CPU handle does not belong to any known descriptor heap");
    return { 0 };
}

void ResourceManager::UploadBufferData(ID3D12Resource* buffer, const void* data, size_t size) {
    if (!buffer || !data || size == 0) {
        LOG_ERROR("Invalid parameters for buffer upload");
        return;
    }

    // Map upload buffer
    void* mappedData = nullptr;
    D3D12_RANGE readRange = { 0, 0 }; // We won't read from this buffer

    HRESULT hr = m_uploadBuffer->Map(0, &readRange, &mappedData);
    if (FAILED(hr)) {
        LOG_ERROR("Failed to map upload buffer (HRESULT: 0x{:08X})", static_cast<uint32_t>(hr));
        return;
    }

    // Copy data
    memcpy(mappedData, data, size);
    m_uploadBuffer->Unmap(0, nullptr);

    // Copy from upload buffer to target buffer
    // Note: This requires a command list, which we'll need to pass in
    // For now, this is a placeholder - actual implementation needs command list
    LOG_WARN("UploadBufferData needs command list integration - data copied to upload buffer only");
}

ResourceManager::UploadAllocation ResourceManager::AllocateUpload(size_t size, size_t alignment) {
    UploadAllocation result;

    // Align offset
    m_uploadHeapOffset = (m_uploadHeapOffset + alignment - 1) & ~(alignment - 1);

    // Check space
    if (m_uploadHeapOffset + size > m_uploadBufferSize) {
        LOG_ERROR("Upload heap full! Requested {} bytes, {} available",
                  size, m_uploadBufferSize - m_uploadHeapOffset);
        return result;
    }

    // Map upload buffer (persistent mapping)
    if (!m_uploadBufferMapped) {
        D3D12_RANGE readRange = { 0, 0 };
        HRESULT hr = m_uploadBuffer->Map(0, &readRange, &m_uploadBufferMapped);
        if (FAILED(hr)) {
            LOG_ERROR("Failed to map upload buffer");
            return result;
        }
    }

    // Return allocation
    result.resource = m_uploadBuffer.Get();
    result.cpuAddress = static_cast<uint8_t*>(m_uploadBufferMapped) + m_uploadHeapOffset;
    result.offset = m_uploadHeapOffset;

    m_uploadHeapOffset += size;
    return result;
}

void ResourceManager::ResetUploadHeap() {
    m_uploadHeapOffset = 0;
}

void ResourceManager::TransitionResource(ID3D12GraphicsCommandList* cmdList,
                                        ID3D12Resource* resource,
                                        D3D12_RESOURCE_STATES from,
                                        D3D12_RESOURCE_STATES to) {
    if (!cmdList || !resource) {
        LOG_ERROR("Invalid parameters for resource transition");
        return;
    }

    if (from == to) {
        return; // No transition needed
    }

    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(resource, from, to);
    cmdList->ResourceBarrier(1, &barrier);
}