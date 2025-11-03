#include "ProbeGridSystem.h"
#include "../core/Device.h"
#include "../utils/ResourceManager.h"
#include "../utils/Logger.h"
#include "../utils/d3dx12/d3dx12.h"

#include <fstream>
#include <vector>
#include <d3dcompiler.h>

bool ProbeGridSystem::Initialize(Device* device, ResourceManager* resources) {
    if (m_initialized) {
        LOG_WARN("ProbeGridSystem already initialized");
        return true;
    }

    m_device = device;
    m_resources = resources;

    LOG_INFO("Initializing Hybrid Probe Grid System...");
    LOG_INFO("  Grid size: {}³ ({} probes)", m_gridSize, m_gridSize * m_gridSize * m_gridSize);
    LOG_INFO("  Grid spacing: {:.2f} units", m_gridSpacing);
    LOG_INFO("  Rays per probe: {}", m_raysPerProbe);
    LOG_INFO("  Update interval: {} frames", m_updateInterval);

    // Create probe buffer
    if (!CreateProbeBuffer()) {
        LOG_ERROR("Failed to create probe buffer");
        return false;
    }

    // Create compute pipelines
    if (!CreatePipelines()) {
        LOG_ERROR("Failed to create probe grid pipelines");
        return false;
    }

    m_initialized = true;
    LOG_INFO("Hybrid Probe Grid System initialized successfully!");
    LOG_INFO("  Memory: Probe buffer = {:.2f} MB",
             (m_gridSize * m_gridSize * m_gridSize * sizeof(Probe)) / (1024.0f * 1024.0f));

    return true;
}

bool ProbeGridSystem::CreateProbeBuffer() {
    ID3D12Device* device = m_device->GetDevice();

    uint32_t totalProbes = m_gridSize * m_gridSize * m_gridSize;
    size_t bufferSize = totalProbes * sizeof(Probe);

    LOG_INFO("Creating probe buffer:");
    LOG_INFO("  Total probes: {}", totalProbes);
    LOG_INFO("  Probe size: {} bytes", sizeof(Probe));
    LOG_INFO("  Buffer size: {:.2f} MB", bufferSize / (1024.0f * 1024.0f));

    // Create probe buffer (structured buffer)
    D3D12_RESOURCE_DESC bufferDesc = {};
    bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufferDesc.Width = bufferSize;
    bufferDesc.Height = 1;
    bufferDesc.DepthOrArraySize = 1;
    bufferDesc.MipLevels = 1;
    bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
    bufferDesc.SampleDesc.Count = 1;
    bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    bufferDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    HRESULT hr = device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(&m_probeBuffer));

    if (FAILED(hr)) {
        LOG_ERROR("Failed to create probe buffer! HRESULT: 0x{:08X}", static_cast<uint32_t>(hr));
        return false;
    }

    m_probeBuffer->SetName(L"ProbeGridSystem::ProbeBuffer");

    // Create SRV (for particle shader sampling)
    m_probeBufferSRV = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = DXGI_FORMAT_UNKNOWN;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Buffer.FirstElement = 0;
    srvDesc.Buffer.NumElements = totalProbes;
    srvDesc.Buffer.StructureByteStride = sizeof(Probe);

    device->CreateShaderResourceView(m_probeBuffer.Get(), &srvDesc, m_probeBufferSRV);

    // Store GPU handle for shader binding
    m_probeBufferSRV_GPU = m_resources->GetGPUHandle(m_probeBufferSRV);

    LOG_INFO("  Probe buffer SRV created: GPU handle 0x{:016X}",
             static_cast<uint64_t>(m_probeBufferSRV_GPU.ptr));

    // Create UAV (for probe update shader)
    m_probeBufferUAV = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = DXGI_FORMAT_UNKNOWN;
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uavDesc.Buffer.FirstElement = 0;
    uavDesc.Buffer.NumElements = totalProbes;
    uavDesc.Buffer.StructureByteStride = sizeof(Probe);

    device->CreateUnorderedAccessView(m_probeBuffer.Get(), nullptr, &uavDesc, m_probeBufferUAV);

    m_probeBufferUAV_GPU = m_resources->GetGPUHandle(m_probeBufferUAV);

    LOG_INFO("  Probe buffer UAV created: GPU handle 0x{:016X}",
             static_cast<uint64_t>(m_probeBufferUAV_GPU.ptr));

    // Create constant buffer for probe updates
    size_t cbSize = ((sizeof(ProbeUpdateConstants) + 255) / 256) * 256; // Align to 256 bytes

    D3D12_RESOURCE_DESC cbDesc = {};
    cbDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    cbDesc.Width = cbSize;
    cbDesc.Height = 1;
    cbDesc.DepthOrArraySize = 1;
    cbDesc.MipLevels = 1;
    cbDesc.Format = DXGI_FORMAT_UNKNOWN;
    cbDesc.SampleDesc.Count = 1;
    cbDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    cbDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    D3D12_HEAP_PROPERTIES uploadHeap = {};
    uploadHeap.Type = D3D12_HEAP_TYPE_UPLOAD;

    hr = device->CreateCommittedResource(
        &uploadHeap,
        D3D12_HEAP_FLAG_NONE,
        &cbDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&m_updateConstantBuffer));

    if (FAILED(hr)) {
        LOG_ERROR("Failed to create probe update constant buffer! HRESULT: 0x{:08X}",
                  static_cast<uint32_t>(hr));
        return false;
    }

    m_updateConstantBuffer->SetName(L"ProbeGridSystem::UpdateConstants");

    LOG_INFO("Probe buffer created successfully!");
    return true;
}

bool ProbeGridSystem::CreatePipelines() {
    LOG_INFO("Creating probe grid compute pipelines...");

    auto d3dDevice = m_device->GetDevice();
    HRESULT hr;

    // ===================================================================
    // Probe Update Pipeline
    // ===================================================================

    // Create root signature for probe update shader
    {
        CD3DX12_ROOT_PARAMETER1 rootParams[5];

        // b0: ProbeUpdateConstants (constant buffer)
        rootParams[0].InitAsConstantBufferView(0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE, D3D12_SHADER_VISIBILITY_ALL);

        // t0: Particle buffer (SRV, structured buffer)
        rootParams[1].InitAsShaderResourceView(0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE, D3D12_SHADER_VISIBILITY_ALL);

        // t1: Light buffer (SRV, structured buffer)
        rootParams[2].InitAsShaderResourceView(1, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE, D3D12_SHADER_VISIBILITY_ALL);

        // t2: Particle TLAS (SRV, raytracing acceleration structure)
        rootParams[3].InitAsShaderResourceView(2, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE, D3D12_SHADER_VISIBILITY_ALL);

        // u0: Probe buffer (UAV, structured buffer)
        rootParams[4].InitAsUnorderedAccessView(0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE, D3D12_SHADER_VISIBILITY_ALL);

        CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSigDesc;
        rootSigDesc.Init_1_1(_countof(rootParams), rootParams, 0, nullptr,
                            D3D12_ROOT_SIGNATURE_FLAG_NONE);

        ComPtr<ID3DBlob> signature;
        ComPtr<ID3DBlob> error;
        hr = D3D12SerializeVersionedRootSignature(&rootSigDesc, &signature, &error);

        if (FAILED(hr)) {
            if (error) {
                LOG_ERROR("Failed to serialize probe update root signature: {}",
                          static_cast<char*>(error->GetBufferPointer()));
            }
            return false;
        }

        hr = d3dDevice->CreateRootSignature(0, signature->GetBufferPointer(),
                                            signature->GetBufferSize(),
                                            IID_PPV_ARGS(&m_updateProbeRS));

        if (FAILED(hr)) {
            LOG_ERROR("Failed to create probe update root signature! HRESULT: 0x{:08X}",
                      static_cast<uint32_t>(hr));
            return false;
        }

        m_updateProbeRS->SetName(L"ProbeGridSystem::UpdateProbeRS");
        LOG_INFO("Probe update root signature created");
    }

    // Load and create PSO
    {
        // Load compiled shader
        std::ifstream shaderFile("shaders/probe_grid/update_probes.dxil", std::ios::binary);
        if (!shaderFile) {
            LOG_ERROR("Failed to open update_probes.dxil shader file!");
            return false;
        }

        std::vector<uint8_t> shaderData((std::istreambuf_iterator<char>(shaderFile)),
                                        std::istreambuf_iterator<char>());
        shaderFile.close();

        LOG_INFO("Loaded probe update shader: {} bytes", shaderData.size());

        // Create PSO
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = m_updateProbeRS.Get();
        psoDesc.CS.pShaderBytecode = shaderData.data();
        psoDesc.CS.BytecodeLength = shaderData.size();

        hr = d3dDevice->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_updateProbePSO));

        if (FAILED(hr)) {
            LOG_ERROR("Failed to create probe update PSO! HRESULT: 0x{:08X}",
                      static_cast<uint32_t>(hr));
            return false;
        }

        m_updateProbePSO->SetName(L"ProbeGridSystem::UpdateProbePSO");
        LOG_INFO("Probe update PSO created");
    }

    LOG_INFO("Probe grid pipelines created successfully");
    return true;
}

void ProbeGridSystem::UpdateProbes(
    ID3D12GraphicsCommandList* commandList,
    ID3D12Resource* particleTLAS,
    ID3D12Resource* particleBuffer,
    uint32_t particleCount,
    ID3D12Resource* lightBuffer,
    uint32_t lightCount,
    uint32_t frameIndex) {

    if (!m_initialized || !m_enabled) {
        return;
    }

    // TODO: Implement probe update dispatch
    // This will be implemented after shader creation (Day 2-3)

    m_frameCount++;
}

void ProbeGridSystem::SetGridSize(uint32_t size) {
    if (m_initialized) {
        LOG_WARN("Cannot change grid size after initialization");
        return;
    }

    m_gridSize = size;

    // Recalculate grid spacing
    float worldSize = m_gridMax.x - m_gridMin.x; // Assumes cubic world
    m_gridSpacing = worldSize / static_cast<float>(size);

    LOG_INFO("Grid size set to {}³ (spacing: {:.2f} units)", size, m_gridSpacing);
}

ProbeGridSystem::ProbeGridParams ProbeGridSystem::GetProbeGridParams() const {
    ProbeGridParams params;
    params.gridMin = m_gridMin;
    params.gridSpacing = m_gridSpacing;
    params.gridSize = m_gridSize;
    params.totalProbes = m_gridSize * m_gridSize * m_gridSize;
    params.padding0 = 0;
    params.padding1 = 0;
    return params;
}
