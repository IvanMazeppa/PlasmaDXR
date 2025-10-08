#include "ParticleSystem.h"
#include "../core/Device.h"
#include "../utils/ResourceManager.h"
#include "../utils/Logger.h"
#include "../utils/d3dx12/d3dx12.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <d3dcompiler.h>

#pragma comment(lib, "d3dcompiler.lib")

ParticleSystem::~ParticleSystem() {
    Shutdown();
}

bool ParticleSystem::Initialize(Device* device, ResourceManager* resources, uint32_t particleCount) {
    m_device = device;
    m_resources = resources;
    m_particleCount = particleCount;

    LOG_INFO("Initializing ParticleSystem with {} particles...", particleCount);

    // Create particle buffer like working PlasmaDX project
    // Key: Initial state is UAV, NO CPU initialization, physics shader initializes on first frame!
    size_t bufferSize = particleCount * sizeof(Particle);

    CD3DX12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE_DEFAULT);
    CD3DX12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(
        bufferSize,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
    );

    HRESULT hr = m_device->GetDevice()->CreateCommittedResource(
        &heapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,  // CRITICAL: Start in UAV, not COMMON!
        nullptr, IID_PPV_ARGS(&m_particleBuffer));

    if (FAILED(hr)) {
        LOG_ERROR("Failed to create particle buffer: {}", static_cast<int>(hr));
        return false;
    }

    LOG_INFO("Particle buffer created ({} bytes, state: UAV)", bufferSize);
    LOG_INFO("Physics shader will GPU-initialize particles when totalTime < 0.01");

    // NO CPU initialization, NO upload buffer, NO copy command!
    // Physics shader will initialize on first frame (totalTime < 0.01)

    LOG_INFO("ParticleSystem initialized successfully");
    LOG_INFO("  Particles: {}", particleCount);
    LOG_INFO("  Inner radius: {} (Schwarzschild radii)", INNER_STABLE_ORBIT);
    LOG_INFO("  Outer radius: {}", OUTER_DISK_RADIUS);
    LOG_INFO("  Disk thickness: {}", DISK_THICKNESS);

    // Create physics compute pipeline
    if (!CreateComputePipeline()) {
        LOG_ERROR("Failed to create particle physics pipeline");
        return false;
    }

    return true;
}

void ParticleSystem::InitializeAccretionDisk() {
    // Physics shader handles GPU initialization - this is just a placeholder
    LOG_INFO("Particles will be GPU-initialized by physics shader");
}

// Removed all CPU initialization code - physics shader handles GPU init

bool ParticleSystem::CreateComputePipeline() {
    HRESULT hr;

    // Load physics compute shader - try multiple paths for different working directories
    std::vector<std::string> shaderPaths = {
        "shaders/particles/particle_physics.dxil",           // From project root
        "../shaders/particles/particle_physics.dxil",        // From build/
        "../../shaders/particles/particle_physics.dxil"      // From build/Debug/
    };

    std::ifstream shaderFile;
    std::string foundPath;
    for (const auto& path : shaderPaths) {
        shaderFile.open(path, std::ios::binary);
        if (shaderFile) {
            foundPath = path;
            LOG_INFO("Found particle_physics.dxil at: {}", path);
            break;
        }
        shaderFile.clear();
    }

    if (!shaderFile) {
        LOG_ERROR("Failed to open particle_physics.dxil - tried {} paths", shaderPaths.size());
        for (const auto& path : shaderPaths) {
            LOG_ERROR("  - {}", path);
        }
        return false;
    }

    std::vector<char> shaderData((std::istreambuf_iterator<char>(shaderFile)), std::istreambuf_iterator<char>());
    Microsoft::WRL::ComPtr<ID3DBlob> computeShader;
    hr = D3DCreateBlob(shaderData.size(), &computeShader);
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create blob for physics shader");
        return false;
    }
    memcpy(computeShader->GetBufferPointer(), shaderData.data(), shaderData.size());

    // Create root signature
    // b0: ParticleConstants
    // u0: RWStructuredBuffer<Particle> particles
    CD3DX12_ROOT_PARAMETER1 rootParams[2];
    rootParams[0].InitAsConstants(32, 0);  // b0: ParticleConstants (multiple of 4 DWORDs)
    rootParams[1].InitAsUnorderedAccessView(0);  // u0: particles UAV

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSigDesc;
    rootSigDesc.Init_1_1(2, rootParams, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);

    Microsoft::WRL::ComPtr<ID3DBlob> signature, error;
    hr = D3DX12SerializeVersionedRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1_1, &signature, &error);
    if (FAILED(hr)) {
        if (error) {
            LOG_ERROR("Physics root signature serialization failed: {}", (char*)error->GetBufferPointer());
        }
        return false;
    }

    hr = m_device->GetDevice()->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(),
                                                     IID_PPV_ARGS(&m_computeRootSignature));
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create physics root signature");
        return false;
    }

    // Create compute PSO
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = m_computeRootSignature.Get();
    psoDesc.CS = CD3DX12_SHADER_BYTECODE(computeShader.Get());

    hr = m_device->GetDevice()->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_computePSO));
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create physics compute PSO");
        return false;
    }

    LOG_INFO("Particle physics pipeline created successfully");
    return true;
}

void ParticleSystem::Update(float deltaTime, float totalTime) {
    if (!m_computePSO || !m_computeRootSignature) {
        return;  // Physics not initialized
    }

    m_totalTime = totalTime;

    // CRITICAL DEBUG: Log first few frames to verify GPU initialization
    static int s_debugFrameCount = 0;
    s_debugFrameCount++;
    if (s_debugFrameCount <= 5) {
        LOG_INFO("=== PHYSICS UPDATE {} ===", s_debugFrameCount);
        LOG_INFO("  deltaTime: {}", deltaTime);
        LOG_INFO("  totalTime: {}", totalTime);
        LOG_INFO("  Will GPU-init: {}", (totalTime < 0.01f) ? "YES" : "NO");
        LOG_INFO("  innerRadius: {}, outerRadius: {}", INNER_STABLE_ORBIT, OUTER_DISK_RADIUS);
    }

    auto cmdList = m_device->GetCommandList();

    // Set compute pipeline
    cmdList->SetPipelineState(m_computePSO.Get());
    cmdList->SetComputeRootSignature(m_computeRootSignature.Get());

    // Setup physics constants
    struct ParticleConstants {
        float deltaTime;
        float totalTime;  // GPU initializes when < 0.01, then runs physics
        float blackHoleMass;
        float gravityStrength;
        DirectX::XMFLOAT3 blackHolePosition;
        float turbulenceStrength;
        DirectX::XMFLOAT3 diskAxis;
        float dampingFactor;
        float innerRadius;
        float outerRadius;
        float diskThickness;
        float viscosity;
        float angularMomentumBoost;
        uint32_t constraintShape;
        float constraintRadius;
        float constraintThickness;
        float particleCount;
    } constants = {};

    constants.deltaTime = deltaTime;
    constants.totalTime = totalTime;  // FIXED: Pass real totalTime so shader can GPU-init when < 0.01
    constants.blackHoleMass = BLACK_HOLE_MASS;
    constants.gravityStrength = m_gravityStrength;
    constants.blackHolePosition = m_blackHolePosition;
    constants.turbulenceStrength = m_turbulenceStrength;
    constants.diskAxis = m_diskAxis;
    constants.dampingFactor = m_dampingFactor;
    constants.innerRadius = INNER_STABLE_ORBIT;
    constants.outerRadius = OUTER_DISK_RADIUS;
    constants.diskThickness = DISK_THICKNESS;
    constants.viscosity = m_viscosity;
    constants.angularMomentumBoost = m_angularMomentumBoost;
    constants.constraintShape = m_constraintShape;
    constants.constraintRadius = 50.0f;
    constants.constraintThickness = 5.0f;
    constants.particleCount = static_cast<float>(m_particleCount);

    cmdList->SetComputeRoot32BitConstants(0, sizeof(constants) / 4, &constants, 0);
    cmdList->SetComputeRootUnorderedAccessView(1, m_particleBuffer->GetGPUVirtualAddress());

    // Dispatch compute shader
    uint32_t threadGroupCount = (m_particleCount + 63) / 64;
    cmdList->Dispatch(threadGroupCount, 1, 1);

    // Log every 60 frames
    static int s_updateCount = 0;
    s_updateCount++;
    if (s_updateCount % 60 == 0) {
        LOG_INFO("Physics update {} (totalTime={}, dispatched {} thread groups)",
                 s_updateCount, totalTime, threadGroupCount);
    }

    // Barrier: UAV -> SRV for rendering
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = m_particleBuffer.Get();
    cmdList->ResourceBarrier(1, &barrier);

    // DON'T close/execute here - let the main render loop handle it!
    // The physics compute will execute as part of the frame's command list
}

void ParticleSystem::Shutdown() {
    m_particleBuffer.Reset();
    m_computeRootSignature.Reset();
    m_computePSO.Reset();

    LOG_INFO("ParticleSystem shut down");
}

void ParticleSystem::DebugReadbackParticles(int count) {
    if (!m_particleBuffer || !m_device) {
        LOG_ERROR("Cannot readback - particle buffer or device not initialized");
        return;
    }

    count = (std::min)(count, (int)m_particleCount);
    LOG_INFO("=== GPU Particle Readback ({} particles) ===", count);

    // Create readback buffer
    Microsoft::WRL::ComPtr<ID3D12Resource> readbackBuffer;
    D3D12_HEAP_PROPERTIES readbackHeapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
    D3D12_RESOURCE_DESC readbackDesc = CD3DX12_RESOURCE_DESC::Buffer(count * sizeof(Particle));

    HRESULT hr = m_device->GetDevice()->CreateCommittedResource(
        &readbackHeapProps, D3D12_HEAP_FLAG_NONE, &readbackDesc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readbackBuffer));

    if (FAILED(hr)) {
        LOG_ERROR("Failed to create readback buffer, HRESULT={}", static_cast<int>(hr));
        return;
    }

    LOG_INFO("Readback buffer created successfully");

    // Reset command list to ensure clean state
    m_device->ResetCommandList();
    auto cmdList = m_device->GetCommandList();

    // Transition particle buffer to copy source
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = m_particleBuffer.Get();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmdList->ResourceBarrier(1, &barrier);

    // Copy particles
    cmdList->CopyBufferRegion(readbackBuffer.Get(), 0, m_particleBuffer.Get(), 0, count * sizeof(Particle));

    // Transition back to UAV
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    cmdList->ResourceBarrier(1, &barrier);

    // Execute and wait
    hr = cmdList->Close();
    if (FAILED(hr)) {
        LOG_ERROR("Failed to close command list, HRESULT={}", static_cast<int>(hr));
        m_device->ResetCommandList();
        return;
    }

    LOG_INFO("Command list closed, executing...");
    m_device->ExecuteCommandList();
    m_device->WaitForGPU();
    LOG_INFO("GPU synchronized, mapping readback buffer...");

    // Map and read
    void* readbackData = nullptr;
    hr = readbackBuffer->Map(0, nullptr, &readbackData);
    if (SUCCEEDED(hr)) {
        LOG_INFO("Map succeeded, reading particles...");
        Particle* particles = static_cast<Particle*>(readbackData);

        for (int i = 0; i < count; i++) {
            LOG_INFO("Particle {}: pos=({}, {}, {}) vel=({}, {}, {}) temp={} dens={}",
                i,
                particles[i].position.x, particles[i].position.y, particles[i].position.z,
                particles[i].velocity.x, particles[i].velocity.y, particles[i].velocity.z,
                particles[i].temperature, particles[i].density);
        }

        readbackBuffer->Unmap(0, nullptr);
    } else {
        LOG_ERROR("Failed to map readback buffer, HRESULT={}", static_cast<int>(hr));
    }

    m_device->ResetCommandList();
    LOG_INFO("===========================================");
}