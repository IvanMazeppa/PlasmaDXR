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

    // Create particle buffer on GPU
    ResourceManager::BufferDesc bufferDesc = {};
    bufferDesc.size = particleCount * sizeof(Particle);
    bufferDesc.heapType = D3D12_HEAP_TYPE_DEFAULT;
    bufferDesc.flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    bufferDesc.initialState = D3D12_RESOURCE_STATE_COMMON;

    m_particleBuffer = m_resources->CreateBuffer("ParticleBuffer", bufferDesc);
    if (!m_particleBuffer) {
        LOG_ERROR("Failed to create particle buffer");
        return false;
    }

    // Create upload buffer for initial data
    bufferDesc.heapType = D3D12_HEAP_TYPE_UPLOAD;
    bufferDesc.flags = D3D12_RESOURCE_FLAG_NONE;
    bufferDesc.initialState = D3D12_RESOURCE_STATE_GENERIC_READ;

    m_particleUploadBuffer = m_resources->CreateBuffer("ParticleUploadBuffer", bufferDesc);
    if (!m_particleUploadBuffer) {
        LOG_ERROR("Failed to create particle upload buffer");
        return false;
    }

    // Initialize particles on CPU
    std::vector<Particle> particles(particleCount);
    InitializeAccretionDisk();

    // Map upload buffer and copy data
    void* mappedData = nullptr;
    D3D12_RANGE readRange = { 0, 0 };
    HRESULT hr = m_particleUploadBuffer.Get()->Map(0, &readRange, &mappedData);
    if (FAILED(hr)) {
        LOG_ERROR("Failed to map particle upload buffer");
        return false;
    }

    // Generate particle data
    std::mt19937 rng(12345); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    Particle* particleData = static_cast<Particle*>(mappedData);

    for (uint32_t i = 0; i < particleCount; i++) {
        float theta = dist(rng) * 2.0f * 3.14159265f;
        float radiusFactor = powf(dist(rng), 0.5f); // Square root for better distribution
        float radius = INNER_STABLE_ORBIT + radiusFactor * (OUTER_DISK_RADIUS - INNER_STABLE_ORBIT);
        float height = (dist(rng) - 0.5f) * DISK_THICKNESS * expf(-radiusFactor * 2.0f); // Thinner at edges

        particleData[i].position.x = radius * cosf(theta);
        particleData[i].position.y = height;
        particleData[i].position.z = radius * sinf(theta);

        // Keplerian orbital velocity
        float v = sqrtf(BLACK_HOLE_MASS / radius) * INITIAL_ANGULAR_MOMENTUM * 0.01f;
        particleData[i].velocity.x = -v * sinf(theta);
        particleData[i].velocity.y = 0.0f;
        particleData[i].velocity.z = v * cosf(theta);

        // Temperature: hotter near black hole (blackbody radiation)
        particleData[i].temperature = 10000.0f / (radius * 0.1f);

        // Density: higher near black hole, exponential falloff
        particleData[i].density = expf(-radiusFactor * 2.0f) * 2.0f;
    }

    m_particleUploadBuffer.Get()->Unmap(0, nullptr);

    // Copy from upload buffer to GPU buffer
    // This needs to be done on the command list
    auto cmdList = m_device->GetCommandList();

    // Transition particle buffer to copy dest
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = m_particleBuffer.Get();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    cmdList->ResourceBarrier(1, &barrier);

    // Copy data
    cmdList->CopyResource(m_particleBuffer.Get(), m_particleUploadBuffer.Get());

    // Transition to UAV for compute access
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    cmdList->ResourceBarrier(1, &barrier);

    // Execute and wait
    cmdList->Close();
    m_device->ExecuteCommandList();
    m_device->WaitForGPU();
    m_device->ResetCommandList();

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
    // NASA-quality distribution calculated in Initialize()
    // This is just a placeholder for future enhancements
    LOG_INFO("Generating NASA-quality accretion disk distribution...");
}

bool ParticleSystem::CreateComputePipeline() {
    HRESULT hr;

    // Load physics compute shader
    std::ifstream shaderFile("shaders/particles/particle_physics.dxil", std::ios::binary);
    if (!shaderFile) {
        LOG_ERROR("Failed to open particle_physics.dxil");
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

    auto cmdList = m_device->GetCommandList();

    // Set compute pipeline
    cmdList->SetPipelineState(m_computePSO.Get());
    cmdList->SetComputeRootSignature(m_computeRootSignature.Get());

    // Setup physics constants
    struct ParticleConstants {
        float deltaTime;
        float totalTime;
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
    constants.totalTime = totalTime;
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

    // Barrier: UAV -> SRV for rendering
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = m_particleBuffer.Get();
    cmdList->ResourceBarrier(1, &barrier);

    // Execute physics compute work immediately
    cmdList->Close();
    m_device->ExecuteCommandList();
    m_device->WaitForGPU();  // Wait for physics to complete before rendering
    // Note: Render() will reset the command list
}

void ParticleSystem::Shutdown() {
    m_particleBuffer.Reset();
    m_particleUploadBuffer.Reset();
    m_computeRootSignature.Reset();
    m_computePSO.Reset();

    LOG_INFO("ParticleSystem shut down");
}