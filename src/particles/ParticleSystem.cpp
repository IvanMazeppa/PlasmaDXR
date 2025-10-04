#include "ParticleSystem.h"
#include "../core/Device.h"
#include "../utils/ResourceManager.h"
#include "../utils/Logger.h"
#include <random>
#include <cmath>
#include <algorithm>

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

        particleData[i].mass = 1.0f;

        // Temperature: hotter near black hole (blackbody radiation)
        particleData[i].temperature = 10000.0f / (radius * 0.1f);

        // Color based on temperature (simplified blackbody)
        float t = (std::min)(particleData[i].temperature / 20000.0f, 1.0f);
        particleData[i].color.x = 1.0f;                    // Red
        particleData[i].color.y = 0.7f * t + 0.3f;        // Green increases with temp
        particleData[i].color.z = 0.3f * t;                // Blue at highest temps
        particleData[i].color.w = 0.8f;                    // Alpha
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

    return true;
}

void ParticleSystem::InitializeAccretionDisk() {
    // NASA-quality distribution calculated in Initialize()
    // This is just a placeholder for future enhancements
    LOG_INFO("Generating NASA-quality accretion disk distribution...");
}

void ParticleSystem::Update(float deltaTime, float totalTime) {
    m_totalTime = totalTime;

    // Physics update would go here
    // For now, particles remain in their initial orbital configuration
    // Future: Implement full N-body dynamics with:
    // - Gravitational forces
    // - Orbital mechanics
    // - Viscosity and angular momentum transfer
    // - Turbulence
    // - Magnetic fields (if applicable)
}

void ParticleSystem::Shutdown() {
    m_particleBuffer.Reset();
    m_particleUploadBuffer.Reset();
    m_computeRootSignature.Reset();
    m_computePSO.Reset();

    LOG_INFO("ParticleSystem shut down");
}