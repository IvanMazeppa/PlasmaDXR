#include "ParticleSystem.h"
#include "../core/Device.h"
#include "../utils/ResourceManager.h"
#include "../utils/Logger.h"
#include "../utils/d3dx12/d3dx12.h"
#include "../ml/PINNPhysicsSystem.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <d3dcompiler.h>

#pragma comment(lib, "d3dcompiler.lib")

ParticleSystem::~ParticleSystem() {
    delete m_pinnPhysics;
    Shutdown();
}

bool ParticleSystem::Initialize(Device* device, ResourceManager* resources, uint32_t particleCount) {
    m_device = device;
    m_resources = resources;
    m_activeParticleCount = particleCount;  // Initially all particles active
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

    // Sprint 1: Initialize Material System
    InitializeMaterialProperties();  // Set up vibrant material presets
    if (!CreateMaterialPropertiesBuffer()) {
        LOG_ERROR("Failed to create material properties buffer");
        return false;
    }
    LOG_INFO("Material system initialized: 5 material types with distinct properties");

    // Initialize PINN ML Physics System (optional)
    m_pinnPhysics = new PINNPhysicsSystem();
    // Path relative to project root (working directory)
    if (m_pinnPhysics->Initialize("ml/models/pinn_accretion_disk.onnx")) {
        m_pinnPhysics->SetEnabled(false);  // Start disabled (user can enable via 'P' key)
        m_pinnPhysics->SetHybridMode(true);
        m_pinnPhysics->SetHybridThreshold(10.0f);  // 10× R_ISCO
        LOG_INFO("PINN physics available! Press 'P' to toggle PINN physics");
        LOG_INFO("  {}", m_pinnPhysics->GetModelInfo());

        // Allocate CPU buffers for PINN inference
        m_cpuPositions.resize(particleCount);
        m_cpuVelocities.resize(particleCount);
        m_cpuForces.resize(particleCount);

        // Initialize particles on CPU (avoid GPU readback entirely)
        LOG_INFO("[PINN] Initializing particles on CPU (no GPU readback needed)");
        InitializeAccretionDisk_CPU();
        m_particlesOnCPU = true;

        // Upload CPU particles to GPU for rendering
        UploadParticleData(m_cpuPositions, m_cpuVelocities);
        m_device->ExecuteCommandList();
        m_device->WaitForGPU();
        m_device->ResetCommandList();

        LOG_INFO("[PINN] CPU initialization complete - {} particles ready", particleCount);
    } else {
        LOG_INFO("PINN not available (ONNX Runtime not installed or model not found)");
        LOG_INFO("  GPU physics will be used exclusively");
    }

    return true;
}

void ParticleSystem::InitializeAccretionDisk() {
    // Physics shader handles GPU initialization - this is just a placeholder
    LOG_INFO("Particles will be GPU-initialized by physics shader");
}

void ParticleSystem::InitializeAccretionDisk_CPU() {
    // CPU-based initialization for PINN mode (matches GPU shader logic)
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> radiusDist(INNER_STABLE_ORBIT, OUTER_DISK_RADIUS);
    std::uniform_real_distribution<float> angleDist(0.0f, 2.0f * 3.14159265f);
    std::uniform_real_distribution<float> heightDist(-DISK_THICKNESS * 0.5f, DISK_THICKNESS * 0.5f);

    for (uint32_t i = 0; i < m_particleCount; i++) {
        // Cylindrical coordinates
        float radius = radiusDist(rng);
        float angle = angleDist(rng);
        float height = heightDist(rng);

        // Convert to Cartesian
        m_cpuPositions[i].x = radius * cosf(angle);
        m_cpuPositions[i].y = height;
        m_cpuPositions[i].z = radius * sinf(angle);

        // Keplerian orbital velocity: v = sqrt(GM/r)
        float orbitalSpeed = sqrtf(m_gravityStrength / radius) * m_angularMomentumBoost;

        // Tangential velocity in disk plane
        m_cpuVelocities[i].x = -orbitalSpeed * sinf(angle);
        m_cpuVelocities[i].y = 0.0f;  // No initial vertical velocity
        m_cpuVelocities[i].z = orbitalSpeed * cosf(angle);

        // NOTE: albedo and materialType will be initialized in UploadParticleData
        // when particles are copied to GPU (Particle struct includes these fields)
    }

    LOG_INFO("[PINN] CPU-initialized {} particles in accretion disk", m_particleCount);
}

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
    // Apply timescale to deltaTime (allows slowing down/speeding up simulation)
    deltaTime *= m_timeScale;
    m_totalTime = totalTime;

    // Dispatch to PINN or GPU physics
    if (m_usePINN && m_pinnPhysics && m_pinnPhysics->IsAvailable()) {
        UpdatePhysics_PINN(deltaTime, totalTime);
    } else {
        UpdatePhysics_GPU(deltaTime, totalTime);
    }
}

void ParticleSystem::UpdatePhysics_GPU(float deltaTime, float totalTime) {
    if (!m_computePSO || !m_computeRootSignature) {
        return;  // Physics not initialized
    }

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
    constants.blackHoleMass = m_blackHoleMass;  // NOW RUNTIME ADJUSTABLE!
    constants.gravityStrength = m_gravityStrength;
    constants.blackHolePosition = m_blackHolePosition;
    constants.turbulenceStrength = m_turbulenceStrength;
    constants.diskAxis = m_diskAxis;
    constants.dampingFactor = m_dampingFactor;
    constants.innerRadius = INNER_STABLE_ORBIT;
    constants.outerRadius = OUTER_DISK_RADIUS;
    constants.diskThickness = DISK_THICKNESS;
    constants.viscosity = m_alphaViscosity;  // NOW USES m_alphaViscosity (Shakura-Sunyaev α)
    constants.angularMomentumBoost = m_angularMomentumBoost;
    constants.constraintShape = m_constraintShape;
    constants.constraintRadius = 50.0f;
    constants.constraintThickness = 5.0f;
    constants.particleCount = static_cast<float>(m_activeParticleCount);  // Use active count for physics

    cmdList->SetComputeRoot32BitConstants(0, sizeof(constants) / 4, &constants, 0);
    cmdList->SetComputeRootUnorderedAccessView(1, m_particleBuffer->GetGPUVirtualAddress());

    // Dispatch compute shader
    uint32_t threadGroupCount = (m_activeParticleCount + 63) / 64;  // Use active count for dispatch
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

// ===================================================================
// PINN Physics Implementation
// ===================================================================

void ParticleSystem::UpdatePhysics_PINN(float deltaTime, float totalTime) {
    if (!m_pinnPhysics || !m_pinnPhysics->IsEnabled()) {
        return;
    }

    if (!m_particlesOnCPU) {
        LOG_ERROR("[PINN] Particles not on CPU - cannot use PINN!");
        return;
    }

    // Step 1: Predict forces using PINN (works directly on CPU particles)
    bool success = m_pinnPhysics->PredictForcesBatch(
        m_cpuPositions.data(),
        m_cpuVelocities.data(),
        m_cpuForces.data(),
        m_activeParticleCount,
        totalTime
    );

    if (!success) {
        LOG_ERROR("[PINN] Force prediction failed");
        return;
    }

    // Step 2: Integrate forces (Velocity Verlet) on CPU particles
    IntegrateForces(m_cpuForces, deltaTime);

    // Step 3: Upload updated CPU particles to GPU for rendering
    UploadParticleData(m_cpuPositions, m_cpuVelocities);

    // Log performance metrics every 60 frames
    static int s_pinnUpdateCount = 0;
    s_pinnUpdateCount++;
    if (s_pinnUpdateCount % 60 == 0) {
        auto metrics = m_pinnPhysics->GetPerformanceMetrics();
        LOG_INFO("[PINN] Update {} - Inference: {:.2f}ms, {} particles, CPU-only: {}",
                 s_pinnUpdateCount, metrics.inferenceTimeMs, metrics.particlesProcessed,
                 m_particlesOnCPU ? "YES" : "NO");
    }
}

// ReadbackParticleData() removed - particles are initialized on CPU when PINN is available
// This eliminates GPU crashes caused by copying particle buffer while RT acceleration structures reference it

void ParticleSystem::UploadParticleData(
    const std::vector<DirectX::XMFLOAT3>& positions,
    const std::vector<DirectX::XMFLOAT3>& velocities) {

    // Create upload buffer
    size_t bufferSize = m_particleCount * sizeof(Particle);
    Microsoft::WRL::ComPtr<ID3D12Resource> uploadBuffer;

    CD3DX12_HEAP_PROPERTIES uploadProps(D3D12_HEAP_TYPE_UPLOAD);
    CD3DX12_RESOURCE_DESC uploadDesc = CD3DX12_RESOURCE_DESC::Buffer(bufferSize);

    HRESULT hr = m_device->GetDevice()->CreateCommittedResource(
        &uploadProps, D3D12_HEAP_FLAG_NONE, &uploadDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(&uploadBuffer)
    );

    if (FAILED(hr)) {
        LOG_ERROR("[PINN] Failed to create upload buffer");
        return;
    }

    // Map and copy data
    void* uploadData = nullptr;
    hr = uploadBuffer->Map(0, nullptr, &uploadData);
    if (FAILED(hr)) {
        LOG_ERROR("[PINN] Failed to map upload buffer");
        return;
    }

    Particle* particles = static_cast<Particle*>(uploadData);
    for (uint32_t i = 0; i < m_activeParticleCount; i++) {
        particles[i].position = positions[i];
        particles[i].velocity = velocities[i];
        // Keep temperature and density from GPU (PINN doesn't modify these)

        // Sprint 1: Initialize new material system fields
        // Default to PLASMA material with warm orange albedo (backward compatible)
        particles[i].albedo = DirectX::XMFLOAT3(1.0f, 0.4f, 0.1f);  // Hot plasma orange/red
        particles[i].materialType = static_cast<uint32_t>(ParticleMaterialType::PLASMA);  // Type 0
    }

    uploadBuffer->Unmap(0, nullptr);

    // Copy to GPU
    auto cmdList = m_device->GetCommandList();

    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = m_particleBuffer.Get();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmdList->ResourceBarrier(1, &barrier);

    cmdList->CopyResource(m_particleBuffer.Get(), uploadBuffer.Get());

    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    cmdList->ResourceBarrier(1, &barrier);

    // Don't execute - let main loop handle it
}

void ParticleSystem::IntegrateForces(const std::vector<DirectX::XMFLOAT3>& forces, float deltaTime) {
    // Velocity Verlet integration
    for (uint32_t i = 0; i < m_activeParticleCount; i++) {
        // Update velocity: v' = v + a * dt
        float ax = forces[i].x;
        float ay = forces[i].y;
        float az = forces[i].z;

        m_cpuVelocities[i].x += ax * deltaTime;
        m_cpuVelocities[i].y += ay * deltaTime;
        m_cpuVelocities[i].z += az * deltaTime;

        // Update position: p' = p + v' * dt
        m_cpuPositions[i].x += m_cpuVelocities[i].x * deltaTime;
        m_cpuPositions[i].y += m_cpuVelocities[i].y * deltaTime;
        m_cpuPositions[i].z += m_cpuVelocities[i].z * deltaTime;
    }
}

// PINN Control Methods

bool ParticleSystem::IsPINNAvailable() const {
    return m_pinnPhysics && m_pinnPhysics->IsAvailable();
}

bool ParticleSystem::IsPINNEnabled() const {
    return m_usePINN;
}

void ParticleSystem::SetPINNEnabled(bool enabled) {
    if (!IsPINNAvailable()) {
        LOG_WARN("[PINN] Cannot enable - PINN not available");
        return;
    }

    if (!m_particlesOnCPU) {
        LOG_WARN("[PINN] Particles not initialized on CPU - cannot use PINN");
        return;
    }

    m_usePINN = enabled;
    if (m_pinnPhysics) {
        m_pinnPhysics->SetEnabled(enabled);
    }

    LOG_INFO("[PINN] Physics {} (particles already on CPU)", enabled ? "ENABLED" : "DISABLED");
}

void ParticleSystem::TogglePINNPhysics() {
    SetPINNEnabled(!m_usePINN);
}

bool ParticleSystem::IsPINNHybridMode() const {
    return m_pinnPhysics ? m_pinnPhysics->IsHybridMode() : false;
}

void ParticleSystem::SetPINNHybridMode(bool hybrid) {
    if (m_pinnPhysics) {
        m_pinnPhysics->SetHybridMode(hybrid);
        LOG_INFO("[PINN] Hybrid mode {}", hybrid ? "ENABLED" : "DISABLED");
    }
}

float ParticleSystem::GetPINNHybridThreshold() const {
    return m_pinnPhysics ? m_pinnPhysics->GetHybridThreshold() / 6.0f : 10.0f;  // Return in × R_ISCO
}

void ParticleSystem::SetPINNHybridThreshold(float radiusMultiplier) {
    if (m_pinnPhysics) {
        m_pinnPhysics->SetHybridThreshold(radiusMultiplier);
    }
}

std::string ParticleSystem::GetPINNModelInfo() const {
    return m_pinnPhysics ? m_pinnPhysics->GetModelInfo() : "PINN not available";
}

ParticleSystem::PINNMetrics ParticleSystem::GetPINNMetrics() const {
    PINNMetrics metrics;
    if (m_pinnPhysics) {
        auto pinnMetrics = m_pinnPhysics->GetPerformanceMetrics();
        metrics.inferenceTimeMs = pinnMetrics.inferenceTimeMs;
        metrics.particlesProcessed = pinnMetrics.particlesProcessed;
        metrics.avgBatchTimeMs = pinnMetrics.avgBatchTimeMs;
    }
    return metrics;
}

// ============================================================================
// Sprint 1: Material System Implementation
// ============================================================================

void ParticleSystem::InitializeMaterialProperties() {
    // Initialize 5 material type presets with VIBRANT, SPECTACULAR colors
    // Goal: Transform muted brown appearance into exciting galactic core visuals

    // Material 0: PLASMA (Legacy - hot accretion disk plasma)
    // Vibrant orange/red, high emission, forward scattering
    // FIX: Increased opacity/scattering to restore volumetric appearance (was too thin at 0.6)
    m_materialProperties.materials[0].albedo = DirectX::XMFLOAT3(1.0f, 0.4f, 0.1f);  // Hot orange/red
    m_materialProperties.materials[0].opacity = 1.0f;                 // Fully volumetric (was 0.6 - too thin)
    m_materialProperties.materials[0].emissionMultiplier = 2.5f;      // Strong self-emission
    m_materialProperties.materials[0].scatteringCoefficient = 2.5f;   // Higher scattering (was 1.5 - too weak)
    m_materialProperties.materials[0].phaseG = 0.7f;                  // Strong forward scatter (was 0.3 - too mild)

    // Material 1: STAR_MAIN_SEQUENCE (Brilliant white/yellow stars)
    // Intense white light with yellow tint, maximum emission
    m_materialProperties.materials[1].albedo = DirectX::XMFLOAT3(1.0f, 0.95f, 0.7f);  // Brilliant white-yellow
    m_materialProperties.materials[1].opacity = 0.9f;                 // Nearly opaque
    m_materialProperties.materials[1].emissionMultiplier = 8.0f;      // VERY high emission (stars glow!)
    m_materialProperties.materials[1].scatteringCoefficient = 0.5f;   // Low scattering (self-luminous)
    m_materialProperties.materials[1].phaseG = 0.0f;                  // Isotropic

    // Material 2: GAS_CLOUD (Wispy nebula clouds - purples, blues, pinks)
    // Colorful, low density, backward scattering creates wispy appearance
    m_materialProperties.materials[2].albedo = DirectX::XMFLOAT3(0.4f, 0.6f, 0.95f);  // Vibrant blue/purple
    m_materialProperties.materials[2].opacity = 0.3f;                 // Very transparent (wispy)
    m_materialProperties.materials[2].emissionMultiplier = 0.8f;      // Gentle glow
    m_materialProperties.materials[2].scatteringCoefficient = 2.5f;   // High scattering (diffuse)
    m_materialProperties.materials[2].phaseG = -0.4f;                 // Backward scattering (rim lighting)

    // Material 3: ROCKY_BODY (Asteroids, rocky particles)
    // Deep grey/brown, minimal emission, absorbs light
    m_materialProperties.materials[3].albedo = DirectX::XMFLOAT3(0.35f, 0.32f, 0.3f);  // Deep grey
    m_materialProperties.materials[3].opacity = 1.0f;                 // Fully opaque
    m_materialProperties.materials[3].emissionMultiplier = 0.05f;     // Almost no emission
    m_materialProperties.materials[3].scatteringCoefficient = 0.3f;   // Low scattering (solid)
    m_materialProperties.materials[3].phaseG = 0.2f;                  // Slight forward scatter

    // Material 4: ICY_BODY (Comets, icy moons - bright reflective blues/whites)
    // Brilliant white/blue, highly reflective, backward scattering creates bright rims
    m_materialProperties.materials[4].albedo = DirectX::XMFLOAT3(0.9f, 0.95f, 1.0f);  // Bright icy blue-white
    m_materialProperties.materials[4].opacity = 0.85f;                // Mostly opaque
    m_materialProperties.materials[4].emissionMultiplier = 0.3f;      // Minimal emission
    m_materialProperties.materials[4].scatteringCoefficient = 3.0f;   // Very high scattering (reflective)
    m_materialProperties.materials[4].phaseG = -0.6f;                 // Strong backward scatter (rim glow)

    // Zero out padding to avoid undefined behavior
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 9; j++) {
            m_materialProperties.materials[i].padding[j] = 0.0f;
        }
    }

    LOG_INFO("[Material System] Initialized 5 vibrant material presets:");
    LOG_INFO("  0: PLASMA          - Hot orange/red, emission 2.5×");
    LOG_INFO("  1: STAR            - Brilliant white-yellow, emission 8.0×");
    LOG_INFO("  2: GAS_CLOUD       - Wispy blue/purple, backward scatter");
    LOG_INFO("  3: ROCKY_BODY      - Deep grey, minimal emission");
    LOG_INFO("  4: ICY_BODY        - Bright blue-white, reflective");
}

bool ParticleSystem::CreateMaterialPropertiesBuffer() {
    // Create upload buffer for material properties (320 bytes)
    size_t bufferSize = sizeof(MaterialPropertiesConstants);

    CD3DX12_HEAP_PROPERTIES uploadProps(D3D12_HEAP_TYPE_UPLOAD);
    CD3DX12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(bufferSize);

    HRESULT hr = m_device->GetDevice()->CreateCommittedResource(
        &uploadProps,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,  // Upload heap starts in GENERIC_READ
        nullptr,
        IID_PPV_ARGS(&m_materialPropertiesBuffer)
    );

    if (FAILED(hr)) {
        LOG_ERROR("[Material System] Failed to create material properties buffer: {}", static_cast<int>(hr));
        return false;
    }

    // Map and upload material data
    void* mappedData = nullptr;
    hr = m_materialPropertiesBuffer->Map(0, nullptr, &mappedData);
    if (FAILED(hr)) {
        LOG_ERROR("[Material System] Failed to map material properties buffer: {}", static_cast<int>(hr));
        return false;
    }

    memcpy(mappedData, &m_materialProperties, bufferSize);
    m_materialPropertiesBuffer->Unmap(0, nullptr);

    LOG_INFO("[Material System] Material properties buffer created and uploaded ({} bytes)", bufferSize);
    return true;
}