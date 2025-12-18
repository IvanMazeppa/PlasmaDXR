#include "ParticleSystem.h"
#include "../core/Device.h"
#include "../utils/ResourceManager.h"
#include "../utils/Logger.h"
#include "../utils/d3dx12/d3dx12.h"
#include "../ml/PINNPhysicsSystem.h"
#include "../ml/SIRENVortexField.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <d3dcompiler.h>
#include <execution>  // C++17 parallel algorithms
#include <numeric>    // std::iota for index generation

#pragma comment(lib, "d3dcompiler.lib")

ParticleSystem::~ParticleSystem() {
    delete m_pinnPhysics;
    delete m_sirenVortex;
    Shutdown();
}

bool ParticleSystem::Initialize(Device* device, ResourceManager* resources, uint32_t particleCount) {
    m_device = device;
    m_resources = resources;
    m_activeParticleCount = particleCount;  // Initially all particles active
    m_particleCount = particleCount;

    // Phase 2C: Initialize explosion pool (last EXPLOSION_POOL_SIZE particles reserved)
    if (particleCount > EXPLOSION_POOL_SIZE) {
        m_explosionPoolStart = particleCount - EXPLOSION_POOL_SIZE;
        m_activeParticleCount = m_explosionPoolStart;  // Accretion disk uses first N particles
    } else {
        m_explosionPoolStart = 0;  // Not enough particles for pool
    }
    m_nextExplosionIndex = 0;
    m_explosionPoolUsed = 0;

    LOG_INFO("Initializing ParticleSystem with {} particles...", particleCount);
    LOG_INFO("  Accretion disk: {} particles, Explosion pool: {} particles",
             m_activeParticleCount, particleCount - m_explosionPoolStart);

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
    LOG_INFO("Material system initialized: 9 material types with distinct properties");

    // Phase 2C: Create persistent upload buffer for explosions
    if (m_explosionPoolStart > 0) {
        CD3DX12_HEAP_PROPERTIES uploadProps(D3D12_HEAP_TYPE_UPLOAD);
        CD3DX12_RESOURCE_DESC uploadDesc = CD3DX12_RESOURCE_DESC::Buffer(EXPLOSION_UPLOAD_BUFFER_SIZE);

        hr = m_device->GetDevice()->CreateCommittedResource(
            &uploadProps, D3D12_HEAP_FLAG_NONE, &uploadDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
            IID_PPV_ARGS(&m_explosionUploadBuffer)
        );

        if (FAILED(hr)) {
            LOG_WARN("[Explosion] Failed to create upload buffer - explosions disabled");
            m_explosionPoolStart = 0;  // Disable explosion pool
        } else {
            LOG_INFO("[Explosion] Upload buffer created ({} KB)", EXPLOSION_UPLOAD_BUFFER_SIZE / 1024);
        }
    }

    // Initialize PINN ML Physics System (optional)
    m_pinnPhysics = new PINNPhysicsSystem();
    // Path relative to project root (working directory)
    // Try v3 TOTAL FORCES model first (gravity + viscosity + MRI)
    // Falls back to v2 models if v3 not available
    bool pinnLoaded = false;
    if (m_pinnPhysics->Initialize("ml/models/pinn_v3_total_forces.onnx")) {
        LOG_INFO("[PINN] Loaded v3 TOTAL FORCES model (gravity + viscosity + MRI)");
        LOG_INFO("[PINN] This model outputs gravitational forces directly for proper orbital motion");
        pinnLoaded = true;
    } else if (m_pinnPhysics->Initialize("ml/models/pinn_v2_turbulent.onnx")) {
        LOG_INFO("[PINN] Loaded v2 turbulent model (MRI + Kolmogorov physics)");
        LOG_WARN("[PINN] v2 may have rotation issues - recommend using v3");
        pinnLoaded = true;
    } else if (m_pinnPhysics->Initialize("ml/models/pinn_v2_param_conditioned.onnx")) {
        LOG_INFO("[PINN] Loaded v2 standard param-conditioned model");
        LOG_WARN("[PINN] v2 may have rotation issues - recommend using v3");
        pinnLoaded = true;
    }

    if (pinnLoaded) {
        m_pinnPhysics->SetEnabled(false);  // Start disabled (user can enable via 'P' key)
        m_pinnPhysics->SetHybridMode(true);
        m_pinnPhysics->SetHybridThreshold(10.0f);  // 10× R_ISCO
        LOG_INFO("PINN physics available! Press 'P' to toggle PINN physics");
        LOG_INFO("  {}", m_pinnPhysics->GetModelInfo());

        // Allocate CPU buffers for PINN inference
        m_cpuPositions.resize(particleCount);
        m_cpuVelocities.resize(particleCount);
        m_cpuForces.resize(particleCount);

        // Create persistent upload buffer for PINN (CRITICAL: must outlive GPU command execution)
        // This fixes the crash caused by using a temporary buffer that gets destroyed before GPU executes
        CD3DX12_HEAP_PROPERTIES pinnUploadProps(D3D12_HEAP_TYPE_UPLOAD);
        CD3DX12_RESOURCE_DESC pinnUploadDesc = CD3DX12_RESOURCE_DESC::Buffer(bufferSize);

        hr = m_device->GetDevice()->CreateCommittedResource(
            &pinnUploadProps, D3D12_HEAP_FLAG_NONE, &pinnUploadDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
            IID_PPV_ARGS(&m_pinnUploadBuffer)
        );

        if (FAILED(hr)) {
            LOG_ERROR("[PINN] Failed to create persistent upload buffer - PINN disabled");
            delete m_pinnPhysics;
            m_pinnPhysics = nullptr;
        } else {
            LOG_INFO("[PINN] Persistent upload buffer created ({} KB)", bufferSize / 1024);

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
        }
    } else {
        LOG_INFO("PINN not available (ONNX Runtime not installed or model not found)");
        LOG_INFO("  GPU physics will be used exclusively");
    }
    
    // Initialize SIREN Vortex Field (ML-based turbulence)
    m_sirenVortex = new SIRENVortexField();
    if (m_sirenVortex->Initialize("ml/models/vortex_siren.onnx")) {
        LOG_INFO("[SIREN] Loaded SIREN vortex field model!");
        m_sirenVortex->SetEnabled(false);  // Start disabled (user can enable via ImGui)
        m_sirenVortex->SetIntensity(0.5f); // Default moderate intensity
        LOG_INFO("[SIREN] Turbulence available! Enable via ImGui controls");
        
        // Allocate turbulence force buffer
        m_cpuTurbulenceForces.resize(particleCount);
    } else {
        LOG_INFO("[SIREN] Vortex field model not found (optional - turbulence disabled)");
        delete m_sirenVortex;
        m_sirenVortex = nullptr;
    }

    return true;
}

void ParticleSystem::InitializeAccretionDisk() {
    // Physics shader handles GPU initialization - this is just a placeholder
    LOG_INFO("Particles will be GPU-initialized by physics shader");
}

void ParticleSystem::InitializeAccretionDisk_CPU() {
    // CPU-based initialization for PINN mode
    // CRITICAL: Uses PINN normalized units where GM = 1
    // Keplerian velocity: v = sqrt(GM/r) = sqrt(1/r) in normalized units

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> radiusDist(INNER_STABLE_ORBIT, OUTER_DISK_RADIUS * 0.7f);  // Keep particles in well-trained region
    std::uniform_real_distribution<float> angleDist(0.0f, 2.0f * 3.14159265f);

    // Get disk thickness from PINN params if available, else use default
    float diskHR = m_pinnPhysics ? m_pinnPhysics->GetDiskThickness() : 0.1f;

    for (uint32_t i = 0; i < m_particleCount; i++) {
        // Cylindrical coordinates
        float radius = radiusDist(rng);
        float angle = angleDist(rng);

        // Disk height based on H/R ratio (thinner for PINN accuracy)
        float scaleHeight = diskHR * radius;
        float height = std::uniform_real_distribution<float>(-scaleHeight * 0.5f, scaleHeight * 0.5f)(rng);

        // Convert to Cartesian
        m_cpuPositions[i].x = radius * cosf(angle);
        m_cpuPositions[i].y = height;
        m_cpuPositions[i].z = radius * sinf(angle);

        // PINN Normalized Units: Keplerian velocity v = sqrt(GM/r) = sqrt(1/r)
        // Get black hole mass multiplier from PINN params (default 1.0)
        float bhMass = m_pinnPhysics ? m_pinnPhysics->GetBlackHoleMass() : 1.0f;
        float GM = PINN_GM * bhMass;
        float orbitalSpeed = sqrtf(GM / radius);

        // Add small random perturbation (1-2%) for realism - matches training data
        float perturbation = 1.0f + std::uniform_real_distribution<float>(-0.01f, 0.01f)(rng);
        orbitalSpeed *= perturbation;

        // Tangential velocity in disk plane (counter-clockwise orbit)
        m_cpuVelocities[i].x = -orbitalSpeed * sinf(angle);
        m_cpuVelocities[i].y = 0.0f;  // No initial vertical velocity
        m_cpuVelocities[i].z = orbitalSpeed * cosf(angle);

        // NOTE: albedo and materialType initialized in UploadParticleData
    }

    LOG_INFO("[PINN] CPU-initialized {} particles with NORMALIZED units (GM={})",
             m_particleCount, PINN_GM);
    LOG_INFO("[PINN] Orbital speed range: {:.4f} to {:.4f} (at r={} to r={})",
             sqrtf(PINN_GM / (OUTER_DISK_RADIUS * 0.7f)), sqrtf(PINN_GM / INNER_STABLE_ORBIT),
             OUTER_DISK_RADIUS * 0.7f, INNER_STABLE_ORBIT);
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
        uint32_t integrationMethod;  // 0 = Euler, 1 = Velocity Verlet
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
    constants.integrationMethod = m_integrationMethod;  // 0 = Euler, 1 = Verlet

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
    m_explosionUploadBuffer.Reset();  // Phase 2C
    m_pinnUploadBuffer.Reset();       // PINN persistent upload buffer

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

    // DIAGNOSTIC: Log entry to PINN physics update
    static bool firstCall = true;
    if (firstCall) {
        LOG_INFO("[PINN] First call to UpdatePhysics_PINN");
        LOG_INFO("[PINN]   CPU positions size: {}", m_cpuPositions.size());
        LOG_INFO("[PINN]   CPU velocities size: {}", m_cpuVelocities.size());
        LOG_INFO("[PINN]   CPU forces size: {}", m_cpuForces.size());
        LOG_INFO("[PINN]   Active particle count: {}", m_activeParticleCount);
        LOG_INFO("[PINN]   Positions pointer: {}", (void*)m_cpuPositions.data());
        LOG_INFO("[PINN]   Velocities pointer: {}", (void*)m_cpuVelocities.data());
        LOG_INFO("[PINN]   Forces pointer: {}", (void*)m_cpuForces.data());
        firstCall = false;
    }

    // Validate vector sizes match
    if (m_cpuPositions.size() != m_particleCount ||
        m_cpuVelocities.size() != m_particleCount ||
        m_cpuForces.size() != m_particleCount) {
        LOG_ERROR("[PINN] Vector size mismatch! pos={}, vel={}, force={}, expected={}",
            m_cpuPositions.size(), m_cpuVelocities.size(), m_cpuForces.size(), m_particleCount);
        return;
    }

    // Step 1: Predict orbital forces using PINN (works directly on CPU particles)
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

    // Step 1.5: Add SIREN turbulence forces (if enabled)
    // PINN v4 is trained to be robust to velocity perturbations, so SIREN turbulence is safe
    if (m_sirenVortex && m_sirenVortex->IsEnabled()) {
        // Compute turbulent forces: F_turb = cross(velocity, vorticity)
        bool sirenSuccess = m_sirenVortex->ComputeTurbulentForcesBatch(
            m_cpuPositions.data(),
            m_cpuVelocities.data(),
            m_cpuTurbulenceForces.data(),
            m_activeParticleCount,
            totalTime
        );

        if (sirenSuccess) {
            // Add turbulence to orbital forces: F_total = F_orbital + F_turbulence
            for (uint32_t i = 0; i < m_activeParticleCount; i++) {
                m_cpuForces[i].x += m_cpuTurbulenceForces[i].x;
                m_cpuForces[i].y += m_cpuTurbulenceForces[i].y;
                m_cpuForces[i].z += m_cpuTurbulenceForces[i].z;
            }
        }
    }

    // Step 2: Integrate forces (Velocity Verlet) on CPU particles
    IntegrateForces(m_cpuForces, deltaTime);

    // Step 3: Upload updated CPU particles to GPU for rendering
    UploadParticleData(m_cpuPositions, m_cpuVelocities);

    // Log performance metrics and force diagnostics every 60 frames
    static int s_pinnUpdateCount = 0;
    s_pinnUpdateCount++;
    if (s_pinnUpdateCount % 60 == 0) {
        auto metrics = m_pinnPhysics->GetPerformanceMetrics();

        // Compute average force magnitude and direction for diagnostics
        float avgFx = 0.0f, avgFy = 0.0f, avgFz = 0.0f;
        float maxForceMag = 0.0f;
        for (uint32_t i = 0; i < m_activeParticleCount; i++) {
            avgFx += m_cpuForces[i].x;
            avgFy += m_cpuForces[i].y;
            avgFz += m_cpuForces[i].z;
            float mag = sqrtf(m_cpuForces[i].x * m_cpuForces[i].x +
                            m_cpuForces[i].y * m_cpuForces[i].y +
                            m_cpuForces[i].z * m_cpuForces[i].z);
            maxForceMag = std::max(maxForceMag, mag);
        }
        avgFx /= m_activeParticleCount;
        avgFy /= m_activeParticleCount;
        avgFz /= m_activeParticleCount;
        float avgNetForce = sqrtf(avgFx*avgFx + avgFy*avgFy + avgFz*avgFz);

        LOG_INFO("[PINN] Frame {} - Inference: {:.2f}ms | Avg force: ({:.4f}, {:.4f}, {:.4f}) mag={:.4f} | Max: {:.4f}",
                 s_pinnUpdateCount, metrics.inferenceTimeMs, avgFx, avgFy, avgFz, avgNetForce, maxForceMag);
    }
}

// ReadbackParticleData() removed - particles are initialized on CPU when PINN is available
// This eliminates GPU crashes caused by copying particle buffer while RT acceleration structures reference it

void ParticleSystem::UploadParticleData(
    const std::vector<DirectX::XMFLOAT3>& positions,
    const std::vector<DirectX::XMFLOAT3>& velocities) {

    // CRITICAL FIX: Use persistent upload buffer instead of temporary
    // The old code created a local ComPtr that was destroyed before GPU executed the copy command
    // This caused a use-after-free crash when the main loop executed the command list

    if (!m_pinnUploadBuffer) {
        LOG_ERROR("[PINN] Persistent upload buffer not initialized!");
        return;
    }

    // Map persistent upload buffer and copy particle data
    void* uploadData = nullptr;
    HRESULT hr = m_pinnUploadBuffer->Map(0, nullptr, &uploadData);
    if (FAILED(hr)) {
        LOG_ERROR("[PINN] Failed to map persistent upload buffer");
        return;
    }

    Particle* particles = static_cast<Particle*>(uploadData);

    // PARALLELIZED: Populate particle data using all available CPU cores
    std::vector<uint32_t> indices(m_activeParticleCount);
    std::iota(indices.begin(), indices.end(), 0);

    std::for_each(std::execution::par, indices.begin(), indices.end(),
        [&positions, &velocities, particles](uint32_t i) {
            particles[i].position = positions[i];
            particles[i].velocity = velocities[i];

            // CRITICAL FIX: Must set temperature and density for particles to be visible!
            // Temperature determines blackbody emission color, density determines opacity.
            // Without these, particles are invisible (temperature=0 means no emission).
            float x = positions[i].x;
            float y = positions[i].y;
            float z = positions[i].z;
            float radius = sqrtf(x * x + y * y + z * z);

            // Temperature based on radius (hotter near black hole, cooler at edge)
            // Inner disk: ~30000K, Outer disk: ~5000K (matches GPU physics shader)
            float normalizedRadius = (radius - INNER_STABLE_ORBIT) / (OUTER_DISK_RADIUS - INNER_STABLE_ORBIT);
            normalizedRadius = (std::max)(0.0f, (std::min)(1.0f, normalizedRadius));
            particles[i].temperature = 30000.0f - normalizedRadius * 25000.0f;  // 30000K → 5000K

            // Density affects opacity - use consistent value for volumetric rendering
            particles[i].density = 0.8f;

            // Sprint 1: Initialize new material system fields
            // Default to PLASMA material with warm orange albedo (backward compatible)
            particles[i].albedo = DirectX::XMFLOAT3(1.0f, 0.4f, 0.1f);  // Hot plasma orange/red
            particles[i].materialType = static_cast<uint32_t>(ParticleMaterialType::PLASMA);  // Type 0

            // Initialize lifetime fields (non-explosive particles are immortal)
            particles[i].lifetime = 0.0f;
            particles[i].maxLifetime = 0.0f;  // 0 = infinite/immortal
            particles[i].spawnTime = 0.0f;
            particles[i].flags = 0;
        });

    m_pinnUploadBuffer->Unmap(0, nullptr);

    // Copy to GPU using the persistent buffer (buffer will remain valid until next frame)
    auto cmdList = m_device->GetCommandList();

    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = m_particleBuffer.Get();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmdList->ResourceBarrier(1, &barrier);

    cmdList->CopyResource(m_particleBuffer.Get(), m_pinnUploadBuffer.Get());

    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    cmdList->ResourceBarrier(1, &barrier);

    // Don't execute - let main loop handle it
    // The persistent m_pinnUploadBuffer will remain valid until destroyed in destructor
}

void ParticleSystem::IntegrateForces(const std::vector<DirectX::XMFLOAT3>& forces, float deltaTime) {
    // Velocity Verlet integration - PARALLELIZED for multi-core CPUs
    // Uses C++17 parallel algorithms to leverage all available cores
    // Includes configurable damping, turbulence, and boundary conditions

    // Create index range for parallel iteration
    std::vector<uint32_t> indices(m_activeParticleCount);
    std::iota(indices.begin(), indices.end(), 0);

    // Get PINN visualization parameters
    const float turbulence = m_pinnTurbulence;
    const float damping = m_pinnDamping;
    const bool enforceBounds = m_pinnEnforceBoundaries;
    const float innerRadius = PINN_R_ISCO * 1.01f;
    const float outerRadius = OUTER_DISK_RADIUS * 0.99f;

    // Random seed for turbulence (different each frame)
    unsigned int seed = static_cast<unsigned int>(m_totalTime * 1000.0f);

    // Parallel integration across all particles
    std::for_each(std::execution::par, indices.begin(), indices.end(),
        [this, &forces, deltaTime, turbulence, damping, enforceBounds,
         innerRadius, outerRadius, seed](uint32_t i) {
            // Update velocity: v' = v + a * dt
            // Let PINN forces work unmodified - they contain the learned physics balance
            float ax = forces[i].x;
            float ay = forces[i].y;
            float az = forces[i].z;

            m_cpuVelocities[i].x += ax * deltaTime;
            m_cpuVelocities[i].y += ay * deltaTime;
            m_cpuVelocities[i].z += az * deltaTime;

            // PINN v3 SAFETY: Clamp velocity to prevent runaway when model outputs garbage
            // Keplerian velocity at inner edge (r=6) is sqrt(100/6)≈4.08
            // Allow up to 3× Keplerian for perturbations, but no more
            const float maxVelMagnitude = 15.0f;  // 3× max Keplerian velocity
            float vx = m_cpuVelocities[i].x;
            float vy = m_cpuVelocities[i].y;
            float vz = m_cpuVelocities[i].z;
            float velMag = sqrtf(vx*vx + vy*vy + vz*vz);
            if (velMag > maxVelMagnitude) {
                float scale = maxVelMagnitude / velMag;
                m_cpuVelocities[i].x *= scale;
                m_cpuVelocities[i].y *= scale;
                m_cpuVelocities[i].z *= scale;
            }

            // Add turbulence (DISABLED for PINN v3 - model can't handle non-Keplerian velocities!)
            // The PINN was trained only on particles with Keplerian orbital velocities.
            // Adding random velocity perturbations pushes particles outside the trained distribution,
            // causing the model to output garbage forces that create runaway feedback loops.
            // TODO: Retrain PINN with velocity perturbations in training data if turbulence is needed.
            // For now, turbulence is intentionally disabled to prevent explosions.
            (void)turbulence;  // Suppress unused variable warning
            (void)seed;

            // Apply configurable damping (default=1.0 for PINN v3)
            // CRITICAL: PINN v3 model already includes learned viscosity forces
            // External damping (< 1.0) will fight the physics and kill orbital velocity!
            // Only use damping < 1.0 for legacy GPU physics or debugging
            m_cpuVelocities[i].x *= damping;
            m_cpuVelocities[i].y *= damping;
            m_cpuVelocities[i].z *= damping;

            // Update position: p' = p + v' * dt
            m_cpuPositions[i].x += m_cpuVelocities[i].x * deltaTime;
            m_cpuPositions[i].y += m_cpuVelocities[i].y * deltaTime;
            m_cpuPositions[i].z += m_cpuVelocities[i].z * deltaTime;

            // Boundary conditions (optional) - SOFTENED for PINN v3
            // PINN model includes gravitational forces that naturally contain particles
            // Only apply very gentle boundary nudging, don't kill velocities!
            if (enforceBounds) {
                float px = m_cpuPositions[i].x;
                float py = m_cpuPositions[i].y;
                float pz = m_cpuPositions[i].z;
                float radius = sqrtf(px*px + pz*pz);

                // Radial boundaries - VERY soft nudge, preserve orbital velocity!
                // Inner boundary: reflect outward gently (particle fell too close to black hole)
                if (radius < innerRadius) {
                    float scale = innerRadius / (radius + 1e-6f);
                    m_cpuPositions[i].x *= scale;
                    m_cpuPositions[i].z *= scale;
                    // DON'T kill velocity - just flip radial component outward
                    // This preserves orbital motion while bouncing from inner edge
                } else if (radius > outerRadius) {
                    float scale = outerRadius / (radius + 1e-6f);
                    m_cpuPositions[i].x *= scale;
                    m_cpuPositions[i].z *= scale;
                    // Gentle slow-down at outer edge (particles escaping disk)
                    m_cpuVelocities[i].x *= 0.98f;
                    m_cpuVelocities[i].z *= 0.98f;
                }

                // Vertical confinement - keep near disk midplane
                float maxHeight = 0.15f * radius;  // H/R ~ 0.15 (slightly thicker)
                m_cpuPositions[i].y = std::clamp(m_cpuPositions[i].y, -maxHeight, maxHeight);
            }
        });
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

// === PINN v2 Physics Parameters ===

bool ParticleSystem::IsPINNParameterConditioned() const {
    return m_pinnPhysics ? m_pinnPhysics->IsParameterConditioned() : false;
}

int ParticleSystem::GetPINNModelVersion() const {
    return m_pinnPhysics ? m_pinnPhysics->GetModelVersion() : 0;
}

float ParticleSystem::GetPINNBlackHoleMass() const {
    return m_pinnPhysics ? m_pinnPhysics->GetBlackHoleMass() : 1.0f;
}

void ParticleSystem::SetPINNBlackHoleMass(float normalized) {
    if (m_pinnPhysics) {
        m_pinnPhysics->SetBlackHoleMass(normalized);
    }
}

float ParticleSystem::GetPINNAlphaViscosity() const {
    return m_pinnPhysics ? m_pinnPhysics->GetAlphaViscosity() : 0.1f;
}

void ParticleSystem::SetPINNAlphaViscosity(float alpha) {
    if (m_pinnPhysics) {
        m_pinnPhysics->SetAlphaViscosity(alpha);
    }
}

float ParticleSystem::GetPINNDiskThickness() const {
    return m_pinnPhysics ? m_pinnPhysics->GetDiskThickness() : 0.1f;
}

void ParticleSystem::SetPINNDiskThickness(float hrRatio) {
    if (m_pinnPhysics) {
        m_pinnPhysics->SetDiskThickness(hrRatio);
    }
}

void ParticleSystem::ReinitializePINNParticles() {
    if (!m_particlesOnCPU) {
        LOG_WARN("[PINN] Cannot reinitialize - particles not on CPU");
        return;
    }

    LOG_INFO("[PINN] Reinitializing particles with current parameters...");

    // Store current PINN state
    bool wasEnabled = m_usePINN;

    // Reinitialize CPU particle positions and velocities
    InitializeAccretionDisk_CPU();

    // Reset total time for fresh simulation
    m_totalTime = 0.0f;

    // Upload to GPU for rendering
    UploadParticleData(m_cpuPositions, m_cpuVelocities);

    LOG_INFO("[PINN] Reinitialized {} particles", m_particleCount);
}

// === PINN Model Selection ===

std::string ParticleSystem::GetPINNModelName() const {
    return m_pinnPhysics ? m_pinnPhysics->GetCurrentModelName() : "none";
}

std::string ParticleSystem::GetPINNModelPath() const {
    return m_pinnPhysics ? m_pinnPhysics->GetCurrentModelPath() : "";
}

std::vector<std::pair<std::string, std::string>> ParticleSystem::GetAvailablePINNModels() const {
    return PINNPhysicsSystem::GetAvailableModels();
}

bool ParticleSystem::LoadPINNModel(const std::string& modelPath) {
    if (!m_pinnPhysics) {
        LOG_ERROR("[PINN] Cannot load model - PINN system not initialized");
        return false;
    }
    
    LOG_INFO("[PINN] Loading model: {}", modelPath);
    
    bool wasEnabled = m_usePINN;
    m_usePINN = false;  // Disable during model switch
    
    bool result = m_pinnPhysics->LoadModel(modelPath);
    
    if (result) {
        LOG_INFO("[PINN] Model loaded successfully: {}", m_pinnPhysics->GetCurrentModelName());
        LOG_INFO("[PINN] Model version: v{}", m_pinnPhysics->GetModelVersion());
        
        // Reinitialize particles for the new model
        if (m_particlesOnCPU) {
            ReinitializePINNParticles();
        }
        
        m_usePINN = wasEnabled;  // Restore enabled state
    } else {
        LOG_ERROR("[PINN] Failed to load model: {}", modelPath);
    }
    
    return result;
}

// === SIREN Vortex Field (ML-based Turbulence) ===

bool ParticleSystem::IsSIRENAvailable() const {
    return m_sirenVortex && m_sirenVortex->IsAvailable();
}

bool ParticleSystem::IsSIRENEnabled() const {
    return m_sirenVortex ? m_sirenVortex->IsEnabled() : false;
}

void ParticleSystem::SetSIRENEnabled(bool enabled) {
    if (m_sirenVortex) {
        m_sirenVortex->SetEnabled(enabled);
        LOG_INFO("[SIREN] Turbulence {}", enabled ? "ENABLED" : "DISABLED");
    }
}

float ParticleSystem::GetSIRENIntensity() const {
    return m_sirenVortex ? m_sirenVortex->GetIntensity() : 0.0f;
}

void ParticleSystem::SetSIRENIntensity(float intensity) {
    if (m_sirenVortex) {
        m_sirenVortex->SetIntensity(intensity);
    }
}

float ParticleSystem::GetSIRENSeed() const {
    return m_sirenVortex ? m_sirenVortex->GetSeed() : 0.0f;
}

void ParticleSystem::SetSIRENSeed(float seed) {
    if (m_sirenVortex) {
        m_sirenVortex->SetSeed(seed);
    }
}

std::string ParticleSystem::GetSIRENModelInfo() const {
    return m_sirenVortex ? m_sirenVortex->GetModelInfo() : "SIREN not available";
}

ParticleSystem::SIRENMetrics ParticleSystem::GetSIRENMetrics() const {
    SIRENMetrics metrics;
    if (m_sirenVortex) {
        auto sirenMetrics = m_sirenVortex->GetPerformanceMetrics();
        metrics.inferenceTimeMs = sirenMetrics.inferenceTimeMs;
        metrics.particlesProcessed = sirenMetrics.particlesProcessed;
        metrics.avgBatchTimeMs = sirenMetrics.avgBatchTimeMs;
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

    // ============================================================================
    // Phase 2: PYRO/EXPLOSION MATERIALS
    // ============================================================================

    // Material 5: SUPERNOVA (Explosive stellar death - EXTREME brilliance)
    // Starts at 100,000K+, expands rapidly, fades over 5-10 seconds
    m_materialProperties.materials[5].albedo = DirectX::XMFLOAT3(1.0f, 0.95f, 0.8f);  // Brilliant white-yellow
    m_materialProperties.materials[5].opacity = 0.95f;                // Nearly opaque core
    m_materialProperties.materials[5].emissionMultiplier = 15.0f;     // EXTREME emission (blinding!)
    m_materialProperties.materials[5].scatteringCoefficient = 1.5f;   // Moderate scattering
    m_materialProperties.materials[5].phaseG = 0.8f;                  // Strong forward scatter (radial glow)
    m_materialProperties.materials[5].expansionRate = 50.0f;          // Fast expansion (50 units/sec)
    m_materialProperties.materials[5].coolingRate = 8000.0f;          // Rapid cooling (8000K/sec)
    m_materialProperties.materials[5].fadeStartRatio = 0.6f;          // Start fading at 60% lifetime

    // Material 6: STELLAR_FLARE (Solar flare ejection - hot plasma burst)
    // Coronal mass ejection, 20,000K, arcing trajectory
    m_materialProperties.materials[6].albedo = DirectX::XMFLOAT3(1.0f, 0.6f, 0.2f);   // Hot orange-yellow
    m_materialProperties.materials[6].opacity = 0.7f;                 // Semi-transparent
    m_materialProperties.materials[6].emissionMultiplier = 6.0f;      // High emission
    m_materialProperties.materials[6].scatteringCoefficient = 2.0f;   // Good scattering
    m_materialProperties.materials[6].phaseG = 0.5f;                  // Forward scatter
    m_materialProperties.materials[6].expansionRate = 20.0f;          // Moderate expansion
    m_materialProperties.materials[6].coolingRate = 3000.0f;          // Slower cooling
    m_materialProperties.materials[6].fadeStartRatio = 0.7f;          // Fade at 70% lifetime

    // Material 7: SHOCKWAVE (Expanding shockwave ring - fast, dramatic)
    // Thin expanding shell, very fast, short-lived
    m_materialProperties.materials[7].albedo = DirectX::XMFLOAT3(0.8f, 0.9f, 1.0f);   // Bright blue-white
    m_materialProperties.materials[7].opacity = 0.4f;                 // Semi-transparent ring
    m_materialProperties.materials[7].emissionMultiplier = 10.0f;     // Very bright edge
    m_materialProperties.materials[7].scatteringCoefficient = 0.5f;   // Low scattering (sharp edge)
    m_materialProperties.materials[7].phaseG = 0.0f;                  // Isotropic (ring visible from all angles)
    m_materialProperties.materials[7].expansionRate = 100.0f;         // VERY fast expansion (shockwave!)
    m_materialProperties.materials[7].coolingRate = 1000.0f;          // Quick fade
    m_materialProperties.materials[7].fadeStartRatio = 0.4f;          // Early fade (shockwaves are brief)

    // ============================================================================
    // LUMINOUS STARS: SUPERGIANT_STAR MATERIAL
    // ============================================================================

    // Material 8: SUPERGIANT_STAR (Luminous star particles with embedded lights)
    // Blue-white supergiant, VERY low opacity so light shines through!
    // These particles contain embedded point lights that illuminate neighbors
    m_materialProperties.materials[8].albedo = DirectX::XMFLOAT3(0.85f, 0.9f, 1.0f);  // Blue-white (25000K+)
    m_materialProperties.materials[8].opacity = 0.15f;                // VERY transparent - light shines through!
    m_materialProperties.materials[8].emissionMultiplier = 15.0f;     // Highest emission (matches SUPERNOVA)
    m_materialProperties.materials[8].scatteringCoefficient = 0.3f;   // Low scattering (self-luminous core)
    m_materialProperties.materials[8].phaseG = 0.0f;                  // Isotropic (glow visible from all angles)
    m_materialProperties.materials[8].expansionRate = 0.0f;           // No expansion (static star)
    m_materialProperties.materials[8].coolingRate = 0.0f;             // No cooling (permanent)
    m_materialProperties.materials[8].fadeStartRatio = 1.0f;          // Never fade

    // Set default expansion/cooling for non-explosive materials (0 = no effect)
    for (int i = 0; i < 5; i++) {
        m_materialProperties.materials[i].expansionRate = 0.0f;
        m_materialProperties.materials[i].coolingRate = 0.0f;
        m_materialProperties.materials[i].fadeStartRatio = 1.0f;  // Never fade
    }

    // Zero out padding to avoid undefined behavior
    for (int i = 0; i < 9; i++) {  // Updated to 9 materials
        for (int j = 0; j < 6; j++) {
            m_materialProperties.materials[i].padding[j] = 0.0f;
        }
    }

    LOG_INFO("[Material System] Initialized 9 material presets:");
    LOG_INFO("  0: PLASMA          - Hot orange/red, emission 2.5×");
    LOG_INFO("  1: STAR            - Brilliant white-yellow, emission 8.0×");
    LOG_INFO("  2: GAS_CLOUD       - Wispy blue/purple, backward scatter");
    LOG_INFO("  3: ROCKY_BODY      - Deep grey, minimal emission");
    LOG_INFO("  4: ICY_BODY        - Bright blue-white, reflective");
    LOG_INFO("  5: SUPERNOVA       - Extreme emission 15×, expansion 50u/s");
    LOG_INFO("  6: STELLAR_FLARE   - Hot plasma burst, emission 6×");
    LOG_INFO("  7: SHOCKWAVE       - Fast expanding ring, emission 10×");
    LOG_INFO("  8: SUPERGIANT_STAR - Blue-white 25000K+, emission 15×, opacity 0.15 (luminous)");
}

bool ParticleSystem::CreateMaterialPropertiesBuffer() {
    // Create upload buffer for material properties (576 bytes: 9 materials × 64 bytes)
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

// ============================================================================
// Phase 2C: Explosion Spawning System Implementation
// ============================================================================

void ParticleSystem::SpawnExplosion(const ExplosionConfig& config) {
    // Queue explosion for processing after command list reset
    // This ensures GPU copy commands are recorded at the correct time
    m_pendingExplosions.push_back(config);
    LOG_INFO("[Explosion] Queued explosion at ({:.1f}, {:.1f}, {:.1f}) type={} (will spawn {} particles)",
             config.position.x, config.position.y, config.position.z,
             static_cast<uint32_t>(config.type), config.particleCount);
}

uint32_t ParticleSystem::ProcessPendingExplosions() {
    if (m_pendingExplosions.empty()) {
        return 0;
    }

    if (!m_device || !m_particleBuffer) {
        LOG_ERROR("[Explosion] Cannot process - system not initialized");
        m_pendingExplosions.clear();
        return 0;
    }

    if (m_explosionPoolStart == 0) {
        LOG_ERROR("[Explosion] Cannot process - no explosion pool available");
        m_pendingExplosions.clear();
        return 0;
    }

    uint32_t totalSpawned = 0;

    // Process all pending explosions
    for (const auto& config : m_pendingExplosions) {
        // Clamp particle count to available pool space
        uint32_t particlesToSpawn = (std::min)(config.particleCount, EXPLOSION_POOL_SIZE);

        LOG_INFO("[Explosion] Processing {} particles at ({:.1f}, {:.1f}, {:.1f}) type={}",
                 particlesToSpawn, config.position.x, config.position.y, config.position.z,
                 static_cast<uint32_t>(config.type));

        // Generate explosion particles
        std::vector<Particle> explosionParticles(particlesToSpawn);

        // Get material properties for albedo
        const auto& matProps = m_materialProperties.materials[static_cast<uint32_t>(config.type)];

        for (uint32_t i = 0; i < particlesToSpawn; i++) {
            Particle& p = explosionParticles[i];

            // Generate random direction on unit sphere (Fibonacci sphere for uniform distribution)
            float phi = acosf(1.0f - 2.0f * (float(i) + 0.5f) / float(particlesToSpawn));
            float theta = 3.14159265f * (1.0f + sqrtf(5.0f)) * float(i);

            float sinPhi = sinf(phi);
            float cosPhi = cosf(phi);
            float sinTheta = sinf(theta);
            float cosTheta = cosf(theta);

            DirectX::XMFLOAT3 direction = {
                sinPhi * cosTheta,
                cosPhi,
                sinPhi * sinTheta
            };

            // Add some randomness to the direction
            float randOffset = 0.1f;
            direction.x += (float(rand()) / RAND_MAX - 0.5f) * randOffset;
            direction.y += (float(rand()) / RAND_MAX - 0.5f) * randOffset;
            direction.z += (float(rand()) / RAND_MAX - 0.5f) * randOffset;

            // Normalize direction
            float len = sqrtf(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);
            if (len > 0.0001f) {
                direction.x /= len;
                direction.y /= len;
                direction.z /= len;
            }

            // Random initial radius within spawn sphere
            float spawnRadius = config.initialRadius * (0.5f + 0.5f * float(rand()) / RAND_MAX);

            // Position: center + direction * initial radius
            p.position.x = config.position.x + direction.x * spawnRadius;
            p.position.y = config.position.y + direction.y * spawnRadius;
            p.position.z = config.position.z + direction.z * spawnRadius;

            // Velocity: outward expansion with some variation
            float speedVariation = 0.8f + 0.4f * float(rand()) / RAND_MAX;
            p.velocity.x = direction.x * config.expansionSpeed * speedVariation;
            p.velocity.y = direction.y * config.expansionSpeed * speedVariation;
            p.velocity.z = direction.z * config.expansionSpeed * speedVariation;

            // Temperature with slight variation
            p.temperature = config.temperature * (0.9f + 0.2f * float(rand()) / RAND_MAX);

            // Density
            p.density = config.density;

            // Material properties
            p.albedo = matProps.albedo;
            p.materialType = static_cast<uint32_t>(config.type);

            // Lifetime fields
            p.lifetime = 0.0f;  // Just spawned
            p.maxLifetime = config.lifetime * (0.8f + 0.4f * float(rand()) / RAND_MAX);  // Vary lifetime
            p.spawnTime = m_totalTime;
            p.flags = static_cast<uint32_t>(ParticleFlags::EXPLOSION);
        }

        // Calculate where to place these particles in the explosion pool
        uint32_t startIndex = m_explosionPoolStart + m_nextExplosionIndex;

        // Upload to GPU
        UploadExplosionParticles(explosionParticles, startIndex);

        // Update pool tracking (circular buffer)
        m_nextExplosionIndex = (m_nextExplosionIndex + particlesToSpawn) % EXPLOSION_POOL_SIZE;
        m_explosionPoolUsed = (std::min)(m_explosionPoolUsed + particlesToSpawn, EXPLOSION_POOL_SIZE);

        LOG_INFO("[Explosion] Spawned {} particles at pool index {} (pool usage: {}/{})",
                 particlesToSpawn, startIndex - m_explosionPoolStart,
                 m_explosionPoolUsed, EXPLOSION_POOL_SIZE);

        totalSpawned += particlesToSpawn;
    }

    // Clear pending queue
    m_pendingExplosions.clear();

    return totalSpawned;
}

void ParticleSystem::SpawnRandomExplosion(ParticleMaterialType type) {
    ExplosionConfig config;
    config.type = type;

    // Random position in the accretion disk area
    float angle = float(rand()) / RAND_MAX * 6.28318f;  // 0 to 2π
    float radius = INNER_STABLE_ORBIT + float(rand()) / RAND_MAX * (OUTER_DISK_RADIUS - INNER_STABLE_ORBIT);
    float height = (float(rand()) / RAND_MAX - 0.5f) * DISK_THICKNESS * 0.5f;

    config.position.x = radius * cosf(angle);
    config.position.y = height;
    config.position.z = radius * sinf(angle);

    // Type-specific parameters
    switch (type) {
        case ParticleMaterialType::SUPERNOVA:
            config.particleCount = 200;
            config.expansionSpeed = 150.0f;
            config.temperature = 80000.0f;
            config.lifetime = 5.0f;
            config.initialRadius = 20.0f;
            config.density = 0.9f;
            break;

        case ParticleMaterialType::STELLAR_FLARE:
            config.particleCount = 80;
            config.expansionSpeed = 80.0f;
            config.temperature = 40000.0f;
            config.lifetime = 2.5f;
            config.initialRadius = 10.0f;
            config.density = 0.7f;
            break;

        case ParticleMaterialType::SHOCKWAVE:
            config.particleCount = 150;
            config.expansionSpeed = 200.0f;
            config.temperature = 25000.0f;
            config.lifetime = 2.0f;
            config.initialRadius = 5.0f;
            config.density = 0.5f;
            break;

        default:
            // Generic explosion
            config.particleCount = 100;
            config.expansionSpeed = 100.0f;
            config.temperature = 50000.0f;
            config.lifetime = 3.0f;
            break;
    }

    SpawnExplosion(config);
}

void ParticleSystem::UploadExplosionParticles(const std::vector<Particle>& particles, uint32_t startIndex) {
    if (particles.empty()) return;

    if (!m_explosionUploadBuffer) {
        LOG_ERROR("[Explosion] Upload buffer not initialized");
        return;
    }

    size_t uploadSize = particles.size() * sizeof(Particle);
    size_t bufferOffset = startIndex * sizeof(Particle);

    // Validate upload size fits in our persistent buffer
    if (uploadSize > EXPLOSION_UPLOAD_BUFFER_SIZE) {
        LOG_ERROR("[Explosion] Upload size {} exceeds buffer size {}", uploadSize, EXPLOSION_UPLOAD_BUFFER_SIZE);
        return;
    }

    // Map persistent upload buffer and copy particle data
    void* uploadData = nullptr;
    HRESULT hr = m_explosionUploadBuffer->Map(0, nullptr, &uploadData);
    if (FAILED(hr)) {
        LOG_ERROR("[Explosion] Failed to map upload buffer");
        return;
    }

    memcpy(uploadData, particles.data(), uploadSize);
    m_explosionUploadBuffer->Unmap(0, nullptr);

    // Copy to particle buffer at the specified offset
    // NOTE: We just record the copy command - it will execute as part of the frame's command list
    auto cmdList = m_device->GetCommandList();

    // Transition particle buffer to copy dest
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = m_particleBuffer.Get();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmdList->ResourceBarrier(1, &barrier);

    // Copy at offset from persistent upload buffer
    cmdList->CopyBufferRegion(m_particleBuffer.Get(), bufferOffset,
                               m_explosionUploadBuffer.Get(), 0, uploadSize);

    // Transition back to UAV for physics/rendering
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    cmdList->ResourceBarrier(1, &barrier);

    // DO NOT execute/wait/reset here - let the main render loop handle command list execution
    // The copy will happen as part of the current frame's command list
}

// ========== Benchmark Support ==========

ParticleSystem::PhysicsSnapshot ParticleSystem::CapturePhysicsSnapshot() const {
    PhysicsSnapshot snap = {};
    
    if (!m_particlesOnCPU || m_cpuPositions.empty() || m_cpuVelocities.empty()) {
        LOG_WARN("[Benchmark] CapturePhysicsSnapshot: No CPU particle data available");
        return snap;
    }
    
    const uint32_t count = m_activeParticleCount;
    if (count == 0) return snap;
    
    // Accumulators
    float totalKE = 0.0f;
    float totalPE = 0.0f;
    float totalL = 0.0f;
    float totalVelX = 0.0f, totalVelZ = 0.0f;  // For covariance
    float totalVelMag = 0.0f;
    float maxVelMag = 0.0f;
    float totalHeight = 0.0f;
    float totalRadius = 0.0f;
    float totalKeplerError = 0.0f;
    float totalForceMag = 0.0f;
    float totalRadialForce = 0.0f;
    uint32_t correctSignCount = 0;
    
    std::vector<float> velocities(count);
    
    for (uint32_t i = 0; i < count; i++) {
        const auto& pos = m_cpuPositions[i];
        const auto& vel = m_cpuVelocities[i];
        
        // Cylindrical radius (disk plane)
        float r_cyl = sqrtf(pos.x * pos.x + pos.z * pos.z);
        float r_3d = sqrtf(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z);
        
        // Velocity magnitude
        float v2 = vel.x * vel.x + vel.y * vel.y + vel.z * vel.z;
        float v = sqrtf(v2);
        velocities[i] = v;
        totalVelMag += v;
        maxVelMag = (std::max)(maxVelMag, v);
        
        // Kinetic energy: 0.5 * m * v² (m=1)
        totalKE += 0.5f * v2;
        
        // Potential energy: -GM/r (avoid division by zero)
        if (r_3d > 1e-6f) {
            totalPE -= PINN_GM / r_3d;
        }
        
        // Angular momentum: L = r × v (z-component for disk plane)
        // Lz = x*vz - z*vx
        totalL += pos.x * vel.z - pos.z * vel.x;
        
        // For covariance calculation
        totalVelX += vel.x;
        totalVelZ += vel.z;
        
        // Disk shape: height and radius
        totalHeight += std::abs(pos.y);
        totalRadius += r_cyl;
        
        // Keplerian velocity error
        if (r_cyl > 1e-6f) {
            float v_kepler = sqrtf(PINN_GM / r_cyl);
            float v_orbital = sqrtf(vel.x * vel.x + vel.z * vel.z);  // Horizontal velocity
            float error = std::abs(v_orbital - v_kepler) / (v_kepler + 1e-6f);
            totalKeplerError += error;
        }
        
        // Force metrics (if forces are populated)
        if (!m_cpuForces.empty() && i < m_cpuForces.size()) {
            const auto& force = m_cpuForces[i];
            float fMag = sqrtf(force.x * force.x + force.y * force.y + force.z * force.z);
            totalForceMag += fMag;
            
            // Radial force component: F · r̂
            if (r_3d > 1e-6f) {
                float f_radial = (force.x * pos.x + force.y * pos.y + force.z * pos.z) / r_3d;
                totalRadialForce += f_radial;
                if (f_radial < 0.0f) {
                    correctSignCount++;  // Gravity should be negative (attractive)
                }
            }
        }
        
        // Boundary checks
        if (r_cyl < PINN_R_ISCO) {
            snap.particlesCollapsed++;
        } else if (r_cyl > OUTER_DISK_RADIUS) {
            snap.particlesEscaped++;
        } else {
            snap.particlesInBounds++;
        }
    }
    
    // Compute averages
    float invCount = 1.0f / count;
    snap.velocityMean = totalVelMag * invCount;
    snap.velocityMax = maxVelMag;
    snap.totalKineticEnergy = totalKE;
    snap.totalPotentialEnergy = totalPE;
    snap.totalEnergy = totalKE + totalPE;
    snap.totalAngularMomentum = totalL;
    snap.keplerianVelocityError = totalKeplerError * invCount * 100.0f;  // As percentage
    snap.avgForceMagnitude = totalForceMag * invCount;
    snap.avgRadialForce = totalRadialForce * invCount;
    snap.correctRadialForceCount = correctSignCount;
    
    // Velocity standard deviation
    float meanVel = snap.velocityMean;
    float sumSqDiff = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        float diff = velocities[i] - meanVel;
        sumSqDiff += diff * diff;
    }
    snap.velocityStdDev = sqrtf(sumSqDiff * invCount);
    
    // Coherent motion: covariance of velocity components
    // High covariance = particles moving together (bad)
    float meanVelX = totalVelX * invCount;
    float meanVelZ = totalVelZ * invCount;
    float covariance = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        covariance += (m_cpuVelocities[i].x - meanVelX) * (m_cpuVelocities[i].z - meanVelZ);
    }
    snap.velocityCovarianceXZ = covariance * invCount;
    
    // Disk thickness ratio: H/R
    float avgHeight = totalHeight * invCount;
    float avgRadius = totalRadius * invCount;
    if (avgRadius > 1e-6f) {
        snap.diskThicknessRatio = avgHeight / avgRadius;
    }
    
    return snap;
}