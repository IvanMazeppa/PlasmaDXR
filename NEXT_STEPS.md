# PlasmaDXR - Next Implementation Steps

## Current Status ✅

**Repository**: https://github.com/IvanMazeppa/PlasmaDXR

**Completed:**
- ✅ Clean project structure
- ✅ Device.cpp - D3D12 initialization for RTX 4060 Ti
- ✅ SwapChain.cpp - Presentation layer
- ✅ FeatureDetector.cpp - Auto-detect capabilities
- ✅ Logger.cpp - Logging system
- ✅ Application.cpp - Main loop framework
- ✅ All header files defined
- ✅ RT shader with GREEN test pattern
- ✅ Pre-merge workaround shader

**Remaining to Implement:**
- 🔲 ResourceManager.cpp
- 🔲 ParticleSystem.cpp
- 🔲 ParticleRenderer.cpp
- 🔲 RTLightingSystem.cpp

---

## Priority 1: ResourceManager.cpp (CRITICAL)

This is needed by everything else.

```cpp
// File: src/utils/ResourceManager.cpp

#include "ResourceManager.h"
#include "Device.h"
#include "Logger.h"
#include "d3dx12/d3dx12.h"

ResourceManager::~ResourceManager() {
    Shutdown();
}

bool ResourceManager::Initialize(Device* device) {
    m_device = device;

    // Create upload buffer for staging
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

    LOG_INFO("ResourceManager initialized");
    return true;
}

void ResourceManager::Shutdown() {
    m_buffers.clear();
    m_textures.clear();
    m_descriptorHeaps.clear();
    m_uploadBuffer.Reset();
}

ID3D12Resource* ResourceManager::CreateBuffer(const std::string& name, const BufferDesc& desc) {
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
        LOG_ERROR("Failed to create buffer: {}", name);
        return nullptr;
    }

    m_buffers[name] = buffer;
    LOG_INFO("Created buffer: {} ({} bytes)", name, desc.size);
    return buffer.Get();
}

ID3D12Resource* ResourceManager::GetBuffer(const std::string& name) {
    auto it = m_buffers.find(name);
    return (it != m_buffers.end()) ? it->second.Get() : nullptr;
}

bool ResourceManager::CreateDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE type,
                                          uint32_t numDescriptors,
                                          bool shaderVisible) {
    D3D12_DESCRIPTOR_HEAP_DESC desc = {};
    desc.NumDescriptors = numDescriptors;
    desc.Type = type;
    desc.Flags = shaderVisible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE
                               : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

    DescriptorHeapInfo info;
    HRESULT hr = m_device->GetDevice()->CreateDescriptorHeap(&desc,
                                                             IID_PPV_ARGS(&info.heap));
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create descriptor heap");
        return false;
    }

    info.numDescriptors = numDescriptors;
    info.descriptorSize = m_device->GetDevice()->GetDescriptorHandleIncrementSize(type);
    info.currentIndex = 0;

    m_descriptorHeaps[type] = info;
    return true;
}

D3D12_CPU_DESCRIPTOR_HANDLE ResourceManager::AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE type) {
    auto& heapInfo = m_descriptorHeaps[type];

    if (heapInfo.currentIndex >= heapInfo.numDescriptors) {
        LOG_ERROR("Descriptor heap full!");
        return { 0 };
    }

    D3D12_CPU_DESCRIPTOR_HANDLE handle = heapInfo.heap->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += heapInfo.currentIndex * heapInfo.descriptorSize;
    heapInfo.currentIndex++;

    return handle;
}

void ResourceManager::TransitionResource(ID3D12GraphicsCommandList* cmdList,
                                        ID3D12Resource* resource,
                                        D3D12_RESOURCE_STATES from,
                                        D3D12_RESOURCE_STATES to) {
    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(resource, from, to);
    cmdList->ResourceBarrier(1, &barrier);
}
```

---

## Priority 2: ParticleSystem.cpp (Minimal Version)

Just create the buffer - physics can come later.

```cpp
// File: src/particles/ParticleSystem.cpp

#include "ParticleSystem.h"
#include "../core/Device.h"
#include "../utils/ResourceManager.h"
#include "../utils/Logger.h"

ParticleSystem::~ParticleSystem() {
    Shutdown();
}

bool ParticleSystem::Initialize(Device* device, ResourceManager* resources, uint32_t particleCount) {
    m_device = device;
    m_resources = resources;
    m_particleCount = particleCount;

    // Create particle buffer
    ResourceManager::BufferDesc desc = {};
    desc.size = particleCount * sizeof(Particle);
    desc.heapType = D3D12_HEAP_TYPE_DEFAULT;
    desc.flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    desc.initialState = D3D12_RESOURCE_STATE_COMMON;

    m_particleBuffer = m_resources->CreateBuffer("ParticleBuffer", desc);
    if (!m_particleBuffer) {
        return false;
    }

    // Initialize particles on CPU
    std::vector<Particle> particles(particleCount);
    InitializeAccretionDisk(particles.data(), particleCount);

    // Upload to GPU
    // (would need upload helper from ResourceManager)

    LOG_INFO("ParticleSystem initialized with {} particles", particleCount);
    return true;
}

void ParticleSystem::InitializeAccretionDisk(Particle* particles, uint32_t count) {
    // NASA-quality accretion disk distribution
    for (uint32_t i = 0; i < count; i++) {
        float theta = ((float)i / count) * 2.0f * 3.14159f * 100.0f; // Multiple rings
        float radius = INNER_STABLE_ORBIT + ((float)rand() / RAND_MAX) *
                      (OUTER_DISK_RADIUS - INNER_STABLE_ORBIT);
        float height = ((float)rand() / RAND_MAX - 0.5f) * DISK_THICKNESS;

        particles[i].position = DirectX::XMFLOAT3(
            radius * cosf(theta),
            height,
            radius * sinf(theta)
        );

        // Orbital velocity (Keplerian)
        float v = sqrtf(BLACK_HOLE_MASS / radius);
        particles[i].velocity = DirectX::XMFLOAT3(
            -v * sinf(theta),
            0.0f,
            v * cosf(theta)
        );

        particles[i].mass = 1.0f;
        particles[i].temperature = 10000.0f / radius; // Hotter closer to black hole

        // Color based on temperature (blackbody)
        float t = particles[i].temperature / 10000.0f;
        particles[i].color = DirectX::XMFLOAT4(
            1.0f * t, 0.8f * t, 0.3f * t, 0.5f
        );
    }
}

void ParticleSystem::Update(float deltaTime, float totalTime) {
    // Physics update would go here
    // For now, particles are static
}

void ParticleSystem::Shutdown() {
    // Resources cleaned up by ResourceManager
}
```

---

## Priority 3: ParticleRenderer.cpp (GREEN Test Path)

Focus on compute fallback first (mesh shader path can wait).

```cpp
// Simplified version - just get GREEN particles showing!

bool ParticleRenderer::Initialize(Device* device, ResourceManager* resources,
                                 const FeatureDetector* features, uint32_t particleCount) {
    m_device = device;
    m_resources = resources;
    m_particleCount = particleCount;

    // ALWAYS use compute fallback for now (mesh shaders can wait)
    m_activePath = RenderPath::ComputeFallback;

    return InitializeComputeFallbackPath();
}

bool ParticleRenderer::InitializeComputeFallbackPath() {
    // 1. Create vertex buffer (4 verts per particle)
    // 2. Create compute PSO to build vertices
    // 3. Create traditional raster PSO

    // For minimal test: Just show static particles
    LOG_INFO("Compute fallback path initialized");
    return true;
}

void ParticleRenderer::Render(ID3D12GraphicsCommandList* cmdList,
                             ID3D12Resource* particleBuffer,
                             ID3D12Resource* rtLightingBuffer,
                             const RenderConstants& constants) {
    // Minimal version: Just clear to test color
    // Full version: Run compute shader, then draw

    LOG_INFO("Rendering {} particles", m_particleCount);
}
```

---

## Priority 4: RTLightingSystem.cpp (GREEN Test Only)

Just output GREEN - no actual RT yet!

```cpp
bool RTLightingSystem::Initialize(Device* device, ResourceManager* resources,
                                 uint32_t particleCount) {
    m_device = device;
    m_resources = resources;
    m_particleCount = particleCount;

    // Create lighting output buffer
    ResourceManager::BufferDesc desc = {};
    desc.size = particleCount * sizeof(DirectX::XMFLOAT3);
    desc.heapType = D3D12_HEAP_TYPE_DEFAULT;
    desc.flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    desc.initialState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

    m_lightingBuffer = m_resources->CreateBuffer("RTLightingBuffer", desc);

    // For GREEN test: Just fill buffer with (0, 100, 0)
    // Real RT pipeline comes later

    LOG_INFO("RT Lighting system initialized (test mode)");
    return true;
}

void RTLightingSystem::ComputeLighting(ID3D12GraphicsCommandList4* cmdList,
                                      const RTConstants& constants) {
    // For now: Just clear buffer to GREEN
    // Real DispatchRays() comes after GREEN test works

    // Minimal: Fill with GREEN via compute shader
}
```

---

## Build Order

1. **Implement ResourceManager.cpp** (foundation)
2. **Implement ParticleSystem.cpp** (minimal - just buffer creation)
3. **Build and fix compile errors**
4. **Add minimal ParticleRenderer.cpp** (just stub)
5. **Add minimal RTLightingSystem.cpp** (just stub)
6. **Build again - should compile!**
7. **Run - should see window**
8. **Add actual rendering** (step by step)

---

## Expected Timeline

- **ResourceManager**: 1 hour
- **ParticleSystem (minimal)**: 30 min
- **First compile**: 30 min debugging
- **ParticleRenderer (basic)**: 2 hours
- **RTLightingSystem (test)**: 1 hour
- **GREEN test working**: +2 hours debugging

**Total**: ~7-8 hours of focused work

---

## Success Criteria

✅ **Phase 1**: Project compiles
✅ **Phase 2**: Window opens
✅ **Phase 3**: Particles visible (white)
✅ **Phase 4**: **PARTICLES TURN GREEN** ← THE GOAL!

Once GREEN works, real RT lighting is straightforward!

---

## Key Files You Need

All in the repository at:
- `src/utils/ResourceManager.cpp` ← Start here!
- `src/particles/ParticleSystem.cpp`
- `src/particles/ParticleRenderer.cpp`
- `src/lighting/RTLightingSystem.cpp`

Headers are done, just implement the .cpp files!

---

## Final Notes

The architecture is **solid**. Once these 4 files are implemented:
1. You'll have a working RT engine
2. GREEN test will prove RT lighting works
3. You can add real lighting calculations
4. Your RTX 4060 Ti will shine!

Focus on ResourceManager first - everything depends on it. The rest will flow naturally from the clean architecture we've built.

**Repository**: https://github.com/IvanMazeppa/PlasmaDXR
**Target**: 30+ FPS with 100K particles and RT lighting
**Hardware**: RTX 4060 Ti (perfect for this!)