# Luminous Star Particles - Technical Reference

**Document Version:** 1.0
**Created:** December 2025
**Status:** Reference Documentation

---

## Table of Contents

1. [Data Structures](#1-data-structures)
2. [Memory Layouts](#2-memory-layouts)
3. [GPU Synchronization Strategies](#3-gpu-synchronization-strategies)
4. [Light-Particle Mapping](#4-light-particle-mapping)
5. [Material Properties](#5-material-properties)
6. [Buffer Sizing](#6-buffer-sizing)
7. [Performance Considerations](#7-performance-considerations)

---

## 1. Data Structures

### 1.1 Light Structure (C++)

**File:** `src/particles/ParticleRenderer_Gaussian.h:24-41`

```cpp
struct Light {
    // === Base Light Properties (32 bytes) ===
    DirectX::XMFLOAT3 position;    // 12 bytes - World space position
    float intensity;               // 4 bytes  - Brightness (0.1 - 20.0)
    DirectX::XMFLOAT3 color;       // 12 bytes - RGB (0.0 - 1.0 per channel)
    float radius;                  // 4 bytes  - Soft shadow/falloff radius

    // === God Ray Parameters (32 bytes) ===
    float enableGodRays;           // 4 bytes  - 0.0 = disabled, 1.0 = enabled
    float godRayIntensity;         // 4 bytes  - Beam brightness
    float godRayLength;            // 4 bytes  - Beam distance
    float godRayFalloff;           // 4 bytes  - Radial sharpness
    DirectX::XMFLOAT3 godRayDirection;  // 12 bytes - Beam direction
    float godRayConeAngle;         // 4 bytes  - Half-angle (radians)
    float godRayRotationSpeed;     // 4 bytes  - Rotation rate
    float _padding;                // 4 bytes  - GPU alignment
};  // Total: 64 bytes (GPU cache-line aligned)
```

### 1.2 Light Structure (HLSL)

**File:** `shaders/particles/particle_gaussian_raytrace.hlsl:126-143`

```hlsl
struct Light {
    float3 position;           // 12 bytes
    float intensity;           // 4 bytes
    float3 color;              // 12 bytes
    float radius;              // 4 bytes

    float enableGodRays;       // 4 bytes
    float godRayIntensity;     // 4 bytes
    float godRayLength;        // 4 bytes
    float godRayFalloff;       // 4 bytes
    float3 godRayDirection;    // 12 bytes
    float godRayConeAngle;     // 4 bytes
    float godRayRotationSpeed; // 4 bytes
    float _padding;            // 4 bytes
};  // Total: 64 bytes
```

### 1.3 Particle Structure (C++)

**File:** `src/particles/ParticleSystem.h:59-75`

```cpp
struct Particle {
    // === LEGACY FIELDS (32 bytes) - DO NOT REORDER ===
    DirectX::XMFLOAT3 position;    // 12 bytes (offset 0)
    float temperature;             // 4 bytes  (offset 12) - Kelvin
    DirectX::XMFLOAT3 velocity;    // 12 bytes (offset 16)
    float density;                 // 4 bytes  (offset 28)

    // === MATERIAL FIELDS (16 bytes) ===
    DirectX::XMFLOAT3 albedo;      // 12 bytes (offset 32)
    uint32_t materialType;         // 4 bytes  (offset 44) - ParticleMaterialType

    // === LIFETIME FIELDS (16 bytes) ===
    float lifetime;                // 4 bytes  (offset 48)
    float maxLifetime;             // 4 bytes  (offset 52)
    float spawnTime;               // 4 bytes  (offset 56)
    uint32_t flags;                // 4 bytes  (offset 60)
};  // Total: 64 bytes (16-byte aligned)
```

### 1.4 Material Properties Structure

**File:** `src/particles/ParticleSystem.h:80-90`

```cpp
struct MaterialTypeProperties {
    DirectX::XMFLOAT3 albedo;      // 12 bytes - Base color
    float opacity;                 // 4 bytes  - 0.0 (transparent) to 1.0 (opaque)
    float emissionMultiplier;      // 4 bytes  - Glow intensity multiplier
    float scatteringCoefficient;   // 4 bytes  - Volumetric scattering
    float phaseG;                  // 4 bytes  - Henyey-Greenstein (-1 to 1)
    float expansionRate;           // 4 bytes  - Radius expansion (explosions)
    float coolingRate;             // 4 bytes  - Temperature decay
    float fadeStartRatio;          // 4 bytes  - Lifetime fade trigger
    float padding[6];              // 24 bytes - Align to 64 bytes
};  // Total: 64 bytes
```

---

## 2. Memory Layouts

### 2.1 Current Memory Layout

```
GPU Particle Buffer (N particles × 64 bytes):
┌─────────────────────────────────────────────────────────────────┐
│ Index 0    │ Index 1    │ ... │ Index N-1001 │ ... │ Index N-1 │
│ (Regular)  │ (Regular)  │     │ (Regular)    │     │ (Explode) │
└─────────────────────────────────────────────────────────────────┘
                                 └──────────────────────────────┘
                                    Explosion Pool (Last 1000)
```

### 2.2 Proposed Memory Layout (With Stars)

```
GPU Particle Buffer (N particles × 64 bytes):
┌─────────────────────────────────────────────────────────────────┐
│ 0-15       │ 16         │ ... │ N-1001       │ ... │ N-1       │
│ STAR POOL  │ (Regular)  │     │ (Regular)    │     │ (Explode) │
│ (16 stars) │            │     │              │     │           │
└─────────────────────────────────────────────────────────────────┘
└────────────┘                                 └──────────────────┘
  Star Pool                                      Explosion Pool
  (First 16)                                     (Last 1000)
```

### 2.3 Light Buffer Layout

**Current (16 lights):**
```
Light Buffer (1024 bytes = 16 × 64 bytes):
┌────────┬────────┬────────┬─────┬─────────┐
│ Light 0│ Light 1│ Light 2│ ... │ Light 15│
│ 64B    │ 64B    │ 64B    │     │ 64B     │
└────────┴────────┴────────┴─────┴─────────┘
```

**Proposed (32 lights):**
```
Light Buffer (2048 bytes = 32 × 64 bytes):
┌────────┬────────┬─────┬──────────┬──────────┬─────┬──────────┐
│Light 0 │Light 1 │ ... │ Light 15 │ Light 16 │ ... │ Light 31 │
│ STAR   │ STAR   │     │ STAR     │ STATIC   │     │ STATIC   │
└────────┴────────┴─────┴──────────┴──────────┴─────┴──────────┘
└──────────────────────────────────┘└──────────────────────────┘
        Star Lights (0-15)              Static Lights (16-31)
```

### 2.4 Material Properties Buffer Layout

**Previous (8 materials, 512 bytes):**
```
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ PLASMA   │ STAR_MS  │ GAS_CLOUD│ ROCKY    │ ICY      │ SUPERNOVA│ FLARE    │ SHOCK    │
│ (0)      │ (1)      │ (2)      │ (3)      │ (4)      │ (5)      │ (6)      │ (7)      │
│ 64 bytes │ 64 bytes │ 64 bytes │ 64 bytes │ 64 bytes │ 64 bytes │ 64 bytes │ 64 bytes │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

**Current (9 materials, 576 bytes) - IMPLEMENTED:**
```
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬───────────┐
│ PLASMA   │ STAR_MS  │ GAS_CLOUD│ ROCKY    │ ICY      │ SUPERNOVA│ FLARE    │ SHOCK    │ SUPERGIANT│
│ (0)      │ (1)      │ (2)      │ (3)      │ (4)      │ (5)      │ (6)      │ (7)      │ (8) ✅    │
│ 64 bytes │ 64 bytes │ 64 bytes │ 64 bytes │ 64 bytes │ 64 bytes │ 64 bytes │ 64 bytes │ 64 bytes  │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴───────────┘
```

---

## 3. GPU Synchronization Strategies

### 3.1 The Challenge

Star light positions must be updated from particle positions each frame. This requires reading GPU particle data to update CPU-side light array.

**Data Flow:**
```
GPU Particle Buffer → (SYNC) → CPU Light Array → GPU Light Buffer
     (physics)                    (positions)        (rendering)
```

### 3.2 Option A: Synchronous GPU Readback

**How it works:**
1. After physics update, copy particle buffer region to readback buffer
2. Map readback buffer, read positions
3. Update light positions
4. Continue with rendering

**Implementation:**
```cpp
// Create readback buffer (one-time)
D3D12_HEAP_PROPERTIES heapProps = {};
heapProps.Type = D3D12_HEAP_TYPE_READBACK;

D3D12_RESOURCE_DESC bufferDesc = {};
bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
bufferDesc.Width = STAR_PARTICLE_COUNT * sizeof(Particle);  // 1024 bytes for 16 stars

device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE,
    &bufferDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
    IID_PPV_ARGS(&m_readbackBuffer));

// Per-frame readback
cmdList->CopyBufferRegion(m_readbackBuffer.Get(), 0,
    m_particleBuffer.Get(), 0,
    STAR_PARTICLE_COUNT * sizeof(Particle));

// Execute and wait (BLOCKING)
cmdQueue->ExecuteCommandLists(1, &cmdList);
fence->Signal(++fenceValue);
fence->SetEventOnCompletion(fenceValue, fenceEvent);
WaitForSingleObject(fenceEvent, INFINITE);

// Read positions
Particle* particles;
m_readbackBuffer->Map(0, nullptr, (void**)&particles);
for (uint32_t i = 0; i < STAR_PARTICLE_COUNT; i++) {
    m_lights[i].position = particles[i].position;
}
m_readbackBuffer->Unmap(0, nullptr);
```

**Performance:**
- Latency: ~0.5-2ms (GPU stall)
- Accuracy: 100% (exact positions)
- Simplicity: Medium

**When to use:** Prototyping, when accuracy is critical

---

### 3.3 Option B: Asynchronous Readback (Double Buffered)

**How it works:**
1. Request readback for frame N
2. Use positions from frame N-2 (2-frame latency)
3. No GPU stalls

**Implementation:**
```cpp
// Double buffered readback
ID3D12Resource* m_readbackBuffers[2];
uint32_t m_readbackIndex = 0;

// Per-frame (non-blocking)
void UpdateStarLights() {
    // Copy current frame to readback buffer
    cmdList->CopyBufferRegion(
        m_readbackBuffers[m_readbackIndex].Get(), 0,
        m_particleBuffer.Get(), 0,
        STAR_PARTICLE_COUNT * sizeof(Particle));

    // Use positions from 2 frames ago
    uint32_t readIndex = (m_readbackIndex + 1) % 2;  // Other buffer

    D3D12_RANGE readRange = { 0, STAR_PARTICLE_COUNT * sizeof(Particle) };
    Particle* particles;
    m_readbackBuffers[readIndex]->Map(0, &readRange, (void**)&particles);
    for (uint32_t i = 0; i < STAR_PARTICLE_COUNT; i++) {
        m_lights[i].position = particles[i].position;
    }
    m_readbackBuffers[readIndex]->Unmap(0, nullptr);

    // Swap buffers
    m_readbackIndex = readIndex;
}
```

**Performance:**
- Latency: ~0.1ms (no stall)
- Accuracy: 2-frame delay (imperceptible at 120 FPS)
- Simplicity: Medium-High

**When to use:** Production, when performance matters

---

### 3.4 Option C: CPU-Side Position Prediction

**How it works:**
1. Track star particle positions and velocities on CPU
2. Integrate positions using same physics (Verlet)
3. Never read from GPU

**Implementation:**
```cpp
// CPU-side tracking
std::vector<DirectX::XMFLOAT3> m_starPositions;
std::vector<DirectX::XMFLOAT3> m_starVelocities;

void UpdateStarLights(float deltaTime) {
    const float GM = 100.0f;  // Must match GPU physics

    for (uint32_t i = 0; i < STAR_PARTICLE_COUNT; i++) {
        DirectX::XMFLOAT3& pos = m_starPositions[i];
        DirectX::XMFLOAT3& vel = m_starVelocities[i];

        // Calculate gravitational acceleration
        float r = sqrtf(pos.x*pos.x + pos.y*pos.y + pos.z*pos.z);
        float accel = -GM / (r * r);
        DirectX::XMFLOAT3 dir = { -pos.x/r, -pos.y/r, -pos.z/r };

        // Velocity Verlet integration
        pos.x += vel.x * deltaTime + 0.5f * dir.x * accel * deltaTime * deltaTime;
        pos.y += vel.y * deltaTime + 0.5f * dir.y * accel * deltaTime * deltaTime;
        pos.z += vel.z * deltaTime + 0.5f * dir.z * accel * deltaTime * deltaTime;

        vel.x += dir.x * accel * deltaTime;
        vel.y += dir.y * accel * deltaTime;
        vel.z += dir.z * accel * deltaTime;

        m_lights[i].position = pos;
    }
}
```

**Performance:**
- Latency: 0ms (no GPU interaction)
- Accuracy: ~95-99% (drift possible over time)
- Simplicity: High

**When to use:** Performance-critical, when slight drift is acceptable

---

### 3.5 Option D: GPU-Only Update (Compute Shader)

**How it works:**
1. Compute shader reads particle positions
2. Writes directly to light buffer
3. Zero CPU involvement

**Implementation:**
```hlsl
// update_star_lights.hlsl
RWStructuredBuffer<Light> g_lights : register(u0);
StructuredBuffer<Particle> g_particles : register(t0);

[numthreads(16, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= 16) return;  // 16 star particles

    Particle p = g_particles[id.x];

    g_lights[id.x].position = p.position;
    // Could also update color based on temperature:
    // g_lights[id.x].color = TemperatureToColor(p.temperature);
}
```

**Performance:**
- Latency: 0ms CPU, <0.1ms GPU
- Accuracy: 100%
- Simplicity: Low (requires new shader pipeline)

**When to use:** Maximum performance, production quality

---

### 3.6 Strategy Comparison

| Strategy | CPU Cost | GPU Cost | Accuracy | Complexity | Recommended For |
|----------|----------|----------|----------|------------|-----------------|
| A: Sync Readback | ~1ms stall | Copy op | 100% | Medium | Prototyping |
| B: Async Readback | ~0.1ms | Copy op | 2-frame delay | Medium-High | Production |
| C: CPU Prediction | ~0.01ms | None | ~95% | Low | Quick test |
| D: GPU Compute | None | ~0.05ms | 100% | High | Best quality |

**Recommendation for MVP:** Start with Option C (CPU Prediction), upgrade to Option D later.

---

## 4. Light-Particle Mapping

### 4.1 Simple 1:1 Mapping

Star particle index directly maps to light index:

```
Particle Index → Light Index
      0        →     0
      1        →     1
      ...      →    ...
      15       →     15
```

**Code:**
```cpp
for (uint32_t i = 0; i < STAR_PARTICLE_COUNT; i++) {
    m_lights[i].position = GetParticlePosition(i);
}
```

### 4.2 Index Table Mapping

Flexible mapping via lookup table:

```cpp
std::vector<uint32_t> m_starParticleIndices;  // e.g., [5, 12, 47, 103, ...]
std::vector<uint32_t> m_starLightIndices;     // e.g., [0, 1, 2, 3, ...]

for (size_t i = 0; i < m_starParticleIndices.size(); i++) {
    uint32_t particleIdx = m_starParticleIndices[i];
    uint32_t lightIdx = m_starLightIndices[i];
    m_lights[lightIdx].position = GetParticlePosition(particleIdx);
}
```

**Advantage:** Stars don't need to be first N particles.

---

## 5. Material Properties

### 5.1 SUPERGIANT_STAR Material (Index 8) - IMPLEMENTED

| Property | Value | Rationale |
|----------|-------|-----------|
| albedo | **(0.85, 0.9, 1.0)** | Blue-white (25000K+ supergiant color) |
| opacity | **0.15** | Very transparent - light shines through particle |
| emissionMultiplier | **15.0** | Highest emission (matches SUPERNOVA brightness) |
| scatteringCoefficient | **0.3** | Low scattering (self-luminous core, not diffuse) |
| phaseG | 0.0 | Isotropic (glow equally in all directions) |
| expansionRate | 0.0 | No expansion (stable star) |
| coolingRate | 0.0 | No cooling (maintains temperature) |
| fadeStartRatio | 1.0 | Never fades |

**Implementation Notes:**
- Albedo changed from warm white to blue-white to match 25000K+ supergiant temperature
- Scattering reduced from 0.8 to 0.3 - supergiants are self-luminous, not diffuse scatterers
- Low opacity (0.15) allows embedded light source to "shine through" the particle

### 5.2 Material Comparison (Updated with Implementation)

| Material | Emission | Opacity | Scattering | Albedo | Use Case |
|----------|----------|---------|------------|--------|----------|
| PLASMA | 2.5× | 1.0 | 2.5 | Orange-red | Accretion disk particles |
| STAR_MAIN_SEQUENCE | 8.0× | 0.9 | 0.5 | White-yellow | Regular stars |
| **SUPERGIANT_STAR** | **15.0×** | **0.15** | **0.3** | **Blue-white** | **Luminous star particles** |
| SUPERNOVA | 15.0× | 0.95 | 1.5 | White-yellow | Explosion events |

### 5.3 Temperature-Color Relationship

Using Wien's law approximation for stellar colors:

| Temperature | Color RGB | Star Type |
|-------------|-----------|-----------|
| 3000K | (1.0, 0.5, 0.2) | M-class (Red dwarf) |
| 4000K | (1.0, 0.7, 0.4) | K-class (Orange) |
| 6000K | (1.0, 0.95, 0.85) | G-class (Sun-like) |
| 8000K | (0.9, 0.95, 1.0) | A-class (White) |
| 15000K | (0.7, 0.85, 1.0) | B-class (Blue-white) |
| 25000K | (0.6, 0.8, 1.0) | O-class (Blue supergiant) |

**Implementation:**
```cpp
DirectX::XMFLOAT3 TemperatureToColor(float kelvin) {
    // Wien's law approximation
    float t = kelvin / 1000.0f;
    float r, g, b;

    if (t <= 6.6f) {
        r = 1.0f;
        g = 0.39f * logf(t) - 0.63f;
        b = 0.12f * logf(t - 1.0f);
    } else {
        r = 1.29f * powf(t - 6.0f, -0.133f);
        g = 1.13f * powf(t - 6.0f, -0.075f);
        b = 1.0f;
    }

    return { saturate(r), saturate(g), saturate(b) };
}
```

---

## 6. Buffer Sizing

### 6.1 Light Buffer

| Configuration | Buffer Size | Memory |
|---------------|-------------|--------|
| Current (16 lights) | 16 × 64 bytes | 1,024 bytes |
| Proposed (32 lights) | 32 × 64 bytes | 2,048 bytes |
| **Increase** | | **+1,024 bytes** |

### 6.2 Material Properties Buffer

| Configuration | Buffer Size | Memory |
|---------------|-------------|--------|
| Current (8 materials) | 8 × 64 bytes | 512 bytes |
| Proposed (9 materials) | 9 × 64 bytes | 576 bytes |
| **Increase** | | **+64 bytes** |

### 6.3 Particle Buffer (No Change)

Star particles use existing particle slots (indices 0-15), no buffer resize needed.

| Configuration | Buffer Size | Memory |
|---------------|-------------|--------|
| 10K particles | 10,000 × 64 bytes | 640 KB |
| 30K particles | 30,000 × 64 bytes | 1.92 MB |
| **Change** | | **None** |

---

## 7. Performance Considerations

### 7.1 Expected Overhead

| Component | Overhead | Notes |
|-----------|----------|-------|
| Light buffer upload | +64 bytes/frame | 1KB → 2KB |
| Shader light loop | +16 iterations | 16 → 32 lights |
| Position sync | +0.1-1.0ms | Depends on strategy |
| Material lookup | None | Same shader code |
| **Total** | **<1ms** | |

### 7.2 Shader Loop Impact

Current shader light loop:
```hlsl
for (uint i = 0; i < lightCount; i++) {  // lightCount = 13
    Light light = g_lights[i];
    // ... lighting calculation
}
```

With 32 lights:
- Loop iterations: 13 → 29 (13 static + 16 star)
- Iteration cost: ~0.01ms each
- Additional cost: ~0.16ms total

### 7.3 Optimization Opportunities

1. **Light culling:** Skip lights too far from current pixel
2. **LOD system:** Reduce light count at distance
3. **Spatial hashing:** Use grid to find nearby lights only
4. **GPU-driven:** Move all sync to GPU (Option D)

---

## Related Documents

- `FEATURE_OVERVIEW.md` - High-level concept
- `ARCHITECTURE_OPTIONS.md` - Implementation approaches
- `IMPLEMENTATION_GUIDE.md` - Step-by-step instructions
- `SHADER_MODIFICATIONS.md` - HLSL changes

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | Dec 2025 | Claude Code | Initial document |
| 1.1 | Dec 2025 | Claude Code | Updated SUPERGIANT_STAR material with actual implemented values (blue-white albedo, 0.3 scattering) |
