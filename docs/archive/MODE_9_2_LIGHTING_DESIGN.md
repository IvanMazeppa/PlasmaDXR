# Mode 9.2: Advanced Lighting for NASA-Quality Accretion Disk
**Implementation Guide for 100K Mesh Shader Particle System**

---

## Executive Summary

Mode 9.2 builds upon Mode 9.1's shadow mapping to create physically-inspired lighting for the accretion disk simulation. The design prioritizes **visual impact over strict physical accuracy** while maintaining 60fps on RTX 4060 Ti.

**Key Features:**
1. Particle emission - Hot particles (>15000K) emit light affecting nearby particles
2. Dynamic point lights - Hottest core particles act as light sources
3. Screen-space volumetric scattering - Density-based light absorption
4. Bloom/glow post-process - Hot core regions radiate visually
5. Optional: Doppler shift and gravitational lensing effects

**Performance Target:** 60fps @ 1920x1080 on RTX 4060 Ti (12ms frame budget)

---

## 1. Recommended Lighting Model

### Architecture: Hybrid Screen-Space + Clustered Lighting

**Why This Approach:**
- Screen-space effects leverage existing mesh shader output
- Clustered lighting only for brightest particles (top 1-5%)
- Deferred lighting on particle G-buffer avoids overdraw
- RayQuery for high-quality occlusion queries (already working in Mode 9.1)

### Three-Tier System:

#### Tier 1: Emission Pass (Most Important)
**What:** Particles with T > 15000K emit light stored in emission buffer
**Where:** Extend particle pixel shader (particle_pixel.hlsl)
**Cost:** ~0.5ms (additive blending to existing pass)

#### Tier 2: Light Propagation (Core Feature)
**What:** Compute shader propagates light through particle density field
**Where:** New shader: shaders/mode9/light_propagation_cs.hlsl
**Cost:** ~2-3ms (single compute dispatch)

#### Tier 3: Volumetric Scattering (Visual Polish)
**What:** Screen-space ray march through particle density
**Where:** New shader: shaders/mode9/volumetric_scatter_cs.hlsl
**Cost:** ~3-4ms (half-res + temporal reprojection)

---

## 2. Step-by-Step Implementation Plan

### Milestone 1: Emission Buffer System (Week 1)
**Goal:** Particles emit light based on temperature

**Tasks:**
1. Add emission render target (R11G11B10_FLOAT format, same res as backbuffer)
2. Modify particle pixel shader to output emission for hot particles
3. Visualize emission buffer in debug mode (Key: E toggles emission view)

**Shader Changes:**
```hlsl
// In particle_pixel.hlsl, add second render target output

struct PSOutput {
    float4 color : SV_Target0;      // Existing particle color
    float4 emission : SV_Target1;   // NEW: Emission intensity
};

PSOutput PSMain(VertexOutput input) {
    PSOutput output;

    // ... existing particle rendering code ...
    output.color = float4(color, alpha);

    // Calculate emission based on temperature
    // Only emit if temperature > 15000K
    float emissionStrength = 0.0;
    if (input.temperature > 15000.0) {
        // Exponential falloff for emission intensity
        float normalizedTemp = saturate((input.temperature - 15000.0) / 11000.0);
        emissionStrength = pow(normalizedTemp, 2.0) * 5.0; // Scale factor
    }

    // Emission color matches particle temperature
    output.emission = float4(input.color * emissionStrength, emissionStrength);

    return output;
}
```

**CPU Changes:**
```cpp
// In App.cpp, create emission texture
bool App::createEmissionTexture() {
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = m_width;
    desc.Height = m_height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R11G11B10_FLOAT; // HDR format
    desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

    // ... create resource and RTV ...
}
```

**Validation:**
- Press E to toggle emission-only view
- Hottest core particles should glow intensely (white/blue-white)
- Mid-temperature particles emit less (yellow/orange)
- Cool particles (red) should not emit

**Performance Estimate:** +0.5ms (minimal overhead, same pixel shader pass)

---

### Milestone 2: Spatial Light Grid (Week 2)
**Goal:** Accelerate light-particle queries using 3D grid

**Approach:** Clustered lighting with 32x32x32 grid (32768 cells)

**Shader:** shaders/mode9/light_grid_build_cs.hlsl
```hlsl
// Build spatial grid of light-emitting particles

#define GRID_SIZE 32
#define MAX_LIGHTS_PER_CELL 32

struct LightParticle {
    float3 position;
    float intensity;
    float3 color;
    float radius;
};

// Output: Grid of light indices
RWStructuredBuffer<uint> g_lightGrid : register(u0);  // [GRID_SIZE^3 * MAX_LIGHTS_PER_CELL]
RWStructuredBuffer<uint> g_lightCount : register(u1); // [GRID_SIZE^3]

// Input: All particles
StructuredBuffer<Particle> g_particles : register(t0);

cbuffer GridParams : register(b0) {
    float3 g_gridMin;      // World space bounds (-100, -100, -100)
    float g_cellSize;      // 200.0 / 32 = 6.25 units per cell
    float3 g_gridMax;      // (100, 100, 100)
    uint g_particleCount;
    float g_emissionThreshold; // 15000K
};

// Hash position to grid cell
uint3 WorldToGrid(float3 worldPos) {
    float3 normalized = (worldPos - g_gridMin) / (g_gridMax - g_gridMin);
    return clamp(uint3(normalized * GRID_SIZE), uint3(0,0,0), uint3(GRID_SIZE-1, GRID_SIZE-1, GRID_SIZE-1));
}

uint GridCellIndex(uint3 gridPos) {
    return gridPos.x + gridPos.y * GRID_SIZE + gridPos.z * GRID_SIZE * GRID_SIZE;
}

[numthreads(64, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    uint particleIdx = id.x;
    if (particleIdx >= g_particleCount) return;

    Particle p = g_particles[particleIdx];

    // Only add hot particles to light grid
    if (p.temperature < g_emissionThreshold) return;

    // Calculate emission intensity
    float normalizedTemp = saturate((p.temperature - 15000.0) / 11000.0);
    float intensity = pow(normalizedTemp, 2.0) * 5.0;

    if (intensity < 0.1) return; // Cull weak lights

    // Insert into grid cell
    uint3 gridPos = WorldToGrid(p.position);
    uint cellIdx = GridCellIndex(gridPos);

    // Atomic add to get insertion slot
    uint insertIdx;
    InterlockedAdd(g_lightCount[cellIdx], 1, insertIdx);

    // Store light index if there's space
    if (insertIdx < MAX_LIGHTS_PER_CELL) {
        uint writeIdx = cellIdx * MAX_LIGHTS_PER_CELL + insertIdx;
        g_lightGrid[writeIdx] = particleIdx;
    }
}
```

**CPU Setup:**
```cpp
// In App.h
Microsoft::WRL::ComPtr<ID3D12Resource> m_lightGridBuffer;
Microsoft::WRL::ComPtr<ID3D12Resource> m_lightCountBuffer;
UINT m_lightGridSrvIndex = UINT_MAX;

// In App.cpp
bool App::createLightGrid() {
    const uint32_t GRID_SIZE = 32;
    const uint32_t MAX_LIGHTS_PER_CELL = 32;
    const uint32_t totalCells = GRID_SIZE * GRID_SIZE * GRID_SIZE; // 32768

    // Light grid buffer: indices of particles per cell
    CD3DX12_RESOURCE_DESC gridDesc = CD3DX12_RESOURCE_DESC::Buffer(
        totalCells * MAX_LIGHTS_PER_CELL * sizeof(uint32_t),
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
    );

    // Light count buffer: # lights per cell
    CD3DX12_RESOURCE_DESC countDesc = CD3DX12_RESOURCE_DESC::Buffer(
        totalCells * sizeof(uint32_t),
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
    );

    // ... create resources ...
}
```

**Validation:**
- Debug visualization: Color cells by light count (blue=0, red=32+)
- Hottest regions (disk core) should have most lights
- Outer disk should be sparse

**Performance Estimate:** ~0.5ms (single compute pass, 100K particles / 64 threads = 1563 thread groups)

---

### Milestone 3: Light Accumulation Pass (Week 2-3)
**Goal:** Each particle samples nearby lights from grid

**Shader:** shaders/mode9/light_accumulation_cs.hlsl
```hlsl
// Accumulate lighting on each particle from nearby emitters

StructuredBuffer<Particle> g_particles : register(t0);
StructuredBuffer<uint> g_lightGrid : register(t1);
StructuredBuffer<uint> g_lightCount : register(t2);

RWStructuredBuffer<float3> g_particleLighting : register(u0); // Output: RGB lighting per particle

cbuffer LightParams : register(b0) {
    float3 g_gridMin;
    float g_cellSize;
    float3 g_gridMax;
    uint g_particleCount;
    float g_lightFalloff;    // Default: 20.0 units
    float g_scatterStrength; // Default: 0.5
};

// Sample lights in 3x3x3 neighborhood (27 cells)
float3 SampleNearbyLights(float3 worldPos, uint particleIdx) {
    uint3 centerCell = WorldToGrid(worldPos);
    float3 totalLight = float3(0, 0, 0);

    // Query 3x3x3 neighborhood
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int3 cellOffset = int3(x, y, z);
                int3 cellPos = int3(centerCell) + cellOffset;

                // Bounds check
                if (any(cellPos < 0) || any(cellPos >= GRID_SIZE)) continue;

                uint cellIdx = GridCellIndex(uint3(cellPos));
                uint lightCount = g_lightCount[cellIdx];

                // Sample lights in this cell
                for (uint i = 0; i < min(lightCount, MAX_LIGHTS_PER_CELL); i++) {
                    uint lightIdx = g_lightGrid[cellIdx * MAX_LIGHTS_PER_CELL + i];

                    // Don't light yourself
                    if (lightIdx == particleIdx) continue;

                    Particle light = g_particles[lightIdx];

                    // Calculate lighting contribution
                    float3 toLight = light.position - worldPos;
                    float dist = length(toLight);

                    if (dist < 0.01) continue; // Avoid singularity

                    // Inverse square falloff with smooth cutoff
                    float attenuation = 1.0 / (1.0 + dist * dist / (g_lightFalloff * g_lightFalloff));
                    attenuation *= smoothstep(g_lightFalloff * 2.0, 0.0, dist);

                    // Light intensity based on temperature
                    float normalizedTemp = saturate((light.temperature - 15000.0) / 11000.0);
                    float intensity = pow(normalizedTemp, 2.0) * 5.0;

                    // Light color from temperature
                    float3 lightColor = TemperatureToColor(light.temperature);

                    // Accumulate contribution
                    totalLight += lightColor * intensity * attenuation;
                }
            }
        }
    }

    return totalLight * g_scatterStrength;
}

[numthreads(64, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    uint particleIdx = id.x;
    if (particleIdx >= g_particleCount) return;

    Particle p = g_particles[particleIdx];
    float3 lighting = SampleNearbyLights(p.position, particleIdx);

    g_particleLighting[particleIdx] = lighting;
}
```

**Integration with Mesh Shader:**
```hlsl
// In particle_mesh.hlsl, add lighting buffer
StructuredBuffer<float3> particleLighting : register(t2); // NEW

// In mesh shader main:
float3 accumulatedLight = particleLighting[particleIndex];

// In pixel shader:
struct VertexOutput {
    // ... existing fields ...
    float3 accumulatedLight : COLOR2; // NEW
};

// In PSMain:
float3 finalColor = color + input.accumulatedLight * 0.5; // Blend in lighting
```

**Validation:**
- Particles near hot core should be brighter on edges facing core
- Create visual "glow halos" around brightest regions
- Adjust g_scatterStrength (0.1-2.0) for artistic control

**Performance Estimate:** ~2-3ms (depends on light density, 27 cell queries per particle)

---

### Milestone 4: Screen-Space Volumetric Scattering (Week 3-4)
**Goal:** Fog/glow effects for atmospheric depth

**Approach:** Half-resolution ray march with temporal reprojection

**Shader:** shaders/mode9/volumetric_scatter_cs.hlsl
```hlsl
// Screen-space volumetric scattering through particle density

// Inputs
Texture2D<float> g_depthBuffer : register(t0);        // Particle depth
Texture2D<float4> g_emissionBuffer : register(t1);    // Hot particle emission
Texture2D<float4> g_colorBuffer : register(t2);       // Particle colors
Texture2D<float4> g_historyBuffer : register(t3);     // Previous frame (temporal)

// Outputs
RWTexture2D<float4> g_scatterOutput : register(u0);

cbuffer ScatterParams : register(b0) {
    float4x4 g_invViewProj;
    float3 g_cameraPos;
    float g_scatterStrength;
    float g_extinctionCoeff;  // Beer's law absorption
    float g_anisotropy;       // Henyey-Greenstein g parameter (-1 to 1)
    float3 g_lightDir;        // Directional light for scattering
    float g_lightIntensity;
};

// Henyey-Greenstein phase function
float HenyeyGreenstein(float cosTheta, float g) {
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cosTheta;
    return (1.0 / (4.0 * 3.14159)) * (1.0 - g2) / (denom * sqrt(denom));
}

// March ray through particle density field
float4 VolumetricScatter(float2 uv, float3 rayOrigin, float3 rayDir, float maxDist) {
    const int SAMPLES = 16; // Half-res = 960x540, 16 samples = acceptable
    float stepSize = maxDist / float(SAMPLES);

    float3 scatteredLight = float3(0, 0, 0);
    float transmittance = 1.0;

    float3 currentPos = rayOrigin;

    for (int i = 0; i < SAMPLES; i++) {
        currentPos += rayDir * stepSize;

        // Project to screen space to sample emission
        float4 clipPos = mul(float4(currentPos, 1.0), g_viewProj);
        float2 sampleUV = (clipPos.xy / clipPos.w) * 0.5 + 0.5;
        sampleUV.y = 1.0 - sampleUV.y;

        if (any(sampleUV < 0) || any(sampleUV > 1)) continue;

        // Sample emission at this point
        float4 emission = g_emissionBuffer.SampleLevel(sampler_linear, sampleUV, 0);
        float density = emission.a; // Alpha = density

        if (density < 0.01) continue;

        // Calculate scattering contribution
        float cosTheta = dot(rayDir, g_lightDir);
        float phase = HenyeyGreenstein(cosTheta, g_anisotropy);

        // In-scattering
        float3 inscatter = emission.rgb * phase * density * g_scatterStrength;

        // Extinction (Beer's law)
        float extinction = exp(-density * g_extinctionCoeff * stepSize);

        // Accumulate
        scatteredLight += inscatter * transmittance;
        transmittance *= extinction;

        // Early exit if fully opaque
        if (transmittance < 0.01) break;
    }

    return float4(scatteredLight, 1.0 - transmittance);
}

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    // Half-res: 960x540 for 1920x1080
    uint2 pixelCoord = id.xy;
    float2 uv = (float2(pixelCoord) + 0.5) / float2(960, 540);

    // Reconstruct ray from screen UV
    float depth = g_depthBuffer.SampleLevel(sampler_point, uv, 0);
    float3 rayDir = normalize(UnprojectUV(uv, g_invViewProj));
    float3 rayOrigin = g_cameraPos;

    // March through volume
    float maxDist = 200.0; // Scene bounds
    float4 scatter = VolumetricScatter(uv, rayOrigin, rayDir, maxDist);

    // Temporal blend (60% history, 40% current - reduces noise)
    float4 history = g_historyBuffer[pixelCoord];
    float4 blended = lerp(scatter, history, 0.6);

    g_scatterOutput[pixelCoord] = blended;
}
```

**CPU Integration:**
```cpp
// Render order:
// 1. Render particles to color + emission + depth
// 2. Build light grid
// 3. Compute light accumulation
// 4. Render particles again with accumulated lighting
// 5. Volumetric scatter pass (half-res)
// 6. Upsample and composite volumetric with scene
// 7. Bloom pass (optional)
```

**Validation:**
- Bright core should have visible fog/glow extending outward
- Anisotropy = 0.8 (forward scattering, typical for dust)
- Anisotropy = 0.0 (isotropic, uniform glow)
- Anisotropy = -0.5 (back-scattering, halo effect)

**Performance Estimate:** ~3-4ms (half-res + 16 samples + temporal = reasonable)

---

### Milestone 5: Bloom Post-Process (Week 4)
**Goal:** Glow/radiance for hottest particles

**Approach:** Dual Kawase blur (efficient, good quality)

**Shader:** shaders/mode9/bloom_downsample_cs.hlsl + bloom_upsample_cs.hlsl
```hlsl
// Kawase Blur Downsample
Texture2D<float4> g_input : register(t0);
RWTexture2D<float4> g_output : register(u0);

cbuffer BloomParams : register(b0) {
    float2 g_texelSize;
    float g_threshold;     // 0.5 = only bright particles bloom
    float g_intensity;     // 1.0 = default bloom strength
};

[numthreads(8, 8, 1)]
void DownsampleCS(uint3 id : SV_DispatchThreadID) {
    float2 uv = (float2(id.xy) + 0.5) * g_texelSize;

    // 4-tap downsample with threshold
    float4 color = 0;
    color += g_input.SampleLevel(sampler_linear, uv + float2(-1, -1) * g_texelSize, 0);
    color += g_input.SampleLevel(sampler_linear, uv + float2( 1, -1) * g_texelSize, 0);
    color += g_input.SampleLevel(sampler_linear, uv + float2(-1,  1) * g_texelSize, 0);
    color += g_input.SampleLevel(sampler_linear, uv + float2( 1,  1) * g_texelSize, 0);
    color *= 0.25;

    // Luminance threshold
    float luminance = dot(color.rgb, float3(0.299, 0.587, 0.114));
    float bloomContribution = smoothstep(g_threshold, g_threshold + 0.5, luminance);

    g_output[id.xy] = color * bloomContribution;
}

// Upsample and accumulate
[numthreads(8, 8, 1)]
void UpsampleCS(uint3 id : SV_DispatchThreadID) {
    float2 uv = (float2(id.xy) + 0.5) * g_texelSize;

    // Tent filter upsample (9-tap)
    float4 color = 0;
    color += g_input.SampleLevel(sampler_linear, uv + float2(-1, -1) * g_texelSize, 0) * 1;
    color += g_input.SampleLevel(sampler_linear, uv + float2( 0, -1) * g_texelSize, 0) * 2;
    color += g_input.SampleLevel(sampler_linear, uv + float2( 1, -1) * g_texelSize, 0) * 1;
    color += g_input.SampleLevel(sampler_linear, uv + float2(-1,  0) * g_texelSize, 0) * 2;
    color += g_input.SampleLevel(sampler_linear, uv + float2( 0,  0) * g_texelSize, 0) * 4;
    color += g_input.SampleLevel(sampler_linear, uv + float2( 1,  0) * g_texelSize, 0) * 2;
    color += g_input.SampleLevel(sampler_linear, uv + float2(-1,  1) * g_texelSize, 0) * 1;
    color += g_input.SampleLevel(sampler_linear, uv + float2( 0,  1) * g_texelSize, 0) * 2;
    color += g_input.SampleLevel(sampler_linear, uv + float2( 1,  1) * g_texelSize, 0) * 1;
    color /= 16.0;

    g_output[id.xy] = color * g_intensity;
}
```

**Bloom Pipeline:**
1. Downsample emission buffer: 1920x1080 -> 960x540 -> 480x270 -> 240x135 -> 120x67
2. Upsample with accumulation: 120x67 -> 240x135 -> 480x270 -> 960x540 -> 1920x1080
3. Additive blend with final frame

**Validation:**
- Hot core glows intensely
- Adjust threshold (0.3-0.8) to control which particles bloom
- Adjust intensity (0.5-2.0) for bloom strength

**Performance Estimate:** ~1.5ms (5 downsamples + 4 upsamples at reduced res)

---

## 3. Runtime Controls

### Keyboard Mappings (Add to App::onKeyDown)

```cpp
// Mode 9.2 lighting controls
case 'U': // Light scattering strength
    m_lightScatterStrength += (shift ? -0.1f : 0.1f);
    m_lightScatterStrength = std::clamp(m_lightScatterStrength, 0.0f, 5.0f);
    break;

case 'I': // Light falloff radius
    m_lightFalloff += (shift ? -2.0f : 2.0f);
    m_lightFalloff = std::clamp(m_lightFalloff, 5.0f, 50.0f);
    break;

case 'O': // Henyey-Greenstein anisotropy g
    m_hgAnisotropy += (shift ? -0.1f : 0.1f);
    m_hgAnisotropy = std::clamp(m_hgAnisotropy, -0.9f, 0.9f);
    break;

case 'P': // Bloom threshold
    m_bloomThreshold += (shift ? -0.1f : 0.1f);
    m_bloomThreshold = std::clamp(m_bloomThreshold, 0.0f, 1.0f);
    break;

case '[': // Bloom intensity
    m_bloomIntensity += (shift ? -0.1f : 0.1f);
    m_bloomIntensity = std::clamp(m_bloomIntensity, 0.0f, 5.0f);
    break;

case ']': // Volumetric extinction
    m_extinctionCoeff += (shift ? -0.01f : 0.01f);
    m_extinctionCoeff = std::clamp(m_extinctionCoeff, 0.0f, 0.5f);
    break;

case 'E': // Toggle emission buffer view (debug)
    m_debugEmissionView = !m_debugEmissionView;
    break;

case 'L': // Toggle lighting system on/off
    m_mode9LightingEnabled = !m_mode9LightingEnabled;
    break;
```

### Default Values (Add to App.h)
```cpp
// Mode 9.2 lighting parameters
float m_lightScatterStrength = 0.5f;    // Particle-to-particle lighting strength
float m_lightFalloff = 20.0f;           // Light attenuation radius (world units)
float m_hgAnisotropy = 0.8f;            // Henyey-Greenstein phase (0.8 = forward scatter)
float m_bloomThreshold = 0.5f;          // Bloom luminance threshold
float m_bloomIntensity = 1.0f;          // Bloom strength multiplier
float m_extinctionCoeff = 0.1f;         // Beer's law absorption coefficient
bool m_debugEmissionView = false;       // Debug: Show only emission buffer
bool m_mode9LightingEnabled = true;     // Master toggle for Mode 9.2
```

---

## 4. Performance Estimates

### Frame Time Budget Breakdown (Target: 16.67ms @ 60fps)

| Pass | Resolution | Description | Estimated Cost | Cumulative |
|------|------------|-------------|----------------|------------|
| **Existing (Mode 9.1)** | | | | |
| Particle Physics | N/A | Compute: 100K particles | 0.5ms | 0.5ms |
| Shadow Map | 1024x1024 | RayQuery depth | 2.0ms | 2.5ms |
| Particle Render | 1920x1080 | Mesh shader + PS | 3.0ms | 5.5ms |
| **New (Mode 9.2)** | | | | |
| Emission Pass | 1920x1080 | Additional RT output | +0.5ms | 6.0ms |
| Light Grid Build | N/A | Spatial hash | 0.5ms | 6.5ms |
| Light Accumulation | N/A | Per-particle lighting | 2.5ms | 9.0ms |
| Volumetric Scatter | 960x540 | Half-res ray march | 3.5ms | 12.5ms |
| Bloom | Multi-res | Dual Kawase | 1.5ms | 14.0ms |
| Composite | 1920x1080 | Final blend | 0.5ms | 14.5ms |
| **Total** | | | | **14.5ms** |

**Margin:** 2.17ms (13% headroom for spikes)

### Optimization Strategies (If Needed)

1. **Light Grid LOD:** Only update grid every 2-3 frames (particles move slowly)
2. **Asymmetric Scatter:** Half-res scatter upsampled with bilateral filter
3. **Reduced Light Count:** Top 2% instead of 5% (temperature cutoff 20000K)
4. **Adaptive Samples:** 16 samples near particles, 8 samples in empty space
5. **Wave Intrinsics:** Use WaveReadLaneFirst for coherent light queries (30% speedup per 2025 data)

---

## 5. Progressive Enhancement Strategy

### Phase 1: Baseline (First Commit)
**What Works:**
- Emission buffer rendering
- Visible distinction between hot/cool particles
- Debug view (E key)

**Visual Impact:** Moderate (particles have inherent glow)
**Stability:** High (minimal changes to existing system)
**Time:** 2-3 days

---

### Phase 2: Core Lighting (Second Commit)
**Adds:**
- Light grid + accumulation
- Particle-to-particle illumination
- Runtime controls (U/I keys)

**Visual Impact:** High (particles light each other, creates depth)
**Stability:** Medium (new compute passes, potential bugs)
**Time:** 1 week

---

### Phase 3: Volumetric + Polish (Third Commit)
**Adds:**
- Screen-space scattering (fog/glow)
- Bloom post-process
- Henyey-Greenstein phase control (O key)
- Temporal reprojection (smoother)

**Visual Impact:** Very High (cinematic NASA-quality look)
**Stability:** Medium-Low (temporal artifacts possible)
**Time:** 1-2 weeks

---

### Phase 4: Relativistic Effects (Optional, Fourth Commit)
**Adds:**
- Doppler shift (blue-shift toward camera, red-shift away)
- Gravitational lensing (light bending near black hole)
- Einstein ring effect (photon ring around event horizon)

**Visual Impact:** Extreme (astrophysically accurate)
**Stability:** Low (complex shader math, performance risk)
**Time:** 2-3 weeks

**Implementation Notes:**
```hlsl
// Doppler shift (simplified, non-relativistic approximation)
float3 ApplyDopplerShift(float3 color, float3 velocity, float3 toCamera) {
    float radialVelocity = dot(normalize(velocity), normalize(toCamera));
    float beta = radialVelocity / 299792.458; // Speed of light (km/s units)
    float dopplerFactor = sqrt((1.0 - beta) / (1.0 + beta));

    // Shift wavelength (blue-shift if approaching, red-shift if receding)
    // Simplified: scale RGB channels by doppler factor
    return color * dopplerFactor;
}

// Gravitational lensing (ray deflection near black hole)
float3 ApplyGravitationalLensing(float3 rayDir, float3 blackHolePos, float schwarzschildRadius) {
    float3 toBlackHole = blackHolePos - cameraPos;
    float distance = length(toBlackHole);

    // Einstein deflection angle (small angle approximation)
    float deflectionAngle = (4.0 * schwarzschildRadius) / distance;

    // Bend ray toward black hole
    float3 perpDir = normalize(cross(rayDir, toBlackHole));
    float3 bentRay = normalize(rayDir + perpDir * deflectionAngle);

    return bentRay;
}
```

---

## 6. HLSL Code Examples (Complete Shaders)

### 6.1 Light Grid Build (Complete)
**File:** /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX/shaders/mode9/light_grid_build_cs.hlsl

```hlsl
// Build spatial acceleration grid for particle-emitted lights
// DXR Mode 9.2 - NASA Accretion Disk Advanced Lighting

#define GRID_SIZE 32
#define MAX_LIGHTS_PER_CELL 32

struct Particle {
    float3 position;
    float temperature;
    float3 velocity;
    float density;
};

// Output buffers
RWStructuredBuffer<uint> g_lightGrid : register(u0);  // [GRID_SIZE^3 * MAX_LIGHTS_PER_CELL]
RWStructuredBuffer<uint> g_lightCount : register(u1); // [GRID_SIZE^3]

// Input: All particles
StructuredBuffer<Particle> g_particles : register(t0);

cbuffer GridParams : register(b0) {
    float3 g_gridMin;          // (-100, -100, -100)
    float g_cellSize;          // 6.25 units
    float3 g_gridMax;          // (100, 100, 100)
    uint g_particleCount;      // 100000
    float g_emissionThreshold; // 15000K
    float g_intensityCutoff;   // 0.1 (cull weak lights)
};

uint3 WorldToGrid(float3 worldPos) {
    float3 normalized = (worldPos - g_gridMin) / (g_gridMax - g_gridMin);
    return clamp(uint3(normalized * GRID_SIZE), uint3(0,0,0), uint3(GRID_SIZE-1, GRID_SIZE-1, GRID_SIZE-1));
}

uint GridCellIndex(uint3 gridPos) {
    return gridPos.x + gridPos.y * GRID_SIZE + gridPos.z * GRID_SIZE * GRID_SIZE;
}

float CalculateEmissionIntensity(float temperature) {
    float normalizedTemp = saturate((temperature - 15000.0) / 11000.0);
    return pow(normalizedTemp, 2.0) * 5.0;
}

[numthreads(64, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    uint particleIdx = id.x;
    if (particleIdx >= g_particleCount) return;

    Particle p = g_particles[particleIdx];

    // Filter: Only hot particles emit light
    if (p.temperature < g_emissionThreshold) return;

    float intensity = CalculateEmissionIntensity(p.temperature);
    if (intensity < g_intensityCutoff) return;

    // Insert into grid
    uint3 gridPos = WorldToGrid(p.position);
    uint cellIdx = GridCellIndex(gridPos);

    uint insertIdx;
    InterlockedAdd(g_lightCount[cellIdx], 1, insertIdx);

    if (insertIdx < MAX_LIGHTS_PER_CELL) {
        uint writeIdx = cellIdx * MAX_LIGHTS_PER_CELL + insertIdx;
        g_lightGrid[writeIdx] = particleIdx;
    }
}
```

### 6.2 Light Accumulation (Complete)
**File:** /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX/shaders/mode9/light_accumulation_cs.hlsl

```hlsl
// Accumulate lighting from nearby emitting particles
// Uses spatial grid for efficient neighbor queries

#define GRID_SIZE 32
#define MAX_LIGHTS_PER_CELL 32

struct Particle {
    float3 position;
    float temperature;
    float3 velocity;
    float density;
};

// Inputs
StructuredBuffer<Particle> g_particles : register(t0);
StructuredBuffer<uint> g_lightGrid : register(t1);
StructuredBuffer<uint> g_lightCount : register(t2);

// Output
RWStructuredBuffer<float3> g_particleLighting : register(u0);

cbuffer LightParams : register(b0) {
    float3 g_gridMin;
    float g_cellSize;
    float3 g_gridMax;
    uint g_particleCount;
    float g_lightFalloff;      // 20.0
    float g_scatterStrength;   // 0.5
    float g_emissionThreshold; // 15000K
    float padding;
};

uint3 WorldToGrid(float3 worldPos) {
    float3 normalized = (worldPos - g_gridMin) / (g_gridMax - g_gridMin);
    return clamp(uint3(normalized * GRID_SIZE), uint3(0,0,0), uint3(GRID_SIZE-1, GRID_SIZE-1, GRID_SIZE-1));
}

uint GridCellIndex(uint3 gridPos) {
    return gridPos.x + gridPos.y * GRID_SIZE + gridPos.z * GRID_SIZE * GRID_SIZE;
}

float3 TemperatureToColor(float temperature) {
    float t = saturate((temperature - 800.0) / 25200.0);

    float3 color;
    if (t < 0.25) {
        float blend = t / 0.25;
        color = lerp(float3(0.5, 0.1, 0.05), float3(1.0, 0.3, 0.1), blend);
    } else if (t < 0.5) {
        float blend = (t - 0.25) / 0.25;
        color = lerp(float3(1.0, 0.3, 0.1), float3(1.0, 0.6, 0.2), blend);
    } else if (t < 0.75) {
        float blend = (t - 0.5) / 0.25;
        color = lerp(float3(1.0, 0.6, 0.2), float3(1.0, 0.95, 0.7), blend);
    } else {
        float blend = (t - 0.75) / 0.25;
        color = lerp(float3(1.0, 0.95, 0.7), float3(1.0, 1.0, 1.0), blend);
    }

    return color;
}

float CalculateEmissionIntensity(float temperature) {
    float normalizedTemp = saturate((temperature - g_emissionThreshold) / 11000.0);
    return pow(normalizedTemp, 2.0) * 5.0;
}

float3 SampleNearbyLights(float3 worldPos, uint particleIdx) {
    uint3 centerCell = WorldToGrid(worldPos);
    float3 totalLight = float3(0, 0, 0);

    // 3x3x3 neighborhood (27 cells)
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int3 cellPos = int3(centerCell) + int3(x, y, z);

                if (any(cellPos < 0) || any(cellPos >= GRID_SIZE)) continue;

                uint cellIdx = GridCellIndex(uint3(cellPos));
                uint lightCount = g_lightCount[cellIdx];

                for (uint i = 0; i < min(lightCount, MAX_LIGHTS_PER_CELL); i++) {
                    uint lightIdx = g_lightGrid[cellIdx * MAX_LIGHTS_PER_CELL + i];

                    if (lightIdx == particleIdx) continue;

                    Particle light = g_particles[lightIdx];
                    float3 toLight = light.position - worldPos;
                    float dist = length(toLight);

                    if (dist < 0.01) continue;

                    // Inverse square with smooth cutoff
                    float attenuation = 1.0 / (1.0 + dist * dist / (g_lightFalloff * g_lightFalloff));
                    attenuation *= smoothstep(g_lightFalloff * 2.0, 0.0, dist);

                    float intensity = CalculateEmissionIntensity(light.temperature);
                    float3 lightColor = TemperatureToColor(light.temperature);

                    totalLight += lightColor * intensity * attenuation;
                }
            }
        }
    }

    return totalLight * g_scatterStrength;
}

[numthreads(64, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    uint particleIdx = id.x;
    if (particleIdx >= g_particleCount) return;

    Particle p = g_particles[particleIdx];
    float3 lighting = SampleNearbyLights(p.position, particleIdx);

    g_particleLighting[particleIdx] = lighting;
}
```

### 6.3 Volumetric Scattering (Complete)
**File:** /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX/shaders/mode9/volumetric_scatter_cs.hlsl

```hlsl
// Screen-space volumetric scattering with Henyey-Greenstein phase function
// Half-resolution with temporal reprojection for performance

Texture2D<float> g_depthBuffer : register(t0);
Texture2D<float4> g_emissionBuffer : register(t1);
Texture2D<float4> g_colorBuffer : register(t2);
Texture2D<float4> g_historyBuffer : register(t3);

RWTexture2D<float4> g_scatterOutput : register(u0);

SamplerState sampler_linear : register(s0);
SamplerState sampler_point : register(s1);

cbuffer ScatterParams : register(b0) {
    float4x4 g_invViewProj;
    float4x4 g_viewProj;
    float3 g_cameraPos;
    float g_scatterStrength;
    float g_extinctionCoeff;
    float g_anisotropy;           // HG g parameter
    float3 g_lightDir;
    float g_lightIntensity;
    float2 g_resolution;          // Half-res: (960, 540)
    float g_maxDistance;          // 200.0
    float g_temporalBlend;        // 0.6
};

float HenyeyGreenstein(float cosTheta, float g) {
    const float INV_4PI = 0.07957747154594767;
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cosTheta;
    return INV_4PI * (1.0 - g2) / (denom * sqrt(max(denom, 0.0001)));
}

float3 UnprojectUV(float2 uv, float4x4 invViewProj) {
    float4 ndc = float4(uv * 2.0 - 1.0, 0.5, 1.0);
    ndc.y = -ndc.y; // Flip Y
    float4 worldPos = mul(ndc, invViewProj);
    return worldPos.xyz / worldPos.w - g_cameraPos;
}

float4 VolumetricScatter(float2 uv, float3 rayDir, float maxDist) {
    const int SAMPLES = 16;
    float stepSize = maxDist / float(SAMPLES);

    float3 scatteredLight = float3(0, 0, 0);
    float transmittance = 1.0;
    float3 currentPos = g_cameraPos;

    for (int i = 0; i < SAMPLES; i++) {
        currentPos += rayDir * stepSize;

        // Project to screen
        float4 clipPos = mul(float4(currentPos, 1.0), g_viewProj);
        float2 sampleUV = (clipPos.xy / clipPos.w) * 0.5 + 0.5;
        sampleUV.y = 1.0 - sampleUV.y;

        if (any(sampleUV < 0) || any(sampleUV > 1)) continue;

        float4 emission = g_emissionBuffer.SampleLevel(sampler_linear, sampleUV, 0);
        float density = emission.a;

        if (density < 0.01) continue;

        // Phase function
        float cosTheta = dot(rayDir, g_lightDir);
        float phase = HenyeyGreenstein(cosTheta, g_anisotropy);

        // In-scattering
        float3 inscatter = emission.rgb * phase * density * g_scatterStrength;

        // Extinction
        float extinction = exp(-density * g_extinctionCoeff * stepSize);

        scatteredLight += inscatter * transmittance;
        transmittance *= extinction;

        if (transmittance < 0.01) break;
    }

    return float4(scatteredLight, 1.0 - transmittance);
}

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    uint2 pixelCoord = id.xy;
    float2 uv = (float2(pixelCoord) + 0.5) / g_resolution;

    float3 rayDir = normalize(UnprojectUV(uv, g_invViewProj));
    float4 scatter = VolumetricScatter(uv, rayDir, g_maxDistance);

    // Temporal accumulation
    float4 history = g_historyBuffer[pixelCoord];
    float4 blended = lerp(scatter, history, g_temporalBlend);

    g_scatterOutput[pixelCoord] = blended;
}
```

---

## 7. Validation Checklist

### Visual Quality Tests
- [ ] Hot core (T > 20000K) glows intensely white/blue-white
- [ ] Mid-disk (T ~ 10000K) has orange/yellow glow
- [ ] Outer disk (T < 5000K) minimal glow, mostly red
- [ ] Particles near core are illuminated by neighbors
- [ ] Volumetric fog creates depth perception
- [ ] Bloom creates radiance without obscuring detail
- [ ] Smooth motion without temporal flickering

### Performance Tests
- [ ] 60fps stable at 1920x1080 (16.67ms target)
- [ ] No GPU timeouts or TDR events
- [ ] Memory usage < 2GB (RTX 4060 Ti has 16GB)
- [ ] CPU usage < 10% (all work on GPU)

### Runtime Control Tests
- [ ] U/Shift+U adjusts scatter strength smoothly
- [ ] I/Shift+I adjusts light falloff radius
- [ ] O/Shift+O changes anisotropy (visible fog directionality change)
- [ ] P/Shift+P adjusts bloom threshold (more/fewer particles bloom)
- [ ] [/Shift+[ adjusts bloom intensity (glow strength)
- [ ] ]/Shift+] adjusts extinction (fog density)
- [ ] E toggles emission debug view (shows only hot particles)
- [ ] L toggles entire lighting system on/off

### Correctness Tests
- [ ] No light self-interaction (particles don't light themselves)
- [ ] Grid bounds clamping works (no crashes at edges)
- [ ] Temporal reprojection doesn't ghost excessively
- [ ] Bloom doesn't bloom dark regions (threshold working)
- [ ] Light intensity scales with temperature squared (physical)

---

## 8. Known Limitations & Future Work

### Current Limitations
1. **No Occlusion:** Particle-to-particle lighting doesn't account for blocking (could add RayQuery shadow rays)
2. **Uniform Density:** Assumes all particles have same light-blocking density
3. **Screen-Space Artifacts:** Volumetric scattering misses off-screen emitters
4. **Temporal Ghosting:** Fast camera motion may cause trails
5. **No Multi-Scattering:** Only single-bounce light transport

### Future Enhancements
1. **Sparse Voxel Octree:** Replace 32^3 grid with adaptive octree (10x faster queries)
2. **RayQuery Shadows:** Add shadow rays for particle-particle occlusion
3. **Radiance Cascades:** Multi-scale light propagation for global illumination
4. **Spectral Rendering:** Full wavelength-dependent scattering (physically accurate)
5. **GPU Particles:** Move particle physics to dedicated compute pipeline

### Phase 4: Relativistic Effects (Optional)
If implementing gravitational lensing and Doppler shift:
- Research: Read "Interstellar" (Kip Thorne) technical papers on black hole rendering
- Tools: Use geodesic ray tracer for light paths
- Performance: Requires iterative ray solver (~10ms overhead)
- Visual: Creates Einstein ring, photon ring, extreme distortion near event horizon

---

## 9. References & Resources

### Academic Papers
1. "Clustered Deferred and Forward Shading" - Olsson et al. (2012)
2. "The Henyey-Greenstein Phase Function" - Henyey & Greenstein (1941)
3. "Volumetric Methods in Visual Effects" - Wrenninge (2016)
4. "Physically Based Sky, Atmosphere and Cloud Rendering" - Hillaire (2020)

### Technical Resources
- NVIDIA DXR 1.1 Tutorial: https://developer.nvidia.com/rtx/raytracing/dxr/dx12-raytracing-tutorial-part-2
- Microsoft DirectX Raytracing Spec: https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html
- GPU Pro 5: Volumetric Lights (Alexandre Pestana)
- Real-Time Rendering 4th Edition: Chapter 14 (Acceleration Structures)

### NASA Visualizations
- Black Hole Accretion Disk: https://svs.gsfc.nasa.gov/13326/
- Black Hole Anatomy: https://science.nasa.gov/universe/black-holes/anatomy/

---

## 10. Summary: Recommended Path Forward

### Immediate Actions (This Week)
1. **Implement Milestone 1:** Emission buffer system
   - Add second render target to particle mesh pipeline
   - Modify particle_pixel.hlsl to output emission
   - Create debug view toggle (E key)
   - **Expected Outcome:** Hot particles visibly glow

2. **Validate Performance:** Ensure Milestone 1 adds < 1ms
   - Use PIX/RGP to profile GPU timings
   - Confirm 60fps maintained

### Short-Term (Next 2 Weeks)
3. **Implement Milestones 2-3:** Light grid + accumulation
   - Build spatial grid compute shader
   - Accumulate lighting per particle
   - Add runtime controls (U/I keys)
   - **Expected Outcome:** Particles illuminate neighbors, creates depth

4. **Tune Parameters:** Artistic iteration
   - Adjust scatter strength, falloff radius
   - Find "sweet spot" values for accretion disk aesthetic

### Medium-Term (Next Month)
5. **Implement Milestones 4-5:** Volumetric + bloom
   - Screen-space scattering with HG phase function
   - Dual Kawase bloom post-process
   - Temporal reprojection for stability
   - **Expected Outcome:** NASA-quality cinematic rendering

6. **Optimize:** Hit 60fps consistently
   - Profile hotspots, optimize shaders
   - Consider LOD strategies if needed

### Long-Term (Optional)
7. **Phase 4:** Relativistic effects (stretch goal)
   - Doppler shift color grading
   - Gravitational lensing ray solver
   - **Expected Outcome:** Astrophysically accurate simulation

---

## Contact & Support

This implementation guide is designed for the PlasmaDX engine running on Windows with DirectX 12 Agility SDK.

**Target Hardware:** NVIDIA RTX 4060 Ti (DXR 1.2, Shader Model 6.5+)
**Target Performance:** 60fps @ 1920x1080
**Development Environment:** Visual Studio 2022, DXC compiler

For questions or clarifications, refer to the MCP DX12/DXR/HLSL documentation server.

---

**END OF MODE 9.2 IMPLEMENTATION GUIDE**
