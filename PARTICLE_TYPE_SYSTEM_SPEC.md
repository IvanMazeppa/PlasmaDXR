# Particle Type System - Technical Specification

**Document Version:** 1.0
**Created:** 2025-10-20
**Status:** Phase 5 Milestone 5.1 Implementation Blueprint
**Related:** PHASE_5_CELESTIAL_RENDERING_PLAN.md

---

## Executive Summary

This document provides the complete technical specification for implementing the Particle Type System in PlasmaDX-Clean. The system enables rendering of varied celestial bodies (stars of different types, gas clouds, dust particles) by extending the current unified particle model with material type differentiation.

**Key Changes:**
- Expand particle structure from 32 → 48 bytes (add 16 bytes)
- Introduce 5 material types with distinct visual properties
- Add material property constant buffer system
- Implement per-type rendering behaviors in Gaussian shader
- Create ImGui preset system for animation scenarios

**Implementation Time:** 1 week (5.1a: 2 days, 5.1b: 2 days, 5.1c: 3 days)

---

## Current System Analysis

### Existing Particle Structure (32 bytes)

**File:** `shaders/particles/particle_physics.hlsl`

```hlsl
struct Particle {
    float3 position;       // 12 bytes (world space coordinates)
    float radius;          // 4 bytes (base Gaussian radius)
    float3 velocity;       // 12 bytes (m/s, used for anisotropic elongation)
    float temperature;     // 4 bytes (Kelvin, 800K-26000K range)
};
// Total: 32 bytes
```

**Current Capabilities:**
- ✅ Temperature-based blackbody emission (800K-26000K)
- ✅ Anisotropic elongation along velocity vectors
- ✅ Radius variation (10.0 - 200.0 units)
- ✅ Opacity variation (algorithmic based on temperature)
- ✅ Henyey-Greenstein phase function scattering

**Current Limitations:**
- ❌ All particles use same material properties (density, albedo, phase function)
- ❌ No type-specific visual characteristics (stars vs gas vs dust)
- ❌ No runtime type distribution control
- ❌ No preset save/load for animation scenarios

---

## New Particle Type System

### Expanded Particle Structure (48 bytes)

**File:** `shaders/particles/particle_physics.hlsl` (modified)

```hlsl
// Material type enumeration (GPU constant)
// Values match C++ enum for upload consistency
#define PARTICLE_TYPE_PLASMA_BLOB        0
#define PARTICLE_TYPE_STAR_MAIN_SEQUENCE 1
#define PARTICLE_TYPE_STAR_GIANT         2
#define PARTICLE_TYPE_GAS_CLOUD          3
#define PARTICLE_TYPE_DUST_PARTICLE      4

struct Particle {
    // === Existing Fields (32 bytes) ===
    float3 position;       // 12 bytes (world space coordinates)
    float radius;          // 4 bytes (base Gaussian radius)
    float3 velocity;       // 12 bytes (m/s)
    float temperature;     // 4 bytes (Kelvin)

    // === NEW FIELDS (16 bytes) ===
    uint particleType;     // 4 bytes (PARTICLE_TYPE_* enumeration)
    float opacity;         // 4 bytes (0.0-1.0, overrides temperature-based opacity)
    float density;         // 4 bytes (kg/m³, affects light scattering)
    float lifetime;        // 4 bytes (seconds since spawn, for gas cloud dissipation)
};
// Total: 48 bytes (GPU alignment already 16-byte aligned)
```

**Field Descriptions:**

**`particleType` (uint):**
- Determines which material property set to use
- Immutable after particle initialization
- Valid values: 0-4 (5 types total)
- Used as index into material property constant buffer

**`opacity` (float):**
- Manual opacity override (0.0 = fully transparent, 1.0 = fully opaque)
- Replaces algorithmic temperature-based opacity calculation
- Allows artistic control for gas clouds (low opacity) vs stars (high opacity)
- Default: 1.0 (fully opaque, current behavior)

**`density` (float):**
- Material density in kg/m³
- Affects Beer-Lambert law absorption coefficient
- Higher density → stronger light absorption → darker shadows
- Range: 1e-5 (tenuous gas) to 1e5 (dense stellar core)
- Default: 1.0 (current unified behavior)

**`lifetime` (float):**
- Time since particle spawn in seconds
- Used for gas cloud dissipation effects (fade out over time)
- Used for dust particle burn-up (increase temperature near star)
- Updated by physics shader each frame
- Default: 0.0 (reset on spawn)

---

### Material Property Constant Buffer

**File:** `shaders/particles/particle_types.hlsl` (NEW FILE)

```hlsl
// Material properties for each particle type
// Uploaded once per frame via constant buffer
struct MaterialProperties {
    // === Base Material Properties ===
    float baseRadius;           // Base Gaussian radius multiplier (0.5 - 3.0)
    float densityScale;         // Density multiplier for Beer-Lambert law
    float opacityScale;         // Opacity multiplier (0.0 - 1.0)
    float emissionStrength;     // Emission intensity multiplier (0.0 - 10.0)

    // === Scattering Properties ===
    float phaseG;               // Henyey-Greenstein g parameter (-1.0 to 1.0)
    float scatteringAlbedo;     // Single-scattering albedo (0.0 - 1.0)
    float absorptionCoeff;      // Absorption coefficient multiplier (0.0 - 10.0)
    float extinctionCoeff;      // Extinction coefficient (absorption + scattering)

    // === Color Properties ===
    float3 tintColor;           // RGB color tint (multiplicative)
    float blackbodyWeight;      // Blend between tint and blackbody (0.0 - 1.0)

    // === Animation Properties ===
    float anisotropyStrength;   // Velocity-based elongation multiplier (0.0 - 5.0)
    float dissipationRate;      // Opacity decay per second (for gas clouds)
    float temperatureFloor;     // Minimum temperature (Kelvin)
    float temperatureCeiling;   // Maximum temperature (Kelvin)
};

// Constant buffer for all 5 material types
cbuffer MaterialTypeConstants : register(b2) {
    MaterialProperties g_materialTypes[5];  // 64 bytes × 5 = 320 bytes total
};
```

**GPU Buffer Layout:**
- Constant buffer slot: `b2` (next available after b0=render constants, b1=physics constants)
- Size: 320 bytes (64 bytes per material × 5 types)
- Update frequency: Once per frame (only when user modifies ImGui controls)
- CPU-side storage: `src/core/Application.h` → `MaterialTypeConfig m_materialConfigs[5]`

---

## Material Type Definitions

### Type 0: PLASMA_BLOB (Default/Fallback)

**Visual Character:** Current behavior, hot volumetric plasma

```cpp
MaterialProperties plasmaBlob = {
    // Base
    .baseRadius = 1.0f,              // Default size
    .densityScale = 1.0f,            // Standard density
    .opacityScale = 1.0f,            // Fully opaque
    .emissionStrength = 1.0f,        // Normal emission

    // Scattering
    .phaseG = 0.3f,                  // Slight forward scattering (current)
    .scatteringAlbedo = 0.15f,       // 15% albedo tint (current)
    .absorptionCoeff = 1.0f,         // Standard absorption
    .extinctionCoeff = 1.15f,        // absorption + scattering

    // Color
    .tintColor = float3(1, 0.8, 0.6), // Warm artistic tint (current)
    .blackbodyWeight = 0.7f,         // 70% blackbody, 30% artistic (current)

    // Animation
    .anisotropyStrength = 1.0f,      // Current anisotropic elongation
    .dissipationRate = 0.0f,         // No decay
    .temperatureFloor = 800.0f,      // Current minimum
    .temperatureCeiling = 26000.0f   // Current maximum
};
```

**Use Cases:**
- Accretion disk particles (current default)
- Fallback type if unknown type value
- General-purpose hot plasma

---

### Type 1: STAR_MAIN_SEQUENCE

**Visual Character:** Bright, dense stellar core with strong emission

```cpp
MaterialProperties starMainSequence = {
    // Base
    .baseRadius = 2.0f,              // 2× larger than default
    .densityScale = 10.0f,           // Very dense core
    .opacityScale = 1.0f,            // Fully opaque
    .emissionStrength = 5.0f,        // 5× brighter emission

    // Scattering
    .phaseG = 0.7f,                  // Strong forward scattering (limb darkening)
    .scatteringAlbedo = 0.05f,       // Low albedo (pure emission dominant)
    .absorptionCoeff = 5.0f,         // Strong self-absorption
    .extinctionCoeff = 5.05f,

    // Color
    .tintColor = float3(1, 1, 1),    // Pure white (use full blackbody)
    .blackbodyWeight = 1.0f,         // 100% blackbody radiation (physically accurate)

    // Animation
    .anisotropyStrength = 0.1f,      // Minimal elongation (spherical stars)
    .dissipationRate = 0.0f,         // Stars don't dissipate
    .temperatureFloor = 3000.0f,     // Red dwarf minimum (M-class)
    .temperatureCeiling = 40000.0f   // Blue giant maximum (O-class)
};
```

**Visual Characteristics:**
- Large, bright, spherical
- Limb darkening effect (center brighter than edges)
- Pure blackbody color (3000K red → 40000K blue)
- Minimal velocity-based elongation

**Use Cases:**
- Central star in stellar system
- Binary star pairs
- Background star field

---

### Type 2: STAR_GIANT (Red Giants, Blue Supergiants)

**Visual Character:** Huge diffuse envelope with lower density

```cpp
MaterialProperties starGiant = {
    // Base
    .baseRadius = 3.0f,              // 3× larger than default
    .densityScale = 0.1f,            // Very tenuous outer envelope
    .opacityScale = 0.6f,            // Partially transparent (see inner layers)
    .emissionStrength = 8.0f,        // Very bright but diffuse

    // Scattering
    .phaseG = 0.5f,                  // Moderate forward scattering
    .scatteringAlbedo = 0.2f,        // Higher scattering (diffuse glow)
    .absorptionCoeff = 0.5f,         // Low absorption (tenuous)
    .extinctionCoeff = 0.7f,

    // Color
    .tintColor = float3(1, 0.4, 0.2), // Reddish tint for red giants
    .blackbodyWeight = 0.8f,         // Mostly blackbody with artistic tint

    // Animation
    .anisotropyStrength = 0.3f,      // Slight elongation (stellar winds)
    .dissipationRate = 0.0f,
    .temperatureFloor = 2500.0f,     // Cool red giant minimum
    .temperatureCeiling = 35000.0f   // Blue supergiant maximum
};
```

**Visual Characteristics:**
- Massive size with diffuse edges
- Partially transparent (layered envelope visible)
- Red tint for giants, blue tint for supergiants (via temperature)
- Subtle elongation from stellar winds

**Use Cases:**
- Red giant stars (Betelgeuse-like)
- Blue supergiants (Rigel-like)
- Late-stage stellar evolution

---

### Type 3: GAS_CLOUD (Nebulae, Interstellar Medium)

**Visual Character:** Low-density, highly scattering, dissipating over time

```cpp
MaterialProperties gasCloud = {
    // Base
    .baseRadius = 1.5f,              // Larger than plasma
    .densityScale = 0.01f,           // Extremely tenuous
    .opacityScale = 0.2f,            // Very transparent (20% opacity)
    .emissionStrength = 0.1f,        // Minimal self-emission

    // Scattering
    .phaseG = -0.3f,                 // BACKWARD scattering (Rayleigh-like)
    .scatteringAlbedo = 0.8f,        // Very high scattering (glowing from external light)
    .absorptionCoeff = 0.05f,        // Minimal absorption
    .extinctionCoeff = 0.85f,        // Scattering-dominated

    // Color
    .tintColor = float3(0.3, 0.5, 1.0), // Blue/cyan tint (emission nebula)
    .blackbodyWeight = 0.1f,         // Mostly artistic color (10% blackbody)

    // Animation
    .anisotropyStrength = 2.0f,      // Strong elongation (turbulent flow)
    .dissipationRate = 0.05f,        // Fade out over 20 seconds
    .temperatureFloor = 100.0f,      // Cold interstellar gas
    .temperatureCeiling = 10000.0f   // Ionized gas maximum
};
```

**Visual Characteristics:**
- Large, diffuse, wispy
- Backward scattering (glows when backlit)
- Fades out over time (lifetime-based opacity)
- Highly turbulent (strong anisotropy)

**Use Cases:**
- Nebula clouds
- Accretion disk outflows
- Interstellar medium

**Dissipation Logic:**
```hlsl
// In physics shader (particle_physics.hlsl)
if (particle.particleType == PARTICLE_TYPE_GAS_CLOUD) {
    particle.lifetime += deltaTime;
    float dissipationFactor = exp(-particle.lifetime * g_materialTypes[3].dissipationRate);
    particle.opacity *= dissipationFactor;  // Exponential decay
}
```

---

### Type 4: DUST_PARTICLE

**Visual Character:** Small, dense, high absorption, cool temperature

```cpp
MaterialProperties dustParticle = {
    // Base
    .baseRadius = 0.5f,              // Small particles
    .densityScale = 100.0f,          // Very dense (solid grains)
    .opacityScale = 1.0f,            // Fully opaque
    .emissionStrength = 0.01f,       // Minimal self-emission (cool)

    // Scattering
    .phaseG = 0.0f,                  // Isotropic scattering (Mie scattering)
    .scatteringAlbedo = 0.3f,        // Moderate scattering
    .absorptionCoeff = 10.0f,        // VERY high absorption (dark dust)
    .extinctionCoeff = 10.3f,        // Absorption-dominated

    // Color
    .tintColor = float3(0.6, 0.5, 0.4), // Brown/gray dust color
    .blackbodyWeight = 0.3f,         // Mostly artistic (30% blackbody)

    // Animation
    .anisotropyStrength = 0.5f,      // Moderate elongation (grain alignment)
    .dissipationRate = 0.0f,         // Dust persists
    .temperatureFloor = 50.0f,       // Very cold dust
    .temperatureCeiling = 2000.0f    // Hot dust near star
};
```

**Visual Characteristics:**
- Small, dark, opaque
- Creates dark lanes in accretion disk
- Strong shadows cast on background particles
- Low emission (cool blackbody)

**Use Cases:**
- Dust lanes in accretion disk
- Protoplanetary disk
- Dust torus around black hole

**Temperature Heating Logic:**
```hlsl
// In physics shader - dust heats up near star
if (particle.particleType == PARTICLE_TYPE_DUST_PARTICLE) {
    float distToStar = length(particle.position);
    float heatingFactor = 1.0 / (distToStar * 0.001);  // Inverse square law
    particle.temperature = min(particle.temperature + heatingFactor * deltaTime, 2000.0f);
}
```

---

## Implementation Task Breakdown

### Milestone 5.1a: Core Structure Expansion (2 days)

**Task 5.1a.1: Expand GPU Particle Buffer (4 hours)**

**Files Modified:**
- `src/particles/ParticleSystem.h` (add new fields to CPU struct)
- `src/particles/ParticleSystem.cpp` (update buffer creation)
- `shaders/particles/particle_physics.hlsl` (expand Particle struct)

**Changes:**

1. **CPU Particle Structure (`src/particles/ParticleSystem.h`)**
```cpp
struct ParticleData {
    DirectX::XMFLOAT3 position;
    float radius;
    DirectX::XMFLOAT3 velocity;
    float temperature;

    // NEW FIELDS (16 bytes)
    uint32_t particleType;      // 0-4 (PARTICLE_TYPE_* enum)
    float opacity;              // 0.0-1.0
    float density;              // kg/m³
    float lifetime;             // seconds
};
// Total: 48 bytes (was 32 bytes)
```

2. **Buffer Creation (`src/particles/ParticleSystem.cpp`)**
```cpp
// Update buffer size calculation
const uint32_t PARTICLE_STRIDE = 48;  // Was 32
const uint32_t bufferSize = particleCount * PARTICLE_STRIDE;

// Update initialization logic
for (uint32_t i = 0; i < particleCount; i++) {
    particleData[i].particleType = 0;  // Default to PLASMA_BLOB
    particleData[i].opacity = 1.0f;    // Fully opaque
    particleData[i].density = 1.0f;    // Standard density
    particleData[i].lifetime = 0.0f;   // Just spawned
}
```

3. **GPU Shader Struct (`shaders/particles/particle_physics.hlsl`)**
```hlsl
// Add type definitions at top of file
#define PARTICLE_TYPE_PLASMA_BLOB        0
#define PARTICLE_TYPE_STAR_MAIN_SEQUENCE 1
#define PARTICLE_TYPE_STAR_GIANT         2
#define PARTICLE_TYPE_GAS_CLOUD          3
#define PARTICLE_TYPE_DUST_PARTICLE      4

struct Particle {
    float3 position;
    float radius;
    float3 velocity;
    float temperature;
    uint particleType;     // NEW
    float opacity;         // NEW
    float density;         // NEW
    float lifetime;        // NEW
};
```

**Testing:**
- Build succeeds with no errors
- Particle buffer uploads correctly (verify in PIX)
- Physics shader reads/writes new fields
- No regression in particle rendering

---

**Task 5.1a.2: Create Material Property System (4 hours)**

**Files Created:**
- `shaders/particles/particle_types.hlsl` (material property definitions)

**Files Modified:**
- `src/core/Application.h` (add MaterialTypeConfig struct)
- `src/core/Application.cpp` (create constant buffer, upload logic)

**Changes:**

1. **Material Property Shader (`shaders/particles/particle_types.hlsl`)**
```hlsl
struct MaterialProperties {
    float baseRadius;
    float densityScale;
    float opacityScale;
    float emissionStrength;
    float phaseG;
    float scatteringAlbedo;
    float absorptionCoeff;
    float extinctionCoeff;
    float3 tintColor;
    float blackbodyWeight;
    float anisotropyStrength;
    float dissipationRate;
    float temperatureFloor;
    float temperatureCeiling;
    // Padding to 64 bytes (GPU alignment)
    float2 _padding;
};

cbuffer MaterialTypeConstants : register(b2) {
    MaterialProperties g_materialTypes[5];
};
```

2. **CPU Configuration (`src/core/Application.h`)**
```cpp
// Material type configuration (CPU-side mirror of GPU constant buffer)
struct MaterialTypeConfig {
    float baseRadius = 1.0f;
    float densityScale = 1.0f;
    float opacityScale = 1.0f;
    float emissionStrength = 1.0f;
    float phaseG = 0.3f;
    float scatteringAlbedo = 0.15f;
    float absorptionCoeff = 1.0f;
    float extinctionCoeff = 1.15f;
    DirectX::XMFLOAT3 tintColor = {1.0f, 0.8f, 0.6f};
    float blackbodyWeight = 0.7f;
    float anisotropyStrength = 1.0f;
    float dissipationRate = 0.0f;
    float temperatureFloor = 800.0f;
    float temperatureCeiling = 26000.0f;
    float _padding[2] = {0, 0};
};

class Application {
private:
    MaterialTypeConfig m_materialConfigs[5];  // One per particle type
    ComPtr<ID3D12Resource> m_materialTypeBuffer;  // GPU constant buffer
    void InitializeDefaultMaterialTypes();  // Set defaults in constructor
    void UploadMaterialTypeConstants();     // Upload to GPU each frame
};
```

3. **Initialization (`src/core/Application.cpp`)**
```cpp
void Application::InitializeDefaultMaterialTypes() {
    // Type 0: PLASMA_BLOB (current behavior)
    m_materialConfigs[0] = {
        .baseRadius = 1.0f,
        .densityScale = 1.0f,
        .opacityScale = 1.0f,
        .emissionStrength = 1.0f,
        .phaseG = 0.3f,
        .scatteringAlbedo = 0.15f,
        .absorptionCoeff = 1.0f,
        .extinctionCoeff = 1.15f,
        .tintColor = {1.0f, 0.8f, 0.6f},
        .blackbodyWeight = 0.7f,
        .anisotropyStrength = 1.0f,
        .dissipationRate = 0.0f,
        .temperatureFloor = 800.0f,
        .temperatureCeiling = 26000.0f
    };

    // Type 1: STAR_MAIN_SEQUENCE
    m_materialConfigs[1] = { /* ... see definitions above ... */ };

    // Type 2: STAR_GIANT
    m_materialConfigs[2] = { /* ... */ };

    // Type 3: GAS_CLOUD
    m_materialConfigs[3] = { /* ... */ };

    // Type 4: DUST_PARTICLE
    m_materialConfigs[4] = { /* ... */ };
}

void Application::UploadMaterialTypeConstants() {
    // Upload all 5 material configs to GPU constant buffer b2
    // Only upload when user modifies via ImGui
    void* mappedData;
    m_materialTypeBuffer->Map(0, nullptr, &mappedData);
    memcpy(mappedData, m_materialConfigs, sizeof(m_materialConfigs));
    m_materialTypeBuffer->Unmap(0, nullptr);
}
```

**Testing:**
- Constant buffer creation succeeds
- Upload completes without errors
- Shader can read g_materialTypes[0-4]
- Default plasma behavior unchanged

---

### Milestone 5.1b: Per-Type Rendering (2 days)

**Task 5.1b.1: Update Gaussian Renderer (6 hours)**

**Files Modified:**
- `shaders/particles/particle_gaussian_raytrace.hlsl` (per-type material application)
- `shaders/particles/gaussian_common.hlsl` (material lookup functions)

**Changes:**

1. **Material Lookup Function (`gaussian_common.hlsl`)**
```hlsl
#include "particle_types.hlsl"

// Get material properties for a particle
MaterialProperties GetMaterialProperties(uint particleType) {
    // Bounds check (fallback to PLASMA_BLOB)
    if (particleType > 4) {
        particleType = 0;
    }
    return g_materialTypes[particleType];
}

// Apply material-specific radius scaling
float GetEffectiveRadius(Particle particle) {
    MaterialProperties mat = GetMaterialProperties(particle.particleType);
    return particle.radius * mat.baseRadius;
}

// Apply material-specific opacity
float GetEffectiveOpacity(Particle particle) {
    MaterialProperties mat = GetMaterialProperties(particle.particleType);
    float baseOpacity = particle.opacity * mat.opacityScale;

    // Apply lifetime-based dissipation (gas clouds)
    if (mat.dissipationRate > 0.0) {
        float dissipationFactor = exp(-particle.lifetime * mat.dissipationRate);
        baseOpacity *= dissipationFactor;
    }

    return saturate(baseOpacity);
}

// Apply material-specific emission
float3 GetParticleEmission(Particle particle, float temperature) {
    MaterialProperties mat = GetMaterialProperties(particle.particleType);

    // Clamp temperature to material range
    temperature = clamp(temperature, mat.temperatureFloor, mat.temperatureCeiling);

    // Calculate blackbody emission
    float3 blackbodyColor = TemperatureToColor(temperature);

    // Blend with artistic tint
    float3 finalColor = lerp(mat.tintColor, blackbodyColor, mat.blackbodyWeight);

    // Apply emission strength
    return finalColor * mat.emissionStrength;
}

// Apply material-specific scattering
float GetPhaseFunction(Particle particle, float cosTheta) {
    MaterialProperties mat = GetMaterialProperties(particle.particleType);
    return HenyeyGreenstein(cosTheta, mat.phaseG);
}
```

2. **Volumetric Ray Marching Update (`particle_gaussian_raytrace.hlsl`)**
```hlsl
// In main ray marching loop:
for (uint i = 0; i < sortedCount; i++) {
    Particle particle = g_particles[sortedIndices[i]];
    MaterialProperties mat = GetMaterialProperties(particle.particleType);

    // Use material-specific radius
    float effectiveRadius = GetEffectiveRadius(particle);

    // Use material-specific opacity
    float effectiveOpacity = GetEffectiveOpacity(particle);

    // Use material-specific density for Beer-Lambert
    float densityAdjusted = particle.density * mat.densityScale;
    float absorption = mat.absorptionCoeff * densityAdjusted * stepSize;
    float transmittance = exp(-absorption);  // Beer-Lambert law

    // Use material-specific emission
    float3 emission = GetParticleEmission(particle, particle.temperature);

    // Use material-specific phase function
    float phase = GetPhaseFunction(particle, cosTheta);

    // Apply scattering albedo
    float3 scatteredLight = incomingLight * mat.scatteringAlbedo * phase;

    // Accumulate color (unchanged logic)
    accumulatedColor += (emission + scatteredLight) * transmittance * effectiveOpacity;
    accumulatedAlpha += effectiveOpacity * (1.0 - accumulatedAlpha);
}
```

**Testing:**
- All 5 material types render with distinct appearances
- PLASMA_BLOB matches previous behavior (regression test)
- Gas clouds show dissipation over time
- Stars appear brighter and more spherical
- Dust particles appear dark and small

---

**Task 5.1b.2: Update Physics Shader (2 hours)**

**Files Modified:**
- `shaders/particles/particle_physics.hlsl` (per-type physics behaviors)

**Changes:**

```hlsl
#include "particle_types.hlsl"

[numthreads(256, 1, 1)]
void main(uint3 threadID : SV_DispatchThreadID) {
    uint particleIndex = threadID.x;
    if (particleIndex >= g_particleCount) return;

    Particle particle = g_particles[particleIndex];
    MaterialProperties mat = g_materialTypes[particle.particleType];

    // === Update lifetime (all types) ===
    particle.lifetime += g_deltaTime;

    // === Gas cloud dissipation ===
    if (particle.particleType == PARTICLE_TYPE_GAS_CLOUD && mat.dissipationRate > 0.0) {
        float dissipationFactor = exp(-particle.lifetime * mat.dissipationRate);
        particle.opacity *= dissipationFactor;

        // Respawn if fully dissipated
        if (particle.opacity < 0.01) {
            particle.opacity = 0.2;  // Reset to default gas opacity
            particle.lifetime = 0.0;
            // Respawn at random position
            particle.position = RandomPositionInDisk();
        }
    }

    // === Dust heating near star ===
    if (particle.particleType == PARTICLE_TYPE_DUST_PARTICLE) {
        float distToCenter = length(particle.position);
        float heatingRate = 100.0 / (distToCenter * distToCenter);  // Inverse square
        particle.temperature = min(particle.temperature + heatingRate * g_deltaTime, mat.temperatureCeiling);
    }

    // === Clamp temperature to material range ===
    particle.temperature = clamp(particle.temperature, mat.temperatureFloor, mat.temperatureCeiling);

    // === Apply material-specific anisotropy ===
    float velocityMagnitude = length(particle.velocity);
    float anisotropy = mat.anisotropyStrength * velocityMagnitude;
    // (anisotropy used in renderer for Gaussian elongation)

    // Write back
    g_particles[particleIndex] = particle;
}
```

**Testing:**
- Gas clouds fade out over 20 seconds
- Dust particles heat up near center
- Stars maintain temperature range
- Lifetime updates correctly

---

### Milestone 5.1c: ImGui Controls & Presets (3 days)

**Task 5.1c.1: ImGui Material Type Editor (4 hours)**

**Files Modified:**
- `src/core/Application.cpp` (add ImGui controls)

**Changes:**

```cpp
void Application::RenderImGui() {
    // ... existing ImGui code ...

    // === NEW: Particle Type System Controls ===
    if (ImGui::CollapsingHeader("Particle Type System", ImGuiTreeNodeFlags_DefaultOpen)) {

        // Type distribution controls
        if (ImGui::TreeNode("Type Distribution")) {
            static int typeToSpawn = 0;
            static float spawnPercentage[5] = {100, 0, 0, 0, 0};  // Default: 100% plasma

            ImGui::Text("Set particle type distribution:");
            ImGui::SliderFloat("Plasma Blob %%", &spawnPercentage[0], 0, 100);
            ImGui::SliderFloat("Main Sequence Star %%", &spawnPercentage[1], 0, 100);
            ImGui::SliderFloat("Giant Star %%", &spawnPercentage[2], 0, 100);
            ImGui::SliderFloat("Gas Cloud %%", &spawnPercentage[3], 0, 100);
            ImGui::SliderFloat("Dust Particle %%", &spawnPercentage[4], 0, 100);

            if (ImGui::Button("Apply Distribution")) {
                // Normalize percentages to sum to 100%
                float total = spawnPercentage[0] + spawnPercentage[1] + spawnPercentage[2] +
                              spawnPercentage[3] + spawnPercentage[4];
                for (int i = 0; i < 5; i++) {
                    spawnPercentage[i] = (spawnPercentage[i] / total) * 100.0f;
                }

                // Reassign particle types based on distribution
                ReinitializeParticleTypes(spawnPercentage);
            }

            ImGui::TreePop();
        }

        // Per-type material property editors
        const char* typeNames[5] = {
            "Plasma Blob", "Main Sequence Star", "Giant Star", "Gas Cloud", "Dust Particle"
        };

        for (int typeIdx = 0; typeIdx < 5; typeIdx++) {
            if (ImGui::TreeNode(typeNames[typeIdx])) {
                MaterialTypeConfig& mat = m_materialConfigs[typeIdx];
                bool changed = false;

                // Base properties
                changed |= ImGui::SliderFloat("Base Radius", &mat.baseRadius, 0.1f, 5.0f);
                changed |= ImGui::SliderFloat("Density Scale", &mat.densityScale, 0.001f, 100.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
                changed |= ImGui::SliderFloat("Opacity Scale", &mat.opacityScale, 0.0f, 1.0f);
                changed |= ImGui::SliderFloat("Emission Strength", &mat.emissionStrength, 0.0f, 10.0f);

                // Scattering properties
                if (ImGui::TreeNode("Scattering")) {
                    changed |= ImGui::SliderFloat("Phase G", &mat.phaseG, -1.0f, 1.0f);
                    changed |= ImGui::SliderFloat("Scattering Albedo", &mat.scatteringAlbedo, 0.0f, 1.0f);
                    changed |= ImGui::SliderFloat("Absorption Coeff", &mat.absorptionCoeff, 0.0f, 10.0f);
                    changed |= ImGui::SliderFloat("Extinction Coeff", &mat.extinctionCoeff, 0.0f, 15.0f);
                    ImGui::TreePop();
                }

                // Color properties
                if (ImGui::TreeNode("Color")) {
                    changed |= ImGui::ColorEdit3("Tint Color", &mat.tintColor.x);
                    changed |= ImGui::SliderFloat("Blackbody Weight", &mat.blackbodyWeight, 0.0f, 1.0f);
                    ImGui::TreePop();
                }

                // Animation properties
                if (ImGui::TreeNode("Animation")) {
                    changed |= ImGui::SliderFloat("Anisotropy Strength", &mat.anisotropyStrength, 0.0f, 5.0f);
                    changed |= ImGui::SliderFloat("Dissipation Rate", &mat.dissipationRate, 0.0f, 1.0f);
                    changed |= ImGui::SliderFloat("Temperature Floor", &mat.temperatureFloor, 50.0f, 10000.0f);
                    changed |= ImGui::SliderFloat("Temperature Ceiling", &mat.temperatureCeiling, 1000.0f, 50000.0f);
                    ImGui::TreePop();
                }

                if (changed) {
                    m_materialTypesNeedUpload = true;  // Flag for GPU upload
                }

                ImGui::TreePop();
            }
        }

        // Reset to defaults button
        if (ImGui::Button("Reset All to Defaults")) {
            InitializeDefaultMaterialTypes();
            m_materialTypesNeedUpload = true;
        }
    }
}

// In Update() method:
if (m_materialTypesNeedUpload) {
    UploadMaterialTypeConstants();
    m_materialTypesNeedUpload = false;
}
```

**Testing:**
- All sliders update material properties in real-time
- Changes apply immediately (no restart required)
- Reset button restores defaults
- No performance impact from ImGui rendering

---

**Task 5.1c.2: Preset Save/Load System (4 hours)**

**Files Created:**
- `configs/particle_types/` (directory for preset JSON files)
- `configs/particle_types/default.json`
- `configs/particle_types/accretion_disk.json`
- `configs/particle_types/stellar_nursery.json`
- `configs/particle_types/dust_torus.json`

**Files Modified:**
- `src/core/Application.h` (add preset manager)
- `src/core/Application.cpp` (save/load logic)

**Changes:**

1. **Preset JSON Format (`configs/particle_types/default.json`)**
```json
{
    "name": "Default Plasma Blob",
    "description": "100% plasma particles, current behavior",
    "typeDistribution": {
        "PLASMA_BLOB": 100.0,
        "STAR_MAIN_SEQUENCE": 0.0,
        "STAR_GIANT": 0.0,
        "GAS_CLOUD": 0.0,
        "DUST_PARTICLE": 0.0
    },
    "materialTypes": {
        "PLASMA_BLOB": {
            "baseRadius": 1.0,
            "densityScale": 1.0,
            "opacityScale": 1.0,
            "emissionStrength": 1.0,
            "phaseG": 0.3,
            "scatteringAlbedo": 0.15,
            "absorptionCoeff": 1.0,
            "extinctionCoeff": 1.15,
            "tintColor": [1.0, 0.8, 0.6],
            "blackbodyWeight": 0.7,
            "anisotropyStrength": 1.0,
            "dissipationRate": 0.0,
            "temperatureFloor": 800.0,
            "temperatureCeiling": 26000.0
        }
    }
}
```

2. **Preset Manager (`src/core/Application.cpp`)**
```cpp
bool Application::LoadParticleTypePreset(const std::string& presetPath) {
    // Read JSON file
    std::ifstream file(presetPath);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open particle type preset: {}", presetPath);
        return false;
    }

    nlohmann::json preset;
    file >> preset;

    // Parse type distribution
    auto& dist = preset["typeDistribution"];
    float percentages[5] = {
        dist["PLASMA_BLOB"].get<float>(),
        dist["STAR_MAIN_SEQUENCE"].get<float>(),
        dist["STAR_GIANT"].get<float>(),
        dist["GAS_CLOUD"].get<float>(),
        dist["DUST_PARTICLE"].get<float>()
    };
    ReinitializeParticleTypes(percentages);

    // Parse material properties
    auto& materials = preset["materialTypes"];
    for (int typeIdx = 0; typeIdx < 5; typeIdx++) {
        const char* typeName = GetParticleTypeName(typeIdx);
        if (materials.contains(typeName)) {
            auto& mat = materials[typeName];
            m_materialConfigs[typeIdx].baseRadius = mat["baseRadius"].get<float>();
            m_materialConfigs[typeIdx].densityScale = mat["densityScale"].get<float>();
            // ... load all other fields ...
        }
    }

    m_materialTypesNeedUpload = true;
    LOG_INFO("Loaded particle type preset: {}", presetPath);
    return true;
}

bool Application::SaveParticleTypePreset(const std::string& presetPath) {
    nlohmann::json preset;

    // Save current type distribution
    float percentages[5];
    CalculateCurrentTypeDistribution(percentages);
    preset["typeDistribution"] = {
        {"PLASMA_BLOB", percentages[0]},
        {"STAR_MAIN_SEQUENCE", percentages[1]},
        {"STAR_GIANT", percentages[2]},
        {"GAS_CLOUD", percentages[3]},
        {"DUST_PARTICLE", percentages[4]}
    };

    // Save material properties
    for (int typeIdx = 0; typeIdx < 5; typeIdx++) {
        const char* typeName = GetParticleTypeName(typeIdx);
        auto& mat = m_materialConfigs[typeIdx];
        preset["materialTypes"][typeName] = {
            {"baseRadius", mat.baseRadius},
            {"densityScale", mat.densityScale},
            // ... save all fields ...
        };
    }

    // Write to file
    std::ofstream file(presetPath);
    file << preset.dump(4);  // Pretty-print with 4-space indent

    LOG_INFO("Saved particle type preset: {}", presetPath);
    return true;
}
```

3. **ImGui Preset UI**
```cpp
// In RenderImGui():
if (ImGui::TreeNode("Presets")) {
    // Preset dropdown
    static int selectedPreset = 0;
    const char* presetNames[] = {
        "Default", "Accretion Disk", "Stellar Nursery", "Dust Torus", "Custom"
    };

    if (ImGui::Combo("Preset", &selectedPreset, presetNames, 5)) {
        std::string presetPath = "configs/particle_types/";
        switch (selectedPreset) {
            case 0: presetPath += "default.json"; break;
            case 1: presetPath += "accretion_disk.json"; break;
            case 2: presetPath += "stellar_nursery.json"; break;
            case 3: presetPath += "dust_torus.json"; break;
            case 4: /* Custom - no load */ break;
        }

        if (selectedPreset != 4) {
            LoadParticleTypePreset(presetPath);
        }
    }

    // Save current as new preset
    static char savePresetName[256] = "";
    ImGui::InputText("Save As", savePresetName, 256);
    if (ImGui::Button("Save Preset")) {
        std::string savePath = "configs/particle_types/" + std::string(savePresetName) + ".json";
        SaveParticleTypePreset(savePath);
    }

    ImGui::TreePop();
}
```

**Testing:**
- Load default preset → 100% plasma (regression test)
- Load accretion disk preset → mixed types
- Save custom preset → file created with correct JSON
- Reload custom preset → configuration restored

---

## GPU Buffer Update Strategy

### Particle Type Reassignment Algorithm

**When:** User changes type distribution via ImGui or loads preset

**Method:** CPU-side random assignment, full GPU buffer upload

**Implementation:**
```cpp
void Application::ReinitializeParticleTypes(const float percentages[5]) {
    // Normalize percentages to 100%
    float total = percentages[0] + percentages[1] + percentages[2] + percentages[3] + percentages[4];
    float normalized[5];
    for (int i = 0; i < 5; i++) {
        normalized[i] = percentages[i] / total;
    }

    // Calculate cumulative distribution
    float cumulative[5];
    cumulative[0] = normalized[0];
    for (int i = 1; i < 5; i++) {
        cumulative[i] = cumulative[i-1] + normalized[i];
    }

    // Reassign particle types
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    ParticleData* particles = m_particleSystem->GetParticleData();  // CPU-side buffer

    for (uint32_t i = 0; i < m_particleSystem->GetParticleCount(); i++) {
        float rand = dist(gen);

        // Find type based on cumulative distribution
        uint32_t type = 4;  // Default to DUST_PARTICLE
        for (int t = 0; t < 5; t++) {
            if (rand < cumulative[t]) {
                type = t;
                break;
            }
        }

        particles[i].particleType = type;

        // Reset type-specific fields
        MaterialTypeConfig& mat = m_materialConfigs[type];
        particles[i].opacity = mat.opacityScale;
        particles[i].density = mat.densityScale;
        particles[i].lifetime = 0.0f;

        // Clamp temperature to material range
        particles[i].temperature = std::clamp(particles[i].temperature,
                                              mat.temperatureFloor,
                                              mat.temperatureCeiling);
    }

    // Upload entire buffer to GPU
    m_particleSystem->UploadParticleData();

    LOG_INFO("Particle types reassigned:");
    LOG_INFO("  Plasma: {:.1f}%", normalized[0] * 100.0f);
    LOG_INFO("  Main Sequence Star: {:.1f}%", normalized[1] * 100.0f);
    LOG_INFO("  Giant Star: {:.1f}%", normalized[2] * 100.0f);
    LOG_INFO("  Gas Cloud: {:.1f}%", normalized[3] * 100.0f);
    LOG_INFO("  Dust: {:.1f}%", normalized[4] * 100.0f);
}
```

**Performance:** Full buffer upload (10K particles × 48 bytes = 480 KB) takes ~0.5ms on RTX 4060 Ti

---

## Testing & Validation

### Unit Tests (Manual Verification)

**Test 1: Buffer Expansion**
- Verify particle buffer size increased from 320 KB (10K × 32) to 480 KB (10K × 48)
- Use PIX buffer inspection to confirm 48-byte stride
- Check no corruption in existing fields (position, velocity, temperature)

**Test 2: Material Property Upload**
- Set custom material values via ImGui
- Use PIX constant buffer view to confirm upload to b2
- Verify shader reads correct values from g_materialTypes[0-4]

**Test 3: Per-Type Rendering**
- Create 100% STAR_MAIN_SEQUENCE particles
- Verify large, bright, spherical appearance
- Compare to 100% GAS_CLOUD (should be dim, diffuse, wispy)
- Verify PLASMA_BLOB matches previous behavior (regression test)

**Test 4: Gas Cloud Dissipation**
- Set dissipationRate = 0.1 (10 second fade)
- Observe particles fade out over time
- Verify respawn after full dissipation
- Check lifetime counter increments correctly

**Test 5: Dust Heating**
- Spawn DUST_PARTICLE near center
- Observe temperature increase over time
- Verify temperature caps at temperatureCeiling (2000K)
- Check inverse square law heating rate

**Test 6: Preset System**
- Load default.json → 100% plasma
- Load accretion_disk.json → mixed types
- Save custom preset → file created
- Reload custom preset → same distribution restored

### Performance Validation

**Target:** Maintain 120+ FPS @ 1080p with 10K particles (RTX 4060 Ti)

**Overhead Estimate:**
- Buffer size increase (32 → 48 bytes): +50% memory bandwidth → ~2% FPS impact
- Per-type material lookup: 5 texture fetches/particle → ~3% FPS impact
- Dissipation logic (if-branch): Minimal (branch coherence in similar types)
- Total estimated overhead: ~5% FPS loss

**Expected:** 114-120 FPS (was 120 FPS)

**Mitigation:**
- Use material property cache in shared memory (future optimization)
- Batch particles by type for better branch coherence (future)

---

## Integration with Phase 5 Milestones

### Connection to Milestone 5.2 (Enhanced Physics)

The particle type system enables physics behaviors in Milestone 5.2:

**Constraint Shapes (5.2a):**
- STAR_MAIN_SEQUENCE → SPHERE constraint (stars orbit at fixed radius)
- GAS_CLOUD → TORUS constraint (gas in accretion torus)
- DUST_PARTICLE → DISC constraint (dust in thin disk)

**Black Hole Mass (5.2b):**
- Affects Keplerian velocity: `v = sqrt(GM/r)`
- STAR types orbit slower (larger r)
- DUST/GAS types orbit faster (smaller r)

**Alpha Viscosity (5.2c):**
- GAS_CLOUD experiences strong viscosity (inward spiral)
- DUST_PARTICLE experiences moderate viscosity
- STAR types unaffected (independent orbits)

### Connection to Milestone 5.3 (Streamlined UX)

**Bulk Light Color Controls:**
- Apply color presets to match particle type aesthetic
- "Stellar Nursery" preset → blue/white lights for hot young stars
- "Dust Torus" preset → red/orange lights for cool dusty disk

**In-Scattering Restart:**
- GAS_CLOUD type needs strong backward scattering
- DUST_PARTICLE needs Mie scattering (isotropic)
- Per-type phase function already implemented (reuse for in-scattering)

**Blackbody Completion:**
- STAR types use 100% blackbody (blackbodyWeight = 1.0)
- GAS_CLOUD uses artistic tint (blackbodyWeight = 0.1)
- Already implemented in GetParticleEmission()

---

## Animation Scenario Examples

### Scenario 1: Stellar Nursery

**Particle Type Distribution:**
- 30% STAR_MAIN_SEQUENCE (young hot stars)
- 50% GAS_CLOUD (nebula gas)
- 20% DUST_PARTICLE (protoplanetary dust)

**Material Tuning:**
- Stars: High emission (8.0), blue tint (30000K temperature)
- Gas: High scattering albedo (0.9), blue/cyan tint
- Dust: Low opacity (0.3), small radius (0.3)

**Expected Visuals:**
- Bright blue stars embedded in glowing nebula
- Wispy gas tendrils illuminated by stars
- Dark dust lanes creating contrast

---

### Scenario 2: Red Giant Engulfment

**Particle Type Distribution:**
- 1 particle: STAR_GIANT (red giant core)
- 999 particles: GAS_CLOUD (expanding envelope)

**Material Tuning:**
- Giant: Huge radius (5.0), red tint (3000K), high emission (10.0)
- Gas: Dissipation rate (0.01), red tint, low opacity (0.1)

**Expected Visuals:**
- Massive red core with diffuse expanding shell
- Gas gradually fades out as it dissipates
- Layered envelope visible through transparency

---

### Scenario 3: Accretion Disk with Dust Torus

**Particle Type Distribution:**
- 60% PLASMA_BLOB (hot inner disk)
- 30% DUST_PARTICLE (cool outer disk)
- 10% GAS_CLOUD (vertical outflows)

**Material Tuning:**
- Plasma: High temperature (20000K), blue/white
- Dust: Low temperature (500K), brown tint, high absorption
- Gas: Backward scattering, vertical velocity bias

**Expected Visuals:**
- Bright blue/white inner accretion disk
- Dark brown dust torus at larger radius
- Vertical gas plumes illuminated from below

---

## Open Questions & Future Work

### Open Technical Questions

**Q1: Material property constant buffer caching?**
- Currently reads from constant buffer every particle
- Could cache in shared memory for compute shader
- Estimated speedup: 5-10% (avoid repeated CB reads)
- **Decision:** Defer to post-Phase 5 optimization

**Q2: Dynamic type reassignment during simulation?**
- E.g., dust particles heat up → become plasma at high temp
- Would require type transition logic in physics shader
- **Decision:** Not in scope for Phase 5, consider for Phase 6

**Q3: Per-particle anisotropy override?**
- Currently anisotropy is material-type-specific
- Could add per-particle anisotropy field for finer control
- **Decision:** Not needed, material-level control is sufficient

### Future Enhancements (Phase 6+)

**Performance Optimizations:**
- Material property caching in shared memory
- Particle batching by type (better branch coherence)
- LOD system (reduce detail for distant particles)

**Visual Enhancements:**
- Motion blur for fast-moving particles
- Depth of field blur (bokeh effect)
- Chromatic aberration (atmospheric distortion)

**Physics Extensions:**
- Particle collisions (SPH-like)
- Type transitions (dust → plasma when heated)
- Gravitational lensing (light bending near black hole)

---

## Success Criteria

**Milestone 5.1 is complete when:**

1. ✅ Particle buffer expanded to 48 bytes with no regressions
2. ✅ All 5 material types render with distinct visual characteristics
3. ✅ ImGui controls allow runtime type distribution changes
4. ✅ Preset save/load system working for animation scenarios
5. ✅ Gas cloud dissipation visible over time
6. ✅ Dust particle heating near center working
7. ✅ PLASMA_BLOB matches previous behavior (regression test passes)
8. ✅ Performance maintains 114+ FPS @ 1080p with 10K particles

**Definition of Done:**
- All code committed to branch `0.8.7` (Milestone 5.1)
- Documentation updated (CLAUDE.md, README.md)
- PIX capture showing all 5 types rendering correctly
- User can create custom presets and share JSON files

---

## Appendix: File Checklist

### Files Created
- ✅ `shaders/particles/particle_types.hlsl` (material property definitions)
- ✅ `configs/particle_types/default.json` (default preset)
- ✅ `configs/particle_types/accretion_disk.json` (mixed types)
- ✅ `configs/particle_types/stellar_nursery.json` (stars + nebula)
- ✅ `configs/particle_types/dust_torus.json` (dust + plasma)

### Files Modified
- ✅ `src/particles/ParticleSystem.h` (ParticleData struct expansion)
- ✅ `src/particles/ParticleSystem.cpp` (buffer creation, upload logic)
- ✅ `shaders/particles/particle_physics.hlsl` (Particle struct, per-type physics)
- ✅ `shaders/particles/gaussian_common.hlsl` (material lookup functions)
- ✅ `shaders/particles/particle_gaussian_raytrace.hlsl` (per-type rendering)
- ✅ `src/core/Application.h` (MaterialTypeConfig, preset manager)
- ✅ `src/core/Application.cpp` (ImGui controls, save/load, upload logic)

### Files Read (No Changes)
- `src/lighting/RTXDILightingSystem.h` (no impact on RTXDI)
- `src/core/Device.h` (no device changes needed)
- `src/utils/ResourceManager.h` (uses existing descriptor allocation)

---

**Document Status:** Complete and ready for implementation
**Next Steps:** Begin Task 5.1a.1 (Expand GPU Particle Buffer)
