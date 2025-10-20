# Phase 5: Celestial Rendering - Master Roadmap

**Status:** Planning Phase
**Branch:** 0.8.7 (upcoming)
**Goal:** Transform particle system into animation-ready celestial renderer with varied particle types, enhanced physics, and streamlined controls
**Timeline:** 3-4 weeks (3 major milestones)

---

## Executive Summary

PlasmaDX-Clean has achieved production-ready volumetric RT rendering with RTXDI integration. Phase 5 pivots from technical infrastructure to **creative enablement** - unlocking animation workflows through:

1. **Particle Type System** - Stars, gas clouds, dust with distinct material properties
2. **Enhanced Physics Controls** - Animation-ready parameters (mass, viscosity, constraints)
3. **Streamlined UX** - Bulk light color controls, preset management
4. **Material Rendering** - Per-type scattering, emission, opacity behaviors

**Why Now:**
- ✅ RTXDI M4 complete (weighted reservoir sampling)
- ✅ Multi-light system operational (13-16 lights)
- ✅ Volumetric RT lighting working
- ✅ Foundation solid - time to build creative tools on top

**Deferred:**
- M5 Temporal Accumulation (infrastructure complete, visual effect missing - technical debt)
- In-scattering (broken implementation - will restart from scratch)

---

## Current State Analysis

### ✅ Existing Capabilities (Already Implemented)

**Per-Particle Variation:**
- Temperature (800K - 26000K) → Blackbody emission color
- Radius (algorithmic variation)
- Opacity (temperature-dependent)
- Anisotropic elongation (velocity-based stretching)
- Emission intensity

**Rendering Features:**
- 3D Gaussian splatting (ray-ellipsoid intersection)
- Beer-Lambert volumetric absorption
- Henyey-Greenstein phase function
- Shadow rays (PCSS with temporal filtering)
- RT particle-to-particle lighting
- RTXDI weighted reservoir sampling

**Physics Simulation:**
- Schwarzschild black hole gravity
- Keplerian orbital dynamics
- Temperature-based properties
- Anisotropic Gaussian elongation

### ❌ Missing for Animation Workflows

**Particle Diversity:**
- No distinct particle types (all use same material)
- No per-type material properties
- Color variation purely algorithmic (temperature-based)
- No artistic override controls

**Physics Limitations:**
- Single constraint shape (disc only)
- Fixed black hole mass
- No inward spiral (alpha viscosity)
- No runtime constraint switching

**UX Friction:**
- Tedious per-light color adjustment (13 lights × manual RGB)
- No bulk color presets
- In-scattering broken (never worked despite extensive debugging)
- Blackbody radiation incomplete

---

## Milestone 5.1: Particle Type System

**Duration:** 1 week
**Goal:** Foundation for varied celestial materials
**Branch:** 0.8.7

### Architecture Design

**Particle Structure Enhancement:**
```cpp
enum class ParticleType : uint32_t {
    PLASMA_BLOB = 0,      // Current behavior (baseline)
    STAR_MAIN_SEQUENCE,   // Spherical, high emission, minimal scattering
    STAR_GIANT,           // Large radius, orange/red, soft edges
    GAS_CLOUD,            // High scattering, low emission, wispy opacity
    DUST_PARTICLE,        // Anisotropic, low emission, high absorption
    COUNT
};

struct Particle {
    float3 position;      // 12 bytes
    float3 velocity;      // 12 bytes
    float temperature;    // 4 bytes
    float radius;         // 4 bytes
    uint32_t type;        // 4 bytes ← NEW!
    float materialParam;  // 4 bytes ← NEW! (type-specific property)
    float padding[2];     // 8 bytes (maintain 32-byte alignment)
    // Total: 48 bytes (was 32 bytes - GPU buffer update required)
};
```

**Material Property Tables (Shader Constants):**
```hlsl
struct MaterialProperties {
    float emissionMultiplier;   // Star: 2.0, Gas: 0.3, Dust: 0.1
    float scatteringStrength;   // Star: 0.1, Gas: 5.0, Dust: 2.0
    float phaseAnisotropy;      // Star: 0.0 (isotropic), Gas: 0.7, Dust: 0.5
    float baseOpacity;          // Star: 1.0, Gas: 0.2, Dust: 0.5
    float anisotropicStretch;   // Star: 1.0, Gas: 1.5, Dust: 3.0
    float3 colorTint;           // Artistic override (lerp with blackbody)
    float tintStrength;         // How much to blend tint vs blackbody
};

cbuffer MaterialConstants : register(b1) {
    MaterialProperties g_materials[5];  // One per ParticleType
};
```

### Implementation Tasks

**Task 1.1: Particle Buffer Update (2 hours)**
- Expand particle structure from 32 → 48 bytes
- Update particle physics shader initialization
- Add type randomization (configurable distribution)
- Test buffer creation and GPU upload

**Files Modified:**
- `src/particles/ParticleSystem.h` - Structure definition
- `src/particles/ParticleSystem.cpp` - Buffer creation
- `shaders/particles/particle_physics.hlsl` - Initialization logic

**Task 1.2: Material Property System (3 hours)**
- Create material property constant buffer
- ImGui controls for per-type material editing
- Preset system (save/load material configs)
- Apply material properties in Gaussian renderer

**Files Modified:**
- `src/particles/ParticleRenderer_Gaussian.h` - Material constants
- `src/particles/ParticleRenderer_Gaussian.cpp` - Upload logic
- `src/core/Application.cpp` - ImGui controls
- `shaders/particles/particle_gaussian_raytrace.hlsl` - Material application

**Task 1.3: Rendering Integration (4 hours)**
- Per-type emission calculation
- Per-type scattering behavior
- Per-type opacity modulation
- Per-type phase function (HG parameter override)
- Color tint blending (artistic override)

**Files Modified:**
- `shaders/particles/particle_gaussian_raytrace.hlsl` - Core rendering loop
- `shaders/particles/gaussian_common.hlsl` - Helper functions

### ImGui Control Specification

```
Material Properties
├─ Particle Type Distribution
│  ├─ Plasma Blob: [Slider 0-100%] (default: 40%)
│  ├─ Main Sequence Star: [Slider 0-100%] (default: 20%)
│  ├─ Giant Star: [Slider 0-100%] (default: 10%)
│  ├─ Gas Cloud: [Slider 0-100%] (default: 20%)
│  └─ Dust Particle: [Slider 0-100%] (default: 10%)
│
├─ [Dropdown: Select Material Type]
│
├─ Star Properties (when Star type selected)
│  ├─ Emission Multiplier: [0.1 - 10.0] (default: 2.0)
│  ├─ Scattering: [0.0 - 5.0] (default: 0.1)
│  ├─ Phase Anisotropy: [-1.0 - 1.0] (default: 0.0, isotropic)
│  ├─ Base Opacity: [0.0 - 1.0] (default: 1.0)
│  ├─ Anisotropic Stretch: [1.0 - 5.0] (default: 1.0)
│  ├─ Color Tint: [RGB Picker]
│  └─ Tint Strength: [0.0 - 1.0] (0 = pure blackbody, 1 = pure tint)
│
├─ Gas Cloud Properties
│  ├─ Emission Multiplier: [0.0 - 2.0] (default: 0.3)
│  ├─ Scattering: [0.0 - 10.0] (default: 5.0)
│  ├─ Phase Anisotropy: [0.0 - 1.0] (default: 0.7, forward scattering)
│  ├─ Base Opacity: [0.0 - 1.0] (default: 0.2, wispy)
│  ├─ Opacity Noise: [0.0 - 1.0] (adds perlin noise variation)
│  ├─ Color Tint: [RGB Picker] (nebula colors)
│  └─ Tint Strength: [0.0 - 1.0]
│
├─ Dust Properties
│  ├─ Emission Multiplier: [0.0 - 1.0] (default: 0.1)
│  ├─ Scattering: [0.0 - 5.0] (default: 2.0)
│  ├─ Absorption: [0.0 - 1.0] (default: 0.8, opaque)
│  ├─ Anisotropic Stretch: [1.0 - 5.0] (default: 3.0, elongated)
│  ├─ Metallic Tint: [RGB Picker]
│  └─ Tint Strength: [0.0 - 1.0]
│
└─ [Button: Apply to All Particles] [Button: Reset to Defaults]
```

### Visual Validation Tests

**Test 1: Single Type Rendering**
- Set 100% Main Sequence Star
- Verify spherical, bright, minimal scattering
- Check emission multiplier effect (0.1 → 10.0)

**Test 2: Mixed Type Distribution**
- 30% Stars, 40% Gas, 30% Dust
- Verify distinct visual appearance per type
- Check transitions smooth (no popping)

**Test 3: Color Tint Override**
- Set Gas Cloud tint to cyan (nebula effect)
- Tint strength 0.0 → 1.0 transition
- Verify blackbody → artistic color blend

**Test 4: Material Extreme Values**
- Gas Cloud scattering = 10.0 (maximum wispy)
- Dust anisotropic = 5.0 (extreme elongation)
- Star emission = 10.0 (supernova-like)

### Deliverables (Milestone 5.1)

✅ 5 particle types fully differentiated
✅ Per-type material properties adjustable at runtime
✅ ImGui controls for all parameters
✅ Visual distinction clear and dramatic
✅ Preset save/load system (JSON)
✅ Documentation: `PARTICLE_TYPE_SYSTEM_SPEC.md`

---

## Milestone 5.2: Enhanced Physics Controls

**Duration:** 1 week
**Goal:** Animation-ready physics parameters
**Branch:** 0.8.8

### Features

**Feature 2.1: Constraint Shape System**

**Enum Definition:**
```cpp
enum class ConstraintShape {
    SPHERE,          // Nebula expansion, supernova remnant
    DISC,            // Accretion disk (current default)
    TORUS,           // Ring systems (Saturn, protoplanetary disk)
    ACCRETION_DISK,  // Realistic Keplerian flow with infall
};
```

**Physics Shader Implementation:**
```hlsl
// Constraint force calculation (per-frame)
float3 CalculateConstraintForce(Particle p, ConstraintShape shape) {
    switch (shape) {
        case SPHERE:
            // Radial expansion from origin
            float targetRadius = g_sphereRadius;
            float distFromOrigin = length(p.position);
            float3 radialDir = normalize(p.position);
            return radialDir * (targetRadius - distFromOrigin) * g_expansionSpeed;

        case DISC:
            // Flatten to XZ plane (current behavior)
            float3 toPlane = float3(0, -p.position.y, 0);
            return toPlane * g_flatteningStrength;

        case TORUS:
            // Ring at radius R, thickness T
            float2 xz = p.position.xz;
            float distFromRing = length(xz) - g_torusRadius;
            float3 toRing = normalize(float3(xz.x, 0, xz.y)) * -distFromRing;
            float3 toPlane = float3(0, -p.position.y, 0);
            return toRing * g_ringStrength + toPlane * g_thicknessStrength;

        case ACCRETION_DISK:
            // Keplerian flow + alpha viscosity inward spiral
            // (Implemented in Feature 2.3)
    }
}
```

**ImGui Controls:**
```
Physics Configuration
├─ Constraint Shape: [Dropdown: Sphere | Disc | Torus | Accretion Disk]
├─ Sphere Parameters (when Sphere selected)
│  ├─ Radius: [10 - 500 units] (default: 200)
│  └─ Expansion Speed: [0.0 - 10.0] (default: 1.0)
├─ Disc Parameters
│  ├─ Inner Radius: [10 - 300] (default: 10)
│  ├─ Outer Radius: [100 - 500] (default: 300)
│  └─ Thickness: [10 - 100] (default: 50)
├─ Torus Parameters
│  ├─ Ring Radius: [100 - 400] (default: 250)
│  ├─ Ring Thickness: [10 - 100] (default: 30)
│  └─ Ring Strength: [0.1 - 10.0] (default: 1.0)
```

**Animation Use Case:**
- Start: Compact sphere (radius 50)
- Over 10 seconds: Expand radius 50 → 400 (supernova remnant)
- Particle temperature cools as radius increases (adiabatic expansion)

---

**Feature 2.2: Black Hole Mass Parameter**

**Physics Impact:**
```hlsl
// Keplerian velocity: v = sqrt(GM/r)
float CalculateKeplerianSpeed(float radius, float blackHoleMass) {
    const float G = 6.674e-11;  // Gravitational constant
    const float solarMass = 1.989e30;  // kg
    float M = blackHoleMass * solarMass;
    return sqrt(G * M / radius);
}

// Orbital period: T = 2π * sqrt(r^3 / GM)
float CalculateOrbitalPeriod(float radius, float blackHoleMass) {
    const float G = 6.674e-11;
    const float solarMass = 1.989e30;
    float M = blackHoleMass * solarMass;
    return 2.0 * PI * sqrt(pow(radius, 3) / (G * M));
}
```

**ImGui Controls:**
```
Black Hole Properties
├─ Mass: [1e6 - 1e9 solar masses] (default: 1e7, log scale slider)
├─ Schwarzschild Radius: [Computed: 2GM/c²] (display only)
├─ Inner Stable Orbit (ISCO): [Computed: 3 × Schwarzschild] (display only)
└─ [Checkbox] Show Event Horizon Sphere
```

**Animation Use Case:**
- Start: Mass = 1e6 (slow orbital speeds)
- Over 20 seconds: Increase to 1e8 (speeds increase, particles spiral inward)
- Visual: Accretion disk brightens as infall accelerates

---

**Feature 2.3: Alpha Viscosity (Shakura-Sunyaev)**

**Physics Implementation:**
```hlsl
// Alpha viscosity: drives inward mass transport
float3 CalculateViscousForce(Particle p, float alpha) {
    // Shakura-Sunyaev α parameter (0.01 - 1.0)
    // Higher α → faster inward spiral

    float radius = length(p.position.xz);
    float keplerianSpeed = CalculateKeplerianSpeed(radius, g_blackHoleMass);

    // Viscous torque causes angular momentum transfer
    // Net effect: inward radial velocity
    float inwardSpeed = alpha * keplerianSpeed * 0.01;  // Damped

    float3 inwardDir = -normalize(p.position);
    return inwardDir * inwardSpeed;
}

// Update particle velocity each frame
p.velocity += CalculateViscousForce(p, g_alphaViscosity) * deltaTime;
```

**ImGui Controls:**
```
Accretion Physics
├─ Alpha Viscosity: [0.01 - 1.0] (default: 0.1, log scale)
├─ Inward Drift Speed: [Computed] (display only)
└─ Accretion Timescale: [Computed: ~years] (display only)
```

**Animation Use Case:**
- Alpha = 0.01: Slow, stable disk
- Increase to Alpha = 0.5: Visible inward spiral over 30 seconds
- Particles cross ISCO → temperature spikes (final plunge)

---

**Feature 2.4: Velocity-Based Temperature Heating**

**Physics Model:**
```hlsl
// Shear heating from velocity gradients
float CalculateShearHeating(Particle p) {
    float radius = length(p.position.xz);
    float speed = length(p.velocity);
    float keplerianSpeed = CalculateKeplerianSpeed(radius, g_blackHoleMass);

    // Deviation from Keplerian → turbulence → heating
    float velocityDeviation = abs(speed - keplerianSpeed);
    float heatingRate = velocityDeviation * g_shearHeatingFactor;

    return heatingRate;
}

// Radiative cooling (Stefan-Boltzmann)
float CalculateRadiativeCooling(float temperature) {
    const float sigma = 5.67e-8;  // Stefan-Boltzmann constant
    return sigma * pow(temperature, 4) * g_coolingFactor;
}

// Update temperature each frame
float heating = CalculateShearHeating(p);
float cooling = CalculateRadiativeCooling(p.temperature);
p.temperature += (heating - cooling) * deltaTime;
```

**ImGui Controls:**
```
Temperature Dynamics
├─ Shear Heating Factor: [0.0 - 10.0] (default: 1.0)
├─ Radiative Cooling Factor: [0.0 - 1.0] (default: 0.1)
└─ [Checkbox] Enable Temperature Evolution
```

**Animation Use Case:**
- Enable shear heating + cooling
- Particles in inner disk: High velocity → heating → white-hot
- Particles in outer disk: Low velocity → cooling → red
- Dynamic color gradient emerges naturally

---

### Deliverables (Milestone 5.2)

✅ 4 constraint shapes runtime switchable
✅ Black hole mass adjustable (1e6 - 1e9 solar masses)
✅ Alpha viscosity inward spiral visible
✅ Velocity-based temperature dynamics
✅ All physics parameters runtime adjustable
✅ Animation presets saved to JSON
✅ Documentation: `PHYSICS_ANIMATION_GUIDE.md`

---

## Milestone 5.3: Streamlined UX & Material Rendering

**Duration:** 1 week
**Goal:** Remove UX friction, polish rendering
**Branch:** 0.8.9

### Feature 3.1: Bulk Light Color Controls

**Current Problem:**
- 13 lights × manual RGB adjustment = tedious
- No bulk operations
- Preset colors require editing each light individually

**Solution: Color Palette System**

**ImGui Interface:**
```
Light Color Presets
├─ [Dropdown: Stellar Ring | Binary Star | Nebula Glow | Custom]
│
├─ Quick Color Palette (when Custom selected)
│  ├─ Primary Color: [RGB Picker]
│  ├─ Secondary Color: [RGB Picker]
│  ├─ Accent Color: [RGB Picker]
│  └─ [Button: Apply Gradient to All Lights]
│
├─ Advanced Palette Controls
│  ├─ Gradient Mode: [Linear | Radial | Random]
│  ├─ Color Temperature Range: [2000K - 20000K]
│  ├─ Hue Shift: [-180° - 180°] (rotate entire palette)
│  └─ Saturation: [0.0 - 2.0] (desaturate/boost)
│
└─ Per-Light Override (expandable section)
   ├─ Light 0: [Mini RGB Picker]
   ├─ Light 1: [Mini RGB Picker]
   └─ ... (collapses by default)
```

**Preset Definitions:**
```cpp
struct LightColorPreset {
    std::string name;
    std::vector<float3> colors;  // RGB per light
};

// Built-in presets
presets["Stellar Ring"] = {
    // 13 lights in gradient from blue (inner) → red (outer)
    colors: {
        {0.5, 0.7, 1.0},   // Blue-white (inner)
        {0.6, 0.75, 1.0},
        // ... gradient ...
        {1.0, 0.5, 0.3},   // Orange (outer)
    }
};

presets["Binary Star"] = {
    // 2 distinct colors for binary system
    colors: {
        {0.7, 0.85, 1.0},  // Blue primary
        {1.0, 0.6, 0.4},   // Orange secondary
        // ... fill remaining with dim tertiary
    }
};

presets["Nebula Glow"] = {
    // Cyan/magenta nebula aesthetic
    colors: {
        {0.3, 1.0, 1.0},   // Cyan
        {1.0, 0.3, 1.0},   // Magenta
        {0.3, 1.0, 0.6},   // Teal
        // ... randomized variations
    }
};
```

**Implementation Tasks:**
- Add preset system (save/load JSON)
- Gradient application function (linear, radial, random)
- Hue shift / saturation bulk operations
- ImGui collapsible per-light override section

---

### Feature 3.2: In-Scattering Restart (From Scratch)

**Current Status:** Broken, debugged extensively, never worked

**Decision:** Complete restart with modern approach

**New Architecture:**

**Phase Function Integration:**
```hlsl
// CURRENT (WORKING): Henyey-Greenstein phase function
float HenyeyGreenstein(float cosTheta, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5));
}

// NEW: In-scattering with multiple light sources
float3 CalculateInScattering(RayMarchState state, float3 viewDir) {
    float3 inScatter = float3(0, 0, 0);

    // For each light source
    for (uint i = 0; i < g_lightCount; i++) {
        Light light = g_lights[i];

        float3 lightDir = normalize(light.position - state.position);
        float cosTheta = dot(lightDir, -viewDir);  // Scattering angle

        // Phase function (Henyey-Greenstein)
        float phase = HenyeyGreenstein(cosTheta, g_phaseAnisotropy);

        // Light attenuation
        float lightDist = length(light.position - state.position);
        float attenuation = 1.0 / (1.0 + lightDist * g_attenuationFactor);

        // In-scattered radiance
        float3 lightRadiance = light.color * light.intensity * attenuation;
        inScatter += lightRadiance * phase * g_scatteringCoefficient;
    }

    return inScatter;
}

// Integrate along ray march
totalEmission += CalculateInScattering(state, viewDir) * absorptionCoeff * stepSize;
```

**Key Differences from Old Implementation:**
1. ✅ Uses existing working phase function (HG)
2. ✅ Multi-light support (not just primary light)
3. ✅ Proper Beer-Lambert integration
4. ✅ Scattering coefficient per material type
5. ✅ No separate "in-scattering buffer" (inline calculation)

**ImGui Controls:**
```
In-Scattering (New Implementation)
├─ [Checkbox] Enable In-Scattering
├─ Scattering Coefficient: [0.0 - 10.0] (default: 1.0)
├─ Attenuation Factor: [0.0001 - 0.01] (default: 0.001)
├─ Phase Anisotropy (g): [-1.0 - 1.0] (default: 0.5, forward)
└─ [Info] Uses Henyey-Greenstein phase function (validated)
```

**Validation Tests:**
- Single light → should see glow around particles
- Multiple lights → verify additive scattering
- Phase anisotropy -1.0 (backscatter) vs 1.0 (forward scatter)
- Compare with phase function (F7) - should be complementary

---

### Feature 3.3: Blackbody Radiation Completion

**Current Status:** Incomplete implementation

**Missing Features:**
1. Wien's displacement law (peak wavelength)
2. Stefan-Boltzmann law (total radiated power)
3. Color temperature presets
4. Artistic override blending

**Complete Implementation:**

```hlsl
// Planck's law for blackbody spectrum
float3 BlackbodySpectrum(float temperature) {
    // Simplified RGB approximation (fast, GPU-friendly)
    // Based on color temperature → RGB lookup tables

    if (temperature < 1000.0) {
        return float3(1.0, 0.0, 0.0);  // Deep red
    } else if (temperature < 2000.0) {
        return lerp(float3(1.0, 0.0, 0.0), float3(1.0, 0.3, 0.0), (temperature - 1000.0) / 1000.0);
    } else if (temperature < 6500.0) {
        // Orange → white transition
        float t = (temperature - 2000.0) / 4500.0;
        return lerp(float3(1.0, 0.3, 0.0), float3(1.0, 1.0, 1.0), t);
    } else if (temperature < 10000.0) {
        // White → blue transition
        float t = (temperature - 6500.0) / 3500.0;
        return lerp(float3(1.0, 1.0, 1.0), float3(0.7, 0.85, 1.0), t);
    } else {
        // Hot blue stars
        return float3(0.5, 0.7, 1.0);
    }
}

// Wien's displacement law: λ_peak = b / T
float WiensDisplacement(float temperature) {
    const float b = 2.897e-3;  // Wien's constant (m·K)
    return b / temperature;  // Peak wavelength in meters
}

// Stefan-Boltzmann: Power = σ * T^4 * Area
float StefanBoltzmannPower(float temperature, float radius) {
    const float sigma = 5.67e-8;  // W/(m²·K⁴)
    float area = 4.0 * PI * radius * radius;
    return sigma * pow(temperature, 4) * area;
}

// Apply to emission calculation
float3 emission = BlackbodySpectrum(particle.temperature);
emission *= StefanBoltzmannPower(particle.temperature, particle.radius);

// Artistic override blend
emission = lerp(emission, g_artisticColor, g_artisticBlend);
```

**ImGui Controls:**
```
Blackbody Radiation
├─ [Checkbox] Enable Physical Blackbody
├─ Temperature Range: [800K - 26000K] (adjust particle physics)
├─ Emission Intensity: [0.1 - 10.0] (multiplier)
├─ Artistic Override
│  ├─ Color: [RGB Picker]
│  └─ Blend: [0.0 - 1.0] (0 = pure blackbody, 1 = pure artistic)
└─ [Info] Peak Wavelength: [Computed from Wien's law]
```

---

### Deliverables (Milestone 5.3)

✅ Bulk light color controls (presets + gradients)
✅ In-scattering working (new implementation)
✅ Blackbody radiation complete (Wien's + Stefan-Boltzmann)
✅ All UX friction points resolved
✅ Documentation: `UX_IMPROVEMENTS_SUMMARY.md`

---

## Animation Scenario Library

**Scenario 1: Binary Star Gas Transfer**
- Particle types: 40% Gas Cloud, 40% Plasma, 20% Dust
- Constraint: TORUS (gas stream bridge)
- Physics: Alpha = 0.3 (moderate transfer rate)
- Lights: 2 stars (blue primary, orange secondary)
- Duration: 30 seconds
- Camera: Orbital path around binary system

**Scenario 2: Supernova Remnant Expansion**
- Particle types: 60% Gas Cloud, 30% Dust, 10% Star remnants
- Constraint: SPHERE (expanding shell)
- Physics: Sphere radius 50 → 400 over 20 seconds
- Lights: Single central neutron star (blue-white, pulsing)
- Temperature: Hot inner (10000K) → cool outer (2000K)
- Duration: 20 seconds
- Camera: Slow zoom out

**Scenario 3: Accretion Disk Flare**
- Particle types: 50% Plasma, 30% Gas, 20% Dust
- Constraint: ACCRETION_DISK
- Physics: Black hole mass 1e7 → 1e8 (increases over 30s)
- Alpha viscosity: 0.1 → 0.8 (infall accelerates)
- Lights: 13-light ring (gradient blue → red)
- Temperature: Shear heating enabled (inner disk brightens)
- Duration: 40 seconds
- Camera: Fixed side view

**Scenario 4: Protoplanetary Disk Formation**
- Particle types: 70% Dust, 20% Gas, 10% Plasma
- Constraint: DISC → TORUS transition
- Physics: Start chaotic, settle into Keplerian orbits
- Lights: Single central protostar (yellow)
- Temperature: Cooling over time (26000K → 4000K)
- Duration: 60 seconds
- Camera: Slow rotation above disk

**Scenario 5: Stellar Nebula (Planetary Nebula)**
- Particle types: 80% Gas Cloud, 20% Dust
- Constraint: SPHERE (slow expansion)
- Physics: Sphere radius fixed at 300, opacity noise high
- Lights: Central white dwarf (blue-white, high intensity)
- Material: Gas cloud tint = cyan/magenta gradient
- Duration: 45 seconds
- Camera: Slow fly-through

---

## Agent SDK Strategy

### Specialized Agent Development

**Agent 1: celestial-rendering-specialist**
- **Purpose:** Particle type system design and material properties
- **Tools:** Read, Write, Edit, Bash, Glob, Grep
- **Expertise:** Volumetric rendering, material properties, HLSL shaders
- **Use Case:** "Design material property table for gas cloud particles"

**Agent 2: physics-animation-engineer**
- **Purpose:** Physics parameter implementation and animation design
- **Tools:** Read, Write, Edit, Bash, Glob, Grep
- **Expertise:** Orbital mechanics, Shakura-Sunyaev viscosity, constraint shapes
- **Use Case:** "Implement alpha viscosity inward spiral in particle physics shader"

**Agent 3: volumetric-materials-researcher**
- **Purpose:** Research advanced rendering techniques
- **Tools:** WebSearch, WebFetch, Read, Write
- **Expertise:** Phase functions, scattering models, opacity variations
- **Use Case:** "Research noise-based opacity modulation for gas clouds"

### Multi-Agent Workflow Example

```bash
# Phase 1: Design (parallel research)
@celestial-rendering-specialist "Analyze existing particle rendering pipeline"
@volumetric-materials-researcher "Research phase function variations for different particle types"

# Phase 2: Architecture
/feature-dev:feature-dev "Particle type system with per-type material properties"

# Phase 3: Implementation (sequential, dependency-based)
@celestial-rendering-specialist "Implement material property constant buffer system"
@physics-animation-engineer "Add constraint shape enum and physics calculations"

# Phase 4: Optimization
@physics-performance-agent-v2 "Optimize alpha viscosity compute shader"
@performance-optimizer:performance-engineer "Profile particle type rendering overhead"

# Phase 5: Validation
@feature-dev:code-reviewer "Review particle type system implementation"
```

---

## Technical Debt & Known Issues

### Deferred to Post-Phase 5

**M5 Temporal Accumulation:**
- **Status:** Infrastructure complete, visual effect missing
- **Issue:** Ping-pong buffers implemented correctly, but no visible smoothing
- **Next Step:** PIX buffer inspection to debug shader logic
- **Priority:** Low (temporal smoothing is polish, not core feature)
- **Documentation:** `M5_TEMPORAL_ACCUMULATION_DEFERRED.md`

**In-Scattering (Old Implementation):**
- **Status:** Broken, debugged extensively, never worked
- **Decision:** Complete restart in Milestone 5.3
- **Reason:** Old implementation had fundamental architectural issues
- **New Approach:** Inline calculation using validated phase function

**Brightness Issues:**
- **Status:** External lighting multipliers still require tuning
- **Workaround:** Users can adjust emission strength + lighting strength
- **Root Cause:** Multi-light illumination values reach 50+, requires scaling
- **Priority:** Medium (functional but not optimal)

---

## Success Metrics

### Milestone 5.1 Success Criteria
- [ ] 5 particle types visually distinct
- [ ] Material properties adjustable without recompilation
- [ ] 60+ FPS maintained @ 10K particles (RTX 4060 Ti)
- [ ] User can create custom material presets in <5 minutes

### Milestone 5.2 Success Criteria
- [ ] All 4 constraint shapes working
- [ ] Black hole mass change produces visible effect
- [ ] Alpha viscosity inward spiral visible within 10 seconds
- [ ] Animation scenario playback smooth (no stuttering)

### Milestone 5.3 Success Criteria
- [ ] Bulk light color change takes <10 seconds (was ~2 minutes)
- [ ] In-scattering working with all light presets
- [ ] Blackbody radiation matches reference color tables
- [ ] Zero UX friction points reported by user

### Overall Phase 5 Success
- [ ] User can create animation in <30 minutes (setup → record)
- [ ] 5+ animation scenarios documented and validated
- [ ] Codebase ready for creative workflows (not just technical demos)
- [ ] Documentation complete and user-accessible

---

## Dependencies & Prerequisites

### Required Before Starting
- ✅ RTXDI M4 complete (weighted reservoir sampling)
- ✅ Multi-light system operational
- ✅ Volumetric RT lighting working
- ✅ Build system stable (Debug + DebugPIX)

### External Dependencies
- DirectX 12 Agility SDK 1.618.2 (already upgraded)
- HLSL Shader Model 6.5+ (current)
- PIX for Windows (GPU debugging)
- JSON library (for preset save/load)

### Knowledge Dependencies
- HLSL compute shader programming (have)
- DirectX 12 resource management (have)
- Orbital mechanics basics (need research for Feature 2.2-2.4)
- Color science (blackbody spectra, Wien's law) (need reference tables)

---

## Risk Assessment

### Low Risk (Proven Techniques)
- ✅ Particle type enum expansion (standard GPU buffer update)
- ✅ Material property constant buffers (already using extensively)
- ✅ ImGui controls (established pattern)
- ✅ Bulk light color operations (simple array manipulation)

### Medium Risk (New Implementation)
- ⚠️ In-scattering restart (old implementation failed, but new approach simpler)
- ⚠️ Alpha viscosity physics (requires careful Keplerian math)
- ⚠️ Constraint shape physics (multiple implementations to validate)

### High Risk (Complex Integration)
- 🔴 Performance impact of per-particle material evaluation (mitigate: GPU branching optimization)
- 🔴 Animation timeline system (complex state management, defer to Phase 6)

### Mitigation Strategies
- **Performance:** Profile early, optimize per-type branching
- **In-scattering:** Validate with single light first, then multi-light
- **Physics:** Implement constraint shapes incrementally (SPHERE first, then others)
- **Timeline:** Defer to Phase 6, use manual parameter adjustment for Phase 5

---

## Next Session Preparation

### Before Next Window (Documentation to Create)
1. ✅ `PHASE_5_CELESTIAL_RENDERING_PLAN.md` (this document)
2. ⏳ `PARTICLE_TYPE_SYSTEM_SPEC.md` (technical specification)
3. ⏳ `M5_TEMPORAL_ACCUMULATION_DEFERRED.md` (status + future work)
4. ⏳ `ANIMATION_SCENARIOS.md` (detailed scenario library)

### First Actions in Next Window
1. Create branch `0.8.7` from `0.8.6`
2. Implement particle structure expansion (32 → 48 bytes)
3. Create material property constant buffer
4. Add basic ImGui controls for material type distribution
5. Validate GPU buffer update works correctly

### Questions to Resolve
- Particle buffer size: 32 → 48 bytes acceptable? (50% increase)
- Material property count: 5 types × 8 properties = manageable?
- Performance target: Maintain 60+ FPS with per-particle type branching?
- Animation recording: Use external tool (OBS) or implement in-app?

---

**End of Phase 5 Master Roadmap**
**Last Updated:** 2025-10-20
**Status:** Planning Complete, Ready for Implementation
