# Luminous Star Particles - Shader Modifications

**Document Version:** 1.0
**Created:** December 2025
**Status:** Reference Documentation

---

## Overview

This document details all HLSL shader modifications required for the Luminous Star Particles feature. Changes are minimal but critical for proper operation.

---

## Table of Contents

1. [File Summary](#1-file-summary)
2. [particle_gaussian_raytrace.hlsl](#2-particle_gaussian_raytrachlsl)
3. [particle_physics.hlsl](#3-particle_physicshlsl)
4. [gaussian_common.hlsl](#4-gaussian_commonhlsl)
5. [Shader Compilation](#5-shader-compilation)
6. [Validation Checklist](#6-validation-checklist)

---

## 1. File Summary

| File | Change Type | Lines Changed | Description |
|------|-------------|---------------|-------------|
| `particle_gaussian_raytrace.hlsl` | Constant | 1 | MAX_LIGHTS 16→32 |
| `particle_physics.hlsl` | Init logic | 5 | Set star material type |
| `gaussian_common.hlsl` | None | 0 | Material enum referenced only |

**Total Shader Changes:** ~6 lines

---

## 2. particle_gaussian_raytrace.hlsl

**Location:** `shaders/particles/particle_gaussian_raytrace.hlsl`

### 2.1 MAX_LIGHTS Constant

**Current Code (around line 15):**
```hlsl
#define MAX_LIGHTS 16
```

**Modified Code:**
```hlsl
#define MAX_LIGHTS 32  // Doubled for star particle lights
```

### 2.2 Light Buffer Declaration

**Location:** Line ~146

No change needed - StructuredBuffer is dynamically sized:
```hlsl
StructuredBuffer<Light> g_lights : register(t4);  // No size limit in declaration
```

### 2.3 Light Loop

**Location:** Lines ~1510-1654

No change needed - loop uses `lightCount` from constant buffer:
```hlsl
for (uint lightIdx = 0; lightIdx < lightCount; lightIdx++) {
    Light light = g_lights[lightIdx];
    // ... lighting calculation
}
```

The `lightCount` variable from the constant buffer controls iteration count, so increasing MAX_LIGHTS only affects buffer allocation, not shader logic.

### 2.4 Complete Change Diff

```diff
--- a/shaders/particles/particle_gaussian_raytrace.hlsl
+++ b/shaders/particles/particle_gaussian_raytrace.hlsl
@@ -12,7 +12,7 @@
 // ...

-#define MAX_LIGHTS 16
+#define MAX_LIGHTS 32  // Doubled for star particle lights (16 stars + 16 static)

 // ...
```

---

## 3. particle_physics.hlsl

**Location:** `shaders/particles/particle_physics.hlsl`

### 3.1 Star Particle Initialization

This change sets the material type for star particles during the first frame when particles are initialized.

**Location:** In the initialization block (where `totalTime < 0.01`)

**Current Code Structure (around lines 200-250):**
```hlsl
// First-frame initialization
if (totalTime < 0.01f) {
    // Initialize particle position (accretion disk distribution)
    float radius = ...;
    float angle = ...;
    g_particles[index].position = float3(cos(angle) * radius, height, sin(angle) * radius);

    // Initialize velocity (Keplerian orbit)
    float orbitalSpeed = sqrt(GM / radius);
    g_particles[index].velocity = float3(-sin(angle), 0, cos(angle)) * orbitalSpeed;

    // Initialize temperature based on radius
    g_particles[index].temperature = lerp(30000.0, 5000.0, normalizedRadius);

    // Default material type
    g_particles[index].materialType = 0;  // PLASMA
}
```

**Modified Code:**
```hlsl
// First-frame initialization
if (totalTime < 0.01f) {
    // Initialize particle position (accretion disk distribution)
    float radius = ...;
    float angle = ...;
    g_particles[index].position = float3(cos(angle) * radius, height, sin(angle) * radius);

    // Initialize velocity (Keplerian orbit)
    float orbitalSpeed = sqrt(GM / radius);
    g_particles[index].velocity = float3(-sin(angle), 0, cos(angle)) * orbitalSpeed;

    // Initialize temperature based on radius
    g_particles[index].temperature = lerp(30000.0, 5000.0, normalizedRadius);

    // === STAR PARTICLE INITIALIZATION ===
    // First 16 particles are supergiant stars with light sources
    if (index < 16) {
        g_particles[index].materialType = 8;  // SUPERGIANT_STAR
        g_particles[index].temperature = 25000.0;  // Hot blue-white
        g_particles[index].density = 1.5;  // Higher density for visibility
        g_particles[index].flags = g_particles[index].flags | 0x4;  // FLAG_IMMORTAL
    } else {
        g_particles[index].materialType = 0;  // PLASMA (default)
    }
}
```

### 3.2 Material Type Enum Reference

The shader needs to understand material type indices. Add a comment block for clarity:

```hlsl
// Material Type Indices (must match ParticleMaterialType enum in C++)
// 0 = PLASMA
// 1 = STAR_MAIN_SEQUENCE
// 2 = GAS_CLOUD
// 3 = ROCKY_BODY
// 4 = ICY_BODY
// 5 = SUPERNOVA
// 6 = STELLAR_FLARE
// 7 = SHOCKWAVE
// 8 = SUPERGIANT_STAR (NEW)
```

### 3.3 Complete Change Diff

```diff
--- a/shaders/particles/particle_physics.hlsl
+++ b/shaders/particles/particle_physics.hlsl
@@ -15,6 +15,17 @@
 // Particle flags
 #define FLAG_EXPLOSION    (1 << 0)
 #define FLAG_FADING       (1 << 1)
 #define FLAG_IMMORTAL     (1 << 2)
 #define FLAG_EMISSIVE_ONLY (1 << 3)
+
+// Material Type Indices (must match ParticleMaterialType enum in C++)
+// 0 = PLASMA
+// 1 = STAR_MAIN_SEQUENCE
+// 2 = GAS_CLOUD
+// 3 = ROCKY_BODY
+// 4 = ICY_BODY
+// 5 = SUPERNOVA
+// 6 = STELLAR_FLARE
+// 7 = SHOCKWAVE
+// 8 = SUPERGIANT_STAR

 // ...

@@ -220,7 +231,15 @@
     g_particles[index].temperature = lerp(30000.0, 5000.0, normalizedRadius);

-    // Default material type
-    g_particles[index].materialType = 0;  // PLASMA
+    // === STAR PARTICLE INITIALIZATION ===
+    // First 16 particles are supergiant stars with light sources
+    if (index < 16) {
+        g_particles[index].materialType = 8;  // SUPERGIANT_STAR
+        g_particles[index].temperature = 25000.0;  // Hot blue-white
+        g_particles[index].density = 1.5;  // Higher density for visibility
+        g_particles[index].flags = g_particles[index].flags | FLAG_IMMORTAL;
+    } else {
+        g_particles[index].materialType = 0;  // PLASMA (default)
+    }
 }
```

---

## 4. gaussian_common.hlsl

**Location:** `shaders/particles/gaussian_common.hlsl`

### 4.1 No Changes Required

The material lookup in `gaussian_common.hlsl` already supports any material index:

```hlsl
// Existing code - no changes needed
MaterialTypeProperties mat = g_materials[p.materialType];  // Works for index 0-8
```

The C++ side handles the expanded material array (9 materials instead of 8).

### 4.2 Verification Points

Verify these existing features will work with SUPERGIANT_STAR:

1. **Emission calculation:**
```hlsl
float3 emission = TemperatureToEmission(p.temperature) * mat.emissionMultiplier;
// mat.emissionMultiplier = 15.0 for SUPERGIANT_STAR → High emission
```

2. **Opacity handling:**
```hlsl
float extinction = mat.opacity * lifetimeFade;
// mat.opacity = 0.5 for SUPERGIANT_STAR → Semi-transparent
```

3. **Scattering:**
```hlsl
float scattering = mat.scatteringCoefficient;
float phase = HenyeyGreenstein(dot(-rayDir, lightDir), mat.phaseG);
// mat.phaseG = 0.0 → Isotropic scattering
```

---

## 5. Shader Compilation

### 5.1 Manual Compilation Commands

After modifying shaders, recompile using DXC:

**particle_gaussian_raytrace.hlsl:**
```bash
"/mnt/c/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/dxc.exe" \
    -T cs_6_5 \
    -E main \
    shaders/particles/particle_gaussian_raytrace.hlsl \
    -Fo build/bin/Debug/shaders/particles/particle_gaussian_raytrace.dxil \
    -I shaders
```

**particle_physics.hlsl:**
```bash
"/mnt/c/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/dxc.exe" \
    -T cs_6_5 \
    -E main \
    shaders/particles/particle_physics.hlsl \
    -Fo build/bin/Debug/shaders/particles/particle_physics.dxil \
    -I shaders
```

### 5.2 Build System Compilation

Alternatively, rebuild the entire project to trigger shader compilation:

```bash
"/mnt/c/Program Files/Microsoft Visual Studio/2022/Community/MSBuild/Current/Bin/MSBuild.exe" \
    build/PlasmaDX-Clean.sln \
    /p:Configuration=Debug \
    /p:Platform=x64 \
    /t:PlasmaDX-Clean \
    /v:minimal
```

### 5.3 Verify Compilation Success

Check that .dxil files are newer than .hlsl source files:

```bash
ls -la build/bin/Debug/shaders/particles/particle_gaussian_raytrace.dxil
ls -la shaders/particles/particle_gaussian_raytrace.hlsl

# dxil timestamp should be >= hlsl timestamp
```

---

## 6. Validation Checklist

### 6.1 Pre-Implementation

- [ ] Backup current shader files
- [ ] Note current .dxil timestamps
- [ ] Take screenshot of current rendering (baseline)

### 6.2 After Modification

- [ ] particle_gaussian_raytrace.hlsl: MAX_LIGHTS = 32
- [ ] particle_physics.hlsl: Star particle init block added
- [ ] Shaders compile without errors
- [ ] .dxil files updated (check timestamps)

### 6.3 Runtime Validation

- [ ] Application launches without crash
- [ ] Particles render correctly
- [ ] First 16 particles appear brighter (SUPERGIANT_STAR material)
- [ ] Light count shows 16+ in ImGui
- [ ] No visual artifacts or corruption

### 6.4 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Particles disappear | Material index out of bounds | Verify C++ has 9 materials |
| No brightness change | Stale .dxil | Recompile shaders |
| Crash on startup | Buffer size mismatch | Check MAX_LIGHTS in C++ and HLSL |
| Stars not moving | Init block not reached | Verify totalTime < 0.01 logic |

---

## Optional: GPU-Only Light Sync Shader

If implementing Option D (GPU-only sync), create this new shader:

### update_star_lights.hlsl

**Location:** `shaders/lights/update_star_lights.hlsl`

```hlsl
// update_star_lights.hlsl
// Compute shader that syncs star particle positions to light buffer
// Eliminates CPU readback for maximum performance

#include "../particles/gaussian_common.hlsl"

// Light structure (must match C++ and particle_gaussian_raytrace.hlsl)
struct Light {
    float3 position;
    float intensity;
    float3 color;
    float radius;
    float enableGodRays;
    float godRayIntensity;
    float godRayLength;
    float godRayFalloff;
    float3 godRayDirection;
    float godRayConeAngle;
    float godRayRotationSpeed;
    float _padding;
};

// Buffers
RWStructuredBuffer<Light> g_lights : register(u0);
StructuredBuffer<Particle> g_particles : register(t0);

// Constants
cbuffer StarLightConstants : register(b0) {
    uint starCount;           // Number of star particles (16)
    float globalLuminosity;   // Luminosity multiplier
    float globalOpacity;      // Opacity multiplier (not used for lights)
    float padding;
};

// Temperature to RGB color conversion (Wien's law approximation)
float3 TemperatureToLightColor(float kelvin) {
    float t = kelvin / 1000.0;
    float3 color;

    if (t <= 6.6) {
        color.r = 1.0;
        color.g = saturate(0.39 * log(t) - 0.63);
        color.b = saturate(0.12 * log(max(t - 1.0, 0.01)));
    } else {
        color.r = saturate(1.29 * pow(t - 6.0, -0.133));
        color.g = saturate(1.13 * pow(t - 6.0, -0.075));
        color.b = 1.0;
    }

    return color;
}

[numthreads(16, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= starCount) return;

    // Read particle data
    Particle p = g_particles[id.x];

    // Update light position from particle
    g_lights[id.x].position = p.position;

    // Optionally update color from temperature
    g_lights[id.x].color = TemperatureToLightColor(p.temperature);

    // Apply global luminosity multiplier
    // Note: Base intensity set during initialization, multiply here
    g_lights[id.x].intensity = g_lights[id.x].intensity * globalLuminosity;
}
```

**Compilation:**
```bash
"/mnt/c/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/dxc.exe" \
    -T cs_6_5 \
    -E main \
    shaders/lights/update_star_lights.hlsl \
    -Fo build/bin/Debug/shaders/lights/update_star_lights.dxil \
    -I shaders
```

**Integration:**
This shader would run after particle physics but before rendering:
```
Physics Pass → Star Light Sync Pass → Render Pass
```

---

## Related Documents

- `FEATURE_OVERVIEW.md` - High-level concept
- `ARCHITECTURE_OPTIONS.md` - Implementation approaches
- `TECHNICAL_REFERENCE.md` - Data structures and memory layouts
- `IMPLEMENTATION_GUIDE.md` - Step-by-step instructions

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | Dec 2025 | Claude Code | Initial document |
