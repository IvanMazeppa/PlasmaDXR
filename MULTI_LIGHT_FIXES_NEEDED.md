# Multi-Light System - Remaining Issues

**Date:** 2025-10-16
**Status:** WORKING but 3 quirks to fix

---

## Issue 1: "Sphere Boundary" - Light Disappears Beyond ~300-400 Units

**User Report:** "once it passes that boundary region it just disappears despite there being particles out there to illuminate"

### Suspected Cause
The 300-400 unit sphere matches your **particle physics constraint**:
- `outerRadius: 300.0` in config
- Particles only spawn within 10-300 units
- BUT some particles drift beyond 300 (velocity carries them)

**The light itself doesn't disappear** - there are just WAY FEWER particles beyond 300 units to illuminate!

### Fix Options

**Option A: Increase Particle Outer Radius**
```json
// config_dev.json
"physics": {
    "outerRadius": 1000.0  // Was 300.0
}
```

**Option B: Check Light Position Clamping (if exists)**
Search for any Vec3 clamping in Application.cpp light handling

**Option C: Increase Attenuation Range**
```hlsl
// Line 726 in particle_gaussian_raytrace.hlsl
// BEFORE:
float attenuation = 1.0 / (1.0 + lightDist * 0.01);

// AFTER (10x weaker falloff):
float attenuation = 1.0 / (1.0 + lightDist * 0.001);
```

---

## Issue 2: Light Radius Control Has No Effect

**User Report:** "the light radius control doesn't seem to have an effect"

### Root Cause
The `light.radius` parameter is uploaded to GPU but **never used in shader**!

**Current shader (line 726):**
```hlsl
float attenuation = 1.0 / (1.0 + lightDist * 0.01);  // Ignores light.radius!
```

### Fix
```hlsl
// particle_gaussian_raytrace.hlsl:726
// Use light.radius for soft falloff
float normalizedDist = lightDist / max(light.radius, 1.0);  // Normalize by radius
float attenuation = 1.0 / (1.0 + normalizedDist * normalizedDist);  // Quadratic for soft edge
```

**Effect:** Lights with larger radius = wider illumination area

---

## Issue 3: Can't Disable Original RT Lighting

**User Report:** "i really think that the original main RT light should be made into a preset and also be on by default. i can't fully eliminate its influence"

### Current Confusion
Two separate lighting systems active:
1. **RTLightingSystem (Phase 2.6)** - Particle-to-particle RT lighting (16 rays/particle)
2. **Multi-Light System (Phase 3.5)** - External point lights

Both are active simultaneously. User can't fully disable #1.

### Fixes Needed

#### A. Add RT Lighting Toggle (Application.cpp)
```cpp
// Application.h:118
bool m_enableRTLighting = true;  // NEW

// Application.cpp ImGui (~line 1850, in Multi-Light System section)
if (ImGui::Checkbox("RT Particle-Particle Lighting", &m_enableRTLighting)) {
    // When disabled, set rtLightingStrength to 0 in next frame
}

// Apply in Update() (~line 340)
float effectiveRTStrength = m_enableRTLighting ? m_rtLightingStrength : 0.0f;
// Pass effectiveRTStrength to renderer instead of m_rtLightingStrength
```

#### B. Add "RT Primary" Preset (Application.cpp InitializeLights())
```cpp
// Around line 2050
if (ImGui::Button("RT Primary")) {
    m_lights.clear();
    // Add single light at origin (mimics Phase 2.6 behavior)
    ParticleRenderer_Gaussian::Light primaryLight;
    primaryLight.position = XMFLOAT3(0, 0, 0);
    primaryLight.color = XMFLOAT3(1.0f, 0.95f, 0.9f);  // Warm white
    primaryLight.intensity = 5.0f;
    primaryLight.radius = 50.0f;
    m_lights.push_back(primaryLight);
    m_enableRTLighting = true;  // Enable RT lighting with this preset
}
```

#### C. Make It Default (Application.cpp ~line 194)
```cpp
// BEFORE:
m_lights.clear();  // Start with 0 lights

// AFTER:
// Default: RT Primary preset (single light + RT lighting)
InitializeLights();  // Restore the original call
// But change InitializeLights() to create just 1 light at origin by default
```

---

## Quick Fix Priority

**30 minutes total:**
1. Fix light radius (5 min) - shader edit line 726
2. Add RT lighting toggle (15 min) - Application.cpp/h edits
3. Test sphere boundary with increased outerRadius (10 min)

**Files to Edit:**
- `shaders/particles/particle_gaussian_raytrace.hlsl:726`
- `src/core/Application.h:118` (add bool)
- `src/core/Application.cpp:1850` (add checkbox)
- `src/core/Application.cpp:340` (apply toggle)
- `config_dev.json` (increase outerRadius to 1000?)

---

**User Feedback:** "this is one hell of a brilliant update!!!!!!!!!!!"
**Status:** System working, just needs these 3 polish fixes
