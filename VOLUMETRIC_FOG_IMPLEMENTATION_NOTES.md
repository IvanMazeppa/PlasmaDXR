# Volumetric Fog Implementation & Debugging Notes

## 1. System Overview
The volumetric fog system was designed to create atmospheric "God Rays" (crepuscular rays) and general volumetric depth. It works by ray-marching through the scene volume and accumulating light scattering.

### Key Components
*   **Shader:** `shaders/particles/particle_gaussian_raytrace.hlsl`
*   **Function:** `RayMarchAtmosphericFog`
*   **Technique:** Uniform step ray marching (originally 32 steps, reduced to 8).
*   **Lighting:** Samples the global light list (`g_lights`) and the Probe Grid (`g_probeGrid`).
*   **Occlusion:** Uses DXR 1.1 `TraceRayInline` to cast shadow rays against the particle BVH (`g_particleBVH`).

## 2. Implementation Details

### A. The Shader (`particle_gaussian_raytrace.hlsl`)
The `RayMarchAtmosphericFog` function performs the following:
1.  **Loops** `NUM_STEPS` times along the camera ray.
2.  **Calculates** sample position world coordinates.
3.  **Loops** through all active lights (clamped to 16).
4.  **Casts** an inline shadow ray (`TraceRayInline`) from the sample position to the light.
    *   *Note:* Uses `shadowBias` and `SHADOW_OPACITY_THRESHOLD` (0.99).
5.  **Accumulates** scattered light if the path is unoccluded (or partially occluded based on density).
6.  **Probe Grid Integration:** Also samples the `g_probeGrid` (ambient light) at each step and adds it to the scatter color.

### B. C++ Integration (`Application.cpp` & `ParticleRenderer_Gaussian.cpp`)
*   **Controls:** `m_godRayDensity` (float) controls the fog intensity.
    *   `0.0` = Disabled.
    *   `> 0.0` = Enabled.
*   **Data Transfer:** Passed to the shader via the `GaussianConstants` constant buffer (register `b0`).

## 3. Recent Changes (Debugging & Performance Fixes)

In response to performance issues (~1 FPS) and artifacts (white noise/shapes), the following changes were made:

### Change 1: Disabled by Default in C++
*   **File:** `src/core/Application.cpp`
*   **Action:** Initialization of `m_godRayDensity` forced to `0.0f` in `Application::Initialize`.
*   **Action:** Removed/Commented out the `if` block that allowed `config.json` to override this value.
*   **Action:** Added `--fog` command-line flag to optionally re-enable it.

### Change 2: Hard-Disabled in Shader
*   **File:** `shaders/particles/particle_gaussian_raytrace.hlsl`
*   **Action:** The call to `RayMarchAtmosphericFog` inside the `main` entry point was commented out.
    ```hlsl
    /*
    if (godRayDensity > 0.001) {
        float3 atmosphericFog = RayMarchAtmosphericFog(...);
        finalColor += atmosphericFog;
    }
    */
    ```
*   **Effect:** The GPU will essentially dead-code-eliminate the entire fog function, guaranteeing zero performance cost.

## 4. Investigation Clues: "Dimmed Light" & "Broken Controls"

The user reports that after these changes, the scene is dimmed and "inline RQ controls" (likely RT lighting strength) are not working.

### Hypothesis A: Fog Was Acting as Ambient
If the scene is significantly dimmer, the fog might have been providing a baseline level of ambient illumination (via the Probe Grid sampling inside the fog loop).
*   **Code:** `float3 indirectScatter = indirect * godRayDensity * 0.5;`
*   **Impact:** Removing the fog removes this additive ambient light layer.
*   **Fix:** Adjust `rtMinAmbient` in `config.json` or via ImGui to compensate.

### Hypothesis B: Constant Buffer Misalignment
If "controls are not working" (e.g., sliders do nothing), the C++ struct and HLSL cbuffer might be out of sync.
*   **Check:** `GaussianConstants` layout.
*   **C++:** `ParticleRenderer_Gaussian.h` -> `RenderConstants`
*   **HLSL:** `particle_gaussian_raytrace.hlsl` -> `cbuffer GaussianConstants`
*   **Risk:** Did removing/changing code shift any offsets? (Unlikely if only logic was changed, but `godRayStepMultiplier` usage was removed in C++ logic, though the *struct member* `godRayStepMultiplier` must remain to keep alignment).

### Hypothesis C: Adaptive Quality Interference
The `AdaptiveQualitySystem` was overwriting `m_godRayDensity` and potentially other RT settings (`m_rtLightingStrength`).
*   **Fix Applied:** Commented out `m_godRayDensity = preset.godRayDensity;` in `Application::Update`.
*   **Remaining Risk:** Does it still overwrite `m_rtLightingStrength`? Yes.
    ```cpp
    m_rtLightingStrength = preset.rtLightingStrength;
    ```
    *   **Diagnosis:** If Adaptive Quality is enabled (even if invisible), it might be locking `rtLightingStrength` to a low value, making manual sliders ineffective. Check if Adaptive Quality is truly disabled.

## 5. Recommended Next Steps
1.  **Verify Adaptive Quality:** Explicitly set `m_enableAdaptiveQuality = false;` in `Application::Initialize` to ensure it's not overriding manual controls.
2.  **Check rtMinAmbient:** Increase this value to restore scene brightness if the fog ambient is missed.
3.  **Verify Struct Alignment:** Double-check `RenderConstants` (C++) vs `GaussianConstants` (HLSL) byte-for-byte.
