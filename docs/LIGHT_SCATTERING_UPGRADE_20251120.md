# Light Scattering Upgrade & Probe Grid Integration - 2025-11-20

## Executive Summary

To address the missing "volumetric light scattering" in PlasmaDX-Clean without incurring the performance cost of full path tracing (ReSTIR), we have implemented a **Hybrid Probe Grid Fog** system.

This architecture shifts from simulating "physics rays" (expensive) to simulating a "volumetric medium" (efficient) that is lit by the existing Probe Grid system.

---

## 1. The Core Problem

**Previous State:**
- **Particles:** Lit by direct RT + Shadows (Good)
- **Fog/Space:** Uniform or empty. God rays were calculated using analytical cones, but the "air" itself didn't carry the color of the environment.
- **Result:** "Isolated spheres in space" look. No sense of a cohesive medium connecting the particles.

**ReSTIR Failure:**
- VolumetricReSTIR correctly simulated scattering but converged too slowly (noise) and ignored external lights.

## 2. The Solution: Probe Grid Atmospheric Fog

We have upgraded the `RayMarchAtmosphericFog` function in the primary Gaussian renderer to sample the **Irradiance Probe Grid** at every step of the ray march.

### Architecture

1.  **Probe Grid (Existing):** A 32³ sparse grid captures irradiance from all 13 dynamic lights. It effectively stores the "ambient light field" of the accretion disk.
2.  **Atmospheric Fog (Upgraded):**
    - Ray marches from camera through the scene (32 steps).
    - **Direct Light:** Samples analytical god ray cones (existing).
    - **Indirect Light (NEW):** Samples the Probe Grid at the current ray position.
    - **Combination:** `Fog = DirectShafts + (ProbeIrradiance * AmbientDensity)`

### Key Benefits

-   **Unified Lighting:** The fog now glows with the same light colors as the particles. If a red star is near, the fog turns red.
-   **Volumetric Depth:** Even in shadow (where direct god rays are blocked), the fog is visible due to ambient probe lighting, creating true volumetric depth.
-   **Performance:** Extremely cheap (~1-2ms). Reuses the Probe Grid computed for particle lighting.

---

## 3. Tuning & Controls

The effect is controlled by existing parameters, now repurposed:

| Parameter | Variable | Effect |
| :--- | :--- | :--- |
| **God Ray Density** | `godRayDensity` | Controls global fog density. Higher = thicker fog, stronger shafts, brighter ambient. |
| **Probe Intensity** | `g_probeIntensity` | Controls how bright the probe grid is. Affects both particle bounce lighting AND now fog ambient brightness. |
| **Shadows** | `g_shadowDepth` | Particles cast shadows into the fog (via RayQuery), creating "voids" or shafts in the mist. |

## 4. Implementation Details

**Modified File:** `shaders/particles/particle_gaussian_raytrace.hlsl`

**Changes:**
1.  Moved `SampleProbeGrid` and SH helper functions to be accessible by `RayMarchAtmosphericFog`.
2.  Injected `SampleProbeGrid` into the fog ray marching loop:

```hlsl
// Ray March Loop
for (uint step = 0; step < NUM_STEPS; step++) {
    // ... Direct Light Calculation ...

    // === ADDED: Indirect/Ambient Volumetric Lighting ===
    if (useProbeGrid != 0) {
         float3 indirect = SampleProbeGrid(samplePos, rayDir);
         // 5% of density contributes to ambient glow
         totalFogColor += indirect * godRayDensity * 0.05 * stepSize; 
    }
}
```

## 5. Future Roadmap (Next Steps)

1.  **Density Injection:** Currently, fog density is uniform (`godRayDensity`). The next step is to inject particle density into a low-res 3D texture (Froxel grid) so fog only exists *around* particles, not everywhere.
2.  **Screen-Space Scattering:** Add a post-process radial blur pass to create "bloom/glow" from the bright core, further enhancing the cinematic look.

---

**Status:** ✅ Implemented
**Date:** 2025-11-20

