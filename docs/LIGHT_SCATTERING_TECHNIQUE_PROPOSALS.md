# Light Scattering Technique Proposals - 2025-11-20

## Executive Summary

To achieve the "Interstellar" style volumetric light scattering you're looking for—without the crushing weight of full path tracing—you need to shift your mental model from "simulating physics" (ReSTIR) to "simulating the medium" (Volumetric Fog / Froxels).

Currently, your engine renders solid objects (Gaussians) and empty space (Uniform Fog). You are missing the bridge: a heterogeneous medium that represents the collective density of your accretion disk, which light can scatter through.

Here is the architectural breakdown and prioritized roadmap.

## 1 Architectural Recommendations

Recommendation A: The "Froxel" Grid (Frustum-Aligned Voxel Grid)

Concept: Instead of tracing rays between 10,000 particles (N² problem), you rasterize or "splat" your particle density into a low-resolution 3D frustum-aligned grid (e.g., 160x90x64 voxels).

Lighting: You calculate lighting once per voxel (13 lights + shadows), not per pixel.

Rendering: During your main pass, you ray-march this grid.

Why it works: It creates a continuous volumetric field where light "bleeds" and scatters naturally. It unifies your particles and the space between them.

Cost: Fixed cost (grid size), independent of particle count. Extremely fast on Ada Lovelace.

Recommendation B: Enhanced Probe Grid (Immediate Win)

Concept: You already have a ProbeGridSystem capturing irradiance. Currently, it seems to be under-utilized or not fully coupled to the medium.

Fix: Use the probe grid to act as the "Ambient Volumetric Term".

Integration: In your RayMarchAtmosphericFog function (which currently assumes uniform fog), sample the Probe Grid at each step. This adds the "multi-bounce" light color to the air itself, not just the direct light shafts.

Recommendation C: Screen-Space God Rays (The "Cinematic" Glue)

Concept: A post-process radial blur originating from screen-space light positions.

Why it works: It mimics the camera lens scattering. It's "fake" but looks 10x more volumetric than physical scattering for <1ms.

## 2\. Prioritized List of Approaches

Screen-Space Radial Blur (Post-Process)

Cost: ~0.5ms

Impact: High (Instant "cinematic" feel)

Why: It's the cheapest way to get the "glow" you describe as missing.

Probe Grid Integration for Fog

Cost: ~1-2ms

Impact: Medium-High (Adds colored ambient scatter to the "air")

Why: You already have the data structures; you just need to sample them in the fog loop.

Density-Based Volumetric Fog (Simplified Froxel)

Cost: ~3-5ms

Impact: Very High (True volumetric shadows and shafts from particles)

Why: This solves the "isolated spheres" look by creating a density field around them.

Discard ReSTIR for General Scattering

Decision: Stop trying to make ReSTIR work for the whole disk. Use it strictly for explosions or super-bright cores where self-emission dominates.

## 3\. Implementation Breakdown

Step 1: Screen-Space Radial Scattering (4 Hours)

Don't underestimate this. Most "volumetric" games use this.

Shader: A simple compute shader that samples your MainSceneColor and blurs it radially away from the screen-space position of your black hole/lights.

Integration: Add a PostProcess\_RadialBlur.hlsl.

Performance: Negligible on 4060 Ti.

Step 2: Connect Probe Grid to Fog (6 Hours)

Your RayMarchAtmosphericFog in particle\_gaussian\_raytrace.hlsl currently calculates direct light:

Change it to:

Requirement: Ensure UpdateProbes compute shader is actually running and capturing the 13 lights. (Code inspection suggests it exists but verify the Dispatch call in Application.cpp).

Step 3: "Poor Man's" Density Grid (2 Days)

Instead of full froxels:

Create a 3D Texture (R16\_FLOAT, e.g., 64x64x64) covering the accretion disk bounds.

Compute Shader: InjectDensity.hlsl. Each thread takes a particle, calculates its footprint in the grid, and uses InterlockedAdd (or atomic min/max) to write density.

Modify Fog: In RayMarchAtmosphericFog, sample this 3D texture instead of using global godRayDensity.

Result: Fog only exists where particles are. Light shafts will appear to emanate from the particle cloud itself.

## 4\. Modern Ada Lovelace \& DXR 1.1+ Opportunities

Shader Execution Reordering (SER):

Verdict: SKIP for now. SER helps when rays are incoherent (scrambled). Your Gaussian renderer uses coherent primary rays, and RayQuery for shadows is also fairly coherent. The complexity overhead isn't worth the 5-10% gain here.

RTXDI (Direct Illumination):

Verdict: Keep it handy. If you want every particle to emit light (10,000 lights), RTXDI is the only way. But for 13 lights? Standard loops are faster.

Opacity Micromaps (OMM):

Verdict: Not applicable. Your Gaussians are procedural intersections, not alpha-tested triangles. OMM is for foliage/fences.

## 5\. The "Brutal Truth" on ReSTIR

Is ReSTIR a dead end for this specific goal?

Yes.

Volumetric ReSTIR converges to physical reality, which is often "dark and noisy" without thousands of samples. You want an artistic representation of light scattering (glow, shafts, halos).

Action: Shelf ReSTIR for the general accretion disk. Keep it for "Hero" effects (e.g., a supernova explosion event) where the noise looks like turbulent energy.

Implementation Plan (Next Steps)

I have devised a plan to modify your existing shaders to implement Step 2 (Probe Grid Fog) immediately, as it uses your existing code.

Verify Probe Update: Check Application.cpp to ensure m\_probeGrid->UpdateProbes is called.

Modify particle\_gaussian\_raytrace.hlsl:

Inject SampleProbeGrid into the RayMarchAtmosphericFog loop.

Use particle density (if possible) or a distance function to modulate fog density, creating a "halo" around the disk.

This will immediately make the "empty space" between particles feel connected to the lighting system.

