# Architecture Proposal: PlasmaDX-Clean Redesign

## 1. Executive Summary

Based on the analysis of your current codebase (v0.18.8) and the recent "RT Engine Breakthrough," I strongly recommend **refactoring** PlasmaDX-Clean rather than restarting. You have a working, scientifically accurate DXR 1.1 renderer that already solves the hardest problem: volumetric ray-ellipsoid intersection for Gaussian splatting.

The proposed architecture consolidates your rendering paths into a single **Unified Volumetric Pipeline**. This removes the complexity of maintaining separate "Froxel," "Gaussian," and "ReSTIR" paths. The core strategy is to leverage **RTXDI** for light management (solving the multi-light scale issue) and **Hierarchical LOD** to solve the BLAS/TLAS bottleneck, allowing you to scale from 10k to 100k+ particles.

## 2. Research Findings & Technology Choices

### Key Techniques Selected
1.  **RTXDI (RTX Direct Illumination):** The industry standard for "many lights" (1000+). Essential for your vision of glowing accretion disks.
    *   *Why:* Replaces your custom ReSTIR and Multi-Light arrays. Handles importance sampling automatically.
    *   *Status:* Partially implemented but disabled. Needs fixing.
2.  **Volumetric Ray-Marching Shadows:**
    *   *Why:* PCSS is a surface technique. Your particles are volumetric. Ray-marched shadows (accumulating opacity) are physically correct for gas/dust.
    *   *Optimization:* Combine with **Temporal Accumulation** (reusing your PCSS buffers) to reduce ray cost (1 ray/pixel/frame).
3.  **3D Gaussian Splatting (Core):**
    *   *Why:* You already have this working ("The Breakthrough"). It is the correct primitive for nebulous celestial bodies.
    *   *Enhancement:* **LOD System** to handle 100k particles.

## 3. Proposed Architecture: The Unified Volumetric Pipeline

### 3.1 Core Philosophy
Instead of separate systems for "Fog" and "Particles," everything is a **Gaussian Volume**.
*   **Stars:** High-density, emissive Gaussians.
*   **Gas/Dust:** Low-density, absorptive Gaussians.
*   **Fog:** Large, low-density background Gaussians (eliminating the need for a separate Froxel grid).

### 3.2 Rendering Pipeline (Frame Breakdown)

1.  **Simulation Step (Compute):**
    *   Run Physics (PINN or Keplerian).
    *   Update Particle Buffers (Pos, Temp, Density).
2.  **LOD & Culling (Compute) - *SOLVES TLAS BOTTLENECK*:**
    *   Kernel categorizes particles into **Near** (<500 units), **Mid**, and **Far**.
    *   **Near/Mid:** Added to an `InstanceBuffer` for TLAS build.
    *   **Far:** Rendered as simplified billboards or skipped if occluded.
    *   *Result:* TLAS only contains ~20k active instances, not 100k. Rebuild time drops < 0.5ms.
3.  **Acceleration Structure Build:**
    *   Build BLAS (Procedural AABBs) + TLAS from the culled list.
4.  **RTXDI Light Update:**
    *   Register emissive particles as lights in RTXDI.
5.  **Primary Ray Trace (RayQuery):**
    *   Trace Camera Rays.
    *   Intersect Gaussians (Volumetric).
    *   **Lighting:** Ask RTXDI for the best light sample.
    *   **Shadows:** Trace **1 Volumetric Shadow Ray** to that light.
    *   Accumulate Radiance.
6.  **Denoising / Temporal Accumulation:**
    *   RTXDI Temporal Pass (Fix the "patchwork" here).
    *   Output Blit (HDR -> SDR).

## 4. Addressing Pain Points

### 4.1 The BLAS/TLAS Bottleneck (2.1ms @ 100k)
**Solution: Instance Culling & LOD.**
You are currently rebuilding the AS for *every* particle.
*   **Fix:** Implement a compute shader `CullParticles.hlsl` before the build.
*   Filters out:
    *   Particles behind the camera.
    *   Particles too small to be seen (pixel coverage < threshold).
    *   Very distant particles (render these with a simple "Skybox/Imposter" pass instead).
*   **Impact:** Reduces TLAS build complexity by 50-80%.

### 4.2 Froxel Fog Issues
**Solution: Deprecate Froxels.**
Your Gaussian system *is* volumetric.
*   **Fix:** To simulate "fog," just add a few huge, low-density Gaussian particles that encompass the scene.
*   **Benefit:** Removes the race-condition-prone `inject_density.hlsl` and simplifies the pipeline.

### 4.3 RTXDI "Patchwork" & Complexity
**Solution: Fix M5 Integration.**
The "patchwork" pattern is a classic sign of broken **Spatial Reuse** or incorrect **Temporal History**.
*   **Fix:** Ensure motion vectors are correct (even if zero for now, they must be consistent). Validate reservoir exchange.
*   **Benefit:** Enables "infinite" lights without the code complexity of custom arrays.

## 5. Implementation Roadmap (Refactor Strategy)

**Phase 1: Cleanup (Days 1-3)**
*   [ ] Remove custom ReSTIR (freed 132MB VRAM).
*   [ ] Remove Froxel System (if you agree to the "Huge Gaussian" alternative).
*   [ ] Consolidate shaders.

**Phase 2: RTXDI Repair (Week 1)**
*   [ ] Fix M5 Temporal Accumulation (The "Patchwork" fix).
*   [ ] Enable RTXDI as the primary light handler.
*   [ ] Validate with 13 lights first, then 100.

**Phase 3: The LOD System (Week 2)**
*   [ ] Implement `CullParticles` compute shader.
*   [ ] Modify TLAS build to use `IndirectArgs` (build only active count).
*   [ ] Test 100k particles (Target: < 1ms build time).

**Phase 4: Shadows & Material System (Week 3)**
*   [ ] Implement "Volumetric Shadow Ray" (reuse `RayGaussianIntersection`).
*   [ ] Add Material Types (Star vs. Dust) using the `GaussianConstants`.

## 6. Continue vs. Restart Analysis

**Recommendation: CONTINUE (REFACTOR)**

| Metric | Restart | Refactor (Recommended) |
| :--- | :--- | :--- |
| **Time to MVP** | 4-6 Weeks | 1-2 Weeks |
| **Risk** | High (Re-solving intersection math) | Low (Core math is solid) |
| **Momentum** | Loses "Breakthrough" morale | Builds on recent wins |
| **Tech Debt** | 0 (Initial) | Moderate (Requires cleanup) |

**Why?** You just solved the hardest part (Volumetric Ray-Ellipsoid Intersection). Throwing that away would be a mistake. The "mess" is peripheral (Fog, ReSTIR), not core.

## 7. Next Step
Shall I create a **Todo List** to execute **Phase 1 (Cleanup)** and **Phase 2 (RTXDI Repair)**?

