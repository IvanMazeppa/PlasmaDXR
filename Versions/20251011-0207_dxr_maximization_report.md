# DXR Maximization Report (PlasmaDX)

## Current state (from code and GPT5_RT_CONSULTATION_PROMPT)

- Inline RayQuery pre-pass computes per-particle lighting (`shaders/dxr/particle_raytraced_lighting_cs.hlsl`) and the render path multiplies emission by a scalar term, so lighting mostly shifts brightness without depth-aware scattering.
- Procedural AABBs + TLAS/BLAS are rebuilt each frame with `PREFER_FAST_BUILD`; refit/compaction flags aren’t leveraged yet in `RTLightingSystem_RayQuery.cpp`.
- Volumetric marching combines emission, optional shadow rays, and phase term; in-scattering isn’t integrated along the path with transmittance and directional visibility to emitters.
- `shaders/dxr/dxr12_features.hlsli` contains SER/HitObject placeholders but isn’t wired into lighting or volumetric kernels.

## Why volumetric effects feel like “just brightness changes”

- Lighting is estimated outside the volume integral; a per-particle scalar loses directionality and path-length attenuation by the medium.
- Neighbor sampling records a single committed hit; multi-hit and many-light sampling are underrepresented.
- In-scattering is not accumulated along the ray march, so no volumetric shafts or layer-dependent glow.

## Top changes to fully exploit DXR (prioritized)

1. Integrate single-scatter inside the volumetric march

Replace the scalar pre-pass with in-path integration. At each step s, accumulate:
L += T(s) · [σ_s(s) · Σ L_i(s) · p(cosθ)] · Δs, with Beer–Lambert T(s).
Where L_i(s) comes from a few emissive neighbors via RayQuery occlusion (short rays), or from a small temporal reservoir (see 3). Remove or demote reliance on `g_rtLighting` in the march.

1. Use physically separated coefficients and adaptive stepping

Define σ_t = σ_a + σ_s and albedo = σ_s/σ_t. At each step: T *= exp(-σ_t·Δs). Make σ_a, σ_s functions of density/temperature; adapt Δs by local density; early-out when T is very low.

1. Many-light sampling with temporal reuse (ReSTIR-style reservoirs)

Keep a per-pixel reservoir (K entries) of likely emitters (indices + weights). Each frame, update with a few candidates; during marching, sample 1–2 emitters from the reservoir, cast short RayQuery rays for occlusion, and reuse temporally. This yields shafts/layering with few rays.

1. SER (Shader Execution Reordering) on divergent loops

Gate with a runtime flag and wrap divergent branches (candidate handling, shading). Include `dxr12_features.hlsli` in lighting/march shaders and apply SER hint macros when supported.

1. BLAS/TLAS refit + compaction

Build BLAS with `ALLOW_UPDATE`; compact after warmup. Prefer `PREFER_FAST_TRACE` steady-state. Refactor TLAS to update/refit. Coalesce UAV barriers.

1. Optional: directional irradiance (low-order SH) if a pre-pass is retained

If you keep a pre-pass, store 2nd-order SH (9 coeffs) of incident radiance per particle; in the march, evaluate with view/light directions and phase instead of a scalar.

1. Denoiser for single-scatter (NRD RELAX/ReBLUR or SVGF)

Denoise the in-scattering buffer using history, approximate normals (from Gaussian axes), and a proxy depth. Enables lower spp with stable results.

1. Adaptive budgets and LOD

Distance/importance-based budgets: fewer emitters and shorter max distance far from camera; scale steps by projected size; early-out by transmittance cutoff.

1. Async compute overlap

Overlap AABB generation, BLAS refit, RayQuery lighting, and denoiser with physics or graphics; split queues and fence properly.

1. Telemetry and overlays

Add per-pass GPU timings, ray count histograms, heatmaps of in-scattering cost and hit/miss ratios; on-screen toggles to validate changes.

## Celestial particle expansion (star-like behavior)

- Spectral type/metallicity: modulate blackbody with spectral tints; temperature-driven intensity (T^4 scaled).
- Variability: parameters for flares/pulsation; time-varying emission.
- Limb darkening for large Gaussians: angle-dependent emission factor.
- Jets/hotspots: anisotropic lobes aligned to spin/jet axes; fold into phase/emission.
- Dust cocoon factor: higher σ_a, lower albedo near star-forming regions to produce self-shadowing halos.
- Relativistic effects: extend Doppler and gravitational redshift with a lensing proxy (screen-space deflection) near compact objects.
- Efficiency: bucket emitters into classes (few templates) and sample representatives to keep RT costs low.

## Concrete edit targets

- Shaders: volumetric march CS (from consultation doc) — add single-scatter with RayQuery occlusion or reservoir sampling; remove scalar multiply by `g_rtLighting`.
- Shaders: `shaders/dxr/particle_raytraced_lighting_cs.hlsl` — repurpose to build reservoirs or SH (or retire).
- Shaders: `shaders/dxr/dxr12_features.hlsli` — include in lighting/march CS; gate SER with constants.
- C++: `src/lighting/RTLightingSystem_RayQuery.cpp` — switch AS flags to `ALLOW_UPDATE`, add compaction, prefer `PREFER_FAST_TRACE`, unify UAV barriers; plumb feature/config flags.
- C++: application scheduling — add async compute and denoiser stages; wire telemetry/overlays.

## How to leverage existing agent docs

- `.claude/agents/` (graphics-debugging, systems, volumetric, RT-ML): use as step-by-step checklists in Cursor during tuning and debugging (paste or reference per task).
- `PlasmaDX/agent/` guides (RayQuery, ReSTIR, NRD, pipeline/SBT): template reservoir buffers, candidate proposals, and denoiser IO.
- Optional: add a small “agent harness” doc in `Versions/` that maps milestones (debug → perf → denoise → quality) to the specific agent checklists.

## Quick validation goals

- Visual: clear shafts through dense regions; layered depth; anisotropic halos around bright regions.
- Perf: lower rays/frame via reservoirs; stable FPS uplift; reduced variance with denoise.
- AS: lower build time after refit/compaction; fewer stalls from barrier hygiene.

## Suggested queries

- MCP: Show all uses of `D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS` and flags; where can we set `ALLOW_UPDATE` and compaction in `RTLightingSystem_RayQuery.cpp`?
- MCP: Find where the render path multiplies emission by RT lighting; list the file/line that reads `g_rtLighting` in the volumetric march.
- Web: ReSTIR volumes temporal reuse reservoir sampling implementation
- Web: DXR RayQuery single-scatter integration volumetric rendering
- Web: D3D12 raytracing acceleration structure compaction refit dynamic objects
