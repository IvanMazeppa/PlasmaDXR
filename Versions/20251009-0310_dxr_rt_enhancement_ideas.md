# Raytracing Enhancements (DXR 1.1) — Ranked

1. SER (Shader Execution Reordering) for RayQuery compute — Score: 9.4

Value: Increase wave coherence by reordering divergent rays with SER; place `MaybeReorderThread` around shading branches after hit classification.
Where: `shaders/dxr/particle_raytraced_lighting_cs.hlsl`, `shaders/dxr/raytracing_lib.hlsl`.
Web Search: "DXR Shader Execution Reordering RayQuery MaybeReorderThread sample".

1. Acceleration structure flags: ALLOW_UPDATE + compaction — Score: 9.2

Value: Switch BLAS/TLAS builds to support refit (`ALLOW_UPDATE`), prefer `PREFER_FAST_TRACE` at runtime, and compact post-build to reduce VRAM and traversal cost.
Where: `src/lighting/RTLightingSystem_RayQuery.cpp` in `CreateAccelerationStructures`, `BuildBLAS`, `BuildTLAS`.
MCP Search: "Where are `D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS` and build flags set? Show all occurrences in `RTLightingSystem_RayQuery.cpp`".

1. Optimize procedural AABB intersection shader — Score: 8.9

Value: Early-exit on miss, conservative particle bounds, minimize payload writes and branching in procedural intersection.
Where: `shaders/dxr/particle_intersection.hlsl`, `shaders/dxr/generate_particle_aabbs.hlsl`.
Web Search: "DXR procedural primitive intersection shader optimization AABB particles".

1. Cluster particles and build per-cluster BLAS — Score: 8.7

Value: Spatial clustering (e.g., grid or k-means) to reduce rebuild/refit costs and improve traversal; instance clusters via TLAS.
Where: CPU clustering prepass feeding `CreateAccelerationStructures` path in `RTLightingSystem_RayQuery.cpp`.
Web Search: "DXR TLAS BLAS clustering dynamic objects particles".

1. Ray budget with blue-noise and temporal reuse — Score: 8.6

Value: 1–2 rays/particle with blue-noise sequence, temporal reuse, and reservoir sampling for many-emitters.
Where: `shaders/dxr/particle_raytraced_lighting_cs.hlsl` + per-frame constants.
Web Search: "spatiotemporal reservoir sampling many lights ray tracing".

1. Hybrid path: RayQuery for visibility + RT pipeline for complex shading — Score: 8.4

Value: Keep inline ray tracing fast-path for visibility; introduce RTPSO for complex BRDF/material cases and easier denoiser integration.
Where: Add RT pipeline in `src/lighting/RTLightingSystem.cpp` (keep existing RayQuery path as a mode).
Web Search: "DXR RayQuery vs raytracing pipeline hybrid approach".

1. Denoiser (NRD/SVGF) for the RT lighting buffer — Score: 8.3

Value: Denoise low-spp lighting with NRD (ReBLUR/RELAX) or SVGF; add history, normals, roughness inputs.
Where: After `shaders/compute/merge_rt_lighting.hlsl` or as part of a new denoise chain.
Web Search: "NVIDIA NRD ReBLUR RELAX integration guide".

1. Any-hit transparency culling — Score: 8.1

Value: Early reject low-opacity particles to cut closest-hit cost; mimic via early exits in RayQuery if staying inline.
Where: Any-hit in RTPSO path; RayQuery traversal conditions otherwise.
Web Search: "DXR any-hit shader transparent materials optimization".

1. AS memory residency and barrier hygiene — Score: 8.0

Value: Correct heap types, UAV barriers, and residency transitions to avoid stalls during AS build/use.
Where: `CreateAccelerationStructures`, `BuildBLAS`, `BuildTLAS`, dispatch in `RTLightingSystem_RayQuery.cpp`.
MCP Search: "Find UAV barriers and resource state transitions around AS build/use in `RTLightingSystem_RayQuery.cpp`".

1. Distance-based LOD and adaptive ray count — Score: 7.9

Value: Lower emissive contribution and rays for distant clusters; adapt to on-screen size/importance.
Where: CPU constants + `particle_raytraced_lighting_cs.hlsl` conditionals.
Web Search: "adaptive ray tracing sampling budget distance-based LOD".
