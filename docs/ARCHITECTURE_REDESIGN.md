# Architecture Redesign Prompt for AI Models

## Context

I'm developing **PlasmaDX-Clean**, a DirectX 12 real-time renderer for simulating galactic accretion disks around black holes with physically-based rendering. The current implementation (v0.18.8) has achieved some success but faces architectural challenges, particularly with the Froxel volumetric fog system which has stubborn bugs.

## Current State Summary

**What's Working Well:**
- 3D Gaussian splatting for volumetric particles with ray-ellipsoid intersection
- DXR 1.1 inline ray tracing (RayQuery API) for particle-to-particle lighting
- Multi-light system (13 lights) with realistic shadowing
- GPU physics simulation (black hole gravity, Keplerian dynamics, blackbody emission)
- NVIDIA DLSS 3.7 Super Resolution integration
- 16-bit HDR pipeline with proper tone mapping
- ~120 FPS @ 10K particles on RTX 4060 Ti

**Current Challenges:**
- Froxel volumetric fog system has race conditions and stubborn bugs
- RTXDI integration incomplete (M5 temporal accumulation causing quality issues)
- Multiple rendering paths (Gaussian RT, Volumetric ReSTIR, Probe Grid, Froxel) creating complexity
- Performance bottleneck: BLAS/TLAS rebuild every frame (2.1ms @ 100K particles)
- Code complexity from accumulated systems over time

**Technology Stack:**
- DirectX 12 + Agility SDK
- DXR 1.1 (RayQuery inline ray tracing)
- HLSL Shader Model 6.5+
- CMake + MSBuild + Visual Studio 2022
- Target Hardware: RTX 4060 Ti / RTX 4000-series

## The Vision

**Render a scientifically accurate galactic accretion disk simulation featuring:**

1. **Diverse Stellar Objects:**
   - Main sequence stars (O, B, A, F, G, K, M types) with accurate spectral colors
   - Red giants and blue supergiants
   - Neutron stars with extreme gravitational lensing
   - White dwarfs
   - Gas clouds (molecular clouds, HII regions, planetary nebulae)
   - Dust regions with appropriate scattering properties
   - Supernovae explosions (time-varying pyrotechnic effects)
   - The central supermassive black hole with Schwarzschild photon sphere

2. **Physical Accuracy:**
   - Relativistic effects (Doppler shift, gravitational redshift, time dilation)
   - Keplerian orbital dynamics with turbulence
   - Temperature-based blackbody radiation (800K-50,000K range)
   - Shakura-Sunyaev viscosity for accretion physics
   - Energy conservation and angular momentum

3. **Advanced Rendering Features:**
   - Volumetric ray tracing with proper light transport
   - Multi-light support (dozens of self-emitting particles)
   - Soft shadows with temporal stability
   - Light scattering (Rayleigh, Mie, Henyey-Greenstein phase functions)
   - God rays / volumetric light shafts
   - Gravitational lensing around black hole (Einstein rings)
   - HDR rendering with proper tone mapping
   - Real-time performance (90+ FPS minimum @ 1080p, 60+ FPS @ 1440p)

4. **Scale Requirements:**
   - 10,000 - 100,000 particles minimum
   - Particle LOD system (distant aggregation → individual close-up)
   - Camera can move from distant overview (1000+ units) to close-up (<50 units)
   - Spatial acceleration structures for efficient culling

## Your Task

**Please design a comprehensive rendering architecture that:**

1. **Research Modern Techniques:**
   - Search for cutting-edge volumetric rendering papers (2020-2025)
   - Investigate production techniques (Unreal Engine 5, Unity, Frostbite)
   - Review NVIDIA RTX research (RTXDI, RTXGI, NRD denoiser)
   - Examine heterogeneous volume rendering (VDB, froxels, probe grids)
   - Consider neural rendering techniques (NeRF, 3D Gaussian Splatting variants)

2. **Propose Optimal Architecture:**
   - What rendering technique(s) should be the foundation? (Pure ray tracing? Hybrid raster+RT? Gaussian splatting? VDB volumes?)
   - How to handle heterogeneous particles (stars vs gas vs dust)?
   - What acceleration structures? (BVH? Sparse voxel octree? Light grid? Probe grid?)
   - Shadow algorithm recommendations (PCSS? Raytraced? RTXDI-based?)
   - Light transport strategy (path tracing? ReSTIR? Photon mapping?)
   - Denoising strategy (temporal? spatial? ML-based like NRD?)

3. **Address Current Pain Points:**
   - How to avoid BLAS/TLAS rebuild bottleneck?
   - Better approach than Froxel system for volumetric fog?
   - Simplify the rendering pipeline (currently 4+ separate systems)?
   - Ensure temporal stability without flicker
   - Scale efficiently from 10K to 100K+ particles

4. **Implementation Roadmap:**
   - Should I continue with PlasmaDX-Clean (refactor existing) or restart fresh?
   - What would a minimal viable renderer look like (Phase 1)?
   - How to incrementally add features without creating technical debt?
   - What systems can be deprecated/removed from current implementation?
   - Estimated development timeline for MVP → Full Feature Set

5. **RTX 4000-Series Optimization:**
   - Leverage Ada Lovelace features (SER, Opacity Micromap, DMM)?
   - Memory bandwidth optimization strategies
   - Ray coherence and wavefront path tracing considerations
   - Async compute opportunities for physics simulation

6. **Practical Constraints:**
   - Solo developer (me) with AI assistant support
   - Moderate C++ experience, learning as I go
   - Must maintain ~90 FPS minimum for fluid interaction
   - Prefer elegant simplicity over feature bloat
   - Real-time interactive rendering (not offline/baking)

## Evaluation Criteria

**Your proposed architecture should:**
- Be technically sound based on current research (cite papers/techniques)
- Provide concrete implementation details (not just buzzwords)
- Balance visual quality with performance
- Minimize complexity where possible
- Have a clear incremental development path
- Acknowledge trade-offs and alternatives

## Output Format

Please structure your response as:

1. **Executive Summary** (2-3 paragraphs)
2. **Research Findings** (key papers/techniques discovered)
3. **Proposed Architecture** (detailed technical design)
4. **Rendering Pipeline** (frame-by-frame breakdown)
5. **Data Structures** (particle representation, acceleration structures)
6. **Shader Architecture** (compute vs raytrace vs raster, shader stages)
7. **Continue vs Restart Analysis** (pros/cons with recommendation)
8. **Implementation Roadmap** (phases with time estimates)
9. **Risk Assessment** (potential blockers and mitigations)
10. **References** (papers, blog posts, GitHub repos, etc.)

## Additional Context

**Current PlasmaDX-Clean Architecture (for reference):**
- Particle struct: 48 bytes (position, velocity, temperature, lifetime, radius)
- Gaussian renderer uses ray-ellipsoid intersection per particle
- DXR 1.1 RayQuery for shadow rays and lighting rays
- CMake build system with HLSL shader auto-compilation
- ImGui for runtime parameter tuning
- Multiple MCP servers provide specialized domain expertise

**Documentation Available:**
- CLAUDE.md (comprehensive project guide)
- MASTER_ROADMAP_V2.md (development roadmap)
- Various technical deep-dives in docs/

---

**Question:** Given this context, what is the optimal architecture for achieving the vision of a physically-accurate, real-time galactic accretion disk renderer targeting RTX 4000-series hardware? Should I continue refactoring PlasmaDX-Clean or start fresh with lessons learned?
