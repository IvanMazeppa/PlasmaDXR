# Top 10 RT‑Focused Roadmap Suggestions (Ascending scope/effort)

1. SER toggles in hot RayQuery loops

Why: Improve wave/wrap coherence and cache locality with minimal code.
Where: Shadow rays, volumetric sample branches, and upcoming RTXDI shading. Gate with a runtime flag and use SER hints/macros where supported.

1. BLAS/TLAS refit + compaction

Why: Cut rebuild cost and VRAM; speed up traversal.
Where: Build BLAS with ALLOW_UPDATE and compact after warmup; refit TLAS each frame; prefer PREFER_FAST_TRACE steady‑state; coalesce UAV barriers/residency transitions.

1. Blue‑noise and adaptive ray budgets

Why: Lower variance with few rays; spend rays where they matter.
Where: Use blue‑noise for per‑sample jitter; scale rays/step counts by projected particle size/medium density; early‑out on transmittance; distance‑based LOD.

1. Framegraph + async compute overlap

Why: Hide latency and formalize barriers/residency.
Where: Model passes (AABB gen → BLAS refit → TLAS → RTXDI/lighting → denoise → blit), overlap compute with physics/present; enable transient aliasing and centralized barrier planning.

1. NRD denoiser on single‑scatter/shadows

Why: Hold 1–2 spp stable visually.
Where: Feed history, normals proxy (from Gaussian axes), and depth/hit‑distance to RELAX/ReBLUR (or SVGF) on the single‑scatter/RT shadow outputs.

1. Hybrid RT path: RayQuery + selective RTPSO

Why: Keep simplicity and add flexibility where any‑hit/closest‑hit shines.
Where: Use RayQuery for fast visibility; introduce a small RTPSO only for complex materials (e.g., semi‑transparent gas or any‑hit culling); choose path per mode.

1. RTXDI integration with volumetric coupling

Why: Production‑grade emitter sampling/visibility with temporal stability.
Where: Use RTXDI for light selection/visibility; multiply by in‑medium transmittance and phase function in your volume loop; keep your volumetric self‑shadowing; tune temporal/spatial settings.

1. Material‑aware scattering for celestial bodies

Why: Big visual gain with small constant footprint.
Where: Per‑type σ_s/σ_a/g and emissivity (dust, gas, stars, compact objects); drive phase/shadow behavior and emission; wire small per‑body constants to shaders.

1. Multi‑light + distance clustering

Why: Scale many emitters without exploding cost.
Where: Cluster emitters by distance/region; RTXDI samples cluster representatives; expand to detailed sets only near the camera; couple with distance‑based ray budgets.

1. Optional NRC/light cache for volumes

Why: Amortize multi‑bounce/in‑medium contributions beyond direct lighting.
Where: Add a sparse radiance cache (probe grid or neural cache) blended with RTXDI direct lighting; update on a budget and reuse temporally/spatially.


