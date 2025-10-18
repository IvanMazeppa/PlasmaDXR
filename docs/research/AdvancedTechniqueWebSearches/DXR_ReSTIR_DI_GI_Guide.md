### ReSTIR for Direct and Global Illumination in D3D12/DXR: Implementation Guide

This guide explains how to implement ReSTIR (Resampled Spatio-Temporal Importance Resampling) for Direct Illumination (DI) and extend it toward GI in a DX12/DXR renderer. It focuses on practical data layouts, reservoir math, correctness pitfalls, and performance.

### Why ReSTIR

- Dramatically improves quality at 1–2 spp by reusing good samples across pixels and frames.
- Scales to thousands of lights and high-frequency emissive geometry.
- Pairs naturally with ray-traced visibility (1 shadow ray per pixel) and modern denoisers.

### Core concepts

- A **reservoir** maintains one representative sample s with weight W and sum of weights wSum, plus a running random key u.
- **Temporal resampling**: combine current frame candidates with last frame’s reservoir (after reprojection) to preserve good lights over time.
- **Spatial resampling**: merge reservoirs from neighbors to increase effective candidates without extra light evals.

Reservoir fields per pixel (packed for bandwidth):

- Sample: light identifier (index or triangle emissive barycentrics), position/normal/radiance if needed
- `float wSum` (sum of importance weights)
- `float W` (final MIS-style weight for unbiasedness)
- `uint rngState` and `float u`

### Candidate generation (DI)

For each pixel:

1) Draw K light candidates using a global alias table (or tile-local lists), and optionally one from last frame’s reservoir (temporal reuse).
2) For each candidate i, compute importance `w_i = L_i * BSDF * G / p_i`, where p_i is the PDF of drawing this candidate from the generator.
3) Perform reservoir sampling (Algorithm R with weights): keep one sample with probability proportional to w_i, updating `wSum`.

At the end, compute `W = wSum / (K * w_r)`, where w_r is the weight of the reservoir’s chosen sample. Use `W` to scale the contribution at shading.

### Temporal resampling

- Reproject the previous frame’s reservoir via motion vectors (or world position reprojection).
- Validate with depth/normal/roughness tests to reject mismatches.
- Merge the reprojected reservoir as an additional candidate with its own `p_prev` and weight.

### Spatial resampling

- From a small neighborhood (e.g., a 2×2 or 3×3 pattern with blue-noise rotation), gather N neighbor reservoirs.
- Treat each neighbor’s sample as a candidate with its own weight and selection PDF.
- Perform a second reservoir update to produce the final reservoir for the pixel.

### Visibility and shading

- After final reservoir selection, trace one ray toward the chosen light sample (shadow ray). Use `ACCEPT_FIRST_HIT_AND_END_SEARCH` and clamp `tMax` to the light distance.
- If visible, add contribution `W * Li * BSDF * G / p_sel`. If blocked, contribute 0.

### Data layout and buffers

- Double-buffer reservoirs for temporal reuse.
- Store compact light descriptors (type, transform, power, alias entry) in a bindless `StructuredBuffer`. For emissive triangles, store a light list with packed primitive IDs and CDF/alias tables per mesh.
- Maintain per-tile light lists for clustered lighting to cull far lights before sampling.

### PDFs and correctness

- Track the generator PDF for every candidate (alias-selected light or emissive triangle sample). Include Jacobians if sampling by area/solid angle conversion.
- When merging temporal/spatial reservoirs, use the original candidate’s PDF as `p_i` for unbiasedness.
- Compute `W` using standard ReSTIR formula; ensure numerical stability by clamping extremely small/large weights.

### Extending toward ReSTIR GI (one-bounce)

- Generate candidates from BSDF sampling at the primary hit (one or few directions), plus reprojected/spatial reuse of previous directions.
- For each candidate direction, trace a ray to find an indirect hit; evaluate indirect radiance as the candidate weight.
- Use reservoir resampling to keep the best direction, then trace one final ray for contribution (or reuse the already traced one depending on variant).
- Keep payloads tiny; GI rays should be short with conservative `tMax` in real-time.

### Performance best practices

- Keep K small (e.g., 1–2 candidates) and rely on temporal+spatial reuse to grow effective sample count.
- Use half-resolution reservoirs for GI and upscale after denoising.
- Limit neighborhood size and use blue-noise patterns for spatial gathering.
- Batch shadow rays by tiles for coherence.
- Use instance masks to prune irrelevant geometry for shadow/indirect rays.

### Denoising

- Use NRD ReLAX for indirect/specular signals and SIGMA for shadow visibility.
- Provide stable inputs (normals, roughness, hit distance, motion vectors).
- Clamp history and reset reservoirs on disocclusions or large lighting changes.

### Debugging and validation

- Visualize reservoir `wSum`, selected light index, and `W` to ensure distribution matches intuition.
- Validate PDFs via Monte Carlo checks (compare with brute-force sampling in micro scenes).
- PIX/Nsight: confirm ray counts, shadow ray lengths, and absence of any-hit hotspots.

### Common pitfalls and fixes

- **Wrong PDFs**: forgetting geometry/solid-angle Jacobians biases results.
- **Reservoir weight misuse**: using `W` incorrectly causes energy drift; follow `W = wSum / (m * w_r)`.
- **Reprojection mismatches**: stale reservoirs cause ghosting; gate with depth/normal/roughness tests and reactive masks.
- **Over-large neighborhoods**: blurs details and increases bandwidth; keep small and blue-noise rotated.

### Minimal checklist

- [ ] Temporal + spatial reservoirs double-buffered
- [ ] Correct PDFs tracked per candidate (alias or emissive triangle)
- [ ] One shadow ray per pixel for DI visibility
- [ ] NRD integration and history management
- [ ] Visualizations for `wSum`, chosen light, and PDF sanity checks

### References

- ReSTIR DI (SIGGRAPH 2020): `https://research.nvidia.com/publication/2020-07_spatiotemporal-reservoir-resampling-real-time-ray-tracing-direct-illumination`
- ReSTIR GI (SIGGRAPH 2021): `https://research.nvidia.com/publication/2021-07_restir-gi-path-resampling-gi`
- DirectX-Graphics-Samples: `https://github.com/microsoft/DirectX-Graphics-Samples`
- NRD (ReLAX/SIGMA): `https://github.com/NVIDIAGameWorks/NRD`

---

ReSTIR turns a single visibility ray into film-quality lighting by reusing great samples across time and space. Implement it carefully with correct PDFs and robust history handling, and it will scale to massive light counts gracefully.


