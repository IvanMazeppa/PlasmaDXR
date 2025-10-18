### Ray-Traced Shadows in D3D12/DXR: Correctness, Performance, and Denoising

This guide provides an end-to-end, production-focused approach for ray-traced shadows using Direct3D 12 Raytracing. It covers pipeline choices (RTPSO vs Inline), robust implementation patterns, common pitfalls, and modern denoising strategies. Examples assume DXR 1.0/1.1 and the Agility SDK.

### Goals and scope

- Hard and soft shadows for punctual and area lights
- Efficient single-hit visibility testing per light/sample
- Scaling to many lights using tiled/clustered lighting or ReSTIR DI sampling
- Stable, denoised output at low rays-per-pixel (1–2 rpp)

### Choose your tracing path

Two viable approaches:

- **Inline Ray Tracing (RayQuery)**: Best for binary visibility inside raster/compute lighting passes. Minimal setup, no SBT, highly coherent with your G-buffer. Ideal default for shadows.
- **RTPSO + SBT**: Use when per-material closest-hit logic or special any-hit behavior is needed (complex alpha tests, subsurface decals, etc.). Also suitable if shadows are part of a larger RT pipeline.

Recommendation: Start with Inline for most shadows; switch to RTPSO only if you require complex hit logic that can’t be expressed inline.

### Acceleration structures and masks

- Build BLAS for meshes and a TLAS for the scene (see Acceleration Structures guide).
- Set instance masks to quickly exclude geometry for certain ray types (e.g., exclude particles/foliage from hard shadows if desired).
- Mark opaque geometry as `OPAQUE` to skip any-hit. Reserve any-hit only for alpha-tested content.

### HLSL: minimal hard shadow with RayQuery

```hlsl
RaytracingAccelerationStructure SceneBVH : register(t0);

bool TraceHardShadow(float3 origin, float3 L, float dist, uint mask)
{
    RayDesc r = { origin, L, 0.001, dist - 0.002 };
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_FORCE_OPAQUE> rq;
    rq.TraceRayInline(SceneBVH, 0, mask, r);
    while (rq.Proceed()) {}
    return rq.CommittedStatus() == COMMITTED_NOTHING; // true if visible
}
```

Key flags:

- `ACCEPT_FIRST_HIT_AND_END_SEARCH` ends traversal early.
- `SKIP_CLOSEST_HIT_SHADER` avoids invoking hit shaders (inline path supplies logic).
- `FORCE_OPAQUE` treats all tri geometry as opaque (faster). Omit when alpha-test is required.

### Alpha-tested geometry

Inline path: add a lightweight transparency test in the candidate loop. Keep it as cheap as possible—ideally 1 texel fetch using coherent UVs and LOD:

```hlsl
// Inside while (rq.Proceed())
if (rq.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
    float2 uv = rq.CandidateTriangleBarycentrics();
    float a = Opacity.SampleLevel(LinearClamp, uv, 0);
    if (a < 0.5) { continue; }
    rq.CommitNonOpaqueTriangleHit();
}
```

RTPSO path: implement alpha test in an any-hit shader; call `IgnoreHit()` for transparent texels and keep logic minimal.

### Soft shadows for area lights

Soft shadows approximate penumbra by sampling across an area light and integrating visibility:

- Use 1–2 rays per pixel at half/quarter resolution with temporal accumulation.
- Jitter the light sample point (disk/sphere/rectangle) per frame using blue-noise or Owen scrambling.
- Clamp history and use a robust denoiser (NRD SIGMA/ReLAX) tuned for shadow signals.

Pseudocode (per-pixel in a compute lighting pass):

```hlsl
float ShadowAreaLight(float3 P, float3 N, Light light)
{
    uint2 pix = DispatchRaysIndex().xy; // or SV_DispatchThreadID
    float rand = BlueNoiseHash(pix, frameIndex);
    float2 uv = SampleDiskConcentric(rand);
    float3 xL = light.center + uv.x * light.u + uv.y * light.v;
    float3 L  = normalize(xL - P);
    float  d  = length(xL - P);
    bool vis  = TraceHardShadow(P + N * 0.002, L, d, 0xFF);
    return vis ? 1.0 : 0.0;
}
```

Accumulate temporally and denoise to turn single samples into smooth penumbrae.

### Many lights: tiled/clustered and ReSTIR DI

- Use tiled/clustered lighting to cull lights per tile before tracing.
- For very many lights (thousands), use **ReSTIR Direct Illumination** (reservoir sampling) to select a small set of high-contribution lights per pixel, then trace 1 shadow ray for the chosen light(s). This yields dramatic quality at constant cost.

Core ideas you will implement for ReSTIR DI:

- Sample candidate lights from a global alias table or local tile set.
- Compute importance (power × BSDF × G / p(light)).
- Reservoir resampling across spatial/temporal neighbors to keep a single weighted light.
- Trace one visibility ray for the final selected light per pixel.

### Denoising strategy (NRD, SIGMA/ReLAX)

- Prefer **NRD SIGMA** for shadow signals; **ReLAX** for GI/reflections. Both are real-time and production-proven.
- Provide NRD with stable inputs: hit distance, normal, roughness (if used), motion vectors, and history validity.
- Clamp and reject history on disocclusions; use a reactive mask when lights/geometry change.

Recommended settings starting point:

- Base resolution: half or quarter for soft shadows
- rpp: 1 (temporal + spatial denoiser recovers quality)
- Stabilization strength: medium; history length ~ 30–60 frames with clamping

### Efficiency best practices

- Keep rays short: set `tMax = distance(light) - epsilon`.
- Use instance masks to skip irrelevant geometry (e.g., decals) for shadow rays.
- Batch work by tiles to improve ray coherence.
- Avoid any-hit unless strictly required for alpha test; move all other logic to closest-hit or the inline loop.
- Limit live variables around the trace to reduce register pressure and spills.

### Validation and debugging

- Enable D3D12 debug layer; fix all raytracing warnings.
- Use PIX’s raytracing views to inspect TLAS, instance masks, and candidate/committed hits.
- Add developer visualizations: shadow ray length heatmap, candidate-hit count per pixel, and denoiser history weight visualization.

### Common pitfalls and fixes

- **Overusing any-hit**: kills perf. Use `OPAQUE` and `FORCE_OPAQUE` where possible.
- **Long rays**: unbounded `tMax` increases traversal cost; clamp to light distance.
- **Unstable sampling**: no blue-noise/temporal accumulation causes flicker; integrate TAA-compatible jitter and denoise.
- **Wrong TLAS masks**: missing masks increases needless invocations; configure per-ray-type masks.

### Minimal checklist

- [ ] Choose Inline or RTPSO; implement minimal hard shadow path
- [ ] TLAS masks configured; any-hit only for alpha-test
- [ ] Short, coherent rays with early accept-first-hit
- [ ] Temporal accumulation + NRD SIGMA for soft shadows
- [ ] PIX validation clean; heatmaps show expected traversal lengths

### References

- Microsoft Docs — Inline Ray Tracing: `https://learn.microsoft.com/windows/win32/direct3d12/inline-raytracing`
- DirectX-Graphics-Samples — Raytraced Shadows: `https://github.com/microsoft/DirectX-Graphics-Samples`
- NVIDIA NRD (ReLAX/SIGMA): `https://github.com/NVIDIAGameWorks/NRD`
- ReSTIR DI (SIGGRAPH 2020): `https://research.nvidia.com/publication/2020-07_spatiotemporal-reservoir-resampling-real-time-ray-tracing-direct-illumination`

---

With a single coherent ray per pixel and modern denoising, you can deliver stable, soft ray-traced shadows that scale to many lights via ReSTIR—all without blowing your frame time.


