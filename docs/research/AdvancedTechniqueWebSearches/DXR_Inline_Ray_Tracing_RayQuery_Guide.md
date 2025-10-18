### Inline Ray Tracing (RayQuery) in D3D12: Practical Guide and Best Practices

This guide explains how to implement Inline Ray Tracing using HLSL RayQuery with Direct3D 12 (DXR 1.1). It covers correct usage, integration into raster/compute passes, performance tuning, and image-quality strategies. Use this for effects like shadows, AO, and probes where dedicated RT pipelines are overkill.

### Why inline ray tracing

- Avoids Raytracing Pipeline State (RTPSO) and Shader Binding Table (SBT) setup overhead.
- Integrates directly into existing raster or compute passes; great for coherence when sampling G-buffer.
- Excellent for simple queries: binary shadows, visibility tests, short-distance AO, probe tracing, or single-hit queries.

### Requirements

- DXR 1.1-capable device and driver (Agility SDK recommended).
- Build TLAS/BLAS as usual (see Acceleration Structures guide). RayQuery traverses the same AS.
- Bind TLAS SRV in a descriptor heap accessible to the shader stage where you use RayQuery.

### HLSL: RayQuery basics

Key types and intrinsics:

- `RayQuery<RAY_FLAG_NONE>`: query object used to trace and iterate hits.
- `rayQuery.TraceRayInline(SceneBVH, flags, mask, ray, payload?)`: starts traversal.
- `rayQuery.Proceed()`: advances traversal; returns whether there is more work.
- `rayQuery.CandidateType()`: triangle vs procedural candidate.
- `rayQuery.CommitNonOpaqueTriangleHit()` / `CommitProceduralPrimitiveHit()` to accept a candidate as a hit.
- `rayQuery.CommittedStatus()` / `CommittedPrimitiveIndex()` / `CommittedGeometryIndex()` / `CommittedInstanceID()` to fetch results.

Minimal binary shadow example (HLSL):

```hlsl
// Resources
RaytracingAccelerationStructure SceneBVH   : register(t0);
Texture2D<float> OpacityTex                : register(t1);
SamplerState LinearClamp                   : register(s0);

bool TraceShadow(float3 origin, float3 dir, float tMin, float tMax, uint mask)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = tMin;
    ray.TMax = tMax;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER> rq;
    rq.TraceRayInline(SceneBVH, /*rayFlags*/0, mask, ray);

    // Proceed until either an opaque hit is committed or we finish traversal
    while (rq.Proceed()) {
        if (rq.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
            // Example alpha test (keep minimal; texture LODs should be coherent)
            float2 uv = rq.CandidateTriangleBarycentrics();
            float a = OpacityTex.SampleLevel(LinearClamp, uv, 0);
            if (a < 0.5f) {
                // Transparent: ignore this hit and continue
                continue;
            }
            rq.CommitNonOpaqueTriangleHit();
        }
    }
    return rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
}
```

Notes:

- Use `RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH` for shadow/visibility to stop early.
- Use `RAY_FLAG_SKIP_CLOSEST_HIT_SHADER`; with RayQuery you supply the logic inline.
- Keep any per-candidate work extremely lightweight (a single alpha test at most). Heavy texture work destroys performance as candidates can be numerous.

### Integrating in raster passes

Common pattern: compute direct lighting in a deferred lighting pass; for each light, call `TraceShadow` to test visibility to the light.

Tips for good performance:

- Use **coherent rays**: group pixels by light tile/cluster so shadow rays have similar directions and lengths.
- Prefer **half or quarter resolution** shadows with upsample + temporal stability when soft shadows are desired.
- Clamp max `tMax` to light distance plus penumbra padding; avoid unbounded rays.

### AO with RayQuery

Ambient occlusion needs short, multi-directional rays around the normal.

Recommendations:

- Use a **small radius** (e.g., 0.5–2.0 meters depending on scene scale).
- Cast 2–4 rays per pixel and denoise temporally + spatially; use blue-noise rotation per-pixel.
- Use `RAY_FLAG_FORCE_OPAQUE` to skip any-hit; AO rays should usually treat geometry as opaque.

Example snippet:

```hlsl
float ComputeAO(float3 P, float3 N, uint mask)
{
    const uint kNumRays = 4;
    float occlusion = 0.0;
    for (uint i = 0; i < kNumRays; ++i) {
        float2 xi = BlueNoise(i, DispatchRaysIndex().xy);
        float3 dir = SampleHemisphereCosine(N, xi);
        RayDesc ray = { P + N * 0.01, dir, 0.0, AO_RADIUS };
        RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> rq;
        rq.TraceRayInline(SceneBVH, 0, mask, ray);
        while (rq.Proceed()) {}
        occlusion += (rq.CommittedStatus() == COMMITTED_NOTHING) ? 0.0 : 1.0;
    }
    return 1.0 - occlusion / kNumRays;
}
```

### Resource binding and pipeline setup

- Bind the TLAS as `RaytracingAccelerationStructure` SRV in the descriptor heap visible to the pass using RayQuery.
- No SBT required. Your existing root signatures can remain unchanged except for adding the TLAS SRV and any textures.
- Ray flags are specified per-query. Instance masks in TLAS let you cull ray types at the scene level.

### When to prefer RayQuery vs RTPSO

Prefer RayQuery when:

- You already have a raster/compute pass that can incorporate the ray result inline (shadows, AO, decals projection, light/gi probes).
- You need simple single-hit queries or short-range traces without custom closest-hit logic.
- You want to avoid the complexity and memory of SBT and multiple RT pipelines.

Prefer RTPSO + SBT when:

- You need complex material-dependent shading in closest-hit/any-hit.
- You require multiple ray types with different hit groups or callable shaders.
- You need fine-grained local root data per geometry/material.

### Performance best practices

- **Keep payload small**: although RayQuery does not use payload structs like RTPSO, your live variables around `TraceRayInline` act like a payload. Minimize live state across the loop to avoid spills.
- **Use conservative ray flags**: `ACCEPT_FIRST_HIT_AND_END_SEARCH`, `SKIP_CLOSEST_HIT_SHADER`, `FORCE_OPAQUE`, `CULL_BACK_FACING_TRIANGLES` where applicable.
- **Shorten rays**: Clamp `tMax` to the true query distance; early termination saves traversal.
- **Cohere rays**: Process in screen- or tile-grouped order so rays traverse similar BVH regions.
- **Avoid any-hit logic**: If you must do alpha-test, keep it to a single texture sample with coherent LOD.
- **Instance masks**: Use TLAS masks to exclude irrelevant geometry for a given ray type.

### Debugging and validation

- **PIX**: Capture the pass; use raytracing views to inspect TLAS/BLAS and ray flags. Validate that `CommittedStatus` matches expectation.
- **D3D12 debug layer**: Watch for AS state/usage errors. Ensure UAV barriers after AS builds/updates.
- **Heatmaps**: Visualize number of RayQuery iterations (candidate hits) per pixel to identify overdraw regions.

### Image-quality strategies

- **Temporal accumulation**: Accumulate inline shadow/AO over time; jitter directions and clamp history to avoid lag/ghosting.
- **Blue-noise sampling**: Rotate directions per pixel to minimize structured noise.
- **Denoisers**: Integrate NRD (SVGF-like) for shadows/AO to drop rays per pixel to 1–2 while maintaining stability.
- **Hybrid soft shadows**: Trace penumbra cones or multiple jittered rays at half-res and denoise; upscale with edge-aware filters.

### Common pitfalls and fixes

- **Forgetting UAV barriers on AS**: You still need UAV barriers after AS builds/updates before using RayQuery.
- **Wrong TLAS binding space**: Ensure the TLAS SRV is visible to the shader stage (graphics/compute) that runs the code.
- **Expensive per-candidate logic**: Keep the loop lightweight; heavy texture work can explode cost.
- **Too many long rays**: Clamp `tMax`; use masks to cull irrelevant instances.

### Minimal checklist

- [ ] TLAS/BLAS built and resident; UAV barriers inserted after builds/updates
- [ ] TLAS SRV bound to the pass using RayQuery
- [ ] Ray flags chosen per effect (shadows/AO/probes)
- [ ] Rays clamped in length and directions jittered if accumulating
- [ ] PIX capture shows expected `CommittedStatus` and low candidate counts

### References

- Microsoft Docs — Inline Ray Tracing: `https://learn.microsoft.com/windows/win32/direct3d12/inline-raytracing`
- Microsoft Docs — DXR 1.1: `https://devblogs.microsoft.com/directx/dxr-1-1/`
- DirectX-Graphics-Samples — RayQuery samples: `https://github.com/microsoft/DirectX-Graphics-Samples`
- NVIDIA — Best Practices for RTX (RayQuery guidance applies): `https://developer.nvidia.com/blog/best-practices-using-nvidia-rtx-ray-tracing/`

---

Inline ray tracing lets you add high-value ray-cast effects without full RT pipelines. Keep queries short and coherent, lean on masks and flags, and denoise sparingly for stable, performant results.


