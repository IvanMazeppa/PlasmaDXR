### Ray-Traced Reflections in D3D12/DXR: Hybrid SSR, Denoising, and Performance

This guide describes a robust, production-ready approach to ray-traced reflections (RTR) in Direct3D 12 Raytracing. We focus on a hybrid SSR + RT solution, payload minimization, masking, and modern denoising to achieve stable, high-quality reflections at low cost.

### Design goals

- High-quality reflections with minimal rays per pixel (1 rpp typical)
- Hybrid with Screen-Space Reflections (SSR) for primary hits; RT fallback for off-screen/missing data
- Temporal stability with responsive history clamping
- Content-aware performance knobs (masks, max distance, roughness-based cutoffs)

### Pipeline choices: Inline vs RTPSO

- Use **RTPSO + SBT** when you require material-dependent shading, importance sampling of microfacet lobes, or callable utilities.
- Use **Inline (RayQuery)** for simpler “first-hit only” reflection queries fused into a lighting/resolve pass.

A common architecture: RTPSO for reflections to leverage closest-hit material logic and compact SBT with a small local root (material index).

### Input data and G-buffer

You will need per-pixel:

- World-space position P, view vector V, normal N, roughness α, metallic, and material ID
- Velocity/motion vectors for temporal reprojection
- Depth and surface flags (sky, transmissive, etc.)

### Ray setup

- Reflection direction: `R = reflect(-V, N)`. For rough surfaces, importance sample the GGX lobe around R using a low-discrepancy sequence (blue-noise + frame index).
- Set `tMin` to a small offset along N to avoid self-intersection; `tMax` to a max distance scaled by roughness.
- Instance mask: exclude irrelevant instances (foliage, decals) for reflection rays.

### Hybrid with SSR

Algorithm per pixel:

1) Attempt SSR with hierarchical Z (HiZ) and thickness tests. If a hit is found and parallax is acceptable, use SSR result.
2) If SSR fails or is low confidence (edge cases, disocclusions), cast a single RT reflection ray.
3) Combine with temporal history using confidence weights, then denoise.

This saves many RT rays in view-coherent regions and keeps costs bounded.

### RTPSO configuration for reflections

- Minimal payload: encode radiance (e.g., `float3` or packed uint) + roughness or a small bitfield if needed.
- Shader config: `MaxPayloadSizeInBytes` as small as possible (12–16 bytes typical); attribute size 8 bytes for triangles.
- Pipeline config: recursion depth 1 for most cases (primary rays only). For reflection → shadow checks, prefer a secondary pass or inline shadow queries rather than recursion.
- Hit groups: specialize by material families if it improves I-cache; avoid any-hit unless doing alpha-test.

### Closest-hit shading

- Sample material textures; compute BRDF reflectance term and environment lighting fallback.
- For mirror-like materials, you may directly fetch reflection color from a precomputed probe/skybox when no hit or beyond `tMax`.
- For rough reflections, importance sample one lobe direction (already done by ray dir sampling); keep payload small by accumulating locally and returning a compact result.

### Inline path (RayQuery) snippet

```hlsl
RaytracingAccelerationStructure SceneBVH : register(t0);

float3 TraceReflection(float3 P, float3 N, float3 V, float roughness, uint mask)
{
    float3 R = ImportanceSampleGGX(V, N, roughness, DispatchRaysIndex().xy);
    RayDesc ray = { P + N * 0.003, R, 0.0, REFLECTION_MAX_DISTANCE(roughness) };
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER> rq;
    rq.TraceRayInline(SceneBVH, 0, mask, ray);
    float3 color = 0;
    while (rq.Proceed()) {
        if (rq.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
            rq.CommitNonOpaqueTriangleHit();
        }
    }
    if (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        // Resolve material from instance/primitive IDs; fetch color via bindless material buffers
        uint inst = rq.CommittedInstanceID();
        uint prim = rq.CommittedPrimitiveIndex();
        uint mat  = ResolveMaterial(inst, prim);
        color = ShadeMaterial(mat, rq.CommittedObjectRayDirection(), roughness);
    } else {
        color = SampleEnvironment(R);
    }
    return color;
}
```

Keep per-candidate work tiny. For alpha-test, add one texel read max.

### Denoising reflections (NRD ReLAX or ReBLUR)

- Use **NRD ReLAX** for glossy reflections; **ReBLUR** for very glossy/specular.
- Inputs: current noisy reflection, view-space normals, roughness, depth, motion vectors, and reflection hit distance.
- Stabilize with temporal reprojection; clamp history using neighborhood clamping and hit-distance confidence.
- Run spatial filter afterward with normal/roughness/depth guidance.

Starting parameters:

- rpp = 1 (or 2 for very rough surfaces)
- Max distance scaled by roughness; fade to probes for far rays
- History length 30–60 frames; reactive mask on large lighting/geometry changes

### Performance best practices

- Limit RT rays via SSR first; only fallback when missing/off-screen.
- Use instance masks and exclude small or irrelevant geometry from reflection rays.
- Clamp `tMax` aggressively; far reflections contribute little.
- Use half-resolution for rough reflections; upscale with normal-aware filters.
- Minimize payload and live state across trace to reduce register pressure.
- Avoid any-hit unless absolutely necessary.

### Validation and debugging

- Visualize ray length, hit distance, and SSR confidence. Ensure fallback triggers as intended.
- PIX: inspect RT dispatch, SBT records, and instance masks; confirm no any-hit hotspots.
- Track denoiser inputs for correctness (normals, motion, roughness).

### Common pitfalls and fixes

- **No SSR fallback**: full RT everywhere is expensive. Implement hybrid path early.
- **Oversized payloads**: trim to essentials; pack into 12–16 bytes.
- **Long rays**: cap distances and prefer environment fallback.
- **Any-hit overuse**: hurts perf; move alpha tests to closest-hit or inline candidate loop.

### Minimal checklist

- [ ] SSR first, RT fallback only on failure/low confidence
- [ ] Small payload and recursion depth = 1
- [ ] Aggressive distance limits and instance masks
- [ ] NRD ReLAX/ReBLUR denoising with stable inputs
- [ ] PIX validation clean; heatmaps show bounded ray lengths

### References

- Microsoft Docs — DXR overview: `https://learn.microsoft.com/windows/win32/direct3d12/directx-raytracing`
- DirectX-Graphics-Samples — Reflections samples: `https://github.com/microsoft/DirectX-Graphics-Samples`
- NVIDIA NRD (ReLAX/ReBLUR): `https://github.com/NVIDIAGameWorks/NRD`
- SSR algorithms and HiZ techniques: `https://iryoku.com/downloads/Practical_Real-Time_Strategies_for_Accurate_Indirect_Occlusion.pdf`

---

With a hybrid SSR-first approach, minimal payloads, and modern denoising, ray-traced reflections become practical at 1 rpp, delivering crisp, stable results at real-time frame budgets.


