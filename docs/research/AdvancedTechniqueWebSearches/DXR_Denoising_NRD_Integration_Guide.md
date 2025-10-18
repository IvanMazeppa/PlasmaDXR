### Real-Time Denoising for DXR: Integrating NVIDIA NRD (ReLAX, ReBLUR, SIGMA)

This guide covers integrating NVIDIA’s NRD denoisers into a DX12/DXR renderer for shadows, specular/glossy reflections, and diffuse GI. It focuses on robust inputs, scheduling, parameterization, and debugging for stable, high-quality results at 1–2 spp.

### Which denoiser for which signal

- **SIGMA**: shadow visibility (0/1 or fractional). Ideal for inline/RT shadow signals.
- **ReLAX**: specular/glossy reflections and indirect; good temporal stability.
- **ReBLUR**: very glossy/specular reflections; stronger blur with good detail preservation.

### Required inputs (typical)

- Noisy signal: radiance (RGB) or visibility (shadow) at the working resolution
- **Hit distance** or view-space depth per pixel (crucial for stability)
- **Normal** (preferably world or view-space) and **roughness**
- **Motion vectors** (screen-space) for temporal reprojection
- Optional: material ID, disocclusion masks, confidence metrics (SSR confidence, ReSTIR W)

Ensure inputs are full-resolution or scaled consistently with the denoised signal.

### Scheduling in the frame

1) Produce your noisy signals (shadows, reflections, GI) at half/quarter/full res.
2) Run NRD passes for each signal (can be compute). Provide stable prev-frame history via double-buffered textures.
3) Upscale to full resolution if needed using normal/roughness/depth-guided filters.
4) Composite denoised results into your lighting pipeline.

### Parameterization starting points

- History length: 30–60 frames with clamp to neighborhood min/max
- Antilag (history sensitivity): medium; increase on fast motion
- Disocclusion threshold: tuned via depth/normal deltas; be conservative
- Hit-distance normalization: normalize to [0,1] by dividing by max trace distance; feed to denoiser
- Specular lobe tuning (ReLAX/ReBLUR): increase stability at high roughness

### Input prep details

- Pack normals in octahedral encoding to save bandwidth; decode in denoiser pass if needed.
- Ensure motion vectors account for dynamic resolution and jitter; provide jitter offsets the denoiser expects.
- Provide roughness in perceptual space if your BRDF uses it for variance guidance.
- For shadows (SIGMA), pass visibility and hit distance; for reflections/GI (ReLAX/ReBLUR), pass radiance, normals, roughness, and hit distance.

### Example integration sequence (compute)

1) Shadow pass (inline or RTPSO): write visibility and hit distance
2) SIGMA denoise → shadow mask
3) Reflection pass (SSR + RT fallback): write radiance and hit distance
4) ReLAX/ReBLUR denoise → reflection radiance
5) (Optional) Indirect GI (ReSTIR GI): write radiance + hit distance
6) ReLAX denoise → diffuse/specular indirect
7) Composite

### Quality tips

- Prefer half-res for soft shadows and rough reflections; full-res for mirror-like reflections.
- Use temporal jitter compatible with TAA; clamp history with neighborhood tests.
- Maintain reactive masks for big lighting/geometry changes to reset history.
- Feed denoisers with accurate hit distance; it is the strongest stabilizer.

### Performance best practices

- Tile/dispatch sizes tuned for your GPU; keep UAV writes coalesced.
- Avoid unnecessary format conversions; store inputs in denoiser-native formats.
- Batch denoising passes to reuse descriptor heaps and pipelines.
- Profile: denoisers are typically a few ms at 1080p; budget accordingly.

### Debugging and validation

- Visualize inputs: normals, roughness, motion, hit distance, and noisy signals.
- Visualize denoiser internal outputs (if available) and history validity.
- Check for ghosting by scrubbing frame-by-frame; adjust antilag and disocclusion thresholds.

### Common pitfalls and fixes

- **Missing/incorrect motion vectors**: causes smearing; ensure camera and object velocities are correct.
- **Wrong hit distance scale**: destabilizes history; normalize by max ray distance or surface scale.
- **Inconsistent resolutions**: mismatch between inputs and signal resolution produces artifacts; keep them aligned.
- **Over-aggressive history**: clamp with neighborhood min/max and use reactive masks on large changes.

### Minimal checklist

- [ ] Provide noisy signal, normals, roughness, motion, hit distance
- [ ] Configure denoiser (SIGMA/ReLAX/ReBLUR) per signal
- [ ] Temporal reprojection with jitter awareness
- [ ] History clamping and reactive masks
- [ ] Visual validation and performance profiling

### References

- NRD (ReLAX, ReBLUR, SIGMA): `https://github.com/NVIDIAGameWorks/NRD`
- NVIDIA Technical Blog — NRD & best practices: `https://developer.nvidia.com/blog/rtx-best-practices/`
- DirectX-Graphics-Samples: `https://github.com/microsoft/DirectX-Graphics-Samples`

---

With robust inputs and conservative history, NRD turns 1–2 spp ray-traced signals into stable, production-quality results suitable for shipping games.


