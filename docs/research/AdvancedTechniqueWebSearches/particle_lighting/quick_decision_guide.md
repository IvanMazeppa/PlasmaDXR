# Particle Lighting Without Atomics: Quick Decision Guide

## THE VERDICT: Start with Texture Splatting

**Implementation Order:**
1. Texture Splatting (this week) - validate approach
2. Radix Sort (if quality insufficient) - upgrade later

---

## THREE TECHNIQUES SUMMARY

### 1. RADIX SORT + BINNED ACCUMULATION
**Best for:** Exact neighbor finding, stable long-term solution
- Sort particles by grid cell (AMD FidelityFX Parallel Sort)
- Read neighbors from sorted array
- Direct writes (no atomics)
- **Cost:** 2-4ms, 3-5 days dev time
- **Risk:** Medium

### 2. TEXTURE SPLATTING (RECOMMENDED)
**Best for:** Fast implementation, "good enough" quality
- Render particles into 3D texture (additive blend)
- Sample texture for lighting
- Hardware blending (no atomics)
- **Cost:** 0.7-1.4ms, 2-3 days dev time
- **Risk:** Low

### 3. STENCIL BINNING
**Best for:** Graphics pipeline experts only
- Use stencil buffer for bin slot allocation
- Particle IDs to framebuffer
- Complex but no atomics
- **Cost:** 1.8-3.2ms, 4-6 days dev time
- **Risk:** High

---

## IMPLEMENTATION CHECKLIST: TEXTURE SPLATTING

### Day 1: Setup
- [ ] Create 3D texture (128x128x64, R11G11B10_FLOAT)
- [ ] Map disk volume to texture coordinates
- [ ] Write geometry shader for particle splatting

### Day 2: Splatting Pass
- [ ] Compute shader: Splat emissive particles
- [ ] Manual loop: Write to voxels in radius
- [ ] Test with simple falloff function

### Day 3: Lighting Pass
- [ ] Compute shader: Sample 3D texture per particle
- [ ] Apply trilinear filtering
- [ ] Integrate with existing particle renderer

---

## KEY RESOURCES

**Must Read:**
- AMD FidelityFX Parallel Sort: https://gpuopen.com/fidelityfx-parallel-sort/
- GPU Gems 3 Ch 23 (Particles): https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-23-high-speed-screen-particles

**Reference:**
- NVIDIA Parallel Prefix Sum: https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
- Mike Turitzin Compute Shaders: https://miketuritzin.com/post/rendering-particles-with-compute-shaders/

---

## HLSL STARTER (TEXTURE SPLATTING)

```hlsl
RWTexture3D<float4> lightAccumTexture : register(u0);
StructuredBuffer<Particle> particles : register(t0);

[numthreads(256, 1, 1)]
void SplatEmissiveParticles(uint3 DTid : SV_DispatchThreadID)
{
    uint particleID = DTid.x;
    Particle p = particles[particleID];

    float3 uvw = WorldToTextureUVW(p.position);
    float3 emission = GetEmission(p.temperature);

    // Splat to voxels in radius
    for (int z = -2; z <= 2; z++)
    for (int y = -2; y <= 2; y++)
    for (int x = -2; x <= 2; x++) {
        float3 offset = float3(x, y, z);
        float dist = length(offset);
        if (dist > 2.5) continue;

        float falloff = 1.0 - dist / 2.5;
        uint3 voxel = uint3((uvw + offset / 128.0) * 128.0);
        lightAccumTexture[voxel] += float4(emission * falloff * falloff, 1);
    }
}
```

---

## PERFORMANCE TARGETS (RTX 4060 Ti, 100k particles)

| Technique | Expected Cost | Bottleneck |
|-----------|---------------|------------|
| Texture Splatting | 0.7-1.4ms | Texture writes |
| Radix Sort | 2-4ms | Sort + memory BW |
| Stencil Binning | 1.8-3.2ms | Rasterization |

**Target:** 60fps = 16.67ms budget → Any technique fits!

---

## TROUBLESHOOTING

**If texture splatting is too blurry:**
- Increase 3D texture resolution (128³ → 256³)
- Use R16G16B16A16_FLOAT for higher precision
- Add sharpening filter in sampling pass

**If radix sort is too slow:**
- Reduce sort frequency (every 2-3 frames)
- Use smaller grid cells (faster neighbor search)
- Profile with PIX/NSight - may be prefix sum bottleneck

**If stencil binning overflows:**
- Increase max_particles_per_cell
- Split into multiple binning passes
- Consider hybrid: stencil for dense areas, direct for sparse
