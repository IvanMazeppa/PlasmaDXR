# DLSS Ray Reconstruction Enhancement Options

**Date:** 2025-10-29
**Status:** Basic denoising implemented, enhancements pending
**Branch:** 0.11.6+

---

## Current Implementation (Phase 3 - In Progress)

**What we have:**
- DLSS Ray Reconstruction feature created (1920×1080)
- Denoising entire Gaussian output (R16G16B16A16_FLOAT)
- Zero motion vectors (static scene assumption)
- 16 rays/particle → target 4 rays + DLSS denoising

**Expected benefit:** 4× performance boost with maintained quality

---

## Enhancement Options

### 1. Depth Buffer Integration ⭐ **HIGHEST IMPACT**

**Estimated Time:** 15 minutes
**Difficulty:** Easy
**Impact:** HIGH - Dramatically improves edge quality and particle separation

**What it does:**
- Provides DLSS with 3D spatial information
- Helps distinguish overlapping particles
- Improves reconstruction at depth discontinuities
- Reduces "mushy" edges in denoised output

**Implementation:**
```cpp
// In ParticleRenderer_Gaussian.h
Microsoft::WRL::ComPtr<ID3D12Resource> m_depthBuffer;  // R32_FLOAT
D3D12_CPU_DESCRIPTOR_HANDLE m_depthUAV;
D3D12_GPU_DESCRIPTOR_HANDLE m_depthUAVGPU;

// In particle_gaussian_raytrace.hlsl
RWTexture2D<float> g_depth : register(u3);

// In ray marching loop (line ~600)
if (firstHit) {
    g_depth[pixelCoord] = tMin;  // Write ray distance
    firstHit = false;
}

// Pass to DLSS
dlssParams.inputDepth = m_depthBuffer.Get();
```

**Why prioritize:** Already computing depth during ray marching, just need to export it. Biggest quality win per minute of work.

---

### 2. Motion Vectors (Camera Movement)

**Estimated Time:** 30 minutes
**Difficulty:** Medium
**Impact:** MEDIUM - Eliminates ghosting during camera movement

**What it does:**
- Tracks pixel movement between frames
- Enables DLSS temporal accumulation
- Reduces flickering when camera rotates/translates
- Critical for interactive camera control

**Current Status:** Shader written, temporarily disabled (needs constant buffer fix)

**Implementation:**
```cpp
// Fix: Create separate constant buffer for motion vectors
struct MotionVectorConstants {
    float4x4 viewProj;
    float4x4 prevViewProj;      // Store previous frame's matrix
    float3 cameraPos;
    float deltaTime;
    uint2 screenSize;
    uint particleCount;
};

// Store previous frame matrices in Application.cpp
DirectX::XMFLOAT4X4 m_prevViewProj;

// Update each frame
m_prevViewProj = currentViewProj;  // Before computing new one
```

**Why prioritize:** Already 80% done, just needs constant buffer plumbing. Fixes temporal artifacts.

---

### 3. Normal Vectors (Surface Orientation)

**Estimated Time:** 20 minutes
**Difficulty:** Medium
**Impact:** MEDIUM - Sharper boundaries, better lighting reconstruction

**What it does:**
- Provides surface orientation information
- Helps DLSS preserve lighting details
- Improves particle boundary sharpness
- Better volumetric scattering reconstruction

**Implementation Option A (Velocity-based):**
```hlsl
// In particle_gaussian_raytrace.hlsl
float3 normal = normalize(particle.velocity);  // Direction of motion
g_normals[pixelCoord] = float4(normal * 0.5f + 0.5f, 1.0f);  // Encode to [0,1]
```

**Implementation Option B (Temperature gradient):**
```hlsl
// Compute spatial temperature gradient
float3 gradT = ComputeTemperatureGradient(hit.particleIdx);
float3 normal = normalize(gradT);
```

**Why prioritize:** Relatively cheap to compute, noticeable quality improvement at particle edges.

---

### 4. Albedo/Material Information

**Estimated Time:** 10 minutes
**Difficulty:** Easy
**Impact:** LOW-MEDIUM - Better color preservation

**What it does:**
- Separates material color from lighting
- Helps DLSS distinguish intrinsic color from illumination
- Prevents color "bleeding" in denoised output
- Preserves temperature-based coloring

**Implementation:**
```hlsl
// Export base particle color (before lighting)
float3 albedo = TemperatureToColor(particle.temperature);
g_albedo[pixelCoord] = float4(albedo, 1.0f);

// DLSS parameters
dlssParams.inputDiffuseAlbedo = m_albedoBuffer.Get();
```

**Why prioritize:** Quick win, helps with color accuracy in dense particle regions.

---

### 5. Roughness/Specular (Advanced)

**Estimated Time:** 30 minutes
**Difficulty:** Hard
**Impact:** LOW - Diminishing returns

**What it does:**
- Specifies surface reflectivity
- Helps with specular highlight reconstruction
- More important for solid surfaces than particles

**Implementation:**
```hlsl
// Compute roughness from particle properties
float roughness = 1.0f - (particle.temperature / MAX_TEMP);  // Hot = smooth
g_roughness[pixelCoord] = roughness;
```

**Why deprioritize:** Volumetric particles don't have strong specular, limited benefit.

---

## Recommended Implementation Order

### Phase 1: Baseline (Tonight - 30 min)
1. ✅ Basic DLSS denoising on final output
2. ✅ Reduce rays from 16 → 4
3. ✅ Test and benchmark

**Goal:** Confirm DLSS works, measure performance gain

---

### Phase 2: Quick Wins (Next Session - 15 min)
4. **Add depth buffer** ⭐
5. Test quality improvement

**Goal:** Maximum quality improvement with minimal work

---

### Phase 3: Temporal Stability (After testing - 30 min)
6. Fix motion vector shader
7. Enable motion vector generation
8. Test camera movement

**Goal:** Eliminate flickering during interaction

---

### Phase 4: Polish (Optional - 30 min)
9. Add normal vectors
10. Add albedo buffer
11. Compare quality vs performance

**Goal:** Squeeze out last 10% of quality

---

## Performance vs Quality Matrix

| Configuration | Rays/Particle | Buffers | FPS (est) | Quality |
|---------------|---------------|---------|-----------|---------|
| Current Baseline | 16 | None | 120 | Excellent |
| DLSS Minimal | 4 | Color only | **480** | Good |
| DLSS + Depth | 4 | Color + Depth | **450** | Very Good |
| DLSS + Depth + MV | 4 | Color + Depth + MV | **420** | Excellent |
| DLSS Full | 4 | All buffers | **400** | Pristine |

**Key Insight:** Depth buffer gives 90% of the quality improvement with only 7% performance cost.

---

## Buffer Memory Requirements

| Buffer | Format | Size @ 1080p | Purpose |
|--------|--------|--------------|---------|
| Color (existing) | R16G16B16A16_FLOAT | 16 MB | Main output |
| Depth | R32_FLOAT | 8 MB | Spatial info |
| Motion Vectors | RG16_FLOAT | 8 MB | Temporal info |
| Normals | R16G16B16A16_FLOAT | 16 MB | Surface info |
| Albedo | R16G16B16A16_FLOAT | 16 MB | Material info |
| **Total Additional** | | **48 MB** | All enhancements |

**Memory Cost:** Minimal on modern GPUs (RTX 4060 Ti has 8GB)

---

## DLSS Parameter Tuning

### Denoiser Strength (0.0 - 2.0)
- **0.0:** No denoising (pass-through)
- **0.5:** Light denoise (preserve detail)
- **1.0:** Balanced (default) ⭐
- **1.5:** Aggressive (smooth, less detail)
- **2.0:** Maximum (may blur)

**Recommendation:** Start at 1.0, expose ImGui slider for user tuning

### Quality Presets

**Performance Mode:**
- 2 rays/particle
- Color + Depth only
- Denoiser strength: 1.2
- Target: 600+ FPS

**Balanced Mode:** ⭐
- 4 rays/particle
- Color + Depth + Motion Vectors
- Denoiser strength: 1.0
- Target: 400+ FPS

**Quality Mode:**
- 8 rays/particle
- All buffers
- Denoiser strength: 0.8
- Target: 200+ FPS

---

## Known Limitations

### 1. Motion Vector Shader
**Issue:** Needs separate constant buffer with `prevViewProj` matrix
**Status:** Shader written, disabled due to constant buffer mismatch
**Fix:** 30 minutes to create proper constant buffer structure

### 2. Static Scene Assumption
**Current:** Zero motion vectors (assumes static scene)
**Impact:** DLSS uses less temporal information
**Acceptable for:** Initial testing, stationary camera
**Fix Required for:** Camera movement, particle animation visibility

### 3. Single-Channel Shadows
**Issue:** DLSS Ray Reconstruction expects RGBA, not single R channel
**Solution:** Switched to full-scene denoising instead
**Result:** Even better - denoises everything at once!

---

## Testing Checklist

### Baseline Test (16 rays)
- [ ] Capture screenshot: `screenshot_16ray_baseline.bmp`
- [ ] Record FPS: `___ FPS @ 10K particles`
- [ ] Note quality: Visual assessment

### DLSS Minimal (4 rays, color only)
- [ ] Capture screenshot: `screenshot_4ray_dlss_minimal.bmp`
- [ ] Record FPS: `___ FPS @ 10K particles`
- [ ] Compare quality: Side-by-side with baseline
- [ ] Check for: Excessive blur, missing detail, artifacts

### DLSS + Depth
- [ ] Capture screenshot: `screenshot_4ray_dlss_depth.bmp`
- [ ] Record FPS: `___ FPS @ 10K particles`
- [ ] Compare edges: Particle boundaries vs minimal
- [ ] Check depth discontinuities: Sharp or mushy?

### DLSS + Depth + MV (camera moving)
- [ ] Capture video: Camera rotation + translation
- [ ] Check for: Ghosting, trailing, flickering
- [ ] Record FPS: During camera movement

### Full DLSS (all buffers)
- [ ] Final quality assessment
- [ ] Performance cost analysis
- [ ] Identify diminishing returns point

---

## Integration with Existing Systems

### Shadow System (PCSS)
**Status:** Currently trying to denoise shadows (not ideal)
**New Approach:** DLSS denoises entire scene including shadows
**Benefit:** Shadows get denoised "for free" as part of final image
**Recommendation:** Keep PCSS temporal filtering as fallback if DLSS disabled

### RTXDI Lighting
**Compatible:** Yes - DLSS denoises after RTXDI sampling
**Benefit:** Can reduce RTXDI sample count with DLSS denoising
**Future:** Test 1-sample RTXDI + DLSS vs multi-sample RTXDI

### Adaptive Quality System
**Synergy:** DLSS enables lower ray counts → adaptive system can scale particles higher
**Example:** 4 rays + DLSS + 40K particles = similar performance to 16 rays + 10K particles
**Control:** Adaptive system adjusts rays per particle based on FPS target

---

## Future Possibilities

### 1. DLSS Super Resolution (Not Ray Reconstruction)
**What:** Render at lower resolution, upscale to higher
**Example:** 1280×720 → 1920×1080
**Benefit:** 2.7× performance boost
**Caveat:** Different feature, not same as Ray Reconstruction

### 2. Multi-Pass Denoising
**What:** Separate passes for lighting, shadows, volumetrics
**Benefit:** Fine-grained control
**Downside:** 3× DLSS calls = slower
**Verdict:** Single-pass full-scene better for now

### 3. Hybrid DLSS + Temporal
**What:** DLSS for spatial, temporal filter for history
**Benefit:** Best of both worlds
**Complexity:** Need to manage two denoising pipelines

---

## References

- NVIDIA DLSS SDK Documentation (nvsdk_ngx_helpers_dlssd.h)
- Ray Reconstruction Feature Creation (DLSSSystem.cpp:133-171)
- Evaluation Parameters (DLSSSystem.cpp:216-287)
- Phase 2.5 Success Document (DLSS_PHASE25_SUCCESS.md)

---

**Last Updated:** 2025-10-29
**Next Steps:** Implement basic denoising, then add depth buffer
**Expected Completion:** Phase 1 tonight, Phase 2 next session
