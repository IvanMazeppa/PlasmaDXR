# Volumetric Light Scattering Options Research
**Date:** 2025-11-03
**Focus:** Realtime raytraced volumetric particle light scattering

---

## Research Summary

### NVIDIA RTX Technologies (2024-2025)

#### 1. **RTXGI 2.0** (Updated October 2025)
**Repository:** https://github.com/NVIDIAGameWorks/RTXGI
**Status:** ✅ Actively maintained (last commit: Oct 6, 2025)

**What it provides:**
- Neural Radiance Cache (NRC) - AI-based GI (requires Tensor Cores)
- Spatially Hashed Radiance Cache (SHaRC) - Hardware-agnostic
- DX12 and Vulkan support

**For your use case:**
- ❌ **NOT suitable for volumetric particles**
- Designed for surface-based global illumination
- No mention of volumetric rendering or particle systems
- Uses radiance caching for surface hits during path tracing

**Verdict:** Not appropriate for particle scattering

---

#### 2. **RTX Volumetrics** (Announced GDC 2025)
**Tech:** Volume-based Reservoir Spatio-Temporal Importance Resampling (ReSTIR)
**Example:** Half-Life 2 RTX (released March 18, 2025)

**What it provides:**
- High-contrast volumetric light and shadows
- "Crystal clear beams of light"
- Volume-based ReSTIR algorithm

**Availability:**
- ❌ **No standalone SDK yet**
- Integrated into RTX Remix (for remastering classic games)
- Not available as separate library for custom engines

**Verdict:** This is EXACTLY what you need, but not publicly available yet

---

#### 3. **Academic Volumetric ReSTIR** (SIGGRAPH Asia 2021)
**Repository:** https://github.com/DQLin/VolumetricReSTIRRelease
**Paper:** "Fast Volume Rendering with Spatiotemporal Reservoir Resampling"
**Status:** ⚠️ Archived research code

**What it provides:**
- Volumetric path tracing with ReSTIR
- Implementation in Falcor 4.2 framework
- Optimized for RTX 3090

**Requirements:**
- Specific driver version (471.68) - newer drivers cause slowdown
- VDB/GVDB volumetric data format (voxel grids, NOT particles)
- Windows 10 1809+, Visual Studio 2019

**For your use case:**
- ❌ **NOT suitable for particles**
- Designed for voxel-based volumes (clouds, explosions)
- Requires converting data to `.vbx` format with mipmapping
- Framework dependency (Falcor 4.2)

**Verdict:** Wrong data representation (voxels vs particles)

---

#### 4. **RTXGI DDGI** (Original - Last updated ~2 years ago)
**Repository:** https://github.com/NVIDIAGameWorks/RTXGI-DDGI
**Status:** ⚠️ Superseded by RTXGI 2.0

**What it provided:**
- Dynamic Diffuse Global Illumination
- Probe-based GI (similar to your idea)
- 32×32×32 probe grids

**Why it's outdated:**
- Replaced by RTXGI 2.0 with NRC/SHaRC
- 2-year gap in updates
- Newer techniques are more efficient

**Verdict:** Use RTXGI 2.0 instead, but still not for particles

---

### Alternative Technologies

#### Unreal Engine 5 - Lumen
**What it is:** Real-time global illumination and reflections
**Key features:**
- Software + Hardware ray tracing (GPU-agnostic)
- Kilometer-scale environments
- Fully dynamic (no baking)

**For your use case:**
- ⚠️ **Engine-locked** (Unreal Engine only)
- Designed for surfaces, not volumetric particles
- Would require porting entire project to UE5

**Verdict:** Not viable (different engine)

---

#### Unity HDRP - RTGI/SSGI
**What it is:** Ray-traced + Screen-space global illumination
**Key features:**
- Multiple tracing modes (ray marching, ray tracing, mixed)
- Screen-space and off-screen geometry support
- Includes particles in GI calculations

**For your use case:**
- ⚠️ **Engine-locked** (Unity HDRP only)
- Can handle "on-screen opaque particles"
- Designed for game engine integration

**Verdict:** Not viable (different engine)

---

## What You Actually Need

### Requirements Analysis

**Your specific problem:**
1. ✅ 10K+ volumetric particles
2. ✅ Smooth light scattering (not discrete brightness jumps)
3. ✅ Many-light support (16+ lights minimum, ideally 100-200)
4. ✅ Real-time performance (60+ FPS @ 1440p on RTX 4060 Ti)
5. ✅ DirectX 12 / DXR 1.1 native

**Why current inline RayQuery fails:**
- Only affects overall particle brightness
- No volumetric scattering or light transport
- Creates "flashing" discrete jumps in brightness

**Why Volumetric ReSTIR is failing:**
- Custom implementation is extremely complex
- GPU hang issues at 2045 particles
- 6+ hours of debugging with no resolution
- Reimplementing cutting-edge research is high-risk

---

## Recommendations

### Option A: Wait for RTX Volumetrics SDK ⏳

**Pros:**
- EXACTLY what you need (volumetric ReSTIR for particles)
- Production-tested (Half-Life 2 RTX uses it)
- NVIDIA support and documentation

**Cons:**
- No public release date announced
- May be months or years away
- Might be locked to RTX Remix framework

**Timeline:** Unknown (likely 6-12+ months)

---

### Option B: Hybrid Probe Grid (Custom Implementation) ⚡ **RECOMMENDED**

Build a simplified version of RTXGI probe-based GI specifically for particles:

**Architecture:**
```
32×32×32 Probe Grid → Probes sample 16 lights → Particles query 8 nearest probes
```

**Implementation:**
1. Create sparse 3D grid (32³ = 32,768 probes)
2. Each probe stores spherical harmonics (4 RGB coefficients = 48 bytes)
3. Update probes every 4 frames (amortized cost)
4. Particles do trilinear interpolation between 8 nearest probes
5. Total memory: 32,768 × 48 bytes = 1.5 MB

**Advantages:**
- Much simpler than full Volumetric ReSTIR
- Handles unlimited lights (baked into probes)
- Smooth scattering via interpolation
- ~2-3 weeks implementation time

**Performance estimate:**
- Probe updates: ~8,000 probes × 64 rays = 512K rays (every 4 frames)
- Particle queries: 10K particles × 8 probe reads = 80K reads (every frame)
- Expected: 90+ FPS @ 1440p

**Disadvantages:**
- Lower quality than full path tracing
- Temporal lag (probes update every 4 frames)
- Doesn't handle single-bounce paths perfectly

---

### Option C: ReSTIR DI for Particles (Alternative Approach) 🔄

Use NVIDIA's RTX Dynamic Illumination SDK (ReSTIR DI) adapted for particles:

**What it is:**
- Importance sampling for direct illumination
- Handles many lights efficiently
- Available in NVIDIA RTX Kit

**Adaptation for particles:**
1. Each particle is a "surface" with spherical BRDF
2. Use ReSTIR DI to select important lights per particle
3. Add phase function (Henyey-Greenstein) for scattering
4. Temporal reuse across frames

**Advantages:**
- Production SDK (well-tested)
- Handles 100s of lights efficiently
- Better than your current inline RayQuery

**Disadvantages:**
- Still not true volumetric scattering
- Requires adapting surface-based SDK for particles
- May not solve "discrete brightness jumps" completely

---

### Option D: Continue Debugging Volumetric ReSTIR ⚠️ **NOT RECOMMENDED**

**Time invested:** 6+ hours
**Progress:** Still crashing at 2045 particles
**Complexity:** Extremely high (cutting-edge research)

**Why not recommended:**
- Diminishing returns on debugging effort
- No guarantee it will work even when fixed
- Reimplementing research papers is notoriously difficult
- Your time better spent on alternatives

---

## Specific Recommendations for Your Project

### Immediate (This Week)

1. **Set up log analysis RAG** (2-3 hours)
   - Will help with current debugging AND future issues
   - Strategic investment in developer productivity

2. **Try hybrid probe grid approach** (2-3 days MVP)
   - Simpler than Volumetric ReSTIR
   - Higher chance of success
   - Gets you 80% of the visual quality with 20% of the complexity

### Short Term (Next 2 Weeks)

3. **Implement probe-based particle scattering** (full implementation)
   - 32³ probe grid
   - Spherical harmonics storage
   - Trilinear interpolation for particles
   - Target: Smooth scattering from 16+ lights

### Long Term (1-3 Months)

4. **Monitor RTX Volumetrics SDK release**
   - Check NVIDIA developer site monthly
   - Switch if/when SDK becomes available
   - Your probe grid can serve as placeholder

---

## Technical Comparison

| Technique | Complexity | Quality | Performance | Availability | Particle Support |
|-----------|------------|---------|-------------|--------------|------------------|
| RTX Volumetrics | ??? | Excellent | Excellent | ❌ No SDK | ✅ Yes |
| RTXGI 2.0 | Medium | Excellent | Excellent | ✅ Available | ❌ No |
| Volumetric ReSTIR (DIY) | Extreme | Excellent | Good | ⚠️ Research | ✅ Yes (buggy) |
| Hybrid Probe Grid | Low-Medium | Good | Excellent | ✅ DIY | ✅ Yes |
| ReSTIR DI (adapted) | Medium | Good | Excellent | ✅ Available | ⚠️ Adaptation needed |
| Current inline RayQuery | Low | Poor | Excellent | ✅ Working | ⚠️ No scattering |

---

## Conclusion

**Your problem:** Volumetric particle light scattering with smooth transitions and many-light support

**Best available solution:** **Hybrid probe grid** (custom implementation)
- Achieves your goals without extreme complexity
- 2-3 week implementation vs months of debugging
- Production-ready quality
- Can upgrade to RTX Volumetrics if SDK releases

**What to avoid:**
- ❌ Continuing Volumetric ReSTIR debugging (sunk cost fallacy)
- ❌ RTXGI 2.0 (wrong use case - surfaces not particles)
- ❌ Engine-locked solutions (Lumen, Unity HDRP)

**Timeline:**
- **Week 1:** Log analysis RAG + probe grid MVP
- **Week 2-3:** Full probe-based scattering implementation
- **Week 4:** Polish, optimize, integrate with existing lighting

**Expected result:**
- Smooth volumetric scattering for 10K particles
- Support for 50-100 lights (via probe pre-computation)
- 90+ FPS @ 1440p on RTX 4060 Ti
- No more brightness "flashing" or discrete jumps

---

**Next Action:** Would you like me to:
A) Design the hybrid probe grid architecture
B) Try the GPU fence fix one more time for Volumetric ReSTIR
C) Set up the log analysis RAG system first
