# Screen-Space Shadow System - Implementation Status & Debugging Guide

**Date:** 2025-11-07
**Branch:** 0.14.2
**Status:** Phase 2 Partially Complete (Debugging Minimal Visual Effect)

---

## Current Implementation Status

### ✅ Completed (Phase 1 & 2 Partial)

1. **Depth Pre-Pass Pipeline** (Phase 1)
   - R32_UINT depth buffer created at render resolution (1484×836 with DLSS)
   - Compute shader projects particles to screen space using `InterlockedMin`
   - DLSS-aware: Depth buffer recreated when resolution changes
   - Initialization: Clear shader sets all pixels to 1.0 (far plane) before pre-pass

2. **Screen-Space Shadow Ray March** (Phase 2)
   - `ScreenSpaceShadow()` function in `particle_gaussian_raytrace.hlsl` (lines 377-473)
   - Samples depth buffer in screen space to detect occlusion
   - Contact hardening: Shadows sharper near contact, softer at distance
   - Quality control: 8/16/32 steps (ImGui slider)

3. **Root Signature Fixes**
   - Fixed CBV vs root constants mismatch (was causing crashes)
   - Fixed DLSS resolution mismatch (was causing TDR)
   - Added depth buffer clear to prevent halo artifacts

### 🚧 Current Issues

1. **Minimal/No Visual Effect**
   - Shadows are barely visible or not visible at all
   - Need to debug why occlusion detection isn't working
   - PCSS system may be interfering (both enabled simultaneously)

2. **2044 Particle Limit Crash (TDR)**
   - Same 5-second TDR pause as original probe grid bug
   - Crash occurs during initialization with >2044 particles
   - **Root Cause:** Depth pre-pass likely writing to wrong buffer or BLAS mismatch
   - **Hypothesis:** Depth pre-pass reads full particle buffer (10K) but probe grid BLAS only has 2044
   - **Fix Needed:** Depth pre-pass must read from correct particle range or skip probe particles

3. **PCSS Interaction**
   - Both PCSS (F5) and screen-space shadows can be enabled
   - They may be fighting for the same shadow buffer slots
   - Need to test with PCSS fully disabled

### ⏳ Not Yet Implemented (From Design Doc)

- **Phase 3:** Volumetric Self-Shadowing (density accumulation, Beer-Lambert attenuation)
- **Phase 4:** Enhanced Temporal Accumulation (8-16 frame history)
- **Phase 5:** Full Integration (probe grid, inline RQ, multi-light compatibility)

---

## Architecture Overview

### Depth Pre-Pass Flow

```
1. RenderDepthPrePass() called from Application.cpp (lines 984-989)
   ↓
2. Clear depth buffer to 1.0 (far plane)
   - Shader: depth_buffer_clear.hlsl
   - 8×8 thread groups, dispatch: (width+7)/8 × (height+7)/8
   ↓
3. UAV barrier (ensure clear completes)
   ↓
4. Depth pre-pass compute shader
   - Shader: depth_prepass.hlsl
   - Reads: g_particles (t0: StructuredBuffer<Particle>)
   - Writes: g_depthBuffer (u0: RWTexture2D<uint>)
   - Projects particles to screen space
   - InterlockedMin(g_depthBuffer[pixel], asuint(depth))
   ↓
5. UAV barrier (ensure writes complete)
   ↓
6. Main Gaussian rendering reads g_shadowDepth (t8: Texture2D<uint>)
```

### Shadow Application Flow

```
Multi-Light Loop (particle_gaussian_raytrace.hlsl lines 1106-1140)
   ↓
For each light:
   ↓
   if (useScreenSpaceShadows != 0)
      shadowTerm = ScreenSpaceShadow(pos, lightDir, lightDist, ssSteps)
   else if (useShadowRays != 0)
      shadowTerm = CastPCSSShadowRay(...)
   ↓
   lightContribution = light.color * intensity * attenuation * shadowTerm * phase
```

---

## PIX Buffer Analysis Guide

### Files Captured

1. **g_shadowDepth.bin** (1484×836 × 4 bytes = 4.96 MB)
   - Format: R32_UINT (float depth as uint bits)
   - Each pixel: 4 bytes (uint32)
   - Expected values: 0x3F800000 (1.0 far plane) or particle depths

2. **g_output.hdr** (1484×836 × 8 bytes = 9.92 MB)
   - Format: R16G16B16A16_FLOAT (HDR render target)
   - Each pixel: 8 bytes (4× float16)
   - Final rendered image before DLSS upscaling

3. **g_particleLighting.bin** (2044 particles × 16 bytes = 32 KB)
   - Format: Structured buffer of float4 per particle
   - Each entry: 16 bytes (XYZW lighting color)
   - RT computed lighting per particle

4. **g_probeGrid.bin** (110,592 probes × 128 bytes = 13.5 MB)
   - Format: Probe grid structure (irradiance + visibility)
   - Mostly zeros (sparse data expected)
   - Only irradiance_0 has data, irradiance_1/2 empty

### Parsing Instructions

#### 1. Parse g_shadowDepth.bin (Depth Buffer)

```python
import numpy as np
import struct

# Read depth buffer
width, height = 1484, 836
with open('g_shadowDepth.bin', 'rb') as f:
    depth_data = np.frombuffer(f.read(), dtype=np.uint32)
    depth_data = depth_data.reshape((height, width))

# Convert uint bits back to float depth
def uint_to_float(u):
    return struct.unpack('f', struct.pack('I', u))[0]

depth_float = np.vectorize(uint_to_float)(depth_data)

# Statistics
print(f"Depth range: [{depth_float.min()}, {depth_float.max()}]")
print(f"Far plane pixels (1.0): {np.sum(depth_float == 1.0)} / {width*height}")
print(f"Valid depth pixels: {np.sum(depth_float < 1.0)}")

# Visualize (0=near, 1=far)
import matplotlib.pyplot as plt
plt.imshow(depth_float, cmap='gray', vmin=0, vmax=1)
plt.colorbar(label='Depth (0=near, 1=far)')
plt.title('Shadow Depth Buffer')
plt.savefig('depth_visualization.png')
```

**Expected Results:**
- Most pixels should be 1.0 (far plane, no particles)
- Particle pixels should be 0.0-0.99 (closer to camera)
- Should see particle silhouettes as dark regions

**Debugging Checks:**
- If ALL pixels are 1.0 → depth pre-pass didn't write anything
- If ALL pixels are 0.0 or garbage → clear didn't work
- If particles visible but no depth variation → projection broken

#### 2. Parse g_output.hdr (Rendered Image)

```python
import numpy as np
from PIL import Image

# Read HDR output (R16G16B16A16_FLOAT)
width, height = 1484, 836
with open('g_output.hdr', 'rb') as f:
    hdr_data = np.frombuffer(f.read(), dtype=np.float16)
    hdr_data = hdr_data.reshape((height, width, 4))  # RGBA

# Extract RGB channels
rgb = hdr_data[:, :, :3]

# Tone map for visualization (simple Reinhard)
rgb_tonemapped = rgb / (1.0 + rgb)
rgb_tonemapped = np.clip(rgb_tonemapped, 0, 1)

# Convert to 8-bit for saving
rgb_8bit = (rgb_tonemapped * 255).astype(np.uint8)
img = Image.fromarray(rgb_8bit, mode='RGB')
img.save('output_tonemapped.png')

# Statistics
print(f"HDR range: [{rgb.min()}, {rgb.max()}]")
print(f"Mean brightness: {rgb.mean()}")
print(f"Black pixels: {np.sum(rgb.sum(axis=2) == 0)}")
```

**Compare With/Without Screen-Space Shadows:**
Take two captures:
1. Screen-space shadows ON, PCSS OFF
2. Screen-space shadows OFF, PCSS OFF

Subtract images to see difference:
```python
diff = np.abs(img1 - img2)
print(f"Max difference: {diff.max()}")
print(f"Mean difference: {diff.mean()}")
plt.imshow(diff)
plt.title('Shadow Difference Map')
plt.savefig('shadow_difference.png')
```

#### 3. Parse g_particleLighting.bin

```python
# Read particle lighting buffer
num_particles = 2044
with open('g_particleLighting.bin', 'rb') as f:
    lighting_data = np.frombuffer(f.read(), dtype=np.float32)
    lighting_data = lighting_data.reshape((num_particles, 4))  # XYZW per particle

# Extract RGB (ignore W)
lighting_rgb = lighting_data[:, :3]

# Statistics
print(f"Lighting range: [{lighting_rgb.min()}, {lighting_rgb.max()}]")
print(f"Fully lit particles (>0.9): {np.sum(lighting_rgb.max(axis=1) > 0.9)}")
print(f"Shadowed particles (<0.1): {np.sum(lighting_rgb.max(axis=1) < 0.1)}")
print(f"Zero lighting: {np.sum(lighting_rgb.sum(axis=1) == 0)}")
```

---

## Debugging Checklist

### Issue: Minimal Visual Effect

**Step 1: Verify Depth Buffer Population**
```python
# Parse g_shadowDepth.bin
depth_float = parse_depth_buffer('g_shadowDepth.bin')

# Check 1: Are particles visible?
particle_pixels = np.sum(depth_float < 1.0)
print(f"Particle coverage: {particle_pixels / (1484*836) * 100:.2f}%")
# Expected: 5-20% with 2044 particles

# Check 2: Is depth range reasonable?
particle_depths = depth_float[depth_float < 1.0]
print(f"Particle depth range: [{particle_depths.min()}, {particle_depths.max()}]")
# Expected: 0.1-0.9 (particles between near and far plane)
```

**If particle_pixels == 0:** Depth pre-pass isn't writing
- Check: Is `m_useScreenSpaceShadows` true?
- Check: Is depth pre-pass being called?
- Check: Add debug logging to RenderDepthPrePass

**If depth range is wrong:** Projection matrix issue
- Check: `viewProj` matrix in shader constants
- Check: Particle world positions in reasonable range

**Step 2: Verify Shadow Ray March**
```hlsl
// Add debug output to ScreenSpaceShadow() function
// Replace line 472 with:
float debugOcclusion = occlusionRatio;  // 0=no occlusion, 1=full occlusion
return float3(debugOcclusion, 1.0 - debugOcclusion, 0);  // Red=shadow, Green=lit
```

Rebuild and capture. Output should be:
- Green: Fully lit areas
- Red: Shadowed areas
- Yellow: Partial occlusion

**Step 3: Test Without PCSS Interference**
```cpp
// In Application.h line 276, force disable PCSS
bool m_useShadowRays = false;  // Disable PCSS completely

// Rebuild and test
```

### Issue: 2044 Particle Crash (TDR)

**Root Cause Analysis:**

The depth pre-pass reads ALL particles from the particle buffer:
```cpp
// RenderDepthPrePass dispatches based on particleCount
uint32_t dispatchX = (constants.particleCount + 255) / 256;  // Line 1004
```

But with dual TLAS:
- **Probe Grid BLAS:** First 2044 particles (indices 0-2043)
- **Direct RT BLAS:** Remaining particles (indices 2044-9999)

The depth pre-pass treats all particles identically, which is correct! The crash is likely elsewhere.

**Hypothesis:** The crash isn't in the depth pre-pass, but in the **screen-space shadow sampling** reading out-of-bounds from the TLAS.

**Fix Option 1: Skip Probe Particles in Depth Pre-Pass**
```hlsl
// In depth_prepass.hlsl, line 53:
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint particleIdx = dispatchThreadID.x;

    // Skip probe grid particles (first 2044)
    if (particleIdx < 2044) {
        return;  // Don't render probe particles to depth buffer
    }

    // Rest of shader...
}
```

**Fix Option 2: Use Particle Counts from Dual TLAS**
```cpp
// In Application.cpp, pass only direct RT particle count
RenderConstants gaussianConstants;
gaussianConstants.particleCount = m_rtLighting->GetDirectRTParticleCount();  // 7956 instead of 10000
```

---

## Next Steps (Priority Order)

1. **Debug Minimal Effect (CRITICAL)**
   - Parse g_shadowDepth.bin to verify particle visibility
   - Add debug visualization to ScreenSpaceShadow()
   - Test with PCSS fully disabled
   - Compare captures with/without screen-space shadows

2. **Fix 2044 Particle Crash (HIGH)**
   - Test Fix Option 1: Skip probe particles in depth pre-pass
   - Or Fix Option 2: Pass correct particle count
   - Verify with 5000, 10000 particles

3. **Complete Phase 2 Integration (MEDIUM)**
   - Ensure works with probe grid lighting
   - Ensure works with inline RQ lighting
   - Ensure works with RTXDI lighting

4. **Implement Phase 3-5 (LOW - After Debugging)**
   - Phase 3: Volumetric self-shadowing
   - Phase 4: Temporal accumulation
   - Phase 5: Final polish

---

## Key Files Reference

### C++ Implementation
- `src/particles/ParticleRenderer_Gaussian.cpp` (lines 950-1050): RenderDepthPrePass
- `src/particles/ParticleRenderer_Gaussian.cpp` (lines 1542-1673): Pipeline creation
- `src/particles/ParticleRenderer_Gaussian.cpp` (lines 1388-1455): DLSS recreation
- `src/core/Application.cpp` (lines 984-989): Depth pre-pass call site
- `src/core/Application.h` (lines 276-278): Shadow toggle variables

### Shaders
- `shaders/shadows/depth_buffer_clear.hlsl`: Initialize depth to 1.0
- `shaders/shadows/depth_prepass.hlsl`: Project particles to screen depth
- `shaders/particles/particle_gaussian_raytrace.hlsl` (lines 377-473): ScreenSpaceShadow()
- `shaders/particles/particle_gaussian_raytrace.hlsl` (lines 1120-1128): Shadow integration

### Compiled Shaders (Must Exist)
- `build/bin/Debug/shaders/shadows/depth_buffer_clear.dxil`
- `build/bin/Debug/shaders/shadows/depth_prepass.dxil`

---

## Design Doc Progress

From `ADVANCED_SHADOW_SYSTEM_DESIGN.md`:

- ✅ **Phase 1:** Depth Pre-Pass (COMPLETE)
- 🚧 **Phase 2:** Screen-Space Contact Shadows (PARTIAL - debugging minimal effect)
- ⏳ **Phase 3:** Volumetric Self-Shadowing (NOT STARTED)
- ⏳ **Phase 4:** Enhanced Temporal Accumulation (NOT STARTED)
- ⏳ **Phase 5:** Full Integration (NOT STARTED)

**Estimated Completion:**
- Phase 2 debugging: 1-2 sessions
- Phase 3-5 implementation: 4-6 sessions

---

## Contact & Support

If you encounter issues:
1. Check logs: `build/bin/Debug/logs/`
2. Capture PIX: Use depth pre-pass frame
3. Parse buffers: Use Python scripts above
4. Review this document for debugging steps

**Last Updated:** 2025-11-07 03:30 AM
