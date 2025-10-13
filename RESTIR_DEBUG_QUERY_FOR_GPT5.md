# ReSTIR Implementation Debug Query - Requesting Expert Analysis

## Context

We are implementing **Reservoir-based Spatiotemporal Importance Resampling (ReSTIR)** for a volumetric 3D Gaussian Splatting particle renderer using DirectX 12 DXR 1.1 (RayQuery inline raytracing). The system renders 10,000 particles simulating an accretion disk around a black hole with physically-based temperature emission.

ReSTIR is implemented for **light sampling** - selecting which emissive particles to use as light sources for illuminating other particles. When ReSTIR is OFF (F7 toggle), the scene renders correctly at all camera distances. When ReSTIR is ON, we encounter **distance-dependent artifacts** that we cannot resolve despite multiple attempted fixes.

---

## The Bug - Detailed Symptom Description

### Visual Behavior by Distance

We have a **color-coded corner indicator** that shows ReSTIR sample accumulation state based on M (sample count):

1. **FAR VIEW (Red Indicator: M=0-1)**
   - Camera at 800-1400 units from origin
   - Scene appears **normal and correct**
   - Bright white/yellow core, orange/red accretion disk
   - Colors are vibrant and saturated
   - No artifacts visible
   - ReSTIR vs ReSTIR OFF: **Visually identical** ✅

2. **APPROACHING (Orange Indicator: M=2-8)**
   - Camera at 300-600 units from origin
   - **Thousands of dots appear** across the particle cloud
   - Dots are bright colored specks, evenly distributed
   - Base rendering still visible beneath dots
   - Colors start to look slightly less saturated
   - Still mostly recognizable as accretion disk

3. **CLOSER (Yellow Indicator: M=9-16)**
   - Camera at 150-300 units from origin
   - **Dots become denser and more prominent**
   - Colors shift noticeably toward **brown/muted tones**
   - Orange particles → brownish-orange
   - Yellow core → muddy yellow-brown
   - Saturation drops significantly
   - Overall brightness appears **darker than ReSTIR OFF**

4. **VERY CLOSE (Green Indicator: M=16+)**
   - Camera at 50-150 units from origin
   - **Maximum color shift** - entire scene is brown/desaturated
   - Dots cover most of the view
   - Original accretion disk structure barely recognizable
   - Appears **significantly darker** than ReSTIR OFF (not brighter!)
   - This is the **exact opposite** of the "over-exposure" our analysis predicted

### Critical Observation: NOT Over-Exposure

**IMPORTANT:** Our PIX agent analysis and all debugging assumed we had **over-exposure** (too bright), but the actual visual artifact is **under-exposure** (too dark) with brown/muted color shift. This discrepancy is critical.

### The "Dots" Phenomenon

The dots are the most puzzling aspect:
- Appear suddenly when transitioning from red→orange indicator
- Density increases as camera approaches
- Are NOT single-pixel noise (each dot is ~5-10 pixels)
- Appear to be **individual light samples being visualized**
- Look like "fireflies" or point light sources scattered through the volume
- Do NOT flicker or change frame-to-frame (temporally stable)

**Question:** Are the dots discrete light samples that should be blended/filtered somehow?

---

## Current ReSTIR Implementation

### System Architecture

**Rendering Pipeline:**
1. Physics simulation updates 10,000 particle positions/velocities (120Hz)
2. Acceleration structure (BLAS/TLAS) rebuilt each frame
3. Ray tracing lighting pass (optional, when not using ReSTIR)
4. **Gaussian splatting compute shader** (main renderer):
   - Traces rays from camera through pixels
   - Finds overlapping 3D Gaussians via RayQuery
   - Volume rendering: march through Gaussians (16 steps per Gaussian)
   - **At each volume sample point:** Evaluate ReSTIR lighting
   - Accumulate color with transmittance
   - ACES tone mapping + gamma correction

### ReSTIR Algorithm (per pixel)

**Temporal Resampling (Lines 458-521):**
```hlsl
// 1. Load previous frame's reservoir
Reservoir prevReservoir = g_prevReservoirs[pixelIndex];

// 2. Validate temporal sample (check visibility)
bool temporalValid = ValidateReservoir(prevReservoir, cameraPos);

// 3. Clamp M to prevent unbounded accumulation (RECENTLY ADDED)
const uint maxTemporalM = restirInitialCandidates * 20;  // 320 max
if (prevReservoir.M > maxTemporalM) {
    prevReservoir.weightSum *= float(maxTemporalM) / float(prevReservoir.M);
    prevReservoir.M = maxTemporalM;
}

// 4. Decay temporal samples
float temporalM = prevReservoir.M * restirTemporalWeight;  // 0.9 weight
currentReservoir = prevReservoir;
currentReservoir.M = max(1, uint(temporalM));
currentReservoir.weightSum = prevReservoir.weightSum * restirTemporalWeight;

// 5. Generate NEW candidate samples (16 per frame)
Reservoir newSamples = SampleLightParticles(cameraPos, ray.Direction,
                                             pixelIndex, restirInitialCandidates);

// 6. Merge temporal + new samples
if (newSamples.M > 0) {
    float combinedWeight = currentReservoir.weightSum + newSamples.weightSum;
    currentReservoir.M += newSamples.M;

    // Probabilistic selection
    float random = Hash(pixelIndex + frameIndex * 3000);
    float newProbability = newSamples.weightSum / max(combinedWeight, 0.0001);

    if (random < newProbability) {
        currentReservoir.lightPos = newSamples.lightPos;
        currentReservoir.particleIdx = newSamples.particleIdx;
    }

    currentReservoir.weightSum = combinedWeight;

    // Clamp combined M (RECENTLY ADDED)
    if (currentReservoir.M > maxTemporalM) {
        currentReservoir.weightSum *= float(maxTemporalM) / float(currentReservoir.M);
        currentReservoir.M = maxTemporalM;
    }
}

// 7. Compute final weight W = weightSum / M
currentReservoir.W = currentReservoir.weightSum / float(currentReservoir.M);

// 8. Store for next frame (ping-pong buffers)
g_currentReservoirs[pixelIndex] = currentReservoir;
```

**Light Sampling (Lines 277-394):**
```hlsl
Reservoir SampleLightParticles(float3 rayOrigin, float3 rayDirection,
                                uint pixelIndex, uint numCandidates) {
    Reservoir reservoir;
    reservoir.M = 0;
    reservoir.weightSum = 0;

    // Sample N random emissive particles
    for (uint i = 0; i < numCandidates; i++) {
        // Random particle selection
        float random = Hash(pixelIndex + i * 1000 + frameIndex * 50000);
        uint particleIdx = uint(random * float(particleCount));

        Particle p = g_particles[particleIdx];

        // Compute light weight (importance)
        float dist = length(p.position - rayOrigin);
        float attenuation = 1.0 / max(1.0 + dist * 0.01 + dist * dist * 0.0001, 0.1);

        float intensity = EmissionIntensity(p.temperature);
        float weight = intensity * attenuation;

        // Skip if weight too low
        if (weight < 0.000001) continue;

        // Weighted reservoir update
        UpdateReservoir(reservoir, p.position, particleIdx, weight, random);
    }

    return reservoir;
}
```

**Light Application (Lines 631-657):**
```hlsl
if (useReSTIR != 0 && currentReservoir.M > 0) {
    // Get selected light particle
    Particle lightParticle = g_particles[currentReservoir.particleIdx];
    float3 lightEmission = TemperatureToEmission(lightParticle.temperature);
    float lightIntensity = EmissionIntensity(lightParticle.temperature);
    float dist = length(currentReservoir.lightPos - pos);

    // Same attenuation as sampling
    float attenuation = 1.0 / max(1.0 + dist * 0.01 + dist * dist * 0.0001, 0.1);

    // Evaluate light contribution
    float3 directLight = lightEmission * lightIntensity * attenuation;

    // MIS weight (RECENTLY CHANGED)
    float misWeight = currentReservoir.W * float(restirInitialCandidates)
                      / max(float(currentReservoir.M), 1.0);
    misWeight = clamp(misWeight, 0.0, 2.0);

    rtLight = directLight * misWeight;
} else {
    // Fallback: pre-computed RT lighting (4 rays per particle)
    rtLight = g_rtLighting[hit.particleIdx].rgb;
}

// Apply RT lighting to particle
float3 illumination = float3(1, 1, 1);  // Base self-illumination
illumination += clamp(rtLight * rtLightingStrength, 0.0, 10.0);
```

### Reservoir Structure (32 bytes)
```hlsl
struct Reservoir {
    float3 lightPos;    // 12 bytes - Selected light sample position
    float weightSum;    // 4 bytes  - Accumulated importance weight
    uint M;             // 4 bytes  - Number of samples seen
    float W;            // 4 bytes  - Final weight (weightSum / M)
    uint particleIdx;   // 4 bytes  - Selected particle index
    float pad;          // 4 bytes  - Alignment padding
};
```

**Ping-pong buffers:** 2× 63MB (1920×1080 × 32 bytes)

---

## What We've Tried (Chronological)

### Attempt 1: Fixed M Increment Bug (October 11)
**Problem Identified:** Temporal reuse was copying M without checking `weightSum > 0`, causing invalid state where M > 0 but weightSum = 0.

**Fix Applied:**
```hlsl
// Added weightSum validation before temporal reuse
if (temporalValid && prevReservoir.M > 0 && prevReservoir.weightSum > 0.0001) {
    // ... temporal reuse code
}
```

**Result:** ❌ No change in visual artifacts

---

### Attempt 2: Removed M Scaling (October 12)
**Theory:** W was being multiplied by M, causing exponential brightness with accumulation.

**Fix Applied:**
```hlsl
// BEFORE:
float restirScale = (currentReservoir.W * currentReservoir.M) / float(restirInitialCandidates);
rtLight = lightEmission * lightIntensity * attenuation * restirScale;

// AFTER:
rtLight = lightEmission * lightIntensity * attenuation * currentReservoir.W;
```

**Result:** ❌ No change in visual artifacts

---

### Attempt 3: Added Illumination Clamp (October 12)
**Theory:** Prevent over-brightness from extreme ReSTIR values.

**Fix Applied:**
```hlsl
illumination += clamp(rtLight * rtLightingStrength, 0.0, 10.0);
```

**Result:** ❌ No change in visual artifacts

---

### Attempt 4: PIX Agent Analysis + M Clamping (October 13)
**Analysis:** PIX agent identified unbounded M accumulation as root cause. Recommended clamping M to 20× initial candidates (320 max).

**Fix Applied:**
```hlsl
// Clamp before temporal reuse
const uint maxTemporalM = restirInitialCandidates * 20;
if (prevReservoir.M > maxTemporalM) {
    prevReservoir.weightSum *= float(maxTemporalM) / float(prevReservoir.M);
    prevReservoir.M = maxTemporalM;
}

// Clamp after merging new samples
if (currentReservoir.M > maxCombinedM) {
    currentReservoir.weightSum *= float(maxCombinedM) / float(currentReservoir.M);
    currentReservoir.M = maxCombinedM;
}
```

**Result:** ❌ **NO CHANGE IN VISUAL ARTIFACTS** (still seeing dots + color shift)

---

### Attempt 5: MIS-Compliant W Usage (October 13)
**Theory:** W should be used as MIS weight, not direct brightness multiplier. Normalize by candidate ratio.

**Fix Applied:**
```hlsl
float3 directLight = lightEmission * lightIntensity * attenuation;
float misWeight = currentReservoir.W * float(restirInitialCandidates) / max(float(currentReservoir.M), 1.0);
misWeight = clamp(misWeight, 0.0, 2.0);
rtLight = directLight * misWeight;
```

**Result:** ❌ **NO CHANGE IN VISUAL ARTIFACTS** (still seeing dots + color shift)

---

## PIX Capture Data (from Agent Analysis)

**Reservoir Values at Different Distances:**

1. **Distant (Red, no artifacts):**
   ```
   { lightPos={ 0, 0, 0 }, weightSum=0, M=0, W=0, particleIdx=0 }
   ```

2. **Closer (Red→Orange transition):**
   ```
   { lightPos={ -37.8963, 128.337, 118.75 }, weightSum=2.4655e-05, M=1,
     W=2.4655e-05, particleIdx=3769 }
   ```

3. **Orange→Yellow (dots starting):**
   ```
   { lightPos={ -84.5343, 127.221, 22.9143 }, weightSum=0.00433057, M=9,
     W=0.000481175, particleIdx=2808 }
   ```

4. **Very Close (Green, maximum artifacts):**
   ```
   { lightPos={ -47.9122, 182.491, 89.2122 }, weightSum=0.0292925, M=24,
     W=0.00122052, particleIdx=9518 }
   ```

**Observations:**
- M grows as expected (0 → 1 → 9 → 24)
- weightSum grows proportionally
- W values are extremely small (0.000001 - 0.001 range)
- **All invariants appear correct:** W = weightSum / M ✅
- **M is clamped below 320** (max value observed: M=24)

**PIX File Sizes (correlation with complexity):**
- Far view: 7.1 MB
- Approaching: 86 MB
- Very close: 103 MB

This suggests computational complexity is exploding, but why?

---

## Discrepancy: Analysis vs. Reality

### What PIX Agent Predicted
- **Over-exposure** due to unbounded M and incorrect W usage
- Scene should be **too bright** when close
- Tone mapper saturating to white

### What Actually Happens
- **Under-exposure** - scene is **too dark** when close
- Colors shift to brown/desaturated (not white)
- Dots appear (not uniform brightness)

**This fundamental mismatch suggests our theory is wrong.**

---

## Questions for GPT-5 HIGH

### 1. Root Cause of Darkness
Why does the scene become **darker** (not brighter) when ReSTIR accumulates more samples? This is opposite to expected behavior.

**Hypothesis:** Is `misWeight` calculation systematically underweighting close samples?
```hlsl
misWeight = W * (candidates / M) = W * (16 / 24) = W * 0.666
```
When M=24, we're applying 0.666× scaling. Combined with W=0.001, final weight is ~0.0006. Is this too small?

### 2. The "Dots" Phenomenon
What are these dots, and why do they appear?

**Hypothesis A:** Are they **individual reservoir samples** being rendered without spatial filtering?
- Each pixel uses a single selected light (lightPos from reservoir)
- Should we be blending multiple samples or applying spatial blur?

**Hypothesis B:** Are they **discrete Gaussian centers** being lit instead of volumetric integration?
- Volume rendering marches through Gaussians, but ReSTIR selects one point light
- Point light × Gaussian density → discrete bright spots?

**Hypothesis C:** Are they **numerical precision artifacts**?
- W values are in 10^-6 to 10^-3 range
- Floating-point precision issues when multiplying small values?

### 3. Distance-Dependent Behavior
Why does the artifact severity correlate exactly with camera distance and M accumulation?

The transition points are:
- M=0-1: No artifacts (red)
- M=2-8: Dots appear (orange)
- M=9-16: Color shift begins (yellow)
- M>16: Maximum artifacts (green)

Is there something about the **volumetric integration** that breaks when ReSTIR samples accumulate?

### 4. MIS Weight Formula
Is our MIS weight calculation correct for this use case?

```hlsl
float misWeight = currentReservoir.W * float(restirInitialCandidates) / max(float(currentReservoir.M), 1.0);
```

**In NVIDIA ReSTIR paper (2020), the unbiased estimator is:**
```
L = (1 / M) × sum(f(x_i) × w_i / p(x_i))
  = W × f(x_selected)
```

Should we be using **just W** without any M normalization? Or is the `candidates / M` term needed for our specific sampling strategy?

### 5. Temporal Validation
Our validation checks if the light source is visible from **camera position**, but we're rendering a **volumetric medium**. Should we validate from the **volume sample point** instead?

```hlsl
// Current (potentially wrong):
bool temporalValid = ValidateReservoir(prevReservoir, cameraPos);

// Alternative:
bool temporalValid = ValidateReservoir(prevReservoir, volumeSamplePos);
```

### 6. Spatial vs. Temporal Resampling
We only implement **temporal** reuse (accumulating across frames). The NVIDIA paper also uses **spatial** reuse (sharing samples between neighboring pixels). Could the dots be artifacts from missing spatial filtering?

### 7. Attenuation Mismatch
We compute attenuation from different reference points:

**During sampling (line 357):**
```hlsl
float dist = length(p.position - rayOrigin);  // rayOrigin = camera
```

**During shading (line 636):**
```hlsl
float dist = length(currentReservoir.lightPos - pos);  // pos = volume sample point
```

These distances differ by the ray depth. Is this causing systematic bias?

**Example at close range:**
- Camera at (0, 100, 100) - distance 141 units from origin
- Volume sample at (0, 50, 50) - distance 70 units from origin
- Light particle at origin (0, 0, 0)

Sampling computes: `dist = 141` → `attenuation = 0.0165`
Shading computes: `dist = 70` → `attenuation = 0.0406`

**2.5× attenuation mismatch!** Could this be causing the darkness?

### 8. Weight Threshold
We have a hard threshold in sampling:
```hlsl
if (weight < 0.000001) continue;
```

At close range with high-intensity particles, are we **rejecting valid samples** because the threshold is too high relative to the distance-scaled weights?

### 9. Volume Rendering Integration
Our renderer marches through **multiple Gaussians** (up to 64) with **16 steps each**. Each step evaluates ReSTIR lighting:

```hlsl
for (uint g = 0; g < hitCount; g++) {
    for (uint step = 0; step < 16; step++) {
        float3 pos = ray.Origin + ray.Direction * t;

        // Evaluate ReSTIR lighting at THIS point
        float3 rtLight = EvaluateReSTIR(pos, currentReservoir);

        // ... density, color, accumulation ...
    }
}
```

**Question:** Should ReSTIR lighting be **constant per pixel** (one sample), or **varying per volume step** (recalculate for each pos)?

Currently we use the **same reservoir** (one selected light) for all volume steps. Is this correct?

### 10. Ping-Pong Buffer State
Could there be a **buffer initialization issue**?

First frame (frame 0):
- `g_prevReservoirs` contains uninitialized data
- We read from it anyway
- Could garbage values be propagating?

Do we need an explicit clear operation on first use?

---

## System Configuration

**GPU:** NVIDIA GeForce RTX 4060 Ti (Ada Lovelace, 8GB VRAM)
**Driver:** 580.xx (latest)
**DXR Tier:** 1.1
**Shader Model:** 6.5
**Compiler:** DXC 10.0.26100.0

**Scene Parameters:**
- Particles: 10,000
- Particle radius: 20.0 units
- Disk inner radius: 10 units
- Disk outer radius: 300 units
- Resolution: 1920×1080
- Frame rate: 75-110 FPS (80 FPS average)

**ReSTIR Parameters:**
- Initial candidates: 16
- Temporal weight: 0.9
- Max M: 320 (clamped)
- Weight threshold: 0.000001
- RT lighting strength: 2.0 (adjustable with I/K keys)

---

## Code Locations

**Primary Shader:**
`/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/shaders/particles/particle_gaussian_raytrace.hlsl`

**Critical Sections:**
- Lines 66-74: Reservoir struct definition
- Lines 234-245: UpdateReservoir function
- Lines 248-273: ValidateReservoir function
- Lines 277-394: SampleLightParticles function
- Lines 458-521: Temporal resampling logic
- Lines 631-657: ReSTIR light application
- Lines 560-675: Volume rendering loop (16 steps × 64 Gaussians)

**C++ Configuration:**
`/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/src/particles/ParticleRenderer_Gaussian.cpp`
- Lines 335-373: Reservoir buffer setup and ping-pong swap

---

## What We Need

We need expert guidance on:

1. **Why the darkness?** Theory predicted over-brightness, reality is under-brightness.
2. **What are the dots?** Are they a rendering artifact or ReSTIR sampling artifact?
3. **Is our MIS weight calculation correct** for volumetric rendering with temporal reuse?
4. **Should we validate temporal samples from camera or volume position?**
5. **Is the attenuation mismatch (camera vs. volume sample) the root cause?**
6. **Do we need spatial resampling** in addition to temporal?
7. **Is there a systematic bias** we're missing in the math?

We have PIX captures at 5 different distances showing reservoir state, but we cannot figure out why the behavior diverges from theory.

**All attempted fixes have had zero effect on the visual artifacts.** This suggests we're fixing symptoms, not the root cause.

---

## Reference Materials Available

- NVIDIA ReSTIR paper (Bitterli et al. 2020)
- 2024 ReSTIR MIS weight enhancement paper (Pan et al.)
- PIX captures showing reservoir states at all distances
- Shader source with full implementation
- Previous debug analysis documents

---

## Request

Please analyze this implementation and identify:
1. The root cause of the darkness + dots artifacts
2. What we're misunderstanding about ReSTIR theory
3. Specific code changes needed (with line numbers if possible)
4. Whether our volumetric rendering approach is incompatible with standard ReSTIR

We've spent multiple debugging sessions on this and cannot find the issue. Multiple attempted fixes based on ReSTIR literature have had **zero visual impact**, suggesting a fundamental misunderstanding of the algorithm.

**Thank you for any insights you can provide.**
