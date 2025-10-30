# The Volumetric Lighting Problem: Why Multi-Lights Work and P2P RT Doesn't

**Date:** 2025-10-30
**Author:** Analysis of PlasmaDX-Clean lighting systems
**Context:** Moving from billboard particles to 3D Gaussian volumetric rendering

---

## The Visual Problem

**What you see:**
- **Multi-light system (13 lights):** Beautiful, smooth volumetric glow that radiates through particles like light through fog
- **Particle-to-particle RT lighting:** Twinkling, flashing effect with discrete brightness jumps - no smooth scattering

**What you want:**
- Multi-light quality smoothness
- But with thousands of potential emissive particles (not limited to 16)
- Volumetric light scattering through 3D Gaussian medium

---

## Understanding the Difference: Billboard vs Volumetric Rendering

### Billboard Rendering (Original Design)

```
Camera → Flat sprite → Single brightness value per particle
         ↓
    [Particle]  ← Just a 2D texture facing camera
         ↓
   Brightness × color = Final pixel
```

**Why p2p RT worked for billboards:**
- Each particle is a flat quad facing the camera
- Rendered in a single pass (no ray marching)
- One lookup: `g_rtLighting[particleIdx]` provides brightness
- Discrete jumps less noticeable (2D sprites, quick transitions)

### Volumetric Gaussian Rendering (Current System)

```
Camera → Ray → Multiple sample points THROUGH 3D ellipsoid volume
              ↓      ↓      ↓      ↓      ↓
           [P1]    [P2]   [P3]   [P4]   [P5]  ← Sample points
              ↓      ↓      ↓      ↓      ↓
           Lighting evaluated at EACH point → Accumulated
```

**Why p2p RT fails for volumetric:**
- Ray marches through particle volume (50+ sample points per particle)
- Each sample point may be FAR from particle center
- But we only store ONE lighting value: `g_rtLighting[particleIdx]`
- Result: All sample points get the SAME brightness → discrete jump at particle boundary

---

## The Core Problem: Spatial Sampling Mismatch

### What Happens During Ray Marching

Imagine the camera ray passing through 3 particles at positions A, B, C:

```
Camera →→→ [Particle A] →→→ [Particle B] →→→ [Particle C] →→→

Sample points:
  s1  s2  s3  |  s4  s5  s6  |  s7  s8  s9
              A              B              C
     (inside A)    (inside B)    (inside C)
```

**With per-particle RT lighting (`g_rtLighting[particleIdx]`):**

```
Samples s1, s2, s3 → Look up g_rtLighting[A] = 1.8 (bright)
Samples s4, s5, s6 → Look up g_rtLighting[B] = 0.3 (dim)  ← DISCRETE JUMP!
Samples s7, s8, s9 → Look up g_rtLighting[C] = 1.9 (bright) ← DISCRETE JUMP!
```

Every sample inside particle A gets brightness 1.8, then SUDDENLY drops to 0.3 at B's boundary. This creates the "twinkling/flashing" effect you see - discrete brightness transitions that don't match smooth volumetric reality.

**With multi-lights (proper volumetric):**

Each sample point calculates lighting INDEPENDENTLY:
```
Sample s1 at position (100, 50, 200):
  For each light:
    distance = length(light.pos - s1.pos)
    attenuation = 1.0 / (1 + distance²)
    brightness += light.intensity * attenuation

Sample s2 at position (105, 52, 205):
  [Recalculates with s2's position - DIFFERENT value]

Sample s3 at position (110, 54, 210):
  [Recalculates with s3's position - DIFFERENT value]
```

Result: **Smooth gradient** as sample points move through space, because each point evaluates lighting at ITS OWN POSITION.

---

## Why Multi-Lights Create Beautiful Volumetric Glow

### The Multi-Light Algorithm (Per Sample Point)

For EVERY sample point during ray marching:

```hlsl
for (each sample point P along view ray) {
    float3 totalLight = 0;

    for (each of 13 lights) {
        // 1. Calculate light direction FROM this sample point
        float3 lightDir = normalize(light.position - P);
        float lightDist = distance(light.position, P);

        // 2. Distance attenuation (quadratic falloff)
        float attenuation = 1.0 / (1.0 + (lightDist / light.radius)²);

        // 3. Shadow ray (is path to light blocked?)
        float shadow = CastShadowRay(P, light.position);

        // 4. Phase function (anisotropic scattering)
        float cosTheta = dot(-viewDir, lightDir);
        float phase = HenyeyGreenstein(cosTheta, g=0.7);

        // 5. Combine
        totalLight += light.color * light.intensity * attenuation * shadow * phase;
    }

    // Use totalLight for this sample's contribution
    accumulatedColor += totalLight * density * transmittance;
}
```

**Key insight:** Every sample point gets its own lighting calculation based on ITS POSITION. This creates smooth gradients because:
- Nearby samples have similar (but not identical) distances to lights
- Gradual distance changes → gradual attenuation changes → smooth transitions
- No discrete jumps because there are no discrete boundaries

---

## Why Particle-to-Particle RT Lighting Fails

### The P2P RT Algorithm (Pre-Computed Per Particle)

**Step 1: Pre-computation (once per frame)**
```hlsl
// particle_raytraced_lighting_cs.hlsl
for (each particle P) {
    float3 lighting = 0;

    for (16 random rays from P.center) {
        Cast ray, find nearest lit particle
        lighting += particle.emission / distance²;
    }

    g_rtLighting[P.index] = lighting / 16;  // Store average
}
```

**Step 2: Ray marching (per sample point)**
```hlsl
for (each sample point S along view ray) {
    // Which particle does S belong to?
    uint particleIdx = CurrentParticleIndex(S);

    // Look up PRE-COMPUTED lighting
    float3 light = g_rtLighting[particleIdx];  ← PROBLEM!

    accumulatedColor += light * density * transmittance;
}
```

**The fatal flaw:**
- `g_rtLighting[]` stores ONE value per particle (at particle center)
- But sample points S are distributed THROUGHOUT the particle volume
- All samples inside particle A get `g_rtLighting[A]` (same value)
- All samples inside particle B get `g_rtLighting[B]` (different value)
- **Discrete jump at A/B boundary**

### Why This Was Designed for Billboards

In billboard rendering:
```hlsl
// Billboard particle shader
float4 main() {
    uint particleIdx = GetParticleIndex();
    float3 lighting = g_rtLighting[particleIdx];  // ONE lookup per particle

    return particleColor * lighting;  // Done!
}
```

Each billboard is rendered in one shot - no ray marching, no internal sample points. ONE lookup per particle is perfect because the particle IS a single point (flat sprite).

---

## The Solution: Spatial Scattering with Virtual Lights

### Conceptual Approach

Instead of reading a pre-computed value, **re-evaluate lighting at the sample point** using neighbors as virtual lights:

```
Sample point P is inside particle A, but also near particles B, C, D:

         B (bright: 1.9)
        /
       /
  P ←---- Current sample point
       \
        \
         C (dim: 0.3)
              \
               D (bright: 1.7)

Calculate lighting at P by treating B, C, D as lights:
  - Light B: intensity=1.9, distance=120 units → contributes 0.85
  - Light C: intensity=0.3, distance=150 units → contributes 0.05
  - Light D: intensity=1.7, distance=180 units → contributes 0.45

  P's lighting = (0.85 + 0.05 + 0.45) / 3 = 0.45  ← UNIQUE VALUE for P
```

As the ray moves to the next sample point P+1:
- Different distances to B, C, D
- Different attenuation values
- **Smooth gradient emerges**

### Implementation

```hlsl
float3 InterpolateRTLighting(float3 samplePos, float3 viewDir) {
    float3 totalLight = 0;

    // Find 8 neighbor particles via ray casting
    for (8 directions in Fibonacci sphere) {
        Cast ray from samplePos, find nearest particle N

        if (found N) {
            // Treat N as a virtual light:
            float3 lightColor = g_rtLighting[N.index];  // Pre-computed intensity
            float3 lightPos = N.position;

            // Apply SAME math as multi-lights:
            float3 lightDir = normalize(lightPos - samplePos);
            float dist = length(lightPos - samplePos);

            // Distance attenuation
            float attenuation = 1.0 / (1.0 + dist²);

            // Phase function (view-dependent scattering)
            float cosTheta = dot(-viewDir, lightDir);
            float phase = HenyeyGreenstein(cosTheta, 0.7);

            // Accumulate
            totalLight += lightColor * attenuation * phase;
        }
    }

    return totalLight / 8;
}
```

**Why this works:**
- Each sample point gets its own lighting calculation (like multi-lights)
- Distances to neighbors vary smoothly as ray marches
- Applies proper volumetric scattering (attenuation + phase function)
- Reuses pre-computed `g_rtLighting[]` (no expensive recomputation)

---

## The Mathematics of Smooth vs Discrete Transitions

### Discrete (P2P RT)

```
Lighting = STEP_FUNCTION(particle_index)

         │  1.8 ┌─────────┐
         │      │         │
         │      │         │
Brightness│      │         │  0.3 ┌──────
         │      │         │      │
         │      │         │      │
         │──────┴─────────┴──────┴─────→
              Particle A    B        Distance

= Sharp transitions, visible jumps
```

### Smooth (Multi-Light or Spatial RT)

```
Lighting = CONTINUOUS_FUNCTION(sample_position)

         │     ╱╲
         │    ╱  ╲
Brightness│   ╱    ╲___
         │  ╱         ╲___
         │ ╱              ╲____
         │╱                    ╲____
         │─────────────────────────────→
                Distance from lights

= Gradual transitions, smooth gradients
```

---

## Analogy: Photographing a Foggy Scene

### Billboard Approach (Discrete)

```
Camera → [Measures fog density at specific markers]
              A (thick)    B (thin)     C (thick)

Photo shows: Thick | Thin | Thick  ← Visible bands
```

Each marker has ONE measurement applied to its entire region.

### Volumetric Approach (Smooth)

```
Camera → [Measures fog density CONTINUOUSLY along ray]
         Every centimeter gets its own measurement

Photo shows: Thick → gradually thinning → gradually thickening
```

Continuous sampling creates smooth transitions - like real fog.

---

## Why We Can't Just "Fix" P2P RT

### Attempt 1: Increase Ray Count (2 → 16 rays)

**Result:** Reduces noise, but doesn't fix discrete jumps
**Why:** More rays = better accuracy of `g_rtLighting[particleIdx]`, but still ONE value per particle

### Attempt 2: Reduce Particle Radius

**Result:** Smaller discrete steps, but still visible
**Why:** Smaller particles = more frequent jumps, loses volumetric quality

### Attempt 3: Simple Spatial Interpolation (Weighted Average)

**Result:** Still no scattering - just blended brightness
**Why:** Missing the volumetric math (attenuation, phase function, shadow rays)

### The Right Solution: Volumetric Scattering

**Result:** Multi-light quality smoothness
**Why:** Applies SAME volumetric physics as multi-lights at EVERY sample point

---

## Performance Comparison

### Multi-Light System
- **Cost:** 13 lights × per-sample-point evaluation × shadow rays × phase function
- **Quality:** Perfect volumetric scattering
- **Limitation:** Max 16 lights

### Legacy P2P RT
- **Cost:** Pre-compute once + simple buffer lookup
- **Quality:** Discrete jumps, no volumetric scattering
- **Limitation:** Billboard-era design, doesn't match volumetric needs

### Spatial RT with Scattering (New)
- **Cost:** 8 neighbor searches × per-sample-point × attenuation × phase function
- **Quality:** Multi-light quality smoothness
- **Benefit:** Reuses pre-computed `g_rtLighting[]`, works with thousands of particles

---

## Visual Comparison

### Multi-Light (Target Quality)

```
     ☀️             ☀️             ☀️
      \            |            /
       \           |           /
        \          |          /
         \         |         /
          ╲        │        ╱
           ╲       │       ╱
            ╲      │      ╱
             ╲     │     ╱
              ╲    │    ╱
               ╲   │   ╱
                ╲  │  ╱
                 ╲ │ ╱
                  ╲│╱
                [PARTICLE]
            Smooth radial glow
          (light rays scatter through volume)
```

### P2P RT Legacy (Current Problem)

```
     ☀️             ☀️             ☀️




          ┌──────────┐
          │ BRIGHT   │  ← All internal samples = 1.8
          └──────────┘
          ┌──────────┐
          │   DIM    │  ← All internal samples = 0.3 (JUMP!)
          └──────────┘

     Discrete blocks, visible transitions
```

### Spatial RT Scattering (New Solution)

```
     ☀️             ☀️             ☀️
      \            |            /
       \           |           /
        \          |          /
         \         |         /
          ╲        │        ╱
           ╲       │       ╱
            ╲      │      ╱
             ╲     │     ╱
              ╲    │    ╱
               ╲   │   ╱
                ╲  │  ╱
                 ╲ │ ╱
                  ╲│╱
           [VIRTUAL LIGHTS]
        (neighbors treated as lights)
          Smooth radial glow
    (Same volumetric math as multi-lights!)
```

---

## Conclusion

**The problem:** P2P RT lighting was designed for 2D billboard rendering where ONE brightness value per particle is sufficient. Volumetric 3D Gaussian rendering needs CONTINUOUS lighting evaluation at EVERY sample point during ray marching.

**The solution:** Treat neighbor particles as virtual lights and apply the SAME volumetric scattering math (distance attenuation + phase function) that makes multi-lights look beautiful. This creates smooth gradients while reusing pre-computed `g_rtLighting[]` data.

**Result:** Multi-light quality volumetric glow using thousands of emissive particles instead of being limited to 16 explicit lights.

---

## References

### Code Locations
- Multi-light algorithm: `particle_gaussian_raytrace.hlsl` lines 834-869
- P2P RT compute: `particle_raytraced_lighting_cs.hlsl`
- Spatial RT scattering: `particle_gaussian_raytrace.hlsl` lines 411-493

### Key Concepts
- **Ray marching:** Numerical integration along view ray through volumetric medium
- **Phase function:** Angular scattering distribution (Henyey-Greenstein)
- **Beer-Lambert law:** Absorption of light through participating media
- **Quadratic attenuation:** 1/(1+d²) distance falloff for soft lighting edges

### Papers
- "Real-Time Rendering of Volumetric Clouds" (Schneider 2015) - Volume ray marching
- "Production Volume Rendering" (Wrenninge 2013) - Phase functions
- "Physically Based Sky, Atmosphere and Cloud Rendering" (Hillaire 2020) - Scattering theory
