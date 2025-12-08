# Visual Quality Rubric for PlasmaDX Volumetric Rendering

## Project Context

PlasmaDX renders a black hole accretion disk using:
- **Volumetric 3D Gaussian splatting** (not 2D splats)
- **DXR 1.1 ray tracing** for lighting and shadows
- **Multi-Light system** (13 lights, Fibonacci sphere distribution)
- **RTXDI weighted reservoir sampling** for many-light scenarios (WIP)
- **Froxel volumetric fog** (Phase 8) for atmospheric scattering
- **Probe grid** (Phase 0.13.1) for indirect lighting/SH
- **NanoVDB** (Phase 5.x) for volumetric file loading (experimental)
- **Screen-space shadows** (Phase 2) for contact shadows
- **Adaptive particle radius** (Phase 1.5) for camera-distance sizing
- **Physically-based rendering** with blackbody emission (800K-26000K)

**Target aesthetic:** Photorealistic astrophysical simulation with cinematic volumetric atmosphere

**Metadata Schema:** v4.0 (see screenshot .json sidecar files)

---

## Quality Dimensions

### 1. Volumetric Depth & Atmosphere (CRITICAL)

**What we want:**
- Particles should have **perceivable 3D depth** - closer particles occlude farther ones
- Smooth volumetric density falloff (not hard edges)
- Atmospheric scattering creates **soft, diffuse glow** around bright regions
- "Fog-like" quality where you can see INTO the volume, not just surface splats

**Anti-patterns to avoid:**
- ❌ Flat, billboard-like appearance (2D splats)
- ❌ Hard particle edges (no volumetric blending)
- ❌ No depth perception (everything looks same distance)
- ❌ Harsh cutoffs at particle boundaries

**Visual test:** Can you tell which particles are in front vs behind? Does it feel like a 3D volume or a flat sprite sheet?

---

### 2. Lighting Quality & Rim Lighting (CRITICAL)

**What we want:**
- **Rim lighting halos** on particles when backlit by lights
- Multi-directional shadowing from multiple light sources
- Soft shadow penumbra (PCSS working correctly)
- Light scattering through the volume (Henyey-Greenstein phase function)
- **No black silhouettes** - backlit particles should glow with rim lighting

**Anti-patterns to avoid:**
- ❌ Particles completely black when backlit (missing rim lighting)
- ❌ Hard shadow edges (PCSS not working)
- ❌ Light "pops" or discontinuities when crossing cell boundaries
- ❌ Uniform lighting (no directional information)

**Visual test:** Move a light behind the accretion disk. Do you see glowing halos around particles, or black silhouettes?

---

### 3. Temperature Gradient & Blackbody Emission (HIGH)

**What we want:**
- Smooth color gradient from **hot inner disk (white/blue)** to **cooler outer disk (orange/red)**
- Physically accurate blackbody radiation colors (800K-26000K range)
- Hottest regions near black hole (within ~10× Schwarzschild radius)
- Natural color transitions (not banded or posterized)

**Anti-patterns to avoid:**
- ❌ Uniform color across entire disk (no temperature variation)
- ❌ Banding or posterization in color gradients
- ❌ Unnatural colors (pure green, magenta, etc.)
- ❌ Inverted gradient (cool inside, hot outside)

**Visual test:** Is there a clear hot (white/blue) core transitioning to cooler (orange/red) edges?

---

### 4. RTXDI Light Sampling Quality (HIGH)

**What we want:**
- Smooth, temporally stable lighting (no flickering)
- Even light distribution across visible particles
- **No patchwork pattern** (M4 artifact should be smoothed by M5 temporal accumulation)
- Natural light falloff with distance

**Anti-patterns to avoid:**
- ❌ Rainbow patchwork pattern (M4 debug visualization active)
- ❌ Temporal flickering or "crawling" noise
- ❌ Light "cells" or grid artifacts
- ❌ Sudden brightness changes frame-to-frame

**Visual test:** Does lighting look smooth and stable, or does it flicker/crawl? (Pause physics to isolate RTXDI from particle motion)

---

### 5. Shadow Quality (MEDIUM)

**What we want:**
- Soft shadow penumbra (PCSS working)
- Particle-to-particle occlusion visible
- Shadows match light direction
- Natural shadow density (not too dark, not too light)

**Anti-patterns to avoid:**
- ❌ Hard shadow edges (PCSS disabled or broken)
- ❌ No shadows (RT lighting disabled)
- ❌ Shadows too dark (occlusion factor too high)
- ❌ Shadow direction doesn't match light position

**Visual test:** Can you see soft-edged shadows between particles? Do shadows point away from lights?

---

### 6. Anisotropic Scattering & Phase Function (MEDIUM)

**What we want:**
- Forward scattering dominates (Henyey-Greenstein g=0.6-0.8)
- Backlit particles scatter light toward camera
- Directional scattering (not uniform sphere)

**Anti-patterns to avoid:**
- ❌ Isotropic scattering (uniform in all directions)
- ❌ No scattering (particles don't glow when backlit)
- ❌ Too much backscatter (looks foggy/washed out)

**Visual test:** When a light is behind the disk, do you see directional scattering toward the camera?

---

### 7. Performance & Temporal Stability (MEDIUM)

**What we want:**
- Consistent frame timing (no stutters)
- Smooth camera movement
- Temporal accumulation converges quickly (<100ms)

**Anti-patterns to avoid:**
- ❌ Frame stutters or hitches
- ❌ Slow convergence (>500ms to stabilize)
- ❌ Camera lag or input delay

**Visual test:** Does camera movement feel smooth? Does lighting stabilize quickly after changes?

---

## Comparison Scenarios

### Scenario A: RTXDI M4 vs M5 (Temporal Accumulation)

**Expected improvement:**
- M4: Visible patchwork pattern (rainbow debug or brightness variation)
- M5: Smooth, temporally accumulated lighting (pattern should disappear)

**Key metric:** Temporal stability - M5 should eliminate per-pixel flickering

---

### Scenario B: 1-Ray vs 16-Ray Shadow Sampling

**Expected improvement:**
- 1-ray: Sharp shadow edges, temporal noise
- 16-ray: Soft shadow penumbra, stable

**Key metric:** Shadow softness and stability

---

### Scenario C: Single Light vs Multi-Light (13 lights)

**Expected improvement:**
- Single: Simple directional lighting
- Multi: Complex rim lighting, multi-directional shadows, richer atmosphere

**Key metric:** Volumetric atmosphere depth and lighting complexity

---

### Scenario D: RT Lighting ON vs OFF

**Expected improvement:**
- OFF: Flat, self-emissive only (no particle-to-particle lighting)
- ON: Rich volumetric depth, rim lighting, atmospheric scattering

**Key metric:** Perceivable 3D depth and lighting realism

---

## Reference Images (Golden Standard)

### Best Renders to Date

**Location:** `screenshots/reference/golden_standard/`

**Key examples:**
- `multi_light_rim_lighting.png` - Perfect rim lighting halos (2025-10-17)
- `volumetric_depth_showcase.png` - Clear 3D depth perception (2025-10-15)
- `temperature_gradient_ideal.png` - Smooth hot→cool transition (2025-10-16)

**What makes these "golden":**
- All 7 quality dimensions met
- User feedback: "oh my god the image quality has entered a new dimension!! it looks gorgeous"
- Photorealistic volumetric atmosphere achieved

---

## Analysis Workflow for Agent

### Step 1: Load Reference Context
- Read this rubric
- Load golden standard reference images (if available)
- Understand the 7 quality dimensions

### Step 2: Analyze Screenshot
For each quality dimension:
1. Identify if present/absent
2. Rate quality (Excellent/Good/Fair/Poor/Missing)
3. Note specific issues or achievements
4. Compare to reference images (if provided)

### Step 3: Qualitative Report
Generate report structured as:
```
VOLUMETRIC RENDERING QUALITY ASSESSMENT

Overall Grade: A- (85/100)

=== CRITICAL DIMENSIONS ===

1. Volumetric Depth & Atmosphere: EXCELLENT (95/100)
   ✅ Clear 3D depth perception
   ✅ Smooth volumetric falloff
   ✅ Atmospheric scattering visible
   ⚠️  Minor: Some particles slightly too sharp at edges

2. Lighting Quality & Rim Lighting: GOOD (80/100)
   ✅ Rim lighting halos present
   ✅ Multi-directional shadows
   ❌ Issue: Some backlit particles too dark (missing rim)

=== HIGH PRIORITY ===

3. Temperature Gradient: EXCELLENT (90/100)
   ✅ Smooth hot→cool gradient
   ✅ Physically accurate colors

4. RTXDI Sampling: FAIR (65/100)
   ⚠️  Visible patchwork pattern (M4 active, M5 needed)
   ❌ Temporal flickering observed

=== RECOMMENDATIONS ===

- Enable RTXDI M5 temporal accumulation to eliminate patchwork
- Increase rim lighting intensity for backlit particles
- Overall: Strong volumetric atmosphere, main issue is RTXDI convergence

=== COMPARISON TO REFERENCE ===

vs golden_standard/multi_light_rim_lighting.png:
  Similarity: 87%
  Main difference: Reference has stronger rim lighting

vs golden_standard/volumetric_depth_showcase.png:
  Similarity: 92%
  Main difference: Reference has slightly softer particle edges
```

### Step 4: Actionable Feedback
Suggest specific code/config changes to address issues found.

---

## Agent Training Data

To teach the agent what we want, we'll build a dataset:

### Dataset Structure
```
screenshots/
├── reference/
│   ├── golden_standard/        # Best renders (A+ grade)
│   ├── good/                   # Acceptable (B grade)
│   ├── issues/                 # Known problems (C-D grade)
│   └── failures/               # Broken renders (F grade)
└── annotations/                # JSON files with quality ratings
```

### Annotation Format (JSON)
```json
{
  "screenshot": "multi_light_rim_lighting.png",
  "date": "2025-10-17",
  "grade": "A+",
  "scores": {
    "volumetric_depth": 95,
    "lighting_quality": 92,
    "temperature_gradient": 88,
    "rtxdi_quality": 90,
    "shadow_quality": 85,
    "scattering": 87,
    "temporal_stability": 93
  },
  "notes": "Perfect rim lighting halos. User feedback: 'gorgeous'",
  "settings": {
    "rtxdi_mode": "M4",
    "lights": 13,
    "shadow_rays": 8,
    "particles": 10000
  }
}
```

---

## Usage by Agent

The agent should:

1. **Read this rubric** before analyzing any screenshot
2. **Use GPT-4V/Claude 3.5 vision** to analyze the image
3. **Score each dimension** (0-100)
4. **Compare to reference images** (if available)
5. **Generate actionable recommendations**
6. **Track improvements** over time (regression detection)

**Example agent prompt:**
```
You are analyzing a screenshot from PlasmaDX volumetric black hole renderer.

Reference context: [This rubric + golden standard images]
Screenshot to analyze: screenshots/screenshot_2025-10-24_03-31-46.bmp

Task: Rate the 7 quality dimensions and provide actionable feedback.
Focus on: Volumetric depth, rim lighting, RTXDI stability.
```

---

**Last Updated:** 2025-10-24
**Maintained by:** Ben (user) + Claude Code sessions
