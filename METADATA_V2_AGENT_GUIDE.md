# Metadata v2.0 Agent Guide

**For:** AI agents analyzing PlasmaDX screenshots
**Purpose:** Quick reference for interpreting v2.0 metadata fields
**Schema Version:** 2.0

**⚠️ CRITICAL: Before making ANY recommendation, consult `SETTING_EFFECTS.md` for known issues and actual observed behaviors (not theoretical expectations).**

---

## Quick Schema Detection

```json
{
  "schema_version": "2.0"  // ← Check this first!
}
```

- **v1.0:** Basic rendering config only (~20 fields)
- **v2.0:** Comprehensive config capture (~100+ fields)

**If v1.0:** Use basic interpretation (limited context)
**If v2.0:** Use enhanced interpretation (full context available)

---

## Critical Fields for Recommendations

### 1. Active Lighting System (MUST CHECK FIRST!)

```json
{
  "rendering": {
    "active_lighting_system": "MultiLight"  // or "RTXDI"
  }
}
```

**DO:**
- If `"MultiLight"`: Focus on multi-light recommendations (light positions, colors, count)
- If `"RTXDI"`: Focus on RTXDI recommendations (M5 convergence, light grid coverage)

**DON'T:**
- Recommend RTXDI fixes when Multi-Light is active
- Recommend multi-light optimizations when RTXDI is active

**Example mistake (v1.0):**
> "Your RTXDI M5 is not converging..." (when Multi-Light is actually active)

**Correct (v2.0):**
> "You're using Multi-Light system with 13 lights. RTXDI settings are not relevant."

---

### 2. Quality Preset & Target FPS (MUST USE FOR PERFORMANCE EVAL!)

```json
{
  "quality": {
    "preset": "Ultra",       // Maximum, Ultra, High, Medium, Low
    "target_fps": 30.0       // 0, 30, 60, 120, 165
  },
  "performance": {
    "fps": 34.1,
    "target_fps": 30.0,      // Same as quality.target_fps
    "fps_ratio": 1.137       // current / target (1.0 = on target)
  }
}
```

**Quality Presets:**
- **Maximum:** Any FPS (video/screenshots, not realtime) - target: 0 (ignored)
- **Ultra:** 30 FPS target - High quality shadows, high particle count
- **High:** 60 FPS target - Medium quality shadows, medium particle count
- **Medium:** 120 FPS target (MOST COMMON) - Performance shadows, standard particle count
- **Low:** 165 FPS target - Minimal settings for competitive framerates

**FPS Ratio Interpretation:**
- `fps_ratio >= 1.0`: Meeting or exceeding target ✅
- `fps_ratio >= 0.9`: Close to target, acceptable ⚠️
- `fps_ratio < 0.9`: Below target, optimization needed ❌

**Example mistake (v1.0):**
> "FPS is 34 - should be 120. Major bottleneck!" ❌

**Correct (v2.0):**
> "FPS is 34.1 at Ultra quality (target: 30 FPS). FPS ratio: 1.137 (exceeding by 13.7%). Performance is excellent!" ✅

---

### 3. Feature Status (CHECK BEFORE RECOMMENDING!)

```json
{
  "feature_status": {
    "working": {
      "multi_light": true,
      "shadow_rays": true,
      "phase_function": true,
      "physical_emission": true,
      "anisotropic_gaussians": true
    },
    "wip": {
      "doppler_shift": true,           // Has no visible effect currently
      "gravitational_redshift": true,  // Has no visible effect currently
      "rtxdi_m5": true                 // Temporal accumulation in progress
    },
    "deprecated": {
      "in_scattering": true,  // Never worked, marked for removal
      "god_rays": true        // Shelved due to quality/performance issues
    }
  }
}
```

**DO:**
- Recommend `working` features freely
- Mention `wip` features with caveats (e.g., "Doppler shift currently has no visible effect")
- **NEVER** recommend `deprecated` features

**DON'T:**
- Recommend enabling deprecated features (in-scattering, god rays)
- Expect visible effects from WIP features without mentioning they're WIP

**Example mistake (v1.0):**
> "Enable in-scattering for better volumetric quality" ❌

**Correct (v2.0):**
> "In-scattering is marked as deprecated (never worked). I won't recommend enabling it. For better volumetric quality, consider adjusting phase function strength instead." ✅

---

### 4. Physical Effects (FULL VISIBILITY!)

```json
{
  "physical_effects": {
    "physical_emission": {
      "enabled": false,
      "strength": 1.00,
      "blend_factor": 1.00
    },
    "doppler_shift": {
      "enabled": false,
      "strength": 1.00
    },
    "gravitational_redshift": {
      "enabled": false,
      "strength": 1.00
    },
    "phase_function": {
      "enabled": true,
      "strength": 5.00
    },
    "anisotropic_gaussians": {
      "enabled": true,
      "strength": 1.00
    }
  }
}
```

**DO:**
- Check if effects are already enabled before recommending
- Consider current strength values when suggesting adjustments
- Cross-reference with `feature_status` to know what actually works

**DON'T:**
- Recommend enabling already-enabled effects
- Ignore strength values when making suggestions

**Example mistake (v1.0):**
> "Screenshot looks flat. Enable physical emission for temperature-based glow." (when it's already enabled at strength 5.0!)

**Correct (v2.0):**
> "Physical emission is already enabled at maximum strength (5.0). The flatness is likely due to lighting distribution, not emission settings. Consider adjusting light positions or enabling phase function scattering." ✅

---

### 5. Light Configuration (FULL DETAILS!)

```json
{
  "rendering": {
    "lights": {
      "count": 13,
      "light_list": [
        {
          "position": [1200.0, 0.0, 0.0],
          "color": [0.400, 0.600, 1.000],  // Cool blue
          "intensity": 1.50,
          "radius": 150.0
        },
        {
          "position": [-1200.0, 0.0, 0.0],
          "color": [1.000, 0.400, 0.200],  // Warm orange
          "intensity": 1.50,
          "radius": 150.0
        },
        // ... 11 more lights
      ]
    }
  }
}
```

**DO:**
- Reference specific lights by index when making recommendations
- Analyze color distribution (all cool? all warm? balanced?)
- Check position distribution (clustered? evenly distributed?)
- Consider intensity and radius when suggesting adjustments

**DON'T:**
- Make generic "adjust lighting" recommendations
- Ignore existing light configuration

**Example mistake (v1.0):**
> "Adjust lighting for better disk illumination." (too generic)

**Correct (v2.0):**
> "Light #7 at position [800.0, 400.0, 200.0] has pure white color [1.0, 1.0, 1.0]. For better disk heating effect, shift to warm orange [1.0, 0.5, 0.2] to simulate blackbody radiation." ✅

---

### 6. Shadow Configuration

```json
{
  "rendering": {
    "shadows": {
      "preset": "Performance",           // Performance, Balanced, Quality, Custom
      "rays_per_light": 1,
      "temporal_filtering": true,
      "temporal_blend": 0.100
    }
  }
}
```

**Presets:**
- **Performance:** 1 ray + temporal filtering (115-120 FPS target) - MOST COMMON
- **Balanced:** 4 ray PCSS (90-100 FPS target)
- **Quality:** 8 ray PCSS (60-75 FPS target)
- **Custom:** User-defined settings

**DO:**
- Reference preset name when discussing shadows
- Consider temporal filtering status
- Account for shadow quality in performance recommendations

**DON'T:**
- Recommend increasing shadow rays without checking performance headroom
- Ignore temporal filtering when it's enabled

**Example:**
> "You're using Performance shadow preset (1 ray + temporal filtering). This converges to 8-ray quality in ~67ms. For instant soft shadows, consider Balanced preset (4 rays), but expect ~15-20 FPS drop." ✅

---

### 7. Particles & Physics

```json
{
  "particles": {
    "count": 10000,
    "radius": 50.0,
    "gravity_strength": 1.00,
    "physics_enabled": true,
    "inner_radius": 10.0,      // Accretion disk inner edge
    "outer_radius": 300.0,     // Accretion disk outer edge
    "disk_thickness": 50.0     // Disk vertical extent
  }
}
```

**DO:**
- Reference exact particle count in recommendations
- Consider inner/outer radius when discussing disk structure
- Account for disk thickness in vertical camera positioning

**DON'T:**
- Recommend increasing particle count without checking FPS headroom
- Ignore disk geometry when discussing camera angles

---

### 8. Camera State (with pitch!)

```json
{
  "camera": {
    "position": [695.3, 1200.0, 401.7],
    "look_at": [0.0, 0.0, 0.0],
    "distance": 800.0,
    "height": 1200.0,
    "angle": 0.523,
    "pitch": 0.000
  }
}
```

**DO:**
- Reference exact camera position when discussing viewpoint
- Consider height/distance ratio for angle recommendations
- Check pitch for vertical tilt

**DON'T:**
- Make camera recommendations without checking current position
- Ignore height when discussing top-down vs edge-on views

---

## Recommendation Workflow

### Step 1: Check Schema Version
```
IF schema_version == "1.0":
    Use basic interpretation (limited context)
ELSE IF schema_version == "2.0":
    Use enhanced interpretation (full context)
```

### Step 2: Identify Active System
```
IF active_lighting_system == "MultiLight":
    Focus on multi-light optimizations
    Ignore RTXDI settings
ELSE IF active_lighting_system == "RTXDI":
    Focus on RTXDI optimizations
    Check M5 status, grid coverage
```

### Step 3: Evaluate Performance
```
fps_ratio = performance.fps / performance.target_fps

IF fps_ratio >= 1.0:
    "Meeting or exceeding target" ✅
    "You have headroom to enable more features"
ELSE IF fps_ratio >= 0.9:
    "Close to target" ⚠️
    "Minor optimizations may help"
ELSE:
    "Below target" ❌
    "Optimization needed"
```

### Step 4: Check Feature Status
```
FOR each feature in recommendation:
    IF feature in feature_status.deprecated:
        DO NOT recommend
        Mention it's deprecated if user asks
    ELSE IF feature in feature_status.wip:
        Recommend with caveat
        Mention it's WIP and may not show visible effect
    ELSE IF feature in feature_status.working:
        Recommend freely
```

### Step 5: Analyze Physical Effects
```
FOR each physical_effect:
    IF effect.enabled:
        Check if strength is optimal
        Suggest adjustments if needed
    ELSE:
        Consider recommending if appropriate for scene
        Check FPS headroom before recommending
```

### Step 6: Make Specific Recommendations
```
INSTEAD OF: "Adjust lighting"
USE: "Light #7 at position [X, Y, Z] has color [R, G, B]. Consider changing to [R', G', B'] for X effect."

INSTEAD OF: "Improve performance"
USE: "You're at 85% of target FPS. Reduce shadow rays from 4 to 1 (Performance preset) to regain 15% performance."

INSTEAD OF: "Enable RTXDI"
USE: "You're using Multi-Light system which is appropriate for 13 lights. RTXDI would only provide benefits above 20+ lights."
```

---

## Common Mistakes to Avoid

### ❌ Mistake 1: Ignoring active_lighting_system
**Bad:** "Your RTXDI M5 is not converging..."
**Good:** "You're using Multi-Light (not RTXDI), so M5 settings don't apply."

### ❌ Mistake 2: Wrong FPS target assumption
**Bad:** "FPS is 34, should be 120 - major bottleneck!"
**Good:** "FPS is 34 at Ultra quality (target: 30), exceeding by 13%."

### ❌ Mistake 3: Recommending deprecated features
**Bad:** "Enable in-scattering for better quality"
**Good:** "In-scattering is deprecated (never worked). Use phase function instead."

### ❌ Mistake 4: Ignoring current settings
**Bad:** "Enable physical emission for glow"
**Good:** "Physical emission already enabled at strength 5.0. Consider adjusting blend_factor instead."

### ❌ Mistake 5: Generic recommendations
**Bad:** "Adjust lighting for better results"
**Good:** "Light #3 at [pos] has pure white color. Shift to warm orange [1.0, 0.5, 0.2] for disk heating effect."

---

## Example: Full Analysis Workflow

```json
{
  "schema_version": "2.0",
  "rendering": {
    "active_lighting_system": "MultiLight",
    "lights": { "count": 13 }
  },
  "quality": {
    "preset": "Medium",
    "target_fps": 120.0
  },
  "performance": {
    "fps": 105.3,
    "fps_ratio": 0.878
  },
  "physical_effects": {
    "physical_emission": { "enabled": false },
    "phase_function": { "enabled": true, "strength": 5.0 }
  },
  "feature_status": {
    "working": { "physical_emission": true },
    "wip": { "doppler_shift": true },
    "deprecated": { "in_scattering": true }
  }
}
```

**Analysis:**
1. ✅ **Schema:** v2.0 - Full context available
2. ✅ **System:** Multi-Light (not RTXDI)
3. ⚠️ **Performance:** 105.3 FPS at Medium quality (target: 120), ratio: 0.878 - Slightly below target
4. ✅ **Physical emission:** Working feature, currently disabled
5. ✅ **Phase function:** Already enabled at max strength (5.0)
6. ❌ **In-scattering:** Deprecated, don't recommend

**Recommendations:**
> "Your FPS is 105.3 at Medium quality (target: 120 FPS), which is 87.8% of target - slightly below but acceptable. You have a 15 FPS deficit to make up.
>
> **Current Configuration:**
> - System: Multi-Light (13 lights)
> - Physical emission: ❌ Disabled
> - Phase function: ✅ Enabled at maximum (5.0)
>
> **Suggestions:**
> 1. **Consider enabling physical emission** (strength: 1.5-2.0) for temperature-based glow. This will cost ~5-10 FPS but significantly enhance visual quality. You have enough headroom (15 FPS) to enable this without dropping below 90 FPS.
>
> 2. **If FPS becomes critical:** Switch shadow preset from Performance (1 ray) to... wait, checking shadows config... (you're already at Performance preset). No further shadow optimization available without quality loss.
>
> 3. **Alternative:** Reduce particle count from 10,000 to 8,000 to regain 10-15 FPS, then enable physical emission for net quality gain.
>
> **What NOT to do:**
> - ❌ Don't enable in-scattering (deprecated, never worked)
> - ❌ Don't enable Doppler shift (currently has no visible effect, WIP)
> - ❌ Don't worry about RTXDI M5 convergence (you're using Multi-Light, not RTXDI)"

---

## Summary: Key v2.0 Fields

**Must check:**
1. `schema_version` - Determines available fields
2. `rendering.active_lighting_system` - MultiLight or RTXDI
3. `quality.preset` + `quality.target_fps` - Performance evaluation baseline
4. `performance.fps_ratio` - Quick performance health check
5. `feature_status.deprecated` - Never recommend these!

**Should check:**
6. `physical_effects.*` - What's currently enabled
7. `rendering.lights.light_list` - Specific light configurations
8. `rendering.shadows.preset` - Shadow quality level

**Nice to have:**
9. `particles.*` - Disk geometry and particle count
10. `camera.*` - Current viewpoint for framing suggestions
11. `ml_quality.*` - PINN/adaptive quality status

---

## Critical Reference Documents

**BEFORE making recommendations, consult these documents:**

### 1. `SETTING_EFFECTS.md` - Actual Observed Behaviors (MUST READ!)

This document contains **empirical reality** vs theoretical expectations:
- **Particle radius > 30** → Cube artifacts (never recommend!)
- **Phase function** → Requires particle radius 20-30 (prerequisite)
- **RTXDI M5** → Still shows patchwork despite being enabled (WIP)
- **Doppler/Redshift** → No visible effect (WIP, don't recommend)
- **In-scattering** → Deprecated, never worked
- **God rays** → Shelved due to issues
- **Saturated colored lights** → Worsen RTXDI patchwork visibility

**Golden rule:** Theory says "should work" but SETTING_EFFECTS.md says "doesn't work" → Trust SETTING_EFFECTS.md

### 2. `FEATURE_STATUS.md` - Feature Completeness Audit

Comprehensive feature audit with:
- ✅ Complete and working
- 🔄 WIP (visible but not fully functional)
- ⚠️ Deprecated (should not use)
- 🚫 Removed (no longer in code)

### 3. `METADATA_V2_AGENT_GUIDE.md` - This Document

How to interpret v2.0 metadata fields and make recommendations.

---

**For MCP server integration:** See `agents/rtxdi-quality-analyzer/` for Python parsing code.


## Critical Reference Documents

**BEFORE making recommendations, consult these documents:**

### 1. `SETTING_EFFECTS.md` - Actual Observed Behaviors (MUST READ\!)

This document contains **empirical reality** vs theoretical expectations:
- **Particle radius > 30** → Cube artifacts (never recommend\!)
- **Phase function** → Requires particle radius 20-30 (prerequisite)
- **RTXDI M5** → Still shows patchwork despite being enabled (WIP)
- **Doppler/Redshift** → No visible effect (WIP, don't recommend)
- **In-scattering** → Deprecated, never worked
- **God rays** → Shelved due to issues
- **Saturated colored lights** → Worsen RTXDI patchwork visibility

**Golden rule:** Theory says "should work" but SETTING_EFFECTS.md says "doesn't work" → Trust SETTING_EFFECTS.md

### 2. `FEATURE_STATUS.md` - Feature Completeness Audit

Comprehensive feature audit with:
- ✅ Complete and working
- 🔄 WIP (visible but not fully functional)
- ⚠️ Deprecated (should not use)
- 🚫 Removed (no longer in code)

### 3. `METADATA_V2_AGENT_GUIDE.md` - This Document

How to interpret v2.0 metadata fields and make recommendations.
