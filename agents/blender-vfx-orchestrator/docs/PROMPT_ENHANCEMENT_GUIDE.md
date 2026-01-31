# VFX Prompt Enhancement Guide

## Purpose

This document defines the **Structured VFX Prompt Pattern** — a template for writing detailed, unambiguous asset requests that produce significantly better results from the VFX orchestrator.

**Why this matters:** Vague prompts like "make a fire effect" force the AI to make dozens of assumptions. Detailed prompts front-load creative decisions, resulting in:
- Fewer wasted iterations
- Results that match your vision
- Better starting points for refinement

---

## The Structured VFX Prompt Template

Every VFX request should include these sections:

### 1. Effect Type (Required)
Single word classification for pipeline routing:
- `fire` — Flames, candles, torches, explosions
- `smoke` — Smoke plumes, fog, mist, steam
- `water` — Liquids, splashes, pours, rain
- `explosion` — Fireballs, blasts, debris
- `nebula` — Cosmic gas clouds, space effects
- `solar` — Sun, corona, solar flares, prominences

### 2. Scene Description (Required)
**What is physically happening in the scene?**

Write this as if describing a film shot to a cinematographer. Include:
- Main subject and its state (a burning candle, a pouring liquid, etc.)
- Action/motion (what's moving, how fast, in what direction)
- Supporting elements (what else is in frame)
- Temporal arc (what changes from start to end of the shot)

**Good example:**
> A single beeswax taper candle stands in an ornate brass holder. The flame burns with hypnotic slowness — each flicker stretched into graceful movement. Thin wisps of smoke curl upward from the flame's peak.

**Bad example:**
> A candle is burning.

### 3. Mood and Look (Required)
**What emotional/aesthetic quality should the shot have?**

Include:
- Overall atmosphere (intimate, dramatic, ethereal, violent)
- Lighting quality (warm, cold, harsh, soft, single source, ambient)
- Color palette (specific colors, not just "warm" or "cool")
- Reference touchstones (films, paintings, photographs — even if not provided as files)
- What should be visible vs hidden in shadow

**Good example:**
> Deep, contemplative atmosphere of a Victorian-era private library at night. The candle is the sole light source, creating an intimate sphere of warm illumination that fades into velvety darkness. Color palette: warm ambers, deep browns, shadows with hints of burgundy.

**Bad example:**
> Make it look nice and warm.

### 4. Camera (Required)
**How is the shot framed?**

Include:
- Shot size (extreme close-up, close-up, medium, wide)
- Angle (eye level, low angle, high angle, Dutch)
- Movement (static, slow push, orbit, handheld)
- Depth of field (deep focus, shallow with what in focus)
- Lens character (wide angle distortion, telephoto compression)

**Good example:**
> Tight close-up on the flame, macro-style framing. Very shallow depth of field — flame tack-sharp, background books blur into soft shapes. Camera slightly below flame height, looking subtly upward. Static.

**Bad example:**
> Close up.

### 5. Physical Details (Required)
**Specific measurements and material properties.**

This section should read like a prop breakdown. Include:
- Dimensions (in cm/mm — be specific)
- Materials and their visual properties
- Colors with specificity (not "red" but "deep burgundy with ruby highlights in thin areas")
- Surface qualities (matte, glossy, textured, translucent)
- Any relevant physics (viscosity, temperature, density)

**Good example:**
```
- Candle: Beeswax taper, cream/honey colored, ~2cm diameter
- Flame height: ~3-4cm, teardrop shape with dancing tip
- Flame structure: Blue core (1cm), yellow body, orange-tipped wisps
- Smoke: Thin wispy trails, NOT heavy smoke
- Wax pool: ~1.5cm diameter molten area, slightly concave
```

**Bad example:**
> Use a normal candle with fire on top.

### 6. Motion/Timing Details (Conditional)
**Required for slow-motion, time-lapse, or complex motion.**

For slow-motion especially, describe what the slowed movement should reveal:
- What becomes visible that wouldn't be at normal speed?
- How should different elements move relative to each other?
- What's the emotional quality of the slowed motion?

**Good example:**
> At slow motion, the flame should reveal its internal structure — the stable blue cone, the turbulent yellow convection, the way orange wisps form, stretch, and break away. The flame should "breathe" — expanding and contracting as air currents affect it.

### 7. Hard Constraints (Required)
**Technical requirements that MUST be followed.**

Always include:
- **Simulation system**: Mantaflow liquid/gas, rigid body, cloth, etc.
- **Domain type**: LIQUID, GAS, etc.
- **Key simulation parameters**: Resolution, reaction speed, vorticity
- **Renderer**: Cycles GPU (standard)
- **Sample count**: Usually 256 max
- **Frame range**: Start and end frames
- **cache_type**: Usually `ALL`
- **Reference**: Path to reference image/video, or `None`

**Example:**
```
- Fluid simulation: Blender Mantaflow smoke domain with fire enabled
- Domain type: GAS with use_noise for detail
- Fire reaction speed: Low (0.3-0.5) for slow burn
- Smoke amount: Minimal (0.05-0.1)
- Vorticity: Medium (0.5-0.7)
- Renderer: Cycles GPU
- Samples: 256 max
- Frame range: 1-120
- Resolution: 64-96
- cache_type: ALL
- Reference: None
```

---

## Complete Template

Copy and fill in:

```
Effect Type: [fire|smoke|water|explosion|nebula|solar]

Description:

SCENE DESCRIPTION:
[2-4 sentences describing what is physically happening]

MOOD AND LOOK:
[2-4 sentences describing atmosphere, lighting, color, emotion]

CAMERA:
[1-3 sentences describing framing, angle, movement, DOF]

PHYSICAL DETAILS:
- [Subject]: [dimensions, material, color]
- [Element 2]: [details]
- [Element 3]: [details]
[Continue as needed]

[MOTION/TIMING DETAILS: — if applicable]
[Description of slow-motion behavior or timing]

HARD CONSTRAINTS:
- Fluid simulation: [system and type]
- Domain type: [LIQUID|GAS]
- [Key parameter 1]: [value]
- [Key parameter 2]: [value]
- Renderer: Cycles GPU
- Samples: [count] max
- Frame range: [start]-[end]
- Resolution: [value]
- cache_type: ALL
- Reference: [path or None]

Research documentation, patterns, and APIs to find the optimal starting approach.
```

---

## Effect-Specific Guidance

### Fire Effects
Key parameters to specify:
- Flame height and shape
- Flame color gradient (blue core → yellow → orange tips)
- Smoke amount (none, wispy, heavy)
- Reaction speed (fast flickering vs slow graceful burn)
- Heat distortion visibility

### Water/Liquid Effects
Key parameters to specify:
- Liquid viscosity (water, wine, honey, etc.)
- Surface tension behavior
- Splash characteristics
- Transparency and color at different depths (subsurface scattering)
- Caustics visibility
- `use_plane_init: True` for inflow emitters

### Smoke Effects
Key parameters to specify:
- Smoke density and opacity
- Dissipation rate
- Vorticity (turbulent vs laminar)
- Color (pure white, gray, colored)
- Interaction with light (volumetric scattering)

### Explosion Effects
Key parameters to specify:
- Explosion scale and intensity
- Fire-to-smoke ratio over time
- Debris inclusion
- Shockwave visibility
- Expansion speed

---

## Anti-Patterns (What NOT to Do)

### ❌ Vague descriptions
> "Make an explosion that looks cool"

### ❌ Contradictory requirements
> "Slow motion but only 30 frames"

### ❌ Missing physical scale
> "A fire" (How big? A match? A bonfire? A forest fire?)

### ❌ Unspecified mood
> "Good lighting" (Warm? Cold? Dramatic? Flat?)

### ❌ No hard constraints
The system will guess, and it will guess wrong.

---

## Integration with Orchestrator

The orchestrator automatically parses prompts looking for these sections. Well-structured prompts enable:

1. **TechniqueSelector** to choose optimal simulation approach
2. **ScriptWriter** to generate accurate Blender Python code
3. **QualityAnalyst** to evaluate against your stated intentions
4. **ModificationStrategist** to make targeted improvements

Poorly structured prompts force these agents to make assumptions, leading to iteration loops and budget waste.

---

## Version History

| Date | Change |
|------|--------|
| 2026-01-31 | Initial version |
