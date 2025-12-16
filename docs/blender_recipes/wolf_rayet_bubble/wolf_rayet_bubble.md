# GPT-5.2 — Wolf-Rayet Bubble Nebula — Blender OpenVDB Recipe

**Difficulty:** Intermediate
**Method:** Mantaflow (Gas / Smoke) + Three Wind Model (3 emitters)
**Blender Version:** 5.0+
**Export Format:** OpenVDB (`density` is the primary grid)
**Primary Output:** `vdb_cache/` OpenVDB sequence + `.blend` + optional renders
**Script:** `docs/blender_recipes/GPT-5-2_Scripts_Docs_Advice/blender_wolf_rayet_bubble.py`

---

## What this creates

This recipe generates a **Wolf-Rayet bubble nebula**: a multi-shell structure created by the intense stellar winds of a Wolf-Rayet star interacting with material from previous mass-loss epochs.

Key visual characteristics:
- **Bubble/shell morphology** (like NGC 6888 Crescent Nebula, Sharpless 308)
- **Blue-green OIII emission** (500.7nm line dominates)
- **Asymmetric "break-out" structures** where fast wind punches through weak spots
- **Clumpy, filamentary detail** from turbulent wind-ISM interaction

---

## Visual Reference

### Real-World Examples
- **NGC 6888 (Crescent Nebula)** - Classic WR bubble with break-out lobes
- **Sharpless 308** - Nearly spherical WR bubble
- **WR 31a** - Blue bubble nebula
- **Thor's Helmet (NGC 2359)** - Asymmetric WR bubble

### Key Visual Features
- **Color:** Predominantly blue-green (OIII), with some pink (H-alpha)
- **Structure:** Bubble/shell with clumpy edges and asymmetric bulges
- **Central cavity:** Evacuated by fast stellar wind
- **Swept-up shell:** Denser outer ring of compressed material

---

## Astrophysical Properties

| Property | Value | Notes |
|----------|-------|-------|
| WR Star Temperature | 30,000 - 200,000 K | Extremely hot, compact stars |
| Wind Velocity | 1,000 - 3,000 km/s | Supersonic stellar wind |
| Mass Loss Rate | 10⁻⁵ M☉/year | Among highest for any star type |
| Bubble Diameter | 1 - 30 parsecs | In Blender: 100-3000 units |
| Shell Expansion | 20-100 km/s | Much slower than driving wind |

### The Three Wind Model

Wolf-Rayet nebulae form through three distinct mass-loss phases:

1. **Main Sequence Wind** (oldest, outermost)
   - Slow, low-density wind from early stellar evolution
   - Forms the outer boundary of the nebula

2. **Red Supergiant Wind** (intermediate)
   - Dense, slow wind from post-main-sequence phase
   - Gets swept up to form the visible shell

3. **WR Wind** (current, innermost)
   - Fast, hot wind from current WR phase
   - Creates inner cavity, drives shell expansion
   - Responsible for "break-out" structures

### Emission Characteristics
- **OIII (500.7nm):** Dominant emission line, gives blue-green color
- **H-alpha (656.3nm):** Secondary emission, adds pink accents
- **Forbidden lines:** [NII], [SII] in cooler regions
- **Forward scattering:** From dust grains in the shell

---

## Quick Start (headless, one command)

Run from repo root:

```bash
blender -b -P docs/blender_recipes/GPT-5-2_Scripts_Docs_Advice/blender_wolf_rayet_bubble.py -- \
  --output_dir "/abs/path/to/out/WolfRayetBubble" \
  --name "GPT-5-2_WolfRayetBubble" \
  --resolution 128 \
  --domain_size 8.0 \
  --bubble_radius 2.5 \
  --frame_end 120 \
  --bake 1 \
  --render_still 0 \
  --still_frame 80 \
  --render_anim 0 \
  --render_res 1920 1080
```

**Outputs (inside `--output_dir`)**

- **Blend file:** `WolfRayetBubble.blend`
- **VDB cache directory:** `vdb_cache/`
- **Renders:** `renders/WolfRayetBubble_still.png` (and/or animation frames if enabled)

---

## Converting to NanoVDB for PlasmaDX

```bash
python scripts/convert_vdb_to_nvdb.py "/abs/path/to/out/WolfRayetBubble/vdb_cache/<frame>.vdb" \
  "/abs/path/to/out/WolfRayetBubble/WolfRayetBubble_density.nvdb" \
  --grid density
```

---

## The Three Wind Model Implementation

### Emitter 1: Inner Fast Wind (WR Wind)

**Purpose:** Drives the bubble expansion by creating a hot, fast-moving cavity.

**Script Function:** `create_inner_wind_emitter()`

| Setting | Value | Why |
|---------|-------|-----|
| Shape | Small UV Sphere (15% of bubble radius) | Central WR star |
| Density | 0.6 | Hot wind is less dense |
| Temperature | 5.0 | Very hot (30,000+ K equivalent) |
| Velocity Normal | 1.8 | Strong outward push |
| Velocity Random | 0.2 | Wind variability |

### Emitter 2: Outer Slow Shell (RSG Wind)

**Purpose:** Provides the denser material that gets swept up into the visible shell.

**Script Function:** `create_outer_shell_emitter()`

| Setting | Value | Why |
|---------|-------|-----|
| Shape | Large UV Sphere (85% of bubble radius) | Previous mass-loss shell |
| Density | 1.8 | Denser, cooler material |
| Temperature | 1.5 | Cooler RSG-phase wind |
| Velocity Normal | 0.4 | Slower expansion |
| Velocity Random | 0.3 | More clumpy structure |

### Emitter 3: Break-out Structure (Optional)

**Purpose:** Creates asymmetric bulges where fast wind punches through weak spots.

**Script Function:** `create_breakout_emitter()`

| Setting | Value | Why |
|---------|-------|-----|
| Shape | Tilted Cone | Directed outflow |
| Location | Offset from center | Asymmetry |
| Density | 1.2 | Moderate |
| Temperature | 4.0 | Hot outflow |
| Velocity Normal | 2.0 | Very fast breakout |

### Animation

All emitters are animated to create evolving structure:
- **Inner wind:** Pulsates (irregular mass loss)
- **Outer shell:** Slow precession
- **Breakout:** Sweeping motion over time

---

## Domain Settings Reference

| Setting | Value | Why |
|---------|-------|-----|
| Domain Type | GAS | Smoke simulation |
| Resolution | 96-128 | Good detail for bubble structure |
| Adaptive Domain | ON | Memory optimization |
| Gravity | (0, 0, 0) | Space-like environment |
| Alpha (density buoyancy) | 0.0 | No rising smoke |
| Beta (heat buoyancy) | 0.0 | No heat-driven motion |
| Vorticity | 0.85 | Creates turbulent structure |
| Noise | ON | High-frequency detail |
| Noise Scale | 3 | Large-scale turbulence |
| Noise Strength | 1.5 | Strong detail |
| Dissolve Speed | 180 | Slow fade to maintain bubble |

### Cache Settings

| Setting | Value | Notes |
|---------|-------|-------|
| Cache Type | ALL | Single bake operation |
| Data Format | OpenVDB | Required for NanoVDB conversion |
| Compression | ZIP | ⚠️ BLOSC removed in Blender 5.0 |
| Precision | HALF | 16-bit, good balance |

---

## Volume Material (Principled Volume)

| Setting | Value | Why |
|---------|-------|-----|
| Color | (0.3, 0.75, 0.85) | Blue-green OIII emission |
| Density | 1.2 | Visible shell structure |
| Anisotropy | -0.2 | Slight backward scattering |
| Absorption Color | (0.05, 0.1, 0.15) | Blue-tinted absorption |
| Emission Strength | 8.0 | Strong self-illumination |
| Emission Color | (0.4, 0.8, 0.9) | Blue-green glow |

---

## Parameter Reference

### Shape & Scale

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `--domain_size` | 8.0 | 6-12 | Overall simulation bounds |
| `--bubble_radius` | 2.5 | 1.5-4.0 | Size of emitter arrangement |
| `--enable_breakout` | 1 | 0/1 | Include asymmetric breakout |

### Quality & Performance

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `--resolution` | 96 | 64-256 | Voxel detail level |
| `--frame_end` | 120 | 60-250 | Animation length |

### Render Outputs

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--render_still` | 0 | Render single frame |
| `--still_frame` | 80 | Frame to render |
| `--render_anim` | 0 | Render full animation |

---

## PlasmaDX Integration

### Material Type Mapping

Wolf-Rayet bubble nebula should use material type: `GAS_CLOUD` or `PLASMA`

| Property | PlasmaDX Value | Notes |
|----------|----------------|-------|
| Opacity | 0.4 | Semi-transparent shell |
| Scattering | 1.5 | Moderate scattering |
| Emission | 0.8 | Self-luminous gas |
| Phase Function G | -0.2 | Backward scattering |

### Coordinate System
- Blender: Z-up, right-handed
- PlasmaDX: Y-up - rotate -90° around X axis when loading

### Expected Visual Result
In PlasmaDX-Clean, this nebula should appear as:
- Bubble-shaped structure with visible inner cavity
- Blue-green glowing shell
- Asymmetric bulges if breakout enabled
- Turbulent, filamentary detail
- Central star illumination from point light

---

## Variations

### Variation 1: Symmetric Bubble (Sharpless 308 Style)
```bash
--enable_breakout 0
```
Produces a more spherical, symmetric bubble without break-out features.

### Variation 2: High-Detail Hero Shot
```bash
--resolution 192 --frame_end 200
```
Higher resolution for close-up rendering (longer bake time).

### Variation 3: Quick Preview
```bash
--resolution 64 --frame_end 60 --bake 1 --render_still 1
```
Fast iteration for testing pipeline.

---

## Comparison to Other Recipes

| Feature | Supergiant Star | Bipolar Nebula | Wolf-Rayet Bubble |
|---------|----------------|----------------|-------------------|
| Structure | Solid sphere | Jets + ring | Hollow bubble |
| Morphology | Spherical interior | Axisymmetric | Bubble with breakouts |
| Emitters | 2 (core + hotspot) | 3 (ring + 2 jets) | 3 (inner + outer + breakout) |
| Emission | Orange-red | Cyan-magenta | Blue-green |
| Motion | Convective turbulence | Bipolar jets | Expanding shell |

---

## Troubleshooting

### Issue: Bubble looks too solid (no inner cavity)
**Solution:**
- Increase inner wind `velocity_normal` (try 2.0-2.5)
- Decrease outer shell density
- Increase dissolve speed for inner region

### Issue: No visible break-out structure
**Solution:**
- Ensure `--enable_breakout 1`
- Increase breakout emitter temperature and velocity
- Wait for later frames (break-out develops over time)

### Issue: Simulation is too uniform
**Solution:**
- Increase `vorticity` (try 0.9-1.0)
- Increase `noise_strength` (try 2.0)
- Add more emitter animation

### Issue: VDB bakes but PlasmaDX shows nothing
**Solution:**
- Convert with `--grid density`
- Check coordinate system rotation
- See: `docs/NanoVDB/TROUBLESHOOTING_ASSET_VISIBILITY_AND_ANIMATION.md`

### Issue: TDR timeout during bake/render
**Solution:**
- The script uses TDR-safe defaults (`--tdr_safe 1`)
- For rendering, use `--cycles_device CPU` (default)
- See: `docs/blender_recipes/GPT-5-2_Scripts_Docs_Advice/TDR_SAFE_WORKFLOW.md`

---

## Astronomical Sources

- [Wolf-Rayet star - Wikipedia](https://en.wikipedia.org/wiki/Wolf%E2%80%93Rayet_star)
- [Wolf-Rayet nebula - Wikipedia](https://en.wikipedia.org/wiki/Wolf%E2%80%93Rayet_nebula)
- [WISE morphological study of Wolf-Rayet nebulae (A&A 2015)](https://www.aanda.org/articles/aa/full_html/2015/06/aa25706-15/aa25706-15.html)
- [The Asymmetric Nebula Surrounding Wolf-Rayet Star 18 - NASA](https://science.nasa.gov/asymmetric-nebula-surrounding-wolf-rayet-star-18/)

---

## Related Recipes

- [Supergiant Star](supergiant_star.md) - Solid stellar atmosphere
- [Planetary Nebula (Bipolar)](planetary_nebula.md) - Bipolar jets structure
- [Hydrogen Cloud](../emission_nebulae/hydrogen_cloud.md) - Basic emission nebula

---

*Recipe Version: 1.0.0*
*Last Updated: 2025-12-15*
*Tested with: Blender 5.0+*
*Author: Claude Code (Opus 4.5)*
*Based on: GPT-5.2 script patterns + astronomical research*
