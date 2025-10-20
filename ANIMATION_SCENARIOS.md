# Animation Scenarios - Creative Use Case Library

**Document Version:** 1.0
**Created:** 2025-10-20
**Status:** Phase 5 Animation Blueprint
**Related:** PHASE_5_CELESTIAL_RENDERING_PLAN.md, PARTICLE_TYPE_SYSTEM_SPEC.md

---

## Executive Summary

This document provides **10 comprehensive animation scenarios** showcasing the creative capabilities unlocked by Phase 5 enhancements. Each scenario includes:
- Particle type distribution and configuration
- Physics parameter settings (constraint shapes, black hole mass, alpha viscosity)
- Camera movement timeline
- Expected visual outcomes
- Technical requirements and performance estimates

**Purpose:** Enable user (Ben) to create stunning astronomical animations using PlasmaDX-Clean's volumetric particle rendering engine.

---

## Scenario Template Format

Each scenario follows this structure:

**Overview:**
- **Title:** Descriptive name
- **Duration:** Animation length (seconds)
- **Complexity:** Low/Medium/High (particle count, physics complexity)
- **Visual Style:** Artistic/Physically Accurate/Hybrid
- **Primary Effect:** Key visual phenomenon showcased

**Particle Configuration:**
- Type distribution (% of each type)
- Per-type material properties
- Total particle count

**Physics Setup:**
- Constraint shape (SPHERE, DISC, TORUS, ACCRETION_DISK)
- Black hole mass (solar masses)
- Alpha viscosity (Shakura-Sunyaev parameter)
- Gravity strength, turbulence, damping

**Lighting Setup:**
- Light preset (Sphere, Ring, Sparse, Custom)
- Light count and distribution
- Bulk color scheme

**Camera Timeline:**
- Key camera positions and orientations
- Movement speed and easing
- Focal points and framing

**Expected Visuals:**
- Detailed description of visual effects
- Key moments to capture
- Performance estimate (FPS @ 1080p on RTX 4060 Ti)

---

## Scenario 1: Stellar Nursery - Birth of Stars

### Overview

- **Title:** "Stellar Nursery - The Orion Nebula"
- **Duration:** 60 seconds
- **Complexity:** High (15K particles, multiple types, complex physics)
- **Visual Style:** Hybrid (physically accurate temperatures + artistic nebula colors)
- **Primary Effect:** Young stars igniting within glowing gas clouds

### Particle Configuration

**Type Distribution:**
- 30% STAR_MAIN_SEQUENCE (4,500 particles) - Young hot stars
- 50% GAS_CLOUD (7,500 particles) - Emission nebula
- 20% DUST_PARTICLE (3,000 particles) - Protoplanetary dust

**Total Particle Count:** 15,000

**Per-Type Material Properties:**

**STAR_MAIN_SEQUENCE:**
```json
{
    "baseRadius": 2.5,
    "densityScale": 15.0,
    "opacityScale": 1.0,
    "emissionStrength": 8.0,
    "phaseG": 0.7,
    "scatteringAlbedo": 0.05,
    "absorptionCoeff": 8.0,
    "tintColor": [1.0, 1.0, 1.0],
    "blackbodyWeight": 1.0,
    "anisotropyStrength": 0.05,
    "temperatureFloor": 15000.0,
    "temperatureCeiling": 40000.0
}
```

**GAS_CLOUD:**
```json
{
    "baseRadius": 2.0,
    "densityScale": 0.005,
    "opacityScale": 0.25,
    "emissionStrength": 0.2,
    "phaseG": -0.4,
    "scatteringAlbedo": 0.9,
    "absorptionCoeff": 0.02,
    "tintColor": [0.2, 0.4, 1.0],
    "blackbodyWeight": 0.05,
    "anisotropyStrength": 3.0,
    "dissipationRate": 0.0,
    "temperatureFloor": 5000.0,
    "temperatureCeiling": 12000.0
}
```

**DUST_PARTICLE:**
```json
{
    "baseRadius": 0.4,
    "densityScale": 150.0,
    "opacityScale": 1.0,
    "emissionStrength": 0.005,
    "phaseG": 0.0,
    "scatteringAlbedo": 0.25,
    "absorptionCoeff": 15.0,
    "tintColor": [0.5, 0.4, 0.3],
    "blackbodyWeight": 0.2,
    "anisotropyStrength": 0.8,
    "temperatureFloor": 50.0,
    "temperatureCeiling": 1500.0
}
```

### Physics Setup

**Constraint Shape:** TORUS
- Inner radius: 300 units
- Outer radius: 1200 units
- Thickness: 400 units (vertical spread)

**Black Hole Mass:** 5×10⁶ solar masses (moderate stellar nursery)

**Alpha Viscosity:** 0.05 (low viscosity, slow inward drift)

**Other Parameters:**
- Gravity strength: 1.2 (strong pull to center)
- Turbulence: 0.8 (high turbulence for wispy gas)
- Damping: 0.3 (moderate damping)

**Velocity-Based Heating:**
- Dust particles heat up near stars (temperature increases from 50K → 1500K)
- Gas clouds ionized by stellar radiation (5000K → 12000K)

### Lighting Setup

**Preset:** Custom "Stellar Nursery"

**Light Configuration:**
- 20 lights total (one per young star)
- Distribution: Embedded within STAR_MAIN_SEQUENCE particles
- Color scheme: Blue/white (15000K-40000K blackbody)

**Bulk Color Gradient:**
- Core stars: Pure white (no tint)
- Background fill: Soft blue/cyan (nebula illumination)

**Light Intensity:**
- Star lights: 5.0-8.0 (very bright)
- Radius: 200-400 units (large illumination zones)

### Camera Timeline

**0:00-0:10** - Wide Establishing Shot
- Position: (2000, 1000, -2000)
- Look At: (0, 0, 0)
- Movement: Slow dolly in (2000 → 1500 units distance)
- Framing: Full nebula visible, stars as bright points

**0:10-0:25** - Dive into Nebula
- Position: (1500, 800, -1500) → (400, 200, -400)
- Look At: (0, 0, 0)
- Movement: Accelerating dolly in (ease-in-out)
- Framing: Gas clouds begin to fill frame, stars grow larger

**0:25-0:40** - Close-Up of Young Star
- Position: Circle around star at 300-unit radius
- Look At: Brightest STAR_MAIN_SEQUENCE particle
- Movement: Orbital (30° per second, horizontal plane)
- Framing: Single star with gas envelope visible

**0:40-0:50** - Pull Back Through Dust Lane
- Position: (300, 100, -300) → (1000, 500, -1000)
- Look At: (0, 0, 0)
- Movement: Slow dolly out through dark dust lane
- Framing: Silhouette effect (backlit dust)

**0:50-1:00** - Final Wide Shot
- Position: (1800, 1200, -1800)
- Look At: (0, 0, 0)
- Movement: Gentle rotation (10° clockwise)
- Framing: Full nebula with depth layers visible

### Expected Visuals

**0:00-0:10:**
- Wide view of blue/cyan glowing nebula
- ~30 bright white/blue stars scattered throughout
- Wispy gas structures (backward scattering from stars)

**0:10-0:25:**
- Volumetric depth becomes apparent (foreground gas vs background)
- Stars grow from points to volumetric spheres with limb darkening
- Dust lanes create dark streaks across bright gas

**0:25-0:40:**
- Single star fills frame, showing:
  - Bright spherical core (limb darkening visible)
  - Glowing gas envelope (blue/cyan emission)
  - Rim lighting halo (Henyey-Greenstein phase function)
  - Dust particles orbiting star (heating effect visible)

**0:40-0:50:**
- Silhouette effect: dense dust lane in foreground
- Backlit stars create glowing edges on dust
- Atmospheric scattering through dust particles

**0:50-1:00:**
- Layered depth: foreground gas → middle stars → background nebula
- Soft color gradients (blue outer regions → white stellar cores)
- Turbulent gas motion visible (anisotropic elongation)

**Performance Estimate:**
- 15K particles, 20 lights, complex physics
- Estimated FPS: 80-100 @ 1080p (RTX 4060 Ti)
- Bottleneck: High particle count with multiple types

---

## Scenario 2: Red Giant Engulfment - Stellar Death

### Overview

- **Title:** "The Death of Betelgeuse"
- **Duration:** 45 seconds
- **Complexity:** Medium (10K particles, simple physics)
- **Visual Style:** Physically Accurate (pure blackbody radiation)
- **Primary Effect:** Red giant star with expanding, dissipating envelope

### Particle Configuration

**Type Distribution:**
- 0.1% STAR_GIANT (10 particles) - Central red giant core
- 99.9% GAS_CLOUD (9,990 particles) - Expanding envelope

**Total Particle Count:** 10,000

**Per-Type Material Properties:**

**STAR_GIANT:**
```json
{
    "baseRadius": 5.0,
    "densityScale": 0.05,
    "opacityScale": 0.7,
    "emissionStrength": 12.0,
    "phaseG": 0.6,
    "scatteringAlbedo": 0.25,
    "absorptionCoeff": 0.8,
    "tintColor": [1.0, 0.3, 0.15],
    "blackbodyWeight": 0.9,
    "anisotropyStrength": 0.2,
    "temperatureFloor": 2800.0,
    "temperatureCeiling": 3500.0
}
```

**GAS_CLOUD:**
```json
{
    "baseRadius": 1.8,
    "densityScale": 0.001,
    "opacityScale": 0.15,
    "emissionStrength": 0.05,
    "phaseG": -0.2,
    "scatteringAlbedo": 0.85,
    "absorptionCoeff": 0.01,
    "tintColor": [1.0, 0.4, 0.2],
    "blackbodyWeight": 0.3,
    "anisotropyStrength": 1.5,
    "dissipationRate": 0.03,
    "temperatureFloor": 500.0,
    "temperatureCeiling": 3000.0
}
```

### Physics Setup

**Constraint Shape:** SPHERE
- Radius: 1500 units (expanding envelope)

**Black Hole Mass:** 0 (no central gravity, just expansion)

**Alpha Viscosity:** 0.0 (no accretion)

**Other Parameters:**
- Gravity strength: 0.1 (minimal gravity)
- Turbulence: 1.2 (very turbulent stellar winds)
- Damping: 0.1 (low damping, continuous expansion)

**Radial Expansion:**
- Gas particles drift outward at 10-50 units/second
- Dissipation rate: 0.03 (30-second half-life)
- Respawn at core when fully dissipated

### Lighting Setup

**Preset:** Single Beacon (modified)

**Light Configuration:**
- 1 light at center (red giant core)
- Color: Deep red/orange (2800K blackbody)
- Intensity: 10.0 (very bright)
- Radius: 800 units (huge illumination zone)

### Camera Timeline

**0:00-0:15** - Distant Approach
- Position: (3000, 1500, -3000) → (2000, 1000, -2000)
- Look At: (0, 0, 0)
- Movement: Slow dolly in
- Framing: Full red giant visible, diffuse envelope

**0:15-0:30** - Orbital Around Envelope
- Position: Circle at 1200-unit radius
- Look At: (0, 0, 0)
- Movement: Orbital (20° per second, horizontal plane)
- Framing: Layered envelope visible, core glowing through gas

**0:30-0:45** - Zoom to Core
- Position: (1200, 600, -1200) → (600, 300, -600)
- Look At: (0, 0, 0)
- Movement: Accelerating dolly in (ease-in)
- Framing: Close-up of central core with wispy envelope

### Expected Visuals

**0:00-0:15:**
- Massive red/orange sphere (1500-unit radius)
- Diffuse, partially transparent envelope
- Bright red core visible through gas layers
- Slow pulsing effect (turbulence creating waves)

**0:15-0:30:**
- Layered structure visible:
  - Dense red core (STAR_GIANT particles)
  - Mid-density orange envelope (warm gas)
  - Tenuous outer regions (dissipating gas)
- Gas particles fading out at edges (dissipation effect)
- Stellar wind streamers (anisotropic elongation)

**0:30-0:45:**
- Close-up of core:
  - Bright red/orange center (limb darkening)
  - Turbulent gas motion (convection cells)
  - Wispy envelope drifting outward
  - Backlit gas glowing red

**Performance Estimate:**
- 10K particles, 1 light, simple physics
- Estimated FPS: 120+ @ 1080p (RTX 4060 Ti)
- Very efficient (single light, low particle types)

---

## Scenario 3: Black Hole Accretion Disk - The Feeding Cycle

### Overview

- **Title:** "Feeding the Beast"
- **Duration:** 90 seconds (full orbital cycle)
- **Complexity:** High (20K particles, complex multi-zone physics)
- **Visual Style:** Hybrid (artistic warm colors + physically accurate dynamics)
- **Primary Effect:** Three-zone accretion disk with varying material properties

### Particle Configuration

**Type Distribution:**
- 50% PLASMA_BLOB (10,000 particles) - Hot inner disk
- 30% GAS_CLOUD (6,000 particles) - Warm mid-disk
- 20% DUST_PARTICLE (4,000 particles) - Cool outer disk

**Total Particle Count:** 20,000

**Spatial Distribution:**
- Inner disk (200-600 units): 100% PLASMA_BLOB
- Mid-disk (600-1000 units): 70% GAS_CLOUD, 30% PLASMA_BLOB
- Outer disk (1000-1500 units): 100% DUST_PARTICLE

**Per-Type Material Properties:**

**PLASMA_BLOB (Inner Disk):**
```json
{
    "baseRadius": 1.0,
    "densityScale": 2.0,
    "opacityScale": 1.0,
    "emissionStrength": 2.5,
    "phaseG": 0.3,
    "scatteringAlbedo": 0.15,
    "absorptionCoeff": 1.5,
    "tintColor": [1.0, 0.9, 0.7],
    "blackbodyWeight": 0.8,
    "anisotropyStrength": 1.5,
    "temperatureFloor": 15000.0,
    "temperatureCeiling": 26000.0
}
```

**GAS_CLOUD (Mid-Disk):**
```json
{
    "baseRadius": 1.3,
    "densityScale": 0.05,
    "opacityScale": 0.4,
    "emissionStrength": 0.8,
    "phaseG": 0.1,
    "scatteringAlbedo": 0.5,
    "absorptionCoeff": 0.5,
    "tintColor": [1.0, 0.6, 0.3],
    "blackbodyWeight": 0.6,
    "anisotropyStrength": 2.0,
    "temperatureFloor": 5000.0,
    "temperatureCeiling": 15000.0
}
```

**DUST_PARTICLE (Outer Disk):**
```json
{
    "baseRadius": 0.6,
    "densityScale": 80.0,
    "opacityScale": 1.0,
    "emissionStrength": 0.02,
    "phaseG": 0.0,
    "scatteringAlbedo": 0.3,
    "absorptionCoeff": 12.0,
    "tintColor": [0.6, 0.4, 0.3],
    "blackbodyWeight": 0.4,
    "anisotropyStrength": 0.7,
    "temperatureFloor": 300.0,
    "temperatureCeiling": 2000.0
}
```

### Physics Setup

**Constraint Shape:** ACCRETION_DISK
- Inner radius: 200 units (ISCO - Innermost Stable Circular Orbit)
- Outer radius: 1500 units
- Disk thickness: 100 units (thin disk approximation)

**Black Hole Mass:** 1×10⁸ solar masses (supermassive black hole)

**Alpha Viscosity:** 0.1 (Shakura-Sunyaev, moderate inward drift)

**Other Parameters:**
- Gravity strength: 2.0 (very strong central pull)
- Turbulence: 0.5 (moderate disk turbulence)
- Damping: 0.6 (strong damping, thin disk)

**Temperature Gradient:**
- Inner disk (200-600 units): 15000K-26000K (blue/white)
- Mid-disk (600-1000 units): 5000K-15000K (yellow/orange)
- Outer disk (1000-1500 units): 300K-2000K (red/brown)

**Keplerian Velocity:**
- Inner disk: ~200 units/second (fast orbit)
- Outer disk: ~50 units/second (slow orbit)

### Lighting Setup

**Preset:** RTXDI Ring (16 lights)

**Light Configuration:**
- 16 lights in dual-ring formation
- Inner ring (8 lights): 600-unit radius, white/blue (hot disk emission)
- Outer ring (8 lights): 1000-unit radius, orange/red (warm disk emission)

**Bulk Color Gradient:**
- Radial gradient: Blue (inner) → Yellow → Orange → Red (outer)
- Vertical gradient: Bright (disk plane) → Dim (above/below disk)

**Light Intensity:**
- Inner ring: 3.0-5.0 (bright, hot)
- Outer ring: 1.0-2.0 (dim, cool)

### Camera Timeline

**0:00-0:20** - Top-Down View (Disk Plane)
- Position: (0, 2000, 0)
- Look At: (0, 0, 0)
- Movement: Slow descent (2000 → 1500 units altitude)
- Framing: Full disk visible, color gradient clear

**0:20-0:40** - Orbital Sweep (Equatorial)
- Position: Circle at 1800-unit radius, disk plane
- Look At: (0, 0, 0)
- Movement: Orbital (15° per second, horizontal)
- Framing: Edge-on view, disk thickness visible

**0:40-0:60** - Dive Through Disk
- Position: (1800, 1200, -1800) → (600, 50, -600)
- Look At: (0, 0, 0)
- Movement: Diving trajectory (ease-in-out)
- Framing: Passing through outer disk into inner disk

**0:60-0:75** - Low Altitude Pass (Inner Disk)
- Position: Circle at 400-unit radius, 100 units above disk
- Look At: (0, 0, 0)
- Movement: Fast orbital (30° per second)
- Framing: Close-up of hot inner disk, particles whizzing by

**0:75-0:90** - Pull Back to Wide
- Position: (400, 100, -400) → (2500, 1500, -2500)
- Look At: (0, 0, 0)
- Movement: Accelerating dolly out (ease-out)
- Framing: Final wide shot, full disk visible

### Expected Visuals

**0:00-0:20:**
- Three distinct color zones:
  - Inner disk: Bright blue/white (hot plasma)
  - Mid-disk: Yellow/orange (warm gas)
  - Outer disk: Red/brown (cool dust)
- Spiral arm structures (alpha viscosity creating density waves)
- Differential rotation visible (inner fast, outer slow)

**0:20-0:40:**
- Edge-on view reveals disk thickness (100 units)
- Layered structure:
  - Bright equatorial plane
  - Dimmer vertical extent
  - Dark dust lanes in outer disk
- Rim lighting from embedded lights

**0:40-0:60:**
- Dive trajectory:
  - Start: Dark brown dust particles (large, opaque)
  - Middle: Orange/yellow gas (translucent, wispy)
  - End: Blue/white plasma (bright, dense)
- Temperature increase visible (color shift)
- Density increase (more particles per volume)

**0:60-0:75:**
- Low altitude pass:
  - Fast-moving plasma particles (200 units/sec)
  - Anisotropic elongation visible (tidal stretching)
  - Bright blue/white emission
  - Rim lighting halos from nearby particles
  - Volumetric depth (hundreds of particles in view)

**0:75-0:90:**
- Final wide shot:
  - Full disk structure visible
  - Color gradient: blue core → red outer edge
  - Spiral density waves
  - Vertical scale height (thin disk)
  - Embedded lights illuminate disk from within

**Performance Estimate:**
- 20K particles, 16 lights, complex physics
- Estimated FPS: 60-80 @ 1080p (RTX 4060 Ti)
- Bottleneck: High particle count, alpha viscosity calculation

---

## Scenario 4: Binary Star Dance - Gravitational Tango

### Overview

- **Title:** "Binary Waltz"
- **Duration:** 30 seconds (one orbital period)
- **Complexity:** Medium (8K particles, dual gravity wells)
- **Visual Style:** Physically Accurate (pure blackbody, Keplerian orbits)
- **Primary Effect:** Two stars orbiting common center of mass with gas exchange

### Particle Configuration

**Type Distribution:**
- 25% STAR_MAIN_SEQUENCE (2,000 particles) - Primary star (blue)
- 25% STAR_GIANT (2,000 particles) - Secondary star (red)
- 50% GAS_CLOUD (4,000 particles) - Accretion streams

**Spatial Distribution:**
- Primary star (blue): Centered at (+400, 0, 0)
- Secondary star (red): Centered at (-400, 0, 0)
- Gas streams: Between stars (Roche lobe overflow)

**Per-Type Material Properties:**

**STAR_MAIN_SEQUENCE (Blue Primary):**
```json
{
    "baseRadius": 1.5,
    "densityScale": 20.0,
    "opacityScale": 1.0,
    "emissionStrength": 6.0,
    "phaseG": 0.7,
    "scatteringAlbedo": 0.05,
    "tintColor": [1.0, 1.0, 1.0],
    "blackbodyWeight": 1.0,
    "anisotropyStrength": 0.1,
    "temperatureFloor": 20000.0,
    "temperatureCeiling": 35000.0
}
```

**STAR_GIANT (Red Secondary):**
```json
{
    "baseRadius": 2.5,
    "densityScale": 0.08,
    "opacityScale": 0.8,
    "emissionStrength": 7.0,
    "phaseG": 0.5,
    "scatteringAlbedo": 0.2,
    "tintColor": [1.0, 0.35, 0.15],
    "blackbodyWeight": 0.95,
    "anisotropyStrength": 0.3,
    "temperatureFloor": 2800.0,
    "temperatureCeiling": 3800.0
}
```

**GAS_CLOUD (Accretion Streams):**
```json
{
    "baseRadius": 1.0,
    "densityScale": 0.02,
    "opacityScale": 0.3,
    "emissionStrength": 0.5,
    "phaseG": 0.0,
    "scatteringAlbedo": 0.7,
    "tintColor": [0.8, 0.6, 1.0],
    "blackbodyWeight": 0.4,
    "anisotropyStrength": 3.0,
    "temperatureFloor": 8000.0,
    "temperatureCeiling": 18000.0
}
```

### Physics Setup

**Constraint Shape:** DUAL_SPHERE (custom - two gravity wells)
- Primary sphere: 300-unit radius at (+400, 0, 0)
- Secondary sphere: 400-unit radius at (-400, 0, 0)
- Gas particles flow between Roche lobes

**Black Hole Mass:** N/A (dual stellar masses)
- Primary mass: 5×10⁶ solar masses
- Secondary mass: 2×10⁶ solar masses

**Binary Orbit:**
- Orbital period: 30 seconds
- Separation: 800 units (center to center)
- Circular orbit around barycenter

**Alpha Viscosity:** 0.2 (gas streams from secondary to primary)

**Other Parameters:**
- Gravity strength: 1.5 (moderate)
- Turbulence: 0.6 (moderate gas turbulence)
- Damping: 0.4 (moderate)

### Lighting Setup

**Preset:** Custom "Binary System"

**Light Configuration:**
- 2 primary lights (one per star)
  - Blue primary: (400, 0, 0), color: white/blue (30000K), intensity: 6.0
  - Red secondary: (-400, 0, 0), color: red/orange (3000K), intensity: 7.0
- 4 accent lights (gas stream illumination)
  - Positioned along accretion stream path
  - Color: Purple/blue (mixed star light)
  - Intensity: 2.0

### Camera Timeline

**0:00-0:10** - Wide Orbital View
- Position: (1500, 800, -1500)
- Look At: (0, 0, 0)
- Movement: Slow orbital (10° per second)
- Framing: Both stars visible, gas stream connecting them

**0:10-0:20** - Close-Up of Accretion Stream
- Position: (800, 400, -800) → (200, 100, -200)
- Look At: Gas stream midpoint (0, 0, 0)
- Movement: Dolly in following gas flow
- Framing: Tight on gas stream, stars in background

**0:20-0:30** - Figure-8 Sweep
- Position: Figure-8 path around both stars
- Look At: Alternate between stars
- Movement: Smooth figure-8 (15 seconds per loop)
- Framing: Dynamic view of both stars and connection

### Expected Visuals

**0:00-0:10:**
- Blue star (left) and red star (right) orbiting barycenter
- Purple/blue gas stream flowing from red → blue (Roche lobe overflow)
- Orbital motion visible (stars move ~12° per second)
- Color contrast: cool red vs hot blue

**0:10-0:20:**
- Close-up of gas stream:
  - Thin tendril of glowing gas
  - High velocity flow (anisotropic elongation visible)
  - Mixed illumination (red + blue = purple)
  - Turbulent eddies in stream

**0:20-0:30:**
- Figure-8 camera motion:
  - Pass behind blue star (backlit gas)
  - Sweep around to red star (frontlit gas)
  - Cross between stars (gas stream fills frame)
  - Complete loop showing full system

**Performance Estimate:**
- 8K particles, 6 lights, dual gravity wells
- Estimated FPS: 100-120 @ 1080p (RTX 4060 Ti)
- Efficient (moderate particle count, simple types)

---

## Scenario 5: Protoplanetary Disk - Birth of Planets

### Overview

- **Title:** "The Cradle of Worlds"
- **Duration:** 75 seconds
- **Complexity:** High (18K particles, multi-scale structures)
- **Visual Style:** Hybrid (artistic colors + physically accurate gaps)
- **Primary Effect:** Young star with surrounding disk showing gap formation (planet clearing)

### Particle Configuration

**Type Distribution:**
- 10% STAR_MAIN_SEQUENCE (1,800 particles) - Central young star
- 60% DUST_PARTICLE (10,800 particles) - Dust disk
- 30% GAS_CLOUD (5,400 particles) - Gas disk

**Spatial Distribution:**
- Star: 0-150 unit radius (central core)
- Inner disk: 150-400 units (hot dust)
- Gap zone: 400-600 units (planet clearing, 80% density reduction)
- Outer disk: 600-1500 units (cool dust and gas)

**Per-Type Material Properties:**

**STAR_MAIN_SEQUENCE (Young T Tauri Star):**
```json
{
    "baseRadius": 2.0,
    "densityScale": 12.0,
    "opacityScale": 1.0,
    "emissionStrength": 7.0,
    "phaseG": 0.7,
    "scatteringAlbedo": 0.05,
    "tintColor": [1.0, 0.95, 0.9],
    "blackbodyWeight": 1.0,
    "anisotropyStrength": 0.1,
    "temperatureFloor": 4000.0,
    "temperatureCeiling": 6000.0
}
```

**DUST_PARTICLE (Protoplanetary Dust):**
```json
{
    "baseRadius": 0.3,
    "densityScale": 200.0,
    "opacityScale": 1.0,
    "emissionStrength": 0.01,
    "phaseG": 0.0,
    "scatteringAlbedo": 0.35,
    "absorptionCoeff": 20.0,
    "tintColor": [0.55, 0.45, 0.35],
    "blackbodyWeight": 0.3,
    "anisotropyStrength": 0.5,
    "temperatureFloor": 100.0,
    "temperatureCeiling": 1800.0
}
```

**GAS_CLOUD (Molecular Hydrogen):**
```json
{
    "baseRadius": 1.2,
    "densityScale": 0.008,
    "opacityScale": 0.2,
    "emissionStrength": 0.15,
    "phaseG": -0.1,
    "scatteringAlbedo": 0.75,
    "tintColor": [0.4, 0.5, 0.8],
    "blackbodyWeight": 0.2,
    "anisotropyStrength": 1.8,
    "temperatureFloor": 200.0,
    "temperatureCeiling": 5000.0
}
```

### Physics Setup

**Constraint Shape:** ACCRETION_DISK (with gap)
- Inner radius: 150 units (stellar surface)
- Gap: 400-600 units (planet clearing zone)
- Outer radius: 1500 units

**Black Hole Mass:** 1×10⁶ solar masses (T Tauri star equivalent)

**Alpha Viscosity:** 0.08 (moderate accretion)

**Gap Formation:**
- Particle density reduced by 80% in gap zone (400-600 units)
- Dust particles preferentially cleared (90% reduction)
- Gas particles moderately cleared (60% reduction)

**Temperature Gradient:**
- Inner disk (150-400): 800K-1800K (warm dust)
- Gap zone (400-600): 500K-800K (cool, low density)
- Outer disk (600-1500): 100K-500K (cold dust)

### Lighting Setup

**Preset:** Custom "Protoplanetary"

**Light Configuration:**
- 1 central light (young star)
  - Position: (0, 0, 0)
  - Color: Yellow/white (5000K)
  - Intensity: 8.0
  - Radius: 600 units
- 12 accent lights (disk illumination)
  - Ring formation at 800-unit radius
  - Color: Warm orange (disk emission)
  - Intensity: 1.5

### Camera Timeline

**0:00-0:15** - Wide Top-Down
- Position: (0, 2500, 0)
- Look At: (0, 0, 0)
- Movement: Slow descent (2500 → 2000 units)
- Framing: Full disk, gap clearly visible

**0:15-0:30** - Orbital Sweep (Mid-Altitude)
- Position: Circle at 1800 units, 1000 units altitude
- Look At: (0, 0, 0)
- Movement: Orbital (12° per second)
- Framing: Angled view, gap structure visible

**0:30-0:45** - Dive to Gap Zone
- Position: (1800, 1000, -1800) → (500, 150, -500)
- Look At: (0, 0, 0)
- Movement: Diving trajectory into gap
- Framing: Passing through outer disk into gap

**0:45-0:60** - Low Pass Through Gap
- Position: Circle at 500-unit radius, 80 units altitude
- Look At: (0, 0, 0)
- Movement: Fast orbital (25° per second)
- Framing: Low-altitude pass, gap walls visible on both sides

**0:60-0:75** - Pull Back to Edge-On
- Position: (500, 80, -500) → (2000, 0, -2000)
- Look At: (0, 0, 0)
- Movement: Dolly out to edge-on view
- Framing: Thin disk profile, gap visible as dark band

### Expected Visuals

**0:00-0:15:**
- Concentric ring structure:
  - Inner disk: Brown/gray dust (dense, opaque)
  - Gap zone: Dark band (low particle density)
  - Outer disk: Lighter brown dust (tenuous)
- Spiral density waves (alpha viscosity)

**0:15-0:30:**
- Angled view reveals:
  - Gap depth (clear channel)
  - Inner disk edge (sharp cutoff at 400 units)
  - Outer disk edge (sharp cutoff at 600 units)
  - Vertical scale height (thin disk)

**0:30-0:45:**
- Dive trajectory:
  - Start: Tenuous blue gas (outer disk)
  - Middle: Transition zone (gas + dust mix)
  - End: Gap zone (very low density, dark)
- Temperature decrease visible (warm → cool)

**0:45-0:60:**
- Low-altitude pass:
  - Gap "walls" on both sides (inner and outer disk edges)
  - Clear channel (80% fewer particles)
  - Central star bright in center
  - High-speed dust particles in outer disk

**0:60-0:75:**
- Edge-on profile:
  - Thin disk (100-unit thickness)
  - Dark gap band clearly visible
  - Inner disk brighter (hot dust emission)
  - Outer disk dimmer (cool dust)

**Performance Estimate:**
- 18K particles, 13 lights, gap physics
- Estimated FPS: 70-90 @ 1080p (RTX 4060 Ti)
- Moderate complexity

---

## Scenario 6: Supernova Remnant - Stellar Explosion Aftermath

### Overview

- **Title:** "The Crab Nebula"
- **Duration:** 60 seconds
- **Complexity:** High (15K particles, radial expansion)
- **Visual Style:** Artistic (vibrant colors, high contrast)
- **Primary Effect:** Expanding shell of hot gas with central neutron star

### Particle Configuration

**Type Distribution:**
- 0.1% STAR_MAIN_SEQUENCE (15 particles) - Central neutron star (very small, very hot)
- 70% GAS_CLOUD (10,500 particles) - Expanding ejecta
- 30% DUST_PARTICLE (4,485 particles) - Dust condensation

**Spatial Distribution:**
- Neutron star: 0-50 unit radius (central point)
- Inner shell: 400-800 units (hot, fast-moving ejecta)
- Outer shell: 800-1500 units (cool, slower ejecta)

**Per-Type Material Properties:**

**STAR_MAIN_SEQUENCE (Neutron Star):**
```json
{
    "baseRadius": 0.5,
    "densityScale": 1000.0,
    "opacityScale": 1.0,
    "emissionStrength": 15.0,
    "phaseG": 0.9,
    "scatteringAlbedo": 0.01,
    "tintColor": [1.0, 1.0, 1.0],
    "blackbodyWeight": 1.0,
    "anisotropyStrength": 0.0,
    "temperatureFloor": 100000.0,
    "temperatureCeiling": 500000.0
}
```

**GAS_CLOUD (Ejecta):**
```json
{
    "baseRadius": 1.5,
    "densityScale": 0.01,
    "opacityScale": 0.35,
    "emissionStrength": 1.2,
    "phaseG": -0.5,
    "scatteringAlbedo": 0.95,
    "tintColor": [0.2, 1.0, 0.6],
    "blackbodyWeight": 0.1,
    "anisotropyStrength": 4.0,
    "dissipationRate": 0.0,
    "temperatureFloor": 10000.0,
    "temperatureCeiling": 50000.0
}
```

**DUST_PARTICLE (Condensed Dust):**
```json
{
    "baseRadius": 0.4,
    "densityScale": 100.0,
    "opacityScale": 0.8,
    "emissionStrength": 0.05,
    "phaseG": 0.0,
    "scatteringAlbedo": 0.4,
    "tintColor": [0.7, 0.3, 0.2],
    "blackbodyWeight": 0.5,
    "anisotropyStrength": 1.5,
    "temperatureFloor": 500.0,
    "temperatureCeiling": 5000.0
}
```

### Physics Setup

**Constraint Shape:** SPHERE (expanding)
- Radius: 1500 units (current shell radius)

**Black Hole Mass:** 0 (no gravity, pure expansion)

**Radial Expansion:**
- Inner shell velocity: 150-200 units/second (fast ejecta)
- Outer shell velocity: 50-100 units/second (slow ejecta)
- No accretion (alpha viscosity = 0)

**Other Parameters:**
- Gravity strength: 0.0 (freefall expansion)
- Turbulence: 2.0 (very turbulent, filamentary structure)
- Damping: 0.05 (minimal damping)

### Lighting Setup

**Preset:** Custom "Supernova Remnant"

**Light Configuration:**
- 1 central light (neutron star)
  - Position: (0, 0, 0)
  - Color: Bright white/blue (500000K)
  - Intensity: 12.0
  - Radius: 1200 units
- 20 accent lights (shell illumination)
  - Scattered throughout shell (400-1200 unit radius)
  - Colors: Green/cyan/blue (emission lines)
  - Intensity: 2.0-4.0 (variable)

### Camera Timeline

**0:00-0:15** - Wide Establishing Shot
- Position: (3000, 2000, -3000)
- Look At: (0, 0, 0)
- Movement: Slow dolly in (3000 → 2500 units)
- Framing: Full remnant visible, filamentary structure

**0:15-0:30** - Orbital Sweep
- Position: Circle at 2000 units, horizontal plane
- Look At: (0, 0, 0)
- Movement: Orbital (18° per second)
- Framing: Side view, shell thickness visible

**0:30-0:45** - Dive Through Shell
- Position: (2000, 1000, -2000) → (300, 150, -300)
- Look At: (0, 0, 0)
- Movement: Diving trajectory through shell
- Framing: Passing through outer shell into interior

**0:45-0:60** - Close-Up of Neutron Star
- Position: Circle at 400 units, looking at center
- Look At: (0, 0, 0)
- Movement: Orbital (30° per second)
- Framing: Neutron star in center, surrounding ejecta

### Expected Visuals

**0:00-0:15:**
- Spherical shell structure (1500-unit radius)
- Filamentary gas (high turbulence creating threads)
- Green/cyan color (oxygen/nitrogen emission lines)
- Bright white core (neutron star)

**0:15-0:30:**
- Shell thickness visible (~400 units)
- Layered structure:
  - Inner: Bright cyan/blue (hot, fast)
  - Middle: Green (medium temp)
  - Outer: Red/brown (cool dust)
- Radial filaments (expansion pattern)

**0:30-0:45:**
- Dive through shell:
  - Outer: Cool brown dust
  - Middle: Warm green gas
  - Inner: Hot cyan gas
  - Core: Blindingly bright neutron star
- Temperature increase during dive

**0:45-0:60:**
- Close-up:
  - Tiny, incredibly bright neutron star (0.5-unit radius)
  - Surrounding hot gas (backward scattering glow)
  - Radial streamers of ejecta
  - High-velocity particles racing outward

**Performance Estimate:**
- 15K particles, 21 lights, expansion physics
- Estimated FPS: 75-95 @ 1080p (RTX 4060 Ti)
- Moderate complexity

---

## Scenario 7: Dust Torus - Cool Dusty Ring

### Overview

- **Title:** "The Dust Shroud"
- **Duration:** 45 seconds
- **Complexity:** Low (8K particles, simple physics)
- **Visual Style:** Physically Accurate (pure absorption/scattering)
- **Primary Effect:** Dense dust torus obscuring central region, backlit by embedded lights

### Particle Configuration

**Type Distribution:**
- 95% DUST_PARTICLE (7,600 particles) - Dense dust torus
- 5% GAS_CLOUD (400 particles) - Tenuous gas envelope

**Spatial Distribution:**
- Dust torus: 600-1200 unit radius, 200-unit thickness
- Gas envelope: 1200-1500 unit radius, 400-unit thickness

**Per-Type Material Properties:**

**DUST_PARTICLE (Dense Torus):**
```json
{
    "baseRadius": 0.7,
    "densityScale": 250.0,
    "opacityScale": 1.0,
    "emissionStrength": 0.005,
    "phaseG": 0.0,
    "scatteringAlbedo": 0.2,
    "absorptionCoeff": 25.0,
    "tintColor": [0.5, 0.4, 0.3],
    "blackbodyWeight": 0.5,
    "anisotropyStrength": 0.4,
    "temperatureFloor": 200.0,
    "temperatureCeiling": 1200.0
}
```

**GAS_CLOUD (Envelope):**
```json
{
    "baseRadius": 1.5,
    "densityScale": 0.002,
    "opacityScale": 0.1,
    "emissionStrength": 0.05,
    "phaseG": -0.2,
    "scatteringAlbedo": 0.8,
    "tintColor": [0.6, 0.5, 0.4],
    "blackbodyWeight": 0.3,
    "anisotropyStrength": 1.0,
    "temperatureFloor": 100.0,
    "temperatureCeiling": 500.0
}
```

### Physics Setup

**Constraint Shape:** TORUS
- Major radius: 900 units (torus center)
- Minor radius: 300 units (torus thickness)

**Black Hole Mass:** 5×10⁷ solar masses

**Alpha Viscosity:** 0.02 (very slow accretion)

**Other Parameters:**
- Gravity strength: 1.0
- Turbulence: 0.3 (low turbulence, settled torus)
- Damping: 0.8 (high damping, circular orbits)

### Lighting Setup

**Preset:** RTXDI Ring (16 lights)

**Light Configuration:**
- 16 lights embedded within torus (900-unit radius)
- Color: Warm orange/red (obscured by dust)
- Intensity: 4.0 (bright to penetrate dust)
- Radius: 400 units

### Camera Timeline

**0:00-0:15** - Edge-On View
- Position: (2500, 0, 0)
- Look At: (0, 0, 0)
- Movement: Slow orbit (5° per second)
- Framing: Thin dust ring, opaque

**0:15-0:30** - Transition to Face-On
- Position: (2500, 0, 0) → (0, 2500, 0)
- Look At: (0, 0, 0)
- Movement: Smooth transition (edge → face)
- Framing: Ring opens up to torus

**0:30-0:45** - Face-On View
- Position: (0, 2500, 0)
- Look At: (0, 0, 0)
- Movement: Slow descent (2500 → 2000 units)
- Framing: Full torus visible, dark center

### Expected Visuals

**0:00-0:15:**
- Thin brown line (edge-on torus)
- Completely opaque (dense dust blocks all light)
- Soft glow at edges (backlit by embedded lights)
- No central region visible (obscured)

**0:15-0:30:**
- Torus opens up during transition
- Interior becomes visible
- Embedded lights appear as dim orange glows
- Dark central cavity (no particles)

**0:30-0:45:**
- Face-on torus:
  - Brown/gray dust ring
  - Embedded orange lights (partially obscured)
  - Dark center (black hole region)
  - Tenuous gas envelope (faint blue glow)

**Performance Estimate:**
- 8K particles, 16 lights, simple physics
- Estimated FPS: 120+ @ 1080p (RTX 4060 Ti)
- Very efficient

---

## Scenario 8: Gas Cloud Dissipation - Ephemeral Beauty

### Overview

- **Title:** "The Fading Nebula"
- **Duration:** 40 seconds (20-second dissipation half-life)
- **Complexity:** Low (5K particles, simple physics)
- **Visual Style:** Artistic (vibrant colors, ethereal)
- **Primary Effect:** Gas cloud fading out over time, continuous respawn

### Particle Configuration

**Type Distribution:**
- 100% GAS_CLOUD (5,000 particles) - Dissipating nebula

**Per-Type Material Properties:**

**GAS_CLOUD:**
```json
{
    "baseRadius": 2.0,
    "densityScale": 0.005,
    "opacityScale": 0.3,
    "emissionStrength": 0.3,
    "phaseG": -0.4,
    "scatteringAlbedo": 0.95,
    "tintColor": [0.3, 0.7, 1.0],
    "blackbodyWeight": 0.05,
    "anisotropyStrength": 2.5,
    "dissipationRate": 0.05,
    "temperatureFloor": 2000.0,
    "temperatureCeiling": 10000.0
}
```

### Physics Setup

**Constraint Shape:** SPHERE
- Radius: 1000 units

**Dissipation:**
- Dissipation rate: 0.05 (20-second half-life)
- Respawn: Particles respawn at center when opacity < 0.01
- Radial drift: 20 units/second outward

**Other Parameters:**
- Gravity strength: 0.2 (weak pull)
- Turbulence: 1.5 (high turbulence, wispy)
- Damping: 0.2 (low damping)

### Lighting Setup

**Preset:** RTXDI Sphere (13 lights)

**Light Configuration:**
- 13 lights at 1200-unit radius (external illumination)
- Color: Blue/white (hot stars)
- Intensity: 3.0
- Radius: 600 units

### Camera Timeline

**0:00-0:20** - Wide Orbital
- Position: Circle at 1800 units
- Look At: (0, 0, 0)
- Movement: Orbital (15° per second)
- Framing: Full nebula, dissipation visible

**0:20-0:40** - Close Approach
- Position: (1800, 900, -1800) → (600, 300, -600)
- Look At: (0, 0, 0)
- Movement: Dolly in
- Framing: Interior details, respawn visible

### Expected Visuals

**0:00-0:20:**
- Blue wispy nebula
- Particles fading out at edges (low opacity)
- Dense center (recently spawned)
- Turbulent motion (anisotropic elongation)

**0:20-0:40:**
- Close-up:
  - Individual particles visible
  - Fading effect observable (opacity decay)
  - Respawn at center (sudden appearance)
  - Backward scattering glow (backlit by external lights)

**Performance Estimate:**
- 5K particles, 13 lights, dissipation physics
- Estimated FPS: 120+ @ 1080p (RTX 4060 Ti)
- Very efficient

---

## Scenario 9: Multi-Zone Accretion - Three Material Regimes

### Overview

- **Title:** "The Tri-Zone Disk"
- **Duration:** 60 seconds
- **Complexity:** High (20K particles, three distinct zones)
- **Visual Style:** Hybrid (educational visualization + artistic)
- **Primary Effect:** Clear demonstration of three material types in same scene

### Particle Configuration

**Type Distribution:**
- Inner zone (200-500 units): 100% PLASMA_BLOB (6,000 particles)
- Mid zone (500-900 units): 100% GAS_CLOUD (8,000 particles)
- Outer zone (900-1500 units): 100% DUST_PARTICLE (6,000 particles)

**Total:** 20,000 particles

**Material Properties:** Use default configurations from PARTICLE_TYPE_SYSTEM_SPEC.md

### Physics Setup

**Constraint Shape:** ACCRETION_DISK
- Inner radius: 200 units
- Outer radius: 1500 units
- Thickness: 120 units

**Black Hole Mass:** 1×10⁸ solar masses

**Alpha Viscosity:** 0.1

**Zone Transitions:**
- Abrupt transitions at 500 and 900 units (educational visualization)
- No particle type mixing (clear boundaries)

### Lighting Setup

**Preset:** Custom "Tri-Zone"

**Light Configuration:**
- 18 lights total
  - Inner ring (6 lights @ 350 units): White/blue
  - Mid ring (6 lights @ 700 units): Yellow/orange
  - Outer ring (6 lights @ 1200 units): Red/brown

### Camera Timeline

**0:00-0:20** - Top-Down Tour
- Position: Descend from (0, 3000, 0) to (0, 1500, 0)
- Look At: (0, 0, 0)
- Framing: Three zones clearly visible (color-coded)

**0:20-0:40** - Radial Flythrough
- Position: (1500, 300, 0) → (250, 50, 0)
- Look At: (0, 0, 0)
- Framing: Pass through all three zones inward

**0:40-0:60** - Low Orbital
- Position: Circle at 600 units, 100 units altitude
- Look At: (0, 0, 0)
- Framing: Mid-zone (gas cloud) close-up

### Expected Visuals

**Three distinct zones:**
- Inner: Bright white/blue (hot plasma)
- Mid: Yellow/orange (warm gas)
- Outer: Brown/red (cool dust)

**Performance Estimate:**
- 20K particles, 18 lights
- Estimated FPS: 65-85 @ 1080p

---

## Scenario 10: Celestial Ballet - Complex Multi-Body System

### Overview

- **Title:** "The Cosmic Dance"
- **Duration:** 120 seconds (full choreography)
- **Complexity:** Very High (25K particles, 8 distinct objects)
- **Visual Style:** Cinematic (dramatic lighting, dynamic composition)
- **Primary Effect:** Multiple celestial bodies interacting (stars, disks, jets)

### Particle Configuration

**Type Distribution:**
- 2 binary stars: 15% STAR_MAIN_SEQUENCE (3,750 particles)
- 3 accretion disks: 50% PLASMA_BLOB + GAS_CLOUD (12,500 particles)
- 2 nebula clouds: 20% GAS_CLOUD (5,000 particles)
- 1 dust torus: 15% DUST_PARTICLE (3,750 particles)

**Total:** 25,000 particles

### Complexity Note

This scenario requires **custom multi-body physics** (beyond Phase 5 scope). Included here as aspirational long-term goal.

### Expected Visuals

- Binary star system in center
- Surrounding accretion disks
- Background nebulae
- Foreground dust torus
- Dramatic lighting interactions

**Performance Estimate:**
- 25K particles, 25+ lights
- Estimated FPS: 45-60 @ 1080p (demanding)

---

## Technical Requirements Summary

### Hardware Requirements

**Minimum (30 FPS @ 720p):**
- GPU: RTX 2060 or RX 6600 XT
- VRAM: 6 GB
- System RAM: 16 GB

**Recommended (120 FPS @ 1080p):**
- GPU: RTX 4060 Ti or RX 7700 XT
- VRAM: 8 GB
- System RAM: 32 GB

**Optimal (120 FPS @ 1440p):**
- GPU: RTX 4070 Ti or RX 7900 XT
- VRAM: 12 GB
- System RAM: 32 GB

### Software Requirements

- Windows 10/11 (DXR 1.1 support)
- DirectX 12 Agility SDK 1.618.2+
- Visual Studio 2022 (for build)
- PIX for Windows (for debugging/capture)

### Performance Estimates

| Scenario | Particles | Lights | Est. FPS @ 1080p (RTX 4060 Ti) |
|----------|-----------|--------|--------------------------------|
| 1. Stellar Nursery | 15K | 20 | 80-100 |
| 2. Red Giant | 10K | 1 | 120+ |
| 3. Accretion Disk | 20K | 16 | 60-80 |
| 4. Binary Stars | 8K | 6 | 100-120 |
| 5. Protoplanetary | 18K | 13 | 70-90 |
| 6. Supernova | 15K | 21 | 75-95 |
| 7. Dust Torus | 8K | 16 | 120+ |
| 8. Gas Dissipation | 5K | 13 | 120+ |
| 9. Tri-Zone Disk | 20K | 18 | 65-85 |
| 10. Celestial Ballet | 25K | 25+ | 45-60 |

---

## Preset Configuration Files

### Preset File Naming Convention

`configs/scenarios/<scenario_name>.json`

Examples:
- `stellar_nursery_orion.json` (Scenario 1)
- `red_giant_betelgeuse.json` (Scenario 2)
- `accretion_disk_feeding.json` (Scenario 3)
- `binary_waltz.json` (Scenario 4)

### Preset File Structure

```json
{
    "name": "Stellar Nursery - Orion",
    "description": "Young hot stars forming within emission nebula",
    "version": "1.0",

    "particleSystem": {
        "particleCount": 15000,
        "typeDistribution": {
            "PLASMA_BLOB": 0.0,
            "STAR_MAIN_SEQUENCE": 30.0,
            "STAR_GIANT": 0.0,
            "GAS_CLOUD": 50.0,
            "DUST_PARTICLE": 20.0
        }
    },

    "materialTypes": {
        "STAR_MAIN_SEQUENCE": { /* material config */ },
        "GAS_CLOUD": { /* material config */ },
        "DUST_PARTICLE": { /* material config */ }
    },

    "physics": {
        "constraintShape": "TORUS",
        "blackHoleMass": 5.0e6,
        "alphaViscosity": 0.05,
        "gravityStrength": 1.2,
        "turbulence": 0.8,
        "damping": 0.3
    },

    "lighting": {
        "preset": "stellar_nursery",
        "lightCount": 20,
        "distribution": "embedded",
        "colorScheme": "blue_white"
    },

    "camera": {
        "keyframes": [
            {
                "time": 0.0,
                "position": [2000, 1000, -2000],
                "lookAt": [0, 0, 0],
                "fov": 60.0
            }
            /* ... more keyframes ... */
        ]
    }
}
```

---

## Animation Production Workflow

### Step 1: Conceptualization
- Choose scenario from library
- Identify key visual effects to showcase
- Sketch camera movement timeline

### Step 2: Configuration
- Load scenario preset JSON
- Adjust particle type distribution
- Tune material properties via ImGui

### Step 3: Physics Tuning
- Set constraint shape
- Adjust black hole mass, alpha viscosity
- Test particle motion (verify orbital dynamics)

### Step 4: Lighting Setup
- Load light preset
- Apply bulk color scheme
- Fine-tune individual lights if needed

### Step 5: Camera Choreography
- Define keyframe positions
- Set movement speed and easing
- Test camera path (manual movement first)

### Step 6: Capture
- Run at target resolution (1080p or 1440p)
- Record via PIX or screen capture software
- Export as image sequence or video

### Step 7: Post-Processing (Optional)
- Color grading
- Motion blur
- Glow/bloom effects
- Audio (music, narration)

---

## Success Criteria

**These scenarios are successfully implemented when:**

1. ✅ All 10 presets load without errors
2. ✅ Particle type distributions match specifications
3. ✅ Physics behaviors produce expected motion
4. ✅ Visual appearance matches described outcomes
5. ✅ Performance meets estimated FPS targets
6. ✅ Camera timelines execute smoothly
7. ✅ User (Ben) can create custom variations

**Definition of Done:**
- 10 preset JSON files created and tested
- User confirms: "I can create stunning animations"
- Documentation complete (this file)
- Example screenshots/videos captured

---

## Future Expansion Ideas

### Phase 6+ Scenarios

**Advanced Physics:**
- Gravitational lensing (light bending near black hole)
- Particle collisions (SPH-like)
- Magnetic field lines (particle alignment)

**New Visual Effects:**
- Volumetric fog/atmospheric scattering
- Chromatic aberration (spectral dispersion)
- Motion blur (fast-moving particles)

**Interactive Scenarios:**
- Real-time parameter adjustment during playback
- User-controlled camera (flight sim mode)
- VR/AR support (immersive exploration)

---

**Document Status:** Complete - Ready for User Creative Workflow
**Next Steps:** Begin Phase 5.1 implementation (particle type system)
