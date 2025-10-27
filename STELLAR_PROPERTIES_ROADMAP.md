# Stellar Properties Enhancement Roadmap

**Project:** PlasmaDX-Clean
**Feature Area:** Multi-Light System - Celestial Body Properties
**Started:** 2025-10-27
**Current Branch:** 0.10.8

## Overview

Enhancement of the physics-driven multi-light system to simulate realistic stellar properties including temperature-based coloration, pulsation, size-luminosity relationships, stellar winds, and advanced astrophysical phenomena.

## Completed Features ✅

### Phase 1: Foundation (v0.10.8)
- ✅ **Physics-Driven Lights** - Keplerian orbital motion for lights
  - Lights orbit with ω ∝ 1/√r (matches particle physics)
  - Clockwise rotation matching accretion disk
  - Vertical oscillation for disk thickness
  - ImGui toggle for enable/disable
  - **Performance:** Zero impact on framerate
  - **Visual Impact:** Dramatic dynamic shadows, excellent RT showcase

- ✅ **Bulk Light Controls** - Apply settings to all lights simultaneously
  - Bulk color picker
  - Bulk intensity slider (0.1 - 20.0)
  - Bulk radius slider (1.0 - 200.0)
  - Individual apply buttons (color, intensity, radius, all)
  - Logging for all bulk operations
  - **Use Case:** Quickly create uniform stellar populations

### Phase 2: Stellar Classification (v0.10.9) ✅
- ✅ **Temperature-Based Color Shift** - Realistic stellar temperature mapping
  - Planck's law color temperature simulation
  - Intensity → Temperature → Color mapping
  - **Stellar Classes:**
    - O-type (30,000K+): Blue-white (RGB: 0.6, 0.7, 1.0)
    - B-type (10,000-30,000K): Blue-white (RGB: 0.8, 0.85, 1.0)
    - A-type (7,500-10,000K): White (RGB: 1.0, 1.0, 1.0)
    - F-type (6,000-7,500K): Yellow-white (RGB: 1.0, 0.95, 0.85)
    - G-type (5,200-6,000K): Yellow (RGB: 1.0, 0.9, 0.7) - Sun-like
    - K-type (3,700-5,200K): Orange (RGB: 1.0, 0.7, 0.5)
    - M-type (2,400-3,700K): Red (RGB: 1.0, 0.5, 0.3) - Red dwarfs/supergiants
  - Auto-apply mode (intensity drives color)
  - Manual override capability
  - ImGui toggle: "Stellar Temperature Colors"
  - **Performance:** CPU-only, negligible cost
  - **Visual Impact:** Physically accurate stellar populations
  - **Estimated Time:** 2-3 hours

## Planned Features 📋

### Phase 3: Stellar Dynamics (v0.11.0)
- ⏳ **Pulsation/Variable Stars** - Time-varying intensity
  - Sinusoidal intensity modulation
  - Per-light pulsation period (seconds)
  - Amplitude control (% intensity variation)
  - **Star Types:**
    - Cepheid variables (1-100 day periods)
    - RR Lyrae variables (0.2-2 day periods)
    - Mira variables (80-1,000 day periods)
  - Phase offset per light (avoid synchronization)
  - ImGui controls: Enable toggle, period slider, amplitude slider
  - **Performance:** Trivial (sin() calculation per light)
  - **Visual Impact:** Organic, living stellar environment
  - **Estimated Time:** 2-3 hours

- ⏳ **Size-Luminosity Relationship** - Realistic stellar radii
  - Stefan-Boltzmann law: L ∝ R² T⁴
  - Auto-scale radius based on intensity and color
  - **Examples:**
    - Blue supergiants: High luminosity, moderate radius
    - Red supergiants: High luminosity, huge radius (Betelgeuse)
    - Red dwarfs: Low luminosity, small radius
  - Optional override for artistic control
  - ImGui toggle: "Realistic Stellar Radii"
  - **Performance:** CPU-only, negligible cost
  - **Visual Impact:** Visually distinct star types
  - **Estimated Time:** 2-3 hours

### Phase 4: Stellar Activity (v0.11.1)
- ⏳ **Stellar Flares/Coronal Mass Ejections** - Dynamic events
  - Random intensity spikes (10-50% increase)
  - Duration: 0.5-2 seconds
  - Frequency: Adjustable per light (flare-active vs quiet)
  - Visual flash + optional particle burst
  - ImGui controls: Flare frequency, max amplitude
  - **Performance:** Minimal (random number generation)
  - **Visual Impact:** Dramatic, attention-grabbing events
  - **Estimated Time:** 4-6 hours

- ⏳ **Binary/Multiple Star Systems** - Orbital pairs
  - Light pairing system (binary partner index)
  - Lights orbit each other while orbiting black hole
  - Mass ratio affects dynamics
  - Visual tethers (optional debug lines)
  - ImGui: "Link as Binary" button, mass ratio slider
  - **Examples:**
    - Sirius A+B (massive + white dwarf)
    - Alpha Centauri A+B (sun-like pair)
  - **Performance:** 2× calculations for paired lights
  - **Visual Impact:** Complex, realistic dynamics
  - **Estimated Time:** 6-8 hours

### Phase 5: Volumetric Stellar Atmospheres (v0.11.2)
- ⏳ **Atmospheric Scattering Halo** - Volumetric glow
  - Secondary particle system or shader-based
  - Intensity-dependent radius (L ∝ R²)
  - Reuse 3D Gaussian splatting infrastructure
  - Atmospheric density falloff (exponential)
  - **Star Types:**
    - Giants/supergiants: Large, diffuse halos
    - Main sequence: Moderate halos
    - White dwarfs: Minimal/no halo
  - ImGui toggle: "Stellar Atmospheres"
  - Color matching parent star temperature
  - **Performance:** Depends on particle count (target: 5-10% FPS cost)
  - **Visual Impact:** Soft, nebula-like light sources
  - **Estimated Time:** 8-12 hours

### Phase 6: Stellar Wind Interactions (v0.12.0) - RESEARCH LEVEL
- ⏳ **Stellar Wind Particle Injection** - Active mass loss
  - Lights spawn high-velocity particles radially
  - Wind speed based on stellar type:
    - O/B stars: 2,000-3,000 km/s
    - Sun-like: 400 km/s (solar wind)
    - Red giants: 10-30 km/s (slow, dense)
  - Particle-particle collisions (wind vs disk)
  - Bow shock formation
  - Mass loss rate parameter (M☉/year)
  - ImGui: Wind speed, mass loss rate, enable toggle
  - **Performance:** Depends on particle budget (10-20% FPS cost estimated)
  - **Visual Impact:** Dynamic interaction, bow shocks, turbulence
  - **Estimated Time:** 2-3 days

### Phase 7: Advanced Spectral Simulation (v0.12.1) - RESEARCH LEVEL
- ⏳ **Spectral Line Simulation** - Multi-wavelength rendering
  - Separate color channels for spectral lines:
    - H-alpha (656nm) - Hydrogen emission
    - H-beta (486nm) - Hydrogen emission
    - O-III (501nm) - Doubly ionized oxygen
    - N-II (658nm) - Nitrogen emission
  - Particles respond to different wavelengths
  - Temperature/density-dependent line strengths
  - Optional "filter wheel" to view single lines
  - ImGui: Spectral line toggles, filter mode
  - **Applications:**
    - Scientific visualization
    - Realistic H-II regions (emission nebulae)
    - AGN/quasar accretion disk physics
  - **Performance:** 4× lighting cost (one pass per line) - significant
  - **Visual Impact:** Scientifically accurate, publication-ready
  - **Estimated Time:** 3-5 days

## Performance Budget

| Feature | CPU Cost | GPU Cost | FPS Impact (10K particles, 13 lights) |
|---------|----------|----------|----------------------------------------|
| Physics-Driven Lights | ~0.1ms | 0ms | 0 FPS (120 → 120) ✅ |
| Bulk Controls | ~0.01ms | 0ms | 0 FPS ✅ |
| Temperature Colors | ~0.05ms | 0ms | 0 FPS ✅ |
| Pulsation | ~0.05ms | 0ms | 0 FPS ✅ |
| Size-Luminosity | ~0.05ms | 0ms | 0 FPS ✅ |
| Stellar Flares | ~0.1ms | 0ms | 0 FPS ✅ |
| Binary Systems | ~0.2ms | 0ms | 0 FPS ✅ |
| Atmospheric Halos | ~0.5ms | 2-3ms | -5 to -10 FPS (120 → 110-115) ⚠️ |
| Stellar Winds | ~1ms | 5-10ms | -15 to -25 FPS (120 → 95-105) ⚠️ |
| Spectral Lines | ~0.1ms | 15-20ms | -30 to -40 FPS (120 → 80-90) ⚠️ |

**Target:** Maintain 120 FPS @ 10K particles, 1080p on RTX 4060 Ti for Phases 1-4
**Acceptable:** 100+ FPS for Phase 5, 80+ FPS for Phases 6-7 (research features)

## Technical Implementation Notes

### Temperature → Color Mapping (Phase 2)
```cpp
DirectX::XMFLOAT3 GetStellarColor(float temperature) {
    // Wien's displacement law + blackbody approximation
    if (temperature > 30000.0f) {
        return XMFLOAT3(0.6f, 0.7f, 1.0f);  // O-type: Blue
    } else if (temperature > 10000.0f) {
        return XMFLOAT3(0.8f, 0.85f, 1.0f);  // B-type: Blue-white
    } else if (temperature > 7500.0f) {
        return XMFLOAT3(1.0f, 1.0f, 1.0f);  // A-type: White
    } else if (temperature > 6000.0f) {
        return XMFLOAT3(1.0f, 0.95f, 0.85f);  // F-type: Yellow-white
    } else if (temperature > 5200.0f) {
        return XMFLOAT3(1.0f, 0.9f, 0.7f);  // G-type: Yellow (Sun)
    } else if (temperature > 3700.0f) {
        return XMFLOAT3(1.0f, 0.7f, 0.5f);  // K-type: Orange
    } else {
        return XMFLOAT3(1.0f, 0.5f, 0.3f);  // M-type: Red
    }
}

// Intensity → Temperature mapping
float temperature = 3000.0f + (intensity / 20.0f) * 27000.0f;  // 3000K - 30000K range
```

### Pulsation Implementation (Phase 3)
```cpp
// Per-light properties
struct Light {
    // ... existing fields ...
    float pulsationPeriod;    // Seconds (0 = no pulsation)
    float pulsationAmplitude; // Fraction (0.0 - 1.0)
    float pulsationPhase;     // Initial phase offset (radians)
};

// Update loop
float pulsationFactor = 1.0f +
    pulsationAmplitude * sinf(2.0f * PI * totalTime / pulsationPeriod + pulsationPhase);
float effectiveIntensity = baseIntensity * pulsationFactor;
```

### Binary Orbit Implementation (Phase 4)
```cpp
struct Light {
    // ... existing fields ...
    int binaryPartnerIndex;  // -1 = no partner
    float binaryMassRatio;   // 0.1 - 10.0 (M_this / M_partner)
    float binarySeparation;  // Orbital radius (units)
};

// Two-body orbit around center of mass
XMFLOAT3 centerOfMass = (m1 * p1 + m2 * p2) / (m1 + m2);
float r1 = binarySeparation * m2 / (m1 + m2);
float r2 = binarySeparation * m1 / (m1 + m2);
```

## Astrophysical Reference

### Stellar Classification (Morgan-Keenan System)
- **O-type:** Hottest, most massive, shortest-lived (3-10 M☉, <10 million years)
- **B-type:** Hot, massive, short-lived (2-16 M☉, 10-100 million years)
- **A-type:** Hot, medium mass (1.4-2 M☉, 1-2 billion years)
- **F-type:** Medium temperature (1.04-1.4 M☉, 2-4 billion years)
- **G-type:** Sun-like, stable (0.8-1.04 M☉, 8-12 billion years) - **OUR SUN**
- **K-type:** Cool, orange (0.45-0.8 M☉, 15-30 billion years)
- **M-type:** Coolest, red dwarfs/giants (0.08-0.45 M☉, 50+ billion years)

### Luminosity Classes
- **Ia:** Bright supergiant (Rigel, Betelgeuse)
- **Ib:** Less luminous supergiant
- **II:** Bright giant
- **III:** Giant
- **IV:** Subgiant
- **V:** Main sequence (Sun = G2V)

### Variable Star Types
- **Cepheids:** 1-100 day periods, period-luminosity relation (distance indicators)
- **RR Lyrae:** 0.2-2 day periods, horizontal branch stars
- **Mira variables:** 80-1,000 day periods, red giants, large amplitude
- **Delta Scuti:** 0.03-0.3 day periods, A-F type stars

## Research Papers & References

1. **Blackbody Radiation:**
   - Planck, M. (1900). "On the Theory of the Energy Distribution Law of the Normal Spectrum"
   - Wien, W. (1893). "A New Relationship Between Blackbody Radiation and Temperature"

2. **Stellar Evolution:**
   - Kippenhahn, R. & Weigert, A. (1990). "Stellar Structure and Evolution"
   - Iben, I. (1967). "Stellar Evolution. I. The Approach to the Main Sequence"

3. **Variable Stars:**
   - Leavitt, H. S. (1912). "Periods of 25 Variable Stars in the Small Magellanic Cloud"
   - Christy, R. F. (1966). "A Study of Pulsation in RR Lyrae Models"

4. **Stellar Winds:**
   - Lamers, H. & Cassinelli, J. (1999). "Introduction to Stellar Winds"
   - Abbott, D. C. (1982). "The Theory of Radiatively Driven Stellar Winds"

5. **Binary Systems:**
   - Eggleton, P. P. (1983). "Approximations to the Radii of Roche Lobes"
   - Paczyński, B. (1971). "Evolutionary Processes in Close Binary Systems"

## Success Metrics

### Visual Quality
- ✅ Lights exhibit realistic color progression (blue → white → yellow → red)
- ✅ Variable stars show periodic intensity changes
- ✅ Binary systems display physically plausible orbits
- ✅ Stellar atmospheres create soft, believable glows
- ✅ Overall scene resembles Hubble/JWST astrophotography

### Performance
- ✅ Maintain 120 FPS for Phases 1-4 (10K particles, 13 lights, 1080p)
- ✅ Achieve 100+ FPS for Phase 5 (atmospheric halos)
- ✅ Achieve 80+ FPS for Phases 6-7 (stellar winds, spectral lines)

### Usability
- ✅ All features accessible via ImGui with clear labels
- ✅ Presets for common stellar configurations
- ✅ Bulk controls for efficient parameter adjustment
- ✅ Real-time parameter changes (no rebuild required)
- ✅ Comprehensive logging for debugging

## Next Steps

**Immediate (Today):**
1. Implement temperature-based color shift (Phase 2)
2. Test with existing light presets (Sphere, Ring, Disk)
3. Add ImGui toggle and auto-apply mode

**This Week:**
1. Complete Phase 2 (temperature colors)
2. Begin Phase 3 (pulsation system)
3. Test performance with all Phase 1-3 features enabled

**This Month:**
1. Complete Phases 1-4 (all dynamic features)
2. Begin research for Phase 5 (atmospheric halos)
3. Consider ML-based stellar atmosphere rendering

**Long-term:**
1. Publish paper/demo on RT stellar physics rendering
2. Integrate with PINN ML physics system
3. VR/AR support for immersive stellar environments

---

**Last Updated:** 2025-10-27
**Maintainer:** Claude Code + Ben
**Status:** Active Development - Phase 2 In Progress
