# Luminous Star Particles - Feature Overview

**Document Version:** 1.0
**Created:** December 2025
**Status:** Planning Phase
**Target Version:** 0.19.0

---

## Executive Summary

Luminous Star Particles is a new graphical feature that combines the 3D Gaussian volumetric particle system with the multi-light system to create physics-driven light sources. Instead of static lights floating in space, lights will be placed *inside* special "supergiant star" particles, creating authentic volumetric stellar glow that moves with the accretion disk physics simulation.

---

## The Problem

Currently, PlasmaDX-Clean has two separate systems that don't interact:

1. **Multi-Light System** - 13-16 static lights positioned around the accretion disk
   - Lights illuminate particles with beautiful scattered glow
   - Lights are stationary (don't move with physics)
   - Lights are invisible (just point sources, no visual representation)

2. **Particle System** - 10K-30K+ particles simulating accretion disk
   - Particles have temperature-based blackbody emission (self-glow)
   - Particles move with physics (gravity, Keplerian orbits)
   - But particles don't actually *illuminate* other particles significantly

**The Gap:** There's no way to have a visible, glowing object that also illuminates its surroundings while moving with physics.

---

## The Solution: Luminous Star Particles

Combine lights and particles into a single entity:

```
┌─────────────────────────────────────────────────────────┐
│                  LUMINOUS STAR PARTICLE                 │
│                                                         │
│    ┌─────────────────────────────────────────────┐     │
│    │            3D Gaussian Particle              │     │
│    │    ┌───────────────────────────────┐        │     │
│    │    │         Light Source          │        │     │
│    │    │    (position syncs from       │        │     │
│    │    │     particle each frame)      │        │     │
│    │    └───────────────────────────────┘        │     │
│    │                                             │     │
│    │  Material: SUPERGIANT_STAR                  │     │
│    │  - 15x emission multiplier                  │     │
│    │  - 0.15 opacity (light shines through!)     │     │
│    │  - Temperature: 25000K+ (blue-white)        │     │
│    └─────────────────────────────────────────────┘     │
│                                                         │
│    Physics: Moves with black hole gravity               │
│    Rendering: Volumetric glow + illuminates neighbors   │
└─────────────────────────────────────────────────────────┘
```

---

## Key Benefits

### 1. Visual Authenticity
- Stars are now *visible volumetric objects*, not invisible point lights
- The glow comes from *inside* the star, creating authentic halos
- Semi-transparent particles allow light to "shine through"

### 2. Physics Integration
- Star lights move with accretion disk rotation
- Stars follow Keplerian orbits around the black hole
- Dynamic lighting changes as stars orbit

### 3. Performance
- System already handles 30K+ particles at 120+ FPS
- Adding 16 special particles is negligible
- Light buffer expansion (16→32) is minimal overhead

### 4. Future Potential
- Convert ALL existing lights to star particles (32 total)
- Add different star types (red giants, white dwarfs, pulsars)
- Binary star systems with shared orbits
- Supernova events (star particle → explosion)

---

## Feature Specifications

### Core Requirements (APPROVED)

| Requirement | Specification | Notes |
|-------------|---------------|-------|
| Star particle count | 16 | First 16 particles in buffer |
| Material type | `SUPERGIANT_STAR` (index 8) | New material type |
| Emission multiplier | 15× | Brightest material |
| Opacity | **0.15** (very transparent) | Light shines through particle |
| Temperature | 25000K+ | Blue-white supergiant |
| Light radius | 100-200 units | Configurable via ImGui |
| Light intensity | 8.0-15.0 | Configurable via ImGui |
| Physics | Full accretion disk simulation | Stars orbit with disk |
| Max lights | 32 (unified buffer) | 16 stars + 16 static |

### Initial Configuration

For first implementation, spawn 16 star particles in these positions:
- **4 spiral arm stars** - Blue supergiants at 200-unit radius, 0°/90°/180°/270°
- **8 disk hotspot stars** - Main sequence stars scattered 100-400 units
- **4 random stars** - Mixed types (red giants, white dwarfs)

### Performance Targets

| Metric | Current | With Feature | Impact |
|--------|---------|--------------|--------|
| Frame time | 8.3ms | 8.5ms | +0.2ms |
| Light count | 13 | 29 | +16 |
| Particle count | 10K | 10K+16 | +0.16% |
| GPU memory | 64MB | 65MB | +1MB |

---

## User Experience

### Visual Effect

When enabled, users will see:
1. **Glowing star particles** - Bright volumetric spheres that stand out from regular particles
2. **Dynamic shadows** - As stars orbit, shadows shift across the accretion disk
3. **Light scattering** - Nearby particles glow brighter when close to stars
4. **Orbital motion** - Stars visibly orbit the black hole with the disk

### ImGui Controls (Planned)

```
[Star Particle System]
├── Enable Star Particles: [x]
├── Active Stars: 16 / 32
├── Global Luminosity: [====|====] 1.0
├── Global Opacity: [==|======] 0.5
├── [Spawn Spiral Arm Stars (4)]
├── [Spawn Disk Hotspots (8)]
├── [Spawn Random Stars]
├── [Clear All Stars]
└── [+] Individual Stars
    ├── Star 0: T=25000K, L=10.0, Pos=(200, 0, 0)
    ├── Star 1: T=25000K, L=10.0, Pos=(0, 0, 200)
    └── ...
```

---

## Success Criteria

### Minimum Viable Product (MVP)
- [ ] 16 star particles spawn at hardcoded positions
- [ ] Light positions sync from particle positions each frame
- [ ] Star particles render with high emission (visible glow)
- [ ] Other particles receive illumination from star lights
- [ ] Performance impact <1ms

### Full Feature
- [ ] All MVP criteria
- [ ] ImGui controls for spawn/clear/adjust
- [ ] Multiple star presets (blue supergiant, red giant, etc.)
- [ ] Runtime adjustable luminosity and opacity
- [ ] Star particle info display (temperature, position)

### Stretch Goals
- [ ] Pulsating stars (intensity modulation)
- [ ] Binary star systems
- [ ] Convert existing 13 lights to star particles (29→32 total)
- [ ] Supernova trigger (star → explosion)

---

## Technical Approach Summary

Two implementation approaches are available:

### Approach 1: Minimal (~54 lines)
- Direct integration into existing systems
- No new classes
- First 16 particles reserved as stars
- Simple and fast to implement

### Approach 2: StarParticleSystem Class (~300 lines)
- Dedicated manager class
- Better separation of concerns
- More extensible
- Full ImGui integration

**Recommendation:** Start with Approach 1 (minimal) to validate the concept, then refactor to Approach 2 if more features are needed.

---

## Related Documents

- `ARCHITECTURE_OPTIONS.md` - Detailed comparison of implementation approaches
- `IMPLEMENTATION_GUIDE.md` - Step-by-step implementation instructions
- `SHADER_MODIFICATIONS.md` - Required HLSL changes
- `TECHNICAL_REFERENCE.md` - Data structures, memory layouts, GPU sync strategies

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | Dec 2025 | Claude Code | Initial document |
