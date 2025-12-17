# Luminous Star Particles Documentation

**Feature:** Physics-driven light sources inside 3D Gaussian particles
**Status:** Architecture Approved, Implementation In Progress
**Target Version:** PlasmaDX 0.19.0
**Chosen Architecture:** Clean Architecture (LuminousParticleSystem class)

---

## Quick Start

1. **New to this feature?** Start with [FEATURE_OVERVIEW.md](FEATURE_OVERVIEW.md)
2. **Architecture details?** Read [ARCHITECTURE_OPTIONS.md](ARCHITECTURE_OPTIONS.md)
3. **Track implementation?** Follow [MILESTONE_CHECKLIST.md](MILESTONE_CHECKLIST.md) **(BUILD & TEST AFTER EACH)**
4. **Ready to implement?** Follow [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)
5. **Need technical details?** Reference [TECHNICAL_REFERENCE.md](TECHNICAL_REFERENCE.md)
6. **Modifying shaders?** Check [SHADER_MODIFICATIONS.md](SHADER_MODIFICATIONS.md)

---

## Document Index

| Document | Purpose | Key Content |
|----------|---------|-------------|
| [FEATURE_OVERVIEW.md](FEATURE_OVERVIEW.md) | High-level concept | Problem, solution, benefits, specs |
| [ARCHITECTURE_OPTIONS.md](ARCHITECTURE_OPTIONS.md) | Design choices | Minimal vs Clean Architecture (CLEAN CHOSEN) |
| [MILESTONE_CHECKLIST.md](MILESTONE_CHECKLIST.md) | **Implementation tracking** | Build-test checklist for each milestone |
| [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) | Step-by-step | Code changes, compilation, testing |
| [TECHNICAL_REFERENCE.md](TECHNICAL_REFERENCE.md) | Technical details | Data structures, memory, GPU sync |
| [SHADER_MODIFICATIONS.md](SHADER_MODIFICATIONS.md) | HLSL changes | Shader code, compilation commands |

---

## Feature Summary

### The Concept

Transform static lights into **living stars** by placing light sources inside 3D Gaussian particles. The result:

- **Visible volumetric stars** (not invisible point lights)
- **Physics-driven motion** (stars orbit with accretion disk)
- **Authentic stellar glow** (light shines through semi-transparent particle)
- **Dynamic shadowing** (shadows shift as stars move)

### Technical Specs (APPROVED)

| Specification | Value | Notes |
|---------------|-------|-------|
| Star particles | 16 | Fixed count, first 16 particles in buffer |
| Material type | SUPERGIANT_STAR (index 8) | New material type |
| Emission | 15× multiplier | Brightest material |
| Opacity | **0.15** (very transparent) | Light shines through particle |
| Temperature | 25000K (blue supergiant) | Hot blue-white glow |
| Max lights | **32** (unified buffer) | 16 stars + 16 static |
| Light radius | 100-200 units | Soft shadows |
| Light intensity | 8.0-15.0 | Configurable via ImGui |

### Chosen Architecture: Clean Architecture

| Aspect | Decision |
|--------|----------|
| **Architecture** | LuminousParticleSystem class |
| **Light buffer** | Single unified 32-light buffer |
| **Position sync** | CPU prediction (GPU readback optional) |
| **ImGui controls** | Yes - luminosity, opacity, radius |
| **Future effects** | Corona/halo (Phase 2) |

**Implementation:** See [MILESTONE_CHECKLIST.md](MILESTONE_CHECKLIST.md) for build-test workflow.

---

## Decision Checklist (ALL APPROVED)

All architectural decisions have been made:

- [x] **Material type:** SUPERGIANT_STAR (index 8)
- [x] **Star count:** 16 (first 16 particles)
- [x] **Opacity:** 0.15 (very transparent - light shines through)
- [x] **Max lights:** 32 total (unified buffer)
- [x] **Initial positions:** First 16 particles (physics-driven)
- [x] **Implementation option:** Clean Architecture (LuminousParticleSystem class)
- [x] **GPU sync strategy:** CPU prediction (Option C)
- [x] **Future effects:** Corona/halo (planned for Phase 2)

---

## File Structure

```
PlasmaDX-LuminousStars/
├── docs/
│   └── Luminous_stars/
│       ├── README.md                 ← You are here
│       ├── FEATURE_OVERVIEW.md       ← Start here
│       ├── ARCHITECTURE_OPTIONS.md   ← Design decisions
│       ├── MILESTONE_CHECKLIST.md    ← BUILD-TEST TRACKING (NEW)
│       ├── IMPLEMENTATION_GUIDE.md   ← Step-by-step code
│       ├── TECHNICAL_REFERENCE.md    ← Data structures
│       └── SHADER_MODIFICATIONS.md   ← HLSL changes
│
├── src/particles/
│   ├── LuminousParticleSystem.h      ← NEW FILE (to create)
│   └── LuminousParticleSystem.cpp    ← NEW FILE (to create)
│
└── shaders/particles/
    ├── particle_gaussian_raytrace.hlsl  ← MAX_LIGHTS change
    └── particle_physics.hlsl            ← Star particle init
```

---

## Related PlasmaDX Files

Files that will be modified/created for this feature:

### New Files (Clean Architecture)
- `src/particles/LuminousParticleSystem.h` - New class declaration
- `src/particles/LuminousParticleSystem.cpp` - New class implementation

### C++ Modifications
- `src/particles/ParticleSystem.h` - Add SUPERGIANT_STAR enum, expand materials[9]
- `src/particles/ParticleSystem.cpp` - Add material 8 properties
- `src/particles/ParticleRenderer_Gaussian.cpp` - MAX_LIGHTS 16→32
- `src/core/Application.h` - Add LuminousParticleSystem member
- `src/core/Application.cpp` - Initialize, Update, integrate star lights

### Shader Modifications
- `shaders/particles/particle_gaussian_raytrace.hlsl` - MAX_LIGHTS 16→32
- `shaders/particles/particle_physics.hlsl` - Star particle initialization (first 16)

### Build System
- `CMakeLists.txt` - Add new LuminousParticleSystem files

---

## Performance Impact

Expected overhead with 16 star particles (32 total lights):

| Component | Impact | Notes |
|-----------|--------|-------|
| Light buffer | +1KB (1KB → 2KB) | Negligible |
| Shader loop | +16 iterations | 13 → 29 lights |
| Position sync | <0.5ms/frame | CPU prediction |
| Shadow rays | +16 rays/pixel | Main cost |
| **Total** | **~5-10ms** | ~60-80 FPS expected |

**Accepted trade-off:** User approved ~60-80 FPS with 32 lights (down from ~120 FPS with 13 lights).

**Optimization potential:** Light LOD culling, reduce shadow rays for distant stars.

---

## Glossary

| Term | Definition |
|------|------------|
| **Luminous Star** | A particle with an attached light source |
| **SUPERGIANT_STAR** | New material type (index 8) for star particles |
| **Light-particle sync** | Updating light positions from particle positions |
| **CPU prediction** | Tracking positions on CPU to avoid GPU readback |
| **Fibonacci sphere** | Algorithm for evenly distributing points on a sphere |

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | Dec 2025 | Claude Code | Initial documentation |

---

## Next Steps

Architecture is approved. Implementation workflow:

1. **Open [MILESTONE_CHECKLIST.md](MILESTONE_CHECKLIST.md)** - Track progress
2. **Milestone 1:** Expand light buffer to 32 → Build → Test
3. **Milestone 2:** Add SUPERGIANT_STAR material → Build → Test
4. **Milestone 3:** Create LuminousParticleSystem class → Build → Test
5. **Milestone 4:** Integrate into Application → Build → Test
6. **Milestone 5:** Initialize star particles in shader → Build → Test
7. **Milestone 6:** Implement light position sync → Build → Test
8. **Milestone 7:** Add ImGui controls → Build → Test
9. **Milestone 8:** Performance validation
10. **Milestone 9:** Final polish and commit

**CRITICAL:** Build and test after EVERY milestone to catch bugs early!

---

*This documentation was generated to support the Luminous Star Particles feature for PlasmaDX-Clean.*
