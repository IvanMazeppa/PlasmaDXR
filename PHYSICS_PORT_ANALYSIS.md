# Physics Engine Port Analysis
## PlasmaVulkan → PlasmaDX-Clean

**Document Version:** 1.0
**Created:** 2025-10-15
**Status:** Planning Phase

---

## Executive Summary

This document analyzes the PlasmaVulkan physics implementation and identifies which features to port, which to skip, and which to stub for future implementation.

**Key Principle:** Add features incrementally, testing thoroughly between each addition to avoid breaking the working simulation.

---

## Feature Analysis

### ✅ Features to Port (Priority Order)

#### 1. **Constraint Shapes System** - HIGH PRIORITY
**Status:** PARTIALLY IMPLEMENTED (needs completion)
**Complexity:** MEDIUM
**Current State:** Code exists in DX12 shader but not fully exposed to runtime controls

**What to Port:**
- `applyConstraints()` function (lines 40-104 in Vulkan)
- SPHERE constraint (line 41-50)
- DISC constraint (line 51-70)
- TORUS constraint (line 71-103)
- ACCRETION_DISK mode logic (line 183-199)

**Runtime Controls Needed:**
```cpp
enum ConstraintShape {
    NONE = 0,
    SPHERE = 1,
    DISC = 2,
    TORUS = 3,
    ACCRETION_DISK = 4
};

float constraintRadius;      // Outer radius (runtime adjustable)
float constraintThickness;   // Disk/torus thickness (runtime adjustable)
uint32_t constraintShape;    // Shape selector
```

**ImGui Controls:**
- Dropdown: Constraint Shape (None, Sphere, Disc, Torus, Accretion Disk)
- Slider: Outer Radius (10.0 - 1000.0)
- Slider: Thickness (10.0 - 200.0)
- Hotkeys:
  - `5`: Cycle constraint shapes
  - `[` / `]`: Adjust radius
  - `-` / `=`: Adjust thickness

**Implementation Notes:**
- Current DX12 shader already has `ApplyConstraints()` at lines 34-81
- Already has `constraintShape`, `constraintRadius`, `constraintThickness` in constants
- Just needs full exposure to ImGui and hotkey controls
- NO TORUS WIREFRAME VISUALIZATION (Vulkan had ugly debug mesh, we'll do better)

---

#### 2. **Black Hole Mass Parameter** - HIGH PRIORITY
**Status:** PARTIALLY IMPLEMENTED
**Complexity:** LOW
**Current State:** Hardcoded constant, not runtime adjustable

**What to Port:**
- Black hole mass as runtime parameter (line 28 in Vulkan push constants)
- Keplerian velocity calculation using mass (line 251 in Vulkan)
- Paczynski-Wiita pseudo-Newtonian potential (lines 185-191)
- ISCO (Innermost Stable Circular Orbit) calculations (line 180, 194-199)

**Runtime Controls Needed:**
```cpp
float blackHoleMass;  // In solar masses (default: 4.3e6 for Sgr A*)
```

**ImGui Controls:**
- Slider: Black Hole Mass (1.0 - 1e10 solar masses, logarithmic scale)
- Presets:
  - Stellar Mass: 10 M☉
  - Intermediate: 1000 M☉
  - Sgr A*: 4.3e6 M☉ (default)
  - Quasar: 1e9 M☉
- Hotkey:
  - `Ctrl+M` / `Shift+M`: Adjust mass (log scale)

**Implementation Notes:**
- Affects Keplerian orbital velocity: `v = sqrt(GM/r)`
- Schwarzschild radius: `r_s = 2GM/c² ≈ 0.5 * mass` (in our units)
- ISCO at 3 Schwarzschild radii: particles sink inside this

---

#### 3. **Alpha Viscosity (Inward Spiral)** - MEDIUM PRIORITY
**Status:** NOT IMPLEMENTED
**Complexity:** LOW
**Current State:** Not present in DX12 version

**What to Port:**
- Alpha viscosity parameter (line 29 in Vulkan)
- Radial drift calculation (lines 257-261)
- Viscous heating (line 453)

**Physics Background:**
- Shakura-Sunyaev α-disk model (1973)
- α represents turbulent viscosity strength
- Realistic range: 0.01 - 0.4
- Causes particles to slowly spiral inward while orbiting

**Runtime Controls Needed:**
```cpp
float alphaViscosity;  // Shakura-Sunyaev α (0.0 - 1.0)
```

**ImGui Controls:**
- Slider: Alpha Viscosity (0.0 - 1.0)
- Default: 0.1
- Hotkey:
  - `Ctrl+V` / `Shift+V`: Adjust viscosity

**Implementation Notes:**
- Very small effect: `radialDrift = -normalize(toCenter) * α * 0.01`
- Only active in ACCRETION_DISK mode
- Creates gradual inward motion (accretion)
- Generates viscous heating (temperature increase)

---

#### 4. **Enhanced Temperature Models** - MEDIUM PRIORITY
**Status:** BASIC VERSION EXISTS
**Complexity:** MEDIUM
**Current State:** DX12 has distance-based temperature, needs velocity-based enhancement

**What to Port:**
- Orbital velocity-based temperature (lines 432-476 in Vulkan)
- Keplerian velocity deviation heating (lines 445-450)
- Viscous dissipation heating (line 453)
- Conservative white-hot core model (lines 455-470)

**Physics Background:**
- Temperature from orbital shear, not just distance
- Particles at "wrong" orbital speed heat up (friction)
- Viscous dissipation converts orbital energy to heat
- Allows white-hot cores in high-shear regions

**Runtime Controls Needed:**
```cpp
// No new parameters - uses existing velocity/density data
// Just different temperature calculation
```

**Implementation Notes:**
- Replaces simple distance-based temperature model
- More physically accurate
- Allows greater temperature range (white-hot cores)
- Smoother color transitions

---

### ❌ Features NOT to Port

#### 1. **SPH (Smoothed Particle Hydrodynamics)** - DO NOT PORT
**Reason:** Never worked well, too complex, different use case

**What it was:**
- Fluid simulation mode (lines in `sph.comp`, `sph_simple.comp`)
- Pressure forces between particles
- Density-based interactions
- Spatial hashing for neighbor finding

**Why skip:**
- Very computationally expensive (requires spatial hash grid)
- Incompatible with current Gaussian splatting renderer
- Better suited for fluid sim, not accretion disk
- Would require major architectural changes

**Alternative:**
- Current curl noise turbulence is sufficient
- Density field already provides fluid-like appearance
- RT lighting + Gaussian splatting gives volumetric look

---

#### 2. **Torus Wireframe Visualization** - DO NOT PORT
**Reason:** Ugly debug mesh, we can do better

**What it was:**
- Debug wireframe mesh showing torus boundary
- Helped visualize constraint shape
- Very basic rendering

**Why skip:**
- Not visually appealing
- DX12 project has better UI (ImGui overlays)
- If needed, create proper torus debug visualization later

**Alternative:**
- Show constraint parameters in ImGui text
- Possibly add particle coloring based on constraint violations
- Could add RT-based constraint visualization later (glow at boundaries)

---

#### 3. **Dual Galaxy Collision Mode** - DO NOT PORT (YET)
**Reason:** Never worked correctly, but might be worth revisiting with DX12

**What it was:**
- Two gravitational centers (lines 121-163 in Vulkan)
- Particles assigned to Galaxy A or Galaxy B
- Tidal interactions between galaxies
- Complex orbital mechanics

**Why it failed in Vulkan:**
- Particles immediately mixed between galaxies
- No clear separation or structure
- Gravity calculations were unstable
- Visual result was just a messy blob

**Possible Future Revisit:**
- DX12 has more stable physics foundation
- Could work with proper particle initialization
- Needs galaxy-specific initial conditions
- Would be dramatic visual feature if working

**Decision:** Create a STUB but don't implement yet
```cpp
// STUB: Dual galaxy mode (not implemented)
// uint32_t dualGalaxyMode = 0;
// vec3 gravityCenter2;
// float blackHoleMass2;
```

---

### 🔧 Features to STUB (Future Implementation)

#### 1. **Relativistic Jets** - STUB FOR RT-BASED IMPLEMENTATION
**Reason:** Particles aren't the right approach, RT engine can do better

**What it was in Vulkan:**
- Polar particle ejection (lines 314-382)
- High-velocity particle streams
- Collimation forces
- Temperature marking

**Why not port as-is:**
- Wastes particles on jet (want them in disk)
- Jets should be volumetric, not particle-based
- RT engine can render volumetric jets separately
- Better physics: separate jet system from disk particles

**Future RT-Based Approach:**
1. Detect high central density (jet trigger condition)
2. Spawn volumetric emission regions along poles
3. RT lighting system renders jets as emissive volumes
4. Particles stay in disk, jets are pure visual effect
5. Could even add synchrotron radiation (magnetic field spirals)

**Stub Implementation:**
```cpp
// STUB: Relativistic jets
// Future: RT-based volumetric jet emission system
// Trigger when central density > threshold
// Render as emissive volumes with RT lighting
struct JetSystemStub {
    bool jetsEnabled = false;        // Future feature toggle
    float jetStrength = 1.0f;        // Emission intensity
    float jetOpeningAngle = 10.0f;   // Degrees from pole
    // Implementation: Separate from particle physics
};
```

**ImGui Placeholder:**
- Checkbox: "Enable Relativistic Jets (Coming Soon)"
- Tooltip: "Future feature: RT-based volumetric jet rendering"

---

#### 2. **Dual Galaxy Collision** - STUB FOR FUTURE ATTEMPT
**Create placeholder for potential future feature**

```cpp
// STUB: Dual galaxy collision mode
// Not implemented - original Vulkan version was unstable
// Could revisit with proper initialization and physics
struct DualGalaxyStub {
    bool dualGalaxyMode = false;          // Feature toggle
    DirectX::XMFLOAT3 gravityCenter2;     // Second galaxy center
    float blackHoleMass2 = 4.3e6f;        // Second BH mass
    float galaxySeparation = 500.0f;      // Distance between centers
    // Implementation: Needs careful particle initialization
};
```

---

## Implementation Order (Incremental)

### Week 1: Constraint Shapes (Days 1-2)

**Goal:** Get all 4 constraint shapes working with runtime controls

**Day 1:**
- [ ] Add constraint parameters to `ParticleSystem.h`
- [ ] Verify shader already has `ApplyConstraints()` function
- [ ] Add ImGui dropdown for constraint shape selection
- [ ] Add ImGui sliders for radius and thickness
- [ ] Test SPHERE constraint

**Day 2:**
- [ ] Test DISC constraint
- [ ] Test TORUS constraint
- [ ] Test ACCRETION_DISK mode
- [ ] Add hotkeys for constraint controls
- [ ] Verify smooth transitions between shapes

**Success Criteria:**
- Can switch between constraint shapes at runtime
- Particles stay within boundaries
- No crashes or instability
- Smooth parameter adjustments

---

### Week 1: Black Hole Mass (Day 3)

**Goal:** Black hole mass affects orbital velocity

**Tasks:**
- [ ] Add `blackHoleMass` parameter to `ParticleSystem.h`
- [ ] Update Keplerian velocity calculation in shader
- [ ] Implement Paczynski-Wiita potential
- [ ] Add ISCO detection and particle sinking
- [ ] Add ImGui slider (logarithmic scale)
- [ ] Add mass presets (Stellar, Sgr A*, Quasar)
- [ ] Test with different masses (10 M☉ to 1e9 M☉)

**Success Criteria:**
- Higher mass → faster orbital speeds
- Particles spiral inward inside ISCO
- Realistic accretion disk behavior
- Stable across mass range

---

### Week 2: Alpha Viscosity (Day 4)

**Goal:** Particles slowly spiral inward (accretion)

**Tasks:**
- [ ] Add `alphaViscosity` parameter to `ParticleSystem.h`
- [ ] Implement radial drift force in shader
- [ ] Only active in ACCRETION_DISK mode
- [ ] Add ImGui slider (0.0 - 1.0)
- [ ] Test with α = 0.0 (no accretion) vs α = 0.4 (fast accretion)
- [ ] Verify gradual inward motion
- [ ] Ensure disk remains stable

**Success Criteria:**
- Visible inward spiral over time
- Disk doesn't collapse too quickly
- Realistic accretion behavior
- No instabilities

---

### Week 2: Enhanced Temperature (Day 5)

**Goal:** Velocity-based temperature for realistic colors

**Tasks:**
- [ ] Port orbital velocity temperature model from Vulkan
- [ ] Implement Keplerian deviation heating
- [ ] Add viscous dissipation heating
- [ ] Test white-hot core generation
- [ ] Compare with old distance-based model
- [ ] Ensure smooth color transitions

**Success Criteria:**
- More realistic color distribution
- White-hot cores in high-shear regions
- Smooth transitions (no flashing)
- Better matches astrophysics expectations

---

### Week 2: Stubs and Documentation (Day 6-7)

**Tasks:**
- [ ] Add jets stub structure
- [ ] Add dual galaxy stub structure
- [ ] Update all ImGui tooltips
- [ ] Create physics parameter reference doc
- [ ] Test all features together
- [ ] Performance profiling
- [ ] Create demo videos

---

## Runtime Controls Summary

### Existing Controls (Keep)
```
Camera:
  W/A/S/D/Q/E: Move camera
  Mouse: Look around (hold right button)
  [ / ]: Adjust particle size

Physics:
  Up/Down: Gravity strength
  Left/Right: Angular momentum
  Ctrl+Up/Down: Turbulence
  Shift+Up/Down: Damping
```

### New Controls (Add)

```
Constraint System:
  5: Cycle constraint shapes (None→Sphere→Disc→Torus→Accretion Disk)
  [ / ]: Adjust constraint radius
  - / =: Adjust constraint thickness

Black Hole:
  Ctrl+M / Shift+M: Adjust black hole mass (log scale)

Viscosity:
  Ctrl+V / Shift+V: Adjust alpha viscosity

Quick Presets:
  F5: Sgr A* preset (default)
  F6: Stellar Mass BH preset
  F7: Quasar preset
  F8: Custom preset
```

### ImGui Panel Layout

```
Physics Parameters
├─ Constraint Shape: [Dropdown: Accretion Disk v]
│  ├─ Outer Radius: [Slider: 300.0]
│  └─ Thickness: [Slider: 50.0]
├─ Black Hole
│  └─ Mass: [Slider (log): 4.3e6 M☉]
├─ Accretion Physics
│  └─ Alpha Viscosity: [Slider: 0.1]
├─ Forces
│  ├─ Gravity Strength: [Slider: 500.0]
│  ├─ Angular Momentum: [Slider: 1.0]
│  ├─ Turbulence: [Slider: 15.0]
│  └─ Damping: [Slider: 0.99]
└─ Future Features
   ├─ [x] Relativistic Jets (Coming Soon)
   └─ [x] Dual Galaxy Mode (Coming Soon)
```

---

## Testing Strategy

### After Each Feature Addition

1. **Visual Test:**
   - Does it look correct?
   - Are there visual artifacts?
   - Does it match expectations?

2. **Stability Test:**
   - Run for 60 seconds
   - Adjust parameters to extremes
   - Check for crashes/NaN values

3. **Performance Test:**
   - Measure FPS impact
   - Check GPU usage
   - Ensure <5% performance cost per feature

4. **Integration Test:**
   - Test with RT lighting ON
   - Test with Gaussian splatting
   - Test with 10K, 50K, 100K particles

### Final Integration Test

- [ ] All constraint shapes work
- [ ] Black hole mass affects behavior correctly
- [ ] Alpha viscosity causes inward spiral
- [ ] Temperature model produces realistic colors
- [ ] All hotkeys function
- [ ] ImGui controls work
- [ ] No crashes with extreme parameters
- [ ] Performance is acceptable (>30 FPS at 50K particles)

---

## Risk Mitigation

### High Risk: Breaking Existing Physics

**Mitigation:**
- Git commit before each feature
- Test existing modes after each change
- Keep old temperature model as fallback option
- Add "Classic Physics" vs "Enhanced Physics" toggle

### Medium Risk: Performance Degradation

**Mitigation:**
- Profile after each addition
- Use conditional compilation for expensive features
- Add quality presets (Low/Medium/High physics complexity)

### Low Risk: Parameter Instability

**Mitigation:**
- Clamp all input parameters
- Add parameter validation
- Show warnings for extreme values
- Add "Reset to Defaults" button

---

## Success Criteria

### Phase 1 Complete When:
- ✅ All 4 constraint shapes working
- ✅ Runtime adjustable radius/thickness
- ✅ ImGui controls functional
- ✅ No regressions in existing features

### Phase 2 Complete When:
- ✅ Black hole mass affects orbital velocity
- ✅ ISCO physics working
- ✅ Logarithmic mass slider
- ✅ Presets functional

### Phase 3 Complete When:
- ✅ Alpha viscosity causes inward spiral
- ✅ Accretion behavior realistic
- ✅ Stable over long simulations

### Phase 4 Complete When:
- ✅ Velocity-based temperature
- ✅ White-hot cores
- ✅ Smooth color transitions

### Overall Success:
- ✅ All ported features working
- ✅ Performance acceptable
- ✅ No regressions
- ✅ Ready for Phase 3.2 (Splash Screen)

---

## Next Steps

1. **Get user approval** on this plan
2. **Start with Week 1, Day 1** (Constraint Shapes)
3. **Test thoroughly** before moving to next feature
4. **Document progress** in daily logs
5. **Create git commits** after each working feature

---

**Document Status:** Ready for Review
**Last Updated:** 2025-10-15
**Author:** Claude (with user guidance)
