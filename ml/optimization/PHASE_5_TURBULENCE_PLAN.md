# Phase 5: Constrained Turbulence Optimization

## Status: READY TO START (2025-11-30)

**Duration Estimate:** 4-6 hours total
**Prerequisites:** Phase 3 lessons learned, foundation fixes
**Goal:** Optimize turbulence parameters for realistic accretion disk dynamics

---

## Lessons Learned from Phase 3 Failure

### What Went Wrong
- GA predicted fitness 73.79, actual validation 22.0 (70% error!)
- Optimized parameters were 17% WORSE than baseline
- Visual quality collapsed by 69%

### Root Causes Identified
1. ❌ **No particle size awareness** - PINN doesn't know particles are r=20 units
2. ❌ **No settling time** - Benchmarks started with chaotic initial conditions
3. ❌ **Containment too small** - 500 units for 40-unit particles (12.5× not 100×+)
4. ❌ **No equilibrium check** - Optimized for random chaos, not stable orbits

### Critical Fixes Required BEFORE Phase 5

---

## Foundation Fixes (MUST DO FIRST)

### Fix 1: Particle Size Awareness ⏱️ 30 min

**Problem:** PINN input doesn't include particle radius

**Solution:**
```python
# Update PINN input from 7D to 8D
input_features = [
    r, theta, phi,           # Position (spherical)
    v_r, v_theta, v_phi,     # Velocity (spherical)
    particle_radius          # NEW: Physical particle size
]
```

**Impact:** Model can now account for volumetric collisions, pressure

**Files to modify:**
- `ml/pinn_accretion_disk.py` - Update input dimension
- `src/ml/PINNPhysicsSystem.cpp` - Pass particle_radius to inference

---

### Fix 2: Settling Time / Warmup Period ⏱️ 15 min

**Problem:** Benchmarks start before particles settle into orbits

**Solution:**
```cpp
// In BenchmarkRunner.cpp
m_warmupFrames = 200;  // Was 100 - increase to 200
m_measurementStartFrame = 200;  // Don't measure until settled

// Add equilibrium check
bool IsSettled() {
    float energyDrift = abs((currentEnergy - initialEnergy) / initialEnergy);
    float velocityStdDev = ComputeVelocityStdDev();
    return energyDrift < 0.05 && velocityStdDev < 0.1;
}
```

**Impact:** Benchmarks measure stable disk dynamics, not initialization chaos

---

### Fix 3: Boundary/Containment ⏱️ 10 min

**Problem:** 500-unit boundary too small for 20-unit particles

**Options:**

**A) Disable Entirely** (RECOMMENDED for Phase 5)
```json
{
  "physics": {
    "boundary_enabled": false
  }
}
```

**B) Enlarge to 5000+ units**
```json
{
  "physics": {
    "boundary_enabled": true,
    "boundary_radius": 5000.0,
    "boundary_mode": 1  // Reflective
  }
}
```

**Impact:** Particles achieve stable orbits naturally

---

### Fix 4: World Scale Adjustment ⏱️ 5 min

**Problem:** Inner radius (10) < particle diameter (40)

**Solution:**
```json
{
  "physics": {
    "inner_radius": 50.0,    // Was 10.0 - now 2.5× particle radius
    "outer_radius": 1000.0,  // Was 300.0 - now 50× particle radius
    "disk_thickness": 100.0, // Was 50.0 - now 5× particle radius
    "particle_radius": 20.0  // Explicit setting
  }
}
```

**Impact:** Particles have room to breathe, realistic disk geometry

---

## Phase 5: Turbulence Parameters

### Parameters to Optimize (3-5 total)

| Parameter | Range | Current | Purpose |
|-----------|-------|---------|---------|
| `alpha_viscosity` | 0.01-1.0 | 0.1 | Shakura-Sunyaev viscosity |
| `turbulence_intensity` | 0.0-2.0 | 0.0 | Vorticity strength |
| `turbulence_scale` | 10-200 | 50 | Eddy size (units) |
| `turbulence_frequency` | 0.1-10.0 | 1.0 | Temporal variation (Hz) |
| `mri_strength` | 0.0-5.0 | 1.0 | Magnetorotational instability |

**Why fewer parameters?**
- Easier to optimize (3D-5D vs 12D space)
- Less prone to overfitting
- Turbulence is inherently stochastic → needs robust parameters

---

## Phase 5 Implementation Steps

### Step 1: Foundation Fixes (1 hour)
1. ✅ Add `particle_radius` to PINN input
2. ✅ Increase warmup frames to 200
3. ✅ Disable boundary (or set to 5000)
4. ✅ Increase world scale (50/1000 instead of 10/300)
5. ✅ Test baseline with fixes

### Step 2: Add Turbulence to Physics (1.5 hours)
1. ✅ Extend particle_physics.hlsl with turbulence forces
2. ✅ Add SIREN vortex field sampling
3. ✅ Implement MRI (magnetorotational instability) term
4. ✅ Add turbulence parameters to cbuffer
5. ✅ Test visual appearance

### Step 3: Extend GA for Turbulence (30 min)
1. ✅ Add 3-5 turbulence parameters to genetic_optimizer
2. ✅ Update parameter bounds
3. ✅ Keep population small (20-30) for speed
4. ✅ Reduce generations (20-30) for iteration speed

### Step 4: Run Optimization (1-2 hours compute)
1. ✅ Quick test (5 pop × 5 gen = 25 evals, ~10 min)
2. ✅ Full run (30 pop × 30 gen = 900 evals, ~90 min)
3. ✅ Validate best parameters
4. ✅ Visual comparison

### Step 5: Validation (1 hour)
1. ✅ Benchmark baseline (no turbulence)
2. ✅ Benchmark optimized (with turbulence)
3. ✅ Visual screenshots
4. ✅ Document results

---

## Fitness Function for Phase 5

**Updated weights (emphasize visual quality):**

```python
def compute_fitness_turbulence(results):
    summary = results.get('summary', {})

    # Phase 5: Turbulence focuses on visual realism
    stability = summary.get('stability_score', 0.0)
    accuracy = summary.get('accuracy_score', 0.0)
    performance = summary.get('performance_score', 0.0)
    visual = summary.get('visual_score', 50.0)

    # NEW WEIGHTS - visual quality matters more for turbulence
    fitness = (
        0.25 * stability +      # Was 0.35 - still important but less critical
        0.15 * accuracy +       # Was 0.30 - turbulence is inherently chaotic
        0.20 * performance +    # Same - still need 90+ FPS
        0.40 * visual           # Was 0.15 - KEY METRIC for turbulence!
    )

    # Bonus for realistic turbulent features
    coherent_motion = results.get('visual_quality', {}).get('coherent_motion_index', 0.0)
    velocity_jerk = results.get('visual_quality', {}).get('velocity_jerk', 0.0)

    # Reward moderate jerk (turbulence) but not chaos
    if 0.1 < velocity_jerk < 0.5:  # Sweet spot
        fitness += 5.0

    # Reward coherent but dynamic motion
    if 0.3 < coherent_motion < 0.7:  # Not too ordered, not too chaotic
        fitness += 5.0

    return fitness
```

**Key change:** Visual quality weight increased from 15% → 40% because turbulence is all about APPEARANCE!

---

## Success Criteria

### Must Have ✅
- [ ] Visual quality > 50/100 (was 13.2 in failed GA)
- [ ] Particle retention > 90% (escape rate < 10%)
- [ ] FPS > 120 @ 10K particles
- [ ] Realistic swirls/eddies visible in rendering
- [ ] Validated fitness within 10% of GA prediction

### Nice to Have 🎯
- [ ] Energy conservation < 20% drift (realistic for turbulent disk)
- [ ] Coherent motion index 0.3-0.7 (structured chaos)
- [ ] Temporal stability (not flickering)

---

## Dual Machine Strategy (Simple Approach)

### Machine 1 (5950x - 32 threads) - Primary Optimization
```bash
cd ml/optimization
python genetic_optimizer_parallel.py \
    --workers 28 \
    --population 30 \
    --generations 30 \
    --output results/turbulence_5950x.json
```

### Machine 2 (6900k - 20 threads) - Visual Experiments
```bash
# Test different turbulence intensities manually
for intensity in 0.0 0.5 1.0 1.5 2.0; do
    ./PlasmaDX-Clean.exe --turbulence-intensity $intensity --capture-screenshots
done
```

**OR** run a shorter, wider search:
```bash
python genetic_optimizer_parallel.py \
    --workers 18 \
    --population 50 \
    --generations 10 \
    --output results/turbulence_6900k_wide.json
```

**No network setup needed** - just copy results to one machine for analysis!

---

## Timeline

| Task | Duration | Machine |
|------|----------|---------|
| Foundation fixes | 1 hour | Dev work |
| Turbulence physics | 1.5 hours | Dev work |
| GA setup | 30 min | Dev work |
| Quick test | 10 min | 5950x |
| Full optimization | 90 min | 5950x |
| Visual experiments | 90 min | 6900k (parallel!) |
| Validation | 1 hour | Either |
| **Total** | **4-6 hours** | Both machines |

**With both machines:** Effectively 3-4 hours of YOUR time (rest is compute)

---

## Next Steps (In Order)

1. **Document GA failure** (this file) ✅ DONE
2. **Fix particle size awareness** (modify PINN input)
3. **Increase settling time** (warmup frames 100→200)
4. **Disable boundary** (or set to 5000)
5. **Add turbulence to physics shader**
6. **Extend GA optimizer**
7. **Run optimization** (both machines!)
8. **Validate and celebrate** 🎉

---

## Files to Modify

### C++ Code
- `src/benchmark/BenchmarkRunner.cpp` - Settling time, boundary settings
- `src/particles/ParticleSystem.h` - Add particle_radius parameter
- `src/ml/PINNPhysicsSystem.cpp` - Pass particle_radius to PINN

### Shaders
- `shaders/particles/particle_physics.hlsl` - Add turbulence forces

### Python
- `ml/pinn_accretion_disk.py` - Update input dimension (7D→8D)
- `ml/optimization/genetic_optimizer_parallel.py` - Add turbulence parameters

### Configs
- `configs/user/default.json` - Update physics defaults
- `configs/scenarios/turbulence_test.json` - New test scenario

---

## Key Takeaway

**Phase 3 failed because we optimized BEFORE fixing the foundation.**

**Phase 5 will succeed because we're fixing the foundation FIRST, THEN optimizing!**

Foundation → Turbulence → Optimization → Success! 🚀
