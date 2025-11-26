# PINN-Default Migration Plan

**Goal:** Make PINN the primary physics engine, remove legacy GPU physics complexity

**Status:** v3 model training in progress (ETA: 20-30 min)

---

## Phase 1: PINN v3 Model Integration ✅ IN PROGRESS

### Step 1.1: Train v3 Model with Total Forces
- ✅ Created `pinn_v3_total_forces.py`
- ✅ Generated 100K training samples (gravity + viscosity + MRI)
- 🔄 Training 200 epochs (~20-30 min)

**Key Fix:** Outputs TOTAL gravitational forces, not net forces after centrifugal cancellation

### Step 1.2: Test v3 Model
```bash
# After training completes:
./build/bin/Debug/PlasmaDX-Clean.exe --use-pinn 1

# Check logs for force diagnostics:
# Should see: Avg force mag ~0.0004 (vs old 0.0001)
# Should see: Particles orbit smoothly without external forces
```

---

## Phase 2: Simplify Physics Architecture (2-3 hours)

### Files to Modify:

#### **src/particles/ParticleSystem.h**
Changes:
1. Remove `m_usePINN` toggle → always use PINN
2. Remove `m_computePSO`, `m_computeRootSignature` (GPU physics pipeline)
3. Remove `m_pinnVelocityMultiplier` (broken concept)
4. Keep: `m_pinnTurbulence`, `m_pinnDamping`, `m_pinnEnforceBoundaries`
5. Simplify to single physics path: `Update()` → `UpdatePhysics_PINN()`

```cpp
// OLD (hybrid):
if (m_usePINN && m_pinnPhysics) {
    UpdatePhysics_PINN();
} else {
    UpdatePhysics_GPU();
}

// NEW (PINN-only):
UpdatePhysics_PINN();  // Always use PINN
```

#### **src/particles/ParticleSystem.cpp**
Remove:
- `UpdatePhysics_GPU()` function (~80 lines)
- `InitializeComputePipeline()` (~100 lines)
- GPU shader loading logic
- All `m_computePSO` usage

Keep:
- `InitializeAccretionDisk_CPU()` ✅ (already working)
- `UpdatePhysics_PINN()` ✅ (core physics)
- `IntegrateForces()` ✅ (clean Verlet integration)

#### **src/core/Application.cpp**
Remove UI:
- "Use PINN Physics (P)" toggle
- "Velocity Multiplier" slider (already removed)
- Legacy black hole physics controls when PINN active

Keep UI:
- Time Scale (0-50×) ✅
- Turbulence (0-1.0) ✅
- Damping (0.9-1.0) ✅
- Enforce Boundaries checkbox ✅

#### **src/ml/PINNPhysicsSystem.cpp**
Changes:
- Update model loader to try v3 first:
  ```cpp
  if (Initialize("ml/models/pinn_v3_total_forces.onnx")) {
      LOG_INFO("[PINN] Loaded v3 model (total forces: gravity + viscosity + MRI)");
  } else if (Initialize("ml/models/pinn_v2_turbulent.onnx")) {
      // Fallback
  }
  ```

---

## Phase 3: Remove Legacy Shaders (30 min)

### Files to Delete:
- `shaders/particles/particle_physics.hlsl` (legacy GPU physics)
- Corresponding `.dxil` compiled shaders

### CMakeLists.txt Changes:
- Remove shader compilation target for `particle_physics.hlsl`

---

## Phase 4: Add Launch Flags (optional, 30 min)

For backwards compatibility testing:

```cpp
// Application.cpp
bool useLegacyPhysics = false;
for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--legacy-physics") == 0) {
        useLegacyPhysics = true;
    }
}
```

---

## Phase 5: Testing & Validation (1-2 hours)

### Test Cases:

1. **Baseline Stability**
   - Load 10K particles
   - Let run for 60 seconds
   - Check: Particles maintain stable orbits
   - Check: No NaN/Inf in force logs

2. **Time Scale Scaling**
   - Test 1×, 5×, 10×, 20×, 50× speeds
   - Check: Rotation speeds scale linearly
   - Check: No instabilities at high speed

3. **Turbulence Response**
   - Apply 0.1, 0.3, 0.5, 1.0 turbulence
   - Check: Gentle chaotic motion (not coherent translation)
   - Check: Particles don't scatter instantly

4. **Force Diagnostics**
   - Monitor every 60 frames
   - Check: Average force ~0.0003-0.0005
   - Check: No large bias (|avgFx|, |avgFy|, |avgFz| < 0.01)

5. **Performance**
   - Measure FPS @ 10K, 50K, 100K particles
   - Target: 120+ FPS @ 10K (same as before)

---

## Expected Outcomes

### Before (v2 turbulent model):
```
Frame 60: Avg force: (-0.0000, 0.0000, -0.0001) mag=0.0001
          ❌ No rotation (forces too small)
Frame 660 (turbulence): Avg force: (0.0226, -0.1170, -0.0031) mag=0.1192
                        ❌ Coherent translation (1000× force spike)
```

### After (v3 total forces model):
```
Frame 60: Avg force: (-0.0002, 0.0000, -0.0003) mag=0.0004
          ✅ Smooth orbital rotation
Frame 660 (turbulence): Avg force: (-0.0003, 0.0001, -0.0004) mag=0.0005
                        ✅ Gentle chaotic motion (forces stable)
```

---

## Rollback Plan

If v3 has issues:
1. Keep v2 turbulent model as fallback in model loader
2. Add `--use-pinn-v2` flag
3. Restore velocity multiplier temporarily (with fixes)

---

## Timeline

| Phase | Task | Duration | Status |
|-------|------|----------|--------|
| 1.1 | Train v3 model | 20-30 min | 🔄 In Progress |
| 1.2 | Test v3 in app | 15 min | ⏳ Pending |
| 2 | Simplify architecture | 2-3 hours | ⏳ Pending |
| 3 | Remove legacy shaders | 30 min | ⏳ Pending |
| 4 | Add launch flags | 30 min | ⏳ Pending |
| 5 | Testing & validation | 1-2 hours | ⏳ Pending |
| **TOTAL** | | **5-7 hours** | |

---

## Code Removal Summary

### Lines to Remove (~700 total):
- `ParticleSystem.cpp`: ~200 lines (GPU physics)
- `ParticleSystem.h`: ~50 lines (GPU pipeline vars)
- `Application.cpp`: ~100 lines (legacy UI)
- `particle_physics.hlsl`: ~350 lines (GPU shader)

### Lines to Keep (~800 total):
- PINN integration: ~400 lines ✅
- CPU initialization: ~200 lines ✅
- Integration/damping: ~100 lines ✅
- UI (time scale, turbulence): ~100 lines ✅

**Net Result:** Simpler, cleaner codebase with ~100 fewer lines!

---

## Next Steps (After Training)

1. **Check training progress:** `tail -f ml/training_log_v3.txt`
2. **When complete:** Copy model to build directory
   ```bash
   cp ml/models/pinn_v3_total_forces.onnx build/bin/Debug/ml/models/
   ```
3. **Test in app:** Enable PINN (P key) and check force diagnostics
4. **If successful:** Proceed with Phase 2 migration
5. **If issues:** Debug model output, retrain if needed

---

**Expected v3 Benefits:**
- ✅ Smooth orbital rotation (PINN outputs proper gravitational forces)
- ✅ Turbulence works without coherent translation
- ✅ Time scale controls speed correctly (1-50×)
- ✅ Physically accurate (gravity + viscosity + MRI)
- ✅ Simpler integration (no velocity multiplier hacks)
