# PINN v3 Testing Instructions - Quick Start

**Date:** 2025-11-27
**Status:** ✅ Fix deployed, ready for testing
**Estimated time:** 15 minutes

---

## 🎯 **What Was Fixed**

The v3 model was outputting **Cartesian forces** (Fx, Fy, Fz) but the C++ code was treating them as **spherical forces** (F_r, F_θ, F_φ) and running them through a coordinate transformation that completely destroyed them.

**Result before fix:**
- Radial expansion instead of orbital rotation
- Forces 50× too weak (0.0007 vs 0.03)
- Coherent translation under turbulence

**Expected after fix:**
- Circular orbits
- Forces 100× stronger (0.01-0.03)
- Proper rotational motion

---

## 🚀 **Quick Test (5 minutes)**

### Step 1: Launch Application

```bash
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean
./build/bin/Debug/PlasmaDX-Clean.exe
```

### Step 2: Check Console Log

**Look for these messages in the first few seconds:**

```
[PINN] Loaded v3 TOTAL FORCES model              ← ✅ v3 model loaded
[PINN v3 DEBUG] RAW ONNX particle[0]: F=(...) mag=0.0099 | r=100.00 F_radial=-0.0098
                                                  ← ✅ Force magnitude ~0.01 (strong!)
                                                  ← ✅ F_radial NEGATIVE (attractive)
```

**Success criteria:**
- ✅ Force magnitude between 0.005 and 0.050 (should be around 0.01-0.02)
- ✅ F_radial is **negative** (e.g., -0.0098, -0.0150)
- ❌ If mag < 0.001 → still broken
- ❌ If F_radial > 0 → catastrophic sign error

### Step 3: Visual Check

**What you should see:**
- Particles forming a **rotating disk** around the black hole
- **Curved trajectories** (not straight radial lines)
- Disk **spinning** (even at time scale 1×, but slow)

**Recommended settings for initial test:**
- Particle count: 2000-5000 (low count for fast rendering)
- Time scale: 1× initially, then try 10-20×
- Turbulence: 0.0 (disabled for clean test)
- Damping: 1.0 (no damping)

**What rotation looks like:**
- At **time scale 1×:** Very slow rotation (might take 30-60 seconds to see clear movement)
- At **time scale 10×:** Should see obvious spinning motion
- At **time scale 50×:** Fast rotation (full disk spins in ~5-10 seconds)

---

## 📊 **Detailed Verification (10 minutes)**

### Test A: Force Magnitude Verification

**Goal:** Confirm forces are 100× stronger than before fix

**Steps:**
1. Watch console log for frame 60 update:
   ```
   [PINN] Frame 60 - Avg force: (...) mag=0.0132 | Max: 0.0520
   ```

2. **Check avg force magnitude:**
   - ✅ **Good:** 0.010 to 0.030 (strong forces, orbital motion possible)
   - ⚠️ **Weak:** 0.001 to 0.005 (marginal, might need higher time scale)
   - ❌ **Broken:** <0.001 (still has bug, coordinate transform still corrupting)

### Test B: Orbital Motion Verification

**Goal:** Confirm particles orbit instead of expanding/collapsing

**Steps:**
1. Set time scale to **10×** (for visible motion)
2. Watch particles for 30 seconds
3. **Look for:**
   - ✅ Disk rotating around black hole (clockwise or counterclockwise)
   - ✅ Particles following **curved arcs** (not straight lines)
   - ✅ Stable orbits (particles stay at roughly same radius)

4. **Red flags:**
   - ❌ Particles moving **radially outward** (expansion) → still broken
   - ❌ Particles **spiraling inward** (collapse) → sign error or too much damping
   - ❌ No motion at all → forces too weak or time scale issue

### Test C: Time Scale Scaling

**Goal:** Verify time scale correctly speeds up motion

**Steps:**
1. Start at time scale **1×**, observe rotation speed (very slow)
2. Increase to **10×**, rotation should be 10× faster
3. Increase to **50×**, rotation should be VERY fast (full rotation in ~5-10 sec)

**Success criteria:**
- ✅ Rotation speed increases proportionally with time scale
- ✅ At 50×, disk spins rapidly but remains stable (no chaos)
- ❌ If no change in speed → time scale not being applied to forces

### Test D: Turbulence Behavior (Optional)

**Goal:** Verify turbulence creates gentle chaos, not coherent translation

**Prerequisites:** Tests A-C passed (orbital motion working)

**Steps:**
1. Set turbulence to **0.1** (low)
2. Watch for 30 seconds
3. **Expected:** Slight perturbations, particles wobble individually
4. Increase turbulence to **0.5** (moderate)
5. **Expected:** More chaotic motion, but still orbiting overall

**Red flags:**
- ❌ Entire cloud moving in same direction (rigid body translation) → still has coherent bias
- ❌ Particles ejected from disk → turbulence too strong (reduce to 0.1-0.2)

---

## ✅ **Success Checklist**

After testing, check off these items:

- [ ] v3 model loaded (log shows "Loaded v3 TOTAL FORCES model")
- [ ] Raw ONNX force magnitude 0.01-0.03 (log shows "mag=0.01...")
- [ ] Radial force component negative (log shows "F_radial=-0.01...")
- [ ] Frame 60 avg force mag ≥ 0.010 (not 0.0007!)
- [ ] Particles form rotating disk (visual)
- [ ] Rotation visible at time scale 10× (smooth spinning)
- [ ] Rotation FAST at time scale 50× (full rotation in seconds)
- [ ] Turbulence creates gentle chaos (not rigid body motion)

**If all ✅ checked:** Fix successful! PINN v3 is fully operational.

**If any ❌:** See troubleshooting section below.

---

## 🐛 **Troubleshooting**

### Problem: Force magnitude still < 0.001

**Possible causes:**
1. **Model file not updated** - old model still deployed
   - Check: `ls -l build/bin/Debug/ml/models/pinn_v3_total_forces.onnx`
   - Should show recent timestamp (today's date)
   - Fix: `cp ml/models/pinn_v3_total_forces.onnx* build/bin/Debug/ml/models/`

2. **Time scale dividing forces** - integration bug
   - Check: Line 709 in `ParticleSystem.cpp`: `m_cpuVelocities[i].x += ax * deltaTime;`
   - Should be `deltaTime = dt * timeScale`, not `deltaTime = dt / timeScale`

3. **Wrong model loaded** - v2 or v1 fallback
   - Check log: Must say "Loaded v3 TOTAL FORCES model"
   - If says v2/v1 → v3 model file missing or corrupted

### Problem: F_radial is positive (repulsive)

**This is CATASTROPHIC - sign error in physics!**

**Check:**
1. Training script `pinn_v3_total_forces.py` line ~100:
   ```python
   F_grav = -GM * M_bh * r_hat / (r_mag**2)  # Should have negative sign!
   ```

2. Integration code `ParticleSystem.cpp` line 709:
   ```cpp
   m_cpuVelocities[i].x += ax * deltaTime;  // Should be +=, not -=
   ```

**If both correct:** Model was trained with wrong sign. Retrain with corrected script.

### Problem: Rotation still not visible at time scale 50×

**Unlikely with fix, but if it happens:**

1. **Verify raw ONNX output is strong:**
   - Check log: "RAW ONNX particle[0]: ... mag=0.01..."
   - If mag < 0.005 → model problem (retrain)
   - If mag > 0.01 → integration/time scale problem

2. **Disable all legacy physics:**
   - Turbulence: 0.0
   - Damping: 1.0
   - Boundary enforcement: off

3. **Check deltaTime calculation:**
   ```cpp
   // In ParticleSystem::UpdatePhysics_PINN()
   // Should be: deltaTime = dt * timeScale
   // NOT: deltaTime = dt / timeScale
   ```

### Problem: Particles expand radially (still broken)

**This means coordinate transformation is STILL happening!**

**Check:**
1. Rebuild completed successfully? (no compilation errors)
2. Running correct executable? (`build/bin/Debug/PlasmaDX-Clean.exe`)
3. Code change applied? Check line 334 in `PINNPhysicsSystem.cpp`:
   ```cpp
   if (m_isV3Model) {  // ← This line should exist
       // v3: Use raw ONNX output...
   ```

**If code correct but still broken:**
- Clean rebuild: `MSBuild.exe build/PlasmaDX-Clean.sln /t:Rebuild`
- Check `m_isV3Model` flag is set correctly (add log in Initialize())

---

## 📝 **Reporting Results**

### If Tests Pass:

**Report back with:**
```
✅ PINN v3 FIX SUCCESSFUL

Force magnitude: 0.0132 (100× stronger!)
F_radial: -0.0098 (negative, attractive)
Visual: Rotating disk, smooth motion at 10× time scale
Turbulence: Gentle chaos (no rigid body motion)

Ready for production use!
```

### If Tests Fail:

**Capture and report:**
1. **Console log** (first 100 lines showing PINN initialization)
2. **Screenshot** of visual behavior
3. **Settings used** (time scale, turbulence, particle count)
4. **Specific symptoms:**
   - Force magnitude observed: ???
   - F_radial sign: positive or negative?
   - Visual behavior: expansion, collapse, rotation, or static?

---

## 🚀 **Next Steps After Successful Test**

### Optional Cleanup:
1. **Remove diagnostic logging** (line 343-360 in `PINNPhysicsSystem.cpp`)
   - Keeps console log clean
   - Reduces per-frame overhead (minimal, but cleaner code)

### Performance Testing:
1. Increase particle count to 10K-50K
2. Measure FPS impact of PINN vs GPU physics
3. Test hybrid mode (PINN for outer disk, GPU for inner disk)

### Legacy Physics Removal:
If v3 works perfectly, consider executing the legacy removal plan:
- See `PINN_COMPREHENSIVE_ANALYSIS.md` Phase 1-4 (4-6 hours)
- Removes ~700 lines of GPU physics shader code
- Simplifies codebase to PINN-only

---

## 📖 **Reference Documents**

- `PINN_V3_COORDINATE_BUG_FIX.md` - Detailed explanation of the bug and fix
- `PINN_COMPREHENSIVE_ANALYSIS.md` - Original investigation that identified this bug
- `PINN_TRAINING_GUIDE.md` - How to retrain v3 model if needed
- `PINN_SESSION_SUMMARY.md` - Training history

---

**Last Updated:** 2025-11-27 03:20 (UTC)
**Estimated Testing Time:** 5-15 minutes
**Success Rate (Expected):** 95% (fix targets root cause)

**GOOD LUCK! 🎉**
