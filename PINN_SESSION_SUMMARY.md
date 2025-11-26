# PINN Physics System - Session Summary

**Date:** 2025-11-26
**Status:** ✅ v3 model RETRAINED with GM=100 (100× stronger forces) - **READY FOR TESTING**

---

## 🚀 **CONTINUATION SESSION (2025-11-26 21:19-21:28) - CRITICAL FIX APPLIED**

### ✅ Successfully Retrained v3 with GM=100.0

**Actions Completed:**
1. **Fixed normalization** in `pinn_v3_total_forces.py` line 29:
   - Changed `GM = 1.0` → `GM = 100.0` (100× stronger gravitational forces)
2. **Regenerated training data** with corrected physics:
   - 100K samples with avg force magnitude **0.0329** (vs previous 0.0003)
   - Force ranges: Fx=[-1.16, 1.13], Fy=[-0.27, 0.25], Fz=[-1.19, 1.19]
3. **Trained new v3 model** (200 epochs, ~9 minutes on GPU):
   - Final loss: **0.000073** (excellent convergence)
   - Model saved: `ml/models/pinn_v3_total_forces.onnx` (265KB + 3KB metadata)
4. **Deployed to build directory**:
   - Copied to `build/bin/Debug/ml/models/pinn_v3_total_forces.*`
   - Application will auto-load on next run

**Expected Results:**
- Force magnitude: **~0.003-0.005** (100× stronger than broken v3)
- Visible orbital rotation at time scale 1×
- Fast rotation at time scale 10-20×
- Turbulence should create gentle chaos (not coherent translation)

**Next Step:** Test the application and verify force diagnostics!

---

## 🎯 **Original Session Goal (Achieved)**
Successfully diagnosed root cause of PINN physics issues and created v3 model infrastructure.

---

## 📊 **Key Findings from Force Diagnostics**

### v2 Turbulent Model (broken):
```
Frame 60: Avg force mag=0.0001 (too small → no rotation)
Frame 660 (turbulence): mag=0.1192 (1000× spike → coherent translation)
```

### v3 Total Forces Model (loads but broken):
```
Frame 60: Avg force mag=0.0000 (STILL too small!)
Frame 1140 (turbulence): mag=0.0005 (slight increase, but baseline still ~zero)
```

**Root Cause:** Training data generation has forces that are too small due to normalization.

---

## 🔧 **What Was Fixed This Session**

### 1. **Diagnosed v2 Model Issues**
   - v2 outputs near-zero forces (mag ~0.0001)
   - Turbulence causes 1000× force spike with coherent bias
   - Problem: Training data had `F_net = F_gravity + F_centrifugal ≈ 0` for Keplerian orbits

### 2. **Created PINN v3 Training Script**
   - **File:** `pinn_v3_total_forces.py`
   - **Fix attempt:** Output TOTAL gravitational forces, not net forces
   - **Training:** 200 epochs, final loss 0.000001 (excellent convergence)
   - **Model:** 67,843 parameters, 10D Cartesian input

### 3. **Updated C++ Integration**
   - Added v3 model detection (10D input vs 7D)
   - Support Cartesian coordinates: `(x, y, z, vx, vy, vz, t, M_bh, α, H/R)`
   - Falls back to v2 if v3 unavailable
   - Force diagnostics every 60 frames

### 4. **Removed Broken Features**
   - ✅ Removed velocity multiplier (caused exponential growth)
   - ✅ Increased time scale to 50× (was clamped at 10×)
   - ✅ Reduced debug logging (was 100K+ lines)

---

## ❌ **Remaining Problem: v3 Forces Still Near-Zero**

**Expected:** Force magnitude ~0.0004 (from training data stats)
**Actual:** Force magnitude ~0.0000 (model outputs zeros)

### Why This Happened:

**Training data issue:** Gravitational forces at r=100 in normalized units (GM=1):
```python
F_gravity = -GM / r² = -1.0 / (100²) = -0.0001  # Too small!
```

**The model learned correctly but the training data had forces that are too weak to maintain visible orbits.**

---

## 🚀 **Solution for Next Session**

### **Option 1: Retrain v3 with Stronger Forces (RECOMMENDED)**

**Change normalization to make forces ~100× larger:**

```python
# ml/pinn_v3_total_forces.py - Line ~30
GM = 100.0  # Was 1.0 - makes forces 100× stronger
R_ISCO = 6.0  # Keep same

# Training data generation - Line ~340
F_grav = -GM * M_bh * r_hat / (r_mag**2)
# Now for r=100: F = -100 / 10000 = -0.01 (100× larger!)
```

**Expected result:**
- Force magnitude: ~0.004 (visible orbital motion)
- Time scale 10× should show fast rotation
- Turbulence won't need coherent translation fix

### **Option 2: Scale Forces During Inference**

Add force scaling in `PINNPhysicsSystem::PredictForcesBatch()`:

```cpp
// After ONNX inference, scale forces 100×
for (uint32_t i = 0; i < pinnParticleCount; i++) {
    float fx = outputData[i * 3 + 0] * 100.0f;  // Scale up
    float fy = outputData[i * 3 + 1] * 100.0f;
    float fz = outputData[i * 3 + 2] * 100.0f;
    // Convert to Cartesian and store...
}
```

**Pros:** Quick fix, no retraining
**Cons:** Hacky, doesn't fix underlying physics issue

---

## 📋 **Quick Start for Next Session**

### **Test Current v3 Model (confirm it's broken):**
```bash
./build/bin/Debug/PlasmaDX-Clean.exe
# Press 'P' to enable PINN
# Check log: Should see "Loaded v3 TOTAL FORCES model"
# Check forces: Will show mag=0.0000 (broken)
```

### **Retrain v3 with GM=100:**
```bash
# Edit ml/pinn_v3_total_forces.py line ~30: GM = 100.0
ml/venv/bin/python3 ./pinn_v3_total_forces.py --generate-data
ml/venv/bin/python3 ./pinn_v3_total_forces.py --epochs 200

# Copy new model
cp ml/models/pinn_v3_total_forces.onnx* build/bin/Debug/ml/models/

# Test - should now see mag ~0.004
```

### **OR Apply 100× Scaling Hack:**
Edit `src/ml/PINNPhysicsSystem.cpp` line ~330 (force output loop) and multiply by 100.

---

## 📁 **Key Files Modified This Session**

### Created:
- `pinn_v3_total_forces.py` - v3 training script (needs GM fix)
- `PINN_V3_TEST_GUIDE.md` - Testing procedures
- `PINN_DEFAULT_MIGRATION_PLAN.md` - Roadmap to remove legacy physics

### Modified:
- `src/particles/ParticleSystem.cpp` - Model loader, removed velocity multiplier, added diagnostics
- `src/particles/ParticleSystem.h` - Increased time scale to 50×
- `src/ml/PINNPhysicsSystem.cpp` - v3 detection, 10D Cartesian input support
- `src/ml/PINNPhysicsSystem.h` - Added m_isV3Model flag
- `src/core/Application.cpp` - Removed velocity multiplier UI, updated tooltips

---

## 🐛 **Active Bugs**

### 1. **v3 Forces Near-Zero (CRITICAL)**
   - **Symptom:** Force mag ~0.0000 instead of ~0.004
   - **Fix:** Retrain with GM=100 or scale output 100×

### 2. **Turbulence Coherent Translation**
   - **Symptom:** All particles move in same direction when turbulence applied
   - **Cause:** Random perturbations might have bias OR PINN interprets perturbed states as "restore to training distribution"
   - **Status:** Likely fixes itself when v3 forces are corrected

### 3. **Slow Rotation Even at 50× Time Scale**
   - **Cause:** Forces too weak (see bug #1)
   - **Status:** Will fix with GM=100 retraining

---

## ✅ **Success Criteria (for next session)**

After retraining v3 with GM=100:

1. ✅ Log shows "Loaded v3 TOTAL FORCES model"
2. ✅ Force magnitude **~0.003-0.005** (vs current 0.0000)
3. ✅ **Visible rotation** at time scale 1×
4. ✅ **Fast rotation** at time scale 10-20×
5. ✅ Turbulence 0.1-0.5 creates **gentle chaos** (not coherent translation)
6. ✅ Forces stable (no spikes >0.01)

---

## 🎓 **Lessons Learned**

### 1. **Normalized Units Must Match Physics Scale**
   - GM=1 made forces too weak (0.0001 at r=100)
   - Need GM=100 for visible orbital forces (~0.01)

### 2. **Training Loss Can Be Deceptive**
   - v3 converged perfectly (loss=0.000001)
   - But learned to output near-zero forces because training data had near-zero forces
   - **Lesson:** Always validate training data physics before training

### 3. **Force Diagnostics Are Essential**
   - Without logging avg force magnitude every 60 frames, we'd never have caught this
   - Keep diagnostics in production code

### 4. **Velocity Multiplier Was Fundamentally Broken**
   - Multiplying velocity every frame → exponential growth
   - No correct way to apply it without breaking physics
   - Solution: Use time scale instead

---

## 📝 **Notes for Future Work**

### After v3 Works:
1. **Remove legacy GPU physics** (~700 lines, see PINN_DEFAULT_MIGRATION_PLAN.md)
2. **Add SIREN vortex field** (visible turbulent swirls) - ml/vortex_field/
3. **Implement adaptive PINN** (higher resolution near ISCO, coarse far away)
4. **Train on full GR metrics** (current uses simplified Schwarzschild)

### Alternative Approaches to Consider:
- **Hamiltonian Neural Networks** (energy-preserving by design)
- **Lagrangian particle dynamics** (exact conservation laws)
- **Hybrid: PINN for viscosity, analytic for gravity** (best of both worlds)

---

## 🔗 **Reference Documentation**

- `MASTER_ROADMAP_V2.md` - Overall project roadmap
- `PINN_IMPLEMENTATION_SUMMARY.md` - Original PINN architecture
- `ml/PINN_README.md` - Training guide
- `PINN_V3_TEST_GUIDE.md` - Testing procedures

---

## 🚨 **Critical Action Items**

1. **IMMEDIATE:** Retrain v3 with `GM = 100.0` in pinn_v3_total_forces.py
2. **Verify:** Force magnitude ~0.003-0.005 (100× current values)
3. **Test:** Visible rotation at time scale 1×, fast rotation at 10-20×
4. **If successful:** Remove legacy GPU physics (Phase 2 of migration plan)

---

**Last Updated:** 2025-11-26 21:10
**Next Session Goal:** Retrain v3 with GM=100, verify orbital motion works
**Estimated Time:** 30 min training + 15 min testing = **45 min total**
