# PINN v3 Testing Guide

**Model:** `pinn_v3_total_forces.onnx` - Outputs TOTAL forces (gravity + viscosity + MRI)

**Status:** ✅ Ready to test!

---

## Quick Test

1. **Run the application:**
   ```bash
   ./build/bin/Debug/PlasmaDX-Clean.exe
   ```

2. **Enable PINN:** Press **P** key

3. **Check the log for v3 confirmation:**
   ```
   [PINN] Loaded v3 TOTAL FORCES model (gravity + viscosity + MRI)
   [PINN] This model outputs gravitational forces directly for proper orbital motion
   ```

4. **Wait ~5 seconds, then check force diagnostics** (logged every 60 frames):
   ```
   [PINN] Frame 60 - Inference: X.XXms | Avg force: (X.XXXX, X.XXXX, X.XXXX) mag=0.0004 | Max: X.XXXX
   ```

---

## Expected Results

### ✅ **GOOD - v3 Working:**
```
Frame 60: Avg force: (-0.0002, 0.0000, -0.0003) mag=0.0004
          → Forces ~0.0004 magnitude (4× larger than v2's 0.0001)
          → Particles orbit smoothly with visible rotation

Frame 120: Avg force: (-0.0003, 0.0001, -0.0002) mag=0.0004
           → Forces remain stable (not spiking)
```

**Visual:** Particles rotate around black hole in smooth circular/elliptical orbits

---

### ❌ **BAD - v2 Loaded Instead:**
```
[PINN] Loaded v2 turbulent model (MRI + Kolmogorov physics)
[WARN] v2 may have rotation issues - recommend using v3

Frame 60: Avg force: (-0.0000, 0.0000, -0.0001) mag=0.0001
          → Forces too small (~0.0001)
          → No visible rotation
```

**Fix:** Check that both `pinn_v3_total_forces.onnx` AND `pinn_v3_total_forces.onnx.data` are in `build/bin/Debug/ml/models/`

---

## Testing Scenarios

### 1. **Baseline Stability (turbulence=0.0, time scale=1.0)**
   - **Expected:** Smooth circular orbits, slow rotation
   - **Force range:** 0.0003-0.0005 magnitude
   - **Check:** Particles stay in disk plane (y ≈ 0)

### 2. **Time Scale Boost (set to 10×)**
   - **Expected:** Rotation 10× faster, orbits still stable
   - **Force range:** Same (0.0003-0.0005)
   - **Check:** No particles flying off

### 3. **Turbulence 0.1-0.3**
   - **Expected:** Gentle chaotic motion, NOT coherent translation
   - **Force range:** 0.0004-0.0006 (slight increase, NOT 1000×!)
   - **Check:** Particles swirl randomly, don't all move in one direction

### 4. **Turbulence 0.5-1.0 (stress test)**
   - **Expected:** More chaotic, but still contained
   - **Force range:** 0.0005-0.0008
   - **Check:** System doesn't explode

---

## Force Diagnostic Interpretation

### What to Look For:

1. **Average Force Magnitude:**
   - **Good:** 0.0003-0.0005 (gravitational forces maintaining orbits)
   - **Bad (v2):** ~0.0001 (too weak for rotation)
   - **Bad (turbulence bug):** >0.01 (coherent translation)

2. **Force Components (Fx, Fy, Fz):**
   - **Good:** All components small (<0.001), no strong bias
   - **Bad:** One component dominates (e.g., avgFy = -0.26 → pushing all particles down)

3. **Max Force:**
   - **Good:** 0.001-0.003 (individual particles near ISCO experience higher forces)
   - **Bad:** >1.0 (particles being violently ejected)

---

## Troubleshooting

### **Issue: "ONNX Runtime exception: model input count mismatch"**
**Cause:** v3 expects 10D input (Cartesian + params), but code sends 7D (spherical)

**Fix:** Update `PINNPhysicsSystem::PredictForcesBatch()` to send:
```cpp
// v3 input format: [x, y, z, vx, vy, vz, t, M_bh, alpha, H_R]
input_data[i*10 + 0] = pos.x;  // x
input_data[i*10 + 1] = pos.y;  // y
input_data[i*10 + 2] = pos.z;  // z
input_data[i*10 + 3] = vel.x;  // vx
input_data[i*10 + 4] = vel.y;  // vy
input_data[i*10 + 5] = vel.z;  // vz
input_data[i*10 + 6] = t;      // time
input_data[i*10 + 7] = M_bh;   // black hole mass
input_data[i*10 + 8] = alpha;  // viscosity
input_data[i*10 + 9] = H_R;    // disk thickness
```

---

### **Issue: Particles still don't rotate**
**Possible causes:**
1. v2 model loaded instead of v3 → Check log for "v3 TOTAL FORCES"
2. PINN not enabled → Press 'P' key
3. Time scale too low → Increase to 5-10×
4. Damping too high → Set damping to 0.999 or 1.0

---

### **Issue: Turbulence causes coherent translation**
**Diagnosis:** Check force logs:
```
Frame 660: Avg force: (0.04, -0.27, 0.11) mag=0.29
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           If magnitude >0.1, this is the old bug
```

**Cause:** v2 model loaded (has the bug) or turbulence implementation issue

**Fix:** Verify v3 is loaded, check that turbulence strength is small (0.001 scale factor)

---

## Success Criteria

**v3 model is working correctly if ALL are true:**

- ✅ Log shows "Loaded v3 TOTAL FORCES model"
- ✅ Average force magnitude ~0.0004 (vs v2's 0.0001)
- ✅ Particles orbit with visible rotation
- ✅ Time scale 1-50× works smoothly
- ✅ Turbulence 0.1-0.5 creates gentle chaos (NOT coherent translation)
- ✅ No force spikes >0.01 when turbulence applied

**If all criteria met:** Proceed to Phase 2 (remove legacy physics) ✅

**If criteria NOT met:** Debug model loading or retrain v3 with different parameters

---

## Next Steps After Successful Test

1. **Screenshot the beautiful orbits!** 📸
2. **Review PINN_DEFAULT_MIGRATION_PLAN.md**
3. **Start Phase 2:** Remove legacy GPU physics (~2-3 hours)
4. **Enjoy simpler, cleaner codebase!** 🎉
