# RTXDI Visual Difference Validation Test Plan

**Date**: 2025-10-19
**Build**: 0.7.7 + RTXDI lighting scale fix
**Test Duration**: 15 minutes
**Objective**: Validate RTXDI looks visually different from multi-light mode

---

## What Changed

**File Modified**: `shaders/particles/particle_gaussian_raytrace.hlsl` (line 553)

**Change Summary**:
```hlsl
// BEFORE (INCORRECT):
illumination += totalLighting * 10.0;  // Applied to BOTH modes

// AFTER (CORRECT):
if (useRTXDI != 0) {
    illumination += totalLighting;       // RTXDI: NO 10× boost
} else {
    illumination += totalLighting * 10.0;  // Multi-light: Keep 10× boost
}
```

**Why This Fixes the "Identical Visuals" Problem**:
- Multi-light mode: 13 lights × 10× multiplier = ~130× brightness boost
- RTXDI mode (before): 1 light × 10× multiplier = ~10× brightness boost → accidentally looked similar
- RTXDI mode (after): 1 light × 1× (no boost) = ~1× → now **obviously dimmer**

**Expected Result**: RTXDI will be **~10× darker** than multi-light mode.

---

## Test Steps

### Step 1: Launch Application in Multi-Light Mode

**Command**:
```powershell
cd D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean
.\build\Debug\PlasmaDX-Clean.exe
```

**Expected State**:
- Default lighting system: Multi-light (13 lights brute force)
- ImGui shows "Multi-Light: All 13 lights evaluated"
- Bottom-left corner: "Lighting System (F3 to toggle)"

**Visual Baseline**:
- Note overall brightness (bright, warm accretion disk)
- Note illumination uniformity (smooth, all lights contribute)
- Note any visible light sources (rim lighting from multiple directions)

**Take Screenshot**: `screenshots/test_2025-10-19_multi-light_baseline.png`

**FPS Baseline**: Record FPS (should be ~20 FPS @ 10K particles)

---

### Step 2: Switch to RTXDI Mode

**Keyboard Shortcut**: Press **F3**

**OR**

**ImGui Control**:
1. Open "Rendering Features" section
2. Find "Lighting System (F3 to toggle)"
3. Click radio button: "RTXDI (1 sampled light per pixel)"

**Expected State**:
- ImGui shows "RTXDI: Weighted reservoir sampling"
- Log file should show: `"Switched to RTXDI system"`

**Expected Visual Change** (CRITICAL):
- **Brightness**: RTXDI should be **~10× DIMMER** than multi-light
- **Spatial Variation**: May see slight variation across pixels (different lights selected)
- **Temporal Variation**: May see slight flicker each frame (random selection changes)
- **Overall**: Should look **OBVIOUSLY DIFFERENT** from multi-light

**Take Screenshot**: `screenshots/test_2025-10-19_rtxdi_after_fix.png`

**FPS Check**: Record FPS (should be similar to multi-light, ~20 FPS)

---

### Step 3: Toggle Between Modes (Rapid Comparison)

**Action**: Press **F3** repeatedly to toggle between modes

**Observations to Make**:
1. **Is the visual difference obvious?** (MUST BE YES)
2. **Is RTXDI dimmer than multi-light?** (MUST BE YES)
3. **Does the toggle work smoothly?** (no crashes, no delays)
4. **Do the ImGui labels update correctly?** (Multi-Light ↔ RTXDI)
5. **Is there any temporal flicker in RTXDI?** (expected - different light each frame)

**Success Criteria**:
- ✅ Visual difference is **obvious** (user can immediately tell modes apart)
- ✅ RTXDI is **dimmer** than multi-light (as expected from 1 light vs 13 lights)
- ✅ No crashes or rendering artifacts
- ✅ Smooth transition between modes

---

### Step 4: Validate Log Messages

**Log File Location**: `logs/PlasmaDX-Clean_YYYYMMDD_HHMMSS.log`

**Check for**:
1. **Mode Switch Messages**:
   ```
   [INFO] Switched to RTXDI system
   [INFO] Switched to Multi-Light system
   ```

2. **RTXDI DispatchRays Execution** (when in RTXDI mode):
   ```
   [INFO] RTXDI DispatchRays executed (1920x1080, frame XXXX)
   ```

3. **Render Constants Upload** (when in RTXDI mode):
   ```
   [INFO] Gaussian Render Constants:
   [INFO]   useRTXDI: 1
   ```

**Command to Check Logs**:
```powershell
# Find latest log file
Get-ChildItem logs\*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Get-Content | Select-String "RTXDI|Multi-Light"
```

---

### Step 5: Optional - PIX Capture Validation

**If you have PIX installed and want to validate GPU state:**

**Capture Command**:
```powershell
.\build\DebugPIX\PlasmaDX-Clean-PIX.exe
# Wait for auto-capture at frame 120
```

**Or Manual Capture**:
1. Launch PIX for Windows
2. Attach to `PlasmaDX-Clean.exe`
3. Press F12 to capture frame
4. Save to `PIX/Captures/RTXDI_visual_fix_validation.wpix`

**PIX Validation Checklist**:
- [ ] RTXDI output buffer (u0) populated with light indices in R channel
- [ ] Gaussian shader reads `g_rtxdiOutput` texture (t6)
- [ ] Render constants show `useRTXDI = 1` when in RTXDI mode
- [ ] Lighting calculation uses `if (useRTXDI != 0)` branch
- [ ] No 10× multiplier applied in RTXDI branch (line 558)

---

## Expected Results Summary

| Test Case | Expected Behavior | Pass/Fail |
|-----------|-------------------|-----------|
| **Visual Difference** | RTXDI looks **obviously different** from multi-light | [ ] |
| **Brightness** | RTXDI is **~10× dimmer** than multi-light | [ ] |
| **Smooth Toggle** | F3 key switches modes without crashes | [ ] |
| **ImGui Labels** | UI shows correct mode name | [ ] |
| **Log Messages** | "Switched to RTXDI system" appears | [ ] |
| **FPS Stable** | Both modes run at similar FPS (~20 FPS) | [ ] |
| **Spatial Variation** | RTXDI may show pixel-to-pixel variation (different lights) | [ ] |
| **Temporal Variation** | RTXDI may show slight frame-to-frame flicker (random selection) | [ ] |

---

## What If RTXDI Still Looks Identical?

**This should NOT happen** after the fix, but if it does:

### Debug Steps

1. **Check Build Timestamp**:
   ```powershell
   Get-Item build\Debug\PlasmaDX-Clean.exe | Select-Object LastWriteTime
   Get-Item build\Debug\shaders\particles\particle_gaussian_raytrace.dxil | Select-Object LastWriteTime
   ```
   - Shader should be newer than 2025-10-19 (today)
   - If not, shader didn't rebuild → manually delete `.dxil` and rebuild

2. **Verify useRTXDI Flag**:
   - Check log file for `useRTXDI: 1` when in RTXDI mode
   - If always 0, the mode switch isn't working

3. **Check for Multiple Issues**:
   - Maybe light grid is broken (all cells empty → RTXDI selects 0xFFFFFFFF → falls back to multi-light)
   - Maybe RTXDI output buffer isn't bound correctly (shader reads garbage)

4. **Deploy Autonomous Debugging**:
   ```
   Use MCP pix-debug tools:
   mcp__pix-debug__diagnose_visual_artifact symptom="RTXDI still looks identical to multi-light after lighting scale fix"
   ```

---

## What If RTXDI Looks TOO Dark?

**This IS expected** after the fix, but if it's unacceptably dark:

### Possible Causes

1. **Missing Unbiased Weight Correction**:
   - Current implementation doesn't use ReSTIR's `W = weightSum / M` correction
   - This means selected lights contribute with full weight (no bias correction)
   - Result: May be dimmer than ideal

2. **Sparse Light Grid**:
   - Only 0.5% of cells populated (152 out of 27,000)
   - Many pixels may select 0xFFFFFFFF (no light) → fall back to ambient only
   - Result: Large dark regions

3. **Weak Default Lights**:
   - Even after M3 boost (15× radius increase), lights may be too weak
   - RTXDI selects 1 light → if that light is weak, result is dim

### Solutions (for Future Session)

**Short-term**:
1. Boost light intensity in ImGui (increase from 0.5 → 2.0)
2. Increase light radius (increase from 75 → 150 units)
3. Add ambient lighting boost for RTXDI mode

**Long-term** (CORRECT FIX):
1. Implement ReSTIR unbiased weight correction (`W = weightSum / M`)
2. Store `W` in RTXDI output buffer (G channel)
3. Apply `W` when calculating light contribution in Gaussian shader
4. This will make RTXDI brightness **mathematically correct**

**See RTXDI_VALIDATION_ANALYSIS.md Option 2 for full implementation plan.**

---

## Success Criteria

**Test PASSES if**:
- ✅ RTXDI looks **visually different** from multi-light (user can tell modes apart)
- ✅ RTXDI is **dimmer** than multi-light (expected from 1 light vs 13 lights)
- ✅ No crashes or artifacts when switching modes
- ✅ Log messages confirm RTXDI is executing

**Test FAILS if**:
- ❌ RTXDI still looks **identical** to multi-light (original problem)
- ❌ Application crashes when switching modes
- ❌ Visual artifacts appear in RTXDI mode (black screen, garbage pixels)

---

## Next Steps After Validation

### If Test PASSES (RTXDI looks different)

**Immediate Actions**:
1. ✅ Close this GitHub issue / user report (visual difference now obvious)
2. ✅ Document the fix in CLAUDE.md
3. ✅ Commit changes:
   ```bash
   git add shaders/particles/particle_gaussian_raytrace.hlsl
   git commit -m "fix: Remove 10× brightness boost from RTXDI mode

   RTXDI mode was accidentally inheriting the 10× multiplier designed for
   multi-light mode (13 lights). This made 1 selected light look identical
   to 13 accumulated lights.

   Fix: Apply 10× multiplier only to multi-light mode, not RTXDI.

   Expected result: RTXDI is now ~10× dimmer than multi-light (CORRECT).

   Future: Implement ReSTIR unbiased weight correction (W = weightSum / M)
   to achieve mathematically correct brightness.

   Fixes: Visual identity crisis reported 2025-10-19"
   ```

**Future Work** (Next Session):
1. Implement ReSTIR unbiased weight correction (Option 2 from analysis)
2. Add temporal reuse (Phase 2)
3. Add spatial reuse (Phase 3)
4. Performance profiling (ensure <1ms overhead)

---

### If Test FAILS (RTXDI still looks identical)

**Immediate Actions**:
1. Check shader rebuild timestamp (may not have compiled)
2. Check `useRTXDI` flag in logs (may be stuck at 0)
3. Deploy autonomous debugging:
   - `@pix-debugger-v3 analyze PIX/Captures/RTXDI_visual_fix_validation.wpix`
   - `mcp__pix-debug__diagnose_visual_artifact symptom="RTXDI identical after fix"`

4. Investigate alternative root causes:
   - Light grid broken (all cells empty)
   - RTXDI output buffer not bound
   - Shader reading wrong resource
   - Multiple bugs compounding

---

## Test Report Template

**Copy this after testing and fill in results:**

```markdown
## RTXDI Visual Difference Test - Results

**Date**: 2025-10-19
**Tester**: [Your Name]
**Build**: 0.7.7 + lighting scale fix
**Test Duration**: [XX minutes]

### Visual Comparison

**Multi-Light Mode**:
- Brightness: [Bright / Medium / Dim]
- Uniformity: [Smooth / Patchy / Noisy]
- FPS: [XX FPS]
- Screenshot: screenshots/test_2025-10-19_multi-light_baseline.png

**RTXDI Mode**:
- Brightness: [Bright / Medium / Dim] (Expected: **Dimmer** than multi-light)
- Spatial Variation: [None / Slight / Obvious]
- Temporal Flicker: [None / Slight / Noticeable]
- FPS: [XX FPS]
- Screenshot: screenshots/test_2025-10-19_rtxdi_after_fix.png

**Visual Difference**:
- [ ] **OBVIOUS** - I can immediately tell the modes apart (SUCCESS)
- [ ] Subtle - I can tell if I look carefully
- [ ] Identical - They look the same (FAIL - re-run debug steps)

### Functional Tests

- [ ] F3 toggle works smoothly (no crashes)
- [ ] ImGui labels update correctly (Multi-Light ↔ RTXDI)
- [ ] Log messages show "Switched to RTXDI system"
- [ ] RTXDI DispatchRays messages appear in RTXDI mode
- [ ] Render constants show `useRTXDI: 1` in RTXDI mode

### Overall Result

- [ ] **PASS** - RTXDI looks visually different from multi-light
- [ ] **FAIL** - RTXDI still looks identical (need further debugging)

**Notes**:
[Add any observations, anomalies, or questions here]
```

---

**END OF TEST PLAN**

**Estimated Test Duration**: 15 minutes
**Required Tools**: PlasmaDX-Clean.exe (Debug build), screenshot tool
**Optional Tools**: PIX for Windows (GPU validation)
