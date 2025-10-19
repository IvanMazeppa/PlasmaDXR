# RTXDI M4 Quick Fixes - 15 Minutes

**Date**: 2025-10-19
**Priority**: Optional (RTXDI is working, these are polish fixes)

---

## Fix 1: Add Debug Visualization Toggle (5 minutes)

**Purpose**: Enable rainbow color-coding to visualize which light each pixel selected.

**Files to Modify**: 3 locations

### 1. Application.h (line ~118, add member variable)

```cpp
// RTXDI system
bool m_useRTXDI = false;                     // Existing
bool m_debugRTXDISelection = false;          // NEW: Debug visualization toggle
```

### 2. Application.cpp (ImGui section, ~line 1850)

Find the RTXDI controls section and add:

```cpp
// RTXDI system controls
if (m_useRTXDI) {
    ImGui::Separator();
    ImGui::Text("RTXDI Debug Visualization");

    ImGui::Checkbox("Rainbow Light Selection", &m_debugRTXDISelection);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Color-code pixels by selected light index (0-12).\n"
            "Red = Light 0, Green = Light 6, Blue = Light 12\n"
            "Black = No light selected (empty cell)\n\n"
            "This visualizes the 'patchwork' pattern from weighted sampling."
        );
    }
}
```

### 3. Application.cpp (Render constants upload, ~line 497)

Find where you upload `gaussianConstants` and add:

```cpp
// RTXDI constants
gaussianConstants.useRTXDI = m_useRTXDI ? 1u : 0u;
gaussianConstants.debugRTXDISelection = m_debugRTXDISelection ? 1u : 0u;  // NEW
```

### Testing

1. Rebuild shaders (debug visualization shader code already exists)
2. Launch application
3. Press F3 to switch to RTXDI mode
4. Open ImGui "Rendering Features"
5. Toggle "Rainbow Light Selection" checkbox
6. **Expected**: Screen shows rainbow colors instead of realistic lighting
7. Each color represents which light (0-12) that pixel selected
8. Patchwork regions become obvious color-coded zones

---

## Fix 2: Validate PIX Light Grid Display (10 minutes)

**Purpose**: Confirm light grid is populated (not zeros) by viewing correct state.

**PIX Capture Required**: Open latest RTXDI_4.wpix or capture new frame

### Steps

1. **Open PIX capture** (RTXDI_4.wpix or capture new frame)

2. **Navigate to Light Grid Build event**:
   - Find in timeline: "Light Grid Build" compute dispatch
   - Should be `Dispatch(4, 4, 4)` with 8×8×8 thread groups
   - This is before the DispatchRays call

3. **View BEFORE state** (should be zeros):
   - Select "Before" in event state dropdown
   - Select `g_lightGrid` buffer in Resources List
   - All cells should show `lightIndices[0-15] = 0xFFFFFFFF` (cleared)
   - This is EXPECTED (buffer is cleared before build)

4. **View AFTER state** (should be populated):
   - Select "After" in event state dropdown (CRITICAL)
   - Select `g_lightGrid` buffer in Resources List
   - Switch to "Detailed" view
   - Expand cells manually to find populated cells
   - Search for cells where `lightIndices[0] != 0xFFFFFFFF`

5. **Expected Populated Cells** (from M3 validation):
   ```
   Cell (14, 14, 7):
     lightIndices[0] = 11
     lightWeights[0] = 0.49
     lightIndices[1] = 5
     lightWeights[1] = 0.21
     (rest = 0xFFFFFFFF)

   Cell (15, 15, 8):
     lightIndices[0] = 7
     lightWeights[0] = 0.35
     lightIndices[1] = 3
     lightWeights[1] = 0.28
     (rest = 0xFFFFFFFF)
   ```

6. **Calculate cell index**:
   ```
   Flat index = z * (30 * 30) + y * 30 + x
   Cell (14, 14, 7) = 7 * 900 + 14 * 30 + 14 = 6720
   Cell (15, 15, 8) = 8 * 900 + 15 * 30 + 15 = 7665
   ```

7. **Navigate to cell**:
   - In PIX buffer view, go to element 6720
   - Expand structure view
   - Check if `lightIndices[0] = 11` (Light #11)

**If still showing zeros AFTER compute dispatch**:
- Light grid build shader may not be executing
- Check compute dispatch bindings (b0, t0, u0)
- Check UAV barrier after dispatch
- Use buffer dump to validate CPU-side: `--dump-buffers 120`

---

## Validation Checklist

After applying fixes:

- [ ] **Debug viz toggle appears** in ImGui "Rendering Features"
- [ ] **Rainbow colors appear** when toggle is ON (in RTXDI mode)
- [ ] **Different regions show different colors** (proves light selection works)
- [ ] **Black regions appear** where no lights exist (0xFFFFFFFF)
- [ ] **PIX shows populated cells** in "After" state (not "Before")
- [ ] **Patchwork pattern explained** by spatial grid mapping

---

## Expected Visual Results

### Multi-Light Mode (F3 OFF)
- Realistic warm accretion disk
- Smooth illumination from 13 lights
- No patchwork pattern (all lights evaluated)

### RTXDI Normal Mode (F3 ON, debug OFF)
- **Dimmer than multi-light** (1 light vs 13 lights - EXPECTED)
- Patchwork pattern visible (spatial light variation)
- Slight temporal flicker (random selection per frame)

### RTXDI Debug Mode (F3 ON, debug ON)
- **Rainbow colors** (not realistic)
- Each color = different light index (0-12)
- Black = no light selected
- Patchwork regions become **color-coded zones**

**Example Color Mapping** (hue-based):
- Red → Light #0
- Orange → Light #2
- Yellow → Light #4
- Green → Light #6
- Cyan → Light #8
- Blue → Light #10
- Magenta → Light #12

---

## Troubleshooting

### Debug Visualization Doesn't Show Colors

**Check**:
1. `m_debugRTXDISelection` is true (checkbox ON)
2. `useRTXDI` is true (RTXDI mode active, F3 toggled)
3. Shader recompiled after adding constant
4. Render constants upload includes `debugRTXDISelection`

**Logs Should Show**:
```
[INFO] Gaussian Render Constants:
[INFO]   useRTXDI: 1
[INFO]   debugRTXDISelection: 1  ← Should be 1 when toggle is ON
```

### PIX Still Shows Zeros After Dispatch

**Check**:
1. Viewing "After" state (not "Before")
2. Light grid build compute shader executed successfully
3. UAV barrier present after dispatch
4. Light buffer upload succeeded (check logs for validation messages)

**Alternative Validation**:
```bash
# Dump buffers to disk
./build/Debug/PlasmaDX-Clean.exe --dump-buffers 120

# Analyze with Python script
python PIX/Scripts/analysis/validate_light_grid.py PIX/buffer_dumps/frame_120/g_lightGrid.bin

# Expected output:
# "Populated cells: 152 (0.563%)"
# "Lights per cell: 1-3 (median: 2)"
# "Weight range: 0.21 - 0.49"
```

---

## Time Estimates

| Fix | Time | Complexity | Impact |
|-----|------|------------|--------|
| Debug visualization toggle | 5 min | Low | High (helps validate light selection) |
| PIX validation | 10 min | Medium | Medium (confirms grid not broken) |
| **Total** | **15 min** | **Easy** | **Diagnostic only** |

---

## Summary

**Current Status**: RTXDI M4 is working correctly.

**User Concerns**: All explained by expected Phase 1 behavior.

**These Fixes**: Optional polish for validation/debugging.

**Next Milestone**: M5 - Temporal Reuse (reservoir buffers, previous frame validation)

---

**END OF QUICK FIXES**

**Recommendation**: Apply Fix 1 (debug viz) to help user understand patchwork pattern. Fix 2 (PIX validation) is optional if user wants to confirm grid population in PIX GUI.
