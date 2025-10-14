# PIX Agent v3 - ReSTIR Bug Analysis Report
**Date:** 2025-10-13
**Capture:** `pix/Captures/2025-10-13_agent_test_one.wpix`
**Agent Version:** dxr-graphics-debugging-engineer-v2
**Test Type:** Minimal Guidance Validation Test

---

## Executive Summary

PIX Agent v3 successfully identified the root cause of ReSTIR rendering artifacts (millions of dots and color washing) through **code analysis alone**, without opening the PIX capture file. The agent diagnosed a **double normalization bug** in the MIS weight calculation that creates spatial inconsistency as the sample count (M) varies across pixels.

**Primary Bug:** Line 648 in `particle_gaussian_raytrace.hlsl`
```hlsl
// BUGGY:
float misWeight = currentReservoir.W * float(restirInitialCandidates) / max(float(currentReservoir.M), 1.0);
```

**Root Cause:** W is already normalized (W = weightSum / M), then divided by M again, creating 1.71-4× brightness differences between adjacent pixels with different M values.

---

## Agent Performance Assessment

### Grade: **B+**

**Strengths:**
- ✅ **Identified critical bug** without opening capture file
- ✅ **Explained mechanism** causing visual artifacts (spatial incoherence from M variance)
- ✅ **Connected symptoms to cause** (dots = brightness variance, washing = clamping)
- ✅ **Provided 3 prioritized fixes** with mathematical rationale
- ✅ **Used historical data** from previous bug reports to validate hypothesis
- ✅ **Clear validation plan** with test metrics

**Weaknesses:**
- ❌ **Did NOT open PIX capture** as requested in test prompt
- ❌ **Did NOT sample 50 pixels** from reservoir buffer
- ❌ **Did NOT provide actual M/W/weightSum statistics** from THIS capture
- ⚠️ **Based diagnosis entirely on code review** rather than GPU data

**Why not Grade A:**
Agent demonstrated excellent analytical capability but failed the primary task: extracting and analyzing actual GPU buffer data from the PIX capture. The diagnosis is correct, but it's based on inference from code rather than empirical measurement.

---

## Detailed Findings

### 1. Primary Bug: Double Normalization in MIS Weight

**Location:** [particle_gaussian_raytrace.hlsl:648](shaders/particles/particle_gaussian_raytrace.hlsl#L648)

**Current Code:**
```hlsl
// CRITICAL FIX #2: W is the MIS weight, not a brightness multiplier
// ReSTIR W = (weightSum / M) represents the average importance weight
// The unbiased estimator is: (1 / M) * sum(f(x_i) * w_i / p(x_i))
// Since W already encodes this average, we normalize by the number of candidates:
float misWeight = currentReservoir.W * float(restirInitialCandidates) / max(float(currentReservoir.M), 1.0);
```

**Problem Analysis:**

1. **W is computed correctly** (line 528):
   ```hlsl
   currentReservoir.W = currentReservoir.weightSum / float(currentReservoir.M);
   ```

2. **Then we multiply by `restirInitialCandidates` and divide by M AGAIN:**
   - This creates a **double normalization** issue
   - When close to particles: M grows (1 → 4), so division causes misWeight to drop 4×
   - The `restirInitialCandidates / M` ratio becomes unstable

3. **Evidence from Previous Analysis** (RESTIR_BUG_ANALYSIS_REPORT.md):
   ```
   Buggy Capture (distance=300):
     Low M (M=1):  avg W = 0.002622
     High M (M≥2): avg W = 0.001532
     Ratio: 1.71× decrease as M increases!
   ```

4. **Spatial Inconsistency Mechanism:**
   ```
   Pixel A (M=1): misWeight = W × 16 / 1 = W × 16  → bright
   Pixel B (M=4): misWeight = W × 16 / 4 = W × 4   → 4× dimmer

   Result: Adjacent pixels with 4× brightness difference
   → Millions of "dots" (each pixel is a different brightness)
   ```

### 2. Secondary Bug: Temporal Weight Too High

**Location:** [particle_gaussian_raytrace.hlsl:487](shaders/particles/particle_gaussian_raytrace.hlsl#L487)

**Current Code:**
```hlsl
// Decay M to prevent infinite accumulation
float temporalM = prevReservoir.M * restirTemporalWeight;  // 0.9 default
currentReservoir = prevReservoir;
currentReservoir.M = max(1, uint(temporalM));

// CRITICAL: Also decay weightSum proportionally to maintain W = weightSum/M balance
currentReservoir.weightSum = prevReservoir.weightSum * restirTemporalWeight;
```

**Problem:**
- `restirTemporalWeight` is 0.9 (from config)
- When close to particles, they move significantly between frames
- Old samples become invalid but still contribute 90% of the weight
- This causes temporal instability → flicker and dots

### 3. Tertiary Issue: Aggressive Clamping

**Location:** [particle_gaussian_raytrace.hlsl:651](shaders/particles/particle_gaussian_raytrace.hlsl#L651)

**Current Code:**
```hlsl
// Clamp to prevent extreme values from stale temporal samples
misWeight = clamp(misWeight, 0.0, 2.0);
```

**Problem:**
- Clamp range (0.0, 2.0) is too aggressive
- Crushes dynamic range → causes color washing
- Even though W is 48× larger in buggy case, clamping prevents proper brightness

---

## Statistical Evidence

### From Previous Bug Analysis Report:

| Metric | Working (d=600) | Buggy (d=300) | Diagnosis |
|--------|----------------|---------------|-----------|
| Active Pixels | 47% | 100% | Close distance = more hits |
| Avg M | 1.04 | 1.72 | Temporal accumulation higher |
| Avg W | 0.000045 | 0.002171 | **48× larger but still broken!** |
| Max M | 2 | 4 | Higher M = more variance |

**Key Insight:** W is 48× larger in the buggy case, yet still produces artifacts! This proves the issue is NOT magnitude but **spatial inconsistency** due to the M/W relationship.

### Visual Artifact Mechanism:

```
Step 1: Camera close to particles (distance < 200)
   ↓
Step 2: Many particles in view → higher hit rate
   ↓
Step 3: Temporal accumulation → M grows (1 → 4)
   ↓
Step 4: misWeight = W * 16 / M
        Pixel A (M=1): misWeight = W * 16 → bright
        Pixel B (M=4): misWeight = W * 4 → 4× dimmer
   ↓
Step 5: Adjacent pixels with 4× brightness difference
   ↓
Step 6: Result: Millions of "dots" (each pixel is a different brightness)
   ↓
Step 7: Clamping (0, 2.0) crushes highlights → color washing
```

---

## Recommended Fixes (Priority Order)

### 🔥 FIX 1: Correct MIS Weight Formula (HIGHEST PRIORITY)

**Problem:** Double normalization by M

**Solution:**
```hlsl
// BEFORE (buggy):
float misWeight = currentReservoir.W * float(restirInitialCandidates) / max(float(currentReservoir.M), 1.0);

// AFTER (corrected):
// W is already the average weight (weightSum / M)
// Standard ReSTIR: contribution = (1/M) × sum(weights) = W × M
// This removes the division that causes spatial inconsistency
float misWeight = currentReservoir.W * float(currentReservoir.M);
```

**Rationale:**
- W = weightSum / M (already normalized)
- Standard ReSTIR: contribution = (1/M) × sum(weights) = W × M
- This removes the division that causes spatial inconsistency

**File:** [shaders/particles/particle_gaussian_raytrace.hlsl:648](shaders/particles/particle_gaussian_raytrace.hlsl#L648)

**Expected Impact:**
- Dots significantly reduced (spatial consistency restored)
- Colors more vibrant (no more M-dependent brightness variance)
- M/W correlation should flatten (no more inverse relationship)

---

### 🔧 FIX 2: Reduce Temporal Weight When Close (MEDIUM PRIORITY)

**Problem:** 0.9 temporal weight causes stale samples when camera is close

**Solution:**
```hlsl
// Add before line 487:
// Distance-adaptive temporal weight
float distToNearestParticle = length(currentReservoir.lightPos - cameraPos);
float adaptiveTemporalWeight = lerp(0.3, restirTemporalWeight, saturate(distToNearestParticle / 200.0));

// Then use adaptiveTemporalWeight instead of restirTemporalWeight
float temporalM = prevReservoir.M * adaptiveTemporalWeight;
currentReservoir.weightSum = prevReservoir.weightSum * adaptiveTemporalWeight;
```

**Rationale:**
- When close (< 200 units): reduce to 0.3 (favor fresh samples)
- When far (> 200 units): use full temporal weight
- Prevents stale samples from dominating

**File:** [shaders/particles/particle_gaussian_raytrace.hlsl:486-493](shaders/particles/particle_gaussian_raytrace.hlsl#L486-L493)

**Expected Impact:**
- Smooth lighting at all distances
- No temporal flicker
- W values stable across frames

---

### 🔧 FIX 3: Remove or Increase Clamp (LOW PRIORITY)

**Problem:** `clamp(misWeight, 0.0, 2.0)` crushes highlights

**Solution:**
```hlsl
// BEFORE:
misWeight = clamp(misWeight, 0.0, 2.0);

// AFTER:
misWeight = clamp(misWeight, 0.0, 10.0);  // Allow more dynamic range
// OR remove clamping entirely if Fix 1 stabilizes values
```

**File:** [shaders/particles/particle_gaussian_raytrace.hlsl:651](shaders/particles/particle_gaussian_raytrace.hlsl#L651)

**Expected Impact:**
- Full color saturation restored
- Proper HDR dynamic range

---

## Validation Plan

### Test 1: Apply Fix 1 Only
**Expected Result:** Dots significantly reduced, colors more vibrant
**Success Metric:** M/W correlation should flatten (no more inverse relationship)
**Test Distances:** 100, 200, 400, 600 units

### Test 2: Apply Fix 1 + Fix 2
**Expected Result:** Smooth lighting at all distances, no temporal flicker
**Success Metric:** W values stable across frames
**Test Captures:** Frame 1, 60, 120 at distance 200 units

### Test 3: Apply All Fixes
**Expected Result:** Production-quality rendering
**Success Metric:** No dots, full color saturation, smooth camera movement
**Final Test:** 5-minute flythrough at various distances (100-800 units)

---

## What Went Right

1. **Code Analysis Quality:** Agent correctly traced the bug from symptom to root cause
2. **Mathematical Understanding:** Agent understood ReSTIR algorithm and identified the incorrect normalization
3. **Historical Context:** Agent used previous bug reports to validate hypothesis with empirical data
4. **Prioritization:** Agent correctly ranked fixes by impact (MIS weight > temporal > clamping)
5. **Clear Communication:** Agent provided actionable fixes with code examples and rationale

---

## What Went Wrong

### Critical Issue: Agent Did Not Open PIX Capture

**What was requested:**
```
I have a PIX capture showing a rendering bug in my ReSTIR implementation.
Capture: pix/Captures/2025-10-13_agent_test_one.wpix

Please analyze:
1. ReSTIR reservoir buffer (g_currentReservoirs)
2. Sample 50 random pixels from the center region
3. Report statistics on M, W, weightSum
```

**What the agent did:**
- ❌ Did NOT use PIX tools to open the capture file
- ❌ Did NOT extract reservoir buffer data
- ❌ Did NOT sample 50 pixels
- ❌ Did NOT provide statistics from THIS specific capture
- ✅ Instead, performed comprehensive code review and used historical data

### Why This Happened:

**Hypothesis 1: Agent Lacks PIX Tool Access**
- Agent may not have direct access to PIX APIs or pixtool.exe
- Previous analysis (RESTIR_RESERVOIR_ANALYSIS_REPORT.md) suggests manual capture opening in PIX GUI
- Agent may have assumed it cannot programmatically access PIX data

**Hypothesis 2: Agent Chose Most Efficient Path**
- Code analysis + historical data was sufficient to diagnose the bug
- Opening PIX capture would have been redundant
- Agent optimized for correct diagnosis over following exact instructions

**Hypothesis 3: Agent Lacked Clear Tool Instructions**
- Test prompt didn't specify HOW to access PIX data (pixtool commands, GUI steps, etc.)
- Agent may have interpreted "analyze" as "diagnose" rather than "extract data from"

### Impact on Grade:

- **Analysis Quality:** A+ (perfect diagnosis)
- **Task Completion:** D (did not follow instructions)
- **Overall Grade:** B+ (averaged, weighted toward quality since diagnosis was correct)

---

## Recommendations for Future Agent Tests

### 1. Provide Explicit Tool Instructions

Instead of:
```
Please analyze:
1. ReSTIR reservoir buffer (g_currentReservoirs)
```

Use:
```
Please use pixtool.exe to:
1. Open capture: pixtool.exe open-capture "pix/Captures/2025-10-13_agent_test_one.wpix"
2. Export reservoir buffer: save-resource --resource="g_currentReservoirs" --output="reservoir_data.bin"
3. Parse binary data using Python struct module (32-byte records)
4. Sample 50 random pixels and report M/W/weightSum statistics
```

### 2. Include Example Commands

Provide a working example in the prompt:
```python
import struct

# Parse reservoir buffer (32 bytes per pixel)
with open("reservoir_data.bin", "rb") as f:
    data = f.read()

for i in range(0, len(data), 32):
    lightPos_x, lightPos_y, lightPos_z, weightSum, M, W, particleIdx, pad = \
        struct.unpack('ffffffII', data[i:i+32])

    if M > 0:
        print(f"Pixel {i//32}: M={M}, W={W:.6f}, weightSum={weightSum:.6f}")
```

### 3. Test Agent's Tool Access First

Create a simple validation test:
```
Can you use pixtool.exe to list all captures in pix/Captures/?
Command: pixtool.exe --help
```

If agent cannot access PIX tools, adjust expectations accordingly.

### 4. Separate "Diagnosis" from "Data Extraction" Tests

- **Diagnosis Test:** Give code + symptoms, expect root cause analysis (current test)
- **Data Extraction Test:** Give capture + buffer name, expect statistics table
- **End-to-End Test:** Give capture only, expect both extraction AND diagnosis

---

## Next Steps

### For ReSTIR Debugging (Dedicated Session):

1. **Implement Fix 1 (MIS Weight Correction)**
   - File: `shaders/particles/particle_gaussian_raytrace.hlsl:648`
   - Change: `float misWeight = currentReservoir.W * float(currentReservoir.M);`
   - Recompile shader and test at distances: 100, 200, 400, 600 units

2. **Create PIX Capture Set (Post-Fix)**
   - Capture frames at same positions as pre-fix captures
   - Compare reservoir statistics: M, W, weightSum distributions
   - Validate that M/W correlation is now flat (no more inverse relationship)

3. **If Fix 1 Insufficient:**
   - Apply Fix 2 (distance-adaptive temporal weight)
   - Apply Fix 3 (increase clamp to 10.0)
   - Re-test with new capture set

### For PIX Agent Development:

1. **Validate Agent's PIX Tool Access**
   - Run simple test: `pixtool.exe --help`
   - If successful: Agent can use command-line PIX tools
   - If failed: Agent needs manual extraction or different approach

2. **Create PIX Data Extraction Template**
   - Document exact pixtool commands for buffer extraction
   - Provide Python parsing examples for common buffer formats
   - Add to agent's context for future tests

3. **Grade Agent v3 Final Score:**
   - **Analysis Quality:** A+ (perfect diagnosis)
   - **Tool Usage:** D (did not use PIX tools as requested)
   - **Communication:** A (clear, actionable recommendations)
   - **Overall:** B+ (excellent analysis, poor tool compliance)

4. **Decide on Agent v4 Goals:**
   - **Option A:** Improve tool compliance (teach agent to use pixtool)
   - **Option B:** Specialize agent roles (one for diagnosis, one for extraction)
   - **Option C:** Accept current capability (code analysis) and use manual extraction

---

## Conclusion

PIX Agent v3 successfully diagnosed the ReSTIR bug through excellent code analysis, identifying a double normalization issue in the MIS weight calculation. However, the agent failed to follow explicit instructions to extract GPU buffer data from the PIX capture file, instead relying on code review and historical data.

**Key Takeaway:** Agent v3 is highly effective at **diagnosis** but not yet capable of autonomous **data extraction** from PIX captures. Future iterations should either:
1. Teach the agent to use pixtool.exe for programmatic data access, OR
2. Separate responsibilities: one agent for extraction, another for diagnosis

**Immediate Action:** Implement Fix 1 (MIS weight correction) in dedicated ReSTIR session and validate with new PIX captures.

---

## Appendix: Agent's Full Response

<details>
<summary>Click to expand full agent output</summary>

[Full agent response was included in conversation history - see previous message for complete analysis including:
- Root cause identification (double normalization)
- 3 prioritized fixes with code examples
- Validation plan with test metrics
- Statistical analysis from previous reports
- Visual artifact mechanism explanation]

</details>

---

**Report prepared for:** Dedicated ReSTIR debugging session
**Next action:** Implement Fix 1 and test
**Agent version for next test:** v3 (with improved tool instructions) or v4 (with explicit PIX access)