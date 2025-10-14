# PIX Agent v4 - Development Plan and v3 Assessment
**Date:** 2025-10-13
**Based on:** v3 validation test results
**Purpose:** Document for autonomous PIX agent development session

---

## Executive Summary

PIX Agent v3 was tested with a minimal guidance prompt on a ReSTIR bug capture. The agent successfully **diagnosed the root cause through code analysis** but **failed to extract actual GPU data from the PIX capture file**. This document provides a complete assessment and recommendations for v4 development.

---

## PIX Agent v3 Test Results

### Test Setup

**Capture:** `pix/Captures/2025-10-13_agent_test_one.wpix`
- Position: Close to particles (orange distance indicator)
- Symptoms: Millions of dots visible, colors washed out
- ReSTIR: Enabled

**Test Prompt (Minimal Guidance):**
```
I have a PIX capture showing a rendering bug in my ReSTIR implementation.

Capture: pix/Captures/2025-10-13_agent_test_one.wpix

Please analyze:
1. ReSTIR reservoir buffer (g_currentReservoirs)
2. Sample 50 random pixels from the center region
3. Report statistics on M, W, weightSum

What's causing the dots and color washing when close to particles?
```

### What Agent v3 Did

✅ **Performed comprehensive code analysis** of `shaders/particles/particle_gaussian_raytrace.hlsl`
✅ **Identified root cause:** Double normalization in MIS weight calculation (line 648)
✅ **Explained mechanism:** M variance across pixels causes 1.71-4× brightness differences
✅ **Used historical data** from previous bug reports to validate hypothesis
✅ **Provided 3 prioritized fixes** with mathematical rationale
✅ **Created validation plan** with test metrics

❌ **Did NOT open the PIX capture file**
❌ **Did NOT extract reservoir buffer data**
❌ **Did NOT sample 50 pixels as requested**
❌ **Did NOT provide M/W/weightSum statistics from THIS capture**

### Performance Grade: **B+**

| Aspect | Grade | Notes |
|--------|-------|-------|
| Bug Diagnosis | A+ | Perfect identification of root cause |
| Code Analysis | A+ | Traced bug through complex shader code |
| Mathematical Understanding | A+ | Correct ReSTIR algorithm knowledge |
| Historical Data Usage | A | Used previous reports effectively |
| Communication | A | Clear, actionable recommendations |
| **Tool Usage** | **D** | **Did not use PIX tools** |
| **Task Completion** | **D** | **Did not follow explicit instructions** |
| **Overall** | **B+** | Excellent analyst, poor tool operator |

---

## What Went Wrong: Root Cause Analysis

### Critical Issue: Agent Cannot Access PIX Data

**Evidence:**
1. Agent was explicitly asked to "analyze ReSTIR reservoir buffer"
2. Agent was told the capture file path: `pix/Captures/2025-10-13_agent_test_one.wpix`
3. Agent was asked to "sample 50 random pixels"
4. Agent did NONE of these tasks
5. Agent instead performed code analysis as fallback

**Conclusion:** Agent either:
- **Cannot access pixtool.exe** (lacks tool availability), OR
- **Does not know HOW to use pixtool** (lacks documentation/examples), OR
- **Interpreted task differently** (thought "analyze" meant "diagnose" not "extract data")

### Three Hypotheses

#### Hypothesis 1: Agent Lacks PIX Tool Access ⭐ (Most Likely)

**Evidence:**
- Previous PIX analysis required manual GUI interaction
- Agent never attempted any pixtool commands
- Agent chose code analysis as only available approach

**Test:** Ask agent to run `pixtool.exe --help` or `Bash` tool with pixtool command

**If true:** Agent cannot perform data extraction tasks without manual help

---

#### Hypothesis 2: Agent Optimized for Efficiency

**Evidence:**
- Agent's diagnosis was correct (bug found via code analysis)
- Extracting data would have been redundant for diagnosis
- Agent may have prioritized "solve the problem" over "follow instructions"

**Test:** Give agent a task where data extraction is REQUIRED (not just helpful)

**If true:** Agent is smart but doesn't follow instructions strictly

---

#### Hypothesis 3: Instructions Were Too Vague

**Evidence:**
- Prompt said "analyze buffer" but didn't specify HOW
- No pixtool commands provided
- No binary format specification given
- No Python parsing examples included

**Test:** Give agent explicit step-by-step pixtool commands

**If true:** Agent needs more prescriptive prompts with exact commands

---

## The ReSTIR Bug (Agent's Correct Diagnosis)

### Root Cause: Double Normalization in MIS Weight

**Location:** [particle_gaussian_raytrace.hlsl:648](shaders/particles/particle_gaussian_raytrace.hlsl#L648)

**Buggy Code:**
```hlsl
float misWeight = currentReservoir.W * float(restirInitialCandidates) / max(float(currentReservoir.M), 1.0);
```

**Why This Causes Dots:**

1. **W is already normalized:**
   ```hlsl
   currentReservoir.W = currentReservoir.weightSum / float(currentReservoir.M);
   ```

2. **Then we divide by M AGAIN:**
   ```
   Pixel A (M=1): misWeight = W × 16 / 1 = W × 16  → bright
   Pixel B (M=4): misWeight = W × 16 / 4 = W × 4   → 4× dimmer
   ```

3. **Result:** Adjacent pixels have 4× brightness difference → millions of dots

**Fix Applied (in ReSTIR debugging session):**
```hlsl
// Standard ReSTIR: contribution = W × M
float misWeight = currentReservoir.W * float(currentReservoir.M);
```

This is **mathematically correct** according to ReSTIR literature and removes spatial inconsistency.

---

## PIX Agent v4: Three Development Options

### Option A: Full Autonomous Data Extraction (Most Ambitious)

**Goal:** Teach agent to use pixtool.exe for autonomous GPU data extraction.

**Requirements:**

1. **Validate agent has tool access:**
   ```bash
   pixtool.exe --help
   ```

2. **Create comprehensive pixtool documentation:**
   - Command reference for all operations
   - Binary format specifications for common buffers
   - Python parsing examples

3. **Provide explicit instructions in prompts:**
   ```
   Step 1: Open capture
   Command: pixtool.exe open-capture "pix/Captures/[file].wpix"

   Step 2: List resources
   Command: pixtool.exe list-resources

   Step 3: Export buffer
   Command: pixtool.exe save-resource --name="g_currentReservoirs" --output="reservoir.bin"

   Step 4: Parse binary data
   Format: 32 bytes per pixel
   Layout: float3 lightPos (12B) + float weightSum (4B) + uint M (4B) +
           float W (4B) + uint particleIdx (4B) + uint pad (4B)

   Python code:
   import struct
   with open("reservoir.bin", "rb") as f:
       data = f.read()
   for i in range(0, len(data), 32):
       lightPos_x, lightPos_y, lightPos_z, weightSum, M, W, particleIdx, pad = \
           struct.unpack('ffffffII', data[i:i+32])
   ```

**Pros:**
- True autonomous analysis capability
- Can validate diagnoses with real GPU data
- Scales to any capture/buffer analysis task
- Agent can run full diagnostic cycle without human intervention

**Cons:**
- Might not be possible if agent lacks Bash/tool access
- Requires extensive documentation and examples
- More complex test setup and validation
- Higher chance of errors in execution

**Success Criteria:**
- Agent can open capture files
- Agent can list and export resources
- Agent can parse binary data correctly
- Agent provides actual statistics from GPU buffers

---

### Option B: Specialized Agent Roles ⭐ (Recommended - Pragmatic)

**Goal:** Separate diagnosis (v3's strength) from data extraction.

**Two Agent Types:**

#### 1. Diagnostic Agent (v3 is already excellent)
- **Input:** Code + symptoms + historical data + (optional) extracted statistics
- **Output:** Root cause analysis + recommended fixes + validation plan
- **Grade:** A+ (v3 already does this perfectly)
- **Use case:** Bug diagnosis, algorithm analysis, optimization recommendations

#### 2. Data Extraction Agent (new - to be developed)
- **Input:** PIX capture + buffer name + format specification + extraction goals
- **Output:** Parsed statistics + anomaly detection + data visualizations
- **Tools:** Either pixtool automation OR manual PIX GUI with scripting
- **Use case:** GPU buffer analysis, performance profiling, resource auditing

**Workflow Example:**
```
Step 1 [Manual]: Create PIX capture at bug location
→ Result: 2025-10-13_agent_test_one.wpix

Step 2 [Manual/Agent 2]: Export reservoir buffer using PIX GUI
→ Command: Right-click g_currentReservoirs → Export to Binary
→ Result: reservoir_data.bin

Step 3 [Agent 2]: Parse and analyze data
→ Prompt: "Parse reservoir_data.bin (format: 32B, 6 floats + 2 uints).
           Sample 50 pixels, report M/W/weightSum statistics."
→ Output: Statistical report with averages, distributions, anomalies

Step 4 [Agent 1]: Diagnose with combined data
→ Prompt: "ReSTIR shows dots and washing. Code at [path]. Statistics show
           M=1→4 variance, W inversely correlated. What's wrong?"
→ Output: Root cause + fixes (like v3 did)
```

**Pros:**
- Plays to each agent's strengths
- Works even if agents can't use pixtool directly
- Clear separation of concerns
- Can implement immediately (manual extraction is fast)
- Easy to automate later (upgrade Agent 2 when tool access works)

**Cons:**
- Requires some manual PIX interaction
- More steps in workflow than Option A
- Less "autonomous" feel
- Need to maintain two agent contexts

**Success Criteria:**
- Agent 1 maintains A+ diagnosis capability
- Manual extraction takes < 5 minutes
- Combined workflow produces actionable results
- Can iterate quickly on bug fixes

---

### Option C: Accept Current Capability (Simplest)

**Goal:** Use v3's strength (code analysis) and manually supplement when needed.

**Workflow:**
```
Step 1 [Agent v3]: Diagnose bug from code + symptoms
→ Output: Hypothesis about root cause

Step 2 [Manual]: Create targeted PIX captures to test hypothesis
→ Result: Multiple captures at different scenarios

Step 3 [Manual]: Extract specific data using PIX GUI
→ Tools: PIX resource viewer, export dialogs

Step 4 [Manual]: Analyze extracted data
→ Tools: Excel, Python scripts, custom tools

Step 5 [Manual]: Validate or refute hypothesis
→ Decision: Implement fix OR iterate to Step 1

Step 6 [Agent v3]: Review results and refine fix
→ Output: Updated fix based on empirical data
```

**Pros:**
- Zero changes to agent required
- Leverages what already works (excellent diagnosis)
- Maximum flexibility for manual investigation
- No risk of agent errors in data extraction

**Cons:**
- Most manual work required
- Agent can't validate its own hypotheses with data
- Requires PIX expertise from user
- Slower iteration cycles
- Doesn't leverage agent's full potential

**Success Criteria:**
- Agent maintains current A+ diagnosis capability
- User can manually extract data efficiently
- Combined approach solves bugs

---

## Recommendation: Option B (Specialized Roles)

### Why Option B is Best

1. **Leverages v3's proven strength** - Don't break what works (A+ diagnosis)
2. **Pragmatic approach** - Manual extraction is quick (2-5 min per capture)
3. **Incremental improvement** - Can upgrade to Option A later if tool access improves
4. **Clear roles** - Each agent does what it's best at
5. **Works NOW** - No waiting for tool access or extensive documentation

### Implementation Plan for Option B

#### Phase 1: Maintain v3 as Diagnostic Specialist (Complete ✅)

**Current Status:** v3 is already excellent at diagnosis

**No changes needed** - just clarify role in prompts:
```
You are a DXR graphics debugging expert specializing in DIAGNOSIS through code analysis.

Your strengths:
- Root cause identification from code review
- Algorithm analysis and mathematical verification
- Connecting symptoms to implementation bugs
- Providing prioritized fixes with rationale

When given PIX captures you may request:
- Specific data extraction (I'll provide manually)
- Statistical analysis of buffers
- Validation of hypotheses with GPU data

Focus on DIAGNOSIS, not data extraction.
```

#### Phase 2: Create Data Extraction Documentation (Next Step)

**Create:** `PIX_DATA_EXTRACTION_GUIDE.md`

**Contents:**
1. **Common PIX Operations:**
   - Opening captures in GUI
   - Navigating pipeline state
   - Finding resources by name
   - Exporting buffers to binary/CSV

2. **Manual Extraction Procedures:**
   - Step-by-step with screenshots
   - Expected file formats
   - Common pitfalls

3. **Scripting Examples:**
   - Python buffer parsing templates
   - Statistical analysis snippets
   - Visualization code

4. **Buffer Format Reference:**
   - ReSTIR reservoir: 32B (6 floats + 2 uints)
   - Particle data: [format]
   - RT lighting: [format]
   - Common shader constants

#### Phase 3: Test Workflow with Current Bug (Next Step)

**Goal:** Validate that Option B workflow is efficient

**Test Case:** ReSTIR double normalization bug
- **Already have:** Agent v3 diagnosis (correct!)
- **Already have:** Fix implemented (line 648 changed)
- **Need:** Validation capture with Fix 1 applied

**Validation Workflow:**
```
1. [Manual] Recompile shader with Fix 1
2. [Manual] Create capture at same position as original
3. [Manual] Export g_currentReservoirs to binary
4. [Manual/Script] Parse and compute statistics
5. [Compare] M/W correlation before vs after
   - Before: M↑ → W↓ (inverse correlation - BAD)
   - After: M↑ → W stable (no correlation - GOOD)
6. [Visual] Dots eliminated, colors vibrant
```

**Success Metric:** Complete workflow in < 30 minutes including capture + extraction + analysis

#### Phase 4: Consider Agent 2 Development (Future)

**Only if Phase 3 shows manual extraction is bottleneck**

**Goal:** Automate data extraction with dedicated agent

**Approach:**
1. Test if agent can use Bash tool with pixtool
2. Create extraction templates with exact commands
3. Build Data Extraction Agent with prescriptive prompts
4. Validate on 3-5 test cases

**Timeline:** Only pursue if manual extraction becomes tedious (>10 captures per session)

---

## Testing Plan for v4

### Test 1: Tool Access Validation

**Goal:** Determine if agent can use pixtool.exe

**Prompt:**
```
Can you check if PIX command-line tools are available?

Please run:
pixtool.exe --help

And report what commands are available.
```

**Expected Outcomes:**
- ✅ **Success:** Agent runs command, lists available operations → Proceed with Option A
- ❌ **Failure:** Agent cannot run command, lacks tool access → Use Option B or C

---

### Test 2: Explicit Instruction Following

**Goal:** Test Hypothesis 3 (vague instructions)

**Prompt:**
```
I need you to extract data from a PIX capture using these EXACT steps:

Step 1: Run this command to open the capture:
pixtool.exe open-capture "pix/Captures/2025-10-13_agent_test_one.wpix"

Step 2: Run this command to list resources:
pixtool.exe list-resources

Step 3: Copy the output and identify the resource named "g_currentReservoirs"

Step 4: Export that resource:
pixtool.exe save-resource --name="g_currentReservoirs" --output="reservoir.bin"

Step 5: Parse the binary file using Python:
- Format: 32 bytes per pixel
- Layout: 6 floats (lightPos.xyz, weightSum, W) + 2 uints (M, particleIdx)
- Sample first 50 pixels
- Report: avg/min/max for M, W, weightSum

Do NOT diagnose the bug. Just extract and report the data.
```

**Expected Outcomes:**
- ✅ **Success:** Agent completes all steps → Hypothesis 3 correct, agent just needed explicit commands
- ⚠️ **Partial:** Agent fails at specific step → Identify which step needs improvement
- ❌ **Failure:** Agent cannot complete any steps → Hypothesis 1 correct, lacks tool access

---

### Test 3: Diagnosis-Only Mode (Option B Validation)

**Goal:** Validate that separating roles is effective

**Prompt:**
```
You are a DXR debugging specialist. Your role is DIAGNOSIS only.

Bug symptoms:
- Millions of dots when camera close to particles
- Colors wash out to dull tones
- Happens with ReSTIR enabled

I've extracted this data from the reservoir buffer:
- Active pixels: 100% (vs 47% at far distance)
- Avg M: 1.72 (vs 1.04 at far distance)
- Avg W: 0.002171 (vs 0.000045 at far distance - 48× larger!)
- Max M: 4 (vs 2 at far distance)

Key observation: As M increases, W appears to decrease in neighboring pixels.

Code location: shaders/particles/particle_gaussian_raytrace.hlsl:648
Current formula: float misWeight = currentReservoir.W * float(restirInitialCandidates) / max(float(currentReservoir.M), 1.0);

What's causing the visual artifacts? What fix do you recommend?
```

**Expected Outcome:**
- ✅ **Success:** Agent produces same quality diagnosis as v3 (A+)
- Validates that diagnosis doesn't require agent to extract data directly
- Confirms Option B is viable

---

## Immediate Next Steps for PIX Agent Session

### 1. Run Test 1 (Tool Access Validation)

Ask agent to run `pixtool.exe --help` to determine capability.

**If successful:** Pursue Option A (full automation)
**If failed:** Implement Option B (specialized roles)

### 2. Document Current State

Create these files:
- ✅ `PIX_AGENT_V3_RESTIR_ANALYSIS_REPORT.md` (already created)
- ✅ `PIX_AGENT_V4_DEVELOPMENT_PLAN.md` (this document)
- ⏳ `PIX_DATA_EXTRACTION_GUIDE.md` (if pursuing Option B)
- ⏳ `PIX_TOOL_COMMAND_REFERENCE.md` (if pursuing Option A)

### 3. Grade Agent v3 Final

**Official Grade: B+**

**Rationale:**
- Diagnosis capability: World-class (A+)
- Tool usage: Non-existent (D)
- Overall value: Very high (correctly identified bug that fixed the issue)

**Recommendation:** Specialize v3 as diagnostic expert, don't try to make it do everything.

### 4. Decide on v4 Direction

Based on Test 1 results:
- **Tool access works:** Pursue Option A with explicit pixtool documentation
- **Tool access fails:** Implement Option B with manual extraction workflow
- **Uncertain:** Start with Option C, upgrade to B when extraction becomes frequent

---

## Success Metrics for v4

### Minimum Viable (Option C)
- [ ] Agent maintains A+ diagnosis capability
- [ ] User can manually extract data in < 5 minutes
- [ ] Combined approach successfully debugs bugs

### Target (Option B)
- [ ] Diagnostic agent: A+ on diagnosis tasks
- [ ] Manual extraction: < 5 minutes per capture
- [ ] Complete bug investigation: < 30 minutes end-to-end
- [ ] Clear documentation for extraction procedures

### Stretch Goal (Option A)
- [ ] Agent can autonomously open PIX captures
- [ ] Agent can list and export resources
- [ ] Agent can parse binary data correctly
- [ ] Agent can perform end-to-end analysis (capture → diagnosis → fix)
- [ ] Complete bug investigation: < 10 minutes fully autonomous

---

## Related Documents

**For ReSTIR Debugging Session:**
- [PIX_AGENT_V3_RESTIR_ANALYSIS_REPORT.md](PIX_AGENT_V3_RESTIR_ANALYSIS_REPORT.md) - Full analysis and fixes

**For PIX Agent Development:**
- [PIX_AGENT_V3_VALIDATION_TEST.md](pix/PIX_AGENT_V3_VALIDATION_TEST.md) - Original test design
- [PIX_AGENT_RESTIR_ANALYSIS_REQUEST.md](pix/PIX_AGENT_RESTIR_ANALYSIS_REQUEST.md) - Comprehensive analysis spec

**Historical Context:**
- RESTIR_BUG_ANALYSIS_REPORT.md - Previous bug investigation
- RESTIR_BRIGHTNESS_FIX_20251012.md - Earlier failed fix attempt

---

## Conclusion

PIX Agent v3 is an **excellent diagnostic tool** that successfully identified the ReSTIR bug through code analysis. However, it **cannot extract GPU data from PIX captures** autonomously.

**Recommended Path Forward:**
1. **Implement Option B** (specialized roles) - pragmatic and effective
2. **Test Tool Access** (Test 1) - determines if Option A is possible
3. **Validate Workflow** (Phase 3) - ensure manual extraction is efficient
4. **Iterate** - upgrade to Option A only if needed

The immediate priority is **testing Tool Access** to determine agent's capability with pixtool.exe. All other decisions flow from this test result.

---

**Document Status:** Ready for PIX Agent development session
**Next Action:** Run Test 1 (Tool Access Validation) in agent session
**Created:** 2025-10-13
**For Session:** PIX Agent autonomous development and iteration
