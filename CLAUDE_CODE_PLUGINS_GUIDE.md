# Claude Code Plugins & Debugging Agents - Complete Guide

**Date:** 2025-10-17
**Project:** PlasmaDX-Clean
**Purpose:** Comprehensive guide to Claude Code plugin system and available debugging agents

---

## Table of Contents

1. [What Are Claude Code Plugins?](#what-are-claude-code-plugins)
2. [Two Types of Agent Tools](#two-types-of-agent-tools)
3. [Built-In Debugging Agents (Available Now)](#built-in-debugging-agents-available-now)
4. [Agent SDK vs Built-In Agents](#agent-sdk-vs-built-in-agents)
5. [Your Current PIX Agent System](#your-current-pix-agent-system)
6. [Multi-Light System Analysis (pix-debugging-agent)](#multi-light-system-analysis-pix-debugging-agent)
7. [When to Use Each Agent Type](#when-to-use-each-agent-type)
8. [How to Invoke Agents](#how-to-invoke-agents)
9. [Future Agent SDK Use Cases](#future-agent-sdk-use-cases)
10. [Quick Reference](#quick-reference)

---

## What Are Claude Code Plugins?

Claude Code has a **plugin system** that provides specialized agents and tools for specific tasks. You can install plugins using the `/plugin` command.

**Currently installed plugins:**
- `feature-dev` - Guided feature development
- `agent-sdk-dev` - Build standalone Claude Agent SDK applications
- `pr-review-toolkit` - Code review and PR analysis
- `commit-commands` - Git commit automation

**Key concept:** Plugins provide either:
1. **Built-in specialized agents** (like debugging experts) that Claude can invoke during conversations
2. **Tools to build your own agents** (Agent SDK) that run independently

---

## Two Types of Agent Tools

### Type 1: Built-In Specialized Agents

**What they are:** Expert agents that Claude invokes during a conversation to help with specific tasks.

**How they work:**
- You ask Claude to solve a problem
- Claude launches a specialized agent (like `pix-debugging-agent`)
- Agent analyzes your code, buffers, shaders, etc.
- Agent returns diagnosis and recommendations
- All happens within your Claude Code session

**Analogy:** Like calling in a specialist consultant for 20 minutes.

### Type 2: Agent SDK (Standalone Agents)

**What it is:** A framework for building **standalone agent applications** that run outside Claude Code.

**How it works:**
- Use `/agent-sdk-dev:new-sdk-app [name]` to create a new agent project
- Write custom tools and workflows (Python or TypeScript)
- Deploy as a standalone application
- Runs independently (can run 24/7, in CI/CD, etc.)

**Analogy:** Like building a permanent automation system for your codebase.

---

## Built-In Debugging Agents (Available Now)

These are specialized agents Claude can invoke to help you debug. You don't need to install anything - they're already available.

### Graphics/Rendering Agents

#### 1. **pix-debugging-agent** ⭐ RECOMMENDED FOR YOUR PROJECT

**Capabilities:**
- Autonomous PIX GPU capture analysis
- DXR performance debugging
- ReSTIR validation
- Buffer inspection (works with your `--dump-buffers` system!)

**When to use:**
- Multi-light system not rendering correctly
- ReSTIR artifacts (dots, color washing)
- Shader logic errors
- Buffer data validation
- Performance profiling

**Example invocation:**
```
Ask Claude: "Use the pix-debugging-agent to diagnose why my multi-light
system isn't showing any visible lighting effects despite correct buffer uploads."
```

**What it does:**
- Reads shader code (HLSL)
- Analyzes buffer structures
- Traces data flow (CPU → GPU)
- Identifies root causes
- Provides specific fixes with file:line references
- Estimates fix time and risk

---

#### 2. **dxr-graphics-debugging-engineer-v2**

**Capabilities:**
- Diagnosing DXR rendering issues
- Missing effects, black frames
- Crashes or performance problems
- DXR pipeline state validation

**When to use:**
- Black screen / no rendering
- DXR crashes or validation errors
- Missing ray tracing effects
- BLAS/TLAS build failures

**Example invocation:**
```
Ask Claude: "Use the dxr-graphics-debugging-engineer-v2 to figure out
why my DXR pipeline is crashing on BLAS build."
```

---

#### 3. **hlsl-volumetric-implementation-engineer-v2**

**Capabilities:**
- Volumetric rendering shader implementation
- Ray marching algorithms
- Lighting integration with DXR
- Beer-Lambert law, phase functions

**When to use:**
- Implementing new volumetric effects
- Ray-ellipsoid intersection issues
- Gaussian splatting shader development
- Volumetric lighting integration

**Example invocation:**
```
Ask Claude: "Use the hlsl-volumetric-implementation-engineer-v2 to help
me implement anisotropic scattering in my Gaussian renderer."
```

---

#### 4. **dxr-systems-engineer-v2**

**Capabilities:**
- DXR pipeline infrastructure design
- State objects, Shader Binding Tables (SBT)
- Raytracing Tier 1.1 upgrades
- Acceleration structure optimization

**When to use:**
- RTXDI integration planning
- Upgrading to newer DXR versions
- Refactoring acceleration structure builds
- Optimizing BLAS/TLAS performance

**Example invocation:**
```
Ask Claude: "Use the dxr-systems-engineer-v2 to plan my RTXDI integration
and identify what infrastructure changes are needed."
```

---

### Performance Agents

#### 5. **physics-performance-agent-v2**

**Capabilities:**
- Physics simulation optimization
- Eliminating GPU stalls
- Scaling particle/metaball systems to high framerates

**When to use:**
- Accretion disk physics becomes bottleneck
- GPU compute shader performance issues
- Need to scale beyond 100K particles
- Optimizing particle updates

**Example invocation:**
```
Ask Claude: "Use the physics-performance-agent-v2 to optimize my
particle physics compute shader - currently limited to 100K particles."
```

---

#### 6. **dx12-mesh-shader-engineer-v2**

**Capabilities:**
- Mesh/Amplification shader pipeline planning
- DXR hybrid rendering
- Mesh shader optimization

**When to use:**
- Issues with mesh shader fallback system
- Planning mesh shader implementation
- Debugging descriptor access issues (like the RTX 40-series bug you had)

**Example invocation:**
```
Ask Claude: "Use the dx12-mesh-shader-engineer-v2 to diagnose why
mesh shaders are failing on RTX 4060 Ti."
```

---

### Research Agents

#### 7. **rt-ml-technique-researcher-v2**

**Capabilities:**
- Researching cutting-edge ray tracing techniques
- ML denoising and neural rendering papers
- AI lighting techniques from recent research

**When to use:**
- Researching RTXDI implementation details
- Planning neural denoising integration
- Understanding latest RT research papers
- Evaluating new rendering techniques

**Example invocation:**
```
Ask Claude: "Use the rt-ml-technique-researcher-v2 to research RTXDI
and summarize the key implementation requirements for my DXR 1.1 pipeline."
```

---

## Agent SDK vs Built-In Agents

### Use **Built-In Agents** when you need:

✅ **Quick diagnosis** of current issues (minutes to hours)
✅ **One-time investigations** during development
✅ **Expert analysis** during active coding sessions
✅ **Immediate help** with bugs or design decisions

**How it works:**
1. Ask Claude to use a specific agent
2. Agent runs within your conversation
3. Returns analysis/recommendations
4. Continue working with Claude

**Example workflow:**
```
You: "My multi-light system isn't rendering. Use pix-debugging-agent."
Claude: [Invokes agent, waits for analysis]
Agent: [Analyzes code, identifies 3 issues with fixes]
Claude: "The agent found the problems. Want me to apply the fixes?"
You: "Yes, apply them."
Claude: [Applies fixes using Edit tool]
```

---

### Use **Agent SDK** when you need:

✅ **Long-running validation** (overnight tests, CI/CD)
✅ **Standalone automation tools** for your team
✅ **Production monitoring** agents (24/7 operation)
✅ **Custom debugging infrastructure** specific to your project

**How it works:**
1. Use `/agent-sdk-dev:new-sdk-app [name]` to scaffold project
2. Define custom tools (run app, capture buffers, parse data, etc.)
3. Write agent logic (autonomous decision-making)
4. Deploy as standalone Python/TypeScript application
5. Runs independently of Claude Code sessions

**Example Agent SDK application for your project:**

**"Multi-Light Validator Agent"**

```
Agent capabilities:
- Runs PlasmaDX-Clean.exe with different configs
- Captures frames at various camera distances
- Dumps GPU buffers (g_lights, g_particles, g_output)
- Parses buffer data and validates:
  ✓ Light count matches config
  ✓ Light radius affects attenuation correctly
  ✓ No lights beyond configured outer radius
  ✓ Illumination visible at expected distances
- Generates regression test report
- Runs on every git commit (CI/CD integration)
```

**Tools this agent would have:**
```typescript
// Tool definitions
{
  name: "run_app_with_config",
  description: "Launch PlasmaDX-Clean.exe with specific config",
  input: { config_path: string, frame_count: number }
}

{
  name: "dump_gpu_buffers",
  description: "Trigger buffer dump via --dump-buffers flag",
  input: { frame: number, dump_dir: string }
}

{
  name: "parse_light_buffer",
  description: "Parse g_lights.bin and extract Light structs",
  input: { buffer_path: string }
}

{
  name: "validate_lighting",
  description: "Check if lighting appears in output at expected positions",
  input: { output_path: string, expected_lights: Light[] }
}

{
  name: "generate_report",
  description: "Create validation report with pass/fail status",
  input: { results: ValidationResult[] }
}
```

**Autonomous workflow:**
```
Agent receives task: "Validate multi-light system with 5 test scenarios"

Agent autonomously:
1. Runs app with config A (close distance, 1 light)
2. Dumps buffers at frame 120
3. Validates light appears in output
4. Repeats for configs B-E (far distance, multiple lights, etc.)
5. Detects regression: "Light disappears at 400+ units in config D"
6. Generates report with screenshots and metrics
7. Files GitHub issue if regression found
```

---

## Your Current PIX Agent System

You already built a **hybrid approach** using Python scripts and buffer dumps:

**What you have:**
- `--dump-buffers` in-app GPU buffer dumping (Application.cpp)
- `analyze_restir_manual.py` for parsing binary buffer data
- `pix_autonomous_agent.py` for orchestration
- Multiple analysis scripts in `PIX/Scripts/analysis/`
- Config-based capture system (`configs/agents/pix_agent.json`)

**Current capabilities:**
- Automatic capture at frame N
- Buffer dumps (g_particles, g_currentReservoirs, g_prevReservoirs, g_rtLighting)
- Binary parsing and statistical analysis
- Hypothesis testing (5 hypotheses for ReSTIR bugs)
- Report generation

**How it compares:**

| Feature | Your Current System | Agent SDK Version |
|---------|---------------------|-------------------|
| Buffer dumps | ✅ Working perfectly | ✅ Same capability |
| Binary parsing | ✅ Python scripts | ✅ More structured |
| Autonomous capture | ⚠️ Manual config edits | ✅ Self-configuring |
| Multi-scenario testing | ⚠️ Requires bash scripts | ✅ Built-in workflows |
| Error recovery | ❌ Script crashes | ✅ Automatic retry |
| Report generation | ✅ Text reports | ✅ Structured reports |
| CI/CD integration | ⚠️ Possible but manual | ✅ Designed for it |
| Maintenance | ⚠️ Scattered scripts | ✅ Unified framework |

**Recommendation:**
Your current system works great for **manual investigation**. Consider Agent SDK when you want **fully autonomous testing** (like running 100 test scenarios overnight and comparing results).

---

## Multi-Light System Analysis (pix-debugging-agent)

**Date:** 2025-10-17
**Task:** Diagnose 3 multi-light system issues in MULTI_LIGHT_FIXES_NEEDED.md

### Agent's Complete Analysis

#### **Issue 1: Light Disappears Beyond ~300-400 Units**

**Root Cause:** Incorrect attenuation formula in shader (too steep falloff)

**Evidence:**
```hlsl
// Line 726 in particle_gaussian_raytrace.hlsl:
float attenuation = 1.0 / (1.0 + lightDist * 0.01);
```

**Why it fails:**
- At `lightDist = 300`: `attenuation = 0.25` (visible)
- At `lightDist = 400`: `attenuation = 0.20` (barely visible)
- At `lightDist = 500`: `attenuation = 0.167` (disappears)

**Fix (Priority 1):**
```hlsl
// shaders/particles/particle_gaussian_raytrace.hlsl:726
// BEFORE:
float attenuation = 1.0 / (1.0 + lightDist * 0.01);

// AFTER:
// Use light.radius for soft falloff
float normalizedDist = lightDist / max(light.radius, 1.0);
float attenuation = 1.0 / (1.0 + normalizedDist * normalizedDist);
```

**Expected impact:** Fixes both Issue 1 AND Issue 2!

---

#### **Issue 2: Light Radius Control Has No Effect**

**Root Cause:** `light.radius` parameter completely unused in shader

**Evidence:**
- Light struct defines `float radius` (line 56-61)
- C++ uploads it correctly (logs confirm)
- Shader hardcodes attenuation constant instead of using radius

**Fix:** Same as Issue 1 (Priority 1 fix addresses both issues)

**After fix:**
- Small radius (10.0) → sharp falloff
- Large radius (100.0) → soft falloff
- Each light independently controllable

---

#### **Issue 3: Can't Disable Original RT Lighting**

**Root Cause:** No toggle for RTLightingSystem, only strength slider

**Current behavior:**
- `m_rtLightingStrength` slider exists (F10/Shift+F10)
- Setting to 0.0 doesn't fully disable RT lighting
- User can't compare pure multi-light vs hybrid

**Fix (Priority 3):**

**Step 3A: Add member variable**
```cpp
// src/core/Application.h:118
bool m_enableRTLighting = true;
```

**Step 3B: Add ImGui checkbox**
```cpp
// src/core/Application.cpp:~1742 (in RT features section)
ImGui::Checkbox("RT Particle-Particle Lighting", &m_enableRTLighting);
ImGui::SameLine();
if (ImGui::Button("?##RTLightingHelp")) {
    ImGui::SetTooltip("Toggle the particle-to-particle RT lighting system.\n"
                      "When OFF, only multi-light system illuminates particles.\n"
                      "Disable to compare pure multi-light vs hybrid.");
}

// Only show strength slider when enabled
if (m_enableRTLighting) {
    ImGui::SliderFloat("RT Lighting Strength (F10)", &m_rtLightingStrength, 0.0f, 10.0f);
} else {
    ImGui::BeginDisabled();
    float zeroStrength = 0.0f;
    ImGui::SliderFloat("RT Lighting Strength (F10)", &zeroStrength, 0.0f, 10.0f);
    ImGui::EndDisabled();
}
```

**Step 3C: Apply toggle**
```cpp
// src/core/Application.cpp:~497 (where gaussianConstants.rtLightingStrength is set)
// BEFORE:
gaussianConstants.rtLightingStrength = m_rtLightingStrength;

// AFTER:
gaussianConstants.rtLightingStrength = m_enableRTLighting ? m_rtLightingStrength : 0.0f;
```

---

#### **Priority 4: Add "RT Primary" Preset (Optional Polish)**

```cpp
// src/core/Application.cpp:~2050 (in multi-light ImGui section)
if (ImGui::Button("RT Primary")) {
    m_lights.clear();
    ParticleRenderer_Gaussian::Light primaryLight;
    primaryLight.position = XMFLOAT3(0, 0, 0); // Black hole center
    primaryLight.color = XMFLOAT3(1.0f, 0.95f, 0.9f); // Warm white
    primaryLight.intensity = 5.0f;
    primaryLight.radius = 50.0f;
    m_lights.push_back(primaryLight);
    m_enableRTLighting = true;
    LOG_INFO("Applied 'RT Primary' preset: 1 light at origin");
}
```

---

### Summary of Fixes

| Priority | Issue | File | Lines | Time | Impact |
|----------|-------|------|-------|------|--------|
| 1 | Light radius not used | `particle_gaussian_raytrace.hlsl` | 726 | 5 min | Fixes Issue 1 & 2 |
| 2 | Tune attenuation range | Same file | 726 | 5 min | Extended reach |
| 3 | Add RT toggle | `Application.h`, `Application.cpp` | 118, 1742, 497 | 15 min | User control |
| 4 | Add presets | `Application.cpp` | 2050 | 10 min | UX polish |

**Total fix time:** 25 minutes
**Rebuild required:** Yes (shader recompilation)
**Risk level:** LOW (all changes are toggleable and reversible)

---

### Validation Plan

1. **Test light radius (Priority 1)**
   - Rebuild shaders
   - Add 1 light at (0, 0, 500)
   - Adjust radius: 10 → 500
   - **Expected:** Light reaches 500+ units when radius > 200

2. **Test attenuation range (Priority 2)**
   - Move light to (0, 0, 800)
   - Set radius to 1000
   - **Expected:** Particles at 300-600 units illuminated

3. **Test RT toggle (Priority 3)**
   - Add 3-5 multi-lights
   - Toggle "RT Particle-Particle Lighting" OFF
   - **Expected:** Only external lights visible, no self-illumination

4. **Test presets (Priority 4)**
   - Click "RT Primary" button
   - **Expected:** 1 light at origin, matches Phase 2.6 appearance

---

### Buffer Dumps: Not Needed

**Agent's assessment:** Buffer dumps not required for these issues.

**Why:**
- Issue 1 & 2 are shader math errors (attenuation formula)
- Issue 3 is UI/application logic (missing toggle)
- Buffer contents confirmed correct (logs show "Updated light buffer: 13 lights")

**When dumps WOULD help:**
- If lights still don't appear after shader fixes
- If attenuation still wrong after formula change
- Validating fixes worked correctly

---

## When to Use Each Agent Type

### Use **pix-debugging-agent** for:
- ✅ Multi-light system issues (shader logic, buffer uploads)
- ✅ ReSTIR debugging (reservoir analysis, weight calculations)
- ✅ Buffer validation (data flow CPU → GPU)
- ✅ Performance profiling with PIX captures
- ✅ Shader logic errors

### Use **dxr-graphics-debugging-engineer-v2** for:
- ✅ Black screen / no rendering
- ✅ DXR crashes or validation errors
- ✅ BLAS/TLAS build failures
- ✅ Ray tracing effects not appearing

### Use **hlsl-volumetric-implementation-engineer-v2** for:
- ✅ New volumetric shader features
- ✅ Ray marching algorithm issues
- ✅ Gaussian splatting improvements
- ✅ Phase function implementation

### Use **dxr-systems-engineer-v2** for:
- ✅ RTXDI integration planning
- ✅ DXR pipeline refactoring
- ✅ Acceleration structure optimization
- ✅ Upgrading to newer DXR tiers

### Use **physics-performance-agent-v2** for:
- ✅ Particle count scaling (beyond 100K)
- ✅ GPU compute shader bottlenecks
- ✅ Accretion disk physics optimization

### Use **rt-ml-technique-researcher-v2** for:
- ✅ Researching RTXDI implementation
- ✅ Neural denoising techniques
- ✅ Latest RT research papers
- ✅ Evaluating new rendering approaches

---

## How to Invoke Agents

### Method 1: Direct Request (Recommended)

Just ask Claude to use a specific agent:

```
You: "Use the pix-debugging-agent to diagnose why my multi-light
system shows no visible lighting."
```

Claude will:
1. Invoke the agent automatically
2. Wait for agent's analysis
3. Present results to you
4. Offer to apply fixes if applicable

---

### Method 2: General Problem Description

Describe your problem, and Claude will choose the appropriate agent:

```
You: "My DXR pipeline is crashing when building the BLAS with 100K particles."

Claude: "I'll use the dxr-graphics-debugging-engineer-v2 to diagnose this crash."
[Invokes agent automatically]
```

---

### Method 3: Multiple Agents in Parallel

For complex issues, Claude can invoke multiple agents simultaneously:

```
You: "I need to optimize my particle physics AND diagnose why
volumetric rendering is slower than expected."

Claude: [Invokes physics-performance-agent-v2 AND pix-debugging-agent in parallel]
```

---

## Future Agent SDK Use Cases

Here are **specific Agent SDK applications** you could build for PlasmaDX:

### 1. **Multi-Light Regression Validator**

**Purpose:** Prevent multi-light system breakage on every commit

**Tools:**
- `run_app(config)` - Launch with specific config
- `dump_buffers(frame)` - Trigger buffer dump
- `parse_lights(buffer)` - Extract Light structs
- `validate_attenuation(lights, output)` - Check expected illumination
- `screenshot(frame)` - Capture visual output
- `compare_images(baseline, current)` - Pixel diff

**Workflow:**
1. Run 10 test scenarios (close/far distance, 1/13 lights, various radii)
2. Dump buffers and screenshots for each
3. Compare against baseline (known-good captures)
4. Generate pass/fail report
5. If failed, file GitHub issue with diff images

**When to run:** On every `git push` (CI/CD integration)

---

### 2. **RTXDI Integration Validator**

**Purpose:** Validate RTXDI integration correctness

**Tools:**
- `run_app(rtxdi_config)` - Launch with RTXDI enabled
- `dump_rtxdi_buffers()` - Dump RTXDI-specific buffers (reservoirs, light grid)
- `validate_reservoir_convergence()` - Check ReSTIR convergence
- `measure_fps(duration)` - Profile performance
- `compare_visual(rtxdi_on, rtxdi_off)` - Visual correctness check

**Workflow:**
1. Run with RTXDI disabled (baseline)
2. Run with RTXDI enabled
3. Validate visual similarity (lighting should be better, not different)
4. Check performance target (>100 FPS @ 100K particles)
5. Test edge cases (1 light, 1000 lights, moving lights)
6. Generate integration report

**When to run:** During RTXDI development, then nightly

---

### 3. **Performance Regression Detector**

**Purpose:** Alert if FPS drops below targets

**Tools:**
- `run_benchmark(config)` - Standardized performance test
- `measure_frame_times()` - Detailed GPU timing
- `compare_baseline(current, historical)` - Trend analysis
- `profile_bottlenecks()` - Identify slowest operations
- `generate_pix_capture()` - Auto-capture if slow

**Workflow:**
1. Run 5 standard benchmark scenarios
2. Measure avg/min/max FPS for each
3. Compare against historical baseline
4. If >10% regression detected:
   - Generate PIX capture
   - Analyze bottleneck (BLAS build? shader? barriers?)
   - File issue with profiling data
5. Track performance trends over time

**When to run:** Nightly, on every tagged release

---

### 4. **Autonomous PIX Debugger v2**

**Purpose:** Fully autonomous GPU debugging (upgrade your current system)

**Tools:**
- `capture_pix(frame)` - Programmatic PIX capture
- `dump_all_buffers()` - Dump g_particles, g_lights, g_reservoirs, g_output
- `parse_buffer(name, format)` - Generic binary parser
- `test_hypothesis(hypothesis)` - Statistical validation
- `edit_shader(file, fix)` - Apply shader fixes
- `verify_fix(before, after)` - Compare results

**Workflow:**
```
Agent receives: "ReSTIR shows dots at close distance"

Agent autonomously:
1. Captures baseline (before fix)
2. Dumps reservoirs
3. Analyzes M/W distribution
4. Forms hypothesis: "Double normalization in MIS weight"
5. Reads shader code
6. Confirms hypothesis
7. Applies fix to shader
8. Rebuilds project
9. Captures after fix
10. Validates dots are gone
11. Generates before/after report with code changes
```

**When to run:** On-demand for complex debugging tasks

---

### 5. **Shader Optimization Advisor**

**Purpose:** Suggest shader optimizations based on PIX data

**Tools:**
- `analyze_shader_occupancy()` - GPU utilization
- `identify_divergent_branches()` - Warp efficiency
- `measure_memory_bandwidth()` - Bandwidth bottlenecks
- `suggest_optimizations()` - AI-based recommendations
- `apply_optimization(suggestion)` - Automated refactoring
- `benchmark_improvement()` - Measure speedup

**Workflow:**
1. Profile current shaders with PIX
2. Analyze GPU metrics (occupancy, bandwidth, etc.)
3. Identify bottlenecks
4. Generate optimization suggestions
5. Apply safest optimizations automatically
6. Benchmark improvement
7. Report results with before/after comparisons

**When to run:** Monthly, or when targeting new performance goals

---

## Quick Reference

### Installed Plugins

```bash
# List installed plugins
/plugin

# Install a new plugin
/plugin install [plugin-name]

# Update plugin
/plugin update [plugin-name]
```

---

### Built-In Debugging Agents

| Agent | Primary Use Case | Typical Invocation |
|-------|-----------------|-------------------|
| `pix-debugging-agent` | Multi-light, ReSTIR, buffers | "Use pix-debugging-agent to diagnose..." |
| `dxr-graphics-debugging-engineer-v2` | DXR crashes, black screens | "Use dxr-graphics-debugging-engineer-v2..." |
| `hlsl-volumetric-implementation-engineer-v2` | Volumetric shaders | "Use hlsl-volumetric-implementation-engineer-v2..." |
| `dxr-systems-engineer-v2` | RTXDI, DXR infrastructure | "Use dxr-systems-engineer-v2 to plan..." |
| `physics-performance-agent-v2` | Particle physics optimization | "Use physics-performance-agent-v2..." |
| `dx12-mesh-shader-engineer-v2` | Mesh shader issues | "Use dx12-mesh-shader-engineer-v2..." |
| `rt-ml-technique-researcher-v2` | Research RTXDI, neural techniques | "Use rt-ml-technique-researcher-v2..." |

---

### Agent SDK Commands

```bash
# Create new Agent SDK application
/agent-sdk-dev:new-sdk-app [project-name]

# Verify Agent SDK app (Python)
# (Claude will invoke agent-sdk-verifier-py automatically)

# Verify Agent SDK app (TypeScript)
# (Claude will invoke agent-sdk-verifier-ts automatically)
```

---

### Your Current PIX Agent System

**Quick start:**
```bash
# Single capture with auto-dump at frame 120
./build/Debug/PlasmaDX-Clean.exe --dump-buffers 120 --gaussian

# Analyze dumped buffers
python PIX/Scripts/analysis/analyze_restir_manual.py \
    --current PIX/buffer_dumps/g_currentReservoirs.bin \
    --prev PIX/buffer_dumps/g_prevReservoirs.bin
```

**Key files:**
- Config: `configs/agents/pix_agent.json`
- Implementation: `src/core/Application.cpp` (lines 990-1171)
- Analysis: `PIX/Scripts/analysis/analyze_restir_manual.py`
- Documentation: `PIX/PIX_AUTONOMOUS_AGENT.md`

---

### Decision Tree: Which Agent/Tool to Use?

```
Problem: Graphics rendering issue
├─ Black screen / crashes
│  └─ Use: dxr-graphics-debugging-engineer-v2
├─ Shader logic errors (colors, lighting wrong)
│  └─ Use: pix-debugging-agent
├─ Need new volumetric effect
│  └─ Use: hlsl-volumetric-implementation-engineer-v2
└─ Planning major refactor (RTXDI)
   └─ Use: dxr-systems-engineer-v2

Problem: Performance too slow
├─ Particle physics bottleneck
│  └─ Use: physics-performance-agent-v2
├─ Shader/GPU bottleneck
│  └─ Use: pix-debugging-agent
└─ General optimization
   └─ Use: dxr-systems-engineer-v2

Need: Research/planning
├─ RTXDI implementation
│  └─ Use: rt-ml-technique-researcher-v2
├─ Neural denoising
│  └─ Use: rt-ml-technique-researcher-v2
└─ Feature planning
   └─ Use: /feature-dev:feature-dev

Need: Standalone automation
├─ CI/CD testing
│  └─ Use: Agent SDK
├─ Nightly validation
│  └─ Use: Agent SDK
└─ Production monitoring
   └─ Use: Agent SDK
```

---

## Best Practices

### 1. **Use Built-In Agents for Immediate Problems**

Don't try to build an Agent SDK app when a built-in agent can solve it in 20 minutes.

**Good:** "Use pix-debugging-agent to find why lights aren't visible"
**Bad:** "Let me build an Agent SDK app to debug this one issue"

---

### 2. **Use Agent SDK for Repeated Tasks**

If you're running the same test workflow 10+ times, automate it with Agent SDK.

**Good:** Building "Multi-Light Validator" that runs on every commit
**Bad:** Manually testing 10 scenarios every time you change shader code

---

### 3. **Let Claude Choose the Agent**

You don't need to know which agent to use - just describe the problem.

**Good:** "My DXR pipeline crashes on BLAS build"
Claude: [Invokes dxr-graphics-debugging-engineer-v2 automatically]

**Also Good:** "Use pix-debugging-agent to analyze this" (if you know which you want)

---

### 4. **Agents Can Work in Parallel**

For complex issues, agents can run simultaneously:

**Good:** "Optimize physics AND debug rendering issues"
Claude: [Invokes physics-performance-agent-v2 + pix-debugging-agent in parallel]

---

### 5. **Agents Return to You for Decisions**

Agents analyze and recommend, but **you** decide what to do.

**Example:**
1. Agent: "I found 3 issues. Priority 1 takes 5 min, Priority 3 takes 15 min."
2. You: "Apply Priority 1 only, skip the rest."
3. Claude: [Applies just Priority 1 fix]

---

### 6. **Save Agent Reports**

Agent analyses are valuable - save them for future reference.

**Good practice:**
- Agent completes analysis
- Ask Claude: "Save this analysis to PIX/reports/multi_light_diagnosis_20251017.md"
- Reference it later when implementing fixes

---

## Common Workflows

### Workflow 1: Debug Rendering Issue

```
1. You notice problem (e.g., "lights not visible")
2. Describe to Claude: "Multi-light system shows no lighting"
3. Claude invokes: pix-debugging-agent
4. Agent analyzes: shaders, buffers, data flow
5. Agent returns: Root causes + fixes
6. You review: "Apply Priority 1 and 3, skip 2 and 4"
7. Claude applies: Selected fixes
8. You validate: Rebuild and test
9. If fixed: Done!
10. If not fixed: Agent suggests buffer dumps for deeper analysis
```

---

### Workflow 2: Plan Major Feature (RTXDI)

```
1. You decide: "Ready to integrate RTXDI"
2. Ask Claude: "Use rt-ml-technique-researcher-v2 to research RTXDI"
3. Agent researches: Papers, NVIDIA docs, sample code
4. Agent returns: Implementation requirements
5. Ask Claude: "Use dxr-systems-engineer-v2 to plan integration"
6. Agent plans: Infrastructure changes needed
7. You review: Approve plan or request adjustments
8. Claude helps: Implement step-by-step
9. Later: Build Agent SDK "RTXDI Validator" for ongoing testing
```

---

### Workflow 3: Performance Optimization

```
1. You notice: FPS dropped from 140 to 100
2. Describe to Claude: "Performance regression, was 140 FPS, now 100"
3. Claude invokes: pix-debugging-agent (for profiling)
4. Agent analyzes: GPU timings, bottlenecks
5. Agent identifies: BLAS rebuild taking 2.1ms → 3.5ms
6. Ask Claude: "Use dxr-systems-engineer-v2 to optimize BLAS"
7. Agent suggests: BLAS update (not rebuild), instance culling
8. You decide: "Implement BLAS update only"
9. Claude implements: BLAS update logic
10. You validate: FPS back to 140
```

---

## Next Steps

### Immediate (Today)

1. **Apply multi-light fixes** from pix-debugging-agent analysis:
   - Priority 1: Fix light radius (5 min)
   - Priority 3: Add RT lighting toggle (15 min)

2. **Validate fixes work:**
   - Test light radius slider
   - Test RT toggle checkbox

---

### Short Term (This Week)

1. **Use agents for RTXDI research:**
   - Ask: "Use rt-ml-technique-researcher-v2 to summarize RTXDI implementation"
   - Ask: "Use dxr-systems-engineer-v2 to plan RTXDI integration"

2. **Explore feature-dev plugin:**
   - Try: `/feature-dev:feature-dev` for guided feature development

---

### Medium Term (This Month)

1. **Consider Agent SDK** if you find yourself:
   - Running same test scenarios repeatedly
   - Wishing for automated validation
   - Wanting CI/CD integration

2. **Build first Agent SDK app:**
   - Start with: "Multi-Light Validator" (simplest)
   - Use: `/agent-sdk-dev:new-sdk-app multi-light-validator`

---

## Resources

### Documentation in Your Project

- `PIX/PIX_AUTONOMOUS_AGENT.md` - Your current PIX agent system
- `PIX_AGENT_V4_DEVELOPMENT_PLAN.md` - Agent development roadmap
- `MULTI_LIGHT_FIXES_NEEDED.md` - Current issues (analyzed by pix-debugging-agent)
- `CLAUDE.md` - Project guidance for Claude Code

---

### Ask Claude

You can always ask Claude about plugins and agents:

```
"What plugins do I have installed?"
"How do I use the pix-debugging-agent?"
"Should I use Agent SDK for this task?"
"Which agent is best for DXR crashes?"
```

Claude will guide you to the right tool for your specific problem.

---

**Document Version:** 1.0
**Last Updated:** 2025-10-17
**Purpose:** Complete reference for Claude Code plugin system and debugging agents
**Audience:** Developer with no prior plugin experience
**Related Files:** `MULTI_LIGHT_FIXES_NEEDED.md`, `PIX/PIX_AUTONOMOUS_AGENT.md`, `CLAUDE.md`