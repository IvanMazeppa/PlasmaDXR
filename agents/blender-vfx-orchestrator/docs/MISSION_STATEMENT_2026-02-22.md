# Blender VFX Orchestrator — Mission Statement

**Version:** 2.0
**Last Updated:** 2026-02-21
**Status:** Canonical reference for all development decisions

---

## What This Document Is

This is the single source of truth for what this project is, what it does, how it works, and why every architectural decision exists. Any AI agent, human contributor, or future version of this system should read this document first and treat it as the ultimate tiebreaker when design choices conflict.

This document was created through a structured interview process capturing the creator's vision, validated against 188+ pipeline runs, and informed by research into the OpenAI Agents SDK, autonomous multi-agent patterns, and Blender 5.0's Python API.

---

## 1. The Vision

Build an **autonomous, adaptive AI system** that allows anyone — regardless of 3D graphics experience — to create high-quality volumetric VFX assets in Blender using natural language.

The user describes what they want: *"a wine glass being filled with red wine on a marble table."* The system researches how to build it, writes a complete Blender Python script that creates every element from scratch (geometry, materials, physics, lighting, camera, environment), executes it, evaluates the result, and iterates until the quality threshold is met — or honestly reports what went wrong and what it plans to try next.

**The system should behave like a skilled VFX artist who happens to have perfect recall:** creative when given novel challenges, reliable when repeating known techniques, and honest about what it doesn't know. It produces entire scenes — not just physics simulations. A wine pour prompt doesn't just create liquid dynamics; it creates the glass (with proper refraction material), the table (with marble surface), ambient set dressing, three-point lighting, and camera composition. The LLM's creative ability to "run with an idea" and produce a complete, coherent scene is the core value proposition of this system.

### What Makes This Different

This is not a parameter tuner, template filler, or Mantaflow wrapper. This system:

- **Creates complete scenes from scratch** — every mesh, material, physics setup, light, and camera is generated procedurally in a single Python script
- **Handles any visual effect Blender supports** — Mantaflow gas/liquid, rigid body destruction, particle systems, cloth simulation, geometry nodes, and combinations
- **Learns from every run** — successful patterns are captured and validated; failures are diagnosed and avoided
- **Uses LLM intelligence where it creates real value** — creative scene composition, research, code generation, visual critique — while keeping deterministic operations deterministic
- **Progresses from guided to autonomous** — earning independence through demonstrated reliability, not calendar time

### Proof It Works

The system has produced renders that demonstrate the vision is achievable:

- **Wine pour** (`images/wine_render.png`): A procedurally-generated wine glass with correct glass refraction, red liquid Mantaflow simulation, marble table surface, warm lighting — all created from a single generated Blender Python script. Quality score: 70/100.
- **Kitchen leak** (`kitchen_leak` traces): Mantaflow liquid flowing from a faucet into a sink cavity, with environment geometry, proper materials, and camera framing. Best quality score: 58/100.

These results prove the system CAN create high-quality, complete scenes. The challenge is making it do so **reliably and consistently**.

---

## 2. What This Project Is NOT

- **Not a parameter tuner.** The system generates full Blender Python scripts, not JSON configs for a template. Templates and parameter schemas may exist as accelerators for known patterns, but they are not the primary mode of operation.
- **Not a Mantaflow-only tool.** Mantaflow is one physics system among many. A user prompt like "a window getting smashed" should route to rigid body physics, "sparks from a grinder" to particle systems, "a flag waving in the wind" to cloth simulation. The architecture must not assume fluid simulation is the only kind of VFX.
- **Not a deterministic pipeline with LLM decoration.** The LLM agents are creative problem-solvers. The wine glass render succeeded because the LLM made dozens of creative decisions (glass shape, material properties, table placement, liquid color, lighting mood). Replacing that creativity with hardcoded templates would destroy the system's core value.
- **Not a one-shot generator.** Iteration is expected and designed for. The system improves renders through multiple passes, learning from each attempt. Even "failed" runs (low quality scores) produce valuable learning signal.

---

## 3. How It Works: End-to-End User Experience

### Step 1: The User Writes a Prompt

The user describes what they want in natural language. This can be brief:

> "A candle flame on a table"

### Step 2: Prompt Enhancement

Before the pipeline begins, a prompt enhancement layer expands the brief description into a structured VFX specification using the [Prompt Enhancement Guide](PROMPT_ENHANCEMENT_GUIDE.md). An LLM takes the short prompt and produces a rich scene description covering:

- **Scene description** — What is physically happening, what objects exist, what motion occurs
- **Mood and look** — Atmosphere, lighting quality, color palette, emotional tone
- **Camera** — Shot size, angle, movement, depth of field
- **Physical details** — Dimensions, materials, surface properties, physics parameters
- **Motion/timing** — Slow motion behavior, temporal arc
- **Hard constraints** — Physics system, renderer, frame range, resolution, cache type

The enhanced prompt transforms "a candle flame on a table" into a complete creative brief that reads like a cinematographer's shot description — specifying the candle's dimensions, flame structure (blue core, yellow body, orange tips), wax pool characteristics, table material, lighting mood, camera framing, and simulation parameters.

### Step 3: Human Approval

The user reviews the enhanced prompt and approves, modifies, or redirects before any API credits are spent. This is the primary HITL checkpoint.

### Step 4: The Pipeline Runs

The orchestrator coordinates a team of specialized sub-agents through an iterative cycle:

1. **Research** — The system researches how to build what was requested, consulting documentation, the knowledge base, and (where available) Blender's own API
2. **Technique Selection** — Based on research, the system selects appropriate Blender physics/generation systems
3. **Script Generation** — A code-writing agent produces a complete Blender Python script (typically 500-1000+ lines) that creates the entire scene from scratch
4. **Validation** — The script is checked for known API errors before execution
5. **Execution** — The script runs in headless Blender, baking physics and rendering frames
6. **Quality Evaluation** — ML metrics and vision-based analysis evaluate the render
7. **Learning** — Results (good and bad) are recorded in the knowledge base
8. **Decision** — Pass (score >= 60, no critical issues) or iterate with specific feedback

Each iteration takes 5-30 minutes depending on simulation complexity. The system runs up to N iterations (configurable), improving with each pass.

### Step 5: The User Gets Results

- **Rendered frames** (.png) for immediate visual assessment
- **The Blender Python script** that created the scene (reusable, editable)
- **The .blend file** for manual refinement in Blender
- **Quality metrics** showing what the system thinks of its own output
- **Iteration history** showing how the system improved (or didn't)
- *(Future)* **Cache data** for the PlasmaDXR volumetric renderer pipeline

---

## 4. The Creative Pipeline

### What Each Phase Does

#### Research Phase
The system investigates how to build what was requested. This is NOT a formality — it is one of the most important phases, and intelligence is required here. The research agent consults:

- **Blender manual** — How do the relevant physics systems, materials, and rendering features actually work? What workflows exist? What techniques are available? What parameter combinations produce specific visual effects?
- **Blender Python API reference** — What attributes exist on the relevant `bpy.types` classes? What are their types, valid ranges, and defaults?
- **Knowledge base** — Have we successfully built something similar before? What worked? What failed?
- **API introspection** — Runtime truth pack data confirming what actually exists in the running Blender binary (see Section 6)

**Critical: Both the manual AND the API reference are essential, but they serve fundamentally different purposes:**

The **API reference** tells you WHAT exists — `FluidDomainSettings.burning_rate` is a float between 0.01 and 4.0. The **manual** tells you WHY, WHEN, and HOW — when to use fire vs smoke vs a hybrid, what burning_rate values produce a candle flame vs a roaring bonfire, how noise strength interacts with resolution, and what alternative techniques exist beyond the most obvious one.

**Without the manual, the agent cannot learn new techniques.** It will default to whatever its training data suggests — which means the same handful of approaches for every prompt, regardless of what Blender actually supports. This is a known failure mode (see Section 12: "Technique monotony"). The research agent MUST be able to search the manual for conceptual understanding, workflow tutorials, and technique comparisons — not just API signatures.

**The manual should be optimized for LLM consumption.** The raw Blender manual is written for humans — it contains UI navigation instructions, screenshot descriptions, and verbose explanations optimized for human readers. For the research agent, this content should be processed into a denser, more structured format that maximizes information per token: factual content, parameter relationships, technique comparisons, and workflow steps — without the UI-centric padding.

The research agent's output should likewise be **dense and structured for LLM consumption**, not verbose human-readable prose. Context bloat from research output is a known problem that directly degrades downstream agent performance.

#### Technique Selection
Based on research, the system selects which Blender systems to use. This must be **prompt-driven, not default-driven**:

| Prompt Signal | Technique | NOT This |
|--------------|-----------|----------|
| fire, flame, smoke, explosion | Mantaflow Gas | Always Mantaflow |
| water, pour, splash, liquid | Mantaflow Liquid | Always Mantaflow |
| shatter, smash, destroy, collapse | Rigid Body | Mantaflow |
| sparks, debris, rain, snow, dust | Particle System | Mantaflow |
| fabric, cloth, flag, curtain | Cloth Simulation | Mantaflow |
| procedural, scatter, growth | Geometry Nodes | Mantaflow |
| Complex scenes | **Combinations** | Single system only |

"A flag waving in the wind near an explosion" requires cloth + Mantaflow gas + particle debris. The system must support multi-technique scenes.

#### Script Generation
The code writer produces a complete Blender Python script. This is where LLM creativity produces the most value. The script creates:

- Scene setup (clearing defaults, setting renderer, frame range)
- **Object geometry** — Meshes for every scene element, created with Blender primitives (add cube, add cylinder, etc.) shaped through transforms, extrusions, modifiers
- **Materials** — Principled BSDF configurations, volume shaders, glass materials with proper IOR, subsurface scattering, etc.
- **Physics setup** — Domain configuration, flow sources, effectors, bake settings
- **Environment** — Background elements, set dressing, ground planes/tables/surfaces
- **Lighting** — Key, fill, and rim lights with appropriate energy, color temperature, and positioning for the scene's mood
- **Camera** — Positioned and aimed to frame the subject as described in the prompt
- **Render settings** — Resolution, samples, denoising, output path
- **Bake and render commands** — Physics simulation bake followed by render

**Script quality indicator:** Scripts under 500 lines are generally too basic — they produce oversimplified scenes that lack the detail needed for quality renders. Good scripts are 500-700 lines. Excellent scripts that produce the best results are 700-1000+ lines.

#### Execution
The script runs in headless Blender 5.0 on WSL2. This is a **deterministic subprocess operation** — no LLM intelligence is needed to run a Python script. The executor manages:

- Setting up the Blender subprocess with correct environment variables
- Capturing stdout/stderr for error diagnosis
- Managing output directories and file paths
- Timeout enforcement
- Parsing execution results (success/failure, output files produced)

#### Quality Evaluation
A multi-signal evaluation combining:

- **ML metrics** (CLIP, LPIPS, TOPIQ) — Objective structural and perceptual quality
- **Vision-based critique** (OpenAI vision model) — Semantic analysis of what's in the render, composition quality, lighting assessment, identification of problems

The vision analyst produces excellent critiques but has a critical limitation: **it cannot see the script code**. It can identify that a render is too dark, but it cannot tell you that the cause is a missing light object at line 342. This means its improvement suggestions must be treated as **symptoms, not diagnoses**. The system should pair visual critique with script analysis for accurate root cause identification.

**Output format:** All evaluation output must be structured, dense, and LLM-optimized — not verbose human-readable paragraphs. Context bloat from evaluation output directly degrades the next iteration's code writing quality.

#### Learning
Every run — successful or failed — produces signal. The system must:

- **Record** what parameters, techniques, and approaches were used
- **Record** what worked (quality scores, successful patterns) and what failed (errors, hallucinated attributes, quality issues)
- **Validate** learnings through repeated evidence before promoting them to trusted knowledge (evidence-gating)
- **Decay** stale or unreinforced knowledge over time (Ebbinghaus-inspired memory management)
- **Apply** validated knowledge to future runs via dynamic instructions

The knowledge base must be curated, not a dumping ground. A bloated KB with stale or wrong patterns will actively hurt performance by overwhelming agents with irrelevant information.

---

## 5. Where Intelligence Lives

**Core principle:** Use LLM intelligence where it creates real value (creative reasoning, natural language understanding, code generation, visual critique). Use deterministic code where the task is computable (execution, validation, camera geometry, budget tracking).

### Needs LLM Intelligence

| Task | Why Intelligence is Required |
|------|------------------------------|
| **Prompt enhancement** | Understanding natural language intent, making creative decisions about scene composition, mood, lighting |
| **Research** | Reading and synthesizing documentation, understanding which techniques apply to a given effect, comparing approaches |
| **Script writing** | Generating creative, complete Blender Python that creates an entire scene — geometry, materials, physics, environment. This is the highest-value use of LLM intelligence in the system. |
| **Visual analysis** | Looking at a render and understanding what's good, what's wrong, and what could be improved from a compositional/aesthetic perspective |
| **Error diagnosis** | Reading a Python traceback in context, understanding what went wrong, and suggesting a specific fix |
| **Learning/cataloguing** | Interpreting outcomes (good and bad), extracting reusable patterns, deciding what's worth remembering |

### Should Be Deterministic

| Task | Why It Should Be Computed |
|------|--------------------------|
| **Script execution** | Running `blender --background --python script.py` is subprocess management, not reasoning |
| **API validation** | Checking attribute names against a truth pack is pattern matching, not understanding |
| **Camera placement verification** | Ensuring camera is outside scene bounds and pointed at the subject is geometry |
| **Light energy bounds** | Clamping values to sane per-effect ranges is arithmetic |
| **Budget tracking** | Counting tokens and dollars is accounting |
| **Quality threshold comparison** | `score >= 60` is a comparison, not a judgment call |
| **Circuit breakers** | "Same error 3 times = stop retrying" is counting |
| **Collision effector injection** | For liquid scenes, ensuring containment geometry has effector modifiers is rule-following |

### Strategic Decision: What Agents Exist

The system uses a team of specialized sub-agents coordinated by an orchestrator. Each agent has a specific role, a set of tools, and produces structured output. The exact agent lineup is an architectural decision that should be informed by the needs above and the Agents SDK's capabilities — but the principle is clear: **agents for creative/reasoning tasks, functions for computable tasks**.

The current agent roster includes: Research Agent, Documentation Expert, Technique Selector, API Spec Agent, Script Writer/Code Writer, Executor, Quality Analyst, Learning Agent, Quality Gate Judge, and Modification Strategist. Whether all of these need to be LLM-powered agents vs deterministic functions is an active design question — but the key creative agents (Research, Script Writer, Visual Analyst) must retain their intelligence.

---

## 6. Anti-Hallucination Strategy

### The Problem

Over 188 runs, the #1 failure mode has been **LLMs hallucinating deprecated Blender 4.x attributes** — using `resolution_divisions` (which doesn't exist) instead of `resolution_max` (which does). A 57-rule regex fixer was built to catch these, but:

- It's reactive (fixes after generation, not before)
- It's fragile (every new hallucination needs a new rule)
- It can't keep up (the LLM invents new wrong attributes faster than rules can be added)
- It made the system rigid (trying new techniques inevitably hits unmapped attributes)

### The Solution: Truth Pack + Validation

**Layer 1: PREVENT — Runtime Introspection (Truth Pack)**

Before script generation, interrogate the actual running Blender binary for ground truth:

```python
# Runs INSIDE Blender headless — returns every valid attribute with types, ranges, defaults
props = bpy.types.FluidDomainSettings.bl_rna.properties
for name, prop in props.items():
    print(f"{name}: type={prop.type}, range=[{prop.hard_min}, {prop.hard_max}]")
```

This "truth pack" is injected into the script writer's context, giving it the ONLY valid reference for attribute names. The LLM receives a complete list of valid attributes — if `resolution_divisions` isn't in the list, it cannot use it. Cost: $0.00 (Blender subprocess, no LLM).

The truth pack is built dynamically based on which `bpy.types` the current scene needs (determined during research/technique selection). For novel techniques, the system can discover available types by inspecting `dir(bpy.types)` — a catalog of everything Blender offers.

**Layer 2: DETECT — Static Validation**

After script generation, validate every attribute access against the truth pack before sending the script to Blender. This catches hallucinations before a 30-second Blender execution fails. Cost: $0.00 (regex/AST matching).

**Layer 3: CORRECT — Deterministic Substitution**

When validation catches an invalid attribute, use `difflib.get_close_matches()` against the truth pack to suggest corrections (e.g., `resolution_divisions` → `resolution_max`). Apply automatically. Cost: $0.00.

**Layer 4: The Existing Regex Fixer**

The 57-rule fixer remains as a last-resort safety net for patterns that slip through layers 1-3. But it should shrink over time as the truth pack handles more cases.

### Why Not Just Fix the Fixer?

Adding more regex rules is an infinite arms race. The truth pack approach is finite: Blender has a fixed number of attributes, and `bl_rna.properties` enumerates all of them. One introspection call replaces hundreds of regex rules.

---

## 7. Quality and Learning

### Quality Evaluation

A render **passes** when:
- `overall_score >= 60` (0-100 scale)
- No critical issues present

**Critical issues (auto-fail regardless of score):**
- `ZERO_LIGHTS_ACTIVE` — No lighting in scene
- `BLACK_SCREEN` — Completely dark render
- `WHITE_SCREEN` — Completely overexposed
- `CLIPPING_ARTIFACTS` — Volume clipping at boundaries
- `CAMERA_INSIDE_GEOMETRY` — Camera placed inside an object

**Known quality evaluation limitations:**
- The visual analyst sees the render image but NOT the script code. Its suggestions about what to fix are based on symptoms ("too dark") not causes ("missing Area Light at line 342"). The system must combine visual critique with script analysis for accurate diagnosis.
- Verbose evaluation output causes context bloat. All evaluation must be structured and concise — optimized for LLM consumption, not human reading.

### Learning Architecture

**Evidence-gating:** Nothing enters trusted knowledge without proof.

| Trust Level | Criteria | Usage |
|-------------|----------|-------|
| **Untrusted** | Single occurrence | Recorded for analytics only |
| **Emerging** | 2 occurrences, mixed outcomes | Available but flagged as experimental |
| **Trusted** | 3+ successful uses | Injected into dynamic instructions |
| **Deprecated** | Success rate < 20% after 10+ uses | Actively excluded from instructions |

**Memory decay:** Knowledge entries that are not reinforced by successful outcomes should naturally lose weight over time. This prevents the KB from growing unbounded and overwhelming agents with stale patterns.

**What to learn:**
- Which techniques work for which effect types
- Camera placement parameters that produce good framing per effect type
- Light energy ranges that produce good exposure per effect type
- Specific parameter values that produce quality results
- Common failure patterns and their root causes
- Script patterns/code snippets from successful runs

**What NOT to learn from:**
- Single successes (one lucky run isn't a pattern)
- LLM opinions about why something worked (not evidence)
- Partial successes (a score of 45 doesn't mean the parameters are "close")

---

## 8. Experimentation and Exploration

### The Exploration Principle

The system should actively explore Blender's capabilities — running diagnostic scripts, inspecting APIs, testing small code snippets — just as a human developer would. If the system encounters a novel prompt, it should research and experiment, not fall back to the only technique it knows.

### Sandbox Mode

Between production runs (or as part of a run when stuck), the system should be able to:

- **Run small experimental scripts** — Test a specific technique, material, or parameter in isolation before committing to a full scene
- **Introspect Blender capabilities** — Discover what types, modifiers, and physics systems are available
- **Record experimental findings** — Document what was tried, what happened, and what was learned
- **Feed findings into the knowledge base** — Successful experiments become available for future production runs

This sandbox mode can run between real tests, after failed runs, or proactively when the system detects it's about to attempt something novel. The goal is to build practical experience with Blender's capabilities through hands-on experimentation, not just documentation reading.

### Exploration Triggers

- Novel prompt that doesn't match any known technique pattern
- Escape velocity level 2+ (stuck on same issue repeatedly)
- New Blender version installed (truth pack changes detected)
- User-requested exploration ("experiment with cloth simulation")

---

## 9. Human Collaboration

### Autonomy Spectrum

The system operates on a spectrum from fully guided to fully autonomous, with the human always able to step in. As reliability improves, the human steps back. But the human can always step forward.

### HITL Checkpoints

| Checkpoint | When | What the Human Sees |
|-----------|------|---------------------|
| **Prompt approval** | After enhancement, before pipeline starts | Enhanced prompt for review |
| **Stall detection** | 3+ iterations with no score improvement | Score history, last render, suggested actions |
| **Budget warning** | Cumulative cost exceeds threshold | Cost breakdown, quality trajectory |
| **Critical issue** | BLACK_SCREEN, WHITE_SCREEN, camera failure | Render image, diagnosis |
| **Escalation** | System is stuck and has exhausted its strategies | Full diagnostic report, request for guidance |

### Monitoring

The runs are long (5-30 minutes per iteration) and the user often steps away. The system must be self-monitoring. Ideas being explored:

- **A monitoring agent/layer** that watches pipeline progress and can detect early failure signals (quality analyst's incorrect feedback loop, tool overuse, context bloat, oscillating parameters)
- **Leveraging SDK capabilities** like RunHooks for lifecycle monitoring, streaming for real-time progress visibility, and native HITL approval gates
- **Artifact-based status reporting** — write structured status updates to files rather than inline context, so a human (or monitoring agent) can check progress without reading full traces

### Early Failure Signals

From practical experience, these indicate a run is heading for failure:

| Signal | What It Means | When Detected |
|--------|--------------|---------------|
| Script < 500 lines | Scene too basic, lacks detail | After generation |
| Black/white render | Camera, lighting, or material failure | After first render |
| Quality analyst suggests wrong fix | It sees symptoms, not causes — feedback loop broken | After evaluation |
| Same error 3+ times | System is stuck in a loop | Across iterations |
| Parameter oscillation | Overcorrection (e.g., light energy 50 → 2500 → 10) | Across iterations |

---

## 10. Autonomy Progression

The system earns autonomy through demonstrated reliability:

| Level | Requirement | What Changes |
|-------|-------------|--------------|
| **0: Guided** | Default | Human approves prompt, sees every evaluation, all HITL active |
| **1: Assisted** | 10+ runs for this effect type, some passes | Prompt approval can be skipped for known types, stall detection still active |
| **2: Semi-Autonomous** | Pass rate > 50% across 3+ effect types | Only budget and critical issue checks. System selects techniques independently. |
| **3: Autonomous (Known)** | Pass rate > 70% for this specific effect type, 50+ runs | Pipeline runs to completion. Human sees only final result. |
| **4: Autonomous (Novel)** | Stable Level 3 + external review | System discovers new techniques, creates new knowledge, explores without prompting |

**Autonomy is earned by evidence, not time.** A system that reliably produces quality fire effects but has never attempted rigid body physics is Level 3 for fire and Level 0 for destruction.

---

## 11. Technical Foundations

### OpenAI Agents SDK (v0.9.0+)

The system is built on the OpenAI Agents SDK. Key capabilities currently used or planned:

| Capability | Status | Purpose |
|-----------|--------|---------|
| **Agents-as-tools** | In use | Sub-agents called as tools by the orchestrator, control returns to caller |
| **Dynamic instructions** | In use | Runtime instruction injection based on pipeline state (escape level, budget, known errors) |
| **RunHooks** | In use | Lifecycle monitoring, loop detection, doc query enforcement |
| **Structured outputs** | In use | Pydantic models ensure parseable agent responses |
| **Tracing** | In use | Pipeline execution traces for debugging and analysis |
| **Sessions** | Planned | `AdvancedSQLiteSession` for branching (try different techniques from same point), token tracking, persistence |
| **Tool guardrails** | Planned | Deterministic API validation as tool guardrails, zero LLM cost |
| **tool_use_behavior** | Planned | `stop_on_first_tool` for deterministic agents; `StopAtTools` for quality gate |
| **Conditional tool enabling** | Planned | Hide expensive tools when budget is low (`is_enabled` callback) |
| **HITL (needs_approval)** | Planned | Native approval gates with state serialization for pipeline pauses |
| **Streaming** | Planned | Real-time monitoring of long-running agent operations |
| **call_model_input_filter** | Planned | Trim conversation history to prevent context bloat in long iteration sessions |
| **Parallelism** | Partial | `asyncio.gather()` for independent operations; `parallel_tool_calls` in ModelSettings |

### Models

| Model | Role | Why |
|-------|------|-----|
| **gpt-5.2** | Script writer, coordinator | Best creative reasoning and code generation |
| **gpt-5-mini** | Research, diagnosis, evaluation | Good reasoning at lower cost |
| **gpt-5.3-codex** | Script writing (alternative) | Specialized for code generation |
| **o3** | Complex debugging (when stuck) | Deep reasoning for hard problems |
| **o4-mini** | Fast decisions, technique selection | Quick structured reasoning |

### Blender 5.0

- Headless execution on WSL2 via `blender --background --python script.py`
- Cycles GPU renderer
- Mantaflow for fluid simulation (gas and liquid)
- Rigid body, particle systems, cloth, geometry nodes for other effect types
- **Known gotchas documented in CLAUDE.md** — `resolution_max` not `resolution_divisions`, `ShaderNodeMix` not `ShaderNodeMixRGB`, etc.

### Budget

**Monthly limit:** $20 for all API calls.

This budget constraint is a first-class design concern, not an afterthought. Every LLM call must justify its existence. Model routing (expensive model for creative tasks, cheap model for structured decisions) and deterministic alternatives (truth pack validation, camera geometry computation) directly serve the budget constraint.

**Target cost per run:** ~$0.20-0.50 for a typical 3-iteration run. This enables 40-100 runs per month — enough for meaningful iteration and learning.

---

## 12. Current State (Honest Assessment)

### What Works
- The system CAN produce impressive, complete scenes (wine pour: 70/100)
- Prompt enhancement guide produces well-structured scene descriptions
- Multi-agent pipeline architecture is sound in concept
- Quality evaluation (ML metrics + vision critique) provides useful signal
- Tracing and analysis tools provide visibility into pipeline behavior
- The OpenAI Agents SDK is the right foundation

### What's Broken
- **Reliability is too low** — 188 runs, 0 formal passes at the 60/100 threshold. Best scores: 58, 18, 0 in the S1.5 benchmark.
- **Hallucinated attributes** — The #1 cause of script execution failure. The 57-rule regex fixer can't keep up.
- **Camera placement** — Catastrophically wrong in 2/3 benchmark scenarios (camera inside wine glass, camera too close to candle)
- **Technique monotony** — System defaults to Mantaflow for everything regardless of prompt. Root cause diagnosed (2026-02-22): the doc search system was returning only API reference results, completely blocking manual/tutorial content. The agent could not discover new techniques because it could not read the manual. Code fix applied but the underlying manual content is still human-oriented with low info density for LLM consumption.
- **Context bloat** — Verbose agent output fills context windows, degrading downstream agent performance
- **Knowledge base quality** — May contain stale/wrong patterns that mislead the research agent
- **Quality analyst feedback loop** — Analyst sees renders but not code, so its fix suggestions target symptoms not causes
- **Light energy oscillation** — Overcorrection between iterations (50 → 2500 → 10)
- **Missing collision effectors** — Liquid falls through containment geometry
- **Cost per run** — Too high due to unnecessary LLM calls for tasks that should be deterministic

### What's Promising But Unproven
- Truth pack (runtime introspection) for hallucination prevention
- Sandbox/experimentation mode for capability exploration
- Monitoring agent for early failure detection
- Artifact-based info sharing to combat context bloat
- Evidence-gated learning with memory decay
- SDK features like tool guardrails, conditional enabling, AdvancedSQLiteSession

---

## 13. Design Principles (Ranked)

These principles resolve conflicts when two design goals compete. Higher-ranked principles override lower ones.

### P1: Reliability Before Capability

A system that reliably produces evaluable renders (even low-scoring ones) is more valuable than a system that occasionally produces amazing renders but usually crashes. Every render — even a bad one — is learning data. A crash produces nothing.

*Near-term success = every run produces a render the quality analyst can evaluate.*

### P2: LLM Creativity Is the Core Value

The wine glass render exists because an LLM made dozens of creative decisions — glass shape, material properties, table surface, liquid color, wax pool detail. Replacing that creativity with templates or hardcoded paths would destroy what makes this system special. Protect the LLM's creative role. Give it better tools, better context, better guardrails — don't replace it.

### P3: Blender Is the Source of Truth for Its Own API

When an LLM needs to know what attributes exist on `FluidDomainSettings`, the answer comes from Blender itself (`bl_rna.properties`), not from training data, not from a static allowlist, not from a regex fixer. Runtime introspection is the ground truth. Everything else is a cache or approximation.

### P4: Compute What You Can, Generate What You Must

Camera placement is geometry. Light energy bounds are physics. Collision effectors are topology. These are computable. The LLM should decide *what* to create (a candle flame on a table). Deterministic code should verify *where to put the camera* (outside the scene bounds, pointed at the subject) and *what parameters are safe* (light energy within sane bounds for this effect type).

### P5: Context Is Precious — Every Token Must Earn Its Place

Agent output that fills context windows with verbose prose degrades every downstream operation. All inter-agent communication should be **dense, structured, and optimized for LLM consumption** — not human reading. Artifact-based sharing (store full reports in files, pass only summaries) over inline context.

### P6: Every Run Produces Learning Signal

Even failed runs have value — if the system captures what was attempted, what happened, and why. Learning must be evidence-gated (proven patterns only) and subject to decay (stale knowledge is worse than no knowledge).

### P7: Earn Autonomy Through Evidence

No aspect of the system becomes "trusted" or "autonomous" without repeated proof. Three successful uses of a pattern before it enters trusted knowledge. Pass rate thresholds before HITL checkpoints are relaxed. Calendar time proves nothing; outcomes prove everything.

---

## 14. Success Criteria

### The system is succeeding when:

1. **Every run produces a render** — even if the quality score is low, the pipeline completes without crashing
2. **Known effect types (fire, liquid, smoke) reliably score >= 60** on quality evaluation
3. **Novel prompts produce reasonable first attempts** — "a chandelier falling and shattering" creates a scene with rigid body physics, not a Mantaflow simulation
4. **The system demonstrably improves over time** — the same prompt produces better results after 50 runs than after 5
5. **Failed runs produce useful diagnostics** — actionable information about what went wrong and what to try next
6. **The user can intervene at any point** and redirect without losing progress
7. **Budget is respected** — cost per run stays under $0.50 for typical scenarios

### The system is failing when:

1. **Runs crash without producing evaluable renders** — lost API spend with zero learning signal
2. **The same errors recur across sessions** without the system learning to avoid them
3. **The human has to manually fix problems** that the system has encountered before
4. **The system loops on a failing approach** without trying alternatives
5. **Every prompt produces the same technique** regardless of what was requested
6. **Context bloat degrades quality** as iterations progress within a run

---

## 15. Scope of Capabilities

The system should progressively support these Blender physics and generation systems:

| Priority | System | Examples | Status |
|----------|--------|----------|--------|
| **P0 (Now)** | Mantaflow Gas | Fire, smoke, explosions | Working (best: 58/100) |
| **P0 (Now)** | Mantaflow Liquid | Water, pours, splashes | Working (best: 70/100 manual, 0/100 benchmark) |
| **P1 (Next)** | Rigid Body | Destruction, collapse, shattering | Architecture must support |
| **P1 (Next)** | Particle Systems | Debris, sparks, rain, snow | Architecture must support |
| **P2 (Later)** | Geometry Nodes | Procedural effects, scattering | Architecture must support |
| **P2 (Later)** | Cloth/Soft Body | Fabric, deformation | Architecture must support |
| **P3 (Future)** | Combinations | Multi-physics scenes (flag + explosion) | Architecture must support |
| **P3 (Future)** | PlasmaDXR Pipeline | NanoVDB export → DXR volumetric renderer | Future integration |

**Architectural constraint:** Every design decision must be evaluated against P1-P3 scope. A decision that solves a Mantaflow problem but makes rigid body support harder is a bad decision.

---

## Appendix A: Key References

| Document | Purpose |
|----------|---------|
| [Prompt Enhancement Guide](PROMPT_ENHANCEMENT_GUIDE.md) | How to write good VFX prompts |
| [VERSION_TRUTH.md](VERSION_TRUTH.md) | Blender 5.0 API ground truth |
| [AI_OPERATION_MANUAL.md](AI_OPERATION_MANUAL.md) | Operational guide for AI agents |
| [MASTER_ROADMAP_2026-01-26.md](MASTER_ROADMAP_2026-01-26.md) | Implementation roadmap |
| [ARCHITECTURE_FROM_SCRATCH_V2.md](ARCHITECTURE_FROM_SCRATCH_V2.md) | Independent architecture analysis (truth pack proposal) |

## Appendix B: Relevant Research Findings

These patterns from external systems may inform future development:

| Pattern | Source | Relevance |
|---------|--------|-----------|
| Stateless iteration + file memory | Ralph | Fresh context per iteration eliminates degradation |
| Artifact-based info sharing | Anthropic multi-agent system | Store full results in files, pass only summaries |
| Ebbinghaus memory decay | SAGE | Knowledge entries decay if not reinforced by success |
| Metacognitive monitoring layer | MASC research | Secondary monitor detects stuck loops, oscillation, context bloat |
| Maker-checker with iteration caps | Microsoft Azure patterns | Formalized quality gate with fallback behavior |
| Multi-grader quality evaluation | OpenAI self-evolving agents | Combine deterministic + ML + LLM graders for robust scoring |
| Dynamic task ledger | Microsoft Magentic | Adaptive pipeline that adjusts plan based on runtime evidence |
| Prompt versioning with rollback | OpenAI cookbook | Track which system prompts produce better results over time |

## Appendix C: Glossary

| Term | Meaning |
|------|---------|
| **Truth pack** | Runtime introspection data from Blender's `bl_rna.properties`, providing ground truth for valid attribute names, types, and ranges |
| **Evidence-gating** | Requiring N successful uses before a pattern becomes trusted knowledge |
| **Escape velocity** | Escalating exploration strategy when the system is stuck — from parameter tweaks to technique switches to human escalation |
| **Context bloat** | Excessive verbose text in agent context windows, degrading downstream reasoning quality |
| **Artifact-based sharing** | Storing detailed results in files and passing only lightweight references between agents |
| **HITL** | Human-in-the-loop — checkpoints where the human reviews and approves system decisions |
| **Quality threshold** | Score of 60/100 with no critical issues — the bar for a "passing" render |
