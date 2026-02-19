# Blender VFX Orchestrator — Mission Statement

**Version:** Draft v0.1
**Purpose:** Define what this project is, what it is not, and what guides every architectural decision. This document is the ultimate tiebreaker when design choices conflict.

---

## The Mission

Build an **autonomous, adaptive AI system** that generates high-quality volumetric VFX assets in Blender — capable of handling any visual effect a user can describe in natural language, learning from every attempt, and improving over time without human intervention.

The system should behave like a **skilled VFX artist who happens to have perfect recall**: creative when given novel challenges, reliable when repeating known techniques, and honest about what it doesn't know.

---

## Core Objectives

### 1. Generality Over Specialization

The system must handle the full range of Blender's physics and visual capabilities — not just Mantaflow fluid simulations. This includes but is not limited to:

- Fluid dynamics (liquid, gas, fire, smoke)
- Rigid body physics (destruction, collapse, shattering)
- Particle systems (debris, sparks, rain, snow)
- Cloth and soft body simulation
- Geometry Nodes procedural effects
- Volumetric rendering (fog, clouds, atmospherics)
- Combinations of the above in a single scene

A user prompt like "a window getting smashed" or "a pile of bricks collapsing" should be within scope — not as a future goal, but as a design constraint that shapes every decision we make today.

### 2. Intelligent Adaptation, Not Parameter Filling

The LLM agents are **creative problem-solvers**, not form-fillers. When the system encounters a novel prompt:

- It should **research** the right Blender approach (which physics system? which techniques? what are the constraints?)
- It should **explore** Blender's capabilities directly (run diagnostic scripts, inspect available APIs, test hypotheses)
- It should **reason** about what might work and why
- It should **learn** from the outcome and apply that knowledge to future runs

Templates and parameter schemas exist as **accelerators for known patterns** — a fast path when the system already knows what works. They are not the primary mode of operation. The primary mode is intelligent, adaptive code generation guided by knowledge and experience.

### 3. Self-Learning and Self-Improvement

Every run — successful or failed — produces signal. The system must:

- **Capture** what worked and what didn't, at the parameter level, the technique level, and the approach level
- **Validate** learnings through repeated measurement before promoting them to trusted knowledge
- **Apply** validated knowledge automatically via dynamic instructions that adapt agent behavior in real time
- **Expand** its capabilities by discovering new techniques, understanding new physics systems, and creating new templates from successful novel approaches

Self-improvement is gated by evidence, not by calendar. The system earns autonomy by demonstrating reliability.

### 4. Honest Execution and Truthful Feedback

The system never lies about outcomes. When a run fails:

- Exit codes are never overridden
- Errors are reported with full context
- Quality scores reflect actual render quality, not optimistic interpretations
- The system tells the user exactly what went wrong and what it plans to try next

### 5. Human Collaboration, Not Human Replacement

The system operates on a spectrum from **fully guided** to **fully autonomous**, with the human always able to step in:

- **Human-in-the-loop** at critical decision points (technique selection, quality judgement, novel approaches)
- **Transparent reasoning** so the user can understand and redirect the system's decisions
- **Graceful escalation** when the system is stuck — ask for help rather than loop endlessly
- **Respect for budget** — never burn API credits on speculative actions without user awareness

As reliability improves, the human steps back. But the human can always step in.

---

## What This Project Is NOT

- **Not a parameter tuner.** Filling bounded fields in a template is a useful optimization for known effects, but it is not the core capability.
- **Not a Mantaflow-only tool.** Mantaflow is one physics system among many. The architecture must not assume fluid simulation is the only kind of VFX.
- **Not a deterministic pipeline.** Deterministic components (camera placement, API validation, parameter bounds) exist as **safety nets and accelerators**. They catch errors and provide sane defaults. They do not replace creative decision-making.
- **Not a one-shot generator.** Iteration is expected. The system improves renders through multiple passes, learning from each attempt.

---

## Design Principles

These principles resolve conflicts when two design goals compete:

### P1: Flexibility Before Rigidity

When choosing between a flexible approach that might fail and a rigid approach that works for known cases, prefer the flexible approach **with proper guardrails**. Guardrails constrain the failure mode, not the creative space.

*Example: An API allowlist catches hallucinated attributes, but does not prevent the agent from using valid Blender APIs it hasn't seen before.*

### P2: Exploration Is a Feature, Not a Bug

The system should actively explore Blender's capabilities — running diagnostic scripts, querying documentation, testing small code snippets — just as a human developer would. If the agent gets stuck, it should debug the problem programmatically, not loop on the same failing approach.

*Example: When an attribute name is wrong, run `dir(bpy.types.FluidDomainSettings)` to find the correct one instead of guessing or looping through the API fixer.*

### P3: Deterministic Systems Support Adaptive Ones

Templates, parameter schemas, camera placement functions, and API truth packs exist to make the adaptive system better — by providing fast paths for known patterns and safety nets for common failures. They do not replace the adaptive system.

*Example: A template for fire effects provides a fast, reliable starting point. But if the user asks for "fire spreading through a forest," the system should be able to go beyond the template — combining rigid body trees, particle embers, and Mantaflow fire.*

### P4: Every Run Produces Learning Signal

Even failed runs have value. The system must extract and store actionable knowledge from every execution — what parameters were used, what errors occurred, what the quality score was, and why. This signal feeds dynamic instructions and guides future decisions.

### P5: Use the SDK's Full Capabilities

The OpenAI Agents SDK provides tools for adaptive, human-collaborative agent systems — dynamic instructions, human-in-the-loop approval, streaming, parallelism, tool guardrails, conditional tool enabling, session persistence. These capabilities exist to make agents flexible and reliable. Use them.

### P6: Parallelism and Speed Matter

Where operations are independent, run them concurrently. Research and documentation lookup can happen in parallel. Multiple evaluation metrics can run simultaneously. Blender can bake while the agent plans the next iteration. Throughput directly impacts how much the system can learn per dollar spent.

---

## Success Criteria

The system is succeeding when:

1. A novel prompt (e.g., "a chandelier falling and shattering") produces a reasonable first attempt without requiring a pre-built template
2. Known effect types (fire, liquid, smoke) reliably score >= 60 on quality evaluation
3. The system demonstrably improves over time — the same prompt produces better results after 50 runs than after 5
4. Failed runs produce useful diagnostic information and actionable next steps
5. The user can intervene at any point and redirect the system without losing progress
6. Budget is respected — the system makes intelligent decisions about when to spend API credits

The system is failing when:

1. Novel prompts are rejected or produce empty/broken results
2. The same errors recur across sessions without the system learning to avoid them
3. The human has to manually fix problems that the system has encountered before
4. The system loops on a failing approach without trying alternatives
5. Architectural decisions make it harder to add new physics types or techniques

---

## Autonomy Progression

The system earns autonomy through demonstrated reliability:

| Level | Requirement | Capability |
|-------|-------------|------------|
| **0: Guided** | Baseline | Templates + curated knowledge + HITL at decision points |
| **1: Assisted** | 10+ successful runs per effect type | Parameter memory, reduced HITL frequency |
| **2: Semi-Autonomous** | Stable pass rate > 50% across types | Bayesian parameter optimization, automated technique selection |
| **3: Autonomous (Known)** | Stable Level 2 for 4+ weeks | Template self-creation from successful Lane B runs |
| **4: Autonomous (Novel)** | External review + stable Level 3 | Discover new physics systems, create new technique categories, full self-improvement |

---

## Scope of Blender Capabilities

The system should progressively support these Blender physics and generation systems:

| Priority | System | Examples |
|----------|--------|----------|
| **P0 (Now)** | Mantaflow Gas | Fire, smoke, explosions |
| **P0 (Now)** | Mantaflow Liquid | Water, pours, splashes |
| **P1 (Next)** | Rigid Body | Destruction, collapse, shattering |
| **P1 (Next)** | Particle Systems | Debris, sparks, rain, snow |
| **P2 (Later)** | Geometry Nodes | Procedural effects, scattering |
| **P2 (Later)** | Cloth/Soft Body | Fabric, deformation |
| **P3 (Future)** | Combinations | Multi-physics scenes |
