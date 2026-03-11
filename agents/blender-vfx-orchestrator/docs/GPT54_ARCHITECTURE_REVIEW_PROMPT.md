# GPT-5.4 Architecture Review Prompt

Use with: `gpt-5.4`, reasoning effort: `xhigh`, verbosity: `high`

---

## The Prompt

```
<system>
You are a senior software architect specializing in autonomous multi-agent systems built on the OpenAI Agents SDK. You have deep expertise in:
- OpenAI Agents SDK v0.10.5+ architecture patterns
- Blender 5.0 Python API and headless pipeline automation
- Iterative quality improvement loops with ML-powered evaluation
- Self-learning systems with evidence-gated knowledge bases
- LLM-as-code-generator patterns (not template fillers)

You are being asked to perform a comprehensive architectural review of a real, working system with ~55,000 lines of Python. The system works — it has produced quality renders — but it fails too often. The creator wants you to identify root causes, not symptoms, and propose a new roadmap that addresses them systemically.
</system>

<task>
Perform a deep architectural review of the Blender VFX Orchestrator — an autonomous multi-agent system that generates high-quality volumetric VFX assets in Blender using natural language prompts. Identify the root causes of persistent quality and reliability problems, and propose a new development roadmap that addresses them systemically rather than through incremental patches.

This is a complex, multi-step analysis task. Use your full reasoning capability.
</task>

<research_mode>
Do research in 3 passes:
1) Analyze: identify the core architectural tensions and failure modes from the evidence provided
2) Diagnose: trace each failure mode to its root cause in the architecture
3) Propose: design a roadmap that resolves root causes, not symptoms
Stop only when you have addressed every failure mode with a specific architectural intervention.
</research_mode>

<output_contract>
Return exactly these sections in this order:
1. EXECUTIVE SUMMARY (3-5 sentences)
2. ROOT CAUSE ANALYSIS (numbered list, each with: symptom → root cause → evidence)
3. ARCHITECTURAL INTERVENTIONS (numbered, each with: what changes, why it fixes the root cause, estimated complexity)
4. PROPOSED ROADMAP (phased, with dependencies between phases)
5. SDK FEATURES TO LEVERAGE (specific Agents SDK capabilities that address identified problems)
6. WHAT TO STOP DOING (things the current system does that should be removed or simplified)
7. OPEN QUESTIONS (things you cannot determine from the provided context)

Do not add sections beyond these seven.
</output_contract>

<completeness_contract>
- Address EVERY failure mode listed in the OBSERVED FAILURE MODES section
- For each root cause, trace it to a specific architectural decision or missing capability
- For each roadmap phase, specify what it unblocks and what depends on it
- Do not propose solutions that conflict with the design principles (ranked P1-P7)
</completeness_contract>

<verification_loop>
Before finalizing:
- Does every observed failure mode have a root cause identified?
- Does every root cause have at least one intervention proposed?
- Does the roadmap address root causes in dependency order (prerequisites first)?
- Do the interventions respect P2 (LLM creativity is the core value) — i.e., no intervention replaces creative LLM work with templates?
- Is the total proposed work realistic for a solo developer with a $20/month API budget?
</verification_loop>

---

## THE SYSTEM: BLENDER VFX ORCHESTRATOR

### Mission (from canonical Mission Statement v2.0)

Build an autonomous, adaptive AI system that allows anyone — regardless of 3D graphics experience — to create high-quality volumetric VFX assets in Blender using natural language.

The user describes what they want: "a wine glass being filled with red wine on a marble table." The system researches how to build it, writes a complete Blender Python script that creates every element from scratch (geometry, materials, physics, lighting, camera, environment), executes it, evaluates the result, and iterates until the quality threshold is met.

The system should behave like a skilled VFX artist who happens to have perfect recall: creative when given novel challenges, reliable when repeating known techniques, and honest about what it doesn't know.

**What makes this different:** This is NOT a parameter tuner, template filler, or Mantaflow wrapper. The LLM's creative ability to "run with an idea" and produce a complete, coherent scene is the core value proposition. A wine pour prompt doesn't just create liquid dynamics — it creates the glass (with proper refraction material), the table (with marble surface), ambient set dressing, three-point lighting, and camera composition.

### Design Principles (Ranked — higher overrides lower)

P1: Reliability Before Capability — every run must produce an evaluable render
P2: LLM Creativity Is the Core Value — do NOT replace creative LLM work with templates
P3: Blender Is the Source of Truth — runtime bl_rna introspection, not training data
P4: Compute What You Can, Generate What You Must — deterministic where possible
P5: Context Is Precious — every token must earn its place
P6: Every Run Produces Learning Signal — even failures have value
P7: Earn Autonomy Through Evidence — nothing becomes trusted without proof

### Current Architecture

```
Codebase: ~55,000 lines Python
Main orchestrator: orchestrator.py (2,980 lines) — monolithic pipeline loop
SDK: OpenAI Agents SDK v0.10.5
Models: gpt-5.2 (script writing), gpt-5-mini (research, evaluation)
Blender: 5.0.1, headless on WSL2, Cycles GPU

Pipeline phases (sequential):
  Phase 0: Research (ResearchAgent → DocsExpert)
  Phase 0.5: Technique Selection (TechniqueSelector coordinator)
  Phase 1: Script Generation (ScriptWriter agent)
  Phase 1.5: API Validation (truth pack + API fixer, deterministic)
  Phase 2: Execution (Executor agent → Blender subprocess)
  Phase 3: Quality Evaluation (QualityAnalyst — ML metrics + vision)
  Phase 3.5: Quality Gate (QualityGateJudge coordinator)
  Phase 4: Learning (LearningAgent — experiment tracker, knowledge base)
  Phase 5: Decision (continue/modify/switch technique/escalate)

Agent pattern: agents-as-tools (sub-agents called as tools by coordinators)
Anti-hallucination: truth pack (bl_rna introspection) + API fixer (regex) + hardcoded fixes
Self-learning: vector store docs, knowledge distillation, proactive research, code pattern memory, escape velocity
```

### Key Files

| File | Lines | Purpose |
|------|-------|---------|
| orchestrator.py | 2,980 | Monolithic pipeline loop with all phases |
| blender_api_fixer.py | 2,056 | Regex-based API corrections (57+ rules) |
| blender_docs_tools.py | 1,675 | Documentation search (vector store + keyword) |
| script_generator_tools.py | 1,584 | Script generation/modification tools |
| dynamic_instructions.py | 1,149 | Runtime instruction injection |
| blender_executor_tools.py | 979 | Blender process management |
| asset_evaluator_tools.py | 976 | ML quality metrics |
| shared_context.py | 945 | Pydantic models for session state |
| truth_pack.py | 933 | Runtime Blender API introspection |
| enforcement_hooks.py | 900 | RunHooks for loop detection |

---

## OBSERVED FAILURE MODES

These are real, documented problems from 200+ pipeline runs and detailed trace analysis:

### F1: Technique Monotony
**Symptom:** System defaults to the same techniques regardless of prompt. "Glass shattering" routes to Mantaflow gas. "Sparks" routes to Mantaflow gas. Everything is Mantaflow.
**What we've tried:** Fixed doc search bug (API results were crowding out manual results). Seeded vector store with technique docs. Added destruction-specific code patterns.
**Current state:** Technique selection now picks correct names (e.g., "cell_fracture_constraint_breaking") but the Script Writer still generates manual mesh-cutting code instead of using the selected technique's actual Blender operators.

### F2: Script Writer Ignores Technique Research
**Symptom:** Research phase finds the right technique. Technique selector picks it. Script writer then generates a 900-line script that implements everything from scratch using basic mesh operations instead of using the researched technique (e.g., Cell Fracture addon operators).
**What we've tried:** Added explicit instructions to dynamic_instructions.py. Added HARDCODED_FIXES to truth pack. Added API fixer injection. Seeded code patterns.
**Current state:** Despite all this guidance, the script writer's training data preference for generating its own mesh manipulation code overrides the injected instructions.

### F3: Recovery Scripts Lose Quality
**Symptom:** When a script fails execution and the recovery path triggers, the rewritten script drops advanced features (Cell Fracture, complex materials, detailed geometry) and produces a much simpler script that scores lower.
**What we've tried:** Nothing systemic — each fix is manual.
**Current state:** Recovery consistently produces worse scripts than the original attempt.

### F4: Scene Geometry Doesn't Fit Together
**Symptom:** Objects float in space, gaps between surfaces that should touch, objects intersect incorrectly. A glass on a table has a gap between glass bottom and table surface.
**What we've tried:** Nothing systemic.
**Current state:** Persistent across all runs. The script writer places objects by absolute coordinates without spatial reasoning about relationships.

### F5: Quality Scores Plateau at 28-34
**Symptom:** Across 4 recent E2E runs targeting glass shatter, best scores were 28, 30, 34, 30. The system never breaks through to the 60+ passing threshold.
**What we've tried:** Improving technique selection, adding code patterns, fixing doc search.
**Current state:** The improvements we've made (technique selection, truth pack, doc search) address infrastructure but not the fundamental quality of generated scripts.

### F6: Research Agent Exhausts Turns Without Synthesizing
**Symptom:** Research agent uses all max_turns (8) doing sequential doc searches but never produces a synthesized ResearchOutput. It runs out of turns mid-research.
**What we've tried:** Bumped max_turns from 4 to 6 to 8.
**Current state:** Still hitting turn limits. The agent's multi-pass research_mode prompt causes it to do 7+ searches before attempting synthesis.

### F7: Context Bloat Degrades Downstream Agents
**Symptom:** By iteration 3+, the script writer's context is so full of previous iteration history, evaluation output, and modification strategies that it produces worse code than iteration 1.
**What we've tried:** Session compaction at ~25K tokens.
**Current state:** Compaction helps but doesn't solve the fundamental issue of verbose inter-agent communication.

### F8: Orchestrator Monolith (2,980 lines)
**Symptom:** All pipeline logic lives in one file. Adding new features (HITL, monitoring, new phases) means modifying a 3,000-line file. High coupling between phases.
**Current state:** Every change risks breaking other phases. Difficult to test individual phases.

### F9: Vision Analyst Can't See Code
**Symptom:** Quality evaluation identifies visual problems ("too dark", "objects floating") but can't connect them to code causes ("missing Area Light at line 342", "object Z-position wrong at line 218").
**What we've tried:** Added QA diagnosis bridge that pairs visual critique with script code.
**Current state:** Bridge exists but the feedback loop still produces symptom-level suggestions that the modification strategist can't act on precisely.

### F10: Rigid Body Constraint Context Error
**Symptom:** `bpy.ops.anim.keyframe_insert_by_name` fails in headless Blender because it requires animation context that doesn't exist without a UI.
**Current state:** Blocks Cell Fracture + Rigid Body Constraints workflow entirely.

### F11: Incremental Fix Complexity
**Symptom:** Each problem gets its own fix layer: truth pack, API fixer, hardcoded fixes, dynamic instructions, code patterns, enforcement hooks. The system now has 6+ overlapping correction mechanisms that interact unpredictably.
**Current state:** Adding a new technique requires touching 5+ files. The correction layers sometimes conflict (API fixer undoes truth pack fixes, etc.).

---

## WHAT WORKS WELL

- Truth pack (bl_rna introspection) — catches most hallucinated attributes before execution
- Quality evaluation (ML metrics + vision) — provides real signal about render quality
- Escape velocity system — correctly escalates from parameter tweaks to technique switches to human guidance
- Evidence-gated learning — prevents single lucky runs from becoming trusted knowledge
- Pipeline infrastructure — sessions save/resume, tracing works, budget tracking works
- The system CAN produce impressive complete scenes (wine pour: 70/100 with manual prompting)

---

## AGENTS SDK CAPABILITIES (v0.10.5+)

Currently used:
- agents-as-tools (sub-agents called as tools)
- Dynamic instructions (runtime injection)
- RunHooks (lifecycle monitoring)
- Structured outputs (Pydantic models)
- Tracing

Available but NOT used:
- tool_use_behavior: stop_on_first_tool — deterministic agent control
- AdvancedSQLiteSession — branching, token tracking, persistence
- Tool guardrails (ToolInputGuardrail) — $0 validation at tool boundaries
- Conditional tool enabling (is_enabled callback) — hide expensive tools
- call_model_input_filter — trim conversation history before LLM call
- Streaming (run_streamed) — real-time monitoring
- parallel_tool_calls in ModelSettings
- needs_approval — native HITL with state serialization

GPT-5.4 specific capabilities:
- 1M token context window
- Native compaction support (trained for it)
- Tool search (deferred tool loading)
- Preambles (pre-tool-call explanations)
- phase parameter (commentary vs final_answer)
- Allowed tools (constrain available tools per turn)
- Custom tools with freeform inputs and CFG constraints

---

## CONSTRAINTS

- Solo developer (novice programmer, strong AI/ML intuition)
- $20/month API budget (~$0.20-0.50 per run target)
- WSL2 Linux environment, Blender 5.0.1 headless
- Must support any Blender physics system (not just Mantaflow)
- LLM creativity is non-negotiable — the system must NOT become a template filler
- Must be self-learning — the system should get better over time without manual intervention

---

## SPECIFIC QUESTIONS

1. **Why does the script writer ignore technique research?** The research phase correctly identifies Cell Fracture as the right approach, the technique selector picks it, the dynamic instructions explicitly say to use it — but the script writer still generates manual mesh-cutting code. Is this a prompt engineering problem, a context problem, or an architectural problem? What would actually fix it?

2. **Should the orchestrator be decomposed?** The 2,980-line monolith is painful. But decomposing it risks breaking the carefully-tuned phase interactions. What's the right decomposition strategy that preserves the pipeline's sequential guarantees while enabling independent phase testing?

3. **How should recovery work?** Currently, when a script fails, the system rewrites from scratch (losing quality). Should recovery instead: (a) patch the failing script, (b) maintain a "last known good" baseline and modify from there, (c) use the SDK's session branching to try multiple approaches, or (d) something else?

4. **Is the multi-layer correction approach (truth pack + API fixer + hardcoded fixes + dynamic instructions + code patterns + enforcement hooks) fundamentally wrong?** Should these be consolidated into fewer, more powerful mechanisms?

5. **How can scene geometry quality be improved?** Objects floating, gaps between surfaces, incorrect intersections. Is this solvable through better prompts, deterministic post-processing, or does it require a fundamentally different approach to scene composition?

6. **What GPT-5.4 and SDK features would have the highest impact?** Given the failure modes above, which unused SDK capabilities and GPT-5.4 features would provide the most leverage?

7. **Is the current agent roster correct?** The system has 10+ agents. Are some unnecessary? Are any missing? Should some agents be merged or split differently?

8. **How should the system handle Blender features it hasn't been trained on?** The Cell Fracture problem reveals that LLMs default to generating their own code rather than calling addon operators they haven't seen in training data. This will be a recurring problem for any advanced Blender feature. What's the general solution?

---

## WHAT A GOOD ANSWER LOOKS LIKE

- Identifies 3-5 root causes that explain most of the 11 failure modes (many failures share common roots)
- Proposes architectural changes, not more patches on top of patches
- Respects the design principles (especially P2: LLM creativity is the core value)
- Leverages specific SDK features with concrete implementation suggestions
- Provides a phased roadmap where each phase unblocks the next
- Identifies things to REMOVE (not just things to add) — the system may be doing too much
- Is honest about what it can't determine from this context alone
```
