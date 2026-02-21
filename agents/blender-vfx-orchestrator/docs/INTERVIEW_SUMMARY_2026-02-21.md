# VFX Orchestrator Vision Interview — Complete Summary

**Date:** 2026-02-21
**Participants:** Ben (creator) + Claude Opus 4.6
**Purpose:** Capture the true definition, vision, and operational reality of the Blender VFX Orchestrator through structured interview, to prevent future misinterpretation by AI assistants.
**Context:** Written at 93% context usage. This document preserves every insight, decision, and finding from the session.

---

## Background: Why This Interview Happened

Ben asked Claude to read the existing mission statement (`MISSION_STATEMENT_DRAFT.md` v1) and design a multi-agent system from scratch to satisfy it. Claude produced `ARCHITECTURE_FROM_SCRATCH_V2.md` — a proposal centered on minimizing LLM usage, reducing agents from 9 to 3, and making most operations deterministic.

**Ben's reaction revealed a fundamental misunderstanding.** Claude had interpreted the project as primarily a physics simulation generator where LLM involvement was the problem. In reality:

1. The system's core value IS LLM creativity — it produced a wine glass with correct refraction, marble table, liquid simulation, all from scratch in one script
2. The existing system WORKS — it scored 70/100 on a wine pour — but it's unstable, slow, and inefficient
3. The hallucination fix (57-rule regex fixer) was overcorrected and made the system rigid
4. The scope is much broader than Mantaflow — rigid body, particles, cloth, geometry nodes, and combinations are all in scope
5. The system creates COMPLETE SCENES (geometry, materials, physics, environment, lighting, camera), not just physics setups

This mismatch prompted the structured interview to capture Ben's true vision.

---

## Interview Round 1: The User Experience

### Q1: What happens from the user's perspective?

The user types a prompt. The prompt enhancement layer (LLM-powered) fleshes out the scene into a detailed, structured specification covering: elaborated scene premise, set dressing, object specifics, materials, scene geometry, physics choices, camera, lighting, and more. The user reviews and approves the enhanced prompt before the pipeline begins.

**Key insight:** The prompt enhancement is currently run MANUALLY as a separate step, not yet integrated into the automated pipeline. It uses the Prompt Enhancement Guide (`PROMPT_ENHANCEMENT_GUIDE.md`).

### Q2: Where does the user interact?

Currently CLI/development environment. If there's interest, Ben envisions a web UI as a limited demonstration. The delivery method is not finalized — confidence is low due to regression following promising tests.

### Q3: What does "enhance your prompt" mean?

Takes a short prompt (e.g., "candle on a table") and expands it into multiple structured categories: elaborated premise, set dressing, object specifics, materials, scene geometry, physics simulation choices, camera framing, lighting mood, and more.

---

## Interview Round 2: What Gets Created

### Q4: Does the Blender script create everything from scratch?

**YES.** The wine render (`images/wine_render.png`) was created entirely from a single generated Python script — the glass mesh, glass refraction material, liquid simulation, marble table surface, lighting, camera. Ben's jaw hit the floor when he saw it. It scored 70/100 on the vision quality test, and might have scored higher if the camera framing had been better (too zoomed in).

The system also corrected some script errors during the run — the code writer's initial script had a hallucinated attribute that the agent fixed itself before execution succeeded.

### Q5: How are objects created?

Ben doesn't know the specifics — he's a complete newcomer to Blender/3D graphics. **This is the whole point of the project** — allowing someone without 3D expertise to create VFX through natural language. The LLM figures out how to create the geometry.

### Q6: Set dressing and environment creation?

The wine scene had key points in the prompt, and the LLM "ran with the idea." Ben emphasized: "this is where having an LLM subagent can produce real magic." The user provides creative direction; the LLM makes dozens of detailed decisions about scene composition, materials, and environment.

**Critical insight Ben raised:** The system isn't just about physics simulation. It also handles environments, set dressing, and model creation of objects. This was a major scope element Claude had missed.

---

## Interview Round 3: The Pipeline and Where Intelligence Lives

### Q7: Ben's correction on LLM minimization

**"Lessening the role of LLMs is not my idea."** Ben has been recommended by others to shift to deterministic approaches, but his vision is an autonomous multi-agent system that performs complex tasks well. He's had success — the wine render proves it. The severe overcorrection of the hallucination issue has degraded everything, and the agent is currently broken.

### Q8: Pipeline flow

A Python script sets up the initial run, then the orchestrator takes over. The OpenAI dashboard traces show the agent flow: Research Agent → Documentation Expert → Technique Selector → API Spec Agent → Code Writer → Executor → Quality Analyst → Learning Agent → Quality Gate Judge.

Metrics show:
- Runs have 15-46+ LLM calls
- Duration: 5-160+ minutes
- Tool calls: 0-83 per run
- Quality scores range from 6 to 58 in benchmark
- Wine pour 3-iteration run: 47 LLM calls, 6 iterations, 20 minutes, 4 Blender runs, 3 successful

### Q9: Error correction during runs

The code writer's initial script had a hallucinated attribute. The agent fixed it itself and then executed successfully. Ben noted: **99% of "show-stopping" errors are just fictitious attributes.** The errors can be identified and fixed easily, making it extremely frustrating that they end test runs.

### Q10: What needs intelligence vs what should be deterministic

Ben agreed some things should be deterministic (e.g., the executor doesn't need to be LLM-based). But intelligence is needed for:

- **Research agent** — Reading the Blender manual (how elements work in B5, how to use them). Intelligence required to make the right choices.
- **Script writer** — Certainly needs LLM, with high reasoning ability
- **Visual analyst** — Important, uses OpenAI vision to critique renders
- **Learning agent** — Needs to catalogue everything learned, good and bad
- **Monitoring agent** (proposed) — Would monitor iteration progress, intervene if necessary

Ben also mentioned ideas like a running monitoring agent that oversees each iteration, intervening when needed. This comes from his own experience and Agents SDK capabilities that have never been leveraged (sessions, tracing, agents-as-tools, parallelism).

---

## Interview Round 4: Deep Dives

### Q11: Research agent effectiveness

Up to this point, the research agent worked on every run but despite that, similar results and repeated mistakes persisted. It was researching the manual (visible in context), but wasn't learning — possibly because the local KB was overwhelming it. Ben hasn't checked KB contents.

**Decisions made:**
- Research should run at start of each run, and again if the visual analyst suggests a technique change
- Research should cache what it learns
- The entire learning architecture needs overhaul and the existing KB should be wiped
- The visual analyst is good and presents excellent critique BUT is too verbose, causing context bloat
- **All agent output should be optimized for LLM readers, not humans** — as info-dense as possible
- Context bloat is a HUGE problem and waste of tokens
- They've already tried shifting to artifact-based info sharing (file-based) instead of inline context

### Q12: Truth pack for hallucination prevention

**Ben accepted the truth pack idea.** He noted the Blender 4.5→5.0 changelog is at `https://docs.blender.org/api/5.0/change_log.html`. He pointed out they already have a script that scans Blender scripts for errors — errors that CAN be identified and fixed easily but currently end test runs. This has been "extremely frustrating."

### Q13: What should learning look like?

Ben said: "This is a question I'm not qualified to answer." He asked Claude to research learning and autonomous operation in multi-agent systems.

His intuitions:
- The system should reflect on outcomes, good and bad
- The agent already has capability to use many SDK features (sessions, tracing, agents-as-tools, parallelism) that have never been leveraged
- A running/monitoring agent could keep things on track, avoiding common issues like tool overuse violations
- He referenced Ralph (`https://github.com/snarktank/ralph`) as a system to research
- He referenced the Agents SDK repo (`https://github.com/openai/openai-agents-python/tree/main`) for capability research

---

## Interview Round 5: Practical Reality

### Q14: What does near-term success look like?

**"Reliability."** Consistently producing a result the quality analyst can evaluate. Even low scores are good because it's more data to learn from. Novel prompts producing a real output would be excellent.

**This is the single most important insight from the interview.** Near-term success is NOT "score >= 60". It's "every run produces a render." Crashes with zero output are the worst outcome because they produce zero learning signal.

### Q15: How does prompt enhancement work?

Currently run as a separate manual step. Ben gives a basic scene description plus anything important. The enhancement script uses the Prompt Enhancement Guide (`PROMPT_ENHANCEMENT_GUIDE.md`) to expand it into the full structured format.

### Q16: When do you know a run is failing?

Ben often steps away during runs (5-30 minutes per iteration) and studies traces afterward. Early failure signals:

- **Black/white screen** — Often from incorrect camera positioning, excessive light, or default materials left unchanged
- **Quality analyst gives wrong advice** — Because it can't see the script code, its suggestions target symptoms not causes. This incorrect feedback gets passed to the start of the next iteration, causing several cascading failures.
- **Script too short** — < 500 lines = too basic. A decent script needs 500-700 lines. Good scripts are 700+ to over 1000 lines.
- **HITL or a monitoring agent could help** — Catching problems before they cascade

### Additional points Ben raised:

**Experimentation/sandbox mode:** Should be incorporated somehow — even a small sandbox where agents explore different ideas practically and record their thoughts. Could run between real tests or afterward.

**Monitoring flexibility:** Uncertain how feasible a running monitoring agent is, but if HITL works it could coordinate with a monitoring agent to catch issues and reduce failures.

**Openness to major changes:** Ben is happy to make major changes, fork the project, or even start again if needed. He asked Claude to create a new version of the mission statement based on this interview.

---

## Research Conducted During Interview

Three research agents were dispatched in parallel:

### Research 1: OpenAI Agents SDK v0.9.0+ Capabilities

**Source:** GitHub repo + web search

Key findings — SDK features NOT currently used that directly solve identified problems:

| Feature | Solves |
|---------|--------|
| **Tool guardrails** (`ToolInputGuardrail`) | API validation without LLM — deterministic checking as a tool wrapper, $0 cost |
| **`tool_use_behavior: "stop_on_first_tool"`** | Forces Executor to call one tool and return, no LLM interpretation overhead |
| **`AdvancedSQLiteSession`** | Built-in branching (try different techniques from same point), token tracking, keyword search |
| **Conditional tool enabling** (`is_enabled`) | Expensive tools disappear from agent's view when budget is low |
| **`call_model_input_filter`** | Trim conversation history before LLM calls — prevents context bloat in long sessions |
| **Native HITL** (`needs_approval`) | Pipeline pauses, state serialized, human approves, pipeline resumes. Replaces custom polling. |
| **Streaming** (`run_streamed()`) | Real-time monitoring of script generation — detect hallucinations mid-stream |
| **`Agent.clone()`** | Create effect-type-specific agent variants without duplicating config |
| **`reset_tool_choice`** | Force first turn to call tool, then allow free-form — useful for deterministic routing |
| **Per-agent model settings** | Different models per role: gpt-5.2 for creative, gpt-5-mini for structured, o3 for debugging |
| **`prompt_cache_retention="24h"`** | Cache system prompts for research/docs agents whose prompts rarely change |

### Research 2: Autonomous Multi-Agent Patterns

**Sources:** Ralph, Anthropic's multi-agent research system, SAGE, MASC, Microsoft Azure patterns, OpenAI self-evolving agents cookbook

Key patterns applicable to VFX orchestrator:

**HIGH IMPACT:**

1. **Stateless iteration + file memory (Ralph)** — Each iteration gets fresh context. Memory persists through files (session_state.json, learnings.txt). Eliminates context degradation.

2. **Artifact-based info sharing (Anthropic)** — Agents store full results in files, pass only lightweight summaries to coordinator. Context grows by 1 line instead of 50 paragraphs.

3. **Ebbinghaus memory decay (SAGE)** — Knowledge entries not reinforced by successful outcomes naturally fade. Prevents KB bloat without manual curation.

4. **Metacognitive monitoring layer (MASC)** — Secondary monitor watches for: repeated failures, score oscillation, same script structure repeatedly, budget/cost exceeded. Can force technique switch, clamp parameters, halt pipeline. This IS the monitoring agent Ben described.

**MEDIUM IMPACT:**

5. **Maker-checker with iteration caps (Microsoft)** — Formalized ScriptWriter(maker)/QualityGate(checker) loop with fallback behavior at cap.

6. **Task specification templates (Anthropic)** — Each agent gets structured spec: objective, output format, constraints, tools to use, task boundaries.

7. **Multi-grader quality (OpenAI cookbook)** — Combine deterministic checks + ML metrics + LLM-as-judge. Require majority agreement.

**LOWER IMPACT:**

8. **Prompt versioning with rollback (OpenAI)** — Track which system prompts produce better results.

9. **Effort scaling (Anthropic)** — Different iteration budgets per effect complexity.

10. **Dynamic task ledger (Microsoft Magentic)** — Replace fixed state machine with adaptive plan.

### Research 3: Blender 5.0 Python API Changelog

**Source:** `https://docs.blender.org/api/5.0/change_log.html` + individual feature pages

Key findings — items NOT in the existing API fixer:

| Pattern | Fix | Priority |
|---------|-----|----------|
| `ShaderNodeMixRGB` | `ShaderNodeMix` | HIGH |
| `ShaderNodeSeparateHSV` | `ShaderNodeSeparateColor` | MEDIUM |
| `ShaderNodeCombineHSV` | `ShaderNodeCombineColor` | MEDIUM |
| `.inputs['Sheen']` | `.inputs['Sheen Weight']` | MEDIUM |
| `scene.node_tree` | `scene.compositing_node_group` | HIGH (if compositor used) |
| `BLENDER_EEVEE_NEXT` | `BLENDER_EEVEE` | HIGH |
| `material.use_nodes = True` | No-op (always True in 5.0) | LOW |
| `PointCache.compression` | Remove line (always ZSTD now) | LOW |
| `bpy.data.grease_pencils` | `bpy.data.annotations` | LOW |

Also documented: 30+ renamed attributes, 40+ removed attributes/functions, 20+ new additions, and 13+ silent behavior changes (different rendering results without errors).

---

## Documents Produced During This Session

1. **`ARCHITECTURE_FROM_SCRATCH_V2.md`** — Independent architecture proposal (truth pack, 3-agent design, deterministic pipeline). Valuable for the truth pack concept and anti-hallucination strategy, but the "minimize LLM" philosophy was explicitly rejected by Ben.

2. **`MISSION_STATEMENT_DRAFT.md` v2.0** — Complete rewrite of the mission statement incorporating all interview findings. 15 sections covering vision, user experience, pipeline, intelligence allocation, anti-hallucination, quality/learning, experimentation, human collaboration, autonomy progression, technical foundations, current state, design principles, success criteria, scope, and research appendices.

3. **This document** (`INTERVIEW_SUMMARY_2026-02-21.md`) — Complete summary of the interview and all research.

---

## Key Decisions Made During This Session

| Decision | Rationale |
|----------|-----------|
| **LLM creativity is the core value, not the problem** | Wine render proves it. System scored 70/100 when the LLM was creative. |
| **Truth pack approach accepted** | Runtime introspection of `bl_rna.properties` as prevention layer. Existing regex fixer as last resort. |
| **KB needs to be wiped and rebuilt** | Current KB may be poisoning the research agent with stale/wrong patterns. |
| **All agent output must be LLM-optimized** | Dense, structured, minimal. Not verbose human-readable prose. Context bloat is a top-3 problem. |
| **Near-term success = reliability** | Every run produces a render. Even low scores are learning data. Crashes produce nothing. |
| **Monitoring agent/layer is needed** | For catching problems early: oscillation, stuck loops, context bloat, wrong feedback cascades. |
| **Experimentation mode is wanted** | Sandbox for exploring new techniques between production runs. |
| **Script quality correlates with length** | <500 = too basic, 500-700 = decent, 700-1000+ = good. |
| **Quality analyst feedback loop is broken** | Can't see code → suggests wrong fixes → cascading failures. Need to pair visual critique with script analysis. |
| **Open to major changes including restart** | Ben prioritizes getting the vision right over preserving existing code. |

---

## Unresolved Questions for Next Session

1. Ben's "fairly minor" thoughts on the mission statement doc — specific amendments needed
2. Implementation plan — what to build first, in what order, with what milestones
3. Whether to fork, refactor, or rewrite the existing codebase
4. Specific agent definitions and tool sets for the redesigned pipeline
5. How to integrate the prompt enhancement layer into the automated pipeline
6. Detailed design of the monitoring agent/layer
7. Detailed design of the experimentation/sandbox mode
8. How to structure the new knowledge base (schema, what to store, decay parameters)
9. How to implement the quality analyst fix (pairing visual critique with script analysis)
10. Budget model for the redesigned system — cost per run estimates

---

## Context for Future AI Sessions

**If you are an AI reading this document to work on the VFX Orchestrator:**

1. Read the Mission Statement (`MISSION_STATEMENT_DRAFT.md` v2.0) first — it IS the canonical reference
2. The system's value comes from LLM CREATIVITY, not from minimizing LLM usage
3. The wine render (`images/wine_render.png`) is proof the system works — study it
4. The #1 near-term goal is RELIABILITY (every run produces a render), not quality (score >= 60)
5. The truth pack (runtime introspection) is the accepted approach for hallucination prevention
6. Context bloat is a top-3 problem — keep all agent output dense and structured
7. Ben is a novice programmer who prefers brutal honesty and wants to understand the "why"
8. The existing codebase may be refactored, forked, or rewritten — do not assume it's sacred
9. The Agents SDK has many unused capabilities (see Research 1 above) — leverage them
10. The autonomous patterns research (see Research 2 above) provides proven patterns — use them
