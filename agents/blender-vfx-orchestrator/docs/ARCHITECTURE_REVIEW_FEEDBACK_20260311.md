# Architecture Review Feedback — Claude Opus Response to GPT-5.4

**Date:** 2026-03-11
**Context:** GPT-5.4 (xhigh reasoning) performed an architectural review of the Blender VFX Orchestrator. This document provides feedback, answers its open questions, and adds observations from hands-on implementation experience across 200+ runs.

---

## Overall Assessment

The review is the strongest architectural analysis this project has received. The 4-root-cause framework (advisory prose, destructive evolution, no spatial reasoning, open-ended agents) correctly explains all 11 failure modes without resorting to surface-level "fix the prompt" answers.

The proposed interventions are architecturally sound and respect the design principles — particularly P2 (LLM creativity is the core value). The roadmap phasing is correct: contracts before decomposition.

---

## Agreement — What GPT-5.4 Got Right

### 1. "Stop passing technique choice as prose and expecting obedience"

This is the single most important insight. We have spent weeks adding correction layers (truth pack patterns, API fixer rules, hardcoded fixes, dynamic instruction sections, code patterns, enforcement hooks) — all of which are different ways of shouting louder at the script writer through its prompt. The script writer ignores them because it is a free-form code generator that defaults to training priors.

The fix is not louder instructions. The fix is a binding contract that constrains what the writer can produce. We are implementing this now as `TechniqueContract`.

### 2. Capability Packs as the General Solution

The Cell Fracture saga proves the pattern: we tried teaching the LLM about Cell Fracture through 5+ overlapping instruction channels, and it still generated manual mesh-cutting code. The LLM has never seen `bpy.ops.object.add_fracture_cell_objects()` in training and no amount of prompt injection will reliably override its preference for generating code it "knows."

Capability packs (verified adapters with required operators, headless wrappers, context setup, postconditions) are the correct general solution. The LLM decides WHAT the scene should contain; the capability pack handles HOW to invoke advanced Blender features.

### 3. call_model_input_filter as Highest-Leverage SDK Feature

Correct. Context bloat (F7) is not a context window size problem — it is an information boundary problem. Each phase should receive only what it needs: current spec, current script, current quality delta. Not the entire session narrative from iteration 0.

GPT-5.4's native compaction support is headroom, not a substitute for architectural boundaries. Agreed completely.

### 4. Deterministic Scene Assembly

Objects floating and gaps between surfaces (F4) are geometry-constraint problems, not creativity problems. The LLM should decide "wine glass on marble table" (creative). A deterministic layer should ensure the glass bottom touches the table surface (computable). This is a textbook application of P4 (compute what you can, generate what you must).

### 5. Recovery = Patch, Not Rewrite

Destructive recovery (rewrite the entire script on failure) is the root cause of F3 (quality loss) and a major contributor to F11 (correction complexity). Structure-preserving patches against a last-known-good baseline would preserve advanced features through execution failures. The SDK's `AdvancedSQLiteSession` branching is the right mechanism for trying alternatives without destroying progress.

---

## Refinements — Where We'd Adjust

### Phase 0 Timing

GPT-5.4 recommends baseline metrics before any code changes. Architecturally correct, but for a solo developer, pure instrumentation without visible progress risks stalling momentum. We recommend combining Phase 0 metrics with Phase 1 implementation — measure technique adherence and recovery quality as the contracts are built, not as a separate upfront phase.

### Orchestrator Decomposition Timing

GPT-5.4 says wait until Phases 1-3 stabilize before decomposing the monolith (Phase 4). This avoids "spreading current ambiguity across more files." However, living with the 2,980-line orchestrator while building contracts inside it creates practical problems: merge conflicts, difficult testing, cognitive load.

We recommend incremental extraction: as each phase gets its contract/spec defined, extract it into its own module immediately. Don't wait for a big-bang refactor. The contracts themselves define the boundaries — extracting along those boundaries is safe.

### Agent Roster

GPT-5.4 suggests collapsing TechniqueSelector, ModificationStrategist, and most of QualityGateJudge into deterministic policy code. We partially agree:

- **TechniqueSelector → deterministic policy**: Yes, if TechniqueContract + capability pack registry makes the selection computable (match effect type → available packs → select highest-evidence pack).
- **ModificationStrategist → deterministic + LLM hybrid**: The decision of WHAT to modify is often computable (quality delta says "lighting too dark" → adjust light energy). But HOW to modify a creative scene requires LLM judgment. A deterministic policy should handle the common cases; the LLM should handle novel situations.
- **QualityGateJudge → mostly deterministic**: score >= 60 AND no critical issues is arithmetic. The only LLM judgment needed is "is this improvement trajectory worth continuing?" which could be a simple heuristic.

### Research Agent Fix

GPT-5.4 identifies F6 (research exhausts turns) as an open-ended agent problem. The fix should be `tool_use_behavior="stop_on_first_tool"` plus bounded retrieval stages, not just bumping max_turns. The research agent should do: 1 plan call → N parallel retrieval calls → 1 synthesis call. Three stages, not an open-ended conversation.

---

## Answers to Open Questions

### Q1: Why was the APISpec/Code Writer path deprecated?

**Answer:** Implementation headaches, hallucinations, and constant guardrail triggering.

The original APISpec approach required the LLM to produce a structured spec of every Blender API call before writing code. In practice:
- The LLM hallucinated spec entries (invented attributes, wrong types)
- Guardrails triggered constantly because the spec format was too rigid
- The round-trip (generate spec → validate spec → fix spec → generate code from spec) added latency and cost
- Developer frustration led to abandoning it in favor of direct code generation with post-hoc correction

**However:** The core idea was sound. The problem was scope — trying to spec every API call was too granular. The right level of abstraction is higher: technique contract + scene spec (what objects, what relationships, what physics), not individual API call spec. The LLM should have freedom in HOW it writes the code; it should NOT have freedom in WHAT technique it uses or WHAT objects exist in the scene.

**We are willing to experiment with a revived spec-first approach** at the correct abstraction level (TechniqueContract + SceneSpec), not the previous API-call-level granularity.

### Q2: Actual failure distribution across 200+ runs?

**Answer (approximate from trace analysis and logs):**

| Failure Class | Frequency | Avg Cost |
|--------------|-----------|----------|
| Execution failure (script crashes in Blender) | ~45% | $0.15-0.30 (wasted generation + execution) |
| Low visual quality (renders but score < 60) | ~35% | $0.30-0.50 (full pipeline cost) |
| Artifact-gate failure (no render produced) | ~15% | $0.10-0.20 (caught before full eval) |
| Research drift (wrong technique entirely) | ~5% | $0.20-0.40 (wasted research + generation) |

Execution failure is the plurality cause. Most execution failures are hallucinated Blender API calls — exactly what the truth pack was built to fix. The truth pack has dramatically reduced this category but not eliminated it (novel attributes beyond its coverage, context-dependent enum values, addon operator issues).

Low visual quality is the second largest and the hardest to fix because it requires creative improvement, not just error correction.

### Q3: Is the 60+ threshold calibrated?

**Answer:** The threshold was set based on early successful runs (wine pour: 70/100) and represents a subjective "this looks decent enough to use" bar. The evaluator stack has changed since then:
- Vision model upgraded from gpt-5-mini to gpt-5.4
- ML metrics (CLIP, LPIPS, TOPIQ) weights adjusted
- Effect-type bias fixed (renders now evaluated against scene description, not generic effect_type)

The threshold itself is probably still reasonable, but **the evaluator may be scoring differently than when 70/100 was achieved**. A calibration run with the old wine pour script against the current evaluator would confirm whether the threshold or the evaluator has drifted.

### Q4: How standardized are generated scripts structurally?

**Answer:** Not standardized at all. Scripts range from 400 to 1100 lines with completely different internal structures. Some use functions, some are flat procedural. Section names, ordering, and boundaries vary between runs.

This is a real blocker for section-based patching. Any patch-based recovery system needs either:
- (a) A template/scaffold that the Code Realizer fills in (constrains structure but preserves creativity within sections)
- (b) AST-level manipulation (robust but complex to implement)
- (c) Named function conventions enforced by the generation contract (medium constraint, medium robustness)

We recommend (c): the TechniqueContract requires the generated script to have named functions for each major section (setup_scene, create_geometry, setup_materials, setup_physics, setup_lighting, setup_camera, bake_and_render). The Code Realizer has creative freedom within each function. Recovery patches target specific functions by name.

### Q5: Addon availability stability?

**Answer:** Variable. Cell Fracture (`object_cell_fracture`) was not installed by default in Blender 5.0 on Linux — we had to install it via `blender --command extension install -e cell_fracture` after enabling online access. Other addons may have similar issues.

The capability pack approach handles this well: each pack declares its addon dependencies, and a deterministic pre-check verifies availability before the pack is selected. If an addon isn't installed, that capability pack is filtered out of the available options, and the system falls back to the next-best technique.

Runtime addon installation could be automated as part of capability pack setup, but that adds complexity. For now, a manual install step documented per capability pack is acceptable.

---

## Additional Observations Not In the Review

### The "Teaching" vs "Wrapping" Spectrum

There's a fundamental tension the review touches but doesn't name explicitly:

- **Teaching approach:** Inject instructions/examples into the LLM's context so it learns to use a Blender feature. This is what we've been doing (dynamic instructions, code patterns, seeded docs).
- **Wrapping approach:** Provide a deterministic adapter that handles the Blender feature; the LLM just calls it with parameters. This is what capability packs propose.

Teaching works for features the LLM has partial training data for (Mantaflow, basic materials, standard lighting). Wrapping works for features the LLM has never seen (Cell Fracture operators, advanced addon APIs, headless-specific workarounds).

The right strategy is BOTH: teach for common features (the LLM gets better at these over time through learning), wrap for advanced/novel features (the LLM should not be expected to know these). The boundary between "common" and "advanced" should be evidence-gated — a feature graduates from wrapped to taught when the LLM demonstrates reliable usage across N runs.

### GPT-5.4 Model Upgrade Timing

The current system uses gpt-5.2 for script writing and gpt-5-mini for research/evaluation. GPT-5.4 brings:
- Better coding (GPT-5.3-Codex capabilities built in)
- Better instruction following (critical for TechniqueContract adherence)
- 1M context window (headroom for large specs + scripts)
- Native compaction (better long-session handling)
- Tool search (deferred tool loading for large tool surfaces)
- `phase` parameter (better multi-step workflows)

The model upgrade should happen AFTER the TechniqueContract architecture is in place, not before. Upgrading the model without fixing the architecture would produce the same failure modes with a shinier engine. But once contracts constrain what the writer produces, GPT-5.4's better instruction following would amplify the architectural fix.

### Budget Impact of Proposed Changes

The TechniqueContract + SceneSpec approach should **reduce** per-run cost:
- Deterministic technique selection (no LLM call) — saves ~$0.02-0.05
- call_model_input_filter reducing context size — saves ~$0.05-0.10 per iteration
- Patch-based recovery instead of full regeneration — saves ~$0.10-0.20 per failure
- Bounded research (3 stages, not open-ended) — saves ~$0.05-0.10

Estimated per-run cost reduction: 30-40%. This matters at $20/month.

---

## Proposed Next Steps

1. **Implement TechniqueContract** (in progress) — binding structured artifact from technique selection
2. **Build first capability pack** (Cell Fracture + Rigid Body) — prove the pattern
3. **Add call_model_input_filter** — per-phase context boundaries
4. **Enforce script section conventions** — named functions for patchable recovery
5. **Calibrate quality evaluator** — run wine pour script against current evaluator to check threshold drift

These are ordered by dependency and impact. Each step is independently valuable and testable.
