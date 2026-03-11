# Revised Architecture Review - Blender VFX Orchestrator

Date: 2026-03-11

This revision incorporates the implementation feedback in `docs/ARCHITECTURE_REVIEW_FEEDBACK_20260311.md`, answers to the previously open questions, and additional runtime observations from 200+ pipeline runs.

## 1. EXECUTIVE SUMMARY

The original review remains directionally correct: the system's recurring failures come from advisory handoffs, destructive script evolution, missing deterministic scene assembly, and open-ended control/context boundaries. The key refinement is abstraction: the system should not revive API-call-level `APISpec` generation, because that was too granular and brittle; it should bind the LLM with `TechniqueContract + SceneSpec`, wrap advanced Blender features in capability packs, and preserve creativity inside named scene sections. Feedback from trace analysis indicates execution failure is still the largest failure class at roughly 45%, followed by low visual quality at roughly 35%, so the roadmap should prioritize contract binding, capability wrapping, bounded context, and patch-based recovery before evaluator tuning or model upgrades. The orchestrator should be decomposed incrementally as each contract lands, not in a late big-bang refactor. If implemented in that order, the architecture should become both more reliable and cheaper per run.

## 2. ROOT CAUSE ANALYSIS

1. Symptom: F1 Technique Monotony, F2 Script Writer Ignores Technique Research, and part of F5 Quality Plateau. Root cause: the pipeline discovers structured technique knowledge, then collapses it into prose before generation, so the active Script Writer can silently substitute training priors for selected technique. Evidence: `TechniqueDecision` is passed as prompt text in `orchestrator.py:1325`, research is truncated before generation in `orchestrator.py:1342`, the active writer is still a free-form tool user in `specialized_agents/script_writer.py`, and `script.technique_used` overwrites the selected technique in `orchestrator.py:1927`.

2. Symptom: F3 Recovery Scripts Lose Quality, F9 Vision Analyst Cannot See Code Precisely, F10 Rigid Body Constraint Context Error, and much of F11 Correction Complexity. Root cause: script evolution is destructive and text-first; failures are handled by whole-script rewrites, regex patching, and stacked correction layers rather than structure-preserving edits against a last-known-good baseline. Evidence: recovery explicitly tells the writer to regenerate a complete script in `phases/execution.py:274`, recovery disables doc-query enforcement in `hooks/enforcement_hooks.py:737`, `_modify_script_impl()` is regex-oriented in `tools/script_generator_tools.py:867`, the QA diagnosis bridge is symptom-to-parameter matching in `tools/qa_diagnosis_bridge.py`, and execution failures remain the plurality failure class in the feedback analysis.

3. Symptom: F4 Scene Geometry Does Not Fit Together and another large share of F5. Root cause: the architecture has no deterministic spatial reasoning layer, so support, contact, containment, and intersection are treated as creative prompt-following problems instead of computable geometry constraints. Evidence: geometry correctness currently lives in dynamic instruction text such as the cube-scale guidance and scene-design rules in `tools/dynamic_instructions.py`, but there is no dedicated placement/contact phase in the runtime pipeline and no spatial post-pass before render.

4. Symptom: F6 Research Agent Exhausts Turns, F7 Context Bloat Degrades Later Iterations, and F8 Orchestrator Monolith. Root cause: the system uses open-ended agent conversations and session-level history management where bounded stage logic and per-agent input filtering should exist. Evidence: the Research Agent is prompted to do PLAN -> RETRIEVE -> SYNTHESIZE with `TURNS: MAX 8` in `orchestrator.py:857`, research still runs with `max_turns=8` in `phases/research.py:80`, compaction is session-level in `orchestrator.py:267`, and the main control loop still centralizes phase orchestration in `orchestrator.py`.

5. Symptom: F1/F2 recur for Blender features that are weakly represented in training data, with Cell Fracture as the clearest example, and the same pattern will recur for future addon- or headless-specific features. Root cause: the architecture has been trying to teach novel Blender features through prompt injection and correction layers when it should wrap them behind verified capabilities. Evidence: the Cell Fracture instructions are hardcoded into `tools/dynamic_instructions.py`, the writer still defaults to code it "knows," addon availability is variable in the real runtime environment, and the original API-spec path failed not because binding is wrong, but because it bound at the wrong level of detail.

## 3. ARCHITECTURAL INTERVENTIONS

1. What changes: introduce a higher-abstraction spec-first boundary built around `TechniqueContract`, `SceneSpec`, and a split between `CreativePlanner` and `CodeRealizer`. The contract should bind technique, required capabilities, object inventory, semantic relationships, and quality targets, while leaving implementation details, materials, lighting, and artistic choices open. Why it fixes the root cause: it prevents the writer from re-deciding technique and scene structure while preserving P2, and it corrects the core mistake of the old `APISpec` approach by constraining "what must exist" rather than every API call. Estimated complexity: High.

2. What changes: build a capability-pack registry for advanced or weakly-trained Blender features. Each pack should declare addon dependencies, required operators, headless-safe wrappers, context setup, postconditions, and fallback behavior; technique selection should only choose from packs that are actually available in the runtime environment. Why it fixes the root cause: it moves the system from teaching to wrapping where teaching is unreliable, solves the Cell Fracture class of problems generally, and makes technique selection deterministic when the available capability set makes the choice computable. Estimated complexity: Medium-High.

3. What changes: standardize generated scripts around named sections or functions such as `setup_scene`, `create_geometry`, `setup_materials`, `setup_physics`, `setup_lighting`, `setup_camera`, and `bake_and_render`, and pair them with a sidecar manifest. Recovery should patch specific functions by name against a last-known-good baseline; full regeneration should be a deliberate escalation path, not the default. Why it fixes the root cause: it makes repair structure-preserving, supports targeted QA-to-code feedback, and gives `AdvancedSQLiteSession` branching a stable unit of modification. Estimated complexity: Medium.

4. What changes: add a deterministic scene-assembly and geometry-validation layer driven by `SceneSpec`. This layer should resolve support/contact, enforce containment, detect small penetrations or gaps, and optionally snap objects into physically plausible relationships before render. Why it fixes the root cause: floating glasses and visible gaps are not creativity failures; they are solvable geometry constraints and should be handled under P4. Estimated complexity: Medium.

5. What changes: turn research into a bounded three-stage flow: one planning step, bounded retrieval, one synthesis step. Use deterministic or parallel retrieval where possible and keep the research artifact compact and structured rather than conversational. Why it fixes the root cause: it directly addresses F6 and reduces wasted turns, while producing a cleaner input for `TechniqueContract` generation. Estimated complexity: Medium.

6. What changes: decompose the monolith incrementally as contracts land. Extract each phase into its own module at the moment its input/output contract stabilizes, and replace prose-heavy coordinators with deterministic policy where the decision is computable. Why it fixes the root cause: it reduces cognitive load immediately, avoids a dangerous late-stage refactor, and lets the new architecture pay for itself while it is being built. Estimated complexity: Medium.

7. What changes: recalibrate the evaluator against known-good historical assets and delay the GPT-5.4 model upgrade until after the contract architecture is in place. Why it fixes the root cause: evaluator drift may be masking progress, but changing models before fixing the architecture would only reproduce the same failure modes with better raw capability. Estimated complexity: Low-Medium.

## 4. PROPOSED ROADMAP

1. Phase 1: finish `TechniqueContract` and a minimal `SceneSpec`, and add the core metrics at the same time: technique adherence, recovery quality delta, failure class, and prompt/context size. Dependencies: none. Unblocks: capability packs, input filtering, bounded research, and patchable script conventions.

2. Phase 2: build the capability registry and the first wrapped pack, starting with Cell Fracture plus rigid body constraints. Include addon pre-checks, availability filtering, headless-safe wrappers, and deterministic technique selection whenever the registry makes the choice computable. Dependencies: Phase 1. Unblocks: reliable non-Mantaflow technique selection and the wrap-vs-teach boundary.

3. Phase 3: harden context and research boundaries. Add `call_model_input_filter`, convert research into the three-stage bounded flow, constrain tool surfaces per phase, and stop passing the entire session narrative downstream. Dependencies: Phase 1; benefits from Phase 2 registry information. Unblocks: lower cost, less drift, and more stable iteration quality.

4. Phase 4: enforce named script sections and implement patch-based recovery with a last-known-good baseline. Use `AdvancedSQLiteSession` branches only for ambiguous repair alternatives, and define explicit escalation from patch -> branch -> regenerate -> switch technique. Dependencies: Phases 1 and 3; benefits from Phase 2 capability wrappers. Unblocks: F3, a major share of F11, and lower recovery cost.

5. Phase 5: build the deterministic scene-assembly pass and geometry validation. Start with tabletop/contact scenes and container relationships, because those failures are common, visually obvious, and easy to score. Dependencies: Phase 1 for `SceneSpec`; benefits from Phase 4 patchability. Unblocks: F4 and meaningful progress on the low-quality bucket of failures.

6. Phase 6: extract stabilized phases from the orchestrator as they mature, rather than waiting for a late monolith split. Convert TechniqueSelector into deterministic policy where registry evidence is sufficient, make QualityGate mostly deterministic, and keep ModificationStrategy as a deterministic-plus-LLM hybrid for novel cases. Dependencies: each prior phase contract as it stabilizes. Unblocks: maintainability, targeted testing, and safer feature work.

7. Phase 7: calibrate the current evaluator with known-good assets, then upgrade to GPT-5.4 where it amplifies the new contracts rather than compensating for missing ones. Dependencies: Phases 1 through 6 for a stable architecture. Unblocks: trustworthy pass/fail semantics and better instruction-following once the system has something worth following.

8. Phase 8: retarget learning so it stores evidence-backed deltas against contracts, capability packs, and section-level repair patterns, not more prompt text. Dependencies: all prior phases. Unblocks: self-improvement that aligns with P6 and P7 without recreating the current correction-layer sprawl.

## 5. SDK FEATURES TO LEVERAGE

1. `call_model_input_filter`: this remains the single highest-leverage SDK feature. It should define exact per-phase information boundaries so later agents see current spec, current script, and current delta only.

2. `AdvancedSQLiteSession`: use it to branch from a last-known-good state during repair or technique-switch experiments, with strict branch limits to keep cost within budget.

3. `tool_input_guardrails` and `tool_output_guardrails`: use them to enforce capability-pack availability, technique adherence, section naming conventions, and patch-vs-regenerate policy at tool boundaries.

4. `tool_use_behavior="stop_on_first_tool"`: use it on helper agents or retrieval wrappers so they do one bounded thing and stop, especially in research and diagnostic subflows.

5. `parallel_tool_calls`: use it in bounded retrieval stages where several independent doc or artifact lookups can be issued together without opening an unconstrained conversation loop.

6. `is_enabled` and GPT-5.4 `allowed_tools`: dynamically hide tools that are irrelevant to the current phase or unavailable in the current runtime, especially capability packs gated by addon presence.

7. `needs_approval`: keep it for high-value HITL boundaries such as a second repair branch, a costly technique switch, or a run that exceeds its budget envelope.

8. `run_streamed`: use it for long-running iteration monitoring and early loop detection, not as a core reliability mechanism.

9. GPT-5.4 native compaction, large context, tool search, and better instruction following: these become valuable after contracts exist. They should be used as multipliers on a good architecture, not as substitutes for missing boundaries.

## 6. WHAT TO STOP DOING

1. Stop passing technique choice as prose into a generic writer and expecting obedience.

2. Stop reviving API-call-level `APISpec` generation; the old failure mode was over-constraining at the wrong layer, not the idea of binding itself.

3. Stop trying to teach novel Blender features solely through dynamic instructions, code patterns, and prompt examples when they should be wrapped.

4. Stop making whole-script regeneration the default response to execution failure.

5. Stop letting `script.technique_used` redefine the selected technique without an explicit adherence check.

6. Stop using open-ended research conversations as a substitute for staged retrieval and synthesis.

7. Stop storing operational truth in five or six overlapping fix layers.

8. Stop treating compaction or bigger context windows as the primary answer to context drift.

9. Stop postponing orchestrator extraction until one giant future refactor; extract along contract boundaries as soon as they stabilize.

10. Stop upgrading models before the evaluator is recalibrated and the contract architecture is in place.

## 7. OPEN QUESTIONS

1. What evidence threshold should move a feature from "wrapped capability" to "taught behavior"? This boundary should be explicit or the system will drift back toward prompt accumulation.

2. Which scene families should define version one of the deterministic spatial solver after tabletop/contact scenes: container liquids, room interiors, fracture debris, or atmospheric volumes?

3. Should addon installation remain manual per capability pack for now, or is automated capability-pack setup required for the intended user experience?

4. How much structural enforcement can the Code Realizer tolerate before it begins to measurably reduce creativity or scene diversity?

5. What is the exact branch budget for repair experiments before the system should escalate to HITL or stop, given the current monthly spend target?
