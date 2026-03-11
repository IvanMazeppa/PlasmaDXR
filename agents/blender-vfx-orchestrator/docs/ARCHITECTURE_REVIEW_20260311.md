• 1. EXECUTIVE SUMMARY

  The main failures are architectural, not prompt-engineering failures. The pipeline often discovers the right technique and constraints, then downgrades them into prose before generation and recovery, so the
  Script Writer re-decides technique, layout, and repair strategy from its priors. The second systemic issue is destructive script evolution: execution failures trigger whole-script rewrites and stacked fix
  layers instead of structure-preserving patches and verified capability adapters. The right roadmap is to restore a spec-first boundary, add deterministic spatial assembly and headless-safe capability packs,
  and then simplify the orchestrator around those artifacts; that preserves P2 because the LLM still invents the scene, but stops it from re-litigating facts that upstream phases already decided. Most of this
  is deterministic or reusable from code already in the repo, so it is realistic for a solo developer if repair branching is capped tightly.

  2. ROOT CAUSE ANALYSIS
  3. Symptom: F1 Technique Monotony, F2 Script Writer Ignores Technique Research, and a large part of F5 Quality Scores Plateau. Root cause: research and technique selection are advisory prose, not a binding
     generation contract, so the active writer is still a generic free-form code generator that can silently substitute its own learned priors. Evidence: the selected technique is flattened into prompt text
     at orchestrator.py:1325, research is truncated before generation at orchestrator.py:1342, the active Script Writer exposes only free-form write_script/modify_script/validate_script at
     script_writer.py:129, the input guardrail only checks for generic “research context” at script_guardrails.py:177, and the writer’s returned technique_used becomes authoritative session state at
     orchestrator.py:1927.
  4. Symptom: F3 Recovery Scripts Lose Quality, F9 Vision Analyst Can’t See Code, F10 Rigid Body Constraint Context Error, and F11 Correction Complexity. Root cause: the system evolves scripts destructively,
     through whole-script rewrites and regex patches, instead of preserving structure and routing advanced Blender features through verified headless-safe capability adapters. Evidence: recovery explicitly
     tells the writer to generate a fixed version of the complete script at execution.py:274, error-recovery hooks disable doc-query enforcement at enforcement_hooks.py:737, _modify_script_impl() is
     fundamentally regex-based at script_generator_tools.py:867, the QA bridge is keyword-to-parameter matching rather than causal diagnosis at qa_diagnosis_bridge.py:31, and pre-execution correction is
     layered across validator, truth-pack fix, and legacy fixer at orchestrator.py:1961, execution.py:169, and blender_executor_tools.py:512.
  5. Symptom: F4 Scene Geometry Doesn’t Fit Together and another large share of F5. Root cause: there is no deterministic scene-assembly layer that reasons about support, contact, containment, or
     intersection; spatial correctness is left to prompt following. Evidence: geometry guidance exists only as instruction prose at dynamic_instructions.py:373 and dynamic_instructions.py:394, but there is no
     pipeline phase between generation and execution that solves object relationships, and the active modification machinery remains scalar/code-line oriented at script_generator_tools.py:867.
  6. Symptom: F6 Research Agent Exhausts Turns, F7 Context Bloat Degrades Downstream Agents, and F8 Orchestrator Monolith. Root cause: open-ended agents and session-wide prompt accumulation are being used
     where bounded stage logic and per-agent input filtering should exist. Evidence: the Research Agent is explicitly prompted to do PLAN/RETRIEVE/SYNTHESIZE with TURNS: MAX 8 at orchestrator.py:857, research
     runs with max_turns=8 at research.py:80, compaction is only session-level at orchestrator.py:267, presets still default to stateless_iterations=True at agent_config.py:108, and the core pipeline loop
     still lives in orchestrator.py:1018.
  7. Symptom: F1/F2 recur whenever Blender features are weakly represented in training, with Cell Fracture being the clearest example. Root cause: the architecture is trying to “teach” unseen Blender features
     via prompt injection and correction layers instead of exposing them as verified capabilities with required operators, context rules, and success checks. Evidence: Cell Fracture instructions are hardcoded
     into dynamic instructions at dynamic_instructions.py:278, the technique-aware catalog path still exists but is not on the active writer tool list at script_generator_tools.py:518 and
     script_writer.py:138, and the runtime authority is currently split across truth pack, tool guardrails, hooks, validator, and legacy fixer rather than a single capability layer.
  8. ARCHITECTURAL INTERVENTIONS
  9. What changes: reinstate a spec-first boundary, but broader than the old APISpec: ResearchBrief -> TechniqueContract -> SceneSpec -> CodeRealizerInput. Use the existing truth_pack_to_api_spec() and
     deprecated Code Writer pattern as starting points, not as dead history. Why it fixes the root cause: technique, required operators/addons, object graph, spatial relations, and headless constraints become
     binding inputs to generation instead of advisory prose, which directly fixes F1/F2 and raises the ceiling on F5. Estimated complexity: High.
  10. What changes: replace rewrite-on-failure with structure-preserving recovery. Generated scripts should have named sections/functions plus a sidecar manifest, repairs should target those sections with
     patch ops, and full regeneration should be allowed only when the spec is invalid or the technique changes; keep a last-known-good script/spec pair. Why it fixes the root cause: recovery stops being the
     least-grounded path, advanced features survive execution fixes, and QA feedback can point to concrete sections instead of forcing a fresh 900-line rewrite; this addresses F3/F9/F10/F11. Estimated
     complexity: Medium.
  11. What changes: add a deterministic scene-assembly pass driven by SceneSpec, with support/contact graphs, containment constraints, bounding-box sanity checks, and pre-render snap/resolve for small gaps or
     penetrations. Why it fixes the root cause: floating glasses, gaps, and incorrect intersections are geometry-constraint problems, not creativity problems, so they should be solved deterministically after
     the LLM decides what the scene should be. Estimated complexity: Medium.
  12. What changes: consolidate correction logic into technique capability packs plus two validation boundaries. Each capability pack should declare required addons/operators, headless-safe wrappers, context
     requirements, doc refs, and postconditions; the only global authorities should be pre-generation capability validation and pre-execution artifact validation. Why it fixes the root cause: this replaces
     six overlapping correction mechanisms with one way to represent “how to do Cell Fracture / rigid body constraints / headless-safe baking,” which is the general solution for advanced Blender features
     beyond training data. Estimated complexity: Medium-High.
  13. What changes: decompose orchestration around typed phase runners and a deterministic policy engine. Keep Research Retriever/Synthesizer, Creative Planner, Code Realizer, Quality Analyst, and Learning
     Agent; keep execution deterministic; collapse Technique Selector, Modification Strategist, and most of Quality Gate into policy code over typed artifacts and metrics; keep DocsExpert as tools, not a
     conversational agent. Why it fixes the root cause: fewer agents will be making prose-level adjacent decisions, phases become independently testable, and F6/F7/F8 stop being emergent properties of one
     giant loop. Estimated complexity: Medium.
  14. PROPOSED ROADMAP
  15. Phase 0: add baseline metrics for technique adherence, recovery quality delta, spatial-contact failures, artifact-gate failure class, and prompt/context size per agent. Dependencies: none. Unblocks:
     objective validation of every later change and budget control.
  16. Phase 1: introduce TechniqueContract and SceneSpec, and split the current Script Writer into Creative Planner and Code Realizer. Dependencies: Phase 0. Unblocks: binding technique choice, capability
     packs, patch-based recovery, and deterministic geometry.
  17. Phase 2: enforce a patchable script format and implement last-known-good recovery with bounded branching. Dependencies: Phase 1, because patches need stable section names and manifest structure.
     Unblocks: quality-preserving execution recovery and a more precise QA-to-code loop.
  18. Phase 3: build the scene-assembly layer and the first capability packs, starting with Cell Fracture, rigid body constraints/headless animation, and tabletop/support-contact scenes. Dependencies: Phase 1
     for scene/spec contracts; benefits from Phase 2 for repair. Unblocks: non-Mantaflow breadth, geometry quality, and the most visible score improvements.
  19. Phase 4: decompose the orchestrator around those artifacts, add per-agent input filtering, and replace prose-heavy coordinators with deterministic policy functions. Dependencies: Phases 1-3 must
     stabilize the contract between stages first; decomposing earlier would just spread the current ambiguity across more files. Unblocks: independent phase tests, safer HITL, lower token burn, and easier
     future features.
  20. Phase 5: retarget learning so it stores validated deltas against specs/capability packs, not more prompt text. Dependencies: Phases 1-4. Unblocks: self-improvement that respects P6/P7 without recreating
     today’s instruction sprawl.
  21. SDK FEATURES TO LEVERAGE
  22. call_model_input_filter: highest leverage. Give each stage only the current spec, current script, and current delta, instead of a growing session narrative; this is the real fix for F7, not bigger
     context windows.
  23. AdvancedSQLiteSession: use branches for recovery alternatives and technique-switch experiments, always from a last-known-good baseline, and cap branch count tightly to protect budget.
  24. tool_input_guardrails and tool_output_guardrails: enforce that planner output names a valid capability pack, write_script adheres to the selected TechniqueContract, and repair tools cannot escalate to
     full regeneration unless the policy engine explicitly allows it.
  25. tool_use_behavior="stop_on_first_tool" plus is_enabled: use them on retriever/diagnosis/helper agents so they do one bounded tool action and stop wandering; also hide irrelevant or expensive tools
     outside the phase that needs them.
  26. needs_approval: use native HITL only at high-value boundaries such as second repair branch, technique switch after repeated failure, or budget overrun; that gives you supervised autonomy without custom
     control flow.
  27. run_streamed, GPT-5.4 allowed_tools, and tool search: stream long runs for loop detection and monitoring, constrain each stage to the exact tool surface it should have, and defer loading heavy tool sets
     until the stage actually needs them.
  28. GPT-5.4 1M context and native compaction: use them for artifacts and branch comparisons, not as permission to keep whole iteration histories in every prompt. They are headroom, not a substitute for
     architectural boundaries.
  29. WHAT TO STOP DOING
  30. Stop passing technique choice as prose into a generic writer and expecting obedience.
  31. Stop letting script.technique_used overwrite the selected technique without an adherence check.
  32. Stop making full-script regeneration the default response to execution failure.
  33. Stop storing Blender technique knowledge simultaneously in dynamic instructions, truth-pack fixes, validator tables, API fixer rules, and guardrails.
  34. Stop using dynamic instructions as the primary carrier of geometry rules, recovery policy, and advanced technique tutorials.
  35. Stop treating compaction or larger context windows as the main answer to context drift.
  36. Stop keeping separate conversational agents for decisions that can be deterministic policy over typed artifacts and metrics.
  37. OPEN QUESTIONS
  38. Why was the spec-first APISpec/Code Writer path deprecated in practice: cost, latency, quality regression, or implementation pain? That affects whether to revive it directly or redesign it.
  39. Across the 200+ runs, what is the actual failure distribution by count and cost: execution failure, artifact-gate failure, low visual quality, or research drift? That determines how much branch-repair
     budget is justified.
  40. Is the current 60+ pass threshold calibrated to the present evaluator stack, or has score distribution shifted since earlier successful runs? The plateau still looks architectural, but threshold drift
     would affect rollout strategy.
  41. How standardized are current generated scripts structurally? If function/section boundaries vary too much, section-based patching may require a one-time generator-format migration.
  42. How stable is addon availability in the real runtime environment, especially object_cell_fracture and other advanced systems you want to treat as first-class capabilities?