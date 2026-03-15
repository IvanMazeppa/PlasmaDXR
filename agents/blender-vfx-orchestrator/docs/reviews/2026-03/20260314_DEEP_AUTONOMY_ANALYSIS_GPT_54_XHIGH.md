# Deep Autonomy Analysis

Date: 2026-03-14

Scope:
- `agents/blender-vfx-orchestrator/`
- Mission-first, current-wave-aware, code-grounded analysis
- Incorporates online research on the OpenAI Agents SDK, Blender docs, and agent/evaluation design

## 1. EXECUTIVE SUMMARY

The Blender VFX Orchestrator has crossed an important threshold: it is no longer primarily blocked by basic technique selection or Blender API hallucinations. `TechniqueContract`, truth-pack validation, stale-render invalidation, section patching infrastructure, and GPT-5.4 adoption have moved the system from “can it ever work?” to “can it adapt correctly, learn honestly, and earn autonomy through evidence?” The answer is still “not yet,” but the reasons are now sharper: there is no single authoritative adaptation controller, the learning substrate is not yet provenance-safe, the evaluation loop still lacks typed repair semantics, and autonomy remains more aspirational than benchmark-gated.

Online research reinforces rather than overturns this direction. The current OpenAI Agents SDK publicly exposes durable HITL via `needs_approval` and `RunState`, and its guardrail docs make clear that tool guardrails only apply to function tools, which directly sharpens the repo’s existing concern about hot-path bypasses. Blender’s official cloth and collision docs also support a core architectural conclusion: some failures are inherently structural workflow problems, not scalar-parameter problems, so an adaptation loop that defaults to numeric tweaking will plateau even when evaluation is correct. The next phase should therefore prioritize authoritative repair routing, canonical iteration evidence, evaluator calibration, and a real experiment ledger before autonomy claims, more coordinators, or more memory layers.

## 2. WHAT THE CURRENT ROADMAP ALREADY LIKELY FIXES

1. Technique binding is materially improved. The combination of `TechniqueContract`, capability packs, truth-pack build, and contract adherence checks appears to have addressed the older “everything becomes Mantaflow” failure mode at the architecture level, even if some downstream quality problems remain.

2. Blender API hallucination risk is much lower than before. Runtime truth-pack grounding, validation, and deterministic correction mean the project is no longer primarily losing runs to basic deprecated-API mistakes.

3. Iteration honesty improved with the stale-render fix. The execution/evaluation boundary is now much closer to a trustworthy artifact boundary than it was before, which is a prerequisite for any legitimate learning or escalation logic.

4. The system has real structural-repair infrastructure, even if it is not yet authoritative. Section patching exists, patch budgets exist, and the runtime has a code-repair path available.

5. Context management is on a better track. `call_model_input_filter`, artifact-first handoffs, and general movement away from prompt-bloat all reduce one of the earlier systemic drags.

6. The overall architecture is converging toward the right macro-shape. The repo is less “agent swarm” and more “deterministic orchestration with bounded LLM specialists,” which is consistent with the mission and with external guidance favoring simpler composable patterns over ever-larger agent stacks.

## 3. WHAT ONLINE RESEARCH CHANGED OR SHARPENED

1. The current public OpenAI Agents SDK surface strengthens the case for hybrid HITL, but not as the immediate next priority. The official HITL docs show `needs_approval` on `function_tool`, `Agent.as_tool`, `ShellTool`, and `ApplyPatchTool`, with durable pause/resume via `RunState` and `result.to_state()`.[1] This confirms the repo should not keep treating native approvals as hypothetical. However, it also implies that HITL pays off only after repair routing is trustworthy enough to gate meaningful actions.

2. Official SDK guardrail semantics sharpen the repo’s “hot-path bypass” problem. The guardrails docs say tool guardrails run on every custom function-tool invocation, but only for tools that actually go through the function-tool pipeline.[2] This reinforces a key repo-local issue: direct `_modify_script_impl()` and deterministic execution/helper paths can bypass the per-tool safety/validation model you may think you have.

3. Public SDK distribution/release surfaces are themselves somewhat inconsistent in how easy they are to interpret at a glance. PyPI clearly shows `openai-agents` 0.12.2 published on March 14, 2026,[3] while cached release surfaces can lag or be misleading. This strengthens the repo’s emerging conclusion that version truth should be captured from the installed runtime and written into artifacts automatically rather than manually trusted from scattered docs.

4. Anthropic’s “Building effective agents” guidance strongly supports the repo’s move away from abstract multi-agent complexity and toward simpler workflows with explicit programmatic checks.[4] Two points are especially relevant: successful systems use simple composable patterns, and agents need ground truth from the environment at each step.[4] That maps directly onto the need for canonical manifests, deterministic `RepairIntent`, and a trustworthy iteration evidence plane.

5. Blender’s official cloth workflow docs support a major architectural conclusion about the Ireland Flag/cloth-class failures: correct outcomes depend on mesh setup, pinning, modifier ordering, collision objects, shared visibility, and bake workflow, not just scalar stiffness values.[5][6][7] This directly supports the repo’s need for explicit structural-vs-parameter repair semantics.

6. DeepMind’s MONA work sharpens the reward-integrity concern.[8] While it is a research result in a broader setting, its practical lesson translates well: if optimization is allowed to exploit approval/reward gaps across steps, the system can look locally reasonable while pursuing globally bad trajectories. In this repo’s setting, that means parameter-tweak loops can become “reward-compatible” even when they are not actually on a path to a quality scene.

## 4. REMAINING ROOT CAUSES

1. There is still no single authoritative adaptation controller.
Why it still matters after the current roadmap:
The repo now has the pieces for structural adaptation, but not a canonical place where repair mode is chosen.
Evidence basis:
`modify_code` exists, section patching exists, and direct parameter modifications exist, but the hot path still allows parameter-first paths to become authoritative before structural repair logic has to decide anything.
Consequence if ignored:
The system will keep looking like it has code-repair capability while mostly behaving like a parameter tuner under pressure.

2. Iteration-state authority is still fragmented.
Why it still matters after the current roadmap:
You cannot claim honest learning or autonomy if score history, stuck-state, issue persistence, and session truth are updated through multiple overlapping channels.
Evidence basis:
The repo still splits operational truth across `SessionState`, `SharedContext`, `SessionManager`, artifact files, and agent/coordinator outputs; some of the most advanced stuck-state logic exists but is not the live authority.
Consequence if ignored:
Plateau detection, escalation, learning, and retrospective analysis will continue to disagree about what actually happened.

3. The evaluation-to-repair boundary is still prose-heavy instead of typed.
Why it still matters after the current roadmap:
The evaluator can identify real issues, but the system still has to infer from prose whether a defect is structural, parametric, or technique-level.
Evidence basis:
`QualityOutput` is still centered on `primary_issue`, `issues`, and `suggestions` as text, with no first-class issue typing or repair-mode hints.
Consequence if ignored:
Correct visual diagnosis will keep getting translated into the wrong class of intervention.

4. The learning loop still lacks a clean causal experiment ledger.
Why it still matters after the current roadmap:
Evidence-gating only works if the evidence is provenance-safe, versioned, and attributable to real changes.
Evidence basis:
The repo records patterns and outcomes, but not yet as a tight, canonical per-iteration experiment object that captures script hash, valid artifact provenance, evaluator version, repair mode, and actual structural deltas.
Consequence if ignored:
The project can accumulate “learning” that is partly evaluator drift, partly repair-routing artifacts, and only partly genuine improvement.

5. Autonomy is not yet operationally defined in a benchmark-gated way.
Why it still matters after the current roadmap:
“Autonomous” still risks meaning “fewer checkpoints” instead of “proven reliable in a measured envelope.”
Evidence basis:
The mission statement is strong on philosophy, but the live repo still lacks a concise, enforced promotion framework by effect family, repair reliability, and evaluator trustworthiness.
Consequence if ignored:
Autonomy claims will remain rhetorical and difficult to defend.

6. Capability acquisition remains weaker than capability reuse.
Why it still matters after the current roadmap:
The mission is not just to repeat known techniques more cleanly. It is to discover and operationalize new Blender capabilities over time.
Evidence basis:
The repo is much better at binding known techniques and validating known APIs than it is at running bounded micro-experiments that can promote a novel feature into trusted capability.
Consequence if ignored:
The system may plateau as a better-known-technique orchestrator without becoming a true self-expanding agent.

## 5. ACTIONABLE INTERVENTIONS

1. Add a deterministic `RepairIntent` controller immediately after quality evaluation.
Why it matters:
This becomes the authoritative bridge between evaluator output and actual repair mode.
Covered by current roadmap:
Partially, but not explicitly enough.
Implementation shape:
Introduce a structured `RepairIntent` model that consumes typed quality issues, score trend, section availability, stuck-state, and code-grounded feedback; only after that should the runtime choose param edits, section patching, full rewrite, or technique switch.
Rough complexity / effort:
Medium.
How to verify it worked:
An Ireland Flag-style structural defect should reach `modify_code` by iteration 2 without requiring more iterations or coordinator luck.

```python
class RepairIntent(BaseModel):
    mode: Literal["modify_params", "modify_code", "switch_technique", "request_guidance"]
    trigger: Literal["parameter_issue", "structural_issue", "plateau", "technique_exhausted", "execution_failure"]
    confidence: float
    target_sections: list[str] = []
    target_params: dict[str, Any] = {}
```

2. Extend quality outputs or the QA bridge with typed repair semantics.
Why it matters:
The evaluator should not stop at free-text critique if the runtime depends on repair class.
Covered by current roadmap:
Mostly missed.
Implementation shape:
Add structured issues with fields like `kind`, `repair_mode_hint`, `target`, and `confidence`; optionally let the QA diagnosis bridge attach section hints and script targets.
Rough complexity / effort:
Medium.
How to verify it worked:
A small gold set of past traces should classify structural vs parameter vs technique issues with high agreement against human review.

3. Replace split iteration truth with a canonical run/iteration manifest plus one mutation API.
Why it matters:
The system needs one source of truth for what changed, what score was valid, what issue persisted, and why the next action happened.
Covered by current roadmap:
Partially.
Implementation shape:
Route iteration completion through a single function that updates session state, plateau counters, same-issue counters, best score, and a serialized per-iteration manifest; keep `SessionManager` as a derived read model only.
Rough complexity / effort:
Medium.
How to verify it worked:
The same information appears consistently in traces, persisted sessions, HITL checkpoints, and later analysis.

```python
class IterationManifest(BaseModel):
    iteration: int
    script_hash: str
    run_dir: str
    render_path: str | None
    artifact_valid: bool
    evaluator_profile: str
    repair_mode: str
    primary_issue: str | None
    score: float
```

4. Split learning into two layers: deterministic experiment ledger and slower distillation.
Why it matters:
Learning should not be doing too much hot-path actuation.
Covered by current roadmap:
Partially.
Implementation shape:
Move run logging, provenance capture, and delta recording into deterministic code; keep the Learning Agent for periodic pattern distillation, negative-knowledge synthesis, and optional advisory work when confidence is high.
Rough complexity / effort:
Medium.
How to verify it worked:
You can replay the experiment ledger without any LLM calls and still reconstruct what changed and what the measured outcome was.

5. Add an autonomy promotion contract by effect family.
Why it matters:
Autonomy should be benchmark-gated, not intuition-gated.
Covered by current roadmap:
Weakly.
Implementation shape:
Define thresholds for valid artifact rate, repair success rate, evaluator calibration confidence, cost envelope, and pass rate by effect family before relaxing HITL or claiming “autonomous.”
Rough complexity / effort:
Low-Medium.
How to verify it worked:
A dashboard or report can answer, for each effect family, whether it is Guided, Assisted, or Autonomous with evidence.

6. Build a bounded capability-incubation path distinct from production iteration.
Why it matters:
Production runs and novelty discovery should not be the same workflow.
Covered by current roadmap:
Partially.
Implementation shape:
Add an explicit `sandbox_experiment` mode: retrieve docs, generate a minimal script, run it, capture artifacts, and only then decide whether the feature becomes a pack candidate, truth-pack extension, or known anti-pattern.
Rough complexity / effort:
Medium.
How to verify it worked:
At least one previously weak feature is operationalized through the sandbox path without requiring a full-scene first attempt.

## 6. PRIORITIZED IMPLEMENTATION PLAN

Now

1. Implement `RepairIntent` and put all direct parameter paths behind it.
Dependency:
None beyond current code.
Owner assumption:
Solo developer.
Expected payoff:
Stops the system from defaulting into parameter loops on structural failures.
What it unblocks:
Real `modify_code` traces, meaningful section-patching proof, and cleaner evaluator feedback usage.

2. Add typed structural-vs-parameter issue semantics to the QA boundary.
Dependency:
Can begin in parallel with `RepairIntent`.
Owner assumption:
Solo developer.
Expected payoff:
Makes evaluator output actionable instead of interpretive prose.
What it unblocks:
Deterministic repair routing and better regression tests.

3. Unify iteration state and manifests.
Dependency:
Best done alongside `RepairIntent`.
Owner assumption:
Solo developer.
Expected payoff:
Makes plateau, same-issue, and evidence-gating trustworthy.
What it unblocks:
Evaluator calibration, learning validity, autonomy scoring, and HITL confidence.

Next

4. Prove `modify_code -> patch_script_section` end to end in a live case.
Dependency:
Repair routing and typed issue semantics.
Owner assumption:
Solo developer.
Expected payoff:
Turns code repair from “available” into “operational.”
What it unblocks:
Any credible claim that the system can structurally improve scenes.

5. Build an evaluator calibration ladder by physics family.
Dependency:
Canonical manifests and valid artifacts.
Owner assumption:
Solo developer.
Expected payoff:
Separates model weakness from evaluator weakness and reduces proxy optimization risk.
What it unblocks:
Learning promotion, autonomy gates, and meaningful baselines.

6. Move learning to an experiment-ledger-plus-distiller design.
Dependency:
Canonical manifests and calibrated evaluation.
Owner assumption:
Solo developer.
Expected payoff:
Makes “self-learning” less rhetorical and more auditable.
What it unblocks:
Negative knowledge, confidence-aware memory, and safer reuse.

Later

7. Add capability incubation / sandbox mode as a first-class workflow.
Dependency:
Reliable evaluation and manifests.
Owner assumption:
Solo developer.
Expected payoff:
Lets the system actually expand its repertoire instead of only improving known paths.
What it unblocks:
Novel technique acquisition, better pack growth, and more honest autonomy at Level 4.

8. Reintroduce deterministic scene assembly and first-render correctness checks.
Dependency:
Repair routing and evaluator calibration should exist first.
Owner assumption:
Solo developer.
Expected payoff:
Raises the quality ceiling on scene fit, support/contact relations, and obvious composition failures.
What it unblocks:
The jump from “evaluable” to “convincingly good” renders.

9. Continue coordinator reduction and semantic extraction from `orchestrator.py`.
Dependency:
Stable seams from the earlier steps.
Owner assumption:
Solo developer.
Expected payoff:
Less policy-by-prompt, better testability, lower ambiguity.
What it unblocks:
Safer future features and easier maintenance.

## 7. BENCHMARKS / EXPERIMENTS / MEASUREMENTS TO ADD

1. Structural-repair regression suite.
Create 5-10 cases where the expected fix is clearly structural, not parametric: reversed stripe order, wrong attachment topology, missing collider, wrong camera framing, absent key light. Success means the runtime chooses `modify_code` and either patches successfully or fails honestly.

2. Repair-mode classification benchmark.
Build a labeled dataset from historical traces with human labels for `modify_params`, `modify_code`, `switch_technique`, and `request_guidance`. Use it to measure agreement of the new typed QA/`RepairIntent` pipeline.

3. Evaluator calibration ladder by effect family.
Maintain a small fixed corpus of good, bad, and misleading renders for cloth, Mantaflow, rigid body, and mixed scenes. Track score bands, false-positive structural critiques, and disagreement cases.

4. Canonical-manifest integrity test.
For each completed iteration, verify that the persisted manifest, session state, trace summary, and scorecard all agree on render path, score, repair mode, and primary issue.

5. Learning provenance audit.
Periodically sample promoted patterns and confirm that each promotion has multiple clean underlying manifests with valid artifacts, clear deltas, and consistent evaluator profiles.

6. Capability-incubation benchmark.
Choose a feature that is not yet comfortable for the system, run it through doc retrieval -> micro-test -> artifact capture -> candidacy decision, and record whether the result becomes reusable knowledge.

## 8. WHAT TO STOP DOING

1. Stop letting direct parameter edits become the de facto repair controller.

2. Stop treating `iterate` as a sufficient action vocabulary for a system that must distinguish parameter tuning from structural repair.

3. Stop using prose-only evaluator feedback as the main repair-routing interface.

4. Stop assuming that because section patching exists, structural repair is already operational.

5. Stop treating more coordinator reasoning as a substitute for a deterministic adaptation boundary.

6. Stop using doc truth as a manual maintenance task; capture runtime truth into artifacts automatically.

7. Stop talking about autonomy mainly in terms of turning checkpoints on or off; define and enforce promotion thresholds.

8. Stop assuming capability packs are the same thing as capability acquisition.

## 9. OPEN QUESTIONS

1. Should typed structural-vs-parameter classification live in `QualityOutput`, in the QA diagnosis bridge, or in a deterministic repair controller that consumes both?

2. What is the minimum per-iteration manifest that is rich enough for learning validity without becoming burdensome to maintain?

3. Should one `modify_code` attempt be the default response to high-confidence structural issues, or should the runtime require section hints and confidence thresholds first?

4. Which effect family should be the first to receive an autonomy promotion scoreboard: cloth, liquid, rigid body, or a simpler canonical scene class?

5. How much of the Learning Agent should remain online versus moving to periodic offline distillation?

6. What is the cleanest way to separate `production`, `recovery`, and `sandbox_experiment` run modes without bloating the state machine again?

## 10. SOURCES

Local sources:
- `docs/MISSION_STATEMENT_2026-02-22.md`
- `docs/VERSION_TRUTH.md`
- `docs/reviews/2026-03/20260312_WAVE2_STATUS_AND_PATH_FORWARD.md`
- `docs/reviews/2026-03/20260313_ARCHITECTURE_REVIEW_WAVE2_RESPONSE.md`
- `docs/postmortems/2026-03/20260314_WAVE2A_POSTMORTEM_AND_MODIFY_CODE_CRISIS.md`
- `docs/reviews/2026-03/20260314_ARCHITECTURE_REVIEW_MODIFY_CODE_RESPONSE.md`
- `orchestrator.py`
- `phases/execution.py`
- `models/pipeline_models.py`
- `models/shared_context.py`
- `session_manager.py`

External sources:
1. OpenAI Agents SDK HITL docs: https://openai.github.io/openai-agents-python/human_in_the_loop/
2. OpenAI Agents SDK guardrails docs: https://openai.github.io/openai-agents-python/guardrails/
3. OpenAI Agents SDK PyPI release history: https://pypi.org/project/openai-agents/
4. Anthropic, “Building Effective AI Agents”: https://www.anthropic.com/engineering/building-effective-agents
5. Blender 5.0 Manual, cloth examples: `physics/cloth/examples.html`
6. Blender 5.0 Manual, cloth collisions: `physics/cloth/settings/collisions.html`
7. Blender Python API, `bpy.types.ClothModifier`: `bpy.types.ClothModifier.html`
8. Google DeepMind, “MONA: Myopic Optimization with Non-myopic Approval Can Mitigate Multi-step Reward Hacking”: https://deepmind.google/research/publications/mona-myopic-optimization-with-non-myopic-approval-can-mitigate-multi-step-reward-hacking/
