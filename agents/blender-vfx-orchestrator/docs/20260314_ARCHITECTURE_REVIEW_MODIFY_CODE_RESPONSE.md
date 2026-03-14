# Architecture Review Response - The `modify_code` Crisis

Date: 2026-03-14

This document responds to:
- `docs/20260314_WAVE2A_POSTMORTEM_AND_MODIFY_CODE_CRISIS.md`
- the live runtime code after the Wave 2A stale-render fix
- the existing Wave 2 review sequence

## 1. EXECUTIVE SUMMARY

The Ireland Flag postmortem identifies the right new bottleneck, but its root-cause framing is still one layer too shallow. The system is not missing a `modify_code` capability; `modify_code`, section patching, plateau logic, and escalation concepts already exist in the runtime. The actual problem is that repair routing is fragmented and parameter-first fast paths become authoritative before structural repair logic ever gets a vote. In practice, the pipeline is structurally biased toward cheap parameter edits, while the signals that should trigger `modify_code` live in separate state owners and mostly non-authoritative LLM outputs. The next roadmap change should therefore be: build one deterministic `RepairIntent` boundary after quality evaluation, make it authoritative, and only then decide whether to apply parameters, patch code, switch technique, or escalate.

## 2. ROOT CAUSE ANALYSIS

1. Symptom: structural problems still produce `modify_params` behavior. Root cause: parameter-first short-circuits run before the Modification Coordinator and before any authoritative structural routing step. Evidence: in `orchestrator.py`, the hot path first applies `learning.parameter_modifications` directly, then runs deterministic parameter fallback from `map_quality_issues_to_params()`, and only consults the Modification Coordinator if those paths fail. This means `modify_code` is not the primary repair path; it is the fallback after parameter paths are exhausted.

2. Symptom: plateau and escape signals do not meaningfully drive live routing. Root cause: the system has two partially disconnected stuck-state systems, and the more capable one is bypassed in the hot path. Evidence: `models/shared_context.py` already has `StuckDetectionState.update_from_iteration()` with both `same_issue_count` and `plateau_count`, but the runtime appends directly to `session.iterations` instead of calling `SessionState.record_iteration()`. At the same time, `SessionManager` separately tracks `consecutive_same_issue`, and then the Quality Gate overwrites `session.stuck_state.escape_level` from coordinator output. Plateau detection exists in code, but it is not authoritative in routing.

3. Symptom: the QA can describe structural defects accurately, but the pipeline does not act structurally on them. Root cause: the quality-to-repair boundary is prose-only. `QualityOutput` carries `primary_issue`, `issues`, and `suggestions` as strings, but no typed repair semantics such as `issue_kind=STRUCTURAL` or `repair_mode=modify_code`. Evidence: the Modification Coordinator prompt asks the model to infer `modify_params` vs `modify_code` from prose, but that prompt is often never reached because earlier parameter fast paths already succeeded.

4. Symptom: the Learning Agent repeatedly returns `iterate` while also supplying parameter edits, so the loop keeps nudging numbers even when structure is wrong. Root cause: the action vocabulary is too coarse and the prompt bias is toward parametric modification. Evidence: `LearningOutput.next_action` only supports `iterate`, `switch_technique`, and `complete`, while the learning prompt explicitly emphasizes `parameter_modifications` from script analysis. The runtime then treats concrete parameter modifications as immediately actionable, which effectively collapses `iterate` into `modify_params`.

5. Symptom: increasing `max_iterations` looks tempting as a fix. Root cause: the routing bug masquerades as an iteration-budget problem. Evidence: in the Ireland Flag case, three iterations were enough to demonstrate that structural problems were being observed but routed into the wrong repair mode. With the current architecture, five iterations would mostly create a longer parameter loop, not a more intelligent escalation path.

## 3. ARCHITECTURAL INTERVENTIONS

1. What changes: add an authoritative deterministic `RepairIntent` stage between quality evaluation and any modification path. It should consume the quality result, score history, code-grounded feedback, section availability, and current stuck-state, then emit one of `modify_params`, `modify_code`, `switch_technique`, or `request_guidance`. Why it fixes the root cause: it prevents parameter fast paths from preempting structural repair and gives the orchestrator one canonical repair decision instead of inferring it from several unrelated signals. Estimated complexity: Medium.

```python
class RepairIntent(BaseModel):
    mode: Literal["modify_params", "modify_code", "switch_technique", "request_guidance"]
    trigger: Literal["parameter_issue", "structural_issue", "plateau", "execution_failure", "technique_exhausted"]
    confidence: float = 0.0
    target_sections: list[str] = []
    target_params: dict[str, Any] = {}
    reasoning: str = ""


def choose_repair_intent(
    quality: QualityOutput,
    code_feedback: CodeGroundedFeedback | None,
    state: IterationState,
    script_sections: list[str],
) -> RepairIntent:
    if quality_has_structural_issue(quality, code_feedback):
        return RepairIntent(mode="modify_code", trigger="structural_issue", target_sections=section_hints(code_feedback))
    if state.plateau_count >= 2:
        return RepairIntent(mode="modify_code", trigger="plateau")
    if quality_has_parametric_issue(quality):
        return RepairIntent(mode="modify_params", trigger="parameter_issue", target_params=extract_param_targets(quality))
    if state.same_issue_count >= 3:
        return RepairIntent(mode="switch_technique", trigger="technique_exhausted")
    return RepairIntent(mode="modify_params", trigger="parameter_issue")
```

2. What changes: extend the quality boundary from free-text critique to typed repair semantics. Add structured issue entries such as `kind`, `repair_mode_hint`, `target`, and `confidence` to the QA output or to the QA diagnosis bridge. Why it fixes the root cause: the system should not have to rediscover from prose that “wrong stripe orientation” is structural while “raise bump strength” is parametric. Estimated complexity: Medium.

```python
class QualityIssue(BaseModel):
    summary: str
    kind: Literal["parameter", "structural", "technique", "camera", "lighting"]
    repair_mode_hint: Literal["modify_params", "modify_code", "switch_technique"]
    target: str | None = None
    confidence: float = 0.0


class QualityOutput(BaseModel):
    overall_score: float
    passed: bool
    primary_issue: str | None = None
    issues: list[str] = []
    structured_issues: list[QualityIssue] = []
    suggestions: list[str] = []
    vision_assessment: str = ""
```

3. What changes: unify live iteration-state authority. Route iteration completion through one mutation API that updates score history, `same_issue_count`, `plateau_count`, escape level, and persistence, then let `SessionManager` become a derived read model instead of a second operational truth source. Why it fixes the root cause: plateau logic and stuck-state escalation only matter if the hot path actually uses them. Estimated complexity: Medium.

4. What changes: demote the Learning Agent from repair-mode authority to repair-content advisory. It can still propose parameter values, extract patterns, and support distillation, but it should not be the thing that implicitly decides whether the system is in param-edit mode or code-edit mode. Why it fixes the root cause: right now the presence of `parameter_modifications` becomes a routing decision by accident. Estimated complexity: Low-Medium.

5. What changes: make `modify_code` proof a first-class integration target. Add one production-style regression where the expected path is `modify_code -> patch_script_section -> execute -> evaluate`, and another where plateau or structural classification forces `modify_code` by iteration 2. Why it fixes the root cause: the pipeline should not declare structural repair operational until the live loop has actually used it on a real failure class. Estimated complexity: Low.

## 4. PROPOSED ROADMAP

1. Wave 2A.1 - Make repair routing authoritative. Content: implement `RepairIntent`, move all direct parameter application behind it, and add typed structural-vs-parameter classification to quality feedback or the QA bridge. Dependencies: Wave 2A stale-render fix is already complete. Exit criteria: a structural defect case such as Ireland Flag reaches `modify_code` no later than iteration 2.

2. Wave 2A.2 - Unify stuck-state and iteration authority. Content: route iteration completion through one mutation path, make plateau and same-issue counts visible in the live decision context, and stop treating coordinator escape-level output as the canonical stuck-state. Dependencies: Wave 2A.1, because repair intent should consume the unified state. Exit criteria: score plateaus, issue persistence, and escape level all come from one source of truth.

3. Wave 2B - Prove structural repair and calibrate reward. Content: run a real section-patching proof, calibrate evaluator score bands by physics family, and only then add hybrid HITL approvals to high-impact repair actions. Dependencies: Wave 2A.1 and 2A.2. Exit criteria: `modify_code` works in traces, evaluator behavior is measured, and HITL is gating meaningful decisions rather than a broken routing loop.

4. Wave 2C - Reduce cost and coordinator ambiguity. Content: bounded research retrieval, narrower coordinator roles, and deterministic policy for obvious threshold decisions. Dependencies: Wave 2B benefits this work, but parts can start once routing is fixed. Exit criteria: fewer conversational control decisions, lower token spend, and less prompt-based policy simulation.

5. Wave 3 - Raise the quality ceiling. Content: deterministic scene assembly / spatial-contact logic, first-render correctness checks for obvious structural errors, and novelty incubation through bounded sandbox experiments. Dependencies: reliable repair routing and evaluator calibration. Exit criteria: the system can both recognize and structurally correct scene-level defects, not just patch parameters around them.

Clarifying note: do not increase default `max_iterations` to 5 as the primary response to this postmortem. A larger iteration budget is useful only after structural repair routing is trustworthy. Before that, it mostly hides the bug by making the rut longer.

## 5. SDK FEATURES TO LEVERAGE

1. Structured outputs: this is the immediate leverage point. Use them to make structural-vs-parameter repair hints explicit in QA output, `RepairIntent`, and modification decisions instead of inferring them from prose.

2. Tool output guardrails: use them to require that quality or diagnosis tools return a valid repair classification before the orchestrator enters modification. This is more useful right now than additional agent creativity.

3. `call_model_input_filter`: keep it in place so the new repair intent stage sees compact, current evidence rather than verbose history.

4. `tool_use_behavior="stop_on_first_tool"`: useful on the section-patching path so the patching agent reads the script, patches the targeted section, and stops instead of wandering through unnecessary tool loops.

5. `needs_approval` + `RunState`: still valuable, but only after `modify_code` routing works. The first approvals to add should be high-cost full rewrites or technique switches, not the core repair-mode decision itself.

6. `parallel_tool_calls`: low priority for this crisis. The bottleneck is decision authority, not lack of parallelism.

## 6. WHAT TO STOP DOING

1. Stop letting direct parameter modifications run before the system has authoritatively decided whether the issue is structural or parametric.

2. Stop treating `iterate` as if it were synonymous with “tune parameters.”

3. Stop relying on exact-string `primary_issue` stability as the main way to detect repeated structural failure.

4. Stop overwriting canonical stuck-state with coordinator escape-level output before the persistent state has been updated from the iteration result.

5. Stop treating a higher `max_iterations` setting as a fix for missing structural repair routing.

6. Stop assuming the Modification Coordinator is the decision-maker if earlier deterministic param paths can preempt it entirely.

7. Stop making HITL and baseline measurement the next highest priorities until the loop can actually choose `modify_code` when the evidence demands it.

## 7. OPEN QUESTIONS

1. Should structural-vs-parameter classification live in the Quality Analyst output, the QA diagnosis bridge, or a deterministic `RepairIntent` function that consumes both? My view: make the final classification deterministic, but feed it typed QA signals.

2. Is one `modify_code` attempt the right default for structural issues before falling back to `modify_params`, or should the system require a confidence threshold and section-target hints first?

3. Should full rewrites be allowed directly from `RepairIntent`, or should the sequence always be `modify_code -> patch_script_section -> full rewrite only if patching fails`?

4. How much of the Learning Agent’s hot-path role should remain after `RepairIntent` exists? It may be better as a distiller plus param adviser than as a generic next-action recommender.

5. Which first-render correctness checks belong in Wave 3 versus the immediate repair-routing fix? My view: stripe order, camera outside geometry, and “no visible deformation” can all be folded into the same structural issue taxonomy instead of becoming a separate ad hoc validator.
