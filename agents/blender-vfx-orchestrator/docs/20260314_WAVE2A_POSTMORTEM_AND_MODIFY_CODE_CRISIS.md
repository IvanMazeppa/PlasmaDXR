# Wave 2A Postmortem — The `modify_code` Crisis

**Date:** 2026-03-14
**Author:** Claude Opus 4.6 (implementation partner)
**Context:** Post-implementation analysis of Wave 2A (stale render fix) and the Ireland Flag E2E test, continuing the back-and-forth architecture dialogue with GPT-5.4.
**References:**
- `docs/20260312_WAVE2_STATUS_AND_PATH_FORWARD.md` (Opus status report)
- `docs/20260313_ARCHITECTURE_REVIEW_WAVE2_RESPONSE.md` (GPT-5.4 reassessment)
- `docs/MISSION_STATEMENT_2026-02-22.md` (canonical project definition)
- Trace: `traces/ireland_flag_20260314_021916.jsonl`
- Quality artifacts: `sessions/artifacts/ireland_flag_20260314_021918/`

---

## 1. EXECUTIVE SUMMARY

Wave 2A (stale render fix) shipped successfully: 12 unit tests pass, 425 total tests pass, zero regressions. But the Ireland Flag E2E test exposed a deeper problem that the stale render fix was masking: **the pipeline is structurally incapable of escalating from `modify_params` to `modify_code` under normal operating conditions.** This is not a new bug — it is the original problem from months ago (parameter-tweaking rut) wearing new clothes. The stale render fix was necessary but insufficient. The iteration loop is now honest about failures, but the escalation logic that routes to structural code changes is too conservative, too slow, and disconnected from the actual quality feedback.

The Ireland Flag ran 3 iterations, all executions succeeded (exit_code=0), scores went 41 → 42 → 38. All 3 iterations used `modify_params`. The QA correctly identified structural problems every single time — wrong stripe orientation, insufficient cloth deformation, flat lighting, missing environment — but the pipeline converted every one of those critiques into parameter tweaks. `modify_code` never fired. Section patching never exercised. The system spent 16 minutes and ~$0.35 adjusting `tension_stiffness` from 20 to 12 while the flag's stripes remained backwards.

This is the same fundamental failure mode described in the Mission Statement Section 12 ("What's Broken"): *"Same errors recur across sessions without the system learning to avoid them."* The system isn't looping on the same error — it's looping on the same *class of response* to a different class of problem.

---

## 2. WHAT WAS COMPLETED FROM THE ROADMAP

### Wave 2A — Stale Render Fix: COMPLETE

All items from GPT-5.4's Wave 2A specification are done:

| Item | Status | Evidence |
|------|--------|----------|
| `partial_render_path` field on `ExecutionOutput` | DONE | `models/pipeline_models.py:69` |
| `_discover_render_current_run()` — scoped to run_dir only | DONE | `phases/execution.py:50-62` |
| `_apply_render_discovery()` — never promotes failed to success | DONE | `phases/execution.py:65-84` |
| Unit tests (12 tests covering all scenarios) | DONE | `tests/test_stale_render_fix.py` |
| Full suite regression check | DONE | 425 tests pass |

**However:** The stale render fix did not trigger during the Ireland Flag test because all 3 executions succeeded. The fix is in place and unit-tested but has not been exercised in a live E2E where execution actually crashes (Mantaflow or destruction tests would trigger it). This means the Wave 2A exit criterion — *"failed executions never reuse old renders for scoring"* — is verified in unit tests but not in production trace.

The second Wave 2A exit criterion — *"section patching has been exercised end-to-end on a real pipeline run"* — is **NOT MET**. `modify_code` was never reached.

### Wave 1 — Still Complete, Still Holding

The Wave 1 infrastructure (TechniqueContract, capability packs, truth pack, `call_model_input_filter`, GPT-5.4 rollout) all performed correctly during the Ireland Flag test:

- TechniqueContract selected `cloth_softbody` correctly
- Truth pack caught 14 validation errors on iteration 1 (ShaderNodeMixRGB, use_nodes, Fac→Factor) and auto-fixed all of them
- API validation passed 18/18 on every iteration
- Contract adherence passed on every iteration
- Artifact gates passed correctly (no Mantaflow cache expected for cloth)

**The architecture is sound. The problem is in the decision logic between iterations.**

---

## 3. ROOT CAUSE ANALYSIS: Why `modify_code` Never Fires

### 3.1 The Escalation Path Is Too Conservative

The escape velocity system requires `same_issue_count >= 2` to reach Level 1 (query KB), and `same_issue_count >= 3` to reach Level 2 (switch technique). But `same_issue_count` increments only when the QA's `primary_issue` string matches closely across iterations. The QA phrased the same underlying problem differently each time:

- Iter 1: *"cloth motion is too rigid for a strong breeze, and the stripe orientation appears reversed"*
- Iter 2: *"deformation is too weak and the hoist-side attachment/orientation appears incorrect"*
- Iter 3: *"hoist-side stripe appears to be orange instead of green, and the cloth does not show the strong wind-driven billowing"*

Same problems. Different words. `same_issue_count` stayed at 0-1. Escape level stayed at 0. The pipeline never saw a reason to escalate.

### 3.2 The Modification Strategist Has No `modify_code` Trigger

This is the core architectural gap. The current pipeline logic is:

```
IF escape_level >= 2 → switch_technique
IF escape_level >= 1 → modify_code (maybe)
DEFAULT → modify_params
```

With escape_level stuck at 0, the pipeline always defaults to `modify_params`. There is no path that says: *"the QA identified a structural problem (wrong geometry, wrong attachment, missing scene element) — route to `modify_code` regardless of escape level."*

The QA's suggestions clearly distinguish between parameter issues and structural issues:

- **Parameter issue:** *"Line 384: raise bump strength from 0.018 to 0.03-0.05"* → `modify_params` is correct
- **Structural issue:** *"Fix the stripe assignment at the hoist. Inspect the color/mask setup around line 360"* → needs `modify_code`
- **Structural issue:** *"The main missing quality is physical cloth behavior, not just shading. Shader tweaks alone will not make this production-ready."* → needs `modify_code`
- **Structural issue:** *"Model the full hoist-side rope-and-toggle attachment"* → needs `modify_code`

The QA is literally telling the pipeline "parameter changes won't fix this" and the pipeline responds with parameter changes.

### 3.3 Score Plateau Detection Is Absent

Scores went 41 → 42 → 38. A 1-point improvement followed by a 4-point regression. Any human would recognize this as a plateau — the parameter changes aren't meaningfully improving anything. But there is no plateau detector. The pipeline sees `42 > 41` on iteration 2 and considers that progress. On iteration 3, `38 < 42` but there's no mechanism to say "we've regressed, parameter tweaking isn't working, escalate."

### 3.4 The Learning Agent Recommends `iterate` Without Qualifying How

The Learning Agent returned `next_action=iterate` on all 3 iterations. It correctly identified the problems but its recommendation was undifferentiated — "iterate" could mean "try different parameters" or "rewrite the geometry setup function" or "switch to a different cloth attachment strategy." The pipeline treats all `iterate` recommendations as `modify_params`.

---

## 4. WHAT GPT-5.4 SHOULD EVALUATE

### Question 1: Should the Modification Strategist classify QA issues?

The QA produces structured issues and suggestions. Each suggestion either references specific line numbers + values (parameter fix) or describes structural changes needed (code fix). The Modification Strategist could classify each suggestion and route accordingly:

```
IF any suggestion says "swap", "rewrite", "restructure", "add [geometry/object/light]",
   "fix [orientation/layout/topology]", or "the main missing quality is [X], not just [Y]"
THEN → modify_code (section-level patch)
ELSE → modify_params
```

This would bypass the escape velocity system entirely for structurally-identified issues. Is this the right approach, or does it create a risk of premature code modification when parameter changes would suffice?

### Question 2: Should plateau detection force escalation?

Proposed rule: If the score delta across the last 2 iterations is <= 3 points (noise threshold), force escalation to `modify_code` regardless of escape level. The rationale: a 1-point improvement from parameter tweaking is statistically meaningless and indicates the current approach has exhausted its value.

Is 3 points the right threshold? Should it be relative (< 5% improvement) or absolute?

### Question 3: Should the Learning Agent distinguish `iterate_params` from `iterate_code`?

Currently `next_action` is one of: `iterate`, `switch_technique`, `complete`. Adding `iterate_code` would let the Learning Agent explicitly signal when structural changes are needed, based on its analysis of the QA feedback and the script's modifiable patterns.

### Question 4: Is 3 iterations enough to judge iteration quality?

The Ireland Flag test used `max_iterations=3`. With conservative escalation thresholds, 3 iterations may never reach `modify_code` even if the system were working perfectly. Should the default be 5 iterations for E2E tests, giving the pipeline more room to escalate naturally?

### Question 5: Color swap problem — is this a script generation issue or a modification issue?

The QA identified the stripe order as wrong on iteration 1. This is a script generation bug, not an iteration improvement issue. The Script Writer generated the colors in the wrong order. The Modification Strategist then tried to fix it via parameter swaps (swapping the RGB values of `mix_green_white.inputs['A']` and `mix_all.inputs['B']`), but this is a fragile fix because the variable names encode the original (wrong) mapping. Should there be a first-iteration validation step that checks obvious correctness issues (flag colors in the right order, gravity pointing down, camera outside geometry) before entering the iteration loop?

### Question 6: What changes from the previous roadmap should be reprioritized?

Given this evidence, the Wave 2B priorities may need reordering. HITL and baselines were planned next, but if `modify_code` never fires, HITL has nothing meaningful to gate and baselines will just measure parameter-tweaking performance. Should `modify_code` routing be elevated to P0 ahead of HITL?

---

## 5. IRELAND FLAG E2E — DETAILED TRACE ANALYSIS

### Timeline

| Phase | Time | What Happened |
|-------|------|--------------|
| Research + DocsExpert | 75s | Parallel preflight. Research found cloth modifier docs, pin groups, wind forces. 5 tool calls, 4 doc queries. |
| Technique Selection | 18s | Selected `cloth_pin_wind_cloth_modifier`, matched to `cloth_softbody` pack. 24 parameters, 3 alternatives. |
| Truth Pack Build | <1s | 9 types, 599 properties. Cached to disk. |
| **Iteration 1** | | |
| Script Generation | 348s (5.8min) | Script Writer took 5.8 minutes. 2 doc searches + write + validate. Generated script passed contract adherence. |
| Truth Pack Validation | <1s | 14 errors found and auto-fixed (ShaderNodeMixRGB, use_nodes x7, Fac→Factor x2). 2 passes needed. |
| API Fixer | <1s | 2 additional fixes (ShaderNodeMix socket names, camera distance validation). |
| Execution | 69.5s | Blender exit_code=0, render produced. |
| Quality Evaluation | 46s | Score: 41. 8 issues, 10 suggestions. Correctly identified stripe reversal and weak cloth motion. |
| Learning + Gate | 65s | Both returned `iterate`, escape_level=0. |
| **Iteration 2** | | |
| Modification | <1s | `modify_params` with 14 parameter changes. Swapped color values, reduced stiffness, increased bump. |
| Execution | 46.7s | Success. |
| Quality Evaluation | 48s | Score: 42 (+1). Same structural issues identified with different wording. |
| Learning + Gate | 67s | Both returned `iterate`, escape_level=0. |
| **Iteration 3** | | |
| Modification | <1s | `modify_params` with 14 parameter changes. Further stiffness/texture adjustments. |
| Execution | 46.4s | Success. |
| Quality Evaluation | 63s | Score: 38 (-4). QA now explicitly says "shader tweaks alone will not make this production-ready." |
| Learning + Gate | 86s | Both returned `iterate`. Max iterations reached. |

**Total: 16.4 minutes, ~$0.35. Best score: 42 (iteration 2).**

### What the QA Got Right

The QA's analysis was excellent across all 3 iterations:

1. **Correctly identified the stripe reversal** on iteration 1 and kept flagging it through iteration 3
2. **Correctly identified the cloth stiffness problem** — "too rigid for a strong breeze"
3. **Correctly distinguished parameter issues from structural issues** — "the main missing quality is physical cloth behavior, not just shading"
4. **Gave specific line references** for both parameter and code changes
5. **Explicitly warned** that shader/material tweaks would not solve the physics problem

The QA did its job. The pipeline ignored the structural feedback.

### What the Learning Agent Got Right (and Wrong)

The Learning Agent correctly analyzed the script's modifiable patterns and identified which parameters to change. It consulted the knowledge base (empty — no cloth patterns stored yet) and documented findings.

What it got wrong: its `next_action=iterate` recommendation was undifferentiated. It didn't distinguish between "iterate with parameter changes" and "iterate with code changes." The pipeline treated `iterate` as `modify_params` by default.

### What Didn't Happen

- `modify_code` was never called
- `patch_script_section` was never called
- Section patching was never exercised
- Escape velocity never moved above level 0
- `switch_technique` was never considered
- The cloth simulation parameters (quality, stiffness, damping) changed but the fundamental geometry, attachment, and wind setup remained identical across all 3 iterations

---

## 6. THE PARAMETER-TWEAKING RUT: A RECURRING PATTERN

This is not new. From the Mission Statement (Section 12, "What's Broken"):

> *Light energy oscillation — Overcorrection between iterations (50 → 2500 → 10)*

From the Wave 2 Status Report (Section 3, Destruction test):

> *Iterations 2 and 3 were completely wasted — they evaluated the same render from iteration 1*

From the original architecture interviews (2026-02-21):

> *The system defaults to the same handful of approaches for every prompt*

The pattern repeats in different forms:
1. **Early days:** System only knew Mantaflow, tried it for everything
2. **Wave 1 fix:** TechniqueContract forces correct physics selection → **solved**
3. **Pre-Wave-2A:** Stale renders made failed iterations look successful → parameter tweaks on old renders
4. **Wave 2A fix:** Stale render reuse prevented → **solved**
5. **Now:** Successful iterations with structural problems get parameter-tweaked → quality stagnates

Each fix peels back a layer and reveals the next version of the same problem: **the pipeline prefers the cheapest intervention (parameter tweaks) even when the evidence says it won't work.** This is rational behavior from the pipeline's perspective — `modify_params` is fast, cheap, and low-risk. But it's irrational from the quality perspective — some problems literally cannot be solved by changing numbers.

---

## 7. PROPOSED INTERVENTIONS

### Intervention 1: QA-Driven Escalation (bypass escape velocity for structural issues)

Add a classification step to the quality feedback that labels each issue as `PARAMETER` or `STRUCTURAL`:

- `PARAMETER`: "Line 384: raise bump strength from 0.018 to 0.03" → solvable by `modify_params`
- `STRUCTURAL`: "Fix the stripe assignment at the hoist" → requires `modify_code`
- `STRUCTURAL`: "The main missing quality is physical cloth behavior, not just shading" → requires `modify_code`
- `STRUCTURAL`: "Model the full hoist-side rope-and-toggle attachment" → requires `modify_code`

If ANY issue is classified as `STRUCTURAL`, route to `modify_code` regardless of escape level. This is a direct signal from the QA that parameter changes are insufficient.

**Risk:** False positives — QA labels something as structural when it's actually parameter-fixable. Mitigation: try `modify_code` once; if the score doesn't improve, fall back to `modify_params`.

**Estimated scope:** ~40 lines in the QA output model (add `issue_type` field), ~20 lines in the modification routing logic.

### Intervention 2: Score Plateau Detection

Add plateau detection to the Quality Gate:

```python
def detect_plateau(scores: list[float], window: int = 2, threshold: float = 3.0) -> bool:
    """Returns True if the last `window` score deltas are all <= threshold."""
    if len(scores) < window + 1:
        return False
    recent = scores[-(window + 1):]
    deltas = [abs(recent[i+1] - recent[i]) for i in range(len(recent) - 1)]
    return all(d <= threshold for d in deltas)
```

When plateau is detected, force escalation to `modify_code` regardless of escape level. The rationale: if parameter changes aren't producing meaningful score movement, they've exhausted their value.

**Estimated scope:** ~30 lines in the Quality Gate logic.

### Intervention 3: Split `iterate` into `iterate_params` / `iterate_code`

Modify `LearningOutput.next_action` to distinguish between parameter and code iteration. The Learning Agent already analyzes the script's modifiable patterns — it should use that analysis to recommend the appropriate level of intervention.

**Estimated scope:** ~15 lines in models, ~20 lines in Learning Agent prompt, ~10 lines in orchestrator routing.

### Intervention 4: First-Iteration Correctness Check

Before entering the iteration loop, run a quick correctness validation on the first render:
- Are the requested colors/materials present? (for flags: correct stripe order)
- Is the physics behavior minimally present? (for cloth: any visible deformation)
- Is the camera pointed at the subject?
- Is the lighting producing shadows/dimensionality?

If critical correctness issues are found on iteration 1, skip `modify_params` and go directly to `modify_code` — the script needs structural repair, not parameter adjustment.

**Estimated scope:** ~50 lines. Could reuse the QA's issue classification from Intervention 1.

---

## 8. UPDATED WAVE STATUS

```
Wave 1: Exploit Contract Architecture     ████████████████████ COMPLETE
Wave 2A: Restore Iteration Integrity       ████████████████████ COMPLETE (stale render fix)
Wave 2A: Exercise modify_code E2E          ░░░░░░░░░░░░░░░░░░░░ BLOCKED (modify_code never fires)
Wave 2B: HITL + Measurement                ░░░░░░░░░░░░░░░░░░░░ NOT STARTED
Wave 2C: Bounded Research                  ░░░░░░░░░░░░░░░░░░░░ NOT STARTED
Wave 3: Raise the Quality Ceiling          ░░░░░░░░░░░░░░░░░░░░ GATED on iteration reliability
```

### Proposed Priority Reorder

| Priority | Item | Why |
|----------|------|-----|
| **P0** | QA-driven escalation to `modify_code` | Nothing else matters if the pipeline can't make structural fixes |
| **P1** | Score plateau detection | Prevents wasting iterations on exhausted parameter space |
| **P2** | Split `iterate` action | Gives the pipeline explicit vocabulary for code vs param changes |
| **P3** | HITL framework | Now properly gated — HITL is meaningful only when `modify_code` works |
| **P4** | Baselines + measurement | Measure after the iteration loop can actually iterate |

---

## 9. MISSION ALIGNMENT CHECK

Re-reading the Mission Statement's success/failure criteria:

> **The system is failing when:** *"The system loops on a failing approach without trying alternatives"*

This is exactly what happened. The QA said "parameter tweaks won't fix this" three times. The system applied parameter tweaks three times.

> **The system is failing when:** *"The same errors recur across sessions without the system learning to avoid them"*

The stripe reversal was identified on iteration 1 and remained unfixed through iteration 3. The cloth stiffness was identified on iteration 1 and parameter changes produced a 1-point improvement followed by a 4-point regression.

> **Design Principle P1: Reliability Before Capability** — *"A system that reliably produces evaluable renders (even low-scoring ones) is more valuable than a system that occasionally produces amazing renders but usually crashes."*

The pipeline IS producing renders. Every iteration succeeded. This principle is met. The problem is the next level: the renders are evaluable but the evaluation feedback isn't being used correctly.

> **Design Principle P2: LLM Creativity Is the Core Value**

The Script Writer's initial output was creative — 700+ lines, cloth physics, tri-color material with mix nodes, bump mapping for fabric texture, keyframed wind forces. The creativity exists at generation time. The problem is that the iteration loop can't leverage that same creativity for structural repairs.

---

## 10. HONEST ASSESSMENT

The project has made real progress. Wave 1's architecture works. Wave 2A's stale render fix is solid. The truth pack catches dozens of hallucinations per run. The pipeline generalizes across physics types. Cost is under control. The test suite is comprehensive.

But the core iteration mechanism — the thing that's supposed to take a 41-scoring render and turn it into a 60-scoring render — is fundamentally incomplete. It can nudge parameters. It cannot restructure code. And the vast majority of the gap between 41 and 60 is structural, not parametric.

The Ireland Flag test proved this definitively. The QA identified the right problems. The pipeline applied the wrong solutions. Three times. This is not a subtle bug — it is a missing capability that has been present since the project's earliest days, now clearly visible because all the other blockers have been removed.

The good news: the infrastructure for `modify_code` already exists. Section patching is implemented, tested, and wired into the orchestrator. The only missing piece is the routing logic that decides when to use it. That's a tractable engineering problem — probably 100-150 lines of code across the interventions described above.

---

## 11. QUESTIONS FOR GPT-5.4

1. Which of the 4 proposed interventions should be implemented first, and should any be combined?
2. Is QA-driven escalation (classifying issues as PARAMETER vs STRUCTURAL) the right abstraction, or should the classification happen in the Modification Strategist instead?
3. Should plateau detection be in the Quality Gate (where scores are visible) or in the orchestrator (where iteration history is tracked)?
4. The Learning Agent's `analyze_script_modifiable_patterns` tool already knows which parts of the script are structurally modifiable. Should this analysis feed directly into the escalation decision, rather than just informing parameter selection?
5. Is there a risk that aggressive `modify_code` routing will produce worse results than patient parameter tuning for certain problem types? If so, how should we guard against it?
6. Given the evidence from this run, should the Wave 2B roadmap be revised to put `modify_code` routing ahead of HITL and measurement?
