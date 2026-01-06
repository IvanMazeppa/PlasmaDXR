# VFX Pipeline AI Operator Manual

**Version:** 1.0
**Date:** 2026-01-06
**Audience:** AI Orchestrators (Claude Code, GPT-based agents)
**Purpose:** Systematic asset creation and quality improvement

---

## System Overview

You are operating a distributed VFX asset creation pipeline consisting of 7 specialized MCP servers. Your role is to orchestrate these tools to create high-quality Blender volumetric assets (VDB files, renders) through intelligent iteration.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    VFX PIPELINE ARCHITECTURE                         │
│                                                                      │
│   YOU (AI Orchestrator)                                             │
│     │                                                                │
│     ├──► blender-librarian ──► Documentation + Vision diagnosis     │
│     │                                                                │
│     ├──► script-generator ──► Blender Python script creation        │
│     │                                                                │
│     ├──► blender-executor ──► CLI execution + output capture        │
│     │                                                                │
│     ├──► asset-evaluator ──► Quality metrics (LPIPS, SigLIP, VLM)   │
│     │                                                                │
│     ├──► experiment-tracker ──► Learning + knowledge accumulation   │
│     │                                                                │
│     ├──► iteration-controller ──► State management + suggestions    │
│     │                                                                │
│     └──► blender-manual ──► Official Blender 5.0 documentation      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Workflow

### Phase 1: Session Initialization

**Goal:** Establish context, load relevant knowledge, set quality targets.

```
STEP 1.1: Start experiment session
┌─────────────────────────────────────────────────────────────────────┐
│ Tool: mcp__experiment-tracker__start_experiment_session             │
│                                                                      │
│ Parameters:                                                          │
│   asset_name: "solar_prominence_v1"                                 │
│   effect_type: "sun"                                                │
│   description: "Sun with visible prominences and limb darkening"   │
│   reference_path: "assets/reference_images/star/frame_00639.jpg"   │
│   semantic_query: "realistic sun with solar flares"                 │
│                                                                      │
│ Returns: session_id for tracking                                    │
└─────────────────────────────────────────────────────────────────────┘

STEP 1.2: Query existing knowledge
┌─────────────────────────────────────────────────────────────────────┐
│ Tool: mcp__experiment-tracker__query_knowledge_base                 │
│                                                                      │
│ Parameters:                                                          │
│   query: "sun prominences limb darkening"                           │
│                                                                      │
│ Purpose: Load learnings from past sun-related experiments           │
│ Action: Incorporate relevant learnings into your approach           │
└─────────────────────────────────────────────────────────────────────┘

STEP 1.3: Check for technique recommendations
┌─────────────────────────────────────────────────────────────────────┐
│ Tool: mcp__script-generator__recommend_technique                    │
│                                                                      │
│ Parameters:                                                          │
│   effect_type: "pyro"                                               │
│   description: "sun with prominences"                               │
│                                                                      │
│ Purpose: UCB1 algorithm balances exploration vs exploitation        │
│ Returns: technique_name, confidence, selection_reason               │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Phase 2: Script Generation

**Goal:** Create a Blender Python script that will produce the desired effect.

```
STEP 2.1: Generate initial script
┌─────────────────────────────────────────────────────────────────────┐
│ Tool: mcp__script-generator__generate_script                        │
│                                                                      │
│ Parameters:                                                          │
│   effect_type: "pyro"                                               │
│   description: "A sun with visible prominences extending from the   │
│                 surface, showing limb darkening at the edges,       │
│                 temperature around 5778K with emission glow"        │
│   output_name: "solar_prominence_v1"                                │
│   resolution: 96  (start low for fast iteration)                    │
│   frame_end: 50                                                     │
│   technique_name: <from step 1.3>                                   │
│                                                                      │
│ Returns: script_path, script_content                                │
└─────────────────────────────────────────────────────────────────────┘

STEP 2.2: Validate script before execution
┌─────────────────────────────────────────────────────────────────────┐
│ Tool: mcp__script-generator__validate_script                        │
│                                                                      │
│ Parameters:                                                          │
│   script_path: <from step 2.1>                                      │
│   strict: false                                                     │
│                                                                      │
│ Purpose: Catch syntax errors, invalid parameters before Blender     │
│ Action: If validation fails, fix issues before proceeding          │
└─────────────────────────────────────────────────────────────────────┘
```

**IMPORTANT:** If you need documentation while generating scripts:
```
┌─────────────────────────────────────────────────────────────────────┐
│ Tool: mcp__blender-manual__search_python_api                        │
│   operation: "FluidDomainSettings"                                  │
│                                                                      │
│ Tool: mcp__blender-manual__search_vdb_workflow                      │
│   query: "export openvdb mantaflow"                                 │
│                                                                      │
│ Tool: mcp__blender-manual__search_bpy_types                         │
│   typename: "FluidFlowSettings"                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Phase 3: Execution

**Goal:** Run the Blender script, capture outputs, handle errors.

```
STEP 3.1: Execute script
┌─────────────────────────────────────────────────────────────────────┐
│ Tool: mcp__blender-executor__execute_blender_script                 │
│                                                                      │
│ Parameters:                                                          │
│   script_path: <from step 2.1>                                      │
│   output_dir: "build/vdb_output/solar_prominence_v1"                │
│   timeout_seconds: 600  (10 minutes for baking)                     │
│                                                                      │
│ Returns: success, stdout, stderr, vdb_files, render_files           │
└─────────────────────────────────────────────────────────────────────┘

STEP 3.2: If execution failed, parse errors
┌─────────────────────────────────────────────────────────────────────┐
│ Tool: mcp__blender-executor__parse_blender_errors                   │
│                                                                      │
│ Parameters:                                                          │
│   stderr: <from step 3.1>                                           │
│   stdout: <from step 3.1>                                           │
│                                                                      │
│ Returns: Structured errors with suggested_fix                       │
│ Action: Fix script and retry (goto step 2.1 with modifications)    │
└─────────────────────────────────────────────────────────────────────┘

STEP 3.3: List outputs
┌─────────────────────────────────────────────────────────────────────┐
│ Tool: mcp__blender-executor__list_run_outputs                       │
│                                                                      │
│ Parameters:                                                          │
│   run_dir: <output_dir from step 3.1>                               │
│                                                                      │
│ Returns: vdb_files, render_files with paths                         │
│ Action: Select representative render for evaluation (e.g., frame 30)│
└─────────────────────────────────────────────────────────────────────┘
```

---

### Phase 4: Quality Evaluation

**Goal:** Assess render quality using multiple metrics, identify issues.

```
STEP 4.1: Primary evaluation (RECOMMENDED)
┌─────────────────────────────────────────────────────────────────────┐
│ Tool: mcp__asset-evaluator__evaluate_render_v2                      │
│                                                                      │
│ Parameters:                                                          │
│   render_path: "build/vdb_output/solar_prominence_v1/render_0030.png"
│   reference_path: "assets/reference_images/star/frame_00639.jpg"   │
│   effect_type: "sun"                                                │
│   profile: "standard"  (or "quick" for fast iteration)              │
│   include_diagnostics: true                                         │
│   include_suggestions: true                                         │
│                                                                      │
│ Returns:                                                             │
│   - overall_score: 0-100                                            │
│   - passed: true if score >= 60                                     │
│   - metric_scores: {lpips, siglip, topiq, structural_dino, etc.}   │
│   - diagnostics: VLM-identified issues                              │
│   - suggestions: Parameter changes to try                           │
└─────────────────────────────────────────────────────────────────────┘

EVALUATION PROFILES:
  - quick: LPIPS + SigLIP only (~2 seconds)
  - standard: + TOPIQ, feature_cv (~10 seconds) ← DEFAULT
  - comprehensive: + DINOv2, VLM diagnosis (~30 seconds)
```

**DECISION POINT:**
```
IF overall_score >= 70:
    → Asset is GOOD. Proceed to Phase 6 (Finalization).

IF overall_score >= 50 AND iteration < max_iterations:
    → Asset is ACCEPTABLE but improvable. Proceed to Phase 5 (Iteration).

IF overall_score < 50:
    → Asset has SIGNIFICANT ISSUES. Diagnose deeper before iterating.
```

---

### Phase 4b: Deep Diagnosis (When Needed)

**Use when:** Score < 50, or issues are unclear, or iteration isn't improving.

```
STEP 4b.1: VLM-powered issue diagnosis
┌─────────────────────────────────────────────────────────────────────┐
│ Tool: mcp__asset-evaluator__diagnose_issues_v2                      │
│                                                                      │
│ Parameters:                                                          │
│   render_path: <render to diagnose>                                 │
│   reference_path: <reference image>                                 │
│   effect_type: "sun"                                                │
│   known_issues: "too dark, wrong color"  (hints from evaluation)   │
│                                                                      │
│ Returns:                                                             │
│   - issues: List with severity ratings                              │
│   - primary_issue: Most critical to fix                             │
│   - overall_assessment: Summary                                     │
└─────────────────────────────────────────────────────────────────────┘

STEP 4b.2: Get librarian diagnosis (uses GPT-5.2 vision)
┌─────────────────────────────────────────────────────────────────────┐
│ Tool: mcp__blender-librarian__diagnose_render_issue                 │
│                                                                      │
│ Parameters:                                                          │
│   render_path: <render to diagnose>                                 │
│   reference_path: <reference image>                                 │
│   effect_type: "sun"                                                │
│   current_issues: <JSON array from step 4.1>                        │
│   current_score: <score from step 4.1>                              │
│                                                                      │
│ Returns: diagnosis, primary_issue, severity                         │
│ Cost: ~$0.05-0.10 (vision API call)                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Phase 5: Iteration

**Goal:** Improve the asset based on evaluation feedback.

```
STEP 5.1: Get modification advice
┌─────────────────────────────────────────────────────────────────────┐
│ Tool: mcp__blender-librarian__get_modification_advice               │
│                                                                      │
│ Parameters:                                                          │
│   effect_type: "sun"                                                │
│   issues: <JSON array of issues from evaluation>                    │
│   current_params: <JSON of current script parameters>               │
│   evaluator_scores: <JSON of metric scores>                         │
│                                                                      │
│ Returns:                                                             │
│   - modifications: Dict of parameter changes                        │
│   - rationale: Why these changes should help                        │
│   - confidence: 0-1                                                 │
│   - source: "playbook" (free) or "gpt" (paid)                      │
└─────────────────────────────────────────────────────────────────────┘

STEP 5.2: Check warnings before applying changes
┌─────────────────────────────────────────────────────────────────────┐
│ Tool: mcp__experiment-tracker__get_warnings_before_change           │
│                                                                      │
│ Parameters:                                                          │
│   parameter: "flame_max_temp"                                       │
│   change_type: "increase"                                           │
│                                                                      │
│ Returns: Warnings from past experiments                             │
│ Purpose: Avoid known pitfalls                                       │
└─────────────────────────────────────────────────────────────────────┘

STEP 5.3: Modify script with improvements
┌─────────────────────────────────────────────────────────────────────┐
│ Tool: mcp__script-generator__modify_script                          │
│                                                                      │
│ Parameters:                                                          │
│   script_path: <current script>                                     │
│   modifications: {                                                  │
│     "flame_max_temp": 6000,                                         │
│     "emission_intensity": 2.5,                                      │
│     "turbulence": 0.15                                              │
│   }                                                                 │
│   output_name: "solar_prominence_v1_iter2"                          │
│                                                                      │
│ Returns: Modified script path                                       │
└─────────────────────────────────────────────────────────────────────┘

STEP 5.4: Record baseline before iteration
┌─────────────────────────────────────────────────────────────────────┐
│ Tool: mcp__experiment-tracker__record_baseline                      │
│                                                                      │
│ Parameters:                                                          │
│   params: <JSON of current parameters>                              │
│   scores: <JSON of current scores>                                  │
│   render_path: <current best render>                                │
│   script_path: <current script>                                     │
└─────────────────────────────────────────────────────────────────────┘

→ Now return to Phase 3 (Execution) with the modified script
→ Then Phase 4 (Evaluation)
→ Compare new score to previous
```

---

### Phase 5b: Compare Iterations

**Use after each iteration to track progress.**

```
STEP 5b.1: Compare renders
┌─────────────────────────────────────────────────────────────────────┐
│ Tool: mcp__asset-evaluator__compare_renders_v2                      │
│                                                                      │
│ Parameters:                                                          │
│   render_a: <previous iteration render>                             │
│   render_b: <new iteration render>                                  │
│   reference_path: <reference image>                                 │
│   comparison_type: "iteration"                                      │
│                                                                      │
│ Returns:                                                             │
│   - winner: "A" or "B"                                              │
│   - score_a, score_b: Individual scores                             │
│   - improvements: What got better                                   │
│   - regressions: What got worse                                     │
│   - recommendation: What to try next                                │
└─────────────────────────────────────────────────────────────────────┘

STEP 5b.2: Record experiment result
┌─────────────────────────────────────────────────────────────────────┐
│ Tool: mcp__experiment-tracker__record_experiment_result             │
│                                                                      │
│ Parameters:                                                          │
│   hypothesis: "Increasing flame_max_temp will improve brightness"   │
│   issue_addressed: "too_dark"                                       │
│   result_params: <JSON of new parameters>                           │
│   result_scores: <JSON of new scores>                               │
│   result_render: <new render path>                                  │
│   result_script: <new script path>                                  │
│   success: true/false (did score improve?)                          │
│   observed_effects: ["brightness increased", "color shifted warm"]  │
│   learnings: ["flame_max_temp > 5500 improves sun brightness"]      │
│   warnings: ["flame_max_temp > 7000 causes color clipping"]         │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Phase 6: Finalization

**Goal:** Record success, save learnings, clean up.

```
STEP 6.1: Record successful fix (if using librarian)
┌─────────────────────────────────────────────────────────────────────┐
│ Tool: mcp__blender-librarian__record_fix_outcome                    │
│                                                                      │
│ Parameters:                                                          │
│   session_id: <from librarian session>                              │
│   issue: "too_dark"                                                 │
│   modifications: <JSON of what worked>                              │
│   success: true                                                     │
│   score_improvement: 15.5  (new_score - old_score)                 │
│   reason: "Increased flame_max_temp and emission solved darkness"  │
└─────────────────────────────────────────────────────────────────────┘

STEP 6.2: Add to playbook (for future FREE lookups)
┌─────────────────────────────────────────────────────────────────────┐
│ Tool: mcp__blender-librarian__add_to_playbook                       │
│                                                                      │
│ Parameters:                                                          │
│   effect_type: "sun"                                                │
│   symptom: "too_dark"                                               │
│   fix: <JSON of modifications that worked>                          │
│   confidence: 0.85                                                  │
└─────────────────────────────────────────────────────────────────────┘

STEP 6.3: Record technique outcome (for UCB1 learning)
┌─────────────────────────────────────────────────────────────────────┐
│ Tool: mcp__script-generator__record_technique_outcome               │
│                                                                      │
│ Parameters:                                                          │
│   technique_name: <technique used>                                  │
│   effect_type: "sun"                                                │
│   success: true                                                     │
│   final_score: 78.5                                                 │
│   iterations: 3                                                     │
└─────────────────────────────────────────────────────────────────────┘

STEP 6.4: End experiment session
┌─────────────────────────────────────────────────────────────────────┐
│ Tool: mcp__experiment-tracker__end_experiment_session               │
│                                                                      │
│ Parameters:                                                          │
│   final_status: "completed"  (or "abandoned", "max_iterations")    │
│   best_score: 78.5                                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Decision Trees

### When to Use Each Evaluation Profile

```
START
  │
  ▼
Is this early iteration (< 3)?
  │
  ├─YES─► Use profile="quick" (2 seconds)
  │       Just need directional feedback
  │
  └─NO──► Is score improving consistently?
            │
            ├─YES─► Use profile="standard" (10 seconds)
            │       Default for most evaluations
            │
            └─NO──► Use profile="comprehensive" (30 seconds)
                    Need deep diagnosis of what's wrong
```

### When to Use Librarian vs Direct Evaluation

```
START
  │
  ▼
Is issue category known? (too_dark, wrong_color, etc.)
  │
  ├─YES─► Check playbook first (FREE)
  │       │
  │       └─► mcp__blender-librarian__get_modification_advice
  │           (Returns playbook fix if available)
  │
  └─NO──► Need diagnosis
          │
          ├─► mcp__asset-evaluator__diagnose_issues_v2 (VLM, cheaper)
          │
          └─► If still unclear:
              mcp__blender-librarian__diagnose_render_issue (GPT-5.2 vision)
```

### When to Stop Iterating

```
STOP if ANY of these are true:

1. Score >= quality_threshold (success!)

2. Iterations >= max_iterations (give up, return best)

3. Score decreased for 2 consecutive iterations (local optimum)
   → Consider: Different technique, not parameter tweaks

4. Score plateau (< 2 point change for 3 iterations)
   → Consider: Reference image may be wrong target

5. Budget exhausted
   → mcp__blender-librarian__get_budget_status returns remaining < $0.50
```

---

## Budget Management

### Cost Breakdown (Approximate)

| Operation | Cost | Tool |
|-----------|------|------|
| Playbook lookup | FREE | get_modification_advice (if playbook hit) |
| Doc search (intelligent) | ~$0.02-0.05 | search_docs_intelligent |
| Doc search (agents) | ~$0.02-0.08 | search_docs_with_agents_sdk |
| Vision diagnosis | ~$0.05-0.10 | diagnose_render_issue |
| Asset evaluation | FREE | evaluate_render_v2 (local ML) |
| VLM diagnosis | FREE | diagnose_issues_v2 (local Moondream) |

### Budget Strategy

```
MONTHLY BUDGET: $20
  - Vision: $10
  - Documentation: $8
  - Buffer: $2

STRATEGY:
1. Always try playbook FIRST (free)
2. Use local evaluation (evaluate_render_v2) for metrics
3. Use local VLM (diagnose_issues_v2) before GPT vision
4. Reserve GPT vision for stuck situations
5. Check budget every 5 iterations:
   mcp__blender-librarian__get_budget_status
```

---

## Common Workflows

### Workflow A: Quick Iteration (Speed Priority)

```python
# Minimal tool calls, fast feedback loop
for iteration in range(max_iterations):
    1. generate_script / modify_script
    2. execute_blender_script
    3. evaluate_render_v2(profile="quick")

    if score >= threshold:
        break

    4. get_modification_advice  # Uses playbook (free)
    5. Repeat
```

### Workflow B: Quality Iteration (Quality Priority)

```python
# Full evaluation, deep diagnosis
1. start_experiment_session
2. query_knowledge_base
3. recommend_technique

for iteration in range(max_iterations):
    4. generate_script / modify_script
    5. validate_script
    6. execute_blender_script
    7. evaluate_render_v2(profile="standard")

    if score >= threshold:
        break

    8. diagnose_issues_v2
    9. get_warnings_before_change
    10. get_modification_advice
    11. record_experiment_result
    12. Repeat

13. end_experiment_session
14. record_technique_outcome
```

### Workflow C: Research Mode (Learning Priority)

```python
# When exploring new effect types
1. Browse blender-manual hierarchy
2. search_tutorials for the effect
3. search_python_api for relevant types
4. list_techniques to see options

5. generate_script with technique_name

6. Run full quality workflow (B)

7. Record ALL learnings:
   - add_manual_learning for parameter insights
   - add_to_playbook for successful fixes
   - record_technique_outcome for UCB1
```

---

## Error Recovery

### Script Execution Failures

```
ERROR: "Python script error at line X"
  │
  ├─► parse_blender_errors → Get suggested_fix
  │
  ├─► If API error: search_bpy_types for correct API
  │
  └─► modify_script with fix, retry execution
```

### Quality Score Stuck

```
SITUATION: Score not improving after 3 iterations
  │
  ├─► Try different technique (not just parameters)
  │   └─► recommend_technique with force_random=true
  │
  ├─► Re-examine reference image
  │   └─► Is it actually achievable with Mantaflow?
  │
  └─► Query knowledge base for similar stuck situations
    └─► query_knowledge_base("stuck", effect_type)
```

### Budget Exhausted

```
SITUATION: get_budget_status shows < $0.50 remaining
  │
  ├─► Switch to FREE-only workflow:
  │   - evaluate_render_v2 (local ML)
  │   - diagnose_issues_v2 (local VLM)
  │   - get_modification_advice (playbook only)
  │
  └─► Do NOT call:
      - diagnose_render_issue (GPT vision)
      - search_docs_intelligent (GPT)
      - search_docs_with_agents_sdk (GPT)
```

---

## State Persistence

### Saving State (For Context Recovery)

```
When approaching context limits, save state:

mcp__iteration-controller__save_iteration_state(
    session_id="solar_v1_20260106",
    asset_name="solar_prominence_v1",
    effect_type="sun",
    current_iteration=3,
    best_score=62.5,
    best_iteration=2,
    parameters_current=<JSON>,
    issues_current=<JSON array>,
    next_action="adjust_params",
    status="in_progress"
)
```

### Resuming State

```
In new context, restore state:

state = mcp__iteration-controller__load_iteration_state(
    session_id="solar_v1_20260106"
)

# Continue from where you left off
```

---

## Metrics Reference

### Overall Score Interpretation

| Score | Quality | Action |
|-------|---------|--------|
| 90-100 | Production Ready | Ship it |
| 70-89 | Good | Minor polish optional |
| 50-69 | Acceptable | Worth iterating |
| 30-49 | Poor | Needs significant work |
| 0-29 | Failed | Consider different approach |

### Individual Metrics

| Metric | Range | Good Value | Meaning |
|--------|-------|------------|---------|
| LPIPS | 0-1 | < 0.35 | Perceptual similarity (lower=better) |
| SigLIP | 0-1 | > 0.5 | Semantic similarity (higher=better) |
| TOPIQ | 0-1 | > 0.6 | Technical quality (higher=better) |
| Feature CV | 0-100 | > 20 | Texture variety (higher=more natural) |
| DINOv2 | 0-1 | > 0.5 | Structural similarity (higher=better) |

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────────┐
│                     QUICK REFERENCE                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  START SESSION                                                       │
│    experiment-tracker: start_experiment_session                     │
│    experiment-tracker: query_knowledge_base                         │
│    script-generator:   recommend_technique                          │
│                                                                      │
│  GENERATE                                                            │
│    script-generator:   generate_script                              │
│    script-generator:   validate_script                              │
│                                                                      │
│  EXECUTE                                                             │
│    blender-executor:   execute_blender_script                       │
│    blender-executor:   list_run_outputs                             │
│                                                                      │
│  EVALUATE                                                            │
│    asset-evaluator:    evaluate_render_v2 ← PRIMARY                 │
│    asset-evaluator:    diagnose_issues_v2 (if needed)               │
│    blender-librarian:  diagnose_render_issue (if stuck)             │
│                                                                      │
│  ITERATE                                                             │
│    blender-librarian:  get_modification_advice ← TRY FIRST          │
│    experiment-tracker: get_warnings_before_change                   │
│    script-generator:   modify_script                                │
│    asset-evaluator:    compare_renders_v2                           │
│    experiment-tracker: record_experiment_result                     │
│                                                                      │
│  FINALIZE                                                            │
│    blender-librarian:  record_fix_outcome                           │
│    blender-librarian:  add_to_playbook                              │
│    script-generator:   record_technique_outcome                     │
│    experiment-tracker: end_experiment_session                       │
│                                                                      │
│  DOCUMENTATION (when needed)                                         │
│    blender-manual:     search_python_api                            │
│    blender-manual:     search_vdb_workflow                          │
│    blender-manual:     search_bpy_types                             │
│                                                                      │
│  STATE MANAGEMENT                                                    │
│    iteration-controller: save_iteration_state                       │
│    iteration-controller: load_iteration_state                       │
│                                                                      │
│  BUDGET CHECK                                                        │
│    blender-librarian:  get_budget_status                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Appendix: Effect-Specific Guidelines

### Sun/Star Effects

**Key parameters:** flame_max_temp (5778K), emission_intensity, limb_darkening
**Common issues:** Too orange (temp too low), flat (missing limb darkening), no prominences
**Reference dataset:** assets/reference_images/star/

### Explosions

**Key parameters:** turbulence (0.6-0.8), vorticity, burning_rate, smoke presence
**Common issues:** Too uniform, wrong color gradient, no mushroom shape
**Technique hint:** "rising_mushroom" for classic mushroom cloud

### Fire

**Key parameters:** flame_smoke (1-3), burning_rate, temperature gradient
**Common issues:** Too static, wrong color, smoke/flame imbalance

### Nebula

**Key parameters:** smoke_density (0.2-0.4), emission low, turbulence low
**Common issues:** Too bright, too uniform, missing dust lanes

---

**Document Version:** 1.0
**Last Updated:** 2026-01-06
**Maintained by:** AI Operator System Documentation
