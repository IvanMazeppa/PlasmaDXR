# AI Operation Manual - Blender VFX Orchestrator

**Version:** 3.4.1
**Last Updated:** 2026-01-26
**Target Audience:** AI Agents (Claude, GPT-5.2, or similar LLMs)
**Purpose:** Autonomous VFX asset generation with minimal human intervention

> **Architecture Note:** This system uses **code-based pipeline orchestration** with **3 Coordinator agents** for intelligent decisions, protected by **RunHooks + Guardrails** for defense-in-depth validation. All agents share **SDK Session context** for conversation persistence. The deprecated handoff-based `create_asset()` method should NOT be used.

---

## Quick Reference

### Start Asset Generation

```python
from orchestrator import create_vfx_asset

result = await create_vfx_asset(
    asset_name="explosion_001",
    description="Large fiery explosion with rising smoke column",
    effect_type="explosion",
    quality_threshold=60.0,
    max_iterations=5
)
```

### Resume Interrupted Session

```python
from orchestrator import resume_vfx_session

result = await resume_vfx_session("session_20260118_explosion_001")
```

### Check Budget Before Starting

```python
from orchestrator import get_orchestrator

orchestrator = await get_orchestrator()
budget = orchestrator.get_budget_status()

if not budget["can_afford_evaluation"]:
    print(f"Budget exhausted: ${budget['total_spent']:.2f} / ${budget['monthly_limit']}")
```

---

## Autonomy Architecture Addendum (2026-01-29)

### State Machine (Required)
All runs must follow a deterministic state flow:
```
PLAN → GENERATE → VALIDATE → EXECUTE → EVALUATE → DECIDE
```
Failure routing rules:
- VALIDATE fail → back to GENERATE with fixes.
- EXECUTE fail → DIAGNOSE → FIX → EXECUTE.
- EVALUATE fail → IMPROVE → GENERATE.

### Artifact-First Handoffs
Agents must hand off **file references**, not inline dumps. Required artifacts:
- Script path
- Run log path
- Render output path
- Cache path
- Scorecard path

### Run Manifest + Scorecard
Each run must emit:
- **Run manifest** (inputs, outputs, env, decisions)
- **Scorecard** (quality score, cache size, render count, critical issues)

### Bake/Render Gates
Runs must be rejected if:
- Cache is missing or too small
- Render set count is wrong
- Critical errors exist in logs

### SDK Feature Notes (2026-01-29)
- Detailed SDK upgrade plan: `docs/SDK_FEATURES_UPGRADE_PLAN_2026-01-29.md`
- Patch implementation notes: `docs/ORCHESTRATOR_PATCH_RUNCONFIG_PARALLEL_PREFLIGHT_2026-01-29.md`

## Core Operating Principles

### Principle 1: Quality Gates Are Non-Negotiable

**PASS** requires ALL of:
- `overall_score >= 60`
- No critical issues (ZERO_LIGHTS, BLACK_SCREEN, CLIPPING)
- If reference provided: acceptable LPIPS similarity

**NEVER** mark a session as PASSED if critical issues exist, regardless of score.

### Principle 2: Proactive Research Prevents Wasted Iterations

**BEFORE** each modification iteration (after iteration 1):

1. Call `pre_iteration_research()` with current state
2. Check `warning_level` in response
3. If `warning_level != "none"`, follow the `escape_action`

```python
# Example pre-iteration research
research = pre_iteration_research(
    current_issue="smoke too thin",
    current_approach="increasing flame_smoke",
    iteration_history=json.dumps([
        {"iteration": 1, "score": 35, "issue": "smoke too thin"},
        {"iteration": 2, "score": 38, "issue": "smoke too thin"}
    ]),
    effect_type="pyro"
)

if research["warning_level"] == "early":
    # Check knowledge base BEFORE trying another variation
    patterns = search_code_patterns(issue="smoke too thin", effect_type="pyro")
    if patterns["patterns_found"] > 0:
        # Apply proven pattern instead of guessing
        apply_pattern_to_script(...)
```

### Principle 3: Code Patterns Over Parameter Guessing

**ALWAYS** search for existing patterns before modifying scripts:

```python
# CORRECT: Check pattern library first
patterns = search_code_patterns(
    issue="lacks density",
    effect_type="explosion",
    min_confidence=70
)

if patterns["patterns_found"] > 0:
    best_pattern = patterns["patterns"][0]
    # Apply proven fix
    apply_pattern_to_script(
        script_path=current_script,
        pattern_id=best_pattern["pattern_id"]
    )
else:
    # No pattern found - generate fix and record if successful
    # ... make modification ...
    if improvement >= 5.0:
        record_code_pattern(
            issue="lacks density",
            code_snippet=extracted_code,
            effect_type="explosion",
            improvement=improvement,
            experiment_id=current_experiment_id
        )
```

### Principle 4: Escape Velocity Prevents Infinite Loops

Track and respond to escape level:

| Level | Detection | Required Action |
|-------|-----------|-----------------|
| 0 | Normal | Proceed with planned modification |
| 1 | Plateau 2x | Query knowledge base BEFORE modifying |
| 2 | Same issue 2x | **DO NOT MODIFY** - generate NEW script with DIFFERENT technique |
| 3 | Same issue 3x | Mine documentation with semantic search |
| 4 | No progress 4x | **STOP** - report stuck and request guidance |

**Critical:** At Level 2+, do NOT continue modifying the same script. Switch techniques.

```python
# Level 2+ handling
if escape_level >= 2:
    # Get untried techniques
    untried = stuck_state.get_untried_techniques(available_techniques)

    if untried:
        # Generate completely new script
        new_script = generate_script(
            effect_type=effect_type,
            technique=untried[0]  # Try first untried
        )
        # Reset stuck counters for fair evaluation
        stuck_state.reset_for_new_technique()
    else:
        # All techniques tried - escalate to Level 3
        # Mine documentation for novel approaches
        alternatives = find_alternative_approaches(
            current_approach=current_technique,
            issue=primary_issue,
            effect_type=effect_type,
            exclude_techniques=techniques_tried
        )
```

### Principle 5: Defense-in-Depth Validation

The system uses two validation layers to ensure robust agent behavior:

#### RunHooks (Tool-Level)

RunHooks intercept tool calls and enforce requirements:

```python
# These are automatically applied in create_asset_pipeline()

# Loop Detection
if tool_call_count["semantic_search"] > 3:
    raise LoopDetectedError("Tool called too many times")

# Documentation Requirement
if tool_name == "write_script" and not doc_query_made:
    raise DocQueryRequiredError("Research required before scripting")
```

**Exceptions to Handle:**
- `LoopDetectedError` - Stop and return best result
- `DocQueryRequiredError` - Force research before continuing
- `TurnBudgetExceededError` - Stop agent and return partial result

#### Guardrails (Agent-Level)

Guardrails validate agent inputs and outputs:

**Input Guardrails:**
- `require_research_context` - Script Writer must receive research findings
- `validate_effect_type` - Must specify valid effect type
- `check_budget_before_quality` - Block if budget exhausted

**Output Guardrails:**
- `validate_script_output` - Must have script_path, technique_used
- `validate_quality_output` - Score 0-100, passed boolean, no critical issues if passed=True
- `validate_technique_decision` - Must have selected_technique, reasoning
- `validate_modification_decision` - Must have action, parameter_changes if modify_params
- `validate_quality_decision` - passed/next_action must be consistent

**Handling Guardrail Failures:**
```python
try:
    result = await Runner.run(agent, prompt)
except InputGuardrailTripwireTriggered as e:
    # Input validation failed - check e.guardrail_result.output_info
    print(f"Input validation failed: {e.guardrail_result.output_info['reason']}")
except OutputGuardrailTripwireTriggered as e:
    # Output validation failed - agent produced invalid output
    print(f"Output validation failed: {e.guardrail_result.output_info['errors']}")
```

---

### Principle 6: Conversation Context is Shared (SDK Sessions)

All agents within a VFX session share **conversation context** via SDK Sessions. This means:

1. **Agents can reference previous phases** - Script Writer can see Research Agent findings
2. **Context persists across iterations** - Quality Analyst remembers previous evaluations
3. **No manual history management** - SDK handles conversation threading automatically

**How It Works:**

```python
# At pipeline start, an SDK session is created
sdk_session = get_or_create_sdk_session(session_id)

# All Runner.run() calls share this session
result = await Runner.run(research_agent, prompt, session=sdk_session, ...)
result = await Runner.run(script_writer, prompt, session=sdk_session, ...)
# Script Writer automatically sees Research Agent's output!
```

**What You Can Reference:**

| Agent | Can Reference |
|-------|---------------|
| TechniqueSelector | Research findings |
| Script Writer | Research + Technique decision |
| Quality Analyst | Script generated, execution results |
| Learning Agent | Quality evaluation, iteration history |
| QualityGateJudge | All previous phases in current iteration |

**Important:** Session context is per-VFX-session, not global. Each asset generation gets its own conversation thread stored in `sessions/sdk/vfx_conversations.db`.

---

### Principle 7: Record Everything for Future Sessions

After EVERY iteration, record:

1. **Experiment outcome** (regardless of success/failure)
2. **Pattern** (if improvement >= 5 points)
3. **Pattern feedback** (if a stored pattern was applied)

```python
# After each iteration
record_experiment_result(
    experiment_id=exp_id,
    score=current_score,
    improvement=score_delta,
    success=score_delta > 0,
    observed_effects=["density increased", "color shifted slightly"],
    learnings=["flame_smoke above 2.5 creates visible density"]
)

# If significant improvement
if score_delta >= 5.0:
    extract_successful_pattern(
        original_path=original_script,
        modified_path=modified_script,
        issue=issue_fixed,
        improvement=score_delta,
        effect_type=effect_type,
        experiment_id=exp_id
    )

# If pattern was used
if applied_pattern_id:
    report_pattern_outcome(
        pattern_id=applied_pattern_id,
        success=score_delta > 0,
        improvement=score_delta
    )
```

---

## Instruction Style Optimization (Machine-First Prompts)

The current prompts are prose-heavy and occasionally contradictory. The agents behave more reliably with compact, machine-optimized instruction blocks that make tool order, stopping conditions, and output schema explicit.

### Prompt Design Rules
- Short imperative lines (no narrative).
- Explicit tool order (T1/T2/T3).
- Explicit stop conditions (max turns or max tool calls).
- Schema-first output spec (fields, types, required/optional).
- No contradictory rules (e.g., "never use hasattr" vs "always use hasattr").

### Template: Script Writer (Machine-Optimized)
```
ROLE: Generate or modify Blender Python code.
INPUTS: effect_type, description, research_summary, technique, constraints.
TOOLS: semantic_search_blender_docs, search_blender_api_by_intent, write_script, validate_script, modify_script.
TURNS: MAX 5.

T1: If API uncertain -> one doc search.
T2: Write complete code (no templates).
T3: write_script(code, output_name, technique_name).
T4: validate_script(script_path).
T5: If validation fails -> one modify_script, then return.

OUTPUT (ScriptOutput):
- script_path (str, required)
- technique_used (str, required)
- parameters_set (dict, optional)
- validation_passed (bool, required)
- validation_errors (list, optional)
STOP after T5 regardless of quality.
```

### Template: Quality Analyst (Machine-Optimized)
```
ROLE: Evaluate render quality strictly.
TOOLS: analyze_with_vision, find_reference_images, compare_to_reference.
TURNS: MAX 3.

T1: analyze_with_vision(render_path). If reference exists -> find_reference_images.
T2: compare_to_reference if reference available.
T3: Return QualityOutput.

OUTPUT (QualityOutput): overall_score (0-100), passed (bool), primary_issue, issues[], suggestions[], vision_assessment, reference_similarity.
```

### Template: Learning Agent (Machine-Optimized)
```
ROLE: Mandatory pre-generation exploration controller + record outcomes.
TOOLS: query_knowledge_base, search_code_patterns, record_experiment_result, (Docs Expert if missing doc refs).
TURNS: MAX 3.

T1: query_knowledge_base + search_code_patterns (parallel ok).
T2: Propose doc-grounded candidates + micro-experiments (Blender 5 only).
T3: record_experiment_result (exactly once), return LearningOutput.

OUTPUT (LearningOutput):
- proposals[] (technique, params, expected_effect, doc_refs[])
- micro_experiments[] (minimal script + success_criteria + doc_refs[])
- anti_patterns[]
- experiment_recorded, pattern_extracted, pattern_id, next_action, suggested_modifications[], parameter_modifications{}.
```

### Doc-Gating Rules (Blender 5 Only)
- Every proposal must include `doc_refs` from Blender 5 docs.
- If `doc_refs` are missing, **force Docs Expert** before proposal is accepted.
- New API usage requires **at least one** micro-experiment before full integration.

### Template: Executor / Docs Expert (Machine-Optimized)
```
Executor:
T1 execute_blender_script; T2 parse_blender_errors or list_run_outputs; T3 return ExecutionOutput.

Docs Expert:
T1 semantic_search_blender_docs or search_blender_api_by_intent
T2 validate_parameter_range (if needed)
T3 return DocsOutput
```

### Contradictions Resolved (v3.3.0)

The following contradictions were identified and fixed in `tools/dynamic_instructions.py`:

| Issue | Resolution |
|-------|------------|
| "No compatibility checks" vs "Always use hasattr()" | Removed generic hasattr() guidance. Use `.get()` only for KNOWN Principled BSDF socket renames. |
| "No backward compatibility" vs ".get() fallback" | Clarified: `.get()` is for KNOWN API changes (socket renames), not version detection. |
| Prompt turn budgets vs `max_turns` | Aligned: Prompts specify 3-5 turns, `max_turns` set to 6-8 to allow buffer. |

**Current Blender 5.0 API Guidance:**

```python
# KNOWN socket renames - use .get() for these specific sockets:
bsdf.inputs.get('Emission Color', bsdf.inputs.get('Emission')).default_value = (1,1,1,1)
bsdf.inputs.get('Specular IOR Level', bsdf.inputs.get('Specular')).default_value = 0.5

# Direct property access (Blender 5.0):
obj.visible_shadow = False  # Not cycles_visibility.shadow

# Compositor setup:
scene.use_nodes = True
if scene.node_tree is not None:  # Guard for None, not version
    nt = scene.node_tree
```

## Standard Workflows

### Workflow 1: New Asset Generation

```
1. Receive AssetRequest
2. Create session with generate_session_id()
3. Check budget availability
4. PHASE 0: RESEARCH
   - Research Agent gathers best approach for effect type
   - Extract alternative approaches from findings
5. PHASE 0.5: TECHNIQUE SELECTION (Coordinator)
   - TechniqueSelector Coordinator analyzes research
   - Returns TechniqueDecision with selected_technique, reasoning, key_parameters
6. ITERATION LOOP:
   a. IF iteration > 1:
      - PHASE 1.1: MODIFICATION STRATEGY (Coordinator)
      - ModificationStrategist decides: modify_params OR switch_technique
      - Returns ModificationDecision with parameter_changes
   b. IF iteration == 1:
      - Use technique from TechniqueSelector
   c. PHASE 0.75: LEARNING (Doc-Grounded)
      - Learning Agent proposes candidates + micro-experiments (Blender 5 doc refs required)
   d. PHASE 1: SCRIPT GENERATION
      - Script Writer generates/modifies script from Learning Agent proposals
   e. PHASE 1.5: API VALIDATION
      - Validate Blender 5.0 API calls, apply corrections
   f. PHASE 2: EXECUTION
      - Execute script in Blender
      - Parse errors if failed, fix and retry (max 2)
   f2. PHASE 2.7: ARTIFACT GATES (Deterministic)
      - Validate cache exists and size >= 1MB (for simulation effects)
      - Validate at least 1 render file exists
      - If gates fail: skip quality evaluation, route to next iteration
      - This catches "empty bake" bugs WITHOUT LLM involvement
   g. PHASE 3: QUALITY EVALUATION
      - Quality Analyst evaluates render with vision + metrics
   h. PHASE 4: LEARNING (Post-Eval)
      - Learning Agent records experiment, updates knowledge base
   i. PHASE 5: QUALITY GATE (Coordinator)
      - QualityGateJudge interprets results
      - Returns QualityDecision with passed, next_action, escape_level
   i. IF passed: complete session
   j. IF failed:
      - Extract patterns if improvement >= 5
      - Continue loop based on next_action
7. Return SessionState
```

### Workflow 2: Session Resumption

```
1. Load SessionState from persistence
2. Check status (skip if PASSED/CANCELLED)
3. Restore stuck_state
4. Build resumption prompt with:
   - Recent iteration history
   - Current script path
   - Best score achieved
   - Escape level
5. Continue from ITERATION LOOP step 4a
```

### Workflow 3: Pattern Application

```
1. Identify primary issue from quality evaluation
2. Search pattern library:
   patterns = search_code_patterns(issue, effect_type, min_confidence=70)
3. IF high-confidence pattern found:
   a. Get full pattern code: get_pattern_code(pattern_id)
   b. Apply to script: apply_pattern_to_script(...)
   c. Execute modified script
   d. Evaluate quality
   e. Report outcome: report_pattern_outcome(pattern_id, success, improvement)
4. IF no pattern found:
   a. Generate fix using knowledge base suggestions
   b. If successful (improvement >= 5): record_code_pattern(...)
```

### Workflow 4: Escape Velocity Handling

```
1. After each iteration, update stuck_state:
   escape_level = stuck_state.update_from_iteration(
       score=current_score,
       primary_issue=primary_issue,
       technique_used=current_technique
   )

2. Route based on escape_level:

   LEVEL 0 (NORMAL):
   → Proceed with standard modification

   LEVEL 1 (KNOWLEDGE_CHECK):
   → Query knowledge base: query_knowledge_base(issue)
   → Check pattern library: search_code_patterns(issue)
   → If failure rate > 50% for current approach: escalate to Level 2
   → Otherwise: apply best suggestion

   LEVEL 2 (SWITCH_TECHNIQUE):
   → DO NOT modify current script
   → Get untried techniques: stuck_state.get_untried_techniques()
   → Generate NEW script with different technique
   → Call stuck_state.reset_for_new_technique()

   LEVEL 3 (MINE_DOCS):
   → Use semantic search: find_alternative_approaches(...)
   → Search with exclusions: semantic_search_blender_docs(...)
   → Try first novel approach found
   → If nothing found: escalate to Level 4

   LEVEL 4 (REQUEST_GUIDANCE):
   → STOP iteration loop
   → Report summary:
     - Total iterations
     - Techniques tried
     - Best score achieved
     - Persistent issues
   → Request human guidance OR return best result
```

---

## Tool Usage Reference

### Search Tools Priority

When searching for solutions, use tools in this order:

1. **search_code_patterns** - Highest priority, returns working code
2. **query_knowledge_base** - Second priority, returns learnings and warnings
3. **semantic_search_blender_docs** - Third priority, finds related documentation
4. **search_manual** - Fourth priority, keyword-based fallback

### When to Use Each Semantic Search Tool

| Tool | Use When |
|------|----------|
| `semantic_search_blender_docs` | Need conceptually related documentation |
| `find_alternative_approaches` | Current approach failing, need fundamentally different solution |
| `search_blender_api_by_intent` | Know what you want to DO but not which API |

### Pattern Tools Decision Tree

```
Is there a known fix for this issue?
│
├─ YES: search_code_patterns() returns results
│  │
│  ├─ High confidence (>70%): Apply pattern directly
│  │  → apply_pattern_to_script()
│  │  → After execution: report_pattern_outcome()
│  │
│  └─ Low confidence (<70%): Consider but verify
│     → apply_pattern_to_script()
│     → Watch for side effects
│     → After execution: report_pattern_outcome()
│
└─ NO: No patterns found
   │
   ├─ Generate fix manually
   │
   └─ If fix achieves >= 5 point improvement:
      → record_code_pattern()
      → OR extract_successful_pattern() for diff-based extraction
```

---

## Error Handling

### Blender Execution Errors

| Error Type | Detection | Recovery |
|------------|-----------|----------|
| Syntax error | `parse_blender_errors()` returns syntax issue | Fix script and retry |
| Missing dependency | Error mentions import/module | Add missing import |
| API error | Error mentions bpy.types/ops | Check docs, fix API call |
| Memory error | "Out of memory" | Reduce resolution |
| Timeout | Execution exceeds limit | Reduce complexity |

### Quality Evaluation Errors

| Error Type | Detection | Recovery |
|------------|-----------|----------|
| No render output | `render_path` is None | Check script's render settings |
| Black screen | Critical issue detected | Check lighting, camera |
| API rate limit | OpenAI error | Wait and retry |
| Budget exceeded | Budget check fails | Return best result |

### Session Recovery

| State | Recovery Action |
|-------|-----------------|
| `IN_PROGRESS` | Resume from last iteration |
| `PAUSED` | Resume from saved state |
| `FAILED` | Can retry with different parameters |
| `MAX_ITERATIONS` | Return best result or retry with higher limit |

### Artifact Gate Errors (Phase 2.7)

Artifact gates are **deterministic** checks that run after execution but before quality evaluation.
They catch structural failures without LLM involvement.

| Gate | Detection | Likely Cause | Recovery |
|------|-----------|--------------|----------|
| `CACHE_SIZE` | Cache < 1MB or missing | Empty bake: emitter misconfigured | Fix emitter setup (e.g., `use_plane_init=True` for planar liquid) |
| `RENDER_COUNT` | No render files found | Render skipped or path invalid | Check camera, output path, render settings |
| `VDB_VALIDITY` | VDB files too small | Volume export failed | Check OpenVDB addon, domain cache |

**When gates fail:**
- Quality evaluation is **skipped** (saves LLM cost)
- Iteration is recorded with score=0
- Failure diagnosis is stored in context for next iteration
- Pipeline continues to next iteration with the gate failure as the primary issue

**Key principle:** Deterministic gates trump LLM opinions. If cache is empty, the bake failed
regardless of what the LLM might say about the (non-existent) render.

### Enforcement Errors (RunHooks + Guardrails)

| Error Type | Detection | Recovery |
|------------|-----------|----------|
| `LoopDetectedError` | RunHooks: Same tool >3x | Return best result, log warning |
| `DocQueryRequiredError` | RunHooks: Script write without research | Force research phase, then retry |
| `TurnBudgetExceededError` | RunHooks: Agent exceeded turns | Stop agent, use partial result |
| `InputGuardrailTripwireTriggered` | Guardrails: Invalid agent input | Fix input and retry |
| `OutputGuardrailTripwireTriggered` | Guardrails: Invalid agent output | Use fallback decision |

---

## Budget Management

### Cost Estimation per Iteration

| Component | Estimated Cost |
|-----------|---------------|
| Script generation (GPT-5.2) | $0.05-0.15 |
| Error handling (GPT-5.2) | $0.02-0.05 |
| Quality evaluation (vision, gpt-5-mini) | $0.05-0.20 |
| Documentation search | $0.02-0.05 |
| **Total per iteration** | **$0.20-0.55** |

### Budget Checks

```python
# Before starting session
if not budget_tracker.can_afford_evaluation():
    raise BudgetExceededError("Monthly budget exhausted")

# Remaining budget helper (BudgetTracker does not expose get_remaining)
remaining = max(0.0, budget_tracker.monthly_limit - budget_tracker.get_spent())

# At 80% budget
if remaining < budget_tracker.monthly_limit * 0.2:
    # Switch to reduced evaluation profile
    evaluation_profile = "quick"

# At 100% budget
if remaining <= 0:
    # Return best result immediately
    return best_result
```

---

## Logging and Debugging

### Log Levels

| Level | When Used |
|-------|-----------|
| `INFO` | Iteration start/end, Coordinator decisions, phase transitions |
| `DEBUG` | Tool calls, parameter changes |
| `WARNING` | Escape level changes, budget warnings, Coordinator failures |
| `ERROR` | Execution failures, API errors |

### Key Log Points

```python
# Phase transitions
print(f"[Pipeline] PHASE 0.5: Technique Selection (Coordinator)", file=sys.stderr)

# Coordinator decisions
print(f"[Pipeline] Coordinator selected: {technique}", file=sys.stderr)
print(f"[Pipeline] Reasoning: {reasoning[:60]}...", file=sys.stderr)

# Iteration start
print(f"[Pipeline] ====== ITERATION {n}/{max} ======", file=sys.stderr)

# Quality Gate
print(f"[Pipeline] Quality Gate: passed={passed}, next={next_action}", file=sys.stderr)

# Escape level change
print(f"[Pipeline] Escape Level: {level}, Reasoning: {reason[:50]}...", file=sys.stderr)

# Session complete
print(f"[Pipeline] ✓ QUALITY GATE PASSED at iteration {n}", file=sys.stderr)
```

---

## Common Pitfalls

### Pitfall 1: Modifying at High Escape Level

**WRONG:**
```python
if escape_level >= 2:
    # Still trying to modify the same script
    modify_script(script_path, {"flame_smoke": 3.5})  # BAD!
```

**CORRECT:**
```python
if escape_level >= 2:
    # Generate completely new script with different technique
    untried = stuck_state.get_untried_techniques(techniques)
    generate_script(effect_type, technique=untried[0])
    stuck_state.reset_for_new_technique()
```

### Pitfall 2: Ignoring Pattern Feedback

**WRONG:**
```python
apply_pattern_to_script(script, pattern_id)
execute_script(script)
evaluate_quality()
# Forgot to report outcome!
```

**CORRECT:**
```python
apply_pattern_to_script(script, pattern_id)
execute_script(script)
result = evaluate_quality()
report_pattern_outcome(  # REQUIRED
    pattern_id=pattern_id,
    success=result.improvement > 0,
    improvement=result.improvement
)
```

### Pitfall 3: Skipping Pre-Iteration Research

**WRONG:**
```python
for iteration in range(max_iterations):
    modify_script(...)  # Immediately modifying
    execute(...)
    evaluate(...)
```

**CORRECT:**
```python
for iteration in range(max_iterations):
    if iteration > 0:  # Skip first iteration
        research = pre_iteration_research(...)
        if research["warning_level"] != "none":
            # Handle warning before proceeding
            handle_escape_action(research["escape_action"])

    # Now proceed with modification
    modify_script(...)
```

### Pitfall 4: Not Recording Successful Patterns

**WRONG:**
```python
# Made a fix that improved score by 15 points
# ... moved on without recording ...
```

**CORRECT:**
```python
if improvement >= 5.0:
    # Record for future sessions
    extract_successful_pattern(
        original_path=old_script,
        modified_path=new_script,
        issue=issue_fixed,
        improvement=improvement,
        effect_type=effect_type,
        experiment_id=exp_id
    )
```

### Pitfall 5: Ignoring Guardrail Failures

**WRONG:**
```python
# Just retry without understanding why it failed
for attempt in range(3):
    try:
        result = await Runner.run(agent, prompt)
        break
    except:  # Catching all exceptions blindly
        continue
```

**CORRECT:**
```python
try:
    result = await Runner.run(agent, prompt)
except InputGuardrailTripwireTriggered as e:
    # Understand the specific failure
    reason = e.guardrail_result.output_info.get("reason")
    if "research context" in reason:
        # Need to add research findings to prompt
        prompt = f"{research_findings}\n\n{prompt}"
        result = await Runner.run(agent, prompt)
    elif "budget exhausted" in reason:
        # Return best result, don't retry
        return best_result
except OutputGuardrailTripwireTriggered as e:
    # Output was invalid - log and fallback
    errors = e.guardrail_result.output_info.get("errors", [])
    logger.warning(f"Agent output invalid: {errors}")
    # Use fallback decision
    decision = create_fallback_decision()
```

---

## Issue → Parameter Mapping (Learning Agent)

The Learning Agent must output concrete parameter values in `parameter_modifications`, not text descriptions. Use this mapping table as a starting point:

| Quality Issue | parameter_modifications |
|---------------|------------------------|
| "overexposed/clipped/white" | `{"blackbody_intensity": 2.0, "emission_strength": 5.0}` |
| "too dark/underexposed" | `{"blackbody_intensity": 8.0, "emission_strength": 15.0}` |
| "static/no animation" | `{"temperature": 3.0, "fuel_amount": 2.0}` |
| "no surface detail" | `{"noise_strength": 2.0, "flame_vorticity": 0.8}` |
| "blob/not spherical" | `{"domain_scale": 2.0}` |
| "rises/sinks in space" | `{"beta": 0.0, "alpha": 0.0}` |
| "too fast" | `{"time_scale": 0.3, "burning_rate": 0.5}` |
| "too slow" | `{"time_scale": 2.0, "burning_rate": 2.0}` |
| "no corona/glow" | `{"emission_strength": 20.0}` |
| "fuzzy/soft edges" | `{"density": 8.0, "scatter_anisotropy": 0.8}` |

**Parameter Ranges:**

| Category | Parameter | Range | Notes |
|----------|-----------|-------|-------|
| Mantaflow | temperature | -10.0 to 10.0 | Flow temperature |
| Mantaflow | density | 0.0 to 10.0 | Volume density |
| Mantaflow | fuel_amount | 0.0 to 10.0 | Fuel for fire |
| Mantaflow | burning_rate | 0.01 to 4.0 | Combustion speed |
| Mantaflow | flame_smoke | 0.0 to 8.0 | Smoke from flames |
| Mantaflow | flame_vorticity | 0.0 to 2.0 | Flame turbulence |
| Mantaflow | beta | -5.0 to 5.0 | Density buoyancy (0.0 for space) |
| Mantaflow | alpha | -5.0 to 5.0 | Thermal buoyancy (0.0 for space) |
| Mantaflow | resolution_max | 32 to 512 | Simulation resolution |
| Shader | blackbody_intensity | 0.0 to 20.0 | Emission brightness |
| Shader | emission_strength | 0.0 to 100.0 | Glow intensity |
| Shader | scatter_anisotropy | -1.0 to 1.0 | Scattering direction |

**Usage in LearningOutput:**

```python
# CORRECT: Concrete values for direct script modification
LearningOutput(
    experiment_recorded=True,
    next_action="iterate",
    suggested_modifications=["Reduce emission to fix clipping"],
    parameter_modifications={"blackbody_intensity": 2.0, "emission_strength": 5.0}
)

# WRONG: Empty or text descriptions
LearningOutput(
    parameter_modifications={}  # BAD - Learning Agent should always provide concrete values
)
```

---

## Quality Issue Resolution Guide

### Issue: "smoke too thin"

1. Check patterns: `search_code_patterns("smoke too thin", "pyro")`
2. If no pattern, try:
   - Increase `flame_smoke` (1.0 → 2.5)
   - Increase `smoke_density` (0.5 → 0.8)
   - Decrease `dissolve_speed` (10 → 5)
3. Record pattern if improvement >= 5

### Issue: "lacks brightness/emission"

1. Check patterns: `search_code_patterns("lacks emission", effect_type)`
2. If no pattern, try:
   - Increase `emission_intensity` (1.0 → 5.0)
   - Increase `flame_max_temp` (for fire effects)
   - Add/adjust emission shader
3. Record pattern if improvement >= 5

### Issue: "clipping at boundaries"

1. Check patterns: `search_code_patterns("clipping", effect_type)`
2. If no pattern, try:
   - Increase domain size
   - Adjust camera position
   - Reduce simulation resolution to fit
3. Record pattern if improvement >= 5

### Issue: "wrong color temperature"

1. Check patterns: `search_code_patterns("color temperature", effect_type)`
2. If no pattern, try:
   - Adjust `flame_max_temp` for blackbody color
   - Modify color ramp in shader
   - Use `get_parameter_defaults(effect_type)` as baseline
3. Record pattern if improvement >= 5

---

## Autonomous Decision Matrix

| Situation | Decision | Confidence Required |
|-----------|----------|---------------------|
| Pattern found with 80%+ confidence | Apply automatically | 80% |
| Pattern found with 50-79% confidence | Apply with monitoring | 50% |
| Pattern found with <50% confidence | Consider but don't auto-apply | - |
| No pattern, knowledge base has warning | Apply alternative | 70% |
| No pattern, no warnings | Generate fix from first principles | - |
| Escape level 4 reached | Stop and report | - |
| Budget at 80% | Switch to quick evaluation | - |
| Budget exhausted | Return best result | - |

---

## Session State Reference

### Key Fields to Track

```python
session.current_iteration  # Current iteration number
session.best_score        # Highest score achieved
session.best_iteration    # Which iteration achieved best
session.stuck_state.escape_level  # Current escape level
session.stuck_state.techniques_tried  # All techniques used
session.stuck_state.same_issue_count  # Same issue persistence
session.current_issues    # Issues from last iteration
```

### State Persistence

Session state is automatically saved after each iteration to:
```
{STATE_DIR}/sessions/{session_id}.json
```

Fields persisted:
- All AssetRequest parameters
- Complete iteration history
- Stuck detection state
- Best results paths

---

## Changelog

### v3.3.0 (2026-01-23) - Multi-Agent Optimization

**Instruction Contradictions Resolved:**
- Fixed conflicting Blender 5.0 API guidance in `tools/dynamic_instructions.py`
- Removed generic "always use hasattr()" rule that contradicted "no backwards compatibility"
- Clarified: use `.get()` ONLY for known Principled BSDF socket renames
- Consolidated to single coherent rule set

**Issue → Parameter Mapping Added:**
- Added concrete issue→parameter mapping table to Learning Agent instructions
- Learning Agent now outputs actual parameter values in `parameter_modifications`
- Mapping table provides starting values for common quality issues
- Added parameter ranges reference for Mantaflow and Shader parameters

**Prompt Optimization:**
- Removed `prompt_with_handoff_instructions()` wrapper from standalone agents
- Standalone agents (Research, Script Writer, Executor, Quality Analyst, Learning) don't use handoffs
- Reduces token usage and prompt confusion

**Turn-Limited Agent Wrappers (SDK Compliance):**
- Replaced `agent.as_tool()` with `function_tool` wrappers that call `Runner.run()` with explicit `max_turns`
- SDK limitation: `agent.as_tool()` does not accept `max_turns`
- Solution per SDK docs/tools.md: wrap in custom function_tool
- Turn limits: Research=4, Script=6, Executor=3, Quality=4, Learning=3, API Validator=3
- New functions: `create_agent_tool_wrappers()`, `create_research_tool_wrapper()`

**Deprecated Handoff Architecture:**
- Added deprecation notice to handoff-based agents section in `orchestrator.py`
- Handoff agents kept only for `resume_session()` backwards compatibility
- Future: Create `resume_session_pipeline()` to fully remove

**Files Modified:**
- `tools/dynamic_instructions.py` - Fixed contradictions, added mapping table
- `orchestrator.py` - Removed handoff wrappers, added mapping, turn-limited wrappers, deprecation notices
- `specialized_agents/api_validator.py` - Turn-limited `get_api_validator_as_tool()`
- `docs/AI_OPERATION_MANUAL.md` - This changelog, mapping documentation

**Related Documents:**
- [MULTI_AGENT_OPTIMIZATION_ANALYSIS_2026-01-23.md](./MULTI_AGENT_OPTIMIZATION_ANALYSIS_2026-01-23.md) - Full analysis
- [WORKFLOW_ANALYSIS_2026-01-21.md](./WORKFLOW_ANALYSIS_2026-01-21.md) - Original problem identification

### v3.4.0 (2026-01-23) - Dynamic Instructions Enabled

**Dynamic Instructions Fully Implemented:**
- Created wrapper functions in `tools/dynamic_instructions.py`:
  - `dynamic_script_writer_standalone_instructions(ctx, agent)`
  - `dynamic_quality_analyst_standalone_instructions(ctx, agent)`
  - `dynamic_learning_agent_standalone_instructions(ctx, agent)`
- Wrappers call base dynamic functions + append pipeline-specific rules
- Standalone agents now receive KB-injected learnings at runtime

**How It Works:**
1. At agent run start, wrapper function is called with `(ctx, agent)`
2. Base function queries knowledge base for validated learnings (success_rate > 0.7)
3. Learnings are formatted and injected into instructions
4. Pipeline-specific rules (efficiency, output format) are appended
5. Complete instruction string is returned to agent

**Benefits:**
- Agents receive physics rules that emerged from experimentation
- No hardcoded rules - knowledge comes from validated experiments
- Context-aware instructions based on effect type and session state

**Files Modified:**
- `tools/dynamic_instructions.py` - Added standalone wrapper functions
- `orchestrator.py` - Standalone agents use dynamic instruction functions
- `docs/SDK_ENFORCEMENT_PROTOCOL.md` - Added section 8: Dynamic Instructions
- `docs/DYNAMIC_INSTRUCTIONS_GUIDE_2026-01-23.md` - Updated with implementation details

---

### v3.4.1 (2026-01-26) - SDK Enforcement + Budget Fixes

**Doc-Query Enforcement Enabled:**
- Script Writer must perform a doc query before `write_script`/`modify_script`.
- Enforced via RunHooks (`require_doc_query_before`).

**SDK v0.7.0 Alignment:**
- `agent.as_tool(max_turns=...)` is supported natively; wrappers only needed for custom logic.

**Budget Guardrail Consistency:**
- Budget checks use `get_spent()` + `monthly_limit` (no `get_remaining()`).

**Vision Model Default:**
- Vision evaluation defaults to `gpt-5-mini` (override via `VISION_MODEL` env var).

---

*End of Operation Manual*
