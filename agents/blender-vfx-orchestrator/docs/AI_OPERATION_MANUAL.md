# AI Operation Manual - Blender VFX Orchestrator

**Version:** 2.0.0
**Target Audience:** AI Agents (Claude, GPT-5.2, or similar LLMs)
**Purpose:** Autonomous VFX asset generation with minimal human intervention

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

### Principle 5: Record Everything for Future Sessions

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

## Standard Workflows

### Workflow 1: New Asset Generation

```
1. Receive AssetRequest
2. Create session with generate_session_id()
3. Check budget availability
4. ITERATION LOOP:
   a. IF iteration > 1:
      - Call pre_iteration_research()
      - Check warning_level and escape_level
      - If escape_level >= 2: switch technique
   b. Search code patterns for known fixes
   c. Generate/modify script (delegate_to_script_writer)
   d. Execute script (delegate_to_executor)
   e. IF execution failed:
      - Parse error
      - Fix script
      - Re-execute (max 2 retries)
   f. Evaluate quality (delegate_to_quality_analyst)
   g. Record experiment outcome (delegate_to_learning_agent)
   h. IF passed: complete session
   i. IF failed:
      - Extract patterns if improvement >= 5
      - Update stuck_state
      - Continue loop
5. Return SessionState
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

---

## Budget Management

### Cost Estimation per Iteration

| Component | Estimated Cost |
|-----------|---------------|
| Script generation (GPT-5.2) | $0.05-0.15 |
| Error handling (GPT-5.2) | $0.02-0.05 |
| Quality evaluation (vision) | $0.10-0.30 |
| Documentation search | $0.02-0.05 |
| **Total per iteration** | **$0.20-0.55** |

### Budget Checks

```python
# Before starting session
if not budget_tracker.can_afford_evaluation():
    raise BudgetExceededError("Monthly budget exhausted")

# At 80% budget
if budget_tracker.get_remaining() < budget_tracker.monthly_limit * 0.2:
    # Switch to reduced evaluation profile
    evaluation_profile = "quick"

# At 100% budget
if budget_tracker.get_remaining() <= 0:
    # Return best result immediately
    return best_result
```

---

## Logging and Debugging

### Log Levels

| Level | When Used |
|-------|-----------|
| `INFO` | Iteration start/end, handoffs |
| `DEBUG` | Tool calls, parameter changes |
| `WARNING` | Escape level changes, budget warnings |
| `ERROR` | Execution failures, API errors |

### Key Log Points

```python
# Iteration start
logger.info(f"[Iteration {n}] Starting with escape_level={escape_level}")

# Handoff
logger.info(f"[Handoff] Delegating to {agent_name}: {reason}")

# Escape level change
logger.warning(f"[Escape] Level escalated: {old} -> {new}, reason: {trigger}")

# Pattern applied
logger.info(f"[Pattern] Applied {pattern_id} to {script_path}")

# Session complete
logger.info(f"[Complete] Session {session_id}: score={final_score}, iterations={n}")
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

*End of Operation Manual*
