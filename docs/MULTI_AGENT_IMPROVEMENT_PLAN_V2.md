# Multi-Agent Pipeline Improvement Plan v2

**Based on:** `docs/MULTI_AGENT_PIPELINE_ANALYSIS.md`, `docs/MULTI_AGENT_IMPROVEMENT_PLAN.md`
**Additional Research:** Repo analysis, MCP best practices, Framework documentation, SpecFlow gap analysis
**Goal:** Address all identified issues with formal specifications for critical decisions
**Created:** 2026-01-03
**Updated:** 2026-01-04 (Phase 0, Phase 0.5, Phase 1 COMPLETE + Task 5.4 JSON fix)

---

## Executive Summary

The original improvement plan identified 5 issues. This updated plan adds **8 newly discovered problems** and provides **formal specifications** for the 5 critical blockers that must be resolved before implementation begins.

### Critical Architecture Decision: Skills + MCP (No Agent SDK)

**Decision:** Use Claude Code Skills + MCP servers instead of Claude Agent SDK.

**Rationale:**
- **Cost:** Agent SDK requires separate API billing (per-token). Max subscription only covers Claude.ai web interface.
- **Capability:** Claude Code already provides an agent runtime with tool execution loops, MCP connections, and session management.
- **Existing Infrastructure:** The `blender-orchestrator` skill already defines a multi-stage iterative workflow.

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                 SKILL.md (Workflow Definition)              │
│  Defines: Stages, conditionals, loop control, hard stops   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Claude Code Session                        │
│  Provides: Context persistence, tool execution loop         │
│  Limits: Context window (~200K tokens), requires running    │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│script-generator│   │blender-executor│   │asset-evaluator│
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              State & Knowledge Layer                         │
│  iteration-controller │ experiment-tracker │ blender-manual │
└─────────────────────────────────────────────────────────────┘
```

**Trade-offs:**
| Feature | Skill + MCP | Agent SDK |
|---------|-------------|-----------|
| Cost | Included in Max subscription | Per-token API billing |
| Autonomous loops | Within Claude Code session | True headless automation |
| State persistence | Via MCP tools | Built-in sessions |
| Headless/CI operation | Not supported | Supported |
| Learning curve | Low (Markdown skill) | Medium (Python/TS) |

**Conclusion:** For interactive VFX development, Skill + MCP is sufficient and cost-effective.

### Important Clarification: Skill-Driven vs. Tool-Driven Execution

The `blender-orchestrator` MCP server provides tools (`create_asset`, `get_status`, etc.) but these tools do **NOT** execute the full iteration loop autonomously. They:
- Return workflow plans/recommendations
- Provide session management
- Offer status queries

**The actual iteration loop is executed by Claude Code** interpreting the SKILL.md workflow definition. This means:
- `create_asset()` returns immediately with a plan, not after N iterations
- Claude Code must manually call `generate_script`, `execute_blender`, `evaluate_quality` in sequence
- Session state is persisted via `iteration-controller` between conversation turns

**User Expectation vs. Reality:**
| Expectation | Reality |
|-------------|---------|
| Call `create_asset()`, wait 5 minutes, get result | `create_asset()` returns immediately with recommendations |
| Autonomous headless execution | Claude Code session required as runtime |
| Single blocking API call | Multi-turn conversation with tool calls |

For headless/automated execution without Claude Code, consider the Claude Agent SDK (requires separate API billing).

### Issue Summary

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | Orchestrator describes tools but may not execute them | CRITICAL | Original |
| 2 | Technique selection is too random | HIGH | Original |
| 3 | Knowledge base not automatically consulted | MEDIUM | Original |
| 4 | External research capability missing | LOW | Original |
| 5 | Session resumption unreliable | LOW | Original |
| 6 | Missing core files (tools.py, orchestrator.py) | CRITICAL | ✅ **FIXED Phase 0.1** |
| 7 | Dual state persistence conflict | CRITICAL | ✅ **FIXED Phase 0.3** |
| 8 | claude_agent_sdk import is unnecessary (not needed for MCP architecture) | HIGH | ✅ **FIXED Phase 0.2** |
| 9 | No circuit breakers for autonomous loops | HIGH | ✅ **FIXED Phase 0.4** |
| 10 | No pre-execution validation for Blender scripts | HIGH | **NEW** |
| 11 | LPIPS 528MB model timeout risk | MEDIUM | **NEW** |
| 12 | No formal state machine definition | MEDIUM | **NEW** |
| 13 | No convergence safety criteria | HIGH | ✅ **FIXED Phase 0.4/0.5** |
| 14 | Volumetric-only effect types (no mesh physics) | HIGH | **NEW (Gemini Feedback)** |
| 15 | Mesh physics evaluation fails (penalizes clean geometry) | HIGH | **NEW (Gemini Feedback)** |
| 16 | JSON serialization crashes on numpy types | MEDIUM | **NEW (Gemini Feedback)** |
| 17 | `create_asset()` non-blocking behavior causes confusion | MEDIUM | ✅ **CLARIFIED Above** |

---

## Critical Blockers (Must Resolve Before Phase 1)

These 5 decisions must be made and documented before any implementation begins.

### Blocker 1: State Persistence Authority

**Problem:** Two parallel state persistence mechanisms exist:
- `SessionState` in `blender-orchestrator/orchestrator.py`
- `OrchestrationState` in `iteration-controller/server.py`

**Decision Required:** Which system is authoritative?

**Recommended Resolution:**
```
AUTHORITY: iteration-controller (single source of truth)
ROLE of blender-orchestrator: In-memory working state only, syncs to iteration-controller

State Ownership:
- iteration-controller: Persists to disk, handles resume, validates schema
- blender-orchestrator: Ephemeral session state, calls iteration-controller for persistence
```

**Implementation:**
1. Remove `_save_state()` from blender-orchestrator
2. Add `sync_state_to_controller()` that calls `save_iteration_state()` MCP tool
3. On session start, load from iteration-controller via `load_iteration_state()`

---

### Blocker 2: Orchestrator Execution Detection

**Problem:** Orchestrator may describe MCP tool calls in its response text without actually executing them.

**Root Cause:** LLM generates tool descriptions as prose, no execution layer verifies actual calls.

**Recommended Resolution:**
```
DETECTION MECHANISM: Pre/Post Execution Hooks + Result Verification

Pre-Execution:
1. Before generating LLM response, inject system prompt requiring structured output
2. Response must include explicit "TOOL_CALL:" markers if tools are needed

Post-Execution:
1. Parse LLM response for tool call markers
2. Verify each claimed tool call has corresponding MCP result
3. If mismatch: log discrepancy, retry with explicit instruction

Result Verification:
- Every tool call must return non-null result
- Result must pass schema validation (not just "success": true)
- Log: timestamp, tool_name, input_params, output_result, latency_ms
```

**Implementation:**
```python
class ToolExecutionVerifier:
    def __init__(self):
        self.call_log = []

    async def wrap_tool_call(self, tool_name: str, params: dict) -> dict:
        start = time.time()
        try:
            result = await self._execute_mcp_tool(tool_name, params)
            self.call_log.append({
                "timestamp": datetime.now().isoformat(),
                "tool": tool_name,
                "params": params,
                "result": result,
                "latency_ms": (time.time() - start) * 1000,
                "status": "success"
            })
            return result
        except Exception as e:
            self.call_log.append({
                "timestamp": datetime.now().isoformat(),
                "tool": tool_name,
                "params": params,
                "error": str(e),
                "status": "failed"
            })
            raise

    def verify_execution(self, claimed_tools: list[str]) -> bool:
        executed = {log["tool"] for log in self.call_log if log["status"] == "success"}
        missing = set(claimed_tools) - executed
        if missing:
            logger.error(f"Tools claimed but not executed: {missing}")
            return False
        return True
```

---

### Blocker 3: Convergence Safety Criteria

**Problem:** No stopping conditions defined. Autonomous loops can run infinitely without human oversight.

**Recommended Resolution:**
```
CIRCUIT BREAKERS (any triggers immediate stop):

1. MAX_ITERATIONS = 10
   - Hard limit on improvement attempts per session
   - After 10 iterations, MUST stop and report best result

2. MAX_WALL_TIME = 30 minutes
   - Total session duration limit
   - Prevents runaway Blender executions

3. NO_IMPROVEMENT_THRESHOLD = 3
   - If best_score unchanged for 3 consecutive iterations, stop
   - Indicates local optimum reached

4. QUALITY_FLOOR = 40
   - If score drops below 40 for 2 iterations, pause for review
   - Indicates technique fundamentally unsuitable

5. BLENDER_FAILURE_LIMIT = 2
   - If Blender crashes/times out twice in a row, stop
   - Indicates script generation bug

PAUSE TRIGGERS (request human review):
- Trust score < 0.3 AND quality < 60
- Parameters pushed to API limits (> 90% of max range)
- Knowledge base has conflicting suggestions (see Blocker 5)
```

**Implementation:**
```python
@dataclass
class ConvergenceCriteria:
    max_iterations: int = 10
    max_wall_time_minutes: int = 30
    no_improvement_threshold: int = 3
    quality_floor: int = 40
    blender_failure_limit: int = 2

    def should_stop(self, state: SessionState) -> tuple[bool, str]:
        if state.iteration >= self.max_iterations:
            return True, "MAX_ITERATIONS reached"

        elapsed = (datetime.now() - state.start_time).total_seconds() / 60
        if elapsed > self.max_wall_time_minutes:
            return True, "MAX_WALL_TIME exceeded"

        if state.iterations_without_improvement >= self.no_improvement_threshold:
            return True, "NO_IMPROVEMENT for 3 iterations"

        if state.consecutive_low_scores >= 2:
            return True, "QUALITY_FLOOR breached twice"

        if state.consecutive_blender_failures >= self.blender_failure_limit:
            return True, "BLENDER_FAILURE_LIMIT reached"

        return False, ""
```

---

### Blocker 4: Quality Evaluation Decision Tree

**Problem:** Multiple metrics can conflict. No defined resolution when LPIPS says "good" but VFX score says "bad".

**Recommended Resolution:**
```
METRIC HIERARCHY (in order of priority):

1. VFX Quality Score (0-100) - PRIMARY for standalone evaluation
   - No reference needed, provides actionable issues
   - Score >= 60 = PASS

2. LPIPS (0-1) - SECONDARY for reference comparison
   - Requires reference image
   - Score < 0.35 = PASS (lower = more similar)
   - Only used when reference_path provided

3. CLIP (0-1) - TERTIARY for semantic matching
   - Requires semantic_query
   - Score > 0.60 = PASS (higher = better match)
   - Used to validate effect matches description

CONFLICT RESOLUTION MATRIX:
| VFX | LPIPS | CLIP | Decision |
|-----|-------|------|----------|
| PASS | PASS | PASS | ✅ Accept |
| PASS | FAIL | * | ⚠️ Accept with note (style differs from reference) |
| PASS | * | FAIL | ⚠️ Accept with note (semantic drift) |
| FAIL | PASS | * | ❌ Reject (quality issues despite similarity) |
| FAIL | * | FAIL | ❌ Reject (fundamental problems) |

DECISION TREE:
1. Always run VFX diagnostics (standalone, no reference needed)
2. If VFX < 40: REJECT immediately, don't bother with LPIPS/CLIP
3. If VFX >= 60: Check LPIPS if reference provided
4. If LPIPS < 0.35 OR no reference: Check CLIP if query provided
5. Final decision based on matrix above
```

**Implementation:**
```python
def evaluate_quality(render_path: str, reference_path: str = None,
                     semantic_query: str = None) -> QualityResult:
    # Always run VFX diagnostics first
    vfx = evaluate_vfx_quality(render_path, effect_type)

    if vfx.score < 40:
        return QualityResult(passed=False, reason="VFX score below floor",
                            vfx_score=vfx.score, issues=vfx.issues)

    lpips_score = None
    clip_score = None

    if reference_path:
        lpips_score = compare_lpips(render_path, reference_path)

    if semantic_query:
        clip_score = compare_clip(render_path, semantic_query)

    # Apply decision matrix
    vfx_pass = vfx.score >= 60
    lpips_pass = lpips_score is None or lpips_score < 0.35
    clip_pass = clip_score is None or clip_score > 0.60

    if vfx_pass and lpips_pass and clip_pass:
        return QualityResult(passed=True, confidence="high")
    elif vfx_pass and not lpips_pass:
        return QualityResult(passed=True, confidence="medium",
                            note="Style differs from reference")
    elif not vfx_pass:
        return QualityResult(passed=False, reason="VFX quality issues",
                            issues=vfx.issues)
```

---

### Blocker 5: Knowledge Base Conflict Resolution

**Problem:** Multiple experiments may have conflicting learnings for the same parameter.

**Example:** "Increase domain_scale fixes clipping" vs "Increase domain_scale causes density loss"

**Recommended Resolution:**
```
CONFLICT DETECTION:
- When querying knowledge base, check for contradictory rules
- Contradiction = same parameter, opposite direction, both marked successful

RESOLUTION STRATEGY:
1. RECENCY: More recent experiments weighted 2x
2. SUCCESS_RATE: Higher success rate wins (min 3 experiments)
3. CONFIDENCE: Experiments with human feedback weighted 3x
4. CONTEXT_MATCH: If issue context matches exactly, weight 5x

CONFLICT RESPONSE:
If confidence difference < 20%:
  - Present both options to user (if trust < 0.5)
  - OR try higher-success option first, track outcome (if trust >= 0.5)

If one option clearly wins (>20% confidence difference):
  - Use winning option
  - Log that conflict existed for future analysis
```

**Implementation:**
```python
def resolve_knowledge_conflict(parameter: str, conflicts: list[Knowledge]) -> Knowledge:
    scored = []
    now = datetime.now()

    for k in conflicts:
        score = k.success_rate * 100

        # Recency bonus (2x for last 7 days)
        days_old = (now - k.last_used).days
        if days_old < 7:
            score *= 2

        # Human feedback bonus (3x)
        if k.has_human_feedback:
            score *= 3

        # Context match bonus (5x)
        if k.context_matches_current:
            score *= 5

        scored.append((score, k))

    scored.sort(reverse=True)
    winner, runner_up = scored[0], scored[1] if len(scored) > 1 else (0, None)

    confidence_diff = (winner[0] - runner_up[0]) / winner[0] * 100 if runner_up else 100

    if confidence_diff < 20:
        return KnowledgeResult(
            suggestion=winner[1],
            conflict=True,
            alternative=runner_up[1],
            requires_human_review=True
        )

    return KnowledgeResult(
        suggestion=winner[1],
        conflict=True,
        conflict_logged=True
    )
```

---

## Phase 0.5: Skill Enhancement (NEW - Parallel with Phase 0)

**Problem:** The existing SKILL.md lacks explicit loop control, circuit breakers, and mandatory knowledge integration. Claude Code follows the skill instructions but the skill doesn't enforce iteration limits or require knowledge base consultation.

### Task 0.5.1: Add Explicit Loop Control to SKILL.md

**File:** `.claude/skills/blender-orchestrator/SKILL.md`

**Current Gap:** Stage 5 says "Return to Stage 3" but doesn't track iteration count or enforce limits.

**Enhancement:**
```markdown
### Iteration Tracking (MANDATORY)

At the START of each iteration, you MUST:
1. Increment iteration counter: `current_iteration += 1`
2. Log: "=== ITERATION {n} of {max} ==="
3. Check circuit breakers (see below)

Track these variables throughout the session:
- `current_iteration`: Starts at 0, increments each loop
- `best_score`: Highest VFX score achieved
- `best_iteration`: Which iteration achieved best_score
- `iterations_without_improvement`: Counter, resets when score improves
- `session_start_time`: Timestamp when session began
```

### Task 0.5.2: Add Circuit Breakers to SKILL.md

**Enhancement:**
```markdown
### Circuit Breakers (HARD STOPS)

Before EVERY iteration, check these conditions. If ANY trigger, STOP immediately:

| Breaker | Condition | Action |
|---------|-----------|--------|
| MAX_ITERATIONS | current_iteration >= 10 | STOP, report best result |
| MAX_WALL_TIME | elapsed > 30 minutes | STOP, save state for resume |
| NO_IMPROVEMENT | iterations_without_improvement >= 3 | STOP, local optimum reached |
| QUALITY_FLOOR | score < 40 for 2 consecutive iterations | PAUSE, request human review |
| BLENDER_FAILURES | 2 consecutive Blender crashes | STOP, script has fundamental issue |

When a circuit breaker triggers:
1. Log: "⚠️ CIRCUIT BREAKER: {breaker_name} triggered"
2. Save session state via `mcp__iteration-controller__save_iteration_state()`
3. Report final status with best score and output paths
4. DO NOT continue iterating
```

### Task 0.5.3: Add Mandatory Knowledge Base Consultation

**Enhancement:**
```markdown
### Knowledge Base Consultation (MANDATORY)

Before EVERY parameter modification, you MUST:

1. Query warnings:
   ```
   Call: mcp__experiment-tracker__get_warnings_before_change(
       parameter=<param_being_changed>,
       change_type="increase" or "decrease"
   )
   ```

2. If warnings returned with severity "critical":
   - Apply the suggested mitigation
   - OR skip that parameter change
   - Log: "⚠️ Skipped {param} due to warning: {reason}"

3. Query suggestions for current issues:
   ```
   Call: mcp__experiment-tracker__suggest_experiments(
       issue=<primary_issue_from_evaluation>,
       current_params=<current_params_json>
   )
   ```

4. If suggestion confidence > 0.6, USE IT instead of heuristic fix

This is NOT optional. Skipping knowledge consultation wastes learned experience.
```

### Task 0.5.4: Add Research Integration

**Enhancement:**
```markdown
### Research Integration (When Stuck)

If iterations_without_improvement >= 2 AND no high-confidence suggestions:

1. Research techniques in Blender documentation:
   ```
   Call: mcp__blender-manual__search_tutorials(
       topic=<effect_type>,
       technique=<current_technique>
   )
   ```

2. Search for alternative approaches:
   ```
   Call: mcp__blender-manual__search_vdb_workflow(
       query="<effect_type> <issue> solution"
   )
   ```

3. If promising technique found, consider switching techniques:
   ```
   Call: mcp__script-generator__list_techniques(effect_type=<effect_type>)
   ```

   Select a DIFFERENT technique and restart from Stage 2.
```

### Task 0.5.5: Add Quality Decision Tree to SKILL.md

**Enhancement:**
```markdown
### Quality Evaluation Decision Tree

Follow this EXACT sequence when evaluating quality:

1. **Always run VFX diagnostics first** (no reference needed):
   ```
   result = mcp__asset-evaluator__evaluate_vfx_quality(...)
   ```

2. **If VFX score < 40**: REJECT immediately
   - Do NOT run LPIPS/CLIP (waste of time)
   - Diagnose issues and iterate

3. **If VFX score >= 40 but < 60**: Check secondary metrics
   - If reference_path provided: Run LPIPS
   - If semantic_query provided: Run CLIP
   - Use decision matrix below

4. **If VFX score >= 60**: Quality gate passed
   - Still run LPIPS/CLIP for completeness
   - Note any warnings but ACCEPT

**Decision Matrix:**
| VFX | LPIPS | CLIP | Decision |
|-----|-------|------|----------|
| >=60 | <0.35 | >0.60 | ✅ ACCEPT |
| >=60 | >=0.35 | * | ⚠️ ACCEPT (style differs from reference) |
| >=60 | * | <=0.60 | ⚠️ ACCEPT (semantic drift) |
| 40-59 | <0.35 | >0.60 | 🔄 ITERATE (close, minor fixes) |
| 40-59 | * | * | 🔄 ITERATE (needs work) |
| <40 | * | * | ❌ REJECT (fundamental issues) |
```

---

## Phase 0: Foundation Fixes (Must Complete First)

**Problem:** Core infrastructure issues prevent Phase 1 from working.

### Task 0.1: Create Missing Core Files

**Files to Create:**
- `agents/blender-orchestrator/tools.py` - MCP tool wrappers
- `agents/blender-orchestrator/orchestrator.py` - Main orchestration class

**Why Missing:** Import statements reference these but they don't exist. Current code catches `ImportError` silently.

**Implementation:**
```python
# tools.py
"""MCP Tool Wrappers for Blender Orchestrator"""
import json
from typing import Any

class MCPToolWrapper:
    """Wraps MCP tool calls with logging and verification."""

    def __init__(self, mcp_client):
        self.client = mcp_client
        self.call_history = []

    async def call(self, tool_name: str, **params) -> dict:
        """Execute MCP tool with full logging."""
        result = await self.client.call_tool(tool_name, params)
        self.call_history.append({
            "tool": tool_name,
            "params": params,
            "result": result
        })
        return result

    # Typed wrappers for each tool
    async def generate_script(self, effect_type: str, description: str,
                              output_name: str, **kwargs) -> dict:
        return await self.call("generate_script",
                               effect_type=effect_type,
                               description=description,
                               output_name=output_name,
                               **kwargs)

    async def execute_blender(self, script_path: str, **kwargs) -> dict:
        return await self.call("execute_blender_script",
                               script_path=script_path,
                               **kwargs)

    async def evaluate_quality(self, render_path: str, **kwargs) -> dict:
        return await self.call("evaluate_vfx_quality",
                               image_path=render_path,
                               **kwargs)
```

### Task 0.2: Remove claude_agent_sdk Dependency

**Problem:** Code imports `claude_agent_sdk` which doesn't exist.

**Why Agent SDK Is Not Needed:**
The current architecture uses **Claude Code as the agent runtime**. Claude Code (CLI) already provides:
- Autonomous tool execution loop
- MCP server connections
- Session management
- Subagent spawning via Task tool

The Agent SDK is only needed for *headless* automation (CI/CD, server-side agents).
For interactive development with Claude Code, MCP servers are sufficient.

**Cost Consideration:** Agent SDK requires separate API billing (per-token).
Max subscription only covers Claude.ai web interface, not API access.
Using Agent SDK would mean "paying twice" - keeping MCP-only avoids this.

**Solution:** Remove the import and use FastMCP directly.

**File:** `agents/blender-orchestrator/server.py`
```python
# REMOVE:
try:
    from claude_agent_sdk import create_sdk_mcp_server
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# REPLACE WITH:
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("blender-orchestrator")
```

### Task 0.3: Consolidate State Persistence

**Per Blocker 1 Decision:**
- Remove `SessionState._save_state()` from blender-orchestrator
- Add `sync_to_controller()` method that calls iteration-controller
- iteration-controller becomes single source of truth

### Task 0.4: Add Circuit Breakers

**Per Blocker 3 Decision:**
- Add `ConvergenceCriteria` class to orchestrator
- Check criteria before each iteration
- Add `MAX_ITERATIONS`, `MAX_WALL_TIME`, etc.

---

## Phase 1: Orchestrator Reliability (Priority: CRITICAL)

**Original scope retained, enhanced with verification layer.**

### Task 1.1: MCP Tool Execution Verification

**File:** `agents/blender-orchestrator/tool_executor.py` (new)

**Implementation:** Per Blocker 2 specification above.

**Key Components:**
- `ToolExecutionVerifier` class
- Pre/post execution hooks
- Result schema validation
- Execution logging with latency

### Task 1.2: Workflow Tracing

**File:** `agents/blender-orchestrator/workflow_tracer.py` (new)

**Trace Format:**
```json
{
  "session_id": "sun_v1_20260103",
  "traces": [
    {
      "timestamp": "2026-01-03T10:00:00Z",
      "state": "GENERATE_SCRIPT",
      "tool_called": "generate_script",
      "tool_params": {"effect_type": "sun", "...": "..."},
      "tool_result": {"status": "success", "script_path": "..."},
      "next_state": "EXECUTE_BLENDER",
      "latency_ms": 1250
    }
  ]
}
```

### Task 1.3: MCP Server Health Checks

**File:** `agents/blender-orchestrator/health_check.py` (new)

**Servers to Check:**
1. script-generator
2. blender-executor
3. asset-evaluator
4. experiment-tracker
5. iteration-controller
6. blender-manual

**Health Check Protocol:**
```python
async def check_server_health(server_name: str) -> HealthStatus:
    try:
        # Each server has a lightweight ping tool
        result = await mcp_client.call_tool(f"{server_name}__ping", {})
        return HealthStatus(server_name, healthy=True)
    except Exception as e:
        return HealthStatus(server_name, healthy=False, error=str(e))
```

### Task 1.4: Formal State Machine Definition (NEW)

**File:** `agents/blender-orchestrator/state_machine.py` (new)

**States:**
```python
class WorkflowState(Enum):
    SESSION_START = "session_start"
    GENERATE_SCRIPT = "generate_script"
    VALIDATE_SCRIPT = "validate_script"      # NEW
    EXECUTE_BLENDER = "execute_blender"
    EVALUATE_QUALITY = "evaluate_quality"
    DECIDE_NEXT_ACTION = "decide_next_action"
    RECORD_LEARNING = "record_learning"
    CHECK_CONVERGENCE = "check_convergence"  # NEW
    SESSION_END = "session_end"
    ERROR_RECOVERY = "error_recovery"
    AWAITING_APPROVAL = "awaiting_approval"
```

**Transitions:**
```python
TRANSITIONS = {
    SESSION_START: [GENERATE_SCRIPT],
    GENERATE_SCRIPT: [VALIDATE_SCRIPT, ERROR_RECOVERY],
    VALIDATE_SCRIPT: [EXECUTE_BLENDER, GENERATE_SCRIPT],  # Loop back if invalid
    EXECUTE_BLENDER: [EVALUATE_QUALITY, ERROR_RECOVERY],
    EVALUATE_QUALITY: [DECIDE_NEXT_ACTION, ERROR_RECOVERY],
    DECIDE_NEXT_ACTION: [RECORD_LEARNING, GENERATE_SCRIPT, SESSION_END, AWAITING_APPROVAL],
    RECORD_LEARNING: [CHECK_CONVERGENCE],
    CHECK_CONVERGENCE: [GENERATE_SCRIPT, SESSION_END],  # Circuit breaker check
    ERROR_RECOVERY: [GENERATE_SCRIPT, SESSION_END],
    AWAITING_APPROVAL: [GENERATE_SCRIPT, SESSION_END],
    SESSION_END: []
}
```

---

## Phase 2: Pre-Execution Validation (NEW - Priority: HIGH)

**Problem:** Generated Blender scripts are not validated before execution.

### Task 2.1: Blender Parameter Validation

**File:** `agents/script-generator/validator.py` (new)

**Validation Checks:**
1. All parameters within Blender 5.0 API ranges
2. No syntax errors in generated Python
3. Required attributes present (domain, flow, material)
4. Output paths valid

**Integration:**
```python
@mcp.tool
def validate_script(script_path: str) -> ValidationResult:
    """Validate Blender script before execution."""
    with open(script_path) as f:
        content = f.read()

    # Syntax check
    try:
        ast.parse(content)
    except SyntaxError as e:
        return ValidationResult(valid=False, error=f"Syntax error: {e}")

    # Parameter range check
    params = extract_parameters(content)
    for name, value in params.items():
        if name in PARAMETER_RANGES:
            min_val, max_val = PARAMETER_RANGES[name]
            if not (min_val <= value <= max_val):
                return ValidationResult(
                    valid=False,
                    error=f"{name}={value} out of range [{min_val}, {max_val}]"
                )

    return ValidationResult(valid=True)
```

### Task 2.2: Add VALIDATE_SCRIPT State

**Workflow Change:**
```
GENERATE_SCRIPT → VALIDATE_SCRIPT → EXECUTE_BLENDER
                       ↓ (if invalid)
               GENERATE_SCRIPT (with error context)
```

---

## Phase 2.5: Simulation Type Extensibility (NEW - Priority: HIGH)

**Problem:** Pipeline is hard-coded for volumetric fluid simulations. Mesh-based physics (soft body, cloth, rigid body) are not supported. Discovered during Gemini 3 Pro's "Jelly Rabbit" test.

**Context:** PlasmaDX is upgrading to support mesh physics simulations. The pipeline should be extensible without requiring a complete rewrite.

### Task 2.5.1: Effect Type Registry

**File:** `agents/script-generator/effect_registry.py` (new)

**Implementation:**
```python
"""Effect Type Registry - Maps effect types to simulation categories and patterns."""

EFFECT_TYPES = {
    # Volumetric (VDB output) - Original scope
    "pyro": {
        "category": "volumetric",
        "output": "vdb",
        "templates": ["explosion", "fire", "smoke"],
        "simulation_pattern": "bake_export"
    },
    "nebula": {
        "category": "volumetric",
        "output": "vdb",
        "templates": ["emission_nebula", "dust_cloud", "hydrogen_cloud"],
        "simulation_pattern": "bake_export"
    },
    "sun": {
        "category": "volumetric",
        "output": "vdb",
        "templates": ["stellar_surface", "corona", "prominences"],
        "simulation_pattern": "bake_export"
    },
    "supernova": {
        "category": "volumetric",
        "output": "vdb",
        "templates": ["shockwave", "remnant"],
        "simulation_pattern": "bake_export"
    },

    # Mesh-based (Blend/Alembic output) - NEW
    "soft_body": {
        "category": "mesh",
        "output": "blend",
        "templates": ["jelly", "bounce", "squish"],
        "simulation_pattern": "live_render"
    },
    "cloth": {
        "category": "mesh",
        "output": "blend",
        "templates": ["fabric", "flag", "curtain", "drape"],
        "simulation_pattern": "live_render"
    },
    "rigid_body": {
        "category": "mesh",
        "output": "blend",
        "templates": ["destruction", "dominos", "pile", "shatter"],
        "simulation_pattern": "live_render"
    },

    # Generic (LLM-driven, no template) - NEW
    "custom": {
        "category": "custom",
        "output": "auto",
        "templates": [],
        "simulation_pattern": "auto"
    }
}

def get_effect_category(effect_type: str) -> str:
    """Return the simulation category for an effect type."""
    return EFFECT_TYPES.get(effect_type, {}).get("category", "volumetric")

def get_simulation_pattern(effect_type: str) -> str:
    """Return the appropriate simulation pattern for this effect."""
    return EFFECT_TYPES.get(effect_type, {}).get("simulation_pattern", "bake_export")

def get_output_format(effect_type: str) -> str:
    """Return the expected output format for this effect."""
    return EFFECT_TYPES.get(effect_type, {}).get("output", "vdb")

def list_effect_types(category: str = None) -> list[str]:
    """List all effect types, optionally filtered by category."""
    if category:
        return [k for k, v in EFFECT_TYPES.items() if v["category"] == category]
    return list(EFFECT_TYPES.keys())
```

### Task 2.5.2: Live Render Execution Pattern

**File:** `agents/blender-executor/execution_patterns.py` (new)

**Problem:** Standard `bpy.ops.ptcache.bake_all()` is unreliable in headless Blender. Soft body, cloth, and rigid body simulations require frame-by-frame evaluation with explicit dependency graph updates.

**Implementation:**
```python
"""Execution patterns for different simulation types."""

LIVE_RENDER_TEMPLATE = '''
def run_live_render_loop(output_dir: str, frame_start: int, frame_end: int):
    """
    Robust pattern for mesh physics simulations in headless Blender.

    Why this works:
    - Mimics viewport timeline playback behavior
    - view_layer.update() forces physics solver to advance one timestep
    - Rendering immediately captures valid state before next frame

    This pattern is REQUIRED for:
    - Soft Body
    - Cloth
    - Rigid Body (when not using baked cache)
    """
    import bpy
    import os

    scene = bpy.context.scene
    os.makedirs(output_dir, exist_ok=True)

    # 1. Force Reset to start frame
    scene.frame_set(frame_start)

    # 2. Strict Linear Loop (NO frame skipping!)
    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)

        # 3. CRITICAL: Force Dependency Graph Update
        # This tells Blender's modifier stack to compute physics
        # for the current frame based on previous frame's state.
        bpy.context.view_layer.update()

        # 4. Render immediately
        scene.render.filepath = os.path.join(output_dir, f"render_{frame:04d}.png")
        bpy.ops.render.render(write_still=True)

        print(f"Frame {frame}/{frame_end} rendered")

    print(f"Live render complete: {frame_end - frame_start + 1} frames")
'''

BAKE_EXPORT_TEMPLATE = '''
def run_bake_export(output_dir: str, frame_start: int, frame_end: int):
    """
    Standard pattern for volumetric simulations (Fluid/MantaFlow).

    Steps:
    1. Bake simulation to cache
    2. Export VDB files
    3. Optionally render preview frames
    """
    import bpy
    import os

    scene = bpy.context.scene
    scene.frame_start = frame_start
    scene.frame_end = frame_end

    # Find domain object
    domain = None
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.modifiers:
            for mod in obj.modifiers:
                if mod.type == 'FLUID' and mod.fluid_type == 'DOMAIN':
                    domain = obj
                    break

    if not domain:
        raise RuntimeError("No fluid domain found")

    # Bake simulation
    bpy.context.view_layer.objects.active = domain
    bpy.ops.fluid.bake_all(bake=True)

    # VDB export handled by domain cache settings
    print(f"Bake complete: {domain.modifiers['Fluid'].domain_settings.cache_directory}")
'''

def get_execution_pattern(pattern_name: str) -> str:
    """Get the execution pattern template code."""
    patterns = {
        "live_render": LIVE_RENDER_TEMPLATE,
        "bake_export": BAKE_EXPORT_TEMPLATE,
    }
    return patterns.get(pattern_name, BAKE_EXPORT_TEMPLATE)

def inject_execution_pattern(script_content: str, pattern_name: str) -> str:
    """Inject the appropriate execution pattern into a generated script."""
    pattern_code = get_execution_pattern(pattern_name)
    return script_content.replace("# EXECUTION_PATTERN_PLACEHOLDER", pattern_code)
```

### Task 2.5.3: Custom/Generic Effect Type Support

**File:** `agents/script-generator/server.py`

**Enhancement to `generate_script()` tool:**
```python
@mcp.tool
async def generate_script(
    effect_type: str,
    description: str,
    output_name: str,
    resolution: int = 96,
    frame_start: int = 1,
    frame_end: int = 50,
    template_name: str = None,
    technique_name: str = None,
    force_random_technique: bool = False
) -> dict:
    """Generate a Blender script with support for all effect types."""

    # Import effect registry
    from effect_registry import get_effect_category, get_simulation_pattern

    category = get_effect_category(effect_type)

    if effect_type == "custom" or category == "custom":
        # Bypass template selection - use pure LLM generation
        script = await generate_custom_script(
            description=description,
            output_name=output_name,
            frame_start=frame_start,
            frame_end=frame_end
        )
    elif category == "mesh":
        # Use mesh physics templates with live render pattern
        template = select_mesh_template(effect_type, technique_name)
        script = await generate_from_template(
            template=template,
            description=description,
            output_name=output_name,
            resolution=resolution,
            frame_start=frame_start,
            frame_end=frame_end
        )
        # Inject live render execution pattern
        script = inject_execution_pattern(script, "live_render")
    else:
        # Existing volumetric flow (unchanged)
        template = select_template(effect_type, technique_name)
        script = await generate_from_template(
            template=template,
            description=description,
            output_name=output_name,
            resolution=resolution,
            frame_start=frame_start,
            frame_end=frame_end
        )

    return {
        "status": "success",
        "script_path": f"assets/blender_scripts/generated/{output_name}.py",
        "effect_type": effect_type,
        "category": category,
        "simulation_pattern": get_simulation_pattern(effect_type)
    }


async def generate_custom_script(description: str, output_name: str,
                                  frame_start: int, frame_end: int) -> str:
    """Generate a script purely from description without templates.

    Uses LLM to write the entire Blender Python script based on
    the natural language description. No template constraints.
    """
    # This would call out to an LLM or use advanced code generation
    # For now, return a skeleton that requires manual completion
    return f'''
# Custom Blender Script: {output_name}
# Description: {description}
# Generated without template - requires manual verification

import bpy

# TODO: Implement based on description
# Frame range: {frame_start} to {frame_end}

print("Custom script placeholder - implement based on description")
'''
```

### Task 2.5.4: Soft Body Stability Settings (Knowledge Base Entry)

**File:** `agents/experiment-tracker/knowledge_base/soft_body_stability.json` (new)

Based on Gemini 3 Pro's discoveries during the Jelly Rabbit task:

```json
{
  "category": "soft_body",
  "title": "Soft Body Stability Settings for Procedural Meshes",
  "source": "Gemini 3 Pro - Jelly Rabbit Task (2026-01-03)",
  "rules": [
    {
      "parameter": "step_min",
      "recommended_value": 20,
      "default_value": 5,
      "rationale": "Low substeps cause jitter explosions in procedural meshes"
    },
    {
      "parameter": "step_max",
      "recommended_value": 100,
      "default_value": 10,
      "rationale": "Allow adaptive stepping for collision resolution"
    },
    {
      "parameter": "damping",
      "recommended_value": 2.0,
      "default_value": 0.5,
      "rationale": "High damping prevents energy buildup and NaN explosions"
    },
    {
      "parameter": "goal_spring",
      "recommended_value": 0.3,
      "default_value": 0.7,
      "rationale": "Lower goal spring allows more jelly-like deformation",
      "note": "Blender 5.0+ uses 'goal_spring', earlier versions use 'stiffness'"
    }
  ],
  "warnings": [
    {
      "condition": "Joined primitives (e.g., multiple spheres)",
      "problem": "Internal geometry overlaps cause self-collision explosion",
      "solution": "Apply Voxel Remesh modifier before soft body to create manifold skin",
      "code": "bpy.ops.object.modifier_add(type='REMESH'); mod.mode='VOXEL'; mod.voxel_size=0.05"
    },
    {
      "condition": "Using bpy.ops.ptcache.bake_all() headless",
      "problem": "Often fails silently or produces empty cache",
      "solution": "Use live_render pattern with view_layer.update() per frame"
    }
  ]
}
```

---

## Phase 3: Intelligent Technique Selection (Priority: HIGH)

**Enhanced from original plan with formal exploration/exploitation.**

### Task 3.1: Technique Performance Table

**File:** `agents/experiment-tracker/database.py`

**Schema:**
```sql
CREATE TABLE technique_performance (
    id INTEGER PRIMARY KEY,
    technique_name TEXT NOT NULL,
    effect_type TEXT NOT NULL,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    total_iterations INTEGER DEFAULT 0,
    avg_final_score REAL,
    avg_iterations_to_pass REAL,
    last_used TIMESTAMP,
    exploration_bonus REAL DEFAULT 1.0,  -- Decays as technique is used
    UNIQUE(technique_name, effect_type)
);
```

### Task 3.2: UCB1 Exploration/Exploitation Algorithm

**File:** `agents/script-generator/technique_selector.py` (new)

**Algorithm:** Upper Confidence Bound (UCB1)
```python
def select_technique(effect_type: str, techniques: list[str]) -> str:
    """UCB1 algorithm for technique selection."""
    total_trials = sum(get_trial_count(t, effect_type) for t in techniques)

    scores = []
    for technique in techniques:
        trials = get_trial_count(technique, effect_type)
        if trials == 0:
            # Untried techniques get infinite score (exploration)
            scores.append((float('inf'), technique))
            continue

        avg_reward = get_avg_score(technique, effect_type) / 100  # Normalize to 0-1
        exploration_term = math.sqrt(2 * math.log(total_trials) / trials)
        ucb_score = avg_reward + exploration_term

        scores.append((ucb_score, technique))

    scores.sort(reverse=True)
    return scores[0][1]
```

**Properties:**
- Balances exploitation (high avg_reward) with exploration (high uncertainty)
- Untried techniques automatically prioritized
- Converges to best technique over time
- No need for hardcoded 80/20 split

### Task 3.3: Technique Recommendation MCP Tool

**File:** `agents/script-generator/server.py`

```python
@mcp.tool
def recommend_technique(effect_type: str, description: str) -> TechniqueRecommendation:
    """Recommend technique using UCB1 + keyword matching."""
    candidates = get_techniques_for_effect(effect_type)

    # First filter by keyword relevance
    if description:
        keywords = extract_keywords(description)
        candidates = [t for t in candidates if any(k in t.keywords for k in keywords)]

    # If no keyword matches, use all techniques
    if not candidates:
        candidates = get_techniques_for_effect(effect_type)

    # Apply UCB1 selection
    selected = select_technique(effect_type, [t.name for t in candidates])

    return TechniqueRecommendation(
        technique=selected,
        confidence=calculate_confidence(selected, effect_type),
        alternatives=get_alternatives(selected, candidates),
        rationale=explain_selection(selected, effect_type)
    )
```

---

## Phase 4: Knowledge Base Integration (Priority: MEDIUM)

**Enhanced with conflict resolution per Blocker 5.**

### Task 4.1: Mandatory Warning Checks

**File:** `agents/blender-orchestrator/orchestrator.py`

**Integration Point:** Before every `modify_script()` call:
```python
async def modify_with_warnings(self, script_path: str, modifications: dict) -> dict:
    # Query warnings for each parameter being modified
    for param, new_value in modifications.items():
        warnings = await self.tools.get_warnings_before_change(
            parameter=param,
            change_type="increase" if new_value > current_value else "decrease"
        )

        if warnings.severity == "critical":
            # Apply mitigation or skip modification
            if warnings.mitigation:
                modifications.update(warnings.mitigation)
            else:
                del modifications[param]
                self.log_skipped_modification(param, warnings.reason)

    return await self.tools.modify_script(script_path, modifications)
```

### Task 4.2: Knowledge Base Conflict Resolution

**Per Blocker 5 specification.**

**Implementation:** Add `resolve_knowledge_conflict()` to experiment-tracker.

### Task 4.3: Cross-Session Knowledge Preloading

**At session start:**
```python
async def preload_knowledge(self, effect_type: str) -> KnowledgeContext:
    """Load relevant knowledge before starting iterations."""
    relevant = await self.tools.query_knowledge_base(query=effect_type)

    return KnowledgeContext(
        common_issues=[k for k in relevant if k.type == "issue"],
        proven_solutions=[k for k in relevant if k.success_rate > 0.7],
        warnings=[k for k in relevant if k.type == "warning"],
        effect_specific_rules=[k for k in relevant if k.effect_type == effect_type]
    )
```

---

## Phase 5: Evaluation Reliability (NEW - Priority: MEDIUM)

### Task 5.1: Guaranteed LPIPS Lazy Loading

**File:** `agents/asset-evaluator/server.py`

**Problem:** 528MB model weights can cause MCP 30-second timeout.

**Solution:**
```python
class LazyLPIPS:
    _instance = None
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            import lpips
            cls._model = lpips.LPIPS(net='alex')
        return cls._model

# Use in tool
@mcp.tool
def compare_lpips(image1_path: str, image2_path: str) -> LPIPSResult:
    model = LazyLPIPS.get_model()  # Only loads on first call
    # ...
```

### Task 5.2: Quality Evaluation Decision Tree

**Per Blocker 4 specification.**

**File:** `agents/asset-evaluator/decision_tree.py` (new)

### Task 5.3: Effect Type Evaluation Registry (NEW - Gemini Feedback)

**Problem:** Evaluator penalizes clean mesh geometry because it expects high-frequency noise typical of volumetric effects. A perfectly rendered mesh scores "NO STRUCTURE" / "LOW CONTRAST".

**File:** `agents/asset-evaluator/effect_evaluators.py` (new)

**Implementation:**
```python
"""Effect Type Evaluation Registry - Different metrics for different effect categories."""

EVALUATORS = {
    "volumetric": {
        "name": "Volumetric VFX Evaluator",
        "metrics": [
            "edge_density",        # High-frequency detail
            "noise_frequency",     # Turbulent patterns
            "density_variance",    # Volume variation
            "warm_ratio",          # Fire/explosion warmth
            "dynamic_range"        # Brightness variation
        ],
        "thresholds": {
            "structure": 0.3,      # Expect visible texture
            "dynamic_range": 0.4,  # Expect contrast
            "edge_density": 0.2    # Expect detail edges
        },
        "pass_score": 60,
        "floor_score": 40
    },
    "mesh": {
        "name": "Mesh Physics Evaluator",
        "metrics": [
            "surface_smoothness",  # Clean geometry is GOOD
            "silhouette_clarity",  # Clear object boundary
            "specular_highlights", # Material response
            "motion_blur_quality", # Animation smoothness
            "shadow_definition"    # Contact shadows
        ],
        "thresholds": {
            "smoothness": 0.7,     # Expect smooth surfaces
            "clarity": 0.6,        # Clear edges
            "shadow": 0.3          # Visible shadows
        },
        "pass_score": 60,
        "floor_score": 40
    },
    "custom": {
        "name": "Semantic-Only Evaluator",
        "metrics": [
            "clip_score"           # Only semantic matching
        ],
        "thresholds": {
            "clip": 0.55
        },
        "pass_score": 55,          # Lower bar for custom
        "floor_score": 30
    }
}

def get_evaluator_for_effect(effect_type: str) -> dict:
    """Get the appropriate evaluator configuration for an effect type."""
    from effect_registry import get_effect_category
    category = get_effect_category(effect_type)
    return EVALUATORS.get(category, EVALUATORS["volumetric"])

def evaluate_with_category_awareness(image_path: str, effect_type: str) -> dict:
    """Evaluate using category-appropriate metrics."""
    evaluator = get_evaluator_for_effect(effect_type)

    if evaluator["name"] == "Mesh Physics Evaluator":
        # Use mesh-appropriate metrics (smooth is good, not bad)
        return evaluate_mesh_render(image_path, evaluator)
    elif evaluator["name"] == "Semantic-Only Evaluator":
        # Skip visual metrics, use CLIP only
        return evaluate_semantic_only(image_path)
    else:
        # Default volumetric evaluation
        return evaluate_volumetric_render(image_path, evaluator)
```

### Task 5.4: JSON Serialization Fix (NEW - Gemini Feedback)

**Problem:** `extract_vfx_diagnostics` crashes with `Object of type bool is not JSON serializable` because numpy types aren't handled.

**File:** `agents/asset-evaluator/server.py`

**Implementation:**
```python
import json
import numpy as np

class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.void):
            return None
        return super().default(obj)


def safe_json_dumps(obj: any) -> str:
    """Safely serialize any object to JSON, handling numpy types."""
    return json.dumps(obj, cls=NumpyJSONEncoder)


# Apply to all MCP tool returns that use JSON:

@mcp.tool
def extract_vfx_diagnostics(image_path: str) -> str:
    """Extract VFX-specific quality diagnostics from an image."""
    result = _compute_diagnostics(image_path)
    return safe_json_dumps(result)  # Use safe encoder


@mcp.tool
def evaluate_vfx_quality(image_path: str, effect_type: str = "explosion") -> str:
    """Compute VFX quality score with actionable issues."""
    result = _evaluate_quality(image_path, effect_type)
    return safe_json_dumps(result)  # Use safe encoder
```

**Testing:**
```python
# Test cases for numpy serialization
import numpy as np

test_obj = {
    "bool_value": np.bool_(True),
    "int_value": np.int64(42),
    "float_value": np.float32(3.14),
    "array_value": np.array([1, 2, 3]),
    "nested": {
        "np_bool": np.bool_(False)
    }
}

result = safe_json_dumps(test_obj)
# Should not raise "Object of type bool is not JSON serializable"
assert '"bool_value": true' in result
```

---

## Phase 6: Session Resumption (Priority: LOW)

**Retained from original plan.**

### Task 6.1: State Schema Versioning

**File:** `agents/iteration-controller/state_schema.py` (new)

```python
STATE_SCHEMA_VERSION = "1.0.0"

SCHEMA_V1 = {
    "session_id": str,
    "asset_name": str,
    "effect_type": str,
    "current_iteration": int,
    "best_score": float,
    "best_iteration": int,
    "parameters_current": dict,
    "issues_current": list,
    "next_action": str,
    "status": str,
    "schema_version": str  # NEW
}

def migrate_state(state: dict) -> dict:
    """Migrate old state formats to current schema."""
    version = state.get("schema_version", "0.0.0")

    if version == "0.0.0":
        # Pre-versioning state
        state["schema_version"] = "1.0.0"
        state.setdefault("issues_current", [])

    return state
```

### Task 6.2: File Existence Validation

```python
def validate_state_files(state: dict) -> ValidationResult:
    """Check that all referenced files exist."""
    missing = []

    if state.get("script_path") and not Path(state["script_path"]).exists():
        missing.append(f"Script: {state['script_path']}")

    if state.get("render_path") and not Path(state["render_path"]).exists():
        missing.append(f"Render: {state['render_path']}")

    if missing:
        return ValidationResult(valid=False, missing_files=missing)

    return ValidationResult(valid=True)
```

---

## Phase 7: External Research (Priority: LOW)

**Retained from original plan, deferred to last phase.**

---

## Implementation Order

```
Phase 0: Foundation Fixes ✅ COMPLETE (2026-01-03)
├── Task 0.1: Create missing core files (tools.py, orchestrator.py) ✅
├── Task 0.2: Remove claude_agent_sdk dependency ✅
├── Task 0.3: Consolidate state persistence ✅
└── Task 0.4: Add circuit breakers ✅

Phase 0.5: Skill Enhancement ✅ COMPLETE (2026-01-03)
├── Task 0.5.1: Add explicit loop control to SKILL.md ✅
├── Task 0.5.2: Add circuit breakers to SKILL.md ✅
├── Task 0.5.3: Add mandatory knowledge base consultation ✅
├── Task 0.5.4: Add research integration ✅
└── Task 0.5.5: Add quality decision tree ✅

Phase 1: Orchestrator Reliability ✅ COMPLETE (2026-01-04)
├── Task 1.1: MCP tool execution verification ✅
├── Task 1.2: Workflow tracing ✅
├── Task 1.3: Health checks ✅
└── Task 1.4: Formal state machine ✅

Phase 2: Pre-Execution Validation
├── Task 2.1: Blender parameter validation
└── Task 2.2: Add VALIDATE_SCRIPT state

Phase 2.5: Simulation Type Extensibility (NEW - Gemini Feedback)
├── Task 2.5.1: Effect type registry
├── Task 2.5.2: Live render execution pattern
├── Task 2.5.3: Custom/generic effect type support
└── Task 2.5.4: Soft body stability knowledge base entry

Phase 3: Intelligent Technique Selection
├── Task 3.1: Technique performance table
├── Task 3.2: UCB1 exploration/exploitation
└── Task 3.3: Recommendation MCP tool

Phase 4: Knowledge Base Integration
├── Task 4.1: Mandatory warning checks
├── Task 4.2: Conflict resolution
└── Task 4.3: Cross-session preloading

Phase 5: Evaluation Reliability
├── Task 5.1: LPIPS lazy loading guarantee
├── Task 5.2: Quality decision tree
├── Task 5.3: Effect type evaluation registry (NEW - Gemini Feedback)
└── Task 5.4: JSON serialization fix ✅ (NEW - Gemini Feedback)

Phase 6: Session Resumption
├── Task 6.1: State schema versioning
└── Task 6.2: File existence validation

Phase 7: External Research (if time permits)
```

---

## Files to Modify/Create

### Create (New Files):
| File | Phase | Purpose | Status |
|------|-------|---------|--------|
| `agents/blender-orchestrator/tools.py` | 0.1 | MCP tool wrappers | ✅ DONE |
| `agents/blender-orchestrator/circuit_breakers.py` | 0.4 | Circuit breaker classes | ✅ DONE |
| `agents/blender-orchestrator/orchestrator.py` | 0.1 | Main orchestration class | ✅ EXISTS (enhanced) |
| `agents/blender-orchestrator/tool_executor.py` | 1.1 | Execution verification | ✅ DONE |
| `agents/blender-orchestrator/workflow_tracer.py` | 1.2 | Session tracing | ✅ DONE |
| `agents/blender-orchestrator/health_check.py` | 1.3 | Server health checks | ✅ DONE |
| `agents/blender-orchestrator/state_machine.py` | 1.4 | Formal state machine | ✅ DONE |
| `agents/script-generator/validator.py` | 2.1 | Script validation | Pending |
| `agents/script-generator/technique_selector.py` | 3.2 | UCB1 algorithm | Pending |
| `agents/asset-evaluator/decision_tree.py` | 5.2 | Quality decision tree | Pending |
| `agents/asset-evaluator/effect_evaluators.py` | 5.3 | Effect type evaluation registry | Pending |
| `agents/script-generator/effect_registry.py` | 2.5.1 | Effect type registry | Pending |
| `agents/blender-executor/execution_patterns.py` | 2.5.2 | Live render pattern | Pending |
| `agents/experiment-tracker/knowledge_base/soft_body_stability.json` | 2.5.4 | Soft body knowledge | Pending |
| `agents/iteration-controller/state_schema.py` | 6.1 | Schema versioning | Pending |

### Modify (Existing Files):
| File | Phase | Changes | Status |
|------|-------|---------|--------|
| `.claude/skills/blender-orchestrator/SKILL.md` | 0.5 | Add loop control, circuit breakers, knowledge integration, research | ✅ DONE |
| `agents/blender-orchestrator/server.py` | 0.2 | Remove claude_agent_sdk import | ✅ DONE |
| `agents/blender-orchestrator/orchestrator.py` | 0.3, 0.4 | State sync, circuit breaker integration | ✅ DONE |
| `agents/blender-orchestrator/config.yaml` | 0.4 | Add circuit breaker config | Pending |
| `agents/experiment-tracker/database.py` | 3.1 | Add technique_performance table | Pending |
| `agents/script-generator/server.py` | 3.3 | Add recommend_technique tool | Pending |
| `agents/iteration-controller/server.py` | 4.2, 6.2 | Add conflict resolution, validation | Pending |
| `agents/asset-evaluator/server.py` | 5.1 | Guarantee lazy loading | Pending |

### Deleted (Cleanup):
| File | Reason | Status |
|------|--------|--------|
| `agents/blender-orchestrator/tools/__init__.py` | Consolidated into tools.py | ✅ DONE |
| `agents/blender-orchestrator/tools/create_asset.py` | Consolidated into tools.py | ✅ DONE |
| `agents/blender-orchestrator/tools/get_status.py` | Consolidated into tools.py | ✅ DONE |
| `agents/blender-orchestrator/tools/list_sessions.py` | Consolidated into tools.py | ✅ DONE |
| `agents/blender-orchestrator/tools/resume_session.py` | Consolidated into tools.py | ✅ DONE |
| `agents/blender-orchestrator/test_mcp_simple.py` | Renamed to _DEPRECATED | ✅ DONE |

---

## Success Metrics

| Metric | Current | Target | Phase | Status |
|--------|---------|--------|-------|--------|
| Core files exist | 2/2 | 2/2 | 0 | ✅ DONE |
| SKILL.md has loop control | Yes | Yes | 0.5 | ✅ DONE |
| SKILL.md has circuit breakers | Yes | Yes | 0.5 | ✅ DONE |
| SKILL.md mandates knowledge consultation | Yes | Yes | 0.5 | ✅ DONE |
| Tool execution verified | 0% | 100% | 1 | Pending |
| Circuit breakers active | Yes | Yes | 0.4 | ✅ DONE |
| Scripts validated before execution | 0% | 100% | 2 | Pending |
| Technique selection informed by history | 0% | 100% | 3 | Pending |
| Knowledge base consulted per iteration | ~0% | 100% | 4 | Pending |
| LPIPS lazy loading guaranteed | Partial | Yes | 5 | Pending |
| Session resume success rate | Unknown | >90% | 6 | Pending |
| Average iterations to pass quality | ~3-5 | <3 | All | Pending |

---

## Testing Strategy

### Unit Tests
- Each new module gets corresponding test file
- Mock MCP client for tool wrapper tests
- Test UCB1 algorithm convergence properties

### Integration Tests
- End-to-end workflow with mock MCP servers
- Circuit breaker triggering scenarios
- State persistence and resume

### Regression Tests
- Existing functionality not broken
- Backward compatibility with saved states

### Manual Validation
- Run real asset generation after each phase
- Verify circuit breakers stop runaway loops
- Confirm knowledge base suggestions applied

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| UCB1 converges too slowly | Add minimum exploration trials (3 per technique) |
| Circuit breakers too aggressive | Make limits configurable, start conservative |
| Knowledge conflicts cause thrashing | Cap conflict resolution to 1 retry per parameter |
| State migration breaks old sessions | Keep migration tests, support last 2 schema versions |

---

## Related Documents

- `docs/MULTI_AGENT_PIPELINE_ANALYSIS.md` - Full system documentation
- `docs/MULTI_AGENT_IMPROVEMENT_PLAN.md` - Original improvement plan (v1)
- `.claude/skills/blender-orchestrator/SKILL.md` - Skill definition
- `agents/blender-orchestrator/config.yaml` - Configuration reference
