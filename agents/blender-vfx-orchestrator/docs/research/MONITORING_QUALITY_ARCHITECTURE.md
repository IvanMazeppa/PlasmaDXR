# Monitoring, Context Management, and Quality Feedback Architecture

**Author:** Monitoring & Quality Architect Agent
**Date:** 2026-02-22
**Status:** Design proposal for Phase 2+ implementation
**Depends on:** Phase 1 Reliability (completed), SDK features research (Task #1), Codebase analysis (Task #2)

---

## Executive Summary

This document designs three interconnected systems that address the top failure modes identified in the mission statement and 188+ pipeline runs:

1. **Monitoring Layer** — Deterministic + hook-based system that detects parameter oscillation, stuck loops, context bloat, wrong feedback cascades, and budget overruns
2. **Context Management** — Per-agent history trimming via `call_model_input_filter`, artifact-based sharing, and token budget allocation
3. **Quality Feedback Loop** — Multi-grader evaluation with code-grounded diagnosis that prevents overcorrection

These three systems are tightly coupled: the monitoring layer watches for quality feedback oscillation, context management determines what feedback survives between iterations, and the feedback loop architecture determines what signals the monitor observes.

---

## Part 1: Monitoring Layer

### 1.1 Design Decision: Hybrid (Deterministic Core + Optional LLM Escalation)

**Recommendation: 95% deterministic monitor, 5% LLM escalation.**

Rationale:
- Monitoring must be $0 cost per iteration (budget is $20/month, ~$0.20-0.50/run)
- Failure signals are computable (score deltas, parameter diffs, line counts, error counts)
- LLM intelligence is only needed for the 5% case: "should we switch technique or try one more parameter tweak?" — and even this can be handled by the existing QualityGateJudge
- A full LLM monitoring agent would add 1-2 API calls per iteration ($0.02-0.10), consuming 5-20% of the run budget for oversight

The monitor should be a Python class using SDK RunHooks callbacks, NOT an Agent.

### 1.2 What to Monitor

| Signal | Detection Method | Threshold | Action |
|--------|-----------------|-----------|--------|
| **Parameter oscillation** | Track param values across iterations; detect sign-alternating deltas | Same param changes direction 2x in 3 iterations | Clamp to bounded midpoint; flag for review |
| **Stuck loop (same error)** | Count consecutive identical `primary_issue` | 3x same issue | Force technique switch (escape velocity L2) |
| **Score plateau** | `abs(score_delta) < 3.0` for N iterations | 3 consecutive | Escalate escape velocity |
| **Context bloat** | Sum `len(str(item))` for conversation history | >100K chars (existing) | Trigger compaction |
| **Wrong feedback cascade** | QA suggests fix → score drops after applying it | Score decreases by >5 after applying QA suggestion | Override QA; revert to previous params |
| **Budget overrun** | Track `session_cost` vs per-run budget | >$0.50 per run | Disable expensive tools; force early termination |
| **Script quality (line count)** | `wc -l` on generated script | <500 lines | Warn; inject "scene too basic" feedback |
| **Render quality (critical)** | Check for BLACK_SCREEN, WHITE_SCREEN, ZERO_LIGHTS in QA output | Any critical issue | Skip normal iteration; route to targeted fix |
| **Agent turn overrun** | Count turns per agent per call | >2x target turns | Force output production (existing EnforcementHooks) |
| **Tool cost spike** | Track which tools are called and their cost | Vision tool called >2x in one iteration | Throttle; use cached result |

### 1.3 Architecture: `PipelineMonitor` Class

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

class MonitorAlert(Enum):
    """Alert severity levels."""
    INFO = "info"           # Logged, no action
    WARNING = "warning"     # Logged, may adjust parameters
    CRITICAL = "critical"   # Action required: clamp, switch, pause
    FATAL = "fatal"         # Pipeline must stop

@dataclass
class ParameterHistory:
    """Track a single parameter's value across iterations."""
    name: str
    values: List[float] = field(default_factory=list)
    iterations: List[int] = field(default_factory=list)

    def is_oscillating(self) -> bool:
        """Detect sign-alternating deltas (e.g., 50 -> 2500 -> 10)."""
        if len(self.values) < 3:
            return False
        deltas = [self.values[i+1] - self.values[i] for i in range(len(self.values)-1)]
        # Check last 3 deltas for alternating signs
        recent = deltas[-3:] if len(deltas) >= 3 else deltas[-2:]
        if len(recent) < 2:
            return False
        for i in range(len(recent) - 1):
            if recent[i] * recent[i+1] >= 0:  # Same sign
                return False
        return True

    def get_bounded_midpoint(self) -> float:
        """Return the midpoint of the last 3 values as a stable target."""
        recent = self.values[-3:]
        return sum(recent) / len(recent)

@dataclass
class MonitorState:
    """Accumulated monitoring state across iterations."""
    # Parameter tracking
    param_history: Dict[str, ParameterHistory] = field(default_factory=dict)

    # Score tracking
    scores: List[float] = field(default_factory=list)
    score_deltas: List[float] = field(default_factory=list)

    # Issue tracking
    primary_issues: List[str] = field(default_factory=list)
    qa_suggestions_applied: List[Dict] = field(default_factory=list)

    # Cost tracking
    iteration_costs: List[float] = field(default_factory=list)
    total_cost: float = 0.0

    # Alert log
    alerts: List[Tuple[int, MonitorAlert, str]] = field(default_factory=list)


class PipelineMonitor:
    """
    Deterministic pipeline monitor.

    Runs between pipeline phases (not as RunHooks — see 1.4 for why).
    Called by the orchestrator at specific checkpoints.
    """

    def __init__(self, budget_limit: float = 0.50, score_threshold: float = 60.0):
        self.state = MonitorState()
        self.budget_limit = budget_limit
        self.score_threshold = score_threshold

    def check_after_generation(self, script_path: str, iteration: int) -> List[Tuple[MonitorAlert, str]]:
        """Check script quality after generation."""
        alerts = []
        line_count = sum(1 for _ in open(script_path))
        if line_count < 500:
            alerts.append((
                MonitorAlert.WARNING,
                f"Script too basic: {line_count} lines (<500). "
                "Scene likely lacks environmental detail, lighting variety, or material complexity."
            ))
        return alerts

    def check_after_evaluation(
        self,
        iteration: int,
        score: float,
        primary_issue: str,
        issues: List[str],
        params_used: Dict[str, float],
        qa_suggestions: List[str],
    ) -> List[Tuple[MonitorAlert, str, Optional[Dict]]]:
        """
        Check for problems after quality evaluation.

        Returns list of (alert_level, message, action_data) tuples.
        action_data may contain:
          - {"clamp": {param: value}} for oscillation fixes
          - {"switch_technique": True} for stuck detection
          - {"revert_params": {param: value}} for cascade detection
        """
        alerts = []

        # Track score
        if self.state.scores:
            delta = score - self.state.scores[-1]
            self.state.score_deltas.append(delta)
        self.state.scores.append(score)
        self.state.primary_issues.append(primary_issue)

        # 1. Parameter oscillation detection
        for name, value in params_used.items():
            if name not in self.state.param_history:
                self.state.param_history[name] = ParameterHistory(name=name)
            ph = self.state.param_history[name]
            ph.values.append(value)
            ph.iterations.append(iteration)

            if ph.is_oscillating():
                midpoint = ph.get_bounded_midpoint()
                alerts.append((
                    MonitorAlert.CRITICAL,
                    f"OSCILLATION: {name} = {ph.values[-3:]} -> clamping to {midpoint:.2f}",
                    {"clamp": {name: midpoint}}
                ))

        # 2. Wrong feedback cascade detection
        if len(self.state.score_deltas) >= 1 and self.state.score_deltas[-1] < -5.0:
            # Score dropped significantly — was a QA suggestion applied last iteration?
            if self.state.qa_suggestions_applied:
                last_suggestion = self.state.qa_suggestions_applied[-1]
                alerts.append((
                    MonitorAlert.WARNING,
                    f"FEEDBACK CASCADE: Score dropped {self.state.score_deltas[-1]:+.1f} "
                    f"after applying QA suggestion: {last_suggestion.get('suggestion', 'unknown')[:60]}",
                    {"revert_params": last_suggestion.get("params_before", {})}
                ))

        # 3. Critical render issues (immediate action)
        critical_keywords = ["ZERO_LIGHTS", "BLACK_SCREEN", "WHITE_SCREEN", "CLIPPING"]
        for issue in issues:
            if any(kw in issue.upper() for kw in critical_keywords):
                alerts.append((
                    MonitorAlert.CRITICAL,
                    f"CRITICAL RENDER ISSUE: {issue}",
                    None
                ))

        # 4. Budget check
        if self.state.total_cost > self.budget_limit:
            alerts.append((
                MonitorAlert.CRITICAL,
                f"BUDGET EXCEEDED: ${self.state.total_cost:.2f} > ${self.budget_limit:.2f}",
                None
            ))

        # Record QA suggestions for next iteration's cascade check
        self.state.qa_suggestions_applied.append({
            "iteration": iteration,
            "suggestion": primary_issue,
            "params_before": params_used.copy(),
        })

        # Log alerts
        for alert in alerts:
            self.state.alerts.append((iteration, alert[0], alert[1]))

        return alerts

    def get_parameter_bounds(self, param_name: str) -> Optional[Tuple[float, float]]:
        """
        Get safe bounds for a parameter based on history.

        If oscillation detected, returns (lower, upper) that prevents overshoot.
        """
        if param_name not in self.state.param_history:
            return None
        ph = self.state.param_history[param_name]
        if len(ph.values) < 2:
            return None
        # Use min/max of observed values as bounds, with 20% margin
        lo, hi = min(ph.values), max(ph.values)
        margin = (hi - lo) * 0.2
        return (lo + margin, hi - margin)

    def get_status_report(self) -> str:
        """Dense status report for artifact file."""
        lines = ["## Pipeline Monitor Status"]
        lines.append(f"Iterations: {len(self.state.scores)}")
        lines.append(f"Scores: {self.state.scores}")
        lines.append(f"Cost: ${self.state.total_cost:.2f}")

        oscillating = [
            name for name, ph in self.state.param_history.items()
            if ph.is_oscillating()
        ]
        if oscillating:
            lines.append(f"OSCILLATING PARAMS: {oscillating}")

        critical_alerts = [
            (it, msg) for it, level, msg in self.state.alerts
            if level in (MonitorAlert.CRITICAL, MonitorAlert.FATAL)
        ]
        if critical_alerts:
            lines.append(f"Critical alerts: {len(critical_alerts)}")
            for it, msg in critical_alerts[-3:]:
                lines.append(f"  iter {it}: {msg}")

        return "\n".join(lines)
```

### 1.4 Integration Strategy: Orchestrator Checkpoints, NOT RunHooks

**Why NOT RunHooks for the primary monitor:**

The existing `EnforcementHooks` and `DiagnosticHooks` operate at the **intra-agent** level (tool calls within a single agent run). The PipelineMonitor needs to operate at the **inter-iteration** level (comparing state across iterations). RunHooks cannot see across Runner.run() invocations.

RunHooks remain appropriate for:
- Loop detection within agents (existing `EnforcementHooks`)
- Turn budget enforcement (existing)
- Tool call counting and cost tracking per agent

**The PipelineMonitor integrates at orchestrator checkpoints:**

```python
# In create_asset_pipeline():

monitor = PipelineMonitor(budget_limit=0.50)

for iteration in range(1, max_iterations + 1):
    # ... GENERATE phase ...

    # Checkpoint 1: After generation
    gen_alerts = monitor.check_after_generation(script.script_path, iteration)
    for alert_level, message in gen_alerts:
        if alert_level == MonitorAlert.WARNING:
            # Inject warning into modification prompt
            context.monitoring_warnings.append(message)

    # ... EXECUTE + EVALUATE phases ...

    # Checkpoint 2: After evaluation (most important)
    eval_alerts = monitor.check_after_evaluation(
        iteration=iteration,
        score=quality.overall_score,
        primary_issue=quality.primary_issue or "",
        issues=quality.issues,
        params_used=extract_params_from_script(script.script_path),
        qa_suggestions=quality.suggestions,
    )

    for alert_level, message, action_data in eval_alerts:
        if alert_level == MonitorAlert.CRITICAL and action_data:
            if "clamp" in action_data:
                # Override parameter suggestions with clamped values
                for param, value in action_data["clamp"].items():
                    context.parameter_overrides[param] = value
            if "switch_technique" in action_data:
                session.stuck_state.escape_level = EscapeLevel.SWITCH_TECHNIQUE
            if "revert_params" in action_data:
                context.parameter_overrides.update(action_data["revert_params"])

    # Checkpoint 3: Write status artifact
    monitor_artifact = artifact_mgr.write_monitor_status(
        iteration, monitor.get_status_report()
    )
```

### 1.5 Integration with Existing Escape Velocity

The monitor and escape velocity system serve complementary roles:

| System | Scope | Trigger | Action |
|--------|-------|---------|--------|
| **Escape Velocity** (existing) | Same-issue persistence, score plateau | After N iterations with same pattern | Escalate: L0→L1→L2→L3→L4 |
| **PipelineMonitor** (new) | Parameter oscillation, feedback cascades, budget, script quality | After each checkpoint | Clamp, revert, warn, pause |

**They should share data but not duplicate logic:**
- Monitor feeds `StuckDetectionState` with additional signals (oscillation counts as "same issue")
- Escape velocity remains the decision-maker for technique switching
- Monitor adds parameter clamping (escape velocity doesn't do this)
- Monitor detects feedback cascades (escape velocity doesn't track QA suggestion→outcome)

```python
# Monitor can escalate escape velocity:
if monitor.detected_oscillation_count >= 2:
    session.stuck_state.same_issue_count = max(
        session.stuck_state.same_issue_count,
        2  # Force at least L2 consideration
    )
```

### 1.6 RunHooks Enhancement: Per-Agent Cost Tracking

While the main monitor doesn't use RunHooks, we should **enhance the existing hooks** to track cost:

```python
class CostTrackingHooks(RunHooks):
    """Track token usage per agent for budget allocation."""

    def __init__(self):
        self.agent_costs: Dict[str, float] = {}
        self._current_agent: Optional[str] = None

    async def on_agent_start(self, context, agent):
        self._current_agent = agent.name
        if agent.name not in self.agent_costs:
            self.agent_costs[agent.name] = 0.0

    async def on_tool_end(self, context, agent, tool, result):
        # Track vision API calls (most expensive)
        if tool.name in ("analyze_with_vision", "evaluate_render", "compare_renders"):
            self.agent_costs[agent.name] = self.agent_costs.get(agent.name, 0) + 0.05

    def get_total_cost(self) -> float:
        return sum(self.agent_costs.values())
```

---

## Part 2: Context Management Strategy

### 2.1 The Problem

Context bloat is a top-3 failure mode (Mission Statement Section 12). Evidence:
- Agent output is verbose human-oriented prose, not LLM-optimized
- Quality evaluation can produce 500+ tokens of feedback
- Research agent output can be 1000+ tokens
- By iteration 3+, context windows fill with stale iteration history
- Session compaction was **completely disabled** (`lambda _: False`) until Phase 1 fixed it

Phase 1 enabled compaction at ~25K tokens (~100K chars). But compaction is a blunt instrument — it summarizes everything equally. What's needed is **selective trimming per agent type**.

### 2.2 call_model_input_filter Design

The SDK's `call_model_input_filter` is a function on the Agent that transforms conversation history before each LLM call. This is the precision tool for context management.

**Signature:**
```python
def my_filter(
    context: RunContextWrapper[SharedContext],
    items: list[TResponseInputItem]
) -> list[TResponseInputItem]:
    """Transform conversation history before LLM call."""
    return filtered_items
```

**Per-Agent Trimming Strategy:**

| Agent | What to Keep | What to Trim | Token Budget |
|-------|-------------|-------------|--------------|
| **ScriptWriter** | Current research, truth pack, last QA feedback, monitor warnings | All previous iteration details, old research, old QA feedback | ~8K tokens input |
| **QualityAnalyst** | Current render path, script parameter summary (top 15), scoring criteria | All previous evaluations, research, script generation history | ~4K tokens input |
| **LearningAgent** | Current score + delta, script analysis, KB query results, last 2 iteration summaries | Full research, full script text, old evaluation details | ~6K tokens input |
| **ResearchAgent** | Effect type, user prompt, KB learnings, alternative approaches | All iteration history (research runs once at start) | ~6K tokens input |
| **ModificationStrategist** | Code-grounded feedback, score history (last 3), monitor alerts, parameter bounds | Full evaluation text, full research, old script details | ~4K tokens input |

**Implementation:**

```python
def create_script_writer_filter():
    """Keep only what ScriptWriter needs: research + feedback + truth pack."""
    def _filter(context, items):
        # Keep system message (instructions + truth pack)
        # Keep the last user message (the prompt with research + QA feedback)
        # Drop everything else (old tool calls, old responses)
        if len(items) <= 2:
            return items
        # Keep first (system) and last (current prompt)
        return [items[0], items[-1]]
    return _filter

def create_quality_analyst_filter():
    """Keep only what QA needs: render path + script params."""
    def _filter(context, items):
        # QA should NOT see previous QA results (prevents self-reinforcing bias)
        # Keep system message + current evaluation prompt only
        if len(items) <= 2:
            return items
        return [items[0], items[-1]]
    return _filter

def create_learning_agent_filter():
    """Keep recent context: last 2 iterations of summaries."""
    def _filter(context, items):
        if len(items) <= 4:
            return items
        # Keep system + last 3 messages (current prompt + up to 2 previous turns)
        return [items[0]] + items[-3:]
    return _filter
```

**Applying to agents:**

```python
script_writer = Agent(
    name="Script Writer",
    instructions=dynamic_script_writer_instructions,
    call_model_input_filter=create_script_writer_filter(),
    # ... tools, model, etc.
)
```

### 2.3 Artifact-Based Info Sharing (Extend Existing Pattern)

The orchestrator already uses `ArtifactManager` for file-based handoffs. This pattern should be deepened:

**Current state:**
- Research output → `research_iter0.md`
- Quality output → `quality_iter{N}.md`
- Scorecard → `scorecard_iter{N}.md`

**Proposed additions:**

| Artifact | Content | Written By | Read By |
|----------|---------|-----------|---------|
| `monitor_iter{N}.md` | Parameter bounds, oscillation alerts, cascade warnings | PipelineMonitor | ModificationStrategist, ScriptWriter |
| `params_iter{N}.json` | All extracted parameter values with line numbers | Script analysis (deterministic) | Monitor, LearningAgent, ModificationStrategist |
| `diagnosis_iter{N}.md` | Code-grounded feedback (QA bridge output) | QA Diagnosis Bridge | ModificationStrategist, ScriptWriter |
| `iteration_summary.md` | Rolling 3-iteration summary (overwritten each iter) | Orchestrator | All agents on next iteration |

**Key principle:** Agents receive **file paths** in their prompts, not inline data. They read what they need via tools.

```python
# Instead of:
prompt = f"""... Score: {quality.overall_score}
Issues: {quality.issues}
Suggestions: {quality.suggestions}
Vision Assessment: {quality.vision_assessment}
Code Grounded: {code_grounded_feedback}
Monitor Alerts: {monitor_alerts}
..."""  # 2000+ tokens

# Use:
prompt = f"""Modify script to fix quality issues.
Script: {script.script_path}
Quality: {quality_artifact_path}
Diagnosis: {diagnosis_artifact_path}
Monitor: {monitor_artifact_path}
Iteration summary: {summary_artifact_path}
Score: {quality.overall_score:.1f} | Issue: {quality.primary_issue}"""  # ~200 tokens
```

**Agents that need to read artifacts get a `read_artifact` tool:**

```python
@function_tool
def read_artifact(artifact_path: str) -> str:
    """Read a pipeline artifact file. Use for detailed context when needed."""
    path = Path(artifact_path)
    if not path.exists():
        return f"Artifact not found: {artifact_path}"
    return path.read_text()[:4000]  # Cap at 4K chars
```

This puts the agent in control of its own context budget — it reads details only when relevant to its task.

### 2.4 Token Budget Allocation Per Iteration

**Target: $0.20-0.50 per 3-iteration run = ~$0.07-0.17 per iteration.**

At gpt-5.2 pricing (~$5/M input, ~$15/M output), this means:
- ~10K-30K input tokens per iteration across all agents
- ~2K-5K output tokens per iteration

**Budget allocation per agent per iteration:**

| Agent | Input Budget | Output Budget | Model | Notes |
|-------|-------------|---------------|-------|-------|
| ScriptWriter | 8K tokens | 3K tokens | gpt-5.2 | Largest budget — creative code generation |
| QualityAnalyst | 4K tokens | 500 tokens | gpt-5-mini | Vision API is separate cost; structured output is small |
| LearningAgent | 4K tokens | 500 tokens | gpt-5-mini | KB queries + experiment recording |
| ModificationStrategist | 3K tokens | 500 tokens | o4-mini | Quick structured decision |
| QualityGateJudge | 2K tokens | 200 tokens | o4-mini | Pass/fail decision with action |
| **Total** | **~21K tokens** | **~4.7K tokens** | | **~$0.18/iteration** |

**Enforcement:** The `call_model_input_filter` handles input budgets. Output budgets are handled by `max_turns` limits on Runner.run().

### 2.5 Integration with Phase 1 Session Compaction

Phase 1 enabled compaction at 100K chars. With `call_model_input_filter` in place, compaction becomes a **safety net** rather than the primary mechanism:

- `call_model_input_filter` handles per-agent trimming (precision)
- Session compaction handles the overall session history if it somehow grows past the threshold (safety net)
- SharedContext (truth pack, session state) survives both mechanisms (Python-managed)

**Compaction should be tightened:**
```python
# Current: 100K chars (~25K tokens)
# Proposed: 60K chars (~15K tokens) — earlier trigger since filters handle per-agent
should_trigger_compaction=lambda history: sum(
    len(str(item)) for item in (history or [])
) > 60_000
```

### 2.6 What Survives Compaction (SharedContext Fields)

These fields are Python-managed and always available regardless of compaction:

| Field | Survives | Notes |
|-------|----------|-------|
| `truth_pack` | Yes | Built from Blender introspection, injected into prompts |
| `session.stuck_state` | Yes | Escape velocity state |
| `session.iterations` | Yes | Full iteration history (Pydantic models) |
| `session.current_script_path` | Yes | For code-grounded feedback |
| `session.research_text` | Yes | For ScriptWriter context |
| `api_spec` | Yes | For code validation |
| Conversation history | Compacted | Summarized by SDK compaction |
| Artifact files on disk | Yes | Files persist independently |

---

## Part 3: Quality Feedback Loop Architecture

### 3.1 Current State (Post-Phase 1)

Phase 1 added `qa_diagnosis_bridge.py` which pairs QA visual critique with script code analysis:
- QA says "too dark" → Bridge finds `Line 245: light.energy = 10 (range [0, 1000000])`
- 12 issue keyword mappings (dark, dim, smoke, fire, density, camera, liquid, etc.)
- Outputs dense, LLM-optimized feedback with line numbers

**Remaining problems:**
1. QA still makes suggestions that the bridge may not catch (unmapped keywords)
2. No protection against overcorrection (oscillating parameter changes)
3. QA can reinforce its own previous wrong assessment (sees previous output in context)
4. Multi-grader evaluation is incomplete (vision + some ML, but no deterministic checks)

### 3.2 Multi-Grader Evaluation Architecture

**Three-tier evaluation, run in order of cost:**

```
Tier 1: DETERMINISTIC CHECKS ($0)
  ├── Script line count (< 500 = warning)
  ├── Artifact gates (render exists? correct size? not blank?)
  ├── Light count check (< 3 = warning)
  ├── Camera bounds check (inside geometry? too close?)
  └── Known critical patterns (ZERO_LIGHTS, BLACK_SCREEN)

Tier 2: ML METRICS ($0 — local models)
  ├── TOPIQ aesthetic score
  ├── LPIPS perceptual quality (if reference available)
  ├── SigLIP semantic match (does it look like the description?)
  └── DINOv2 structural quality

Tier 3: LLM VISION ($0.01-0.05 — API call)
  ├── OpenAI vision model analyzes render
  ├── Composition, lighting, effect quality assessment
  ├── Identification of specific visual problems
  └── Suggestions (treated as SYMPTOMS, not diagnoses)
```

**Short-circuit logic:**
- If Tier 1 finds a critical issue (no render, blank image), skip Tiers 2 and 3
- If Tier 2 score is >80 with no critical issues, Tier 3 can be skipped (save $0.05)
- Tier 3 is always run on the first iteration (baseline assessment)

**Combined scoring:**

```python
def compute_combined_score(
    deterministic: DeterministicCheckResult,
    ml_metrics: MLMetricsResult,
    vision: Optional[VisionResult],
) -> Tuple[float, bool, List[str]]:
    """
    Combine multi-grader results into a single score.

    Weights:
    - Deterministic: pass/fail gates (no score contribution, but can auto-fail)
    - ML metrics: 60% of score (objective, reproducible)
    - Vision: 40% of score (subjective but valuable)

    If vision unavailable, ML metrics = 100% of score.
    """
    # Auto-fail on deterministic critical issues
    if deterministic.has_critical_issues:
        return 0.0, False, deterministic.critical_issues

    # ML score (0-100)
    ml_score = ml_metrics.composite_score  # Already 0-100

    if vision:
        vision_score = vision.overall_score  # 0-100
        combined = ml_score * 0.6 + vision_score * 0.4
    else:
        combined = ml_score

    # Apply deterministic warnings as score penalties
    penalty = len(deterministic.warnings) * 2.0  # -2 per warning
    combined = max(0, combined - penalty)

    passed = combined >= 60.0 and not deterministic.has_critical_issues
    issues = deterministic.all_issues + (vision.issues if vision else [])

    return combined, passed, issues
```

### 3.3 Preventing Overcorrection: Bounded Modification Strategy

The #1 quality feedback failure mode is **parameter oscillation**: QA says "too dark" → increase light energy 50→500 → QA says "overexposed" → decrease 500→20 → QA says "too dark" again.

**Solution: Parameter clamping with damped convergence.**

```python
@dataclass
class ParameterBound:
    """Safe range for a parameter, refined over iterations."""
    name: str
    min_value: float
    max_value: float
    step_size: float  # Maximum change per iteration

    def clamp(self, proposed: float) -> float:
        """Clamp to safe range."""
        return max(self.min_value, min(self.max_value, proposed))

    def damped_change(self, current: float, target: float) -> float:
        """Apply damped change — move at most step_size toward target."""
        delta = target - current
        if abs(delta) > self.step_size:
            delta = self.step_size if delta > 0 else -self.step_size
        return self.clamp(current + delta)


# Per-effect-type parameter bounds
PARAMETER_BOUNDS: Dict[str, Dict[str, ParameterBound]] = {
    "fire": {
        "energy": ParameterBound("energy", 20.0, 200.0, step_size=50.0),
        "density": ParameterBound("density", 1.0, 15.0, step_size=3.0),
        "blackbody_intensity": ParameterBound("blackbody_intensity", 0.5, 10.0, step_size=2.0),
        "temperature": ParameterBound("temperature", -5.0, 10.0, step_size=2.0),
    },
    "liquid": {
        "energy": ParameterBound("energy", 30.0, 300.0, step_size=80.0),
        "viscosity_base": ParameterBound("viscosity_base", 0.0, 5.0, step_size=1.0),
        "resolution_max": ParameterBound("resolution_max", 64, 256, step_size=32),
    },
    # ... other effect types
}
```

**Integration with modification pipeline:**

```python
# In the modification phase:
proposed_changes = learning.parameter_modifications  # From LearningAgent

# 1. Get bounds for current effect type
bounds = PARAMETER_BOUNDS.get(effect_type, {})

# 2. Check monitor for oscillation
monitor_bounds = monitor.get_parameter_bounds(param_name)

# 3. Apply most restrictive bound
for param, proposed_value in proposed_changes.items():
    if param in bounds:
        # Damped change
        current = extract_current_value(script_path, param)
        clamped = bounds[param].damped_change(current, proposed_value)

        # Additional monitor restriction if oscillating
        if monitor_bounds:
            clamped = max(monitor_bounds[0], min(monitor_bounds[1], clamped))

        proposed_changes[param] = clamped
```

### 3.4 QA Isolation: Preventing Self-Reinforcing Bias

**Problem:** If the QA agent sees its previous evaluation in the conversation history, it tends to repeat and reinforce its previous assessment rather than evaluating fresh.

**Solution:** `call_model_input_filter` for QA (Section 2.2) already handles this — QA only sees the system prompt and current evaluation request. No previous QA results leak into its context.

**Additional safeguard:** The QA prompt should explicitly state:

```
Evaluate this render independently. Do NOT reference previous evaluations.
Base your assessment ONLY on what you see in this image and the scoring criteria.
```

### 3.5 Improved QA → Diagnosis → Modification Pipeline

**Current flow (Phase 1):**
```
QA Evaluation → Code-Grounded Feedback → Modification Strategist → Script Writer
```

**Proposed flow (Phase 2):**
```
Tier 1 Deterministic ──┐
                        ├── Combined Score + Issues ──→ Code-Grounded Diagnosis
Tier 2 ML Metrics ──────┤                                      │
                        │                               Monitor Alerts
Tier 3 LLM Vision ─────┘                                      │
                                                        Parameter Bounds
                                                               │
                                                    ┌──────────▼──────────┐
                                                    │ Modification Decision │
                                                    │  (bounded, damped)    │
                                                    └──────────┬──────────┘
                                                               │
                                                    ┌──────────▼──────────┐
                                                    │    Script Writer     │
                                                    │ (with clamped params)│
                                                    └─────────────────────┘
```

**Key improvements:**
1. **Deterministic checks first** — catch critical issues at $0 before spending on vision
2. **Code-grounded diagnosis** — every visual issue is paired with actual script parameters (existing QA bridge, extended)
3. **Monitor alerts injected** — oscillation warnings, cascade warnings, parameter bounds
4. **Bounded modifications** — all parameter changes are clamped and damped
5. **QA isolation** — QA evaluates fresh each iteration, no self-reinforcing bias

### 3.6 Extending the QA Diagnosis Bridge

The existing `qa_diagnosis_bridge.py` maps 12 keywords. Extend it with:

**a) Automatic parameter extraction (no keyword needed):**

```python
def extract_all_modifiable_params(script_path: str) -> Dict[str, Dict]:
    """
    Extract every modifiable parameter with line number, current value, and type.

    No issue matching — just provides the complete parameter inventory.
    Used by the monitor for tracking and by the modification pipeline for bounds.
    """
    analysis = analyze_script_structure(script_path)
    params = {}
    for name, pattern in analysis.settings_assignments.items():
        params[name] = {
            "line": pattern.line_number,
            "value": pattern.current_value,
            "type": pattern.pattern_type,
        }
    # ... same for config_values, shader_node_inputs
    return params
```

**b) Truth-pack-enhanced ranges:**

When the truth pack is available, every parameter in the diagnosis should include its valid range:
```
Line 245: domain_settings.resolution_max = 64 [range: 1-10000, default: 32]
Line 312: light.energy = 50 [range: 0-1000000, default: 10]
```

This is already partially implemented (`_format_param_with_truth_pack`) — ensure it's always used.

**c) Delta feedback from monitor:**

```
Line 312: light.energy = 50 [range: 0-1000000]
  MONITOR: Changed 3x in last 3 iterations (20 → 200 → 50). OSCILLATING.
  BOUND: Next change limited to [35, 65] (damped convergence).
```

---

## Part 4: Implementation Roadmap

### Phase 2a: Core Monitoring (Estimated: 2-3 sessions)

1. Implement `PipelineMonitor` class in `tools/pipeline_monitor.py`
2. Add checkpoint calls in `orchestrator.py` (after generation, after evaluation)
3. Add `monitor_iter{N}.md` artifact writing
4. Wire parameter bounds into modification pipeline
5. Test: Verify oscillation detection with synthetic parameter histories

### Phase 2b: Context Management (Estimated: 2-3 sessions)

1. Implement `call_model_input_filter` for ScriptWriter, QualityAnalyst, LearningAgent
2. Add `read_artifact` tool to agents that need detailed context
3. Convert inline data in prompts to artifact references
4. Tighten compaction threshold to 60K chars
5. Test: Verify agent prompts contain only relevant context

### Phase 2c: Multi-Grader Quality (Estimated: 1-2 sessions)

1. Extract deterministic checks into `tools/deterministic_quality_checks.py`
2. Implement short-circuit logic (critical issue → skip vision)
3. Implement combined scoring with weights
4. Add parameter bounds per effect type
5. Test: Verify scoring consistency and cost savings

### Phase 2d: Integration & Testing (Estimated: 1-2 sessions)

1. Run E2E with all three systems active
2. Verify: No parameter oscillation across 5-iteration run
3. Verify: Context stays within budget per agent
4. Verify: Monitor catches critical issues and prevents cascades
5. Tune thresholds based on real pipeline behavior

---

## Appendix A: SDK Feature Dependencies

| Feature | Used In | Status |
|---------|---------|--------|
| `RunHooks.on_tool_start/end` | Cost tracking hooks | Existing, extend |
| `call_model_input_filter` | Per-agent context trimming | New, SDK-native |
| `is_enabled` (conditional tools) | Hide expensive tools when budget low | New, SDK-native |
| `tool_use_behavior: stop_on_first_tool` | QualityGateJudge (deterministic routing) | New, SDK-native |
| `OpenAIResponsesCompactionSession` | Session compaction (safety net) | Existing, tune threshold |

## Appendix B: File Inventory (New Files)

| File | Purpose | Lines (est) |
|------|---------|-------------|
| `tools/pipeline_monitor.py` | PipelineMonitor class | ~300 |
| `tools/deterministic_quality_checks.py` | Tier 1 checks | ~200 |
| `tools/parameter_bounds.py` | Per-effect-type bounds + damped changes | ~150 |
| `tools/context_filters.py` | call_model_input_filter implementations | ~100 |

**Total estimated new code: ~750 lines.**

## Appendix C: Interaction Diagram

```
                    ITERATION N
                         │
    ┌────────────────────▼────────────────────┐
    │           SCRIPT GENERATION              │
    │  (ScriptWriter with call_model_filter)   │
    │  Context: research + truth pack + bounds  │
    └────────────────────┬────────────────────┘
                         │
    ┌────────────────────▼────────────────────┐
    │      MONITOR CHECKPOINT 1: POST-GEN     │
    │  Check: line count, known anti-patterns  │
    └────────────────────┬────────────────────┘
                         │
    ┌────────────────────▼────────────────────┐
    │           EXECUTION (Blender)            │
    │  (Deterministic subprocess)              │
    └────────────────────┬────────────────────┘
                         │
    ┌────────────────────▼────────────────────┐
    │      TIER 1: DETERMINISTIC CHECKS       │
    │  render exists? blank? lights? camera?   │
    │  Cost: $0 | Short-circuit on critical    │
    └────────────────────┬────────────────────┘
                         │
    ┌────────────────────▼────────────────────┐
    │        TIER 2: ML METRICS               │
    │  TOPIQ, LPIPS, SigLIP, DINOv2           │
    │  Cost: $0 (local) | Objective scores    │
    └────────────────────┬────────────────────┘
                         │
    ┌────────────────────▼────────────────────┐
    │        TIER 3: LLM VISION               │
    │  OpenAI vision critique                  │
    │  Cost: $0.01-0.05 | Skip if Tier2 > 80  │
    └────────────────────┬────────────────────┘
                         │
    ┌────────────────────▼────────────────────┐
    │     COMBINED SCORING + DIAGNOSIS        │
    │  QA Bridge: visual issues → code params  │
    │  Cost: $0 (deterministic)                │
    └────────────────────┬────────────────────┘
                         │
    ┌────────────────────▼────────────────────┐
    │   MONITOR CHECKPOINT 2: POST-EVAL       │
    │  oscillation? cascade? budget? stuck?    │
    │  → clamp params, warn, escalate          │
    └────────────────────┬────────────────────┘
                         │
    ┌────────────────────▼────────────────────┐
    │    LEARNING + QUALITY GATE (parallel)    │
    │  Record experiment, extract patterns     │
    │  Pass/fail decision with bounded params  │
    └────────────────────┬────────────────────┘
                         │
                    ITERATION N+1
                    (or COMPLETE)
```
