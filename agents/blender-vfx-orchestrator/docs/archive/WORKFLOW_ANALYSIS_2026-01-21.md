# Workflow Analysis: Critical Problems & Solutions

**Date:** 2026-01-21
**Status:** Analysis Complete, Implementation Pending
**Context:** After multiple E2E tests showing identical scripts despite iteration

---

## Executive Summary

The self-learning architecture (Phase 1-4 in SELF_LEARNING_ARCHITECTURE.md) removed hardcoded physics rules but didn't establish a working feedback loop. The system generates scripts, evaluates them, but **never actually modifies parameters** because the learning chain is broken at multiple points.

---

## Critical Problems Identified

### Problem 1: `record_baseline()` Never Called

**Location:** `orchestrator.py:615` says "handled elsewhere" but it's never called.

**Evidence:**
```python
# Learning Agent instructions say:
- DO NOT call start_experiment_session or record_baseline (handled elsewhere)

# But orchestrator never calls it either!
```

**Impact:** `record_experiment_result()` fails or produces meaningless comparisons without a baseline.

**Fix:** Orchestrator must call `record_baseline()` before iteration 2+.

---

### Problem 2: Research Agent Only Runs Once

**Location:** `orchestrator.py:741-759` (PHASE 0)

**Evidence:**
```python
# When switch_technique is recommended:
if learning.next_action == 'switch_technique':
    print(f"[Pipeline] Learning Agent recommends switching technique", file=sys.stderr)
    # Research agent already found alternatives in research_text
    # Next iteration will get modified feedback to try different approach
```

**Impact:** System can't explore new ideas. The "alternatives" from initial research may not address the actual problems discovered during iteration.

**Fix:** Re-run Research Agent when `escape_level >= 2` or `switch_technique` recommended.

---

### Problem 3: Learning Agent Doesn't Output Concrete Parameters

**Location:** `LearningOutput` schema and Learning Agent instructions

**Evidence:**
- Schema has `parameter_modifications: Dict[str, Any]`
- Instructions say to include concrete values
- But instructions don't explain HOW to derive values from quality issues
- Learning Agent outputs empty `{}` for parameter_modifications

**Impact:** Direct modification code (`orchestrator.py:803-833`) never triggers because `learning.parameter_modifications` is always empty.

**Fix:** Add Issue Resolver agent that maps quality issues → parameter changes using KB.

---

### Problem 4: No Iteration Context Passed to Agents

**Location:** All agent prompts in orchestrator

**Evidence:**
```python
# Script Writer prompt for iteration > 1:
script_prompt = f"""Modify the existing script to fix quality issues.
## Current Script
Path: {previous_script.script_path}
...
"""
# No history of what was tried, what failed, why
```

**Impact:** Agents repeat the same mistakes. No learning accumulation within a session.

**Fix:** Pass iteration history summary to agents:
- What parameters were tried
- What scores resulted
- What issues persisted

---

### Problem 5: Tools Return JSON Strings, Agents Output Prose

**Location:** All `@function_tool` definitions

**Evidence:**
```python
@function_tool
async def record_experiment_result(...) -> str:
    """..."""
    return json.dumps({"success": True, ...})  # JSON string

# Agent then describes this in prose instead of forwarding structured data
```

**SDK Pattern:** Tools should return `ToolOutputText` or structured objects that flow through without prose interpretation.

**Impact:** Information loss at every tool boundary. Agents "summarize" instead of forwarding.

**Fix:** Use SDK's structured tool outputs and `tool_use_behavior="stop_on_first_tool"` where appropriate.

---

### Problem 6: No Lifecycle Hooks for Auto-Recording

**Location:** Missing from architecture

**SDK Pattern:**
```python
class ExperimentHooks(AgentHooksBase):
    async def on_tool_end(self, ctx, agent, tool, result):
        if tool.name == "modify_script":
            self.session_manager.record_modification(tool.arguments, result)
```

**Impact:** Recording depends on agent explicitly calling tools. Agents skip recording to save turn budget.

**Fix:** Add lifecycle hooks that automatically record tool outcomes.

---

### Problem 7: Direct Modification Bypasses Learning

**Location:** `orchestrator.py:803-833`

**Evidence:**
```python
if learning and learning.parameter_modifications and previous_script:
    # Call _modify_script_impl directly
```

**Impact:** Good idea, but `learning.parameter_modifications` is always empty (see Problem 3).

**Fix:** This code is correct but needs Problem 3 fixed first.

---

## Proposed Architecture: Issue-Driven Learning Loop

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Issue-Driven Learning Loop                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  PHASE 0: Research (runs on iter 1 AND when stuck)                      │
│  ┌──────────────┐                                                        │
│  │   Research   │ ← Re-runs when escape_level >= 2                      │
│  │    Agent     │                                                        │
│  └──────────────┘                                                        │
│         │                                                                │
│         ▼                                                                │
│  PHASE 1: Script Generation                                              │
│  ┌──────────────┐     Receives: research + iteration_history            │
│  │Script Writer │                                                        │
│  └──────────────┘                                                        │
│         │                                                                │
│         ▼                                                                │
│  PHASE 2: Execution                                                      │
│  ┌──────────────┐                                                        │
│  │   Executor   │                                                        │
│  └──────────────┘                                                        │
│         │                                                                │
│         ▼                                                                │
│  PHASE 3: Quality Evaluation                                             │
│  ┌──────────────┐     Outputs: structured issues list                   │
│  │   Quality    │                                                        │
│  │   Analyst    │                                                        │
│  └──────────────┘                                                        │
│         │                                                                │
│         ▼                                                                │
│  PHASE 3.5: Issue Resolution (NEW)                                       │
│  ┌──────────────┐     Maps: issues → {param: value} changes             │
│  │    Issue     │     Uses: KB patterns, parameter knowledge             │
│  │   Resolver   │     Outputs: concrete parameter_modifications          │
│  └──────────────┘                                                        │
│         │                                                                │
│         ▼                                                                │
│  PHASE 4: Learning (Simplified)                                          │
│  ┌──────────────┐     Just records: baseline vs result                  │
│  │   Learning   │     Auto-recorded via lifecycle hooks                  │
│  │    Agent     │     Decides: iterate / switch_technique / complete    │
│  └──────────────┘                                                        │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Session Manager (Python-side, not LLM)                           │   │
│  │  - Tracks baseline/result pairs automatically                     │   │
│  │  - Maintains iteration_history for context                        │   │
│  │  - Uses SDK lifecycle hooks for auto-recording                    │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Order (Recommended)

### Phase A: Foundation (Do First)
1. **Add Session Manager** - Python class (not LLM) that tracks state
2. **Call record_baseline** - In orchestrator before iteration 2+
3. **Pass iteration_history** - To all agent prompts

### Phase B: Issue Resolution (Do Second)
4. **Add Issue Resolver Agent** - Maps issues → parameters
5. **Update Quality Analyst output** - Structured issue list with severity

### Phase C: Research Loop (Do Third)
6. **Re-run Research when stuck** - escape_level >= 2 or switch_technique
7. **Add alternative_approaches tracking** - Don't repeat failed approaches

### Phase D: Automation (Do Last)
8. **Add lifecycle hooks** - Auto-record tool outcomes
9. **Simplify Learning Agent** - Just decide next action, don't record

---

## New Components Needed

### 1. Session Manager (Python class)

```python
class SessionManager:
    """Manages experiment state without LLM intervention."""

    def __init__(self):
        self.baseline: Optional[ExperimentState] = None
        self.iteration_history: List[IterationRecord] = []
        self.issues_seen: Dict[str, int] = {}  # issue → count
        self.params_tried: Dict[str, List[float]] = {}  # param → values tried

    def record_baseline(self, params: dict, score: float, script_path: str):
        """Called by orchestrator before each iteration."""
        self.baseline = ExperimentState(params, score, script_path)

    def record_result(self, params: dict, score: float, issues: list):
        """Called by orchestrator after quality evaluation."""
        delta = score - self.baseline.score if self.baseline else 0
        self.iteration_history.append(IterationRecord(
            params=params,
            score=score,
            delta=delta,
            issues=issues
        ))

    def get_iteration_summary(self) -> str:
        """Returns context for agent prompts."""
        lines = ["## Iteration History"]
        for i, rec in enumerate(self.iteration_history, 1):
            lines.append(f"- Iter {i}: score={rec.score:.1f}, delta={rec.delta:+.1f}")
            if rec.issues:
                lines.append(f"  Issues: {', '.join(rec.issues[:3])}")
        return "\n".join(lines)

    def get_params_to_avoid(self) -> dict:
        """Returns parameter values that made things worse."""
        # Track which param values correlated with score drops
        ...
```

### 2. Issue Resolver Agent

```python
class IssueResolverOutput(BaseModel):
    """Output from Issue Resolver - concrete parameter changes."""
    parameter_modifications: Dict[str, float] = Field(
        description="Concrete {param: value} changes to apply"
    )
    reasoning: str = Field(
        description="Why these changes should fix the issues"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in this fix (0-1)"
    )
    requires_technique_change: bool = Field(
        default=False,
        description="True if issues can't be fixed with params alone"
    )

# Agent definition
issue_resolver = Agent[SharedContext](
    name="Issue Resolver",
    instructions="""You map quality issues to concrete parameter changes.

Given a list of issues from Quality Analyst, determine SPECIFIC parameter values
that should fix them. Use the knowledge base for proven fixes.

## Issue → Parameter Mapping Examples
- "overexposed/clipped highlights" → {"blackbody_intensity": 2.0, "emission_strength": 5.0}
- "no animation/static" → {"temperature": 3.0, "fuel_amount": 2.0}
- "wrong shape/not spherical" → {"domain_scale": 2.0}
- "no surface detail" → {"noise_strength": 2.0, "flame_vorticity": 0.8}

## Rules
1. ALWAYS output concrete numbers, not descriptions
2. Check get_parameter_knowledge() for each param before setting
3. If issues require structural changes (not just params), set requires_technique_change=True
4. Confidence should reflect how well the KB supports this fix
""",
    tools=[query_knowledge_base, get_parameter_knowledge],
    output_type=IssueResolverOutput
)
```

### 3. Lifecycle Hooks

```python
from agents import AgentHooks

class ExperimentHooks(AgentHooks):
    """Auto-records tool outcomes without agent intervention."""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    async def on_tool_end(
        self,
        context: RunContextWrapper,
        agent: Agent,
        tool: Tool,
        result: str
    ) -> None:
        """Called after every tool execution."""
        if tool.name == "generate_script":
            self.session_manager.record_script_generated(result)
        elif tool.name == "modify_script":
            self.session_manager.record_script_modified(tool.arguments, result)
        elif tool.name == "execute_blender_script":
            self.session_manager.record_execution(result)
```

---

## Success Metrics

1. **Scripts differ between iterations** - Config class parameters actually change
2. **Scores improve over iterations** - Not random, correlated with param changes
3. **Issues resolve** - Same issue doesn't appear 3+ times in a row
4. **KB grows** - Successful fixes recorded as patterns
5. **Research re-runs when stuck** - New approaches discovered mid-session

---

## References

- [OpenAI Agents SDK - Agents](https://github.com/openai/openai-agents-python/blob/main/docs/agents.md)
- [OpenAI Agents SDK - Tools](https://github.com/openai/openai-agents-python/blob/main/docs/tools.md)
- [OpenAI Agents SDK - Guardrails](https://github.com/openai/openai-agents-python/blob/main/docs/guardrails.md)
- [OpenAI Agents SDK - Results](https://github.com/openai/openai-agents-python/blob/main/docs/results.md)
- [SELF_LEARNING_ARCHITECTURE.md](./SELF_LEARNING_ARCHITECTURE.md) - Previous work
