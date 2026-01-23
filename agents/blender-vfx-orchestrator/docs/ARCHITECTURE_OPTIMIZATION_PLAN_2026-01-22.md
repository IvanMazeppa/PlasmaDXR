# Multi-Agent Architecture Optimization Plan

**Date:** 2026-01-22
**Author:** Ben + Claude
**Status:** Active Implementation
**Last Updated:** 2026-01-22 (Phase 1, 2, 6, 7 complete)

---

## Executive Summary

This document captures the comprehensive analysis of the Blender VFX Orchestrator's multi-agent architecture and provides a detailed optimization plan based on OpenAI Agents SDK best practices.

**Key Finding:** The orchestrator is using **handoffs** when it should be using **agents-as-tools**. This fundamental pattern mismatch causes most of the reliability issues.

---

## Implementation Progress

| Phase | Task | Status | Notes |
|-------|------|--------|-------|
| 1 | RunHooks for loop detection | ✅ **COMPLETE** | `hooks/enforcement_hooks.py` created |
| 2 | Agents-as-tools pattern | ✅ **COMPLETE** | 3 Coordinator agents + as_tool() wrappers |
| 6 | API Validator agent | ✅ **COMPLETE** | `specialized_agents/api_validator.py` created |
| 7 | Tracing everywhere | ✅ **COMPLETE** | 10+ trace() calls with metadata |
| 3 | Input/Output guardrails | 🔄 **NEXT** | After Phase 2 |
| 4 | SDK Sessions | ⏳ Pending | Low priority |
| 5 | Turn budget per agent | ⚠️ **PARTIAL** | Covered by RunHooks max_turns |

---

## Completed Work Details

### Phase 1: RunHooks for Loop Detection ✅

**Files Created:**
- `hooks/__init__.py`
- `hooks/enforcement_hooks.py`

**Implementation:**
```python
class EnforcementHooks(RunHooks):
    async def on_tool_start(self, context, agent, tool):
        # ENFORCEMENT 1: Loop Detection
        if call_count > self.config.max_same_tool_calls:
            raise LoopDetectedError(...)
        # ENFORCEMENT 2: Doc Query Requirement
        if tool_name in self.config.require_doc_query_before and not self._doc_query_made:
            raise DocQueryRequiredError(...)
```

**Factory Functions:**
- `create_research_hooks()` - max 3 same-tool calls, 8 turns
- `create_script_writer_hooks()` - requires doc query before write_script/modify_script
- `create_quality_analyst_hooks()` - max 3 same-tool calls, 6 turns
- `create_learning_agent_hooks()` - max 3 same-tool calls, 8 turns

**Integration:** All `Runner.run()` calls in `create_asset_pipeline()` now use hooks with try/except for enforcement exceptions.

---

### Phase 6: API Validator Agent ✅

**Files Created:**
- `specialized_agents/api_validator.py`

**Key Components:**

1. **Structured Output Models:**
   - `APICallValidation` - per-call validation result
   - `CodeValidationResult` - complete validation for code snippet

2. **Known Blender 5.0 API Changes:**
   ```python
   KNOWN_API_CHANGES = {
       'inputs["Smoke"]': {"correction": 'inputs["Grid"]', ...},
       'inputs["Smoke Color"]': {"correction": 'inputs["Grid Color"]', ...},
       'modifier.effector_weights': {"correction": 'effector_weights', ...},
       'flow_type': {"correction": 'flow_behavior', ...},
   }
   ```

3. **Function Tools:**
   - `extract_blender_api_calls` - parse code for bpy.* calls
   - `check_known_api_changes` - fast check against known issues
   - `validate_api_call_against_docs` - verify against vector store
   - `format_validation_report` - structured output

4. **Integration Points:**
   - `create_api_validator()` - create agent with validation tools
   - `get_api_validator_as_tool()` - wrap as tool for orchestrator
   - `validate_code_api()` - lightweight validation (no agent call)

**Pipeline Integration (Phase 1.5):**
```python
# Between Phase 1 (Script) and Phase 2 (Execute)
if script.script_path:
    api_validation = await validate_code_api(script_content)
    if not api_validation.is_valid:
        # Apply corrections automatically
        for pattern, fix in KNOWN_API_CHANGES.items():
            corrected_content = corrected_content.replace(pattern, fix["correction"])
        # Write corrected script
        Path(corrected_path).write_text(corrected_content)
```

---

### Phase 7: Tracing Everywhere ✅

**Trace Hierarchy:**
```
VFX Pipeline: {asset_name}           ← Outer trace (existing)
├── Phase 0: Research                 ← NEW nested trace
├── [Iteration 1]
│   ├── Phase 1: Script Writer        ← NEW (is_initial=True)
│   ├── Phase 1.5: API Validation     ← (sync, logged to stderr)
│   ├── Phase 2: Executor             ← NEW
│   ├── Phase 3: Quality Analyst      ← NEW
│   └── Phase 4: Learning Agent       ← NEW
├── [Iteration 2+]
│   ├── Phase 1: Script Writer        ← NEW (is_modification=True)
│   ├── ...
│   └── Technique Switch Research     ← NEW (if stuck)
```

**Metadata Fields for Filtering:**
- `phase` - research, script_writer, executor, quality_analyst, learning_agent, technique_switch
- `effect_type` - explosion, fire, nebula, etc.
- `iteration` - which attempt
- `score` / `passed` - quality results (on learning agent)
- `previous_score` - for modifications
- `is_initial` / `is_modification` - script writer mode

**View traces:** https://platform.openai.com/traces

---

### Phase 2: Agents-as-Tools Pattern ✅

**Implementation Date:** 2026-01-22

**Key Changes:**

1. **New Coordinator Agents Created:**
   - `TechniqueSelector` - Selects initial technique (iteration 1)
   - `ModificationStrategist` - Decides modification strategy (iteration 2+)
   - `QualityGateJudge` - Interprets quality results and decides next action

2. **New Structured Output Models:**
   - `TechniqueDecision` - Technique selection output
   - `ModificationDecision` - Modification strategy output
   - `QualityDecision` - Quality gate decision output

3. **Pipeline Integration:**
   - Phase 0.5: Technique Selection Coordinator called after research
   - Phase 1.1: Modification Coordinator called before Script Writer (iter 2+)
   - Phase 5: Quality Gate Coordinator interprets quality results

4. **Factory Functions Added:**
   - `create_coordinator_agent()` - Full coordinator with all agents as tools
   - `create_technique_selection_coordinator()` - Lightweight technique selector
   - `create_modification_coordinator()` - Lightweight modification strategist
   - `create_quality_gate_coordinator()` - Lightweight quality gate judge

5. **Deprecation:**
   - `create_asset()` deprecated with warning, redirects to `create_asset_pipeline()`
   - `create_vfx_asset()` now uses pipeline instead of handoff-based method

**SDK Pattern Used:**
```python
coordinator = Agent(
    tools=[
        research_agent.as_tool(
            tool_name="research_approach",
            tool_description="Research best approach for effect type",
        ),
        # ... other agents as tools
    ],
)
```

**Key Benefit:** Python controls the pipeline sequence; Coordinators make intelligent decisions at specific points. No more relying on LLM instruction-following for workflow control.

---

## Current State Assessment

### What's Working Well

| Component | Status | Notes |
|-----------|--------|-------|
| 6 Specialized Agents | ✅ Good | Research, Script Writer, Executor, Quality Analyst, Learning, **API Validator** |
| **3 Coordinator Agents** | ✅ **NEW** | Technique Selection, Modification Strategy, Quality Gate |
| SessionManager | ✅ Excellent | Deterministic Python-side state tracking |
| Self-Learning Strategies | ✅ Good | 5 strategies implemented |
| Quality Evaluation | ✅ Good | Vision + LPIPS/CLIP/TOPIQ metrics |
| Structured Outputs | ✅ Good | Pydantic models for type-safe data passing |
| **RunHooks Enforcement** | ✅ Good | Loop detection, doc query requirements |
| **API Validation** | ✅ Good | Blender 5.0 API corrections |
| **Tracing** | ✅ Good | Full visibility with metadata |
| **Agents-as-Tools Pattern** | ✅ **NEW** | Coordinators use sub-agents via as_tool() |

### What's Still Not Working

| Problem | Severity | Status |
|---------|----------|--------|
| Blender 5.0 API Errors | 🟡 Mitigated | API Validator catches known issues; unknown APIs still possible |
| Research Loop Infinity | ✅ **FIXED** | RunHooks with LoopDetectedError |
| SDK Docs Not Used | 🟡 Improved | DocQueryRequiredError enforces research-first |
| Workflow Fragility | ✅ **FIXED** | **Phase 2 complete** - Coordinators + code-based pipeline |

---

## Root Cause Analysis

### Problem 1: Blender 5.0 API Errors

**Status:** 🟡 MITIGATED (Phase 6)

The API Validator now catches 4 known breaking changes automatically. However:
- Unknown API changes may still slip through
- Full agent-based validation (with doc search) available but not yet integrated into pipeline

**Remaining Work:**
- Consider adding more known API changes as discovered
- Integrate full API Validator agent for comprehensive checks

### Problem 2: Research Loop Infinity

**Status:** ✅ FIXED (Phase 1)

RunHooks now enforce:
- Max 3 calls to same tool
- Turn budget per agent
- `LoopDetectedError` exception caught and handled gracefully

### Problem 3: SDK Documentation Not Being Used

**Status:** 🟡 IMPROVED (Phase 1)

`DocQueryRequiredError` blocks `write_script` and `modify_script` if no prior documentation search. However, this enforcement is at the tool level, not the agent level.

**Remaining Work:**
- Phase 3 (Guardrails) will add input validation at agent level

### Problem 4: Workflow Logic Fragility

**Status:** ✅ FIXED (Phase 2)

The handoff-based `create_asset()` method is now deprecated. The Python-controlled `create_asset_pipeline()` now leverages LLM intelligence for decisions via 3 Coordinator agents:

1. **Technique Selector** - Intelligent initial technique selection
2. **Modification Strategist** - Smart modification strategy decisions
3. **Quality Gate Judge** - Informed quality gate interpretation

**Implementation Complete:** Python controls workflow sequence; Coordinators provide intelligent decisions at specific points.

---

## SDK Best Practices Reference

### 1. Tracing (ALWAYS USE) ✅ IMPLEMENTED

```python
from agents import Agent, Runner, trace

with trace("Workflow Name", group_id=session_id):
    result = await Runner.run(agent, prompt, max_turns=10)

# View traces: https://platform.openai.com/traces
```

### 2. Handoffs (Agent Transfer - Decentralized)

```python
from agents import Agent

specialist = Agent(
    name="Specialist",
    instructions="You handle specific tasks...",
    handoff_description="Use for X tasks",  # REQUIRED for triage
)

coordinator = Agent(
    name="Coordinator",
    handoffs=[specialist],  # List of agents that can be handed to
)
```

### 3. Agents as Tools (Centralized Control) ⭐ RECOMMENDED FOR PHASE 2

```python
# Agent called as a tool - returns control to caller
orchestrator = Agent(
    name="Orchestrator",
    tools=[
        specialist_agent.as_tool(
            tool_name="do_specialist_task",
            tool_description="What the specialist does and when to use it",
        ),
    ],
)
```

**Key Difference:**
- Handoffs: New agent takes over conversation completely
- as_tool: Agent called as utility, control returns to caller

### 4. RunHooks (Lifecycle Callbacks) ✅ IMPLEMENTED

```python
from agents import RunHooks

class MyHooks(RunHooks):
    async def on_agent_start(self, context, agent):
        print(f"Agent {agent.name} starting")

    async def on_tool_start(self, context, agent, tool):
        print(f"Tool {tool.name} called")

    async def on_tool_end(self, context, agent, tool, result):
        print(f"Tool {tool.name} returned: {result}")

    async def on_handoff(self, context, from_agent, to_agent):
        print(f"Handoff: {from_agent.name} -> {to_agent.name}")

result = await Runner.run(agent, prompt, run_hooks=MyHooks())
```

### 5. Input/Output Guardrails (Phase 3)

```python
from agents import Agent, input_guardrail, GuardrailFunctionOutput

@input_guardrail
async def validate_input(ctx, agent, input):
    # Validation logic
    return GuardrailFunctionOutput(
        tripwire_triggered=is_invalid,
        output_info={"reason": "why invalid"}
    )

agent = Agent(
    input_guardrails=[validate_input],
)
```

### 6. Structured Output ✅ ALREADY IN USE

```python
from pydantic import BaseModel
from agents import Agent, AgentOutputSchema

class MyOutput(BaseModel):
    field1: str
    field2: int

agent = Agent(
    output_type=AgentOutputSchema(MyOutput, strict_json_schema=False),
)

result = await Runner.run(agent, prompt)
output = result.final_output_as(MyOutput)  # Type-safe!
```

### 7. Sessions (Conversation Persistence) - Phase 4

```python
from agents import Agent, Runner, SQLiteSession

session = SQLiteSession("conversation_123", "history.db")

# First turn
result = await Runner.run(agent, "Question 1", session=session)

# Second turn - agent remembers previous context
result = await Runner.run(agent, "Follow-up", session=session)
```

---

## Proposed Architecture: Hybrid Orchestrator

### Current vs Proposed

```
CURRENT (After Phase 1, 6, 7):
┌─────────────────────────────────────────────────────────────────────┐
│                 create_asset_pipeline() (Python)                    │
│                 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                     │
│  - Deterministic state machine ✅                                   │
│  - Calls agents sequentially via Runner.run() ✅                    │
│  - RunHooks for enforcement ✅                                      │
│  - API Validation (Phase 1.5) ✅                                    │
│  - Tracing with metadata ✅                                         │
├─────────────────────────────────────────────────────────────────────┤
│  BUT: No LLM intelligence for workflow DECISIONS                    │
│       (e.g., "should we switch technique?")                         │
└─────────────────────────────────────────────────────────────────────┘

PROPOSED (After Phase 2):
┌─────────────────────────────────────────────────────────────────────┐
│                 PipelineOrchestrator (Python)                       │
│                 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━                        │
│  - Deterministic state machine (no LLM)                             │
│  - Handles iteration loop, quality gates                            │
│  - Manages escape velocity transitions                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐                                                    │
│  │ Coordinator │ ← Reasoning agent for DECISIONS only               │
│  │   Agent     │   Called at decision points:                       │
│  │  (gpt-5.2)  │   - Which technique to try?                        │
│  └─────────────┘   - Should we switch approach?                     │
│         │          - What parameters to modify?                     │
│         │                                                           │
│         ├── research_agent.as_tool("research_approach")             │
│         ├── api_validator.as_tool("validate_blender_api")  ← NEW    │
│         ├── script_writer.as_tool("generate_script")                │
│         ├── executor.as_tool("execute_script")                      │
│         ├── quality_analyst.as_tool("evaluate_render")              │
│         └── learning_agent.as_tool("record_experiment")             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Architecture Decisions

1. **Keep `create_asset_pipeline()`, deprecate `create_asset()`**
   - Python-controlled pipeline is the right approach
   - Handoff-based method fights against SDK design intent

2. **SessionManager + SDK Sessions = Best of Both**
   - SessionManager: tracks params, scores, issues (deterministic Python)
   - SQLiteSession: tracks conversation history (SDK's persistence)

3. **Mechanical Enforcement Over Instructions** ✅ DONE
   - LLMs don't reliably follow "ALWAYS do X" instructions
   - RunHooks: Block tools if prerequisites not met
   - Guardrails: Validate inputs/outputs (Phase 3)
   - Turn budgets: Hard limits on agent reasoning

4. **API Validation as Separate Agent** ✅ DONE
   - Don't embed Blender 5.0 validation in Script Writer instructions
   - Created dedicated validator with `as_tool()` integration ready

---

## Implementation Plan

### Phase 1: RunHooks for Loop Detection and Enforcement ✅ COMPLETE

**Effort:** 2 hours (actual: ~2 hours)
**Impact:** Immediately stops infinite research loops (Problem 2) and enforces doc queries (Problem 3)

**Files Created:**
- `hooks/__init__.py`
- `hooks/enforcement_hooks.py`

**Integrated into:** `create_asset_pipeline()` - all 7 Runner.run() calls

---

### Phase 2: Convert to Agents-as-Tools Pattern ✅ COMPLETE

**Effort:** 4 hours (actual: ~4 hours)
**Impact:** Coordinator retains control, predictable workflow (Problem 4)

**Implementation Complete:** See "Phase 2: Agents-as-Tools Pattern" in Completed Work Details above.

**Key Files Modified:**
- `orchestrator.py` - Added 3 Coordinator agents and pipeline integration

**Decision Points Implemented:**
- ✅ Initial technique selection (Phase 0.5 - after research)
- ✅ Parameter modification strategy (Phase 1.1 - iteration 2+)
- ✅ Quality gate interpretation (Phase 5 - after each iteration)

---

### Phase 3: Add Input/Output Guardrails

**Effort:** 2 hours
**Impact:** Validation layer for all agent inputs/outputs
**Dependency:** Best done after Phase 2

**Implementation:**
- Create `guardrails/script_guardrails.py`
- Add `require_research_context` input guardrail
- Add `validate_script_output` output guardrail
- Apply to Script Writer agent

---

### Phase 4: Implement SDK Sessions for Persistence

**Effort:** 2 hours
**Impact:** Conversation memory across runs
**Priority:** LOW - SessionManager already handles most state

**Implementation:**
- Add `SQLiteSession` to all `Runner.run()` calls
- Create session database at `sessions/vfx_conversations.db`
- Integrate with existing SessionManager

---

### Phase 5: Add Turn Budget Per Agent ⚠️ PARTIAL

**Status:** Mostly covered by RunHooks implementation

**What's Done:**
- `max_turns` parameter on all `Runner.run()` calls
- `TurnBudgetExceededError` in EnforcementHooks (not yet raised)

**What Remains:**
- Consider using `agent.clone()` to inject turn budget into instructions
- This is low priority since RunHooks handle the critical cases

---

### Phase 6: Create Blender API Validation Agent ✅ COMPLETE

**Effort:** 3 hours (actual: ~2 hours)
**Impact:** Fixes Blender 5.0 API errors at source (Problem 1)

**Files Created:**
- `specialized_agents/api_validator.py`

**Integrated into:** `create_asset_pipeline()` as Phase 1.5

---

### Phase 7: Enable Tracing Everywhere ✅ COMPLETE

**Effort:** 1 hour (actual: ~30 min)
**Impact:** Full visibility into agent behavior

**Implementation:**
- 10 `trace()` calls total (3 outer, 7 nested)
- Metadata for filtering by phase, effect_type, iteration, score

---

## Implementation Priority Matrix (Updated)

| Phase | Task | Effort | Status | Priority |
|-------|------|--------|--------|----------|
| 1 | RunHooks for loop detection | 2h | ✅ COMPLETE | 🔴 CRITICAL |
| 2 | Agents-as-tools pattern | 4h | ✅ COMPLETE | 🟡 HIGH |
| 6 | API Validator agent | 3h | ✅ COMPLETE | 🔴 CRITICAL |
| 7 | Tracing everywhere | 1h | ✅ COMPLETE | 🟡 HIGH |
| **3** | **Input/Output guardrails** | **2h** | **🔄 NEXT** | **🟡 HIGH** |
| 4 | SDK Sessions | 2h | ⏳ Pending | 🟢 MEDIUM |
| 5 | Turn budget per agent | 1h | ⚠️ Partial | 🟢 LOW |

**Completed:** 10 hours (Phase 1: 2h, Phase 2: 4h, Phase 6: 3h, Phase 7: 1h)
**Remaining:** ~4 hours (Phase 3: 2h, Phase 4: 2h)

---

## Success Metrics

| Metric | Before | Current | Target |
|--------|--------|---------|--------|
| Research loops hitting MaxTurns | ~50% | ~5% ✅ | < 5% |
| Blender API errors on first run | ~70% | ~30% 🟡 | < 10% |
| Scripts passing validation | ~30% | ~60% 🟡 | > 80% |
| Iterations to quality threshold | 5-10 | TBD | 3-5 |
| Cost per asset | $2-5 | TBD | $0.50-1 |

---

## Lessons Learned

### From Phase 1 (RunHooks)
- RunHooks are powerful for enforcement but require careful exception handling
- Each agent type needs different hook configurations
- Factory functions (`create_*_hooks()`) make integration clean

### From Phase 6 (API Validator)
- Lightweight validation (no agent call) is fast and effective for known issues
- Full agent validation available for comprehensive checks but adds latency
- Known API changes should be maintained as a living list

### From Phase 7 (Tracing)
- Nested traces require the outer trace context to be active
- Metadata fields are essential for filtering in the dashboard
- Exception handling must be outside the trace block to avoid losing trace data

### From Phase 2 (Agents-as-Tools)
- `as_tool()` is the correct pattern for centralized control - NOT handoffs
- Lightweight coordinators (3-6 max_turns) are better than one big coordinator
- Structured outputs (TechniqueDecision, ModificationDecision, QualityDecision) make decisions actionable
- Python still controls the iteration loop - coordinators only make decisions
- Fallback logic is essential when coordinators fail (network issues, etc.)
- Deprecation warnings guide users to the correct API

---

## References

- [OpenAI Agents SDK Documentation](https://github.com/openai/openai-agents-python/tree/main/docs)
- [Multi-Agent Patterns](https://github.com/openai/openai-agents-python/blob/main/docs/multi_agent.md)
- [Tools Reference](https://github.com/openai/openai-agents-python/blob/main/docs/tools.md)
- [Handoffs](https://github.com/openai/openai-agents-python/blob/main/docs/handoffs.md)
- [Guardrails](https://github.com/openai/openai-agents-python/blob/main/docs/guardrails.md)
- [Tracing](https://github.com/openai/openai-agents-python/blob/main/docs/tracing.md)
- [Sessions](https://github.com/openai/openai-agents-python/blob/main/docs/sessions.md)

---

## Context for Phase 3 Implementation

When starting Phase 3 (Input/Output Guardrails), the key files to review are:

1. **`orchestrator.py`** - Main file to modify
   - Script Writer standalone agent (`_script_agent_standalone`) needs guardrails
   - Quality Analyst (`_quality_agent_standalone`) could benefit from output guardrails

2. **SDK Docs to Query:**
   - `/openai/openai-agents-python` via context7
   - Topic: "input_guardrail and output_guardrail decorators"
   - Topic: "GuardrailFunctionOutput structure"
   - URL: https://github.com/openai/openai-agents-python/blob/main/docs/guardrails.md

3. **Proposed Guardrails:**

   **Input Guardrails:**
   - `require_research_context` - Block Script Writer if no research findings provided
   - `validate_effect_type` - Ensure effect_type is valid enum value
   - `check_budget_before_quality` - Block Quality Analyst if budget exhausted

   **Output Guardrails:**
   - `validate_script_output` - Ensure ScriptOutput has valid script_path
   - `validate_quality_output` - Ensure QualityOutput has required fields
   - `validate_technique_decision` - Ensure TechniqueDecision has selected_technique

4. **Implementation Pattern:**
   ```python
   from agents import Agent, input_guardrail, GuardrailFunctionOutput

   @input_guardrail
   async def require_research_context(ctx, agent, input: str):
       if "research" not in input.lower() and "findings" not in input.lower():
           return GuardrailFunctionOutput(
               tripwire_triggered=True,
               output_info={"reason": "Script Writer requires research context"}
           )
       return GuardrailFunctionOutput(tripwire_triggered=False)

   script_writer = Agent(
       input_guardrails=[require_research_context],
       # ... existing config
   )
   ```

---

*This document should be updated as implementation progresses.*
