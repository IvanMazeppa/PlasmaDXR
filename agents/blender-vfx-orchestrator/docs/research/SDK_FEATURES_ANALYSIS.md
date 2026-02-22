# OpenAI Agents SDK v0.9.3 — Feature Analysis for VFX Orchestrator

**Date:** 2026-02-22
**Researcher:** SDK Specialist (Claude Opus 4.6)
**SDK Version:** v0.9.3 (released 2026-02-20)
**Codebase Baseline:** `orchestrator.py` — uses `Runner.run()`, `Agent`, `ModelSettings`, `function_tool`, `SQLiteSession`, `OpenAIResponsesCompactionSession`, `RunHooks`, input/output guardrails, `Agent.as_tool()`, structured outputs, tracing

---

## Executive Summary

The VFX orchestrator currently uses ~40% of the SDK's available capabilities. The remaining ~60% directly solve identified problems: context bloat, budget waste, lack of deterministic control, no branching for technique experiments, and no native HITL. This analysis covers 11 specific features plus 2 bonus discoveries (ToolOutputTrimmer, function tool timeouts).

**Priority tiers:**
- **P0 (Ship with Phase 2):** `call_model_input_filter`, `tool_use_behavior`, `is_enabled`, Tool Guardrails
- **P1 (Ship next):** `AdvancedSQLiteSession`, `needs_approval`, `Agent.clone()`
- **P2 (Nice to have):** `run_streamed()`, `parallel_tool_calls`, `prompt_cache_retention`, `reset_tool_choice`

---

## 1. AdvancedSQLiteSession

### What It Does

Enhanced SQLite session with conversation **branching**, **token usage tracking**, and **structured queries**. Extends the basic `SQLiteSession` the orchestrator currently uses.

```python
from agents.extensions.memory import AdvancedSQLiteSession

session = AdvancedSQLiteSession(
    session_id="explosion_run_001",
    db_path="conversations.db",
    create_tables=True
)

# After each agent run — automatic token tracking
result = await Runner.run(agent, "Generate fire script", session=session)
await session.store_run_usage(result)

# Branch from turn 2 to try a different technique
branch_id = await session.create_branch_from_turn(2)

# Get token usage per turn
turn_usage = await session.get_turn_usage()
for turn in turn_usage:
    print(f"Turn {turn['user_turn_number']}: {turn['total_tokens']} tokens")

# Get session-wide usage
usage = await session.get_session_usage()
print(f"Total tokens: {usage['total_tokens']}, Requests: {usage['requests']}")

# Search conversation by content
branch = await session.create_branch_from_content("fire technique")

# List all branches
branches = await session.list_branches()

# Switch between branches
await session.switch_to_branch("alternative_path")
```

### Key Methods

| Method | Purpose |
|--------|---------|
| `store_run_usage(result)` | Track tokens/requests per turn |
| `get_session_usage(branch_id=None)` | Total tokens, input/output breakdown, request count |
| `get_turn_usage(user_turn_number=None)` | Per-turn token metrics |
| `create_branch_from_turn(n)` | Branch conversation from turn N |
| `create_branch_from_content(search)` | Branch by searching conversation content |
| `list_branches()` | All branches with turn/message counts |
| `switch_to_branch(id)` | Navigate between branches |
| `delete_branch(id, force=False)` | Remove a branch |
| `get_conversation_turns()` | All turns with branching capability info |
| `find_turns_by_content(keyword)` | Search turns by keyword |
| `get_tool_usage()` | Statistics on tool invocations per turn |

### What Problem It Solves

**Technique experimentation without losing state.** Currently when escape velocity triggers a technique switch (L2+), the orchestrator starts fresh. With branching:
1. Run mantaflow_fire to iteration 3, score 45
2. Branch from technique selection (turn 1)
3. Try shader_volume on the branch
4. If shader_volume scores 55, keep it. If not, switch back.

**Budget tracking with per-turn granularity.** Currently `BudgetTracker` is a custom implementation. `AdvancedSQLiteSession` provides built-in token tracking per turn, per branch, and per session — for free.

**Conversation search.** `find_turns_by_content("resolution_max")` lets the system recall what it previously tried without scanning full conversation history.

### Where It Integrates

| File | Change |
|------|--------|
| `orchestrator.py:367-401` | Replace `get_or_create_sdk_session()` — swap `SQLiteSession` + `OpenAIResponsesCompactionSession` for `AdvancedSQLiteSession` (can still wrap with compaction) |
| `orchestrator.py:2264` (technique switch) | Before switching technique: `branch_id = await session.create_branch_from_turn(technique_selection_turn)` |
| `orchestrator.py:3843` (iteration cost) | Replace custom `budget_spent_start` tracking with `session.store_run_usage(result)` + `session.get_turn_usage()` |
| `session_manager.py` | Add `branches: Dict[str, BranchInfo]` to SessionState for persistence |

### Implementation Complexity: **Medium**

- Replacing SQLiteSession with AdvancedSQLiteSession is ~20 lines
- Token tracking integration replaces custom budget tracking: ~50 lines
- Branching for technique experiments: ~100 lines (new logic in escape velocity handler)
- **Risk:** Must verify compaction session wrapping still works with AdvancedSQLiteSession as underlying

### Priority: **P1**

Branching is a game-changer for technique experimentation. Token tracking replaces custom code. But the existing session system works — this improves it, doesn't fix a blocker.

---

## 2. call_model_input_filter

### What It Does

A callback on `RunConfig` that intercepts and modifies the input **right before every LLM call**. Receives the full conversation history + instructions, returns modified versions.

```python
from agents import Runner, RunConfig
from agents.run_config import CallModelData, ModelInputData

def trim_old_history(data: CallModelData) -> ModelInputData:
    """Keep only the last 5 conversation items to prevent context bloat."""
    trimmed = data.model_data.input[-5:]
    return ModelInputData(
        input=trimmed,
        instructions=data.model_data.instructions
    )

result = await Runner.run(
    agent,
    "Generate fire script",
    run_config=RunConfig(call_model_input_filter=trim_old_history),
)
```

### Advanced Usage for VFX Orchestrator

```python
def vfx_context_trimmer(data: CallModelData) -> ModelInputData:
    """Smart trimming: keep system prompt + truth pack + last N items."""
    items = data.model_data.input

    # Always keep first 2 items (system setup + truth pack injection)
    preserved = items[:2] if len(items) > 2 else items

    # Keep last 8 items (current iteration context)
    recent = items[-8:] if len(items) > 8 else items[2:]

    # Deduplicate (preserved might overlap with recent)
    seen_ids = set()
    final = []
    for item in preserved + recent:
        item_id = id(item)
        if item_id not in seen_ids:
            final.append(item)
            seen_ids.add(item_id)

    return ModelInputData(
        input=final,
        instructions=data.model_data.instructions
    )
```

### What Problem It Solves

**Context bloat — the #3 identified problem.** Long-running pipeline iterations accumulate verbose agent output in conversation history. By iteration 4-5, the context is so bloated that the Script Writer's quality degrades. The current solution (session compaction at ~25K tokens) is blunt — it summarizes everything. `call_model_input_filter` enables **surgical trimming**: keep the truth pack, keep the latest iteration context, drop everything else.

This is strictly superior to compaction for the VFX pipeline because:
- Compaction costs an LLM call ($)
- Compaction can lose important details in summarization
- The filter is deterministic, free ($0), and controllable

### Where It Integrates

| File | Change |
|------|--------|
| `orchestrator.py:910-923` (`_run_agent`) | Add `call_model_input_filter` to the `RunConfig` passed to `Runner.run()` |
| New: `utils/context_trimmer.py` (~80 lines) | VFX-specific trimming logic: preserve truth pack injection, current script path, latest quality feedback; drop intermediate iterations |
| `orchestrator.py:367-401` | Can potentially reduce/remove compaction dependency if filter is aggressive enough |

### Implementation Complexity: **Low**

- The `_run_agent` wrapper already supports `RunConfig` — just needs the filter function
- Filter logic is ~30-50 lines of deterministic Python
- No SDK API changes needed, just a new function wired into existing RunConfig

### Priority: **P0**

Context bloat is a known top-3 problem. This is the cheapest, most direct fix. Implement immediately.

---

## 3. needs_approval (Human-in-the-Loop)

### What It Does

Native SDK approval gates that **pause execution**, **serialize state**, and **resume after human review**. Supports per-tool approval with state persistence.

```python
from agents import function_tool, Runner
from agents.run import RunState

# Unconditional approval
@function_tool(needs_approval=True)
async def execute_blender_script(script_path: str) -> str:
    """Execute a Blender script — requires human approval."""
    return await run_blender(script_path)

# Conditional approval
async def requires_approval_check(ctx, params, call_id) -> bool:
    """Only require approval if budget is low or iteration > 3."""
    return ctx.context.budget_remaining < 5.0 or ctx.context.iteration > 3

@function_tool(needs_approval=requires_approval_check)
async def evaluate_render(render_path: str) -> str:
    """Evaluate render quality — may require approval."""
    return await run_quality_eval(render_path)
```

**State serialization for durable pauses:**

```python
# Pipeline pauses when needs_approval triggers
result = await Runner.run(agent, prompt)

if result.interruptions:
    # Serialize state for later resumption
    state = result.to_state()
    STATE_PATH.write_text(state.to_string())

    # ... time passes, human reviews ...

    # Resume from serialized state
    stored = json.loads(STATE_PATH.read_text())
    state = await RunState.from_json(agent, stored)

    for interruption in result.interruptions:
        if human_approves(interruption):
            state.approve(interruption, always_approve=False)
        else:
            state.reject(interruption)

    # Continue execution
    result = await Runner.run(agent, state)
```

**Decision caching:**

```python
# Auto-approve all future calls to this tool
state.approve(interruption, always_approve=True)

# Auto-reject all future calls to this tool
state.reject(interruption, always_reject=True)
```

### What Problem It Solves

**Pipeline pauses at stall detection.** The Mission Statement (Section 9) defines 5 HITL checkpoints: prompt approval, stall detection (3+ iterations with no improvement), budget warning, critical issues, and escalation. Currently NONE of these are implemented natively — the pipeline just runs to completion or crashes.

With `needs_approval`:
- Executor tool gets `needs_approval=True` → human sees the script before Blender runs it
- Quality evaluation gets conditional approval → only pauses when budget is low or quality is stalling
- State serializes to disk → human can review hours later and resume

### Where It Integrates

| File | Change |
|------|--------|
| `tools/blender_executor_tools.py` | Add `needs_approval=True` to `execute_blender_script` function_tool |
| `tools/asset_evaluator_tools.py` | Add conditional `needs_approval` to `evaluate_render` based on budget/iteration |
| `orchestrator.py:_run_agent` | Handle `result.interruptions` — serialize state, notify user, wait for approval |
| `session_manager.py` | Add `pending_approvals: List[ApprovalRequest]` to SessionState |
| New: `utils/hitl_handler.py` (~100 lines) | Approval logic: serialize state to file, poll for approval, resume |

### Implementation Complexity: **Medium-High**

- Adding `needs_approval` to tools: 5 lines each
- The approval flow handler (serialize, poll, resume) is ~100 lines of new code
- Must handle edge cases: what if human never approves? Timeout? State corruption?
- `RunState.from_json()` requires exact agent definition match — agent code changes break serialized state

### Priority: **P1**

Ben explicitly wants this (Interview Q16, Mission Statement Section 9). But it requires careful design for the file-based approval UX. Not a blocker — the system runs without it.

---

## 4. is_enabled (Conditional Tool Enabling)

### What It Does

Dynamically show/hide tools from the LLM at runtime based on context. Disabled tools are **completely invisible** to the model — it doesn't even know they exist.

```python
from agents import Agent, function_tool, RunContextWrapper

def budget_allows_vision(ctx: RunContextWrapper, agent) -> bool:
    """Only show vision evaluation tool when budget allows."""
    return ctx.context.budget_tracker.can_afford_evaluation()

def budget_allows_docs(ctx: RunContextWrapper, agent) -> bool:
    """Only show doc search when budget allows."""
    return ctx.context.budget_tracker.get_remaining() > 2.0

quality_analyst = Agent(
    name="Quality Analyst",
    tools=[
        evaluate_render_vision.as_tool(
            tool_name="evaluate_with_vision",
            is_enabled=budget_allows_vision,  # Hidden when budget exhausted
        ),
        evaluate_render_metrics,  # Always available (free, local ML)
    ],
)

research_agent = Agent(
    name="Research Agent",
    tools=[
        search_blender_docs.as_tool(
            tool_name="search_docs",
            is_enabled=budget_allows_docs,  # Hidden when budget < $2
        ),
        search_truth_pack,  # Always available (free, local)
        search_code_patterns,  # Always available (free, local)
    ],
)
```

### What Problem It Solves

**Budget waste on expensive tools.** Currently `check_budget_before_quality` is an input guardrail that raises `InputGuardrailTripwireTriggered` when budget is exhausted. This works but is clunky — the agent tries to use the tool, gets blocked, and has to recover. With `is_enabled`, the tool **never appears** in the agent's tool list when budget is low, so it naturally falls back to cheaper alternatives without error handling.

Also enables:
- **Feature gating:** Hide experimental tools during production runs
- **Phase-based tools:** Only show modification tools after iteration 1
- **Effect-type filtering:** Hide Mantaflow tools when doing rigid body

### Where It Integrates

| File | Change |
|------|--------|
| `specialized_agents/quality_analyst.py` | Add `is_enabled=budget_allows_vision` to vision evaluation tool |
| `specialized_agents/docs_expert.py` | Add `is_enabled=budget_allows_docs` to doc search tools |
| `orchestrator.py:create_*_coordinator()` | Wrap expensive agent-as-tool calls with `is_enabled` callbacks |
| `tools/asset_evaluator_tools.py` | Move budget check from guardrail to `is_enabled` |

### Implementation Complexity: **Low**

- Each tool change is 1-3 lines (add `is_enabled=callback`)
- Callback functions are 3-5 lines each
- No architectural changes needed
- Can coexist with existing guardrails during transition

### Priority: **P0**

Trivial to implement, immediate budget savings, cleaner than guardrail-based budget checks. Do it first.

---

## 5. tool_use_behavior: stop_on_first_tool / StopAtTools

### What It Does

Controls what happens after an agent calls a tool. Three modes beyond the default (`run_llm_again`):

```python
from agents import Agent
from agents.agent import StopAtTools

# Mode 1: Stop after ANY tool call — tool output IS the agent output
executor = Agent(
    name="Executor",
    tools=[execute_blender_script],
    tool_use_behavior="stop_on_first_tool",  # No LLM post-processing
)

# Mode 2: Stop only at specific tools
quality_gate = Agent(
    name="Quality Gate",
    tools=[evaluate_metrics, evaluate_vision, make_decision],
    tool_use_behavior=StopAtTools(stop_at_tool_names=["make_decision"]),
)

# Mode 3: Custom function
from agents.agent import ToolsToFinalOutputResult, ToolsToFinalOutputFunction

def handle_execution_result(ctx, tool_results):
    """Stop if execution succeeded, continue if failed."""
    for result in tool_results:
        if "SUCCESS" in str(result.output):
            return ToolsToFinalOutputResult(
                is_final_output=True,
                final_output=result.output
            )
    return ToolsToFinalOutputResult(is_final_output=False, final_output=None)

executor = Agent(
    name="Smart Executor",
    tools=[execute_blender_script],
    tool_use_behavior=handle_execution_result,
)
```

### What Problem It Solves

**Unnecessary LLM calls for deterministic operations.** The Executor agent currently runs a script in Blender, then the LLM processes the result to produce a response. That LLM call is pure waste — the execution output (success/failure + paths) is the complete answer. Same for the Learning Agent when it just records an experiment.

With `stop_on_first_tool`:
- Executor calls `execute_blender_script` → result IS the output. No LLM follow-up. **Saves ~$0.02/call.**
- Learning Agent calls `record_experiment` → result IS the output. **Saves ~$0.01/call.**

With `StopAtTools`:
- Quality Gate calls `evaluate_metrics` and `evaluate_vision` (LLM processes both), then calls `make_decision` → decision IS the output. Saves one LLM call.

### Where It Integrates

| File | Change |
|------|--------|
| `specialized_agents/executor.py` | Set `tool_use_behavior="stop_on_first_tool"` on Executor agent |
| `specialized_agents/learning_agent.py` | Set `tool_use_behavior="stop_on_first_tool"` for recording-only modes |
| `orchestrator.py:create_quality_gate_coordinator()` | Use `StopAtTools(stop_at_tool_names=["make_quality_decision"])` |
| `specialized_agents/api_validator.py` | Set `tool_use_behavior="stop_on_first_tool"` — validation result is the output |

### Implementation Complexity: **Low**

- 1 line per agent definition change
- No new code needed
- Immediate cost savings

### Priority: **P0**

Every run has 15-46+ LLM calls. Eliminating unnecessary post-tool LLM calls for deterministic agents saves ~$0.05-0.10/run. With 40-100 runs/month, that's $2-10/month savings from a single-line change.

---

## 6. ToolInputGuardrail / ToolOutputGuardrail

### What It Does

Guardrails that wrap **individual tools** (not agents). Run before/after every invocation of that specific tool. Can skip the call, replace the output, or raise a tripwire. **Cost: $0** (pure Python, no LLM).

```python
from agents import (
    function_tool,
    tool_input_guardrail,
    tool_output_guardrail,
    ToolGuardrailFunctionOutput,
)

@tool_input_guardrail
def validate_script_against_truth_pack(data):
    """Check script for hallucinated attributes BEFORE execution."""
    args = json.loads(data.context.tool_arguments or "{}")
    script_path = args.get("script_path", "")

    if script_path:
        truth_pack = data.context.context.truth_pack
        errors = validate_against_truth_pack(script_path, truth_pack)
        if errors:
            fix_report = auto_fix_errors(script_path, errors)
            return ToolGuardrailFunctionOutput.reject_content(
                f"Script had {len(errors)} hallucinated attributes. "
                f"Auto-fixed: {fix_report}. Re-run with fixed script."
            )
    return ToolGuardrailFunctionOutput.allow()


@tool_output_guardrail
def check_execution_output(data):
    """Validate execution output for known failure patterns."""
    output = str(data.output or "")
    if "CRITICAL: ZERO_LIGHTS" in output or "BLACK_SCREEN" in output:
        return ToolGuardrailFunctionOutput.reject_content(
            "Critical rendering failure detected. Do not iterate — "
            "switch technique or escalate."
        )
    return ToolGuardrailFunctionOutput.allow()


@function_tool(
    tool_input_guardrails=[validate_script_against_truth_pack],
    tool_output_guardrails=[check_execution_output],
)
async def execute_blender_script(script_path: str) -> str:
    """Execute a Blender Python script."""
    return await run_blender(script_path)
```

### What Problem It Solves

**Deterministic API validation at $0.** Currently the truth pack validation runs as a separate pipeline step orchestrated by Python. Tool guardrails move this validation INTO the tool itself — every time any agent calls `execute_blender_script`, the script is automatically validated against the truth pack first. If validation fails, the tool is skipped and the agent receives an error message explaining what was wrong.

This is strictly better than the current approach because:
- **Can't be bypassed** — validation runs automatically on every tool call
- **No pipeline coordination needed** — it's part of the tool definition
- **$0 cost** — pure Python, no LLM calls
- **Agent gets actionable feedback** — the rejection message tells it exactly what to fix

Also useful for:
- Output guardrails on quality evaluation (catch known failure patterns before the LLM interprets them)
- Input guardrails on doc search (prevent duplicate queries)

### Where It Integrates

| File | Change |
|------|--------|
| `tools/blender_executor_tools.py` | Wrap `execute_blender_script` with truth pack input guardrail |
| `tools/asset_evaluator_tools.py` | Wrap `evaluate_render` with output guardrail for critical failures |
| `tools/script_generator_tools.py` | Wrap `generate_script` with output guardrail to check script length (< 500 lines = warn) |
| New: `guardrails/tool_guardrails.py` (~100 lines) | Define all tool-level guardrails |

### Implementation Complexity: **Low-Medium**

- Guardrail functions are 10-20 lines each
- Wiring into existing `@function_tool` decorators: 1-2 lines per tool
- Need to ensure `data.context.context.truth_pack` is populated before tools run

### Priority: **P0**

Deterministic validation as a tool wrapper is the cheapest, most reliable way to enforce the truth pack. This replaces an entire pipeline step with a zero-cost guardrail.

---

## 7. run_streamed()

### What It Does

Returns a streaming result object that emits events as the agent runs. Three event types: raw LLM responses, high-level run items, and agent change events.

```python
from agents import Agent, Runner, ItemHelpers
from openai.types.responses import ResponseTextDeltaEvent

async def monitor_script_generation(agent, prompt):
    """Stream script generation with real-time monitoring."""
    result = Runner.run_streamed(agent, prompt)

    generated_text = []
    tool_calls = []

    async for event in result.stream_events():
        if event.type == "raw_response_event":
            if isinstance(event.data, ResponseTextDeltaEvent):
                generated_text.append(event.data.delta)
                # Could detect hallucinated attributes in real-time here

        elif event.type == "run_item_stream_event":
            if event.item.type == "tool_call_item":
                tool_calls.append(event.item)
                print(f"[Monitor] Tool called: {event.item}")
            elif event.item.type == "tool_call_output_item":
                print(f"[Monitor] Tool result: {event.item.output[:100]}")
            elif event.item.type == "message_output_item":
                print(f"[Monitor] Message: {ItemHelpers.text_message_output(event.item)[:200]}")

        elif event.type == "agent_updated_stream_event":
            print(f"[Monitor] Agent changed: {event.new_agent.name}")

    return result.final_output
```

### What Problem It Solves

**Monitoring long-running agent operations.** Pipeline runs take 5-30 minutes per iteration. Currently the user gets no feedback until the iteration completes. Streaming enables:
1. **Real-time progress display** — see what the Script Writer is generating
2. **Early hallucination detection** — scan generated text for known bad patterns mid-stream
3. **Tool call monitoring** — track which tools are called and in what order
4. **Monitoring agent feed** — the proposed monitoring layer could consume stream events

### Where It Integrates

| File | Change |
|------|--------|
| `orchestrator.py:_run_agent()` | Add `_run_agent_streamed()` variant that uses `Runner.run_streamed()` |
| New: `utils/stream_monitor.py` (~100 lines) | Event consumer that logs progress, detects hallucinations, feeds monitoring |
| `orchestrator.py:create_vfx_asset()` | Use streamed variant for Script Writer (longest-running agent) |

### Implementation Complexity: **Medium**

- `Runner.run_streamed()` has the same interface as `Runner.run()` — easy to swap
- The event consumer loop requires async iteration patterns
- Must handle the case where streaming + compaction session interact
- Real-time hallucination detection in the stream is an advanced feature

### Priority: **P2**

Nice for monitoring and UX, but not a reliability or cost fix. The system works without it. Implement after P0/P1 features.

---

## 8. Agent.clone()

### What It Does

Create modified copies of an agent without duplicating the full configuration.

```python
from agents import Agent

# Base script writer with common configuration
base_script_writer = Agent(
    name="Script Writer",
    instructions="...(common instructions)...",
    model="gpt-5.2",
    tools=[generate_script, validate_api, search_docs],
    model_settings=ModelSettings(temperature=0.7),
)

# Effect-type-specific variants
fire_writer = base_script_writer.clone(
    name="Fire Script Writer",
    instructions="...(fire-specific instructions with known fire patterns)...",
)

liquid_writer = base_script_writer.clone(
    name="Liquid Script Writer",
    instructions="...(liquid-specific instructions with collision effector rules)...",
)

rigid_body_writer = base_script_writer.clone(
    name="Rigid Body Script Writer",
    instructions="...(rigid body-specific instructions)...",
)
```

### What Problem It Solves

**Effect-type specialization.** The Script Writer currently uses dynamic instructions to inject effect-type-specific context. This works but puts everything in one agent's instructions. With `clone()`:
- Base agent has common tools and settings
- Cloned variants have effect-type-specific instructions, known patterns, and parameter ranges
- Each clone can have different `model_settings` (e.g., higher temperature for creative effects, lower for precise physics)

Also enables:
- **Escape velocity technique variants** — clone the Research Agent with different instructions for each escape level
- **Budget-aware model switching** — clone agents with cheaper models when budget is low

### Where It Integrates

| File | Change |
|------|--------|
| `specialized_agents/script_writer.py` | Define base agent, create `get_writer_for_effect(effect_type)` factory using `clone()` |
| `orchestrator.py:create_technique_selection_coordinator()` | Clone coordinator with effect-type-specific instructions |
| `orchestrator.py` (technique switch) | When switching technique, clone Script Writer with new technique's instructions |

### Implementation Complexity: **Low**

- `clone()` is a single method call
- The work is in crafting good per-effect-type instructions
- No architectural changes needed

### Priority: **P1**

Useful for specialization, but dynamic instructions already solve this problem adequately. Clone is cleaner but not urgent.

---

## 9. reset_tool_choice

### What It Does

Boolean (default `True`) that automatically resets `tool_choice` to `"auto"` after a tool call, preventing infinite tool-use loops.

```python
agent = Agent(
    name="Executor",
    model_settings=ModelSettings(tool_choice="execute_blender_script"),
    reset_tool_choice=True,  # After calling the tool, resets to "auto"
)
```

### What Problem It Solves

**Preventing tool-use loops.** The orchestrator already uses `max_turns` to prevent runaway agent loops. `reset_tool_choice` is a complementary safeguard — if an agent is configured with `tool_choice="required"` or a specific tool name, it won't loop calling the same tool forever.

### Where It Integrates

Already defaults to `True` on all agents. **No changes needed** unless we add agents with forced tool choice (which we should for deterministic agents).

For deterministic agents (Executor, API Validator), combine with `tool_use_behavior="stop_on_first_tool"`:

```python
executor = Agent(
    name="Executor",
    model_settings=ModelSettings(tool_choice="execute_blender_script"),
    tool_use_behavior="stop_on_first_tool",  # Output IS the result
    reset_tool_choice=True,  # Safety: no loops if stop_on_first_tool fails
)
```

### Implementation Complexity: **Trivial**

Already enabled by default. Just verify it's not accidentally disabled.

### Priority: **P2**

Already working. Only relevant when adding forced tool choice to agents.

---

## 10. prompt_cache_retention

### What It Does

Cache system prompts in memory or for 24 hours, reducing input token costs for agents whose instructions rarely change.

```python
from agents import Agent, ModelSettings

research_agent = Agent(
    name="Research Agent",
    instructions="...(long, stable system prompt)...",
    model_settings=ModelSettings(
        prompt_cache_retention="24h",  # Cache for 24 hours
    ),
)
```

### What Problem It Solves

**Reducing input token costs for stable agents.** The Research Agent, Docs Expert, and Technique Selector have long system prompts that rarely change between calls. Caching these prompts server-side means OpenAI doesn't charge full input token price on subsequent calls.

OpenAI's prompt caching provides ~50% discount on cached input tokens. If a research agent's system prompt is 2000 tokens and it runs 5 times per pipeline run, that's 8000 tokens saved (cached instead of re-processed).

### Where It Integrates

| File | Change |
|------|--------|
| `specialized_agents/docs_expert.py` | Add `prompt_cache_retention="24h"` to ModelSettings |
| `specialized_agents/script_writer.py` | **Do NOT cache** — instructions change per iteration (truth pack injection) |
| `orchestrator.py:create_technique_selection_coordinator()` | Add `prompt_cache_retention="24h"` to ModelSettings |
| `orchestrator.py:create_modification_coordinator()` | Add `prompt_cache_retention="in_memory"` (changes more often) |

### Implementation Complexity: **Trivial**

1 line per agent. No code changes.

### Priority: **P2**

Small cost savings (~$0.01-0.02/run). Nice optimization but not impactful at current scale.

---

## 11. parallel_tool_calls in ModelSettings

### What It Does

Allows the LLM to emit multiple tool calls in a single turn, which are then executed concurrently.

```python
from agents import Agent, ModelSettings

research_agent = Agent(
    name="Research Agent",
    model_settings=ModelSettings(
        parallel_tool_calls=True,  # Allow multiple tools per turn
    ),
    tools=[search_blender_docs, search_knowledge_base, search_code_patterns],
)
```

### What Problem It Solves

**Research latency.** The Research Agent currently calls tools sequentially: search docs → search KB → search patterns. With parallel tool calls, the LLM can emit all three searches in one turn, and they execute concurrently. This reduces research phase latency by ~60%.

**Already partially implemented.** The orchestrator uses `asyncio.gather()` for Python-level parallelism (e.g., `_run_parallel_preflight`). `parallel_tool_calls` adds LLM-level parallelism — the model decides which tools to call in parallel.

### Where It Integrates

| File | Change |
|------|--------|
| All agent definitions | Add `parallel_tool_calls=True` to `ModelSettings` where safe |
| `specialized_agents/docs_expert.py` | Enable — doc searches are independent |
| `specialized_agents/quality_analyst.py` | Enable — metrics + vision can run in parallel |
| `specialized_agents/executor.py` | **Do NOT enable** — execution must be sequential |

### Implementation Complexity: **Trivial**

1 line per agent. No code changes. The SDK handles parallel execution automatically.

**Caution:** Some tools have side effects that depend on order. Only enable for agents whose tools are truly independent.

### Priority: **P2**

Latency improvement, not reliability or cost. The default (provider decides) is usually fine.

---

## Bonus Features

### B1. Function Tool Timeouts (Added in v0.9.0)

```python
@function_tool(timeout=120.0)  # 2 minute timeout
async def execute_blender_script(script_path: str) -> str:
    """Execute a Blender Python script with timeout."""
    return await run_blender(script_path)
```

Timeout behaviors:
- `"error_as_result"` (default) — returns timeout message to the model
- `"raise_exception"` — raises `ToolTimeoutError`, fails the run

**Where it helps:** Blender execution can hang on complex simulations. A 5-minute timeout prevents infinite waits.

**Integration:** `tools/blender_executor_tools.py` — add `timeout=300.0` to the execute tool.

**Priority:** P1 — prevents hung pipelines.

### B2. Custom Tool Error Functions (v0.9.0+)

```python
def blender_error_handler(context, error):
    """Provide actionable error messages for Blender execution failures."""
    if "ModuleNotFoundError" in str(error):
        return "Blender Python environment missing module. Check script imports."
    if "MemoryError" in str(error):
        return "Blender ran out of memory. Reduce resolution_max or particle count."
    return f"Blender execution failed: {error}. Check script for API errors."

@function_tool(failure_error_function=blender_error_handler)
async def execute_blender_script(script_path: str) -> str:
    return await run_blender(script_path)
```

**Where it helps:** Currently tool errors produce generic Python tracebacks that confuse the LLM. Custom error functions translate exceptions into actionable feedback.

**Priority:** P1 — improves error recovery quality.

---

## Version Delta: v0.9.0 → v0.9.3

| Version | Key Changes |
|---------|-------------|
| **v0.9.0** (Feb 13) | Dropped Python 3.9. Added function tool timeouts (`timeout`, `timeout_behavior`, `timeout_error_function`). Added `ToolOutputTrimmer`. Narrowed `Agent.as_tool()` return type to `FunctionTool`. |
| **v0.9.1** (Feb 17) | Fixed iterable input history. Added tracing spans for shell/patch/computer tools. Fixed fork hazards in tracing. Improved nested agent-tool state persistence for JSON serialization. Bumped RunState schema to v1.1. |
| **v0.9.2** (Feb 19) | Added `reasoning_item_id_policy: 'omit'` for reasoning model 400 errors. Fixed reasoning item filtering from nested handoff inputs. |
| **v0.9.3** (Feb 20) | Fixed `total_tokens` field in tracing usage payloads. |

**Key takeaway:** v0.9.0 was the feature release. v0.9.1-v0.9.3 are stability patches. The features documented in this analysis (sessions, guardrails, tool_use_behavior, etc.) were available before v0.9.0 and are stable.

---

## Implementation Priority Matrix

### P0 — Ship with Phase 2 (immediate impact, low effort)

| Feature | Effort | Impact | Cost Savings |
|---------|--------|--------|-------------|
| `call_model_input_filter` | Low (~80 lines) | Fixes context bloat (#3 problem) | $0 (filter is free) |
| `tool_use_behavior` | Trivial (1 line/agent) | Eliminates unnecessary LLM calls | ~$0.05-0.10/run |
| `is_enabled` | Low (~20 lines) | Clean budget-aware tool hiding | Prevents budget waste |
| Tool Guardrails | Low-Med (~100 lines) | Deterministic truth pack enforcement on every tool call | Replaces LLM validation |

**Estimated total effort:** 200-300 lines of new/modified code
**Estimated monthly savings:** $2-5/month (10-25% of budget)

### P1 — Ship next (high value, medium effort)

| Feature | Effort | Impact |
|---------|--------|--------|
| `AdvancedSQLiteSession` | Medium (~170 lines) | Branching for technique experiments, built-in token tracking |
| `needs_approval` (HITL) | Medium-High (~200 lines) | Native pipeline pauses with state serialization |
| `Agent.clone()` | Low (~30 lines) | Effect-type specialization |
| Function tool timeouts | Trivial (1 line) | Prevents hung Blender executions |
| Custom error functions | Low (~50 lines) | Better error recovery feedback |

**Estimated total effort:** 400-500 lines of new/modified code

### P2 — Nice to have (optimization, not critical)

| Feature | Effort | Impact |
|---------|--------|--------|
| `run_streamed()` | Medium (~200 lines) | Real-time monitoring, hallucination detection |
| `prompt_cache_retention` | Trivial (1 line/agent) | Small input token savings |
| `parallel_tool_calls` | Trivial (1 line/agent) | Latency reduction for research |
| `reset_tool_choice` | None (already default) | Loop prevention (already working) |

---

## Recommended Implementation Sequence

```
Week 1: P0 Features
├── [1] is_enabled on expensive tools (30 min)
├── [2] tool_use_behavior on Executor, Learning Agent, API Validator (30 min)
├── [3] call_model_input_filter with VFX-specific trimmer (2-3 hours)
└── [4] Tool guardrails for truth pack validation (3-4 hours)

Week 2: P1 Features
├── [5] Function tool timeouts on Blender executor (15 min)
├── [6] Custom error functions for Blender tools (1-2 hours)
├── [7] Agent.clone() for effect-type specialization (2-3 hours)
├── [8] AdvancedSQLiteSession migration (4-6 hours)
└── [9] needs_approval HITL framework (6-8 hours)

Week 3+: P2 Features (as needed)
├── [10] run_streamed() for monitoring layer
├── [11] prompt_cache_retention optimization
└── [12] parallel_tool_calls tuning
```

---

## Appendix: Current SDK Usage in Orchestrator

| SDK Feature | Status | Where |
|-------------|--------|-------|
| `Agent` | In use | All agents |
| `Runner.run()` | In use | `orchestrator.py:_run_agent()` |
| `function_tool` | In use | All tools |
| `Agent.as_tool()` | In use | Coordinator agents |
| `ModelSettings` | In use | All agent definitions |
| `SQLiteSession` | In use | `orchestrator.py:get_or_create_sdk_session()` |
| `OpenAIResponsesCompactionSession` | In use | Session management |
| `RunHooks` (EnforcementHooks) | In use | Loop detection, doc query enforcement |
| Input/Output Guardrails | In use | Research, Script, Quality, Coordinator agents |
| Structured outputs (Pydantic) | In use | All agent output types |
| `trace()` | In use | Pipeline tracing |
| `RunConfig` | Partial | Supported in `_run_agent()` but rarely used |
| Dynamic instructions | In use | `tools/dynamic_instructions.py` |
| `RunContextWrapper` | In use | SharedContext injection |
