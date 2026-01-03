# Migration Plan: Claude Agent SDK → Direct Anthropic API

**Date:** 2026-01-02
**Status:** PLANNING
**Goal:** Eliminate double-payment (Max subscription + API keys) by removing Claude Agent SDK dependency and using direct Anthropic API calls.

---

## Executive Summary

The blender-orchestrator currently uses the Claude Agent SDK, but:
1. **Already bypasses** the SDK's core feature (inner agent tool execution) via `_direct_mode = True`
2. **All sophisticated features** (workflow, autonomy, guardrails, persistence) are custom-built
3. **SDK caused problems** (unsupported `cwd` field, naming mismatches, $7 debugging cost)
4. **Double payment** - Max subscription + API keys required

**Recommendation:** Remove SDK, use direct Anthropic API, keep all custom orchestration logic.

**Estimated effort:** 4-6 hours
**Risk level:** Low (direct mode already proves the architecture works without SDK inner agent)

---

## Current Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TEST SCRIPT (test_explosion.py)              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              BLENDER ORCHESTRATOR AGENT                         │
│                  (orchestrator.py)                              │
│                                                                 │
│  Custom Components (KEEP ALL):                                  │
│  ├── BlenderOrchestratorAgent  - Main orchestrator              │
│  ├── AutonomyController        - Trust score management         │
│  ├── TokenGuardrails           - Cost/token limit enforcement   │
│  ├── WorkflowStateMachine      - Stage transitions              │
│  └── SessionManager            - State persistence              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ Uses Claude Agent SDK 0.1.18
                                │ (ClaudeSDKClient, ClaudeAgentOptions)
                                │ ⚠️ BYPASSED via _direct_mode = True
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 INNER CLAUDE AGENT (SDK)                        │
│           ❌ NOT ACTUALLY USED - direct mode active             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MCP SERVERS (6 total)                        │
│  ✅ Called directly as Python modules (not via SDK)            │
│                                                                 │
│  script-generator, blender-executor, asset-evaluator,          │
│  experiment-tracker, blender-manual, iteration-controller      │
└─────────────────────────────────────────────────────────────────┘
```

### Current SDK Usage (orchestrator.py)

```python
# Lines 49-54: SDK imports
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

# Lines 373-420: SDK client creation
def create_options(self) -> ClaudeAgentOptions:
    options = ClaudeAgentOptions(
        cwd=str(self.project_root),
        system_prompt=self._create_system_prompt(),
        mcp_servers=self._create_mcp_config(),
        allowed_tools=self._create_allowed_tools(),
        max_budget_usd=max_budget,
        permission_mode='acceptEdits',
    )
    return options

# Lines 1088-1150: Query method (BYPASSED when _direct_mode=True)
async def _query_agent(self, prompt: str) -> str:
    if self._direct_mode:
        return await self._query_direct(prompt)  # ← ACTUAL PATH USED
    # SDK path below is dead code when _direct_mode=True
    await self.client.query(prompt)
    ...
```

### What _direct_mode Already Does

```python
# Lines 245-280: Direct tool loading
async def _load_direct_tools(self):
    """Load MCP server modules directly for hybrid mode."""
    # Imports script_generator, blender_executor, etc. as Python modules
    # Calls their functions directly, bypassing JSON-RPC entirely

# Lines 1050-1080: Direct tool execution
async def _call_tool_direct(self, tool_name: str, **kwargs) -> Any:
    """Call MCP tool directly as Python function."""
    # e.g., script_generator.generate_script(**kwargs)
```

---

## Target Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TEST SCRIPT (test_explosion.py)              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              BLENDER ORCHESTRATOR AGENT                         │
│                  (orchestrator.py)                              │
│                                                                 │
│  Custom Components (UNCHANGED):                                 │
│  ├── BlenderOrchestratorAgent  - Main orchestrator              │
│  ├── AutonomyController        - Trust score management         │
│  ├── TokenGuardrails           - Cost/token limit enforcement   │
│  ├── WorkflowStateMachine      - Stage transitions              │
│  └── SessionManager            - State persistence              │
│                                                                 │
│  NEW: AnthropicClient wrapper (replaces SDK)                    │
│  ├── Direct API calls for LLM reasoning                         │
│  ├── Tool definitions from MCP servers                          │
│  └── Tool result handling                                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ Direct Anthropic API
                                │ (AsyncAnthropic from anthropic package)
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 ANTHROPIC API (claude-sonnet-4-20250514)               │
│           ✅ Direct calls, no SDK overhead                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MCP SERVERS (6 total)                        │
│  ✅ Called directly as Python modules (UNCHANGED)              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Migration Steps

### Phase 1: Create Anthropic Client Wrapper (NEW FILE)

**New file:** `agents/blender-orchestrator/anthropic_client.py`

```python
"""
Lightweight Anthropic API wrapper replacing Claude Agent SDK.

This provides the minimal LLM interaction needed for orchestration
decisions without the overhead of the full SDK.
"""

import os
from typing import Optional, List, Dict, Any
from anthropic import AsyncAnthropic
import logging

logger = logging.getLogger(__name__)


class AnthropicOrchestrationClient:
    """
    Direct Anthropic API client for orchestration decisions.

    Replaces ClaudeSDKClient with:
    - Direct API calls (no inner agent abstraction)
    - Simple tool definition format
    - Streaming support for progress feedback
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
    ):
        self.client = AsyncAnthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.conversation_history: List[Dict[str, Any]] = []

        # Token tracking for guardrails integration
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    async def query(
        self,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Send a query to Claude and get a response.

        Args:
            prompt: The user message
            tools: Optional list of tool definitions
            temperature: Sampling temperature

        Returns:
            Response dict with 'content', 'tool_calls', and 'usage'
        """
        messages = self.conversation_history + [
            {"role": "user", "content": prompt}
        ]

        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "temperature": temperature,
        }

        if self.system_prompt:
            kwargs["system"] = self.system_prompt

        if tools:
            kwargs["tools"] = tools

        try:
            response = await self.client.messages.create(**kwargs)

            # Track token usage
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens

            # Extract content and tool calls
            result = {
                "content": "",
                "tool_calls": [],
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                "stop_reason": response.stop_reason,
            }

            for block in response.content:
                if block.type == "text":
                    result["content"] += block.text
                elif block.type == "tool_use":
                    result["tool_calls"].append({
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

            # Update conversation history
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({
                "role": "assistant",
                "content": response.content,
            })

            return result

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    async def submit_tool_result(
        self,
        tool_use_id: str,
        result: str,
    ) -> Dict[str, Any]:
        """
        Submit a tool result and get the next response.

        Args:
            tool_use_id: The ID from the tool_use block
            result: The tool execution result (as string)

        Returns:
            Response dict with 'content', 'tool_calls', and 'usage'
        """
        # Add tool result to history
        self.conversation_history.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": result,
            }],
        })

        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self.conversation_history,
        }

        if self.system_prompt:
            kwargs["system"] = self.system_prompt

        response = await self.client.messages.create(**kwargs)

        # Track tokens
        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens

        # Extract content
        result_dict = {
            "content": "",
            "tool_calls": [],
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            "stop_reason": response.stop_reason,
        }

        for block in response.content:
            if block.type == "text":
                result_dict["content"] += block.text
            elif block.type == "tool_use":
                result_dict["tool_calls"].append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        # Update history
        self.conversation_history.append({
            "role": "assistant",
            "content": response.content,
        })

        return result_dict

    def get_token_usage(self) -> Dict[str, int]:
        """Get cumulative token usage for guardrails."""
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
        }

    def reset_conversation(self):
        """Clear conversation history for new session."""
        self.conversation_history = []

    def estimate_cost(self, input_per_million: float = 3.0, output_per_million: float = 15.0) -> float:
        """
        Estimate cost based on token usage.

        Default pricing is for Claude Sonnet (cheaper than Opus).
        """
        input_cost = (self.total_input_tokens / 1_000_000) * input_per_million
        output_cost = (self.total_output_tokens / 1_000_000) * output_per_million
        return input_cost + output_cost
```

---

### Phase 2: Update orchestrator.py

**Changes to `orchestrator.py`:**

#### 2.1 Replace SDK imports (lines 49-54)

```python
# BEFORE:
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

# AFTER:
from .anthropic_client import AnthropicOrchestrationClient
```

#### 2.2 Update __init__ method (lines ~150-200)

```python
# BEFORE:
self.client: Optional[ClaudeSDKClient] = None

# AFTER:
self.client: Optional[AnthropicOrchestrationClient] = None
```

#### 2.3 Replace create_options() with create_client() (lines 373-420)

```python
# BEFORE:
def create_options(self) -> ClaudeAgentOptions:
    options = ClaudeAgentOptions(
        cwd=str(self.project_root),
        system_prompt=self._create_system_prompt(),
        mcp_servers=self._create_mcp_config(),
        allowed_tools=self._create_allowed_tools(),
        max_budget_usd=max_budget,
        permission_mode='acceptEdits',
    )
    return options

# AFTER:
def create_client(self) -> AnthropicOrchestrationClient:
    """Create direct Anthropic API client."""
    return AnthropicOrchestrationClient(
        model="claude-sonnet-4-20250514",  # Cheaper than Opus, sufficient for orchestration
        max_tokens=4096,
        system_prompt=self._create_system_prompt(),
    )
```

#### 2.4 Remove SDK-specific methods

**DELETE these methods (no longer needed):**
- `_create_mcp_config()` - MCP servers loaded directly
- `_create_hooks()` - SDK-specific hooks
- Lines related to `ClaudeAgentOptions`

#### 2.5 Simplify start() method (lines ~450-500)

```python
# BEFORE:
async def start(self):
    options = self.create_options()
    self.client = ClaudeSDKClient(options=options)
    await self.client.__aenter__()
    if self._direct_mode:
        await self._load_direct_tools()

# AFTER:
async def start(self):
    """Initialize orchestrator."""
    self.client = self.create_client()
    await self._load_direct_tools()  # Always load direct tools
    logger.info("Orchestrator started with direct Anthropic API")
```

#### 2.6 Simplify stop() method

```python
# BEFORE:
async def stop(self):
    if self.client:
        await self.client.__aexit__(None, None, None)

# AFTER:
async def stop(self):
    """Cleanup orchestrator."""
    if self.client:
        usage = self.client.get_token_usage()
        cost = self.client.estimate_cost()
        logger.info(f"Session complete. Tokens: {usage['total_tokens']}, Est. cost: ${cost:.4f}")
    self.client = None
```

#### 2.7 Update _query_agent() to use new client (lines 1088-1150)

```python
# BEFORE (with SDK):
async def _query_agent(self, prompt: str) -> str:
    if self._direct_mode:
        return await self._query_direct(prompt)
    await self.client.query(prompt)
    full_response = ""
    async for message in self.client.receive_response():
        full_response += str(message)
    return full_response

# AFTER (direct API):
async def _query_agent(self, prompt: str) -> str:
    """
    Query Claude for orchestration decisions.

    Uses direct Anthropic API - no SDK inner agent.
    Tool execution happens via _call_tool_direct().
    """
    # Check guardrails before query
    can_proceed, message = self.guardrails.can_proceed(estimated_tokens=1000)
    if not can_proceed:
        raise RuntimeError(f"Guardrails blocked query: {message}")

    # Build tool definitions for Claude (optional - for reasoning about tools)
    tools = self._build_tool_definitions() if self._include_tools_in_prompt else None

    # Query Claude
    response = await self.client.query(prompt, tools=tools)

    # Update guardrails with actual usage
    usage = response["usage"]
    self.guardrails.record_usage(
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
    )

    # Handle any tool calls Claude requested
    while response["tool_calls"]:
        for tool_call in response["tool_calls"]:
            tool_name = tool_call["name"]
            tool_input = tool_call["input"]

            logger.info(f"Executing tool: {tool_name}")
            try:
                result = await self._call_tool_direct(tool_name, **tool_input)
                result_str = str(result) if not isinstance(result, str) else result
            except Exception as e:
                result_str = f"Error: {e}"
                logger.error(f"Tool {tool_name} failed: {e}")

            # Submit result back to Claude
            response = await self.client.submit_tool_result(
                tool_use_id=tool_call["id"],
                result=result_str,
            )

            # Update guardrails
            self.guardrails.record_usage(
                input_tokens=response["usage"]["input_tokens"],
                output_tokens=response["usage"]["output_tokens"],
            )

    return response["content"]
```

#### 2.8 Add tool definition builder (new method)

```python
def _build_tool_definitions(self) -> List[Dict[str, Any]]:
    """
    Build Anthropic tool definitions from loaded MCP tools.

    This allows Claude to reason about available tools without
    the SDK's MCP server abstraction.
    """
    tools = []

    # Define tools based on what's loaded in _direct_tools
    tool_schemas = {
        "generate_script": {
            "name": "generate_script",
            "description": "Generate a Blender Python script for VFX",
            "input_schema": {
                "type": "object",
                "properties": {
                    "effect_type": {"type": "string", "description": "Type of effect (pyro, explosion, fire, etc.)"},
                    "description": {"type": "string", "description": "Description of the effect"},
                    "output_name": {"type": "string", "description": "Name for output script"},
                    "resolution": {"type": "integer", "default": 96},
                    "frame_end": {"type": "integer", "default": 50},
                },
                "required": ["effect_type", "description", "output_name"],
            },
        },
        "execute_blender_script": {
            "name": "execute_blender_script",
            "description": "Execute a Blender Python script to generate VDB assets",
            "input_schema": {
                "type": "object",
                "properties": {
                    "script_path": {"type": "string", "description": "Path to .py script"},
                    "timeout_seconds": {"type": "integer", "default": 600},
                },
                "required": ["script_path"],
            },
        },
        "evaluate_vfx_quality": {
            "name": "evaluate_vfx_quality",
            "description": "Evaluate VFX quality of a rendered image (0-100 score)",
            "input_schema": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path to rendered image"},
                    "effect_type": {"type": "string", "default": "explosion"},
                },
                "required": ["image_path"],
            },
        },
        # Add more as needed...
    }

    for name, schema in tool_schemas.items():
        if name in self._direct_tools or f"mcp__script-generator__{name}" in str(self._direct_tools):
            tools.append(schema)

    return tools
```

---

### Phase 3: Update requirements.txt

**File:** `agents/blender-orchestrator/requirements.txt`

```diff
# REMOVE:
- claude-agent-sdk>=0.1.17

# ADD (if not present):
+ anthropic>=0.40.0
```

---

### Phase 4: Update tests

**File:** `agents/blender-orchestrator/test_mcp_simple.py`

Update to test direct API instead of SDK:

```python
#!/usr/bin/env python3
"""
Minimal test for direct Anthropic API orchestration.
"""

import asyncio
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.blender_orchestrator.anthropic_client import AnthropicOrchestrationClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_direct_api():
    """Test direct Anthropic API call."""
    client = AnthropicOrchestrationClient(
        model="claude-sonnet-4-20250514",
        system_prompt="You are a test agent. Respond briefly.",
    )

    response = await client.query("What is 2 + 2? Answer in one word.")

    logger.info(f"Response: {response['content']}")
    logger.info(f"Tokens: {response['usage']}")
    logger.info(f"Total cost: ${client.estimate_cost():.6f}")

    return "four" in response["content"].lower()


def main():
    result = asyncio.run(test_direct_api())
    if result:
        logger.info("TEST PASSED")
        sys.exit(0)
    else:
        logger.error("TEST FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

---

### Phase 5: Remove SDK-specific code

**Delete or comment out:**
1. All `ClaudeAgentOptions` references
2. `_create_hooks()` method
3. `_create_mcp_config()` method (MCP stdio config)
4. SDK version checks
5. `permission_mode`, `enable_file_checkpointing`, `enable_session_forking` (SDK features)

**Keep:**
1. `_direct_mode` flag (can be removed later, but harmless)
2. `_load_direct_tools()` - still needed
3. `_call_tool_direct()` - still needed
4. All custom systems (workflow, autonomy, guardrails, state)

---

## Files Changed Summary

| File | Action | Changes |
|------|--------|---------|
| `anthropic_client.py` | **CREATE** | New lightweight API wrapper |
| `orchestrator.py` | **MODIFY** | Replace SDK with direct API |
| `requirements.txt` | **MODIFY** | Remove SDK, ensure anthropic present |
| `test_mcp_simple.py` | **MODIFY** | Update test for direct API |
| `test_explosion.py` | **MODIFY** | Minor updates if needed |

---

## Rollback Plan

If issues arise:
1. Revert `orchestrator.py` changes
2. Restore `claude-agent-sdk` to requirements.txt
3. Delete `anthropic_client.py`

The `_direct_mode` architecture means the current code is already close to the target state.

---

## Success Criteria

1. **Tests pass:** `test_mcp_simple.py` and `test_explosion.py` complete successfully
2. **No SDK dependency:** `pip show claude-agent-sdk` returns nothing
3. **Cost reduction:** API calls use Sonnet pricing ($3/$15) instead of Opus ($15/$75)
4. **Full workflow works:** Asset generation completes through all stages
5. **Guardrails work:** Token limits enforced correctly
6. **State persistence works:** Sessions save/load correctly

---

## Cost Comparison

| Scenario | SDK (Opus) | Direct API (Sonnet) | Savings |
|----------|------------|---------------------|---------|
| 10K tokens in, 2K out | $0.30 | $0.06 | 80% |
| Full asset gen (~50K tokens) | $1.50 | $0.30 | 80% |
| Daily limit ($20) | 13 sessions | 66 sessions | 5x more |

**Note:** Can still use Opus for complex decisions if needed, but Sonnet is sufficient for orchestration logic.

---

## Timeline

| Phase | Estimated Time | Dependencies |
|-------|---------------|--------------|
| Phase 1: Create anthropic_client.py | 30 min | None |
| Phase 2: Update orchestrator.py | 2 hours | Phase 1 |
| Phase 3: Update requirements.txt | 5 min | None |
| Phase 4: Update tests | 30 min | Phases 1-2 |
| Phase 5: Remove SDK code | 1 hour | Phases 1-4 |
| Testing & debugging | 1-2 hours | All phases |

**Total:** 4-6 hours

---

## Questions to Resolve Before Starting

1. **Model choice:** Use Sonnet for all orchestration, or Opus for complex decisions?
2. **Max subscription API access:** Verify Max subscription includes API access (check anthropic.com account)
3. **Tool definitions:** Include full tool schemas in prompts, or rely on direct tool calls only?
4. **Streaming:** Need streaming responses for progress feedback, or batch responses sufficient?

---

**End of Migration Plan**
