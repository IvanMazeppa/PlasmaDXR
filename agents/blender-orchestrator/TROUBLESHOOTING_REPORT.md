# Blender VFX Orchestrator - Troubleshooting Report

**Date:** 2026-01-01
**Status:** BLOCKED - Inner agent not executing MCP tools
**Cost Burned:** ~$7 in failed test runs

---

## 1. System Architecture Overview

### 1.1 What This System Is

The **Blender VFX Orchestrator** is a Claude Agent SDK agent that autonomously generates volumetric VFX assets (explosions, fire, smoke, etc.) using Blender's Mantaflow simulation system. It coordinates multiple MCP (Model Context Protocol) servers to:

1. Generate Blender Python scripts
2. Execute those scripts in Blender
3. Evaluate the visual quality of rendered outputs
4. Iterate with parameter adjustments until quality thresholds are met

### 1.2 Technology Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    TEST SCRIPT (test_explosion.py)              │
│                         Python asyncio                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              BLENDER ORCHESTRATOR AGENT                         │
│                  (orchestrator.py)                              │
│                                                                 │
│  Components:                                                    │
│  ├── BlenderOrchestratorAgent  - Main agent class              │
│  ├── AutonomyController        - Trust score management        │
│  ├── TokenGuardrails           - Cost/token limit enforcement  │
│  ├── WorkflowStateMachine      - Stage transitions             │
│  └── SessionManager            - State persistence             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ Uses Claude Agent SDK 0.1.18
                                │ (ClaudeSDKClient, ClaudeAgentOptions)
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 INNER CLAUDE AGENT                              │
│           (Bundled Claude Code CLI via SDK)                     │
│                                                                 │
│  This is the actual LLM that receives prompts and should       │
│  call MCP tools. Created by ClaudeSDKClient.                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ Should call MCP tools via
                                │ mcp_servers configuration
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MCP SERVERS (6 total)                        │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ script-generator│  │ blender-executor│  │ asset-evaluator │ │
│  │                 │  │                 │  │                 │ │
│  │ • generate_     │  │ • execute_      │  │ • evaluate_     │ │
│  │   script()      │  │   blender_      │  │   vfx_quality() │ │
│  │ • modify_       │  │   script()      │  │ • compare_vfx_  │ │
│  │   script()      │  │ • parse_        │  │   iterations()  │ │
│  │ • validate_     │  │   blender_      │  │                 │ │
│  │   parameters()  │  │   errors()      │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │experiment-      │  │iteration-       │  │ blender-manual  │ │
│  │tracker          │  │controller       │  │                 │ │
│  │                 │  │                 │  │ • search_       │ │
│  │ • record_       │  │ • diagnose_     │  │   python_api()  │ │
│  │   experiment_   │  │   vfx_issues()  │  │ • search_       │ │
│  │   result()      │  │ • get_next_     │  │   nodes()       │ │
│  │ • suggest_      │  │   iteration_    │  │                 │ │
│  │   experiments() │  │   params()      │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Intended Workflow

The orchestrator follows a state machine with these stages:

```
SESSION_START
     │
     ▼
GENERATE_SCRIPT  ◄──────────────────┐
     │                              │
     ▼                              │
EXECUTE_BLENDER                     │
     │                              │
     ▼                              │
EVALUATE_QUALITY                    │
     │                              │
     ▼                              │
DECIDE_NEXT_ACTION ─────────────────┤
     │                              │
     │ (if score < threshold        │
     │  and iterations < max)       │
     │                              │
     ▼                              │
SESSION_END (if passed or max iterations)
```

**Stage Details:**

1. **SESSION_START**: Initialize session, load configs
2. **GENERATE_SCRIPT**: Query inner agent to call `script-generator.generate_script()`
3. **EXECUTE_BLENDER**: Query inner agent to call `blender-executor.execute_blender_script()`
4. **EVALUATE_QUALITY**: Query inner agent to call `asset-evaluator.evaluate_vfx_quality()`
5. **DECIDE_NEXT_ACTION**: Check scores, decide to iterate or end
6. **ERROR_RECOVERY**: Handle failures with optional file rewind
7. **SESSION_END**: Finalize and save results

### 1.4 How the Orchestrator Queries the Inner Agent

The orchestrator uses `_query_agent()` method to send prompts to the inner Claude agent:

```python
async def _query_agent(self, prompt: str) -> str:
    """Send query to Claude agent and get response."""
    await self.client.query(prompt)

    full_response = ""
    async for message in self.client.receive_response():
        message_text = str(message)
        full_response += message_text

    return full_response
```

The prompt tells the inner agent what MCP tools to call:

```python
prompt = f"""Generate initial Blender script for:

Asset: {asset_name}
Effect Type: pyro
Description: A dramatic fiery explosion...

Actions:
1. Use script-generator.generate_script() to create the Blender Python script
2. Validate parameters using validate_parameters()
3. Report the script path and key parameters
"""
```

### 1.5 MCP Server Configuration

The orchestrator configures MCP servers in `_create_mcp_config()`:

```python
def _create_mcp_config(self) -> List[Dict[str, Any]]:
    """Create MCP server configuration for Claude Agent SDK."""
    servers = []
    mcp_config = self.config.get("mcp_servers", {})

    for name, settings in mcp_config.items():
        if settings.get("enabled", True):
            server_cwd = self.project_root / settings.get("cwd", "")
            servers.append({
                "name": name.replace("_", "-"),
                "command": str(server_cwd / settings.get("command", "./run_server.sh")),
                "cwd": str(server_cwd),
                "timeout": settings.get("timeout", 60)
            })

    return servers
```

---

## 2. The Current Problem

### 2.1 Symptom

The orchestrator is stuck in an infinite loop between `GENERATE_SCRIPT` and `ERROR_RECOVERY`:

```
04:00:07 - Workflow transition: generate_script -> error_recovery (outcome=failure)
04:00:29 - Workflow transition: error_recovery -> generate_script (outcome=success)
04:00:45 - Workflow transition: generate_script -> error_recovery (outcome=failure)
04:01:00 - Workflow transition: error_recovery -> generate_script (outcome=success)
... (repeats until $5 budget exhausted)
```

### 2.2 Root Cause Analysis

The `_stage_generate_script()` method returns `FAILURE` because `_extract_script_path()` returns `None`:

```python
async def _stage_generate_script(self) -> StageResult:
    prompt = f"""Generate initial Blender script for:
    ...
    Actions:
    1. Use script-generator.generate_script() to create the Blender Python script
    ...
    """

    response = await self._query_agent(prompt)

    # THIS IS RETURNING None BECAUSE THE RESPONSE DOESN'T CONTAIN A SCRIPT PATH
    script_path = self._extract_script_path(response)

    return StageResult(
        outcome=WorkflowOutcome.SUCCESS if script_path else WorkflowOutcome.FAILURE,
        error="Failed to extract script path" if not script_path else None
    )
```

The `_extract_script_path()` method looks for patterns like:
- `Script path: /path/to/script.py`
- `Generated: /path/to/script.py`
- `saved to /path/to/script.py`
- `blender_scripts/*.py`

### 2.3 Why the Response Doesn't Contain a Script Path

**Hypothesis 1: Inner agent is not calling MCP tools**

The inner Claude agent (spawned by Claude Agent SDK) might not be actually invoking the MCP tools. It might just be responding with text like "I will call script-generator.generate_script()..." without actually calling it.

**Hypothesis 2: MCP servers are not starting**

The MCP server configuration might be incorrect:
- Wrong paths to `run_server.sh`
- Missing virtual environments
- Port conflicts

**Hypothesis 3: SDK client configuration issue**

The `ClaudeAgentOptions` might be missing required fields for MCP tool execution.

### 2.4 Evidence

1. **No MCP tool call logs**: The orchestrator logs show stage transitions but no evidence of actual MCP tool invocations.

2. **Response content unknown**: The `_query_agent()` method returns the response but we're not logging it, so we can't see what the inner agent is actually saying.

3. **Rapid failures**: Each generate_script attempt fails in ~15 seconds, which is too fast for actual Blender script generation + execution.

---

## 3. SDK Configuration Details

### 3.1 Current ClaudeAgentOptions

```python
def create_options(self) -> ClaudeAgentOptions:
    options = ClaudeAgentOptions(
        cwd=str(self.project_root),
        system_prompt=self._create_system_prompt(),
        mcp_servers=self._create_mcp_config(),
        allowed_tools=self._create_allowed_tools(),
        max_budget_usd=max_budget,           # SDK 0.1.6+
        max_thinking_tokens=max_thinking,     # SDK 0.1.6+
        enable_file_checkpointing=True,       # SDK 0.1.15+
        hooks=self._create_hooks(),           # SDK 0.1.3+
        permission_mode='acceptEdits',
    )
    return options
```

### 3.2 MCP Server Config from config.yaml

```yaml
mcp_servers:
  script_generator:
    enabled: true
    command: "./run_server.sh"
    cwd: "agents/script-generator"
    timeout: 30

  blender_executor:
    enabled: true
    command: "./run_server.sh"
    cwd: "agents/blender-executor"
    timeout: 900

  # ... etc
```

### 3.3 Allowed Tools List

```python
def _create_allowed_tools(self) -> List[str]:
    return [
        # script-generator
        "mcp__script-generator__generate_script",
        "mcp__script-generator__modify_script",
        "mcp__script-generator__list_techniques",
        "mcp__script-generator__validate_parameters",
        # blender-executor
        "mcp__blender-executor__execute_blender_script",
        "mcp__blender-executor__parse_blender_errors",
        # ... etc (43 tools total)
    ]
```

---

## 4. Files Involved

### 4.1 Orchestrator Files

| File | Purpose |
|------|---------|
| `orchestrator.py` | Main agent class, workflow stages, SDK integration |
| `workflow.py` | WorkflowStateMachine, stage definitions |
| `autonomy.py` | AutonomyController, trust score management |
| `guardrails.py` | TokenGuardrails, cost limit enforcement |
| `state.py` | SessionManager, SessionState, AssetRequest |
| `config.yaml` | Configuration for autonomy, guardrails, MCP servers |
| `test_explosion.py` | Test script that creates explosion asset |

### 4.2 MCP Server Files

Each MCP server has:
```
agents/<server-name>/
├── run_server.sh       # Entry point
├── server.py           # MCP server implementation
├── venv/               # Python virtual environment
└── requirements.txt    # Dependencies
```

---

## 5. What We've Already Fixed

### 5.1 Hook Bug (FIXED)

**Problem:** Pre-tool-use hook was calling `check_limits(estimated_tokens, 0)` but the method signature was `check_limits(self)`.

**Fix:** Changed to `can_proceed(estimated_tokens)` which has the correct signature.

```python
# Before (WRONG)
can_proceed, status = self.guardrails.check_limits(estimated_tokens, 0)

# After (CORRECT)
can_proceed, message = self.guardrails.can_proceed(estimated_tokens)
```

### 5.2 Autonomy Override (CONFIGURED)

Set `autonomy.override: "autonomous"` in config.yaml to prevent blocking on approval requests during testing.

---

## 6. Questions for Investigation

1. **Is the inner Claude agent actually receiving the MCP server configuration?**
   - How does Claude Agent SDK pass MCP servers to the inner CLI?

2. **Is the inner agent calling tools or just generating text?**
   - Can we add logging to see the actual tool calls?

3. **Are the MCP servers starting successfully?**
   - Can we verify the servers are running?

4. **What is the actual response content from `_query_agent()`?**
   - We need to log the full response to see what the inner agent is saying.

5. **Is the SDK client configured correctly for autonomous tool execution?**
   - Are there missing options like `tools`, `tool_choice`, etc.?

---

## 7. Suggested Next Steps

1. **Add verbose logging** to `_query_agent()` to see full response content

2. **Verify MCP servers are running** by testing them independently:
   ```bash
   cd agents/script-generator
   ./run_server.sh  # Should start without errors
   ```

3. **Check Claude Agent SDK documentation** for:
   - Correct MCP server configuration format
   - Required options for tool execution
   - How to verify tool calls are being made

4. **Consider simplifying** by testing the inner agent directly with a single MCP tool call before using the full orchestrator

5. **Review SDK examples** for MCP integration patterns

---

## 8. Test Command

To reproduce the issue:

```bash
cd /home/maz3ppa/projects/PlasmaDXR/agents/blender-orchestrator
source venv/bin/activate
python test_explosion.py 2>&1 | tee test_output.log
```

---

## 9. Key Code Locations

| What | File:Line |
|------|-----------|
| Main agent class | `orchestrator.py:67` |
| SDK client creation | `orchestrator.py:373` (create_options) |
| MCP server config | `orchestrator.py:430` (_create_mcp_config) |
| Query inner agent | `orchestrator.py:1080` (_query_agent) |
| Generate script stage | `orchestrator.py:689` (_stage_generate_script) |
| Extract script path | `orchestrator.py:1114` (_extract_script_path) |
| Hooks definition | `orchestrator.py:304` (_create_hooks) |

---

## 10. Environment Details

- **OS:** WSL2 Ubuntu on Windows
- **Python:** 3.12
- **Claude Agent SDK:** 0.1.18
- **Blender:** 5.0 (for actual execution)
- **API Key:** ANTHROPIC_API_KEY set in environment

---

**End of Report**
