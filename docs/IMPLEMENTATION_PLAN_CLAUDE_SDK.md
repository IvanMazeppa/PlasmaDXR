# Implementation Plan – Claude Agent SDK Setup

Created: 2025‑11‑14

---

## 1) Install & Verify
- Python 3.10+
- Node.js
- Claude Code 2.0+: `npm install -g @anthropic-ai/claude-code`
- SDK: `pip install claude-agent-sdk`

References:
- SDK repo: https://github.com/anthropics/claude-agent-sdk-python
- SDK overview: https://docs.claude.com/en/docs/agent-sdk/overview

---

## 2) Process Model – Mixed Servers
- In‑process SDK servers for orchestration/lightweight utilities: mission-control, knowledge-archivist, particle-pipeline-runner, imageops-agent.
- External stdio MCP servers for heavy/long‑running ML/GPU: dxr-image-quality-analyst, log-analysis-rag, dxr-shadow-engineer, dxr-volumetric-pyro-specialist, pix-debuggers, material-system-engineer, gaussian-analyzer.

---

## 3) Options Template
```python
from pathlib import Path
from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server

# In‑process servers
mission_control = create_sdk_mcp_server(
    name="mission-control", version="1.0.0", tools=[dispatch_plan, record_decision, trigger_review]
)

pipeline_runner = create_sdk_mcp_server(
    name="pipeline-runner", version="1.0.0", tools=[rebuild_shaders, run_plasmadx, capture_screenshot, collect_logs]
)

options = ClaudeAgentOptions(
    cwd=Path("/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean"),
    mcp_servers={
        # In‑process (fast)
        "mission-control": mission_control,
        "pipeline-runner": pipeline_runner,
        # External (stdio)
        "dxr-image-quality-analyst": {
            "type": "stdio",
            "command": "bash",
            "args": ["/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/dxr-image-quality-analyst/run_server.sh"],
            "cwd": "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/dxr-image-quality-analyst"
        },
        "log-analysis-rag": {
            "type": "stdio",
            "command": "bash",
            "args": ["/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/log-analysis-rag/run_server.sh"],
            "cwd": "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/log-analysis-rag"
        },
        # ... add the rest of external agents similarly
    },
    allowed_tools=[
        # Start restrictive; grant per task
        "mcp__mission-control__dispatch_plan",
        "mcp__pipeline-runner__capture_screenshot",
    ],
    permission_mode="manual"  # require explicit approval for edits/commands
)
```

---

## 4) Hooks (Safety Rails)
```python
from claude_agent_sdk import HookMatcher

async def guard_bash(input_data, tool_use_id, context):
    if input_data.get("tool_name") != "Bash":
        return {}
    cmd = input_data.get("tool_input", {}).get("command", "")
    forbidden = ["rm -rf", "format", "diskpart"]
    if any(x in cmd for x in forbidden):
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "Dangerous command"}}
    return {}

options.hooks = {
    "PreToolUse": [HookMatcher(matcher="Bash", hooks=[guard_bash])]
}
```

Use `permission_mode='acceptEdits'` only for trusted file‑ops agents (e.g., material-system-engineer) and only when you want fully autonomous edits.

---

## 5) MCP Registration (External Servers)
```bash
# Example: register dxr-image-quality-analyst
claude mcp add --transport stdio dxr-image-quality-analyst \
  --env PROJECT_ROOT=/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean \
  -- /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/dxr-image-quality-analyst/run_server.sh

claude mcp list
```

---

## 6) Session Forking
Use `ClaudeSDKClient` when running long operations in parallel so orchestration remains responsive. Each fork gets its own `ClaudeAgentOptions` (same `cwd`, narrowed `allowed_tools`).

---

## 7) CWD Discipline
Set `cwd` per server in options or wrapper scripts. Do not rely on shell `cd` inside tools; it causes path drift.

---

## 8) Artifact Directories
- Screenshots: `screenshots/`
- PIX: `PIX/Captures/`, `PIX/buffer_dumps/`
- Logs: `logs/`
- Reports: `docs/sessions/`

Pipeline runner should ensure these exist and tag each session with a UUID/timestamp.

---

## 9) Minimal Bring‑Up Checklist
- [ ] Register in‑process servers (mission-control, pipeline-runner)
- [ ] Register external servers (image QA, RAG, shadow, pyro, materials)
- [ ] Configure hooks and restrictive `allowed_tools`
- [ ] Validate each tool with a trivial call
- [ ] Run first end‑to‑end: build → run → capture screenshot → LPIPS → ingest logs → publish summary
