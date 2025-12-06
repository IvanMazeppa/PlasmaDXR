# Mission-Control Agent

**Strategic Orchestrator for PlasmaDX-Clean Multi-Agent RAG System**

Mission-Control coordinates 4 specialist councils (rendering, materials, physics, diagnostics) in a hierarchical workflow based on NVIDIA's multi-agent RAG architecture. It provides strategic task orchestration, decision recording, status aggregation, and quality gate enforcement for the PlasmaDX-Clean DirectX 12 volumetric rendering project.

## Architecture Overview

Mission-Control has two modes of operation:

1. **MCP Server** (`mcp_server.py`) - FastMCP server that Claude Code connects to via stdio
   - Exposes 4 strategic tools to Claude Code
   - Runs as a standard MCP server

2. **Autonomous Agent** (`autonomous_agent.py`) - Claude Agent SDK application with AI reasoning
   - Uses Anthropic API for autonomous decision-making
   - Connects to other MCP specialist servers as a client
   - Requires `ANTHROPIC_API_KEY` with funds

**Typical use**: Claude Code connects to the MCP server, which provides strategic coordination tools.

---

## Features

### Core Capabilities

- **Task Orchestration**: Dispatch work to specialist councils based on priority and dependencies
- **Decision Recording**: Log all strategic decisions with rationale and artifact links
- **Status Aggregation**: Query councils and publish consolidated status reports
- **Quality Gates**: Enforce thresholds (LPIPS visual similarity, FPS performance)
- **Context Persistence**: Maintain state across sessions (via ChromaDB RAG store, coming soon)

### Custom Tools

| Tool | Purpose | Parameters |
|------|---------|------------|
| `dispatch_plan` | Send tasks to specialist councils | `plan`, `priority`, `council` |
| `record_decision` | Log decisions to session files | `decision`, `rationale`, `artifacts` |
| `publish_status` | Generate consolidated status reports | None |
| `handoff_to_agent` | Delegate to specialist agents | `agent`, `task`, `context` |

### Architecture

```
Mission-Control (Strategic Tier)
├── Rendering Council
│   ├── DXR raytracing
│   ├── RTXDI lighting
│   └── Shadow systems
├── Materials Council
│   ├── Particle properties
│   ├── Gaussian splatting
│   └── Material systems
├── Physics Council
│   ├── PINN ML physics
│   ├── GPU physics
│   └── Accretion disk dynamics
└── Diagnostics Council
    ├── PIX debugging
    ├── Buffer analysis
    └── Performance profiling
```

---

## Installation

### Prerequisites

- **Python 3.10+** (required for type hints and pattern matching)
- **pip** package manager
- **Anthropic API Key** (optional for Max subscribers running via Claude Code)

### Setup

1. **Navigate to the agent directory**:
   ```bash
   cd agents/mission-control
   ```

2. **Install production dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install development dependencies** (optional):
   ```bash
   pip install -r requirements-dev.txt
   ```

4. **Configure environment** (optional):
   ```bash
   cp .env.example .env
   # Edit .env and set ANTHROPIC_API_KEY if needed
   ```

   **Note**: Max subscribers running via Claude Code don't need an API key - the SDK will use your session credentials automatically.

5. **Verify installation**:
   ```bash
   python -c "import claude_agent_sdk; print(f'Claude Agent SDK v{claude_agent_sdk.__version__}')"
   ```

---

## Usage

### Interactive Mode (Recommended)

Run mission-control in continuous conversation mode:

```bash
python server.py
```

Example interaction:
```
> Dispatch a task to optimize RTXDI M5 temporal accumulation

Mission-Control: I'll dispatch that to the rendering council...
[Agent uses dispatch_plan tool]

> What's the current status across all councils?

Mission-Control: Let me query all councils...
[Agent uses publish_status tool]
```

### Single Query Mode

Execute a one-off task and exit:

```bash
python server.py "Check current FPS performance and quality metrics"
```

### Programmatic Usage

```python
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, create_sdk_mcp_server
from tools.dispatch import dispatch_plan
from tools.status import publish_status

# Create MCP server
server = create_sdk_mcp_server(
    name="mission-control",
    version="0.1.0",
    tools=[dispatch_plan, publish_status]
)

# Configure options
options = ClaudeAgentOptions(
    mcp_servers={"mission-control": server},
    allowed_tools=["mcp__mission-control__dispatch_plan"]
)

# Run query
async def main():
    async for message in query("Dispatch rendering optimization task", options=options):
        print(message)

asyncio.run(main())
```

---

## Project Structure

```
agents/mission-control/
├── server.py              # Main entry point with ClaudeSDKClient
├── tools/                 # Custom tool implementations
│   ├── __init__.py        # Tool exports
│   ├── dispatch.py        # dispatch_plan tool
│   ├── record.py          # record_decision tool
│   ├── status.py          # publish_status tool
│   └── handoff.py         # handoff_to_agent tool
├── hooks.py               # Safety guards (guard_bash, guard_file_write)
├── requirements.txt       # Production dependencies
├── requirements-dev.txt   # Development dependencies
├── pyproject.toml         # mypy, black, ruff configuration
├── .env.example           # Environment template
├── .gitignore             # Python-specific ignores
└── README.md              # This file
```

---

## Tool Documentation

### `dispatch_plan`

Dispatch a task to a specialist council for execution.

**Parameters**:
- `plan` (str): Detailed task description with objectives and acceptance criteria
- `priority` (str): Priority level (`critical`, `high`, `medium`, `low`)
- `council` (str): Target council (`rendering`, `materials`, `physics`, `diagnostics`)

**Returns**: Task ID, status, and council acknowledgment

**Example**:
```python
result = await dispatch_plan({
    "plan": "Optimize RTXDI M5 temporal accumulation to fix patchwork pattern",
    "priority": "high",
    "council": "rendering"
})
```

---

### `record_decision`

Log a strategic decision to the session file with rationale.

**Parameters**:
- `decision` (str): What was decided
- `rationale` (str): Why it was decided (reasoning and trade-offs)
- `artifacts` (list[str]): Paths to supporting files (PIX captures, screenshots, etc.)

**Returns**: Confirmation with file path and timestamp

**Output**: Writes to `docs/sessions/SESSION_<date>.md`

**Example**:
```python
result = await record_decision({
    "decision": "Enable RTXDI M5 as primary renderer",
    "rationale": "Quality issues resolved, FPS improved by 15%",
    "artifacts": ["PIX/Captures/rtxdi_m5_final.wpix", "screenshots/comparison.png"]
})
```

---

### `publish_status`

Generate consolidated status report from all councils.

**Parameters**: None

**Returns**:
- Council status (active tasks, completion counts)
- Quality gates (LPIPS, FPS, visual quality scores)
- Blocked tasks and dependencies
- Summary overview

**Example**:
```python
status = await publish_status({})
# Returns comprehensive status across all councils
```

---

### `handoff_to_agent`

Delegate a task to a specialist agent (existing MCP agents).

**Parameters**:
- `agent` (str): Target agent name (`dxr-image-quality-analyst`, `pix-debug`, `log-analysis-rag`, etc.)
- `task` (str): Task description with clear objectives
- `context` (dict): Additional context (file paths, parameters)

**Returns**: Handoff confirmation and estimated completion time

**Available Agents**:
- `dxr-image-quality-analyst` - Visual quality analysis, LPIPS comparison
- `pix-debug` - PIX capture analysis, buffer validation
- `log-analysis-rag` - Log ingestion, diagnostic queries
- `material-system-engineer` - Particle materials, Gaussian properties
- `path-and-probe` - Probe grid optimization, lighting coverage

**Example**:
```python
result = await handoff_to_agent({
    "agent": "dxr-image-quality-analyst",
    "task": "Compare before/after screenshots for RTXDI M5 optimization",
    "context": {
        "before": "screenshots/rtxdi_m4.png",
        "after": "screenshots/rtxdi_m5.png"
    }
})
```

---

## Development

### Type Checking

Run mypy with strict mode:
```bash
mypy server.py tools/
```

Expected output: `Success: no issues found`

### Code Formatting

Format code with black:
```bash
black server.py tools/
```

### Linting

Run ruff linter:
```bash
ruff check server.py tools/
```

### Running Tests

```bash
# TODO: Add pytest tests
pytest tests/
```

---

## Integration with Existing Agents

Mission-Control can invoke existing MCP agents via `handoff_to_agent` tool:

```python
# Example: Integrate with log-analysis-rag agent
from claude_agent_sdk import ClaudeAgentOptions

# TODO: Add external MCP server integration
# external_agents = {
#     "log-analysis-rag": existing_mcp_server_instance
# }
#
# options = ClaudeAgentOptions(
#     mcp_servers={
#         "mission-control": mission_control_server,
#         **external_agents
#     }
# )
```

---

## Roadmap

### Current Status (v0.1.0)

✅ Basic agent structure with 4 custom tools
✅ In-process MCP server setup
✅ Safety hooks framework
✅ Type-safe implementation with mypy

### Next Steps

- [ ] Implement actual council dispatch logic (currently placeholder)
- [ ] Add real file I/O for decision recording to `docs/sessions/`
- [ ] Integrate with existing external MCP agents
- [ ] Add ChromaDB for RAG-based context persistence
- [ ] Implement quality gate threshold checking (LPIPS < 0.05, FPS > 90)
- [ ] Add session context management
- [ ] Build knowledge-steward agent (sibling strategic agent)
- [ ] Build backlog-scribe agent (sibling strategic agent)

---

## Reference Documentation

**Project Context**:
- [`CLAUDE.md`](../../CLAUDE.md) - Project overview and architecture
- [`docs/MASTER_ROADMAP_V2.md`](../../docs/MASTER_ROADMAP_V2.md) - Development roadmap
- [`docs/CELESTIAL_RAG_IMPLEMENTATION_ROADMAP.md`](../../docs/CELESTIAL_RAG_IMPLEMENTATION_ROADMAP.md) - Multi-agent RAG architecture
- [`docs/NVIDIA_MULTI_AGENT_RAG_WORKFLOW.md`](../../docs/NVIDIA_MULTI_AGENT_RAG_WORKFLOW.md) - NVIDIA's multi-agent design
- [`docs/IMPLEMENTATION_PLAN_CLAUDE_SDK.md`](../../docs/IMPLEMENTATION_PLAN_CLAUDE_SDK.md) - SDK integration patterns

**Claude Agent SDK**:
- [Overview](https://docs.claude.com/en/api/agent-sdk/overview)
- [Python SDK Reference](https://docs.claude.com/en/api/agent-sdk/python)
- [Custom Tools Guide](https://docs.claude.com/en/api/agent-sdk/custom-tools)
- [MCP Integration](https://docs.claude.com/en/api/agent-sdk/mcp)

---

## License

This agent is part of the PlasmaDX-Clean project. See project LICENSE for details.

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'claude_agent_sdk'"

Install the SDK:
```bash
pip install claude-agent-sdk==0.1.6
```

### "TypeError: 'dict' object is not callable"

Check tool decorator syntax. Should be:
```python
@tool(name="tool_name", description="...", parameters={...})
async def tool_function(args: dict[str, Any]) -> dict[str, Any]:
    ...
```

### "mypy: Command not found"

Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

### API Key Issues

If you're a Max subscriber running via Claude Code, you don't need an API key. The SDK automatically uses your session credentials.

For standalone usage, get an API key from: https://console.anthropic.com/

---

**Version**: 0.1.0
**Last Updated**: 2025-11-16
**Maintained by**: Claude Code sessions
