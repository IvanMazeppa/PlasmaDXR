# Agents SDK Visualization Guide (2026-01-23)

This guide summarizes the official Agents SDK visualization feature and shows how to apply it to the Blender VFX Orchestrator.

Source: https://github.com/openai/openai-agents-python/blob/main/docs/visualization.md

## What Visualization Provides
The SDK can generate a **Graphviz** diagram of your agent topology:
- Agents: yellow rectangles
- Tools: green ellipses
- MCP servers: grey rectangles
- Solid arrows: agent handoffs
- Dotted arrows: tool invocations
- Dashed arrows: MCP server invocations

This is best for **static structure**, not runtime timing.

## Installation
The visualization feature is in the optional `viz` dependency group:
```
pip install "openai-agents[viz]"
```

## Minimal Example (from SDK docs)
```
from agents.extensions.visualization import draw_graph

triage_agent = Agent(...)
draw_graph(triage_agent)
```

## Recommended Usage for This Project
Target the **coordinator or top-level orchestrator agent** so the graph shows:
- coordinator
- specialized sub-agents (as tools)
- MCP servers and tools

Example (adapt to your agent factory):
```
from agents.extensions.visualization import draw_graph

coordinator = create_coordinator_agent()
draw_graph(coordinator, filename="vfx_orchestrator_graph")
```

This will generate `vfx_orchestrator_graph.png` in the working directory.

## Notes / Limitations
- Visualization is structural, not a timeline.
- It won’t show tracing spans or step-by-step execution order.
- MCP server nodes require a recent `agents` version (per SDK doc note).

## Suggested Placement
Add a one-off script under `agents/blender-vfx-orchestrator/tools/` or `scripts/` to generate the graph on demand, e.g.:
```
python tools/generate_agent_graph.py
```

