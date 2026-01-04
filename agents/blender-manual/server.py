#!/usr/bin/env python3
"""
Compatibility wrapper for the Blender Manual MCP server.

The canonical implementation lives in `blender_server.py`, but several parts of
the pipeline (and external tooling) assume an agent exposes `server.py`.

This wrapper:
- Re-exports the FastMCP instance (`mcp`) and tool functions
- Allows the orchestrator's "direct tool import" mode to load blender-manual
  the same way it loads other agents (script-generator, asset-evaluator, etc.)
"""

from blender_server import *  # noqa: F401,F403


