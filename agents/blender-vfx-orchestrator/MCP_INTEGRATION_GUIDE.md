# Blender VFX Orchestrator MCP Integration Guide

**Date:** 2026-01-26
**Server Location:** `agents/blender-vfx-orchestrator`

## Overview
This document describes how to integrate the **Blender VFX Orchestrator** as an MCP (Model Context Protocol) server. This server exposes the orchestrator's capabilities to MCP clients like Claude Desktop or other AI assistants.

## Server Configuration

### 1. Locate the Startup Script
The entry point is the shell script that handles environment setup (venv activation, .env loading):
- **Path:** `/home/maz3ppa/projects/PlasmaDXR/agents/blender-vfx-orchestrator/run_server.sh`

### 2. Configure Your Client

#### For Claude Desktop
Add the following configuration to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "blender-vfx-orchestrator": {
      "command": "/bin/bash",
      "args": [
        "/home/maz3ppa/projects/PlasmaDXR/agents/blender-vfx-orchestrator/run_server.sh"
      ],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

#### For Gemini CLI
Add the following configuration to your `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "blender-vfx-orchestrator": {
      "command": "/bin/bash",
      "args": [
        "/home/maz3ppa/projects/PlasmaDXR/agents/blender-vfx-orchestrator/run_server.sh"
      ],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

> **Note:** Adjust the absolute path if your project root differs from `/home/maz3ppa/projects/PlasmaDXR`.

## Verification

1.  **Restart** your MCP client.
2.  **Check Connection:** The client should show "blender-vfx-orchestrator" as connected.
3.  **Test Tool:** Ask the assistant to "List the available presets for the VFX orchestrator" or use a tool like `list_presets` (if exposed) to verify functionality.

## Troubleshooting

-   **Permission Denied:** Ensure `run_server.sh` is executable:
    ```bash
    chmod +x /home/maz3ppa/projects/PlasmaDXR/agents/blender-vfx-orchestrator/run_server.sh
    ```
-   **Missing Dependencies:** Ensure the virtual environment at `agents/blender-vfx-orchestrator/venv` is fully populated.
-   **Environment Variables:** The script automatically loads `.env` from the server directory if present.

## Internal Details
-   **Server Script:** `server.py`
-   **Transport:** Stdio (Standard Input/Output)
-   **Dependencies:** `mcp`, `openai-agents`, `pydantic` (see `requirements.txt`)
