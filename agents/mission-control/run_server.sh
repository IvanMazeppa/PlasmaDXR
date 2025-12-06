#!/bin/bash
# Mission-Control MCP Server Launcher
# Starts the FastMCP server for Claude Code integration

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the mission-control directory
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi

# Set project root if not already set
if [ -z "$PROJECT_ROOT" ]; then
    export PROJECT_ROOT="$(cd ../.. && pwd)"
fi

# Run the FastMCP server (not the Agent SDK server)
# mcp_server.py = Claude Code MCP interface (this)
# autonomous_agent.py = Autonomous agent using Agent SDK (separate)
exec python mcp_server.py
