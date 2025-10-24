#!/bin/bash
# RTXDI Quality Analyzer MCP Server Launcher

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the agent directory (required for module execution)
cd "$SCRIPT_DIR"

# Activate virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

# Run the MCP server as a Python module (original working method)
exec python -m src.agent
