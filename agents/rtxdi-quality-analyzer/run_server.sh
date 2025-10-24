#!/bin/bash
# RTXDI Quality Analyzer MCP Server Launcher

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the agent directory
cd "$SCRIPT_DIR"

# Use the project root venv (which has all packages installed)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$PROJECT_ROOT/venv/bin/activate"

# Run the flat MCP server directly
exec python rtxdi_server.py
