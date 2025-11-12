#!/bin/bash
# DXR Image Quality Analyst MCP Server Launcher

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the agent directory
cd "$SCRIPT_DIR"

# Use the local venv in the agent directory
source "$SCRIPT_DIR/venv/bin/activate"

# Run the MCP server directly
exec python rtxdi_server.py
