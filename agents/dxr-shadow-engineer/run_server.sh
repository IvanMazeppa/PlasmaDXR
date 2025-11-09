#!/bin/bash
# Launch script for DXR Shadow Engineer MCP Server

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtual environment if it exists
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# Run MCP server
cd "$SCRIPT_DIR"
exec python3 dxr_shadow_server.py
