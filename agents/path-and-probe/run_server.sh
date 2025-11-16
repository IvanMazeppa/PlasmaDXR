#!/bin/bash
# MCP Server launcher for path-and-probe specialist

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate virtual environment
cd "$SCRIPT_DIR" || exit 1
source venv/bin/activate

# Run MCP server
exec python server.py
