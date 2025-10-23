#!/bin/bash
# RTXDI Quality Analyzer MCP Server Launcher

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the script directory
cd "$SCRIPT_DIR"

# Activate virtual environment and run the agent
source venv/bin/activate
exec python -m src.agent
