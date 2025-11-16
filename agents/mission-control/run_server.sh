#!/bin/bash
# Mission-Control MCP Server Launcher
# Activates virtual environment and starts the server

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the mission-control directory
cd "$SCRIPT_DIR"

# Activate virtual environment
source venv/bin/activate

# Set project root if not already set
if [ -z "$PROJECT_ROOT" ]; then
    export PROJECT_ROOT="$(cd ../.. && pwd)"
fi

# Run the server in MCP mode
exec python server.py
