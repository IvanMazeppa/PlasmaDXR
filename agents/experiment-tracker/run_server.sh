#!/bin/bash
# Experiment Tracker MCP Server startup script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the server
exec python3 "$SCRIPT_DIR/server.py" "$@"
