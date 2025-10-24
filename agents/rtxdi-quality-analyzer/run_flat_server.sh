#!/bin/bash
# Flat server launcher - exact pix-debug pattern

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

# Run flat server directly (like pix-debug)
exec python "$SCRIPT_DIR/flat_server.py"
