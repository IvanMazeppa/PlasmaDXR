#!/bin/bash
# RTXDI Quality Analyzer - Flat server launcher

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

# Run flat server directly
exec python "$SCRIPT_DIR/rtxdi_server.py"
