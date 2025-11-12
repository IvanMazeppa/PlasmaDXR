#!/bin/bash
# 2-tool test server

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/venv/bin/activate"
exec python "$SCRIPT_DIR/rtxdi_server_2tools.py"
