#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

# Source .env file if it exists (more reliable than python-dotenv in MCP context)
if [[ -f ".env" ]]; then
  set -a  # Auto-export variables
  source .env
  set +a
fi

if [[ -x "./venv/bin/python" ]]; then
  exec ./venv/bin/python server.py
fi

exec python3 server.py
