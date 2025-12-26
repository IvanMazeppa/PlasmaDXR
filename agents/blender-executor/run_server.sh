#!/usr/bin/env bash
# Blender Executor MCP Server runner
#
# Usage:
#   ./run_server.sh         # Run server (stdio transport)
#   ./run_server.sh --help  # Show help
#
# Environment variables:
#   BLENDER_EXE     - Path to Blender executable (default: /mnt/e/.../blender.exe)
#   PROJECT_ROOT    - Path to PlasmaDXR project root (default: auto-detect)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "$(dirname "$SCRIPT_DIR")")}"

# Activate virtual environment if it exists
if [[ -f "$PROJECT_ROOT/venv/bin/activate" ]]; then
    source "$PROJECT_ROOT/venv/bin/activate"
elif [[ -f "$SCRIPT_DIR/venv/bin/activate" ]]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# Export for the server
export PROJECT_ROOT

# Run the server
exec python3 "$SCRIPT_DIR/server.py" "$@"
