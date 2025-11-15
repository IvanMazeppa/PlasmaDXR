#!/bin/bash
# MCP Server launcher for log-analysis-rag agent

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate virtual environment
cd "$SCRIPT_DIR" || exit 1
source venv/bin/activate

# Export NVIDIA_API_KEY from Windows environment if available
if [ -z "$NVIDIA_API_KEY" ]; then
    # Get from Windows environment (always available in WSL)
    WIN_NVIDIA_KEY=$(cmd.exe /c "echo %NVIDIA_API_KEY%" 2>/dev/null | tr -d '\r\n')
    if [ "$WIN_NVIDIA_KEY" != "%NVIDIA_API_KEY%" ]; then
        export NVIDIA_API_KEY="$WIN_NVIDIA_KEY"
    fi
fi

# Verify API key is set (for debugging)
if [ -z "$NVIDIA_API_KEY" ]; then
    echo "Warning: NVIDIA_API_KEY not found in environment" >&2
fi

# Run MCP server
exec python server.py
