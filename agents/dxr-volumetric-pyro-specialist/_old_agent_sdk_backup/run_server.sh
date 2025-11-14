#!/bin/bash
#
# MCP Server Launch Script for DXR Volumetric Pyro Specialist
#
# This script launches the pyro specialist agent as an MCP server that can be
# invoked by other agents in the autonomous pipeline (gaussian-analyzer,
# material-system-engineer, dxr-image-quality-analyst).
#
# Usage:
#   ./run_server.sh [--port PORT] [--log-level LEVEL]
#
# Options:
#   --port PORT         TCP port for MCP server (default: auto-assigned)
#   --log-level LEVEL   Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
#
# Requirements:
#   - Python 3.10+ installed
#   - Virtual environment activated (or dependencies installed globally)
#   - .env file configured with PROJECT_ROOT
#
# MCP Registration:
#   Add to ~/.claude/mcp_settings.json:
#   {
#     "mcpServers": {
#       "dxr-pyro-specialist": {
#         "command": "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/dxr-volumetric-pyro-specialist/run_server.sh",
#         "args": [],
#         "env": {}
#       }
#     }
#   }
#

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if .env file exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found"
    echo "Copy .env.example to .env and configure PROJECT_ROOT"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "WARNING: Virtual environment not found at venv/"
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Installing dependencies..."
    ./venv/bin/pip install -r requirements.txt
fi

# Activate virtual environment
source venv/bin/activate

# Default values
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--port PORT] [--log-level LEVEL]"
            exit 1
            ;;
    esac
done

# Export log level
export LOG_LEVEL="$LOG_LEVEL"

# Launch MCP server
echo "=========================================="
echo "DXR Volumetric Pyro Specialist MCP Server"
echo "=========================================="
echo "Script directory: $SCRIPT_DIR"
echo "Log level: $LOG_LEVEL"
echo ""

# Run the agent in MCP server mode
# Note: The agent SDK automatically detects MCP server mode when invoked by Claude
exec python3 main.py
