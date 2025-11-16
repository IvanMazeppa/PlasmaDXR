#!/bin/bash
# Quick Start: Mission-Control Autonomous Agent

set -e

echo "======================================================================"
echo "Mission-Control Autonomous Agent - Quick Start"
echo "======================================================================"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✅ Dependencies installed"

# Check for ANTHROPIC_API_KEY
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo ""
    echo "⚠️  WARNING: ANTHROPIC_API_KEY not set!"
    echo ""
    echo "For Claude Max subscribers:"
    echo "  export CLAUDE_CODE_OAUTH_TOKEN=\$(claude auth token)"
    echo ""
    echo "For API key users:"
    echo "  export ANTHROPIC_API_KEY=your_api_key_here"
    echo ""
    echo "Continuing anyway (will fail if key not set)..."
fi

echo ""
echo "======================================================================"
echo "Choose mode:"
echo "======================================================================"
echo ""
echo "1) Interactive Mode     - Test autonomous reasoning in terminal"
echo "2) HTTP Bridge Mode     - Run as service for Claude Code integration"
echo "3) Single Query Mode    - One-off autonomous task"
echo ""
read -p "Enter choice (1/2/3): " choice

case $choice in
    1)
        echo ""
        echo "Starting interactive mode..."
        echo "Type your queries and watch autonomous reasoning in action!"
        echo ""
        python autonomous_agent.py
        ;;
    2)
        echo ""
        echo "Starting HTTP bridge on port 8001..."
        echo "Access at: http://localhost:8001"
        echo "Docs at: http://localhost:8001/docs"
        echo ""
        echo "Test with:"
        echo "  curl -X POST http://localhost:8001/query \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{\"prompt\": \"analyze probe grid lighting\"}'"
        echo ""
        python http_bridge.py
        ;;
    3)
        echo ""
        read -p "Enter your query: " query
        echo ""
        echo "Processing autonomous query..."
        echo ""
        python autonomous_agent.py "$query"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac
