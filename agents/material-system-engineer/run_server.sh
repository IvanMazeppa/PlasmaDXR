#!/bin/bash
# Material System Engineer - MCP Server Launcher

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "⚠️  Warning: Virtual environment not found. Run setup first:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Check if .env exists, if not copy from .env.example
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "📋 Creating .env from .env.example..."
        cp .env.example .env
        echo "✅ .env created. Please verify PROJECT_ROOT path is correct."
    else
        echo "❌ Error: .env.example not found!"
        exit 1
    fi
fi

# Run the MCP server
echo "🚀 Starting Material System Engineer MCP Server..."
python material_engineer_server.py
