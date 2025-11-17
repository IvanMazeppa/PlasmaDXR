#!/bin/bash
# Setup script for Rendering Council Agent SDK agent

set -e

echo "Setting up Rendering Council Agent SDK agent..."

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To use the agent:"
echo "  1. Set your API key: export ANTHROPIC_API_KEY='your-key-here'"
echo "  2. Activate venv: source venv/bin/activate"
echo "  3. Run agent: python rendering_council_agent.py \"<task>\""
echo ""
echo "Example:"
echo "  python rendering_council_agent.py \"Analyze Gaussian rendering and propose fixes\""
