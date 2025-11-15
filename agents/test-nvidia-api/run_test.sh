#!/bin/bash
# NVIDIA API Test Runner for PlasmaDX

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/.installed" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    touch venv/.installed
fi

# Ensure NVIDIA_API_KEY is set (Windows env var should be visible in WSL)
if [ -z "$NVIDIA_API_KEY" ]; then
    # Try to get from Windows environment
    export NVIDIA_API_KEY=$(powershell.exe -Command 'echo $env:NVIDIA_API_KEY' 2>/dev/null | tr -d '\r')
fi

# Run test
echo ""
echo "Running NVIDIA API tests..."
echo ""
python test_nvidia_api.py
