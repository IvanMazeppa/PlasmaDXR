#!/bin/bash
# Get NVIDIA_API_KEY from Windows and export to WSL

# Get key from Windows environment
WIN_KEY=$(powershell.exe -Command 'echo $env:NVIDIA_API_KEY' 2>/dev/null | tr -d '\r\n ')

if [ -z "$WIN_KEY" ]; then
    echo "ERROR: NVIDIA_API_KEY not found in Windows environment"
    echo ""
    echo "Please set it in Windows PowerShell:"
    echo '  $env:NVIDIA_API_KEY = "nvapi-your-key-here"'
    echo '  [System.Environment]::SetEnvironmentVariable("NVIDIA_API_KEY", "nvapi-your-key-here", "User")'
    exit 1
fi

# Export for this session
export NVIDIA_API_KEY="$WIN_KEY"

echo "✅ NVIDIA_API_KEY exported to WSL"
echo "Key starts with: ${NVIDIA_API_KEY:0:10}..."
echo "Key length: ${#NVIDIA_API_KEY}"

# Save to .env file for future use
echo "NVIDIA_API_KEY=$NVIDIA_API_KEY" > .env
echo "✅ Saved to .env file"
