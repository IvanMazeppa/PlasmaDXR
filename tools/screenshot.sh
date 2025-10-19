#!/bin/bash
# Bash wrapper for PowerShell screenshot tool
# Usage: ./screenshot.sh [description]

DESCRIPTION="${1:-screenshot}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POWERSHELL_SCRIPT="$SCRIPT_DIR/take-screenshot.ps1"

# Convert WSL path to Windows path for PowerShell
WINDOWS_SCRIPT_PATH=$(wslpath -w "$POWERSHELL_SCRIPT")

# Execute PowerShell script
powershell.exe -ExecutionPolicy Bypass -File "$WINDOWS_SCRIPT_PATH" "$DESCRIPTION"
