#!/bin/bash
# WSL wrapper to call Windows PowerShell screenshot script

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
powershell.exe -ExecutionPolicy Bypass -File "$SCRIPT_DIR/screenshot.ps1"
