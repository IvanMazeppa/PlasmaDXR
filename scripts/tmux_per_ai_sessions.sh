#!/usr/bin/env bash
# Start tmux sessions for Blender and NanoVDB worktrees
set -euo pipefail

ROOT="/mnt/d/Users/dilli/AndroidStudioProjects"

# Create sessions if they don't exist
tmux has-session -t claude-blender 2>/dev/null || tmux new-session -d -s claude-blender -c "$ROOT/PlasmaDX-Blender"
tmux has-session -t claude-nanovdb 2>/dev/null || tmux new-session -d -s claude-nanovdb -c "$ROOT/PlasmaDX-NanoVDB"

echo "Sessions ready:"
tmux ls
echo ""
echo "Attach with: tb (blender) or tn (nanovdb)"
echo "Switch with: F9 (blender) or F10 (nanovdb)"
