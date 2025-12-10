#!/usr/bin/env bash
# Start separate tmux sessions per Claude CLI to keep them isolated.
# Assumes WSL/Ubuntu paths; update ROOT if needed.
set -euo pipefail

ROOT="/mnt/d/Users/dilli/AndroidStudioProjects"

declare -A SESSIONS=(
  [claude-pinn]="$ROOT/PlasmaDX-PINN-v4"
  [claude-blender]="$ROOT/PlasmaDX-Blender"
  [claude-multi]="$ROOT/PlasmaDX-MultiAgent"
)

for name in "${!SESSIONS[@]}"; do
  dir="${SESSIONS[$name]}"
  tmux has-session -t "$name" 2>/dev/null || tmux new-session -d -s "$name" -c "$dir"
done

echo "Available sessions:"
tmux ls
echo "Attach with: tmux attach -t <session-name> (e.g., tmux attach -t claude-blender)"

