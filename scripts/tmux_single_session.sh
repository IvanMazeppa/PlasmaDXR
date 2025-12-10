#!/usr/bin/env bash
# Set up a single tmux session with windows for each worktree and Claude CLIs.
# Assumes WSL/Ubuntu paths; update ROOT if your paths differ.
set -euo pipefail

ROOT="/mnt/d/Users/dilli/AndroidStudioProjects"
SESSION="plasma"

# Ordered pairs: window-name then directory.
WINDOWS=(
  blender        "$ROOT/PlasmaDX-Blender"
  pinn           "$ROOT/PlasmaDX-PINN-v4"
  multi          "$ROOT/PlasmaDX-MultiAgent"
  claude-blender "$ROOT/PlasmaDX-Blender"
  claude-pinn    "$ROOT/PlasmaDX-PINN-v4"
  claude-multi   "$ROOT/PlasmaDX-MultiAgent"
)

main_name="${WINDOWS[0]}"
main_dir="${WINDOWS[1]}"

# Start the session (detached) with the first window.
tmux has-session -t "$SESSION" 2>/dev/null || \
  tmux new-session -d -s "$SESSION" -n "$main_name" -c "$main_dir"

# Create the rest of the windows.
for ((i=2; i<${#WINDOWS[@]}; i+=2)); do
  name="${WINDOWS[i]}"
  dir="${WINDOWS[i+1]}"
  tmux new-window -t "$SESSION" -n "$name" -c "$dir"
done

# Jump to the first window and attach.
tmux select-window -t "$SESSION:1"
tmux attach -t "$SESSION"

