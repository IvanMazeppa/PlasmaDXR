#!/bin/bash
# B2 Shutdown Script - Run from B1 before turning off machines
# Saves work, syncs conversations, safely shuts down B2

echo "=== B2 Safe Shutdown ==="
echo ""

# Step 1: Check if B2 is reachable
echo "[1/4] Checking B2 connection..."
if ! ssh b2 "echo 'B2 OK'" 2>/dev/null; then
    echo "ERROR: Cannot reach B2. Is it on?"
    exit 1
fi
echo "  ✓ B2 is reachable"

# Step 2: Sync conversation history FROM B2 to B1
echo ""
echo "[2/4] Syncing conversation history from B2 to B1..."
rsync -av --quiet b2:~/.claude/projects/ ~/.claude/projects/
rsync -av --quiet b2:~/.claude/history.jsonl ~/.claude/history.jsonl
echo "  ✓ Conversations synced"

# Step 3: Check for uncommitted work on B2
echo ""
echo "[3/4] Checking for uncommitted work on B2..."
WORKTREES=(
    "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean"
    "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Blender"
    "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-NanoVDB"
    "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-MultiAgent"
    "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-LuminousStars"
)

DIRTY=0
for wt in "${WORKTREES[@]}"; do
    STATUS=$(ssh b2 "git -C '$wt' status --porcelain 2>/dev/null" | head -1)
    if [ -n "$STATUS" ]; then
        echo "  WARNING: Uncommitted changes in $(basename $wt)"
        DIRTY=1
    fi
done

if [ $DIRTY -eq 1 ]; then
    echo ""
    read -p "There are uncommitted changes. Continue anyway? (y/N): " CONFIRM
    if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
        echo "Aborted. Please commit your work first."
        exit 1
    fi
else
    echo "  ✓ All worktrees clean"
fi

# Step 4: Shutdown B2
echo ""
echo "[4/4] Shutting down B2..."
read -p "Shut down B2 now? (y/N): " CONFIRM
if [ "$CONFIRM" = "y" ] || [ "$CONFIRM" = "Y" ]; then
    ssh b2 "sudo shutdown now" 2>/dev/null
    echo "  ✓ B2 shutdown initiated"
    echo ""
    echo "=== Done! Safe to turn off B1 now ==="
else
    echo "  Skipped shutdown. B2 still running."
fi
