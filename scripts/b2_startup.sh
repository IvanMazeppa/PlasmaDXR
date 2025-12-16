#!/bin/bash
# B2 Startup Script - Run from B1 after turning on B2
# Waits for B2, starts services, syncs conversations

echo "=== B2 Startup ==="
echo ""

# Step 1: Wait for B2 to come online
echo "[1/4] Waiting for B2 to come online..."
echo "  (Make sure B2 is powered on and WSL is running)"
echo ""

ATTEMPTS=0
MAX_ATTEMPTS=30
while [ $ATTEMPTS -lt $MAX_ATTEMPTS ]; do
    if ssh -o ConnectTimeout=2 b2 "echo ok" 2>/dev/null; then
        echo "  ✓ B2 is online"
        break
    fi
    ATTEMPTS=$((ATTEMPTS + 1))
    echo "  Waiting... ($ATTEMPTS/$MAX_ATTEMPTS)"
    sleep 2
done

if [ $ATTEMPTS -eq $MAX_ATTEMPTS ]; then
    echo "ERROR: Could not connect to B2 after $MAX_ATTEMPTS attempts"
    echo "Make sure:"
    echo "  1. B2 is powered on"
    echo "  2. WSL is running (open Ubuntu terminal on B2)"
    echo "  3. Run 'tstart' manually on B2"
    exit 1
fi

# Step 2: Start SSH server on B2 (in case it's not running)
echo ""
echo "[2/4] Ensuring SSH server is running on B2..."
ssh b2 "sudo service ssh start 2>/dev/null || echo 'SSH already running'"
echo "  ✓ SSH server ready"

# Step 3: Create tmux sessions
echo ""
echo "[3/4] Creating tmux sessions on B2..."
ssh b2 'tmux new-session -d -s claude-main -c "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean" 2>/dev/null
tmux new-session -d -s claude-blender -c "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Blender" 2>/dev/null
tmux new-session -d -s claude-nanovdb -c "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-NanoVDB" 2>/dev/null
tmux new-session -d -s claude-multi -c "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-MultiAgent" 2>/dev/null
tmux new-session -d -s claude-luminous -c "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-LuminousStars" 2>/dev/null
tmux ls'
echo "  ✓ tmux sessions created"

# Step 4: Sync conversation history TO B2
echo ""
echo "[4/4] Syncing conversation history to B2..."
rsync -av --quiet ~/.claude/projects/ b2:~/.claude/projects/
rsync -av --quiet ~/.claude/history.jsonl b2:~/.claude/history.jsonl
echo "  ✓ Conversations synced"

echo ""
echo "=== B2 Ready! ==="
echo ""
echo "Available commands:"
echo "  b2m     - Main (PlasmaDX-Clean)"
echo "  b2b     - Blender integration"
echo "  b2n     - NanoVDB"
echo "  b2multi - Multi-agent"
echo "  b2l     - LuminousStars"
echo "  b2ls    - List all sessions"
