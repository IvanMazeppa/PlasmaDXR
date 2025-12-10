# Multi-Window + tmux: Best of Both Worlds

**Last Updated:** 2025-12-10
**Status:** WORKING CONFIGURATION

This doc explains the hybrid approach: **multiple Cursor windows** for visual workflow + **tmux persistence** for reliability and remote access.

---

## Quick Start

**Server PC (multiple monitors):**
1. Open Cursor → PlasmaDX-Clean (Monitor 1)
2. Open Cursor → PlasmaDX-Blender (Monitor 2)
3. Run `claude` in each terminal

**For PINN/MultiAgent (tmux-backed):**
```bash
tp          # Attach to PINN worktree (Claude running)
tm          # Attach to MultiAgent worktree (Claude running)
# Ctrl+b d  # Detach (keeps running)
```

**From Client PC (SSH):**
```bash
ssh server-ip
tp    # or tm
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SERVER PC (Ryzen 9) - Multiple Monitors                                    │
│                                                                             │
│  ┌─ Monitor 1 ─────────────┐  ┌─ Monitor 2 ─────────────┐                  │
│  │ Cursor Window           │  │ Cursor Window           │                  │
│  │ PlasmaDX-Clean          │  │ PlasmaDX-Blender        │                  │
│  │ Claude instance #1      │  │ Claude instance #2      │                  │
│  └─────────────────────────┘  └─────────────────────────┘                  │
│                                                                             │
│  ┌─ tmux (background, persistent) ──────────────────────────────────────┐  │
│  │ claude-pinn    → PlasmaDX-PINN-v4     │ Attach: tp                   │  │
│  │ claude-multi   → PlasmaDX-MultiAgent  │ Attach: tm                   │  │
│  │ claude-blender → PlasmaDX-Blender     │ Attach: tb (backup)          │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ SSH
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  CLIENT PC (Intel 16-core)                                                  │
│                                                                             │
│  ┌─ Terminal ──────────────┐  ┌─ Terminal ──────────────┐                  │
│  │ ssh server → tp         │  │ ssh server → tm         │                  │
│  │ Claude on PINN          │  │ Claude on MultiAgent    │                  │
│  └─────────────────────────┘  └─────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Why This Hybrid Approach?

| Feature | Cursor Windows | tmux Sessions |
|---------|----------------|---------------|
| Visual (separate monitors) | ✅ | ❌ |
| Persistence (survives crashes) | ❌ | ✅ |
| Remote access (SSH) | ❌ | ✅ |
| Full IDE features | ✅ | Terminal only |

**Best practice:** Use Cursor windows for your primary work, tmux for persistence + remote access.

---

## Current Working Configuration

### Git Worktrees

```bash
$ git worktree list
/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean       [0.22.12]
/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Blender     [feature/blender-integration]
/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-MultiAgent  [feature/multi-agent-workflow]
/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-PINN-v4     [feature/pinn-v4-siren-optimizations]
/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Test        [test/learn-worktrees]
```

### tmux Sessions

```bash
$ tmux ls
claude-blender: 1 windows  # → PlasmaDX-Blender (feature/blender-integration)
claude-multi:   1 windows  # → PlasmaDX-MultiAgent (feature/multi-agent-workflow)
claude-pinn:    1 windows  # → PlasmaDX-PINN-v4 (feature/pinn-v4-siren-optimizations)
```

### Shell Aliases (in ~/.bashrc)

```bash
alias tb='tmux attach -t claude-blender'   # Blender worktree
alias tp='tmux attach -t claude-pinn'       # PINN v4 worktree
alias tm='tmux attach -t claude-multi'      # MultiAgent worktree
alias tls='tmux ls'                         # List all sessions
alias tstart='scripts/tmux_per_ai_sessions.sh'  # Start all sessions
alias thelp='...'                           # Show help
```

---

## tmux Keyboard Reference

| Action | Keys |
|--------|------|
| Detach (keeps running) | `Ctrl+b d` |
| Window list | `Ctrl+b w` |
| New window | `Ctrl+b c` |
| Vertical split | `Ctrl+b |` |
| Horizontal split | `Ctrl+b -` |
| Switch panes | `Ctrl+b arrow` |
| Kill current pane | `Ctrl+b x` |

---

## Daily Workflow

### From Server (Local IDE)

1. Open single Cursor window
2. In terminal: `tp` (or `tb`, `tm`) to attach to worktree
3. Work with Claude Code in that worktree
4. `Ctrl+b d` to detach
5. Switch: `tb` for Blender, `tm` for MultiAgent, etc.

### From Client PC (SSH)

1. SSH into server: `ssh your-server`
2. Attach: `tp` (or `tb`, `tm`)
3. Work with Claude Code
4. `Ctrl+b d` to detach when done
5. Sessions keep running on server

---

## Initial Setup (One-Time)

### 1. Create tmux config

```bash
cat > ~/.tmux.conf << 'EOF'
set -g mouse on
set -g base-index 1
setw -g pane-base-index 1
set -g renumber-windows on
setw -g automatic-rename on
set -g history-limit 50000
set -g default-terminal "screen-256color"
bind | split-window -h -c "#{pane_current_path}"
bind - split-window -v -c "#{pane_current_path}"
EOF
```

### 2. Start sessions (script or manual)

**Using script:**
```bash
chmod +x scripts/tmux_per_ai_sessions.sh
./scripts/tmux_per_ai_sessions.sh
```

**Manual:**
```bash
ROOT="/mnt/d/Users/dilli/AndroidStudioProjects"
tmux new-session -d -s claude-blender -c "$ROOT/PlasmaDX-Blender"
tmux new-session -d -s claude-pinn -c "$ROOT/PlasmaDX-PINN-v4"
tmux new-session -d -s claude-multi -c "$ROOT/PlasmaDX-MultiAgent"
```

### 3. Add aliases to ~/.bashrc

```bash
alias tb='tmux attach -t claude-blender'
alias tp='tmux attach -t claude-pinn'
alias tm='tmux attach -t claude-multi'
alias tls='tmux ls'
```

### 4. Start Claude Code in each session

```bash
tb                    # Attach to Blender session
claude                # Start Claude Code (or with --model opus)
# Ctrl+b d to detach

tp                    # Attach to PINN session
claude --model opus   # Start with Opus
# Ctrl+b d to detach

# etc.
```

---

## Troubleshooting

### Session points to wrong directory

```bash
# Kill and recreate
tmux kill-session -t claude-blender
tmux new-session -d -s claude-blender -c "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Blender"
```

### Check session working directory

```bash
tmux attach -t claude-blender
pwd  # Should show the worktree path
```

### Server reboot - sessions lost

```bash
# Just run the startup script again
./scripts/tmux_per_ai_sessions.sh
# Then start Claude Code in each session
```

### "Session already exists" error

```bash
# That's fine - it just means it's already running
tmux attach -t <session-name>
```

---

## Why This Setup Works

| Problem | Solution |
|---------|----------|
| Anthropic blocks multiple Claude terminals per IDE | Each tmux session is isolated |
| Opening multiple IDE windows is heavy | Single IDE + tmux switching |
| Need persistent sessions | tmux survives disconnects |
| Multi-PC development | SSH into tmux sessions from anywhere |
| Context switching between features | Each worktree has its own session/branch |

---

## Files

- `scripts/tmux_per_ai_sessions.sh` - Creates separate sessions per worktree
- `scripts/tmux_single_session.sh` - Alternative: single session with windows
- `~/.tmux.conf` - tmux configuration
- `~/.bashrc` - Shell aliases (tb, tp, tm, tls)
