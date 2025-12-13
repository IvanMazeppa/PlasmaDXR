# PlasmaDX Worktree System Documentation

**Last Updated:** 2025-12-12

This folder contains all documentation for the multi-instance Claude Code workflow using git worktrees and tmux.

---

## Quick Start (After Server Reboot)

```bash
# 1. Start tmux sessions for all worktrees
tstart

# 2. Attach to whichever session you need
tb    # Blender worktree
tp    # PINN worktree
tm    # MultiAgent worktree

# 3. In each session, start Claude Code
claude

# 4. Detach when done (session keeps running)
# Press: Ctrl+b d
```

---

## Documentation Index

| File | Purpose |
|------|---------|
| `tmux_worktrees_setup.md` | **Main guide** - Architecture, setup, daily workflow |
| `4_INSTANCE_WORKFLOW_GUIDE.md` | 4-instance layout across 2 PCs |
| `PARALLEL_CLAUDE_CODE_WORKFLOW.md` | Quick reference for git worktrees |
| `SSH_SETUP_TROUBLESHOOTING_PROMPT.md` | SSH configuration for remote access |
| `SCRIPTS_README.txt` | Windows client scripts usage guide |

---

## Scripts

### Server (Linux/WSL)
| Script | Purpose |
|--------|---------|
| `tmux_per_ai_sessions.sh` | Create separate tmux sessions per worktree |
| `tmux_single_session.sh` | Single tmux session with multiple windows |

### Client (Windows)
| Script | Purpose |
|--------|---------|
| `open_worktrees.bat` | Open worktrees in Cursor via SSH-Remote |
| `open_worktrees.ps1` | PowerShell version with -List and -Help |
| `connect_worktree.bat` | Interactive menu to SSH into a worktree |
| `setup_ssh_key.bat` | One-time SSH key setup (passwordless login) |

**All scripts use absolute paths - can be run from anywhere.**

---

## Shell Aliases (in ~/.bashrc)

```bash
alias tb='tmux attach -t claude-blender'
alias tp='tmux attach -t claude-pinn'
alias tm='tmux attach -t claude-multi'
alias tls='tmux ls'
alias tstart='/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/scripts/tmux_per_ai_sessions.sh'
alias thelp='echo "tb=blender, tp=pinn, tm=multi, tls=list, tstart=create sessions"'
```

**To add these permanently:**
```bash
cat >> ~/.bashrc << 'EOF'
# PlasmaDX tmux worktree aliases
alias tb='tmux attach -t claude-blender'
alias tp='tmux attach -t claude-pinn'
alias tm='tmux attach -t claude-multi'
alias tls='tmux ls'
alias tstart='/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/scripts/tmux_per_ai_sessions.sh'
alias thelp='echo "tb=blender, tp=pinn, tm=multi, tls=list, tstart=create sessions"'
EOF
source ~/.bashrc
```

---

## Current Worktrees

```
PlasmaDX-Clean       → main branch (0.23.x)
PlasmaDX-Blender     → feature/blender-integration
PlasmaDX-PINN-v4     → feature/pinn-v4-siren-optimizations
PlasmaDX-MultiAgent  → feature/multi-agent-workflow
```

---

## tmux Quick Reference

| Action | Keys |
|--------|------|
| Detach (keeps running) | `Ctrl+b d` |
| List windows | `Ctrl+b w` |
| Kill session | `tmux kill-session -t name` |
| List sessions | `tmux ls` or `tls` |

---

## Troubleshooting

### Script can't run from outside Cursor
The `tstart` alias uses an absolute path, so it works from any terminal. If you want to run the script directly:
```bash
# Use absolute path
/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/scripts/tmux_per_ai_sessions.sh

# Or add scripts folder to PATH in ~/.bashrc
export PATH="$PATH:/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/scripts"
```

### Sessions lost after reboot
Just run `tstart` again. Sessions don't persist across reboots.

### Wrong working directory
```bash
tmux kill-session -t claude-blender
# Then run tstart again
```
