# B1 Multi-Monitor Workspace Layout Specification

**Status:** DRAFT - To be implemented
**Machine:** B1 (Main Work PC, 4 monitors)
**Last Updated:** 2025-12-12

---

## Machine Architecture

| Machine | Role | Key Details |
|---------|------|-------------|
| **B1** | Main development PC | 4 monitors, tmux system, primary workspace |
| **B2** | Support/offload PC | i7-6900K, 32GB RAM, SSH to B1, resource offloading |

---

## B1 Workspace Layout (4 Monitors)

### Overview

Two logical "workspace pairs" across 4 monitors:

```
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   MONITOR 1     │ │   MONITOR 2     │ │   MONITOR 3     │ │   MONITOR 4     │
│                 │ │                 │ │                 │ │                 │
│  Blender        │ │  Claude (tmux)  │ │  PlasmaDX-Clean │ │  Claude         │
│  Worktree       │ │  tb/tm/tn       │ │  (main repo)    │ │  --continue     │
│                 │ │                 │ │                 │ │                 │
│  2 bash prompts │ │  Worktree AI    │ │  No tmux        │ │  Main project   │
│  tmux in #2     │ │  sessions       │ │  Standard setup │ │  AI assistant   │
└─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘
     LEFT                                                        RIGHT
```

### Workspace Pair 1: Worktree Development (Monitors 1-2)

**Monitor 1 - Blender Worktree IDE**
- Cursor window: maximized, standard layout (no agent pane)
- Project: `PlasmaDX-Blender` worktree
- Terminal setup:
  - Bash prompt 1: General commands
  - Bash prompt 2: tmux launched here (attach to `claude-blender`)

**Monitor 2 - Worktree Claude Sessions**
- Claude terminal (fullscreen)
- Connected via tmux to worktree sessions:
  - `tb` - claude-blender
  - `tm` - claude-multi
  - `tn` - claude-nanovdb
- Switch between sessions as needed

### Workspace Pair 2: Main Project Development (Monitors 3-4)

**Monitor 3 - PlasmaDX-Clean IDE**
- Cursor window: maximized, standard layout (no agent pane)
- Project: `PlasmaDX-Clean` (main repository)
- No tmux - direct terminal usage
- Standard development workflow

**Monitor 4 - Main Project Claude**
- Claude terminal (fullscreen)
- Launched with `--continue` flag
- Dedicated to main PlasmaDX-Clean work
- Maintains conversation context across sessions

---

## Cursor Window Configuration

All Cursor windows use:
- **Layout:** Maximized, typical layout
- **Agent pane:** Hidden (no extra agent pane)
- **Terminal:** Integrated terminal visible

---

## Startup Sequence (To Be Automated)

### Manual Steps Currently Required:

1. Open Cursor windows (4 total)
2. Position each on correct monitor
3. Open correct project folder in each
4. Set up terminal layout in each
5. Launch tmux in Monitor 1's second terminal
6. Start Claude sessions

### Desired Automation:

```bash
# Ideal single command to set up entire workspace
workspace-start
```

Would configure:
- Window positions per monitor
- Project folders auto-loaded
- Terminal splits configured
- tmux sessions attached
- Claude launched with correct flags

---

## B2 Remote Connection Workflow

When B2 connects to B1 via SSH:

1. SSH connection established
2. Cursor opens with Remote-SSH extension
3. **Desired:** Project folder auto-loads (currently manual)
4. Ready to work on B1 resources from B2

### Current Pain Point:
- Must manually do `Ctrl+Shift+P` → `File: Open Folder...`
- Enter path `/mnt/d/Users/dilli/AndroidStudioProjects/...`
- Repeat for each worktree

### Desired Behavior:
- Script/config that opens Cursor with correct remote folder
- Potentially: `cursor --remote ssh-remote+B1 /path/to/project`

---

## Implementation Notes

### Windows Automation Options:
- PowerShell with window positioning APIs
- AutoHotkey for window management
- Cursor CLI with `--goto` and window flags
- Windows Terminal with saved layouts

### tmux Integration:
- Existing `tstart` script handles session creation
- Need to add window positioning logic
- Consider `tmux` session restore for terminal layouts

---

## Files Related to This System

- `scripts/tmux_per_ai_sessions.sh` - tmux session management
- `~/.bashrc` - Aliases (tb, tp, tm, tn, tstart, topen, thelp)
- `docs/worktree-system/` - Documentation

---

## TODO

- [ ] Design window positioning script (PowerShell or AutoHotkey)
- [ ] Test Cursor CLI remote folder opening
- [ ] Create unified startup command
- [ ] Document B2 remote connection automation
- [ ] Test full workflow end-to-end
