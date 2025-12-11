# Tmux + Git Worktrees + Claude Code Workflow

Single-window workflow for running multiple Claude instances across git worktrees.

---

## Setup Summary

| Component | Location |
|-----------|----------|
| Worktrees | `/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-*` |
| Tmux config | `~/.tmux.conf` |
| Shell aliases | `~/.bashrc` |
| Session saves | `~/.tmux/resurrect/` (auto-managed) |

---

## Daily Workflow

### Start of Day
```bash
tmux                    # Start tmux
Ctrl+B, Ctrl+R          # Restore yesterday's sessions
tb                      # Attach to blender session (or tm, tp, etc.)
claude                  # Start Claude in the session
```

### End of Day
```bash
Ctrl+B, Ctrl+S          # Save all sessions (or just type: tsave)
# Now safe to shut down
```

### Quick Reference
```bash
tls                     # List all sessions
tb                      # Attach to claude-blender
tm                      # Attach to claude-multi
tp                      # Attach to claude-pinn
tsave                   # Save sessions (before shutdown)
trestore                # Restore sessions (after startup)
tshutdown               # Save + confirmation message
tstart                  # Create fresh sessions (if needed)
```

---

## Working Inside Tmux

### While Claude is Running - Access Shell

**Option 1: Split pane** (recommended)
```
Ctrl+B, -               # Split horizontal (shell below Claude)
Ctrl+B, |               # Split vertical (shell beside Claude)
Ctrl+B, ↑/↓/←/→         # Navigate between panes
Ctrl+B, x               # Close current pane
Ctrl+B, z               # Zoom pane (toggle fullscreen)
```

**Option 2: New window** (like a tab)
```
Ctrl+B, c               # Create new window
Ctrl+B, n / p           # Next / Previous window
Ctrl+B, 1/2/3           # Jump to window number
Ctrl+B, w               # Visual window picker
```

### Session Management
```
Ctrl+B, d               # Detach (leave everything running)
Ctrl+B, s               # Switch between sessions
Ctrl+B, $               # Rename current session
Ctrl+B, ,               # Rename current window
```

### Scrolling / Copy Mode
```
Ctrl+B, [               # Enter scroll mode
q                       # Exit scroll mode
↑/↓ or PgUp/PgDn        # Scroll
```
Mouse wheel also works (scrolling enters copy mode automatically).

---

## Creating New Worktrees

From a shell pane (Ctrl+B, - to split):

```bash
# Create new worktree
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean
git worktree add ../PlasmaDX-MultiAgent -b feature/multi-agent

# Create tmux session for it
tmux new-session -d -s claude-multi -c /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-MultiAgent

# Save the new session layout
tsave
```

---

## Session Persistence & Crash Protection

### Automatic Protection (tmux-continuum)
- **Auto-saves every 5 minutes** - protects against crashes
- **Auto-restores on tmux start** - just run `tmux` after a crash

### What Gets Saved
- All sessions and their names
- Window layouts and pane splits
- Working directory of each pane
- Scroll history

### What Doesn't Get Saved
- Running processes (Claude won't auto-restart)
- Unsaved editor buffers
- **Uncommitted git changes** - commit frequently!

### Manual Save/Restore
```
Ctrl+B, Ctrl+S          # Save (inside tmux)
Ctrl+B, Ctrl+R          # Restore (inside tmux)
```
Or from shell:
```bash
tsave                   # Save from command line
trestore                # Restore from command line
```

### Crash Recovery
After a crash, just run:
```bash
tmux                    # Auto-restores last saved state
```
Then start Claude in each session again.

### Best Practices for Unreliable Systems
1. **Commit early, commit often** - git commits survive crashes
2. **Push to GitHub** - protects against disk failure
3. **Claude conversations are auto-saved** to `~/.claude/`
4. tmux auto-saves every 5 min, but layouts only (not running processes)

---

## MCP Servers

MCP servers are configured at **user scope** in `~/.claude.json` and are automatically available in all worktrees. No per-project configuration needed.

To check available servers:
```bash
claude mcp list
```

---

## Troubleshooting

### "No sessions" after restart
```bash
tmux
Ctrl+B, Ctrl+R          # Restore saved sessions
# OR
tstart                  # Create fresh sessions
```

### Session won't restore
```bash
ls ~/.tmux/resurrect/   # Check if saves exist
# If empty, sessions weren't saved before shutdown
```

### Kill a stuck session
```bash
tmux kill-session -t session-name
```

### Claude won't start in session
- Check you're in the right directory: `pwd`
- Check Claude is installed: `which claude`
- Check auth: `claude --version`

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  Your Terminal (one window)                             │
│  ┌───────────────────────────────────────────────────┐  │
│  │  tmux                                             │  │
│  │  ┌─────────────────┬─────────────────────────┐   │  │
│  │  │ [1] blender     │ [2] multi    [3] pinn   │   │  │
│  │  │                 │                         │   │  │
│  │  │  Claude Code    │  Claude Code            │   │  │
│  │  │  (Blender wt)   │  (Multi-Agent wt)       │   │  │
│  │  │                 │                         │   │  │
│  │  │─────────────────┤                         │   │  │
│  │  │  Shell pane     │                         │   │  │
│  │  │  (git, etc.)    │                         │   │  │
│  │  └─────────────────┴─────────────────────────┘   │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

Switch between windows with `Ctrl+B, 1/2/3` or `Ctrl+B, w`.

---

---

## Worktree Types & Risk Levels

### Safe Worktrees (Documentation/Scripts Only)
Examples: `PlasmaDX-Blender`, agent development, recipe writing

**These don't touch C++/HLSL code, so they can't break the build.**

Workflow:
```bash
# Work on your changes
git add -A && git commit -m "Add blender recipes"
git push

# Can merge to main anytime via PR - no testing needed
```

### Risky Worktrees (Code Changes)
Examples: `PlasmaDX-MultiAgent` (acceleration structures), `PlasmaDX-PINN-v4`

**These modify C++/HLSL and could cause crashes.**

Workflow:
```bash
# 1. Make changes and commit
git add -A && git commit -m "WIP: optimize BLAS rebuild"
git push

# 2. Build and test IN THE WORKTREE (see below)
# 3. Only create PR after testing passes
```

---

## Building in Worktrees

Each worktree needs its own build directory. **Do not share builds between worktrees.**

### Quick Commands (Recommended)
```bash
wt-init      # First-time setup: creates build/ and runs cmake
wt-build     # Compile the project
wt-run       # Run the executable
```

### Manual Commands (if needed)

**First-Time Setup:**
```bash
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-MultiAgent
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
```

**Rebuild:**
```bash
"/mnt/c/Program Files/Microsoft Visual Studio/2022/Community/MSBuild/Current/Bin/MSBuild.exe" \
  build/PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64 /t:PlasmaDX-Clean /v:minimal
```

**Run:**
```bash
./build/bin/Debug/PlasmaDX-Clean.exe
```

---

## Safe Merge Protocol

### For Safe Changes (docs, scripts, agents)
```
1. Commit and push to your branch
2. Create PR on GitHub
3. Merge immediately (no testing needed)
```

### For Risky Changes (C++/HLSL code)
```
1. Commit and push to your branch
2. Build in the worktree (cmake + MSBuild)
3. Run and test:
   - Does it launch without crashing?
   - Do the changes work as intended?
   - Is performance acceptable?
4. If tests pass → Create PR
5. Merge to main
6. Rebuild PlasmaDX-Clean to verify
```

### Emergency Rollback
If something breaks main after merging:
```bash
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean
git revert HEAD      # Creates a new commit that undoes the last one
git push
```

---

## Commit Message Convention

Use prefixes to indicate risk level:

| Prefix | Meaning | Example |
|--------|---------|---------|
| `docs:` | Documentation only | `docs: add hydrogen cloud recipe` |
| `agent:` | Agent prompts/scripts | `agent: add blender guardrails` |
| `feat:` | New feature (needs testing) | `feat: add BLAS update optimization` |
| `fix:` | Bug fix (needs testing) | `fix: resolve particle flicker` |
| `perf:` | Performance change (needs testing) | `perf: optimize acceleration structure` |
| `WIP:` | Work in progress (don't merge yet) | `WIP: experimenting with mesh shaders` |

---

## Git Push Commands

### From Any Worktree
```bash
git push                           # Push current branch
git push -u origin branch-name     # First push of new branch
```

### If Worktree Push Fails (fallback)
```bash
gpush                              # Uses main repo to push
```

---

*Last updated: 2025-12-11*
