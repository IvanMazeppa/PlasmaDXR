# B2 Quick Reference

## Commands (run from B1)

| Command | What it does |
|---------|--------------|
| `b2m` | Attach to Main session |
| `b2b` | Attach to Blender session |
| `b2n` | Attach to NanoVDB session |
| `b2l` | Attach to Luminous session |
| `b2status` | Check B2 status |
| `b2init` | Recreate sessions after reboot |
| `b2sync LuminousStars` | Sync worktree from B2 to B1 |
| `b2sync all` | Sync all worktrees |

## Build Commands
- `build-main` - Build main project
- `build-nanovdb` - Build NanoVDB worktree
- `build-luminous` - Build LuminousStars worktree

## Inside tmux
- Type `claude` to start Claude Code
- `Ctrl+B d` - Detach (leave running)
- `Shift+Left/Right` - Switch windows

## After B2 reboot
Run: `b2init`
