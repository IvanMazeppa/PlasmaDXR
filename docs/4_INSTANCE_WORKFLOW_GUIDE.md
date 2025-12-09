# 4-Instance Parallel Claude Code Workflow

**Your Setup:** 2 machines, 4 parallel Claude Code sessions

---

## Instance Layout

| Instance | Location | Worktree | Branch | Purpose |
|----------|----------|----------|---------|---------|
| **1** | This PC | `/mnt/d/.../PlasmaDX-Clean` | `0.22.10` | NanoVDB optimisations|
| **2** | This PC | `../PlasmaDX-Blender` | `feature/blender-integration` | Blender .vdb integration |
| **3** | Other PC | Remote SSH → `../PlasmaDX-MultiAgent` | `feature/multi-agent-workflow` | Multi-agent optimisations |
| **4** | Other PC | Remote SSH → `../PlasmaDX-PINN-v4` | `feature/pinn-v4-siren-optimizations` | PINN v4 / SIREN v2 |

---

## Opening Instances on This PC

```bash
# Instance 1 (already open): Main repo
# (This current Cursor window)

# Instance 2: Blender integration
cursor ../PlasmaDX-Blender
```

**In each Cursor window:**
1. Open folder via Remote WSL
2. Start Claude Code terminal
3. Work independently

---

## Connecting Other PC

### Option A: SSH into This PC's WSL (Recommended)

**On your other PC:**

```bash
# SSH into this PC's WSL
ssh your-user@this-pc-ip

# Navigate to worktrees
cd /mnt/d/Users/dilli/AndroidStudioProjects/

# Open Instance 3
cursor PlasmaDX-MultiAgent

# Open Instance 4
cursor PlasmaDX-PINN-v4
```

### Option B: Cursor Remote-SSH Extension

1. Install "Remote - SSH" extension on other PC's Cursor
2. Connect to this PC: `Ctrl+Shift+P` → "Remote-SSH: Connect to Host"
3. Enter: `ssh://your-user@this-pc-ip`
4. Open folder: `/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-MultiAgent`
5. Repeat for PINN-v4 worktree

---

## Daily Workflow

### Starting Work Session

```bash
# Check worktrees
git worktree list

# Each worktree is independent - open in separate Cursor windows
cursor ../PlasmaDX-Blender          # Window 1
cursor ../PlasmaDX-MultiAgent       # Window 2
cursor ../PlasmaDX-PINN-v4          # Window 3
```

### Committing Changes

Each worktree operates independently:

```bash
# In each worktree
git add .
git commit -m "Feature: Description"
git push origin feature/branch-name
```

### Merging Back to Main

```bash
# When feature is complete, merge from main repo
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean
git checkout main
git merge feature/blender-integration
git push

# Clean up worktree
git worktree remove ../PlasmaDX-Blender
git branch -d feature/blender-integration
```

### Ending Work Session

Don't need to do anything! Worktrees persist. Just close Cursor windows.

---

## Git LFS Note

**Issue:** Windows git-lfs causes problems with worktree creation.

**Current workaround:** LFS filters temporarily disabled during worktree creation, then restored.

**Permanent fix (when you have time):**
```bash
sudo apt-get install git-lfs
git lfs install
```

---

## Troubleshooting

### "Can't create worktree - branch exists"

```bash
# Delete orphaned branch
git branch -D feature/branch-name

# Try again
git worktree add ../PlasmaDX-NewWork -b feature/branch-name
```

### "Working tree clean, nothing to commit"

This is fine! Means all changes are already committed.

### IDE froze during git operation

Known issue. Restart Cursor. Your work is safe (check `git status`).

### Other PC can't see worktrees

Make sure:
1. SSH connection works: `ssh your-user@this-pc-ip`
2. WSL is accessible from network
3. Using correct paths: `/mnt/d/Users/...`

---

## Quick Reference Commands

```bash
# List all worktrees
git worktree list

# Create new worktree
git worktree add ../PlasmaDX-NewFeature -b feature/name

# Remove worktree when done
git worktree remove ../PlasmaDX-NewFeature

# Check status across all worktrees (from main repo)
git worktree list
git branch -vv
```

---

**Last Updated:** 2025-12-08
**Machine:** This PC (WSL Ubuntu) + Other PC (remote)
