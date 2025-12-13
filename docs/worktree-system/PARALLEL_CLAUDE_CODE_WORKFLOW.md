# Parallel Claude Code Workflow with Git Worktrees

**Problem:** Claude Code 2.0.61 removed multi-terminal support within a single IDE window.

**Solution:** Git worktrees + multiple Cursor windows.

---

## Quick Setup

```bash
# From main repo
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean

# Create worktrees for parallel work
git worktree add ../PlasmaDX-NanoVDB feature/nanovdb-improvements
git worktree add ../PlasmaDX-PINN feature/pinn-integration
git worktree add ../PlasmaDX-Optimize optimize/performance

# Open each in separate Cursor windows
cursor ../PlasmaDX-NanoVDB
cursor ../PlasmaDX-PINN
cursor ../PlasmaDX-Optimize
```

Each window can have its own Claude Code terminal - the restriction only applies to multiple terminals *within the same window*.

---

## Useful Commands

```bash
# List all worktrees
git worktree list

# Remove a worktree when done
git worktree remove ../PlasmaDX-NanoVDB

# Prune stale worktree references
git worktree prune
```

---

## Git Flow Integration (Optional)

Use `/git-workflow:feature <name>` to create properly named branches, then create worktrees for them:

```bash
# In Claude Code, create feature branch
/git-workflow:feature nanovdb-sampling

# Then create worktree for it
git worktree add ../PlasmaDX-NanoVDB feature/nanovdb-sampling
```

---

## Key Points

- **Worktrees** = Multiple working directories (parallel Claude Code sessions)
- **Git Flow** = Branch naming convention (organization)
- **Disk space**: Each worktree duplicates working files (~1-2GB with build dirs)
- **Shared**: Git history, remotes, stashes all shared across worktrees
- **Can't reuse branches**: Each worktree must be on a different branch

---

## Cleanup

When finished with parallel work:

```bash
# Merge your feature branch
git checkout main
git merge feature/nanovdb-improvements

# Remove the worktree
git worktree remove ../PlasmaDX-NanoVDB

# Delete the branch if no longer needed
git branch -d feature/nanovdb-improvements
```
