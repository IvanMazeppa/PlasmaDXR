# Two-Worktree Development Architecture

**Version:** 1.0
**Status:** Active
**Last Updated:** 2025-12-11

---

## Overview

PlasmaDX uses a **dual-worktree architecture** to separate concerns between asset creation (Blender) and rendering development (NanoVDB). This enables parallel development workflows while maintaining a single source of truth.

---

## Worktree Structure

```
PlasmaDX Repository Worktrees
├── PlasmaDX-Clean/           # Main development (branch: main, 0.23.x)
│   ├── Primary renderer development
│   ├── Core engine features
│   └── Integration of all systems
│
├── PlasmaDX-Blender/         # Asset creation (branch: feature/blender-integration)
│   ├── Blender 5.0 workflow development
│   ├── Recipe library authoring
│   ├── VDB export automation
│   └── Documentation for artists
│
├── PlasmaDX-NanoVDB/         # Volumetric rendering (branch: feature/nanovdb-animated-assets)
│   ├── NanoVDB system development
│   ├── Shader optimization
│   ├── Animation system
│   └── Performance profiling
│
└── (Other worktrees)
    ├── PlasmaDX-PINN-v4/     # ML physics research
    └── PlasmaDX-GaussianImageQ/ # Quality analysis
```

---

## Data Flow Between Worktrees

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      BLENDER WORKTREE                                     │
│            PlasmaDX-Blender (feature/blender-integration)                 │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Agents:                        Outputs:                                  │
│  - blender-manual (MCP)         - .blend source files                    │
│  - blender-scripting            - .vdb cached simulations                │
│  - celestial-body-curator       - Recipe documentation                   │
│  - blender-diagnostics          - Automation scripts                     │
│                                                                           │
│  Key Files:                                                               │
│  - docs/blender_recipes/        (Recipe library)                         │
│  - agents/blender-*/            (Agent specs)                            │
│  - VDBs/Blender_projects/       (Blender source files)                   │
│                                                                           │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     │ VDB Export
                                     │ (OpenVDB → NanoVDB conversion)
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      SHARED ASSET DIRECTORY                               │
│                       VDBs/ (on main branch)                              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  VDBs/                                                                    │
│  ├── NanoVDB/                   (Production .nvdb files)                 │
│  │   ├── cloud_01.nvdb          (31 MB, test cloud)                      │
│  │   └── chimney_smoke/         (Animated sequence)                      │
│  ├── Blender_projects/          (Source .blend files)                    │
│  │   ├── hydrogen_cloud.blend                                            │
│  │   └── vdb_cache/             (Baked VDB output)                       │
│  ├── Clouds/                    (Cloud assets)                           │
│  └── Smoke/                     (Smoke assets)                           │
│                                                                           │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     │ Load .nvdb files
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      NANOVDB WORKTREE                                     │
│          PlasmaDX-NanoVDB (feature/nanovdb-animated-assets)               │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Development Focus:              Key Files:                               │
│  - NanoVDBSystem.h/cpp           - src/rendering/NanoVDBSystem.*         │
│  - Ray marching shader           - shaders/volumetric/nanovdb_raymarch.* │
│  - Animation system              - scripts/convert_vdb_to_nvdb.py        │
│  - Performance optimization      - scripts/inspect_vdb.py                │
│                                                                           │
│  Agents Available:                                                        │
│  - gaussian-analyzer             (Material validation)                   │
│  - materials-council             (Property mapping)                      │
│  - dxr-volumetric-pyro-specialist (Explosion design)                     │
│                                                                           │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     │ Merge to main
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       MAIN WORKTREE                                       │
│                   PlasmaDX-Clean (main / 0.23.x)                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Integrates all features:                                                │
│  - NanoVDB volumetric rendering                                          │
│  - Blender workflow documentation                                        │
│  - Agent ecosystem                                                       │
│  - Production builds                                                     │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Workflow Scenarios

### Scenario 1: Create New Celestial Asset

```
1. BLENDER WORKTREE
   └─► Open VDBs/Blender_projects/hydrogen_cloud.blend
   └─► Follow recipe from docs/blender_recipes/
   └─► Bake simulation (Mantaflow)
   └─► Export to VDBs/Blender_projects/vdb_cache/

2. MAIN WORKTREE (conversion)
   └─► python scripts/convert_vdb_to_nvdb.py VDBs/Blender_projects/vdb_cache/smoke_####.vdb
   └─► Move .nvdb files to VDBs/NanoVDB/

3. NANOVDB WORKTREE (testing)
   └─► Test loading in PlasmaDX
   └─► Adjust rendering parameters
   └─► Profile performance

4. MAIN WORKTREE (integration)
   └─► Merge from feature/nanovdb-animated-assets
   └─► Update documentation
   └─► Commit asset to repository
```

### Scenario 2: Debug Rendering Issue

```
1. MAIN WORKTREE
   └─► Identify visual artifact with NanoVDB rendering

2. NANOVDB WORKTREE
   └─► git checkout feature/nanovdb-animated-assets
   └─► Debug NanoVDBSystem.cpp or nanovdb_raymarch.hlsl
   └─► Test fix

3. MAIN WORKTREE
   └─► git merge feature/nanovdb-animated-assets
   └─► Verify fix in production context
```

### Scenario 3: Add New Recipe

```
1. BLENDER WORKTREE
   └─► git checkout feature/blender-integration
   └─► Create new recipe in docs/blender_recipes/
   └─► Test workflow manually
   └─► Add automation script if needed

2. MAIN WORKTREE
   └─► git merge feature/blender-integration
   └─► Update recipe library index
```

---

## Git Commands Reference

### List All Worktrees

```bash
git worktree list
```

Output:
```
/mnt/d/.../PlasmaDX-Clean           47f3061 [0.23.3]
/mnt/d/.../PlasmaDX-Blender         2ebec6b [feature/blender-integration]
/mnt/d/.../PlasmaDX-NanoVDB         1cb73f6 [feature/nanovdb-animated-assets]
```

### Create New Worktree

```bash
# Create worktree for existing branch
git worktree add ../PlasmaDX-NewFeature feature/new-feature

# Create worktree with new branch
git worktree add -b feature/new-feature ../PlasmaDX-NewFeature main
```

### Remove Worktree

```bash
git worktree remove ../PlasmaDX-OldFeature
```

### Update Worktree from Main

```bash
cd ../PlasmaDX-NanoVDB
git fetch origin
git rebase origin/main  # or git merge origin/main
```

### Merge Feature to Main

```bash
cd ../PlasmaDX-Clean
git merge feature/nanovdb-animated-assets
git push origin main
```

---

## Branch Strategy

```
main (0.23.x)
├── feature/blender-integration
│   └── Blender workflow, recipes, documentation
├── feature/nanovdb-animated-assets
│   └── NanoVDB system development
├── feature/pinn-v4-siren-optimizations
│   └── ML physics research
└── (release branches)
    └── 0.24.x, 0.25.x, etc.
```

### Branch Responsibilities

| Branch | Purpose | Merge Target |
|--------|---------|--------------|
| `main` | Production-ready code | N/A |
| `feature/blender-integration` | Asset creation workflow | `main` |
| `feature/nanovdb-animated-assets` | Volumetric rendering | `main` |
| `feature/pinn-*` | ML physics research | `main` when stable |

---

## Agent Distribution

### Blender Worktree Agents

| Agent | Location | Purpose |
|-------|----------|---------|
| blender-manual | `agents/blender-manual/` | MCP server for Blender docs |
| blender-scripting | `agents/blender-scripting/` | Python script assistance |
| celestial-body-curator | `agents/celestial-body-curator/` | Recipe management |
| blender-diagnostics | `agents/blender-diagnostics/` | Scene troubleshooting |

### NanoVDB/Main Worktree Agents

| Agent | Location | Purpose |
|-------|----------|---------|
| gaussian-analyzer | `agents/gaussian-analyzer/` | Material validation |
| materials-council | `agents/materials-council/` | Property mapping |
| dxr-volumetric-pyro-specialist | `agents/dxr-volumetric-pyro-specialist/` | Explosion design |
| mission-control | `agents/mission-control/` | Strategic coordination |

---

## File Synchronization Notes

### Shared Files (All Worktrees)

These files exist in all worktrees and should stay synchronized:

- `CLAUDE.md` - Project context
- `docs/` - Documentation (merge changes carefully)
- `agents/` - Agent specifications

### Worktree-Specific Files

These files are primarily developed in one worktree:

| File | Primary Worktree | Notes |
|------|------------------|-------|
| `src/rendering/NanoVDBSystem.*` | NanoVDB | Core implementation |
| `shaders/volumetric/*` | NanoVDB | Rendering shaders |
| `docs/blender_recipes/` | Blender | Recipe library |
| `scripts/blender/` | Blender | Blender automation |
| `VDBs/` | Main | Shared assets |

---

## Troubleshooting

### "fatal: not a git repository"

You may have navigated outside a worktree. Check current directory:
```bash
pwd
git rev-parse --git-dir  # Should show path to .git
```

### Merge Conflicts After Rebase

```bash
# In feature worktree
git status  # See conflicted files
# Edit files to resolve conflicts
git add <resolved-files>
git rebase --continue
```

### Stale Worktree Reference

```bash
# After deleting worktree directory manually
git worktree prune  # Clean up stale references
```

---

## Related Documentation

- [NanoVDB System Overview](./NANOVDB_SYSTEM_OVERVIEW.md)
- [VDB Pipeline Agents](./VDB_PIPELINE_AGENTS.md)
- [Blender Workflow Spec](../BLENDER_PLASMADX_WORKFLOW_SPEC.md)

---

*Document maintained by: Claude Code Agent Ecosystem*
