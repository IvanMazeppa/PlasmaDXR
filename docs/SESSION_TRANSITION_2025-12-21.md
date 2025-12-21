# Session Transition Document - 2025-12-21

## Quick Start for New Session

**Open this folder in Cursor/VS Code:**
```
~/projects/PlasmaDXR
```

Or via WSL Remote:
```
code ~/projects/PlasmaDXR
```

---

## What Changed This Session

### 1. Project Migration to WSL Native Filesystem

| Before | After |
|--------|-------|
| `/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean` | `~/projects/PlasmaDXR` |
| 43 GB / 255k files | 26 GB / 112k files |
| Windows filesystem (9P overhead) | Native ext4 (faster) |

**Git Status**: Clean migration, branch `0.24.6/shader-binary-memory-cache`, 1 unpushed commit

### 2. Cleanup Completed

- Removed 17 GB of regenerable artifacts (venvs, pycache, build cache)
- Rewrote `.gitignore` from 512 lines to 171 lines with proper patterns
- Created `scripts/cleanup_project.sh` for future cleanups

### 3. NanoVDB Pipeline Created

**Root Cause of Blocky VDBs**: Blender Mantaflow domain resolution too low (64-128 voxels vs 256-512+ for professional assets).

**New Files Created:**

| File | Purpose |
|------|---------|
| `docs/NanoVDB/BLENDER_HIGH_RES_EXPORT_GUIDE.md` | How to fix blocky VDBs, resolution targets |
| `docs/NanoVDB/SHADOW_RAYS_FOR_NANOVDB.md` | Shadow ray implementation plan (3 options) |
| `scripts/blender_export_nvdb.py` | Blender headless NanoVDB export |
| `scripts/export_nvdb.sh` | Easy CLI wrapper for batch export |

---

## Current Branch & Git State

```bash
cd ~/projects/PlasmaDXR
git status
# On branch 0.24.6/shader-binary-memory-cache
# Your branch is ahead of 'origin/0.24.6/shader-binary-memory-cache' by 1 commit

# Push when ready:
git push
```

**Symlinks still point to Windows** (`/mnt/d/...`):
- `dlss` → DLSS SDK (fine for Windows builds)
- `VDBs` → Shared VDB assets
- `shared_*` → Shared resources

These work because the built `.exe` runs on Windows anyway.

---

## Pending Tasks

1. **Shadow Rays for NanoVDB** - Implementation plan ready in `docs/NanoVDB/SHADOW_RAYS_FOR_NANOVDB.md`
   - Phase 1: Single-light in-shader shadow marching (quick win)
   - Phase 2: Multi-light with temporal accumulation
   - Phase 3: TLAS integration (optional)

2. **Create feature branch for NanoVDB pivot** (when ready):
   ```bash
   git checkout -b feature/nanovdb-volumetric-engine
   ```

---

## Key Technical Context

### NanoVDB System Status
- **Working**: Volumetric rendering with multi-light support, procedural fog
- **Fixed this week**: Blue channel bug, multi-light support (only 4 of 9 lights were working)
- **Not working**: Shadow rays have no effect on NanoVDB (documented in SHADOW_RAYS doc)

### Blocky VDB Fix
```
Blender > Physics > Fluid > Settings > Resolution Divisions: 300+
Blender > Physics > Fluid > Cache > Precision: Full (not Half!)
```

### Quick Export Command
```bash
./scripts/export_nvdb.sh my_smoke.blend --output ./volumes --resolution 300
```

### File Size Reference (Quality Indicator)
| Size | Quality |
|------|---------|
| 30-60 MB | Hero volume (smooth) |
| 10-20 MB | Background element |
| 3-5 MB | Blocky (too low res) |

---

## Build System

```bash
# Generate build files (one-time after migration)
cd ~/projects/PlasmaDXR
mkdir -p build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64

# Build (from WSL, calls Windows MSBuild)
"/mnt/c/Program Files/Microsoft Visual Studio/2022/Community/MSBuild/Current/Bin/MSBuild.exe" \
    build/PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64 /t:PlasmaDX-Clean /v:minimal
```

**Note**: Build outputs go to `build/bin/Debug/` and run on Windows. The project lives in WSL for faster git/editor operations, but compilation targets Windows.

---

## Backup Location

Full backup made to B2 before migration (as mentioned by user).

Original Windows folder still exists at:
```
/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean
```

---

## MCP Servers Available

These should auto-reconnect when opening in Cursor:
- `dxr-shadow-engineer` - Shadow technique research
- `gaussian-analyzer` - Particle analysis
- `material-system-engineer` - Material/shader generation
- `log-analysis-rag` - PIX/log analysis
- `path-and-probe` - Probe grid specialist
- `blender-manual` - Blender 5 documentation search
- `dxr-image-quality-analyst` - Screenshot quality analysis

---

## Conversation Context Summary

### What We Discussed
1. NanoVDB volumetrics work beautifully - particles were the problem
2. Want to pivot to NanoVDB-first architecture
3. Blender-generated VDBs look blocky (resolved: resolution issue)
4. Shadow rays can work with NanoVDB (3 options documented)
5. Dev environment cleanup and WSL migration (completed)

### User Preferences (from CLAUDE.md)
- **Brutal honesty preferred** over sugar-coating
- **Explain the "why"** when correcting
- **Validate effort** even when technically incorrect
- Break complex problems into manageable steps

### Project Vision
- Real-time RT video production using Blender + PlasmaDX pipeline
- DLSS 4, RTXDI infrastructure already in place
- NanoVDB for volumetrics, physics-driven luminous star particles for dynamic lighting

---

## Next Session Checklist

1. [ ] Open `~/projects/PlasmaDXR` in Cursor
2. [ ] Run `git status` to verify clean state
3. [ ] Optionally push the cleanup commit: `git push`
4. [ ] Continue with shadow ray implementation or other NanoVDB work

---

*Document created: 2025-12-21*
*For: Ben (PlasmaDXR project owner)*
