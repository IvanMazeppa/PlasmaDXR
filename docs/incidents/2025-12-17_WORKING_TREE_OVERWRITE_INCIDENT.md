# Incident Report: Working Tree Overwrite (2025-12-17)

**Date:** December 17, 2025
**Severity:** Medium (work recovered, no permanent loss)
**Affected Branch:** `feature/luminous-stars`
**Affected Commit:** `edb3f11` (Luminous Star Particles - Phase 3.9)

---

## Summary

Completed work on the Luminous Star Particles feature was unexpectedly removed from the working tree on both B1 and B2 machines, despite the commit (`edb3f11`) remaining intact. The git commit preserved all changes, allowing full recovery via `git restore`.

---

## Timeline

| Time | Event |
|------|-------|
| Dec 16, ~19:49 | Commit `edb3f11` created: "feat: complete Luminous Star Particles (Phase 3.9)" |
| Dec 16, evening | Work completed, user logged off |
| Dec 17, ~00:25 | User returns, discovers integration code missing from working tree |
| Dec 17, ~00:30 | Investigation reveals working tree files overwritten with older versions |
| Dec 17, ~00:35 | All files restored via `git restore` |

---

## What Was Affected

### Files with Removed Code (Working Tree Only)

| File | Lines Removed | Content Lost |
|------|---------------|--------------|
| `src/core/Application.cpp` | ~100 lines | LuminousParticleSystem integration, ImGui controls, Update/Render hooks |
| `src/core/Application.h` | ~10 lines | `#include`, member variable, toggle flag |
| `shaders/particles/particle_physics.hlsl` | 52 lines | Star particle GPU initialization (first 16 particles) |
| `CMakeLists.txt` | 2 lines | LuminousParticleSystem.cpp/.h in build |
| `CLAUDE.md` | ~30 lines | Luminous Star documentation section |

### Symlinks Deleted

| Symlink | Target |
|---------|--------|
| `DLSS` | `/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/dlss` |
| `external/imgui` | `/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/external/imgui` |

### What Was NOT Affected

- Git commit history (all commits intact)
- `.git/objects/` (immutable commit data)
- `LuminousParticleSystem.cpp` and `.h` (new files, not overwritten)
- Untracked files

---

## Root Cause Analysis

### Confirmed Cause: Reverse Sync Direction (B2 → B1)

**What Actually Happened:**

1. Luminous Stars work was completed on **B1** and committed (`edb3f11`)
2. **B2 never received the new code** - no `git push` or sync TO B2 was performed
3. `b2sync LuminousStars` ran (manually or accidentally) - syncing FROM B2 TO B1
4. B2's older files overwrote B1's newer files
5. New files (`LuminousParticleSystem.cpp/.h`) survived because they didn't exist on B2

```
What should have happened:
[B1 - new work] ──sync──> [B2 - old state]

What actually happened:
[B2 - old state] ──sync──> [B1 - new work]  ← WRONG DIRECTION
                           (new work overwritten)
```

**Why new files survived:**
- `LuminousParticleSystem.cpp` and `.h` were NEW files created on B1
- They didn't exist on B2, so the sync had nothing to overwrite them with
- Modified files (Application.cpp, etc.) DID exist on B2 in their older state
- The sync replaced B1's newer versions with B2's older versions

**Contributing factors:**
- Sleep deprivation leading to running wrong sync command
- No pre-sync verification to check for uncommitted/unpushed work
- Sync script doesn't verify direction or warn about potential overwrites

---

## Resolution

All files restored from the intact commit using git:

```bash
# Restore core integration files
git restore src/core/Application.cpp src/core/Application.h

# Restore shader with star initialization
git restore shaders/particles/particle_physics.hlsl

# Restore build system and documentation
git restore CMakeLists.txt CLAUDE.md

# Restore symlinks
git restore DLSS external/imgui
```

**Verification:**
```bash
# Confirmed 17 references restored
grep -c "m_luminousParticles" src/core/Application.cpp
# Output: 17
```

---

## Lessons Learned

### Why Git Saved the Day

1. **Commits are immutable** - Once committed, data is safe in `.git/objects/`
2. **Working tree is expendable** - Can always be regenerated from commits
3. **Frequent commits matter** - If this work hadn't been committed, it would be permanently lost

### Warning Signs to Watch For

- Unexpected file size changes between machines
- Symlinks disappearing or becoming regular files
- `git status` showing large unexpected diffs
- Modified files appearing "older" than expected

---

## Recommendations

### Immediate Actions

1. **Audit backup/sync configuration**
   - Check sync direction (which machine is source vs destination)
   - Verify sync excludes `.git/` directory properly
   - Check symlink handling settings

2. **Check sync logs**
   - Look for sync operations that ran between Dec 16 evening and Dec 17 morning
   - Identify which tool performed the sync (rsync, rclone, robocopy, etc.)

3. **Push commits promptly**
   ```bash
   git push origin feature/luminous-stars
   ```
   This creates an off-machine backup in the remote repository.

### Long-Term Preventions

1. **Add sync safeguards**
   ```bash
   # Example: rsync exclude pattern
   --exclude='.git'
   --exclude='*.cpp'
   --exclude='*.h'
   --exclude='*.hlsl'
   # Only sync build outputs, assets, etc.
   ```

2. **Use git-aware sync**
   - Consider `git fetch/pull/push` instead of file-level sync
   - Or use `git bundle` for offline transfer

3. **Pre-sync verification script**
   ```bash
   #!/bin/bash
   # Check for uncommitted changes before sync
   if ! git diff-index --quiet HEAD --; then
       echo "WARNING: Uncommitted changes detected!"
       echo "Commit or stash before syncing."
       exit 1
   fi
   ```

4. **Worktree isolation**
   - Keep worktrees on local storage, not shared mounts
   - Sync only the bare repo or use git remotes

5. **Backup verification**
   - After sync, verify with `git status`
   - Alert if unexpected changes detected

---

## Technical Details

### Git Internals Explanation

```
Repository Structure:
├── .git/
│   ├── objects/      # Immutable commit data (SAFE)
│   │   ├── ed/
│   │   │   └── b3f11...  # Commit edb3f11 blob
│   │   └── ...
│   ├── refs/         # Branch pointers (SAFE)
│   └── HEAD          # Current branch pointer (SAFE)
├── src/              # Working tree (VULNERABLE to sync)
├── shaders/          # Working tree (VULNERABLE to sync)
└── ...
```

**Why commits survive file overwrites:**
- Commits are stored as SHA-1 hashed objects in `.git/objects/`
- These are compressed and not human-readable
- Most sync tools either skip `.git/` or copy it intact
- Working tree files are regenerated from commits on demand

### Recovery Commands Reference

```bash
# Restore single file from HEAD
git restore <file>

# Restore file from specific commit
git restore --source=<commit> <file>

# Restore entire working tree
git restore .

# View what's in a commit without checking out
git show <commit>:<file>

# List files changed in a commit
git show --stat <commit>
```

---

## Appendix: Affected Code Summary

### Application.cpp Integration (Restored)

```cpp
// Initialize Luminous Star Particles (Phase 3.9)
m_luminousParticles = std::make_unique<LuminousParticleSystem>();
if (!m_luminousParticles->Initialize(16)) {
    LOG_ERROR("Failed to initialize Luminous Star Particles");
    m_luminousParticles.reset();
}

// In Render() - Update and merge star lights
if (m_enableLuminousStars && m_luminousParticles) {
    m_luminousParticles->Update(m_deltaTime, m_physicsTimeMultiplier);
    const auto& starLights = m_luminousParticles->GetStarLights();
    for (size_t i = 0; i < starLights.size() && i < 16; i++) {
        m_lights[i] = starLights[i];
    }
}
```

### Physics Shader Star Init (Restored)

```hlsl
// First 16 particles are supergiant stars
if (particleIndex < 16) {
    p.materialType = 8;  // SUPERGIANT_STAR
    p.temperature = 25000.0;
    p.density = 1.5;
    p.albedo = float3(0.85, 0.9, 1.0);
    p.flags = FLAG_IMMORTAL;
    // ... Fibonacci sphere positioning
    // ... Keplerian orbital velocity
}
```

---

**Document Author:** Claude Code
**Last Updated:** 2025-12-17
