# B2 Headless Setup - CONTINUATION REQUIRED

**Date:** 2025-12-14
**Status:** IN PROGRESS - SSH key setup remaining
**Priority:** HIGH - User has been working on this all day

---

## CRITICAL CONTEXT FOR NEXT SESSION

**READ THIS FIRST.** The user (Ben) spent hours setting up a multi-machine tmux workflow that did NOT actually offload resources. This caused significant frustration. DO NOT repeat these mistakes:

1. **SSH direction matters:** B2 SSH into B1 = B1 does all work (USELESS for offloading)
2. **Correct direction:** B1 SSH into B2 = B2 does work, B1 just displays (THIS IS WHAT WE WANT)
3. **Be brutally honest** about what actually saves resources vs what doesn't

---

## What Was Wrong (Old Setup)

- B2 used Cursor Remote-SSH to connect to B1's worktrees
- All Claude processes ran on B1 (where files live)
- B2 was just a display - NO actual resource offloading
- **This was sold as a solution but provided zero benefit**

---

## What We're Building (New Setup)

```
B1 (User sits here)          B2 (Headless compute)
├── Keyboard/Mouse           ├── WSL with full 32GB RAM
├── Blender (Windows)        ├── PlasmaDX-Clean (cloned)
├── SSH client ──────────────┤── PlasmaDX-Blender (worktree)
│   connects TO B2           ├── PlasmaDX-NanoVDB (worktree)
└── Just displays terminal   └── tmux sessions + Claude runs HERE
```

**Result:** User types on B1, Claude runs on B2's RAM, B1 free for Blender.

---

## Current State (Where We Left Off)

### Completed:
- [x] B2 WSL has cloned repo at `/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean`
- [x] B2 has worktrees: PlasmaDX-Blender, PlasmaDX-NanoVDB
- [x] B2 has tmux sessions: claude-main, claude-blender, claude-nanovdb
- [x] B2 SSH server running
- [x] B2 Windows port forwarding configured (port 22 → WSL)
- [x] B2 Windows firewall rule added

### REMAINING (Start Here):
- [ ] Add B1's SSH key to B2's authorized_keys
- [ ] Test passwordless SSH from B1 to B2
- [ ] Add SSH alias on B1 for easy connection
- [ ] Test full workflow: B1 → SSH → B2 tmux → Claude

---

## Exact Next Steps

### Step 1: Add SSH Key to B2

On B2's WSL, run:
```bash
mkdir -p ~/.ssh && echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDiVL7prp3TGi7Np57lydvMEB899gp5neuGeBjMxsG6q" >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys
```

### Step 2: Test Connection from B1

On B1's WSL:
```bash
ssh dillixrudpefy@192.168.0.74 "echo 'SUCCESS' && hostname"
```

### Step 3: Add Alias on B1

On B1's WSL:
```bash
cat >> ~/.ssh/config << 'EOF'

Host b2
    HostName 192.168.0.74
    User dillixrudpefy
EOF
```

### Step 4: Test Full Workflow

From B1:
```bash
ssh b2           # Connect to B2
tb               # Attach to blender tmux session
claude           # Start Claude - runs on B2's RAM!
# Ctrl+b d       # Detach when done
```

---

## Machine Details

| Machine | IP | WSL User | Role |
|---------|-----|----------|------|
| B1 | 192.168.0.237 | maz3ppa | Interactive (user sits here) |
| B2 | 192.168.0.74 | dillixrudpefy | Headless compute |

---

## Git Sync Workflow

Since B2 has its own clone, sync via git:

```bash
# On B1: Push your work
git add . && git commit -m "msg" && git push

# On B2: Pull updates
git pull

# Work on B2...

# On B2: Push when done
git add . && git commit -m "msg" && git push

# On B1: Pull updates
git pull
```

---

## Files to Read for Full Context

1. `docs/worktree-system/WORKTREE_SYSTEM_MAJOR_OVERHAUL_SESSION_COPY.md` - Full session transcript
2. `docs/worktree-system/README.md` - Original tmux setup docs
3. `docs/worktree-system/tmux_worktrees_setup.md` - tmux configuration details

---

**DO NOT suggest SSH from B2 into B1 for "offloading" - this does NOT work.**
