/
# SSH Setup Troubleshooting Prompt


**Purpose:** Copy this prompt to Cursor's AI or Claude to get help fixing SSH/remote development setup.

---

## Prompt

```
I need help fixing my SSH remote development setup between my Windows client PC and a Linux server (WSL2 on another Windows machine). I'm a novice with SSH/networking but experienced with programming.

## My Setup

**Client Machine:** Windows 11, using Cursor IDE (VS Code fork) with Remote SSH extension
**Server Machine:** Windows with WSL2 (Ubuntu), running on 192.168.0.237
**Goal:** Connect to different git worktrees on the server for development

## Current SSH Config (on client: ~/.ssh/config)

```
# PlasmaDX Development Server (Main PC with WSL2)
Host plasmadx-server
    HostName 192.168.0.237
    User maz3ppa
    Port 22

# Alias for PINN-v4 work
Host pinn-v4
    HostName 192.168.0.237
    User maz3ppa
    Port 22

# Alias for MultiAgent work
Host multi-agent
    HostName 192.168.0.237
    User maz3ppa
    Port 22

# Alias for Blender work
Host blender
    HostName 192.168.0.237
    User maz3ppa
    Port 22

# Alias for NanoVDB work
Host nanovdb
    HostName 192.168.0.237
    User maz3ppa
    Port 22
```

## Git Worktrees on Server

All located under `/mnt/d/Users/dilli/AndroidStudioProjects/`:
- PlasmaDX-Clean (main)
- PlasmaDX-Blender (feature/blender-integration)
- PlasmaDX-NanoVDB (feature/nanovdb-animated-assets)
- PlasmaDX-MultiAgent (feature/multi-agent-v3)
- PlasmaDX-PINN-v4 (feature/pinn-v4-siren-optimizations)

## Problem 1: Platform Detection Error

When connecting via Cursor Remote SSH, I get this error:
```
bash: line 1: powershell: command not found
```

The log shows:
```
Using configured platform windows for remote host nanovdb
```

But my remote is Linux (WSL2), not Windows! Cursor is trying to run PowerShell on Linux.

**Attempted fix:** I was told to add this to Cursor settings.json:
```json
{
    "remote.SSH.remotePlatform": {
        "nanovdb": "linux",
        "plasmadx-server": "linux",
        "pinn-v4": "linux",
        "multi-agent": "linux",
        "blender": "linux"
    }
}
```

Please help me verify this is correct and where exactly to add it.

## Problem 2: Constant Password Prompts

Every SSH connection asks for my password repeatedly. I want to set up SSH key authentication so I don't have to type passwords.

I need step-by-step instructions for Windows client → WSL2 server key setup.

## Problem 3: Making This More Robust

I'm prone to accidentally breaking things. I'd like:
1. A way to verify SSH is working before trying Cursor
2. A checklist to validate the setup
3. Any improvements to make the config more resilient

## What I Need

1. Fix the platform detection issue so Cursor connects properly
2. Set up SSH keys to eliminate password prompts
3. Verify each host alias connects to the right worktree (ideally auto-cd to the worktree directory)
4. A simple test I can run to verify everything works

## My Skill Level

I'm a novice with SSH/networking. Please give me explicit commands to copy-paste, explain what each does, and warn me about common mistakes.
```

---

## Additional Context (if needed)

### Recent Issues I've Had

1. **Git LFS corruption** - LFS was enabled system-wide and corrupted worktree checkouts. Fixed by:
   - Deleting `/etc/gitconfig` on server
   - Disabling `/usr/bin/git-lfs`
   - Recommitting files with real content

2. **Worktree confusion** - Multiple worktrees pointing to same repo, needed to understand they share `.git`

3. **Tmux sessions** - Was using tmux for persistent sessions but got confused about which session was which

### What Working Setup Should Look Like

1. `ssh nanovdb` from client terminal → lands in `/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-NanoVDB`
2. Cursor "Remote SSH: Connect to Host" → nanovdb → opens that worktree folder
3. No password prompts (uses SSH key)
4. Claude Code can run in the remote terminal

### Files on Server That May Help

- `/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/docs/NanoVDB/CLAUDE_SESSION_PROMPT.md` - Context for NanoVDB development
- `/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/docs/BLENDER_SESSION_CONTEXT.md` - Context for Blender development

### Validation Commands

Once setup is working, these should all succeed:
```bash
# From client terminal
ssh nanovdb "pwd && git status"
ssh blender "pwd && git status"
ssh pinn-v4 "pwd && git status"
```

---

## Quick Reference: Expected Paths

| Host Alias | Expected Working Directory |
|------------|---------------------------|
| plasmadx-server | /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean |
| nanovdb | /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-NanoVDB |
| blender | /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Blender |
| multi-agent | /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-MultiAgent |
| pinn-v4 | /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-PINN-v4 |

---

*This document created to help get AI assistance on the client machine when server-side Claude Code can't help with client-side SSH issues.*
