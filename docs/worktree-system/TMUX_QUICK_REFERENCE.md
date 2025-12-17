# tmux Quick Reference - Worktree Sessions

**Four sessions for parallel development.**

---

## Session Switching

| Key | Session | Worktree | Purpose |
|-----|---------|----------|---------|
| **F9** | claude-blender | PlasmaDX-Blender | Blender VDB export |
| **F10** | claude-nanovdb | PlasmaDX-NanoVDB | NanoVDB integration |
| **F11** | claude-luminous | PlasmaDX-LuminousStars | Luminous star particles |
| **F12** | claude-agentforge | PlasmaDX-MultiAgent | Multi-agent optimization |

Or use `Ctrl+b 1-4` for each session.

---

## Terminal Aliases

| Alias | Action |
|-------|--------|
| `tstart` | Start both sessions |
| `tb` | Attach to Blender |
| `tn` | Attach to NanoVDB |
| `tls` | List sessions |

---

## Inside tmux

| Keys | Action |
|------|--------|
| **F9** | Switch to Blender |
| **F10** | Switch to NanoVDB |
| **F11** | Switch to LuminousStars |
| **F12** | Switch to AgentForge |
| **q** | Snap back to prompt (after scrolling) |
| `Ctrl+b d` | Detach (session keeps running) |
| `Ctrl+b s` | Session picker |

---

## Scrolling

When you scroll with mousewheel, you enter "copy mode" (viewing history).

**To get back to the prompt:**
- Press **q**
- Or press **Escape**

---

## Session Prompts

Copy these into new Claude sessions:

| Session | Prompt File |
|---------|-------------|
| Blender | `docs/NanoVDB/BLENDER_SESSION_PROMPT.md` |
| NanoVDB | `docs/NanoVDB/BLENDER_NANOVDB_SESSION_PROMPT.md` |

---

## Quick Start

```bash
tstart    # Start sessions (blender + nanovdb)
tb        # Attach to Blender
# Press F9-F12 to switch between sessions
# Press q if prompt scrolled away
```

## Starting Additional Sessions

```bash
# LuminousStars (F11)
tmux new-session -d -s claude-luminous -c /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-LuminousStars

# AgentForge (F12) - uses MultiAgent worktree
tmux new-session -d -s claude-agentforge -c /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-MultiAgent
```

---

*Last Updated: 2025-12-16*
