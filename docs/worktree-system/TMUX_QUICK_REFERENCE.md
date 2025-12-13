# tmux Quick Reference - Blender + NanoVDB

**Two sessions for volumetric asset development.**

---

## Session Switching

| Key | Session | Worktree |
|-----|---------|----------|
| **F9** | claude-blender | PlasmaDX-Blender |
| **F10** | claude-nanovdb | PlasmaDX-NanoVDB |

Or use `Ctrl+b 1` (blender) and `Ctrl+b 2` (nanovdb).

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
tstart    # Start sessions
tb        # Attach to Blender
# Press F10 to switch to NanoVDB
# Press F9 to switch back to Blender
# Press q if prompt scrolled away
```

---

*Last Updated: 2025-12-13*
