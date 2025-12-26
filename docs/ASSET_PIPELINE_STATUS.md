# Self-Improving NanoVDB Asset Pipeline - Status

**Date:** 2025-12-25
**Context:** Pipeline for generating NanoVDB assets from Blender using AI-driven iteration

---

## Pipeline Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ script-generator│────▶│ blender-executor │────▶│ asset-evaluator │
│   (create/mod)  │     │  (run Blender)   │     │ (LPIPS/CLIP)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        ▲                                                 │
        │                                                 │
        └─────────────── feedback loop ◀──────────────────┘
```

---

## MCP Servers - Completed

### 1. blender-executor ✅ CONNECTED
**Location:** `agents/blender-executor/`

**Tools:**
| Tool | Purpose |
|------|---------|
| `execute_blender_script` | Run scripts via Blender CLI, capture output |
| `parse_blender_errors` | Parse errors with suggested fixes (Blender 5.0 API) |
| `list_run_outputs` | Find VDB/renders/logs from execution |
| `get_latest_run` | Get most recent run info |
| `list_available_scripts` | List scripts in project |

**Config:**
- Blender: `/home/maz3ppa/apps/blender-5.0.1-linux-x64/blender`
- Logs: `build/blender_cli_logs/`

---

### 2. asset-evaluator ✅ CONNECTED
**Location:** `agents/asset-evaluator/`

**Tools:**
| Tool | Purpose |
|------|---------|
| `compare_lpips` | Perceptual similarity (0-1, lower=better, ~92% human correlation) |
| `compare_clip` | Semantic similarity (text or image query, higher=better) |
| `evaluate_render` | Combined pass/fail with thresholds |
| `find_reference_images` | Search by keyword in reference library |
| `list_recent_renders` | List renders from build output |

**Dependencies:** torch, lpips, clip (lazy-loaded to avoid MCP timeout)

**Reference Images:** `assets/reference_images/{nebula,explosion,star,gas_cloud,liquid}/`

---

### 3. script-generator ✅ CONNECTED
**Location:** `agents/script-generator/`

**Tools:**
| Tool | Purpose |
|------|---------|
| `list_templates` | List existing scripts as templates |
| `get_template` | Get full content of template |
| `analyze_script` | Extract type, features, parameters |
| `generate_script` | Create new script from effect type + description |
| `modify_script` | Adjust parameters based on feedback |

**Templates:** `assets/blender_scripts/GPT-5.2/`
**Output:** `assets/blender_scripts/generated/`

**Built-in effect types:** pyro, liquid, explosion, nebula, smoke, fire, gas

---

### 4. iteration-controller ✅ CONNECTED
**Location:** `agents/iteration-controller/`

**Tools:**
| Tool | Purpose |
|------|---------|
| `create_asset` | Full pipeline: generate → execute → evaluate → improve loop |
| `run_iteration` | Execute one generate→evaluate cycle |
| `get_history` | View iteration history for an asset |
| `list_sessions` | List all asset creation sessions |
| `compare_iterations` | Compare quality across attempts |

**Session Storage:** `build/iteration_history/`

---

## Configuration Files

| File | Purpose |
|------|---------|
| `~/.claude.json` | Global MCP server registration |
| `agents/*/run_server.sh` | Server startup scripts (Unix LF) |
| `agents/*/.mcp.json` | Local MCP config (reference) |

---

## Key Technical Details

### Blender Integration
- **Version:** Blender 5.0.1 Linux (WSL)
- **Mode:** Headless CLI (`--background`)
- **Simulation:** Mantaflow (GAS/LIQUID domains)
- **Export:** OpenVDB format
- **Known API change:** `openvdb_cache_compress_type` removed (use ZIP/NONE)

### Evaluation Metrics
- **LPIPS threshold:** 0.35 (fair) - lower = more similar
- **CLIP threshold:** 0.60 (fair+) - higher = better match
- **Pass logic:** Either metric passes, or both if `require_both=True`

### Path Updates (Windows → WSL)
All MCP servers updated from:
```
/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/
```
To:
```
/home/maz3ppa/projects/PlasmaDXR/
```

---

## Next Steps

1. ~~**Build iteration-controller**~~ ✅ COMPLETE
2. **Test full pipeline** - Generate script → Execute → Evaluate → Improve
3. **Add reference images** - Populate `assets/reference_images/` with targets
4. **Create example workflow** - Document end-to-end asset creation

---

## Pipeline Complete!

All 4 MCP servers are now connected:
- `blender-executor` ✅
- `asset-evaluator` ✅
- `script-generator` ✅
- `iteration-controller` ✅

Run `/mcp` to reconnect and verify all servers.

---

## Quick Reference

**Connect MCP servers:**
```
/mcp
```

**Test blender-executor:**
```python
mcp__blender-executor__list_available_scripts()
```

**Test asset-evaluator:**
```python
mcp__asset-evaluator__find_reference_images("nebula")
```

**Test script-generator:**
```python
mcp__script-generator__list_templates("pyro")
```
