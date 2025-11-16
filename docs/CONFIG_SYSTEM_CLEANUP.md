# Config System Cleanup - Minor Priority Task

**Priority:** MINOR (can wait unless deemed critical)
**Impact:** MEDIUM (enables agents to select configs for different tasks)
**Created:** 2025-11-15
**Status:** PENDING

---

## Problem Statement

The configuration system has become **scattered and broken** after recent updates:

1. **Scattered Config Files:**
   - Root level configs (should be in `configs/`):
     - `config_dev.json`
     - `config_dev.json.original`
     - `config_pix_analysis.json`
     - `config_pix_close.json`
     - `config_pix_far.json`
     - `config_pix_inside.json`
     - `config_pix_veryclose.json`

   - Proper location: `configs/` directory (mostly empty or stale)

2. **Config System "Useless" After Updates:**
   - Recent lighting simplification changes (global ambient, physical emission, dynamic emission disabled)
   - Configs likely have outdated defaults
   - No clear config selection for agent workflows

3. **No Agent-Friendly Configs:**
   - Agents can't easily select "probe grid diagnostic" vs "performance test" configs
   - Would be invaluable for automated testing and diagnostics

---

## Impact Assessment

### Current Impact: MEDIUM
- **Workaround exists:** Agents can use runtime defaults or PIX configs
- **Not blocking:** Development can continue without this fix
- **User assessment:** "Can wait unless critical"

### Future Impact: HIGH (if left unfixed)
- Agents unable to autonomously test different scenarios
- Manual config management required for each test
- Inconsistent diagnostic environments

---

## Proposed Solution

### Phase 1: Consolidation (30 minutes)
1. **Move all configs to `configs/` directory:**
   ```bash
   mv config_*.json configs/
   ```

2. **Create organized subdirectories:**
   ```
   configs/
   ├── agents/          # Agent-specific configs
   │   ├── diagnostic_default.json
   │   ├── probe_grid_focus.json
   │   ├── performance_test.json
   │   └── visual_quality.json
   ├── scenarios/       # Test scenarios (already exists)
   ├── user/            # User development configs
   │   ├── default.json
   │   └── dev.json
   └── pix/             # PIX capture configs
       ├── close.json
       ├── far.json
       ├── inside.json
       └── veryclose.json
   ```

3. **Update `.gitignore`:**
   ```gitignore
   # Root-level config files (force use of configs/ directory)
   /config*.json
   !/configs/
   ```

### Phase 2: Config System Update (1 hour)
1. **Update config loading hierarchy** in `src/config/Config.cpp`:
   ```cpp
   // Priority order:
   1. Command-line: --config=<path>
   2. Environment: PLASMADX_CONFIG=<path>
   3. Build directory: ./config.json (deprecated, log warning)
   4. Configs directory: configs/user/default.json
   5. Hardcoded defaults
   ```

2. **Add config validation:**
   - Warn about outdated config values
   - Auto-update deprecated fields
   - Log config source on startup

### Phase 3: Agent Integration (30 minutes)
1. **Create agent-friendly configs:**
   - `diagnostic_default.json` - Safe defaults for diagnostics
   - `probe_grid_focus.json` - Probe grid isolated (all other lighting disabled)
   - `performance_test.json` - 100K particles, DLSS enabled, FPS target
   - `visual_quality.json` - Quality-focused settings for LPIPS comparisons

2. **Document config selection API:**
   ```bash
   # For agents launching PlasmaDX
   ./build/bin/Debug/PlasmaDX-Clean.exe --config=configs/agents/probe_grid_focus.json
   ```

---

## Config Template: Probe Grid Focus

**Purpose:** Isolate probe grid for diagnostics (all competing lighting disabled)

**File:** `configs/agents/probe_grid_focus.json`

```json
{
  "schema_version": "2.0",
  "description": "Probe grid focus - all competing lighting disabled for diagnostics",

  "particles": {
    "count": 10000,
    "radius": 11.0,
    "physics_enabled": true
  },

  "rendering": {
    "renderer_type": "Gaussian",
    "active_lighting_system": "ProbeGrid"
  },

  "lighting": {
    "probe_grid": {
      "enabled": true,
      "grid_size": 48,
      "rays_per_probe": 16,
      "update_interval": 4,
      "intensity": 800.0
    },
    "multi_light": {
      "enabled": false,
      "light_count": 0
    },
    "rtxdi": {
      "enabled": false
    },
    "global_ambient": 0.0,
    "physical_emission": {
      "enabled": false,
      "strength": 0.0
    },
    "dynamic_emission": {
      "enabled": false,
      "strength": 0.0
    }
  },

  "quality": {
    "preset": "Medium",
    "target_fps": 120.0
  },

  "camera": {
    "distance": 800.0,
    "height": 1200.0,
    "pitch": 0.700
  }
}
```

---

## Implementation Checklist

### When Ready to Fix:

**Phase 1: Consolidation** (30 min)
- [ ] Create `configs/agents/` subdirectory
- [ ] Create `configs/pix/` subdirectory
- [ ] Move root-level `config_*.json` to appropriate subdirs
- [ ] Update `.gitignore` to prevent root-level configs
- [ ] Test config loading still works

**Phase 2: Config System Update** (1 hour)
- [ ] Update `Config.cpp` loading hierarchy
- [ ] Add deprecation warnings for root-level configs
- [ ] Add config validation and auto-update
- [ ] Log config source on startup
- [ ] Test with all loading methods (CLI, env, default)

**Phase 3: Agent Integration** (30 min)
- [ ] Create `diagnostic_default.json`
- [ ] Create `probe_grid_focus.json`
- [ ] Create `performance_test.json`
- [ ] Create `visual_quality.json`
- [ ] Update agent docs with config selection examples
- [ ] Test agents can launch with specific configs

**Validation:**
- [ ] All agents can launch PlasmaDX with appropriate configs
- [ ] No root-level configs remain (all in `configs/`)
- [ ] Config loading logs source clearly
- [ ] Deprecated configs auto-update or warn

---

## Decision: Fix Now or Later?

### Reasons to Fix NOW:
1. **Enables autonomous agent testing** - Critical for buffer dump automation
2. **Cleans up workspace** - Scattered configs are messy
3. **Quick fix** - Total time: ~2 hours
4. **Foundation for Phase 0.2** - Buffer dump agent needs config selection

### Reasons to WAIT:
1. **User says "can wait"** - Not currently blocking
2. **Other critical issues** - Probe grid dim, buffer dump broken
3. **Can use workarounds** - Runtime defaults, PIX configs sufficient

### **Recommendation:** FIX NOW
**Rationale:**
- Required for buffer dump automation (next critical task)
- Fast fix (~2 hours)
- Unblocks agent autonomous testing
- Clean workspace improves productivity

---

## Related Issues

- **Buffer Dump Automation:** Needs config selection to test different scenarios
- **Probe Grid Diagnostics:** Would benefit from isolated config
- **Agent Collaboration Testing:** Needs reproducible config profiles

---

**Next Steps:**
1. User decides: Fix now or defer?
2. If now: Create subtask in current session
3. If defer: Add to Phase 0.2 roadmap as blocker for buffer dump automation

---

**Last Updated:** 2025-11-15
**Priority:** MINOR (unless blocking buffer dump automation - then IMMEDIATE)
