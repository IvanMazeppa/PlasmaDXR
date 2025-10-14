# Config System Reorganization Plan
**Date:** 2025-10-14
**Status:** Implementation Ready

---

## Current Issues

1. **Too many config files in root** (8+ JSON files cluttering workspace)
2. **Inconsistent naming** (config_pix_close.json, config_pix_far.json, etc.)
3. **No dedicated agent configs** (PIX debugging agent needs its own config)
4. **No build-specific configs** (Debug vs DebugPIX builds should have separate defaults)
5. **Poor organization** (hard to tell which config is for what purpose)

---

## Proposed Structure

```
PlasmaDX-Clean/
│
├── configs/                              # NEW: Centralized config directory
│   ├── README.md                         # Config system documentation
│   │
│   ├── user/                             # User/Development configs
│   │   ├── default.json                  # Default user config (your daily driver)
│   │   └── user_custom.json              # User customizations (gitignored)
│   │
│   ├── builds/                           # Build-specific configs
│   │   ├── Debug.json                    # Debug build default
│   │   └── DebugPIX.json                 # DebugPIX build default (for PIX agent)
│   │
│   ├── agents/                           # AI agent configs (autonomous debugging)
│   │   ├── pix_agent.json                # PIX debugging agent
│   │   ├── restir_debug.json             # ReSTIR-specific debugging
│   │   ├── performance_test.json         # Performance testing
│   │   └── capture_scenarios.json        # Multi-scenario capture config
│   │
│   ├── scenarios/                        # Test scenario configs
│   │   ├── close_distance.json           # Camera close to particles
│   │   ├── medium_distance.json          # Medium distance baseline
│   │   ├── far_distance.json             # Far distance test
│   │   └── stress_test.json              # High particle count
│   │
│   └── templates/                        # Config templates
│       ├── minimal.json                  # Minimal viable config
│       └── full_featured.json            # All features enabled
│
├── config.json → configs/user/default.json  # Symlink for backward compat
│
└── build/
    ├── Debug/
    │   └── config.json → ../../configs/builds/Debug.json
    └── DebugPIX/
        └── config.json → ../../configs/builds/DebugPIX.json
```

---

## Key Improvements

### 1. Centralized Organization
- **All configs in `configs/` directory** (clean root)
- **Categorized by purpose** (user, builds, agents, scenarios)
- **Easy to find and manage**

### 2. Build-Specific Defaults
```
Debug build:     Uses configs/builds/Debug.json
DebugPIX build:  Uses configs/builds/DebugPIX.json
```

Each build looks for its config in its output directory first, then falls back to default.

### 3. Agent Autonomy
**PIX Agent config:** `configs/agents/pix_agent.json`
- Agent can modify this file to enable/disable features
- Isolated from user's daily config
- Documented parameters for agent customization

**Example agent workflow:**
```bash
# PIX Agent modifies its config before launch
echo '{"features": {"enableReSTIR": true}}' > configs/agents/pix_agent.json

# Launch with agent config
./build/DebugPIX/PlasmaDX-Clean.exe --config=configs/agents/pix_agent.json
```

### 4. Scenario-Based Testing
**Scenario configs** allow quick switching between test cases:
```bash
# Test close distance issue
./build/Debug/PlasmaDX-Clean.exe --config=configs/scenarios/close_distance.json

# Test far distance (working baseline)
./build/Debug/PlasmaDX-Clean.exe --config=configs/scenarios/far_distance.json
```

---

## Migration Path

### Phase 1: Create New Structure ✅

1. Create `configs/` directory tree
2. Move existing configs into appropriate folders
3. Create new build and agent configs

### Phase 2: Update Code ✅

1. **Config.cpp line 147:** Change default from `config_dev.json` to `configs/user/default.json`
2. **Add build-specific config detection:**
   ```cpp
   // Try build-specific config first (e.g., build/Debug/config.json)
   std::string buildConfig = std::filesystem::current_path() / "config.json";
   if (std::filesystem::exists(buildConfig)) {
       return LoadFromFile(buildConfig);
   }
   ```

### Phase 3: Create Symlinks ✅

For backward compatibility:
```bash
# Root symlink
ln -s configs/user/default.json config.json

# Build directory symlinks
ln -s ../../configs/builds/Debug.json build/Debug/config.json
ln -s ../../configs/builds/DebugPIX.json build/DebugPIX/config.json
```

### Phase 4: Update Documentation ✅

1. **README.md:** Add config system section
2. **CONFIG_REFERENCE.md:** Update file paths
3. **Session start docs:** Document new structure for Claude

---

## Config File Mapping

### Old → New

| Old Location | New Location | Purpose |
|--------------|--------------|---------|
| `config.json` | `configs/user/default.json` | Default user config |
| `config_dev.json` | `configs/user/default.json` | (merged with default) |
| `config_user.json` | `configs/user/user_custom.json` | User customizations |
| `config_pix_analysis.json` | `configs/agents/pix_agent.json` | PIX debugging agent |
| `config_pix_close.json` | `configs/scenarios/close_distance.json` | Test scenario |
| `config_pix_far.json` | `configs/scenarios/far_distance.json` | Test scenario |
| `config_pix_inside.json` | `configs/scenarios/stress_test.json` | Stress test |
| `config_pix_veryclose.json` | (deleted - duplicate of close) | N/A |
| *(new)* | `configs/builds/Debug.json` | Debug build default |
| *(new)* | `configs/builds/DebugPIX.json` | DebugPIX build default |

---

## Build-Specific Configs

### configs/builds/Debug.json
**Purpose:** Your daily driver for development

```json
{
  "profile": "dev",
  "rendering": {
    "particleCount": 10000,
    "rendererType": "gaussian"
  },
  "features": {
    "enableReSTIR": false,           // Disabled by default (not working yet)
    "enableShadowRays": true,
    "enablePhaseFunction": true
  },
  "debug": {
    "enableDebugLayer": true,        // D3D12 debug layer ON
    "enablePIX": false,               // PIX instrumentation OFF (performance)
    "showFPS": true
  }
}
```

### configs/builds/DebugPIX.json
**Purpose:** PIX agent debugging (instrumented build)

```json
{
  "profile": "pix_analysis",
  "rendering": {
    "particleCount": 10000,
    "rendererType": "gaussian"
  },
  "features": {
    "enableReSTIR": true,            // Agent can override this
    "enableShadowRays": true,
    "enablePhaseFunction": true
  },
  "debug": {
    "enableDebugLayer": false,       // Disabled for PIX (conflicts)
    "enablePIX": true,                // PIX instrumentation ON
    "pixAutoCapture": false,          // Agent controls capture timing
    "showFPS": true
  },
  "pix_analysis": {
    "capturePrefix": "agent_",
    "enableReservoirLogging": true,
    "enablePerformanceCounters": true
  }
}
```

---

## Agent Config System

### configs/agents/pix_agent.json
**Purpose:** PIX debugging agent autonomous configuration

```json
{
  "profile": "pix_agent",
  "rendering": {
    "particleCount": 10000,
    "rendererType": "gaussian"
  },
  "features": {
    "enableReSTIR": true,             // Agent can toggle this
    "restirCandidates": 16,
    "enableShadowRays": true,
    "enablePhaseFunction": true,
    "rtLightingStrength": 2.0
  },
  "camera": {
    "startDistance": 200.0,           // Agent can position camera
    "startHeight": 100.0,
    "startPitch": -0.3
  },
  "debug": {
    "enablePIX": true,
    "pixAutoCapture": false,          // Agent controls capture
    "pixCaptureFrame": 120,
    "enableDebugLayer": false
  },
  "pix_analysis": {
    "capturePrefix": "autonomous_",
    "enableReservoirLogging": true,
    "enablePerformanceCounters": true,
    "trackResourceUsage": true
  },
  "_agent_notes": {
    "purpose": "PIX debugging agent configuration",
    "can_modify": [
      "features.enableReSTIR",
      "camera.startDistance",
      "camera.startHeight",
      "debug.pixCaptureFrame"
    ],
    "do_not_modify": [
      "rendering.particleCount",
      "debug.enablePIX"
    ]
  }
}
```

### Agent Workflow Example

**Agent task:** "Analyze ReSTIR at 3 distances"

**Agent actions:**
1. Modify `configs/agents/pix_agent.json`:
   ```json
   {"camera": {"startDistance": 100.0}}
   ```
2. Launch: `./build/DebugPIX/PlasmaDX-Clean.exe --config=configs/agents/pix_agent.json --dump-buffers 120`
3. Repeat for 200.0 and 500.0 distances
4. Analyze dumps automatically

---

## Config Loading Priority

**New priority order:**

1. **Command-line:** `--config=<path>` (highest priority)
2. **Environment:** `PLASMADX_CONFIG=<path>`
3. **Build directory:** `./config.json` (symlink to build-specific)
4. **Default user:** `configs/user/default.json`
5. **Hardcoded:** Fallback defaults in code

**Example:**
```bash
# Uses configs/builds/Debug.json (via symlink)
./build/Debug/PlasmaDX-Clean.exe

# Uses agent config (command-line override)
./build/Debug/PlasmaDX-Clean.exe --config=configs/agents/pix_agent.json

# Uses scenario config
./build/Debug/PlasmaDX-Clean.exe --config=configs/scenarios/stress_test.json
```

---

## Benefits for AI Agents

### PIX Debugging Agent

**Before:**
- Had to guess config location
- Unclear which config to modify
- Risk of breaking user's config

**After:**
- Clear agent config: `configs/agents/pix_agent.json`
- Documented modifiable parameters
- Isolated from user configs

### Autonomous Testing

**Before:**
- Hard to script multiple test scenarios
- Configs scattered everywhere
- Unclear which config is for what

**After:**
- **Scenario configs** make testing trivial:
  ```bash
  for scenario in configs/scenarios/*.json; do
      ./build/Debug/PlasmaDX-Clean.exe --config=$scenario --dump-buffers 120
  done
  ```

---

## Implementation Checklist

### File Operations
- [x] Create `configs/` directory structure
- [ ] Move existing configs to new locations
- [ ] Create new build-specific configs
- [ ] Create agent configs
- [ ] Create scenario configs
- [ ] Create config templates
- [ ] Create symlinks for backward compatibility

### Code Changes
- [ ] Update `Config.cpp` default path (line 147)
- [ ] Add build directory config detection
- [ ] Update config path resolution logic
- [ ] Test all config loading paths

### Documentation
- [ ] Create `configs/README.md`
- [ ] Update root `README.md`
- [ ] Update `CONFIG_REFERENCE.md`
- [ ] Create `SESSION_START_CHECKLIST.md` for Claude
- [ ] Document agent config system

### Testing
- [ ] Test backward compatibility (old paths still work)
- [ ] Test command-line override
- [ ] Test build-specific configs
- [ ] Test agent config workflow
- [ ] Test scenario switching

---

## Rollout Plan

### Step 1: Non-Breaking Setup (This Session)
1. Create `configs/` directory tree
2. Copy existing configs to new locations (leave old ones)
3. Create new configs (builds, agents, scenarios)
4. Create symlinks
5. Update documentation

### Step 2: Test Period (Next Session)
1. Test all config paths work
2. Verify PIX agent can use its config
3. Test scenario switching
4. Validate backward compatibility

### Step 3: Cleanup (After Validation)
1. Delete old config files from root
2. Update `.gitignore` for `configs/user/user_custom.json`
3. Final documentation review

---

## Success Criteria

✅ **Organization:** All configs in `configs/` directory, categorized by purpose
✅ **Build Configs:** Each build has its own default config
✅ **Agent Autonomy:** PIX agent has dedicated config it can safely modify
✅ **Scenario Testing:** Easy to switch between test scenarios
✅ **Backward Compat:** Old paths still work (via symlinks)
✅ **Documentation:** Clear docs for both humans and AI agents

---

**Status:** Ready for implementation
**Next Action:** Create directory structure and migrate configs