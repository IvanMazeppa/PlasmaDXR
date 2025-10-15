# Configuration System
**Last Updated:** 2025-10-14
**For:** PlasmaDX-Clean v1.0

---

## Quick Start

### For Users (Daily Development)

```bash
# Uses default user config automatically
./build/Debug/PlasmaDX-Clean.exe

# Use custom config
./build/Debug/PlasmaDX-Clean.exe --config=configs/user/my_custom.json
```

### For AI Agents (Autonomous Debugging)

```bash
# PIX debugging agent
./build/DebugPIX/PlasmaDX-Clean.exe --config=configs/agents/pix_agent.json

# Agent can modify its config before launch:
# 1. Edit configs/agents/pix_agent.json
# 2. Change camera.startDistance, features.enableReSTIR, etc.
# 3. Launch with modified config
```

### For Testing Scenarios

```bash
# Test close distance (where ReSTIR bugs appear)
./build/Debug/PlasmaDX-Clean.exe --config=configs/scenarios/close_distance.json

# Test far distance (working baseline)
./build/Debug/PlasmaDX-Clean.exe --config=configs/scenarios/far_distance.json

# Batch test all scenarios
for scenario in configs/scenarios/*.json; do
    ./build/Debug/PlasmaDX-Clean.exe --config=$scenario --dump-buffers 120
done
```

---

## Directory Structure

```
configs/
├── README.md                         # This file
│
├── user/                             # User/Development configurations
│   ├── default.json                  # Default config (used if no --config specified)
│   └── user_custom.json              # User customizations (gitignored)
│
├── builds/                           # Build-specific defaults
│   ├── Debug.json                    # Debug build (daily development)
│   └── DebugPIX.json                 # DebugPIX build (PIX agent)
│
├── agents/                           # AI agent configurations
│   ├── pix_agent.json                # PIX debugging agent
│   ├── restir_debug.json             # ReSTIR-specific debugging
│   └── performance_test.json         # Performance testing
│
├── scenarios/                        # Test scenario configurations
│   ├── close_distance.json           # Camera at 100-200 units (bugs appear here)
│   ├── medium_distance.json          # Camera at 400 units (working baseline)
│   ├── far_distance.json             # Camera at 800+ units (test distance falloff)
│   └── stress_test.json              # High particle count test
│
└── templates/                        # Configuration templates
    ├── minimal.json                  # Minimal viable config
    └── full_featured.json            # All features enabled (kitchen sink)
```

---

## Config Loading Priority

The application loads configs in this order (first found wins):

1. **Command-line:** `--config=<path>` (highest priority)
2. **Environment variable:** `PLASMADX_CONFIG=<path>`
3. **Build directory:** `./config.json` (if exists in exe directory)
4. **User default:** `configs/user/default.json`
5. **Hardcoded defaults:** Built-in fallback

**Example:**
```bash
# Priority 1: Uses scenario config (command-line override)
./build/Debug/PlasmaDX-Clean.exe --config=configs/scenarios/stress_test.json

# Priority 2: Uses environment variable
export PLASMADX_CONFIG=configs/agents/pix_agent.json
./build/Debug/PlasmaDX-Clean.exe

# Priority 3: Uses build directory config (symlink)
./build/Debug/PlasmaDX-Clean.exe  # → build/Debug/config.json → configs/builds/Debug.json

# Priority 4: Uses user default
./PlasmaDX-Clean.exe  # → configs/user/default.json
```

---

## Build-Specific Configs

### Debug Build (builds/Debug.json)

**Purpose:** Your daily driver for development

**Key settings:**
- `enableReSTIR: false` - ReSTIR disabled (not working yet)
- `enableDebugLayer: true` - D3D12 debug validation ON
- `enablePIX: false` - PIX instrumentation OFF (better performance)
- `startDistance: 800.0` - Far from particles (safe starting position)

**Use when:**
- Daily feature development
- Testing non-ReSTIR features
- Performance profiling (no PIX overhead)

---

### DebugPIX Build (builds/DebugPIX.json)

**Purpose:** PIX debugging agent (instrumented build)

**Key settings:**
- `enableReSTIR: true` - ReSTIR enabled for debugging
- `enableDebugLayer: false` - D3D12 debug OFF (conflicts with PIX)
- `enablePIX: true` - PIX instrumentation ON
- `startDistance: 200.0` - Medium distance where ReSTIR bugs appear

**Use when:**
- PIX agent autonomous debugging
- Capturing GPU buffer dumps
- Analyzing ReSTIR reservoir data
- Performance counters analysis

**⚠️ Warning:** Do NOT run DebugPIX build for daily work - PIX overhead is significant!

---

## Agent Configuration System

### configs/agents/pix_agent.json

**Purpose:** PIX debugging agent autonomous configuration

**Agent can modify:**
- `features.enableReSTIR` - Toggle ReSTIR on/off
- `camera.startDistance` - Position camera for testing
- `camera.startHeight` - Adjust vertical position
- `debug.pixCaptureFrame` - Control when to capture

**Agent should NOT modify:**
- `rendering.particleCount` - Keep consistent for comparisons
- `debug.enablePIX` - Must always be true for PIX agent
- `debug.enableDebugLayer` - Must be false (conflicts with PIX)

**Example agent workflow:**

```python
# Agent modifies config before launch
import json

config = json.load(open("configs/agents/pix_agent.json"))
config["camera"]["startDistance"] = 100.0  # Test close distance
config["features"]["enableReSTIR"] = True
json.dump(config, open("configs/agents/pix_agent.json", "w"), indent=2)

# Launch with modified config
os.system("./build/DebugPIX/PlasmaDX-Clean.exe --config=configs/agents/pix_agent.json --dump-buffers 120")
```

---

## Scenario Configurations

### When to Use Scenarios

**Scenarios** are pre-configured test cases for specific debugging needs:

| Scenario | Purpose | Camera Position | Expected Behavior |
|----------|---------|-----------------|-------------------|
| `close_distance.json` | Test ReSTIR bugs | 100-200 units | Dots visible, colors muted |
| `medium_distance.json` | Working baseline | 400 units | Smooth rendering, no artifacts |
| `far_distance.json` | Test attenuation | 800+ units | Test distance falloff |
| `stress_test.json` | Performance test | Variable | High particle count (50K+) |

### Example: Testing ReSTIR at Multiple Distances

```bash
# Batch test script
for scenario in close medium far; do
    ./build/Debug/PlasmaDX-Clean.exe \
        --config=configs/scenarios/${scenario}_distance.json \
        --dump-buffers 120 \
        --dump-dir "analysis/${scenario}"
done

# Then analyze all dumps
python PIX/scripts/analysis/analyze_5_captures.py analysis/*
```

---

## Configuration Parameters

### Rendering Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `particleCount` | int | 10000 | Number of particles (10K-50K) |
| `rendererType` | string | "gaussian" | "gaussian" or "billboard" |
| `width` | int | 1920 | Window width |
| `height` | int | 1080 | Window height |

### Features Section (ReSTIR)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enableReSTIR` | bool | false | Enable temporal resampling |
| `restirCandidates` | int | 16 | Initial candidate samples (8-32) |
| `restirTemporalReuse` | bool | true | Use previous frame's samples |
| `restirTemporalWeight` | float | 0.9 | Temporal reuse strength (0.0-1.0) |

### Physics Section (NEW)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `innerRadius` | float | 10.0 | Inner disk radius (Schwarzschild radii) |
| `outerRadius` | float | 300.0 | Outer disk radius |
| `diskThickness` | float | 50.0 | Disk thickness (vertical extent) |
| `timeStep` | float | 0.008333 | Physics timestep (120Hz) |
| `physicsEnabled` | bool | true | Enable GPU physics simulation |
| `blackHoleMass` | float | 4300000.0 | Black hole mass in solar masses (M☉) |
| `alphaViscosity` | float | 0.1 | Shakura-Sunyaev α (accretion parameter, 0.0-1.0) |

**New Parameters:**
- **blackHoleMass**: Controls Keplerian orbital velocities. Higher mass = faster orbits. Realistic values: 10 (stellar), 4.3e6 (Sgr A*), 1e9 (quasar)
- **alphaViscosity**: Controls inward spiral (accretion). 0.0 = no accretion, 0.1 = realistic, 1.0 = fast accretion

### Camera Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `startDistance` | float | 800.0 | Camera distance from center |
| `startHeight` | float | 400.0 | Camera height above disk plane |
| `startPitch` | float | -0.3 | Camera pitch angle (radians) |
| `particleSize` | float | 50.0 | Particle rendering size |

### Debug Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enableDebugLayer` | bool | true | D3D12 debug validation |
| `enablePIX` | bool | false | PIX instrumentation |
| `pixAutoCapture` | bool | false | Auto-capture at frame N |
| `pixCaptureFrame` | int | 120 | Frame number for auto-capture |
| `showFPS` | bool | true | Display FPS counter |

### PIX Analysis Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `capturePrefix` | string | "capture_" | Prefix for PIX capture files |
| `enableReservoirLogging` | bool | false | Log reservoir statistics |
| `enablePerformanceCounters` | bool | false | Track GPU performance |
| `trackResourceUsage` | bool | false | Monitor VRAM usage |

---

## Creating Custom Configs

### Method 1: Copy Template

```bash
# Start from minimal template
cp configs/templates/minimal.json configs/user/my_config.json

# Edit as needed
nano configs/user/my_config.json

# Test
./build/Debug/PlasmaDX-Clean.exe --config=configs/user/my_config.json
```

### Method 2: Modify Existing

```bash
# Copy working config
cp configs/scenarios/medium_distance.json configs/user/my_test.json

# Modify specific settings
# (e.g., change startDistance, enable/disable features)

# Test
./build/Debug/PlasmaDX-Clean.exe --config=configs/user/my_test.json
```

### Method 3: Generate from Code

The application can generate default configs programmatically:

```cpp
// In C++ code
Config::ConfigManager::GenerateDefaultConfigs();
// Creates: config_dev.json, config_user.json, config_pix_analysis.json
```

---

## Common Workflows

### For Daily Development

```bash
# Just run - uses Debug build default
./build/Debug/PlasmaDX-Clean.exe

# Override specific setting
./build/Debug/PlasmaDX-Clean.exe --config=configs/user/default.json
```

### For PIX Agent Debugging

```bash
# 1. Agent modifies its config
#    (edit configs/agents/pix_agent.json)

# 2. Launch with agent config
./build/DebugPIX/PlasmaDX-Clean.exe --config=configs/agents/pix_agent.json

# 3. Capture buffers at frame 120
./build/DebugPIX/PlasmaDX-Clean.exe \
    --config=configs/agents/pix_agent.json \
    --dump-buffers 120

# 4. Analyze dumps
python PIX/scripts/analysis/analyze_restir_manual.py \
    --current PIX/buffer_dumps/g_currentReservoirs.bin \
    --prev PIX/buffer_dumps/g_prevReservoirs.bin
```

### For Multi-Scenario Testing

```bash
# Test all scenarios in batch
for scenario in configs/scenarios/*.json; do
    echo "Testing: $scenario"
    ./build/Debug/PlasmaDX-Clean.exe \
        --config=$scenario \
        --dump-buffers 120 \
        --dump-dir "analysis/$(basename $scenario .json)"
done

# Compare results
python PIX/scripts/analysis/analyze_5_captures.py analysis/*
```

---

## Troubleshooting

### Config Not Loading

**Symptom:** Application uses default hardcoded config

**Fixes:**
1. Check file path is correct (use absolute path if unsure)
2. Verify JSON syntax (use JSONLint or VS Code)
3. Check logs for "Failed to load config" messages
4. Try with `--config=` (full path)

### Agent Can't Modify Config

**Symptom:** Agent changes don't take effect

**Fixes:**
1. Ensure agent uses `--config=configs/agents/pix_agent.json`
2. Verify file permissions (must be writable)
3. Check agent writes valid JSON
4. Verify no syntax errors after modification

### Build Uses Wrong Config

**Symptom:** Debug build loads DebugPIX config (or vice versa)

**Fixes:**
1. Check symlinks in build directories are correct
2. Verify command-line doesn't override with `--config=`
3. Check `PLASMADX_CONFIG` environment variable
4. Delete config.json from build directory to force user default

---

## Best Practices

### For Users

1. **Don't modify configs/ files directly** - Copy to user/ first
2. **Use version control** - Commit config changes for reproducibility
3. **Document custom configs** - Add `_comments` section explaining purpose
4. **Test before committing** - Verify config loads without errors

### For AI Agents

1. **Only modify agent configs** - Don't touch user/ or builds/ configs
2. **Validate JSON before writing** - Use `json.loads()` to check syntax
3. **Document changes** - Add `_agent_notes` with reasoning
4. **Restore defaults after testing** - Don't leave agent in weird state

### For Debugging

1. **Use scenario configs** - Don't modify build defaults for quick tests
2. **Name configs descriptively** - `restir_close_bug_v2.json` not `test.json`
3. **Keep working configs** - Archive configs that produce useful results
4. **Compare systematically** - Use batch testing for multi-scenario analysis

---

## Version Control

### .gitignore Rules

```gitignore
# User custom configs (personal settings)
configs/user/user_custom.json
configs/user/*_local.json

# Temporary test configs
configs/scenarios/*_temp.json
configs/scenarios/*_test*.json

# Agent-modified configs (regenerated each session)
configs/agents/*_temp.json
```

### Committed Files

**DO commit:**
- All default configs (builds/, scenarios/, templates/)
- Agent base configs (agents/pix_agent.json template)
- This README

**DON'T commit:**
- User customizations (user_custom.json)
- Temporary test configs (*_temp.json, *_test.json)
- Agent-modified configs (agent regenerates them)

---

## Migration Notes

### From Old System (Pre-2025-10-14)

**Old configs → New locations:**

- `config.json` → `configs/user/default.json`
- `config_dev.json` → `configs/user/default.json` (merged)
- `config_user.json` → `configs/user/user_custom.json`
- `config_pix_analysis.json` → `configs/agents/pix_agent.json`
- `config_pix_close.json` → `configs/scenarios/close_distance.json`
- `config_pix_far.json` → `configs/scenarios/far_distance.json`

**Backward compatibility:**

- Root `config.json` still works (symlink to user/default.json)
- Old `--config=config_dev.json` paths work (symlinks preserved)
- Environment variable `PLASMADX_CONFIG` still supported

---

## Additional Resources

- **Full config reference:** [../CONFIG_REFERENCE.md](../CONFIG_REFERENCE.md)
- **PIX agent system:** [../PIX/PIX_AUTONOMOUS_AGENT.md](../PIX/PIX_AUTONOMOUS_AGENT.md)
- **Buffer dump analysis:** [../PIX/docs/BUFFER_DUMP_IMPLEMENTATION.md](../PIX/docs/BUFFER_DUMP_IMPLEMENTATION.md)
- **Session start guide:** [../SESSION_START_CHECKLIST.md](../SESSION_START_CHECKLIST.md)

---

**Questions or Issues?**

1. Check this README first
2. Check CONFIG_REFERENCE.md for parameter details
3. Check logs for config loading messages
4. Create GitHub issue if bug found

**Last Updated:** 2025-10-14
