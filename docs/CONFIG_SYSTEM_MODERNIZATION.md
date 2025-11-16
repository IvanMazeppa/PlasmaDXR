# Config System Modernization

**Created**: 2025-11-15
**Purpose**: Update config system to support modern lighting/effects for agent-driven testing

---

## New Config Templates Created

**Location**: `configs/scenarios/`

1. **multi_light_only.json** - Multi-light isolated (quality reference)
2. **probe_grid_only.json** - Probe-grid isolated (performance comparison)
3. **hybrid_lighting.json** - Both enabled (target: 90+ FPS, 90% quality)

---

## Code Changes Needed

### **1. Add Config Loading for New Lighting Systems**

**File**: `src/core/Application.cpp`

**Location**: In command-line parsing or LoadConfig() function

```cpp
// Add new config sections
struct LightingConfig {
    struct MultiLightConfig {
        bool enabled = true;
        uint32_t lightCount = 16;
        std::string preset = "stellar_ring";
        float intensity = 1.0f;
        float blendWeight = 1.0f;
    } multiLight;

    struct ProbeGridConfig {
        bool enabled = false;
        uint32_t gridSize = 32;
        uint32_t raysPerProbe = 16;
        float intensity = 800.0f;
        uint32_t updateInterval = 4;
        float blendWeight = 1.0f;
    } probeGrid;

    struct RTXDIConfig {
        bool enabled = false;
        std::string mode = "M5";
        float temporalWeight = 0.9f;
    } rtxdi;
};

// Parse from JSON (using nlohmann/json or similar)
if (config.contains("lighting")) {
    auto& lighting = config["lighting"];

    if (lighting.contains("multiLight")) {
        m_useMultiLight = lighting["multiLight"]["enabled"];
        // ... parse other multiLight params
    }

    if (lighting.contains("probeGrid")) {
        m_useProbeGrid = lighting["probeGrid"]["enabled"];
        m_probeGridIntensity = lighting["probeGrid"]["intensity"];
        // ... parse other probeGrid params
    }

    if (lighting.contains("rtxdi")) {
        m_lightingSystem = lighting["rtxdi"]["enabled"]
            ? LightingSystem::RTXDI
            : LightingSystem::MultiLight;
        // ... parse RTXDI params
    }
}
```

### **2. Add Effects Config Support**

```cpp
struct EffectsConfig {
    struct ShadowRaysConfig {
        bool enabled = true;
        std::string preset = "balanced";  // performance, balanced, quality
        uint32_t raysPerPixel = 4;
        bool temporalFiltering = true;
    } shadowRays;

    struct VolumetricConfig {
        bool anisotropicGaussians = true;
        float anisotropyStrength = 1.0f;
        bool adaptiveRadius = true;
        bool rtParticleToParticle = true;
        float rtLightingStrength = 2.0f;
    } volumetric;
};

// Load from config["effects"]
```

### **3. Add Screenshot Auto-Capture Support**

```cpp
struct ScreenshotConfig {
    bool autoCapture = false;
    uint32_t captureFrame = 120;
    std::string outputPrefix = "screenshot_";
    std::string outputFormat = "bmp";
};

// In render loop:
if (m_screenshotConfig.autoCapture && m_frameCount == m_screenshotConfig.captureFrame) {
    std::string filename = m_screenshotConfig.outputPrefix + GetTimestamp() + "." + m_screenshotConfig.outputFormat;
    CaptureScreenshot(filename);
}
```

---

## Agent Integration

### **For Agents to Use Configs:**

**Option 1: Command-line** (simplest, works today)
```bash
# Agent launches PlasmaDX with specific config
./PlasmaDX-Clean.exe --config=configs/scenarios/multi_light_only.json
# Wait for frame 120
# Press F2 or auto-capture screenshot
```

**Option 2: Agent Tool** (more control)
```python
# New tool in particle-pipeline-runner agent (future)
async def run_plasmadx_with_config(config_path: str, duration_seconds: int):
    """Launch PlasmaDX with config, wait, capture screenshot"""
    # 1. Launch: subprocess.Popen(["./PlasmaDX-Clean.exe", f"--config={config_path}"])
    # 2. Wait: time.sleep(duration_seconds) OR watch for frame 120 in log
    # 3. Capture: Send F2 keypress OR read auto-captured screenshot
    # 4. Terminate: process.terminate()
    # 5. Return: screenshot path
```

### **Lighting Comparison Skill Integration:**

Update `~/.claude/skills/lighting-quality-comparison.md`:

```markdown
### 1. **Capture Current State**

**Automated Approach (Future)**:
```
Agent: particle-pipeline-runner
Tool: run_plasmadx_with_config

# Multi-light baseline
run_plasmadx_with_config(
    config="configs/scenarios/multi_light_only.json",
    duration=10
) → screenshots/multi_light_2025-11-15.bmp

# Probe-grid comparison
run_plasmadx_with_config(
    config="configs/scenarios/probe_grid_only.json",
    duration=10
) → screenshots/probe_grid_2025-11-15.bmp

# Automatic LPIPS comparison
compare_screenshots_ml(before, after)
```

**Manual Approach (Today)**:
```
You: Launch with --config=configs/scenarios/multi_light_only.json
You: Press F2 at frame 120
You: Launch with --config=configs/scenarios/probe_grid_only.json
You: Press F2 at frame 120
Me: Compare screenshots automatically
```
```
---

## Implementation Priority

### **Phase 1: Minimal (1-2 hours)**
- [ ] Parse `lighting.multiLight.enabled` → `m_useMultiLight`
- [ ] Parse `lighting.probeGrid.enabled` → `m_useProbeGrid`
- [ ] Parse `lighting.rtxdi.enabled` → `m_lightingSystem`
- [ ] Test: Launch with each config, verify correct systems enabled

### **Phase 2: Full Config Support (2-3 hours)**
- [ ] Parse all lighting params (intensity, blend weights, etc.)
- [ ] Parse effects params (shadow preset, volumetric settings)
- [ ] Parse screenshot auto-capture
- [ ] Test: All 3 scenarios run correctly

### **Phase 3: Agent Automation (3-4 hours)**
- [ ] Create `particle-pipeline-runner` agent (if doesn't exist)
- [ ] Add `run_plasmadx_with_config` tool
- [ ] Add screenshot capture automation (F2 keypress or ImageGrab)
- [ ] Update lighting-quality-comparison skill to use automation

---

## Benefits for Your Workflow

**Before** (Manual):
```
1. Launch PlasmaDX
2. F3 to disable RTXDI
3. ImGui → disable probe-grid
4. ImGui → set multi-light count to 16
5. Wait for stable frame
6. F2 screenshot
7. Repeat for probe-grid...
```

**After** (Agent-driven):
```
You: "Compare probe-grid to multi-light quality"

Skill automatically:
1. Launches with multi_light_only.json
2. Captures screenshot at frame 120
3. Launches with probe_grid_only.json
4. Captures screenshot at frame 120
5. Runs LPIPS comparison
6. Provides detailed report

Total time: 30 seconds vs 5+ minutes
```

---

## Config Schema Reference

```json
{
  "lighting": {
    "multiLight": { "enabled": bool, "lightCount": int, "intensity": float },
    "probeGrid": { "enabled": bool, "gridSize": int, "raysPerProbe": int },
    "rtxdi": { "enabled": bool, "mode": "M4|M5|M6" }
  },
  "effects": {
    "shadowRays": { "enabled": bool, "preset": "performance|balanced|quality" },
    "inScattering": { "enabled": bool, "strength": float },
    "phaseFunction": { "enabled": bool, "strength": float },
    "volumetric": { "anisotropicGaussians": bool, "adaptiveRadius": bool }
  },
  "screenshot": {
    "autoCapture": bool,
    "captureFrame": int,
    "outputPrefix": string
  }
}
```

---

**Last Updated**: 2025-11-15
**Status**: Config templates created, code changes pending
**Owner**: Ben + Claude Code
