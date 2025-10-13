# PlasmaDX-Clean Configuration Reference

**Version:** 1.0
**Last Updated:** October 12, 2025
**Purpose:** Complete reference for all configuration parameters and runtime controls

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration System Overview](#configuration-system-overview)
3. [Parameter Groups](#parameter-groups)
4. [Runtime Keyboard Controls](#runtime-keyboard-controls)
5. [Configuration Profiles](#configuration-profiles)
6. [PIX Agent Integration](#pix-agent-integration)

---

## Quick Start

### Loading a Config File

```bash
# Default config (config.json in project root)
./PlasmaDX-Clean.exe

# Specific config file
./PlasmaDX-Clean.exe --config=config_restir_test.json

# Environment variable override
export PLASMADX_CONFIG=config_pix_analysis.json
./PlasmaDX-Clean.exe
```

### Config Priority Order

1. **Command-line:** `--config=file.json`
2. **Environment variable:** `PLASMADX_CONFIG=file.json`
3. **Default:** `config.json` (if exists)
4. **Fallback:** Hardcoded defaults in source

---

## Configuration System Overview

### File Format

- **Format:** JSON with embedded comments (fields starting with `_` are ignored)
- **Location:** Project root directory
- **Hot-reload:** Not supported (restart required)

### Metadata Fields

All fields starting with underscore are documentation:
- `_note`: Parameter description
- `_runtime`: Runtime keyboard control (if available)
- `_cmdline`: Command-line argument (if available)
- `_envvar`: Environment variable (if available)
- `_warning`: Important usage warnings

---

## Parameter Groups

### 1. Rendering Configuration

Controls particle count, renderer type, and resolution.

| Parameter | Type | Default | Range | Runtime Control |
|-----------|------|---------|-------|-----------------|
| `particleCount` | int | 10000 | 1000-100000 | None |
| `renderer` | string | "Gaussian" | "Gaussian", "Billboard" | None |
| `resolutionWidth` | int | 1920 | 800-3840 | None |
| `resolutionHeight` | int | 1080 | 600-2160 | None |

**Command-line overrides:**
- `--gaussian` or `-g`: Use Gaussian renderer
- `--billboard` or `-b`: Use Billboard renderer
- `--particles 20000`: Set particle count

---

### 2. Camera Configuration

Initial camera position and movement speed.

| Parameter | Type | Default | Range | Runtime Control |
|-----------|------|---------|-------|-----------------|
| `startDistance` | float | 800.0 | 100-5000 | None |
| `startHeight` | float | 1200.0 | 0-5000 | None |
| `startAngle` | float | 0.0 | 0-2π | None |
| `startPitch` | float | 0.0 | -π/2 to π/2 | None |
| `moveSpeed` | float | 100.0 | 10-1000 | None |
| `rotateSpeed` | float | 0.5 | 0.1-5.0 | None |
| `particleSize` | float | 50.0 | 10-200 | **+/- keys** |

**Runtime controls:**
- **WASD:** Camera movement
- **CTRL+Mouse:** Camera rotation
- **+ (Plus):** Increase particle size
- **- (Minus):** Decrease particle size

---

### 3. Physics Simulation

Black hole accretion disk physics parameters.

| Parameter | Type | Default | Range | Runtime Control |
|-----------|------|---------|-------|-----------------|
| `innerRadius` | float | 10.0 | 3-50 | None |
| `outerRadius` | float | 300.0 | 50-1000 | None |
| `diskThickness` | float | 50.0 | 10-200 | None |
| `timeStep` | float | 0.00833 | 0.001-0.1 | None |
| `physicsEnabled` | bool | true | - | **P key** |
| `gravityStrength` | float | 200.0 | 0-1000 | **V key** |
| `angularMomentum` | float | 2.5 | 0-10 | **N key** |
| `turbulence` | float | 10.0 | 0-50 | **B key** |
| `damping` | float | 0.95 | 0-1 | **M key** |

**Runtime controls:**
- **P:** Toggle physics on/off
- **CTRL+V / SHIFT+V:** Increase/decrease gravity
- **CTRL+N / SHIFT+N:** Increase/decrease angular momentum
- **CTRL+B / SHIFT+B:** Increase/decrease turbulence
- **CTRL+M / SHIFT+M:** Increase/decrease damping

---

### 4. Ray Tracing Features

DXR 1.1 lighting features for Gaussian renderer.

| Parameter | Type | Default | Range | Runtime Control |
|-----------|------|---------|-------|-----------------|
| `enableShadowRays` | bool | true | - | **F5 key** |
| `enableInScattering` | bool | false | - | **F6 key** |
| `inScatterStrength` | float | 1.0 | 0-10 | **F9 key** |
| `enablePhaseFunction` | bool | true | - | **F8 key** |
| `phaseStrength` | float | 5.0 | 0-20 | **F8 + modifiers** |
| `rtLightingStrength` | float | 2.0 | 0-10 | **F10 key** |

**Runtime controls:**
- **F5:** Toggle shadow rays
- **F6:** Toggle volumetric in-scattering
- **F8:** Toggle phase function (CTRL+F8/SHIFT+F8 adjust strength)
- **F9 / SHIFT+F9:** Increase/decrease in-scatter strength
- **F10 / SHIFT+F10:** Increase/decrease RT lighting strength

---

### 5. ReSTIR Configuration

Reservoir-based Spatio-Temporal Importance Resampling.

| Parameter | Type | Default | Range | Runtime Control |
|-----------|------|---------|-------|-----------------|
| `enableReSTIR` | bool | false | - | **F7 key** |
| `restirInitialCandidates` | int | 16 | 4-32 | None |
| `restirTemporalReuse` | bool | true | - | None |
| `restirSpatialReuse` | bool | false | - | None |
| `restirTemporalWeight` | float | 0.95 | 0.8-0.99 | **F7 + modifiers** |

**Runtime controls:**
- **F7:** Toggle ReSTIR on/off
- **CTRL+F7:** Increase temporal weight (+0.1)
- **SHIFT+F7:** Decrease temporal weight (-0.1)

**⚠️ Warning:** ReSTIR may cause brightness issues at close distances (actively being debugged).

---

### 6. Gaussian Splatting

Anisotropic Gaussian parameters (velocity-based stretching).

| Parameter | Type | Default | Range | Runtime Control |
|-----------|------|---------|-------|-----------------|
| `useAnisotropicGaussians` | bool | true | - | **F11 key** |
| `anisotropyStrength` | float | 2.0 | 0-5 | **F12 key** |

**Runtime controls:**
- **F11:** Toggle anisotropic Gaussians
- **F12 / SHIFT+F12:** Increase/decrease anisotropy strength

---

### 7. Physical Effects (Relativistic)

Advanced physical simulation features.

| Parameter | Type | Default | Range | Runtime Control |
|-----------|------|---------|-------|-----------------|
| `usePhysicalEmission` | bool | false | - | **E key** |
| `emissionStrength` | float | 1.0 | 0-5 | **E + modifiers** |
| `useDopplerShift` | bool | false | - | **G key** |
| `dopplerStrength` | float | 1.0 | 0-5 | **G + modifiers** |
| `useGravitationalRedshift` | bool | false | - | **None** |
| `redshiftStrength` | float | 1.0 | 0-5 | **None** |

**Runtime controls:**
- **E:** Toggle physical emission (CTRL+E/SHIFT+E adjust strength)
- **G:** Toggle Doppler shift (CTRL+G/SHIFT+G adjust strength)
- **I/K:** Increase/Decrease RT light intensity (×2 / ×0.5)
- **O/L:** Increase/Decrease RT max distance (+100 / -100)

---

### 8. Debug & Diagnostics

Debugging and logging options.

| Parameter | Type | Default | Range | Runtime Control |
|-----------|------|---------|-------|-----------------|
| `enableDebugLayer` | bool | false | - | None |
| `logLevel` | string | "Info" | "Info", "Debug", "Trace" | None |
| `enablePIX` | bool | false | - | None |
| `pixAutoCapture` | bool | false | - | **Environment var** |
| `pixCaptureFrame` | int | 120 | 1-10000 | **Environment var** |
| `showFPS` | bool | true | - | None |
| `showParticleStats` | bool | true | - | None |
| `logPhysicsUpdates` | bool | false | - | None |

**Environment variables:**
- `PIX_AUTO_CAPTURE=1`: Enable automatic capture
- `PIX_CAPTURE_FRAME=120`: Set capture frame number

**⚠️ Warning:** Debug layer causes 5-10× slowdown. Use DebugPIX build instead.

---

### 9. PIX Automated Analysis

Configuration for PIX Agent v2/v3 automated testing.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `captureFrames` | int[] | [1,60,120,300] | Frames to capture |
| `capturePrefix` | string | "analysis_" | Filename prefix |
| `enableReservoirLogging` | bool | false | Log ReSTIR states |
| `enablePerformanceCounters` | bool | false | GPU timing mode |
| `trackResourceUsage` | bool | false | VRAM tracking |

---

## Runtime Keyboard Controls

### Movement & Camera

| Key | Action |
|-----|--------|
| **W** | Move forward |
| **A** | Move left |
| **S** | Move backward |
| **D** | Move right |
| **CTRL+Mouse** | Look around (camera rotation) |
| **+ (Plus)** | Increase particle size |
| **- (Minus)** | Decrease particle size |
| **P** | Toggle physics simulation |
| **ESC** | Exit application |

### Rendering Features (Function Keys)

| Key | Action | Modifiers |
|-----|--------|-----------|
| **F5** | Toggle shadow rays | - |
| **F6** | Toggle in-scattering | - |
| **F7** | Toggle ReSTIR | CTRL: +weight, SHIFT: -weight |
| **F8** | Toggle phase function | CTRL: +strength, SHIFT: -strength |
| **F9** | Adjust in-scatter strength | SHIFT: decrease |
| **F10** | Adjust RT lighting strength | SHIFT: decrease |
| **F11** | Toggle anisotropic Gaussians | - |
| **F12** | Adjust anisotropy strength | SHIFT: decrease |

### Physics Controls

| Key | Action | Modifiers |
|-----|--------|-----------|
| **V** | Adjust gravity | CTRL: increase, SHIFT: decrease |
| **N** | Adjust angular momentum | CTRL: increase, SHIFT: decrease |
| **B** | Adjust turbulence | CTRL: increase, SHIFT: decrease |
| **M** | Adjust damping | CTRL: increase, SHIFT: decrease |

### Physical Effects & RT Lighting

| Key | Action | Modifiers |
|-----|--------|-----------|
| **E** | Toggle physical emission | CTRL: +strength, SHIFT: -strength |
| **G** | Toggle Doppler shift | CTRL: +strength, SHIFT: -strength |
| **I** | Increase RT light intensity | Doubles intensity (×2) |
| **K** | Decrease RT light intensity | Halves intensity (×0.5) |
| **O** | Increase RT max distance | +100 units |
| **L** | Decrease RT max distance | -100 units (min 50) |

---

## Configuration Profiles

### Available Profiles

1. **config.json** (Default)
   - Balanced settings for general use
   - Gaussian renderer, 10K particles
   - ReSTIR disabled by default

2. **config_dev.json** (Developer)
   - Fast iteration mode
   - Minimal logging
   - Lower particle count (5K)
   - Debug features disabled

3. **config_user.json** (User Experience)
   - Maximum visual quality
   - All effects enabled
   - Higher particle count (20K)
   - Optimized for showcase

4. **config_pix_analysis.json** (PIX Analysis)
   - Diagnostic features enabled
   - Reservoir logging active
   - Performance counters on
   - For automated testing

5. **config_restir_test.json** (ReSTIR Testing)
   - ReSTIR enabled by default
   - Frame 120 capture
   - For brightness fix verification

### Loading Profiles

```bash
# Command-line
./PlasmaDX-Clean.exe --config=config_user.json

# Environment variable
export PLASMADX_CONFIG=config_pix_analysis.json
./PlasmaDX-Clean.exe

# PIX Agent v3 (Python)
import json
config = json.load(open("config_restir_test.json"))
config["restir"]["enableReSTIR"] = True
json.dump(config, open("config.json", "w"))
# Launch app
```

---

## PIX Agent Integration

### Modifying Config from Python

```python
import json

# Load template
with open("config_restir_test.json", "r") as f:
    config = json.load(f)

# Modify parameters
config["restir"]["enableReSTIR"] = True
config["restir"]["restirInitialCandidates"] = 32
config["debug"]["pixCaptureFrame"] = 120

# Save for app to load
with open("config.json", "w") as f:
    json.dump(config, f, indent=2)

# Launch app (will read config.json)
subprocess.run(["./PlasmaDX-Clean-PIX.exe"])
```

### Automated Testing Workflow

```python
# Test matrix: ReSTIR ON vs OFF at different distances
test_configs = [
    {"enableReSTIR": False, "captureFrame": 120, "startDistance": 800},
    {"enableReSTIR": True, "captureFrame": 120, "startDistance": 800},
    {"enableReSTIR": False, "captureFrame": 120, "startDistance": 400},
    {"enableReSTIR": True, "captureFrame": 120, "startDistance": 400},
]

for i, test in enumerate(test_configs):
    # Modify config
    config["restir"]["enableReSTIR"] = test["enableReSTIR"]
    config["camera"]["startDistance"] = test["startDistance"]
    config["debug"]["pixCaptureFrame"] = test["captureFrame"]

    # Save and launch
    save_config("config.json", config)
    capture_path = f"pix/Captures/test_{i}.wpix"
    launch_and_capture(capture_path)

    # Analyze
    analyze_reservoirs(capture_path)
    compare_brightness(capture_path, baseline_path)
```

---

## Best Practices

### For Development

1. **Use Debug build** for fast iteration (no PIX overhead)
2. **Start with low particle count** (5-10K) for faster load times
3. **Disable expensive effects** (ReSTIR, in-scattering) until needed
4. **Enable logPhysicsUpdates** only when debugging physics

### For Visual Quality

1. **Increase particle count** to 20-50K
2. **Enable all RT features** (shadows, phase function, in-scattering)
3. **Adjust rtLightingStrength** to taste (F10 key)
4. **Try anisotropic Gaussians** for motion blur effect (F11)

### For PIX Analysis

1. **Use DebugPIX build** (has PIX support compiled in)
2. **Disable debug layer** (use PIX's debug features instead)
3. **Set captureFrame** to 120+ for temporal convergence
4. **Enable reservoir logging** for ReSTIR debugging

### For Performance Testing

1. **Disable all optional features** (baseline measurement)
2. **Enable one feature at a time** to isolate impact
3. **Use multiple particle counts** (5K, 10K, 20K, 50K, 100K)
4. **Track GPU timings** with PIX timing mode

---

## Troubleshooting

### Config Not Loading

- Check file exists in project root
- Verify JSON syntax (no trailing commas, proper quotes)
- Ignore fields starting with `_` (they're comments)
- Check console output for parsing errors

### Runtime Keys Not Working

- Ensure application window has focus
- Some keys require modifiers (CTRL/SHIFT)
- Function keys (F1-F12) work without modifiers
- Dvorak layout shouldn't affect function keys

### PIX Capture Issues

- Use DebugPIX build, not Debug build
- Set environment variables before launch
- Check `enablePIX` is false (use DebugPIX build instead)
- Verify capture path is writable

### ReSTIR Artifacts

- See [RESTIR_BRIGHTNESS_FIX_20251012.md](RESTIR_BRIGHTNESS_FIX_20251012.md)
- Toggle with F7 to compare ON vs OFF
- Adjust temporal weight (CTRL+F7 / SHIFT+F7)
- Try different candidate counts (4, 8, 16, 32)

---

## Additional Resources

- **PIX Agent v3 Documentation:** `pix/PIX_AGENT_V3_README.md`
- **ReSTIR Debugging Guide:** `RESTIR_DEBUG_BRIEFING.md`
- **Build System:** `PIX_DUAL_BINARY_SETUP.md`
- **Keyboard Controls Quick Ref:** Printed at app startup

---

**Document Version:** 1.0
**Last Updated:** October 12, 2025
**Maintained by:** PlasmaDX-Clean Development Team
**For AI Agents:** This document is optimized for compact chat windows and cross-session knowledge transfer.
