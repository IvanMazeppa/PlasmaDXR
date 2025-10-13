# Configuration System Guide

## Overview

The PlasmaDX-Clean configuration system provides three profile-based JSON configurations for different use cases:
1. **`config_dev.json`** - Fast developer iteration (default)
2. **`config_user.json`** - User experience and experimentation
3. **`config_pix_analysis.json`** - PIX debugging and analysis

## Quick Start

### Using Default Config (Dev Mode)
```bash
# Automatically loads config_dev.json if present
.\build\Debug\PlasmaDX-Clean.exe
```

### Using Specific Config
```bash
# Load user experience config
.\build\Debug\PlasmaDX-Clean.exe --config=config_user.json

# Load PIX analysis config with DebugPIX build
.\build\DebugPIX\PlasmaDX-Clean-PIX.exe --config=config_pix_analysis.json
```

### Command-Line Overrides
```bash
# Override config settings via command line
.\build\Debug\PlasmaDX-Clean.exe --config=config_user.json --particles=50000 --restir
```

## Configuration Profiles

### config_dev.json - Developer Mode (Default)
**Purpose:** Fast iteration, daily development, code debugging

**Key Settings:**
- **Particles:** 10,000 (fast)
- **ReSTIR:** Disabled (faster)
- **In-Scattering:** Disabled (faster)
- **Debug Layer:** Enabled
- **PIX:** Disabled (zero overhead)

**Use when:**
- Writing new features
- Debugging C++ code
- Testing physics or rendering changes
- You want maximum performance
- Quick iteration cycles

### config_user.json - User Experience Mode
**Purpose:** Visual quality, experimentation, showcasing

**Key Settings:**
- **Particles:** 20,000 (better quality)
- **ReSTIR:** Enabled (realistic lighting)
- **In-Scattering:** Disabled (can enable with F6)
- **Debug Layer:** Disabled (production-like)
- **PIX:** Disabled (no overhead)

**Use when:**
- Running the demo for others
- Experimenting with visual settings
- Recording videos or screenshots
- Showcasing the renderer

### config_pix_analysis.json - PIX Analysis Mode
**Purpose:** GPU debugging, performance analysis, ReSTIR investigation

**Key Settings:**
- **Particles:** 10,000 (manageable for analysis)
- **ReSTIR:** Enabled (for debugging)
- **Camera:** Close to black hole (100 distance, 200 height) - triggers ReSTIR issues
- **Debug Layer:** Disabled (PIX provides validation)
- **PIX:** Enabled (auto-capture at frame 120)
- **Logging:** Debug level (verbose)
- **ReSTIR Tracking:** Enabled (reservoir state logging)

**Use when:**
- Debugging GPU rendering issues
- Analyzing performance with PIX
- Investigating ReSTIR over-exposure
- Running automated captures via agent
- Need detailed diagnostic data

## Configuration Structure

### Rendering Settings
```json
"rendering": {
  "particleCount": 10000,
  "rendererType": "gaussian|billboard",
  "width": 1920,
  "height": 1080
}
```

### Features (RT Effects)
```json
"features": {
  "enableReSTIR": false,
  "restirCandidates": 16,
  "restirTemporalReuse": true,
  "restirSpatialReuse": true,
  "restirTemporalWeight": 0.9,

  "enableInScattering": false,
  "inScatterStrength": 1.0,

  "enableShadowRays": true,
  "enablePhaseFunction": true,
  "phaseStrength": 5.0,

  "useAnisotropicGaussians": true,
  "anisotropyStrength": 1.0,

  "rtLightingStrength": 2.0,

  "usePhysicalEmission": false,
  "emissionStrength": 1.0,

  "useDopplerShift": false,
  "dopplerStrength": 1.0,

  "useGravitationalRedshift": false,
  "redshiftStrength": 1.0
}
```

### Physics Settings
```json
"physics": {
  "innerRadius": 10.0,
  "outerRadius": 300.0,
  "diskThickness": 50.0,
  "timeStep": 0.008333,
  "physicsEnabled": true
}
```

### Camera Configuration
```json
"camera": {
  "startDistance": 800.0,
  "startHeight": 1200.0,
  "startAngle": 0.0,
  "startPitch": 0.0,
  "moveSpeed": 100.0,
  "rotateSpeed": 0.5,
  "particleSize": 50.0
}
```

### Debug Settings
```json
"debug": {
  "enableDebugLayer": false,
  "logLevel": "info|debug|trace",
  "enablePIX": false,
  "pixAutoCapture": false,
  "pixCaptureFrame": 120,
  "showFPS": true,
  "showParticleStats": true
}
```

### PIX Analysis Settings
```json
"pix_analysis": {
  "capturePrefix": "analysis_",
  "enableReservoirLogging": true,
  "enablePerformanceCounters": true,
  "trackResourceUsage": true
}
```

## Command-Line Options

### Config Selection
```bash
--config=<file>          # Load specific JSON config file
```

### Renderer Override
```bash
--gaussian, -g           # Use Gaussian Splatting renderer
--billboard, -b          # Use Billboard renderer
```

### Particle Count Override
```bash
--particles <count>      # Set particle count (overrides config)
```

### ReSTIR Override
```bash
--restir                 # Enable ReSTIR (overrides config)
--no-restir              # Disable ReSTIR (overrides config)
```

### Help
```bash
--help, -h               # Show command-line options
```

## Configuration Priority

Settings are applied in this order (later overrides earlier):
1. **Hardcoded defaults** (in code)
2. **Config file** (JSON profile)
3. **Environment variable** (`PLASMADX_CONFIG`)
4. **Command-line arguments** (highest priority)

## Environment Variables

```bash
# Override config file via environment variable
export PLASMADX_CONFIG=config_pix_analysis.json
./build/Debug/PlasmaDX-Clean.exe
```

## Runtime Hotkeys

Even with config loaded, you can toggle features at runtime:

### Visual Features
- **F5**: Toggle shadow rays
- **F6**: Toggle in-scattering
- **F7**: Toggle ReSTIR (Ctrl/Shift+F7 adjust temporal weight)
- **F8**: Toggle phase function (Ctrl/Shift+F8 adjust strength)
- **F9/Shift+F9**: Adjust in-scattering strength
- **F10/Shift+F10**: Adjust RT lighting strength
- **F11**: Toggle anisotropic Gaussians
- **F12/Shift+F12**: Adjust anisotropy strength

### Camera Controls
- **Arrow Keys**: Move camera height/angle
- **W/A**: Adjust camera distance
- **CTRL+LMB Drag**: Mouse look

### Physics Controls
- **V/Ctrl+V/Shift+V**: Adjust gravity strength
- **N/Ctrl+N/Shift+N**: Adjust angular momentum
- **B/Ctrl+B/Shift+B**: Adjust turbulence
- **M/Ctrl+M/Shift+M**: Adjust damping

### Debug
- **P**: Toggle physics simulation
- **D**: Read back particle data from GPU
- **C**: Log current camera state
- **ESC**: Exit application

## Examples

### Example 1: Quick Test with High Particle Count
```bash
# Use dev config but override to 50K particles
.\build\Debug\PlasmaDX-Clean.exe --particles=50000
```

### Example 2: User Mode with ReSTIR Disabled
```bash
# Use user config but disable ReSTIR for performance
.\build\Debug\PlasmaDX-Clean.exe --config=config_user.json --no-restir
```

### Example 3: PIX Analysis with Custom Capture Frame
```bash
# Use PIX config, modify auto-capture frame in JSON first
# Edit config_pix_analysis.json: "pixCaptureFrame": 300
.\build\DebugPIX\PlasmaDX-Clean-PIX.exe --config=config_pix_analysis.json
```

### Example 4: Custom Camera Position
```bash
# Edit config_pix_analysis.json camera settings:
{
  "camera": {
    "startDistance": 50.0,    # Very close to black hole
    "startHeight": 100.0,     # Low angle
    ...
  }
}
```

## Creating Custom Configs

### Method 1: Copy Existing Config
```bash
# Copy and modify
cp config_dev.json config_custom.json
# Edit config_custom.json as needed
.\build\Debug\PlasmaDX-Clean.exe --config=config_custom.json
```

### Method 2: Minimal Config
Create a minimal JSON with only the settings you want to override:
```json
{
  "profile": "custom",
  "rendering": {
    "particleCount": 30000
  },
  "features": {
    "enableReSTIR": true
  }
}
```
All other settings will use hardcoded defaults.

## Troubleshooting

### Config Not Loading
**Problem:** Application doesn't seem to load config file

**Solutions:**
1. Check config file is in the same directory as the executable
2. Verify JSON syntax is valid (use a JSON validator)
3. Check log output: `=== Configuration Loaded ===` message
4. Use `--config=` with full path if needed

### ReSTIR Not Working
**Problem:** ReSTIR enabled in config but not seeing effect

**Solutions:**
1. Check log: `useReSTIR: 1` should appear in Gaussian Constants
2. Press F7 at runtime to toggle manually
3. Ensure Gaussian renderer is selected (not Billboard)
4. ReSTIR effects are subtle - look for faster lighting convergence

### PIX Auto-Capture Not Triggering
**Problem:** PIX config loaded but capture doesn't happen

**Solutions:**
1. Use **DebugPIX build** (`PlasmaDX-Clean-PIX.exe`), not Debug build
2. Check `enablePIX: true` and `pixAutoCapture: true` in config
3. Verify `pixCaptureFrame` is set correctly
4. Look for PIX log messages about capture starting/ending

### Config Changes Not Applying
**Problem:** Changed config but app still uses old settings

**Solutions:**
1. Restart application (configs are loaded once at startup)
2. Verify you're editing the correct config file
3. Check command-line arguments aren't overriding your changes
4. Make sure JSON syntax is valid (trailing commas are not allowed)

## Best Practices

### Development Workflow
1. **Daily work:** Use `config_dev.json` (default)
2. **Feature complete:** Test with `config_user.json` (realistic settings)
3. **Performance issues:** Use `config_pix_analysis.json` + DebugPIX build
4. **ReSTIR debugging:** Load `config_pix_analysis.json`, position camera close to light

### Git Workflow
- **Commit** default configs (config_dev.json, config_user.json, config_pix_analysis.json)
- **Don't commit** custom test configs (add to .gitignore if needed)
- **Document** any config changes that affect behavior

### PIX Analysis Workflow
1. Build DebugPIX: `MSBuild /p:Configuration=DebugPIX`
2. Launch with PIX config: `.\build\DebugPIX\PlasmaDX-Clean-PIX.exe --config=config_pix_analysis.json`
3. Let it run to auto-capture frame (default 120)
4. Application will exit after capture
5. Check `pix/Captures/` for the `.wpix` file
6. Analyze with PIX tool or agent scripts

## Technical Details

### Config Loading Process
1. Check command-line for `--config=` argument
2. Check environment variable `PLASMADX_CONFIG`
3. Check for `config_dev.json` in current directory
4. Fall back to hardcoded defaults if nothing found
5. Log which config was loaded

### Config Parser
- Lightweight custom JSON parser (no external dependencies)
- Supports strings, numbers, booleans
- Handles nested objects
- Ignores unknown keys (forward compatible)
- Tolerant of missing keys (uses defaults)

### Performance Impact
- Config loading: ~1-5ms at startup (negligible)
- No runtime overhead (loaded once at init)
- Hot-reload not currently supported (restart required)

## See Also

- [BUILD_GUIDE.md](BUILD_GUIDE.md) - Build system documentation
- [PIX_DUAL_BINARY_SETUP.md](PIX_DUAL_BINARY_SETUP.md) - PIX integration details
- [PIX_ANALYSIS_REPORT_V1.md](pix/Analysis/PIX_ANALYSIS_REPORT_V1.md) - Example PIX analysis

## Status: Complete ✅

The configuration system is fully implemented and integrated. All three profile configs are ready to use.

---

**Last Updated:** October 12, 2025
**Version:** 1.0
**Status:** ✅ Production Ready
