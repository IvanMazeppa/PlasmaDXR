# PIX Manual Capture Workflow - ReSTIR Analysis

## Goal
Capture 4 views with ReSTIR enabled at progressively closer distances to observe color changes and over-exposure issues near the light source.

## Prerequisites
- DebugPIX build is compiled ✓
- WinPixGpuCapturer.dll is in build/DebugPIX/ ✓
- ReSTIR is enabled in all configs ✓
- Particle size set to 20 ✓

## Capture Workflow

### Capture 1: Far View (Distance 800)
1. Copy config: `copy config_pix_far.json config_dev.json`
2. Launch: `build\DebugPIX\PlasmaDX-Clean-PIX.exe`
3. Wait for particles to settle (~60 frames, 1 second)
4. In PIX GUI: **File → Attach to Process → PlasmaDX-Clean-PIX.exe**
5. Click **"GPU Capture"** button in PIX
6. Close app
7. Save capture as: `pix\Captures\restir_far_800.wpix`

### Capture 2: Close View (Distance 300)
1. Copy config: `copy config_pix_close.json config_dev.json`
2. Launch: `build\DebugPIX\PlasmaDX-Clean-PIX.exe`
3. Wait for particles to settle (~60 frames)
4. Attach PIX and capture
5. Save as: `pix\Captures\restir_close_300.wpix`

### Capture 3: Very Close View (Distance 150)
1. Copy config: `copy config_pix_veryclose.json config_dev.json`
2. Launch: `build\DebugPIX\PlasmaDX-Clean-PIX.exe`
3. Wait for particles to settle
4. Attach PIX and capture
5. Save as: `pix\Captures\restir_veryclose_150.wpix`

### Capture 4: Inside Cloud (Distance 50)
1. Copy config: `copy config_pix_inside.json config_dev.json`
2. Launch: `build\DebugPIX\PlasmaDX-Clean-PIX.exe`
3. Wait for particles to settle
4. Attach PIX and capture
5. Save as: `pix\Captures\restir_inside_50.wpix`

## Alternative: Use Keyboard to Navigate (Manual)

If the preset positions don't show the issues:
1. Copy any config to config_dev.json
2. Launch app
3. Use keyboard controls to position camera:
   - **W/A**: Move closer/further
   - **Arrow keys**: Adjust height/angle
   - **+/-**: Adjust particle size (should stay at ~20)
4. When you see ReSTIR issues (color changes, over-exposure), attach PIX and capture
5. Press **C** key to log camera position for documentation

## Verification
After captures, check in PIX that:
- `useReSTIR` = 1 in GaussianConstants buffer
- Frame captured shows actual rendering (not empty/black)
- ReSTIR reservoir buffers are present and populated

## What to Look For
- **Color shifting** as camera approaches light source
- **Over-exposure/blown out highlights** near center
- **Temporal artifacts** (flickering, trailing)
- **Spatial artifacts** (blocky patterns, discontinuities)
- **Incorrect light accumulation** over frames