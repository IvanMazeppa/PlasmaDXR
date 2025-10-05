# PIX Folder Organization

This folder contains all PIX-related files for GPU debugging and analysis of PlasmaDX.

## Directory Structure

```
pix/
├── README.md                                    # This file
├── PIX_SETUP_AND_CAPTURE_GUIDE_FOR_CLAUDE.md  # Original setup guide from GPT-5
├── PIX_AUTOMATION_PROPOSAL_FOR_GPT5.md        # Automation system proposal
├── 0002_pix_capture_analysis_for_gpt5.md      # Analysis of current captures
│
├── Scripts/                                    # Automation scripts
│   ├── analyze_pix_captures.bat              # Basic Windows batch script
│   ├── analyze_pix_detailed.ps1              # PowerShell analysis script
│   ├── capture_all_modes.ps1                 # Automated capture for modes 0..5
│   └── extract_and_analyze.ps1               # Extract timing/events/resources/warnings
│
├── Captures/                                   # PIX capture files
│   ├── off.wpix                              # Normal rendering mode
│   ├── RayDir.wpix                           # Ray direction visualization
│   ├── Bounds.wpix                           # AABB hit/miss visualization
│   └── DensityProbe.wpix                     # Single density sample
│
└── Reports/                                    # Generated analysis reports
    └── (analysis outputs will go here)
```

## Quick Reference

### Manual Capture
1. Open PIX for Windows
2. Launch Process → Set exe to `build-vs2022b/Debug/PlasmaDX.exe`
3. Environment variables:
   - `PLASMADX_NO_DEBUG=1`
   - `PLASMADX_DISABLE_DXR=1`
   - `PLASMADX_NO_QUIT_ON_REMOVAL=1`
4. GPU Capture → 1 frame → Launch

### Automated Capture & Analysis
```powershell
# From PlasmaDX root directory
pwsh -ExecutionPolicy Bypass -File pix\Scripts\capture_all_modes.ps1
pwsh -ExecutionPolicy Bypass -File pix\Scripts\extract_and_analyze.ps1
```
Output: `pix/workflow_claude/pix_analysis_report.md`

### Debug Modes
- Mode 0: Off (normal ray marching)
- Mode 1: RayDir (ray direction visualization)
- Mode 2: Bounds (AABB intersection test)
- Mode 3: DensityProbe (single density sample)
- Mode 4: UVW visualizer (rgb = uvw at entry)
- Mode 5: Step-count heatmap (grayscale)

Press F4 in app to cycle modes. Automation sets `PLASMADX_DEBUG_MODE` per capture.

## Current Issue

**Problem**: Off mode (0) looks identical to DensityProbe mode (3)
**Hypothesis**: Ray marching loop exits after first iteration or debug mode not switching
**Action**: Analyze CBV values in captures to verify g_debugMode and loop parameters








"C:\Program Files\Microsoft PIX\2507.11\pixtool.exe" open-capture "<path>\mode_0.wpix" save-event-list "<path>\test.csv" --counters=*