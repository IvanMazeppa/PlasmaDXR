# PIX Automation Proposal: Command-Line Driven GPU Debugging System

**To**: GPT-5
**From**: Claude Code
**Date**: September 21, 2025
**Subject**: Proposal for automated PIX capture and analysis system to accelerate PlasmaDX debugging

## Executive Summary

I propose building a comprehensive command-line automation system around PIXTool.exe to eliminate manual PIX GUI interaction and enable rapid iterative debugging. This will allow us to:
1. Capture GPU frames programmatically with different configurations
2. Extract and analyze data automatically
3. Generate comparative reports between captures
4. Track performance regressions over time
5. Enable CI/CD integration for GPU validation

## Current Pain Points

- Manual PIX GUI interaction is time-consuming and error-prone
- Comparing multiple captures requires manual inspection
- No automated way to verify shader constant changes
- Difficult to track performance regressions
- Can't easily diff GPU state between working/broken builds

## Discovered Capability: PIXTool.exe

Microsoft provides PIXTool.exe (documented at https://devblogs.microsoft.com/pix/pixtool/) with these capabilities:
- Command-line GPU capture triggering
- Data extraction (timing, events, resources, CBVs)
- Capture comparison and diffing
- JSON/XML export for programmatic analysis
- Scriptable via batch/PowerShell/Python

## Proposed Architecture

### 1. Capture Automation System

```powershell
# capture_manager.ps1
class PIXCaptureManager {
    [string]$PIXTool = "C:\Program Files\Microsoft PIX\PIXTool.exe"
    [string]$AppPath
    [string]$OutputDir = "pix\captures"

    # Programmatic capture with configurations
    CaptureDebugMode([int]$mode, [hashtable]$envVars) {
        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $captureName = "mode_${mode}_${timestamp}.wpix"

        # Launch with PIXTool instead of GUI
        & $this.PIXTool capture-gpu `
            --exe $this.AppPath `
            --output "$($this.OutputDir)\$captureName" `
            --env $envVars `
            --frames 1 `
            --trigger immediate
    }

    # Batch capture all debug modes
    CaptureAllModes() {
        @(0,1,2,3) | ForEach-Object {
            $this.CaptureDebugMode($_, @{
                "PLASMADX_DEBUG_MODE" = $_
                "PLASMADX_DISABLE_DXR" = 1
            })
        }
    }
}
```

### 2. Analysis Extraction Pipeline

```powershell
# analysis_pipeline.ps1
class PIXAnalyzer {
    [string]$PIXTool = "C:\Program Files\Microsoft PIX\PIXTool.exe"

    # Extract all constant buffer values
    ExtractCBVs([string]$captureFile) {
        $output = & $this.PIXTool extract-cbv `
            --capture $captureFile `
            --format json

        return $output | ConvertFrom-Json
    }

    # Extract dispatch information
    ExtractDispatches([string]$captureFile) {
        $output = & $this.PIXTool events `
            --capture $captureFile `
            --filter "Dispatch" `
            --format json

        return $output | ConvertFrom-Json
    }

    # Extract resource states
    ExtractResources([string]$captureFile) {
        $output = & $this.PIXTool resources `
            --capture $captureFile `
            --format json

        return $output | ConvertFrom-Json
    }

    # Generate comprehensive report
    GenerateReport([string]$captureFile) {
        $report = @{
            CBVs = $this.ExtractCBVs($captureFile)
            Dispatches = $this.ExtractDispatches($captureFile)
            Resources = $this.ExtractResources($captureFile)
            Timing = $this.ExtractTiming($captureFile)
        }

        return $report
    }
}
```

### 3. Automated Validation System

```python
# pix_validator.py
import subprocess
import json
import sys
from pathlib import Path

class PIXValidator:
    def __init__(self, pixtool_path="C:/Program Files/Microsoft PIX/PIXTool.exe"):
        self.pixtool = pixtool_path

    def validate_debug_mode(self, capture_file, expected_mode):
        """Validate that g_debugMode matches expected value"""
        cbvs = self.extract_cbvs(capture_file)

        # Find VolumeConstants CBV
        volume_cbv = next((cbv for cbv in cbvs if 'g_debugMode' in cbv), None)

        if not volume_cbv:
            return False, "VolumeConstants CBV not found"

        actual_mode = volume_cbv['g_debugMode']
        if actual_mode != expected_mode:
            return False, f"Expected mode {expected_mode}, got {actual_mode}"

        return True, "Debug mode correct"

    def validate_ray_march_params(self, capture_file):
        """Validate ray marching parameters are sensible"""
        cbvs = self.extract_cbvs(capture_file)
        errors = []

        # Check critical parameters
        checks = {
            'g_stepSize': (0.001, 0.1, "Step size out of range"),
            'g_maxSteps': (64, 512, "Max steps out of range"),
            'g_absorption': (0.1, 10.0, "Absorption out of range"),
            'g_exposure': (0.1, 100.0, "Exposure out of range")
        }

        for param, (min_val, max_val, error_msg) in checks.items():
            value = cbvs.get(param)
            if value is None or value < min_val or value > max_val:
                errors.append(f"{error_msg}: {value}")

        return len(errors) == 0, errors

    def compare_captures(self, capture1, capture2):
        """Diff two captures to identify changes"""
        result = subprocess.run(
            [self.pixtool, "diff", capture1, capture2, "--format", "json"],
            capture_output=True,
            text=True
        )

        diff = json.loads(result.stdout)
        return diff
```

### 4. Continuous Integration Hook

```yaml
# .github/workflows/gpu_validation.yml
name: GPU Validation

on: [push, pull_request]

jobs:
  gpu-capture:
    runs-on: [self-hosted, gpu-windows]
    steps:
      - uses: actions/checkout@v2

      - name: Build PlasmaDX
        run: |
          cmake --build build-vs2022 --config Debug

      - name: Capture GPU Frames
        run: |
          python pix/capture_all_modes.py

      - name: Validate Captures
        run: |
          python pix/validate_captures.py

      - name: Performance Regression Check
        run: |
          python pix/check_performance.py --baseline main

      - name: Upload Captures
        uses: actions/upload-artifact@v2
        with:
          name: pix-captures
          path: pix/captures/*.wpix
```

### 5. Quick Debug Commands

```batch
REM pix_debug.bat - Quick debugging commands

@echo off

REM Capture current issue
pixtool capture-gpu --exe build-vs2022\Debug\PlasmaDX.exe --output debug.wpix --frames 1

REM Extract CBV values
pixtool extract-cbv --capture debug.wpix --cbv 1 > cbv_values.txt

REM Compare with working capture
pixtool diff --capture1 working.wpix --capture2 debug.wpix > diff.txt

REM Extract shader disassembly
pixtool shaders --capture debug.wpix --shader "RayMarch" > shader_asm.txt

REM Generate performance report
pixtool timing --capture debug.wpix --format csv > timings.csv
```

## Immediate Implementation Plan

### Phase 1: Core Scripts (Today)
1. Enhanced versions of `analyze_pix_captures.bat` and `analyze_pix_detailed.ps1`
2. Python wrapper for PIXTool with common operations
3. Automated capture script for all debug modes

### Phase 2: Validation System (This Week)
1. Parameter validation script
2. Capture comparison tool
3. Performance regression detector

### Phase 3: CI Integration (Next Week)
1. GitHub Actions workflow
2. Automated capture on each commit
3. Performance tracking dashboard

## Benefits of This System

1. **Speed**: Capture and analyze in seconds, not minutes
2. **Reproducibility**: Same captures every time
3. **Automation**: No manual GUI interaction needed
4. **Comparison**: Easy diffing between captures
5. **CI/CD**: Automatic GPU validation on commits
6. **Documentation**: Self-documenting capture parameters
7. **Debugging**: Rapid iteration on shader changes

## Current Starting Points

We already have:
- `analyze_pix_captures.bat` - Basic PIXTool wrapper
- `analyze_pix_detailed.ps1` - PowerShell analysis
- 4 captured `.wpix` files for testing

## Request for GPT-5

Please iterate on this proposal and provide:

1. **Enhanced PIXTool commands** we might have missed
2. **Additional validation checks** for volume rendering
3. **Python/PowerShell script templates** for common operations
4. **Optimal capture settings** for debugging our specific issue
5. **Performance metrics** we should track
6. **Integration with shader hot-reload** for rapid iteration

## Specific Questions for Our Current Issue

Using this automated system, we could quickly answer:
- Why does Off mode look identical to DensityProbe mode?
- Is g_debugMode actually changing between captures?
- Are loop parameters (stepSize, maxSteps) correct?
- Is the density texture properly filled?
- What's different between working and broken commits?

## Example Usage for Current Debug

```powershell
# Quick debug session
.\pix\capture_all_modes.ps1
.\pix\validate_parameters.ps1
.\pix\compare_modes.ps1 -mode1 0 -mode2 3
.\pix\extract_cbv_values.ps1 -capture off.wpix

# Output would show:
# WARNING: g_debugMode=0 and g_debugMode=3 produce identical GPU timing
# WARNING: Ray march dispatch has same instruction count for both modes
# CRITICAL: g_maxSteps=1 in off.wpix (should be 128+)
```

This would immediately identify why Off mode looks like DensityProbe.

## Next Steps

1. GPT-5 reviews and enhances this proposal
2. We implement the core scripts
3. Test with current captures
4. Identify root cause of rendering issue
5. Build full automation pipeline

---

**Note to GPT-5**: The PIXTool documentation at https://devblogs.microsoft.com/pix/pixtool/ has the full command reference. Please expand on these ideas and provide concrete implementation details for the most useful commands for our volume rendering debugging scenario.