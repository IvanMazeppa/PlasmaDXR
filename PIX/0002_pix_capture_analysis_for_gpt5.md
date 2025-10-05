# PIX Capture Analysis: PlasmaDX Volume Rendering Debug

**Date**: September 21, 2025
**Issue**: Volume rendering shows "homogeneous colored fog" instead of coherent 3D shapes
**Critical Finding**: Off mode (normal rendering) visually identical to DensityProbe mode (single sample)

## Executive Summary

User captured 4 PIX GPU frames with different debug visualization modes. The most significant finding is that **Off mode (g_debugMode=0) produces identical visual output to DensityProbe mode (g_debugMode=3)**, suggesting the ray marching loop may be exiting after a single sample or accumulation is not happening correctly.

## Capture Files

Located in `build-vs2022\Debug\`:
1. **off.wpix** (42KB) - Normal ray marching, should show volumetric rendering
2. **RayDir.wpix** (11MB) - Ray direction visualization
3. **Bounds.wpix** (2MB) - AABB hit/miss visualization
4. **DensityProbe.wpix** (6MB) - Single density sample at entry point

File size variance suggests different GPU workloads, but visual output similarity between Off and DensityProbe is concerning.

## Environment Configuration

- **Build**: Debug configuration with Agility SDK 1.616.1 (stable)
- **Launch Flags**:
  - `PLASMADX_NO_DEBUG=1` - Debug layer disabled for clean captures
  - `PLASMADX_DISABLE_DXR=1` - Forcing compute-only path (no ray tracing)
  - `PLASMADX_NO_QUIT_ON_REMOVAL=1` - Keep window open on device removal

## Expected vs Actual Behavior

### Expected Behavior by Mode:
1. **Off (0)**: Full volumetric ray marching with Beer-Lambert absorption, showing 3D plasma shape
2. **RayDir (1)**: RGB-encoded ray directions, should vary smoothly across screen
3. **Bounds (2)**: Magenta=miss, Green=hit from outside, Yellow=ray starts inside volume
4. **DensityProbe (3)**: Grayscale sphere silhouette from single density sample

### Actual Behavior:
- Off mode produces same visual as DensityProbe (homogeneous fog)
- This suggests one of:
  1. Debug mode not switching (g_debugMode stuck on 3)
  2. Ray march loop exits after first iteration
  3. Accumulation variables not updating in loop

## Key Code Analysis (ray_march_cs.hlsl)

The shader has proper structure:
```hlsl
// Lines 143-174: Main ray marching loop
float3 accumulatedLight = float3(0, 0, 0);
float transmittance = 1.0;
float t = tNear;
uint stepCount = 0;

while (t < tFar && stepCount < g_maxSteps && transmittance > MIN_TRANSMITTANCE) {
    float3 worldPos = rayOrigin + rayDir * t;
    float density = SampleDensity(worldPos);

    if (density > 0.0001) {
        float3 lighting = ComputeLighting(worldPos, density);
        float absorption = density * g_absorption * g_stepSize;
        float stepTransmittance = exp(-absorption);
        accumulatedLight += lighting * transmittance * (1.0 - stepTransmittance);
        transmittance *= stepTransmittance;
    }

    t += g_stepSize;
    stepCount++;
}
```

## Critical PIX Data Points to Check

When analyzing the captures, focus on:

1. **CBV 1 (VolumeConstants) values**:
   - `g_debugMode`: Must be 0,1,2,3 for respective captures
   - `g_stepSize`: Should be ~0.01 for good quality
   - `g_maxSteps`: Should be 128-256
   - `g_absorption`: Non-zero (typically 0.5-2.0)
   - `g_exposure`: Positive value for brightness
   - `g_volumeMin/Max`: Should be (-1,-1,-1) to (1,1,1)

2. **Dispatch Dimensions**:
   - Ray march: ~(120, 68, 1) for 1920x1080 at 16x16 threads
   - Density fill: (16, 16, 16) for 256³ volume

3. **Resource Bindings**:
   - SRV Texture 0: g_density (3D texture)
   - UAV Texture 0: g_hdrTarget (R16G16B16A16_FLOAT)
   - Verify density texture has sphere data from fill pass

## Hypothesis Priority List

1. **Most Likely**: Loop terminates early
   - Check if `g_maxSteps` is 0 or 1
   - Check if `g_stepSize` is too large (> volume size)
   - Check if `MIN_TRANSMITTANCE` threshold too high

2. **Possible**: Accumulation broken
   - Variables might not persist across iterations
   - Lighting calculation returning 0
   - Absorption coefficient too high (instant opacity)

3. **Less Likely**: Debug mode stuck
   - F4 key handler not updating constant buffer
   - CBV update not flushing before dispatch

## Recommended PIX Investigation Using PIXTool

Run these commands to extract data:
```batch
PIXTool.exe timing off.wpix
PIXTool.exe events off.wpix --filter "Dispatch"
PIXTool.exe gpu-captures off.wpix --export-cbv 1
PIXTool.exe diff off.wpix DensityProbe.wpix
```

## Action Items for GPT-5

1. Verify g_debugMode values differ between captures
2. Check loop parameters (stepSize, maxSteps, absorption)
3. Inspect density texture fill success
4. Compare instruction counts between Off and DensityProbe dispatches
5. Look for early-exit conditions in shader assembly

## Previous Working State

User reports previously seeing a blue "mouth-like" volumetric shape, suggesting:
- The rendering pipeline was functional
- Density generation was creating proper 3D shapes
- Ray marching and accumulation worked correctly

Current "homogeneous fog" regression indicates a recent change broke the accumulation loop or parameters.

## Files to Review

1. `src/volumetric/RayMarcher.cpp` - Check SetDebugMode() and constant buffer updates
2. `src/core/App.cpp:1109` - Verify FillAnalytic() call
3. `shaders/vol/ray_march_cs.hlsl` - Main compute shader
4. Recent commits around volume rendering changes

## Next Steps

1. Run PIXTool commands on the 4 captures to extract CBV data
2. Compare g_debugMode and loop parameters between captures
3. If values are correct, add debug output to track loop iterations
4. Consider reverting to last known working commit for comparison

---

**For Claude**: The automated scripts are ready to run:
- `analyze_pix_captures.bat` - Basic PIXTool analysis
- `analyze_pix_detailed.ps1` - Comprehensive PowerShell analysis

These will extract the actual CBV values and dispatch data needed to confirm the hypothesis.