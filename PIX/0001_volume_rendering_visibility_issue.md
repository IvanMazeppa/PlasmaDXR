# Issue Report #0001: Volume Rendering Visibility Problem

## Problem Description

### User-Reported Symptoms
The user reports that instead of seeing a coherent 3D volumetric shape (like the blue "mouth" shape they had in earlier versions), they are experiencing:

- "Unfortunately i'm still stuck in an space of subtly changing colours"
- "It's really just a homogeneous coloured fog maybe?"
- "Occasionally you'll see a black shape"
- "It's hard to describe"
- "There doesn't seem to be any way to actually see the volume as an actual object like we did before vol0001 (the blue shape that looked like a mouth)"
- "All i see are shades of colour"
- "Feeling around in the dark" and "tesseract-like geometry"

### Expected vs Actual Behavior
- **Expected**: A coherent 3D volumetric shape that responds to camera movement with proper depth and structure
- **Actual**: Abstract color shades, homogeneous fog-like appearance, no discernible 3D structure

## Technical Investigation and Findings

### Root Cause Analysis
I initially suspected that `DensityVolume::FillAnalytic()` was hitting an early return due to uninitialized PSOs, preventing the density texture from being filled. Investigation revealed:

1. **Shader Loading Issue**: The density fill compute shaders (`density_fill.dxil`, `density_slice.dxil`) were not being found due to working directory mismatch
2. **Missing FillAnalytic Call**: The density texture was never being populated before ray marching
3. **PSO Creation Failure**: Pipeline state objects were not being created due to shader loading failures

### Fixes Implemented

#### Fix 1: Shader Path Resolution
Modified `DensityVolume::CreatePipelines()` to try multiple possible paths:
```cpp
std::vector<std::string> possiblePaths = {
    "shaders/density_fill.dxil",
    "../shaders/density_fill.dxil",
    "../../shaders/density_fill.dxil"
};
```

#### Fix 2: Debug Logging
Added comprehensive logging to track:
- Shader loading success/failure
- PSO creation status
- FillAnalytic execution status

#### Fix 3: Verified Call Chain
Confirmed that `App.cpp` line 1109 calls `m_densityVolume->FillAnalytic()` before ray marching.

### Current Status
After fixes, logging confirms:
- ✅ Shaders load successfully: `"Found density_fill.dxil at: shaders/density_fill.dxil"`
- ✅ PSOs create successfully: `"Successfully created density fill PSO"`
- ✅ FillAnalytic executes: `"DensityVolume::FillAnalytic called - PSO:OK RootSig:OK Texture:OK"`
- ✅ Compute dispatch occurs: `"DensityVolume: Dispatching density fill compute shader"`

### Analysis of Persistent Issue

Despite successful density texture filling, the user still reports the same visibility issues. This suggests the problem is **not** in density generation but likely in:

1. **Ray Marching Implementation**: The `ray_march_cs.hlsl` shader may not be correctly:
   - Sampling the density texture
   - Applying Beer-Lambert absorption correctly
   - Generating proper ray directions from camera
   - Accumulating density along rays

2. **Rendering Parameters**: Current parameters may be suboptimal:
   - `densityScale = 1.0f`
   - `absorption = 0.5f`
   - `stepSize = 0.02f`
   - `maxSteps = 128`
   - `exposure = 5.0f`

3. **Density Field Content**: The `density_fill.hlsl` generates an animated torus+sphere pattern, but this may be:
   - Too sparse/thin to be visible with current ray marching settings
   - Positioned outside the sampling volume
   - Not generating sufficient contrast

4. **Camera/Ray Setup**: The camera-to-ray transformation in ray marching may be incorrect

## Recommended Next Steps for GPT-5

1. **Examine Ray Marching Shader**: Review `shaders/vol/ray_march_cs.hlsl` for:
   - Correct density texture sampling
   - Proper ray generation from camera parameters
   - Accurate Beer-Lambert light transport

2. **Parameter Tuning**: Experiment with more aggressive parameters:
   - Increase density scale (try 2.0-5.0)
   - Reduce absorption (try 0.1-0.3)
   - Increase step size for faster convergence
   - Boost exposure significantly

3. **Density Field Validation**: Consider simplifying the density field to a basic sphere or constant value to isolate ray marching issues

4. **Debug Visualization**: Implement a simple debug mode that visualizes:
   - Ray directions as colored pixels
   - Raw density samples as grayscale
   - Step counts as a heatmap

## Context Notes
- Project: PlasmaDX (DirectX 12 + DXR volumetric renderer)
- Current task: VOL_0003 compute ray marcher implementation
- User has confirmed camera movement is detected but no coherent volume structure visible
- This issue prevents progression to VOL_0004 (temporal accumulation) and subsequent features

The fundamental ray marching pipeline is functional (no crashes, proper resource states), but the visual output suggests either sampling or accumulation problems in the compute shader implementation.

## Directory Guardrail
**CRITICAL**: Claude Code sessions must always start in `/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX`, not PlasmaVulkan. PlasmaVulkan has been shelved.