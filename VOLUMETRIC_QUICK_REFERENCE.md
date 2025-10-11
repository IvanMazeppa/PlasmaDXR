# Volumetric Rendering Quick Reference

## Shader Files

- **Original:** `shaders/particles/particle_gaussian_raytrace.hlsl`
- **Enhanced:** `shaders/particles/particle_gaussian_raytrace_volumetric_enhanced.hlsl`

## Key Parameters to Adjust

### In Shader Constants (Must Rebuild)

```hlsl
// Light Configuration (lines 259-262)
volParams.lightPos = float3(0, 500, 200);  // Adjust Y for height, Z for forward/back
volParams.lightColor = float3(10, 10, 10); // Brightness multiplier
volParams.scatteringG = 0.7;               // -1 to 1 (-1=backscatter, 0=isotropic, 1=forward)
volParams.extinction = 2.0;                // Higher = denser shadows

// Density and Steps (lines 38-40)
static const float volumeStepSize = 0.5;   // Smaller = quality, larger = performance
static const float densityMultiplier = 5.0; // Overall volume opacity
static const float shadowBias = 0.1;        // Prevent self-shadowing artifacts

// In-Scattering (lines 101-102)
const uint numSamples = 12;                 // More = smoother halos
const float scatterRange = 150.0;           // Glow reach distance
```

### Runtime Controls (via Constants Buffer)

```cpp
// Enable/Disable Features
useShadowRays = 1;      // Toggle shadows (0/1)
useInScattering = 1;    // Toggle halos/glow (0/1)
usePhaseFunction = 1;   // Toggle forward scatter (0/1)

// Strength Controls
phaseStrength = 2.0;    // 0.5 - 5.0 (forward scatter intensity)
emissionStrength = 1.0; // 0.0 - 2.0 (particle brightness)
```

## Optimal Camera Positions for Effects

### To See Shadows
- Position: Behind particles, looking toward light
- Example: Camera at (0, 1200, -800), Light at (0, 500, 200)

### To See Forward Scattering
- Position: Looking through particles toward light
- Example: Camera at (0, 100, -1000), looking at origin

### To See Halos/Glow
- Position: Particles between camera and light, slightly off-axis
- Example: Camera at (500, 800, 800), looking at origin

## Performance Tuning

### High Quality (30-45 FPS)
```hlsl
volumeStepSize = 0.5;
numSamples = 12;
maxHits = 8;
```

### Balanced (45-60 FPS)
```hlsl
volumeStepSize = 1.0;
numSamples = 8;
maxHits = 6;
```

### Performance (60+ FPS)
```hlsl
volumeStepSize = 2.0;
numSamples = 4;
maxHits = 4;
```

## Troubleshooting

### No Shadows Visible\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
1. Check `useShadowRays = 1`
2. Verify light is outside particle volume
3. Increase `volParams.extinction` to 3.0+
4. Check debug indicator (red bar top-left)

### No Halos/Glow
1. Check `useInScattering = 1`
2. Increase `scatterRange` to 200+
3. Increase in-scatter multiplier (line 213: `* 3.0`)
4. Check debug indicator (green bar top-right)

### Too Dark
1. Increase `volParams.lightColor` to (15, 15, 15)
2. Increase `exposure` to 2.0
3. Increase `volParams.ambientLevel` to 0.2

### Too Bright/Washed Out
1. Decrease `densityMultiplier` to 3.0
2. Decrease `exposure` to 1.0
3. Adjust tone mapping white point (line 386)

## Visual Debug Mode

Debug bars appear when features are active:
```
[RED BAR    ] = Shadow Rays ON     (Top-left)
            [GREEN BAR] = In-Scatter ON (Top-right)
[BLUE BAR   ] = Phase Func ON      (Bottom-left)
```

## Expected Ray Counts

With 1920x1080 resolution and typical particle count:
- Shadow rays: ~500M-1B per frame
- In-scatter rays: ~300M-600M per frame
- Total: 1.8B+ rays with all features

This is EXPECTED and indicates the system is working correctly.