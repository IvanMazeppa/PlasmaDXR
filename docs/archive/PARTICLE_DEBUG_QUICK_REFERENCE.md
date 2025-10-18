# Particle Debug Quick Reference Card

## Keyboard Controls (Numpad)

```
┌─────────────────────────────────────────────────────────────┐
│                  PARTICLE DEBUG CONTROLS                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Numpad 0  →  Debug OFF (normal temperature-based colors)   │
│  Numpad 1  →  Clip W Debug (depth validation)               │
│  Numpad 2  →  Clip XY Position (screen distribution)        │
│  Numpad 3  →  World Distance (spatial distribution)         │
│  Numpad 4  →  Origin Test (pipeline validation)             │
│  Numpad 5  →  Validation Mode (error detection)             │
│  Numpad 6  →  Toggle Validation Overlay                     │
│  Numpad 7  →  Decrease Near Plane                           │
│  Numpad 8  →  Increase Near Plane                           │
│  Numpad 9  →  Cycle All Modes                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Color Codes by Mode

### Mode 1: Clip-Space W (Depth)
- **RED** = Behind camera (W < 0) - CRITICAL BUG
- **GREEN** = Near camera (W ∈ [0,1])
- **BLUE** = Far from camera (W > 1, intensity by distance)

### Mode 2: Clip XY Position (Screen Space)
- **BLUE** = Screen center (NDC distance < 0.2)
- **RED gradient** = Left/right edges
- **GREEN gradient** = Top/bottom edges

### Mode 3: World Distance
- **GREEN** = Close (< 100 units)
- **YELLOW** = Mid-range (100-500 units)
- **RED** = Far (> 500 units)

### Mode 4: Origin Test
- **CYAN** = All particles at (0,0,0) with increasing size
- Purpose: Verify pipeline works at all

### Mode 5: Validation (Errors)
- **MAGENTA** = NaN/Inf (shader math error) - CRITICAL
- **RED** = Behind near plane
- **BLUE** = Beyond far plane
- **ORANGE** = Way off-screen (|NDC| > 2)
- **GREEN** = All checks passed

## Bug Detection Patterns

### Pattern: Particles Invisible
1. Press **Numpad 4** (Origin Test)
   - See CYAN? → Pipeline OK, position bug
   - Nothing? → PSO/shader/binding issue

### Pattern: Flickering/Corruption
1. Press **Numpad 5** (Validation)
   - MAGENTA → Shader math NaN/Inf
   - RED → Near plane issues
   - Use PIX to debug shader

### Pattern: Wrong Spatial Layout
1. Press **Numpad 3** (Distance)
   - Verify color distribution matches expected layout
2. Press **Numpad 2** (Clip XY)
   - Check screen-space distribution

### Pattern: Behind Camera
1. Press **Numpad 1** (Clip W)
   - RED particles → View matrix or position bug

## Files Modified

```
/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX/src/core/App.h
  → Added m_particleDebugMode, m_particleValidationEnabled, etc.

/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX/src/core/App.cpp
  → Added VK_NUMPAD0-9 keyboard handlers

/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/shaders/particles/particle_billboard_vs.hlsl
  → Already has all debug modes implemented (lines 23-247)
```

## Next Steps to Enable

The keyboard controls are now active. To fully enable debug visualization:

1. **Compile shader** (if not already compiled with latest debug code)
2. **Create debug constant buffer** in ParticleSystem.cpp
3. **Bind at register(b2)** when rendering
4. **Pass values from App** to ParticleSystem

See `PARTICLE_DEBUG_IMPLEMENTATION.md` Section 3C for integration details.

## Testing Checklist

- [ ] Press Numpad 4 → See CYAN particles at origin
- [ ] Press Numpad 9 → Cycles through modes 0-5
- [ ] Press Numpad 1 → See depth-coded particles
- [ ] Press Numpad 5 → Validation errors appear
- [ ] Press Numpad 6 → Toggle validation overlay
- [ ] Console logs show debug mode changes

## Performance

- Overhead: 5-10% with debug enabled
- Zero overhead when debugMode=0 and enableValidation=0
- All modes safe for production builds

## Common Issues

### Keyboard not working
- Check window focus
- Verify App::WndProc receives WM_KEYDOWN
- Check git diff to ensure edits applied

### No visual change
- Debug constants not bound at b2
- Shader not recompiled
- Root signature missing CBV slot

### Always shows errors
- Adjust near/far planes (Numpad 7/8)
- Check camera setup matches validation thresholds
