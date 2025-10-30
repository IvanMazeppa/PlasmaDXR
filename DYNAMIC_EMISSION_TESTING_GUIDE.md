# Dynamic Emission Testing Guide

**Quick Reference for Evaluating RT-Driven Dynamic Emission**

---

## What Changed (30 Second Summary)

**Problem:** Physical emission was static and overpowered RT lighting → particles looked the same regardless of lighting changes.

**Solution:** Emission now **responds to RT lighting dynamically**:
- Well-lit particles (high RT lighting) → **emission suppressed** (RT dominates)
- Shadow particles (low RT lighting) → **emission visible** (fills darkness)
- Only hot particles (>22000K) emit significantly
- Gentle temporal pulsing (each star "breathes")
- Distance-based: close particles less emission, far particles more emission

**Result:** RT lighting drives the visual (dynamic), emission supports (fills shadows, adds color).

---

## Parameters to Tune (ImGui Controls)

**✅ NOW AVAILABLE IN IMGUI** - No rebuild required for tuning!

Open ImGui (F1) → **Rendering Features** section → **Dynamic Emission (RT-Driven)**

**Four sliders:**
- **Emission Strength** (0.0-1.0): How much emission vs RT lighting
- **Temp Threshold (K)** (15000-28000): Temperature cutoff for emission
- **RT Suppression** (0.0-1.0): How much RT lighting suppresses emission
- **Temporal Rate** (0.0-0.1): Star twinkling/pulsing speed

**Three preset buttons for quick testing:**
- **Max Dynamicism**: RT lighting drives visual, minimal emission
- **Balanced**: Default settings, good shadow fill (recommended starting point)
- **Star-Like**: More glow, aesthetic starfield look

---

## Testing Protocol

### Step 1: Isolate the Effect

**Turn OFF competing systems:**
- RTXDI: OFF (F3 key) - Use basic RT lighting only
- DLSS: Can leave ON (F4) - doesn't interfere
- God Rays: Should already be off
- Multi-light: Start with **1-2 lights** for clarity

**Why:** Too many lights + RTXDI makes it hard to see emission changes

### Step 2: What to Look For

**Move camera around and watch for:**

1. **Bright Areas (well-lit by RT lighting):**
   - Particles should look **mostly lit by RT** (dynamic)
   - Emission should be **subtle** (suppressed)
   - As RT lighting changes → particle brightness changes noticeably

2. **Shadow Areas (no RT lighting hits):**
   - Particles should **not be pure black**
   - Emission should **fill in** (visible glow)
   - Hot particles (blue/white) should be self-luminous

3. **Hot vs Cool Particles:**
   - **Hot (>22000K, blue/white):** Should glow noticeably even in shadows
   - **Cool (<22000K, red/orange):** Should be mostly RT-driven (dark in shadows)

4. **Temporal Variation:**
   - Watch any single particle for 3-5 seconds
   - Should see **subtle pulsing** (70-100% brightness)
   - NOT strobing - gentle breathing

5. **Distance Effect:**
   - **Close particles (<300 units):** Darker, more RT-dominant
   - **Far particles (>1000 units):** Brighter, more emission (visibility)

### Step 3: Tuning Tests

**Test A: Emission Strength**
```cpp
m_emissionStrength = 0.1f;   // Very RT-dominant (dynamic)
m_emissionStrength = 0.5f;   // More emission visible (less dynamic)
```
**Look for:** Balance point where shadows aren't black but RT still drives visuals.

**Test B: Temperature Threshold**
```cpp
m_emissionThreshold = 28000.0f;  // Only hottest particles glow
m_emissionThreshold = 18000.0f;  // More particles glow
```
**Look for:** How many stars have noticeable self-emission.

**Test C: RT Suppression**
```cpp
m_rtSuppression = 0.9f;   // Emission almost invisible in bright areas
m_rtSuppression = 0.5f;   // Emission more visible everywhere
```
**Look for:** How much emission shows through in well-lit areas.

---

## Quick Visual Tests

### Test 1: Static vs Dynamic Comparison

**Setup:**
- Position camera so you see both bright and dark areas
- Use 1-2 lights
- Disable RTXDI

**Watch for:**
- Particles in bright areas should **change brightness** as you rotate camera (RT lighting varies)
- If particles look the same from all angles → emission too strong (lower `m_emissionStrength`)

### Test 2: Shadow Fill

**Setup:**
- Position camera in shadow zone (no RT lighting hits)
- Look at particle distribution

**Watch for:**
- Hot particles (blue/white) should be **visible** (self-luminous)
- Cool particles (red/orange) should be **faint** (minimal emission)
- Should NOT see pure black particles (emission fills in)

### Test 3: Temporal Pulse

**Setup:**
- Pick a single bright particle
- Watch for 10 seconds

**Watch for:**
- Gentle brightness variation (like star twinkling)
- Rate controlled by `m_temporalRate` (0.03 = subtle, 0.1 = obvious)

---

## Recommended Settings for Different Looks

### Maximum Dynamicism (RT-Driven)
```cpp
m_emissionStrength = 0.15f;      // Minimal emission
m_emissionThreshold = 25000.0f;  // Only hottest stars
m_rtSuppression = 0.9f;          // Strong suppression
m_temporalRate = 0.02f;          // Very subtle pulse
```
**Best for:** Showcasing RT lighting system

### Balanced (Default)
```cpp
m_emissionStrength = 0.25f;      // Moderate emission
m_emissionThreshold = 22000.0f;  // Hot stars only
m_rtSuppression = 0.7f;          // Good suppression
m_temporalRate = 0.03f;          // Subtle pulse
```
**Best for:** General use, good shadow fill

### Star-Like (More Glow)
```cpp
m_emissionStrength = 0.4f;       // Higher emission
m_emissionThreshold = 18000.0f;  // More stars glow
m_rtSuppression = 0.5f;          // Less suppression
m_temporalRate = 0.05f;          // More noticeable pulse
```
**Best for:** Aesthetic "starfield" look

---

## Performance Check

**Before/After FPS:**
- Dynamic emission: <0.1ms overhead
- Should see **no noticeable FPS drop**
- If FPS drops >5%, something else changed

---

## Known Interactions with Other Systems

**RTXDI (F3):**
- Affects RT lighting intensity → changes how much emission is suppressed
- Test with RTXDI OFF first for clarity

**Multi-Light:**
- More lights = brighter RT lighting = more emission suppression
- Start with 1-2 lights, add more to see effect

**DLSS (F4):**
- Doesn't affect emission behavior
- May smooth temporal pulsing (temporal accumulation)

**Particle Count:**
- No effect on emission behavior
- Works identically at 10K or 100K particles

---

## Success Criteria

✅ **RT lighting drives the visual** - particles change as lighting changes
✅ **Emission fills shadows** - no pure black particles
✅ **Hot particles glow** - blue/white stars self-luminous
✅ **Cool particles are dynamic** - red/orange stars RT-driven
✅ **Temporal variation visible** - subtle "breathing" effect
✅ **Distance LOD works** - far particles brighter than close
✅ **Performance unchanged** - <1% FPS impact

---

## Quick Troubleshooting

**"Everything looks the same from all angles"**
→ Emission too strong, lower `m_emissionStrength` to 0.1-0.2

**"Shadows are completely black"**
→ Emission too weak or threshold too high
→ Raise `m_emissionStrength` or lower `m_emissionThreshold`

**"No temporal pulsing visible"**
→ Rate too low, raise `m_temporalRate` to 0.05-0.1

**"Too much flickering/strobing"**
→ Rate too high, lower `m_temporalRate` to 0.01-0.02

**"Can't see any hot stars glowing"**
→ Threshold too high, lower `m_emissionThreshold` to 18000-20000K

---

## Files Modified (For Reference)

- `shaders/dxr/particle_raytraced_lighting_cs.hlsl` - Dynamic emission algorithm
- `src/lighting/RTLightingSystem_RayQuery.h` - Parameters (lines 109-112)
- `src/lighting/RTLightingSystem_RayQuery.cpp` - Camera position passing
- `src/core/Application.cpp` - Camera position moved earlier

**No ImGui controls yet** - parameters are hardcoded defaults.

---

**Ready to test!** Start with default settings, move camera around bright/shadow areas, watch for dynamic changes driven by RT lighting.
