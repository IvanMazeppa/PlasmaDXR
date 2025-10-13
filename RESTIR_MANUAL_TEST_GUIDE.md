# ReSTIR Manual Testing Guide

**Purpose:** Verify the ReSTIR brightness fix works correctly
**Date:** October 12, 2025
**Build Required:** Debug build (PlasmaDX-Clean.exe)

---

## Prerequisites

1. **Build is up to date** with latest shader fix
2. **No errors in compilation**
3. **Working directory** is project root (`D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean`)

---

## Step-by-Step Testing

### Step 1: Launch the Application

```bash
# From project root in WSL
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean

# Launch Debug build
./build/Debug/PlasmaDX-Clean.exe
```

**Or from Windows:**
```cmd
cd D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean
build\Debug\PlasmaDX-Clean.exe
```

**Expected:** Application window opens showing accretion disk

---

### Step 2: Verify Baseline (ReSTIR OFF)

At startup, ReSTIR is **disabled** by default.

**What to observe:**
- Accretion disk with bright white/yellow core
- Orange/red particles in disk
- Smooth, continuous rendering
- Corner indicator shows rendering info

**Test movement:**
- Press **W** to fly toward particles
- Press **S** to fly backward
- Use **CTRL+Mouse** to look around

---

### Step 3: Enable ReSTIR

Press **F7** key

**Expected console output:**
```
ReSTIR: ON (temporal resampling for 10-60x faster convergence)
```

**What should happen:**
- **NO visible change** in brightness or colors
- Scene should look identical to ReSTIR OFF
- Possibly slightly better quality (less noise)

---

### Step 4: Test Distance Behavior

With ReSTIR enabled (F7), fly toward particles:

1. **Far distance (Red indicator)**
   - Press **W** to approach
   - **Expected:** Normal rendering, M=0-1 samples

2. **Medium distance (Orange indicator)**
   - Continue pressing **W**
   - **Expected:** M=2-8 samples, **NO dots**, colors unchanged

3. **Close distance (Yellow indicator)**
   - Keep approaching
   - **Expected:** M=9-16 samples, **NO dots**, **NO brown/muted colors**

4. **Very close (Green indicator)**
   - Fly right into particle cloud
   - **Expected:** M=16-24 samples, smooth bright particles

---

### Step 5: Test RT Lighting Controls

With ReSTIR enabled:

1. **Press I** (Increase RT intensity)
   - **Expected:** Particles get brighter
   - Console shows: `RT Lighting Intensity: 2.0` (doubles each press)

2. **Press K** (Decrease RT intensity)
   - **Expected:** Particles get dimmer
   - Console shows: `RT Lighting Intensity: 0.5` (halves each press)

3. **Press F7** (Toggle ReSTIR OFF)
   - **Expected:** Brightness should stay similar
   - **Success:** ReSTIR ON ≈ ReSTIR OFF brightness

---

### Step 6: Compare ReSTIR ON vs OFF

1. Fly to **medium distance** (orange indicator)
2. Press **F7** to toggle ReSTIR **OFF**
   - Take mental note of scene brightness
3. Press **F7** to toggle ReSTIR **ON**
   - **Success:** Scene looks the same brightness
   - **Failure:** Scene gets much darker/brighter or changes color

---

### Step 7: Test Temporal Weight Adjustment

With ReSTIR enabled:

1. **Press CTRL+F7** (Increase temporal weight)
   - Console shows: `ReSTIR Temporal Weight: 1.0` (increases by 0.1)
   - **Expected:** More temporal reuse, smoother convergence

2. **Press SHIFT+F7** (Decrease temporal weight)
   - Console shows: `ReSTIR Temporal Weight: 0.8` (decreases by 0.1)
   - **Expected:** Less temporal reuse, faster adaptation to changes

---

## Success Criteria

✅ **PASS if ALL of the following are true:**

1. ReSTIR ON/OFF look similar in brightness
2. No "dots" visible at any distance
3. No brown/muted color shift when getting close
4. Corner indicator transitions smoothly (Red → Orange → Yellow → Green)
5. I/K keys adjust brightness correctly
6. No visual artifacts or flickering

❌ **FAIL if ANY of these occur:**

1. Scene gets much darker when enabling ReSTIR
2. Thousands of dots appear when approaching particles
3. Colors shift to brown/muted when close
4. Brightness controls (I/K) stop working
5. Visual artifacts or corruption

---

## Troubleshooting

### Error: "Cannot find shader file"

**Solution:**
- Ensure working directory is project root
- Check `shaders/particles/particle_gaussian_raytrace.dxil` exists
- Rebuild shaders: `/mnt/c/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/dxc.exe -T cs_6_5 -E main "shaders/particles/particle_gaussian_raytrace.hlsl" -Fo "shaders/particles/particle_gaussian_raytrace.dxil"`

### Error: "D3D12 device creation failed"

**Solution:**
- Update GPU drivers
- Check if another D3D12 app is running
- Try Debug build instead of DebugPIX build

### App crashes on startup

**Solution:**
- Check logs in `logs/` folder
- Ensure no PIX environment variables set: `unset PIX_AUTO_CAPTURE PIX_CAPTURE_FRAME`
- Try reducing particle count in config.json

### ReSTIR not toggling (F7 does nothing)

**Solution:**
- Ensure application window has focus (click on it)
- Try holding F7 briefly (don't tap too quickly)
- Check console output for "ReSTIR: ON" message

### Scene still shows dots/artifacts

**Solution:**
- Verify shader was recompiled after fix
- Check shader file timestamp: `ls -l shaders/particles/particle_gaussian_raytrace.dxil`
- If old, recompile shader (see "Cannot find shader file" above)
- Rebuild entire project: `MSBuild PlasmaDX-Clean.sln /t:Rebuild`

---

## Logging Test Results

When reporting results, please include:

1. **Visual behavior:**
   - Does brightness match between ReSTIR ON/OFF?
   - Any dots visible at any distance?
   - Any color shift when getting close?

2. **Console output:**
   - Copy the "ReSTIR: ON/OFF" messages
   - Copy any RT intensity changes (I/K keys)
   - Any warning or error messages

3. **Distance indicators:**
   - What color indicators did you see?
   - At what color did issues occur (if any)?

4. **PIX data (if available):**
   - M and W values at different distances
   - Corner indicator colors
   - Screenshot of problem area

---

## Expected Log Output (Success Case)

```
[INFO] Application initialized successfully
[INFO] === Camera Configuration ===
[INFO]   Position: (0.000000, 1200.000000, 800.000000)
[INFO]   Looking at: (0, 0, 0)
[INFO] === Gaussian RT Controls ===
[INFO]   F5: Shadow Rays [ON]
[INFO]   F6: In-Scattering [OFF]
[INFO]   F7: Phase Function [ON]
[INFO] === DEBUG: Gaussian Constants ===
[INFO]   useShadowRays: 1
[INFO]   useInScattering: 0
[INFO]   usePhaseFunction: 1
[INFO]   useReSTIR: 0
[INFO]   restirInitialCandidates: 16

(User presses F7)
[INFO] ReSTIR: ON (temporal resampling for 10-60x faster convergence)

(User presses I)
[INFO] RT Lighting Intensity: 2.0

(User presses K)
[INFO] RT Lighting Intensity: 1.0

(User presses F7)
[INFO] ReSTIR: OFF
```

---

## Quick Test Script (WSL)

```bash
#!/bin/bash
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean

echo "Building..."
"/mnt/c/Program Files/Microsoft Visual Studio/2022/Community/MSBuild/Current/Bin/MSBuild.exe" \
  "PlasmaDX-Clean.sln" /p:Configuration=Debug /p:Platform=x64 /nologo /v:q

if [ $? -eq 0 ]; then
    echo "Build successful! Launching..."
    ./build/Debug/PlasmaDX-Clean.exe
else
    echo "Build failed. Check errors above."
fi
```

Save as `test_restir.sh`, then:
```bash
chmod +x test_restir.sh
./test_restir.sh
```

---

**Document Version:** 1.0
**Last Updated:** October 12, 2025
**Related:** RESTIR_BRIGHTNESS_FIX_20251012.md
