# PlasmaDX-Clean Development Roadmap
## Phase 3: Splash Screen, Animation System, and Enhanced Physics

**Document Version:** 1.0
**Created:** 2025-10-15
**Status:** Planning Phase

---

## Executive Summary

This roadmap outlines three major feature additions to PlasmaDX-Clean:

1. **Splash Screen System** - A comprehensive preset manager and configuration interface
2. **Animation System** - Frame-by-frame rendering with PNG sequence export and FFmpeg video encoding
3. **Enhanced Physics Engine** - Port advanced features from the PlasmaVulkan project for more realistic accretion disk simulation

**Goal:** Create a professional tool for creating, saving, and rendering high-quality accretion disk simulations with customizable physics parameters.

---

## Table of Contents

1. [Phase 3.1: Enhanced Physics Engine](#phase-31-enhanced-physics-engine)
2. [Phase 3.2: Splash Screen & Preset System](#phase-32-splash-screen--preset-system)
3. [Phase 3.3: Animation & Recording System](#phase-33-animation--recording-system)
4. [Phase 3.4: Integration & Polish](#phase-34-integration--polish)
5. [Technical Architecture](#technical-architecture)
6. [Implementation Timeline](#implementation-timeline)
7. [Risk Assessment](#risk-assessment)

---

## Phase 3.1: Enhanced Physics Engine

**Priority:** HIGH - Foundation for presets and animation
**Estimated Time:** 2-3 days
**Dependencies:** None (can start immediately)

### 3.1.1 Port Physics Parameters from PlasmaVulkan

**Current DX12 Physics (Limited):**
- Gravity strength
- Angular momentum boost
- Turbulence strength
- Damping factor
- Fixed inner/outer radius, disk thickness

**PlasmaVulkan Physics (Rich Feature Set):**
- ✅ Gravity strength + gravity center (runtime movable)
- ✅ Turbulence strength
- ✅ Damping factor
- ✅ Angular momentum boost
- ⭐ **Constraint shapes:** NONE, SPHERE, DISC, TORUS, ACCRETION_DISK, GALAXY_COLLISION
- ⭐ **Runtime adjustable radius and thickness**
- ⭐ **Black hole mass** (affects orbital velocity)
- ⭐ **Alpha viscosity** (Shakura-Sunyaev α parameter for realistic accretion)
- ⭐ **Dual galaxy collision mode** (two gravitational centers)
- ⭐ **SPH (Smoothed Particle Hydrodynamics) mode** for fluid simulation

### 3.1.2 Physics Implementation Plan

#### Step 1: Expand ParticleSystem Constants Structure (src/particles/ParticleSystem.h)

```cpp
// NEW: Extended physics parameters
struct PhysicsConstants {
    // Existing parameters
    float gravityStrength = 500.0f;
    float turbulenceStrength = 15.0f;
    float dampingFactor = 0.99f;
    float angularMomentumBoost = 1.0f;

    // NEW: Constraint system
    uint32_t constraintShape = 4;  // 0=NONE, 1=SPHERE, 2=DISC, 3=TORUS, 4=ACCRETION_DISK, 5=GALAXY_COLLISION
    float constraintRadius = 300.0f;        // Runtime adjustable outer radius
    float constraintThickness = 50.0f;      // Runtime adjustable disk thickness
    float innerStableOrbit = 10.0f;         // Runtime adjustable inner radius

    // NEW: Black hole physics
    float blackHoleMass = 4.3e6f;           // In solar masses (default: Sgr A*)
    float alphaViscosity = 0.1f;            // Shakura-Sunyaev α (0.01-0.4 realistic range)

    // NEW: Gravity center (for galaxy collision mode)
    DirectX::XMFLOAT3 gravityCenter = {0.0f, 0.0f, 0.0f};

    // NEW: Dual galaxy collision mode
    uint32_t dualGalaxyMode = 0;            // 0=single, 1=dual galaxy
    DirectX::XMFLOAT3 gravityCenter2 = {500.0f, 0.0f, 0.0f};
    float blackHoleMass2 = 4.3e6f;

    // NEW: SPH parameters (future expansion)
    float smoothingRadius = 0.5f;
    float restDensity = 1000.0f;
    float pressureConstant = 200.0f;
    float sphViscosity = 0.01f;
    float particleMass = 0.02f;
};
```

#### Step 2: Update Physics Compute Shader (shaders/particles/particle_physics.hlsl)

Add support for:
- Constraint shape switching (if/else or switch statement)
- Runtime radius/thickness adjustment
- Black hole mass affecting Keplerian velocity calculation
- Alpha viscosity for radial accretion (particles slowly spiral inward)
- Dual gravity centers for galaxy collision mode

#### Step 3: Create Physics Preset Presets

Define preset configurations:
- **"Sgr A* (Realistic)"** - Milky Way supermassive black hole
- **"Stellar Mass Black Hole"** - 10 solar masses, rapid accretion
- **"Quasar"** - Massive (1 billion solar masses), extreme luminosity
- **"Galaxy Collision"** - Two black holes approaching
- **"Protoplanetary Disk"** - Low mass, thick disk, low viscosity
- **"Custom"** - User-defined parameters

### 3.1.3 Deliverables

- [ ] Extended `PhysicsConstants` structure in ParticleSystem.h
- [ ] Updated physics compute shader with new features
- [ ] Getters/setters for all new physics parameters
- [ ] Physics preset definitions (C++ constants or JSON file)
- [ ] ImGui controls for new physics parameters
- [ ] Documentation of physics equations and parameters

### 3.1.4 Testing

- [ ] Verify constraint shapes work correctly (sphere, disc, torus, etc.)
- [ ] Test dual galaxy collision mode (two black holes)
- [ ] Validate Keplerian velocity calculation with black hole mass
- [ ] Check alpha viscosity causes gradual inward spiral
- [ ] Ensure runtime parameter changes don't cause crashes

---

## Phase 3.2: Splash Screen & Preset System

**Priority:** MEDIUM - Enhances user experience
**Estimated Time:** 3-4 days
**Dependencies:** Phase 3.1 (physics parameters must be defined)

### 3.2.1 Splash Screen UI Design

**Layout:**
```
+--------------------------------------------------+
|           PlasmaDX-Clean Accretion Disk          |
|              Visualization Studio                |
+--------------------------------------------------+
|                                                  |
|  [Quick Start Presets]                           |
|  +--------------------------------------------+  |
|  | [Sgr A* (Realistic)]  [Quasar]            |  |
|  | [Stellar BH]          [Galaxy Collision]  |  |
|  | [Protoplanetary]      [Custom...]         |  |
|  +--------------------------------------------+  |
|                                                  |
|  [Saved Presets]                                 |
|  +--------------------------------------------+  |
|  | > My Accretion Disk 1      [Load] [Delete]|  |
|  | > High Turbulence Test     [Load] [Delete]|  |
|  | > Collision Animation      [Load] [Delete]|  |
|  +--------------------------------------------+  |
|                                                  |
|  [Custom Configuration]                          |
|  +--------------------------------------------+  |
|  | Particle Count: [20000        ] (slider)  |  |
|  | Resolution:     [1920x1080   v] (dropdown)|  |
|  | Renderer:       [Gaussian    v] (dropdown)|  |
|  |                                            |  |
|  | Physics Parameters:                        |  |
|  | - Inner Radius:     [10.0  ]              |  |
|  | - Outer Radius:     [300.0 ]              |  |
|  | - Disk Thickness:   [50.0  ]              |  |
|  | - Black Hole Mass:  [4.3e6 ] solar masses |  |
|  | - Alpha Viscosity:  [0.1   ]              |  |
|  | - Gravity Strength: [500.0 ]              |  |
|  | - Turbulence:       [15.0  ]              |  |
|  | ... (more parameters)                      |  |
|  |                                            |  |
|  | [Save As Preset...]                        |  |
|  +--------------------------------------------+  |
|                                                  |
|  [Start Simulation]  [Record Animation]          |
+--------------------------------------------------+
```

### 3.2.2 Implementation Architecture

#### Core Components:

**1. SplashScreen Class (src/ui/SplashScreen.h/cpp)**
```cpp
class SplashScreen {
public:
    enum class UserChoice {
        None,
        LoadPreset,
        StartSimulation,
        RecordAnimation,
        Exit
    };

    struct PresetConfig {
        std::string name;
        std::string description;
        Config renderConfig;      // Particle count, resolution, renderer
        PhysicsConstants physics; // All physics parameters
        CameraConfig camera;      // Starting camera position
    };

    bool Initialize(Device* device, HWND hwnd);
    void Shutdown();
    UserChoice Show();  // Blocks until user makes choice

    PresetConfig GetSelectedPreset() const;
    bool IsRecordingMode() const { return m_recordingMode; }

private:
    void RenderUI();
    void LoadPresetsFromDisk();
    void SavePreset(const PresetConfig& preset);
    void DeletePreset(const std::string& name);

    std::vector<PresetConfig> m_builtInPresets;
    std::vector<PresetConfig> m_savedPresets;
    PresetConfig m_currentConfig;
    bool m_recordingMode = false;
};
```

**2. Preset File Format (JSON)**

Location: `presets/` directory

```json
{
  "name": "Sgr A* Realistic",
  "description": "Milky Way supermassive black hole with realistic parameters",
  "version": "1.0",
  "rendering": {
    "particleCount": 50000,
    "rendererType": "gaussian",
    "width": 1920,
    "height": 1080
  },
  "physics": {
    "constraintShape": 4,
    "innerRadius": 10.0,
    "outerRadius": 300.0,
    "diskThickness": 50.0,
    "blackHoleMass": 4300000.0,
    "alphaViscosity": 0.1,
    "gravityStrength": 500.0,
    "turbulenceStrength": 15.0,
    "dampingFactor": 0.99,
    "angularMomentumBoost": 1.0,
    "dualGalaxyMode": false
  },
  "camera": {
    "startDistance": 800.0,
    "startHeight": 1200.0,
    "startAngle": 0.0,
    "startPitch": 0.0,
    "moveSpeed": 100.0,
    "rotateSpeed": 0.5,
    "particleSize": 50.0
  },
  "features": {
    "enableReSTIR": false,
    "enableInScattering": false,
    "enableShadowRays": true,
    "enablePhaseFunction": true,
    "useAnisotropicGaussians": true,
    "rtLightingStrength": 2.0
  }
}
```

**3. Built-in Presets**

Create 5-6 scientifically accurate presets:
- **Sgr A* (Realistic)** - Default preset
- **Stellar Mass Black Hole** - 10 solar masses, rapid rotation
- **Quasar** - 1 billion solar masses, high luminosity
- **Galaxy Collision** - Two approaching black holes
- **Protoplanetary Disk** - 0.1 solar masses, thick, low viscosity
- **Custom** - Empty template

### 3.2.3 Integration with Application

Modify `main.cpp`:
```cpp
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow) {
    // Show splash screen first
    SplashScreen splash;
    splash.Initialize(nullptr, nullptr);  // Minimal init

    auto choice = splash.Show();  // Blocks until user chooses

    if (choice == SplashScreen::UserChoice::Exit) {
        return 0;
    }

    // Get user's configuration
    auto preset = splash.GetSelectedPreset();
    bool recordingMode = splash.IsRecordingMode();

    splash.Shutdown();

    // Initialize application with chosen preset
    Application app;
    if (!app.Initialize(hInstance, preset, recordingMode)) {
        return -1;
    }

    return app.Run();
}
```

### 3.2.4 Deliverables

- [ ] SplashScreen class with ImGui-based UI
- [ ] Preset JSON file format specification
- [ ] 5-6 built-in physics presets
- [ ] Preset save/load system
- [ ] Preset manager (view, load, delete saved presets)
- [ ] Custom configuration editor
- [ ] Integration with Application initialization
- [ ] "Record Animation" mode toggle

### 3.2.5 Testing

- [ ] All built-in presets load correctly
- [ ] Save preset creates valid JSON file
- [ ] Load preset restores all parameters
- [ ] Delete preset removes file from disk
- [ ] Custom configuration validates input ranges
- [ ] Recording mode flag passes correctly to Application

---

## Phase 3.3: Animation & Recording System

**Priority:** MEDIUM-HIGH - Enables offline rendering
**Estimated Time:** 3-4 days
**Dependencies:** Phase 3.1 (physics must be stable)

### 3.3.1 Recording System Architecture

**Design Philosophy:**
- Decouple rendering from real-time (like Vulkan version)
- Render at arbitrary FPS (even 1 FPS) without affecting physics
- Save each frame as PNG
- Use FFmpeg to encode video at target framerate

#### Recording Configuration

```cpp
struct RecordingConfig {
    std::string outputDir = "recordings/";
    uint32_t targetFrameCount = 1800;  // 2 minutes at 15 FPS
    uint32_t targetFPS = 15;           // Final video framerate

    // Quality settings
    uint32_t renderWidth = 1920;
    uint32_t renderHeight = 1080;
    bool enableAllEffects = true;      // Force high quality

    // Camera animation
    enum class CameraMode {
        Static,       // Fixed camera
        Orbit,        // Orbit around disk
        Flythrough,   // Scripted path
        Custom        // User-defined keyframes
    };
    CameraMode cameraMode = CameraMode::Orbit;
    float orbitSpeed = 0.1f;  // Radians per second

    // FFmpeg settings
    std::string videoCodec = "libx264";  // H.264
    std::string videoQuality = "23";     // CRF (0-51, lower = better)
    std::string videoPreset = "slow";    // Encoding speed/quality tradeoff
};
```

### 3.3.2 Frame Capture Implementation

#### Step 1: Add Frame Readback to SwapChain

```cpp
// SwapChain.h
class SwapChain {
public:
    // NEW: Save current backbuffer as PNG
    bool SaveFrameToPNG(const std::string& filename);

private:
    // NEW: Staging buffer for GPU->CPU readback
    Microsoft::WRL::ComPtr<ID3D12Resource> m_readbackBuffer;
};
```

#### Step 2: Create AnimationRecorder Class

```cpp
// src/animation/AnimationRecorder.h
class AnimationRecorder {
public:
    bool Initialize(Device* device, const RecordingConfig& config);
    void Shutdown();

    bool StartRecording();
    void StopRecording();
    bool IsRecording() const { return m_isRecording; }

    // Call after rendering each frame
    bool CaptureFrame(SwapChain* swapChain);

    // Call when all frames captured
    bool EncodeVideo();

    // Progress info
    uint32_t GetCurrentFrame() const { return m_currentFrame; }
    uint32_t GetTotalFrames() const { return m_config.targetFrameCount; }
    float GetProgress() const;

    // Camera animation
    void UpdateCamera(Camera* camera, float simulationTime);

private:
    bool SaveFrameAsPNG(const std::string& filename,
                       const std::vector<uint8_t>& pixelData,
                       uint32_t width, uint32_t height);
    bool RunFFmpegEncode();

    RecordingConfig m_config;
    bool m_isRecording = false;
    uint32_t m_currentFrame = 0;
    std::string m_sessionDir;  // "recordings/session_YYYYMMDD_HHMMSS/"
};
```

### 3.3.3 FFmpeg Integration

**Command Template (from PlasmaVulkan):**
```cpp
std::string ffmpegCmd =
    "ffmpeg -y "                           // Overwrite output
    "-framerate " + std::to_string(fps) +  // Input framerate
    "-i \"" + frameDir + "/frame_%06d.png\" "  // Input frames (000000, 000001, ...)
    "-c:v libx264 "                        // H.264 codec
    "-crf 23 "                             // Quality (lower = better, 18-28 is good)
    "-preset slow "                        // Encoding speed/quality
    "-pix_fmt yuv420p "                    // Compatibility with players
    "\"" + outputVideo + "\"";             // Output file

std::system(ffmpegCmd.c_str());
```

**Alternative: Use file list (more reliable)**
```cpp
// Create file list
std::ofstream fileList(sessionDir + "/filelist.txt");
for (uint32_t i = 0; i < frameCount; i++) {
    fileList << "file 'frame_" << std::setw(6) << std::setfill('0') << i << ".png'\n";
}
fileList.close();

// FFmpeg with file list
std::string ffmpegCmd =
    "ffmpeg -y -f concat -safe 0 -r " + std::to_string(fps) +
    " -i \"" + sessionDir + "/filelist.txt\" "
    "-c:v libx264 -crf 23 -preset slow -pix_fmt yuv420p "
    "\"" + outputVideo + "\"";
```

### 3.3.4 Camera Animation System

**Orbit Mode (Simple):**
```cpp
void AnimationRecorder::UpdateCamera(Camera* camera, float time) {
    if (m_config.cameraMode == CameraMode::Orbit) {
        float angle = time * m_config.orbitSpeed;
        float radius = 800.0f;
        float height = 1200.0f;

        camera->SetPosition(
            radius * cos(angle),
            height,
            radius * sin(angle)
        );
        camera->LookAt(0.0f, 0.0f, 0.0f);
    }
    // TODO: Add Flythrough and Custom keyframe modes
}
```

### 3.3.5 Recording UI (ImGui Overlay During Recording)

```
+----------------------------------+
|  RECORDING IN PROGRESS           |
+----------------------------------+
|  Frame: 120 / 1800 (6.7%)        |
|  [████░░░░░░░░░░░░░] 6%          |
|                                  |
|  Elapsed: 2m 15s                 |
|  Remaining: ~31m 45s             |
|                                  |
|  Output: recordings/session_001/ |
|                                  |
|  [Pause]  [Cancel]               |
+----------------------------------+
```

### 3.3.6 Integration with Application

Modify `Application.cpp`:
```cpp
void Application::Run() {
    if (m_recordingMode) {
        RunRecordingMode();
    } else {
        RunRealtimeMode();
    }
}

void Application::RunRecordingMode() {
    LOG_INFO("Starting recording mode...");

    m_animationRecorder->StartRecording();

    while (m_animationRecorder->IsRecording()) {
        // Fixed timestep (not real-time)
        float deltaTime = 1.0f / 120.0f;  // 120Hz physics regardless of render time

        // Update physics
        m_particleSystem->Update(deltaTime, m_totalTime);
        m_totalTime += deltaTime;

        // Update camera animation
        m_animationRecorder->UpdateCamera(&m_camera, m_totalTime);

        // Render frame (don't care about FPS)
        Render();

        // Capture frame to PNG
        if (!m_animationRecorder->CaptureFrame(m_swapChain.get())) {
            LOG_ERROR("Failed to capture frame!");
            break;
        }

        // Show progress in ImGui
        RenderRecordingUI();

        // Check for user cancel
        if (GetAsyncKeyState(VK_ESCAPE) & 0x8000) {
            LOG_INFO("Recording cancelled by user");
            break;
        }
    }

    LOG_INFO("Encoding video with FFmpeg...");
    if (m_animationRecorder->EncodeVideo()) {
        LOG_INFO("Video created successfully!");
    }

    m_isRunning = false;  // Exit after recording
}
```

### 3.3.7 Deliverables

- [ ] SwapChain::SaveFrameToPNG() implementation
- [ ] AnimationRecorder class with frame capture
- [ ] FFmpeg integration with error handling
- [ ] PNG sequence saving (stb_image_write)
- [ ] Recording progress UI
- [ ] Camera animation system (at least Orbit mode)
- [ ] Application recording mode (separate from realtime)
- [ ] Session directory management (timestamps, cleanup)
- [ ] Video encoding with quality presets

### 3.3.8 Testing

- [ ] Record 10-second test video (150 frames at 15 FPS)
- [ ] Verify PNG frames are saved correctly
- [ ] Verify FFmpeg creates valid MP4/AVI file
- [ ] Test cancelling recording mid-way (cleanup)
- [ ] Test camera orbit animation
- [ ] Test high-quality settings (4K, 60 FPS)
- [ ] Verify physics runs deterministically (same result each time)

---

## Phase 3.4: Integration & Polish

**Priority:** MEDIUM - Quality of life
**Estimated Time:** 2-3 days
**Dependencies:** All previous phases

### 3.4.1 Polish Items

- [ ] Add tooltips to all ImGui controls explaining physics parameters
- [ ] Add "About" section with physics equation explanations
- [ ] Improve error messages (user-friendly)
- [ ] Add confirmation dialogs for destructive actions (delete preset)
- [ ] Create user documentation (README_USER.md)
- [ ] Add sample presets with screenshots
- [ ] Create demo videos showing features
- [ ] Add FFmpeg installation check on startup
- [ ] Auto-detect FFmpeg path (Windows, Linux, macOS)

### 3.4.2 Performance Optimizations

- [ ] Add "Performance Mode" preset (low quality, high FPS)
- [ ] Add particle count presets (10K, 50K, 100K, 200K)
- [ ] Optimize recording mode (disable ImGui during recording)
- [ ] Add GPU memory usage display
- [ ] Add frame timing graph

### 3.4.3 Configuration Management

- [ ] Move all hardcoded values to config files
- [ ] Add `presets/built-in/` and `presets/user/` directories
- [ ] Add preset import/export (share with others)
- [ ] Add preset thumbnails (auto-generated screenshots)

---

## Technical Architecture

### File Structure

```
PlasmaDX-Clean/
├── src/
│   ├── animation/
│   │   ├── AnimationRecorder.h
│   │   ├── AnimationRecorder.cpp
│   │   ├── CameraAnimator.h
│   │   └── CameraAnimator.cpp
│   ├── ui/
│   │   ├── SplashScreen.h
│   │   ├── SplashScreen.cpp
│   │   ├── PresetManager.h
│   │   └── PresetManager.cpp
│   ├── particles/
│   │   ├── ParticleSystem.h (UPDATED)
│   │   └── ParticleSystem.cpp (UPDATED)
│   └── core/
│       ├── Application.h (UPDATED)
│       └── Application.cpp (UPDATED)
├── shaders/
│   └── particles/
│       └── particle_physics.hlsl (UPDATED)
├── presets/
│   ├── built-in/
│   │   ├── sgr_a_realistic.json
│   │   ├── stellar_black_hole.json
│   │   ├── quasar.json
│   │   ├── galaxy_collision.json
│   │   └── protoplanetary_disk.json
│   └── user/
│       └── (user-saved presets)
├── recordings/
│   └── (session directories with frames)
├── external/
│   └── stb/
│       └── stb_image_write.h (PNG saving)
└── docs/
    ├── PHYSICS_EQUATIONS.md
    └── USER_GUIDE.md
```

### Dependencies

**New External Libraries:**
- **stb_image_write.h** - PNG saving (header-only, already have stb_image)
- **FFmpeg** - Video encoding (external executable, not linked)
- No new libraries needed! (Already have ImGui, JSON11/RapidJSON)

### Data Flow

```
1. Launch App
   └─> Show Splash Screen (ImGui window)
       └─> User selects preset or custom config
           └─> Load preset JSON
               └─> Initialize Application with config

2a. Realtime Mode (Normal)
    └─> Run() loop at vsync/uncapped FPS
        └─> Update physics (variable deltaTime)
        └─> Render frame
        └─> Display ImGui controls
        └─> User can adjust parameters live

2b. Recording Mode (Animation)
    └─> RunRecordingMode() loop (ignore real FPS)
        └─> Update physics (FIXED deltaTime, e.g., 1/120s)
        └─> Update camera animation (orbit/path)
        └─> Render frame
        └─> ReadbackBuffer: Copy backbuffer to CPU
        └─> Save frame as "frame_XXXXXX.png"
        └─> Increment frame counter
        └─> Repeat until target frame count reached
        └─> Call FFmpeg to encode PNG sequence -> MP4
```

---

## Implementation Timeline

### Week 1: Enhanced Physics (Days 1-3)

**Day 1:** Extend physics parameters
- Add new members to ParticleSystem
- Update compute shader with constraint shapes
- Create getters/setters for all parameters

**Day 2:** Implement advanced features
- Black hole mass affecting orbital velocity
- Alpha viscosity for inward spiral
- Dual galaxy collision mode
- Constraint shape switching

**Day 3:** Testing and ImGui integration
- Add all new physics controls to ImGui
- Create built-in physics presets
- Test each constraint shape
- Verify realistic behavior

### Week 2: Splash Screen & Presets (Days 4-7)

**Day 4:** Preset system architecture
- Create PresetManager class
- Define JSON preset format
- Implement save/load functions

**Day 5:** Splash screen UI
- Create SplashScreen class
- Design ImGui layout
- Implement preset selection UI

**Day 6:** Built-in presets and integration
- Create 5-6 scientifically accurate presets
- Integrate SplashScreen with main()
- Test preset loading

**Day 7:** Polish and saved presets
- Implement user preset save/delete
- Add custom configuration editor
- Test preset manager

### Week 3: Animation System (Days 8-11)

**Day 8:** Frame capture
- Implement SwapChain::SaveFrameToPNG()
- Create AnimationRecorder class
- Test PNG saving

**Day 9:** Recording mode
- Implement Application::RunRecordingMode()
- Fixed timestep physics
- Recording UI progress bar

**Day 10:** Camera animation
- Implement orbit camera mode
- Test camera animation system
- Record test video (no FFmpeg yet)

**Day 11:** FFmpeg integration
- Implement FFmpeg video encoding
- Test complete recording workflow
- Handle errors gracefully

### Week 4: Polish & Documentation (Days 12-14)

**Day 12:** Testing and bug fixes
- Test all presets
- Test all recording scenarios
- Fix bugs found during testing

**Day 13:** Documentation
- Write USER_GUIDE.md
- Create PHYSICS_EQUATIONS.md
- Add tooltips to ImGui controls

**Day 14:** Final polish
- Add demo videos
- Create preset thumbnails
- Performance optimization pass

**Total Estimated Time:** 14 days (2-3 weeks)

---

## Risk Assessment

### High Risk Items

❗ **FFmpeg Dependency**
- **Risk:** User may not have FFmpeg installed
- **Mitigation:** Add startup check, provide download link, fallback to PNG sequence only

❗ **GPU Memory for High-Resolution Recording**
- **Risk:** 4K recording may exceed VRAM
- **Mitigation:** Add VRAM check, limit max resolution, show warning

❗ **Physics Determinism**
- **Risk:** Physics may not be perfectly deterministic (floating point, GPU differences)
- **Mitigation:** Test extensively, document any limitations

### Medium Risk Items

⚠️ **Preset Compatibility**
- **Risk:** Old presets may break with new features
- **Mitigation:** Add version field to presets, implement migration

⚠️ **Recording Performance**
- **Risk:** PNG saving may be slow, causing lag
- **Mitigation:** Use async I/O, show progress, allow pause/resume

⚠️ **Splash Screen Complexity**
- **Risk:** UI may become cluttered with too many options
- **Mitigation:** Use collapsible sections, good defaults, "Simple" vs "Advanced" modes

### Low Risk Items

✅ **ImGui Integration** - Already working perfectly
✅ **JSON Parsing** - Standard library, well-tested
✅ **PNG Saving** - stb_image_write is battle-tested
✅ **Physics Extensions** - Building on working foundation

---

## Success Criteria

### Phase 3.1 (Physics)
- ✅ All PlasmaVulkan physics features ported
- ✅ Parameters adjustable in real-time
- ✅ No crashes or instability with extreme values
- ✅ Realistic accretion disk behavior

### Phase 3.2 (Splash Screen)
- ✅ User can select from 5+ built-in presets
- ✅ User can save custom presets
- ✅ User can load saved presets
- ✅ All physics parameters configurable
- ✅ Preset files are human-readable JSON

### Phase 3.3 (Animation)
- ✅ Can record PNG sequence at any resolution
- ✅ FFmpeg successfully encodes video
- ✅ Physics runs deterministically (same result each time)
- ✅ Camera animation works smoothly
- ✅ Recording UI shows progress
- ✅ Can cancel recording without corruption

### Phase 3.4 (Polish)
- ✅ User documentation complete
- ✅ All UI has tooltips
- ✅ Error messages are helpful
- ✅ Demo videos created
- ✅ Performance is acceptable

---

## Recommended Next Steps

1. **Review this roadmap** with stakeholder (you!) for approval
2. **Start with Phase 3.1** (Enhanced Physics) - This is the foundation
3. **Iterate on physics** until it feels "right" before moving to UI
4. **Implement phases sequentially** - Don't skip ahead
5. **Test thoroughly** after each phase before continuing
6. **Document as you go** - Don't leave docs until the end

---

## Additional Recommendations

### 1. Physics Validation

Consider adding a "Physics Validation Mode" that:
- Plots particle velocities vs. radius (should follow v ∝ r^(-1/2))
- Checks energy conservation
- Validates Keplerian orbits
- Compares to known solutions

This would be useful for:
- Debugging physics issues
- Creating educational content
- Publishing scientific visualizations

### 2. Preset Sharing

Add ability to export/import presets as shareable files:
- Generate QR codes for easy sharing
- Host a community preset repository
- Add preset ratings/comments

### 3. Advanced Camera Modes

Beyond orbit, consider:
- **Flyby** - Camera flies past disk at high speed
- **Zoom** - Slow zoom from far to close-up
- **Inside Out** - Camera starts inside disk, pulls out
- **Keyframe Editor** - User defines camera path with bezier curves

### 4. Real-time Recording

Add ability to record while running real-time (not fixed timestep):
- Useful for capturing live interactions
- User can adjust parameters during recording
- May not be deterministic, but captures "discovery moments"

### 5. Audio

Consider adding procedurally generated audio:
- Sonification of particle velocities
- Black hole "rumble" based on mass
- Accretion disk "roar" based on density
- Makes videos more engaging

### 6. Export Formats

Beyond MP4, consider:
- **GIF** - For social media (use FFmpeg)
- **WebM** - For web embedding
- **Image Sequence** - For compositing in After Effects/Blender
- **Turntable** - 360° rotation for 3D viewers

### 7. Metadata Embedding

Embed simulation parameters in video metadata:
- Physics settings
- Timestamp
- Software version
- Makes videos scientifically reproducible

---

## Conclusion

This roadmap provides a clear path to transforming PlasmaDX-Clean from a real-time visualization tool into a professional-grade accretion disk simulation and rendering studio.

**Key Benefits:**
- ✨ **User-Friendly:** Splash screen makes it accessible to non-programmers
- 🎬 **Production-Ready:** Animation system enables high-quality video creation
- 🔬 **Scientifically Accurate:** Enhanced physics based on real astrophysics
- 💾 **Flexible:** Save presets, share configurations, iterate on designs
- 🚀 **Future-Proof:** Modular architecture allows easy expansion

**Next Action:** Start with Phase 3.1 (Enhanced Physics) to establish the foundation for all other features.

---

**Document Status:** Ready for Review
**Last Updated:** 2025-10-15
**Author:** Claude (with extensive reference to PlasmaVulkan implementation)
