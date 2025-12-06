# Camera System Upgrade Plan: Orbit to Free-Roam

**Created:** 2025-12-06
**Status:** PLANNED
**Priority:** Medium
**Estimated Effort:** 2-3 hours

---

## Problem Statement

### Current Behavior
The current camera system is an **orbit camera** that always looks at the world origin `(0, 0, 0)`. The camera position is calculated from spherical coordinates (distance, height, angle) but the look-at target is hardcoded.

### Issues This Causes

1. **Volumetric Effects Appear to "Follow" the Camera**
   - The NanoVDB fog sphere at origin is always centered in the viewport
   - As the camera orbits, the fog always appears in the same screen position
   - This creates an illusion that volumetrics are "attached" to the camera
   - Same issue previously affected the deprecated God Ray system

2. **Limited Creative Freedom**
   - Cannot look away from the origin to see volumetrics from true side angles
   - Cannot position camera to frame shots without origin in view
   - Automated camera paths for animations are severely constrained

3. **Physics of Volumetric Scattering**
   - The Henyey-Greenstein phase function creates view-dependent brightness (this is physically correct)
   - With orbit camera, you always see the same scattering angle relative to the volume center
   - Free-roam would let you observe the volume from different scattering angles

---

## Current Implementation

### Key Files
- `src/core/Application.h` - Camera state variables (lines ~510-520)
- `src/core/Application.cpp` - View matrix calculation, input handling

### Camera State Variables (Application.h)
```cpp
// Current orbit camera state
float m_cameraDistance = 800.0f;   // Distance from origin
float m_cameraHeight = 200.0f;     // Height above XZ plane
float m_cameraAngle = 0.0f;        // Orbit angle around Y axis
float m_cameraPitch = 0.0f;        // Pitch angle (limited use currently)
```

### View Matrix Calculation (Application.cpp ~line 834)
```cpp
// Camera position calculated from spherical coords
float camX = m_cameraDistance * sinf(m_cameraAngle);
float camY = m_cameraHeight;
float camZ = m_cameraDistance * cosf(m_cameraAngle);
XMFLOAT3 cameraPosition = { camX, camY, camZ };

// PROBLEM: lookAt is ALWAYS origin
XMVECTOR camPos = XMLoadFloat3(&cameraPosition);
XMVECTOR lookAt = XMVectorSet(0, 0, 0, 1.0f);  // <-- HARDCODED!
XMVECTOR up = XMVectorSet(0, 1, 0, 0);
viewMat = XMMatrixLookAtLH(camPos, lookAt, up);
```

### Current Input Handling
- **Arrow Keys / WASD**: Orbit around origin (change angle/height)
- **Mouse Wheel**: Zoom in/out (change distance)
- **No mouse look**: Camera always faces origin

---

## Proposed Solution: Dual-Mode Camera System

### Design Goals
1. **Free-Roam as Default** - WASD movement + mouse look (FPS-style)
2. **Orbit Mode Optional** - Toggle with Tab key for cinematic shots
3. **Backward Compatible** - Existing configs and screenshots work unchanged
4. **Minimal Code Changes** - Reuse existing infrastructure where possible

### New Camera State Variables
```cpp
// Add to Application.h
enum class CameraMode { FreeRoam, Orbit };

CameraMode m_cameraMode = CameraMode::FreeRoam;  // Default to free-roam

// Free-roam specific state
float m_cameraYaw = 0.0f;      // Horizontal look angle (radians)
float m_cameraPitch = 0.0f;    // Vertical look angle (radians, clamped +-89 degrees)
XMFLOAT3 m_cameraPosition = { 0.0f, 200.0f, 800.0f };  // World position

// Orbit mode state (existing, kept for compatibility)
float m_orbitDistance = 800.0f;
float m_orbitHeight = 200.0f;
float m_orbitAngle = 0.0f;
```

### New View Matrix Calculation
```cpp
XMMATRIX CalculateViewMatrix() {
    if (m_cameraMode == CameraMode::FreeRoam) {
        // Calculate forward direction from yaw/pitch
        float cosP = cosf(m_cameraPitch);
        XMFLOAT3 forward = {
            cosP * sinf(m_cameraYaw),
            sinf(m_cameraPitch),
            cosP * cosf(m_cameraYaw)
        };

        XMVECTOR camPos = XMLoadFloat3(&m_cameraPosition);
        XMVECTOR lookAt = camPos + XMLoadFloat3(&forward);
        XMVECTOR up = XMVectorSet(0, 1, 0, 0);

        return XMMatrixLookAtLH(camPos, lookAt, up);
    } else {
        // Existing orbit camera logic
        float camX = m_orbitDistance * sinf(m_orbitAngle);
        float camY = m_orbitHeight;
        float camZ = m_orbitDistance * cosf(m_orbitAngle);

        XMVECTOR camPos = XMVectorSet(camX, camY, camZ, 1.0f);
        XMVECTOR lookAt = XMVectorSet(0, 0, 0, 1.0f);
        XMVECTOR up = XMVectorSet(0, 1, 0, 0);

        return XMMatrixLookAtLH(camPos, lookAt, up);
    }
}
```

### New Input Handling

#### Free-Roam Mode
| Input | Action |
|-------|--------|
| W/S | Move forward/backward |
| A/D | Strafe left/right |
| Space | Move up |
| Ctrl | Move down |
| Mouse Move | Look around (yaw/pitch) |
| Mouse Wheel | Adjust move speed |
| Tab | Switch to Orbit mode |

#### Orbit Mode (unchanged from current)
| Input | Action |
|-------|--------|
| A/D or Left/Right | Orbit around origin |
| W/S or Up/Down | Raise/lower camera |
| Mouse Wheel | Zoom in/out |
| Tab | Switch to Free-Roam mode |

---

## Implementation Plan

### Phase 1: Add Camera Mode Infrastructure
**Files:** `Application.h`

1. Add `CameraMode` enum
2. Add free-roam state variables (`m_cameraYaw`, `m_cameraPitch`, `m_cameraPosition`)
3. Add `m_cameraMode` with default `FreeRoam`
4. Add `m_mouseCaptured` bool for mouse look
5. Add `m_moveSpeed` for movement speed control

### Phase 2: Update View Matrix Calculation
**Files:** `Application.cpp`

1. Create `CalculateViewMatrix()` helper function
2. Replace all `XMMatrixLookAtLH` calls with the helper
3. Locations to update (search for `XMMatrixLookAtLH`):
   - Line ~834 (main render loop)
   - Line ~1033 (Gaussian renderer)
   - Line ~1085 (DLSS path)
   - Line ~1222 (NanoVDB render)

### Phase 3: Update Input Handling
**Files:** `Application.cpp`

1. Add Tab key handler to toggle camera mode
2. Add mouse capture/release (click to capture, Escape to release)
3. Modify WASD handling based on camera mode
4. Add mouse movement handler for free-roam look
5. Calculate forward/right vectors for movement

### Phase 4: Add ImGui Controls
**Files:** `Application.cpp` (ImGui section)

```cpp
// Add to control panel
if (ImGui::CollapsingHeader("Camera System")) {
    const char* modes[] = { "Free-Roam", "Orbit" };
    int modeInt = static_cast<int>(m_cameraMode);
    if (ImGui::Combo("Mode", &modeInt, modes, 2)) {
        m_cameraMode = static_cast<CameraMode>(modeInt);
    }

    ImGui::SliderFloat("Move Speed", &m_moveSpeed, 10.0f, 500.0f);
    ImGui::SliderFloat("Look Sensitivity", &m_lookSensitivity, 0.001f, 0.01f);

    if (m_cameraMode == CameraMode::FreeRoam) {
        ImGui::Text("Position: (%.1f, %.1f, %.1f)",
                    m_cameraPosition.x, m_cameraPosition.y, m_cameraPosition.z);
        ImGui::Text("Yaw: %.1f, Pitch: %.1f",
                    XMConvertToDegrees(m_cameraYaw), XMConvertToDegrees(m_cameraPitch));
    }

    if (ImGui::Button("Reset to Origin View")) {
        m_cameraPosition = { 0.0f, 200.0f, 800.0f };
        m_cameraYaw = 0.0f;
        m_cameraPitch = 0.0f;
    }
}
```

### Phase 5: Update Config System
**Files:** `Config.h`, `Config.cpp`

1. Add camera mode to `CameraConfig` struct
2. Add save/load for free-roam position and angles
3. Maintain backward compatibility (default to free-roam if not specified)

### Phase 6: Update Screenshot Metadata
**Files:** `Application.cpp` (screenshot section)

1. Save camera mode in metadata
2. Save yaw/pitch for free-roam mode
3. Keep existing orbit metadata for compatibility

---

## Testing Checklist

- [ ] Free-roam WASD movement works in all directions
- [ ] Mouse look works with proper pitch clamping (prevent flipping)
- [ ] Tab toggles between modes correctly
- [ ] Orbit mode still works exactly as before
- [ ] NanoVDB volumetrics render correctly in both modes
- [ ] DLSS works in both modes
- [ ] Gaussian renderer works in both modes
- [ ] Screenshot metadata saves correct camera state
- [ ] Config save/load preserves camera settings
- [ ] ImGui controls work correctly
- [ ] Performance unchanged

---

## Rollback Plan

If issues occur:
1. The existing orbit camera code is preserved (just wrapped in if/else)
2. Change default `m_cameraMode` back to `Orbit` to restore old behavior
3. All changes are additive - no existing functionality removed

---

## Future Enhancements (Out of Scope)

- Camera animation paths (bezier curves, keyframes)
- Cinematic camera modes (dolly, crane, pan)
- Camera collision with scene geometry
- Multiple camera presets with quick switch
- Record/playback camera movements

---

## Notes

- The "volumetrics following camera" effect will be significantly reduced but not eliminated
- The Henyey-Greenstein phase function will still cause physically-correct view-dependent scattering
- This is expected behavior for real volumetric effects (nebulae, fog, smoke)
- If you want the volumetric to look identical from all angles, reduce the phase function `g` parameter toward 0 (isotropic scattering)

---

**Document Author:** Claude Code Session
**Last Updated:** 2025-12-06
