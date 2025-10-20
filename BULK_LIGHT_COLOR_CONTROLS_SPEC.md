# Bulk Light Color Controls - Technical Specification

**Document Version:** 1.0
**Created:** 2025-10-20
**Status:** Phase 5 Milestone 5.3b Implementation Blueprint
**Related:** PHASE_5_CELESTIAL_RENDERING_PLAN.md, GOD_RAY_SYSTEM_SPEC.md

---

## Executive Summary

**Problem:** Adjusting colors for multi-light systems (13-16 lights) is tedious - each light requires opening controls, selecting color picker, and manually setting RGB values.

**Solution:** Bulk light color control system with:
- Color presets (warm, cool, rainbow, monochrome, stellar)
- Gradient application (radial, linear, circular)
- Global color tint/shift
- Quick scenario-based presets

**Implementation Time:** 3-4 hours
**User Impact:** Massive UX improvement for animation workflows

---

## User Workflow Problem

### Current (Tedious) Workflow

To change all 13 lights to warm colors:

1. Expand Light 0 controls
2. Open color picker
3. Set RGB (1.0, 0.8, 0.4)
4. Collapse Light 0 controls
5. Expand Light 1 controls
6. Open color picker
7. Set RGB (1.0, 0.8, 0.4)
8. ... repeat 11 more times ...

**Total time:** ~2-3 minutes per color change
**User feedback:** "for multiple lights this takes some work"

### New (Streamlined) Workflow

To change all 13 lights to warm colors:

1. Open "Bulk Light Color Controls"
2. Click "Warm Sunset" preset
3. Done!

**Total time:** ~5 seconds
**Improvement:** 24-36× faster!

---

## Feature Set

### 1. Color Presets (Quick Apply)

**Preset Categories:**

**A. Temperature Presets (Blackbody):**
- Cool Blue (10000K)
- White (6500K)
- Warm White (4000K)
- Warm Sunset (2500K)
- Deep Red (1800K)

**B. Artistic Presets:**
- Rainbow (HSV gradient across lights)
- Complementary (alternating complementary colors)
- Monochrome Blue/Red/Green
- Neon (vibrant saturated colors)
- Pastel (desaturated soft colors)

**C. Scenario Presets:**
- Stellar Nursery (blue/white for hot stars)
- Red Giant (red/orange for cool giants)
- Accretion Disk (radial gradient: blue → yellow → red)
- Binary System (two-tone: blue + red)
- Dust Torus (brown/orange earth tones)

### 2. Gradient Application

**Gradient Types:**

**A. Radial Gradient:**
- Based on distance from center (0,0,0)
- Inner lights → one color
- Outer lights → different color
- Smooth interpolation in between

**Example:** Accretion disk
- Inner (200-600 units): Blue/white (hot)
- Outer (600-1500 units): Red/brown (cool)

**B. Linear Gradient:**
- Based on position along axis (X, Y, or Z)
- Min axis value → one color
- Max axis value → different color

**Example:** Vertical gradient
- Bottom lights (-Y): Red
- Top lights (+Y): Blue

**C. Circular Gradient:**
- Based on angle around Y-axis
- 0° → start color
- 360° → end color (wraps)

**Example:** Rainbow ring
- 0°: Red
- 60°: Yellow
- 120°: Green
- 180°: Cyan
- 240°: Blue
- 300°: Magenta
- 360°: Red (wrap)

### 3. Global Color Operations

**A. Hue Shift:**
- Rotate all light colors around HSV hue wheel
- Preserves saturation and value
- Quick color scheme variations

**B. Saturation Adjust:**
- Increase/decrease color intensity
- 0.0 = grayscale (white)
- 1.0 = fully saturated

**C. Value/Brightness Adjust:**
- Darken/brighten all lights uniformly
- Preserves relative color relationships

**D. Temperature Shift:**
- Shift all lights warmer (toward red)
- Shift all lights cooler (toward blue)
- Preserves relative temperature differences

### 4. Selection Groups

**Predefined Groups:**
- All Lights
- Inner Ring (distance < threshold)
- Outer Ring (distance > threshold)
- Top Half (Y > 0)
- Bottom Half (Y < 0)
- Even Indices (0, 2, 4, ...)
- Odd Indices (1, 3, 5, ...)

**Custom Ranges:**
- Lights 0-7
- Lights 8-15
- User-defined index ranges

---

## Implementation Architecture

### Data Structures

**Application.h additions:**

```cpp
class Application {
private:
    // === Bulk Light Color Control State ===
    enum class ColorPreset {
        Custom,
        CoolBlue,
        White,
        WarmWhite,
        WarmSunset,
        DeepRed,
        Rainbow,
        Complementary,
        MonochromeBlue,
        MonochromeRed,
        MonochromeGreen,
        Neon,
        Pastel,
        StellarNursery,
        RedGiant,
        AccretionDisk,
        BinarySystem,
        DustTorus
    };

    enum class GradientType {
        Radial,       // Distance from center
        LinearX,      // Position along X
        LinearY,      // Position along Y
        LinearZ,      // Position along Z
        Circular      // Angle around Y-axis
    };

    enum class LightSelection {
        All,
        InnerRing,
        OuterRing,
        TopHalf,
        BottomHalf,
        EvenIndices,
        OddIndices,
        CustomRange
    };

    // Bulk color control state
    ColorPreset m_currentColorPreset = ColorPreset::Custom;
    LightSelection m_lightSelection = LightSelection::All;
    int m_customRangeStart = 0;
    int m_customRangeEnd = 15;
    float m_radialThreshold = 800.0f;  // Distance threshold for inner/outer ring

    // Gradient application state
    GradientType m_gradientType = GradientType::Radial;
    DirectX::XMFLOAT3 m_gradientColorStart = {1.0f, 1.0f, 1.0f};
    DirectX::XMFLOAT3 m_gradientColorEnd = {1.0f, 0.0f, 0.0f};

    // Global color operations
    float m_hueShift = 0.0f;            // -180 to +180 degrees
    float m_saturationAdjust = 1.0f;    // 0.0 to 2.0 (multiplier)
    float m_valueAdjust = 1.0f;         // 0.0 to 2.0 (multiplier)
    float m_temperatureShift = 0.0f;    // -1.0 (cooler) to +1.0 (warmer)

    // Helper functions
    void ApplyColorPreset(ColorPreset preset);
    void ApplyGradient(GradientType type, DirectX::XMFLOAT3 startColor, DirectX::XMFLOAT3 endColor);
    void ApplyGlobalHueShift(float degrees);
    void ApplyGlobalSaturationAdjust(float multiplier);
    void ApplyGlobalValueAdjust(float multiplier);
    void ApplyTemperatureShift(float amount);
    std::vector<int> GetSelectedLightIndices();
    DirectX::XMFLOAT3 RGBtoHSV(DirectX::XMFLOAT3 rgb);
    DirectX::XMFLOAT3 HSVtoRGB(DirectX::XMFLOAT3 hsv);
    DirectX::XMFLOAT3 BlackbodyColor(float temperature);
};
```

### Helper Function Implementations

**Color Space Conversion:**

```cpp
DirectX::XMFLOAT3 Application::RGBtoHSV(DirectX::XMFLOAT3 rgb) {
    float r = rgb.x, g = rgb.y, b = rgb.z;
    float max = std::max({r, g, b});
    float min = std::min({r, g, b});
    float delta = max - min;

    // Hue
    float h = 0.0f;
    if (delta > 0.001f) {
        if (max == r) {
            h = 60.0f * fmod((g - b) / delta, 6.0f);
        } else if (max == g) {
            h = 60.0f * ((b - r) / delta + 2.0f);
        } else {
            h = 60.0f * ((r - g) / delta + 4.0f);
        }
    }
    if (h < 0.0f) h += 360.0f;

    // Saturation
    float s = (max < 0.001f) ? 0.0f : (delta / max);

    // Value
    float v = max;

    return DirectX::XMFLOAT3(h, s, v);
}

DirectX::XMFLOAT3 Application::HSVtoRGB(DirectX::XMFLOAT3 hsv) {
    float h = hsv.x, s = hsv.y, v = hsv.z;

    float c = v * s;
    float x = c * (1.0f - fabs(fmod(h / 60.0f, 2.0f) - 1.0f));
    float m = v - c;

    float r, g, b;
    if (h < 60.0f) {
        r = c; g = x; b = 0.0f;
    } else if (h < 120.0f) {
        r = x; g = c; b = 0.0f;
    } else if (h < 180.0f) {
        r = 0.0f; g = c; b = x;
    } else if (h < 240.0f) {
        r = 0.0f; g = x; b = c;
    } else if (h < 300.0f) {
        r = x; g = 0.0f; b = c;
    } else {
        r = c; g = 0.0f; b = x;
    }

    return DirectX::XMFLOAT3(r + m, g + m, b + m);
}

DirectX::XMFLOAT3 Application::BlackbodyColor(float temperature) {
    // Simplified Planck blackbody approximation
    // Temperature in Kelvin → RGB color

    float t = temperature / 100.0f;
    float r, g, b;

    // Red
    if (t <= 66.0f) {
        r = 1.0f;
    } else {
        r = 1.292936186f * pow(t - 60.0f, -0.1332047592f);
        r = std::clamp(r, 0.0f, 1.0f);
    }

    // Green
    if (t <= 66.0f) {
        g = 0.39008157876f * log(t) - 0.631841444f;
    } else {
        g = 1.129890861f * pow(t - 60.0f, -0.0755148492f);
    }
    g = std::clamp(g, 0.0f, 1.0f);

    // Blue
    if (t >= 66.0f) {
        b = 1.0f;
    } else if (t <= 19.0f) {
        b = 0.0f;
    } else {
        b = 0.543206789f * log(t - 10.0f) - 1.196254089f;
        b = std::clamp(b, 0.0f, 1.0f);
    }

    return DirectX::XMFLOAT3(r, g, b);
}
```

**Selection Helper:**

```cpp
std::vector<int> Application::GetSelectedLightIndices() {
    std::vector<int> indices;

    switch (m_lightSelection) {
    case LightSelection::All:
        for (int i = 0; i < (int)m_lights.size(); i++) {
            indices.push_back(i);
        }
        break;

    case LightSelection::InnerRing:
        for (int i = 0; i < (int)m_lights.size(); i++) {
            float dist = sqrt(m_lights[i].position.x * m_lights[i].position.x +
                            m_lights[i].position.y * m_lights[i].position.y +
                            m_lights[i].position.z * m_lights[i].position.z);
            if (dist < m_radialThreshold) {
                indices.push_back(i);
            }
        }
        break;

    case LightSelection::OuterRing:
        for (int i = 0; i < (int)m_lights.size(); i++) {
            float dist = sqrt(m_lights[i].position.x * m_lights[i].position.x +
                            m_lights[i].position.y * m_lights[i].position.y +
                            m_lights[i].position.z * m_lights[i].position.z);
            if (dist >= m_radialThreshold) {
                indices.push_back(i);
            }
        }
        break;

    case LightSelection::TopHalf:
        for (int i = 0; i < (int)m_lights.size(); i++) {
            if (m_lights[i].position.y > 0.0f) {
                indices.push_back(i);
            }
        }
        break;

    case LightSelection::BottomHalf:
        for (int i = 0; i < (int)m_lights.size(); i++) {
            if (m_lights[i].position.y <= 0.0f) {
                indices.push_back(i);
            }
        }
        break;

    case LightSelection::EvenIndices:
        for (int i = 0; i < (int)m_lights.size(); i += 2) {
            indices.push_back(i);
        }
        break;

    case LightSelection::OddIndices:
        for (int i = 1; i < (int)m_lights.size(); i += 2) {
            indices.push_back(i);
        }
        break;

    case LightSelection::CustomRange:
        for (int i = m_customRangeStart; i <= m_customRangeEnd && i < (int)m_lights.size(); i++) {
            indices.push_back(i);
        }
        break;
    }

    return indices;
}
```

**Color Preset Application:**

```cpp
void Application::ApplyColorPreset(ColorPreset preset) {
    m_currentColorPreset = preset;
    auto indices = GetSelectedLightIndices();

    switch (preset) {
    case ColorPreset::CoolBlue:
        for (int idx : indices) {
            m_lights[idx].color = BlackbodyColor(10000.0f);  // Cool blue
        }
        break;

    case ColorPreset::White:
        for (int idx : indices) {
            m_lights[idx].color = BlackbodyColor(6500.0f);  // Daylight white
        }
        break;

    case ColorPreset::WarmWhite:
        for (int idx : indices) {
            m_lights[idx].color = BlackbodyColor(4000.0f);  // Warm white
        }
        break;

    case ColorPreset::WarmSunset:
        for (int idx : indices) {
            m_lights[idx].color = BlackbodyColor(2500.0f);  // Orange sunset
        }
        break;

    case ColorPreset::DeepRed:
        for (int idx : indices) {
            m_lights[idx].color = BlackbodyColor(1800.0f);  // Deep red
        }
        break;

    case ColorPreset::Rainbow:
        for (size_t i = 0; i < indices.size(); i++) {
            int idx = indices[i];
            float hue = (i * 360.0f) / indices.size();  // Distribute evenly
            DirectX::XMFLOAT3 hsv(hue, 1.0f, 1.0f);
            m_lights[idx].color = HSVtoRGB(hsv);
        }
        break;

    case ColorPreset::Complementary:
        for (size_t i = 0; i < indices.size(); i++) {
            int idx = indices[i];
            float hue = (i % 2 == 0) ? 30.0f : 210.0f;  // Orange vs Blue
            DirectX::XMFLOAT3 hsv(hue, 1.0f, 1.0f);
            m_lights[idx].color = HSVtoRGB(hsv);
        }
        break;

    case ColorPreset::MonochromeBlue:
        for (int idx : indices) {
            m_lights[idx].color = DirectX::XMFLOAT3(0.2f, 0.4f, 1.0f);
        }
        break;

    case ColorPreset::MonochromeRed:
        for (int idx : indices) {
            m_lights[idx].color = DirectX::XMFLOAT3(1.0f, 0.2f, 0.2f);
        }
        break;

    case ColorPreset::MonochromeGreen:
        for (int idx : indices) {
            m_lights[idx].color = DirectX::XMFLOAT3(0.2f, 1.0f, 0.3f);
        }
        break;

    case ColorPreset::Neon:
        for (size_t i = 0; i < indices.size(); i++) {
            int idx = indices[i];
            float hue = (i * 360.0f) / indices.size();
            DirectX::XMFLOAT3 hsv(hue, 1.0f, 1.0f);  // Fully saturated
            m_lights[idx].color = HSVtoRGB(hsv);
        }
        break;

    case ColorPreset::Pastel:
        for (size_t i = 0; i < indices.size(); i++) {
            int idx = indices[i];
            float hue = (i * 360.0f) / indices.size();
            DirectX::XMFLOAT3 hsv(hue, 0.3f, 1.0f);  // Low saturation
            m_lights[idx].color = HSVtoRGB(hsv);
        }
        break;

    case ColorPreset::StellarNursery:
        // Blue/white for hot young stars
        for (int idx : indices) {
            float temp = 15000.0f + (rand() % 10000);  // 15000-25000K variation
            m_lights[idx].color = BlackbodyColor(temp);
        }
        break;

    case ColorPreset::RedGiant:
        // Red/orange for cool giant stars
        for (int idx : indices) {
            float temp = 2800.0f + (rand() % 1000);  // 2800-3800K variation
            m_lights[idx].color = BlackbodyColor(temp);
        }
        break;

    case ColorPreset::AccretionDisk:
        // Radial gradient: blue (inner) → red (outer)
        ApplyGradient(GradientType::Radial,
                     BlackbodyColor(25000.0f),  // Blue/white
                     BlackbodyColor(2000.0f));   // Red
        break;

    case ColorPreset::BinarySystem:
        // Two-tone: first half blue, second half red
        for (size_t i = 0; i < indices.size(); i++) {
            int idx = indices[i];
            if (i < indices.size() / 2) {
                m_lights[idx].color = BlackbodyColor(30000.0f);  // Blue star
            } else {
                m_lights[idx].color = BlackbodyColor(3000.0f);   // Red star
            }
        }
        break;

    case ColorPreset::DustTorus:
        // Earth tones: brown/orange
        for (int idx : indices) {
            m_lights[idx].color = DirectX::XMFLOAT3(0.6f, 0.4f, 0.2f);
        }
        break;
    }
}
```

**Gradient Application:**

```cpp
void Application::ApplyGradient(GradientType type, DirectX::XMFLOAT3 startColor, DirectX::XMFLOAT3 endColor) {
    auto indices = GetSelectedLightIndices();

    // Calculate gradient parameter for each light
    std::vector<float> gradientParams;
    float minParam = FLT_MAX, maxParam = -FLT_MAX;

    for (int idx : indices) {
        float param = 0.0f;

        switch (type) {
        case GradientType::Radial:
            // Distance from origin
            param = sqrt(m_lights[idx].position.x * m_lights[idx].position.x +
                        m_lights[idx].position.y * m_lights[idx].position.y +
                        m_lights[idx].position.z * m_lights[idx].position.z);
            break;

        case GradientType::LinearX:
            param = m_lights[idx].position.x;
            break;

        case GradientType::LinearY:
            param = m_lights[idx].position.y;
            break;

        case GradientType::LinearZ:
            param = m_lights[idx].position.z;
            break;

        case GradientType::Circular:
            // Angle around Y-axis
            param = atan2(m_lights[idx].position.z, m_lights[idx].position.x);
            param = (param + 3.14159f) / (2.0f * 3.14159f);  // Normalize to 0-1
            break;
        }

        gradientParams.push_back(param);
        minParam = std::min(minParam, param);
        maxParam = std::max(maxParam, param);
    }

    // Apply gradient
    for (size_t i = 0; i < indices.size(); i++) {
        int idx = indices[i];

        // Normalize parameter to 0-1 range
        float t = (maxParam - minParam < 0.001f) ? 0.5f : (gradientParams[i] - minParam) / (maxParam - minParam);

        // Lerp between start and end color
        m_lights[idx].color.x = startColor.x * (1.0f - t) + endColor.x * t;
        m_lights[idx].color.y = startColor.y * (1.0f - t) + endColor.y * t;
        m_lights[idx].color.z = startColor.z * (1.0f - t) + endColor.z * t;
    }
}
```

**Global Hue Shift:**

```cpp
void Application::ApplyGlobalHueShift(float degrees) {
    auto indices = GetSelectedLightIndices();

    for (int idx : indices) {
        // Convert RGB → HSV
        DirectX::XMFLOAT3 hsv = RGBtoHSV(m_lights[idx].color);

        // Shift hue
        hsv.x += degrees;
        while (hsv.x < 0.0f) hsv.x += 360.0f;
        while (hsv.x >= 360.0f) hsv.x -= 360.0f;

        // Convert back to RGB
        m_lights[idx].color = HSVtoRGB(hsv);
    }
}
```

---

## ImGui UI Layout

### Bulk Light Color Controls Panel

```cpp
void Application::RenderImGui() {
    // ... existing ImGui code ...

    // === NEW: Bulk Light Color Controls ===
    if (ImGui::CollapsingHeader("Bulk Light Color Controls", ImGuiTreeNodeFlags_DefaultOpen)) {

        // === SECTION 1: Selection ===
        if (ImGui::TreeNode("Light Selection")) {
            const char* selectionModes[] = {
                "All Lights", "Inner Ring", "Outer Ring", "Top Half",
                "Bottom Half", "Even Indices", "Odd Indices", "Custom Range"
            };

            int currentSelection = (int)m_lightSelection;
            if (ImGui::Combo("Select Lights", &currentSelection, selectionModes, 8)) {
                m_lightSelection = (LightSelection)currentSelection;
            }

            // Custom range controls (if Custom Range selected)
            if (m_lightSelection == LightSelection::CustomRange) {
                ImGui::SliderInt("Range Start", &m_customRangeStart, 0, (int)m_lights.size() - 1);
                ImGui::SliderInt("Range End", &m_customRangeEnd, 0, (int)m_lights.size() - 1);
            }

            // Radial threshold (if Inner/Outer Ring selected)
            if (m_lightSelection == LightSelection::InnerRing || m_lightSelection == LightSelection::OuterRing) {
                ImGui::SliderFloat("Radial Threshold", &m_radialThreshold, 100.0f, 2000.0f);
            }

            // Show selected count
            auto selectedIndices = GetSelectedLightIndices();
            ImGui::Text("Selected: %d lights", (int)selectedIndices.size());

            ImGui::TreePop();
        }

        ImGui::Separator();

        // === SECTION 2: Color Presets ===
        if (ImGui::TreeNode("Color Presets")) {

            ImGui::Text("Temperature Presets:");
            if (ImGui::Button("Cool Blue (10000K)")) ApplyColorPreset(ColorPreset::CoolBlue);
            ImGui::SameLine();
            if (ImGui::Button("White (6500K)")) ApplyColorPreset(ColorPreset::White);

            if (ImGui::Button("Warm White (4000K)")) ApplyColorPreset(ColorPreset::WarmWhite);
            ImGui::SameLine();
            if (ImGui::Button("Warm Sunset (2500K)")) ApplyColorPreset(ColorPreset::WarmSunset);

            if (ImGui::Button("Deep Red (1800K)")) ApplyColorPreset(ColorPreset::DeepRed);

            ImGui::Separator();

            ImGui::Text("Artistic Presets:");
            if (ImGui::Button("Rainbow")) ApplyColorPreset(ColorPreset::Rainbow);
            ImGui::SameLine();
            if (ImGui::Button("Complementary")) ApplyColorPreset(ColorPreset::Complementary);

            if (ImGui::Button("Monochrome Blue")) ApplyColorPreset(ColorPreset::MonochromeBlue);
            ImGui::SameLine();
            if (ImGui::Button("Monochrome Red")) ApplyColorPreset(ColorPreset::MonochromeRed);

            if (ImGui::Button("Monochrome Green")) ApplyColorPreset(ColorPreset::MonochromeGreen);
            ImGui::SameLine();
            if (ImGui::Button("Neon")) ApplyColorPreset(ColorPreset::Neon);

            if (ImGui::Button("Pastel")) ApplyColorPreset(ColorPreset::Pastel);

            ImGui::Separator();

            ImGui::Text("Scenario Presets:");
            if (ImGui::Button("Stellar Nursery")) ApplyColorPreset(ColorPreset::StellarNursery);
            ImGui::SameLine();
            if (ImGui::Button("Red Giant")) ApplyColorPreset(ColorPreset::RedGiant);

            if (ImGui::Button("Accretion Disk")) ApplyColorPreset(ColorPreset::AccretionDisk);
            ImGui::SameLine();
            if (ImGui::Button("Binary System")) ApplyColorPreset(ColorPreset::BinarySystem);

            if (ImGui::Button("Dust Torus")) ApplyColorPreset(ColorPreset::DustTorus);

            ImGui::TreePop();
        }

        ImGui::Separator();

        // === SECTION 3: Gradient Application ===
        if (ImGui::TreeNode("Gradient Application")) {

            const char* gradientTypes[] = {
                "Radial (Distance)", "Linear X", "Linear Y", "Linear Z", "Circular (Angle)"
            };

            int currentGradient = (int)m_gradientType;
            if (ImGui::Combo("Gradient Type", &currentGradient, gradientTypes, 5)) {
                m_gradientType = (GradientType)currentGradient;
            }

            ImGui::ColorEdit3("Start Color", &m_gradientColorStart.x);
            ImGui::ColorEdit3("End Color", &m_gradientColorEnd.x);

            if (ImGui::Button("Apply Gradient")) {
                ApplyGradient(m_gradientType, m_gradientColorStart, m_gradientColorEnd);
                m_currentColorPreset = ColorPreset::Custom;
            }

            ImGui::TreePop();
        }

        ImGui::Separator();

        // === SECTION 4: Global Color Operations ===
        if (ImGui::TreeNode("Global Color Operations")) {

            // Hue shift
            ImGui::SliderFloat("Hue Shift (degrees)", &m_hueShift, -180.0f, 180.0f);
            if (ImGui::Button("Apply Hue Shift")) {
                ApplyGlobalHueShift(m_hueShift);
                m_currentColorPreset = ColorPreset::Custom;
            }
            ImGui::SameLine();
            if (ImGui::Button("Reset##hue")) {
                m_hueShift = 0.0f;
            }

            // Saturation adjust
            ImGui::SliderFloat("Saturation Multiplier", &m_saturationAdjust, 0.0f, 2.0f);
            if (ImGui::Button("Apply Saturation")) {
                ApplyGlobalSaturationAdjust(m_saturationAdjust);
                m_currentColorPreset = ColorPreset::Custom;
            }
            ImGui::SameLine();
            if (ImGui::Button("Reset##sat")) {
                m_saturationAdjust = 1.0f;
            }

            // Value adjust
            ImGui::SliderFloat("Brightness Multiplier", &m_valueAdjust, 0.0f, 2.0f);
            if (ImGui::Button("Apply Brightness")) {
                ApplyGlobalValueAdjust(m_valueAdjust);
                m_currentColorPreset = ColorPreset::Custom;
            }
            ImGui::SameLine();
            if (ImGui::Button("Reset##val")) {
                m_valueAdjust = 1.0f;
            }

            // Temperature shift
            ImGui::SliderFloat("Temperature Shift", &m_temperatureShift, -1.0f, 1.0f);
            if (ImGui::Button("Apply Temperature Shift")) {
                ApplyTemperatureShift(m_temperatureShift);
                m_currentColorPreset = ColorPreset::Custom;
            }
            ImGui::SameLine();
            if (ImGui::Button("Reset##temp")) {
                m_temperatureShift = 0.0f;
            }

            ImGui::TreePop();
        }

        ImGui::Separator();

        // === SECTION 5: Quick Actions ===
        if (ImGui::Button("Copy Light 0 Color to All")) {
            if (m_lights.size() > 0) {
                DirectX::XMFLOAT3 color = m_lights[0].color;
                auto indices = GetSelectedLightIndices();
                for (int idx : indices) {
                    m_lights[idx].color = color;
                }
                m_currentColorPreset = ColorPreset::Custom;
            }
        }

        if (ImGui::Button("Randomize Colors")) {
            auto indices = GetSelectedLightIndices();
            for (int idx : indices) {
                float h = (float)(rand() % 360);
                float s = 0.8f + (float)(rand() % 20) / 100.0f;  // 0.8-1.0
                float v = 0.9f + (float)(rand() % 10) / 100.0f;  // 0.9-1.0
                m_lights[idx].color = HSVtoRGB(DirectX::XMFLOAT3(h, s, v));
            }
            m_currentColorPreset = ColorPreset::Custom;
        }
    }
}
```

---

## Use Cases & Examples

### Use Case 1: Quick Warm/Cool Shift

**Scenario:** User wants to try warm vs cool lighting

**Workflow:**
1. Click "Warm Sunset (2500K)" → All lights warm orange
2. Click "Cool Blue (10000K)" → All lights cool blue
3. Click "White (6500K)" → All lights neutral white

**Time:** 3 seconds per change

---

### Use Case 2: Accretion Disk Radial Gradient

**Scenario:** Inner disk blue/white, outer disk red

**Workflow:**
1. Select "All Lights"
2. Open "Gradient Application"
3. Select "Radial (Distance)"
4. Set Start Color: (0.8, 0.9, 1.0) - Blue/white
5. Set End Color: (1.0, 0.3, 0.2) - Red
6. Click "Apply Gradient"

**Result:** Smooth color transition from inner (blue) to outer (red)

---

### Use Case 3: Binary Star Two-Tone

**Scenario:** Half lights blue (star 1), half lights red (star 2)

**Workflow:**
1. Click "Binary System" preset

**Result:** Instant two-tone coloring (blue + red)

---

### Use Case 4: Rainbow Ring

**Scenario:** Rainbow color distribution around disk

**Workflow:**
1. Select "All Lights"
2. Click "Rainbow" preset

**Result:** Evenly distributed rainbow colors (red → orange → yellow → green → blue → magenta)

---

## Performance Impact

### Runtime Cost

**Negligible** - color changes only update CPU-side light array, no GPU pipeline changes.

**Worst case:** Applying gradient to 16 lights
- RGB→HSV conversion: 16 × 10 ops = 160 ops
- Interpolation: 16 × 3 lerps = 48 ops
- HSV→RGB conversion: 16 × 10 ops = 160 ops
- Total: ~400 operations (~1 microsecond on modern CPU)

**No FPS impact** - color upload happens once per frame regardless of changes.

---

## Implementation Checklist

### Phase 1: Core Infrastructure (2 hours)
- ✅ Add color preset enum to Application.h
- ✅ Add gradient type enum
- ✅ Add selection mode enum
- ✅ Implement RGBtoHSV() and HSVtoRGB()
- ✅ Implement BlackbodyColor()
- ✅ Implement GetSelectedLightIndices()
- ✅ Test: Build succeeds, no visual change

### Phase 2: Preset System (1 hour)
- ✅ Implement ApplyColorPreset() for all presets
- ✅ Test: Each preset applies correct colors
- ✅ Test: Selection modes work (All, Inner, Outer, etc.)

### Phase 3: Gradient System (30 minutes)
- ✅ Implement ApplyGradient() for all gradient types
- ✅ Test: Radial gradient (inner→outer)
- ✅ Test: Linear gradients (X, Y, Z axes)
- ✅ Test: Circular gradient (rainbow ring)

### Phase 4: Global Operations (30 minutes)
- ✅ Implement ApplyGlobalHueShift()
- ✅ Implement ApplyGlobalSaturationAdjust()
- ✅ Implement ApplyGlobalValueAdjust()
- ✅ Implement ApplyTemperatureShift()
- ✅ Test: Each operation produces expected result

### Phase 5: ImGui UI (1 hour)
- ✅ Add "Bulk Light Color Controls" collapsing header
- ✅ Add selection controls
- ✅ Add preset buttons (all 17 presets)
- ✅ Add gradient controls
- ✅ Add global operation controls
- ✅ Add quick action buttons
- ✅ Test: All buttons work, UI responsive

**Total Time:** 5 hours (conservative estimate: 3-4 hours realistic)

---

## Success Criteria

**Bulk color control system is complete when:**

1. ✅ All 17 color presets apply correctly
2. ✅ All 5 gradient types work as expected
3. ✅ All 8 selection modes filter lights correctly
4. ✅ All 4 global operations produce expected results
5. ✅ ImGui UI is intuitive and responsive
6. ✅ User confirms: "Color changes are 20× faster than before!"

**Definition of Done:**
- Side-by-side comparison: Manual color change vs preset (time saved)
- User can create custom gradients and presets
- All scenario presets match animation scenarios
- Documentation updated (CLAUDE.md)

---

## Future Enhancements (Phase 6+)

### Advanced Color Features

**1. Animated Color Gradients:**
- Time-based color shifts
- Smooth transitions between presets
- Breathing/pulsing effects

**2. Color Palette Save/Load:**
- Save current light colors as custom preset
- Load custom preset JSON files
- Share presets between users

**3. Color Picker for Gradients:**
- Visual gradient editor (drag handles)
- Live preview of gradient on lights
- Bezier curve interpolation (non-linear gradients)

**4. Light Color Keyframing:**
- Keyframe light colors at specific times
- Animate smooth transitions
- Export as animation timeline

---

**Document Status:** Complete - Ready for Implementation
**Next Steps:** Begin implementation after particle type system complete
