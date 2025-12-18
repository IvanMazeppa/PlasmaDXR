#pragma once

#include <DirectXMath.h>
#include <vector>
#include <cstdint>
#include "ParticleRenderer_Gaussian.h"

// LuminousParticleSystem - Manages physics-driven star lights inside Gaussian particles
// Part of Phase 3.9: Luminous Star Particles feature
// Embeds point lights inside 3D Gaussian particles to create supergiant stars

class LuminousParticleSystem {
public:
    static constexpr uint32_t MAX_STAR_PARTICLES = 16;

    // Binding between a particle and its embedded light source
    struct StarParticleBinding {
        uint32_t particleIndex;
        DirectX::XMFLOAT3 position;
        DirectX::XMFLOAT3 velocity;
        float temperature;      // Kelvin (determines light color via Wien's law)
        float luminosity;       // Light intensity multiplier
        float lightRadius;      // Light falloff radius
        bool active;
    };

    // Star type presets for different stellar classes
    enum class StarPreset {
        BLUE_SUPERGIANT,   // 25000K, intensity 15.0, O/B-class
        RED_GIANT,         // 4000K, intensity 8.0, M-class
        WHITE_DWARF,       // 10000K, intensity 3.0, compact
        MAIN_SEQUENCE      // 6000K, intensity 5.0, G-class (Sun-like)
    };

    LuminousParticleSystem() = default;
    ~LuminousParticleSystem() = default;

    // Initialize with specified number of star particles (default 16)
    bool Initialize(uint32_t starCount = 16);

    // Update star light positions using CPU Keplerian orbit prediction
    // Call AFTER particle physics update each frame
    void Update(float deltaTime, float physicsTimeMultiplier);

    // Access star lights for merging with static lights before GPU upload
    const std::vector<ParticleRenderer_Gaussian::Light>& GetStarLights() const { return m_starLights; }
    uint32_t GetActiveStarCount() const;

    // Runtime enable/disable
    void SetEnabled(bool enabled) { m_enabled = enabled; }
    bool IsEnabled() const { return m_enabled; }

    // Global luminosity multiplier (affects all star lights)
    void SetGlobalLuminosity(float mult);
    float GetGlobalLuminosity() const { return m_globalLuminosity; }

    // Global opacity for star particles (affects material, not lights)
    void SetGlobalOpacity(float opacity) { m_globalOpacity = opacity; }
    float GetGlobalOpacity() const { return m_globalOpacity; }

    // Star spawning patterns
    void SpawnSpiralArmStars(float radius = 200.0f);
    void SpawnDiskHotspots(float minR = 100.0f, float maxR = 400.0f);
    void RespawnAllStars();

    // Debug info for ImGui display
    const std::vector<StarParticleBinding>& GetStarBindings() const { return m_starBindings; }

private:
    // Convert temperature in Kelvin to RGB light color using Wien's law approximation
    DirectX::XMFLOAT3 TemperatureToLightColor(float kelvin) const;

    // Initialize star positions using Fibonacci sphere distribution
    void InitializeStarPositions();

    // Apply Keplerian orbital dynamics (must match GPU physics shader)
    void UpdateKeplerianOrbits(float dt);

    // Get preset configuration for a star type
    void ApplyStarPreset(StarParticleBinding& binding, StarPreset preset);

private:
    std::vector<StarParticleBinding> m_starBindings;
    std::vector<ParticleRenderer_Gaussian::Light> m_starLights;

    bool m_enabled = true;
    bool m_initialized = false;
    float m_globalLuminosity = 1.0f;
    float m_globalOpacity = 0.15f;  // Very transparent - light shines through

    // Physics constants (MUST match GPU particle_physics.hlsl)
    static constexpr float GM = 100.0f;           // Gravitational parameter
    static constexpr float PI = 3.14159265359f;
    static constexpr float GOLDEN_RATIO = 1.6180339887f;
};
