#include "LuminousParticleSystem.h"
#include "../utils/Logger.h"
#include <cmath>
#include <algorithm>

bool LuminousParticleSystem::Initialize(uint32_t starCount) {
    if (starCount > MAX_STAR_PARTICLES) {
        LOG_WARN("LuminousParticleSystem: Requested {} stars, clamping to max {}",
                 starCount, MAX_STAR_PARTICLES);
        starCount = MAX_STAR_PARTICLES;
    }

    m_starBindings.resize(starCount);
    m_starLights.resize(starCount);

    // Initialize star positions using Fibonacci sphere distribution
    InitializeStarPositions();

    m_initialized = true;
    LOG_INFO("LuminousParticleSystem: Initialized {} star particles", starCount);
    return true;
}

void LuminousParticleSystem::InitializeStarPositions() {
    const float phi = GOLDEN_RATIO;
    const uint32_t count = static_cast<uint32_t>(m_starBindings.size());

    for (uint32_t i = 0; i < count; i++) {
        auto& binding = m_starBindings[i];
        auto& light = m_starLights[i];

        binding.particleIndex = i;
        binding.active = true;

        // Fibonacci sphere distribution for even spacing
        float theta = 2.0f * PI * i / phi;
        float y = 1.0f - (2.0f * i + 1.0f) / (2.0f * count);
        float radiusAtY = sqrtf(1.0f - y * y);

        // Orbital radius varies by index (150-300 units)
        float orbitalRadius = 150.0f + (i % 4) * 50.0f;

        binding.position = {
            cosf(theta) * radiusAtY * orbitalRadius,
            y * 20.0f,  // Disk thickness
            sinf(theta) * radiusAtY * orbitalRadius
        };

        // Initialize Keplerian orbital velocity (perpendicular to radial)
        float r = sqrtf(binding.position.x * binding.position.x +
                        binding.position.y * binding.position.y +
                        binding.position.z * binding.position.z);
        if (r > 1.0f) {
            float orbitalSpeed = sqrtf(GM / r);
            // Perpendicular in XZ plane (counter-clockwise)
            binding.velocity = {
                -binding.position.z / r * orbitalSpeed,
                0.0f,
                binding.position.x / r * orbitalSpeed
            };
        } else {
            binding.velocity = { 0.0f, 0.0f, 0.0f };
        }

        // Default to blue supergiant
        ApplyStarPreset(binding, StarPreset::BLUE_SUPERGIANT);

        // Initialize light from binding
        light.position = binding.position;
        light.intensity = binding.luminosity * m_globalLuminosity;
        light.color = TemperatureToLightColor(binding.temperature);
        light.radius = binding.lightRadius;

        // Disable god rays for star lights
        light.enableGodRays = 0.0f;
        light.godRayIntensity = 0.0f;
        light.godRayLength = 0.0f;
        light.godRayFalloff = 1.0f;
        light.godRayDirection = { 0.0f, -1.0f, 0.0f };
        light.godRayConeAngle = 0.0f;
        light.godRayRotationSpeed = 0.0f;
        light._padding = 0.0f;
    }
}

void LuminousParticleSystem::ApplyStarPreset(StarParticleBinding& binding, StarPreset preset) {
    switch (preset) {
        case StarPreset::BLUE_SUPERGIANT:
            binding.temperature = 25000.0f;
            binding.luminosity = 15.0f;
            binding.lightRadius = 150.0f;
            break;
        case StarPreset::RED_GIANT:
            binding.temperature = 4000.0f;
            binding.luminosity = 8.0f;
            binding.lightRadius = 200.0f;
            break;
        case StarPreset::WHITE_DWARF:
            binding.temperature = 10000.0f;
            binding.luminosity = 3.0f;
            binding.lightRadius = 50.0f;
            break;
        case StarPreset::MAIN_SEQUENCE:
        default:
            binding.temperature = 6000.0f;
            binding.luminosity = 5.0f;
            binding.lightRadius = 120.0f;
            break;
    }
}

void LuminousParticleSystem::Update(float deltaTime, float physicsTimeMultiplier) {
    if (!m_enabled || !m_initialized) return;

    float dt = deltaTime * physicsTimeMultiplier;

    // Update orbital positions using Keplerian dynamics
    UpdateKeplerianOrbits(dt);

    // Sync light positions from star bindings
    for (size_t i = 0; i < m_starBindings.size(); i++) {
        auto& binding = m_starBindings[i];
        auto& light = m_starLights[i];

        if (!binding.active) continue;

        light.position = binding.position;
        light.intensity = binding.luminosity * m_globalLuminosity;
        light.color = TemperatureToLightColor(binding.temperature);
        light.radius = binding.lightRadius;
    }
}

void LuminousParticleSystem::UpdateKeplerianOrbits(float dt) {
    for (auto& binding : m_starBindings) {
        if (!binding.active) continue;

        DirectX::XMFLOAT3& pos = binding.position;
        DirectX::XMFLOAT3& vel = binding.velocity;

        // Calculate radius from black hole
        float r = sqrtf(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z);
        if (r < 10.0f) r = 10.0f;  // Prevent singularity

        // Gravitational acceleration magnitude
        float accelMag = GM / (r * r);

        // Direction toward black hole (normalized)
        DirectX::XMFLOAT3 accelDir = {
            -pos.x / r,
            -pos.y / r,
            -pos.z / r
        };

        // Velocity Verlet integration for orbital dynamics
        // Position update: x += v*dt + 0.5*a*dt^2
        pos.x += vel.x * dt + 0.5f * accelDir.x * accelMag * dt * dt;
        pos.y += vel.y * dt + 0.5f * accelDir.y * accelMag * dt * dt;
        pos.z += vel.z * dt + 0.5f * accelDir.z * accelMag * dt * dt;

        // Velocity update: v += a*dt
        vel.x += accelDir.x * accelMag * dt;
        vel.y += accelDir.y * accelMag * dt;
        vel.z += accelDir.z * accelMag * dt;
    }
}

DirectX::XMFLOAT3 LuminousParticleSystem::TemperatureToLightColor(float kelvin) const {
    // Wien's law approximation for stellar colors
    float t = kelvin / 1000.0f;
    float r, g, b;

    if (t <= 6.6f) {
        // Cooler stars (red to yellow-white)
        r = 1.0f;
        g = std::max(0.0f, std::min(1.0f, 0.39f * logf(t) - 0.63f));
        b = std::max(0.0f, std::min(1.0f, 0.12f * logf(std::max(t - 1.0f, 0.01f))));
    } else {
        // Hotter stars (white to blue)
        r = std::max(0.0f, std::min(1.0f, 1.29f * powf(t - 6.0f, -0.133f)));
        g = std::max(0.0f, std::min(1.0f, 1.13f * powf(t - 6.0f, -0.075f)));
        b = 1.0f;
    }

    return { r, g, b };
}

uint32_t LuminousParticleSystem::GetActiveStarCount() const {
    uint32_t count = 0;
    for (const auto& binding : m_starBindings) {
        if (binding.active) count++;
    }
    return count;
}

void LuminousParticleSystem::SetGlobalLuminosity(float mult) {
    m_globalLuminosity = std::max(0.1f, std::min(mult, 10.0f));
}

void LuminousParticleSystem::SpawnSpiralArmStars(float radius) {
    // Place 4 stars at spiral arm positions (0, 90, 180, 270 degrees)
    const uint32_t armCount = 4;
    for (uint32_t i = 0; i < armCount && i < m_starBindings.size(); i++) {
        auto& binding = m_starBindings[i];
        float angle = (i / static_cast<float>(armCount)) * 2.0f * PI;

        binding.position = {
            cosf(angle) * radius,
            0.0f,
            sinf(angle) * radius
        };

        // Keplerian velocity
        float orbitalSpeed = sqrtf(GM / radius);
        binding.velocity = {
            -sinf(angle) * orbitalSpeed,
            0.0f,
            cosf(angle) * orbitalSpeed
        };

        binding.active = true;
        ApplyStarPreset(binding, StarPreset::BLUE_SUPERGIANT);
    }

    LOG_INFO("LuminousParticleSystem: Spawned {} spiral arm stars at radius {}",
             std::min(armCount, static_cast<uint32_t>(m_starBindings.size())), radius);
}

void LuminousParticleSystem::SpawnDiskHotspots(float minR, float maxR) {
    // Scatter stars across the disk at various radii
    for (size_t i = 4; i < m_starBindings.size(); i++) {
        auto& binding = m_starBindings[i];

        float t = static_cast<float>(i - 4) / static_cast<float>(m_starBindings.size() - 4);
        float radius = minR + t * (maxR - minR);
        float angle = static_cast<float>(i) * GOLDEN_RATIO * 2.0f * PI;

        binding.position = {
            cosf(angle) * radius,
            (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 10.0f,  // Slight vertical offset
            sinf(angle) * radius
        };

        // Keplerian velocity
        float orbitalSpeed = sqrtf(GM / radius);
        binding.velocity = {
            -sinf(angle) * orbitalSpeed,
            0.0f,
            cosf(angle) * orbitalSpeed
        };

        binding.active = true;

        // Vary star types
        if (i % 3 == 0) {
            ApplyStarPreset(binding, StarPreset::RED_GIANT);
        } else if (i % 5 == 0) {
            ApplyStarPreset(binding, StarPreset::WHITE_DWARF);
        } else {
            ApplyStarPreset(binding, StarPreset::MAIN_SEQUENCE);
        }
    }

    LOG_INFO("LuminousParticleSystem: Spawned disk hotspot stars between radius {} and {}", minR, maxR);
}

void LuminousParticleSystem::RespawnAllStars() {
    InitializeStarPositions();
    LOG_INFO("LuminousParticleSystem: Respawned all {} stars", m_starBindings.size());
}
