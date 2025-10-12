#pragma once

#include <string>
#include <fstream>
#include <optional>
#include "../../external/json.hpp"
#include "../core/Logger.h"

using json = nlohmann::json;

namespace PlasmaDX {

struct RenderingConfig {
    uint32_t particleCount = 10000;
    std::string renderer = "Gaussian";  // "Gaussian" or "Billboard"
    bool enableRT = true;
    bool preferMeshShaders = true;
    float baseParticleRadius = 50.0f;
};

struct CameraConfig {
    struct Vector3 {
        float x = 0.0f;
        float y = 1200.0f;
        float z = 800.0f;
    };
    Vector3 position{0.0f, 1200.0f, 800.0f};
    Vector3 lookAt{0.0f, 0.0f, 0.0f};
};

struct PhysicsConfig {
    float innerRadius = 10.0f;
    float outerRadius = 300.0f;
    float diskThickness = 50.0f;
    float gravity = 200.0f;
    float angularMomentum = 2.5f;
    float turbulence = 10.0f;
    float damping = 0.95f;
};

struct GaussianRTConfig {
    bool useShadowRays = true;
    bool useInScattering = false;
    bool usePhaseFunction = true;
    float phaseStrength = 5.0f;
    float inScatterStrength = 1.0f;
    float rtLightingStrength = 2.0f;
    bool useReSTIR = false;
    uint32_t restirInitialCandidates = 16;
    float restirTemporalWeight = 0.95f;
    bool useAnisotropicGaussians = true;
    float anisotropyStrength = 2.0f;
};

struct PIXConfig {
    bool autoCapture = false;
    uint32_t captureFrame = 1;
    std::string outputPath = "pix/Captures/auto_capture.wpix";
};

struct DebugConfig {
    bool logPhysicsUpdates = false;
    bool logRenderingStats = false;
    uint32_t maxLoggedFrames = 10;
};

struct AppConfig {
    RenderingConfig rendering;
    CameraConfig camera;
    PhysicsConfig physics;
    GaussianRTConfig gaussianRT;
    PIXConfig pix;
    DebugConfig debug;
};

class ConfigLoader {
public:
    static std::optional<AppConfig> Load(const std::string& configPath = "config.json") {
        try {
            std::ifstream file(configPath);
            if (!file.is_open()) {
                LOG_WARN("Config file not found: {}. Using defaults.", configPath);
                return std::nullopt;
            }

            json j;
            file >> j;

            AppConfig config;

            // Parse rendering config
            if (j.contains("rendering")) {
                auto& r = j["rendering"];
                if (r.contains("particleCount")) config.rendering.particleCount = r["particleCount"];
                if (r.contains("renderer")) config.rendering.renderer = r["renderer"];
                if (r.contains("enableRT")) config.rendering.enableRT = r["enableRT"];
                if (r.contains("preferMeshShaders")) config.rendering.preferMeshShaders = r["preferMeshShaders"];
                if (r.contains("baseParticleRadius")) config.rendering.baseParticleRadius = r["baseParticleRadius"];
            }

            // Parse camera config
            if (j.contains("camera")) {
                auto& c = j["camera"];
                if (c.contains("position")) {
                    config.camera.position.x = c["position"]["x"];
                    config.camera.position.y = c["position"]["y"];
                    config.camera.position.z = c["position"]["z"];
                }
                if (c.contains("lookAt")) {
                    config.camera.lookAt.x = c["lookAt"]["x"];
                    config.camera.lookAt.y = c["lookAt"]["y"];
                    config.camera.lookAt.z = c["lookAt"]["z"];
                }
            }

            // Parse physics config
            if (j.contains("physics")) {
                auto& p = j["physics"];
                if (p.contains("innerRadius")) config.physics.innerRadius = p["innerRadius"];
                if (p.contains("outerRadius")) config.physics.outerRadius = p["outerRadius"];
                if (p.contains("diskThickness")) config.physics.diskThickness = p["diskThickness"];
                if (p.contains("gravity")) config.physics.gravity = p["gravity"];
                if (p.contains("angularMomentum")) config.physics.angularMomentum = p["angularMomentum"];
                if (p.contains("turbulence")) config.physics.turbulence = p["turbulence"];
                if (p.contains("damping")) config.physics.damping = p["damping"];
            }

            // Parse Gaussian RT config
            if (j.contains("gaussianRT")) {
                auto& g = j["gaussianRT"];
                if (g.contains("useShadowRays")) config.gaussianRT.useShadowRays = g["useShadowRays"];
                if (g.contains("useInScattering")) config.gaussianRT.useInScattering = g["useInScattering"];
                if (g.contains("usePhaseFunction")) config.gaussianRT.usePhaseFunction = g["usePhaseFunction"];
                if (g.contains("phaseStrength")) config.gaussianRT.phaseStrength = g["phaseStrength"];
                if (g.contains("inScatterStrength")) config.gaussianRT.inScatterStrength = g["inScatterStrength"];
                if (g.contains("rtLightingStrength")) config.gaussianRT.rtLightingStrength = g["rtLightingStrength"];
                if (g.contains("useReSTIR")) config.gaussianRT.useReSTIR = g["useReSTIR"];
                if (g.contains("restirInitialCandidates")) config.gaussianRT.restirInitialCandidates = g["restirInitialCandidates"];
                if (g.contains("restirTemporalWeight")) config.gaussianRT.restirTemporalWeight = g["restirTemporalWeight"];
                if (g.contains("useAnisotropicGaussians")) config.gaussianRT.useAnisotropicGaussians = g["useAnisotropicGaussians"];
                if (g.contains("anisotropyStrength")) config.gaussianRT.anisotropyStrength = g["anisotropyStrength"];
            }

            // Parse PIX config
            if (j.contains("pix")) {
                auto& p = j["pix"];
                if (p.contains("autoCapture")) config.pix.autoCapture = p["autoCapture"];
                if (p.contains("captureFrame")) config.pix.captureFrame = p["captureFrame"];
                if (p.contains("outputPath")) config.pix.outputPath = p["outputPath"];
            }

            // Parse debug config
            if (j.contains("debug")) {
                auto& d = j["debug"];
                if (d.contains("logPhysicsUpdates")) config.debug.logPhysicsUpdates = d["logPhysicsUpdates"];
                if (d.contains("logRenderingStats")) config.debug.logRenderingStats = d["logRenderingStats"];
                if (d.contains("maxLoggedFrames")) config.debug.maxLoggedFrames = d["maxLoggedFrames"];
            }

            LOG_INFO("Config loaded successfully from {}", configPath);
            return config;

        } catch (const std::exception& e) {
            LOG_ERROR("Failed to parse config file: {}", e.what());
            return std::nullopt;
        }
    }

    static bool Save(const AppConfig& config, const std::string& configPath = "config.json") {
        try {
            json j;

            // Rendering
            j["rendering"]["particleCount"] = config.rendering.particleCount;
            j["rendering"]["renderer"] = config.rendering.renderer;
            j["rendering"]["enableRT"] = config.rendering.enableRT;
            j["rendering"]["preferMeshShaders"] = config.rendering.preferMeshShaders;
            j["rendering"]["baseParticleRadius"] = config.rendering.baseParticleRadius;

            // Camera
            j["camera"]["position"]["x"] = config.camera.position.x;
            j["camera"]["position"]["y"] = config.camera.position.y;
            j["camera"]["position"]["z"] = config.camera.position.z;
            j["camera"]["lookAt"]["x"] = config.camera.lookAt.x;
            j["camera"]["lookAt"]["y"] = config.camera.lookAt.y;
            j["camera"]["lookAt"]["z"] = config.camera.lookAt.z;

            // Physics
            j["physics"]["innerRadius"] = config.physics.innerRadius;
            j["physics"]["outerRadius"] = config.physics.outerRadius;
            j["physics"]["diskThickness"] = config.physics.diskThickness;
            j["physics"]["gravity"] = config.physics.gravity;
            j["physics"]["angularMomentum"] = config.physics.angularMomentum;
            j["physics"]["turbulence"] = config.physics.turbulence;
            j["physics"]["damping"] = config.physics.damping;

            // Gaussian RT
            j["gaussianRT"]["useShadowRays"] = config.gaussianRT.useShadowRays;
            j["gaussianRT"]["useInScattering"] = config.gaussianRT.useInScattering;
            j["gaussianRT"]["usePhaseFunction"] = config.gaussianRT.usePhaseFunction;
            j["gaussianRT"]["phaseStrength"] = config.gaussianRT.phaseStrength;
            j["gaussianRT"]["inScatterStrength"] = config.gaussianRT.inScatterStrength;
            j["gaussianRT"]["rtLightingStrength"] = config.gaussianRT.rtLightingStrength;
            j["gaussianRT"]["useReSTIR"] = config.gaussianRT.useReSTIR;
            j["gaussianRT"]["restirInitialCandidates"] = config.gaussianRT.restirInitialCandidates;
            j["gaussianRT"]["restirTemporalWeight"] = config.gaussianRT.restirTemporalWeight;
            j["gaussianRT"]["useAnisotropicGaussians"] = config.gaussianRT.useAnisotropicGaussians;
            j["gaussianRT"]["anisotropyStrength"] = config.gaussianRT.anisotropyStrength;

            // PIX
            j["pix"]["autoCapture"] = config.pix.autoCapture;
            j["pix"]["captureFrame"] = config.pix.captureFrame;
            j["pix"]["outputPath"] = config.pix.outputPath;

            // Debug
            j["debug"]["logPhysicsUpdates"] = config.debug.logPhysicsUpdates;
            j["debug"]["logRenderingStats"] = config.debug.logRenderingStats;
            j["debug"]["maxLoggedFrames"] = config.debug.maxLoggedFrames;

            std::ofstream file(configPath);
            file << j.dump(2);  // Pretty print with 2-space indent

            LOG_INFO("Config saved to {}", configPath);
            return true;

        } catch (const std::exception& e) {
            LOG_ERROR("Failed to save config: {}", e.what());
            return false;
        }
    }
};

} // namespace PlasmaDX