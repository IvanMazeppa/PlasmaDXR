#include "Config.h"
#include "../utils/Logger.h"
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>

// Simple JSON parser for config files
// This is a lightweight implementation - for more complex needs, consider nlohmann/json

namespace Config {

// Helper function to trim whitespace
static std::string Trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

// Simple JSON value extractor (handles strings, numbers, bools)
static std::string GetJSONValue(const std::string& json, const std::string& key) {
    std::string searchKey = "\"" + key + "\"";
    size_t pos = json.find(searchKey);
    if (pos == std::string::npos) return "";

    pos = json.find(':', pos);
    if (pos == std::string::npos) return "";
    pos++; // Skip ':'

    // Skip whitespace
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

    // Extract value
    std::string value;
    if (json[pos] == '"') {
        // String value
        pos++; // Skip opening quote
        size_t end = json.find('"', pos);
        if (end != std::string::npos) {
            value = json.substr(pos, end - pos);
        }
    } else {
        // Number or bool
        size_t end = json.find_first_of(",}\n\r", pos);
        if (end != std::string::npos) {
            value = Trim(json.substr(pos, end - pos));
        }
    }

    return value;
}

// Helper to get int value
static int GetJSONInt(const std::string& json, const std::string& key, int defaultValue) {
    std::string value = GetJSONValue(json, key);
    if (value.empty()) return defaultValue;
    try {
        return std::stoi(value);
    } catch (...) {
        return defaultValue;
    }
}

// Helper to get float value
static float GetJSONFloat(const std::string& json, const std::string& key, float defaultValue) {
    std::string value = GetJSONValue(json, key);
    if (value.empty()) return defaultValue;
    try {
        return std::stof(value);
    } catch (...) {
        return defaultValue;
    }
}

// Helper to get bool value
static bool GetJSONBool(const std::string& json, const std::string& key, bool defaultValue) {
    std::string value = GetJSONValue(json, key);
    if (value.empty()) return defaultValue;
    return (value == "true" || value == "1");
}

// Get nested JSON object
static std::string GetJSONObject(const std::string& json, const std::string& key) {
    std::string searchKey = "\"" + key + "\"";
    size_t pos = json.find(searchKey);
    if (pos == std::string::npos) return "";

    pos = json.find('{', pos);
    if (pos == std::string::npos) return "";

    // Find matching closing brace
    int braceCount = 1;
    size_t start = pos + 1;
    pos++;

    while (pos < json.size() && braceCount > 0) {
        if (json[pos] == '{') braceCount++;
        else if (json[pos] == '}') braceCount--;
        pos++;
    }

    if (braceCount == 0) {
        return json.substr(start, pos - start - 1);
    }

    return "";
}

ConfigManager::ConfigManager() {
    // Set dev defaults
    SetProfileDefaults(Profile::Dev);
}

ConfigManager::~ConfigManager() {
}

bool ConfigManager::Initialize(int argc, char** argv) {
    // Check command-line arguments
    std::string configPath;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.find("--config=") == 0) {
            configPath = arg.substr(9); // Skip "--config="
            break;
        }
    }

    // Check environment variable if no command-line arg
    if (configPath.empty()) {
        char* envConfig = std::getenv("PLASMADX_CONFIG");
        if (envConfig != nullptr) {
            configPath = envConfig;
        }
    }

    // Try to load specified config
    if (!configPath.empty()) {
        LOG_INFO("Loading config from: {}", configPath);
        if (LoadFromFile(configPath)) {
            return true;
        }
        LOG_WARN("Failed to load config '{}', trying defaults...", configPath);
    }

    // Try default config_dev.json
    if (std::filesystem::exists("config_dev.json")) {
        LOG_INFO("Loading default config: config_dev.json");
        if (LoadFromFile("config_dev.json")) {
            return true;
        }
    }

    // Fall back to hardcoded defaults
    LOG_INFO("Using hardcoded default configuration (dev profile)");
    SetProfileDefaults(Profile::Dev);
    return true;
}

bool ConfigManager::LoadFromFile(const std::string& filepath) {
    // Try multiple search paths to handle different working directories
    std::vector<std::string> searchPaths = {
        filepath,                                    // 1. Exact path specified
        "configs/" + filepath,                        // 2. configs/ subdirectory
        "../../../" + filepath,                       // 3. Project root (from build/bin/Debug/)
        "../../../configs/" + std::filesystem::path(filepath).filename().string()  // 4. Project configs/ folder
    };

    std::string resolvedPath;
    for (const auto& path : searchPaths) {
        if (std::filesystem::exists(path)) {
            resolvedPath = path;
            LOG_INFO("Found config at: {}", path);
            break;
        }
    }

    if (resolvedPath.empty()) {
        LOG_ERROR("Config file not found in any search path: {}", filepath);
        LOG_ERROR("Searched paths:");
        for (const auto& path : searchPaths) {
            LOG_ERROR("  - {}", path);
        }
        return false;
    }

    std::ifstream file(resolvedPath);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open config file: {}", resolvedPath);
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json = buffer.str();
    file.close();

    m_config.configFilePath = resolvedPath;  // Store resolved path
    return ParseJSON(json);
}

bool ConfigManager::ParseJSON(const std::string& json) {
    // Parse profile
    std::string profileName = GetJSONValue(json, "profile");
    if (!profileName.empty()) {
        m_config.profileName = profileName;
        if (profileName == "dev") {
            m_config.profile = Profile::Dev;
        } else if (profileName == "user") {
            m_config.profile = Profile::User;
        } else if (profileName == "pix_analysis") {
            m_config.profile = Profile::PIXAnalysis;
        } else {
            m_config.profile = Profile::Custom;
        }
    }

    // Parse rendering section
    std::string renderingJSON = GetJSONObject(json, "rendering");
    if (!renderingJSON.empty()) {
        m_config.rendering.particleCount = GetJSONInt(renderingJSON, "particleCount", m_config.rendering.particleCount);
        m_config.rendering.resolutionWidth = GetJSONInt(renderingJSON, "width", m_config.rendering.resolutionWidth);
        m_config.rendering.resolutionHeight = GetJSONInt(renderingJSON, "height", m_config.rendering.resolutionHeight);

        std::string rendererType = GetJSONValue(renderingJSON, "rendererType");
        if (rendererType == "gaussian") {
            m_config.rendering.rendererType = RendererType::Gaussian;
        } else if (rendererType == "billboard") {
            m_config.rendering.rendererType = RendererType::Billboard;
        }
    }

    // Parse features section
    std::string featuresJSON = GetJSONObject(json, "features");
    if (!featuresJSON.empty()) {
        m_config.features.enableReSTIR = GetJSONBool(featuresJSON, "enableReSTIR", m_config.features.enableReSTIR);
        m_config.features.restirCandidates = GetJSONInt(featuresJSON, "restirCandidates", m_config.features.restirCandidates);
        m_config.features.restirTemporalReuse = GetJSONBool(featuresJSON, "restirTemporalReuse", m_config.features.restirTemporalReuse);
        m_config.features.restirSpatialReuse = GetJSONBool(featuresJSON, "restirSpatialReuse", m_config.features.restirSpatialReuse);
        m_config.features.restirTemporalWeight = GetJSONFloat(featuresJSON, "restirTemporalWeight", m_config.features.restirTemporalWeight);

        m_config.features.enableInScattering = GetJSONBool(featuresJSON, "enableInScattering", m_config.features.enableInScattering);
        m_config.features.inScatterStrength = GetJSONFloat(featuresJSON, "inScatterStrength", m_config.features.inScatterStrength);

        m_config.features.enableShadowRays = GetJSONBool(featuresJSON, "enableShadowRays", m_config.features.enableShadowRays);
        m_config.features.enablePhaseFunction = GetJSONBool(featuresJSON, "enablePhaseFunction", m_config.features.enablePhaseFunction);
        m_config.features.phaseStrength = GetJSONFloat(featuresJSON, "phaseStrength", m_config.features.phaseStrength);

        m_config.features.useAnisotropicGaussians = GetJSONBool(featuresJSON, "useAnisotropicGaussians", m_config.features.useAnisotropicGaussians);
        m_config.features.anisotropyStrength = GetJSONFloat(featuresJSON, "anisotropyStrength", m_config.features.anisotropyStrength);

        m_config.features.rtLightingStrength = GetJSONFloat(featuresJSON, "rtLightingStrength", m_config.features.rtLightingStrength);

        m_config.features.usePhysicalEmission = GetJSONBool(featuresJSON, "usePhysicalEmission", m_config.features.usePhysicalEmission);
        m_config.features.emissionStrength = GetJSONFloat(featuresJSON, "emissionStrength", m_config.features.emissionStrength);

        m_config.features.useDopplerShift = GetJSONBool(featuresJSON, "useDopplerShift", m_config.features.useDopplerShift);
        m_config.features.dopplerStrength = GetJSONFloat(featuresJSON, "dopplerStrength", m_config.features.dopplerStrength);

        m_config.features.useGravitationalRedshift = GetJSONBool(featuresJSON, "useGravitationalRedshift", m_config.features.useGravitationalRedshift);
        m_config.features.redshiftStrength = GetJSONFloat(featuresJSON, "redshiftStrength", m_config.features.redshiftStrength);
    }

    // Parse lighting section
    std::string lightingJSON = GetJSONObject(json, "lighting");
    if (!lightingJSON.empty()) {
        // Primary lighting system
        std::string systemName = GetJSONValue(lightingJSON, "system");
        if (!systemName.empty()) {
            m_config.lighting.system = systemName;
        }

        // Multi-light settings
        std::string multiLightJSON = GetJSONObject(lightingJSON, "multiLight");
        if (!multiLightJSON.empty()) {
            m_config.lighting.multiLightEnabled = GetJSONBool(multiLightJSON, "enabled", m_config.lighting.multiLightEnabled);
            m_config.lighting.lightCount = GetJSONInt(multiLightJSON, "lightCount", m_config.lighting.lightCount);
            std::string preset = GetJSONValue(multiLightJSON, "preset");
            if (!preset.empty()) {
                m_config.lighting.multiLightPreset = preset;
            }
            m_config.lighting.multiLightIntensity = GetJSONFloat(multiLightJSON, "intensity", m_config.lighting.multiLightIntensity);
        }

        // Probe grid settings
        std::string probeGridJSON = GetJSONObject(lightingJSON, "probeGrid");
        if (!probeGridJSON.empty()) {
            m_config.lighting.probeGridEnabled = GetJSONBool(probeGridJSON, "enabled", m_config.lighting.probeGridEnabled);
            m_config.lighting.probeGridSize = GetJSONInt(probeGridJSON, "gridSize", m_config.lighting.probeGridSize);
            m_config.lighting.raysPerProbe = GetJSONInt(probeGridJSON, "raysPerProbe", m_config.lighting.raysPerProbe);
            m_config.lighting.probeGridIntensity = GetJSONFloat(probeGridJSON, "intensity", m_config.lighting.probeGridIntensity);
            m_config.lighting.probeUpdateInterval = GetJSONInt(probeGridJSON, "updateInterval", m_config.lighting.probeUpdateInterval);
        }

        // RTXDI settings
        std::string rtxdiJSON = GetJSONObject(lightingJSON, "rtxdi");
        if (!rtxdiJSON.empty()) {
            m_config.lighting.rtxdiEnabled = GetJSONBool(rtxdiJSON, "enabled", m_config.lighting.rtxdiEnabled);
            std::string mode = GetJSONValue(rtxdiJSON, "mode");
            if (!mode.empty()) {
                m_config.lighting.rtxdiMode = mode;
            }
            m_config.lighting.rtxdiTemporalWeight = GetJSONFloat(rtxdiJSON, "temporalWeight", m_config.lighting.rtxdiTemporalWeight);
        }
    }

    // Parse physics section
    std::string physicsJSON = GetJSONObject(json, "physics");
    if (!physicsJSON.empty()) {
        // Basic geometry
        m_config.physics.innerRadius = GetJSONFloat(physicsJSON, "innerRadius", m_config.physics.innerRadius);
        m_config.physics.outerRadius = GetJSONFloat(physicsJSON, "outerRadius", m_config.physics.outerRadius);
        m_config.physics.diskThickness = GetJSONFloat(physicsJSON, "diskThickness", m_config.physics.diskThickness);
        m_config.physics.timeStep = GetJSONFloat(physicsJSON, "timeStep", m_config.physics.timeStep);
        m_config.physics.physicsEnabled = GetJSONBool(physicsJSON, "physicsEnabled", m_config.physics.physicsEnabled);

        // GA-optimized parameters (Phase 5)
        m_config.physics.gm = GetJSONFloat(physicsJSON, "gm", m_config.physics.gm);
        m_config.physics.bh_mass = GetJSONFloat(physicsJSON, "bh_mass", m_config.physics.bh_mass);
        m_config.physics.alpha = GetJSONFloat(physicsJSON, "alpha", m_config.physics.alpha);
        m_config.physics.damping = GetJSONFloat(physicsJSON, "damping", m_config.physics.damping);
        m_config.physics.angular_boost = GetJSONFloat(physicsJSON, "angular_boost", m_config.physics.angular_boost);
        m_config.physics.density_scale = GetJSONFloat(physicsJSON, "density_scale", m_config.physics.density_scale);
        m_config.physics.force_clamp = GetJSONFloat(physicsJSON, "force_clamp", m_config.physics.force_clamp);
        m_config.physics.velocity_clamp = GetJSONFloat(physicsJSON, "velocity_clamp", m_config.physics.velocity_clamp);
        m_config.physics.boundary_mode = GetJSONInt(physicsJSON, "boundary_mode", m_config.physics.boundary_mode);
        m_config.physics.time_multiplier = GetJSONFloat(physicsJSON, "time_multiplier", m_config.physics.time_multiplier);
    }

    // Parse SIREN section
    std::string sirenJSON = GetJSONObject(json, "siren");
    if (!sirenJSON.empty()) {
        m_config.siren.enabled = GetJSONBool(sirenJSON, "enabled", m_config.siren.enabled);
        m_config.siren.intensity = GetJSONFloat(sirenJSON, "intensity", m_config.siren.intensity);
        m_config.siren.vortex_scale = GetJSONFloat(sirenJSON, "vortex_scale", m_config.siren.vortex_scale);
        m_config.siren.vortex_decay = GetJSONFloat(sirenJSON, "vortex_decay", m_config.siren.vortex_decay);
    }

    // Parse camera section
    std::string cameraJSON = GetJSONObject(json, "camera");
    if (!cameraJSON.empty()) {
        m_config.camera.startDistance = GetJSONFloat(cameraJSON, "startDistance", m_config.camera.startDistance);
        m_config.camera.startHeight = GetJSONFloat(cameraJSON, "startHeight", m_config.camera.startHeight);
        m_config.camera.startAngle = GetJSONFloat(cameraJSON, "startAngle", m_config.camera.startAngle);
        m_config.camera.startPitch = GetJSONFloat(cameraJSON, "startPitch", m_config.camera.startPitch);
        m_config.camera.moveSpeed = GetJSONFloat(cameraJSON, "moveSpeed", m_config.camera.moveSpeed);
        m_config.camera.rotateSpeed = GetJSONFloat(cameraJSON, "rotateSpeed", m_config.camera.rotateSpeed);
        m_config.camera.particleSize = GetJSONFloat(cameraJSON, "particleSize", m_config.camera.particleSize);
    }

    // Parse debug section
    std::string debugJSON = GetJSONObject(json, "debug");
    if (!debugJSON.empty()) {
        m_config.debug.enableDebugLayer = GetJSONBool(debugJSON, "enableDebugLayer", m_config.debug.enableDebugLayer);
        m_config.debug.enablePIX = GetJSONBool(debugJSON, "enablePIX", m_config.debug.enablePIX);
        m_config.debug.pixAutoCapture = GetJSONBool(debugJSON, "pixAutoCapture", m_config.debug.pixAutoCapture);
        m_config.debug.pixCaptureFrame = GetJSONInt(debugJSON, "pixCaptureFrame", m_config.debug.pixCaptureFrame);
        m_config.debug.showFPS = GetJSONBool(debugJSON, "showFPS", m_config.debug.showFPS);
        m_config.debug.showParticleStats = GetJSONBool(debugJSON, "showParticleStats", m_config.debug.showParticleStats);

        std::string logLevel = GetJSONValue(debugJSON, "logLevel");
        if (logLevel == "debug") {
            m_config.debug.logLevel = LogLevel::Debug;
        } else if (logLevel == "trace") {
            m_config.debug.logLevel = LogLevel::Trace;
        } else {
            m_config.debug.logLevel = LogLevel::Info;
        }
    }

    // Parse pix_analysis section
    std::string pixAnalysisJSON = GetJSONObject(json, "pix_analysis");
    if (!pixAnalysisJSON.empty()) {
        m_config.pixAnalysis.capturePrefix = GetJSONValue(pixAnalysisJSON, "capturePrefix");
        m_config.pixAnalysis.enableReservoirLogging = GetJSONBool(pixAnalysisJSON, "enableReservoirLogging", m_config.pixAnalysis.enableReservoirLogging);
        m_config.pixAnalysis.enablePerformanceCounters = GetJSONBool(pixAnalysisJSON, "enablePerformanceCounters", m_config.pixAnalysis.enablePerformanceCounters);
        m_config.pixAnalysis.trackResourceUsage = GetJSONBool(pixAnalysisJSON, "trackResourceUsage", m_config.pixAnalysis.trackResourceUsage);
    }

    m_config.loadedFromFile = true;

    LOG_INFO("=== Configuration Loaded ===");
    LOG_INFO("Profile: {}", m_config.profileName);
    LOG_INFO("Particles: {}", m_config.rendering.particleCount);
    LOG_INFO("Renderer: {}", m_config.rendering.rendererType == RendererType::Gaussian ? "Gaussian" : "Billboard");
    LOG_INFO("ReSTIR: {}", m_config.features.enableReSTIR ? "ENABLED" : "DISABLED");
    LOG_INFO("PIX: {}", m_config.debug.enablePIX ? "ENABLED" : "DISABLED");

    // Log physics parameters if non-default
    bool hasNonDefaultPhysics = (m_config.physics.gm != 100.0f) ||
                                (m_config.physics.bh_mass != 5.0f) ||
                                (m_config.physics.alpha != 0.1f) ||
                                (m_config.siren.intensity > 0.0f);
    if (hasNonDefaultPhysics) {
        LOG_INFO("Physics: GM={:.2f}, BH_Mass={:.2f}, Alpha={:.4f}, Time_Mult={:.1f}x",
                 m_config.physics.gm, m_config.physics.bh_mass,
                 m_config.physics.alpha, m_config.physics.time_multiplier);
        if (m_config.siren.intensity > 0.0f) {
            LOG_INFO("SIREN: Enabled, Intensity={:.3f}", m_config.siren.intensity);
        }
    }

    LOG_INFO("============================");

    return true;
}

bool ConfigManager::LoadProfile(Profile profile) {
    std::string path = GetProfilePath(profile);
    if (std::filesystem::exists(path)) {
        return LoadFromFile(path);
    }

    // Fall back to hardcoded defaults for this profile
    SetProfileDefaults(profile);
    return true;
}

void ConfigManager::SetProfileDefaults(Profile profile) {
    m_config.profile = profile;

    switch (profile) {
        case Profile::Dev:
            m_config.profileName = "dev";
            m_config.rendering.particleCount = 10000;
            m_config.rendering.rendererType = RendererType::Gaussian;
            m_config.features.enableReSTIR = false;
            m_config.features.enableInScattering = false;
            m_config.debug.enableDebugLayer = true;
            m_config.debug.enablePIX = false;
            m_config.debug.showFPS = true;
            break;

        case Profile::User:
            m_config.profileName = "user";
            m_config.rendering.particleCount = 20000;
            m_config.rendering.rendererType = RendererType::Gaussian;
            m_config.features.enableReSTIR = true;
            m_config.features.enableInScattering = false;
            m_config.debug.enableDebugLayer = false;
            m_config.debug.enablePIX = false;
            m_config.debug.showFPS = true;
            break;

        case Profile::PIXAnalysis:
            m_config.profileName = "pix_analysis";
            m_config.rendering.particleCount = 10000;
            m_config.rendering.rendererType = RendererType::Gaussian;
            m_config.features.enableReSTIR = true;
            m_config.features.enableInScattering = false;
            m_config.debug.enableDebugLayer = false;
            m_config.debug.enablePIX = true;
            m_config.debug.pixAutoCapture = true;
            m_config.debug.pixCaptureFrame = 120;
            m_config.pixAnalysis.enableReservoirLogging = true;
            m_config.pixAnalysis.enablePerformanceCounters = true;
            m_config.pixAnalysis.trackResourceUsage = true;
            break;

        case Profile::Custom:
            m_config.profileName = "custom";
            break;
    }
}

std::string ConfigManager::GetProfilePath(Profile profile) {
    switch (profile) {
        case Profile::Dev: return "config_dev.json";
        case Profile::User: return "config_user.json";
        case Profile::PIXAnalysis: return "config_pix_analysis.json";
        default: return "config.json";
    }
}

bool ConfigManager::SaveToFile(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        LOG_ERROR("Failed to create config file: {}", filepath);
        return false;
    }

    file << "{\n";
    file << "  \"profile\": \"" << m_config.profileName << "\",\n";
    file << "  \"rendering\": {\n";
    file << "    \"particleCount\": " << m_config.rendering.particleCount << ",\n";
    file << "    \"rendererType\": \"" << (m_config.rendering.rendererType == RendererType::Gaussian ? "gaussian" : "billboard") << "\",\n";
    file << "    \"width\": " << m_config.rendering.resolutionWidth << ",\n";
    file << "    \"height\": " << m_config.rendering.resolutionHeight << "\n";
    file << "  },\n";
    file << "  \"features\": {\n";
    file << "    \"enableReSTIR\": " << (m_config.features.enableReSTIR ? "true" : "false") << ",\n";
    file << "    \"restirCandidates\": " << m_config.features.restirCandidates << ",\n";
    file << "    \"restirTemporalReuse\": " << (m_config.features.restirTemporalReuse ? "true" : "false") << ",\n";
    file << "    \"restirSpatialReuse\": " << (m_config.features.restirSpatialReuse ? "true" : "false") << ",\n";
    file << "    \"restirTemporalWeight\": " << m_config.features.restirTemporalWeight << ",\n";
    file << "    \"enableInScattering\": " << (m_config.features.enableInScattering ? "true" : "false") << ",\n";
    file << "    \"inScatterStrength\": " << m_config.features.inScatterStrength << ",\n";
    file << "    \"enableShadowRays\": " << (m_config.features.enableShadowRays ? "true" : "false") << ",\n";
    file << "    \"enablePhaseFunction\": " << (m_config.features.enablePhaseFunction ? "true" : "false") << ",\n";
    file << "    \"phaseStrength\": " << m_config.features.phaseStrength << ",\n";
    file << "    \"useAnisotropicGaussians\": " << (m_config.features.useAnisotropicGaussians ? "true" : "false") << ",\n";
    file << "    \"anisotropyStrength\": " << m_config.features.anisotropyStrength << ",\n";
    file << "    \"rtLightingStrength\": " << m_config.features.rtLightingStrength << ",\n";
    file << "    \"usePhysicalEmission\": " << (m_config.features.usePhysicalEmission ? "true" : "false") << ",\n";
    file << "    \"emissionStrength\": " << m_config.features.emissionStrength << ",\n";
    file << "    \"useDopplerShift\": " << (m_config.features.useDopplerShift ? "true" : "false") << ",\n";
    file << "    \"dopplerStrength\": " << m_config.features.dopplerStrength << ",\n";
    file << "    \"useGravitationalRedshift\": " << (m_config.features.useGravitationalRedshift ? "true" : "false") << ",\n";
    file << "    \"redshiftStrength\": " << m_config.features.redshiftStrength << "\n";
    file << "  },\n";
    file << "  \"physics\": {\n";
    file << "    \"innerRadius\": " << m_config.physics.innerRadius << ",\n";
    file << "    \"outerRadius\": " << m_config.physics.outerRadius << ",\n";
    file << "    \"diskThickness\": " << m_config.physics.diskThickness << ",\n";
    file << "    \"timeStep\": " << m_config.physics.timeStep << ",\n";
    file << "    \"physicsEnabled\": " << (m_config.physics.physicsEnabled ? "true" : "false") << ",\n";
    file << "    \"gm\": " << m_config.physics.gm << ",\n";
    file << "    \"bh_mass\": " << m_config.physics.bh_mass << ",\n";
    file << "    \"alpha\": " << m_config.physics.alpha << ",\n";
    file << "    \"damping\": " << m_config.physics.damping << ",\n";
    file << "    \"angular_boost\": " << m_config.physics.angular_boost << ",\n";
    file << "    \"density_scale\": " << m_config.physics.density_scale << ",\n";
    file << "    \"force_clamp\": " << m_config.physics.force_clamp << ",\n";
    file << "    \"velocity_clamp\": " << m_config.physics.velocity_clamp << ",\n";
    file << "    \"boundary_mode\": " << m_config.physics.boundary_mode << ",\n";
    file << "    \"time_multiplier\": " << m_config.physics.time_multiplier << "\n";
    file << "  },\n";
    file << "  \"siren\": {\n";
    file << "    \"enabled\": " << (m_config.siren.enabled ? "true" : "false") << ",\n";
    file << "    \"intensity\": " << m_config.siren.intensity << ",\n";
    file << "    \"vortex_scale\": " << m_config.siren.vortex_scale << ",\n";
    file << "    \"vortex_decay\": " << m_config.siren.vortex_decay << "\n";
    file << "  },\n";
    file << "  \"camera\": {\n";
    file << "    \"startDistance\": " << m_config.camera.startDistance << ",\n";
    file << "    \"startHeight\": " << m_config.camera.startHeight << ",\n";
    file << "    \"startAngle\": " << m_config.camera.startAngle << ",\n";
    file << "    \"startPitch\": " << m_config.camera.startPitch << ",\n";
    file << "    \"moveSpeed\": " << m_config.camera.moveSpeed << ",\n";
    file << "    \"rotateSpeed\": " << m_config.camera.rotateSpeed << ",\n";
    file << "    \"particleSize\": " << m_config.camera.particleSize << "\n";
    file << "  },\n";
    file << "  \"debug\": {\n";
    file << "    \"enableDebugLayer\": " << (m_config.debug.enableDebugLayer ? "true" : "false") << ",\n";

    std::string logLevelStr = "info";
    if (m_config.debug.logLevel == LogLevel::Debug) logLevelStr = "debug";
    else if (m_config.debug.logLevel == LogLevel::Trace) logLevelStr = "trace";
    file << "    \"logLevel\": \"" << logLevelStr << "\",\n";

    file << "    \"enablePIX\": " << (m_config.debug.enablePIX ? "true" : "false") << ",\n";
    file << "    \"pixAutoCapture\": " << (m_config.debug.pixAutoCapture ? "true" : "false") << ",\n";
    file << "    \"pixCaptureFrame\": " << m_config.debug.pixCaptureFrame << ",\n";
    file << "    \"showFPS\": " << (m_config.debug.showFPS ? "true" : "false") << ",\n";
    file << "    \"showParticleStats\": " << (m_config.debug.showParticleStats ? "true" : "false") << "\n";
    file << "  },\n";
    file << "  \"pix_analysis\": {\n";
    file << "    \"capturePrefix\": \"" << m_config.pixAnalysis.capturePrefix << "\",\n";
    file << "    \"enableReservoirLogging\": " << (m_config.pixAnalysis.enableReservoirLogging ? "true" : "false") << ",\n";
    file << "    \"enablePerformanceCounters\": " << (m_config.pixAnalysis.enablePerformanceCounters ? "true" : "false") << ",\n";
    file << "    \"trackResourceUsage\": " << (m_config.pixAnalysis.trackResourceUsage ? "true" : "false") << "\n";
    file << "  }\n";
    file << "}\n";

    file.close();
    return true;
}

bool ConfigManager::GenerateDefaultConfigs() {
    ConfigManager configMgr;

    // Generate config_dev.json
    configMgr.SetProfileDefaults(Profile::Dev);
    if (!configMgr.SaveToFile("config_dev.json")) {
        return false;
    }
    LOG_INFO("Generated: config_dev.json");

    // Generate config_user.json
    configMgr.SetProfileDefaults(Profile::User);
    if (!configMgr.SaveToFile("config_user.json")) {
        return false;
    }
    LOG_INFO("Generated: config_user.json");

    // Generate config_pix_analysis.json
    configMgr.SetProfileDefaults(Profile::PIXAnalysis);
    if (!configMgr.SaveToFile("config_pix_analysis.json")) {
        return false;
    }
    LOG_INFO("Generated: config_pix_analysis.json");

    return true;
}

} // namespace Config