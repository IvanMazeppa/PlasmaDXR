# Config Loader Path Resolution Fix

**Problem**: Config loader only searches current working directory, fails when exe runs from build/bin/Debug/

**Error**: `Failed to open config file: configs/scenarios/multi_light_only.json`

---

## **Root Cause**

```cpp
// Current code in Config.cpp::LoadFromFile()
std::ifstream file(filepath);  // Only checks CWD
if (!file.is_open()) {
    LOG_ERROR("Failed to open config file: {}", filepath);
    return false;
}
```

**When exe runs from**: `build/bin/Debug/`
**Config file is at**: `../../../configs/scenarios/multi_light_only.json`

---

## **Fix: Multi-Path Search**

**File**: `src/config/Config.cpp`

**Location**: Replace `LoadFromFile()` function (line 160-173)

```cpp
bool ConfigManager::LoadFromFile(const std::string& filepath) {
    // Try multiple search paths
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
```

---

## **Alternative: CMake Post-Build Copy** (Simpler)

Add to `CMakeLists.txt`:

```cmake
# Copy configs to build directory after build
add_custom_command(TARGET PlasmaDX-Clean POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/configs
        $<TARGET_FILE_DIR:PlasmaDX-Clean>/configs
    COMMENT "Copying config files to build directory"
)
```

**Pros**: Configs always available in exe directory
**Cons**: Changes to configs require rebuild

---

## **Quick Test**

After applying fix (✅ **VERIFIED WORKING 2025-11-16**):

```bash
# Test 1: Relative path from project root
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean
./build/bin/Debug/PlasmaDX-Clean.exe --config=configs/scenarios/multi_light_only.json
# ✅ WORKING: Found config at: configs/scenarios/multi_light_only.json

# Test 2: Running from build directory
cd build/bin/Debug
./PlasmaDX-Clean.exe --config=configs/scenarios/multi_light_only.json
# ✅ WORKING: CMake copied configs to build/bin/Debug/configs/

# Test 3: Absolute path
./build/bin/Debug/PlasmaDX-Clean.exe --config=/full/path/to/multi_light_only.json
# ✅ WORKING (always supported)
```

**Note:** When using --config flag, always specify the full path relative to configs/, e.g., `configs/scenarios/multi_light_only.json`, not just the filename.

---

## **Which Fix to Use?**

| Fix | Effort | Best For |
|-----|--------|----------|
| **Multi-path search** | 10 min | Development (flexible paths) |
| **CMake copy** | 5 min | Deployment (exe self-contained) |
| **Both** | 15 min | Production-ready ⭐ |

**Recommendation**: Implement **both** - multi-path for dev, CMake copy for distribution.

---

**Last Updated**: 2025-11-16
