#pragma once

#include <d3d12.h>
#include <wrl/client.h>
#include <string>

// Robust feature detection that ACTUALLY TESTS if features work
// Not just checking capability bits!

class FeatureDetector {
public:
    struct Features {
        // Core features
        bool dxr11 = false;
        bool meshShaders = false;
        bool variableRateShading = false;

        // Specific capabilities
        bool meshShadersCanReadDescriptors = false;  // The critical test!
        bool computeShadersFallback = true;          // Always available

        // Hardware info
        std::string gpuName;
        uint64_t dedicatedVideoMemory = 0;
        uint32_t vendorId = 0;

        // Driver/SDK info
        std::string driverVersion;
        uint32_t agilitySDKVersion = 618;
    };

public:
    FeatureDetector() = default;

    // Main detection routine
    bool Initialize(ID3D12Device* device);

    // Query methods
    bool CanUseMeshShaders() const {
        return m_features.meshShaders && m_features.meshShadersCanReadDescriptors;
    }

    bool CanUseDXR() const {
        return m_features.dxr11;
    }

    bool ShouldUseComputeFallback() const {
        return !CanUseMeshShaders();
    }

    const Features& GetFeatures() const { return m_features; }

    // Log detected features
    void LogFeatures() const;

private:
    // Detection methods
    bool DetectGPUInfo(ID3D12Device* device);
    bool DetectDXRSupport(ID3D12Device* device);
    bool DetectMeshShaderSupport(ID3D12Device* device);

    // CRITICAL: Actually test if mesh shaders can read descriptors
    bool TestMeshShaderDescriptorAccess(ID3D12Device* device);

private:
    Features m_features;
    Microsoft::WRL::ComPtr<ID3D12Device> m_device;
};