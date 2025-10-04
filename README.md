# PlasmaDX-Clean

A clean, modern DirectX 12 particle system with ray-traced lighting.

## Architecture Goals

1. **Clean separation of concerns** - Each module does ONE thing well
2. **Automatic fallbacks** - Works on any hardware configuration
3. **Robust feature detection** - Test features before using them
4. **No monolithic files** - Maximum 500 lines per file
5. **Data-driven design** - Configuration over hard-coding

## Core Features

- 100,000 particle NASA-quality accretion disk
- DXR 1.1 ray-traced particle-to-particle lighting
- Mesh shader rendering with compute shader fallback
- Real-time physics simulation
- Clean, maintainable architecture

## Project Structure

```
src/
├── main.cpp                      // Entry point (< 100 lines)
├── core/
│   ├── Application.h/cpp         // Window management only
│   ├── Device.h/cpp              // D3D12 device setup
│   ├── SwapChain.h/cpp           // Swap chain management
│   └── FeatureDetector.h/cpp    // Capability detection
├── particles/
│   ├── ParticleSystem.h/cpp     // Particle logic
│   ├── ParticleRenderer.h/cpp   // Rendering paths
│   └── ParticlePhysics.h/cpp    // Physics simulation
├── lighting/
│   ├── RTLightingSystem.h/cpp   // RT lighting
│   └── AccelerationStructure.h  // BLAS/TLAS management
└── utils/
    ├── ShaderManager.h/cpp       // Shader loading
    ├── ResourceManager.h/cpp     // Buffer/texture management
    └── Logger.h/cpp              // Logging system
```

## Design Principles

### 1. Feature Detection First
```cpp
// Always test before using
if (detector.CanUseMeshShaders()) {
    renderer = std::make_unique<MeshShaderRenderer>();
} else {
    renderer = std::make_unique<ComputeFallbackRenderer>();
}
```

### 2. Single Responsibility
- Application: Window management ONLY
- Device: D3D12 setup ONLY
- ParticleSystem: Logic ONLY
- ParticleRenderer: Rendering ONLY

### 3. Dependency Injection
```cpp
ParticleRenderer(Device* device, ResourceManager* resources);
// Not: ParticleRenderer(App* app); // NO GOD OBJECTS!
```

### 4. Fail Gracefully
```cpp
if (!CreateMeshShaderPipeline()) {
    LOG("Mesh shaders unavailable, using fallback");
    return CreateComputeFallback();
}
```

## Build Instructions

1. Open `build-vs2022/PlasmaDX-Clean.sln`
2. Build in Debug/Release x64
3. Run from build output directory

## Key Improvements Over Original

1. **No 4,842-line monolith** - Modular design
2. **Automatic fallbacks** - Works everywhere
3. **Clean resource management** - No scattered descriptors
4. **Testable components** - Each module independent
5. **Modern C++ practices** - Smart pointers, RAII