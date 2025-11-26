# GEMINI.md

This file serves as the primary context and instruction manual for Gemini agents working on the **PlasmaDX-Clean** project.

---

## 1. User Context & Interaction Guidelines

**User:** Ben (Novice programmer, experienced with C++/Java/Python).
**Personality:** High-functioning autism; deep passion for AI, ML, and this specific project.
**Communication Style:**
- **Brutal Honesty:** Do not sugar-coat feedback. If something is broken or implemented poorly, state it clearly and directly. This saves time and confusion.
- **Explain the "Why":** Don't just provide a fix; explain *why* the previous approach was incorrect and *why* the new solution works.
- **Be Corrective:** Immediately correct misunderstandings about APIs or architectures.
- **Validate Effort:** Acknowledge reasonable attempts, even if they need significant correction
---

## 2. Project Overview

**PlasmaDX-Clean** is a high-performance **DirectX 12 volumetric particle renderer** simulating a black hole accretion disk. It leverages cutting-edge rendering and ML technologies to achieve real-time cinematic visuals.

**Key Features:**
- **Rendering:** 3D Gaussian Splatting (volumetric ellipsoids), DXR 1.1 Inline Ray Tracing (RayQuery), ReSTIR Global Illumination, PCSS Soft Shadows.
- **Volumetrics:** Froxel-based volumetric fog, Henyey-Greenstein phase function.
- **Physics:** GPU compute shaders (traditional) & PINN (Physics-Informed Neural Networks) for relativistic accretion disk dynamics.
- **Upscaling:** NVIDIA DLSS 3.7 Super Resolution.
- **Architecture:** Modular C++17 codebase with a "Clean Architecture" philosophy (Single Responsibility Principle).

---

## 3. Technical Architecture

### Core Stack
- **Language:** C++17 (Visual Studio 2022)
- **Graphics API:** DirectX 12 Agility SDK
- **Shading Language:** HLSL (Shader Model 6.5+)
- **Build System:** CMake (generating MSBuild solutions)
- **UI:** ImGui
- **ML/AI:** PyTorch (Training), ONNX Runtime (Inference/Integration)

### Directory Structure
- `src/core/`: Window, Device, SwapChain, Main Loop.
- `src/particles/`: Particle simulation and rendering logic.
- `src/lighting/`: RT pipelines, ReSTIR, RTXDI, Probe Grids.
- `src/rendering/`: Volumetric systems (Froxels).
- `src/ml/`: PINN physics, Adaptive Quality (ONNX).
- `shaders/`: HLSL source files (compiled to `.dxil` by CMake).
- `configs/`: JSON-based runtime configuration.

### Rendering Pipeline
1.  **Physics Update:** GPU Compute or PINN inference.
2.  **Structure Update:** BLAS/TLAS rebuild (currently per-frame).
3.  **Shadow Pass:** PCSS / DXR RayQuery shadows.
4.  **Volumetric Pass:** Froxel density injection & lighting.
5.  **Main Render:** Ray-marched 3D Gaussian splatting with RT lighting.
6.  **Post-Process:** DLSS Upscaling -> Tone Mapping (HDR to SDR).

---

## 4. Development Workflow

### Build Commands
The project uses **CMake** to generate a Visual Studio solution. Use `MSBuild` for compilation.

**Debug Build (Fast Iteration):**
```cmd
'/mnt/c/Program Files/Microsoft Visual Studio/2022/Community/MSBuild/Current/Bin/MSBuild.exe' build/PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64
```

**DebugPIX Build (GPU Debugging):**
```cmd
MSBuild PlasmaDX-Clean.sln /p:Configuration=DebugPIX /p:Platform=x64
```

### Running the Engine
Executable location: `build/bin/Debug/PlasmaDX-Clean.exe`

**Arguments:**
- `--config=configs/scenarios/stress_test.json`: Load specific config.
- `--particles 50000`: Override particle count.
- `--billboard`: Force billboard renderer (fallback).

### Shader Compilation
Shaders (`.hlsl`) are compiled to `.dxil` binaries during the build via CMake custom commands.
**Critical:** If visual bugs persist, ensure `.dxil` files in `build/bin/Debug/shaders/` are newer than their source `.hlsl` files.

---

## 5. Critical Mandates & Pitfalls

1.  **Feature Detection:** ALWAYS check for hardware support (e.g., `D3D12_RAYTRACING_TIER_1_1`) before initialization.
2.  **Descriptor Heaps:** Use the central `ResourceManager`. Do not create ad-hoc descriptor heaps.
3.  **Resource Barriers:** Ensure correct state transitions (e.g., `UNORDERED_ACCESS` <-> `NON_PIXEL_SHADER_RESOURCE`) to avoid GPU hangs.
4.  **Root Signatures:** Match C++ root parameter indices EXACTLY with HLSL `register(b#)`/`register(u#)`.
5.  **Stale Shaders:** "Weird" visual glitches are almost always stale `.dxil` files. Rebuild or check timestamps.

---

## 6. Current Focus & Roadmap

**Active Tasks:**
- **PINN Integration:** Completing the C++ integration of the trained Physics-Informed Neural Network using ONNX Runtime.
- **RTXDI M5:** Refining temporal accumulation for the ReSTIR system to fix patchwork artifacts.
- **Froxel System:** Optimizing the newly added volumetric fog.

**Reference Documents:**
- `README.md`: General info.
- `CLAUDE.md`: Detailed agent context and specialized MCP tool info.
- `BUILD_GUIDE.md`: Step-by-step build instructions.
- `MASTER_ROADMAP_V2.md`: The authoritative plan.
