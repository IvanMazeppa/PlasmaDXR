# Repository Guidelines

## Project Structure & Module Organization
- `src/`: C++17 engine code (`src/main.cpp`; subsystems in `src/core/`, `src/particles/`, `src/lighting/`, `src/rendering/`, `src/ml/`, `src/utils/`, `src/config/`).
- `shaders/`: HLSL sources; CMake compiles stage-suffixed files into `build*/bin/<Config>/shaders/`.
- `assets/`, `shared_assets/`: runtime assets (textures, meshes, etc.).
- `configs/` and `config*.json`: runtime configuration profiles and presets.
- `docs/`, `shared_docs/`: reference docs and write-ups; prefer adding new long-form docs here.
- `scripts/` and `tools/`: automation (Blender/VDB utilities, screenshots, session helpers).
- Generated output is gitignored: `build/`, `build-vs2022/`, `logs/`, `venv/`, `PIX/`, `screenshots/`, `VDBs/`.

## Build, Test, and Development Commands
Windows-only (DirectX 12). Prereqs: Visual Studio 2022 (Desktop C++), CMake ≥ 3.20, Windows SDK, and `dxc` on `PATH`.

```bat
REM One-shot build + run (creates build-vs2022/)
BUILD_AND_RUN.bat

REM Manual build
cmake -S . -B build-vs2022 -G "Visual Studio 17 2022" -A x64
cmake --build build-vs2022 --config Debug
.\build-vs2022\bin\Debug\PlasmaDX-Clean.exe --help
```

Headless benchmark mode (no window/swapchain):

```bat
.\build-vs2022\bin\Release\PlasmaDX-Clean.exe --benchmark --help
```

## Coding Style & Naming Conventions
- C++: 4-space indentation, braces on the same line, member fields prefixed `m_`, types/methods in `PascalCase`.
- Files: pair headers/impls (`Foo.h` + `Foo.cpp`) and keep modules in their subsystem folder.
- Shaders: name by stage suffix so CMake compiles them (`*_cs.hlsl`, `*_ps.hlsl`, `*_vs.hlsl`, `*_ms.hlsl`, `*_as.hlsl`, `*_lib.hlsl`).

## Configuration Tips
- Select a config with `--config=<file>` or set `PLASMADX_CONFIG` (defaults to `config_dev.json` when present).
- Keep configs reproducible: avoid machine-specific absolute paths and don’t check in captures/logs.

## Testing Guidelines
- No unit-test harness in-tree; treat “tests” as build + runtime validation.
- Before PR: run representative flags (e.g. `--particles 50000 --gaussian`) and check logs under `build*/bin/<Config>/logs/`. For GPU investigations, use PIX scripts (`pix_*.bat`, `pix_capture*.ps1`).

## Commit & Pull Request Guidelines
- Follow Conventional Commits used in history: `feat:`, `fix:`, `docs:`, `perf:`, `refactor:`, `chore:` (see `.git_commit_template_probe_grid.txt` for a good long-form body).
- PRs: include “What/Why”, repro steps, and for visual changes add a screenshot or PIX capture notes; do not commit generated artifacts or large binaries (check `.gitignore`).

## LLM Initialization: Blender VFX Orchestrator (Summary)

This repo contains **blender-vfx-orchestrator**, an autonomous multi-agent system built on the **OpenAI Agents SDK** to generate Blender 5 VFX assets (smoke/fire/liquid/nebulae) via iterative runs.

### Core Docs (Read First)
- `agents/blender-vfx-orchestrator/docs/MASTER_ROADMAP_2026-01-26.md`
- `agents/blender-vfx-orchestrator/docs/AI_OPERATION_MANUAL.md`
- `agents/blender-vfx-orchestrator/docs/MULTI_AGENT_ARCHITECTURE_ADDENDUM_2026-01-29.md`
- `agents/blender-vfx-orchestrator/docs/SDK_ENFORCEMENT_PROTOCOL.md`
- `agents/blender-vfx-orchestrator/docs/VERSION_TRUTH.md`

### Operational Flow (Required)
State machine:
```
PLAN → GENERATE → VALIDATE → EXECUTE → EVALUATE → DECIDE
```
Failures must route back via DIAGNOSE/FIX/IMPROVE paths (no ad-hoc retries).

### Critical Blender 5 Headless Gotchas
- `bpy.ops.fluid.bake_all()` requires **active domain object** and **OBJECT** mode.
- Liquid inflows need `flow_settings.use_plane_init = True` (auto-injected by API fixer).
- `cache_type = 'REPLAY'` does not produce full bake data; use `ALL`.
- `animation=True` does not emit stills headless; render representative frames only.
- Headless paths must use `BLENDER_OUTPUT_DIR` / `BLENDER_CACHE_DIR`.

### Ground Truth Rule
- Use Blender 5 MCP docs for API verification. Do **not** trust memory or training data.
