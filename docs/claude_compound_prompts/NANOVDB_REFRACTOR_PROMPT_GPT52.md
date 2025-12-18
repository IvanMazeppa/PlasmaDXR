# NanoVDB Refactor — Claude Compound Engineering Prompt (GPT‑5.2)

Copy/paste the block below into Claude Compound Engineering.

---

## The Prompt

```
You are working on PlasmaDX-Clean, a DirectX 12 / DXR engine with a NanoVDB volumetric system.

Goal: make file-loaded NanoVDB assets robust (Blender/OpenVDB → NanoVDB pipeline), fix incompatibilities, improve test UX, and ensure the refactor lands cleanly.

You MUST base your decisions on evidence from:
- the runtime log: build/bin/Debug/logs/PlasmaDX-Clean_20251218_031456.log
- the known-working NanoVDB file: VDBs/NanoVDB/cloud_01.nvdb

### What I observed (high-signal facts)

1) Engine enumerates grids correctly, but fails to load some .nvdb due to codec:

From the log (timestamp ~03:15:45):
  [NanoVDB] Enumerated 1 grids in file:
    [0] 'density' - Float (32-bit) (COMPATIBLE)
  [NanoVDB] Loading grid index 0...
  [NanoVDB] Failed to load grid: ZIP compression codec was disabled during build

This happens for newly generated example assets that were created with ZIP compression.

2) The known-working file uses codec NONE:

`nanovdb_print -v VDBs/NanoVDB/cloud_01.nvdb` shows:
  density float, codec NONE, version 32.4.2

3) Another issue in the log: directory animation load was failing due to a wrong relative path:
  Failed to enumerate directory VDBs/NanoVDB/chimney_smoke: system cannot find the path specified
The executable runs from build/bin/Debug, so paths must be relative from there (e.g. ../../../VDBs/...).

4) Procedural fog sphere renders fine (so the renderer path is OK); the failing point is file decoding / IO compatibility.

### Minimum acceptance criteria

- LoadFromFile should successfully load:
  - VDBs/NanoVDB/cloud_01.nvdb (baseline)
  - a representative density-only .nvdb that is codec NONE (no compression)
  - (optionally) a ZIP-compressed .nvdb *if* we decide to enable ZIP support in the engine build
- LoadAnimationFromDirectory works from a relative directory path (correctly resolved).
- UI supports rapid testing of many .nvdb files without manually typing paths.
- Clear, actionable error messages for incompatibilities:
  - codec unsupported (ZIP/BLOSC)
  - grid type unsupported (FP16/quantized, vec3, etc.)
  - grid name selection issues (density not found)
  - transform / scale / bounds issues

### Tasks (please execute in a sensible order)

#### A) Fix/clarify codec support (ZIP disabled)

Decide one of these approaches and implement it fully:

Option A1 (fastest): Keep ZIP disabled and ensure all example assets are codec NONE.
- Update docs/workflow to generate NanoVDB without ZIP:
  nanovdb_convert -f -g density input.vdb output.nvdb
- Ensure all curated example assets use codec NONE.
- Improve the runtime error to explicitly mention:
  “This engine build does not include ZIP codec support; re-export without ZIP or rebuild NanoVDB with ZIP enabled.”

Option A2 (better long-term): Enable ZIP codec support in the engine build.
- Find where NanoVDB IO codecs are configured (CMake / external/nanovdb build flags).
- Enable ZIP codec and link whatever dependency is needed (zlib/miniz).
- Validate that ZIP-coded files load successfully.

Important: if ZIP is enabled, keep a fallback path or clear message for builds that don’t have it.

#### B) Improve UI for rapid testing

Currently the user can load by pasting a path into a textbox; this is the best “fast path”.
Enhance it to reduce friction:
- Add a “Quick Pick” dropdown for commonly used nvdb paths.
- Add a “Paste from Clipboard” button (copy/paste workflow).
- Add a directory scanner (non-recursive + recursive toggle) that lists *.nvdb and lets the user click one to load.

#### C) Path resolution correctness

Normalize all hardcoded UI sample paths:
- The executable runs from build/bin/Debug.
- Therefore sample paths should use ../../../... to reach repo root.

#### D) Grid compatibility coverage

The system currently logs “Float (32-bit) (COMPATIBLE)”.
Ensure the compatibility check is correct and comprehensive:
- handle multiple grids: prefer “density” but allow selection
- log grid name + value type + codec + voxel size (scale)
- if value type is not float32, either:
  - reject with a clear message, or
  - implement conversion on load (if acceptable), or
  - expand shader support

Also: user reported a “green color change” in volumes. Investigate shader defaults and material parameter mapping to ensure the volume color/emission is intentional and configurable.

### Files to inspect/modify (starting points)

- src/rendering/NanoVDBSystem.cpp / .h  (loading, grid enumeration, error strings)
- src/core/Application.cpp (ImGui NanoVDB panel)
- CMakeLists.txt / external/nanovdb build integration (codec enablement)

### Deliverables

- A short status report: root cause(s) + fixes implemented + how to reproduce.
- If enabling ZIP: document build flags + dependencies.
- If keeping ZIP disabled: document conversion commands and add a “codec NONE” policy for example assets.

When you propose changes, cite the log evidence and explain why each fix is required.
```

---


