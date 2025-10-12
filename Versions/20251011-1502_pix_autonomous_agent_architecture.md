# PIX Autonomous Agent for PlasmaDX-Clean — Architecture & Integration Plan

## Objectives

- Enable fully autonomous captures with pixtool + in-app PIX API, robust under WSL/Windows.
- Preserve a clean separation between capture (app) and analysis (Python agent).
- Provide durable interfaces for CI, Claude agents, and local iteration.

## Minimal app integration (30 min)

- Add `src/debug/PIXCapture.h/.cpp` with `PIXCaptureHelper::Initialize()`, `CheckAutomaticCapture(frame)`, `TriggerCapture()` as specified in PIX docs already collected in `PIX_PROGRAMMATIC_CAPTURE_GUIDE.md`.
- Include `#include <pix3.h>` and load `WinPixGpuCapturer.dll` via `PIXLoadLatestWinPixGpuCapturerLibrary()`.
- Main loop: call `Initialize()` once, then `CheckAutomaticCapture(frame++)` each frame. On trigger, call `PIXBeginCapture/PIXEndCapture` and `PostQuitMessage(0)`.
- Use env vars: `PIX_AUTO_CAPTURE=1`, `PIX_CAPTURE_FRAME=N` for warmup frames.

## Python agent launch contract

- Command (Windows/WSL-safe):
  - `pixtool.exe launch <PlasmaDX-Clean.exe> programmatic-capture --open save-capture <output.wpix>`
- Environment: set `PIX_AUTO_CAPTURE=1` and `PIX_CAPTURE_FRAME=<N>` before launching.
- Agent waits for app exit, verifies `<output.wpix>` exists, then runs existing analysis pipeline.

## WSL compatibility

- Prefer running from Windows PowerShell; or from WSL use Windows Python (`/mnt/c/Python39/python.exe`).
- If invoking from WSL: wrap pixtool with `cmd.exe /c` and convert paths via `wslpath`.
- Avoid writing captures into WSL FS; write to Windows path (project drive) to prevent 9P issues.

## File/dir conventions

- Captures: `pix/Captures/auto_capture_<YYYY_MM_DD__HH_MM_SS>.wpix`.
- Reports: `pix/Reports/<name>_analysis.md` (+ images subdir).
- Analysis JSON: `pix/Analysis/<name>_analysis.json`.
- Config: `pix/config.json` (thresholds/targets).

## Health checks

- App prints: `[PIX] GPU Capturer DLL loaded successfully` on startup.
- App prints: `[PIX] Auto-capture ENABLED - will capture at frame N` if env set.
- On trigger: `[PIX] Capture started/ended successfully` then exits.
- Agent: verifies capture exists and size > 100KB; fails fast otherwise with guidance.

## Failure modes and mitigations

- DLL not found: add PIX include path and `USE_PIX` define; verify install path.
- Capture file missing: ensure app exits after `PIXEndCapture(FALSE)`; check permissions/output path.
- WSL silent failure: run via Windows Python or PowerShell; avoid WSL FS for outputs.
- Timeout: increase agent timeout to 600s; log app stdout/stderr to diagnose.

## Extensibility hooks

- Alternate triggers: command-line arg, named pipe, or socket for on-demand capture.
- Multi-frame sequences: replace Begin/End with `PIXGpuCaptureNextFrames(filename, N)`.
- Event/marker-specific screenshots: extend pixtool calls with `--event` or `--marker`.

## Acceptance criteria

- Single command completes end-to-end in <60s:
  - `python pix/pix_autonomous_agent.py --autonomous --delay=120`
- Capture saved, events CSV extracted, image saved, markdown + JSON reports generated.
- Works from both Windows and WSL (via Windows Python).

## Immediate next steps

- Implement `PIXCaptureHelper` and wire into main loop.
- Test manual pixtool command with env vars.
- Enable autonomous workflow in `pix_autonomous_agent.py` (method provided in GUIDE).
