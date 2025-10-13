# PIX Autonomous Agent Overhaul Report (WSL + pixtool v2509.25)

## Executive summary

- Root issue: unstable capture initiation from WSL due to Windows context, path quoting, and reliance on programmatic PIX APIs that return `E_FAIL` under some conditions.
- Overhaul direction: make the agent Windows-first in execution semantics; standardize on `pixtool.exe launch` → warmup/health-check → `take-capture --frames=1 --open save-capture`; add a robust control plane to change app settings at runtime; keep programmatic PIX APIs as a later, optional path behind feature flags.

## What’s broken (from PIX_AGENT_OVERHAUL.md and current behavior)

- Attach mode from WSL: PID mapping and instrumentation preconditions make `pixtool attach` unreliable.
- Launch from WSL: incorrect shell, path quoting, and working directory cause "not recognized"/no-op failures.
- Programmatic capture: `PIXBeginCapture()` returns `0x80004005` in some runs; likely environment/init sequencing.

## Best-practice guidance for this use case (validated approach)

1. Windows context for all pixtool calls

- Invoke via `cmd.exe /c` (or PowerShell) from WSL; prefer Windows Python if available.
- Use absolute Windows paths for pixtool, exe, working-dir, and outputs; never rely on relative paths from WSL.

1. Launch → warmup → take-capture (no programmatic APIs)

- `pixtool.exe launch <app.exe> --working-directory=<proj> --command-line="<args>"`
- After a warmup delay (or app readiness signal), issue:
  - `pixtool.exe take-capture --frames=1 --open save-capture <out.wpix>`
- This mirrors your working .bat and avoids in-app PIX API fragility.

1. Working directory and command line are mandatory

- Always provide `--working-directory` so shaders/configs resolve.
- Pass runtime options via `--command-line` (e.g., `--config config_pix_close.json`).

1. Robust readiness and verification

- Prefer log-polling for a "READY_FOR_CAPTURE" marker over fixed sleeps.
- Verify capture existence and size (>100KB) and retry once with longer warmup.
- Extract an image and event CSV immediately to validate capture integrity.

1. WSL path discipline

- Convert all paths with a single utility (WSL ↔ Windows); quote only at the shell boundary.
- Write outputs to Windows FS (project drive) to avoid 9P issues.

1. Later, optional: programmatic PIX

- If needed, guard with a flag; add COM init, ensure GPU idle, align DLL versions; consider `PIXGpuCaptureNextFrames()`.

## Timed capture strategies (no attach required)

- "Clocked" batches: relaunch app per capture with different warmup frame counts (fast if startup is light).
- Readiness signal: app logs `READY_FOR_CAPTURE` when camera/scene ready; agent then runs `take-capture`.
- Multi-capture session (if attach becomes viable): take N captures with delays and camera updates via control plane (see below).

## Autonomous runtime control (change settings, move camera)

- Add an in-app control plane (Windows Named Pipe or localhost TCP JSON-RPC):
  - Commands: `set_config`, `enable_feature`, `set_camera`, `reload_preset`, `mark_ready_for_capture`.
  - The agent sends commands between launch and capture; the app acknowledges and logs readiness.
- Keep the existing JSON config system for initial state; use the control plane to adjust live parameters.

## Overhaul plan (phased)

Phase 1 — make one capture rock solid (today)

- Implement a `cmd.exe`-wrapped pixtool runner (absolute Windows paths, quoted once at shell boundary).
- Launch with `--working-directory` and `--command-line` → warmup wait → `take-capture --frames=1 --open save-capture`.
- Verify `.wpix` exists and size >100KB; export events and save an image.

Phase 2 — multi-scenario batch + readiness polling (today)

- Loop over presets (`config_pix_close.json`, `config_pix_far.json`, etc.); produce distinct captures and reports.
- Add log polling (watch `logs/*.log`) for `READY_FOR_CAPTURE` to replace fixed sleeps.

Phase 3 — control plane (1–2 days)

- Implement a minimal Named Pipe JSON-RPC server in the app; expose camera/feature toggles.
- Agent issues: set camera → wait for ACK → mark ready → take-capture → analyze.

Phase 4 — optional programmatic PIX (later)

- Behind a flag, try `PIXGpuCaptureNextFrames()` and/or `PIXBegin/EndCapture()` with:
  - COM init at startup (`CoInitializeEx(nullptr, COINIT_MULTITHREADED)`),
  - `WinPixGpuCapturer.dll` colocated with DebugPIX exe (matching your PIX version),
  - GPU idle fences around capture, and
  - older PIX DLL (if 2509.25 still fails).

## Implementation notes and snippets

- cmd wrapper builder (pseudo-Python):
  - Build: `C:\Program Files\Microsoft PIX\2509.25\pixtool.exe launch "D:\...\PlasmaDX-Clean.exe" --working-directory="D:\...\PlasmaDX-Clean" --command-line="--config config_pix_close.json"`
  - Wrap: `['/mnt/c/Windows/System32/cmd.exe', '/c', cmd_str]`
- Path utils: one canonical `wsl_to_windows_path()` and `windows_to_wsl_path()`; quote only once when building `cmd_str`.
- Verification: size threshold, event CSV export, final RT image extraction.

## Troubleshooting E_FAIL and timing

- Programmatic `E_FAIL` common causes: missing COM init, mismatched DLL vs header, capture called while GPU busy, debug layer conflicts, or PIX version quirks.
- Mitigations (if revisited): COM init, `WaitForGPU()` fencing, disable D3D12 debug layer when PIX detected, colocate correct capturer DLL, try prior PIX (e.g., 2403.x), and test `PIXGpuCaptureNextFrames`.

## Claude Sonnet 4.5 integration

- Single intent: "Capture with preset=close and camera=yaw:15,pitch:-10" →
  - Write/ensure preset, launch via pixtool (Windows context), send control-plane camera command, wait for READY, take-capture, analyze, post summary with links to image/report.
- Add guardrails: if capture missing or small → auto-retry once; if analysis fails → emit actionable diagnostics.

## Queries to run (for additional references)

- Web: Microsoft PIX pixtool "take-capture --frames" best practices; programmatic capture `PIXBeginCapture E_FAIL` causes; `PIXGpuCaptureNextFrames` usage.
- Web: WSL → Windows `cmd.exe /c` invocation patterns with quoted paths containing spaces.

## Success criteria

- 100% success rate for a single capture via launch → take-capture from WSL.
- Batch of 3–5 scenarios produces valid `.wpix` + image + report each in <5 minutes.
- Live camera/feature changes via control plane before capture.
- Optional programmatic path validated in a standalone test build.
