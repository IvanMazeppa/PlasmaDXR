# PIX Autonomous Capture Remediation Plan

## Context

- Current infra (dual Debug/DebugPIX builds, JSON configs, PIX integration) is solid, but programmatic capture via `PIXBeginCapture/PIXEndCapture` returns `E_FAIL (0x80004005)` at runtime.
- A batch workflow using pixtool `launch` followed by `take-capture --frames=1` has produced captures successfully.
- WSL adds friction (env propagation, path translation, 9P FS I/O quirks).

## Target outcome

- Reliable, fully automated capture → analyze → report in <60s per run, robust to WSL/Windows differences.

## Priority plan (do this first)

1. Adopt the “pixtool launch + take-capture” stable path (no programmatic APIs)

- Launch the app with full paths, correct working directory, and command-line switches.
- After a warmup delay, issue a separate `take-capture --frames=1 --open save-capture <out.wpix>`.
- Mirror `pix_capture_working.bat` semantics inside the Python agent.

1. Make the agent Windows-first (with WSL fallback)

- Prefer running from Windows PowerShell or Windows Python (`C:\PythonXX\python.exe`).
- From WSL, invoke via `cmd.exe /c` and convert all paths with `wslpath`.
- Always write captures to Windows paths (project drive), not WSL-only locations.

1. Always set `--working-directory` and pass app args via `--command-line`

- Ensures shaders/log paths resolve and configs load.
- Example:
  - `pixtool.exe launch "...\PlasmaDX-Clean.exe" --command-line="--config config_pix_far.json" --working-directory="...\PlasmaDX-Clean"`
  - After delay: `pixtool.exe take-capture --frames=1 --open save-capture "...\pix\Captures\auto_*.wpix"`

1. Add robust waiting and verification

- Wait N seconds (or poll process readiness) before `take-capture`.
- After capture returns, verify file exists and size >100KB; retry once with a longer delay if missing.

1. Capture multiple scenarios in one run

- Loop over config presets (`config_pix_far.json`, `config_pix_close.json`, etc.).
- For each, update `--command-line` and output name, then launch + wait + capture + analyze.

## Secondary plan (if/when you revisit programmatic capture)

If you still want in-app `PIXBeginCapture`:

- Initialize COM before PIX: `CoInitializeEx(nullptr, COINIT_MULTITHREADED)` once at startup.
- Ensure `WinPixGpuCapturer.dll` version matches `pix3.h`; place DLL next to the PIX-enabled binary (`build/DebugPIX`).
- Disable D3D12 debug layer when PIX capturer is present (already implemented).
- Ensure GPU is idle around capture calls (signal/wait fence):
  - `WaitForGPU(); PIXBeginCapture(...); RenderOneFrame(); WaitForGPU(); PIXEndCapture(...);`
- Try the alternative API: `PIXGpuCaptureNextFrames(L"...\capture.wpix", 1)` prior to the warmup loop.
- Test different PIX versions (e.g., 2403.x) if 2509.25 continues to fail.
- Log `GetProcAddress` results for `PIXBeginCapture`/`PIXEndCapture` and HRESULTs to aid diagnosis.

## Agent changes (concrete)

- Add a "stable-capture" path that:
  - Builds the pixtool `launch` command with `--working-directory` and `--command-line`.
  - Sleeps for `--delay` (or polls the log file for readiness markers).
  - Issues `take-capture --frames=1 --open save-capture <out.wpix>`.
  - Verifies capture exists and size threshold; retries once if necessary.
  - Proceeds to event export, image extraction, and analysis as today.

- Add defensive options:
  - `--shell windows|wsl` (force Windows Python vs WSL wrapper).
  - `--verify-dll` to confirm `WinPixGpuCapturer.dll` presence/bitness near the PIX build.
  - `--timeout N` to override default 300s.

## WSL interop checklist

- Use Windows Python from WSL: `/mnt/c/Python39/python.exe pix/pix_autonomous_agent.py ...`.
- Wrap pixtool calls with `cmd.exe /c` and quote arguments.
- Convert all paths with `wslpath` and prefer Windows drive paths in commands.
- Emit all outputs to the project drive (`D:\...`), not the Linux FS.

## Verification steps

- Manual sanity: replicate `pix_capture_working.bat` results via the agent (one capture).
- Multi-scenario: run 3–4 presets in a row and confirm 3–4 `.wpix` files + reports.
- Regression tests: add a short CI step that runs one capture weekly (Windows runner) and checks report metrics.

## Troubleshooting guide (quick)

- No file created: increase warmup delay; ensure app didn’t exit early; verify working directory.
- Small file (<100KB): incorrect capture timing; increase `--frames` to 2 or delay longer; verify rendering occurs.
- WSL silent success/no output: switch to Windows Python/PowerShell; write to Windows paths.
- Programmatic `E_FAIL`: use stable path above; later test COM init + `PIXGpuCaptureNextFrames` + older PIX.

## Suggested near-term tasks

- Implement the stable capture path in Python (2–4 hours).
- Add multi-config batch run and merged summary report (2 hours).
- Optional: instrument logs with explicit "READY_FOR_CAPTURE" marker after warmup so the agent can poll instead of sleeping.
