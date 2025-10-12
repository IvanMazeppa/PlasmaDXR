# PIX Autonomous Agent — Operational Runbook & Troubleshooting

## One-command autonomous run

```bash
python pix/pix_autonomous_agent.py --autonomous --delay=120
```

- Warmup frames: 120 (≈2s @ 60fps). Adjust as needed.
- Output: `pix/Captures/auto_capture_*.wpix`, reports and JSON in `pix/Reports/`, `pix/Analysis/`.

## Preconditions

- PIX installed at `C:\Program Files\Microsoft PIX\`.
- `pix/pixtool.exe` present (copied from install).
- App built: `build/Debug/PlasmaDX-Clean.exe` exists.
- App integrated with PIX (see Architecture doc): `PIXCaptureHelper` wired, env-based trigger.

## Windows vs WSL

- Windows PowerShell: preferred; runs natively.
- WSL: use Windows Python `/mnt/c/Python39/python.exe`; or wrap pixtool with `cmd.exe /c` and convert paths via `wslpath`.
- Write outputs to Windows filesystem paths (project drive), not WSL-only dirs.

## Manual capture sanity test

```powershell
$env:PIX_AUTO_CAPTURE="1"
$env:PIX_CAPTURE_FRAME="120"
pix\pixtool.exe launch .\build\Debug\PlasmaDX-Clean.exe programmatic-capture --open save-capture pix\Captures\manual_test.wpix
```

Expect:

- App logs PIX loaded, auto-capture enabled, capture started/ended, then exits.
- `pix/Captures/manual_test.wpix` exists and >100KB.

## Agent flow

1) Launch and capture via pixtool with env vars
2) Verify capture exists
3) Export events CSV, extract frame image
4) Run analyzers (DXR, ReSTIR, buffers, perf)
5) Generate markdown + JSON reports

## Common failures and fixes

- DLL not loaded
  - Check include path and `USE_PIX` define; verify PIX install.
- Capture not created
  - Ensure `PIXEndCapture(FALSE)` runs and app exits (`PostQuitMessage(0)`). Use small delay (frame 5) to test.
- WSL silent I/O failure
  - Run from Windows or use Windows Python; avoid WSL FS for outputs.
- Timeout (300s)
  - App didn’t exit or capture didn’t start. Log app output; increase timeout to 600s temporarily.
- Image not extracted
  - Use event delay (e.g., 5–10). Confirm capture opens in PIX UI; verify rendertarget exists.

## Useful commands

- Events CSV: `pix/pixtool.exe open-capture file.wpix save-event-list events.csv --counter-groups=D3D*`
- Save RT image: `pix/pixtool.exe open-capture file.wpix save-resource out.png`
- Marker-specific: `--marker="RT Lighting"`; Depth: `--depth`

## CI integration (example)

```yaml
- name: Autonomous PIX capture
  run: python pix/pix_autonomous_agent.py --autonomous --delay=120

- name: Regression check
  run: python pix/regression_check.py
```

## Acceptance checklist

- [ ] App logs PIX load and auto-capture status
- [ ] `.wpix` created and non-empty
- [ ] CSV and PNG extracted
- [ ] Markdown + JSON generated
- [ ] Works on Windows and from WSL (via Windows Python)
