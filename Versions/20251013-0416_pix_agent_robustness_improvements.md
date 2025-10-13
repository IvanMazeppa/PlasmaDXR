# PIX Agent Robustness & Improvements

## Reliability

- Windows-first execution: default to Windows Python; WSL path is opt-in fallback.
- Hardened process control: timeouts, retries, and clear logging of pixtool stdout/stderr.
- Path discipline: always pass absolute Windows paths to pixtool; set `--working-directory`.
- Output verification: existence + minimum size; on failure, auto-rerun with longer warmup.
- Log polling (optional): app logs a `READY_FOR_CAPTURE` line; agent waits for it instead of fixed sleep.

## Ergonomics

- CLI presets: `--preset close|far|inside` maps to config files (`config_pix_*.json`).
- Batch mode: `--scenarios close,far,inside` runs N captures and produces an aggregate summary.
- Single entrypoint: `--autonomous` orchestrates launch → capture → analyze → report.
- Structured console: phase banners and concise summaries; file paths echoed at end.

## Analysis depth

- Add counters coverage: RT core throughput, rays/thread, BLAS/TLAS timings (already in place) plus UAV barrier counts and copy bandwidth.
- Visual correlation: always extract final RT image; allow `--delay-events N` for pre-present imagery.
- Finding quality: thresholds sourced from `pix/config.json`; emit rationale and expected improvements.
- Diff mode: compare two captures (baseline vs optimized) and compute deltas in key metrics with a short text verdict.

## Code changes (agent)

- Stable capture path (based on working .bat):
  - Build `launch` command with `--command-line` and `--working-directory`.
  - Sleep or poll logs; then run `take-capture --frames=1 --open save-capture <out.wpix>`.
  - Verify, retry once, then proceed to analysis.

- Options:
  - `--shell windows|wsl`, `--delay N`, `--timeout SECONDS`, `--min-size BYTES`.
  - `--scenarios <list>`, `--preset <name>`, `--delay-events N` (image extraction).

## Code changes (app)

- Optional readiness marker: log `[APP] READY_FOR_CAPTURE` after warmup to reduce sleeps.
- Ensure PIX-enabled build places `WinPixGpuCapturer.dll` next to exe (for future programmatic tests).
- Keep PIX code behind `USE_PIX` and config flags to avoid perf impact.

## CI and ops

- Weekly smoke capture on a Windows runner; fail CI if agent can’t produce `.wpix` or if key metrics regress > X%.
- Artifacts: upload `.wpix`, reports, and images; keep last N runs for trend lines.

## Roadmap (short)

1. Implement stable capture path and presets (today).
2. Add batch scenarios + aggregate summary (today).
3. Add "READY_FOR_CAPTURE" polling (tomorrow).
4. Add diff mode (+ simple charts) (this week).
5. Revisit programmatic PIX APIs with COM init and `PIXGpuCaptureNextFrames` (later).
