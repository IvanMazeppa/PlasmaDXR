# PIX Agent v4 — Autonomous Capture + Buffer Analysis Plan (WSL + pixtool)

## Goal

- From WSL, use Windows tools to: launch app → capture on demand → list/export buffers → parse/analyze → report. Support camera/feature changes live and deep dives into GPU data.

## Core architecture

1. Windows execution context for pixtool

- Wrap all commands with `cmd.exe /c` from WSL and use absolute Windows paths.
- Always set `--working-directory` and pass runtime args via `--command-line`.

1. Launch → readiness → take-capture

- Launch the app via:
  - `pixtool.exe launch <exe> --working-directory=<proj> --command-line="--config <preset.json>"`
- Wait for readiness via log polling or control-plane ACK (see below), then:
  - `pixtool.exe take-capture --frames=1 --open save-capture <out.wpix>`

1. Control plane (runtime)

- Add a Named Pipe (or localhost TCP) JSON-RPC server in app.
- Commands: `set_config`, `enable_feature`, `set_camera{pos,yaw,pitch}`, `mark_ready_for_capture`, `list_buffers`, `dump_buffer{name,path}`.
- Agent sends control messages to prep the scene before capture and to request targeted dumps after capture.

1. Data plane (two paths)

- Preferred: use `pixtool.exe` to list and export resources from `.wpix`:
  - `pixtool.exe open-capture <file.wpix> list-resources`
  - `pixtool.exe open-capture <file.wpix> save-resource --name="<resourceName>" <out.bin>`
- Fallback (faster iteration): app-side readback dumps to disk during capture window (binary or CSV) with a fixed struct layout.

## Buffer parsing strategy

- Define stable binary layouts per buffer (e.g., ReSTIR reservoir = 32 bytes: float3 lightPos, float weightSum, uint M, float W, uint particleIdx, uint pad).
- Provide Python parsers with `struct.unpack` templates; sample, compute distributions, emit anomalies.
- Maintain a mapping registry: `resourceName` ↔ `layout` so the agent knows how to parse.

## Timed/on-the-fly capture flows

- Clocked: relaunch app per capture with different warmup frames.
- Event-driven: control plane marks READY; agent immediately triggers `take-capture`.
- Multi-capture session (future): if attach becomes viable, take N captures while app runs; else loop relaunch.

## Robustness features

- Path conversion utilities (WSL ↔ Windows) in one place; quote only once at shell boundary.
- Verification: ensure `.wpix` exists and size >100KB; retry once with longer warmup.
- Image + event CSV extraction after capture to validate content.
- Timeouts with clear logs (stdout/stderr persisted per step).

## Implementation tasks (concrete)

- Agent (Python):
  - Implement `run_pixtool_cmd(args, working_dir)` → wraps `cmd.exe /c` and builds quoted string.
  - Implement `launch_and_capture(preset, out_wpix, delay_s | ready_marker)`.
  - Add `export_resource(capture, resourceName, out_bin)` and `parse_reservoir(out_bin)` utilities.
  - Add batch presets: `close, far, veryclose, inside`; generate distinct outputs and reports.

- App (C++):
  - Add Named Pipe JSON-RPC server (`\\.\pipe\PlasmaDXControl`), non-blocking.
  - Implement handlers: camera set, feature toggles, mark_ready.
  - Optional fast path: `dump_buffer{name,path}` that CPU-reads and writes target buffers.
  - Log a clear `READY_FOR_CAPTURE` line when scene is ready.

## Example command sequence (WSL → Windows)

- Launch:
  - `C:\Program Files\Microsoft PIX\2509.25\pixtool.exe launch "D:\...\PlasmaDX-Clean\build\Debug\PlasmaDX-Clean.exe" --working-directory="D:\...\PlasmaDX-Clean" --command-line="--config config_pix_close.json"`
- Wait for readiness (log polling or control ACK)
- Capture:
  - `C:\Program Files\Microsoft PIX\2509.25\pixtool.exe take-capture --frames=1 --open save-capture "D:\...\pix\Captures\auto_2025_10_14__00_42_00.wpix"`
- Export reservoir:
  - `C:\Program Files\Microsoft PIX\2509.25\pixtool.exe open-capture "D:\...\pix\Captures\auto_*.wpix" save-resource --name="g_currentReservoirs" "D:\...\pix\Analysis\reservoir.bin"`

## Parsing template (Python)

```python
import struct, statistics as stats

def parse_reservoir(path):
    records = []
    with open(path, 'rb') as f:
        data = f.read()
    stride = 32
    for i in range(0, len(data), stride):
        # float3 lightPos (12), float weightSum (4), uint M (4), float W (4), uint particleIdx (4), uint pad (4)
        lightX, lightY, lightZ, weightSum, M, W, particleIdx, pad = struct.unpack('ffffI f I I'.replace(' ', ''), data[i:i+stride])
        records.append((M, W, weightSum))
    Ms = [r[0] for r in records if r[0] > 0]
    Ws = [r[1] for r in records if r[0] > 0]
    Sums = [r[2] for r in records if r[0] > 0]
    return {
        'count': len(records),
        'M_avg': stats.mean(Ms) if Ms else 0,
        'W_avg': stats.mean(Ws) if Ws else 0.0,
        'weightSum_avg': stats.mean(Sums) if Sums else 0.0,
        'M_max': max(Ms) if Ms else 0,
        'W_max': max(Ws) if Ws else 0.0
    }
```

## Deep-dive analysis (examples)

- ReSTIR: detect M/W inverse correlation; flag double normalization bugs; quantify temporal weight impacts.
- DXR: correlate BLAS/TLAS timings, rays/thread, RT core throughput to frame time; flag rebuilds vs updates.
- Buffers: scan for NaN/Inf, zeroed regions, and unexpected strides; visualize histograms.

## Stretch goals

- Attach mode research: if viable, enable multi-capture without relaunch.
- Programmatic capture path with COM init + `PIXGpuCaptureNextFrames` behind feature flag.
- Automated diffing across captures; HTML dashboard.

## Success criteria

- From WSL, reliably produce `.wpix`, exported buffers, parsed stats, and a markdown+JSON report in one run.
- Live camera/feature changes applied between launch and capture.
- Reservoir stats computed from binary export; actionable findings included in the report.
