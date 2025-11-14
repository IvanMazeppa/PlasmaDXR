# Agent Hierarchy and Roles – NVIDIA-Style Multi‑Agent RAG for PlasmaDX

Created: 2025‑11‑14
Owner: Orchestrator (mission-control)

---

## Why a Hierarchy
- Prevent “too many cooks” by introducing clear tiers of authority and scope.
- Keep agents specialized but composable, with mission-control assigning work and verifying artifacts.
- Persist context via artifacts (screenshots, logs, reports) so work survives token resets.

---

## Tiered Structure
1) Orchestration & Governance
- mission-control (in‑process SDK server)
  - Plans sprints, assigns tasks, aggregates QA, produces session summaries.
  - Tools: dispatch_plan, record_decision, trigger_review, publish_summary.

- knowledge-archivist (in‑process SDK server)
  - Maintains registry of artifacts (docs, configs, screenshots, PIX, logs), and cross‑links to tickets.
  - Tools: register_artifact, search_artifacts, summarize_thread, export_memory.

2) Production & Execution
- particle-pipeline-runner (in‑process)
  - Runs shader rebuilds, launches app with scenario configs, drives hotkeys, captures artifacts.
  - Tools: rebuild_shaders, run_plasmadx, capture_screenshot, collect_logs, run_pix.

- imageops-agent (in‑process)
  - Autonomous world traversal (camera paths), image + metadata capture, A/B batch runs.
  - Tools: set_camera_pose, sweep_path, capture_series, tag_metadata.

3) Rendering Technology Specialists (external MCP)
- rt-lighting-engineer
  - RayQuery particle‑to‑particle lighting, emission models, intensity coupling.
  - Scope: `particle_gaussian_raytrace.hlsl`, lighting CBs, emission coupling.

- rt-shadow-engineer (existing dxr-shadow-engineer)
  - RayQuery volumetric shadows; PCSS → raytraced migration; temporal accumulation.
  - Scope: shadow rays, soft shadows, temporal buffers, denoising interfaces.

- sampling-and-distribution (RTXDI/ReSTIR)
  - Light sampling strategies (RTXDI M4/M5/M6), spatial reuse, reservoir diagnostics.
  - Scope: `rtxdi_raygen.hlsl` + temporal passes; configs; quality presets.

- path-and-probe (path tracing & probe grid)
  - Probe‑grid lighting and any path/volume interpolation experiments.
  - Scope: Probe structs/buffers, sampling & trilinear, grid updates.

- volumetric-materials (gaussian-analyzer + material-system-engineer)
  - Material schema, per‑material noise/rim/temporal curves, celestial presets.
  - Scope: `MaterialTypeProperties`, ImGui, JSON presets, shader material blocks.

- pyro-effects (existing dxr-volumetric-pyro-specialist)
  - Supernova/prominence specs: temporal expansion, temperature/emission ramps.

4) Diagnostics & QA (external MCP)
- dxr-image-quality-analyst (existing)
  - LPIPS/SSIM/PSNR, seven‑dimension rubric, heatmaps, FPS parsing.

- log-analysis-rag (existing)
  - Log/PIX ingestion, semantic queries, hypothesis generation, confidence scoring.

- pix-debuggers (existing family)
  - Capture/validate GPU buffers and PIX traces.

---

## Recommended Granularity (RT Domain)
- Start with 4 RT specialists (balanced):
  1) rt-lighting-engineer
  2) rt-shadow-engineer
  3) sampling-and-distribution (RTXDI/ReSTIR)
  4) path-and-probe (probe-grid + path/volume experiments)

- Add granular agents only when tool surfaces exceed ~10 tools or the domain demands isolation (e.g., a dedicated RTXDI M6 agent during heavy development).

- Avoid splitting by API ("RayQuery agent" vs "RTXDI agent") unless work streams are concurrent and deep; prefer splitting by problem domain (lighting vs shadows vs sampling vs probe/path).

Target roster size: 8–12 total agents (including orchestration and QA). This balances specialization with cognitive/manageable overhead.

---

## Division of Labor
- mission-control assigns goals → specialists produce artifacts → diagnostics grade → mission-control accepts/rejects.
- imageops-agent and particle-pipeline-runner provide common services so specialists don’t duplicate screenshot/log/orchestration code.
- knowledge-archivist ensures everything is discoverable across sessions.

---

## Escalation & Hand‑Off Rules
- Any agent discovering blocked dependencies escalates to mission-control with a suggested sub‑plan.
- Specialists must attach artifacts (file paths) and a one‑page summary to every change.
- Diagnostics agents always attach metrics (LPIPS, SSIM/PSNR, FPS, PIX hotspots, log excerpts).

---

## Persistence Strategy (Token‑Proof)
- All decisions summarized to `docs/SESSION_SUMMARY_<date>.md`.
- Artifacts strictly stored under `screenshots/`, `PIX/`, `logs/`, and `docs/` with consistent naming.
- knowledge-archivist maintains an index and answers “where is X?” queries.
