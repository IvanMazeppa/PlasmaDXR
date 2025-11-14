# Runbook – Multi‑Agent RAG (Nightly + Ad‑Hoc)

Created: 2025‑11‑14
Owner: mission-control

---

## Nightly Pipeline (UTC 02:00)
1) Build & Launch
- pipeline-runner.rebuild_shaders
- pipeline-runner.run_plasmadx (standard scenario, 120s)

2) Artifact Capture
- pipeline-runner.capture_screenshot (baseline tag)
- pipeline-runner.collect_logs
- pipeline-runner.run_pix (quick preset @ frame 120)

3) Ingestion & QA
- log-analysis-rag.ingest_logs (include_pix=true)
- dxr-image-quality-analyst.assess_visual_quality (latest screenshot)

4) Reporting
- mission-control.publish_summary → docs/sessions/SESSION_<date>.md
- knowledge-archivist.register_artifact for all new files

5) Alerts
- If QA < threshold or diagnosis confidence > 0.7 for regressions → open TODO items and tag agents

---

## Ad‑Hoc Debug Loop (Developer Triggered)
1) State the problem in plain language (e.g., “RTXDI M5 patchwork persists”).
2) mission-control dispatches a micro‑plan:
- sampling-and-distribution → analyze RTXDI M5 temporal buffers
- dxr-image-quality-analyst → capture + compare before/after
- pix-debuggers → capture buffer stats

3) Specialists attach artifacts and a one‑pager.
4) Diagnostics grade and propose fixes with file:line.
5) mission-control merges decisions and records them.

---

## Quality Gates
- LPIPS change > 0.30 → require visual justification and QA note
- FPS regression > 15% → require perf analysis summary
- Shadow/lighting changes → require PIX trace hotspot list

---

## Artifact Locations
- Screenshots: `screenshots/`
- Logs: `logs/`
- PIX: `PIX/Captures/`, `PIX/buffer_dumps/`
- Session reports: `docs/sessions/`

---

## Recovery Playbook
- If a build fails → build-analyzer (if present) or mission-control escalates
- If crashes at startup → log-analysis-rag.diagnose_issue with last 10 logs
- If visuals black or flat → image QA + shader-analyzer + pix-debuggers resource barrier checks

---

## Governance
- All decisions go through mission-control
- All artifacts registered via knowledge-archivist
- Weekly gallery refresh (golden scenes) and QA grading
