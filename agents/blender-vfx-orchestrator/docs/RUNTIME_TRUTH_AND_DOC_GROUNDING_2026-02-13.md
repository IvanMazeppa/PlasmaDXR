# Runtime Truth and Documentation Grounding (2026-02-13)

**Status:** Authoritative runtime contract for the current orchestrator implementation.  
**Use this to override older historical notes in archived/postmortem docs.**

---

## Runtime Truth (Current)
- **OpenAI Agents SDK runtime pin:** `openai-agents==0.8.3` (`agents/blender-vfx-orchestrator/requirements.txt`).
- **Code-critical rollout preset:** `codex_upgrade` now upgrades `script_writer`, `modification_coordinator`, and `research_agent` to `gpt-5.4` (preset name retained for backward compatibility).
- **Spec-first pipeline mode:** enabled by default via `ORCHESTRATOR_SPEC_FIRST=1`.
- **Trace metadata:** includes `generation_mode=spec_first|legacy` for each run.
- **Script modification path:** orchestrator routes modifications through centralized `_apply_script_modifications(...)`; underlying `_modify_script_impl` is invoked in one guarded path with hallucination scanning.
- **Failure routing:** EXECUTE and artifact-gate failures emit explicit `DIAGNOSE -> FIX` artifacts before retry/next-iteration remediation.
- **Quality artifact policy:** missing/empty quality artifact is a hard runtime failure.

## Documentation Grounding Contract
- `doc_refs` must be **canonical Blender doc paths**, not placeholders.
- Rejected `doc_refs`:
  - sentinels (`doc_search_empty`, `doc_search_unavailable`, etc.),
  - blank/null/unknown values,
  - temporary filenames (`*.md` chunk names).
- Research output must include **at least one API reference** (`bpy.types.*` or `bpy.ops.*` path form).
- If research guardrail grounding fails, run is aborted (no ungrounded fallback path).

## Blender API Truth Notes (Critical)
- `FluidDomainSettings.time_scale` is valid in Blender 5 API docs.
- Treat these as hallucinated/invalid:
  - `resolution_divisions`,
  - `use_adaptive_time_steps`,
  - `timesteps_per_frame`,
  - `timesteps_maximum`,
  - `velocity_multi`.

## Historical-Docs Warning
- Many docs dated before **2026-02-13** are investigative/historical and may describe behavior that has since changed.
- When conflicts exist, prefer:
  1. This file,
  2. `docs/VERSION_TRUTH.md`,
  3. code in `orchestrator.py`, `guardrails/`, and `tools/`.
