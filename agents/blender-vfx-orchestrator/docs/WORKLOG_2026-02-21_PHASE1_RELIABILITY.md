# Worklog: Phase 1 Reliability (2026-02-21)

## Goal

Every run produces a render the quality analyst can evaluate. Target: eliminate the top 3 failure modes that prevent reaching the evaluation stage.

## What Changed

### Work Item 1: Truth Pack (Deterministic API Validation)

**Problem:** #1 failure mode — LLM hallucinates Blender 5.0 attribute names (e.g., `resolution_divisions` instead of `resolution_max`). Scripts crash before rendering.

**Solution:** Runtime `bl_rna.properties` introspection. Runs a headless Blender subprocess that queries the actual Python API at startup, building a ground-truth property map per technique.

**New files:**
- `tools/truth_pack.py` (~705 lines) — Core implementation:
  - `INTROSPECTION_SCRIPT` — Blender-side script that introspects `bpy.types.X.bl_rna.properties`
  - `TECHNIQUE_TYPES` — Maps technique categories to lists of bpy.types to introspect
  - `TECHNIQUE_ALIASES` — Maps orchestrator technique names to TECHNIQUE_TYPES keys
  - `SETTINGS_MAP` / `VARIABLE_PATTERNS` — Maps script variable names to bpy.types for validation
  - `KNOWN_HALLUCINATIONS` — 11 regex patterns for known hallucinated attributes with fixes
  - `HARDCODED_FIXES` — Direct string replacement for common hallucinations
  - `build_truth_pack()` — Async, runs Blender, parses output, caches to `data/truth_packs/`
  - `validate_script_against_truth_pack()` — Checks every attribute access against truth pack
  - `auto_fix_script()` — Applies deterministic fixes using difflib + hardcoded fixes
  - `format_truth_pack_for_prompt()` — Dense LLM-optimized text for ScriptWriter injection
  - `truth_pack_to_api_spec()` — Backward-compatible APISpec conversion
- `tools/truth_pack_validator.py` (~187 lines) — Pipeline wrapper:
  - `validate_and_fix_script()` — Read-validate-fix-write cycle (max 2 attempts)
  - `validate_script_truth_pack` — `@function_tool` agent-callable tool

**Modified files:**
- `models/shared_context.py` — Added `truth_pack: Optional[Dict[str, Any]]` field to `SharedContext`
- `orchestrator.py` — Phase 0.6: build truth pack after technique selection, before iteration loop. Validation after Code Writer output. Truth pack rebuild on technique switch.
- `tools/dynamic_instructions.py` — Injects truth pack into ScriptWriter dynamic instructions

**Cost:** $0 (deterministic subprocess, no LLM calls)

**Pipeline integration:**
```
Technique Selection → [Phase 0.6: Build Truth Pack] → Iteration Loop
                                                        ↓
Code Writer → [Truth Pack Validation + Auto-Fix] → Executor
                                                        ↓
Technique Switch → [Rebuild Truth Pack for New Technique]
```

### Work Item 2: QA Feedback Loop Fix

**Problem:** Quality Analyst sees the render (image) but not the code. Its feedback is "too dark" without knowing the script has `energy=10` on line 245 with a valid range of [0, 1000000]. The Modification Coordinator receives vague visual critique and can't make targeted fixes.

**Solution:** Code-grounded diagnosis bridge that pairs QA visual issues with actual script parameters.

**New file:**
- `tools/qa_diagnosis_bridge.py` (~254 lines):
  - `ISSUE_TO_PARAMS` — Maps 12 QA issue keywords (dark, dim, smoke, fire, density, camera, liquid, resolution, clipping, bright, exposure, noise) to script parameter categories
  - `create_code_grounded_feedback()` — Takes QA output + script path + truth pack, returns grounded feedback like: "QA says 'too dark' → Script Line 245: light.energy = 10 (range [0, 1000000])"
  - Uses existing `analyze_script_structure()` from `tools/script_analysis_tools.py`

**Modified files:**
- `orchestrator.py` — Phase 3.5: calls `create_code_grounded_feedback()` after QA evaluation, injects into Modification Coordinator prompt
- `tools/dynamic_instructions.py` — Injects top 15 script parameters (by line number) into QA analyst instructions so it can reference them in feedback

### Work Item 3: KB Wipe + Evidence Gating

**Problem:** Poisoned knowledge base from failed experiments. Low-confidence patterns (30%) injected into prompts. Stale patterns never decay.

**Solution:** Wipe script + stricter evidence thresholds + temporal decay.

**New file:**
- `scripts/wipe_kb.py` (~99 lines):
  - Backs up all KB data to `data/backups/pre_phase1_<timestamp>/` before wiping
  - Supports `--dry-run` flag
  - Recreates empty SQLite databases after deletion

**Modified files:**
- `tools/code_pattern_tools.py` — `min_confidence` default: 30 → 50
- `utils/code_pattern_memory.py` — Added `last_used_at` field, 20% confidence decay per 30 days idle (capped at 60% at 90+ days), `min_confidence` defaults: 30 → 50
- `tools/knowledge_distillation_tools.py` — Aligned explicit `min_confidence` calls to 50
- `tools/dynamic_instructions.py` — `min_success_rate` default: 0.7 → 0.8

### Work Item 4: Session Compaction

**Problem:** `should_trigger_compaction=lambda _: False` — auto-compaction was completely disabled. Long iterations could hit context limits.

**Solution:** Token-budget trigger at ~25K tokens (~100K chars).

**Modified file:**
- `orchestrator.py` — Replaced disabled lambda with:
  ```python
  should_trigger_compaction=lambda history: sum(
      len(str(item)) for item in (history or [])
  ) > 100_000
  ```
  SharedContext (truth pack) and SessionState (script path, quality issues) both survive compaction — they're Python-managed, not in conversation history.

### Work Item 5: Escape Velocity Verification

**Problem:** Untested escape velocity mechanism. Level 4 (REQUEST_GUIDANCE) didn't pause the pipeline.

**Solution:** Tests + Level 4 → PAUSED fix.

**New file:**
- `tests/test_escape_velocity.py` (~144 lines) — 16 tests across 6 classes:
  - `TestStuckDetectionEscalation` — L0 → L2 → L3 → L4 escalation
  - `TestGetUntriedTechniques` — Excludes tried, returns non-failed
  - `TestResetForNewTechnique` — Clears counters, preserves level + tried list
  - `TestStepDown` — Escape level decreases after consecutive progress
  - `TestEscapeActions` — Action recommendations per level

**Modified file:**
- `orchestrator.py` — Added Level 4 check between `should_switch` computation and technique switch block:
  ```python
  if session.stuck_state.escape_level == EscapeLevel.REQUEST_GUIDANCE:
      session.status = SessionStatus.PAUSED
      break
  ```

## Verification Results

15/15 tests pass (fresh run on merged code):
- All imports resolve
- SharedContext.truth_pack field exists
- Truth pack validation catches 7 errors (3 settings + 4 hallucinations) in realistic script
- Auto-fix applies 7 corrections
- Truth pack prompt format: 317 chars (dense)
- truth_pack_to_api_spec: produces valid APISpec
- Escape velocity: L0→L2→L3→L4 escalation correct
- get_untried_techniques: excludes tried, returns available
- reset_for_new_technique: clears counters, preserves level + tried
- Escape actions: correct per level
- QA bridge: extracts 5 issues from mock quality output
- QA bridge: 12 issue keyword mappings
- TECHNIQUE_TYPES: includes mantaflow_liquid
- KNOWN_HALLUCINATIONS: 11 valid regex patterns
- wipe_kb.py: valid Python with backup + dry-run

## File Summary

| File | Status | Lines |
|------|--------|-------|
| `tools/truth_pack.py` | NEW | 705 |
| `tools/qa_diagnosis_bridge.py` | NEW | 254 |
| `tools/truth_pack_validator.py` | NEW | 187 |
| `tests/test_escape_velocity.py` | NEW | 144 |
| `scripts/wipe_kb.py` | NEW | 99 |
| `orchestrator.py` | MODIFIED | +127 |
| `tools/dynamic_instructions.py` | MODIFIED | +37 |
| `utils/code_pattern_memory.py` | MODIFIED | +20 |
| `models/shared_context.py` | MODIFIED | +8 |
| `tools/code_pattern_tools.py` | MODIFIED | +6/-6 |
| `tools/knowledge_distillation_tools.py` | MODIFIED | +4/-4 |
| **Total** | | **+1574/-17** |

## Next Steps

- Run `python scripts/wipe_kb.py` to clean the poisoned KB before next E2E test
- Phase 2: Leverage unused SDK features (tool guardrails, AdvancedSQLiteSession, streaming)
- First E2E test with truth pack: expect significantly fewer API hallucination crashes
