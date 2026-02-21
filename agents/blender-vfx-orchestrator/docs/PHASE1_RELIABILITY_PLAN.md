# VFX Orchestrator Phase 1: Reliability Plan

## Context

The blender-vfx-orchestrator (~43,500 LOC) is an autonomous multi-agent VFX system that works but is unreliable — 188 runs, 0 formal passes. The wine pour render (70/100) proves the vision, but the system fails more often than it succeeds. The #1 failure mode is hallucinated Blender 5.0 attributes (e.g., `resolution_divisions` instead of `resolution_max`).

**Goal:** "Every run produces a render the quality analyst can evaluate."

**Approach:** Audit + refactor. The codebase is solid — targeted fixes, not a rewrite. Phase 1 ships reliability. Phase 2 (separate plan) layers SDK features (AdvancedSQLiteSession, HITL, context trimming, conditional tools).

---

## Execution Order

```
[1] Truth Pack ─────────────────────── highest impact, blocks 2
[3] KB Wipe ────────────────────────── independent, start in parallel with 1
[4] Session Compaction ─────────────── surgical, ~30 lines
[2] QA Feedback Loop Fix ───────────── depends on truth pack
[5] Escape Velocity Testing ────────── exercises whole system, do last
```

---

## Work Item 1: Truth Pack Implementation

**Impact:** Eliminates the #1 failure mode. Replaces the reactive 57-rule regex fixer + LLM-powered API Spec Agent with a $0 deterministic prevention layer.

### New Files

**`tools/truth_pack.py` (~400 lines)**
- `INTROSPECTION_SCRIPT` — Python that runs inside Blender headless, queries `bpy.types.X.bl_rna.properties`, outputs JSON with property names, types, ranges, defaults, enum values. Blueprint exists at `docs/ARCHITECTURE_FROM_SCRATCH_V2.md:312-361`.
- `TECHNIQUE_TYPES` — Dict mapping technique → list of bpy.types to introspect. Blueprint at `ARCHITECTURE_FROM_SCRATCH_V2.md:364-389`. Covers mantaflow_gas, mantaflow_liquid, rigid_body, particle_system, plus `_common` base.
- `async def build_truth_pack(technique, blender_exe)` — Writes introspection script to temp, runs `blender --background --python <script> -- <types_json>`, parses output between `###TRUTH_PACK_START###` / `###TRUTH_PACK_END###` markers. Cache to `data/truth_packs/{blender_version}_{technique}.json`.
- `def validate_script_against_truth_pack(script, truth_pack)` — AST/regex check of every `domain_settings.X`, `flow_settings.X`, etc. against truth pack property lists. Returns list of ValidationErrors with line numbers.
- `def format_truth_pack_for_prompt(truth_pack)` — Dense LLM-optimized text listing ONLY valid attributes with types/ranges/defaults.
- `def build_substitution_table(truth_pack)` — Uses `difflib.get_close_matches` to suggest corrections for invalid attributes.

**`tools/truth_pack_validator.py` (~200 lines)**
- `def validate_and_fix_script(script_path, truth_pack)` — Reads script, validates against truth pack, auto-fixes using substitution table, writes fixed script. Returns `(fixed_path, list_of_fixes)`.
- `@function_tool async def validate_script_truth_pack(script_path)` — Tool wrapper for agents.

### Modified Files

**`orchestrator.py` (~150 lines changed)**
- After technique selection (Phase 0.5, ~line 2170): call `build_truth_pack(selected_technique)`. ~2s Blender subprocess replaces ~25s API Spec Agent.
- Replace `_run_spec_first_pipeline` method (lines ~1410-1605) with simpler flow: inject truth pack → Code Writer generates → validate against truth pack → fix automatically.
- Store truth pack in SharedContext.
- When escape velocity triggers technique switch: rebuild truth pack for new technique.

**`models/shared_context.py` (~5 lines)**
- Add `truth_pack: Optional[Dict[str, Any]] = Field(default=None)` to SharedContext.

**`tools/dynamic_instructions.py` (~100 lines)**
- Modify `dynamic_script_writer_standalone_instructions` to inject truth pack as "## TRUTH PACK — ONLY valid attributes" section.

### What Becomes Obsolete
- `specialized_agents/api_spec_agent.py` — APISpec can be built from truth pack (function, not LLM agent). Mark deprecated, don't delete yet.
- Most of `tools/blender_api_fixer.py` — Keep as Layer 3 safety net but truth pack should catch everything. Run both in parallel initially.

### Reuse
- `models/api_spec.py` Pydantic models — Create `truth_pack_to_api_spec()` function to populate APISpec from truth pack. Preserves existing Code Writer guardrail (`validate_code_against_spec`).
- `tools/blender_executor_tools.py` — Already spawns Blender subprocess. Reuse execution patterns.

---

## Work Item 2: Fix QA Feedback Loop

**Impact:** Stops cascading wrong suggestions across iterations. QA currently sees renders but not code — suggests "too dark" instead of "line 245: light energy=10, need 100-500".

### New Files

**`tools/qa_diagnosis_bridge.py` (~150 lines)**
- `def create_code_grounded_feedback(quality_output, script_path, truth_pack)` — Pairs QA visual critique with script analysis.
  - Reuses `_extract_config_class_values`, `_extract_settings_assignments`, `_extract_shader_node_inputs` from existing `tools/script_analysis_tools.py:450 LOC`.
  - Maps QA issues to script locations: "too dark" → check light energy values → "Line 245: light energy=10, truth pack range [0, 1000000], suggest 100-500"
  - Output: dense, LLM-optimized, with line numbers and variable names.

### Modified Files

**`orchestrator.py` (~50 lines)**
- After QA evaluation (Phase 3), before Modification Coordinator (Phase 4): insert diagnosis bridge call. Code-grounded feedback appended to Modification Coordinator prompt.

**`tools/dynamic_instructions.py` (~20 lines)**
- Add code-grounded context to quality analyst instructions.

---

## Work Item 3: Wipe and Rebuild KB

**Impact:** Stops poisoned/stale patterns from misleading the research agent.

### New Files

**`scripts/wipe_kb.py` (~60 lines)**
- Backup all data to `data/backups/pre_phase1_<timestamp>/`
- Wipe: `experiments.db` (recreate empty), `data/code_patterns/*.json`, `data/physics_observations/*.json`, `sessions/sdk/vfx_conversations.db`

### Modified Files

**`tools/code_pattern_tools.py` (~20 lines)**
- Raise minimum auto-apply confidence from 70 → 80 until pattern has 3+ successful uses.

**`tools/dynamic_instructions.py` (~10 lines)**
- Change `min_success_rate` default from 0.7 → 0.8 for stricter evidence gating.

**`utils/code_pattern_memory.py` (~20 lines)**
- Add `last_used_at` to CodePattern. Apply 20% confidence decay if unused for 30+ days.

---

## Work Item 4: Enable Session Compaction

**Impact:** Prevents unbounded token growth that degrades multi-iteration runs.

### Modified Files

**`orchestrator.py` line 375-381 (~15 lines)**
- Replace `should_trigger_compaction=lambda _: False` with token-budget trigger:
  ```python
  should_trigger_compaction=lambda history: sum(
      len(str(item)) for item in (history or [])
  ) > 100_000,  # safety net at ~25K tokens
  ```
- Add error logging to existing manual compaction try/catch at line ~4176.

**Context safety:** Truth pack is in SharedContext (Python-managed, not conversation history). Current script path and quality issues are in SessionState. Both survive compaction.

---

## Work Item 5: Test and Fix Escape Velocity

**Impact:** Verifies that the system actually switches techniques when stuck, instead of looping.

### New Files

**`tests/test_escape_velocity.py` (~150 lines)**
- Unit tests for StuckDetectionState escalation (same issue 2x → Level 2, 3x → Level 3, 4x → Level 4).
- Test `get_untried_techniques()` excludes current technique.
- Test `reset_for_new_technique()` clears counters.
- Integration test: simulate stuck scenario, verify technique switch produces different script.

### Modified Files

**`orchestrator.py` (~50 lines)**
- After Modification Coordinator returns `action='switch_technique'`: verify `new_technique != previous_technique`. If same, force-select from `stuck_state.get_untried_techniques()`.
- At Level 4 (REQUEST_GUIDANCE): set `session.status = PAUSED`, log what was tried/failed, save session for resume.

---

## Phase 2 Preview (Separate Plan, After Phase 1 Ships)

Once reliability is proven (3 consecutive runs producing evaluable renders):
- `AdvancedSQLiteSession` — branching for technique experiments
- `call_model_input_filter` — context bloat prevention
- `needs_approval` HITL — pipeline pauses at stall detection
- `is_enabled` conditional tools — budget-aware tool hiding
- `tool_use_behavior: stop_on_first_tool` — deterministic executor
- Monitoring agent/layer design

---

## Verification Plan

### Unit Tests
- `tests/test_truth_pack.py` — Introspection output parsing, caching, validation against known-good/bad scripts
- `tests/test_qa_bridge.py` — Issue-to-code mapping with sample scripts
- `tests/test_escape_velocity.py` — Stuck detection escalation and step-down

### Integration Tests
- `tests/test_truth_pack_integration.py` — Run introspection on actual Blender 5.0, verify `FluidDomainSettings` has `resolution_max` and NOT `resolution_divisions`
- `tests/test_pipeline_single_iteration.py` — One iteration with truth pack, verify no hallucinated attributes

### End-to-End Validation
Run 3 different effect types (fire, liquid, smoke). Each must produce at least one evaluable render. Scores don't need to pass 60 — just produce an image the QA can evaluate.

### Success Criteria
1. 90%+ of runs produce a render (up from ~50%)
2. Zero hallucinated attributes in generated scripts
3. QA feedback references actual script line numbers
4. KB is clean — no stale patterns, new patterns need 3+ successes
5. Escape velocity Level 2 technique switch verified working
6. 5+ iteration runs don't fail from context overflow

---

## Critical Files Reference

| File | Role |
|------|------|
| `orchestrator.py` (3,514 LOC) | Main pipeline — truth pack integration, compaction, escape velocity |
| `docs/ARCHITECTURE_FROM_SCRATCH_V2.md:312-499` | Truth pack design blueprint (introspection script, TECHNIQUE_TYPES, validator) |
| `tools/blender_api_fixer.py` (1,962 LOC) | Existing 57-rule fixer — becomes Layer 3 safety net |
| `specialized_agents/api_validator.py` (907 LOC) | Existing validator — partially replaced by truth pack |
| `specialized_agents/api_spec_agent.py` (295 LOC) | LLM APISpec agent — replaced by truth_pack_to_api_spec() |
| `models/shared_context.py` | Add truth_pack field; StuckDetectionState for escape velocity |
| `models/api_spec.py` (467 LOC) | Pydantic models — REUSE, populate from truth pack |
| `tools/dynamic_instructions.py` (1,082 LOC) | Inject truth pack into ScriptWriter instructions |
| `tools/script_analysis_tools.py` (450 LOC) | Reuse extractors for QA diagnosis bridge |
| `tools/blender_executor_tools.py` (926 LOC) | Reuse subprocess patterns for introspection |
| `tools/code_pattern_tools.py` (492 LOC) | Tighten evidence gating |
| `utils/code_pattern_memory.py` | Add memory decay |

## Estimated Scope
- ~600 lines new code (truth_pack.py, truth_pack_validator.py, qa_diagnosis_bridge.py, wipe_kb.py)
- ~150 lines test code (test_escape_velocity.py + other test files)
- ~450 lines modified across existing files
- **Total: ~1,200 lines of work**


If you need specific details from before exiting plan mode (like exact code snippets, error messages, or content you generated), read the full transcript at: /home/maz3ppa/.claude/projects/-home-maz3ppa-projects-PlasmaDXR/3943460b-35da-4965-9ee8-a501462db5f4.jsonl

If this plan can be broken down into multiple independent tasks, consider using the TeamCreate tool to create a team and parallelize the work.