# Shakedown Test Findings (2026-01-24)

Test: `run_shakedown.sh fire` (2 iterations, preset `quick_test`)

---

## What happened (observed)
- Research phase ran cleanly and produced structured output with `doc_refs`.
- Iteration 1 failed in Executor:
  - **Error:** `'FluidDomainSettings' object has no attribute 'use_caching'`
- Iteration 2 attempted coordinator‑driven fixes but the same error persisted.
- Session ended at **MAX_ITERATIONS** with **best score 0.0** (no render produced).

Additional warning on startup:
```
ExperimentTracker.add_manual_learning() missing 1 required positional argument: 'rule'
```

---

## Root cause analysis
1) **Blender 5.0 API mismatch not fully blocked**
   - `use_caching` removed in Blender 5.0.1 but slipped through into the script.
2) **Coordinator “remove line” instructions were ignored**
   - `_modify_script_impl` only changes Config parameters; it did not remove invalid lines.
3) **Knowledge base loader expects `rule` even for warning‑only entries**
   - JSON warnings/common_issues called `add_manual_learning()` without `rule`.

---

## Fixes applied (2026-01-24)
1) **API Fixer update**
   - Added a Blender 5.0 fix to **remove** any `.use_caching = ...` line before execution.
2) **API Validator update**
   - Added `.use_caching` to `KNOWN_API_CHANGES` with a removal correction.
3) **Modify‑script deletion support**
   - `_modify_script_impl` now recognizes directives like:
     - `"settings.use_caching (delete this line)"`
   - It comments out the matching line safely.
4) **Knowledge base loader**
   - `add_manual_learning()` is now called with `rule=""` for warnings/common issues.

Files changed:
- `tools/blender_api_fixer.py`
- `specialized_agents/api_validator.py`
- `tools/script_generator_tools.py`
- `agents/experiment-tracker/tracker.py`

---

## Next step
Re‑run the 2‑iteration shakedown to confirm:
- `use_caching` is removed or commented out
- Coordinator removal instructions are applied
- Knowledge base warnings load without errors

