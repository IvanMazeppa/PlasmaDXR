# Tracing Shakedown Tasklist (2026-01-25)

> Consolidated issue list: `docs/CURRENT_ISSUES_CONSOLIDATED_2026-01-26.md`

## Purpose
Convert the latest tracing shakedown findings into a concrete, ordered tasklist
with quick wins and code snippets. Code changes listed below have now been applied.

## Evidence (latest run)
- Execution failure: "FluidDomainSettings object has no attribute 'noise_res_factor'"
- Modification coordinator emitted nested/prose parameters, and `_modify_script_impl`
  applied zero changes (changes_made: [])
- Hooks reported doc_query_made: False even though blender_doc_search_bundle ran
- Doc refs now point to real DocPath values (this is improved and should be kept)

---

## Quick Wins (same day)

### QW1. Block/remove `noise_res_factor` at validation time
- [x] Add a KNOWN_API_CHANGES entry so the validator flags/removes it
- Why: API validator currently returns 10/10 valid even with a removed attr
- Target: `agents/blender-vfx-orchestrator/specialized_agents/api_validator.py`

Suggested snippet:
```
# KNOWN_API_CHANGES (add this entry)
".noise_res_factor": {
    "correction": "# noise_res_factor removed in Blender 5.0 - delete this line",
    "reason": "FluidDomainSettings.noise_res_factor removed in Blender 5.0"
},
```

### QW2. Auto-strip `noise_res_factor` before execution
- [x] Add a pre-execution fixer rule to remove the line
- Why: Prevents repeat crashes if the validator misses it
- Target: `agents/blender-vfx-orchestrator/tools/blender_api_fixer.py`

Suggested snippet:
```
# BLENDER_50_FIXES (add a pattern)
(
    r"(?m)^.*\.noise_res_factor\s*=\s*.*$",
    r"# Blender 5.0: noise_res_factor removed (line deleted)",
    "noise_res_factor removed"
),
```

### QW3. Update Script Writer guidance for noise/upres
- [x] Explicitly forbid noise_res_factor and show the correct bake flow
- Why: Prevents the error from being generated at the source
- Target: `agents/blender-vfx-orchestrator/tools/dynamic_instructions.py`

Suggested snippet:
```
### Noise/Upres (Blender 5.0)
- DO NOT use dsettings.noise_res_factor (removed in 5.0)
- Use:
  dsettings.use_noise = True
  dsettings.noise_strength = 0.7
  dsettings.noise_scale = 1.0
  bpy.ops.fluid.bake_data()
  bpy.ops.fluid.bake_noise()
```

### QW4. Fix doc query detection for the new bundle tool
- [x] Add blender_doc_search_bundle to doc_query_tools
- Why: Hooks currently report doc_query_made=False even when bundle runs
- Target: `agents/blender-vfx-orchestrator/hooks/enforcement_hooks.py`

Suggested snippet:
```
doc_query_tools: List[str] = field(default_factory=lambda: [
    "semantic_search_blender_docs",
    "search_blender_api_by_intent",
    "blender_doc_search_bundle",
    "find_alternative_approaches",
    "search_code_patterns",
    "query_docs",
])
```

---

## Short-Term (1-2 days)

### ST1. Enforce flat parameter_changes in Modification Coordinator
- [x] Reject nested/prose structures in `validate_modification_decision`
- Why: `_modify_script_impl` can only apply flat key/value pairs; nested dicts
  result in changes_made: []
- Target: `agents/blender-vfx-orchestrator/guardrails/coordinator_guardrails.py`

Suggested snippet (add inside validate_modification_decision):
```
if action == "modify_params" and isinstance(parameter_changes, dict):
    nested = [k for k, v in parameter_changes.items()
              if isinstance(v, (dict, list))]
    if nested:
        errors.append(
            "parameter_changes must be flat (no nested dict/list). "
            f"Nested keys: {nested}"
        )
```

Optional hardening (if you want to be strict):
```
# Optional allowlist check (define ALLOWED_CONFIG_KEYS elsewhere)
invalid = [k for k in parameter_changes.keys() if k not in ALLOWED_CONFIG_KEYS]
if invalid:
    errors.append(f"Unknown config keys: {invalid}")
```

### ST2. (Optional) Add a fallback mapping layer in modify_script
- [ ] Map human-readable keys to Config keys as a last resort
- Why: Provides a safety net if the coordinator still leaks prose
- Target: `_modify_script_impl` in script generator tooling

Suggested pattern:
```
HUMAN_TO_CONFIG = {
    "remove_property": None,  # handled separately
    "noise_enable": "domain_use_noise",
    # ...
}
```
Note: Only use if guardrail is not enough; guardrail is the preferred fix.

---

## Verification Checklist (after fixes)
- [ ] 2-iteration shakedown completes without noise_res_factor errors
- [ ] changes_made is non-empty after modification coordinator step
- [ ] doc_query_made is True when blender_doc_search_bundle is used

---

## Status Notes
- Doc refs now resolve to real DocPath values. Keep this behavior.
- API validator now blocks noise_res_factor (fix applied).
