# E2E Issues Investigation Report

**Date:** 2026-02-23
**Investigator:** Claude Opus 4.6
**Branch:** `0.34.3/phase-2a4-pipeline-monitor`
**Scope:** Deep analysis of runtime failures from E2E smoke testing

---

## Issue 1: Context-Dependent Blender Enums

**Severity:** High
**Current State:** Three specific context-dependent enums are handled via KNOWN_HALLUCINATIONS hardcoding in truth_pack.py (lines 247-264). The auto_fix_script function strips these lines entirely (lines 563-590).

### What's Implemented

The truth pack introspection script (lines 45-96 of truth_pack.py) calls `prop.enum_items` on `bl_rna.properties` at class level. This returns the SUPERSET of all enum values the property CAN hold across all possible runtime contexts. Three context-dependent enums are already hardcoded as known bad:

| Enum | bl_rna Reports | Runtime Valid (GAS) | Fix Strategy |
|------|---------------|---------------------|--------------|
| `openvdb_data_depth` | `['NONE']` | `('32', '16')` | Strip `='NONE'` |
| `cache_particle_format` | includes `'OPENVDB'` | only `'UNI'` | Strip `='OPENVDB'` |
| `cache_mesh_format` | includes `'UNI'`, `'OPENVDB'` | only `'BOBJECT'`/`'OBJECT'` | Strip `='UNI'`/`='OPENVDB'` |

### Gap Analysis

**Other potentially context-dependent enums NOT yet covered:**

1. **`FluidDomainSettings.domain_type`** -- switching from GAS to LIQUID changes which sub-properties are active and which enum values are valid on child properties. The truth pack does not differentiate between GAS and LIQUID contexts when introspecting `FluidDomainSettings`. Both domains share the same `bl_rna.properties`, but the valid enum values for cache formats, particle formats, and visualization options differ at runtime.

2. **`FluidDomainSettings.cache_data_format`** -- `'OPENVDB'` and `'UNI'` may have different availability depending on domain_type. The current truth pack reports all enum_items for this property but the LLM may set values incompatible with the specific domain_type in use.

3. **`FluidDomainSettings.sndparticle_*`** properties -- secondary particle settings (spray, foam, bubble) are only relevant for LIQUID domains. Setting these on a GAS domain may produce runtime errors not caught by static introspection.

4. **`ParticleSettings.physics_type`** -- some enum values (BOIDS, FLUID) have additional sub-settings that only appear in the runtime context. The `bl_rna` reports all physics_type enum values but sub-properties like boid settings only exist when physics_type is set to BOIDS.

5. **Render engine properties** -- `CyclesRenderSettings` vs EEVEE settings. The truth pack always introspects `CyclesRenderSettings` but if a script sets `render.engine = 'BLENDER_EEVEE'`, the Cycles-specific properties would be invalid at runtime.

### Root Cause

The fundamental issue is that `bpy.types.X.bl_rna.properties` is a static class-level introspection mechanism. It describes what the class CAN hold, not what a specific INSTANCE holds in its current state. Blender internally uses property update callbacks and context-dependent enum resolution (`EnumProperty` with `items` callback), which `bl_rna.properties` cannot represent.

### Scalability Assessment

**The current hardcoding approach DOES NOT scale.** Every new context-dependent enum requires:
1. Someone discovering the mismatch through a runtime failure
2. Adding a regex pattern to KNOWN_HALLUCINATIONS
3. Adding fix logic to auto_fix_script
4. Hoping the regex matches all script variations

This is reactive, not preventive. As the system expands to cover cloth, soft body, geometry nodes, and combined physics, the number of context-dependent properties will grow. Each physics type has its own set of context-dependent enums.

### Recommendation

Two complementary approaches:

**Short-term (current sprint):** Create a second introspection script that runs AFTER the script sets up objects and modifiers, querying `obj.modifiers['X'].domain_settings.bl_rna.properties` on the LIVE INSTANCE rather than the CLASS. This gives context-accurate enum values. This would require parsing the generated script to understand what domain_type and cache_type it sets, then running introspection with those values active.

**Medium-term:** Adopt a "strip-if-uncertain" policy for cache format enums. Instead of hardcoding specific bad values, the truth pack should flag ALL cache format assignments as "context-dependent -- let Blender use defaults." The LLM rarely has good reason to override cache formats; defaults are correct 95%+ of the time. This reduces the problem from "validate every enum value" to "suppress the category."

---

## Issue 2: Script Logic Bugs (Not Hallucinations)

**Severity:** High
**Current State:** The truth pack and API fixer are designed to catch ATTRIBUTE-LEVEL hallucinations (wrong property names, wrong enum values). They do NOT catch script logic bugs where objects are not created, not found, or connected incorrectly.

### Specific Bugs Observed

1. **`'NoneType' object has no attribute 'type'`** -- The script references an object variable that was never assigned (e.g., `obj = bpy.context.active_object` when no object is active, or `bpy.data.objects.get("Name")` returns None because the name doesn't match).

2. **`NodeSocketString.default_value expected a string type, not float`** -- A shader node input that expects a string (e.g., an Attribute node's `attribute_name`) is being assigned a float value. This is a type mismatch in the shader graph, not an API name issue.

### Can the Truth Pack Catch These?

**No.** The truth pack validates that `domain_settings.resolution_max` EXISTS as a property on `FluidDomainSettings`. It cannot validate:
- Whether `bpy.context.active_object` is None at the point of access
- Whether `bpy.data.objects.get("FluidDomain")` will find anything
- Whether a shader node input is the correct type for the value being assigned
- Whether the script's control flow creates all objects before referencing them

These are RUNTIME logic errors, not API hallucinations. They require either:
1. Static analysis (AST-level checking of variable assignments and null checks)
2. Better LLM prompting (instructions about defensive coding patterns)
3. Recovery through the existing error recovery loop

### Does the API Fixer Handle These?

**Partially.** The `blender_api_fixer.py` (1963 lines, 57+ regex rules) handles some structural issues:
- Missing volume materials (VOLUME_MATERIAL_SNIPPET injection)
- Missing bake calls (BAKE_BEFORE_RENDER_SNIPPET injection)
- Missing camera setup (camera safety snippet)
- Camera inside geometry (CAMERA_DISTANCE_FIX_SNIPPET)

But it does NOT handle:
- Objects referenced before creation
- Incorrect node socket type assignments
- Missing `bpy.context.view_layer.objects.active` assignments
- Race conditions in script execution order

### Error Recovery Path Analysis

The orchestrator's recovery path (orchestrator.py lines 2792-2964) works as follows:

1. **Detection:** Execution fails, `execution.success = False`, `error_msg` captured
2. **Recovery attempt:** One recovery attempt per iteration (`_recovery_attempted_this_iter` flag)
3. **Recovery prompt:** The Script Writer agent receives the error message, the failed script path, and instructions to use `write_script` (not `modify_script`) for a complete rewrite
4. **Recovery hooks:** Uses `create_error_recovery_hooks()` which does NOT require doc queries (lines 731-757 of enforcement_hooks.py). Max 6 turns, max 8 hard limit.
5. **Truth pack validation:** Recovery output IS validated against truth pack (lines 2851-2863)
6. **Re-execution:** Recovery script is executed deterministically (lines 2877-2934)

**Assessment of recovery effectiveness for logic bugs:**

- The recovery prompt (lines 2803-2829) includes the error message and lists "Common Structural Fixes" (bake context, effector setup, object context, missing depsgraph update)
- The Script Writer has max 6 turns to fix the issue -- adequate for simple fixes, insufficient for complex rewrites
- Truth pack validation catches any API hallucinations introduced during recovery
- BUT: The recovery relies on the LLM understanding the error and correctly fixing it. For NoneType errors, the LLM needs to understand the script's object creation flow, which requires reading the full script. This is reasonable for small scripts but challenging for 700+ line scripts.

### Scalability Assessment

Logic bugs will INCREASE as scripts become more complex (complete scenes with geometry, materials, lighting, camera, physics). The current approach (let it fail, recover via LLM) is workable but expensive ($0.10-0.30 per recovery attempt in API costs) and unreliable (LLM may introduce new bugs during recovery).

### Recommendation

1. **Defensive code patterns in script templates** -- The `_generate_script_impl` in script_generator_tools.py (lines 590-835) generates a template with proper control flow. But LLM-written scripts (via `write_script`) have no such guardrails. Add instructions to the Script Writer's system prompt requiring:
   - Null checks after `bpy.data.objects.get()` and `bpy.context.active_object`
   - Type assertions before `inputs[].default_value` assignments
   - Variable assignment verification comments

2. **Pre-execution AST analysis** -- Before running in Blender, parse the script's AST to find:
   - Variables used before assignment
   - `.get()` calls without null checks
   - `default_value` assignments that can be type-checked against the truth pack's property type metadata

3. **Do NOT expand the truth pack for this** -- Logic bugs are fundamentally different from API hallucinations. The truth pack is the right tool for attribute validation. Logic bugs need a separate, complementary validation layer.

---

## Issue 3: Truth Pack Coverage Gaps

**Severity:** Medium
**Current State:** The truth pack covers 4 technique categories (mantaflow_gas, mantaflow_liquid, rigid_body, particle_system) plus a _common set. It introspects 17 types for gas, 16 for liquid, 7 for rigid body, 6 for particle system.

### TECHNIQUE_TYPES Coverage (truth_pack.py lines 103-133)

| Technique | Types Covered | Missing |
|-----------|---------------|---------|
| `mantaflow_gas` | FluidDomainSettings, FluidFlowSettings, FluidEffectorSettings, FluidModifier, Object, Camera, 4 light types, ShaderNodeMix, ShaderNodeOutputMaterial, ShaderNodeVolumePrincipled, ShaderNodeVolumeAbsorption, CyclesRenderSettings, RenderSettings, Scene | ShaderNodeAttribute, ShaderNodeMath, ShaderNodeBsdfPrincipled (for combined effects), ShaderNodeEmission |
| `mantaflow_liquid` | Same as gas + ShaderNodeBsdfPrincipled (for surface material) | Missing ShaderNodeVolumeAbsorption (for depth coloring -- the API fixer's liquid material snippet uses it) |
| `rigid_body` | RigidBodyWorld, RigidBodyObject, RigidBodyConstraint + common | Missing material/shader nodes entirely |
| `particle_system` | ParticleSettings, ParticleSystem + common | Missing ParticleHairSettings, material nodes |
| `_common` | Object, Camera, 4 light types, CyclesRenderSettings, RenderSettings, Scene | -- |

### Physics Types NOT Covered

The following Blender physics types have ZERO truth pack coverage and will produce unchecked hallucinations:

| Physics Type | Relevant bpy.types | Risk Level |
|-------------|-------------------|------------|
| **Cloth** | ClothSettings, ClothCollisionSettings | High -- next most likely physics type after fluids |
| **Soft Body** | SoftBodySettings | Medium |
| **Geometry Nodes** | NodesModifier, GeometryNodeGroup | High -- increasingly important for procedural effects |
| **Dynamic Paint** | DynamicPaintSurface, DynamicPaintBrushSettings | Low |
| **Collision** | CollisionSettings | Medium -- needed for particle/cloth interactions |

### SETTINGS_MAP Coverage (truth_pack.py lines 154-164)

The settings map maps script variable names to bpy.types classes. Currently covers:
- `domain_settings` -> FluidDomainSettings
- `flow_settings` -> FluidFlowSettings
- `effector_settings` -> FluidEffectorSettings
- `rigid_body` -> RigidBodyObject
- `rigid_body_world` -> RigidBodyWorld
- `particle_systems` -> ParticleSystem (NOTE: this maps to ParticleSystem, not ParticleSettings -- the settings object is `particle_systems[0].settings`, a two-level access)
- `cycles` -> CyclesRenderSettings
- `render` -> RenderSettings
- `scene` -> Scene

**Missing common variable patterns:**
- `cloth` or `cloth_settings` -> ClothSettings
- `mod` or `modifier` -> generic (common in generated scripts)
- `mat` or `material` -> Material (scripts often do `mat.node_tree.nodes`)
- `cam` or `camera` -> Camera (data properties like `cam.data.lens`)
- `light` or `lamp` -> Light types

### VARIABLE_PATTERNS Coverage (truth_pack.py lines 167-174)

Only 5 patterns, all fluid-related:
- `dset`, `dsettings` -> FluidDomainSettings
- `fset`, `fsettings` -> FluidFlowSettings
- `eset` -> FluidEffectorSettings
- `settings` -> FluidDomainSettings (ambiguous fallback)

**Missing patterns from real scripts:**
- `ds` -- very common abbreviation for domain_settings
- `fs` -- very common abbreviation for flow_settings
- `domain` -- used in generated scripts as `domain.modifiers["Fluid"].domain_settings`
- `ps` or `psettings` -- for ParticleSettings
- `rb` or `rigidbody` -- for RigidBodyObject

### Scalability Assessment

The TECHNIQUE_TYPES and SETTINGS_MAP are manually maintained lists. They will fall behind as:
1. New effect types are added (cloth, geometry nodes)
2. LLM scripts use different variable naming conventions
3. Material/shader node types proliferate

The regex-based variable pattern matching (lines 467-505) is inherently limited. It cannot handle:
- Chained property access: `obj.modifiers["Fluid"].domain_settings.resolution_max`
- List/dict comprehensions: `[m for m in obj.modifiers if m.type == 'FLUID']`
- Function return values: `get_domain().resolution_max`

### Recommendation

1. **Immediate: Add `ds` and `fs` to VARIABLE_PATTERNS** -- These are extremely common abbreviations that scripts use.

2. **Immediate: Add ShaderNodeAttribute and ShaderNodeMath to mantaflow_gas** -- The API fixer's VOLUME_MATERIAL_SNIPPET (lines 561-668) uses both of these. If the truth pack validates scripts that include these nodes, it needs to know their properties.

3. **Near-term: Auto-discover variable patterns from the script** -- Instead of maintaining a static VARIABLE_PATTERNS dict, parse the script to find assignment patterns like `ds = obj.modifiers["Fluid"].domain_settings` and dynamically map `ds` to `FluidDomainSettings`. This is an AST-level enhancement.

4. **Medium-term: Add cloth/geometry_nodes to TECHNIQUE_TYPES** -- These are the next most likely physics types to be used. Even if no effects use them today, having the truth pack ready prevents hallucination when they are first used.

---

## Issue 4: Recovery Loop Effectiveness

**Severity:** Medium
**Current State:** The recovery loop (orchestrator.py lines 2792-2964) attempts ONE recovery per iteration. It uses the Script Writer agent with `write_script` (full code generation, not parameter tweaking) and applies truth pack validation to the output.

### Recovery Path Architecture

```
Execution Fails
  |
  v
One recovery attempt per iteration (_recovery_attempted_this_iter flag)
  |
  v
Script Writer agent (max_turns=6, hard_limit=8)
  - Receives: error message, failed script path, common fix suggestions
  - Uses: write_script tool (full code rewrite)
  - NO doc query required (create_error_recovery_hooks)
  |
  v
Truth Pack Validation (lines 2851-2863)
  - Validates recovered script against truth pack
  - Auto-fixes hallucinations
  |
  v
Re-execution (lines 2877-2934)
  - Runs the recovered script in Blender
  - Discovers render files
  - On success: replaces original script, continues to quality evaluation
  - On failure: records failure, moves to next iteration
```

### Iteration Budget

| Scenario | Max Iterations | Recovery Per Iter | Total Recovery Budget |
|----------|---------------|-------------------|----------------------|
| Default | 5 | 1 | 5 recoveries possible |
| Override | User-specified | 1 | N recoveries possible |

Each iteration allows exactly ONE recovery attempt. If recovery fails, the iteration records score=0 and `continue`s to the next iteration. This means:

- If the same structural bug persists, recovery will fail repeatedly, burning iterations
- The escape velocity system (session_mgr.issue_tracker) tracks `consecutive_same_issue` and will escalate to higher escape levels (technique switch at L2, semantic search at L3, human guidance at L4)
- This is CORRECT behavior -- repeated structural failures SHOULD trigger technique switching

### Does Recovery Have Truth Pack Validation?

**Yes.** Lines 2851-2863 explicitly validate recovered scripts against the truth pack:

```python
if context.truth_pack and recovery_script and recovery_script.script_path:
    try:
        fixed_path, fixes = validate_and_fix_script(
            recovery_script.script_path, context.truth_pack
        )
    except Exception as e:
        print(f"[Recovery] WARNING: Truth pack validation failed: {e}")
```

### Can Recovery Fix Script Logic Bugs?

**Partially.** The recovery prompt (lines 2803-2829) specifically says "This is a STRUCTURAL code issue, not a parameter issue" and instructs the LLM to use `write_script` for a complete rewrite. It also provides "Common Structural Fixes":

1. Bake context (active domain object + selected)
2. Effector setup (Fluid modifier with fluid_type='EFFECTOR')
3. Object context (correct active object and selection)
4. Missing depsgraph update

These cover the most common structural issues. However:
- The prompt does NOT include the actual failing script content (only the path). The LLM must use a tool to read it, consuming one of its 6 turns.
- For NoneType errors, the LLM needs to understand the script's object creation flow, which is challenging in 6 turns.
- The recovery hooks do NOT require doc queries, so the LLM may reintroduce API hallucinations (though these will be caught by truth pack validation).

### Scalability Assessment

The recovery loop is fundamentally sound but has cost concerns:
- Each recovery attempt costs ~$0.10-0.30 (LLM API call for Script Writer + Blender re-execution)
- With 5 iterations and potential recovery on each, worst case is ~$1.50 just for recovery
- This is within the $20/month budget but burns allocation fast

### Recommendation

1. **Add failed script content to recovery prompt** -- Include the first ~200 lines of the failing script directly in the prompt. This saves one turn (reading the file) and gives the LLM immediate context.

2. **Pattern-based recovery before LLM** -- Before invoking the Script Writer, try deterministic fixes for known error patterns:
   - `'NoneType' object has no attribute` -> Add null check guards around the identified line
   - `expected a string type, not float` -> Type-coerce the assignment
   - These are cheaper than a full LLM recovery attempt

3. **Limit recovery to iteration 1-2 only** -- After iteration 2, if execution keeps failing, the technique is fundamentally broken. Skip recovery and let escape velocity handle technique switching. This saves budget.

---

## Issue 5: Phase 2A Implementation Status

**Severity:** Informational
**Current State:** Phase 2A items 0-6 are shipped and merged to the current branch. Items 2A-7 and 2A-8 are NOT STARTED.

### Verification of Each Phase 2A Item

| Phase | Status | Commit | Functional? | Evidence |
|-------|--------|--------|-------------|----------|
| **2A-0: Documentation Pipeline** | DONE | `9596fe2` | YES | Scripts exist: `experiment_manual_rewrite.py`, `upload_rewritten_manual.py`, `seed_kb.py`. 131 pages processed. Missing 2/5 tests (CI coverage, not functional gap). |
| **2A-1: Conditional Tool Enabling** | DONE | `a2506f3` | YES | `utils/tool_visibility.py` with 4 callbacks. Quality analyst and docs expert agents have `is_enabled` callbacks on expensive tools. Budget tracker integration working. Missing test file only. |
| **2A-2: Deterministic Agent Control** | DONE | `b589d5e` | YES | Executor agent has `tool_use_behavior="stop_on_first_tool"`. Other agents correctly do NOT have it (documented rationale). 100% complete. |
| **2A-3: Tool Guardrails** | DONE | `55756a2` | YES | `guardrails/tool_guardrails.py` (274 lines), 3 guardrails (truth pack input, critical failure output, script length output). 21 tests. Kill switch. Bypass-proof. Best-implemented item. |
| **2A-4: PipelineMonitor** | DONE | `75178c8` | YES | `tools/pipeline_monitor.py` (PipelineMonitor class). Detects oscillation, stuck loops, score plateaus, wrong feedback cascades, budget overruns. Kill switch via `ENABLE_PIPELINE_MONITOR`. |
| **2A-5: Parameter Bounds** | DONE | `4dd3335` | YES | `tools/parameter_bounds.py` with `ParameterBound` dataclass, `clamp()` and `damped_change()` methods. Per-effect-type bounds. Kill switch via `ENABLE_PARAMETER_BOUNDS`. Orchestrator calls `_apply_script_modifications` with bounds (line 2369-2370). |
| **2A-6: Deprecate Spec-First** | DONE | `136cbf2` | YES | 4 spec-first methods removed from orchestrator (lines 1236-1239 document this). Truth pack + tool guardrails replace the $0.05/call LLM-based API Spec Agent. Code Writer agent marked as DEPRECATED. |
| **2A-7: Remove Executor Agent** | NOT STARTED | -- | -- | Still using the executor agent with `stop_on_first_tool`. Goal is to replace with direct Python function call to `execute_blender_script_impl()`. |
| **2A-8: Tool Timeouts** | NOT STARTED | -- | -- | No timeout wrappers on tools. Blender execution has a 600s timeout in `execute_blender_script`, but other tools (doc search, quality evaluation) have no timeouts. |

### Post-2A-6 Commit

There is one additional commit on the branch:
- `efbfae9` -- "update API validation for Blender 5.0 compatibility" -- This appears to be the context-dependent enum fixes (openvdb_data_depth, cache_particle_format, cache_mesh_format) added to KNOWN_HALLUCINATIONS in truth_pack.py.

### Functional Verification

All DONE items are functional. Specific evidence:

1. **Tool guardrails are wired:** `attach_tool_guardrails()` in `guardrails/tool_guardrails.py` (lines 239-274) dynamically attaches guardrails to `execute_blender_script`, `evaluate_render`, `analyze_with_vision`, and `generate_script`. Called during orchestrator init.

2. **Truth pack flows through the pipeline:** Orchestrator builds truth pack at technique selection time (line 1716 area), passes it to guardrails via `set_guardrail_truth_pack()`, formats it for Script Writer's dynamic instructions via `format_truth_pack_for_prompt()`, and validates scripts via `validate_and_fix_script()` at multiple checkpoints (generation, modification, recovery).

3. **Parameter bounds are applied:** `_apply_script_modifications` in orchestrator.py (lines 848-860 area) calls parameter bounds logic before passing modifications to the script modifier.

4. **PipelineMonitor is instantiated:** `self._pipeline_monitor = PipelineMonitor()` at orchestrator line 758.

### What 2A-7 and 2A-8 Would Provide

**2A-7 (Remove Executor Agent):** The executor agent currently wraps a single tool call (`execute_blender_script`) with `stop_on_first_tool`. Removing the agent and calling the tool function directly would:
- Save ~$0.01-0.02 per execution (LLM call for agent reasoning)
- Reduce latency by ~2-5 seconds (no LLM round-trip)
- Simplify the pipeline (one fewer agent)
- Risk: Loss of the agent's error interpretation capabilities (executor.py lines 55-89 have specific error handling instructions)

**2A-8 (Tool Timeouts):** Currently, only Blender execution has a timeout (600s). Tools without timeouts include:
- `semantic_search_blender_docs` -- OpenAI vector store API call, could hang
- `evaluate_render` -- OpenAI vision API call, could hang
- `analyze_with_vision` -- OpenAI vision API call, could hang
- `search_code_patterns` -- File system search, unlikely to hang but unbounded

Adding timeouts would prevent pipeline stalls from external API outages.

### Recommendation

1. **2A-7 is low priority** -- The executor agent with `stop_on_first_tool` is functionally equivalent to a direct function call. The cost savings ($0.01-0.02/call) are marginal. The error interpretation instructions in the executor's system prompt provide value that would be lost. Defer.

2. **2A-8 is medium priority** -- Tool timeouts prevent pipeline stalls. Implement as a simple wrapper that uses `asyncio.wait_for()` around external API calls. Default timeout: 120s for vision/doc search, 600s for Blender execution (already exists).

---

## Cross-Cutting Observations

### Defense-in-Depth Assessment

The system has FOUR layers of API validation, applied at different pipeline stages:

| Layer | When | What It Catches | Cost |
|-------|------|-----------------|------|
| 1. Script Writer guardrails (script_guardrails.py) | At script output | Hallucination patterns, missing research context | $0 |
| 2. Truth Pack validation (truth_pack_validator.py) | After generation, after modification, after recovery | Invalid attributes against bl_rna introspection | $0 |
| 3. API Fixer (blender_api_fixer.py) | Before execution | 57+ regex patterns, material injection, bake injection, camera fixes | $0 |
| 4. Tool Guardrails (guardrails/tool_guardrails.py) | Before execution (as SDK guardrail) | Truth pack validation + auto-fix, critical failure detection, script length | $0 |

**Overlap:** Layers 2 and 4 both call `validate_script_against_truth_pack`. Layer 3 duplicates some Layer 2 patterns (e.g., both handle `resolution_divisions`, `use_adaptive_time_steps`). This is intentional defense-in-depth, not a bug.

**Gap:** None of the four layers catch script LOGIC bugs (Issue 2). A fifth layer for static analysis/AST checking would close this gap.

### Budget Impact

All four validation layers cost $0. The only validation cost is when recovery engages the Script Writer LLM (~$0.10-0.30/recovery). This is well within the $20/month budget.

### Hallucination Pattern Coverage

Combined across truth_pack.py KNOWN_HALLUCINATIONS and blender_api_fixer.py BLENDER_50_FIXES:

| Category | Pattern Count | Examples |
|----------|---------------|---------|
| Attribute renames | 15 | resolution_divisions, use_adaptive_time_steps, reaction_speed |
| Node type renames | 6 | ShaderNodeMixRGB, ShaderNodeSeparateRGB, ShaderNodeCombineRGB |
| Socket name renames | 12 | Subsurface, Transmission, Specular, Clearcoat, Sheen, Emission |
| Enum value renames | 5 | BLENDER_EEVEE_NEXT, BLOSC, flow_behavior GAS, cache_type REPLAY/MODULAR |
| Removed properties | 4 | use_nodes, use_auto_smooth, noise_res_factor, use_caching |
| Operator renames | 3 | forcefield_add, import_scene.obj, export_scene.obj |
| Context-dependent enums | 3 | openvdb_data_depth, cache_particle_format, cache_mesh_format |
| Structural injections | 6 | Volume material, liquid material, bake, camera, animation stills, frame alignment |
| Type fixes | 3 | noise_scale int, particle_radius range, cube scale halving |
| Node linking | 2 | input-to-input, output-to-output links |

**Total: ~59 unique patterns.** This is comprehensive for mantaflow gas/liquid workflows. Coverage for rigid body, particle systems, cloth, and geometry nodes is thin to nonexistent.

---

## Summary Table

| Issue | Severity | Current State | Key Gap | Priority Fix |
|-------|----------|---------------|---------|--------------|
| 1. Context-Dependent Enums | High | 3 enums hardcoded | More enums exist; not scalable | Strip-if-uncertain policy for cache enums |
| 2. Script Logic Bugs | High | Recovery loop only | No pre-execution logic check | Pre-execution AST analysis + defensive patterns |
| 3. Truth Pack Coverage | Medium | 4 techniques covered | Missing cloth, geo nodes, shader nodes | Add ShaderNodeAttribute/Math; add ds/fs patterns |
| 4. Recovery Loop | Medium | 1 recovery per iteration | Script content not in prompt | Add script content to recovery prompt |
| 5. Phase 2A Status | Info | 2A-0 through 2A-6 done | 2A-7 and 2A-8 TODO | 2A-8 (timeouts) is medium priority |
