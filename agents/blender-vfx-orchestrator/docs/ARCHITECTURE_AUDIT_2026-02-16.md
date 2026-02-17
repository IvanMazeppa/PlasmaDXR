# Blender VFX Orchestrator: Full Architecture Audit & Path Forward

**Date:** 2026-02-16
**Author:** Claude Opus 4.6 (audit requested by Ben)
**Purpose:** Top-to-bottom honest assessment of the orchestrator's current state, what works, what doesn't, and detailed options for the path forward toward autonomy, self-learning, and self-improvement.

---

## Part 1: Current State — The Raw Numbers

### Session History (188 Total Sessions)

| Status | Count | Percentage |
|--------|-------|------------|
| max_iterations (hit cap, never passed) | 102 | 54% |
| failed (crashed before completing) | 69 | 37% |
| in_progress (interrupted/abandoned) | 14 | 7% |
| passed | 3 | 2% |

**Sessions with score > 0:** 37 out of 188 (20%)
**Sessions meeting 60-point threshold:** 1 (kitchen_leak at exactly 60.0 — but recorded as max_iterations, not passed, because it was a single-iteration fluke)
**All 3 "passed" sessions** used a reduced threshold of 40, not 60. They were smoke effect tests from 2026-01-25.

**The system has never formally passed a session at the standard 60-point quality threshold.**

### Benchmark Scores

| Scenario | Type | Sessions Run | Best Score | Root Cause of Failure |
|----------|------|-------------|------------|----------------------|
| kitchen_leak | Liquid | 16 | **58** | Laminar flow, no spray |
| wine_pour | Liquid | 4 | **0** | Camera inside glass, no collision effectors |
| candle_fire | Gas | 7 | **18** | Camera too close, exposure oscillation |
| liquid_spray | Liquid | 1 | **60** | Single-iteration fluke, never reproduced |
| oil_pour | Liquid | ~5 | **19** | Similar issues to wine_pour |
| explosion | Gas | ~8 | **32** | Best gas result |
| smoke (various) | Gas | ~20 | **45** | Only effect type that "passes" at reduced threshold |

### Codebase Complexity

| File | Lines | Key Metric |
|------|-------|------------|
| `orchestrator.py` | 4,429 | `create_asset_pipeline()` = 2,208 lines, 165 branches |
| `tools/blender_api_fixer.py` | 1,962 | 57 regex fixes, 15 sequential transform steps |
| `tools/dynamic_instructions.py` | 1,082 | Returns empty fallback for every session |
| `tools/semantic_docs_tools.py` | 1,127 | Vector store search (works) |
| `tools/knowledge_distillation_tools.py` | 814 | Pattern extraction (4 patterns stored) |
| `tools/proactive_research_tools.py` | 676 | Escape velocity (string-count heuristics) |
| `tools/code_pattern_tools.py` | 492 | Pattern memory (4 untagged patterns) |
| `tools/script_generator_tools.py` | 1,584 | MCP wrapper for script generation |
| `specialized_agents/api_validator.py` | 907 | 30+ known-bad attribute entries |
| `specialized_agents/code_writer_agent.py` | 445 | Spec-first code generation |
| `models/shared_context.py` | 918 | Data models |
| `hooks/enforcement_hooks.py` | 894 | Loop detection, doc query enforcement |
| `utils/quality_parameter_map.py` | 302 | Deterministic fallback (24 mappings) |
| **Total project code** | **~42,000** | Excluding venv, ~100 Python files |

### Error Handling

- **95 `except Exception` blocks** in `orchestrator.py` alone
- **148 total `except` clauses** across non-test, non-venv files
- **30 `except Exception as e`** in `create_asset_pipeline` alone
- **2 silent `pass` catches** (server.py line 83, asset_evaluator_tools.py line 449)
- **Pattern:** Most exceptions are caught, logged to stderr, and continued past. The system rarely crashes but rarely succeeds. Failures are hidden, not solved.

### Test Coverage

- **37 test files total** (35 at root, 2 in `tests/`)
- **3 files with actual `assert` statements** (test_artifact_manager.py, test_artifact_gates.py, test_parallel_phases.py)
- **34 files are integration scripts** — they call the orchestrator, print output, and rely on "no crash = pass"
- **0 unit tests** for the API fixer's 15-step chain
- **0 unit tests** for individual pipeline phases
- **0 unit tests** for self-learning strategies

---

## Part 2: Component-by-Component Analysis

### 2.1 The Orchestrator (`orchestrator.py` — 4,429 lines)

**The monolith.** `create_asset_pipeline()` spans lines 1998-4206 (2,208 lines in a single async function). It contains:

**Pipeline phases:**
| Phase | Label | What It Does |
|-------|-------|-------------|
| 0 | Research | One-time research at start |
| 0.5 | Technique Selection + API Spec | Parallel coordinator + spec gathering |
| 0.9 | Pre-Iteration Research | Escape velocity check (iter > 1) |
| 0.95 | Self-Learning Reuse | Pattern search and application |
| 1 | Script Generation | Spec-first or legacy code writer |
| 1.0.1 | Pattern Application | Self-learning code injection |
| 1.1 | Modification Strategy | Coordinator decides how to modify |
| 1.5 | API Validation | Blender 5.0 attribute check |
| 2 | Execution | Deterministic Blender run |
| 2.5 | Diagnose | Analyze execution failure |
| 2.6 | Fix | Apply fix for execution failure |
| 2.7 | Error Recovery | AI-driven script rewrite |
| 2.7 | Artifact Gates | Deterministic validation (**NAMING COLLISION**) |
| 2.8 | Diagnose | Analyze artifact gate failure |
| 2.9 | Fix | Apply fix for artifact gate failure |
| 3 | Quality Evaluation | LLM-as-Judge vision scoring |
| 4+5 | Learning + Quality Gate | Parallel or sequential |
| 4.5 | Pattern Extraction | Extract patterns from success |
| 4.6 | Pattern Outcome Reporting | Report pattern effectiveness |
| 5 | Quality Gate | Sequential fallback |

**Problems:**
- Phase 2.7 appears twice with different meanings
- 165 if/elif/else branches make control flow nearly impossible to trace
- 95 exception catches mean failures cascade silently
- No way to test a single phase in isolation
- Adding any fix means touching this 2,208-line function

### 2.2 The API Fixer (`tools/blender_api_fixer.py` — 1,962 lines)

**The most valuable component.** 57 regex-to-replacement entries in `BLENDER_50_FIXES` that catch known LLM hallucinations and replace them with correct Blender 5.0 API calls.

**`validate_and_fix_script()` execution order (15 steps):**
1. Tab → space replacement
2. Apply all `BLENDER_50_FIXES` (regex batch — 57 entries)
3. `_fix_mix_node_sockets` (ShaderNodeMix socket names)
4. `_validate_and_fix_api_calls` (vector store validation)
5. Camera injection (if render present but no camera)
6. `_inject_volume_material_setup` (gas effects)
7. `_inject_liquid_material_setup` (liquid effects)
8. `_inject_bake_before_render` (fluid simulation bake)
9. `_inject_bake_frame_alignment` (cache frame range)
10. `_inject_plane_init_for_liquid_flow` (liquid inflow)
11. `_fix_particle_radius_value` (reset nonsensical values)
12. `_fix_cube_scale_halving` (LLM halves all dimensions)
13. `_inject_camera_distance_fix` (camera inside geometry / too close)
14. `_inject_animation_to_stills` (headless render fix)
15. Syntax validation loop (up to 10 passes)

**Problems:**
- Steps are ordered but can conflict (step 13 could theoretically interact with step 5)
- No idempotency — running twice may produce different output
- No unit tests for individual injection functions
- The fixer is playing whack-a-mole: every new LLM model or Blender update produces new hallucinations

**Fix categories:**
- Shader node renames (ShaderNodeMixRGB→Mix, SeparateRGB→SeparateColor, etc.)
- Principled BSDF input renames (Subsurface, Transmission, Specular, Clearcoat)
- Fluid domain attribute renames (resolution_divisions, use_adaptive_time_steps, etc.)
- Render loop limiter (6 patterns)
- Cache type enforcement
- Path fixes for headless mode
- Object selection API changes
- bmesh parameter renames
- Node link direction validation
- Visibility/density boosts

### 2.3 Self-Learning System (5 Strategies — ~3,700 lines total)

#### Strategy 1: Vector Store Search (1,127 lines)
- **What:** Semantic search over two OpenAI vector stores (Blender manual + API reference)
- **Status:** Implemented and connected. Uses hardcoded vector store IDs.
- **Effectiveness:** Unknown — no tracing captures which queries were useful. The stores exist but whether agents invoke them productively is untraceable.

#### Strategy 2: Knowledge Distillation (814 lines)
- **What:** Diffs two scripts to find what changed, generalizes code, stores as reusable pattern
- **Status:** Connected at Phase 0.95. **4 patterns stored.**
- **Problems:**
  - All 4 patterns have `effect_type=None` — not tagged by type
  - Pattern outcome reporting crashes: `'FunctionTool' object is not callable`
  - No pattern has ever been applied, updated, and confirmed improved on a second run
  - Patterns are unretrievable by effect type search because they're untagged

#### Strategy 3: Proactive Mining / Escape Velocity (676 lines)
- **What:** Checks iteration history for repeated issues, escalates through 5 escape levels
- **Status:** Connected at Phase 0.9 (iteration > 1 only)
- **Implementation:** Simple substring counting in a text history string
- **Effectiveness:** No confirmed instance of a technique switch producing improvement within a single session

#### Strategy 4: Code Pattern Memory (492 lines)
- **What:** Stores/retrieves working Python code snippets as JSON files
- **Status:** Connected at Phase 0.95. Same 4 patterns as Strategy 2 (shared storage).
- **Problems:** Same as Strategy 2 — all untagged, unretrievable by effect type

#### Strategy 5: Dynamic Instructions (1,082 lines)
- **What:** Generates runtime instructions for agents by querying knowledge base for validated learnings
- **Status:** Connected as `Agent(instructions=fn)` for script writer, quality analyst, learning agent
- **Critical problem:** Queries `parameter_knowledge` table for entries with `success_rate >= 0.7`. Actual rates:

| Parameter | success_rate |
|-----------|-------------|
| script | 0.30 |
| domain | 0.40 |
| flow | 0.375 |
| resolution_max | 0.143 |
| noise_scale | **0.00** |
| vorticity | **0.00** |
| noise_strength | **0.00** |

**No parameter crosses the 0.7 threshold.** The function returns `"No validated rules yet for this effect type. Use Blender defaults."` for every single session. 1,082 lines of code to generate an empty string.

#### Self-Learning Summary

| Strategy | Lines | Data Populated | Actually Useful |
|----------|-------|----------------|-----------------|
| Vector Store | 1,127 | External stores exist | Unknown |
| Knowledge Distillation | 814 | 4 patterns (untagged) | No |
| Escape Velocity | 676 | Heuristic only | No confirmed success |
| Code Pattern Memory | 492 | 4 patterns (same as #2) | No |
| Dynamic Instructions | 1,082 | 201 entries, all below threshold | **No — returns empty** |
| **Total** | **4,191** | | **~0% measured impact** |

### 2.4 Deterministic Fallback (`utils/quality_parameter_map.py` — 302 lines)

**The most reliably effective mechanism in the entire system.**

24 keyword→parameter mappings. Zero LLM cost. Always fires when both pattern search and LLM fail.

Examples:
- "overexposed" → `{emission_strength: 0.3, density: 2.0}`
- "too dark" → `{emission_strength: 5.0, density: 0.5, light_energy: 500.0}`
- "domain boundary" → `{resolution_max: 128, domain_size: 4.0}`

**Problems:**
- Mappings are effect-type-agnostic (same params for fire and water)
- No per-effect-type clamping (root cause of candle light overcorrection)
- Only 24 mappings — limited vocabulary

### 2.5 Specialized Agents (8 files, 2,783 lines total)

| Agent | Lines | Status |
|-------|-------|--------|
| API Validator | 907 | Works — static dict of 30+ known-bad attrs |
| Code Writer | 445 | Works with spec-first pipeline |
| API Spec Agent | 295 | Works — gathers verified attrs |
| Docs Expert | 273 | Works — searches Blender docs |
| Learning Agent | 244 | Connected but data-starved |
| Quality Analyst | 194 | Works when budget allows |
| Script Writer (legacy) | 200 | Deprecated, still reachable |
| Executor | 169 | Works |

Plus 3 coordinator agents defined inline in `orchestrator.py`: TechniqueSelector, ModificationStrategist, QualityGateJudge.

### 2.6 Budget Tracking

- Singleton `BudgetTracker` persisted to `budget_tracker.json`
- **File doesn't exist** — spend resets between runs
- Monthly limit: $20 ($10 vision, $6 docs, $2 reasoning, $2 buffer)
- Effective within a single run but no cross-run persistence

### 2.7 Spec-First vs Legacy Pipeline

**Spec-First (active, default):**
1. API Spec Agent gathers verified Blender 5.0 attributes
2. Code Writer uses only those verified APIs
3. Guardrail validates code against spec
4. If API Spec Agent fails → **fail closed** (no fallback)

**Legacy (still reachable):**
- Single script writer agent, no API validation
- Reachable via exception at outer pipeline level (line 2448)
- Contradicts the "fail-closed" principle

---

## Part 3: Root Cause Analysis

### Why 0 passes at threshold=60 after 188 sessions?

**Primary causes (in order of impact):**

1. **Camera placement** — The LLM generates camera positions as part of free-form script generation. In 2/3 benchmarks, the camera is inside the geometry or so close the subject is invisible. No amount of post-generation fixing can produce *good* camera angles — only prevent catastrophic ones.

2. **Free-form code generation** — The LLM writes 700+ lines of Python every time, from scratch. Even with the spec-first pipeline constraining API usage, the LLM still controls scene geometry, lighting, materials, camera, animation, and render settings. Every generation is a fresh opportunity for novel failure modes.

3. **Error hiding** — 95 exception catches mean the system continues past failures that should be fatal. A physics simulation with no collision effectors doesn't crash — it just produces an empty render. The quality evaluator sees a black image and scores it 0, but the system doesn't know *why* it's black.

4. **Accretive complexity** — Each fix adds lines without removing any. The fixer chain grows, the pipeline grows, the exception catches grow. Complexity breeds new failure modes faster than fixes eliminate old ones.

5. **Dead self-learning** — The system was designed to get smarter over time, but the learning infrastructure is data-starved. After 188 sessions, it has 4 untagged patterns and zero knowledge base entries above the utility threshold. The system is not learning.

### The Fundamental Architectural Tension

The system tries to fix LLM output **after** generation. This is an infinite game:
- LLM generates `ShaderNodeMixRGB` → fixer renames to `ShaderNodeMix`
- LLM generates camera at (0.1, -0.08, 0.31) → fixer moves it back
- LLM generates `use_nodes = True` → fixer warns but Blender still errors
- LLM generates nonsensical `particle_radius = 0.005` → fixer resets to 1.0

Every new LLM model, every new Blender version, every new effect type produces new failure modes. The fixer can never be complete because the space of possible LLM hallucinations is unbounded.

**The solution is to constrain the generation space**, not to fix unbounded output.

---

## Part 4: Path Forward — Three Options in Detail

### Option A: Keep Patching (NOT Recommended)

**What:** Continue S2 priority queue:
1. Camera distance fixer (done, needs retest)
2. Light energy bounds per effect type
3. Collision effector auto-injection for liquids
4. `Action.fcurves` API fixer
5. Stale render cleanup

**Effort:** 1-3 days per fix, 2-3 weeks total
**Expected outcome:** kitchen_leak → 70+, candle → 40+, wine_pour → 20+
**Risk:** Low (incremental changes)

**Why not recommended:**
- We've been doing this for a month with 0 passes at threshold=60
- Each fix adds complexity without reducing it elsewhere
- The capability ceiling is set by the free-form generation architecture
- Diminishing returns: each new fix has less impact than the last

### Option B: Targeted Refactor (Moderate)

**What:**
1. **Break up the monolith** — Extract `create_asset_pipeline` into a state machine
   - Each phase becomes a method: `_phase_research()`, `_phase_generate()`, etc.
   - Explicit state transitions with typed `PhaseResult` objects
   - Unit testable phases
2. **Fix the test suite** — Add assertions to integration tests, unit tests for fixer chain
3. **Revive self-learning** — Tag patterns by effect type, lower thresholds to 0.5, manually seed knowledge base
4. **Kill dead code** — Remove legacy script writer path, collapse duplicate phase numbers
5. **Harden error handling** — Convert silent catches to explicit failure modes with retry logic
6. **Hard-fix `use_nodes`** — Replace in fixer instead of warning

**Effort:** 3-5 days of focused work
**Expected outcome:** Cleaner codebase, testable pipeline, self-learning starts producing data. Same capability ceiling — still generating free-form code.
**Risk:** Medium (refactoring a monolith can introduce regressions)

**Verdict:** Good hygiene but doesn't change the fundamental architecture. The system will still generate 700 lines of Python and then try to fix them.

### Option C: Architecture Redesign (Recommended)

**What:** Shift from "generate code → fix it" to "fill template → validate parameters"

#### C.1: Template-Based Generation

**Instead of:**
```
LLM writes 700 lines of Python → API fixer applies 15 transforms → Blender runs it → maybe works
```

**Do:**
```
LLM selects template + fills parameter JSON → Template engine fills slots → Blender runs validated code → works
```

**Templates** are pre-validated Blender Python scripts with parameter slots:
```python
# templates/liquid_mantaflow_pour.py
# Validated on Blender 5.0, bakes and renders correctly
import bpy
# ... (200 lines of validated scene setup, materials, lighting) ...

# === PARAMETER SLOTS (filled by LLM) ===
DOMAIN_RESOLUTION = {{resolution_max}}       # int, range: 64-256
DOMAIN_SIZE = {{domain_size}}                 # float, range: 0.5-5.0
FLOW_VELOCITY = {{flow_velocity}}             # float, range: 0.5-3.0
CAMERA_DISTANCE = {{camera_distance}}         # float, range: 0.5-5.0
CAMERA_ANGLE = {{camera_angle}}               # str, enum: ["front", "3/4", "top", "side"]
KEY_LIGHT_ENERGY = {{key_light_energy}}        # float, range: 100-1000
LIQUID_COLOR = {{liquid_color}}               # tuple[float, float, float, float], RGBA
RENDER_SAMPLES = {{render_samples}}           # int, range: 64-256
# ... etc
```

**The LLM's job reduces to:**
1. Select which template to use (from ~10 validated options)
2. Fill the parameter JSON with appropriate values
3. Optionally describe scene-specific geometry modifications

**What this eliminates:**
- Deprecated attribute hallucinations (templates are pre-validated)
- Camera placement failures (camera is a parameter with valid ranges)
- Material setup errors (materials are pre-built in templates)
- Bake configuration errors (bake setup is pre-validated)
- 80% of the API fixer's workload

**What this preserves:**
- The LLM still makes creative decisions (parameters, geometry)
- The quality evaluation loop still drives improvement
- The API fixer still runs (but with much less to fix)

**Template sources:** The kitchen_leak script that scored 58, the smoke scripts that scored 40-45, manually curated examples from Blender documentation.

#### C.2: Modular Pipeline

Replace the 2,208-line function with a state machine:

```python
class PipelinePhase(Enum):
    RESEARCH = "research"
    TECHNIQUE_SELECTION = "technique_selection"
    GENERATION = "generation"
    VALIDATION = "validation"
    EXECUTION = "execution"
    EVALUATION = "evaluation"
    DECISION = "decision"

class PhaseResult:
    next_phase: PipelinePhase
    data: dict
    error: Optional[str]

class Pipeline:
    async def run_phase(self, phase: PipelinePhase, context: PipelineContext) -> PhaseResult:
        handler = self._handlers[phase]
        return await handler(context)
```

Each phase is a separate method (or even a separate file), with typed input/output, independently testable.

#### C.3: Deterministic Camera System

Camera placement as a pure function:

```python
def compute_camera_position(
    scene_bounds: BoundingBox,
    focal_length: float,
    effect_type: str,
    preferred_angle: str = "3/4"
) -> CameraPlacement:
    """Returns position, rotation, focal length. No LLM involved."""
    diagonal = scene_bounds.diagonal()
    fov = 2 * atan(sensor_width / (2 * focal_length))
    min_distance = diagonal / (2 * tan(fov / 2)) * 1.2
    # Apply effect-type presets (fire: slightly above, liquid: slightly below, etc.)
    # Apply angle preset (front, 3/4, top, side)
    return CameraPlacement(location=..., rotation=..., focal_length=focal_length)
```

#### C.4: Curated Knowledge Base

Replace the 5 self-learning strategies with a simple, manually curated parameter knowledge base:

```json
{
  "liquid": {
    "resolution_max": {"default": 96, "range": [64, 256], "notes": "Higher = slower bake"},
    "camera_distance": {"default": 1.5, "range": [0.5, 5.0], "notes": "Minimum 0.5m for any liquid scene"},
    "key_light_energy": {"default": 300, "range": [100, 800], "notes": "Never exceed 1000 for liquid"},
    "particle_radius": {"default": 1.0, "range": [0.5, 2.0], "notes": "LLM generates 0.005, always wrong"}
  },
  "fire": {
    "resolution_max": {"default": 64, "range": [32, 128]},
    "camera_distance": {"default": 2.0, "range": [1.0, 10.0]},
    "key_light_energy": {"default": 100, "range": [50, 500], "notes": "Fire provides its own light"}
  }
}
```

This replaces 4,191 lines of self-learning code with ~100 lines of JSON that actually works.

**Future:** Once the system reliably produces scores > 60, the curated knowledge base can be extended with automated parameter range updates based on successful runs. But the starting point should be curated, not empty.

#### C.5: Simplified Self-Learning (Phase 2)

After the template system works reliably:
1. **Parameter memory:** Record which parameter values produced which scores
2. **Range tightening:** Narrow parameter ranges based on successful runs
3. **Template evolution:** Create new templates from successful custom scripts
4. **No code pattern learning** — the LLM doesn't write code anymore, so there are no code patterns to learn

#### What We Keep

| Component | Lines | Status |
|-----------|-------|--------|
| Blender execution pipeline | ~900 | Works, keep as-is |
| Quality evaluation (vision + metrics) | ~900 | Works, keep as-is |
| Vector store docs search | ~1,100 | Useful for template creation |
| Experiment tracker | ~1,000 | Useful for logging |
| MCP server interface | ~200 | Keep as-is |
| API fixer (simplified) | ~500 | Keep core regex fixes, remove injections handled by templates |
| Budget tracker | ~200 | Keep, fix persistence |

#### What We Replace

| Old | New | Reduction |
|-----|-----|-----------|
| Free-form script generation (700+ lines per run) | Template + parameter filling | LLM output: 700 lines → 20 params |
| 15-step fixer chain (1,962 lines) | Minimal fixer (templates pre-validated) | ~1,500 lines removed |
| 2,208-line pipeline function | Modular state machine | ~1,500 lines simplified |
| 5 self-learning strategies (4,191 lines) | Curated JSON knowledge base | ~4,000 lines removed |
| Dynamic instructions (1,082 lines) | Static per-effect-type instructions | ~900 lines removed |
| LLM-generated camera/lighting | Deterministic functions | Eliminates #1 failure mode |

**Estimated net reduction:** ~8,000 lines removed, ~2,000 lines added = **~6,000 fewer lines**

#### Effort Estimate

| Phase | Work | Duration |
|-------|------|----------|
| Template creation (3 effect types) | Extract from working scripts, validate on Blender 5.0 | 2-3 days |
| Pipeline refactor | State machine, typed phases | 2-3 days |
| Camera/lighting system | Deterministic functions | 1 day |
| Knowledge base | Curated JSON, parameter ranges | 1 day |
| Integration + testing | Wire it all together, run benchmarks | 2-3 days |
| **Total** | | **8-13 days** |

#### Risk Assessment

| Risk | Mitigation |
|------|------------|
| Templates too rigid for creative scenes | Allow "custom geometry" parameter that lets LLM add objects to template |
| Regression on smoke (current best performer) | Keep smoke template identical to current best-scoring script |
| Template doesn't cover new effect types | Template creation process is documented and repeatable |
| Parameter ranges too narrow | Start wide, tighten based on data |

---

## Part 5: Collaboration Plan — Claude + Codex/Gemini

### Why Multi-AI Collaboration?

- **Claude Opus:** Strong at architecture design, code review, complex reasoning about system behavior
- **Codex/Gemini:** Strong at code generation, Blender Python, exhaustive testing, research

### Proposed Division of Labor

#### Phase 1: Research (2-3 days, parallel)

**Codex/Gemini tasks:**
- Research Blender Python VFX automation best practices (2026 state of the art)
- Survey existing template-based Blender automation systems
- Validate Blender 5.0 API for all fluid, fire, smoke, explosion domains
- Create a comprehensive Blender 5.0 API compatibility matrix
- Research camera placement algorithms for VFX scenes

**Claude tasks:**
- Research multi-agent orchestration patterns (OpenAI SDK, LangGraph, CrewAI)
- Design the modular pipeline state machine
- Design the template parameter schema
- Research quality evaluation best practices (LPIPS, CLIP, TOPIQ thresholds)

#### Phase 2: Template Creation (3-5 days, parallel)

**Codex/Gemini tasks:**
- Extract templates from working scripts (kitchen_leak, smoke tests)
- Create 3 initial templates: liquid_pour, gas_fire, gas_smoke
- Validate each template headless on Blender 5.0
- Create template parameter schemas with valid ranges
- Write unit tests for each template

**Claude tasks:**
- Design the template engine (parameter substitution, validation)
- Design the parameter selector agent (LLM picks template + fills params)
- Implement deterministic camera/lighting functions
- Create the curated knowledge base JSON

#### Phase 3: Pipeline Implementation (3-5 days, parallel)

**Codex/Gemini tasks:**
- Implement template rendering engine
- Update API fixer for template-compatible mode
- Create end-to-end test suite with assertions
- Blender integration testing

**Claude tasks:**
- Implement modular pipeline state machine
- Implement parameter selector agent
- Wire quality evaluation into new pipeline
- Implement simplified self-learning (parameter memory)

#### Phase 4: Integration & Benchmarking (2-3 days, joint)

- Run all 3 benchmark scenarios
- Compare to S1.5 baselines (kitchen=58, wine=0, candle=18)
- Target: kitchen >= 70, wine >= 40, candle >= 40
- Iterate on parameter ranges based on results

---

## Part 6: Toward True Autonomy and Self-Improvement

### Current State: Not Autonomous

The system requires human intervention for:
- Template creation (manual extraction from working scripts)
- Knowledge base curation (manual parameter ranges)
- Bug fixes (each failure mode requires code changes)
- Budget monitoring (no persistent tracking)

### Roadmap to Autonomy

#### Level 1: Reliable Execution (Option C above)
- Templates produce consistent scores > 60
- No human intervention needed per session
- **This is the prerequisite for everything else**

#### Level 2: Parameter Self-Optimization
- System records which parameter values produce which scores
- Bayesian optimization narrows parameter ranges automatically
- New parameter combinations are explored within validated ranges
- **Requires:** Reliable execution (Level 1) + persistent experiment tracking

#### Level 3: Template Self-Creation
- System identifies when existing templates don't cover a request
- Generates candidate template from free-form code (like current system)
- Validates candidate template through multiple test runs
- Promotes validated templates to the template library
- **Requires:** Parameter self-optimization (Level 2) + template validation framework

#### Level 4: Full Autonomy
- System handles novel effect types without human intervention
- Discovers new techniques through documentation search
- Creates, validates, and promotes templates automatically
- Maintains and updates its own knowledge base
- **Requires:** Template self-creation (Level 3) + robust quality evaluation

### Key Insight

**The current system tried to start at Level 4** (full autonomy via free-form code generation + self-learning). This is why it fails — it doesn't have the foundation of reliable execution (Level 1) that autonomy requires.

**Option C builds Level 1.** Once we can reliably produce scores > 60 with templates, we have the foundation to build Levels 2-4 incrementally.

---

## Part 7: Known Bugs to Fix Regardless of Direction

These bugs exist independent of architectural choices:

1. **`'FunctionTool' object is not callable`** — Pattern outcome reporting crash in Phase 4.6
2. **`'IterationResult' has no attribute 'primary_issue'`** — Test script model mismatch
3. **`'Action' object has no attribute 'fcurves'`** — Blender 5.0 animation API change (no fixer entry)
4. **`use_nodes` treated as warning, not fixed** — Causes exit_code=1 in Blender 5.0
5. **`sampling_substeps → subframes` correctness unverified** — Applied by fixer but never tested
6. **Camera distance fix regex bug** — Was matching `# Render engine` comments instead of actual render calls (FIXED 2026-02-16)
7. **Budget tracker not persisted** — Resets between runs, no cross-session spend tracking
8. **Legacy writer path still reachable** — Exception at outer pipeline level bypasses fail-closed

---

## Appendix: File Reference

| File | Lines | Category | Status |
|------|-------|----------|--------|
| `orchestrator.py` | 4,429 | Core | Monolith, needs refactor |
| `tools/blender_api_fixer.py` | 1,962 | Core | Most valuable component |
| `tools/dynamic_instructions.py` | 1,082 | Self-Learning | Inert (returns empty) |
| `tools/semantic_docs_tools.py` | 1,127 | Self-Learning | Works (external stores) |
| `tools/knowledge_distillation_tools.py` | 814 | Self-Learning | Data-starved |
| `tools/proactive_research_tools.py` | 676 | Self-Learning | Heuristic, unproven |
| `tools/code_pattern_tools.py` | 492 | Self-Learning | Data-starved |
| `tools/script_generator_tools.py` | 1,584 | Core | MCP wrapper |
| `tools/blender_executor_tools.py` | 926 | Core | Works |
| `tools/asset_evaluator_tools.py` | 921 | Core | Works |
| `tools/experiment_tracker_tools.py` | 1,034 | Self-Learning | Works (sparse data) |
| `specialized_agents/api_validator.py` | 907 | Core | Works |
| `specialized_agents/code_writer_agent.py` | 445 | Core | Works (spec-first) |
| `specialized_agents/api_spec_agent.py` | 295 | Core | Works |
| `specialized_agents/docs_expert.py` | 273 | Core | Works |
| `specialized_agents/learning_agent.py` | 244 | Self-Learning | Data-starved |
| `specialized_agents/quality_analyst.py` | 194 | Core | Works |
| `specialized_agents/script_writer.py` | 200 | Legacy | Deprecated |
| `specialized_agents/executor.py` | 169 | Core | Works |
| `hooks/enforcement_hooks.py` | 894 | Core | Works |
| `utils/quality_parameter_map.py` | 302 | Core | Most reliable mechanism |
| `models/shared_context.py` | 918 | Core | Works |
