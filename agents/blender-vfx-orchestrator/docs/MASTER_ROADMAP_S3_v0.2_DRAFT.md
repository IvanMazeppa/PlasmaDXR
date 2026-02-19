# S3 Master Roadmap v0.2 — DRAFT

**Status:** Draft for cross-review (merged, pending rebuttal round)
**Authors:** Claude Opus 4.6 + Codex (co-authored)
**Date:** 2026-02-17
**Supersedes:** `MASTER_ROADMAP_2026-01-26.md`, `ARCHITECTURE_OPTIMIZATION_PLAN_2026-01-22.md`
**Change log (v0.2):** Adds scope boundaries, governance protocol, reproducibility contract, lane safety controls, refined rollback policy, and open-questions register.

---

## 1. Problem Statement & Evidence Baseline

### Why This Reboot

After 188 pipeline sessions over 3+ weeks:
- **0 sessions** passed at the standard quality threshold (60/100)
- **3 sessions** passed at a reduced threshold (40) — smoke effect only, months ago
- **Best scores:** kitchen_leak=58, candle=18, wine_pour=0
- The system generates 700+ lines of free-form Python per run, then tries to fix it with a 15-step transformation chain
- Self-learning infrastructure (5 strategies, ~4,200 lines) has produced 4 untagged patterns and 0 knowledge base entries above the utility threshold

### Root Causes (Agreed by Both Audits)

1. **Unbounded generation space.** Free-form LLM code generation produces novel failures faster than guard rails can catch them. Every new model version and Blender update introduces new hallucination patterns.

2. **Legacy fallback paths.** Despite "spec-first fail-closed" claims, runtime code at `orchestrator.py:2448`, `:2837`, and `:2995` falls back to the legacy Script Writer, explicitly noting hallucination risk.

3. **Truth corruption.** Execution status logic contradicts itself — comment says "do NOT flip success" (line 3281), code immediately sets `success = True` (line 3293). This contaminates all downstream learning signals.

4. **Monolithic pipeline.** `create_asset_pipeline()` is 2,208 lines with 165 branches and 95 exception catches. Untestable, unmodifiable without regression risk.

5. **Dead self-learning.** 4 patterns (all untagged), 0 knowledge entries above threshold, dynamic instructions returning empty strings for every session. ~4,200 lines of code with no measurable effect.

6. **Camera placement.** The #1 quality killer — camera inside glass (wine_pour=0), camera 13cm from subject (candle=18). Currently an LLM generation problem with only a post-hoc safety net.

### Evidence Links

| Finding | Source | Lines |
|---------|--------|-------|
| Legacy fallback paths | `orchestrator.py` | 2448, 2450, 2452, 2837, 2995 |
| Execution truth contradiction | `orchestrator.py` | 3281-3293 |
| Serialization mismatch | `orchestrator.py` | 169 vs 3825 |
| Version drift (0.8.3 vs 0.9.0) | 6+ docs files | See Codex audit |
| API fixer fail-open | `blender_api_fixer.py` | 1681, 1684 |
| Guardrail blacklist-only | `api_spec_guardrails.py` | 430, 472 |
| Self-learning data state | `data/code_patterns/` | 4 patterns, all `effect_type=None` |
| Dynamic instructions inert | `tools/dynamic_instructions.py` | All params below 0.7 threshold |
| 188 sessions / 0 passes | `build/orchestrator_state/` | 102 max_iter, 69 failed, 14 in_progress, 3 passed@40 |

---

## 2. Architectural Target State

### North-Star Outcomes

By end of S3:
1. No deprecated or hallucinated Blender API use reaches execution.
2. No hidden fallback path can bypass spec-first constraints.
3. The system reliably produces quality scores >= 60 for known effect types.
4. Every change is validated by reproducible eval gates before acceptance.
5. Self-improvement is evidence-based: propose → sandbox test → measure → promote or rollback.
6. The pipeline is modular, testable, and maintainable.

### Scope Boundaries (v1.0)

**In scope for this roadmap version:**
- 3 template-backed benchmark scenarios (`kitchen_leak`, `wine_pour`, `candle_fire`)
- Dual-lane architecture with Lane A as default and Lane B explicitly guarded
- Modular state-machine pipeline replacing monolithic implementation
- Staged eval harness (3-scenario gate first, then 10-scenario expansion)
- Curated knowledge base + Level 0/1 autonomy gating

**Out of scope until v1.1+:**
- Full effect-library coverage across all Blender VFX domains
- Fully autonomous self-editing/self-deployment behavior
- 25-50 scenario matrix as a hard release gate
- CI/nightly automation before Stage 2 is stable

### Architecture Diagram

```
                    ┌─────────────────────────────────────────────┐
                    │             Asset Request                    │
                    └──────────────────┬──────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────────┐
                    │  PHASE 1: SELECT TEMPLATE                   │
                    │  LLM picks template + technique             │
                    │  Lane A: Known effect → template            │
                    │  Lane B: Novel effect → guarded custom gen  │
                    └──────────────────┬──────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────────┐
                    │  PHASE 2: FILL PARAMETERS                   │
                    │  LLM fills param JSON (structured output)   │
                    │  Validated against schema + knowledge base   │
                    │  Camera/lighting: deterministic defaults     │
                    └──────────────────┬──────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────────┐
                    │  PHASE 3: VALIDATE                          │
                    │  API allowlist check (not blacklist)         │
                    │  Parameter range validation                  │
                    │  Static script lint                          │
                    │  Syntax validation                           │
                    └──────────────────┬──────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────────┐
                    │  PHASE 4: EXECUTE                           │
                    │  Blender headless (existing executor)        │
                    │  Dual-channel outcome:                       │
                    │    process_success (exit code truth)         │
                    │    artifact_viable (render produced)         │
                    └──────────────────┬──────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────────┐
                    │  PHASE 5: EVALUATE                          │
                    │  Vision-based quality scoring                │
                    │  Runs only if artifact_viable=True           │
                    │  Budget-gated                                │
                    └──────────────────┬──────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────────┐
                    │  PHASE 6: DECIDE                            │
                    │  Score >= threshold → ACCEPT                 │
                    │  Score < threshold + iterations remain →     │
                    │    adjust parameters → back to PHASE 2      │
                    │  Max iterations → report best result         │
                    └─────────────────────────────────────────────┘
```

### Dual-Lane Generation

| Lane | When | How | Guard Level |
|------|------|-----|-------------|
| **A: Template** | Known effect type with validated template | LLM fills parameter JSON into pre-validated template | Standard (param range checks) |
| **B: Guarded Custom** | Novel/unsupported effect, no template available | Full code generation with spec-first pipeline + API allowlist + strict validators + explicit risk flag | Maximum (allowlist + lint + eval gate) |

Lane A is the default. Lane B exists as an escape hatch for novel effects, gated behind stricter validation than the current system provides. Over time, successful Lane B outputs can be promoted to Lane A templates.

### Lane Control Policy

- `FORCE_LANE_A=1` forces template-only execution (safest mode, recommended default during stabilization).
- `ENABLE_LANE_B=1` explicitly enables guarded custom generation when needed.
- `FORCE_LANE_B=1` is debug-only and cannot be used in unattended production runs.
- If both force flags are set, `FORCE_LANE_A` wins (fail-safe precedence).

---

## 3. Phased Implementation Plan

### Phase 0: Immediate Fixes (Days 1-2)

**Goal:** Fix known bugs and clean the data foundation. No architectural changes.

| # | Action | Owner | Risk |
|---|--------|-------|------|
| 0.1 | Archive all existing data (`data/*`, `build/orchestrator_state/`, traces) with manifest + timestamp. **Do not delete.** Switch runtime to clean-start dataset. | Either | None (additive) |
| 0.2 | Remove/hard-disable all legacy Script Writer fallback paths (lines 2448, 2837, 2995). Gate behind `ALLOW_LEGACY_WRITER=0` env var (default OFF). | Either | Low — legacy path is already unreliable |
| 0.3 | Fix execution truth contradiction: replace `execution.success = True` override with dual-channel `process_success` + `artifact_viable`. | Either | Medium — touches pipeline control flow |
| 0.4 | Fix serialization mismatch (`execution_time` → `execution_time_seconds`). | Either | Low |
| 0.5 | Align all docs to `openai-agents==0.9.0`. Remove/update all `0.8.3` references. | Either | None |
| 0.6 | Hard-fix `use_nodes` in API fixer (replacement, not warning). | Either | Low |
| 0.7 | Add `Action.fcurves` to API fixer (Blender 5.0 animation API change). | Either | Low |
| 0.8 | Preserve deterministic fallback logic (`quality_parameter_map.py`) by porting useful rules into curated Tier-1 knowledge seed. | Either | Low |
| 0.9 | Freeze model/runtime truth in config: explicit model IDs, `strict` structured outputs, and version stamps for eval reproducibility. | Either | Low |

**Exit criteria:** All 9 items complete. Legacy fallback paths unreachable in default configuration. Execution semantics are truthful.

### Phase 1: Template System (Days 3-7)

**Goal:** Create validated templates for the 3 benchmark effect types. Prove that template + parameter filling produces runnable scripts.

#### 1.1 Extract Templates

| Template | Source Script | Effect Type | Score |
|----------|-------------|-------------|-------|
| `templates/liquid_mantaflow_pour.py` | kitchen_leak best-scoring script (58) | liquid/water | Validated |
| `templates/gas_fire_candle.py` | candle script + fixed camera/lighting | fire | New |
| `templates/gas_smoke_column.py` | smoke test scripts (40-45) | smoke/gas | Validated |

**Process per template:**
1. Extract from best-scoring script
2. Identify all variable parameters, replace with `{{param_name}}` slots
3. Hardcode all Blender API calls (no deprecated attrs possible)
4. Hardcode camera setup with deterministic placement function
5. Hardcode material setup (no `use_nodes`, no `ShaderNodeMixRGB`)
6. Validate: runs headless on Blender 5.0, produces render at default params

#### 1.2 Template Parameter Schema

```python
class TemplateParams(BaseModel):
    """Base schema for all template parameters. Per-effect subclasses add specifics."""

    # Scene geometry
    domain_size: float = Field(ge=0.5, le=5.0, default=1.0)
    resolution_max: int = Field(ge=32, le=256, default=96)

    # Camera (deterministic defaults, LLM can adjust within bounds)
    camera_distance: float = Field(ge=0.3, le=10.0)
    camera_angle: Literal["front", "three_quarter", "top", "side"] = "three_quarter"
    camera_elevation: float = Field(ge=-30, le=60, default=15)
    camera_focal_length: float = Field(ge=24, le=85, default=35)

    # Lighting (per-effect-type defaults and bounds)
    key_light_energy: float = Field(ge=50, le=1000)
    fill_light_ratio: float = Field(ge=0.0, le=0.8, default=0.3)

    # Render
    render_samples: int = Field(ge=32, le=512, default=128)
    render_frame: int = Field(ge=1, le=250, default=40)

class LiquidParams(TemplateParams):
    flow_velocity: float = Field(ge=0.1, le=5.0, default=1.0)
    particle_radius: float = Field(ge=0.5, le=3.0, default=1.0)
    use_spray: bool = False
    use_foam: bool = False
    liquid_color: tuple[float, float, float, float] = (0.1, 0.02, 0.02, 1.0)
    # camera defaults for liquid
    camera_distance: float = Field(ge=0.5, le=5.0, default=1.5)
    key_light_energy: float = Field(ge=100, le=800, default=300)

class FireParams(TemplateParams):
    flame_smoke_ratio: float = Field(ge=0.0, le=1.0, default=0.5)
    burning_rate: float = Field(ge=0.1, le=4.0, default=0.75)
    temperature_diff: float = Field(ge=0.5, le=5.0, default=1.0)
    # camera defaults for fire
    camera_distance: float = Field(ge=1.0, le=10.0, default=2.0)
    key_light_energy: float = Field(ge=50, le=500, default=100)

class SmokeParams(TemplateParams):
    density: float = Field(ge=0.5, le=10.0, default=3.0)
    buoyancy_density: float = Field(ge=-2.0, le=0.0, default=-0.5)
    noise_strength: float = Field(ge=0.0, le=5.0, default=1.0)
    vorticity: float = Field(ge=0.0, le=1.0, default=0.3)
    # camera defaults for smoke
    camera_distance: float = Field(ge=1.5, le=8.0, default=3.0)
    key_light_energy: float = Field(ge=200, le=800, default=500)
```

#### 1.3 Template Engine

Simple Jinja2 or `str.format()` rendering:
1. Load template file
2. Validate `TemplateParams` against schema (Pydantic validation)
3. Render template with validated parameters
4. Run API fixer (simplified — mostly a no-op for templates)
5. Output: complete, syntactically valid Blender Python script

#### 1.4 Template Validation Gates

Each template must pass ALL of these to be accepted (not just "non-black render"):

| Gate | Criterion | How |
|------|-----------|-----|
| API compliance | Zero deprecated/unknown Blender 5.0 attributes | Static lint + allowlist check |
| Syntax | Valid Python AST | `ast.parse()` |
| Execution | Exit code 0 on Blender 5.0 headless | Run with default params |
| Artifact | Render file exists, size > 10KB, not black, not white | Pixel analysis |
| Quality floor | Score >= 30 at default parameters | Vision-based eval |
| Boundary | Runs without crash at min and max parameter extremes | Automated sweep |

**Exit criteria:** 3 templates pass all 6 gates. Template engine renders scripts that execute successfully.

### Phase 2: Modular Pipeline (Days 5-10, overlaps with Phase 1)

**Goal:** Replace the 2,208-line `create_asset_pipeline` with a modular state machine.

#### 2.1 Pipeline Contract

Keep the phase model (agreed by both audits):
```
SELECT → FILL → VALIDATE → EXECUTE → EVALUATE → DECIDE
```

Replace the monolithic implementation with typed, testable phase handlers:

```python
class PipelinePhase(Enum):
    SELECT_TEMPLATE = "select_template"
    FILL_PARAMETERS = "fill_parameters"
    VALIDATE = "validate"
    EXECUTE = "execute"
    EVALUATE = "evaluate"
    DECIDE = "decide"

@dataclass
class PhaseResult:
    next_phase: Optional[PipelinePhase]
    data: dict
    success: bool
    error: Optional[str] = None

class Pipeline:
    """Modular pipeline with typed phase handlers."""

    def __init__(self, handlers: dict[PipelinePhase, PhaseHandler]):
        self._handlers = handlers

    async def run(self, request: AssetRequest) -> PipelineResult:
        context = PipelineContext(request=request)
        phase = PipelinePhase.SELECT_TEMPLATE
        while phase is not None:
            handler = self._handlers[phase]
            result = await handler.execute(context)
            if not result.success:
                # Fail loud — no silent continue
                return PipelineResult(success=False, error=result.error, phase=phase)
            context.update(result.data)
            phase = result.next_phase
        return PipelineResult(success=True, context=context)
```

#### 2.2 Phase Implementations

| Phase | Input | Output | LLM? | Module |
|-------|-------|--------|------|--------|
| SELECT_TEMPLATE | AssetRequest | template_name, lane (A/B) | Yes (simple structured choice) | `phases/select.py` |
| FILL_PARAMETERS | template_name, request, knowledge | TemplateParams JSON | Yes (structured output) | `phases/fill.py` |
| VALIDATE | rendered_script | validation_result | No (deterministic) | `phases/validate.py` |
| EXECUTE | script_path | ExecutionResult (dual-channel) | No (Blender) | `phases/execute.py` |
| EVALUATE | render_path | QualityScore | Yes (vision) | `phases/evaluate.py` |
| DECIDE | score, iteration, history | next_action | Deterministic + optional LLM | `phases/decide.py` |

#### 2.3 Execution Semantics (Dual-Channel)

```python
@dataclass
class ExecutionResult:
    """Truthful execution outcome — never mutated after creation."""
    process_success: bool        # Did Blender exit code == 0?
    artifact_viable: bool        # Does a render file exist and pass basic checks?
    render_path: Optional[str]
    exit_code: int
    execution_time_seconds: float
    errors: list[str]
    warnings: list[str]
```

**Rules:**
- `process_success` is never overridden. It reflects the actual Blender exit code.
- `artifact_viable` is set independently based on file existence + basic pixel checks.
- Evaluation phase runs if `artifact_viable=True`, regardless of `process_success`.
- Learning/experiment tracking receives both channels for honest signal.

#### 2.4 Error Handling Strategy

| Error Type | Behavior | Max Retries |
|------------|----------|-------------|
| Transient (API quota, timeout) | Retry with exponential backoff | 3 |
| Structural (template not found, schema violation) | Fail fast, report to user | 0 |
| Blender crash (segfault, OOM) | Log diagnostics, fail phase | 1 |
| Quality below threshold | Adjust parameters, re-enter FILL phase | Per `max_iterations` |

**No silent `except Exception: continue`.** Every exception either retries, fails the phase, or escalates.

**Exit criteria:** Pipeline runs end-to-end on all 3 templates. Each phase is independently testable with mock inputs.

### Phase 3: Deterministic Camera & Lighting (Days 6-8, overlaps)

**Goal:** Camera placement and lighting are deterministic functions of scene geometry and effect type.

#### 3.1 Camera Placement

```python
def compute_camera_placement(
    scene_bounds: BoundingBox,
    effect_type: EffectType,
    angle: str = "three_quarter",
    focal_length: float = 35.0,
) -> CameraPlacement:
    """
    Deterministic camera position from scene geometry.
    Returns position, rotation, focal length.
    """
    diagonal = scene_bounds.diagonal()
    sensor_width = 36.0  # Blender default full-frame
    hfov = 2 * atan(sensor_width / (2 * focal_length))
    min_distance = diagonal / (2 * tan(hfov / 2)) * 1.3  # 30% safety margin
    min_distance = max(min_distance, 0.5)  # absolute floor

    # Effect-type elevation/azimuth presets
    presets = {
        EffectType.WATER:     {"elevation": 20, "azimuth": 45},
        EffectType.FIRE:      {"elevation": 5,  "azimuth": 30},
        EffectType.SMOKE:     {"elevation": -5, "azimuth": 45},
        EffectType.EXPLOSION: {"elevation": 10, "azimuth": 60},
    }
    preset = presets.get(effect_type, {"elevation": 15, "azimuth": 45})

    # Convert spherical to Cartesian
    center = scene_bounds.center()
    distance = min_distance * 1.5
    # ... (spherical math) ...

    return CameraPlacement(
        location=position,
        rotation=look_at_rotation(position, center),
        focal_length=focal_length,
        distance=distance,
    )
```

**The LLM can adjust** `camera_distance` and `camera_angle` within the template parameter schema bounds. The deterministic function sets the defaults and enforces minimum distances.

#### 3.2 Lighting Presets

| Effect | Key Light Range | Fill Ratio | Notes |
|--------|----------------|------------|-------|
| Liquid | 100-800 | 0.2-0.5 | Moderate; needs caustics visibility |
| Fire | 50-500 | 0.0-0.2 | Fire is self-luminous; low external light |
| Smoke | 200-800 | 0.3-0.6 | Needs backlight for volume visibility |
| Explosion | 100-600 | 0.1-0.3 | Dynamic range important |

These ranges are enforced by the `TemplateParams` schema. The LLM can choose within bounds but cannot exceed them. This prevents the candle whiteout (2500W light energy) that occurred in S1.5.

**Exit criteria:** Camera placement function produces positions outside scene bounding box for all 3 benchmark scenarios. Lighting ranges prevent over/underexposure.

### Phase 4: API Truth & Validation (Days 7-12)

**Goal:** Make deprecated/hallucinated API usage structurally impossible.

#### 4.1 API Truth Pack

A versioned, per-effect-type manifest of valid Blender 5.0 API calls:

```python
# api_truth/blender_5_0.py
FLUID_DOMAIN_ATTRS = {
    "domain_type": {"type": "enum", "values": ["GAS", "LIQUID"]},
    "resolution_max": {"type": "int", "range": [8, 512]},
    "use_adaptive_timesteps": {"type": "bool"},
    "timesteps_max": {"type": "int", "range": [1, 20]},
    "use_flip_particles": {"type": "bool"},
    "particle_radius": {"type": "float", "range": [0.5, 3.0]},
    "cache_type": {"type": "enum", "values": ["ALL", "REPLAY", "MODULAR"]},
    "openvdb_cache_compress_type": {"type": "enum", "values": ["ZIP", "NONE"]},
    # ... complete canonical list
}

DEPRECATED_MAP = {
    "resolution_divisions": "resolution_max",
    "use_adaptive_time_steps": "use_adaptive_timesteps",
    "use_dissolve": "use_dissolve_smoke",
    "sampling_substeps": "subframes",
    "timesteps_per_frame": "timesteps_max",
    "timesteps_maximum": "timesteps_max",
    "reaction_speed": "burning_rate",
    "ShaderNodeMixRGB": "ShaderNodeMix",
    "ShaderNodeSeparateRGB": "ShaderNodeSeparateColor",
    "ShaderNodeCombineRGB": "ShaderNodeCombineColor",
    "BLENDER_EEVEE_NEXT": "BLENDER_EEVEE",
}
```

#### 4.2 Validation Modes

| Mode | When | Unknown API Behavior |
|------|------|---------------------|
| **Strict** (default for templates) | Lane A template rendering | Block: unknown attribute = validation failure |
| **Guarded** (for custom generation) | Lane B custom scripts | Quarantine: flag unknown attrs, require manual review |
| **Debug** | Development/testing only | Warn: log unknown attrs, allow execution |

#### 4.3 Static Script Lint (Pre-Execution)

Before any script reaches Blender:
1. All `bpy.types.*` attribute access checked against truth pack
2. All `bpy.ops.*` calls checked for valid arguments
3. Mandatory preconditions verified (active object for bake ops, OBJECT mode, etc.)
4. Deprecated attribute auto-replacement via existing fixer (retained, simplified)

**Exit criteria:** Zero deprecated attributes reach Blender execution in template path. Lane B scripts pass strict syntax/safety lint, and unknown API usage is quarantined pending explicit human approval.

### Phase 5: Eval & Observability (Days 8-14)

**Goal:** Every run produces measurable, comparable data. No more `status: unknown`.

#### 5.1 Eval Suite (Staged Rollout)

| Stage | Scope | When |
|-------|-------|------|
| **Stage 1** | 3 benchmark scenarios (kitchen_leak, wine_pour, candle) with fixed seeds and assertion-based pass/fail | Immediately |
| **Stage 2** | 10 scenarios across liquid/fire/smoke + failure edge cases | After Stage 1 stable for 2 weeks |
| **Stage 3** | CI merge gate + scheduled nightly runs (manual pre-merge gate if CI is not yet available) | After Stage 2 |
| **Stage 4** | 25-50 scenario matrix when operationally stable | Future |

#### 5.1a Reproducibility Contract

Every benchmark/eval record MUST include:
- fixed seed
- model name + model version
- Blender version
- OpenAI Agents SDK version
- git commit SHA
- config hash
- template hash (or custom script hash for Lane B)
- API truth pack version

Any run missing these fields is non-compliant and cannot be used for promotion decisions.

#### 5.2 Eval Gates

Each eval run must produce a verdict:

| Gate | Criterion | Pass/Fail |
|------|-----------|-----------|
| API compliance | 0 deprecated attributes in rendered script | Hard fail |
| Execution stability | Blender exit code 0 | Soft fail (artifact_viable may still be true) |
| Artifact viability | Render exists, > 10KB, not black/white | Hard fail |
| Quality score | Score >= threshold (60 default) | Soft fail (record for trend analysis) |
| Cost | Total API spend < budget allocation | Soft fail (warning) |

#### 5.3 Observability Contract

Every completed pipeline run MUST produce a structured summary:

```python
@dataclass
class RunSummary:
    """Required fields for every pipeline run. No nulls for completed runs."""
    session_id: str
    timestamp: str
    git_commit: str
    blender_version: str
    agents_sdk_version: str
    model_name: str
    model_version: str
    seed: int
    config_hash: str
    template_hash: Optional[str]
    truth_pack_version: str
    status: Literal["passed", "failed", "max_iterations", "error"]
    template_used: Optional[str]
    lane: Literal["template", "custom"]
    iterations_completed: int
    best_score: float
    best_iteration: int
    process_success: bool
    artifact_viable: bool
    render_path: Optional[str]
    api_fixes_applied: list[str]
    deprecated_attrs_caught: int
    total_api_cost: float
    execution_time_seconds: float
    errors: list[str]
    parameter_history: list[dict]  # params used per iteration
```

**No more `status: unknown` or `quality_score: null` for completed runs.** The trace analyzer must derive terminal values from the event stream or fail explicitly.

#### 5.4 Trend Tracking

Per-effect-type score trends, tracked across runs:
- Mean score (rolling 10-run window)
- Best score
- Pass rate (% runs scoring >= 60)
- Failure mode distribution (API errors, execution crashes, low quality)

**Exit criteria:** All 3 benchmark scenarios produce complete `RunSummary` with no null fields. Score trends are queryable.

### Phase 6: Knowledge Base & Self-Learning (Days 14+, gated)

**Goal:** Build honest, useful knowledge. No autonomous learning until Level 1 is proven.

#### 6.1 Data Policy

| Action | Policy |
|--------|--------|
| Existing data | Archived with manifest + timestamp. Accessible for forensics. NOT used by new pipeline. |
| New experiments | Clean-start dataset. Only new pipeline runs contribute. |
| Purge old data | Only after merged roadmap v1.0 confirms no rollback need. |

#### 6.2 Knowledge Tiers

| Tier | Name | How Populated | Used By |
|------|------|---------------|---------|
| **Tier 1** | `verified_rules` | Human-curated OR eval-validated (statistically significant improvement, no regression) | Pipeline parameter defaults, template selection |
| **Tier 2** | `candidate_hypotheses` | Auto-generated from successful runs | Sandbox testing only — NOT production |

#### 6.3 Curated Knowledge Base (Tier 1, Day 1)

```json
{
  "liquid": {
    "resolution_max": {"default": 96, "min": 64, "max": 256},
    "camera_distance": {"default": 1.5, "min": 0.5, "max": 5.0},
    "key_light_energy": {"default": 300, "min": 100, "max": 800},
    "particle_radius": {"default": 1.0, "min": 0.5, "max": 2.0},
    "rules": [
      "Always use collision effectors for containment geometry",
      "use_plane_init=True required for inflow emitters",
      "cache_type must be ALL for full bake data"
    ]
  },
  "fire": {
    "resolution_max": {"default": 64, "min": 32, "max": 128},
    "camera_distance": {"default": 2.0, "min": 1.0, "max": 10.0},
    "key_light_energy": {"default": 100, "min": 50, "max": 500},
    "burning_rate": {"default": 0.75, "min": 0.1, "max": 4.0},
    "rules": [
      "Fire is self-luminous — low external key light",
      "Never exceed key_light_energy 500 for fire scenes"
    ]
  },
  "smoke": {
    "resolution_max": {"default": 64, "min": 32, "max": 128},
    "camera_distance": {"default": 3.0, "min": 1.5, "max": 8.0},
    "key_light_energy": {"default": 500, "min": 200, "max": 800},
    "density": {"default": 3.0, "min": 0.5, "max": 10.0},
    "rules": [
      "Smoke needs backlight for volume visibility",
      "Higher density requires higher light energy"
    ]
  }
}
```

#### 6.4 Autonomy Progression (Gated)

| Level | Gate | What Unlocks |
|-------|------|-------------|
| **Level 0** (current) | — | Curated knowledge only. No autonomous learning. |
| **Level 1** | 10+ sessions per effect type scoring >= 60 | Parameter memory: record which values produce good scores |
| **Level 2** | 20+ successful runs, stable pass rate > 50% | Bayesian parameter optimization within template bounds |
| **Level 3** | Stable Level 2 for 4+ weeks | Template self-creation: promote successful Lane B scripts to Lane A templates |
| **Level 4** | Stable Level 3, external review | Full autonomy: discover techniques, create templates, update knowledge base |

**Self-improvement protocol (Level 2+):**
1. System generates hypothesis ("increasing resolution_max from 96 to 128 will improve liquid detail")
2. Hypothesis tested in sandbox (3+ eval runs with fixed seed)
3. Measured impact compared to baseline (must show statistically significant improvement AND no regression on other metrics)
4. If improvement confirmed → promote to `verified_rules`
5. If regression detected → automatic rollback, hypothesis rejected
6. All promotions and rejections logged with evidence

#### 6.5 Compatibility Bridge

The existing self-learning modules (`dynamic_instructions.py`, `knowledge_distillation_tools.py`, `proactive_research_tools.py`, `code_pattern_tools.py`) are NOT deleted immediately. Instead:

1. **Phase 0:** Disconnect from pipeline (no longer called by orchestrator)
2. **Phase 1-5:** Remain in codebase but unused — no runtime dependency
3. **Post v1.0:** Confirm zero usage via telemetry/grep, then archive to `deprecated/` directory
4. **Only then:** Delete from active codebase

This gives a one-release-cycle compatibility bridge as Codex's cross-review recommends.

---

## 4. Validation & Eval Strategy

### Benchmark Suite

| Scenario | Effect | Template | S1.5 Baseline | S3 Target | Seed |
|----------|--------|----------|---------------|-----------|------|
| kitchen_leak | liquid | `liquid_mantaflow_pour` | 58 | >= 70 | Fixed |
| wine_pour | liquid | `liquid_mantaflow_pour` | 0 | >= 40 | Fixed |
| candle_fire | fire | `gas_fire_candle` | 18 | >= 40 | Fixed |

### Test Pyramid

| Layer | Count | What | Assert? |
|-------|-------|------|---------|
| Unit tests | ~20 | Individual phase handlers, template rendering, camera math, param validation | Yes (pytest) |
| Integration tests | ~10 | Phase A → Phase B data flow, pipeline end-to-end with mock Blender | Yes (pytest) |
| Benchmark tests | 3 (growing to 10) | Full pipeline on real Blender, score comparison | Yes (score >= threshold) |

### Regression Detection

Before any code change merges to the main branch:
1. All unit tests pass
2. All integration tests pass
3. All 3 benchmark scenarios produce `RunSummary` with no null fields
4. No benchmark score drops more than 10 points or 20% (whichever is stricter) from the rolling 10-run best
5. Once an S3 floor is achieved for 3 consecutive runs, it must not regress below the floor without explicit rollback review:
   - `kitchen_leak >= 60`
   - `wine_pour >= 30`
   - `candle_fire >= 30`

---

## 5. Rollout & Rollback Strategy

### Migration Approach

| Step | Description | Rollback |
|------|-------------|----------|
| 1 | New pipeline code lives alongside old in `pipeline/` directory | Delete new directory |
| 2 | New pipeline wired to a new entry point (`create_asset_pipeline_v2`) | Call old entry point |
| 3 | Old entry point deprecated, new one becomes default | Swap env var |
| 4 | Old code archived to `deprecated/` after 2+ weeks of stable new pipeline | Restore from archive |

### Kill Switch

```bash
# Force old pipeline (emergency rollback)
export PIPELINE_VERSION=v1  # old monolith

# Force new pipeline (default after Phase 2)
export PIPELINE_VERSION=v2  # modular state machine

# Force Lane A only (recommended safety mode during stabilization)
export FORCE_LANE_A=1

# Enable Lane B explicitly (disabled by default in production)
export ENABLE_LANE_B=1

# Force Lane B for all requests (debug custom generation)
export FORCE_LANE_B=1
```

### Rollback Triggers

| Trigger | Action |
|---------|--------|
| Any benchmark score drops > 20 points vs baseline | Freeze merges, force `FORCE_LANE_A=1`, investigate |
| New pipeline produces 3+ consecutive score=0 runs on same fixed-seed scenario | Roll back latest v2 change set and re-run last known-good v2 commit |
| All 3 benchmark scenarios fail artifact viability in two consecutive runs | Emergency fallback to `PIPELINE_VERSION=v1` |
| API cost exceeds 2x budget allocation in a single run | Kill run, alert user |

---

## 6. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Templates too rigid for creative scenes | Medium | Medium | Lane B escape hatch; bounded adaptive overrides for camera/lighting |
| Template extraction encodes hidden bad assumptions | Medium | High | All templates must pass 6-gate validation; boundary parameter sweep |
| 80% failure reduction claim is optimistic | Medium | Low | Treat as hypothesis; measure actual reduction via eval suite |
| Pipeline refactor introduces regressions | Medium | High | Side-by-side execution; kill switch; benchmark regression gate |
| Self-learning never reaches Level 1 gate | Low | Medium | Curated knowledge base provides baseline even without learning |
| OpenAI API quota limits testing bandwidth | High | Medium | Staged eval rollout; budget tracking with persistence |
| Blender 5.x API changes break templates | Low | High | Templates are versioned; API truth pack updated per Blender release |

---

## 7. Ownership & Timeline

### Ownership

| Area | Primary | Red-Team |
|------|---------|----------|
| Template architecture + engine | Claude | Codex |
| Camera/lighting determinism | Claude | Codex |
| Autonomy level progression | Claude | Codex |
| Runtime hardening (legacy removal, truth fix) | Codex | Claude |
| Eval/observability stack | Codex | Claude |
| Migration rollout + kill-switch policy | Codex | Claude |
| API truth pack | Collaborative | — |
| Curated knowledge base | Collaborative | — |

### Governance (Codex + Claude Co-Authoring)

Required pre-freeze artifact:
- A decision matrix with columns: `topic`, `Claude position`, `Codex position`, `agreed decision`, `owner`, `evidence links`, `open risk`.

Before roadmap freeze (`v1.0`), both agents run a reciprocal red-team pass:
- Findings labeled `must_fix`, `should_fix`, or `nice_to_have`
- Only `must_fix` blocks roadmap acceptance
- Unresolved disagreements move to an explicit "Open Questions" section with owner + target decision date

Roadmap change control after freeze:
1. Freeze scope for current sprint at `v1.0`
2. New items go to a `v1.1 Delta Proposals` section
3. No silent scope expansion inside active sprint milestones

### Timeline

| Phase | Days | Dependencies |
|-------|------|-------------|
| Phase 0: Immediate fixes | 1-2 | None |
| Phase 1: Template system | 3-7 | Phase 0 |
| Phase 2: Modular pipeline | 5-10 | Phase 0, overlaps Phase 1 |
| Phase 3: Camera/lighting | 6-8 | Phase 1 (needs templates) |
| Phase 4: API truth + validation | 7-12 | Phase 0, overlaps Phase 2 |
| Phase 5: Eval + observability | 8-14 | Phase 2 (needs pipeline) |
| Phase 6: Knowledge + self-learning | 14+ | Phase 5 (gated by Level 1) |

**Total to functional v2 pipeline:** ~14 days
**Total to self-learning enablement:** Gated by benchmark performance, not calendar

---

## 8. Open Questions & Assumptions

1. Is `quality >= 60` the right default threshold for each effect type, or should thresholds be per-effect calibrated after Stage 1?
2. Should Lane B remain disabled by default until Stage 2 stability, or only in production mode?
3. What is the manual-review SLA for quarantined Lane B runs with unknown APIs?
4. Which 7 additional scenarios enter Stage 2 first (to avoid benchmark overfitting)?
5. At what point is CI automation worth the maintenance cost versus manual pre-merge eval gates?

---

## 9. Quality Checklist (From Codex Merge Notes)

Before this roadmap is frozen as v1.0, all must be true:

- [x] Legacy fallback removal is explicitly scoped (Phase 0.2)
- [x] Monolith replacement is explicit (Phase 2)
- [x] Dual execution semantics are specified (Phase 2.3)
- [x] API truth model (allowlist/contract) is specified (Phase 4)
- [x] Template strategy includes extension path (Lane B + Level 3 promotion)
- [x] Eval gates include thresholds and pass/fail criteria (Phase 5.2)
- [x] Observability schema has required fields (Phase 5.3)
- [x] Rollback triggers are concrete (Section 5)
- [x] Data migration/purge policy is reversible (Phase 6.1)
- [x] Ownership is explicit per phase (Section 7)
- [x] Timeline includes staged scope (Section 7)
- [x] Autonomy is gated by measurable reliability milestones (Phase 6.4)
- [x] Scope boundaries and non-goals are explicit (Section 2)
- [x] Reproducibility metadata contract is explicit (Section 5.1a)
- [x] Lane safety controls include fail-safe defaults (Sections 2 and 5)
- [x] Rollback-to-v1 is reserved for true emergency conditions (Section 5)

---

## Appendix A: Files to Create

| File | Purpose |
|------|---------|
| `templates/liquid_mantaflow_pour.py` | Validated liquid template |
| `templates/gas_fire_candle.py` | Validated fire template |
| `templates/gas_smoke_column.py` | Validated smoke template |
| `template_engine.py` | Parameter rendering + validation |
| `pipeline/pipeline.py` | Modular state machine |
| `pipeline/phases/select.py` | Template/lane selection |
| `pipeline/phases/fill.py` | Parameter filling |
| `pipeline/phases/validate.py` | API truth + lint |
| `pipeline/phases/execute.py` | Blender execution (wraps existing) |
| `pipeline/phases/evaluate.py` | Quality scoring (wraps existing) |
| `pipeline/phases/decide.py` | Iteration decision |
| `camera.py` | Deterministic camera placement |
| `lighting.py` | Per-effect-type lighting |
| `api_truth/blender_5_0.py` | Canonical API truth pack |
| `data/curated_knowledge.json` | Curated parameter ranges |
| `tests/test_template_engine.py` | Template unit tests |
| `tests/test_pipeline_phases.py` | Phase unit tests |
| `tests/test_camera.py` | Camera placement tests |

## Appendix B: Files to Modify

| File | Change |
|------|--------|
| `orchestrator.py` | Add `create_asset_pipeline_v2` entry point delegating to `pipeline/`. Deprecate old method. |
| `tools/blender_api_fixer.py` | Simplify: keep core regex fixes, remove injection steps handled by templates. Hard-fix `use_nodes`. |
| `models/shared_context.py` | Add `ExecutionResult` (dual-channel), `RunSummary`, `TemplateParams` hierarchy |
| `server.py` | Route to `v2` pipeline by default |

## Appendix C: Files to Deprecate (Not Delete)

| File | Reason | When to Archive |
|------|--------|----------------|
| `tools/dynamic_instructions.py` | Returns empty — replaced by curated knowledge | After v1.0 stable 2 weeks |
| `tools/knowledge_distillation_tools.py` | Data-starved — replaced by curated knowledge | After v1.0 stable 2 weeks |
| `tools/proactive_research_tools.py` | Unproven — replaced by simpler iteration logic | After v1.0 stable 2 weeks |
| `tools/code_pattern_tools.py` | Data-starved — replaced by templates | After v1.0 stable 2 weeks |
| `specialized_agents/script_writer.py` | Legacy, deprecated | After legacy fallback confirmed unreachable |
