# Blender VFX Orchestrator: Architecture From Scratch

**Author:** Claude Opus 4.6, designed independently from existing codebase
**Date:** 2026-02-19
**Context:** 188 prior runs, 0 passed quality threshold. Best scores: 58, 18, 0.
**SDK:** OpenAI Agents SDK v0.9.0+
**Budget:** $20/month

---

## Table of Contents

1. [Philosophy and Design Principles](#1-philosophy-and-design-principles)
2. [Agent Architecture](#2-agent-architecture)
3. [Anti-Hallucination Strategy](#3-anti-hallucination-strategy)
4. [Pipeline Design](#4-pipeline-design)
5. [Self-Learning](#5-self-learning)
6. [SDK Patterns](#6-sdk-patterns)
7. [Human-in-the-Loop](#7-human-in-the-loop)
8. [Budget Management](#8-budget-management)
9. [Error Recovery](#9-error-recovery)
10. [What We Explicitly Do NOT Build](#10-what-we-explicitly-do-not-build)
11. [Implementation Order](#11-implementation-order)

---

## 1. Philosophy and Design Principles

### The Diagnosis

188 runs, 0 passes. The previous system was an **LLM apologist** — it generated code from hallucinated knowledge, then tried to repair the damage with regex. That is fundamentally backwards. You cannot out-regex an LLM that confidently invents `resolution_divisions` when the real attribute is `resolution_max`.

The three catastrophic failure modes share a root cause: **the LLM was the source of truth for things it cannot know.**

| Failure | Root Cause |
|---------|-----------|
| Hallucinated Blender 4.x attributes | LLM used training data as truth instead of runtime introspection |
| Camera inside objects | LLM guessed spatial coordinates instead of computing them from scene geometry |
| Mantaflow-only techniques | LLM defaulted to its most-trained pattern instead of researching the prompt |

### Principles (Ranked by Priority)

**P1: Blender Is the Source of Truth, Not the LLM**

The LLM does not know what attributes exist on `FluidDomainSettings`. Blender does. Every attribute name, every enum value, every valid range must come from runtime introspection of the actual Blender process — not from the LLM's training data, not from a static allowlist, not from a regex fixer.

*This is the single most important principle. If you internalize nothing else, internalize this.*

**P2: Compute What You Can, Generate What You Must**

Camera placement is geometry. Light energy bounds are physics. Collision effectors are topology. These are not creative decisions — they are deterministic computations. The LLM should decide *what* to create (a candle flame on a table). Deterministic code should compute *where to put the camera* (3x the scene bounding radius at 30° elevation) and *what parameters are safe* (light energy ∈ [10, 500] for a candle scene).

**P3: Fail Fast, Fail Informatively, Fail Forward**

Every failure must produce:
- An error classification (not just a stack trace)
- A machine-readable diagnosis (what went wrong, which phase, what was attempted)
- A suggested next action (retry with fix, escalate, abandon)

No silent failures. No optimistic exit codes. No "partial success" that hides a broken render.

**P4: The LLM is a Creative Director, Not an Engineer**

The LLM excels at: understanding what "a wine glass being filled with red liquid" means, choosing between Mantaflow liquid vs. particle-based splash, deciding that the scene needs a table and ambient lighting.

The LLM is terrible at: remembering that Blender 5 renamed `resolution_divisions` to `resolution_max`, placing a camera at coordinates that don't intersect geometry, computing bounding boxes.

Architect the system so the LLM does what it's good at and deterministic code does the rest.

**P5: Earn Autonomy Through Evidence**

Every piece of "knowledge" the system uses must have provenance:
- Runtime introspection → trusted immediately
- Successful run outcome → trusted after 3 confirmations
- LLM suggestion → untrusted until validated by Blender

**P6: Budget is a First-Class Constraint**

$20/month means every LLM call must justify its existence. Use the cheapest model that can do the job. Cache aggressively. Never call gpt-5.2 for something gpt-5-mini can handle. Never call any model for something `dir()` can answer.

---

## 2. Agent Architecture

### The Core Insight: Most "Agents" Should Be Functions

The previous system had 9+ agents. Most of them don't need to be agents. An agent is an LLM with tools and instructions — expensive, slow, and unpredictable. A function is cheap, fast, and deterministic.

**Rule: Make it an agent only if the task requires natural language understanding or creative reasoning. Everything else is a function.**

### What IS an Agent

| Agent | Model | Purpose | Why an Agent? |
|-------|-------|---------|---------------|
| **Planner** | gpt-5-mini | Interprets user prompt → scene specification | Requires NLU: "a wine glass being filled" → structured scene description |
| **ScriptWriter** | gpt-5.3-codex | Generates Blender Python from spec + truth pack | Requires code generation with creative problem-solving |
| **Diagnoser** | gpt-5-mini | Classifies execution errors → actionable fix | Requires understanding error messages in context |

That's it. **Three agents.** Everything else is a function, a validator, or a computation.

### What is NOT an Agent

| Component | Implementation | Why Not an Agent? |
|-----------|---------------|-------------------|
| **API Validator** | Function: `validate_script(code, truth_pack) → errors[]` | Pattern matching against known-good data. No NLU needed. |
| **Camera Placer** | Function: `compute_camera(scene_bounds) → transform` | Geometry computation. LLMs are terrible at this. |
| **Quality Evaluator** | Function: `evaluate_render(image_path) → QualityScore` | ML model inference (CLIP, LPIPS). No LLM needed. |
| **Truth Pack Builder** | Function: `introspect_blender(types[]) → TruthPack` | Runtime introspection of Blender process. |
| **Execution Engine** | Function: `run_blender(script_path) → ExecutionResult` | Subprocess management. |
| **Scene Fixer** | Function: `fix_scene(script, diagnosis) → fixed_script` | Deterministic AST transforms based on structured diagnosis. |
| **Budget Tracker** | Function: `check_budget() → BudgetStatus` | Arithmetic. |
| **Knowledge Store** | Function: `query/store_pattern(key, data)` | SQLite operations. |
| **Technique Router** | Function: `select_technique(effect_type, prompt) → Technique` | Lookup table + keyword matching. Only escalate to Planner if ambiguous. |

### Agent Definitions

```python
from agents import Agent, ModelSettings, function_tool
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from typing import Optional


# ─── Shared Context ───────────────────────────────────────────────

@dataclass
class PipelineContext:
    """Shared state across all agents and tools in a single pipeline run."""
    session_id: str
    iteration: int = 0
    max_iterations: int = 5
    truth_pack: dict = field(default_factory=dict)  # Blender introspection data
    scene_spec: Optional[dict] = None
    current_script: Optional[str] = None
    current_errors: list[str] = field(default_factory=list)
    quality_history: list[float] = field(default_factory=list)
    budget_remaining_usd: float = 20.0
    technique: Optional[str] = None
    effect_type: Optional[str] = None


# ─── Structured Outputs ──────────────────────────────────────────

class SceneSpec(BaseModel):
    """What the Planner produces: a structured description of the scene."""
    effect_type: str = Field(description="Primary effect: 'gas_fire', 'gas_smoke', 'liquid_pour', 'liquid_splash', 'rigid_body_destruction', 'particle_debris', etc.")
    technique: str = Field(description="Blender technique: 'mantaflow_gas', 'mantaflow_liquid', 'rigid_body', 'particle_system', 'geometry_nodes'")
    scene_elements: list[str] = Field(description="Objects needed: ['table', 'candle', 'wick', 'flame_domain']")
    physics_description: str = Field(description="Plain English description of the physics setup")
    camera_intent: str = Field(description="Camera framing intent: 'medium shot showing full candle', 'close-up of flame tip', 'wide shot of room'")
    lighting_intent: str = Field(description="Lighting intent: 'warm ambient + key light from upper-left', 'dramatic single spotlight'")
    duration_frames: int = Field(description="Animation length in frames", ge=1, le=500)
    resolution: tuple[int, int] = Field(default=(1920, 1080))

class Diagnosis(BaseModel):
    """What the Diagnoser produces: a structured error classification."""
    error_class: str = Field(description="One of: 'invalid_attribute', 'missing_object', 'physics_setup', 'camera_error', 'render_error', 'bake_error', 'unknown'")
    specific_error: str = Field(description="The specific error message or attribute name")
    fix_strategy: str = Field(description="One of: 'substitute_attribute', 'add_missing_object', 'restructure_physics', 'recompute_camera', 'adjust_render_settings', 'regenerate_script', 'escalate'")
    fix_details: str = Field(description="Specific instructions for the fix, e.g., 'replace resolution_divisions with resolution_max'")
    confidence: float = Field(description="0.0-1.0 confidence in the diagnosis", ge=0.0, le=1.0)


# ─── Agent: Planner ──────────────────────────────────────────────

def planner_instructions(ctx, agent):
    """Dynamic instructions that inject truth pack data."""
    base = (
        "You are a VFX scene planner. Given a user's natural language description, "
        "produce a structured SceneSpec that describes what to build in Blender.\n\n"
        "CRITICAL RULES:\n"
        "1. You decide WHAT to build, not HOW to code it.\n"
        "2. Choose the technique that matches the prompt — NOT always Mantaflow.\n"
        "3. For destruction/shattering → rigid_body, NOT mantaflow.\n"
        "4. For debris/sparks/rain → particle_system, NOT mantaflow.\n"
        "5. For procedural effects → geometry_nodes.\n"
        "6. Camera intent must describe FRAMING, not coordinates.\n"
    )
    # Inject available techniques from truth pack
    if ctx.context.truth_pack.get("available_techniques"):
        techniques = ctx.context.truth_pack["available_techniques"]
        base += f"\nAvailable Blender techniques:\n{techniques}\n"
    return base

planner_agent = Agent[PipelineContext](
    name="Planner",
    instructions=planner_instructions,
    model="gpt-5-mini",  # Cheap — this is NLU, not code gen
    output_type=SceneSpec,
    model_settings=ModelSettings(temperature=0.3),
)


# ─── Agent: ScriptWriter ─────────────────────────────────────────

def writer_instructions(ctx, agent):
    """Dynamic instructions that inject the truth pack — the ONLY source of API knowledge."""
    truth = ctx.context.truth_pack
    spec = ctx.context.scene_spec

    base = (
        "You are a Blender 5.0 Python script writer.\n\n"
        "## ABSOLUTE RULES\n"
        "1. You MUST ONLY use attributes listed in the TRUTH PACK below.\n"
        "2. If an attribute is not in the truth pack, IT DOES NOT EXIST. Do not guess.\n"
        "3. You do NOT set camera position or rotation — leave a `# CAMERA_PLACEHOLDER` comment.\n"
        "4. You do NOT set light energy values — leave a `# LIGHTING_PLACEHOLDER` comment.\n"
        "5. Every fluid domain MUST have `cache_type = 'ALL'`.\n"
        "6. Every liquid flow MUST have `use_plane_init = True`.\n"
        "7. Use `bpy.ops.fluid.bake_all()` after setting up ALL fluid objects.\n"
        "8. Output path: use `os.environ.get('BLENDER_OUTPUT_DIR', '/tmp/vfx_output')`.\n\n"
    )

    # Inject the truth pack — this is the ONLY API reference the LLM gets
    if truth:
        base += "## TRUTH PACK (Blender 5.0 Runtime Introspection)\n"
        base += "These are the ONLY valid attributes. Using anything else will fail.\n\n"
        for type_name, props in truth.items():
            if isinstance(props, dict) and "properties" in props:
                base += f"### {type_name}\n```\n"
                for prop_name, prop_info in props["properties"].items():
                    prop_type = prop_info.get("type", "?")
                    if prop_type == "ENUM":
                        items = prop_info.get("items", [])
                        base += f"  {prop_name}: {prop_type} = {items}\n"
                    elif "range" in prop_info:
                        r = prop_info["range"]
                        base += f"  {prop_name}: {prop_type} [{r[0]}..{r[1]}] default={prop_info.get('default', '?')}\n"
                    else:
                        base += f"  {prop_name}: {prop_type} default={prop_info.get('default', '?')}\n"
                base += "```\n\n"

    # Inject scene spec
    if spec:
        base += f"## SCENE SPECIFICATION\n```json\n{spec}\n```\n\n"

    # Inject previous errors if this is a retry
    if ctx.context.current_errors:
        base += "## ERRORS FROM PREVIOUS ATTEMPT (fix these)\n"
        for err in ctx.context.current_errors:
            base += f"- {err}\n"

    return base

script_writer_agent = Agent[PipelineContext](
    name="ScriptWriter",
    instructions=writer_instructions,
    model="gpt-5.3-codex",  # Best at code generation
    model_settings=ModelSettings(temperature=0.2),
    # No output_type — we want raw Python code as text
)


# ─── Agent: Diagnoser ────────────────────────────────────────────

diagnoser_agent = Agent[PipelineContext](
    name="Diagnoser",
    instructions=(
        "You are a Blender error diagnostician. Given an error traceback and the "
        "script that produced it, classify the error and suggest a specific fix.\n\n"
        "RULES:\n"
        "1. If the error mentions an attribute that doesn't exist, classify as 'invalid_attribute'.\n"
        "2. If the error mentions a missing object/collection, classify as 'missing_object'.\n"
        "3. If the error is about fluid/physics setup, classify as 'physics_setup'.\n"
        "4. Be SPECIFIC in fix_details — name the exact attribute to change and what to change it to.\n"
        "5. If you're not confident (< 0.5), set fix_strategy to 'escalate'.\n"
    ),
    model="gpt-5-mini",
    output_type=Diagnosis,
    model_settings=ModelSettings(temperature=0.1),
)
```

### Why Three Agents, Not Nine

| Eliminated Agent | Replacement | Cost Savings |
|-----------------|-------------|--------------|
| ResearchAgent | Truth pack builder (function) + Planner's technique selection | ~$0.50/run |
| DocsExpert | Runtime introspection replaces doc search | ~$0.30/run |
| QualityAnalyst (LLM) | ML-only quality evaluation function | ~$0.40/run |
| QualityGateJudge | Threshold comparison (arithmetic) | ~$0.20/run |
| TechniqueSelector | Lookup table + Planner | ~$0.20/run |
| ModificationStrategist | Diagnoser + deterministic fix table | ~$0.30/run |
| LearningAgent | Knowledge store functions | ~$0.20/run |
| API Validator (LLM) | Truth pack validator (function) | ~$0.30/run |

**Estimated savings: ~$2.40/run → $0.60/run** (4x reduction)

---

## 3. Anti-Hallucination Strategy

### The Three Layers

```
Layer 1: PREVENT — Don't let the LLM use hallucinated knowledge
Layer 2: DETECT — Catch hallucinations before they reach Blender
Layer 3: CORRECT — When hallucinations slip through, fix them deterministically
```

### Layer 1: PREVENT — The Truth Pack

Before the ScriptWriter ever runs, we build a **Truth Pack** by introspecting live Blender:

```python
import asyncio
import json
import subprocess
from pathlib import Path

# This script runs INSIDE Blender (headless)
INTROSPECTION_SCRIPT = '''
import bpy
import json
import sys

def introspect_type(type_name: str) -> dict:
    """Extract all properties of a bpy.types class with full metadata."""
    bpy_type = getattr(bpy.types, type_name, None)
    if bpy_type is None:
        return {"error": f"Type '{type_name}' not found in bpy.types"}

    result = {
        "type_name": type_name,
        "blender_version": list(bpy.app.version),
        "properties": {},
    }

    for prop_name in bpy_type.bl_rna.properties.keys():
        if prop_name.startswith("rna_"):
            continue
        prop = bpy_type.bl_rna.properties[prop_name]
        info = {
            "type": prop.type,
            "is_readonly": prop.is_readonly,
        }
        if prop.type in ("INT", "FLOAT"):
            info["range"] = [prop.hard_min, prop.hard_max]
            info["default"] = prop.default
        elif prop.type == "BOOLEAN":
            info["default"] = prop.default
        elif prop.type == "ENUM":
            info["items"] = [item.identifier for item in prop.enum_items]
            info["default"] = prop.default
        elif prop.type == "STRING":
            info["default"] = prop.default

        result["properties"][prop_name] = info

    return result

# Types to introspect — driven by the scene spec's technique
types_to_query = json.loads(sys.argv[sys.argv.index("--") + 1])
output = {}
for t in types_to_query:
    output[t] = introspect_type(t)

print("###TRUTH_PACK_START###")
print(json.dumps(output))
print("###TRUTH_PACK_END###")
'''

# Types needed per technique
TECHNIQUE_TYPES = {
    "mantaflow_gas": [
        "FluidDomainSettings", "FluidFlowSettings", "FluidEffectorSettings",
        "FluidModifier", "Object", "Camera", "PointLight", "SpotLight",
        "SunLight", "AreaLight", "ShaderNodeMix", "ShaderNodeOutputMaterial",
        "ShaderNodeVolumePrincipled", "ShaderNodeVolumeAbsorption",
        "CyclesRenderSettings", "RenderSettings", "Scene",
    ],
    "mantaflow_liquid": [
        "FluidDomainSettings", "FluidFlowSettings", "FluidEffectorSettings",
        "FluidModifier", "Object", "Camera", "PointLight", "SpotLight",
        "SunLight", "AreaLight", "ShaderNodeBsdfPrincipled",
        "ShaderNodeMix", "ShaderNodeOutputMaterial",
        "CyclesRenderSettings", "RenderSettings", "Scene",
    ],
    "rigid_body": [
        "RigidBodyWorld", "RigidBodyObject", "RigidBodyConstraint",
        "Object", "Camera", "PointLight", "CyclesRenderSettings",
        "RenderSettings", "Scene",
    ],
    "particle_system": [
        "ParticleSettings", "ParticleSystem",
        "Object", "Camera", "PointLight", "CyclesRenderSettings",
        "RenderSettings", "Scene",
    ],
}


async def build_truth_pack(technique: str) -> dict:
    """Run Blender introspection and return the truth pack."""
    types_needed = TECHNIQUE_TYPES.get(technique, TECHNIQUE_TYPES["mantaflow_gas"])

    # Write introspection script to temp file
    script_path = Path("/tmp/vfx_introspect.py")
    script_path.write_text(INTROSPECTION_SCRIPT)

    proc = await asyncio.create_subprocess_exec(
        "blender", "--background", "--python", str(script_path),
        "--", json.dumps(types_needed),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"Blender introspection failed: {stderr.decode()[:500]}")

    output = stdout.decode()
    start_marker = "###TRUTH_PACK_START###"
    end_marker = "###TRUTH_PACK_END###"

    start = output.index(start_marker) + len(start_marker)
    end = output.index(end_marker)
    return json.loads(output[start:end].strip())
```

**Why this works:** The LLM never guesses attribute names. It receives a complete list of valid attributes with their types, ranges, and defaults. If `resolution_divisions` isn't in the truth pack, the LLM cannot use it — because its instructions say "ONLY use attributes listed in the TRUTH PACK."

**Cost: $0.00** — This is a Blender subprocess call, no LLM involved.

### Layer 2: DETECT — Static Validation Against Truth Pack

After the ScriptWriter generates code, validate it BEFORE sending it to Blender:

```python
import ast
import re
from dataclasses import dataclass


@dataclass
class ValidationError:
    line: int
    attribute: str
    object_type: str
    message: str


def validate_script_against_truth_pack(
    script: str,
    truth_pack: dict,
) -> list[ValidationError]:
    """
    Parse the generated script and check every attribute access against the truth pack.
    Returns a list of validation errors (empty = script is clean).
    """
    errors = []

    # Map of bpy.types class names → known property names
    valid_attrs: dict[str, set[str]] = {}
    for type_name, type_data in truth_pack.items():
        if isinstance(type_data, dict) and "properties" in type_data:
            valid_attrs[type_name] = set(type_data["properties"].keys())

    # Pattern: obj.modifiers["Fluid"].domain_settings.ATTR
    # We check ATTR against FluidDomainSettings
    SETTINGS_MAP = {
        "domain_settings": "FluidDomainSettings",
        "flow_settings": "FluidFlowSettings",
        "effector_settings": "FluidEffectorSettings",
        "rigid_body": "RigidBodyObject",
        "rigid_body_world": "RigidBodyWorld",
        "particle_systems": "ParticleSystem",
        "cycles": "CyclesRenderSettings",
        "data": None,  # Could be Camera, Light, Mesh — context-dependent
    }

    lines = script.split("\n")
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue

        # Check for known settings access patterns
        for settings_attr, type_name in SETTINGS_MAP.items():
            if type_name is None or type_name not in valid_attrs:
                continue

            # Pattern: .settings_attr.SOMETHING = value
            pattern = rf'\.{re.escape(settings_attr)}\.(\w+)'
            for match in re.finditer(pattern, line):
                attr = match.group(1)
                if attr not in valid_attrs[type_name] and not attr.startswith("_"):
                    errors.append(ValidationError(
                        line=i,
                        attribute=attr,
                        object_type=type_name,
                        message=f"'{attr}' is not a valid property of {type_name}. "
                                f"Valid properties: {sorted(valid_attrs[type_name])[:10]}...",
                    ))

    # Check for known hallucination patterns (fast regex)
    KNOWN_HALLUCINATIONS = {
        r'\bresolution_divisions\b': "Use 'resolution_max' instead",
        r'\buse_adaptive_time_steps\b': "Use 'use_adaptive_timesteps' (no extra underscore)",
        r'\buse_dissolve\b(?!_smoke)': "Use 'use_dissolve_smoke' instead",
        r'\btimesteps_per_frame\b': "Use 'timesteps_max' instead",
        r'\btimesteps_maximum\b': "Use 'timesteps_max' instead",
        r'\breaction_speed\b': "Use 'burning_rate' instead",
        r'\bShaderNodeMixRGB\b': "Use 'ShaderNodeMix' in Blender 5.0",
        r'\bShaderNodeSeparateRGB\b': "Use 'ShaderNodeSeparateColor' in Blender 5.0",
        r'\bBLENDER_EEVEE_NEXT\b': "Use 'BLENDER_EEVEE' in Blender 5.0",
        r'\bSubsurface Color\b': "Removed from Principled BSDF in 5.0",
        r"['\"]BLOSC['\"]": "BLOSC compression removed in 5.0. Use 'ZIP' or 'NONE'.",
    }

    for pattern, message in KNOWN_HALLUCINATIONS.items():
        for i, line in enumerate(lines, 1):
            if re.search(pattern, line):
                errors.append(ValidationError(
                    line=i,
                    attribute=pattern,
                    object_type="KNOWN_HALLUCINATION",
                    message=message,
                ))

    return errors
```

**Why this works:** The truth pack is ground truth from Blender itself. Any attribute not in it is invalid. This catches hallucinations BEFORE execution, saving a ~30s Blender subprocess call.

### Layer 3: CORRECT — Deterministic Substitution

When validation catches an error, apply deterministic fixes:

```python
# Auto-fix table: hallucinated → correct (from truth pack)
# These are populated at runtime from the truth pack, not hardcoded
def build_substitution_table(truth_pack: dict) -> dict[str, dict[str, str]]:
    """
    Build a substitution table from the truth pack.
    Key: type_name → {wrong_attr: correct_attr}
    Uses edit distance to suggest corrections.
    """
    import difflib
    table = {}
    for type_name, type_data in truth_pack.items():
        if not isinstance(type_data, dict) or "properties" not in type_data:
            continue
        valid_props = list(type_data["properties"].keys())
        table[type_name] = {}
        # We'll populate this on-demand when errors are found
        table[type_name]["_valid_props"] = valid_props
    return table


def suggest_correction(wrong_attr: str, valid_props: list[str]) -> str | None:
    """Use difflib to find the closest valid attribute name."""
    import difflib
    matches = difflib.get_close_matches(wrong_attr, valid_props, n=1, cutoff=0.6)
    return matches[0] if matches else None


def auto_fix_script(
    script: str,
    errors: list[ValidationError],
    truth_pack: dict,
) -> tuple[str, list[str]]:
    """
    Apply deterministic fixes for validation errors.
    Returns (fixed_script, list_of_fixes_applied).
    """
    fixes_applied = []
    fixed = script

    for error in errors:
        if error.object_type == "KNOWN_HALLUCINATION":
            # Known hallucinations have hardcoded fixes
            # These are the 10-15 most common ones that NEVER change
            HARDCODED_FIXES = {
                "resolution_divisions": "resolution_max",
                "use_adaptive_time_steps": "use_adaptive_timesteps",
                "use_dissolve": "use_dissolve_smoke",
                "timesteps_per_frame": "timesteps_max",
                "timesteps_maximum": "timesteps_max",
                "reaction_speed": "burning_rate",
                "ShaderNodeMixRGB": "ShaderNodeMix",
                "ShaderNodeSeparateRGB": "ShaderNodeSeparateColor",
                "BLENDER_EEVEE_NEXT": "BLENDER_EEVEE",
            }
            for wrong, correct in HARDCODED_FIXES.items():
                if wrong in fixed:
                    fixed = fixed.replace(wrong, correct)
                    fixes_applied.append(f"Replaced '{wrong}' with '{correct}'")

        elif error.object_type in truth_pack:
            # Use truth pack to find correct attribute
            valid_props = list(truth_pack[error.object_type]["properties"].keys())
            suggestion = suggest_correction(error.attribute, valid_props)
            if suggestion:
                fixed = fixed.replace(error.attribute, suggestion)
                fixes_applied.append(
                    f"Replaced '{error.attribute}' with '{suggestion}' "
                    f"on {error.object_type}"
                )

    return fixed, fixes_applied
```

### The Anti-Hallucination Pipeline

```
1. Build truth pack (Blender introspection)     — $0.00, ~5s
2. Inject truth pack into ScriptWriter prompt    — included in generation cost
3. ScriptWriter generates code                   — ~$0.05
4. Validate code against truth pack              — $0.00, <1s
5. Auto-fix any caught errors                    — $0.00, <1s
6. If unfixable errors remain → regenerate       — ~$0.05 (rare)
7. Execute in Blender                            — $0.00, ~30s
8. If runtime error → Diagnoser classifies       — ~$0.02
9. Apply deterministic fix based on diagnosis    — $0.00
10. Loop back to step 4 (max 3 attempts)
```

**Key insight:** Steps 1, 4, 5, 9 are FREE. The previous system's 57-rule regex fixer was doing step 9 only, and never doing steps 1-5 at all.

---

## 4. Pipeline Design

### State Machine

```
                    ┌─────────────────────────────────────────────┐
                    │                                             │
                    ▼                                             │
              ┌──────────┐                                        │
  USER ──────►│ INTROSPECT│──► truth_pack                        │
  PROMPT      └──────────┘                                        │
                    │                                             │
                    ▼                                             │
              ┌──────────┐                                        │
              │   PLAN    │──► scene_spec (SceneSpec)             │
              └──────────┘                                        │
                    │                                             │
                    ▼                                             │
              ┌──────────┐     ┌──────────┐                      │
              │ GENERATE  │────►│ VALIDATE │                      │
              └──────────┘     └──────────┘                      │
                    ▲                │                            │
                    │           valid?                            │
                    │          /     \                            │
                   NO          YES                               │
                    │            │                                │
                    │            ▼                                │
                    │      ┌──────────┐                           │
                    │      │ EXECUTE  │                           │
                    │      └──────────┘                           │
                    │            │                                │
                    │       success?                              │
                    │      /       \                              │
                    │    NO        YES                            │
                    │     │          │                            │
                    │     ▼          ▼                            │
                    │ ┌──────────┐ ┌──────────┐                  │
                    │ │ DIAGNOSE │ │  RENDER   │                  │
                    │ └──────────┘ └──────────┘                  │
                    │     │          │                            │
                    │     │          ▼                            │
                    │     │    ┌──────────┐                      │
                    │     │    │ EVALUATE  │──► quality_score     │
                    │     │    └──────────┘                       │
                    │     │          │                            │
                    │     │     score >= 60?                      │
                    │     │    /          \                       │
                    │     │  NO          YES ──────► DONE ✓      │
                    │     │   │                                   │
                    │     ▼   ▼                                   │
                    │ ┌──────────────┐                            │
                    └─┤  IMPROVE     │ iteration++ < max?         │
                      └──────────────┘       │                   │
                                          NO ──────► DONE ✗     │
```

### Phase Contracts

Each phase has a strict input/output contract. No phase can run without its required inputs.

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


class Phase(str, Enum):
    INTROSPECT = "introspect"
    PLAN = "plan"
    GENERATE = "generate"
    VALIDATE = "validate"
    EXECUTE = "execute"
    RENDER = "render"
    EVALUATE = "evaluate"
    DIAGNOSE = "diagnose"
    IMPROVE = "improve"
    DONE = "done"


class PhaseResult(BaseModel):
    """Every phase returns one of these. No exceptions."""
    phase: Phase
    success: bool
    next_phase: Phase
    data: dict = Field(default_factory=dict)
    error: Optional[str] = None
    cost_usd: float = 0.0


# ─── Phase: INTROSPECT ───────────────────────────────────────────

class IntrospectInput(BaseModel):
    technique: str  # e.g., "mantaflow_gas"

class IntrospectOutput(BaseModel):
    truth_pack: dict  # The full introspection result
    types_queried: list[str]
    blender_version: list[int]


# ─── Phase: PLAN ─────────────────────────────────────────────────

class PlanInput(BaseModel):
    user_prompt: str
    truth_pack: dict

class PlanOutput(BaseModel):
    scene_spec: SceneSpec
    technique: str


# ─── Phase: GENERATE ─────────────────────────────────────────────

class GenerateInput(BaseModel):
    scene_spec: SceneSpec
    truth_pack: dict
    previous_errors: list[str] = Field(default_factory=list)

class GenerateOutput(BaseModel):
    script: str
    model_used: str
    token_count: int


# ─── Phase: VALIDATE ─────────────────────────────────────────────

class ValidateInput(BaseModel):
    script: str
    truth_pack: dict

class ValidateOutput(BaseModel):
    is_valid: bool
    errors: list[str]
    fixed_script: Optional[str] = None  # If auto-fix was applied
    fixes_applied: list[str] = Field(default_factory=list)


# ─── Phase: EXECUTE ──────────────────────────────────────────────

class ExecuteInput(BaseModel):
    script_path: str  # Path to the script file

class ExecuteOutput(BaseModel):
    success: bool
    stdout: str
    stderr: str
    return_code: int
    duration_seconds: float
    output_files: list[str]  # Rendered images, caches, etc.


# ─── Phase: EVALUATE ─────────────────────────────────────────────

class EvaluateInput(BaseModel):
    render_path: str
    scene_spec: SceneSpec

class EvaluateOutput(BaseModel):
    overall_score: float  # 0-100
    subscores: dict  # {"composition": 72, "lighting": 45, ...}
    critical_issues: list[str]  # ["ZERO_LIGHTS", "BLACK_SCREEN", ...]
    passed: bool  # overall_score >= 60 AND no critical issues


# ─── Phase: DIAGNOSE ─────────────────────────────────────────────

class DiagnoseInput(BaseModel):
    error_output: str  # stderr from Blender
    script: str
    truth_pack: dict

class DiagnoseOutput(BaseModel):
    diagnosis: Diagnosis
    can_auto_fix: bool
    fixed_script: Optional[str] = None
```

### The Pipeline Executor

```python
from agents import Runner, trace, RunConfig
import time


class VFXPipeline:
    """
    The pipeline executor. This is NOT an agent — it's a deterministic
    state machine that calls agents when LLM reasoning is needed.
    """

    def __init__(self, context: PipelineContext):
        self.ctx = context
        self.phase_history: list[PhaseResult] = []

    async def run(self, user_prompt: str) -> PhaseResult:
        """Execute the full pipeline for a user prompt."""

        with trace(f"VFX Pipeline: {user_prompt[:50]}"):

            # Phase 0: Introspect (always first, always free)
            technique = self._guess_technique(user_prompt)
            truth_pack = await build_truth_pack(technique)
            self.ctx.truth_pack = truth_pack

            # Phase 1: Plan (LLM call: ~$0.03)
            scene_spec = await self._plan(user_prompt)
            self.ctx.scene_spec = scene_spec

            # If technique changed based on planning, re-introspect
            if scene_spec.technique != technique:
                truth_pack = await build_truth_pack(scene_spec.technique)
                self.ctx.truth_pack = truth_pack

            # Iteration loop
            for iteration in range(self.ctx.max_iterations):
                self.ctx.iteration = iteration

                result = await self._iterate()

                if result.phase == Phase.DONE and result.success:
                    return result

                if result.phase == Phase.DONE and not result.success:
                    # Max retries within iteration exhausted
                    if iteration < self.ctx.max_iterations - 1:
                        continue  # Next full iteration
                    return result

            return PhaseResult(
                phase=Phase.DONE,
                success=False,
                next_phase=Phase.DONE,
                error=f"Max iterations ({self.ctx.max_iterations}) reached",
            )

    async def _iterate(self) -> PhaseResult:
        """One generate → validate → execute → evaluate cycle."""

        # Generate
        script = await self._generate()

        # Validate (free, instant)
        validation = self._validate(script)
        if not validation.is_valid and validation.fixed_script:
            script = validation.fixed_script
        elif not validation.is_valid:
            # Can't auto-fix — tell ScriptWriter what's wrong
            self.ctx.current_errors = validation.errors
            return PhaseResult(
                phase=Phase.VALIDATE,
                success=False,
                next_phase=Phase.GENERATE,
                error=f"Validation failed: {validation.errors}",
            )

        # Apply deterministic scene fixes
        script = self._apply_camera_placement(script)
        script = self._apply_lighting_bounds(script)
        script = self._apply_collision_effectors(script)

        # Execute in Blender
        exec_result = await self._execute(script)
        if not exec_result.success:
            # Diagnose and attempt fix (up to 2 retries)
            for retry in range(2):
                diagnosis = await self._diagnose(exec_result, script)
                if diagnosis.can_auto_fix and diagnosis.fixed_script:
                    script = diagnosis.fixed_script
                    exec_result = await self._execute(script)
                    if exec_result.success:
                        break
                else:
                    break

            if not exec_result.success:
                self.ctx.current_errors = [exec_result.stderr[:500]]
                return PhaseResult(
                    phase=Phase.EXECUTE,
                    success=False,
                    next_phase=Phase.GENERATE,
                    error=exec_result.stderr[:500],
                )

        # Find rendered image
        render_path = self._find_render(exec_result.output_files)
        if not render_path:
            return PhaseResult(
                phase=Phase.RENDER,
                success=False,
                next_phase=Phase.GENERATE,
                error="No render output found",
            )

        # Evaluate quality (ML only, no LLM)
        quality = await self._evaluate(render_path)
        self.ctx.quality_history.append(quality.overall_score)

        if quality.passed:
            return PhaseResult(
                phase=Phase.DONE,
                success=True,
                next_phase=Phase.DONE,
                data={
                    "score": quality.overall_score,
                    "render_path": render_path,
                    "iterations": self.ctx.iteration + 1,
                },
            )

        # Failed quality — feed issues back for next iteration
        self.ctx.current_errors = [
            f"Quality score: {quality.overall_score}/100 (need >= 60)",
            f"Critical issues: {quality.critical_issues}",
            f"Subscores: {quality.subscores}",
        ]

        return PhaseResult(
            phase=Phase.EVALUATE,
            success=False,
            next_phase=Phase.GENERATE,
            data={"score": quality.overall_score},
        )

    def _guess_technique(self, prompt: str) -> str:
        """Fast keyword-based technique guess. No LLM needed."""
        prompt_lower = prompt.lower()
        if any(w in prompt_lower for w in ["fire", "flame", "smoke", "explosion", "candle"]):
            return "mantaflow_gas"
        if any(w in prompt_lower for w in ["water", "liquid", "pour", "splash", "rain"]):
            return "mantaflow_liquid"
        if any(w in prompt_lower for w in ["destroy", "shatter", "collapse", "break", "smash"]):
            return "rigid_body"
        if any(w in prompt_lower for w in ["debris", "sparks", "snow", "dust", "particle"]):
            return "particle_system"
        return "mantaflow_gas"  # Default, will be refined by Planner

    async def _plan(self, user_prompt: str) -> SceneSpec:
        """Call the Planner agent to produce a SceneSpec."""
        result = await Runner.run(
            planner_agent,
            user_prompt,
            context=self.ctx,
            max_turns=3,
        )
        return result.final_output

    async def _generate(self) -> str:
        """Call the ScriptWriter agent to produce a Blender Python script."""
        prompt = "Generate a complete Blender 5.0 Python script for the scene specification."
        if self.ctx.current_errors:
            prompt += (
                "\n\nPREVIOUS ERRORS TO FIX:\n"
                + "\n".join(f"- {e}" for e in self.ctx.current_errors)
            )

        result = await Runner.run(
            script_writer_agent,
            prompt,
            context=self.ctx,
            max_turns=3,
        )
        return result.final_output

    def _validate(self, script: str) -> ValidateOutput:
        """Validate script against truth pack. No LLM. Free."""
        errors = validate_script_against_truth_pack(script, self.ctx.truth_pack)
        if not errors:
            return ValidateOutput(is_valid=True, errors=[])

        # Try auto-fix
        fixed, fixes = auto_fix_script(script, errors, self.ctx.truth_pack)

        # Re-validate
        remaining_errors = validate_script_against_truth_pack(fixed, self.ctx.truth_pack)

        return ValidateOutput(
            is_valid=len(remaining_errors) == 0,
            errors=[e.message for e in remaining_errors],
            fixed_script=fixed if fixes else None,
            fixes_applied=fixes,
        )

    async def _execute(self, script: str) -> ExecuteOutput:
        """Run script in headless Blender."""
        import tempfile
        script_path = tempfile.mktemp(suffix=".py")
        with open(script_path, "w") as f:
            f.write(script)

        start = time.time()
        proc = await asyncio.create_subprocess_exec(
            "blender", "--background", "--python", script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        duration = time.time() - start

        return ExecuteOutput(
            success=proc.returncode == 0,
            stdout=stdout.decode()[-2000:],  # Last 2000 chars
            stderr=stderr.decode()[-2000:],
            return_code=proc.returncode,
            duration_seconds=duration,
            output_files=self._scan_output_dir(),
        )

    async def _diagnose(self, exec_result: ExecuteOutput, script: str) -> DiagnoseOutput:
        """Call Diagnoser agent to classify the error."""
        prompt = (
            f"Error output:\n```\n{exec_result.stderr[:1000]}\n```\n\n"
            f"Script (last 100 lines):\n```python\n"
            f"{chr(10).join(script.split(chr(10))[-100:])}\n```"
        )
        result = await Runner.run(
            diagnoser_agent,
            prompt,
            context=self.ctx,
            max_turns=2,
        )
        diagnosis = result.final_output

        # Attempt auto-fix based on diagnosis
        fixed_script = None
        can_fix = False

        if diagnosis.fix_strategy == "substitute_attribute":
            # Look up correct attribute in truth pack
            if diagnosis.error_class == "invalid_attribute":
                wrong = diagnosis.specific_error
                for type_name, type_data in self.ctx.truth_pack.items():
                    if isinstance(type_data, dict) and "properties" in type_data:
                        suggestion = suggest_correction(
                            wrong, list(type_data["properties"].keys())
                        )
                        if suggestion:
                            fixed_script = script.replace(wrong, suggestion)
                            can_fix = True
                            break

        return DiagnoseOutput(
            diagnosis=diagnosis,
            can_auto_fix=can_fix,
            fixed_script=fixed_script,
        )

    def _apply_camera_placement(self, script: str) -> str:
        """Replace CAMERA_PLACEHOLDER with deterministic camera placement code."""
        if "# CAMERA_PLACEHOLDER" not in script:
            return script

        camera_code = '''
# ─── DETERMINISTIC CAMERA PLACEMENT ─────────────────────────────
# Compute scene bounding box from all mesh objects
import mathutils
all_coords = []
for obj in bpy.data.objects:
    if obj.type == 'MESH' and obj.visible_get():
        bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
        all_coords.extend(bbox_corners)

if all_coords:
    xs = [c.x for c in all_coords]
    ys = [c.y for c in all_coords]
    zs = [c.z for c in all_coords]
    center = mathutils.Vector((
        (min(xs) + max(xs)) / 2,
        (min(ys) + max(ys)) / 2,
        (min(zs) + max(zs)) / 2,
    ))
    radius = max(
        max(xs) - min(xs),
        max(ys) - min(ys),
        max(zs) - min(zs),
    ) / 2
else:
    center = mathutils.Vector((0, 0, 0))
    radius = 2.0

# Camera at 3x bounding radius, 30° elevation, 3/4 view
import math
cam_distance = max(radius * 3.5, 3.0)  # Never closer than 3 units
cam_elevation = math.radians(25)
cam_azimuth = math.radians(45)  # 3/4 view

cam_x = center.x + cam_distance * math.cos(cam_elevation) * math.cos(cam_azimuth)
cam_y = center.y + cam_distance * math.cos(cam_elevation) * math.sin(cam_azimuth)
cam_z = center.z + cam_distance * math.sin(cam_elevation)

camera = bpy.context.scene.camera
if camera is None:
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    bpy.context.scene.camera = camera

camera.location = (cam_x, cam_y, cam_z)
direction = center - camera.location
camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
# Set focal length for good framing
camera.data.lens = 50
# ─── END CAMERA PLACEMENT ───────────────────────────────────────
'''
        return script.replace("# CAMERA_PLACEHOLDER", camera_code)

    def _apply_lighting_bounds(self, script: str) -> str:
        """Replace LIGHTING_PLACEHOLDER with bounded lighting code."""
        if "# LIGHTING_PLACEHOLDER" not in script:
            return script

        # Lighting bounds per effect type
        LIGHT_BOUNDS = {
            "mantaflow_gas": {"key": (200, 800), "fill": (50, 200), "world": (0.02, 0.2)},
            "mantaflow_liquid": {"key": (300, 1000), "fill": (100, 300), "world": (0.05, 0.3)},
            "rigid_body": {"key": (200, 600), "fill": (80, 200), "world": (0.03, 0.15)},
            "particle_system": {"key": (200, 600), "fill": (80, 200), "world": (0.03, 0.15)},
        }

        bounds = LIGHT_BOUNDS.get(
            self.ctx.scene_spec.technique if self.ctx.scene_spec else "mantaflow_gas",
            LIGHT_BOUNDS["mantaflow_gas"],
        )
        key_energy = (bounds["key"][0] + bounds["key"][1]) / 2
        fill_energy = (bounds["fill"][0] + bounds["fill"][1]) / 2
        world_strength = (bounds["world"][0] + bounds["world"][1]) / 2

        lighting_code = f'''
# ─── DETERMINISTIC LIGHTING ─────────────────────────────────────
# Key light: upper-left, warm
bpy.ops.object.light_add(type='AREA', location=(3, -3, 5))
key_light = bpy.context.object
key_light.data.energy = {key_energy}
key_light.data.color = (1.0, 0.95, 0.9)
key_light.data.size = 2.0

# Fill light: opposite side, cooler, dimmer
bpy.ops.object.light_add(type='AREA', location=(-3, 2, 3))
fill_light = bpy.context.object
fill_light.data.energy = {fill_energy}
fill_light.data.color = (0.9, 0.93, 1.0)
fill_light.data.size = 3.0

# World ambient
world = bpy.context.scene.world
if world is None:
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
world.use_nodes = True
bg_node = world.node_tree.nodes.get("Background")
if bg_node:
    bg_node.inputs["Strength"].default_value = {world_strength}
    bg_node.inputs["Color"].default_value = (0.05, 0.05, 0.05, 1.0)
# ─── END LIGHTING ───────────────────────────────────────────────
'''
        return script.replace("# LIGHTING_PLACEHOLDER", lighting_code)

    def _apply_collision_effectors(self, script: str) -> str:
        """Inject collision effectors for liquid scenes."""
        if not self.ctx.scene_spec or "liquid" not in self.ctx.scene_spec.technique:
            return script
        # For liquid scenes, ensure all mesh objects that aren't the domain or
        # flow have effector modifiers. This is injected BEFORE the bake call.
        effector_code = '''
# ─── AUTO-INJECT COLLISION EFFECTORS ────────────────────────────
for obj in bpy.data.objects:
    if obj.type != 'MESH':
        continue
    has_fluid = any(m.type == 'FLUID' for m in obj.modifiers)
    if has_fluid:
        continue  # Skip domain/flow objects
    # Check if this is a container or surface that should collide
    # Add effector modifier
    fluid_mod = obj.modifiers.new("Fluid", 'FLUID')
    fluid_mod.fluid_type = 'EFFECTOR'
    fluid_mod.effector_settings.effector_type = 'COLLISION'
    fluid_mod.effector_settings.use_effector = True
# ─── END COLLISION EFFECTORS ────────────────────────────────────
'''
        # Insert before bake_all
        if "bpy.ops.fluid.bake_all()" in script:
            return script.replace(
                "bpy.ops.fluid.bake_all()",
                effector_code + "\nbpy.ops.fluid.bake_all()",
            )
        return script + "\n" + effector_code

    async def _evaluate(self, render_path: str) -> EvaluateOutput:
        """ML-only quality evaluation. No LLM."""
        # This calls your existing CLIP/LPIPS/TOPIQ evaluation
        # Stubbed here — the real implementation uses asset_evaluator
        score = await _run_ml_evaluation(render_path, self.ctx.scene_spec)
        return score

    def _scan_output_dir(self) -> list[str]:
        """Scan output directory for rendered files."""
        import os
        output_dir = os.environ.get("BLENDER_OUTPUT_DIR", "/tmp/vfx_output")
        if not os.path.exists(output_dir):
            return []
        return [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.endswith((".png", ".jpg", ".exr"))
        ]

    def _find_render(self, files: list[str]) -> str | None:
        """Find the most recent render in the output files."""
        import os
        renders = [f for f in files if f.endswith((".png", ".jpg"))]
        if not renders:
            return None
        return max(renders, key=os.path.getmtime)
```

### Error Routing Table

| Phase | Error Type | Route To | Max Retries |
|-------|-----------|----------|-------------|
| VALIDATE | Invalid attributes | Auto-fix → re-validate | 1 |
| VALIDATE | Unfixable errors | GENERATE (with error feedback) | 2 |
| EXECUTE | Runtime attribute error | DIAGNOSE → auto-fix → re-execute | 2 |
| EXECUTE | Bake failure | DIAGNOSE → GENERATE (new script) | 1 |
| EXECUTE | Crash/timeout | GENERATE (different technique) | 1 |
| EVALUATE | Score < 60, no critical | GENERATE (with quality feedback) | remaining iterations |
| EVALUATE | Critical issue (BLACK_SCREEN) | GENERATE (fix specific issue) | 2 |
| DIAGNOSE | Low confidence (< 0.5) | ESCALATE to human | 0 |

---

## 5. Self-Learning

### What We Learn and How

The system learns from three signal sources. All learning is **evidence-gated** — nothing enters trusted knowledge without proof.

```python
from dataclasses import dataclass, field
from datetime import datetime
import sqlite3
from pathlib import Path


@dataclass
class LearningSignal:
    """One data point from a pipeline run."""
    session_id: str
    timestamp: datetime
    effect_type: str
    technique: str
    iteration: int
    quality_score: float
    errors: list[str]
    fixes_applied: list[str]
    truth_pack_misses: list[str]  # Attributes the LLM tried that weren't in truth pack
    camera_placement: dict  # {distance, elevation, azimuth} used
    light_energy: dict  # {key, fill, world} used
    script_hash: str  # For deduplication


class KnowledgeStore:
    """
    SQLite-backed knowledge store. No LLM needed for any operation.
    Evidence-gated: patterns only become 'trusted' after N successes.
    """

    TRUST_THRESHOLD = 3  # Successful uses before a pattern is trusted

    def __init__(self, db_path: str = "knowledge.db"):
        self.db = sqlite3.connect(db_path)
        self._init_tables()

    def _init_tables(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY,
                effect_type TEXT NOT NULL,
                technique TEXT NOT NULL,
                pattern_key TEXT NOT NULL,
                pattern_value TEXT NOT NULL,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                last_score REAL DEFAULT 0,
                avg_score REAL DEFAULT 0,
                is_trusted BOOLEAN DEFAULT FALSE,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(effect_type, technique, pattern_key)
            );

            CREATE TABLE IF NOT EXISTS hallucination_log (
                id INTEGER PRIMARY KEY,
                wrong_attribute TEXT NOT NULL,
                correct_attribute TEXT,
                bpy_type TEXT NOT NULL,
                occurrence_count INTEGER DEFAULT 1,
                last_seen TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(wrong_attribute, bpy_type)
            );

            CREATE TABLE IF NOT EXISTS run_history (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                effect_type TEXT NOT NULL,
                technique TEXT NOT NULL,
                iterations INTEGER NOT NULL,
                best_score REAL NOT NULL,
                passed BOOLEAN NOT NULL,
                total_cost_usd REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS camera_presets (
                id INTEGER PRIMARY KEY,
                effect_type TEXT NOT NULL,
                scene_radius_min REAL,
                scene_radius_max REAL,
                best_distance_multiplier REAL,
                best_elevation_deg REAL,
                best_azimuth_deg REAL,
                avg_score REAL,
                sample_count INTEGER DEFAULT 0,
                UNIQUE(effect_type)
            );
        """)
        self.db.commit()

    def record_signal(self, signal: LearningSignal):
        """Record a learning signal from a pipeline run."""
        # 1. Update patterns
        key = f"{signal.technique}_{signal.effect_type}"
        is_success = signal.quality_score >= 60.0

        self.db.execute("""
            INSERT INTO patterns (effect_type, technique, pattern_key, pattern_value,
                                  success_count, failure_count, last_score, avg_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(effect_type, technique, pattern_key) DO UPDATE SET
                success_count = success_count + ?,
                failure_count = failure_count + ?,
                last_score = ?,
                avg_score = (avg_score * (success_count + failure_count) + ?) / (success_count + failure_count + 1),
                is_trusted = (success_count + ?) >= ?,
                updated_at = CURRENT_TIMESTAMP
        """, (
            signal.effect_type, signal.technique, key, str(signal.camera_placement),
            int(is_success), int(not is_success), signal.quality_score, signal.quality_score,
            int(is_success), int(not is_success), signal.quality_score, signal.quality_score,
            int(is_success), self.TRUST_THRESHOLD,
        ))

        # 2. Log hallucinations
        for miss in signal.truth_pack_misses:
            self.db.execute("""
                INSERT INTO hallucination_log (wrong_attribute, bpy_type)
                VALUES (?, 'unknown')
                ON CONFLICT(wrong_attribute, bpy_type) DO UPDATE SET
                    occurrence_count = occurrence_count + 1,
                    last_seen = CURRENT_TIMESTAMP
            """, (miss,))

        # 3. Update camera presets (running average)
        if is_success and signal.camera_placement:
            cam = signal.camera_placement
            self.db.execute("""
                INSERT INTO camera_presets
                    (effect_type, best_distance_multiplier, best_elevation_deg,
                     best_azimuth_deg, avg_score, sample_count)
                VALUES (?, ?, ?, ?, ?, 1)
                ON CONFLICT(effect_type) DO UPDATE SET
                    best_distance_multiplier = (best_distance_multiplier * sample_count + ?) / (sample_count + 1),
                    best_elevation_deg = (best_elevation_deg * sample_count + ?) / (sample_count + 1),
                    avg_score = (avg_score * sample_count + ?) / (sample_count + 1),
                    sample_count = sample_count + 1
            """, (
                signal.effect_type,
                cam.get("distance_multiplier", 3.5),
                cam.get("elevation_deg", 25),
                cam.get("azimuth_deg", 45),
                signal.quality_score,
                cam.get("distance_multiplier", 3.5),
                cam.get("elevation_deg", 25),
                signal.quality_score,
            ))

        # 4. Record run history
        self.db.execute("""
            INSERT INTO run_history (session_id, effect_type, technique,
                                     iterations, best_score, passed, total_cost_usd)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            signal.session_id, signal.effect_type, signal.technique,
            signal.iteration + 1, signal.quality_score,
            is_success, 0.0,  # Cost tracked separately
        ))

        self.db.commit()

    def get_trusted_camera_preset(self, effect_type: str) -> dict | None:
        """Get camera preset if we have enough successful runs."""
        row = self.db.execute("""
            SELECT best_distance_multiplier, best_elevation_deg, best_azimuth_deg,
                   avg_score, sample_count
            FROM camera_presets
            WHERE effect_type = ? AND sample_count >= ?
        """, (effect_type, self.TRUST_THRESHOLD)).fetchone()

        if row:
            return {
                "distance_multiplier": row[0],
                "elevation_deg": row[1],
                "azimuth_deg": row[2],
                "avg_score": row[3],
                "sample_count": row[4],
            }
        return None

    def get_top_hallucinations(self, limit: int = 20) -> list[tuple[str, int]]:
        """Get most common hallucinated attributes — useful for reporting."""
        rows = self.db.execute("""
            SELECT wrong_attribute, occurrence_count
            FROM hallucination_log
            ORDER BY occurrence_count DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return rows

    def get_pass_rate(self, effect_type: str | None = None) -> float:
        """Get pass rate, optionally filtered by effect type."""
        if effect_type:
            row = self.db.execute("""
                SELECT COUNT(*) FILTER (WHERE passed), COUNT(*)
                FROM run_history WHERE effect_type = ?
            """, (effect_type,)).fetchone()
        else:
            row = self.db.execute("""
                SELECT COUNT(*) FILTER (WHERE passed), COUNT(*)
                FROM run_history
            """).fetchone()
        if row and row[1] > 0:
            return row[0] / row[1]
        return 0.0
```

### Evidence Gating

```
Signal → Record → Count → Trust

UNTRUSTED (count < 3):
  - Pattern is recorded but NOT injected into agent instructions
  - Used only for analytics and debugging

TRUSTED (count >= 3 successes):
  - Pattern IS injected into dynamic instructions
  - Camera presets override default placement
  - Light bounds are tightened based on successful values
  - Technique preferences are updated

DEPRECATED (success_rate < 20% after 10+ runs):
  - Pattern is flagged as unreliable
  - Actively excluded from agent instructions
```

### What We Do NOT Learn From

- **Single successes.** One lucky run doesn't mean the parameters are good.
- **LLM opinions.** The LLM's assessment of why something worked is not evidence.
- **Partial successes.** A score of 45 doesn't mean the parameters are "close."

---

## 6. SDK Patterns

### What We USE

| SDK Feature | Where | Why |
|-------------|-------|-----|
| **Agent with `output_type`** | Planner, Diagnoser | Structured output guarantees parseable results |
| **Dynamic instructions** | ScriptWriter | Inject truth pack + errors at runtime without rebuilding agent |
| **`RunContextWrapper`** | Everywhere | Share pipeline state across agents and tools |
| **`trace()` context manager** | Pipeline executor | Group all phases of one run into a single trace |
| **`max_turns`** | All agents | Prevent runaway token consumption |
| **`ModelSettings(temperature=...)`** | Per agent | Low temp for code gen, moderate for planning |
| **`RunConfig(model=...)`** | Budget management | Override model per-run when budget is tight |

### What We SKIP (and Why)

| SDK Feature | Why Skip |
|-------------|----------|
| **Handoffs** | We want code-controlled flow, not LLM-decided routing. The pipeline state machine decides what happens next. |
| **`agent.as_tool()`** | Our agents don't call each other. The pipeline executor calls each agent in sequence. This is simpler, cheaper, and more debuggable. |
| **MCP servers** | Tools run in-process. MCP adds latency and complexity for zero benefit in our use case. |
| **Input guardrails** | We validate inputs with Pydantic models, not LLM-based guardrails. Cheaper and more reliable. |
| **Output guardrails** | We validate outputs with the truth pack validator, not LLM-based guardrails. |
| **Session/history** | Each pipeline run is independent. We don't need multi-turn conversation history — the pipeline context carries all state. |
| **Streaming** | We don't display tokens to a user in real-time. Batch execution is fine. |
| **`tool_use_behavior`** | Our agents have simple tool sets. Default behavior is fine. |
| **Human-in-the-loop via `needs_approval`** | We implement HITL at the pipeline level, not the agent level (see section 7). |

### The Key Insight: Code-Driven, Not Agent-Driven

The previous system used agents to orchestrate agents. This created a chain of LLM calls where each one could go wrong. The new system uses **Python code** to orchestrate agents:

```python
# WRONG: Agent-driven orchestration
orchestrator = Agent(
    tools=[
        research_agent.as_tool(...),
        writer_agent.as_tool(...),
        executor_agent.as_tool(...),
        evaluator_agent.as_tool(...),
    ],
    instructions="Orchestrate the VFX pipeline...",
)
# LLM decides tool order, can skip steps, can loop forever

# RIGHT: Code-driven orchestration
async def pipeline(prompt):
    spec = await Runner.run(planner_agent, prompt)       # Step 1
    script = await Runner.run(writer_agent, spec)         # Step 2
    result = await run_blender(script)                    # Step 3 (no LLM)
    score = await evaluate(result)                        # Step 4 (no LLM)
    if score < 60:
        ...                                              # Deterministic routing
```

**Benefits:**
- Predictable execution order
- Deterministic error routing
- Debuggable with standard Python tools
- Each agent call has a clear input/output contract
- Budget is perfectly predictable per iteration

---

## 7. Human-in-the-Loop

### When Does the Human Intervene?

| Trigger | Mechanism | What the Human Sees |
|---------|-----------|---------------------|
| **First run of a new effect type** | Pipeline pauses after PLAN | Scene spec for approval |
| **Diagnoser confidence < 0.5** | Pipeline pauses after DIAGNOSE | Error + diagnosis + suggested actions |
| **3 consecutive iterations with no score improvement** | Pipeline pauses | Score history + last 3 quality evaluations |
| **Budget threshold (>$2 on a single run)** | Pipeline pauses before next iteration | Cost breakdown + quality trajectory |
| **Critical issue detected in evaluation** | Pipeline pauses | Render image + critical issue description |

### Implementation

HITL is at the **pipeline level**, not the agent level. This is simpler and more useful:

```python
from enum import Enum


class HumanDecision(str, Enum):
    CONTINUE = "continue"        # Keep going as-is
    REDIRECT = "redirect"        # Try different technique/approach
    APPROVE = "approve"          # Accept current result
    ABORT = "abort"              # Stop the run
    MANUAL_FIX = "manual_fix"    # Human will fix something


class HITLCheckpoint:
    """Human-in-the-loop checkpoint. Blocks pipeline until human responds."""

    def __init__(self, autonomy_level: int = 0):
        self.autonomy_level = autonomy_level
        self.callback = None  # Set this to an async function that presents choices to human

    async def check_plan(self, scene_spec: SceneSpec) -> HumanDecision:
        """Called after PLAN phase."""
        if self.autonomy_level >= 2:
            return HumanDecision.CONTINUE  # Semi-autonomous: skip plan approval

        # Present to human
        print(f"\n{'='*60}")
        print(f"SCENE PLAN:")
        print(f"  Effect: {scene_spec.effect_type}")
        print(f"  Technique: {scene_spec.technique}")
        print(f"  Elements: {scene_spec.scene_elements}")
        print(f"  Camera: {scene_spec.camera_intent}")
        print(f"  Duration: {scene_spec.duration_frames} frames")
        print(f"{'='*60}")

        if self.callback:
            return await self.callback("plan", scene_spec)

        # CLI fallback
        choice = input("Continue? [c]ontinue / [r]edirect / [a]bort: ").strip().lower()
        return {
            "c": HumanDecision.CONTINUE,
            "r": HumanDecision.REDIRECT,
            "a": HumanDecision.ABORT,
        }.get(choice, HumanDecision.CONTINUE)

    async def check_stall(
        self, quality_history: list[float], iteration: int
    ) -> HumanDecision:
        """Called when quality score isn't improving."""
        if self.autonomy_level >= 3:
            return HumanDecision.CONTINUE  # Fully autonomous for known types

        # Check if we're actually stalling
        if len(quality_history) < 3:
            return HumanDecision.CONTINUE

        recent = quality_history[-3:]
        if max(recent) - min(recent) < 5:  # Less than 5-point spread
            print(f"\n{'='*60}")
            print(f"STALL DETECTED at iteration {iteration}")
            print(f"Recent scores: {recent}")
            print(f"{'='*60}")

            if self.callback:
                return await self.callback("stall", quality_history)

            choice = input("[c]ontinue / [r]edirect technique / [a]bort: ").strip().lower()
            return {
                "c": HumanDecision.CONTINUE,
                "r": HumanDecision.REDIRECT,
                "a": HumanDecision.ABORT,
            }.get(choice, HumanDecision.ABORT)

        return HumanDecision.CONTINUE

    async def check_budget(self, cost_so_far: float, cost_per_iter: float) -> HumanDecision:
        """Called when cumulative cost exceeds threshold."""
        if cost_so_far + cost_per_iter > 2.0:  # $2 per run is a lot on $20/month
            print(f"\n{'='*60}")
            print(f"BUDGET WARNING: ${cost_so_far:.2f} spent, next iteration ~${cost_per_iter:.2f}")
            print(f"{'='*60}")

            if self.callback:
                return await self.callback("budget", cost_so_far)

            choice = input("[c]ontinue / [a]bort: ").strip().lower()
            return {
                "c": HumanDecision.CONTINUE,
                "a": HumanDecision.ABORT,
            }.get(choice, HumanDecision.ABORT)

        return HumanDecision.CONTINUE
```

### Autonomy Progression

From the mission statement, adapted to concrete implementation:

| Level | Gate | What Changes |
|-------|------|-------------|
| **0: Guided** | Default | Human approves plan, sees every evaluation |
| **1: Assisted** | 10+ successful runs for this effect type | Plan approval skipped, stall detection still active |
| **2: Semi-Autonomous** | Pass rate > 50% for 4+ effect types | Only budget and critical issue checks |
| **3: Autonomous (Known)** | Pass rate > 70% for this specific effect type, 50+ runs | Pipeline runs to completion, human sees only final result |

```python
def compute_autonomy_level(knowledge_store: KnowledgeStore, effect_type: str) -> int:
    """Compute autonomy level from evidence."""
    pass_rate = knowledge_store.get_pass_rate(effect_type)
    total_runs = knowledge_store.db.execute(
        "SELECT COUNT(*) FROM run_history WHERE effect_type = ?",
        (effect_type,)
    ).fetchone()[0]

    if total_runs >= 50 and pass_rate >= 0.7:
        return 3
    if pass_rate >= 0.5:
        return 2
    if total_runs >= 10 and pass_rate >= 0.3:
        return 1
    return 0
```

---

## 8. Budget Management

### Cost Model

| Operation | Model | Est. Cost | Per |
|-----------|-------|-----------|-----|
| Planner | gpt-5-mini | ~$0.03 | run |
| ScriptWriter | gpt-5.3-codex | ~$0.05 | iteration |
| Diagnoser | gpt-5-mini | ~$0.02 | error |
| ML Evaluation (CLIP+TOPIQ) | local | $0.00 | iteration |
| Truth Pack (Blender introspection) | local | $0.00 | run |
| Validation | local | $0.00 | iteration |

### Cost Per Run (Estimated)

| Scenario | Iterations | LLM Calls | Est. Cost |
|----------|-----------|-----------|-----------|
| First-try success | 1 | Plan + Generate | ~$0.08 |
| Typical (3 iterations) | 3 | Plan + 3×Generate + 1×Diagnose | ~$0.20 |
| Tough (5 iterations, 2 errors) | 5 | Plan + 5×Generate + 2×Diagnose | ~$0.33 |
| Worst case (5 iter, many errors) | 5 | Plan + 5×Generate + 5×Diagnose | ~$0.43 |

### Budget at $20/month

| Scenario | Runs/month |
|----------|-----------|
| Mostly first-try | ~250 |
| Typical mix | ~100 |
| Tough problems only | ~60 |
| Worst case everything | ~46 |

**This is a 4-10x improvement over the previous system** (estimated $2+/run with 9 agents).

### Model Routing

```python
def select_model(phase: Phase, budget_remaining: float, iteration: int) -> str:
    """
    Route to the cheapest model that can do the job.
    Degrade gracefully as budget decreases.
    """
    if budget_remaining < 2.0:
        # Emergency: use cheapest for everything
        return "gpt-5-mini"

    if budget_remaining < 5.0:
        # Tight: use mini for everything except generation
        if phase == Phase.GENERATE:
            return "gpt-5-mini"  # Downgrade from codex
        return "gpt-5-mini"

    # Normal budget
    MODEL_MAP = {
        Phase.PLAN: "gpt-5-mini",
        Phase.GENERATE: "gpt-5.3-codex",  # Best for code
        Phase.DIAGNOSE: "gpt-5-mini",
    }
    return MODEL_MAP.get(phase, "gpt-5-mini")


class BudgetTracker:
    """Track spending and enforce limits."""

    def __init__(self, monthly_limit: float = 20.0, db_path: str = "budget.db"):
        self.monthly_limit = monthly_limit
        self.db = sqlite3.connect(db_path)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS spending (
                id INTEGER PRIMARY KEY,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT,
                phase TEXT,
                model TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost_usd REAL
            )
        """)
        self.db.commit()

    def record(self, session_id: str, phase: str, model: str,
               input_tokens: int, output_tokens: int, cost_usd: float):
        self.db.execute(
            "INSERT INTO spending (session_id, phase, model, input_tokens, output_tokens, cost_usd) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, phase, model, input_tokens, output_tokens, cost_usd),
        )
        self.db.commit()

    def remaining_this_month(self) -> float:
        row = self.db.execute("""
            SELECT COALESCE(SUM(cost_usd), 0)
            FROM spending
            WHERE timestamp >= date('now', 'start of month')
        """).fetchone()
        return self.monthly_limit - row[0]

    def can_afford(self, estimated_cost: float) -> bool:
        return self.remaining_this_month() >= estimated_cost
```

---

## 9. Error Recovery

### Error Classification

```python
from enum import Enum


class ErrorSeverity(str, Enum):
    RECOVERABLE = "recoverable"    # Can fix and retry
    DEGRADED = "degraded"          # Can continue with reduced quality
    FATAL = "fatal"                # Must regenerate from scratch
    SYSTEMIC = "systemic"          # Infrastructure problem, not script problem


class ErrorCategory(str, Enum):
    # Script errors (RECOVERABLE)
    INVALID_ATTRIBUTE = "invalid_attribute"       # Hallucinated API
    MISSING_IMPORT = "missing_import"             # Forgot to import
    SYNTAX_ERROR = "syntax_error"                 # Bad Python syntax
    TYPE_ERROR = "type_error"                     # Wrong type for attribute

    # Scene errors (RECOVERABLE → DEGRADED)
    MISSING_OBJECT = "missing_object"             # Referenced object doesn't exist
    EMPTY_SCENE = "empty_scene"                   # No objects created
    PHYSICS_SETUP = "physics_setup"               # Bad domain/flow config

    # Execution errors (FATAL)
    BAKE_FAILURE = "bake_failure"                 # Fluid sim failed to bake
    OUT_OF_MEMORY = "out_of_memory"               # OOM during bake/render
    CRASH = "crash"                               # Blender segfault

    # Infrastructure errors (SYSTEMIC)
    BLENDER_NOT_FOUND = "blender_not_found"       # Binary missing
    PERMISSION_DENIED = "permission_denied"       # File system issue
    TIMEOUT = "timeout"                           # Execution exceeded limit

    # Quality errors (DEGRADED)
    BLACK_SCREEN = "black_screen"                 # Render is pure black
    WHITE_SCREEN = "white_screen"                 # Render is pure white
    NO_RENDER = "no_render"                       # No output file produced


# ─── Recovery strategies ──────────────────────────────────────────

RECOVERY_TABLE: dict[ErrorCategory, dict] = {
    ErrorCategory.INVALID_ATTRIBUTE: {
        "severity": ErrorSeverity.RECOVERABLE,
        "strategy": "substitute_from_truth_pack",
        "max_retries": 2,
        "needs_llm": False,
    },
    ErrorCategory.MISSING_IMPORT: {
        "severity": ErrorSeverity.RECOVERABLE,
        "strategy": "add_import",
        "max_retries": 1,
        "needs_llm": False,
    },
    ErrorCategory.SYNTAX_ERROR: {
        "severity": ErrorSeverity.FATAL,
        "strategy": "regenerate_script",
        "max_retries": 0,
        "needs_llm": True,
    },
    ErrorCategory.MISSING_OBJECT: {
        "severity": ErrorSeverity.RECOVERABLE,
        "strategy": "diagnose_and_fix",
        "max_retries": 1,
        "needs_llm": True,
    },
    ErrorCategory.BAKE_FAILURE: {
        "severity": ErrorSeverity.FATAL,
        "strategy": "regenerate_with_simpler_params",
        "max_retries": 0,
        "needs_llm": True,
    },
    ErrorCategory.OUT_OF_MEMORY: {
        "severity": ErrorSeverity.FATAL,
        "strategy": "reduce_resolution_and_regenerate",
        "max_retries": 0,
        "needs_llm": False,  # Just halve resolution_max
    },
    ErrorCategory.CRASH: {
        "severity": ErrorSeverity.FATAL,
        "strategy": "try_different_technique",
        "max_retries": 0,
        "needs_llm": True,
    },
    ErrorCategory.BLENDER_NOT_FOUND: {
        "severity": ErrorSeverity.SYSTEMIC,
        "strategy": "halt_and_report",
        "max_retries": 0,
        "needs_llm": False,
    },
    ErrorCategory.BLACK_SCREEN: {
        "severity": ErrorSeverity.DEGRADED,
        "strategy": "add_lighting",
        "max_retries": 1,
        "needs_llm": False,  # Deterministic lighting injection
    },
}
```

### Circuit Breakers

```python
class CircuitBreaker:
    """
    Prevent the system from burning budget on hopeless runs.
    Three levels: per-iteration, per-run, per-session.
    """

    def __init__(self):
        self.iteration_errors: list[str] = []
        self.run_errors: list[str] = []

    def record_error(self, category: ErrorCategory):
        self.iteration_errors.append(category.value)
        self.run_errors.append(category.value)

    def should_stop_iteration(self) -> tuple[bool, str]:
        """Should we stop retrying within this iteration?"""
        # Same error 3 times → give up on this approach
        if len(self.iteration_errors) >= 3:
            from collections import Counter
            counts = Counter(self.iteration_errors)
            for error, count in counts.items():
                if count >= 3:
                    return True, f"Same error '{error}' occurred 3 times"
        return False, ""

    def should_stop_run(self) -> tuple[bool, str]:
        """Should we stop the entire run?"""
        # 10+ errors total → something is fundamentally broken
        if len(self.run_errors) >= 10:
            return True, f"10+ errors in run: {self.run_errors[-5:]}"

        # 5+ FATAL errors → infrastructure or technique problem
        fatal_count = sum(
            1 for e in self.run_errors
            if RECOVERY_TABLE.get(ErrorCategory(e), {}).get("severity") == ErrorSeverity.FATAL
        )
        if fatal_count >= 5:
            return True, f"{fatal_count} fatal errors — technique may be wrong"

        return False, ""

    def reset_iteration(self):
        self.iteration_errors.clear()
```

---

## 10. What We Explicitly Do NOT Build

### 1. LLM-Based API Validation

**Why not:** The truth pack makes this unnecessary. Blender itself tells us what's valid. An LLM guessing whether `resolution_divisions` is correct is strictly worse than checking `bl_rna.properties`.

### 2. LLM-Based Quality Evaluation

**Why not:** ML models (CLIP, TOPIQ, LPIPS) are cheaper, faster, more consistent, and don't hallucinate. An LLM looking at a render image and saying "7/10" provides no actionable signal. ML metrics like "TOPIQ structural quality: 0.42" do.

### 3. A "Research Agent" That Searches Documentation

**Why not:** The truth pack IS the documentation, extracted at runtime from the actual Blender binary. Searching docs with an LLM is an expensive way to get outdated information that may not match the installed Blender version.

### 4. Template System / Parameter Schemas

**Why not (for now):** Templates are an optimization for known patterns. We don't have known patterns yet — we have 0 passes. Build the adaptive system first. If it works, extract templates from successful runs AUTOMATICALLY via the knowledge store.

### 5. Escape Velocity / Stuck Detection Logic in Agents

**Why not:** The pipeline executor handles iteration logic, stall detection, and technique switching. Putting this in agent instructions creates a second, conflicting control loop. One state machine, not two.

### 6. MCP Server Wrapper

**Why not:** The orchestrator runs as a Python program, not a server. MCP adds latency, complexity, and a serialization layer for zero benefit. If we need to expose the pipeline to an external system later, wrap it in a simple HTTP API.

### 7. Vector Store for Blender Documentation

**Why not:** `bl_rna.properties` is a better source of truth than any document. Runtime introspection gives you the correct answer for YOUR Blender version in <5 seconds. A vector store gives you maybe-correct answers from maybe-your-version in ~2 seconds plus embedding cost.

### 8. Parallel Agent Execution Within a Phase

**Why not (yet):** Our pipeline is sequential by design: plan → generate → validate → execute → evaluate. There's nothing to parallelize within a phase. Between runs, yes — you could run multiple prompts in parallel. But that's a scheduling concern, not an architecture concern.

### 9. Fine-Tuning or LoRA

**Why not:** We don't have enough successful data. With 0 passes, fine-tuning would teach the model to fail better. Get to 50% pass rate first, then consider whether fine-tuning is worth the investment.

### 10. Blender Add-on / Custom Nodes

**Why not:** This would lock us into a specific Blender UI integration. The headless Python script approach is more flexible, testable, and automatable. Custom nodes can come later as a convenience layer.

---

## 11. Implementation Order

### Milestone 0: Truth Pack (2-3 days)

**Goal:** Prove that runtime introspection works and eliminates attribute hallucination.

**Deliverables:**
- `truth_pack.py` — Build truth pack from headless Blender
- `validator.py` — Validate a script against a truth pack
- Test: Generate a known-bad script (with `resolution_divisions`), validate it, confirm detection
- Test: Introspect `FluidDomainSettings`, confirm `resolution_max` is present

**Testable milestone:** `validate_script_against_truth_pack(bad_script, truth_pack)` returns errors for every hallucinated attribute.

```bash
# M0 test
python -c "
import asyncio
from truth_pack import build_truth_pack
from validator import validate_script_against_truth_pack

async def test():
    tp = await build_truth_pack('mantaflow_gas')
    assert 'FluidDomainSettings' in tp
    assert 'resolution_max' in tp['FluidDomainSettings']['properties']
    assert 'resolution_divisions' not in tp['FluidDomainSettings']['properties']

    bad_script = 'domain.resolution_divisions = 128'
    errors = validate_script_against_truth_pack(bad_script, tp)
    assert len(errors) > 0
    print('M0 PASSED')

asyncio.run(test())
"
```

### Milestone 1: Planner + ScriptWriter (3-4 days)

**Goal:** Generate a Blender script that passes validation against the truth pack.

**Deliverables:**
- `agents.py` — Planner and ScriptWriter agent definitions
- `pipeline.py` — Phase 0 (introspect) + Phase 1 (plan) + Phase 2 (generate) + Phase 3 (validate)
- `auto_fix.py` — Deterministic substitution from truth pack

**Testable milestone:** Given "a candle with a flickering flame", the pipeline produces a script with 0 truth-pack validation errors.

```bash
# M1 test
python -c "
import asyncio
from pipeline import VFXPipeline, PipelineContext

async def test():
    ctx = PipelineContext(session_id='m1_test')
    pipe = VFXPipeline(ctx)

    # Run through PLAN + GENERATE + VALIDATE only
    technique = pipe._guess_technique('a candle with a flickering flame')
    truth_pack = await build_truth_pack(technique)
    ctx.truth_pack = truth_pack

    spec = await pipe._plan('a candle with a flickering flame')
    script = await pipe._generate()
    validation = pipe._validate(script)

    assert validation.is_valid, f'Validation errors: {validation.errors}'
    print(f'M1 PASSED — script has {len(script.split(chr(10)))} lines, 0 errors')

asyncio.run(test())
"
```

### Milestone 2: Camera + Lighting (2 days)

**Goal:** Deterministic camera placement and lighting that produce visible renders.

**Deliverables:**
- Camera placeholder replacement with bounding-box-based placement
- Lighting placeholder replacement with bounded energy values
- Collision effector injection for liquid scenes

**Testable milestone:** Run the full pipeline on "a candle with a flickering flame" — the render is not black, not white, and the subject is visible (center of frame).

```bash
# M2 test — visual inspection required
python test_camera_lighting.py --prompt "a candle with a flickering flame"
# Output: /tmp/vfx_output/render_0001.png
# Human checks: is the candle visible? Is it lit? Is it centered?
```

### Milestone 3: Execution + Error Recovery (3-4 days)

**Goal:** Full pipeline through EXECUTE with error recovery.

**Deliverables:**
- `executor.py` — Blender subprocess execution
- Diagnoser agent integration
- Error classification and deterministic recovery
- Circuit breakers

**Testable milestone:** Run pipeline end-to-end on 3 prompts. At least 1 produces a render. Errors are classified correctly.

### Milestone 4: Quality Evaluation (2-3 days)

**Goal:** ML-based quality scoring that correctly identifies critical issues.

**Deliverables:**
- `evaluator.py` — CLIP + TOPIQ scoring (no LLM)
- Critical issue detection (black screen, white screen, no subject)
- Score history tracking

**Testable milestone:** Score a known-good render (>60) and a known-bad render (<30). BLACK_SCREEN detection works on a pure-black image.

### Milestone 5: Iteration Loop (2-3 days)

**Goal:** The system iterates and improves.

**Deliverables:**
- Full iteration loop with quality feedback
- Budget tracking
- HITL checkpoints

**Testable milestone:** Run "a candle with a flickering flame" for 5 iterations. Score improves from iteration 1 to iteration 5. System respects `max_iterations`.

### Milestone 6: Knowledge Store + Self-Learning (3-4 days)

**Goal:** The system remembers what works.

**Deliverables:**
- `knowledge.py` — SQLite knowledge store
- Learning signal recording after every run
- Camera preset storage and retrieval
- Evidence-gated injection into dynamic instructions

**Testable milestone:** Run "candle" 5 times. On run 6, camera placement uses learned presets (if any passed). Hallucination log shows the top 5 most-hallucinated attributes.

### Milestone 7: Multi-Technique Support (4-5 days)

**Goal:** System handles more than just Mantaflow gas.

**Deliverables:**
- Technique type maps for rigid_body, particle_system
- Truth pack building for new physics types
- Planner correctly routes "a window getting smashed" to rigid_body

**Testable milestone:** "a glass shattering" uses rigid_body, not mantaflow. "sparks from a grinder" uses particle_system.

### Total Estimated Timeline: 20-26 days

### What This Gets You

At the end of M7, you have a system that:
1. Never halluccinates Blender 4.x attributes (truth pack prevents it)
2. Always places the camera where the subject is visible (deterministic computation)
3. Routes different prompts to different techniques (not always Mantaflow)
4. Recovers from errors deterministically where possible
5. Learns from successful runs
6. Costs ~$0.20/run instead of ~$2+/run
7. Has clear HITL checkpoints that fade as reliability improves
8. Can be debugged with standard Python tools (no "why did the orchestrator agent decide to skip validation?")

---

## Appendix A: File Structure

```
agents/blender-vfx-orchestrator/
├── pipeline.py              # VFXPipeline — the state machine
├── agents.py                # 3 agent definitions (Planner, ScriptWriter, Diagnoser)
├── truth_pack.py            # Blender runtime introspection
├── validator.py             # Script validation against truth pack
├── auto_fix.py              # Deterministic attribute substitution
├── camera.py                # Deterministic camera placement
├── lighting.py              # Bounded lighting with per-technique defaults
├── effectors.py             # Collision effector injection
├── executor.py              # Blender subprocess execution
├── evaluator.py             # ML-only quality evaluation
├── diagnoser.py             # Error classification (uses Diagnoser agent)
├── knowledge.py             # SQLite knowledge store
├── budget.py                # Cost tracking and model routing
├── hitl.py                  # Human-in-the-loop checkpoints
├── circuit_breaker.py       # Error counting and run-stop logic
├── models.py                # Pydantic models for phase contracts
├── introspection_script.py  # Runs inside Blender to build truth pack
└── tests/
    ├── test_truth_pack.py
    ├── test_validator.py
    ├── test_camera.py
    ├── test_pipeline_m0.py
    ├── test_pipeline_m1.py
    └── test_pipeline_e2e.py
```

## Appendix B: Why This Design Addresses Each Failure Mode

| Failure Mode | Previous System | This Design |
|-------------|----------------|-------------|
| **Hallucinated attributes** | 57-rule regex fixer (reactive) | Truth pack introspection (preventive) + validation (detective) + auto-fix (corrective) |
| **Camera inside objects** | LLM guessed coordinates | Deterministic bounding-box computation. LLM only specifies intent ("medium shot"), code computes position. |
| **Mantaflow-only** | LLM defaulted to training data | Planner explicitly selects technique. Keyword router provides fast initial guess. Technique types are first-class. |
| **$2+/run** | 9 agents, most unnecessary | 3 agents + deterministic functions. ~$0.20/run. |
| **No learning** | Complex learning agent (LLM-based) | SQLite knowledge store. Evidence-gated. No LLM needed. |
| **Silent failures** | Exit codes overridden, "partial success" | Strict phase contracts. No success without proof. |

## Appendix C: Cost Comparison

| Component | Previous (9 agents) | This Design (3 agents) |
|-----------|---------------------|------------------------|
| Planning/Research | ~$0.50 (2 agents) | ~$0.03 (1 agent, mini) |
| Script Generation | ~$0.30 (1 agent) | ~$0.05 (1 agent, codex) |
| API Validation | ~$0.30 (1 LLM agent) | $0.00 (truth pack function) |
| Quality Evaluation | ~$0.40 (1 LLM agent) | $0.00 (ML models only) |
| Error Diagnosis | ~$0.20 (1 agent) | ~$0.02 (1 agent, mini, only on error) |
| Orchestration | ~$0.30 (coordinator agents) | $0.00 (Python state machine) |
| Learning | ~$0.20 (1 agent) | $0.00 (SQLite functions) |
| **Total per iteration** | **~$2.20** | **~$0.10** |
| **Total per run (3 iter)** | **~$6.60** | **~$0.20** |
| **Runs per $20/month** | **~3** | **~100** |
