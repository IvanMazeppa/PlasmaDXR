# CODEX Architecture Proposal

**Date:** 2026-02-19
**Status:** Draft v1.0
**Authors:** Research team (4 agents) + synthesis
**Mission:** [MISSION_STATEMENT_DRAFT.md](./MISSION_STATEMENT_DRAFT.md)

---

## Preamble: Why Start Over

The previous system ran **188 times** and scored **0 passes** (threshold: 60/100). Best scores: 58, 18, 0.

Three root causes:

1. **LLM hallucination of Blender API calls** — LLMs consistently generate Blender 4.x attribute names (`resolution_divisions` instead of `resolution_max`). A 57-rule regex fixer could not keep pace with the long tail of hallucinations.
2. **Catastrophic camera placement** — In 2 of 3 benchmark scenarios, the camera was placed inside objects or so close the subject was invisible. Score: 0.
3. **Technique monoculture** — The system used Mantaflow gas/liquid for every prompt regardless of what was actually requested (rigid body, particles, geometry nodes).

Additionally, the 10-agent architecture suffered from **compounding error** — 10 agents at 95% individual accuracy yields ~60% system reliability (0.95^10 = 0.60). Research from UC Berkeley/NeurIPS 2025 (MAST) found 41-86.7% failure rates across 7 state-of-the-art multi-agent systems in 1,642 execution traces.

This proposal designs a replacement from scratch.

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

Ranked by priority. When two principles conflict, the higher-ranked one wins.

### P1: Code Controls the Loop, LLMs Make Creative Decisions

The pipeline is a **Python `while` loop**, not an LLM-driven orchestrator. Python decides when to generate, validate, execute, evaluate, and retry. LLMs decide *what* code to write, *how good* a render looks, and *what* to fix.

**Rationale:** The previous system let an LLM orchestrator decide flow control. This made the pipeline non-deterministic, hard to debug, and prone to skipping steps or looping forever. OpenAI's own practical guide says: "Orchestrating via code makes tasks more deterministic and predictable, in terms of speed, cost and performance."

### P2: Deterministic Fixers Over Prompt Instructions

When a failure can be prevented or corrected deterministically (regex, AST rewrite, geometric calculation), use code — not a prompt instruction hoping the LLM will comply.

**Rationale:** "Place the camera at an appropriate distance" in a prompt works ~40% of the time. `camera.location = compute_from_scene_bounds(bbox, fov)` works 100% of the time. The kitchen_leak success (58/100) validated this: deterministic light energy scaling + API fixing produced the only non-zero score.

### P3: Fewer Agents, More Function Tools

Every agent adds latency, cost, and a compounding error probability. A component should be an agent only if it requires **creative judgment**. Everything else is a `@function_tool` or plain Python.

**Rationale:** 0.95^3 = 0.857 (3 agents). 0.95^10 = 0.599 (10 agents). Cutting from 10 to 3 agents improves theoretical system reliability by 43%.

### P4: Validate Before Execute, Always

No Blender script runs without passing AST-based API validation against the truth pack. No render is accepted without camera validation. No iteration starts without budget check.

**Rationale:** Executing an invalid script wastes 60-300 seconds of Blender bake time plus the cost of the evaluation call that follows. Pre-execution validation is ~10ms.

### P5: Every Run Produces Learning Signal

Failed runs are more valuable than successful ones — they reveal failure modes the system must learn to avoid. The knowledge store captures outcomes at the parameter, technique, approach, and failure levels. Knowledge is promoted by evidence, not by assumption.

### P6: Flexibility With Guardrails

The system should handle any Blender physics system (not just Mantaflow), but creative freedom is bounded by deterministic safety nets: API allowlists catch hallucinated attributes, camera placement is computed from geometry, light energy has per-effect-type bounds.

### P7: Earn Autonomy Through Demonstrated Reliability

The system starts fully supervised (human approves technique selection). As it accumulates evidence of success, human gates are removed. Autonomy is gated by evidence, not by time.

---

## 2. Agent Architecture

### 2.1 Who Exists and Why

**3 agents + Python orchestration + function tools.**

| Agent | Model | Purpose | Why an Agent? |
|-------|-------|---------|---------------|
| **Code Writer** | `gpt-5.2` | Generate/modify Blender Python scripts | Requires creative code generation — the hardest task |
| **Quality Judge** | `gpt-5-mini` | Evaluate render quality from images | Requires vision + subjective quality assessment |
| **Fix Strategist** | `gpt-5-mini` | Diagnose failures and prescribe fixes | Requires reasoning across script + errors + quality |

**Why exactly 3:**
- **Code Writer** must be the smartest model. Generating correct Blender 5.0 Python code with proper physics setup, materials, camera, and lighting is the #1 bottleneck. This justifies `gpt-5.2`.
- **Quality Judge** needs vision but the task is simpler (look at image, score it, list issues). `gpt-5-mini` is sufficient and 7x cheaper on output tokens.
- **Fix Strategist** bridges the gap — reads quality feedback + script + errors, produces actionable fix instructions. This is structured reasoning, well-suited for `gpt-5-mini`.

### 2.2 Agent Definitions

```python
from agents import Agent
from pydantic import BaseModel, Field


# ─── OUTPUT SCHEMAS ───────────────────────────────────────

class GeneratedScript(BaseModel):
    """Output from Code Writer."""
    script_content: str = Field(description="Complete Blender Python script")
    technique_used: str = Field(description="e.g., 'mantaflow_gas', 'rigid_body', 'geometry_nodes'")
    setup_notes: str = Field(description="What this script sets up and any caveats")


class QualityAssessment(BaseModel):
    """Output from Quality Judge."""
    overall_score: int = Field(ge=0, le=100, description="Quality score 0-100")
    issues: list[str] = Field(description="Specific quality issues found")
    critical_failures: list[str] = Field(description="AUTO-FAIL: BLACK_SCREEN, ZERO_LIGHTS, CAMERA_INSIDE, etc.")
    suggestions: list[str] = Field(description="Actionable improvement suggestions")
    primary_issue: str = Field(description="The single most impactful issue to fix")


class FixStrategy(BaseModel):
    """Output from Fix Strategist."""
    diagnosis: str = Field(description="Root cause of the quality issue")
    fix_instructions: str = Field(description="Specific instructions for the Code Writer")
    should_regenerate: bool = Field(description="True = new script with different technique")
    confidence: float = Field(ge=0.0, le=1.0, description="How confident in this fix strategy")


# ─── AGENT DEFINITIONS ───────────────────────────────────

code_writer = Agent(
    name="Code Writer",
    model="gpt-5.2",
    instructions="""You generate Blender 5.0 Python scripts for VFX effects.

You receive: effect description, technique specification, API reference snippets,
and (on modification) the previous script + fix instructions.

You output: a complete, executable Blender Python script.

RULES:
- Use ONLY the Blender 5.0 API attributes provided in the reference. If an attribute
  is not in the reference, DO NOT USE IT.
- Include complete scene setup: domain, emitters, camera, lights, materials, physics.
- Set render output to the path specified in the prompt.
- Handle headless execution (no bpy.ops.wm.open_mainfile, no GUI calls).
- The script MUST call bpy.ops.fluid.bake_all() for fluid simulations.
- The script MUST set use_plane_init = True for ALL liquid flows.

{dynamic_instructions}""",
    output_type=GeneratedScript,
)

quality_judge = Agent(
    name="Quality Judge",
    model="gpt-5-mini",
    instructions="""You evaluate rendered VFX images for visual quality.

You receive: a rendered image and the effect type requested.

SCORING RUBRIC (0-100):
- 80-100: Professional quality, clear subject, good lighting, realistic physics
- 60-79: Acceptable quality, minor issues, subject clearly visible
- 40-59: Mediocre, significant issues but effect is recognizable
- 20-39: Poor, major issues, effect barely visible or wrong
- 0-19: Failed, black screen / white screen / camera inside object / no effect visible

CRITICAL FAILURES (auto-fail regardless of other quality):
- BLACK_SCREEN: Image is entirely or mostly black
- WHITE_SCREEN: Image is entirely or mostly white/overexposed
- ZERO_LIGHTS: No lighting visible in the scene
- CAMERA_INSIDE: View suggests camera is inside geometry (circular dark edges)
- NO_EFFECT: The requested VFX effect is not visible
- CLIPPING: Volume/geometry clipping at domain boundaries

Be BRUTALLY HONEST. A score of 0 is correct for a black screen. Do not soften scores.""",
    output_type=QualityAssessment,
)

fix_strategist = Agent(
    name="Fix Strategist",
    model="gpt-5-mini",
    instructions="""You diagnose why a Blender VFX render scored poorly and prescribe fixes.

You receive: the quality assessment, the Blender script, and optionally the render image.

Your job:
1. Identify the ROOT CAUSE of the primary issue (not symptoms)
2. Prescribe SPECIFIC fix instructions for the Code Writer
3. Decide: should we MODIFY this script or REGENERATE from scratch?

REGENERATE when:
- Camera is inside an object (structural problem, not a tweak)
- Wrong physics system entirely (e.g., used gas for a liquid pour)
- Score < 10 (script is fundamentally broken)
- Same issue has occurred 2+ times (modification isn't working)

MODIFY when:
- Lighting issues (adjust energy values)
- Material issues (tweak shader parameters)
- Resolution/quality issues (increase simulation resolution)
- Minor physics issues (adjust force fields, timing)""",
    output_type=FixStrategy,
)
```

### 2.3 What is NOT an Agent

| Component | Implementation | Why Not an Agent |
|-----------|---------------|------------------|
| Pipeline orchestration | Python `while` loop | Deterministic flow control. LLM would skip steps or loop. |
| Blender execution | `@function_tool` subprocess call | No judgment needed. Run script, capture output. |
| API validation | Pure Python (AST + truth pack) | Deterministic correctness checking. LLM would hallucinate. |
| API fixing | Pure Python (regex + fuzzy match) | Deterministic find-and-replace. Must be exact. |
| Camera placement | Pure Python (geometry math) | Scene bounds → FOV → distance. Pure math. |
| Blender doc search | `@function_tool` vector store query | Database lookup. No reasoning needed. |
| Quality metrics (LPIPS/CLIP) | `@function_tool` computation | Pure math on images. |
| Budget tracking | `@input_guardrail` | Simple arithmetic. |
| Knowledge store | Plain Python + SQLite | Database CRUD. |
| Session persistence | Plain Python + JSON files | Serialization. |
| Plateau detection | Plain Python | Score trend analysis. |

**The rule:** If it can be expressed as `if/else`, `regex`, `math`, or `SQL`, it is NOT an agent. Agents are exclusively for tasks requiring creative judgment.

---

## 3. Anti-Hallucination Strategy

### 3.1 The Layered Defense

```
Layer 0: PREVENTION  — Truth pack snippets in Code Writer's prompt
Layer 1: DETECTION   — AST parse + truth pack validation (pre-execution)
Layer 2: CORRECTION  — Known renames + fuzzy matching (auto-fix)
Layer 3: CAMERA FIX  — Deterministic placement from scene bounds (injected snippet)
Layer 4: RUNTIME     — hasattr() safety net in Blender (last resort)
```

### 3.2 API Truth Pack — Machine-Generated Ground Truth

Instead of manually maintaining rules, **ask Blender itself** what attributes exist. Blender's RNA reflection system exposes complete metadata for every property.

```python
#!/usr/bin/env python3
"""truth_pack_generator.py — Run inside Blender to extract ground-truth API data.
Usage: blender --background --python truth_pack_generator.py -- output.json
"""
import bpy
import json
import sys
from pathlib import Path


def extract_rna_properties(bpy_type_name: str) -> dict:
    """Extract all RNA properties from a bpy.types class with full metadata."""
    bpy_type = getattr(bpy.types, bpy_type_name, None)
    if bpy_type is None or not hasattr(bpy_type, "bl_rna"):
        return {}

    properties = {}
    for prop in bpy_type.bl_rna.properties:
        if prop.identifier == "rna_type":
            continue

        entry = {
            "identifier": prop.identifier,
            "type": prop.type,
            "description": prop.description,
            "is_readonly": prop.is_readonly,
        }

        if prop.type in ("FLOAT", "INT"):
            entry["hard_min"] = prop.hard_min
            entry["hard_max"] = prop.hard_max
            entry["soft_min"] = prop.soft_min
            entry["soft_max"] = prop.soft_max
            entry["default"] = prop.default

        elif prop.type == "BOOLEAN":
            entry["default"] = prop.default

        elif prop.type == "ENUM":
            entry["enum_items"] = [
                {"identifier": item.identifier, "name": item.name}
                for item in prop.enum_items
            ]
            entry["default"] = prop.default

        properties[prop.identifier] = entry

    return properties


def extract_shader_node_sockets(node_type_name: str) -> dict:
    """Extract input/output socket names for a shader node type."""
    mat = bpy.data.materials.new(name="__truth_temp__")
    mat.use_nodes = True
    tree = mat.node_tree

    try:
        node = tree.nodes.new(type=node_type_name)
        result = {
            "inputs": {s.name: s.type for s in node.inputs},
            "outputs": {s.name: s.type for s in node.outputs},
        }
        tree.nodes.remove(node)
        return result
    except Exception as e:
        return {"error": str(e)}
    finally:
        bpy.data.materials.remove(mat)


def build_truth_pack() -> dict:
    """Build the complete API truth pack."""
    truth = {
        "blender_version": list(bpy.app.version),
        "blender_version_string": bpy.app.version_string,
    }

    # Physics types
    physics_types = [
        "FluidDomainSettings", "FluidFlowSettings", "FluidEffectorSettings",
        "RigidBodyObject", "RigidBodyConstraint",
        "ParticleSettings", "ParticleSystem",
        "ClothSettings", "SoftBodySettings",
    ]
    truth["physics"] = {}
    for t in physics_types:
        props = extract_rna_properties(t)
        if props:
            truth["physics"][t] = props

    # Render types
    for t in ["RenderSettings", "SceneEEVEE", "CyclesRenderSettings"]:
        props = extract_rna_properties(t)
        if props:
            truth.setdefault("render", {})[t] = props

    # Shader node types (valid names)
    truth["shader_node_types"] = [
        name for name in dir(bpy.types)
        if name.startswith("ShaderNode") and hasattr(getattr(bpy.types, name), "bl_rna")
    ]

    # Critical shader node sockets
    truth["shader_nodes"] = {}
    for node_type in [
        "ShaderNodeBsdfPrincipled", "ShaderNodeMix", "ShaderNodeEmission",
        "ShaderNodeVolumePrincipled", "ShaderNodeVolumeAbsorption",
        "ShaderNodeSeparateColor", "ShaderNodeCombineColor",
    ]:
        sockets = extract_shader_node_sockets(node_type)
        if "error" not in sockets:
            truth["shader_nodes"][node_type] = sockets

    return truth


if __name__ == "__main__":
    truth = build_truth_pack()
    out = Path(sys.argv[sys.argv.index("--") + 1]) if "--" in sys.argv else Path("truth_pack.json")
    out.write_text(json.dumps(truth, indent=2, default=str))
    print(f"Truth pack → {out} ({len(truth['physics'])} physics types, "
          f"{len(truth.get('shader_node_types', []))} shader nodes)")
```

Run once on setup, regenerate on Blender version updates:
```bash
blender --background --python truth_pack_generator.py -- truth_pack.json
```

### 3.3 Truth Pack Loader

```python
"""truth_pack.py — Load and query the Blender API truth pack."""
import json
from difflib import get_close_matches
from pathlib import Path


class TruthPack:
    def __init__(self, path: str = "truth_pack.json"):
        with open(path) as f:
            self._data = json.load(f)

        # Build fast lookup indices
        self._attrs: dict[str, set[str]] = {}
        for category in ("physics", "render"):
            for type_name, props in self._data.get(category, {}).items():
                self._attrs[type_name] = set(props.keys())

        self._shader_types: set[str] = set(self._data.get("shader_node_types", []))

        self._sockets: dict[str, dict[str, set[str]]] = {}
        for node_type, sockets in self._data.get("shader_nodes", {}).items():
            self._sockets[node_type] = {
                "inputs": set(sockets.get("inputs", {}).keys()),
                "outputs": set(sockets.get("outputs", {}).keys()),
            }

    def is_valid_attr(self, type_name: str, attr: str) -> bool:
        return attr in self._attrs.get(type_name, set())

    def is_valid_shader_node(self, node_type: str) -> bool:
        return node_type in self._shader_types

    def suggest_attr(self, wrong: str, type_name: str) -> str | None:
        valid = list(self._attrs.get(type_name, set()))
        matches = get_close_matches(wrong, valid, n=1, cutoff=0.5)
        return matches[0] if matches else None

    def suggest_shader_node(self, wrong: str) -> str | None:
        matches = get_close_matches(wrong, list(self._shader_types), n=1, cutoff=0.6)
        return matches[0] if matches else None

    def get_attr_range(self, type_name: str, attr: str) -> dict | None:
        for cat in ("physics", "render"):
            types = self._data.get(cat, {})
            if type_name in types and attr in types[type_name]:
                prop = types[type_name][attr]
                if prop["type"] in ("FLOAT", "INT"):
                    return {k: prop[k] for k in ("hard_min", "hard_max", "soft_min", "soft_max", "default")}
        return None

    def get_valid_enum_values(self, type_name: str, attr: str) -> list[str]:
        for cat in ("physics", "render"):
            types = self._data.get(cat, {})
            if type_name in types and attr in types[type_name]:
                prop = types[type_name][attr]
                if prop["type"] == "ENUM":
                    return [item["identifier"] for item in prop.get("enum_items", [])]
        return []

    def snippet_for_type(self, type_name: str) -> str:
        """Generate a prompt snippet listing valid attributes for a type."""
        attrs = self._attrs.get(type_name, set())
        if not attrs:
            return f"# No known attributes for {type_name}"
        lines = [f"### {type_name} — Valid Attributes"]
        for attr in sorted(attrs):
            rng = self.get_attr_range(type_name, attr)
            if rng:
                lines.append(f"- `{attr}` ({rng['hard_min']}..{rng['hard_max']}, default={rng['default']})")
            else:
                lines.append(f"- `{attr}`")
        return "\n".join(lines)
```

### 3.4 AST-Based Script Validator

```python
"""script_validator.py — AST-based validation of Blender scripts against truth pack."""
import ast
from dataclasses import dataclass
from typing import Optional

from truth_pack import TruthPack


@dataclass
class ValidationIssue:
    line: int
    category: str       # unknown_attr, wrong_node_type, wrong_socket, out_of_range, wrong_enum
    severity: str       # error, warning
    message: str
    suggested_fix: Optional[str] = None


# Variable name → likely Blender type (heuristic)
TYPE_HINTS = {
    "domain": "FluidDomainSettings",
    "domain_settings": "FluidDomainSettings",
    "fluid_domain": "FluidDomainSettings",
    "flow": "FluidFlowSettings",
    "flow_settings": "FluidFlowSettings",
    "effector": "FluidEffectorSettings",
    "effector_settings": "FluidEffectorSettings",
    "render": "RenderSettings",
}


class BlenderScriptValidator(ast.NodeVisitor):
    """Validates Blender API usage in generated scripts."""

    def __init__(self, truth: TruthPack):
        self.truth = truth
        self.issues: list[ValidationIssue] = []

    def validate(self, source: str) -> list[ValidationIssue]:
        self.issues = []
        try:
            tree = ast.parse(source)
            self.visit(tree)
        except SyntaxError as e:
            self.issues.append(ValidationIssue(
                line=e.lineno or 0, category="syntax_error", severity="error",
                message=f"Syntax error: {e.msg}",
            ))
        return self.issues

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                type_name = TYPE_HINTS.get(target.value.id)
                if type_name:
                    self._check_attr(node, type_name, target.attr, node.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Check nodes.new(type='ShaderNodeXxx') calls
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "new"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "nodes"):
            for kw in node.keywords:
                if kw.arg == "type" and isinstance(kw.value, ast.Constant):
                    self._check_shader_node(node, kw.value.value)
            if node.args and isinstance(node.args[0], ast.Constant):
                self._check_shader_node(node, node.args[0].value)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        # Check node.inputs["SocketName"] patterns
        if (isinstance(node.value, ast.Attribute)
                and node.value.attr in ("inputs", "outputs")
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
            direction = node.value.attr
            socket = node.slice.value
            all_sockets = set()
            for nt_sockets in self.truth._sockets.values():
                all_sockets.update(nt_sockets.get(direction, set()))
            if all_sockets and socket not in all_sockets:
                suggestion = get_close_matches(socket, list(all_sockets), n=1, cutoff=0.4)
                self.issues.append(ValidationIssue(
                    line=node.lineno, category="wrong_socket", severity="warning",
                    message=f"Socket '{socket}' not found in known {direction}",
                    suggested_fix=f"Did you mean '{suggestion[0]}'?" if suggestion else None,
                ))
        self.generic_visit(node)

    def _check_attr(self, node, type_name: str, attr: str, value_node):
        valid = self.truth._attrs.get(type_name, set())
        if valid and attr not in valid:
            suggestion = self.truth.suggest_attr(attr, type_name)
            self.issues.append(ValidationIssue(
                line=node.lineno, category="unknown_attr", severity="error",
                message=f"'{attr}' is not a valid attribute of {type_name}",
                suggested_fix=f"Did you mean '{suggestion}'?" if suggestion else None,
            ))
            return
        # Range check for numeric constants
        if isinstance(value_node, ast.Constant) and isinstance(value_node.value, (int, float)):
            rng = self.truth.get_attr_range(type_name, attr)
            if rng and not (rng["hard_min"] <= value_node.value <= rng["hard_max"]):
                self.issues.append(ValidationIssue(
                    line=node.lineno, category="out_of_range", severity="error",
                    message=f"{type_name}.{attr} = {value_node.value} outside [{rng['hard_min']}, {rng['hard_max']}]",
                    suggested_fix=f"Use {rng['soft_min']}..{rng['soft_max']}",
                ))

    def _check_shader_node(self, node, node_type: str):
        if isinstance(node_type, str) and node_type.startswith("ShaderNode"):
            if not self.truth.is_valid_shader_node(node_type):
                suggestion = self.truth.suggest_shader_node(node_type)
                self.issues.append(ValidationIssue(
                    line=node.lineno, category="wrong_node_type", severity="error",
                    message=f"Invalid shader node: '{node_type}'",
                    suggested_fix=f"Did you mean '{suggestion}'?" if suggestion else None,
                ))


from difflib import get_close_matches  # used in visit_Subscript
```

### 3.5 AST-Based Auto-Fixer

```python
"""script_fixer.py — Auto-correct hallucinated Blender API calls via AST transformation."""
import ast
from difflib import get_close_matches
from typing import Optional

from truth_pack import TruthPack

# Known LLM hallucinations → correct Blender 5.0 names
KNOWN_RENAMES = {
    # Attributes
    "resolution_divisions": "resolution_max",
    "use_adaptive_time_steps": "use_adaptive_timesteps",
    "use_dissolve": "use_dissolve_smoke",
    "timesteps_per_frame": "timesteps_max",
    "timesteps_maximum": "timesteps_max",
    "reaction_speed": "burning_rate",
    "sampling_substeps": "subframes",
    # Shader node types
    "ShaderNodeMixRGB": "ShaderNodeMix",
    "ShaderNodeSeparateRGB": "ShaderNodeSeparateColor",
    "ShaderNodeCombineRGB": "ShaderNodeCombineColor",
    # Engine names
    "BLENDER_EEVEE_NEXT": "BLENDER_EEVEE",
    # Socket names
    "Fac": "Factor",
    "Color1": "A",
    "Color2": "B",
    "Transmission": "Transmission Weight",
    "Specular": "Specular IOR Level",
}

TYPE_HINTS = {
    "domain": "FluidDomainSettings",
    "domain_settings": "FluidDomainSettings",
    "flow": "FluidFlowSettings",
    "flow_settings": "FluidFlowSettings",
    "effector": "FluidEffectorSettings",
    "render": "RenderSettings",
}


class BlenderScriptFixer(ast.NodeTransformer):
    """AST transformer that auto-corrects hallucinated Blender API calls."""

    def __init__(self, truth: TruthPack):
        self.truth = truth
        self.fixes: list[str] = []

    def fix(self, source: str) -> tuple[str, list[str]]:
        self.fixes = []
        tree = ast.parse(source)
        fixed = self.visit(tree)
        ast.fix_missing_locations(fixed)
        return ast.unparse(fixed), self.fixes

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, str) and node.value in KNOWN_RENAMES:
            new_val = KNOWN_RENAMES[node.value]
            self.fixes.append(f"Line {node.lineno}: '{node.value}' → '{new_val}'")
            return ast.Constant(value=new_val)
        # Fuzzy fix for shader node types
        if isinstance(node.value, str) and node.value.startswith("ShaderNode"):
            if not self.truth.is_valid_shader_node(node.value):
                suggestion = self.truth.suggest_shader_node(node.value)
                if suggestion:
                    self.fixes.append(f"Line {node.lineno}: node '{node.value}' → '{suggestion}'")
                    return ast.Constant(value=suggestion)
        return node

    def visit_Attribute(self, node: ast.Attribute):
        self.generic_visit(node)
        # Known renames
        if node.attr in KNOWN_RENAMES:
            new_attr = KNOWN_RENAMES[node.attr]
            self.fixes.append(f"Line {node.lineno}: .{node.attr} → .{new_attr}")
            node.attr = new_attr
            return node
        # Fuzzy fix for typed variables
        if isinstance(node.value, ast.Name):
            type_name = TYPE_HINTS.get(node.value.id)
            if type_name:
                valid = self.truth._attrs.get(type_name, set())
                if valid and node.attr not in valid:
                    suggestion = get_close_matches(node.attr, list(valid), n=1, cutoff=0.5)
                    if suggestion:
                        self.fixes.append(f"Line {node.lineno}: .{node.attr} → .{suggestion[0]} (fuzzy)")
                        node.attr = suggestion[0]
        return node


def validate_and_fix(source: str, truth_path: str = "truth_pack.json") -> tuple[str, list[str], list]:
    """Full pipeline: validate → fix → re-validate."""
    truth = TruthPack(truth_path)

    from script_validator import BlenderScriptValidator
    validator = BlenderScriptValidator(truth)
    fixer = BlenderScriptFixer(truth)

    # Validate
    issues = validator.validate(source)

    # Fix
    fixed_source, fixes = fixer.fix(source)

    # Re-validate
    remaining = validator.validate(fixed_source)

    return fixed_source, fixes, remaining
```

### 3.6 Deterministic Camera Placement

The #2 failure mode. The LLM has no spatial reasoning — it generates camera coordinates with no awareness of scene geometry. The fix is purely geometric.

```python
"""camera.py — Deterministic camera placement from scene bounds."""
import math

CAMERA_PRESETS = {
    "fire":      {"elevation_deg": 15, "azimuth_deg": -30, "padding": 2.0, "min_distance": 2.0},
    "candle":    {"elevation_deg": 10, "azimuth_deg": -20, "padding": 3.0, "min_distance": 0.5},
    "smoke":     {"elevation_deg": 5,  "azimuth_deg": -45, "padding": 2.5, "min_distance": 3.0},
    "liquid":    {"elevation_deg": 25, "azimuth_deg": -40, "padding": 2.0, "min_distance": 1.5},
    "explosion": {"elevation_deg": 10, "azimuth_deg": -30, "padding": 3.0, "min_distance": 5.0},
    "default":   {"elevation_deg": 20, "azimuth_deg": -30, "padding": 2.0, "min_distance": 2.0},
}

# This snippet is INJECTED into every generated Blender script, AFTER scene setup
CAMERA_SNIPPET = '''
# === DETERMINISTIC CAMERA PLACEMENT (injected by pipeline) ===
import bpy, math
from mathutils import Vector

def _place_camera():
    cam = bpy.context.scene.camera
    if not cam:
        return

    # Compute world-space bounding box of all mesh objects
    min_pt = Vector((float("inf"),) * 3)
    max_pt = Vector((float("-inf"),) * 3)
    for obj in bpy.context.scene.objects:
        if obj.type not in ("MESH", "EMPTY") or obj == cam:
            continue
        for corner in obj.bound_box:
            wc = obj.matrix_world @ Vector(corner)
            min_pt = Vector(min(min_pt[i], wc[i]) for i in range(3))
            max_pt = Vector(max(max_pt[i], wc[i]) for i in range(3))

    center = (min_pt + max_pt) / 2
    diagonal = (max_pt - min_pt).length
    if diagonal < 0.01:
        diagonal = 2.0  # Fallback for empty scenes

    # Camera distance from FOV geometry
    fov_rad = cam.data.angle  # Camera FOV in radians
    res_x = bpy.context.scene.render.resolution_x
    res_y = bpy.context.scene.render.resolution_y
    aspect = res_x / res_y if res_y > 0 else 16 / 9
    eff_fov = 2 * math.atan(math.tan(fov_rad / 2) / aspect) if aspect >= 1 else fov_rad
    geo_dist = (diagonal / 2) / math.tan(eff_fov / 2)

    padding = {padding}
    min_dist = {min_distance}
    final_dist = max(geo_dist * padding, min_dist)

    elev = math.radians({elevation_deg})
    azim = math.radians({azimuth_deg})
    cam.location = (
        center.x + final_dist * math.cos(elev) * math.sin(azim),
        center.y - final_dist * math.cos(elev) * math.cos(azim),
        center.z + final_dist * math.sin(elev),
    )

    # Point at scene center
    direction = center - Vector(cam.location)
    rot_quat = direction.to_track_quat("-Z", "Y")
    cam.rotation_euler = rot_quat.to_euler()

    # VALIDATE: camera outside all bounding boxes
    cl = Vector(cam.location)
    if all(min_pt[i] <= cl[i] <= max_pt[i] for i in range(3)):
        d = (cl - center).normalized()
        cam.location = center + d * (diagonal * 3)
        direction = center - Vector(cam.location)
        rot_quat = direction.to_track_quat("-Z", "Y")
        cam.rotation_euler = rot_quat.to_euler()

    print(f"[CAMERA] Placed at dist={{final_dist:.2f}} from center={{tuple(round(c,2) for c in center)}}")

_place_camera()
'''


def get_camera_snippet(effect_type: str) -> str:
    """Return the camera placement snippet with effect-appropriate parameters."""
    preset = CAMERA_PRESETS.get(effect_type, CAMERA_PRESETS["default"])
    return CAMERA_SNIPPET.format(**preset)
```

### 3.7 Expected Impact of Layered Defense

| Layer | Previous System | New System |
|-------|----------------|------------|
| Prevention (prompt grounding) | ~0% | ~60% fewer hallucinations |
| Detection (pre-execution) | ~40% (57 regex rules) | ~95% (AST + truth pack) |
| Correction (auto-fix) | ~40% (regex) | ~80% (fuzzy + known renames) |
| Camera | 0% (LLM decides) | ~95% (deterministic geometry) |

---

## 4. Pipeline Design

### 4.1 State Machine

```python
"""pipeline.py — The main pipeline state machine."""
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional
import time


class Phase(Enum):
    PLAN = auto()       # Select technique, gather API reference
    GENERATE = auto()   # Write Blender Python script
    VALIDATE = auto()   # AST + truth pack validation
    EXECUTE = auto()    # Run in Blender headless
    EVALUATE = auto()   # Vision-based quality assessment
    DECIDE = auto()     # Continue, pass, or fail
    COMPLETED = auto()  # Terminal: success
    FAILED = auto()     # Terminal: exhausted budget/retries
    HUMAN_NEEDED = auto()  # Terminal: escalation

TRANSITIONS = {
    Phase.PLAN:      [Phase.GENERATE, Phase.FAILED],
    Phase.GENERATE:  [Phase.VALIDATE, Phase.FAILED],
    Phase.VALIDATE:  [Phase.EXECUTE, Phase.GENERATE],     # fail → re-generate
    Phase.EXECUTE:   [Phase.EVALUATE, Phase.GENERATE, Phase.FAILED],
    Phase.EVALUATE:  [Phase.DECIDE, Phase.FAILED],
    Phase.DECIDE:    [Phase.GENERATE, Phase.COMPLETED, Phase.FAILED, Phase.HUMAN_NEEDED],
    Phase.COMPLETED: [],
    Phase.FAILED:    [],
    Phase.HUMAN_NEEDED: [],
}


@dataclass
class PipelineState:
    session_id: str
    effect_type: str
    description: str
    phase: Phase = Phase.PLAN
    iteration: int = 0
    max_iterations: int = 5
    total_cost_usd: float = 0.0
    budget_limit_usd: float = 2.50
    scores: list[float] = field(default_factory=list)
    issues_history: list[str] = field(default_factory=list)
    technique: Optional[str] = None
    script_path: Optional[str] = None
    render_path: Optional[str] = None

    def transition(self, next_phase: Phase):
        allowed = TRANSITIONS.get(self.phase, [])
        if next_phase not in allowed:
            raise ValueError(f"Illegal: {self.phase.name} → {next_phase.name}")
        self.phase = next_phase

    def is_terminal(self) -> bool:
        return self.phase in (Phase.COMPLETED, Phase.FAILED, Phase.HUMAN_NEEDED)
```

### 4.2 Phase Contracts

Each phase has strict preconditions and postconditions. If postconditions aren't met, the pipeline routes to the appropriate recovery path.

```python
CONTRACTS = {
    Phase.PLAN:     {"pre": ["effect_type", "description"],
                     "post": ["technique", "api_reference"]},
    Phase.GENERATE: {"pre": ["technique", "api_reference"],
                     "post": ["script_path", "script_content"]},
    Phase.VALIDATE: {"pre": ["script_content"],
                     "post": ["validation_passed", "fixed_script"]},
    Phase.EXECUTE:  {"pre": ["script_path", "validation_passed"],
                     "post": ["render_path", "execution_success"]},
    Phase.EVALUATE: {"pre": ["render_path"],
                     "post": ["score", "issues", "primary_issue"]},
    Phase.DECIDE:   {"pre": ["score", "issues"],
                     "post": ["decision"]},
}
```

### 4.3 The Complete Pipeline

```python
"""orchestrator.py — The main pipeline loop."""
import asyncio
from pathlib import Path

from agents import Agent, Runner, trace

from truth_pack import TruthPack
from script_validator import BlenderScriptValidator
from script_fixer import BlenderScriptFixer, validate_and_fix
from camera import get_camera_snippet
from pipeline import Phase, PipelineState
from budget import BudgetTracker
from plateau import PlateauDetector, EscapeLevel
from knowledge_store import KnowledgeStore
from dynamic_instructions import build_dynamic_instructions


async def create_vfx_asset(
    effect_type: str,
    description: str,
    quality_threshold: int = 60,
    max_iterations: int = 5,
    budget_limit: float = 2.50,
) -> dict:
    """Main pipeline: PLAN → GENERATE → VALIDATE → EXECUTE → EVALUATE → DECIDE → loop."""

    state = PipelineState(
        session_id=f"session_{int(time.time())}_{effect_type}",
        effect_type=effect_type,
        description=description,
        max_iterations=max_iterations,
        budget_limit_usd=budget_limit,
    )

    truth = TruthPack("truth_pack.json")
    budget = BudgetTracker(per_asset_limit_usd=budget_limit)
    plateau = PlateauDetector()
    knowledge = KnowledgeStore()

    with trace(f"VFX Pipeline: {effect_type}"):

        # ── PLAN PHASE ──────────────────────────────────────
        state.transition(Phase.PLAN)

        # Build API reference from truth pack (relevant types only)
        type_map = {
            "fire": ["FluidDomainSettings", "FluidFlowSettings"],
            "smoke": ["FluidDomainSettings", "FluidFlowSettings"],
            "liquid": ["FluidDomainSettings", "FluidFlowSettings", "FluidEffectorSettings"],
            "explosion": ["FluidDomainSettings", "FluidFlowSettings"],
        }
        relevant_types = type_map.get(effect_type, ["FluidDomainSettings"])
        api_ref = "\n\n".join(truth.snippet_for_type(t) for t in relevant_types)

        # Dynamic instructions from knowledge store
        dyn_instructions = build_dynamic_instructions(knowledge, effect_type, 0)

        state.technique = effect_type  # Simple for v1; smarter selection in v2
        state.transition(Phase.GENERATE)

        # ── ITERATION LOOP ──────────────────────────────────
        best_score = 0
        best_render = None

        for iteration in range(max_iterations):
            state.iteration = iteration

            # Budget gate
            if not budget.can_afford(0.15):
                state.transition(Phase.FAILED)
                break

            # Escape velocity check
            escape = plateau.get_escape_level()
            if escape >= EscapeLevel.HUMAN_HELP:
                state.transition(Phase.HUMAN_NEEDED)
                break

            # ── GENERATE ────────────────────────────────────
            dyn_instructions = build_dynamic_instructions(
                knowledge, effect_type, iteration,
                previous_issues=state.issues_history[-3:] if state.issues_history else None,
            )

            if iteration == 0 or escape >= EscapeLevel.NEW_TECHNIQUE:
                prompt = (
                    f"Create a {effect_type} effect: {description}\n\n"
                    f"## Blender 5.0 API Reference (ONLY use these attributes):\n{api_ref}\n\n"
                    f"{dyn_instructions}"
                )
            else:
                prompt = (
                    f"Modify this script to fix: {fix_strategy.fix_instructions}\n\n"
                    f"Current script:\n```python\n{script_content}\n```\n\n"
                    f"## API Reference:\n{api_ref}\n\n"
                    f"{dyn_instructions}"
                )

            gen_result = await Runner.run(code_writer, prompt, max_turns=3)
            script: GeneratedScript = gen_result.final_output
            script_content = script.script_content
            budget.record_spend(0.07, f"code_writer iter {iteration}")

            # ── VALIDATE ────────────────────────────────────
            state.transition(Phase.VALIDATE)

            fixed_source, fixes, remaining_issues = validate_and_fix(script_content, "truth_pack.json")
            if fixes:
                script_content = fixed_source

            # Inject camera placement snippet
            camera_snippet = get_camera_snippet(effect_type)
            script_content = script_content + "\n\n" + camera_snippet

            # Save script
            script_path = Path(f"scripts/{state.session_id}_iter{iteration}.py")
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text(script_content)
            state.script_path = str(script_path)

            errors_blocking = [i for i in remaining_issues if i.severity == "error"]
            if len(errors_blocking) > 3:
                # Too many unfixable issues — regenerate
                state.transition(Phase.GENERATE)
                continue

            # ── EXECUTE ─────────────────────────────────────
            state.transition(Phase.EXECUTE)

            exec_result = await execute_blender_script(str(script_path))
            if "ERROR" in exec_result or "Traceback" in exec_result:
                # Execution failed — get fix strategy
                fix_prompt = f"Blender error:\n{exec_result[:2000]}\n\nScript:\n{script_content[:3000]}"
                fix_result = await Runner.run(fix_strategist, fix_prompt, max_turns=2)
                fix_strategy = fix_result.final_output
                budget.record_spend(0.01, f"fix_strategist iter {iteration}")
                state.transition(Phase.GENERATE)
                continue

            render_path = extract_render_path(exec_result)
            state.render_path = render_path

            # ── EVALUATE ────────────────────────────────────
            state.transition(Phase.EVALUATE)

            assessment = await evaluate_render(render_path, effect_type)
            budget.record_spend(0.02, f"quality_judge iter {iteration}")

            state.scores.append(assessment.overall_score)
            state.issues_history.append(assessment.primary_issue)
            plateau.record(assessment.overall_score, assessment.primary_issue)

            if assessment.overall_score > best_score:
                best_score = assessment.overall_score
                best_render = render_path

            # ── DECIDE ──────────────────────────────────────
            state.transition(Phase.DECIDE)

            if (assessment.overall_score >= quality_threshold
                    and not assessment.critical_failures):
                state.transition(Phase.COMPLETED)
                # Record success in knowledge store
                knowledge.record_outcome(
                    f"tech_{effect_type}_{state.technique}", True, effect_type
                )
                return {
                    "status": "PASS",
                    "score": best_score,
                    "render": best_render,
                    "iterations": iteration + 1,
                    "cost": budget.current_asset_spent_usd,
                }

            # Get fix strategy for next iteration
            fix_prompt = (
                f"Quality assessment:\n{assessment.model_dump_json()}\n\n"
                f"Script:\n{script_content[:3000]}"
            )
            fix_result = await Runner.run(fix_strategist, fix_prompt, max_turns=2)
            fix_strategy = fix_result.final_output
            budget.record_spend(0.01, f"fix_strategist iter {iteration}")

            state.transition(Phase.GENERATE)

        # Loop exhausted
        if not state.is_terminal():
            state.transition(Phase.FAILED)

        # Record failure in knowledge store
        if state.issues_history:
            knowledge.add_entry(
                id=f"fail_{effect_type}_{state.issues_history[-1][:30]}",
                level="failure",
                content=f"Effect '{effect_type}' failed with: {state.issues_history[-1]}",
                context={"effect_type": effect_type, "best_score": best_score},
            )

        return {
            "status": state.phase.name,
            "score": best_score,
            "render": best_render,
            "iterations": state.iteration + 1,
            "cost": budget.current_asset_spent_usd,
        }
```

### 4.4 Pipeline Parallelism

| Parallel (asyncio.gather) | Sequential (must wait) |
|---|---|
| Doc lookup + knowledge query (PLAN) | GENERATE must follow PLAN |
| Vision eval + LPIPS + histogram (EVALUATE) | EXECUTE must follow VALIDATE |
| Multiple API validation passes (VALIDATE) | EVALUATE must follow EXECUTE |

---

## 5. Self-Learning

### 5.1 Knowledge Store Schema

Four levels of knowledge, promoted by evidence:

```
CANDIDATE → PROMISING → TRUSTED → (may be DEMOTED)
```

| Level | Example | Promotion Threshold |
|-------|---------|-------------------|
| **PARAMETER** | "particle_radius=0.5 works for kitchen" | 2 successes → PROMISING |
| **TECHNIQUE** | "Mantaflow gas + force fields = explosions" | 3 successes across 2+ effect types → TRUSTED |
| **APPROACH** | "Liquid pours need collision effectors" | 3 successes across 2+ effect types → TRUSTED |
| **FAILURE** | "Camera inside objects = score 0" | Created immediately on critical failure |

Demotion: 2 consecutive failures demotes TRUSTED/PROMISING → DEMOTED.

See Section 3 of the learning-strategist research for the full `KnowledgeStore` SQLite implementation.

### 5.2 Dynamic Instructions

Validated knowledge feeds back into the Code Writer's prompt at runtime:

```python
"""dynamic_instructions.py"""
from knowledge_store import KnowledgeStore


def build_dynamic_instructions(
    store: KnowledgeStore,
    effect_type: str,
    iteration: int,
    previous_issues: list[str] | None = None,
) -> str:
    sections = []

    # FAILURE avoidance (always inject, highest priority)
    failures = store.get_failures(effect_type=effect_type)
    if failures:
        lines = [f"- {f['content']}" for f in failures]
        sections.append("## CRITICAL: Known Failures (DO NOT REPEAT)\n" + "\n".join(lines))

    # TRUSTED techniques
    trusted = store.get_trusted(level="technique", effect_type=effect_type)
    if trusted:
        lines = [f"- {t['content']}" for t in trusted[:3]]
        sections.append("## Proven Techniques\n" + "\n".join(lines))

    # TRUSTED parameters
    params = store.get_trusted(level="parameter", effect_type=effect_type)
    if params:
        lines = [f"- {p['content']}" for p in params[:5]]
        sections.append("## Validated Parameters\n" + "\n".join(lines))

    # Previous iteration issues
    if previous_issues and iteration > 0:
        lines = [f"- {issue}" for issue in previous_issues]
        sections.append(f"## Issues from Iteration {iteration - 1} (FIX THESE)\n" + "\n".join(lines))

    return "\n\n".join(sections) if sections else ""
```

**Key principle:** Only inject TRUSTED/PROMISING knowledge + FAILURE warnings. Never inject CANDIDATE knowledge.

### 5.3 Evidence Gating

Knowledge is promoted by evidence, not by assumption:

| Transition | Evidence Required |
|------------|------------------|
| → CANDIDATE | Created on first observation |
| CANDIDATE → PROMISING | 2+ successes (score improvement when this knowledge was applied) |
| PROMISING → TRUSTED | 3+ successes across 2+ different effect types |
| TRUSTED/PROMISING → DEMOTED | 2 consecutive failures after applying this knowledge |

---

## 6. SDK Patterns

### 6.1 What We Use

| SDK Feature | How We Use It | Why |
|-------------|--------------|-----|
| `@function_tool` | Blender execution, doc search, metrics | Expose Python functions to agents with auto-schema |
| `Agent(output_type=...)` | All 3 agents return Pydantic models | Guaranteed typed output, no regex parsing |
| `Runner.run()` | Called from Python loop, not from agents | Code controls flow, agents handle creativity |
| `RunHooks` | Logging, budget tracking, loop detection | Observability without changing agent code |
| `@input_guardrail` | Budget check before expensive operations | Hard stop on overspend |
| `trace()` | Wraps entire pipeline run | Free tracing in OpenAI dashboard |
| `max_turns` | Set per `Runner.run()` call (3-10) | Prevent runaway agent loops |
| `RunConfig` | Global guardrails, tracing config | Consistent settings across all runs |
| `ModelSettings` | `tool_choice` for single-tool agents | Force executor to call the right tool |

### 6.2 What We Skip

| SDK Feature | Why Skip |
|-------------|----------|
| **Handoffs** | Lose orchestrator context. Use agents-as-tools or direct `Runner.run()`. |
| **MCP in agents** | Tools must run in-process. MCP from agent context fails. |
| **Sessions (SQLiteSession)** | The pipeline loop IS the state. No need for SDK conversation persistence. |
| **Voice/Realtime** | Not relevant. |
| **ShellTool/ApplyPatchTool** | We have our own Blender executor. |
| **o3/o4-mini** | Hidden reasoning tokens make cost unpredictable. Use gpt-5.2 for quality, gpt-5-mini for speed. |

### 6.3 Key SDK Anti-Patterns to Avoid

1. **Agent for Everything** — Blender execution, API validation, file I/O are NOT agents.
2. **LLM-Driven Orchestration** — Python `while` loop, not an agent deciding "what next?"
3. **No `max_turns`** — Always set 3-10 per step. Use `error_handlers={"max_turns": handler}`.
4. **Giant system prompts** — Keep focused. Provide API reference via function tools on-demand. Leverage prompt caching.
5. **Single agent with 20 tools** — Keep 3-5 focused tools per agent.
6. **Not using structured outputs** — Always use `output_type=PydanticModel`.

### 6.4 Prompt Caching Strategy

GPT-5 family gets **90% discount on cached input tokens**. Structure prompts so the static prefix is identical across calls:

```
[STATIC: system instructions + API reference + known failures]  ← cached
[DYNAMIC: iteration-specific context + fix instructions]        ← not cached
```

This alone can cut input costs 5-10x.

---

## 7. Human-in-the-Loop

### 7.1 The Reality

The Python Agents SDK does **not** have native HITL/pause-resume support (as of Feb 2026). The JS SDK has `needsApproval` + `state.approve()`/`state.reject()`, but Python does not (GitHub issue #636).

**Workaround:** CLI-based approval gates + guardrail tripwires.

### 7.2 Implementation

```python
"""hitl.py — Human-in-the-loop approval gates."""
import asyncio


async def cli_approval(action: str, details: str, auto_approve: bool = False) -> bool:
    """Simple CLI approval gate."""
    if auto_approve:
        return True
    print(f"\n{'='*60}")
    print(f"APPROVAL REQUIRED: {action}")
    print(f"Details: {details}")
    print(f"{'='*60}")
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, lambda: input("[y/N]: ").strip().lower())
    return response == "y"
```

### 7.3 Autonomy Progression

The system earns less human oversight by demonstrating reliability:

| Level | Evidence Required | Human Approves | Auto-Approved |
|-------|------------------|----------------|---------------|
| **0: Supervised** | None (start here) | Technique, budget >$1, quality overrides | Nothing |
| **1: Novice** | 3 iterations scoring >40 | New techniques, quality overrides | Small budget decisions |
| **2: Trusted** | 5 assets scoring >60 | Quality gate overrides only | Technique selection, budget |
| **3: Autonomous** | 10 assets >60 across 3+ types | Nothing (alerts on anomalies) | Everything |

**Quality gate overrides always require human approval.** The system must never lie about quality.

### 7.4 When the Human Intervenes

| Situation | Trigger | Human Action |
|-----------|---------|--------------|
| Novel effect type | First time seeing this type | Approve/redirect technique |
| Budget threshold | Asset cost approaching limit | Approve continued spending or stop |
| Escape level 4 | System stuck after all automated options | Provide guidance or abort |
| Quality override | System wants to accept a low score | Approve or reject |
| Regression | Score decreased 2+ iterations in a row | Diagnose and redirect |

---

## 8. Budget Management

### 8.1 Model Pricing (Feb 2026)

| Model | Input/1M | Output/1M | Cached Input | Use For |
|-------|----------|-----------|--------------|---------|
| `gpt-5.2` | $1.75 | $14.00 | $0.175 | Code Writer |
| `gpt-5-mini` | $0.25 | $2.00 | $0.025 | Quality Judge, Fix Strategist |
| `o3` | $2.00 | $8.00 | N/A | **AVOID** (hidden reasoning tokens) |
| `o4-mini` | $1.10 | $4.40 | N/A | **AVOID** (hidden reasoning tokens) |

### 8.2 Cost Per Iteration

| Phase | Model | Input Tokens | Output Tokens | Cost |
|-------|-------|-------------|---------------|------|
| GENERATE | gpt-5.2 | ~5,000 | ~3,000 | ~$0.051 |
| VALIDATE | (local) | — | — | $0.000 |
| EXECUTE | (local) | — | — | $0.000 |
| EVALUATE | gpt-5-mini | ~3,000 | ~500 | ~$0.002 |
| DECIDE/FIX | gpt-5-mini | ~3,000 | ~500 | ~$0.002 |
| **Total** | | | | **~$0.055** |

With prompt caching on repeated instructions (~90% of input cached):
- **Cached cost per iteration: ~$0.03-0.05**

### 8.3 Budget Allocation ($20/month)

| Metric | Value |
|--------|-------|
| Cost per iteration (cached) | ~$0.04 |
| Per-asset budget cap | $2.50 |
| Max iterations per asset | ~50 (budget-limited) |
| Max assets per month | 7 (with $2.50 buffer) |
| Emergency reserve | $2.50 |

### 8.4 Budget Tracker

```python
"""budget.py"""
from dataclasses import dataclass, field
import time


@dataclass
class BudgetTracker:
    monthly_limit_usd: float = 20.00
    per_asset_limit_usd: float = 2.50

    total_spent_usd: float = 0.0
    current_asset_spent_usd: float = 0.0
    log: list = field(default_factory=list)

    def record_spend(self, amount: float, description: str):
        self.total_spent_usd += amount
        self.current_asset_spent_usd += amount
        self.log.append({"amount": amount, "desc": description, "t": time.time()})

    def can_afford(self, estimated: float) -> bool:
        return (self.total_spent_usd + estimated <= self.monthly_limit_usd
                and self.current_asset_spent_usd + estimated <= self.per_asset_limit_usd)

    def reset_for_new_asset(self):
        self.current_asset_spent_usd = 0.0

    @property
    def iterations_remaining(self) -> int:
        return int(min(
            (self.monthly_limit_usd - self.total_spent_usd) / 0.05,
            (self.per_asset_limit_usd - self.current_asset_spent_usd) / 0.05,
        ))
```

---

## 9. Error Recovery

### 9.1 Error Taxonomy

```python
"""errors.py"""
from enum import Enum, auto


class ErrorCategory(Enum):
    # Script errors
    SYNTAX_ERROR = auto()
    API_HALLUCINATION = auto()
    MISSING_IMPORT = auto()

    # Execution errors
    BLENDER_CRASH = auto()
    TIMEOUT = auto()
    OOM = auto()
    BAKE_EMPTY = auto()

    # Quality errors
    CAMERA_INSIDE = auto()
    CAMERA_TOO_CLOSE = auto()
    ZERO_LIGHTS = auto()
    OVEREXPOSED = auto()
    UNDEREXPOSED = auto()
    NO_EFFECTORS = auto()
    SCORE_TOO_LOW = auto()

    # System errors
    BUDGET_EXHAUSTED = auto()
    MAX_ITERATIONS = auto()
    PLATEAU = auto()
```

### 9.2 Error Routing

| Error | Action | Max Retries |
|-------|--------|-------------|
| SYNTAX_ERROR | Re-generate script | 2 |
| API_HALLUCINATION | Auto-fix via AST fixer, re-validate | 2 |
| BLENDER_CRASH | Retry execution (may be transient) | 1 |
| TIMEOUT | Reduce resolution, retry | 1 |
| OOM | Reduce resolution, retry | 1 |
| BAKE_EMPTY | Fix bake settings (use_plane_init, cache_type) | 2 |
| CAMERA_INSIDE | **Deterministic fix** (camera snippet re-injection) | 0 (should not recur) |
| ZERO_LIGHTS | Inject default 3-point lighting | 1 |
| OVEREXPOSED | Clamp light energy (per-effect-type bounds) | 1 |
| UNDEREXPOSED | Boost light energy (per-effect-type bounds) | 1 |
| SCORE_TOO_LOW | Fix Strategist → Code Writer modify | 3 |
| PLATEAU | Switch technique entirely | 1 |
| BUDGET_EXHAUSTED | **Stop immediately** | 0 |
| MAX_ITERATIONS | **Stop, return best result** | 0 |

### 9.3 Circuit Breaker

```python
"""circuit_breaker.py"""
import time
from dataclasses import dataclass, field
from enum import Enum, auto


class BreakerState(Enum):
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


@dataclass
class CircuitBreaker:
    name: str
    fail_threshold: int = 3
    reset_timeout: float = 120.0

    _state: BreakerState = field(default=BreakerState.CLOSED, init=False)
    _failures: int = field(default=0, init=False)
    _last_fail: float = field(default=0.0, init=False)

    @property
    def state(self) -> BreakerState:
        if self._state == BreakerState.OPEN:
            if time.time() - self._last_fail >= self.reset_timeout:
                self._state = BreakerState.HALF_OPEN
        return self._state

    def allow(self) -> bool:
        return self.state != BreakerState.OPEN

    def record_success(self):
        self._failures = 0
        self._state = BreakerState.CLOSED

    def record_failure(self):
        self._failures += 1
        self._last_fail = time.time()
        if self._failures >= self.fail_threshold:
            self._state = BreakerState.OPEN


BREAKERS = {
    "blender": CircuitBreaker("blender", fail_threshold=3, reset_timeout=120),
    "llm": CircuitBreaker("llm", fail_threshold=5, reset_timeout=60),
}
```

### 9.4 Plateau Detection and Escape Velocity

```python
"""plateau.py"""
from dataclasses import dataclass, field
from enum import IntEnum


class EscapeLevel(IntEnum):
    NORMAL = 0
    QUERY_KNOWLEDGE = 1
    NEW_TECHNIQUE = 2
    NEW_APPROACH = 3
    HUMAN_HELP = 4


@dataclass
class PlateauDetector:
    scores: list[float] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    escape_level: EscapeLevel = EscapeLevel.NORMAL

    min_improvement: float = 3.0
    same_issue_threshold: int = 2

    def record(self, score: float, primary_issue: str):
        self.scores.append(score)
        self.issues.append(primary_issue)
        self._detect()

    def _detect(self):
        if len(self.scores) < 2:
            return

        # Score not improving
        if len(self.scores) >= 2:
            if self.scores[-1] - self.scores[-2] < self.min_improvement:
                self.escape_level = EscapeLevel(min(self.escape_level + 1, EscapeLevel.HUMAN_HELP))

        # Same issue repeating
        if len(self.issues) >= self.same_issue_threshold:
            recent = self.issues[-self.same_issue_threshold:]
            if len(set(recent)) == 1:
                self.escape_level = max(self.escape_level, EscapeLevel.NEW_TECHNIQUE)

        # Score regression (oscillation)
        if len(self.scores) >= 3:
            if self.scores[-1] < self.scores[-2] - self.min_improvement:
                self.escape_level = max(self.escape_level, EscapeLevel.QUERY_KNOWLEDGE)

    def get_escape_level(self) -> EscapeLevel:
        return self.escape_level

    def reset_for_new_technique(self):
        self.escape_level = EscapeLevel.NORMAL
```

**Escape level actions:**

| Level | Action | Critical Rule |
|-------|--------|---------------|
| 0: NORMAL | Modify current script | Standard path |
| 1: QUERY_KB | Check knowledge store for alternatives | Use validated data |
| 2: NEW_TECHNIQUE | **Discard script, generate fresh** | NEVER modify at this level |
| 3: NEW_APPROACH | Research novel approaches | Break assumptions |
| 4: HUMAN_HELP | Pause, show full history | System is stuck |

---

## 10. What We Explicitly Do NOT Build

| Component | Verdict | Justification |
|-----------|---------|---------------|
| **TechniqueSelector agent** | NOT an agent | A lookup table + knowledge query. No LLM needed for "which technique for fire?" |
| **ModificationStrategist agent** | Merged into Fix Strategist | One agent handles diagnosis + strategy. Two is redundant. |
| **QualityGateJudge agent** | NOT an agent | `score >= threshold and not critical_failures` is a 5-line function. |
| **ResearchAgent** | NOT an agent | `search_docs(query)` is a function tool. No agent persona needed. |
| **DocsExpert agent** | NOT an agent | Vector store query is a function call. |
| **LearningAgent** | NOT an agent | `store.record_outcome()` is a function call. |
| **ApiValidator agent** | NOT an agent | AST + truth pack is deterministic. LLM would hallucinate. |
| **Vector store for docs** | Use truth pack instead | You need EXACT attribute names, not "similar" results. |
| **Embedding-based pattern memory** | Use SQLite text search | ~100 patterns at this scale. Embeddings add complexity for no gain. |
| **Experiment tracking dashboard** | Use JSON logs | Need the system to WORK first. Visualize later. |
| **Self-modifying prompts** | Defer to v2 | Need a working baseline first. Dynamic instructions are sufficient. |
| **Multi-model routing logic** | Use 2 models only | gpt-5.2 for code, gpt-5-mini for everything else. Add complexity when evidence demands it. |
| **.blend file patching** | Defer to v2 | Generate-from-scratch works for v1. |
| **Streaming quality feedback** | Defer to v2 | Useful for UX, not for correctness. |
| **Distributed execution** | Not needed | One Blender instance is fine for 7 assets/month. |
| **Proactive mining agent** | Not at this scale | Web search tool at escape level 3+ is simpler. |

**The principle:** Every component that was an agent in the previous system and doesn't require creative judgment becomes either a function tool or plain Python. This cuts the agent count from 10 to 3 and eliminates the 0.95^10 compounding error problem.

---

## 11. Implementation Order

Each phase is gated by evidence from the previous phase. No phase starts until its gate is passed.

### Phase 1: Minimal Viable Pipeline

**Goal:** `kitchen_leak` scores >40 with deterministic camera and API fixing.

**Build order:**
1. `truth_pack_generator.py` — Run in Blender, generate `truth_pack.json`
2. `truth_pack.py` — Loader + lookup
3. `script_validator.py` — AST-based validation
4. `script_fixer.py` — AST-based auto-correction
5. `camera.py` — Deterministic camera placement snippet
6. `budget.py` — Budget tracker
7. `pipeline.py` — State machine + phase contracts
8. `orchestrator.py` — Main pipeline loop with 3 agents

**Test:** Run `kitchen_leak` scenario. Must score >40. Must not place camera inside objects. Must not hallucinate deprecated API calls.

**Gate to Phase 2:** kitchen_leak scores >40 on 2 of 3 runs.

### Phase 2: Multi-Effect Reliability

**Goal:** 3 different effect types score >40.

**Build order:**
1. `knowledge_store.py` — SQLite-backed evidence-gated store
2. `dynamic_instructions.py` — Inject validated knowledge into prompts
3. `plateau.py` — Plateau detection + escape levels 0-2
4. `hitl.py` — CLI approval gates
5. Effect-type camera presets (fire, smoke, liquid, explosion)
6. Per-effect-type light energy bounds

**Test:** `kitchen_leak` + `candle_fire` + one explosion all score >40.

**Gate to Phase 3:** 3 different effects score >40 on 2 of 3 runs each.

### Phase 3: Quality Threshold

**Goal:** At least 1 asset scores >60 (pass threshold).

**Build order:**
1. Evidence-gated promotion/demotion (knowledge store active)
2. Cross-effect knowledge transfer
3. Escape levels 3-4 (new approach research, human help)
4. Autonomy level 1 (auto-approve budget after 3 good iterations)

**Test:** Any single asset scores >60 with ≤5 iterations.

**Gate to Phase 4:** 3 assets score >60.

### Phase 4: Earned Autonomy

**Goal:** Fully autonomous for known effect types.

**Build order:**
1. Autonomy level 2 (auto-approve technique for known types)
2. Self-evolving prompts (optimize from accumulated knowledge)
3. Multi-model routing (if evidence supports it)
4. New physics systems (rigid body, particles, geometry nodes)

**Test:** 5 consecutive assets score >50 with zero human intervention.

**Gate:** Ongoing reliability metrics.

---

## Appendix A: File Structure

```
blender-vfx-orchestrator/
├── orchestrator.py          # Main pipeline loop (Python-driven)
├── pipeline.py              # State machine, phase contracts
├── budget.py                # Budget tracking + enforcement
├── plateau.py               # Plateau detection + escape velocity
├── hitl.py                  # Human-in-the-loop approval gates
├── knowledge_store.py       # SQLite evidence-gated knowledge
├── dynamic_instructions.py  # Runtime instruction injection
├── truth_pack.py            # Truth pack loader + query
├── truth_pack_generator.py  # Runs IN Blender to generate truth pack
├── script_validator.py      # AST-based API validation
├── script_fixer.py          # AST-based auto-correction + known renames
├── camera.py                # Deterministic camera placement
├── agents/
│   ├── code_writer.py       # Agent: gpt-5.2, generates Blender scripts
│   ├── quality_judge.py     # Agent: gpt-5-mini, evaluates renders (vision)
│   └── fix_strategist.py    # Agent: gpt-5-mini, diagnoses + prescribes fixes
├── tools/
│   ├── blender_executor.py  # @function_tool: run scripts in Blender
│   ├── doc_search.py        # @function_tool: vector store search
│   └── quality_metrics.py   # @function_tool: LPIPS/CLIP computation
├── models/
│   ├── schemas.py           # Pydantic models (GeneratedScript, QualityAssessment, etc.)
│   └── state.py             # PipelineState, PipelineError
├── truth_pack.json          # Generated by truth_pack_generator.py
├── knowledge.db             # SQLite knowledge store
└── sessions/                # JSON session files for persistence
```

## Appendix B: Dependency Summary

| Dependency | Purpose | Required? |
|------------|---------|-----------|
| `openai` v2.x | OpenAI API client | Yes |
| `agents` (openai-agents) v0.9.0+ | Agent SDK | Yes |
| `pydantic` v2 | Typed schemas | Yes (SDK dependency) |
| Python `ast` | Script validation | Yes (stdlib) |
| Python `difflib` | Fuzzy matching | Yes (stdlib) |
| Python `sqlite3` | Knowledge store | Yes (stdlib) |
| Python `json` | Session persistence, truth pack | Yes (stdlib) |
| Python `asyncio` | Pipeline concurrency | Yes (stdlib) |
| `Pillow` | Image loading for metrics | Optional (quality metrics) |

**Zero external dependencies beyond the SDK itself.** Everything else is stdlib.

## Appendix C: Key Research Sources

- [OpenAI Agents SDK Docs](https://openai.github.io/openai-agents-python/) — SDK API surface
- [OpenAI Practical Guide to Building Agents](https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/) — Architecture patterns
- [MAST: Why Multi-Agent Systems Fail (NeurIPS 2025)](https://arxiv.org/abs/2503.13657) — 41-86.7% failure rates
- [The 0.95^10 Problem](https://www.artiquare.com/why-multi-agent-ai-fails/) — Error compounding math
- [AST-Based API Hallucination Detection (arxiv:2601.19106)](https://arxiv.org/abs/2601.19106) — 100% precision, 77% auto-fix
- [OpenAI Self-Evolving Agents Cookbook](https://developers.openai.com/cookbook/examples/partners/self_evolving_agents/) — Evidence-gated learning
- [Blender 5.0 Python API](https://docs.blender.org/api/current/) — RNA reflection system
- [OpenAI API Pricing](https://platform.openai.com/docs/pricing) — Model costs
- [GitHub Issue #636: HITL for Python SDK](https://github.com/openai/openai-agents-python/issues/636) — No native Python HITL yet
