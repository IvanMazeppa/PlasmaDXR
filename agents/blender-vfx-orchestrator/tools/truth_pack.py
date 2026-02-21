"""
Truth Pack: Deterministic API validation via Blender introspection.

Eliminates the #1 failure mode (hallucinated Blender 5.0 attributes) by
querying bpy.types.X.bl_rna.properties at runtime. Cost: $0.00.

Replaces the reactive 57-rule regex fixer + LLM-powered API Spec Agent
with a deterministic prevention layer.

Usage:
    truth_pack = await build_truth_pack("mantaflow_gas", blender_exe)
    errors = validate_script_against_truth_pack(script_text, truth_pack)
    fixed, fixes = auto_fix_script(script_text, errors, truth_pack)
"""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Blender executable (reuse from blender_executor_tools)
DEFAULT_BLENDER_EXE = "/home/maz3ppa/apps/blender-5.0.1-linux-x64/blender"
BLENDER_EXE = os.getenv("BLENDER_EXE", DEFAULT_BLENDER_EXE)

# Cache directory for truth packs
ORCHESTRATOR_ROOT = Path(__file__).parent.parent
TRUTH_PACK_CACHE_DIR = ORCHESTRATOR_ROOT / "data" / "truth_packs"

# Markers for parsing introspection output
START_MARKER = "###TRUTH_PACK_START###"
END_MARKER = "###TRUTH_PACK_END###"


# =============================================================================
# INTROSPECTION SCRIPT (runs inside Blender headless)
# =============================================================================

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


# Types to introspect — driven by the pipeline's technique selection
types_to_query = json.loads(sys.argv[sys.argv.index("--") + 1])
output = {}
for t in types_to_query:
    output[t] = introspect_type(t)

print("###TRUTH_PACK_START###")
print(json.dumps(output))
print("###TRUTH_PACK_END###")
'''


# =============================================================================
# TECHNIQUE → TYPES MAPPING
# =============================================================================

TECHNIQUE_TYPES: Dict[str, List[str]] = {
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
        "Object", "Camera", "PointLight", "SpotLight", "SunLight", "AreaLight",
        "CyclesRenderSettings", "RenderSettings", "Scene",
    ],
    "particle_system": [
        "ParticleSettings", "ParticleSystem",
        "Object", "Camera", "PointLight", "SpotLight", "SunLight", "AreaLight",
        "CyclesRenderSettings", "RenderSettings", "Scene",
    ],
    # Common types included in ALL techniques
    "_common": [
        "Object", "Camera", "PointLight", "SpotLight", "SunLight", "AreaLight",
        "CyclesRenderSettings", "RenderSettings", "Scene",
    ],
}

# Aliases: map technique names used in the orchestrator to TECHNIQUE_TYPES keys
TECHNIQUE_ALIASES: Dict[str, str] = {
    "mantaflow_smoke": "mantaflow_gas",
    "mantaflow_fire": "mantaflow_gas",
    "mantaflow_fire_smoke": "mantaflow_gas",
    "mantaflow_explosion": "mantaflow_gas",
    "shader_volume": "mantaflow_gas",
    "mantaflow_water": "mantaflow_liquid",
    "mantaflow_pour": "mantaflow_liquid",
    "mantaflow_splash": "mantaflow_liquid",
    "rigid_body_fracture": "rigid_body",
    "rigid_body_destruction": "rigid_body",
}


# =============================================================================
# SETTINGS MAP: Script variable → bpy.types class
# =============================================================================

SETTINGS_MAP: Dict[str, str] = {
    "domain_settings": "FluidDomainSettings",
    "flow_settings": "FluidFlowSettings",
    "effector_settings": "FluidEffectorSettings",
    "rigid_body": "RigidBodyObject",
    "rigid_body_world": "RigidBodyWorld",
    "particle_systems": "ParticleSystem",
    "cycles": "CyclesRenderSettings",
    "render": "RenderSettings",
    "scene": "Scene",
}

# Additional variable name patterns that map to types
VARIABLE_PATTERNS: Dict[str, str] = {
    "dset": "FluidDomainSettings",
    "dsettings": "FluidDomainSettings",
    "fset": "FluidFlowSettings",
    "fsettings": "FluidFlowSettings",
    "eset": "FluidEffectorSettings",
    "settings": "FluidDomainSettings",  # Ambiguous, but domain is most common
}


# =============================================================================
# VALIDATION ERROR
# =============================================================================

@dataclass
class ValidationError:
    """An invalid attribute access found in a generated script."""
    line: int
    attribute: str
    object_type: str
    message: str
    suggestion: Optional[str] = None


# =============================================================================
# KNOWN HALLUCINATION PATTERNS
# =============================================================================

KNOWN_HALLUCINATIONS: Dict[str, Tuple[str, str]] = {
    # pattern → (message, fix)
    r'\bresolution_divisions\b': (
        "Use 'resolution_max' instead", "resolution_max"
    ),
    r'\buse_adaptive_time_steps\b': (
        "Use 'use_adaptive_timesteps' (no extra underscore)", "use_adaptive_timesteps"
    ),
    r'\buse_dissolve\b(?!_smoke)': (
        "Use 'use_dissolve_smoke' instead", "use_dissolve_smoke"
    ),
    r'\btimesteps_per_frame\b': (
        "Use 'timesteps_max' instead", "timesteps_max"
    ),
    r'\btimesteps_maximum\b': (
        "Use 'timesteps_max' instead", "timesteps_max"
    ),
    r'\breaction_speed\b': (
        "Use 'burning_rate' instead", "burning_rate"
    ),
    r'\bShaderNodeMixRGB\b': (
        "Use 'ShaderNodeMix' in Blender 5.0", "ShaderNodeMix"
    ),
    r'\bShaderNodeSeparateRGB\b': (
        "Use 'ShaderNodeSeparateColor' in Blender 5.0", "ShaderNodeSeparateColor"
    ),
    r'\bBLENDER_EEVEE_NEXT\b': (
        "Use 'BLENDER_EEVEE' in Blender 5.0", "BLENDER_EEVEE"
    ),
    r'\bSubsurface Color\b': (
        "Removed from Principled BSDF in 5.0", None
    ),
    r"['\"]BLOSC['\"]": (
        "BLOSC compression removed in 5.0. Use 'ZIP' or 'NONE'.", "'ZIP'"
    ),
    # Blender 5.0: use_nodes is always True, setting it is a no-op that emits deprecation warnings
    r'\b(\w+)\.use_nodes\s*=': (
        "use_nodes is always True in Blender 5.0 — remove this line", None
    ),
    # Blender 5.0: Sheen Weight replaces Sheen in Principled BSDF
    r"['\"]Sheen['\"](?!\s*Weight)": (
        "Use 'Sheen Weight' not 'Sheen' in Blender 5.0 Principled BSDF", "'Sheen Weight'"
    ),
}

# Hardcoded fixes for known hallucinations (simple string replacements)
HARDCODED_FIXES: Dict[str, str] = {
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


# =============================================================================
# BUILD TRUTH PACK
# =============================================================================

async def build_truth_pack(
    technique: str,
    blender_exe: Optional[str] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Run Blender introspection and return the truth pack.

    Args:
        technique: Pipeline technique name (e.g., "mantaflow_smoke")
        blender_exe: Path to Blender executable (defaults to BLENDER_EXE)
        use_cache: Whether to use cached truth packs

    Returns:
        Dict mapping bpy.types class names to their property metadata

    Raises:
        RuntimeError: If Blender introspection fails
        FileNotFoundError: If Blender executable not found
    """
    blender = blender_exe or BLENDER_EXE

    # Resolve technique aliases
    resolved = TECHNIQUE_ALIASES.get(technique, technique)
    types_needed = TECHNIQUE_TYPES.get(resolved, TECHNIQUE_TYPES.get("mantaflow_gas"))

    # Check cache
    if use_cache:
        cached = _load_cached_truth_pack(resolved)
        if cached is not None:
            print(f"[TruthPack] Using cached truth pack for '{resolved}' "
                  f"({len(cached)} types)", file=sys.stderr)
            return cached

    # Verify Blender exists
    if not Path(blender).exists():
        raise FileNotFoundError(
            f"Blender not found at: {blender}. Set BLENDER_EXE env var."
        )

    # Write introspection script to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', prefix='vfx_introspect_', delete=False
    ) as f:
        f.write(INTROSPECTION_SCRIPT)
        script_path = f.name

    try:
        print(f"[TruthPack] Building truth pack for '{resolved}' "
              f"({len(types_needed)} types)...", file=sys.stderr)

        proc = await asyncio.create_subprocess_exec(
            blender, "--background", "--python", script_path,
            "--", json.dumps(types_needed),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error_msg = stderr.decode()[:500] if stderr else "Unknown error"
            raise RuntimeError(
                f"Blender introspection failed (exit {proc.returncode}): {error_msg}"
            )

        output = stdout.decode()

        # Parse output between markers
        try:
            start = output.index(START_MARKER) + len(START_MARKER)
            end = output.index(END_MARKER)
        except ValueError:
            raise RuntimeError(
                f"Truth pack markers not found in Blender output. "
                f"Output length: {len(output)} chars. "
                f"First 200 chars: {output[:200]}"
            )

        truth_pack = json.loads(output[start:end].strip())

        # Validate we got actual data
        types_with_props = sum(
            1 for v in truth_pack.values()
            if isinstance(v, dict) and "properties" in v
        )
        if types_with_props == 0:
            raise RuntimeError(
                "Truth pack contains no valid type data. "
                f"Keys: {list(truth_pack.keys())}"
            )

        print(f"[TruthPack] Built successfully: {types_with_props} types, "
              f"{sum(len(v.get('properties', {})) for v in truth_pack.values() if isinstance(v, dict))} "
              f"total properties", file=sys.stderr)

        # Cache the result
        _cache_truth_pack(resolved, truth_pack)

        return truth_pack

    finally:
        Path(script_path).unlink(missing_ok=True)


def _load_cached_truth_pack(technique: str) -> Optional[Dict[str, Any]]:
    """Load a cached truth pack if it exists and is fresh."""
    TRUTH_PACK_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Find cache file matching technique
    # Include blender version in cache key for safety
    cache_files = list(TRUTH_PACK_CACHE_DIR.glob(f"*_{technique}.json"))
    if not cache_files:
        return None

    cache_file = cache_files[0]

    # Check age — invalidate after 7 days (Blender updates are rare)
    import time
    age_days = (time.time() - cache_file.stat().st_mtime) / 86400
    if age_days > 7:
        print(f"[TruthPack] Cache expired ({age_days:.0f} days old)", file=sys.stderr)
        return None

    try:
        return json.loads(cache_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _cache_truth_pack(technique: str, truth_pack: Dict[str, Any]) -> None:
    """Cache a truth pack to disk."""
    TRUTH_PACK_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Extract blender version from truth pack
    blender_version = "unknown"
    for type_data in truth_pack.values():
        if isinstance(type_data, dict) and "blender_version" in type_data:
            v = type_data["blender_version"]
            blender_version = f"{v[0]}_{v[1]}_{v[2]}" if isinstance(v, list) else str(v)
            break

    cache_file = TRUTH_PACK_CACHE_DIR / f"{blender_version}_{technique}.json"
    cache_file.write_text(json.dumps(truth_pack, indent=2))
    print(f"[TruthPack] Cached to: {cache_file}", file=sys.stderr)


# =============================================================================
# VALIDATE SCRIPT AGAINST TRUTH PACK
# =============================================================================

def validate_script_against_truth_pack(
    script: str,
    truth_pack: Dict[str, Any],
) -> List[ValidationError]:
    """
    Validate a generated script against the truth pack.

    Checks every attribute access pattern (e.g., domain_settings.X) against
    the truth pack's property lists. Returns validation errors with line
    numbers and suggestions.

    Args:
        script: The generated Blender Python script text
        truth_pack: Truth pack from build_truth_pack()

    Returns:
        List of ValidationError (empty = script is clean)
    """
    errors = []

    # Build valid attribute sets from truth pack
    valid_attrs: Dict[str, set] = {}
    for type_name, type_data in truth_pack.items():
        if isinstance(type_data, dict) and "properties" in type_data:
            valid_attrs[type_name] = set(type_data["properties"].keys())

    lines = script.split("\n")

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue

        # Check settings access patterns (domain_settings.X, flow_settings.X, etc.)
        for settings_attr, type_name in SETTINGS_MAP.items():
            if type_name not in valid_attrs:
                continue
            pattern = rf'\.{re.escape(settings_attr)}\.(\w+)'
            for match in re.finditer(pattern, line):
                attr = match.group(1)
                if attr not in valid_attrs[type_name] and not attr.startswith("_"):
                    suggestion = _suggest_correction(attr, list(valid_attrs[type_name]))
                    errors.append(ValidationError(
                        line=i,
                        attribute=attr,
                        object_type=type_name,
                        message=f"'{attr}' is not a valid property of {type_name}",
                        suggestion=suggestion,
                    ))

        # Check variable name patterns (dset.X, dsettings.X, fset.X, etc.)
        for var_name, type_name in VARIABLE_PATTERNS.items():
            if type_name not in valid_attrs:
                continue
            # Match var_name.attr but not var_name_something.attr
            pattern = rf'\b{re.escape(var_name)}\.(\w+)'
            for match in re.finditer(pattern, line):
                attr = match.group(1)
                if attr not in valid_attrs[type_name] and not attr.startswith("_"):
                    suggestion = _suggest_correction(attr, list(valid_attrs[type_name]))
                    errors.append(ValidationError(
                        line=i,
                        attribute=attr,
                        object_type=type_name,
                        message=f"'{attr}' is not a valid property of {type_name}",
                        suggestion=suggestion,
                    ))

    # Check for known hallucination patterns
    for pattern, (message, _fix) in KNOWN_HALLUCINATIONS.items():
        for i, line in enumerate(lines, 1):
            if re.search(pattern, line):
                errors.append(ValidationError(
                    line=i,
                    attribute=pattern,
                    object_type="KNOWN_HALLUCINATION",
                    message=message,
                ))

    return errors


def _suggest_correction(wrong_attr: str, valid_props: List[str]) -> Optional[str]:
    """Use difflib to find the closest valid attribute name."""
    matches = difflib.get_close_matches(wrong_attr, valid_props, n=1, cutoff=0.6)
    return matches[0] if matches else None


# =============================================================================
# AUTO-FIX SCRIPT
# =============================================================================

def auto_fix_script(
    script: str,
    errors: List[ValidationError],
    truth_pack: Dict[str, Any],
) -> Tuple[str, List[str]]:
    """
    Apply deterministic fixes for validation errors.

    Args:
        script: The original script text
        errors: Validation errors from validate_script_against_truth_pack()
        truth_pack: Truth pack for suggestion lookup

    Returns:
        Tuple of (fixed_script, list_of_fixes_applied)
    """
    fixes_applied = []
    fixed = script

    for error in errors:
        if error.object_type == "KNOWN_HALLUCINATION":
            # Special case: strip use_nodes lines entirely (no-op in 5.0)
            if "use_nodes" in (error.attribute or ""):
                lines = fixed.split("\n")
                new_lines = []
                for line in lines:
                    if re.search(r'\.use_nodes\s*=', line):
                        fixes_applied.append(f"Removed use_nodes assignment (no-op in 5.0)")
                    else:
                        new_lines.append(line)
                fixed = "\n".join(new_lines)
                continue
            # Apply hardcoded fixes for known hallucinations
            for wrong, correct in HARDCODED_FIXES.items():
                if wrong in fixed:
                    fixed = fixed.replace(wrong, correct)
                    fixes_applied.append(f"Replaced '{wrong}' with '{correct}'")
        elif error.suggestion:
            # Use the suggestion from difflib
            # Be careful: only replace in the context of the settings access
            old_pattern = f".{error.attribute}"
            new_pattern = f".{error.suggestion}"
            if old_pattern in fixed:
                fixed = fixed.replace(old_pattern, new_pattern)
                fixes_applied.append(
                    f"Line {error.line}: Replaced '{error.attribute}' with "
                    f"'{error.suggestion}' on {error.object_type}"
                )

    return fixed, fixes_applied


# =============================================================================
# BUILD SUBSTITUTION TABLE
# =============================================================================

def build_substitution_table(truth_pack: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    """
    Build a substitution table from the truth pack for auto-correction.

    Returns:
        Dict mapping type_name → {"_valid_props": [...]}
    """
    table = {}
    for type_name, type_data in truth_pack.items():
        if not isinstance(type_data, dict) or "properties" not in type_data:
            continue
        table[type_name] = {
            "_valid_props": sorted(type_data["properties"].keys()),
        }
    return table


# =============================================================================
# FORMAT TRUTH PACK FOR PROMPT
# =============================================================================

def format_truth_pack_for_prompt(truth_pack: Dict[str, Any]) -> str:
    """
    Format truth pack as dense LLM-optimized text for ScriptWriter instructions.

    Only includes VALID attributes with types, ranges, and defaults.
    Designed to be injected into the ScriptWriter's dynamic instructions.

    Args:
        truth_pack: Truth pack from build_truth_pack()

    Returns:
        Formatted string for prompt injection
    """
    sections = []
    sections.append("## TRUTH PACK — ONLY valid Blender 5.0 attributes")
    sections.append("Use ONLY attributes listed below. Any attribute NOT listed here is INVALID.")
    sections.append("")

    for type_name, type_data in sorted(truth_pack.items()):
        if not isinstance(type_data, dict) or "properties" not in type_data:
            if isinstance(type_data, dict) and "error" in type_data:
                sections.append(f"### {type_name}: NOT AVAILABLE ({type_data['error']})")
                sections.append("")
            continue

        props = type_data["properties"]
        if not props:
            continue

        # Group by writable vs read-only
        writable = {k: v for k, v in props.items() if not v.get("is_readonly", False)}
        readonly = {k: v for k, v in props.items() if v.get("is_readonly", False)}

        sections.append(f"### {type_name} ({len(writable)} writable, {len(readonly)} read-only)")

        # Show writable properties (these are what the script sets)
        if writable:
            for name, info in sorted(writable.items()):
                line = f"- {name}: {info['type']}"
                if "default" in info:
                    line += f" = {info['default']}"
                if "range" in info:
                    r = info["range"]
                    # Skip absurdly large ranges that aren't useful
                    if abs(r[0]) < 1e10 and abs(r[1]) < 1e10:
                        line += f" [{r[0]}, {r[1]}]"
                if "items" in info:
                    items = info["items"]
                    if len(items) <= 8:
                        line += f" {{{', '.join(items)}}}"
                    else:
                        line += f" {{{', '.join(items[:6])}, ...+{len(items)-6}}}"
                sections.append(line)

        sections.append("")

    return "\n".join(sections)


# =============================================================================
# TRUTH PACK → API SPEC CONVERSION
# =============================================================================

def truth_pack_to_api_spec(
    truth_pack: Dict[str, Any],
    effect_type: str,
    technique: str,
) -> "APISpec":
    """
    Convert a truth pack to an APISpec for backward compatibility.

    The existing Code Writer guardrail (validate_code_against_spec) expects
    an APISpec. This function populates one from the truth pack, preserving
    the guardrail chain.

    Args:
        truth_pack: Truth pack from build_truth_pack()
        effect_type: VFX effect type
        technique: Pipeline technique name

    Returns:
        APISpec populated from truth pack data
    """
    from models.api_spec import APISpec, APIAttribute, APIOperation

    domain_attrs = []
    flow_attrs = []
    scene_attrs = []
    object_attrs = []

    TYPE_TO_CATEGORY = {
        "FluidDomainSettings": "domain",
        "FluidFlowSettings": "flow",
        "FluidEffectorSettings": "domain",  # Group with domain
        "Scene": "scene",
        "RenderSettings": "scene",
        "CyclesRenderSettings": "scene",
        "Object": "object",
        "Camera": "object",
    }

    for type_name, type_data in truth_pack.items():
        if not isinstance(type_data, dict) or "properties" not in type_data:
            continue

        category = TYPE_TO_CATEGORY.get(type_name, "object")

        for prop_name, prop_info in type_data["properties"].items():
            if prop_info.get("is_readonly", False):
                continue

            attr = APIAttribute(
                object_type=type_name,
                attribute_name=prop_name,
                value_type=prop_info.get("type", "UNKNOWN"),
                doc_ref=f"blender_python_reference_5_0/bpy.types.{type_name}.html#{prop_name}",
                example_value=prop_info.get("default"),
                value_range=str(prop_info["range"]) if "range" in prop_info else None,
                enum_values=prop_info.get("items"),
            )

            if category == "domain":
                domain_attrs.append(attr)
            elif category == "flow":
                flow_attrs.append(attr)
            elif category == "scene":
                scene_attrs.append(attr)
            else:
                object_attrs.append(attr)

    return APISpec(
        effect_type=effect_type,
        technique=technique,
        domain_attributes=domain_attrs,
        flow_attributes=flow_attrs,
        scene_attributes=scene_attrs,
        object_attributes=object_attrs,
        ops=[
            APIOperation(
                op_path="bpy.ops.fluid.bake_all",
                doc_ref="blender_python_reference_5_0/bpy.ops.fluid.html#bpy.ops.fluid.bake_all",
            ),
        ],
        approach_doc_refs=[
            f"blender_python_reference_5_0/bpy.types.FluidDomainSettings.html",
        ],
    )
