"""
Knowledge Distillation Tools for Blender VFX Orchestrator.

Implements Strategy 2: Knowledge Distillation Loop.

These tools transform successful experiment outcomes into reusable code patterns
by analyzing script diffs and extracting the meaningful changes. Unlike Strategy 4
which stores patterns manually, these tools automatically extract patterns from
the difference between original and improved scripts.

Key capabilities:
- Extract patterns from script modifications (diff analysis)
- Generalize patterns into templates with parameters
- Apply stored patterns to new scripts
- Track pattern lineage and effectiveness
"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from agents import function_tool

from utils.code_pattern_memory import get_pattern_memory, CodePattern


# =============================================================================
# DIFF ANALYSIS HELPERS
# =============================================================================

def _extract_diff_chunks(original: str, modified: str) -> List[Dict[str, Any]]:
    """
    Extract meaningful change chunks from a unified diff.

    Returns list of chunks, each containing:
    - type: "addition", "deletion", "modification"
    - original_lines: Lines from original (for modifications/deletions)
    - new_lines: Lines added/modified
    - context_before: Lines before the change
    - context_after: Lines after the change
    - line_number: Approximate line number in modified file
    """
    original_lines = original.splitlines(keepends=True)
    modified_lines = modified.splitlines(keepends=True)

    # Generate unified diff
    diff = list(difflib.unified_diff(
        original_lines,
        modified_lines,
        lineterm='',
        n=3  # Context lines
    ))

    chunks = []
    current_chunk = None
    line_num = 0

    for line in diff:
        if line.startswith('@@'):
            # Parse hunk header for line numbers
            match = re.search(r'@@ -\d+(?:,\d+)? \+(\d+)', line)
            if match:
                line_num = int(match.group(1))
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = {
                "type": "modification",
                "original_lines": [],
                "new_lines": [],
                "context_before": [],
                "context_after": [],
                "line_number": line_num
            }
        elif line.startswith('---') or line.startswith('+++'):
            continue  # Skip file headers
        elif current_chunk is not None:
            if line.startswith('-'):
                current_chunk["original_lines"].append(line[1:])
            elif line.startswith('+'):
                current_chunk["new_lines"].append(line[1:])
            elif line.startswith(' '):
                # Context line
                if current_chunk["new_lines"] or current_chunk["original_lines"]:
                    current_chunk["context_after"].append(line[1:])
                else:
                    current_chunk["context_before"].append(line[1:])

    if current_chunk and (current_chunk["new_lines"] or current_chunk["original_lines"]):
        chunks.append(current_chunk)

    # Classify chunk types
    for chunk in chunks:
        if not chunk["original_lines"]:
            chunk["type"] = "addition"
        elif not chunk["new_lines"]:
            chunk["type"] = "deletion"
        else:
            chunk["type"] = "modification"

    return chunks


def _extract_blender_settings(code: str) -> List[Dict[str, Any]]:
    """
    Extract Blender property assignments from code.

    Returns list of settings found, each with:
    - property: Full property path (e.g., "domain.flame_smoke")
    - value: The assigned value
    - line: The full line of code
    """
    settings = []

    # Pattern for property assignments
    # Matches: obj.property = value or obj.sub.property = value
    pattern = r'(\w+(?:\.\w+)+)\s*=\s*([^#\n]+)'

    for match in re.finditer(pattern, code):
        prop_path = match.group(1)
        value = match.group(2).strip()

        # Skip common non-setting assignments
        if any(skip in prop_path for skip in ['.name', '.data', '.type', '.location', '.rotation', '.scale']):
            continue

        settings.append({
            "property": prop_path,
            "value": value,
            "line": match.group(0)
        })

    return settings


def _generalize_pattern(
    code_snippet: str,
    issue: str,
    effect_type: str
) -> Dict[str, Any]:
    """
    Generalize a code snippet into a parameterized template.

    Replaces hardcoded values with template parameters where appropriate.
    """
    template = code_snippet
    parameters = {}

    # Extract settings from the code
    settings = _extract_blender_settings(code_snippet)

    for i, setting in enumerate(settings):
        param_name = setting["property"].split(".")[-1]

        # Try to parse the value
        value = setting["value"]
        try:
            # Check if it's a number
            if '.' in value:
                float_val = float(value)
                parameters[param_name] = {
                    "default": float_val,
                    "type": "float",
                    "property": setting["property"]
                }
                # Replace with template variable
                template = template.replace(
                    f'{setting["property"]} = {value}',
                    f'{setting["property"]} = {{{param_name}}}'
                )
            else:
                int_val = int(value)
                parameters[param_name] = {
                    "default": int_val,
                    "type": "int",
                    "property": setting["property"]
                }
                template = template.replace(
                    f'{setting["property"]} = {value}',
                    f'{setting["property"]} = {{{param_name}}}'
                )
        except ValueError:
            # Not a simple number, keep as-is
            pass

    return {
        "template": template,
        "parameters": parameters,
        "original_code": code_snippet,
        "settings_count": len(settings)
    }


def _find_insertion_point(script: str, pattern_code: str) -> Optional[int]:
    """
    Find the best insertion point for pattern code in a script.

    Looks for similar context or appropriate section.
    """
    script_lines = script.splitlines()

    # Extract key identifiers from pattern (like domain, fluid, etc.)
    pattern_objects = set(re.findall(r'(\w+)\.', pattern_code))

    # Find where these objects are first used/defined in script
    for i, line in enumerate(script_lines):
        for obj in pattern_objects:
            if f'{obj} =' in line or f'{obj}.' in line:
                # Found relevant section, insert after this block
                # Look for end of the logical block (empty line or different object)
                for j in range(i + 1, len(script_lines)):
                    if not script_lines[j].strip() or (
                        script_lines[j].strip() and
                        not any(o in script_lines[j] for o in pattern_objects)
                    ):
                        return j
                return i + 1

    # Default: insert before the last few lines (usually baking/rendering)
    return max(0, len(script_lines) - 10)


# =============================================================================
# FUNCTION TOOLS
# =============================================================================

@function_tool
def extract_successful_pattern(
    original_script_path: str,
    modified_script_path: str,
    issue_fixed: str,
    improvement_delta: float,
    effect_type: str,
    experiment_id: str
) -> str:
    """
    Extract a reusable pattern from a successful script modification.

    Analyzes the diff between original and modified scripts to identify
    the specific code changes that fixed the issue. Generalizes these
    into a reusable pattern that can be applied to future scripts.

    WHEN TO USE:
    - After a script modification achieves >= 5 point improvement
    - When quality analyst confirms the fix worked
    - Before moving to the next iteration

    This is more automated than record_code_pattern() - it extracts
    the pattern from the file diff rather than requiring you to
    specify the code manually.

    Args:
        original_script_path: Path to script BEFORE modification
        modified_script_path: Path to script AFTER successful modification
        issue_fixed: Description of issue that was fixed
                    Example: "smoke too thin and transparent"
        improvement_delta: Score improvement (e.g., 15.0 for +15 points)
        effect_type: Type of effect (pyro, explosion, fire, smoke, nebula, sun)
        experiment_id: ID of the experiment this came from

    Returns:
        JSON with:
        - success: Whether pattern was extracted
        - pattern_id: Unique identifier for this pattern
        - pattern_name: Human-readable name
        - changes_found: Number of meaningful changes detected
        - code_snippet: The extracted code
        - template: Parameterized version of the code
        - parameters: Extracted parameters with defaults
        - confidence: Initial confidence score
    """
    memory = get_pattern_memory()

    try:
        # Read both scripts
        original_path = Path(original_script_path)
        modified_path = Path(modified_script_path)

        if not original_path.exists():
            return json.dumps({
                "success": False,
                "error": f"Original script not found: {original_script_path}"
            })

        if not modified_path.exists():
            return json.dumps({
                "success": False,
                "error": f"Modified script not found: {modified_script_path}"
            })

        with open(original_path) as f:
            original = f.read()
        with open(modified_path) as f:
            modified = f.read()

        # Extract diff chunks
        chunks = _extract_diff_chunks(original, modified)

        if not chunks:
            return json.dumps({
                "success": False,
                "error": "No meaningful changes detected between scripts",
                "hint": "Scripts may be identical or changes too minor"
            })

        # Combine meaningful additions/modifications into code snippet
        code_lines = []
        context_before = []
        context_after = []

        for chunk in chunks:
            if chunk["type"] in ("addition", "modification"):
                code_lines.extend(chunk["new_lines"])
                if not context_before and chunk["context_before"]:
                    context_before = chunk["context_before"][-3:]  # Last 3 context lines
                if chunk["context_after"]:
                    context_after = chunk["context_after"][:3]  # First 3 context lines

        if not code_lines:
            return json.dumps({
                "success": False,
                "error": "No additions or modifications found",
                "changes_found": len(chunks),
                "chunk_types": [c["type"] for c in chunks]
            })

        code_snippet = "".join(code_lines).strip()

        # Generalize into template
        generalized = _generalize_pattern(code_snippet, issue_fixed, effect_type)

        # Store in pattern memory
        pattern_id = memory.record_successful_pattern(
            issue=issue_fixed,
            code_snippet=code_snippet,
            effect_type=effect_type,
            improvement=improvement_delta,
            experiment_id=experiment_id,
            context_before="".join(context_before),
            context_after="".join(context_after)
        )

        pattern = memory.get_pattern(pattern_id)

        return json.dumps({
            "success": True,
            "pattern_id": pattern_id,
            "pattern_name": pattern.name if pattern else f"{effect_type}_fix",
            "changes_found": len(chunks),
            "code_snippet": code_snippet,
            "template": generalized["template"],
            "parameters": generalized["parameters"],
            "settings_modified": generalized["settings_count"],
            "confidence": pattern.confidence if pattern else 50,
            "context": {
                "before": "".join(context_before),
                "after": "".join(context_after)
            },
            "message": f"Extracted pattern with {len(chunks)} changes, {generalized['settings_count']} settings"
        })

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "message": f"Failed to extract pattern: {e}"
        })


@function_tool
def apply_pattern_to_script(
    script_path: str,
    pattern_id: str,
    output_path: str = "",
    parameter_overrides: str = "{}"
) -> str:
    """
    Apply a stored pattern to an existing script.

    Retrieves a pattern from the library and applies it to the script,
    substituting any provided parameters. Creates a new file with the
    modifications.

    WHEN TO USE:
    - After search_code_patterns() finds a high-confidence match
    - When you want to apply a proven fix without manually copying code
    - For batch-applying patterns to multiple scripts

    Args:
        script_path: Path to script to modify
        pattern_id: ID of pattern to apply (from search_code_patterns)
        output_path: Path for modified script (default: adds "_patched" suffix)
        parameter_overrides: JSON object of parameter overrides
                            Example: '{"flame_smoke": 4.0, "dissolve_speed": 8}'

    Returns:
        JSON with:
        - success: Whether pattern was applied
        - output_path: Path to modified script
        - pattern_applied: Name of the pattern
        - changes_made: Description of what was added/modified
        - insertion_point: Line number where code was inserted
        - parameters_used: Final parameter values used
    """
    memory = get_pattern_memory()

    try:
        # Load pattern
        pattern = memory.get_pattern(pattern_id)
        if not pattern:
            return json.dumps({
                "success": False,
                "error": f"Pattern {pattern_id} not found"
            })

        # Read script
        script_file = Path(script_path)
        if not script_file.exists():
            return json.dumps({
                "success": False,
                "error": f"Script not found: {script_path}"
            })

        with open(script_file) as f:
            script = f.read()

        # Parse parameter overrides
        try:
            overrides = json.loads(parameter_overrides) if parameter_overrides else {}
        except json.JSONDecodeError:
            overrides = {}

        # Get the code to insert
        code_to_insert = pattern.code_snippet

        # Apply any parameter overrides if the pattern was generalized
        # (This is a simple string replacement - could be enhanced)
        for param_name, param_value in overrides.items():
            # Look for {param_name} placeholders or direct values
            code_to_insert = re.sub(
                rf'\{{{param_name}\}}',
                str(param_value),
                code_to_insert
            )

        # Find insertion point
        insertion_line = _find_insertion_point(script, code_to_insert)

        # Insert the code
        script_lines = script.splitlines(keepends=True)

        # Add comment indicating pattern application
        pattern_comment = f"\n# Applied pattern: {pattern.name} (ID: {pattern_id})\n"
        pattern_comment += f"# Fixes: {pattern.issue_category}\n"

        # Insert with proper indentation
        indent = ""
        if insertion_line > 0 and script_lines[insertion_line - 1].strip():
            # Match indentation of surrounding code
            match = re.match(r'^(\s*)', script_lines[insertion_line - 1])
            if match:
                indent = match.group(1)

        # Indent the pattern code
        indented_code = "\n".join(
            indent + line if line.strip() else line
            for line in code_to_insert.splitlines()
        )

        # Insert
        script_lines.insert(insertion_line, pattern_comment + indented_code + "\n\n")
        modified_script = "".join(script_lines)

        # Determine output path
        if not output_path:
            output_path = str(script_file.with_stem(script_file.stem + "_patched"))

        # Write modified script
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(modified_script)

        return json.dumps({
            "success": True,
            "output_path": str(output_file),
            "pattern_applied": pattern.name,
            "pattern_id": pattern_id,
            "issue_addressed": pattern.issue_category,
            "changes_made": f"Inserted {len(code_to_insert.splitlines())} lines of code",
            "insertion_point": insertion_line,
            "parameters_used": overrides if overrides else "defaults",
            "expected_improvement": pattern.average_improvement,
            "pattern_confidence": pattern.confidence,
            "message": f"Applied pattern '{pattern.name}' at line {insertion_line}"
        })

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "message": f"Failed to apply pattern: {e}"
        })


@function_tool
def analyze_script_for_patterns(
    script_path: str,
    effect_type: str = ""
) -> str:
    """
    Analyze a script to find which stored patterns might improve it.

    Examines the script structure and compares against known patterns
    to suggest potential improvements.

    WHEN TO USE:
    - At the start of an iteration to see what patterns could help
    - When reviewing a script before execution
    - To understand what optimizations are available

    Args:
        script_path: Path to script to analyze
        effect_type: Optional filter by effect type

    Returns:
        JSON with:
        - script_analyzed: Path to the script
        - current_settings: Settings found in the script
        - applicable_patterns: Patterns that could be applied
        - recommendations: Suggested improvements
    """
    memory = get_pattern_memory()

    try:
        script_file = Path(script_path)
        if not script_file.exists():
            return json.dumps({
                "success": False,
                "error": f"Script not found: {script_path}"
            })

        with open(script_file) as f:
            script = f.read()

        # Extract current settings
        current_settings = _extract_blender_settings(script)

        # Get all patterns
        patterns = memory.list_patterns(
            effect_type=effect_type if effect_type else None,
            min_confidence=30.0
        )

        # Find applicable patterns
        applicable = []
        current_props = {s["property"].split(".")[-1] for s in current_settings}

        for pattern in patterns:
            # Check if pattern addresses different settings
            pattern_props = set(pattern.parameters_affected)

            # Pattern is applicable if:
            # 1. It modifies settings not currently in script, OR
            # 2. It's high confidence and modifies similar settings
            new_props = pattern_props - current_props
            overlap_props = pattern_props & current_props

            if new_props or (pattern.confidence >= 60 and overlap_props):
                applicable.append({
                    "pattern_id": pattern.pattern_id,
                    "name": pattern.name,
                    "issue_addressed": pattern.issue_category,
                    "confidence": pattern.confidence,
                    "new_settings": list(new_props),
                    "overlapping_settings": list(overlap_props),
                    "expected_improvement": pattern.average_improvement
                })

        # Sort by confidence
        applicable.sort(key=lambda p: p["confidence"], reverse=True)

        # Generate recommendations
        recommendations = []
        if applicable:
            top = applicable[0]
            if top["confidence"] >= 70:
                recommendations.append(
                    f"HIGH: Apply '{top['name']}' (confidence {top['confidence']:.0f}%)"
                )
            elif top["confidence"] >= 50:
                recommendations.append(
                    f"MEDIUM: Consider '{top['name']}' (confidence {top['confidence']:.0f}%)"
                )

        if not applicable:
            recommendations.append("No applicable patterns found. Novel approach needed.")

        return json.dumps({
            "success": True,
            "script_analyzed": str(script_file),
            "current_settings": current_settings[:20],  # Limit output
            "settings_count": len(current_settings),
            "applicable_patterns": applicable[:10],  # Top 10
            "patterns_found": len(applicable),
            "recommendations": recommendations
        })

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


@function_tool
def compare_scripts(
    script_a_path: str,
    script_b_path: str
) -> str:
    """
    Compare two scripts to understand their differences.

    Useful for understanding what changed between iterations or
    comparing different approaches to the same effect.

    Args:
        script_a_path: Path to first script (usually original/baseline)
        script_b_path: Path to second script (usually modified/improved)

    Returns:
        JSON with:
        - chunks_changed: Number of change chunks
        - additions: Lines added
        - deletions: Lines removed
        - modifications: Lines modified
        - settings_changed: Blender settings that differ
        - summary: Human-readable summary
    """
    try:
        path_a = Path(script_a_path)
        path_b = Path(script_b_path)

        if not path_a.exists():
            return json.dumps({"success": False, "error": f"Script A not found: {script_a_path}"})
        if not path_b.exists():
            return json.dumps({"success": False, "error": f"Script B not found: {script_b_path}"})

        with open(path_a) as f:
            script_a = f.read()
        with open(path_b) as f:
            script_b = f.read()

        # Get diff chunks
        chunks = _extract_diff_chunks(script_a, script_b)

        # Categorize changes
        additions = []
        deletions = []
        modifications = []

        for chunk in chunks:
            if chunk["type"] == "addition":
                additions.extend(chunk["new_lines"])
            elif chunk["type"] == "deletion":
                deletions.extend(chunk["original_lines"])
            else:
                modifications.append({
                    "removed": chunk["original_lines"],
                    "added": chunk["new_lines"]
                })

        # Extract settings from both
        settings_a = _extract_blender_settings(script_a)
        settings_b = _extract_blender_settings(script_b)

        # Find setting differences
        props_a = {s["property"]: s["value"] for s in settings_a}
        props_b = {s["property"]: s["value"] for s in settings_b}

        settings_changed = []
        all_props = set(props_a.keys()) | set(props_b.keys())

        for prop in all_props:
            val_a = props_a.get(prop, "NOT SET")
            val_b = props_b.get(prop, "NOT SET")
            if val_a != val_b:
                settings_changed.append({
                    "property": prop,
                    "script_a": val_a,
                    "script_b": val_b
                })

        # Generate summary
        summary_parts = []
        if additions:
            summary_parts.append(f"{len(additions)} lines added")
        if deletions:
            summary_parts.append(f"{len(deletions)} lines removed")
        if modifications:
            summary_parts.append(f"{len(modifications)} sections modified")
        if settings_changed:
            summary_parts.append(f"{len(settings_changed)} settings changed")

        return json.dumps({
            "success": True,
            "script_a": str(path_a),
            "script_b": str(path_b),
            "chunks_changed": len(chunks),
            "additions": len(additions),
            "deletions": len(deletions),
            "modifications": len(modifications),
            "settings_changed": settings_changed,
            "summary": ", ".join(summary_parts) if summary_parts else "No significant changes",
            "added_code": "".join(additions)[:1000] if additions else None,  # Truncate
            "removed_code": "".join(deletions)[:1000] if deletions else None
        })

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


# =============================================================================
# DIRECT CALLABLE VERSIONS (for non-agent code)
# =============================================================================

def extract_pattern_from_diff(
    original: str,
    modified: str,
    issue: str,
    effect_type: str
) -> Tuple[str, Dict[str, Any]]:
    """
    Direct callable version for extracting patterns from script content.

    Returns:
        Tuple of (code_snippet, generalized_template_dict)
    """
    chunks = _extract_diff_chunks(original, modified)

    code_lines = []
    for chunk in chunks:
        if chunk["type"] in ("addition", "modification"):
            code_lines.extend(chunk["new_lines"])

    code_snippet = "".join(code_lines).strip()
    generalized = _generalize_pattern(code_snippet, issue, effect_type)

    return code_snippet, generalized


def find_applicable_patterns(
    script: str,
    effect_type: str = ""
) -> List[CodePattern]:
    """
    Direct callable version for finding applicable patterns.
    """
    memory = get_pattern_memory()

    current_settings = _extract_blender_settings(script)
    current_props = {s["property"].split(".")[-1] for s in current_settings}

    patterns = memory.list_patterns(
        effect_type=effect_type if effect_type else None,
        min_confidence=30.0
    )

    applicable = []
    for pattern in patterns:
        pattern_props = set(pattern.parameters_affected)
        new_props = pattern_props - current_props

        if new_props or pattern.confidence >= 60:
            applicable.append(pattern)

    return applicable
