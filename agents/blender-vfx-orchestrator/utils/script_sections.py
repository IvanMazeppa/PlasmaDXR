"""Script section parsing, extraction, and replacement.

Blender VFX scripts use canonical named functions as sections:
  setup_scene, create_geometry, setup_materials, setup_physics,
  setup_lighting, setup_camera, bake_and_render

This module provides AST-based tools to parse, extract, replace, and
validate these sections without touching the rest of the script.
"""

from __future__ import annotations

import ast
from typing import Any, Optional

CANONICAL_SECTIONS = [
    "setup_scene",
    "create_geometry",
    "setup_materials",
    "setup_physics",
    "setup_lighting",
    "setup_camera",
    "bake_and_render",
]


def parse_sections(source: str) -> dict[str, dict[str, Any]]:
    """Parse a script and return canonical sections found.

    Returns dict keyed by section name with:
      - start_line: first line number (1-indexed)
      - end_line: last line number (1-indexed)
      - source: the function source text
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    lines = source.splitlines(keepends=True)
    sections: dict[str, dict[str, Any]] = {}

    # Collect all top-level function defs with canonical names
    func_defs = [
        node for node in ast.iter_child_nodes(tree)
        if isinstance(node, ast.FunctionDef) and node.name in CANONICAL_SECTIONS
    ]

    for i, func in enumerate(func_defs):
        start = func.lineno  # 1-indexed
        end = func.end_lineno  # 1-indexed, inclusive
        if end is None:
            # Fallback: use next function start or EOF
            if i + 1 < len(func_defs):
                end = func_defs[i + 1].lineno - 1
            else:
                end = len(lines)

        func_source = "".join(lines[start - 1 : end])
        sections[func.name] = {
            "start_line": start,
            "end_line": end,
            "source": func_source,
        }

    return sections


def extract_section(source: str, section_name: str) -> Optional[str]:
    """Extract a single section's source code. Returns None if not found."""
    sections = parse_sections(source)
    if section_name not in sections:
        return None
    return sections[section_name]["source"]


def replace_section(source: str, section_name: str, new_source: str) -> str:
    """Replace a section's source code, preserving everything else.

    Args:
        source: Full script source
        section_name: Canonical section name to replace
        new_source: New function definition (must include `def` line)

    Returns:
        Modified script source

    Raises:
        KeyError: If section_name not found in script
    """
    sections = parse_sections(source)
    if section_name not in sections:
        raise KeyError(
            f"Section '{section_name}' not found. "
            f"Available: {list(sections.keys())}"
        )

    section = sections[section_name]
    lines = source.splitlines(keepends=True)

    # Ensure new_source ends with newline
    if not new_source.endswith("\n"):
        new_source += "\n"

    # Replace lines [start-1 : end] with new source
    before = lines[: section["start_line"] - 1]
    after = lines[section["end_line"] :]

    return "".join(before) + new_source + "".join(after)


def validate_section_structure(source: str) -> tuple[list[str], list[str]]:
    """Check which canonical sections are present/missing.

    Returns:
        (missing_sections, found_sections)
    """
    sections = parse_sections(source)
    found = [name for name in CANONICAL_SECTIONS if name in sections]
    missing = [name for name in CANONICAL_SECTIONS if name not in sections]
    return missing, found


def list_sections(source: str) -> list[dict[str, Any]]:
    """List all canonical sections found with their line ranges.

    Returns list of dicts with: name, start_line, end_line, line_count
    """
    sections = parse_sections(source)
    result = []
    for name in CANONICAL_SECTIONS:
        if name in sections:
            s = sections[name]
            result.append({
                "name": name,
                "start_line": s["start_line"],
                "end_line": s["end_line"],
                "line_count": s["end_line"] - s["start_line"] + 1,
            })
    return result
