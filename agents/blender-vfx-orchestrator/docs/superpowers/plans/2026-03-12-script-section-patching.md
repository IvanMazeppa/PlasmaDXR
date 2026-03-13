# Script Section Patching Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace destructive full-script rewrites with targeted section-level patching, so iterations can fix one aspect (e.g., lighting) without losing working code (e.g., cloth physics setup).

**Architecture:** Generated scripts must use canonical named functions (`setup_scene`, `create_geometry`, `setup_materials`, `setup_physics`, `setup_lighting`, `setup_camera`, `bake_and_render`). A new `patch_script_section` tool replaces a single function body while preserving all others. The Modification Coordinator's `modify_code` path is rewired to use section patching instead of full rewrites. A section-naming guardrail on `write_script` enforces the structure. Branch budget tracking limits repair attempts (max 2 per iteration, max 4 per session) to prevent thrashing.

**Tech Stack:** Python 3.12, OpenAI Agents SDK v0.10.5, AST parsing, `TechniqueContract` capability packs, existing `write_script`/`modify_script`/`validate_script` tools

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `utils/script_sections.py` | Section parsing, extraction, replacement, validation |
| Create | `tests/test_script_sections.py` | Unit tests for section parsing/patching |
| Modify | `tools/script_generator_tools.py:1444-1489` | Add `patch_script_section` tool + section-naming guardrail on `write_script` |
| Modify | `orchestrator.py:1720-1777` | Rewire `modify_code` path to use section patching |
| Modify | `models/shared_context.py` | Add `patch_budget` tracking fields to `SessionState` |
| Modify | `models/pipeline_models.py` | Add `CANONICAL_SECTIONS` constant and `ScriptOutput.sections_found` field |

---

## Chunk 1: Section Parser and Patcher

### Task 1: Section Parser Core (`utils/script_sections.py`)

**Files:**
- Create: `utils/script_sections.py`
- Create: `tests/test_script_sections.py`

The section parser uses Python AST to find top-level function definitions matching canonical names and extract/replace their source. This is more reliable than regex since generated scripts have varying indentation and comment styles.

**Canonical section names:**
```python
CANONICAL_SECTIONS = [
    "setup_scene",        # Scene cleanup, render settings, world background
    "create_geometry",    # All mesh/curve creation
    "setup_materials",    # Material/shader definitions
    "setup_physics",      # Physics modifiers, force fields, constraints
    "setup_lighting",     # Lights, HDRI, environment lighting
    "setup_camera",       # Camera placement, lens settings
    "bake_and_render",    # Cache bake, render call, file output
]
```

Scripts also have a preamble (imports, constants, helpers) and an epilogue (the `main()` call or sequential calls at module level). Sections don't need to cover 100% of the script — material builder helpers, utility functions, and constants live outside sections.

- [ ] **Step 1: Write failing tests for section detection**

```python
# tests/test_script_sections.py
"""Tests for script section parsing, extraction, and patching."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from utils.script_sections import (
    CANONICAL_SECTIONS,
    parse_sections,
    extract_section,
    replace_section,
    validate_section_structure,
    list_sections,
)


MINIMAL_SECTIONED_SCRIPT = '''
import bpy

ASSET_NAME = "test_asset"

def setup_scene():
    """Scene cleanup and render settings."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.context.scene.render.engine = 'CYCLES'

def create_geometry():
    """Create meshes."""
    bpy.ops.mesh.primitive_cube_add(size=1)

def setup_materials():
    """Create materials."""
    mat = bpy.data.materials.new("Base")
    mat.use_nodes = True

def setup_physics():
    """Physics setup."""
    pass

def setup_lighting():
    """Lighting."""
    bpy.ops.object.light_add(type='SUN')

def setup_camera():
    """Camera."""
    bpy.ops.object.camera_add()

def bake_and_render():
    """Bake and render."""
    bpy.context.scene.render.filepath = "/tmp/test.png"
    bpy.ops.render.render(write_still=True)

# Main execution
setup_scene()
create_geometry()
setup_materials()
setup_physics()
setup_lighting()
setup_camera()
bake_and_render()
'''

UNSECTIONED_SCRIPT = '''
import bpy

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
bpy.ops.mesh.primitive_cube_add(size=1)
bpy.ops.render.render(write_still=True)
'''

PARTIAL_SECTIONED_SCRIPT = '''
import bpy

def setup_scene():
    bpy.ops.object.select_all(action='SELECT')

def setup_lighting():
    bpy.ops.object.light_add(type='SUN')

# Inline code — no section
bpy.ops.mesh.primitive_cube_add(size=1)
bpy.ops.render.render(write_still=True)
'''


class TestParseSections:
    def test_finds_all_canonical_sections(self):
        sections = parse_sections(MINIMAL_SECTIONED_SCRIPT)
        assert len(sections) == 7
        for name in CANONICAL_SECTIONS:
            assert name in sections

    def test_unsectioned_returns_empty(self):
        sections = parse_sections(UNSECTIONED_SCRIPT)
        assert len(sections) == 0

    def test_partial_finds_existing_only(self):
        sections = parse_sections(PARTIAL_SECTIONED_SCRIPT)
        assert "setup_scene" in sections
        assert "setup_lighting" in sections
        assert "create_geometry" not in sections

    def test_section_has_line_range(self):
        sections = parse_sections(MINIMAL_SECTIONED_SCRIPT)
        scene = sections["setup_scene"]
        assert "start_line" in scene
        assert "end_line" in scene
        assert scene["start_line"] < scene["end_line"]

    def test_section_has_source(self):
        sections = parse_sections(MINIMAL_SECTIONED_SCRIPT)
        scene = sections["setup_scene"]
        assert "source" in scene
        assert "bpy.ops.object.select_all" in scene["source"]


class TestExtractSection:
    def test_extract_existing(self):
        source = extract_section(MINIMAL_SECTIONED_SCRIPT, "setup_lighting")
        assert "light_add" in source

    def test_extract_missing_returns_none(self):
        source = extract_section(MINIMAL_SECTIONED_SCRIPT, "nonexistent_section")
        assert source is None

    def test_extract_from_unsectioned_returns_none(self):
        source = extract_section(UNSECTIONED_SCRIPT, "setup_scene")
        assert source is None


class TestReplaceSection:
    def test_replace_preserves_other_sections(self):
        new_lighting = """def setup_lighting():
    \"\"\"Better lighting.\"\"\"
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 5))
    light = bpy.context.active_object
    light.data.energy = 500
"""
        result = replace_section(MINIMAL_SECTIONED_SCRIPT, "setup_lighting", new_lighting)
        # New section present
        assert "type='AREA'" in result
        assert "energy = 500" in result
        # Old sections preserved
        assert "def setup_scene():" in result
        assert "def create_geometry():" in result
        assert "def setup_physics():" in result
        assert "def setup_camera():" in result
        assert "def bake_and_render():" in result
        # Old lighting gone
        assert "type='SUN'" not in result

    def test_replace_missing_section_raises(self):
        with pytest.raises(KeyError):
            replace_section(MINIMAL_SECTIONED_SCRIPT, "nonexistent", "def nonexistent(): pass")

    def test_replace_keeps_preamble(self):
        new_scene = """def setup_scene():
    pass
"""
        result = replace_section(MINIMAL_SECTIONED_SCRIPT, "setup_scene", new_scene)
        assert "import bpy" in result
        assert 'ASSET_NAME = "test_asset"' in result

    def test_replace_is_syntactically_valid(self):
        new_lighting = """def setup_lighting():
    bpy.ops.object.light_add(type='POINT')
"""
        result = replace_section(MINIMAL_SECTIONED_SCRIPT, "setup_lighting", new_lighting)
        compile(result, "<test>", "exec")  # Should not raise


class TestValidateSectionStructure:
    def test_fully_sectioned_passes(self):
        missing, found = validate_section_structure(MINIMAL_SECTIONED_SCRIPT)
        assert len(missing) == 0
        assert len(found) == 7

    def test_unsectioned_reports_all_missing(self):
        missing, found = validate_section_structure(UNSECTIONED_SCRIPT)
        assert len(missing) == 7
        assert len(found) == 0

    def test_partial_reports_missing(self):
        missing, found = validate_section_structure(PARTIAL_SECTIONED_SCRIPT)
        assert "create_geometry" in missing
        assert "setup_scene" in found


class TestListSections:
    def test_list_returns_names_and_lines(self):
        result = list_sections(MINIMAL_SECTIONED_SCRIPT)
        assert len(result) == 7
        assert all("name" in s and "start_line" in s for s in result)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_script_sections.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'utils.script_sections'`

- [ ] **Step 3: Implement `utils/script_sections.py`**

```python
"""Script section parsing, extraction, and replacement.

Blender VFX scripts use canonical named functions as sections:
  setup_scene, create_geometry, setup_materials, setup_physics,
  setup_lighting, setup_camera, bake_and_render

This module provides AST-based tools to parse, extract, replace, and
validate these sections without touching the rest of the script.
"""

from __future__ import annotations

import ast
import sys
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_script_sections.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add utils/script_sections.py tests/test_script_sections.py
git commit -m "feat: add script section parser for AST-based section extraction and replacement"
```

---

## Chunk 2: `patch_script_section` Tool and Section Guardrail

### Task 2: Add `patch_script_section` Tool

**Files:**
- Modify: `tools/script_generator_tools.py:1444-1489` (add new tool after `write_script`)
- Create: `tests/test_patch_tool.py`

The `patch_script_section` tool is exposed to agents as a `@function_tool`. It reads a script, replaces one section, validates the result compiles, runs truth pack on it, and writes the patched file.

- [ ] **Step 1: Write failing tests for patch tool**

```python
# tests/test_patch_tool.py
"""Tests for the patch_script_section tool."""

import sys
import os
import json
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import asyncio
from pathlib import Path
from utils.script_sections import CANONICAL_SECTIONS


# Test the implementation function directly (not the @function_tool wrapper)
from tools.script_generator_tools import _patch_script_section_impl


SECTIONED_SCRIPT = '''import bpy

ASSET_NAME = "test_patch"
OUTPUT_DIR = "/tmp/test_patch"

def setup_scene():
    """Scene setup."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def create_geometry():
    """Geometry."""
    bpy.ops.mesh.primitive_cube_add(size=1)

def setup_materials():
    """Materials."""
    mat = bpy.data.materials.new("Mat")

def setup_physics():
    """Physics."""
    pass

def setup_lighting():
    """Lighting."""
    bpy.ops.object.light_add(type='SUN')

def setup_camera():
    """Camera."""
    bpy.ops.object.camera_add()

def bake_and_render():
    """Render."""
    bpy.context.scene.render.filepath = OUTPUT_DIR + "/test.png"
    bpy.ops.render.render(write_still=True)

setup_scene()
create_geometry()
setup_materials()
setup_physics()
setup_lighting()
setup_camera()
bake_and_render()
'''


class TestPatchScriptSection:
    @pytest.fixture
    def script_file(self, tmp_path):
        p = tmp_path / "test_script.py"
        p.write_text(SECTIONED_SCRIPT)
        return str(p)

    def test_patch_replaces_section(self, script_file):
        new_lighting = """def setup_lighting():
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 5))
    light = bpy.context.active_object
    light.data.energy = 500"""
        result_json = _patch_script_section_impl(
            script_path=script_file,
            section_name="setup_lighting",
            new_code=new_lighting,
        )
        result = json.loads(result_json)
        assert result["success"] is True
        assert result["section_patched"] == "setup_lighting"
        # Read the output file
        patched = Path(result["script_path"]).read_text()
        assert "type='AREA'" in patched
        assert "energy = 500" in patched
        # Other sections preserved
        assert "def setup_scene():" in patched
        assert "def create_geometry():" in patched

    def test_patch_invalid_section_fails(self, script_file):
        result_json = _patch_script_section_impl(
            script_path=script_file,
            section_name="nonexistent_section",
            new_code="def nonexistent_section(): pass",
        )
        result = json.loads(result_json)
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_patch_syntax_error_fails(self, script_file):
        bad_code = "def setup_lighting(\n    this is not valid python"
        result_json = _patch_script_section_impl(
            script_path=script_file,
            section_name="setup_lighting",
            new_code=bad_code,
        )
        result = json.loads(result_json)
        assert result["success"] is False
        assert "syntax" in result["error"].lower() or "compile" in result["error"].lower()

    def test_patch_creates_named_output(self, script_file):
        new_code = "def setup_lighting():\n    pass\n"
        result_json = _patch_script_section_impl(
            script_path=script_file,
            section_name="setup_lighting",
            new_code=new_code,
            output_name="test_patched_v2",
        )
        result = json.loads(result_json)
        assert result["success"] is True
        assert "test_patched_v2" in result["script_path"]

    def test_patch_preserves_preamble_and_epilogue(self, script_file):
        new_code = "def setup_physics():\n    bpy.ops.object.modifier_add(type='CLOTH')\n"
        result_json = _patch_script_section_impl(
            script_path=script_file,
            section_name="setup_physics",
            new_code=new_code,
        )
        result = json.loads(result_json)
        patched = Path(result["script_path"]).read_text()
        # Preamble
        assert "import bpy" in patched
        assert 'ASSET_NAME = "test_patch"' in patched
        # Epilogue
        assert "setup_scene()" in patched
        assert "bake_and_render()" in patched
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_patch_tool.py -v`
Expected: FAIL — `ImportError: cannot import name '_patch_script_section_impl'`

- [ ] **Step 3: Implement `_patch_script_section_impl` and `patch_script_section` tool**

Add to `tools/script_generator_tools.py` after the `write_script` function (after line ~1489):

```python
# =============================================================================
# Section-level patching (Wave 2)
# =============================================================================

def _patch_script_section_impl(
    script_path: str,
    section_name: str,
    new_code: str,
    output_name: Optional[str] = None,
) -> str:
    """Replace a single canonical section in a script, preserving everything else.

    Args:
        script_path: Path to the script to patch
        section_name: Canonical section name (e.g., "setup_lighting")
        new_code: New function definition (must include `def` line)
        output_name: Optional output filename (default: adds "_patch_{section}" suffix)

    Returns:
        JSON with: success, script_path, section_patched, error
    """
    from utils.script_sections import (
        CANONICAL_SECTIONS,
        replace_section,
        list_sections,
    )

    # Validate section name
    if section_name not in CANONICAL_SECTIONS:
        return json.dumps({
            "success": False,
            "error": f"'{section_name}' is not a canonical section. "
                     f"Valid: {CANONICAL_SECTIONS}",
        })

    # Read source script
    try:
        source = Path(script_path).read_text()
    except FileNotFoundError:
        return json.dumps({
            "success": False,
            "error": f"Script not found: {script_path}",
        })

    # Replace section
    try:
        patched = replace_section(source, section_name, new_code)
    except KeyError as e:
        return json.dumps({
            "success": False,
            "error": str(e),
        })

    # Validate patched script compiles
    try:
        compile(patched, script_path, "exec")
    except SyntaxError as e:
        return json.dumps({
            "success": False,
            "error": f"Patched script has syntax error: {e}",
        })

    # Write output
    if output_name:
        out_path = SCRIPT_OUTPUT_DIR / f"{output_name}.py"
    else:
        stem = Path(script_path).stem
        out_path = SCRIPT_OUTPUT_DIR / f"{stem}_patch_{section_name}.py"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(patched)

    # Report sections in patched script
    sections_info = list_sections(patched)

    return json.dumps({
        "success": True,
        "script_path": str(out_path),
        "section_patched": section_name,
        "sections_found": len(sections_info),
        "sections": [s["name"] for s in sections_info],
    })


@function_tool
async def patch_script_section(
    script_path: str,
    section_name: str,
    new_code: str,
    output_name: Optional[str] = None,
) -> str:
    """Patch a SINGLE section of a Blender script, preserving all other sections.

    USE THIS instead of write_script when you only need to change one aspect
    of an existing script (e.g., fix lighting without touching physics setup).

    Canonical sections: setup_scene, create_geometry, setup_materials,
    setup_physics, setup_lighting, setup_camera, bake_and_render.

    Args:
        script_path: Path to the existing script to patch
        section_name: Which section to replace (must be a canonical name)
        new_code: Complete new function definition including `def` line
        output_name: Optional output filename (default: adds patch suffix)

    Returns:
        JSON with patched script path and section info
    """
    return _patch_script_section_impl(
        script_path=script_path,
        section_name=section_name,
        new_code=new_code,
        output_name=output_name,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_patch_tool.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add tools/script_generator_tools.py tests/test_patch_tool.py
git commit -m "feat: add patch_script_section tool for section-level script modification"
```

---

### Task 3: Section-Naming Guardrail on `write_script`

**Files:**
- Modify: `guardrails/script_guardrails.py` (add `validate_section_naming` guardrail)
- Modify: `tools/script_generator_tools.py` (attach guardrail to `write_script`)
- Modify: `tests/test_tool_guardrails.py` (add tests)

The guardrail checks that scripts written by `write_script` contain at least 5 of 7 canonical sections. This is a WARNING, not a tripwire — it logs the missing sections but doesn't block. Over time as we gain confidence, it can become a tripwire.

- [ ] **Step 1: Write failing tests for section guardrail**

Add to `tests/test_tool_guardrails.py`:

```python
class TestSectionNamingGuardrail:
    """Test the section naming output guardrail on write_script."""

    @pytest.fixture
    def sectioned_output(self):
        return json.dumps({
            "success": True,
            "script_path": "/tmp/test.py",
            "sections_found": 7,
            "sections": [
                "setup_scene", "create_geometry", "setup_materials",
                "setup_physics", "setup_lighting", "setup_camera",
                "bake_and_render",
            ],
        })

    @pytest.fixture
    def unsectioned_output(self):
        return json.dumps({
            "success": True,
            "script_path": "/tmp/test.py",
            "sections_found": 0,
            "sections": [],
        })

    @pytest.fixture
    def partial_output(self):
        return json.dumps({
            "success": True,
            "script_path": "/tmp/test.py",
            "sections_found": 3,
            "sections": ["setup_scene", "setup_physics", "bake_and_render"],
        })

    def test_fully_sectioned_passes(self, sectioned_output):
        from guardrails.script_guardrails import _check_section_naming
        triggered, info = _check_section_naming(sectioned_output)
        assert triggered is False

    def test_unsectioned_warns(self, unsectioned_output):
        from guardrails.script_guardrails import _check_section_naming
        triggered, info = _check_section_naming(unsectioned_output)
        # Warning only, not tripwire
        assert triggered is False
        assert info.get("warning") is not None

    def test_partial_warns_with_missing(self, partial_output):
        from guardrails.script_guardrails import _check_section_naming
        triggered, info = _check_section_naming(partial_output)
        assert triggered is False
        assert "missing_sections" in info
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_tool_guardrails.py::TestSectionNamingGuardrail -v`
Expected: FAIL — `ImportError: cannot import name '_check_section_naming'`

- [ ] **Step 3: Implement section naming check in `guardrails/script_guardrails.py`**

Add after existing guardrail functions:

```python
def _check_section_naming(tool_output: str) -> tuple[bool, dict]:
    """Check if write_script output contains canonical section structure.

    Returns (tripwire_triggered, info_dict). Currently warning-only (never triggers).
    """
    from utils.script_sections import CANONICAL_SECTIONS

    try:
        data = json.loads(tool_output) if isinstance(tool_output, str) else tool_output
    except (json.JSONDecodeError, TypeError):
        return False, {"status": "could_not_parse"}

    if not data.get("success"):
        return False, {"status": "script_write_failed"}

    sections_found = data.get("sections", [])
    missing = [s for s in CANONICAL_SECTIONS if s not in sections_found]

    if len(missing) == 0:
        return False, {"status": "all_sections_present", "count": len(sections_found)}

    # Warning only — log but don't tripwire
    info = {
        "warning": f"Script missing {len(missing)}/{len(CANONICAL_SECTIONS)} canonical sections",
        "missing_sections": missing,
        "found_sections": sections_found,
    }
    print(
        f"[Guardrail] section_naming WARN: {info['warning']} — {missing}",
        file=sys.stderr,
    )
    return False, info
```

- [ ] **Step 4: Wire section detection into `_write_script_impl`**

In `tools/script_generator_tools.py`, modify `_write_script_impl` to include section info in its return value. After writing the file and before returning, add:

```python
from utils.script_sections import list_sections

sections_info = list_sections(code)
# ... include in return JSON:
# "sections_found": len(sections_info),
# "sections": [s["name"] for s in sections_info],
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_tool_guardrails.py::TestSectionNamingGuardrail -v`
Expected: All PASS

- [ ] **Step 6: Run full test suite**

Run: `venv/bin/python -m pytest tests/ -v`
Expected: All 388+ tests PASS

- [ ] **Step 7: Commit**

```bash
git add guardrails/script_guardrails.py tools/script_generator_tools.py tests/test_tool_guardrails.py
git commit -m "feat: add section naming guardrail (warning-only) and wire section detection into write_script"
```

---

## Chunk 3: Orchestrator Integration and Patch Budget

### Task 4: Add Patch Budget to SessionState

**Files:**
- Modify: `models/shared_context.py` (add fields to `SessionState`)
- Modify: `models/pipeline_models.py` (add `CANONICAL_SECTIONS` re-export, `ScriptOutput.sections_found`)

- [ ] **Step 1: Add patch budget fields to `SessionState`**

Find the `SessionState` class in `models/shared_context.py` and add after the existing `hitl_history` field:

```python
# Wave 2: Section patch budget tracking
patch_count_iteration: int = Field(
    default=0,
    description="Number of section patches attempted in current iteration (max 2)"
)
patch_count_session: int = Field(
    default=0,
    description="Total section patches attempted in session (max 4)"
)
```

- [ ] **Step 2: Add `sections_found` to `ScriptOutput` in `models/pipeline_models.py`**

Find the `ScriptOutput` class and add:

```python
sections_found: List[str] = Field(
    default_factory=list,
    description="Canonical section names found in the script"
)
```

- [ ] **Step 3: Re-export `CANONICAL_SECTIONS` from `models/pipeline_models.py`**

At the top of `models/pipeline_models.py`, add:

```python
from utils.script_sections import CANONICAL_SECTIONS
```

This makes the constant importable from the models layer without tools dependency.

- [ ] **Step 4: Run full test suite**

Run: `venv/bin/python -m pytest tests/ -v`
Expected: All tests PASS (new fields have defaults, backward-compatible)

- [ ] **Step 5: Commit**

```bash
git add models/shared_context.py models/pipeline_models.py
git commit -m "feat: add patch budget tracking to SessionState and section info to ScriptOutput"
```

---

### Task 5: Rewire `modify_code` to Use Section Patching

**Files:**
- Modify: `orchestrator.py:1720-1777` (the `modify_code` path)

This is the key integration point. When the Modification Coordinator says `modify_code`, instead of handing the entire script to the Script Writer for a full rewrite, we:

1. Check if the script has canonical sections
2. If yes AND patch budget allows: ask the Script Writer to write ONLY the target section, then use `patch_script_section` to splice it in
3. If no sections or budget exceeded: fall back to current full rewrite behavior

- [ ] **Step 1: Add `patch_script_section` to Script Writer's tool list**

In `orchestrator.py`, find where `self._script_agent_standalone` is created and add the new tool to its tools list. Search for the Script Writer agent creation (around line 860-930).

Add the import at the top of orchestrator.py:

```python
from tools.script_generator_tools import patch_script_section
```

Add `patch_script_section` to the Script Writer's tools list alongside `write_script`, `modify_script`, `validate_script`.

- [ ] **Step 2: Modify the `modify_code` block in orchestrator.py**

Replace the `modify_code` block at lines ~1720-1777 with logic that:
- Checks `session.patch_count_iteration < 2` and `session.patch_count_session < 4`
- Reads the script and calls `validate_section_structure()` to check if sections exist
- If sections exist and budget allows: builds a prompt asking Script Writer to write ONLY the target section and use `patch_script_section`
- If not: falls back to the existing full-rewrite prompt

```python
# MODIFY CODE: Section-level patching if possible, full rewrite as fallback
if mod_decision.action == 'modify_code' and mod_decision.code_change_description and previous_script and previous_script.script_path:
    from utils.script_sections import validate_section_structure, CANONICAL_SECTIONS

    # Check if script has sections and patch budget allows
    try:
        script_source = Path(previous_script.script_path).read_text()
        missing, found = validate_section_structure(script_source)
        has_sections = len(found) >= 5
    except Exception:
        has_sections = False

    can_patch = (
        has_sections
        and session.patch_count_iteration < 2
        and session.patch_count_session < 4
    )

    if can_patch:
        print(f"[Pipeline] SECTION PATCH → targeting specific section(s)", file=sys.stderr)
        print(f"[Pipeline] Patch budget: iter={session.patch_count_iteration}/2, session={session.patch_count_session}/4", file=sys.stderr)
        print(f"[Pipeline] Available sections: {found}", file=sys.stderr)

        patch_prompt = f"""FIX this script by patching ONLY the section(s) that need changes.

## Current Script
Path: {previous_script.script_path}
Sections available: {', '.join(found)}

## What Must Change (from Coordinator)
{mod_decision.code_change_description}

## Quality Feedback
Score: {previous_score:.1f}
Primary Issue: {quality.primary_issue if quality else 'Unknown'}

## Instructions
1. Read the current script to understand all sections
2. Identify which section(s) need changes (usually 1-2)
3. For EACH section that needs changes:
   a. Write the COMPLETE new function definition
   b. Use patch_script_section to replace ONLY that section
4. Do NOT rewrite the entire script — only patch what's broken
5. Keep all other sections exactly as they are

Output name: {request.asset_name}_iter{iteration}_patch"""

        try:
            patch_hooks = create_error_recovery_hooks()
            patch_result = await self._run_agent(
                self._script_agent_standalone,
                patch_prompt,
                context=context,
                session=iter_session,
                hooks=patch_hooks,
                max_turns=6,
                run_config=self._build_run_config(
                    session=session,
                    request=request,
                    iteration=iteration,
                    phase="section_patch",
                ),
            )
            patch_output = patch_result.final_output
            if patch_output and patch_output.script_path:
                print(f"[Pipeline] Section patch SUCCESS: {patch_output.script_path}", file=sys.stderr)
                script = patch_output
                direct_modification_success = True
                session.patch_count_iteration += 1
                session.patch_count_session += 1
            else:
                print(f"[Pipeline] Section patch produced no output, falling back to full rewrite", file=sys.stderr)
        except Exception as e:
            print(f"[Pipeline] Section patch failed: {e}, falling back to full rewrite", file=sys.stderr)

    if not direct_modification_success:
        # FALLBACK: Full rewrite (existing behavior)
        print(f"[Pipeline] FULL REWRITE → Script Writer rewrite", file=sys.stderr)
        # ... existing code_fix_prompt and _run_agent call ...
```

- [ ] **Step 3: Reset `patch_count_iteration` at iteration start**

In the main iteration loop (after `[Pipeline] ====== ITERATION N/M ======`), add:

```python
session.patch_count_iteration = 0
```

- [ ] **Step 4: Run full test suite**

Run: `venv/bin/python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add orchestrator.py
git commit -m "feat: rewire modify_code to use section patching with patch budget (2/iter, 4/session)"
```

---

### Task 6: Update Script Writer Prompt for Sectioned Output

**Files:**
- Modify: `orchestrator.py` (Script Writer generation prompt, around line ~1600-1650)

The Script Writer's initial generation prompt (Phase 1) should instruct it to organize code into canonical sections. This doesn't break existing behavior — it's prompt guidance, not enforcement.

- [ ] **Step 1: Add section structure guidance to Script Writer prompt**

Find the script generation prompt in orchestrator.py (the Phase 1 prompt that goes to Script Writer). Add this block to the prompt:

```
## SCRIPT STRUCTURE (Required)
Organize your code into these canonical functions:
- setup_scene(): Scene cleanup, render settings, world/sky background
- create_geometry(): All mesh and curve creation
- setup_materials(): Material and shader node definitions
- setup_physics(): Physics modifiers, force fields, constraints, bake settings
- setup_lighting(): Lights, HDRI, environment lighting
- setup_camera(): Camera placement, lens, depth of field
- bake_and_render(): Point cache bake, render call, .blend save

Put imports, constants, and small helper functions BEFORE these sections.
End with sequential calls: setup_scene(); create_geometry(); ... bake_and_render()
```

- [ ] **Step 2: Run cloth wind E2E test (1 iteration) to verify sections appear**

Run: `venv/bin/python test_cloth_wind.py`
After completion, check: `grep "^def " assets/blender_scripts/generated/cloth_wind_sheet.py`
Expected: At least 5 of 7 canonical section functions visible

- [ ] **Step 3: Commit**

```bash
git add orchestrator.py
git commit -m "feat: add canonical section structure guidance to Script Writer generation prompt"
```

---

## Verification

After all tasks are complete:

1. `venv/bin/python -m pytest tests/ -v` — all tests pass (388+ existing + ~25 new)
2. Run cloth E2E (1 iteration) — script should have canonical sections
3. Check section guardrail log: `grep "section_naming" traces/cloth_wind_*.jsonl`
4. Verify patch budget fields in session JSON: `python -c "from models.shared_context import SessionState; s = SessionState(); print(s.patch_count_iteration, s.patch_count_session)"`

## What This Does NOT Change

- `modify_params` path — still uses regex parameter replacement (working fine)
- `switch_technique` path — still generates fresh scripts (correct behavior)
- Recovery path in `phases/execution.py` — still does full rewrite (recovery is a different problem; the script may be structurally broken)
- Quality evaluation — unchanged
- TechniqueContract / capability packs — unchanged
- Learning Agent — unchanged

## Rollback

If section patching causes regressions, the Modification Coordinator can still route to `modify_params` or `switch_technique`. The section patch path has an explicit fallback to full rewrite when `can_patch` is False. To disable entirely, set `session.patch_count_session = 4` at session start (budget immediately exhausted).
