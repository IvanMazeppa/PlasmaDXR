# feat: Blender Manual API Validation for Script Generator

## Overview

Integrate the Blender Manual MCP into the Script Generator to validate Blender Python APIs **before** generating scripts. This prevents runtime failures by catching invalid parameters, removed APIs, and deprecations at generation time.

**Problem:** Script Generator creates Blender scripts using hardcoded templates and training data. It does NOT validate against Blender 5.0 documentation, leading to:
- Invalid parameter values (e.g., `burning_rate = 5.0` when max is 4.0)
- Removed APIs (e.g., `'BLOSC'` compression removed in Blender 5.0)
- Deprecation warnings (e.g., `Material.use_nodes` removed in Blender 6.0)

**Solution:** Script Generator queries Blender Manual MCP to validate APIs before code generation.

---

## Technical Approach

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     VALIDATION FLOW                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐     ┌─────────────────┐     ┌────────────────┐  │
│  │   Script    │────>│   Validation    │────>│   Blender      │  │
│  │  Generator  │     │     Layer       │     │   Manual MCP   │  │
│  └─────────────┘     └─────────────────┘     └────────────────┘  │
│         │                    │                        │          │
│         │                    │ Cache                  │          │
│         │                    ▼                        │          │
│         │            ┌─────────────┐                  │          │
│         │            │  Validation │                  │          │
│         │            │    Cache    │◄─────────────────┘          │
│         │            │  (24h TTL)  │                             │
│         ▼            └─────────────┘                             │
│  ┌─────────────┐                                                 │
│  │  Generated  │                                                 │
│  │   Script    │                                                 │
│  └─────────────┘                                                 │
└──────────────────────────────────────────────────────────────────┘
```

### Implementation Pattern: Direct Function Import

Based on research, the simplest pattern is **direct Python import** since both servers are in the same project:

```python
# In agents/script-generator/server.py

import sys
sys.path.insert(0, str(PROJECT_ROOT / "agents/blender-manual"))
from blender_server import search_python_api, search_bpy_types, read_page

async def validate_api_call(api_name: str) -> ValidationResult:
    """Validate a Blender API against documentation."""
    result = search_python_api(api_name, limit=3, compact=False, api_only=True)
    # Parse and return validation status
    ...
```

**Why not MCP client embedding:** Adds complexity (subprocess management, stdio communication). Direct import is simpler since both are Python in same repo.

---

## Implementation Phases

### Phase 1: Validation Layer Core

**Estimated effort:** 2-3 hours

**Files to create:**
- `agents/script-generator/validation.py` (NEW)

**Files to modify:**
- `agents/script-generator/server.py:893` (generate_script function)

#### validation.py

```python
"""
Blender API Validation Layer

Validates Blender Python APIs against the Blender Manual MCP.
Uses direct function import (not MCP client) for simplicity.
"""

import sys
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict
from enum import Enum

# Import Blender Manual functions directly
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "agents/blender-manual"))
from blender_server import search_python_api, search_bpy_types, read_page


class ValidationSeverity(Enum):
    ERROR = "error"      # Blocks generation
    WARNING = "warning"  # Allows with acknowledgment
    INFO = "info"        # Informational only


@dataclass
class ValidationIssue:
    api: str
    severity: ValidationSeverity
    message: str
    suggested_fix: Optional[str] = None
    valid_range: Optional[str] = None
    documentation_url: Optional[str] = None


@dataclass
class ValidationResult:
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    validated_apis: List[str] = field(default_factory=list)
    validation_time: float = 0.0
    
    def has_errors(self) -> bool:
        return any(i.severity == ValidationSeverity.ERROR for i in self.issues)
    
    def has_warnings(self) -> bool:
        return any(i.severity == ValidationSeverity.WARNING for i in self.issues)


class ValidationCache:
    """24-hour TTL cache for validation results."""
    
    def __init__(self, ttl_hours: int = 24):
        self.cache: Dict[str, dict] = {}
        self.ttl = timedelta(hours=ttl_hours)
    
    def get(self, api: str) -> Optional[dict]:
        if api in self.cache:
            entry = self.cache[api]
            if datetime.now() - entry["timestamp"] < self.ttl:
                return entry["data"]
            del self.cache[api]
        return None
    
    def set(self, api: str, data: dict):
        self.cache[api] = {"data": data, "timestamp": datetime.now()}


# Global cache instance
_cache = ValidationCache()


# Known parameter ranges from Blender 5.0 API (pre-populated for performance)
KNOWN_RANGES = {
    "resolution_max": (6, 10000),
    "burning_rate": (0.01, 4.0),
    "flame_smoke": (0.0, 8.0),
    "flame_vorticity": (0.0, 2.0),
    "flame_max_temp": (1.0, 10.0),
    "vorticity": (0.0, 4.0),
    "alpha": (-5.0, 5.0),
    "beta": (-5.0, 5.0),
    "fuel_amount": (0.0, 10.0),
    "temperature": (-10.0, 10.0),
    "density": (0.0, 10.0),
}


# Known removed/deprecated APIs
REMOVED_APIS = {
    "openvdb_cache_compress_type='BLOSC'": {
        "message": "'BLOSC' compression removed in Blender 5.0",
        "fix": "Use 'ZIP' or 'NONE' instead"
    }
}


DEPRECATED_APIS = {
    "Material.use_nodes": {
        "message": "Property will be removed in Blender 6.0",
        "fix": "Material nodes are always enabled in 6.0+"
    }
}


def validate_parameter_range(param_name: str, value: float) -> Optional[ValidationIssue]:
    """Check if parameter value is within valid range."""
    if param_name in KNOWN_RANGES:
        min_val, max_val = KNOWN_RANGES[param_name]
        if value < min_val or value > max_val:
            return ValidationIssue(
                api=param_name,
                severity=ValidationSeverity.ERROR,
                message=f"Value {value} out of range [{min_val}, {max_val}]",
                suggested_fix=f"Use value between {min_val} and {max_val}",
                valid_range=f"[{min_val}, {max_val}]"
            )
    return None


def validate_api_exists(api_name: str) -> Optional[ValidationIssue]:
    """Check if API exists in Blender 5.0 documentation."""
    # Check cache first
    cached = _cache.get(api_name)
    if cached is not None:
        return cached.get("issue")
    
    # Query Blender Manual MCP
    try:
        result = search_python_api(api_name, limit=3, compact=False, api_only=True)
        
        # Parse result - if no matches found, API doesn't exist
        if "Found 0 results" in result or "No results" in result.lower():
            issue = ValidationIssue(
                api=api_name,
                severity=ValidationSeverity.ERROR,
                message=f"API '{api_name}' not found in Blender 5.0 documentation",
                suggested_fix="Check spelling or verify API exists in Blender 5.0"
            )
            _cache.set(api_name, {"issue": issue})
            return issue
        
        # API found - cache positive result
        _cache.set(api_name, {"issue": None})
        return None
        
    except Exception as e:
        # MCP call failed - return warning, don't block
        return ValidationIssue(
            api=api_name,
            severity=ValidationSeverity.WARNING,
            message=f"Could not validate API: {str(e)}",
            suggested_fix="Manual verification recommended"
        )


def validate_removed_apis(script_content: str) -> List[ValidationIssue]:
    """Check for known removed APIs in script content."""
    issues = []
    for pattern, info in REMOVED_APIS.items():
        if pattern in script_content:
            issues.append(ValidationIssue(
                api=pattern,
                severity=ValidationSeverity.ERROR,
                message=info["message"],
                suggested_fix=info["fix"]
            ))
    return issues


def validate_deprecated_apis(script_content: str) -> List[ValidationIssue]:
    """Check for deprecated APIs that will be removed."""
    issues = []
    for pattern, info in DEPRECATED_APIS.items():
        if pattern in script_content:
            issues.append(ValidationIssue(
                api=pattern,
                severity=ValidationSeverity.WARNING,
                message=info["message"],
                suggested_fix=info["fix"]
            ))
    return issues


def validate_script(script_content: str, validate_apis: bool = True) -> ValidationResult:
    """
    Validate a Blender Python script against documentation.
    
    Args:
        script_content: The Python script to validate
        validate_apis: Whether to query Blender Manual MCP (slower)
    
    Returns:
        ValidationResult with issues found
    """
    import time
    start_time = time.time()
    
    issues = []
    validated_apis = []
    
    # Check for removed APIs
    issues.extend(validate_removed_apis(script_content))
    
    # Check for deprecated APIs
    issues.extend(validate_deprecated_apis(script_content))
    
    # Extract and validate parameter assignments
    # Pattern: settings.param_name = value
    param_pattern = r'settings\.(\w+)\s*=\s*([\d.]+)'
    for match in re.finditer(param_pattern, script_content):
        param_name = match.group(1)
        try:
            value = float(match.group(2))
            issue = validate_parameter_range(param_name, value)
            if issue:
                issues.append(issue)
            validated_apis.append(f"settings.{param_name}")
        except ValueError:
            pass
    
    # Validate key APIs against Blender Manual (expensive, optional)
    if validate_apis:
        key_apis = [
            "bpy.ops.fluid.bake_all",
            "bpy.types.FluidDomainSettings",
            "bpy.types.FluidFlowSettings"
        ]
        for api in key_apis:
            if api.replace(".", " ").lower() in script_content.lower():
                issue = validate_api_exists(api)
                if issue:
                    issues.append(issue)
                validated_apis.append(api)
    
    validation_time = time.time() - start_time
    
    return ValidationResult(
        is_valid=not any(i.severity == ValidationSeverity.ERROR for i in issues),
        issues=issues,
        validated_apis=validated_apis,
        validation_time=validation_time
    )
```

#### Integration in server.py

```python
# agents/script-generator/server.py:893

from validation import validate_script, ValidationSeverity

@mcp.tool()
async def generate_script(
    effect_type: str,
    description: str,
    output_name: str,
    resolution: int = 96,
    frame_start: int = 1,
    frame_end: int = 50,
    template_name: Optional[str] = None,
    skip_validation: bool = False  # NEW: Allow bypass for advanced users
) -> str:
    """
    Generate a new Blender script from a description.
    
    Validates against Blender 5.0 API documentation before returning.
    """
    # ... existing generation logic ...
    
    script_content = generate_from_template(...)
    
    # NEW: Validate generated script
    if not skip_validation:
        validation_result = validate_script(script_content, validate_apis=True)
        
        if validation_result.has_errors():
            # Return error with validation details
            return json.dumps({
                "success": False,
                "error": "Script validation failed",
                "validation": {
                    "issues": [
                        {
                            "api": issue.api,
                            "severity": issue.severity.value,
                            "message": issue.message,
                            "suggested_fix": issue.suggested_fix
                        }
                        for issue in validation_result.issues
                    ],
                    "validated_apis": validation_result.validated_apis,
                    "validation_time_seconds": validation_result.validation_time
                }
            }, indent=2)
        
        if validation_result.has_warnings():
            # Include warnings but proceed
            warnings = [i for i in validation_result.issues 
                       if i.severity == ValidationSeverity.WARNING]
    
    # Write script and return success
    # ... existing write logic ...
    
    return json.dumps({
        "success": True,
        "script_path": str(output_path),
        "validation": {
            "status": "passed",
            "warnings": [w.message for w in warnings] if warnings else [],
            "validated_apis": validation_result.validated_apis,
            "validation_time_seconds": validation_result.validation_time
        }
    }, indent=2)
```

---

### Phase 2: Enhanced API Validation

**Estimated effort:** 2 hours

**Files to modify:**
- `agents/script-generator/validation.py`

#### Add Deep API Validation

```python
def validate_api_property_access(script_content: str) -> List[ValidationIssue]:
    """
    Validate property access patterns like domain.settings.resolution_max.
    """
    issues = []
    
    # Pattern: domain_settings.property = value
    property_pattern = r'domain_settings\.(\w+)\s*='
    for match in re.finditer(property_pattern, script_content):
        prop_name = match.group(1)
        
        # Query Blender Manual for property existence
        result = search_bpy_types(f"FluidDomainSettings {prop_name}", limit=3)
        
        if "Found 0 results" in result:
            issues.append(ValidationIssue(
                api=f"FluidDomainSettings.{prop_name}",
                severity=ValidationSeverity.ERROR,
                message=f"Property '{prop_name}' not found on FluidDomainSettings",
                suggested_fix="Check Blender 5.0 API documentation"
            ))
    
    return issues


def get_api_documentation(api_name: str) -> Optional[str]:
    """
    Fetch detailed documentation for an API from Blender Manual.
    Used to provide suggested fixes.
    """
    try:
        # Map API name to documentation page
        page_map = {
            "FluidDomainSettings": "bpy.types.FluidDomainSettings.html",
            "FluidFlowSettings": "bpy.types.FluidFlowSettings.html",
            "bpy.ops.fluid": "bpy.ops.fluid.html"
        }
        
        for key, page in page_map.items():
            if key in api_name:
                return read_page(page, max_length=2000, source="python_api")
        
        return None
    except Exception:
        return None
```

---

### Phase 3: Template Pre-Validation

**Estimated effort:** 1 hour

**Files to create:**
- `agents/script-generator/validate_templates.py` (NEW)

```python
#!/usr/bin/env python3
"""
Template Pre-Validation Script

Run before committing template changes:
    python validate_templates.py

Validates all templates in assets/blender_scripts/GPT-5.2/
against Blender 5.0 API documentation.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "agents/script-generator"))

from validation import validate_script, ValidationSeverity

TEMPLATE_DIR = PROJECT_ROOT / "assets/blender_scripts/GPT-5.2"


def main():
    templates = list(TEMPLATE_DIR.glob("*.py"))
    print(f"Validating {len(templates)} templates...")
    
    errors = 0
    warnings = 0
    
    for template in templates:
        content = template.read_text()
        result = validate_script(content, validate_apis=False)  # Fast check
        
        if result.has_errors():
            print(f"❌ {template.name}: {len([i for i in result.issues if i.severity == ValidationSeverity.ERROR])} errors")
            for issue in result.issues:
                if issue.severity == ValidationSeverity.ERROR:
                    print(f"   - {issue.message}")
                    if issue.suggested_fix:
                        print(f"     Fix: {issue.suggested_fix}")
            errors += 1
        elif result.has_warnings():
            print(f"⚠️  {template.name}: {len([i for i in result.issues if i.severity == ValidationSeverity.WARNING])} warnings")
            warnings += 1
        else:
            print(f"✅ {template.name}: valid")
    
    print(f"\nSummary: {len(templates)} templates, {errors} with errors, {warnings} with warnings")
    sys.exit(1 if errors > 0 else 0)


if __name__ == "__main__":
    main()
```

---

### Phase 4: Integration with Iteration Controller

**Estimated effort:** 1 hour

**Files to modify:**
- `agents/iteration-controller/server.py`

```python
# In create_asset() and run_iteration()

# Only validate on first iteration for performance
if iteration == 1:
    # Full validation with API checks
    validation_result = await call_script_generator_with_validation(...)
    
    if not validation_result["success"]:
        return {
            "status": "validation_failed",
            "iteration": iteration,
            "validation_errors": validation_result["validation"]["issues"]
        }
else:
    # Skip API validation on subsequent iterations
    # (parameter changes validated locally)
    script_result = await call_script_generator(skip_validation=True, ...)
```

---

## Acceptance Criteria

### Functional Requirements

- [ ] `generate_script()` validates parameters against known ranges before generation
- [ ] `generate_script()` checks for removed APIs (e.g., BLOSC compression)
- [ ] `generate_script()` checks for deprecated APIs with warnings
- [ ] Validation errors block script generation with actionable error messages
- [ ] Validation warnings allow generation but are included in response
- [ ] `skip_validation=True` parameter bypasses validation for advanced users
- [ ] Validation results are cached for 24 hours to avoid repeated MCP calls
- [ ] Template validation script can be run manually before commits

### Non-Functional Requirements

- [ ] Validation adds < 2 seconds to script generation (with caching)
- [ ] MCP failures result in WARNING, not ERROR (graceful degradation)
- [ ] Validation works offline using cached results and known ranges

### Quality Gates

- [ ] All existing templates pass validation without errors
- [ ] Unit tests cover validation logic (mock Blender Manual responses)
- [ ] Integration test with real Blender Manual MCP

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Script generation failures due to invalid API | < 5% | Track in experiment-tracker |
| Validation false positives | < 1% | User-reported issues |
| Validation latency (cached) | < 500ms | Time validation_time field |
| Validation latency (uncached) | < 3s | Time validation_time field |

---

## Dependencies & Prerequisites

- [x] Blender Manual MCP operational (`agents/blender-manual/`)
- [x] Script Generator MCP operational (`agents/script-generator/`)
- [ ] Both servers running in same project root (for direct import)

---

## Risk Analysis & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Blender Manual MCP unavailable | Medium | High | Graceful degradation to WARNING, use cached/hardcoded ranges |
| Validation too slow | Medium | Medium | Aggressive caching, skip API validation on iterations 2+ |
| False positives block valid scripts | Low | High | Allow `skip_validation` override, monitor user feedback |
| Blender version mismatch | High | Medium | Document "validates for Blender 5.0 only", add version check |

---

## Future Considerations

1. **Multi-version support:** Add Blender 4.x validation (requires separate Manual index)
2. **Auto-migration:** Automatically fix deprecated APIs in generated scripts
3. **Resource warnings:** Warn about high memory usage (e.g., resolution > 256)
4. **Context validation:** Check operator poll requirements (complex, lower priority)

---

## References & Research

### Internal References

- Script Generator server: `agents/script-generator/server.py:893`
- Blender Manual MCP: `agents/blender-manual/blender_server.py`
- Blender Executor error suggestions: `agents/blender-executor/server.py:134`

### External References

- MCP Best Practices: https://modelcontextprotocol.info/docs/best-practices/
- Blender 5.0 Python API: Via `blender-manual` MCP tools
- FastMCP Documentation: https://gofastmcp.com/

### Related Work

- Experiment Tracker integration: Uses validation results for learning
- Iteration Controller: Calls validation on first iteration only

---

## Assumptions Made

These assumptions were made based on spec-flow analysis. Revisit if requirements change:

1. **Validation failure = ERROR:** Invalid parameters block generation (not auto-clamped)
2. **MCP failure = WARNING:** Continue with degraded validation if MCP unreachable
3. **Single version:** Validates against Blender 5.0 only (no multi-version support)
4. **Validation scope:** API existence, parameter ranges, deprecations only (no context/dependencies)
5. **Iteration strategy:** Full validation on iteration 1, skip on subsequent iterations
6. **Cache TTL:** 24 hours is reasonable (Blender Manual doesn't change daily)

---

*Plan generated: 2025-12-27*
*AI-assisted research using repo-research-analyst, best-practices-researcher, framework-docs-researcher, spec-flow-analyzer*
