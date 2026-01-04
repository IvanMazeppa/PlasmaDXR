#!/usr/bin/env python3
"""
Blender Script Validator for Pre-Execution Validation

Implements Task 2.1 from MULTI_AGENT_IMPROVEMENT_PLAN_V2.md:
- Full Python syntax validation using AST
- Parameter extraction and range validation
- Required attribute checks (domain, flow, material settings)
- Output path validation

This module is used by the script-generator MCP server to validate
generated scripts before they are passed to blender-executor.
"""

import ast
import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"      # Script will fail
    WARNING = "warning"  # Script may work but has issues
    INFO = "info"        # Informational note


@dataclass
class ValidationIssue:
    """A single validation issue found in the script."""
    severity: ValidationSeverity
    category: str
    message: str
    line: Optional[int] = None
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "severity": self.severity.value,
            "category": self.category,
            "message": self.message,
        }
        if self.line is not None:
            result["line"] = self.line
        if self.suggestion is not None:
            result["suggestion"] = self.suggestion
        return result


@dataclass
class ValidationResult:
    """Result of script validation."""
    valid: bool
    script_path: str
    issues: List[ValidationIssue] = field(default_factory=list)
    extracted_params: Dict[str, Any] = field(default_factory=dict)
    detected_effect_type: Optional[str] = None
    detected_simulation_pattern: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "valid": self.valid,
            "script_path": self.script_path,
            "issues": [i.to_dict() for i in self.issues],
            "extracted_params": self.extracted_params,
            "detected_effect_type": self.detected_effect_type,
            "detected_simulation_pattern": self.detected_simulation_pattern,
            "error_count": sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR),
            "warning_count": sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING),
        }


# Blender 5.0 Python API parameter ranges
# Source: bpy.types.FluidDomainSettings, bpy.types.FluidFlowSettings
BLENDER_PARAM_RANGES = {
    # Domain Gas Settings
    "burning_rate": {"min": 0.01, "max": 4.0, "default": 0.75, "type": "float"},
    "flame_smoke": {"min": 0.0, "max": 8.0, "default": 1.0, "type": "float"},
    "flame_vorticity": {"min": 0.0, "max": 2.0, "default": 0.5, "type": "float"},
    "flame_max_temp": {"min": 1.0, "max": 10.0, "default": 3.0, "type": "float"},
    "flame_ignition": {"min": 0.5, "max": 5.0, "default": 1.5, "type": "float"},
    "alpha": {"min": -5.0, "max": 5.0, "default": 1.0, "type": "float"},
    "beta": {"min": -5.0, "max": 5.0, "default": 1.0, "type": "float"},
    "dissolve_speed": {"min": 1, "max": 10000, "default": 5, "type": "int"},
    "vorticity": {"min": 0.0, "max": 4.0, "default": 0.0, "type": "float"},

    # Noise/Upres Settings
    "noise_scale": {"min": 1, "max": 10, "default": 2, "type": "int"},
    "noise_strength": {"min": 0.0, "max": 10.0, "default": 1.0, "type": "float"},
    "noise_pos_scale": {"min": 0.0001, "max": 10.0, "default": 2.0, "type": "float"},

    # Flow Settings
    "fuel_amount": {"min": 0.0, "max": 10.0, "default": 1.0, "type": "float"},
    "temperature": {"min": -10.0, "max": 10.0, "default": 1.0, "type": "float"},
    "velocity_normal": {"min": -100.0, "max": 100.0, "default": 0.0, "type": "float"},
    "velocity_random": {"min": 0.0, "max": 10.0, "default": 0.0, "type": "float"},

    # General/Domain Settings
    "resolution_max": {"min": 32, "max": 512, "default": 64, "type": "int"},

    # Soft Body Settings (Phase 2.5 extensibility)
    "step_min": {"min": 1, "max": 100, "default": 1, "type": "int"},
    "step_max": {"min": 1, "max": 250, "default": 50, "type": "int"},
    "damping": {"min": 0.0, "max": 50.0, "default": 0.5, "type": "float"},
    "goal_spring": {"min": 0.0, "max": 0.999, "default": 0.7, "type": "float"},
}

# Required patterns for different simulation types
REQUIRED_PATTERNS = {
    "volumetric": {
        "domain": [
            r"\.domain_settings\.",
            r"domain_type\s*=\s*['\"]GAS['\"]",
        ],
        "flow": [
            r"\.flow_settings\.",
            r"flow_type\s*=\s*['\"]FIRE['\"]|flow_type\s*=\s*['\"]SMOKE['\"]|flow_type\s*=\s*['\"]BOTH['\"]",
        ],
        "output": [
            r"\.vdb",
            r"cache_directory|cache_data_format",
        ],
    },
    "mesh": {
        "physics": [
            r"modifier_add\(type=['\"]SOFT_BODY['\"]\)|modifier_add\(type=['\"]CLOTH['\"]\)|modifier_add\(type=['\"]RIGID_BODY['\"]\)",
        ],
        "simulation": [
            r"\.soft_body\.|\.cloth\.|\.rigid_body\.",
        ],
    },
}

# Dangerous patterns that should generate warnings
DANGEROUS_PATTERNS = [
    (r"bpy\.ops\.wm\.quit_blender", "Script calls quit_blender - may exit before completion"),
    (r"os\.system\s*\(", "Script uses os.system - potential security risk"),
    (r"subprocess\.", "Script uses subprocess - potential security risk"),
    (r"eval\s*\(", "Script uses eval() - potential security risk"),
    (r"exec\s*\(", "Script uses exec() - potential security risk"),
    (r"__import__\s*\(", "Script uses __import__ - potential security risk"),
]


class BlenderScriptValidator:
    """
    Validates Blender Python scripts before execution.

    Performs:
    1. Syntax validation using AST
    2. Parameter extraction and range validation
    3. Required attribute checks
    4. Security/safety checks
    5. Output path validation
    """

    def __init__(self, param_ranges: Optional[Dict[str, Dict]] = None):
        """
        Initialize validator.

        Args:
            param_ranges: Custom parameter ranges (uses defaults if None)
        """
        self.param_ranges = param_ranges or BLENDER_PARAM_RANGES

    def validate_script(self, script_path: str) -> ValidationResult:
        """
        Validate a Blender script file.

        Args:
            script_path: Path to the Python script

        Returns:
            ValidationResult with issues and extracted parameters
        """
        result = ValidationResult(valid=True, script_path=script_path)

        # Check file exists
        if not os.path.exists(script_path):
            result.valid = False
            result.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="file",
                message=f"Script file not found: {script_path}",
            ))
            return result

        # Read script content
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            result.valid = False
            result.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="file",
                message=f"Failed to read script: {e}",
            ))
            return result

        # Run all validations
        self._validate_syntax(content, result)
        self._validate_parameters(content, result)
        self._detect_simulation_type(content, result)
        self._check_required_patterns(content, result)
        self._check_dangerous_patterns(content, result)
        self._validate_output_paths(content, result)

        # Set overall validity
        result.valid = not any(
            i.severity == ValidationSeverity.ERROR for i in result.issues
        )

        return result

    def validate_script_content(self, content: str, script_name: str = "<inline>") -> ValidationResult:
        """
        Validate script content directly (without file).

        Args:
            content: Python script content
            script_name: Name for error messages

        Returns:
            ValidationResult with issues and extracted parameters
        """
        result = ValidationResult(valid=True, script_path=script_name)

        self._validate_syntax(content, result)
        self._validate_parameters(content, result)
        self._detect_simulation_type(content, result)
        self._check_required_patterns(content, result)
        self._check_dangerous_patterns(content, result)
        self._validate_output_paths(content, result)

        result.valid = not any(
            i.severity == ValidationSeverity.ERROR for i in result.issues
        )

        return result

    def _validate_syntax(self, content: str, result: ValidationResult) -> None:
        """Validate Python syntax using AST."""
        try:
            ast.parse(content)
        except SyntaxError as e:
            result.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="syntax",
                message=f"Syntax error: {e.msg}",
                line=e.lineno,
                suggestion="Fix the Python syntax error before execution",
            ))

    def _validate_parameters(self, content: str, result: ValidationResult) -> None:
        """Extract and validate parameters from script."""
        extracted = {}

        for param_name, param_info in self.param_ranges.items():
            # Look for parameter assignments
            patterns = [
                rf"{param_name}\s*=\s*([0-9.eE+-]+)",  # Direct assignment
                rf"\.{param_name}\s*=\s*([0-9.eE+-]+)",  # Attribute assignment
                rf"['\"{param_name}['\"]\s*:\s*([0-9.eE+-]+)",  # Dict key
            ]

            for pattern in patterns:
                matches = re.findall(pattern, content)
                if matches:
                    try:
                        value = float(matches[-1])  # Use last match
                        extracted[param_name] = value

                        # Validate range
                        min_val = param_info["min"]
                        max_val = param_info["max"]

                        if value < min_val or value > max_val:
                            result.issues.append(ValidationIssue(
                                severity=ValidationSeverity.WARNING,
                                category="parameter",
                                message=f"{param_name}={value} is outside valid range [{min_val}, {max_val}]",
                                suggestion=f"Clamp to range or use default: {param_info['default']}",
                            ))
                        break
                    except ValueError:
                        pass

        result.extracted_params = extracted

    def _detect_simulation_type(self, content: str, result: ValidationResult) -> None:
        """Detect the simulation type from script content."""
        content_lower = content.lower()

        # Check for volumetric indicators
        volumetric_indicators = [
            "mantaflow", "domain_type", "flow_type", "gas", "smoke", "fire",
            ".vdb", "openvdb", "cache_data_format"
        ]
        volumetric_count = sum(1 for ind in volumetric_indicators if ind in content_lower)

        # Check for mesh physics indicators
        mesh_indicators = [
            "soft_body", "cloth", "rigid_body", "softbody",
            "view_layer.update", "frame_set"
        ]
        mesh_count = sum(1 for ind in mesh_indicators if ind in content_lower)

        # Determine type
        if volumetric_count > mesh_count:
            result.detected_effect_type = "volumetric"
            result.detected_simulation_pattern = "bake_export"
        elif mesh_count > volumetric_count:
            result.detected_effect_type = "mesh"
            result.detected_simulation_pattern = "live_render"
        else:
            result.detected_effect_type = "unknown"
            result.detected_simulation_pattern = "unknown"
            result.issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="detection",
                message="Could not determine simulation type from script content",
                suggestion="Explicitly specify effect_type when generating",
            ))

    def _check_required_patterns(self, content: str, result: ValidationResult) -> None:
        """Check for required patterns based on detected type."""
        if result.detected_effect_type == "volumetric":
            required = REQUIRED_PATTERNS.get("volumetric", {})

            # Check domain setup
            domain_patterns = required.get("domain", [])
            has_domain = any(re.search(p, content) for p in domain_patterns)
            if not has_domain:
                result.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="structure",
                    message="No domain settings detected in volumetric script",
                    suggestion="Ensure script sets up domain with domain_type='GAS'",
                ))

            # Check flow setup
            flow_patterns = required.get("flow", [])
            has_flow = any(re.search(p, content) for p in flow_patterns)
            if not has_flow:
                result.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="structure",
                    message="No flow settings detected in volumetric script",
                    suggestion="Ensure script sets up flow emitter",
                ))

        elif result.detected_effect_type == "mesh":
            required = REQUIRED_PATTERNS.get("mesh", {})

            # Check physics modifier
            physics_patterns = required.get("physics", [])
            has_physics = any(re.search(p, content) for p in physics_patterns)
            if not has_physics:
                result.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="structure",
                    message="No physics modifier detected in mesh script",
                    suggestion="Add SOFT_BODY, CLOTH, or RIGID_BODY modifier",
                ))

    def _check_dangerous_patterns(self, content: str, result: ValidationResult) -> None:
        """Check for potentially dangerous patterns."""
        for pattern, message in DANGEROUS_PATTERNS:
            if re.search(pattern, content):
                result.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="security",
                    message=message,
                ))

    def _validate_output_paths(self, content: str, result: ValidationResult) -> None:
        """Validate output path configuration."""
        # Check for output directory specification
        output_patterns = [
            r"output_dir\s*=",
            r"OUTPUT_DIR\s*=",
            r"render\.filepath\s*=",
            r"cache_directory\s*=",
        ]

        has_output = any(re.search(p, content) for p in output_patterns)
        if not has_output:
            result.issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="output",
                message="No explicit output path configuration detected",
                suggestion="Script may use Blender default paths",
            ))

        # Check for absolute Windows paths (common issue in cross-platform)
        windows_path = re.search(r'["\'][A-Za-z]:\\\\', content)
        if windows_path:
            result.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="output",
                message="Script contains absolute Windows path",
                suggestion="Use relative paths or Config class for cross-platform compatibility",
            ))


def validate_script(script_path: str) -> ValidationResult:
    """
    Convenience function to validate a script.

    Args:
        script_path: Path to the script file

    Returns:
        ValidationResult
    """
    validator = BlenderScriptValidator()
    return validator.validate_script(script_path)


def validate_script_content(content: str, script_name: str = "<inline>") -> ValidationResult:
    """
    Convenience function to validate script content.

    Args:
        content: Script content string
        script_name: Name for error messages

    Returns:
        ValidationResult
    """
    validator = BlenderScriptValidator()
    return validator.validate_script_content(content, script_name)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python validator.py <script_path>")
        sys.exit(1)

    script_path = sys.argv[1]
    result = validate_script(script_path)

    print(json.dumps(result.to_dict(), indent=2))

    sys.exit(0 if result.valid else 1)
