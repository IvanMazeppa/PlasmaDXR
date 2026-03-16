"""Repair-mode classification benchmark.

A small regression set of (QualityOutput, expected_mode) pairs derived from
real pipeline scenarios and hand-labeled.  Running this file measures
agreement between choose_repair_intent() and human labels.

Usage:
    python -m pytest tests/repair_routing_benchmark.py -v
    python tests/repair_routing_benchmark.py   # standalone with summary
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Path setup
_root = str(Path(__file__).parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from models.pipeline_models import QualityIssue, QualityOutput
from phases.repair_routing import choose_repair_intent


@dataclass
class BenchmarkCase:
    """A hand-labeled repair-routing scenario."""
    name: str
    quality: QualityOutput
    code_grounded_feedback: str
    plateau_count: int
    same_issue_count: int
    iteration: int
    escape_level: int
    expected_mode: str
    notes: str = ""


def _q(
    score: float = 50.0,
    primary_issue: str | None = None,
    issues: list[str] | None = None,
    structured_issues: list[QualityIssue] | None = None,
    vision: str = "",
) -> QualityOutput:
    return QualityOutput(
        overall_score=score,
        passed=False,
        primary_issue=primary_issue,
        issues=issues or [],
        structured_issues=structured_issues or [],
        vision_assessment=vision,
    )


# =========================================================================
# Benchmark cases — hand-labeled from real and synthetic scenarios
# =========================================================================

BENCHMARK: list[BenchmarkCase] = [
    # --- Structural defects ---
    BenchmarkCase(
        name="ireland_flag_wrong_order",
        quality=_q(
            score=41,
            primary_issue="Wrong stripe order — green and orange reversed",
            structured_issues=[
                QualityIssue(summary="Wrong stripe order", kind="structural",
                             repair_mode_hint="modify_code", confidence=0.95),
                QualityIssue(summary="Incorrect proportions", kind="structural",
                             repair_mode_hint="modify_code", confidence=0.8),
            ],
        ),
        code_grounded_feedback="setup_geometry() creates stripes as [orange, white, green]",
        plateau_count=0, same_issue_count=0, iteration=1, escape_level=0,
        expected_mode="modify_code",
        notes="Classic Ireland Flag scenario",
    ),
    BenchmarkCase(
        name="missing_collider_geometry",
        quality=_q(
            score=35,
            primary_issue="No collider object — fluid falls through floor",
            structured_issues=[
                QualityIssue(summary="Missing collider geometry", kind="structural",
                             repair_mode_hint="modify_code", confidence=0.9),
            ],
        ),
        code_grounded_feedback="",
        plateau_count=0, same_issue_count=0, iteration=2, escape_level=0,
        expected_mode="modify_code",
        notes="Missing collider is a structural problem, not a parameter tweak",
    ),
    BenchmarkCase(
        name="camera_inside_geometry",
        quality=_q(
            score=15,
            primary_issue="Render is completely occluded — camera inside mesh",
            structured_issues=[
                QualityIssue(summary="Camera placed inside geometry", kind="camera",
                             repair_mode_hint="modify_code", confidence=0.95),
            ],
        ),
        code_grounded_feedback="",
        plateau_count=0, same_issue_count=0, iteration=1, escape_level=0,
        expected_mode="modify_code",
        notes="Camera issues are structural — need code change to reposition",
    ),

    # --- Parametric issues ---
    BenchmarkCase(
        name="fire_too_dim",
        quality=_q(
            score=52,
            primary_issue="Fire is barely visible, too dim",
            structured_issues=[
                QualityIssue(summary="Fire emission too low", kind="parameter",
                             repair_mode_hint="modify_params", confidence=0.85),
                QualityIssue(summary="Smoke density too high", kind="parameter",
                             repair_mode_hint="modify_params", confidence=0.7),
            ],
        ),
        code_grounded_feedback="",
        plateau_count=0, same_issue_count=0, iteration=2, escape_level=0,
        expected_mode="modify_params",
        notes="Pure parameter issues — density and emission",
    ),
    BenchmarkCase(
        name="smoke_dissolves_too_fast",
        quality=_q(
            score=48,
            primary_issue="Smoke vanishes immediately after emission",
            structured_issues=[
                QualityIssue(summary="dissolve_speed too high", kind="parameter",
                             repair_mode_hint="modify_params", confidence=0.9),
            ],
        ),
        code_grounded_feedback="",
        plateau_count=0, same_issue_count=0, iteration=1, escape_level=0,
        expected_mode="modify_params",
        notes="Single parameter fix needed",
    ),

    # --- Technique failures ---
    BenchmarkCase(
        name="wrong_physics_system",
        quality=_q(
            score=20,
            primary_issue="Using particle system instead of Mantaflow for fluid sim",
            structured_issues=[
                QualityIssue(summary="Wrong physics approach", kind="technique",
                             repair_mode_hint="switch_technique", confidence=0.95),
            ],
        ),
        code_grounded_feedback="",
        plateau_count=0, same_issue_count=0, iteration=2, escape_level=0,
        expected_mode="switch_technique",
        notes="Technique-class issue should route to switch_technique",
    ),

    # --- Execution failures ---
    BenchmarkCase(
        name="nameerror_in_script",
        quality=_q(
            score=0,
            primary_issue="Execution failed: NameError: name 'domain_obj' is not defined",
        ),
        code_grounded_feedback="",
        plateau_count=0, same_issue_count=0, iteration=1, escape_level=0,
        expected_mode="modify_code",
        notes="Execution failure always routes to modify_code",
    ),

    # --- Mixed signals ---
    BenchmarkCase(
        name="structural_keyword_no_structured_issues",
        quality=_q(
            score=40,
            primary_issue="Wrong stripe order, missing geometry for middle section",
            issues=["Stripes are in wrong order", "Geometry topology is incorrect"],
        ),
        code_grounded_feedback="",
        plateau_count=0, same_issue_count=0, iteration=1, escape_level=0,
        expected_mode="modify_code",
        notes="No structured_issues — should fall to keyword heuristics and detect structural",
    ),
    BenchmarkCase(
        name="parametric_prose_but_structural_typed",
        quality=_q(
            score=45,
            primary_issue="Fire is too dark and density needs adjustment",
            structured_issues=[
                QualityIssue(summary="Missing emitter object", kind="structural",
                             repair_mode_hint="modify_code", confidence=0.85),
            ],
        ),
        code_grounded_feedback="",
        plateau_count=0, same_issue_count=0, iteration=1, escape_level=0,
        expected_mode="modify_code",
        notes="Prose says parametric, typed says structural — typed wins",
    ),

    # --- Escalation scenarios ---
    BenchmarkCase(
        name="plateau_with_no_issues",
        quality=_q(score=50, primary_issue="Slightly underexposed"),
        code_grounded_feedback="",
        plateau_count=3, same_issue_count=0, iteration=4, escape_level=0,
        expected_mode="modify_code",
        notes="Plateau escalation even when issues seem minor",
    ),
    BenchmarkCase(
        name="same_issue_repeated",
        quality=_q(score=45, primary_issue="Smoke still too thin"),
        code_grounded_feedback="",
        plateau_count=0, same_issue_count=4, iteration=5, escape_level=0,
        expected_mode="switch_technique",
        notes="Repeated same issue — time to switch technique",
    ),
]


# =========================================================================
# Pytest integration
# =========================================================================

import pytest


@pytest.mark.parametrize(
    "case", BENCHMARK, ids=[c.name for c in BENCHMARK]
)
def test_benchmark_case(case: BenchmarkCase):
    intent = choose_repair_intent(
        case.quality,
        case.code_grounded_feedback,
        plateau_count=case.plateau_count,
        same_issue_count=case.same_issue_count,
        iteration=case.iteration,
        escape_level=case.escape_level,
    )
    assert intent.mode == case.expected_mode, (
        f"[{case.name}] expected {case.expected_mode}, got {intent.mode} "
        f"(trigger={intent.trigger}). {case.notes}"
    )


# =========================================================================
# Standalone summary
# =========================================================================

def run_benchmark():
    """Run benchmark and print agreement summary."""
    print("=" * 70)
    print("REPAIR-MODE CLASSIFICATION BENCHMARK")
    print("=" * 70)
    print()

    passed = 0
    failed = 0
    results = []

    for case in BENCHMARK:
        intent = choose_repair_intent(
            case.quality,
            case.code_grounded_feedback,
            plateau_count=case.plateau_count,
            same_issue_count=case.same_issue_count,
            iteration=case.iteration,
            escape_level=case.escape_level,
        )
        match = intent.mode == case.expected_mode
        if match:
            passed += 1
        else:
            failed += 1
        results.append((case, intent, match))

    for case, intent, match in results:
        status = "PASS" if match else "FAIL"
        print(f"  [{status}] {case.name:40s} expected={case.expected_mode:18s} "
              f"got={intent.mode:18s} trigger={intent.trigger}")

    total = passed + failed
    agreement = (passed / total * 100) if total else 0
    print()
    print(f"Agreement: {passed}/{total} ({agreement:.0f}%)")
    print(f"Cases: {total}")
    print("=" * 70)
    return agreement


if __name__ == "__main__":
    agreement = run_benchmark()
    sys.exit(0 if agreement >= 80 else 1)
