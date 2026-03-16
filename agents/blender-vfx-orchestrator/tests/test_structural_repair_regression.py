"""Structural repair regression set.

Verifies that structural defects reach modify_code and don't get trapped
in parameter loops. Each case is a scenario where the repair router
MUST choose modify_code (or switch_technique at high escape levels).

These cases guard against future regressions where structural problems
get mistakenly routed to parameter tuning.

Usage:
    python -m pytest tests/test_structural_repair_regression.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_root = str(Path(__file__).parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from models.pipeline_models import QualityIssue, QualityOutput
from phases.repair_routing import choose_repair_intent


def _q(
    score: float = 40.0,
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
# Structural defects that MUST route to modify_code
# =========================================================================

class TestWrongStripeOrder:
    """Ireland Flag scenario: stripes in wrong order."""

    def test_typed_structural_issues(self):
        q = _q(
            score=35,
            primary_issue="Stripes in wrong order — should be green-white-orange",
            structured_issues=[
                QualityIssue(summary="Wrong stripe order", kind="structural",
                             repair_mode_hint="modify_code", confidence=0.95),
                QualityIssue(summary="Orange and green reversed", kind="structural",
                             repair_mode_hint="modify_code", confidence=0.9),
            ],
        )
        intent = choose_repair_intent(q, "", 0, 0, 1)
        assert intent.mode == "modify_code"

    def test_keyword_only_fallback(self):
        """Even without structured_issues, keywords should catch this."""
        q = _q(
            score=35,
            primary_issue="Wrong stripe order, stripes are reversed",
        )
        intent = choose_repair_intent(q, "", 0, 0, 1)
        assert intent.mode == "modify_code"

    def test_not_trapped_in_params_at_iteration_3(self):
        """After 3 iterations, this must not degrade to modify_params."""
        q = _q(
            score=38,
            primary_issue="Still wrong stripe order after adjustments",
            structured_issues=[
                QualityIssue(summary="Stripe order unchanged", kind="structural",
                             repair_mode_hint="modify_code", confidence=0.9),
            ],
        )
        intent = choose_repair_intent(q, "", plateau_count=2, same_issue_count=2, iteration=3)
        assert intent.mode == "modify_code"


class TestMissingCollider:
    """Fluid falls through floor because collider object is missing."""

    def test_routes_to_modify_code(self):
        q = _q(
            score=25,
            primary_issue="Fluid falls through floor — no collider",
            structured_issues=[
                QualityIssue(summary="Missing collider geometry", kind="structural",
                             repair_mode_hint="modify_code", confidence=0.9),
            ],
        )
        intent = choose_repair_intent(q, "", 0, 0, 1)
        assert intent.mode == "modify_code"

    def test_keyword_detection(self):
        q = _q(
            score=25,
            primary_issue="Missing collider object in scene, geometry not present",
        )
        intent = choose_repair_intent(q, "", 0, 0, 1)
        assert intent.mode == "modify_code"


class TestCameraInsideGeometry:
    """Camera placed inside a mesh — render is completely black/occluded."""

    def test_with_structural_issue(self):
        """Camera inside geometry is a scene setup problem, needs code fix."""
        q = _q(
            score=5,
            structured_issues=[
                QualityIssue(summary="Camera inside mesh", kind="structural",
                             repair_mode_hint="modify_code", confidence=0.95),
            ],
        )
        intent = choose_repair_intent(q, "", 0, 0, 1)
        assert intent.mode == "modify_code"

    def test_with_camera_kind(self):
        """Camera kind is treated as structural — needs code change to reposition."""
        q = _q(
            score=5,
            structured_issues=[
                QualityIssue(summary="Camera inside mesh", kind="camera",
                             repair_mode_hint="modify_code", confidence=0.95),
            ],
        )
        intent = choose_repair_intent(q, "", plateau_count=0, same_issue_count=0, iteration=1)
        assert intent.mode == "modify_code"
        assert intent.trigger == "structured_issues_structural_majority"


class TestWrongAttachmentTopology:
    """Objects attached to wrong parents or wrong transform hierarchy."""

    def test_routes_to_modify_code(self):
        q = _q(
            score=30,
            primary_issue="Flame emitter attached to wrong object",
            structured_issues=[
                QualityIssue(summary="Wrong parent-child topology", kind="structural",
                             repair_mode_hint="modify_code", confidence=0.85),
            ],
        )
        intent = choose_repair_intent(q, "", 0, 0, 1)
        assert intent.mode == "modify_code"


class TestMissingKeyLight:
    """Scene has no key light — render is completely dark."""

    def test_lighting_structural(self):
        q = _q(
            score=10,
            primary_issue="Scene has no lights — completely dark render",
            structured_issues=[
                QualityIssue(summary="No light sources in scene", kind="structural",
                             repair_mode_hint="modify_code", confidence=0.95),
            ],
        )
        intent = choose_repair_intent(q, "", 0, 0, 1)
        assert intent.mode == "modify_code"

    def test_lighting_kind_with_parametric_companion(self):
        """Lighting issue + parametric issue — lighting is structural if majority."""
        q = _q(
            score=20,
            structured_issues=[
                QualityIssue(summary="Missing key light", kind="structural",
                             repair_mode_hint="modify_code", confidence=0.9),
                QualityIssue(summary="Missing fill light", kind="structural",
                             repair_mode_hint="modify_code", confidence=0.8),
                QualityIssue(summary="Smoke too thin", kind="parameter",
                             repair_mode_hint="modify_params", confidence=0.6),
            ],
        )
        intent = choose_repair_intent(q, "", 0, 0, 1)
        assert intent.mode == "modify_code"
        assert intent.trigger == "structured_issues_structural_majority"


class TestAntiRegressions:
    """Cases that should NEVER route to modify_params when structural."""

    def test_parametric_prose_structural_typed_wins(self):
        """Prose says 'too dark' but typed says structural — typed wins."""
        q = _q(
            score=40,
            primary_issue="Fire is too dark, needs more intensity",
            issues=["density too low", "exposure underexposed"],
            structured_issues=[
                QualityIssue(summary="Missing emitter mesh", kind="structural",
                             repair_mode_hint="modify_code", confidence=0.9),
            ],
        )
        intent = choose_repair_intent(q, "", 0, 0, 1)
        assert intent.mode == "modify_code"

    def test_plateau_escalates_parametric(self):
        """After plateau, even parametric issues get escalated to modify_code."""
        q = _q(
            score=45,
            primary_issue="Still too dark after 3 iterations",
            structured_issues=[
                QualityIssue(summary="Emission too low", kind="parameter",
                             repair_mode_hint="modify_params", confidence=0.8),
            ],
        )
        # With structured_issues present, parametric majority → modify_params
        intent = choose_repair_intent(q, "", plateau_count=0, same_issue_count=0, iteration=3)
        assert intent.mode == "modify_params"
        # But with plateau, it escalates
        intent = choose_repair_intent(q, "", plateau_count=3, same_issue_count=0, iteration=5)
        # structured_issues still takes priority (it's authoritative)
        assert intent.mode == "modify_params"

    def test_same_issue_escalates_to_technique_switch(self):
        """Same parametric issue repeated 3+ times → switch_technique."""
        q = _q(
            score=45,
            primary_issue="Still too dark",
        )
        intent = choose_repair_intent(q, "", plateau_count=0, same_issue_count=3, iteration=4)
        assert intent.mode == "switch_technique"

    def test_execution_failure_always_modify_code(self):
        """Execution failures are always modify_code regardless of other signals."""
        q = _q(
            score=0,
            primary_issue="Execution failed: TypeError: unsupported operand",
            structured_issues=[
                QualityIssue(summary="Density too high", kind="parameter",
                             repair_mode_hint="modify_params", confidence=0.9),
            ],
        )
        intent = choose_repair_intent(q, "", 0, 0, 1)
        assert intent.mode == "modify_code"
        assert intent.trigger == "execution_failure"
