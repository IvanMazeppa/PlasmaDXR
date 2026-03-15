"""Tests for deterministic RepairIntent routing."""

import pytest

from models.pipeline_models import QualityIssue, QualityOutput, RepairIntent
from phases.repair_routing import choose_repair_intent


def _make_quality(
    score: float = 50.0,
    primary_issue: str | None = None,
    issues: list[str] | None = None,
    structured_issues: list[QualityIssue] | None = None,
    vision_assessment: str = "",
) -> QualityOutput:
    return QualityOutput(
        overall_score=score,
        passed=False,
        primary_issue=primary_issue,
        issues=issues or [],
        structured_issues=structured_issues or [],
        vision_assessment=vision_assessment,
    )


# ---- Rule 1: Execution failure ----

def test_execution_failure_routes_to_modify_code():
    q = _make_quality(primary_issue="Execution failed: NameError: name 'domain' is not defined")
    intent = choose_repair_intent(q, "", plateau_count=0, same_issue_count=0, iteration=1)
    assert intent.mode == "modify_code"
    assert intent.trigger == "execution_failure"


# ---- Rule 2: Structured issues (structural majority) ----

def test_structured_issues_structural_majority():
    q = _make_quality(
        structured_issues=[
            QualityIssue(summary="Wrong stripe order", kind="structural", repair_mode_hint="modify_code", confidence=0.9),
            QualityIssue(summary="Missing green stripe", kind="structural", repair_mode_hint="modify_code", confidence=0.8),
            QualityIssue(summary="Too dark", kind="parameter", repair_mode_hint="modify_params", confidence=0.6),
        ]
    )
    intent = choose_repair_intent(q, "", plateau_count=0, same_issue_count=0, iteration=1)
    assert intent.mode == "modify_code"
    assert intent.trigger == "structured_issues_structural_majority"


# ---- Rule 3: Keyword scan ----

def test_structural_keyword_routes_to_modify_code():
    q = _make_quality(primary_issue="Wrong stripe order — green and orange stripes are reversed")
    intent = choose_repair_intent(q, "", plateau_count=0, same_issue_count=0, iteration=1)
    assert intent.mode == "modify_code"
    assert intent.trigger == "keyword_structural_majority"


def test_param_keyword_routes_to_modify_params():
    q = _make_quality(primary_issue="Fire is too dark and density is too low")
    intent = choose_repair_intent(q, "", plateau_count=0, same_issue_count=0, iteration=1)
    assert intent.mode == "modify_params"
    assert intent.trigger == "default"


# ---- Rule 4: Plateau ----

def test_plateau_forces_modify_code():
    q = _make_quality(primary_issue="Slightly underexposed")
    intent = choose_repair_intent(q, "", plateau_count=2, same_issue_count=0, iteration=3)
    assert intent.mode == "modify_code"
    assert intent.trigger == "plateau"


# ---- Rule 5: Same issue repeated ----

def test_same_issue_forces_switch_technique():
    q = _make_quality(primary_issue="Slightly underexposed")
    intent = choose_repair_intent(q, "", plateau_count=0, same_issue_count=3, iteration=4)
    assert intent.mode == "switch_technique"
    assert intent.trigger == "same_issue_repeated"


# ---- Rule 6: Escape level ----

def test_escape_level_4_requests_guidance():
    q = _make_quality()
    intent = choose_repair_intent(q, "", plateau_count=0, same_issue_count=0, iteration=5, escape_level=4)
    assert intent.mode == "request_guidance"
    assert intent.trigger == "escape_level_4"


def test_escape_level_2_switches_technique():
    q = _make_quality()
    intent = choose_repair_intent(q, "", plateau_count=0, same_issue_count=0, iteration=3, escape_level=2)
    assert intent.mode == "switch_technique"
    assert intent.trigger == "escape_level_2"


# ---- Rule 7: Default ----

def test_empty_quality_defaults_to_modify_params():
    q = _make_quality()
    intent = choose_repair_intent(q, "", plateau_count=0, same_issue_count=0, iteration=1)
    assert intent.mode == "modify_params"
    assert intent.trigger == "default"


# ---- Ireland Flag scenario ----

def test_ireland_flag_scenario():
    """Ireland Flag E2E: QA says stripes are wrong order, missing geometry.
    Must route to modify_code, not modify_params."""
    q = _make_quality(
        score=41.0,
        primary_issue="Wrong stripe order — green and orange stripes are reversed, flag geometry is incorrect",
        issues=[
            "Stripe order is reversed (should be green-white-orange left to right)",
            "Flag proportions are wrong",
            "Missing proper UV mapping for stripes",
        ],
        structured_issues=[
            QualityIssue(summary="Wrong stripe order", kind="structural", repair_mode_hint="modify_code", confidence=0.95),
            QualityIssue(summary="Incorrect flag proportions", kind="structural", repair_mode_hint="modify_code", confidence=0.8),
        ],
        vision_assessment="The flag shows three vertical stripes but in the wrong order.",
    )
    intent = choose_repair_intent(q, "setup_geometry() creates stripes as [orange, white, green]", plateau_count=0, same_issue_count=0, iteration=1)
    assert intent.mode == "modify_code"


# ---- Priority ordering ----

def test_execution_failure_beats_plateau():
    """Execution failure (rule 1) should beat plateau (rule 4)."""
    q = _make_quality(primary_issue="Execution failed: ModuleNotFoundError")
    intent = choose_repair_intent(q, "", plateau_count=3, same_issue_count=4, iteration=5, escape_level=3)
    assert intent.mode == "modify_code"
    assert intent.trigger == "execution_failure"


def test_structural_keywords_in_code_grounded_feedback():
    """Structural keywords in code_grounded_feedback should also trigger modify_code."""
    q = _make_quality(primary_issue="Visual quality is low")
    intent = choose_repair_intent(
        q, "setup_geometry(): missing object for the green stripe, wrong arrangement of mesh objects",
        plateau_count=0, same_issue_count=0, iteration=1,
    )
    assert intent.mode == "modify_code"


def test_repairintent_model_fields():
    """RepairIntent should serialize cleanly."""
    intent = RepairIntent(
        mode="modify_code",
        trigger="test",
        confidence=0.85,
        target_sections=["setup_geometry"],
        target_params={},
        reasoning="test reason",
    )
    d = intent.model_dump()
    assert d["mode"] == "modify_code"
    assert d["target_sections"] == ["setup_geometry"]
