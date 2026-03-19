"""Tests for iteration manifest fields (Task 8).

Verifies that script_hash, issue_kinds, and change_label are
computed correctly and persisted in iteration artifacts.
"""

import json
import tempfile
from pathlib import Path

import pytest

from utils.artifact_manager import (
    CHANGE_LABELS,
    ArtifactManager,
    IterationArtifact,
    compute_script_hash,
)
from models.shared_context import IterationResult, ScriptModification, BlenderExecution, QualityMetrics


# -- compute_script_hash --

def test_hash_deterministic(tmp_path):
    script = tmp_path / "test.py"
    script.write_text("print('hello')")
    h1 = compute_script_hash(str(script))
    h2 = compute_script_hash(str(script))
    assert h1 == h2
    assert len(h1) == 12  # 12 hex chars


def test_hash_changes_with_content(tmp_path):
    script = tmp_path / "test.py"
    script.write_text("version_a")
    h1 = compute_script_hash(str(script))
    script.write_text("version_b")
    h2 = compute_script_hash(str(script))
    assert h1 != h2


def test_hash_missing_file():
    h = compute_script_hash("/nonexistent/path/file.py")
    assert h == ""


def test_hash_hex_only(tmp_path):
    script = tmp_path / "test.py"
    script.write_text("content")
    h = compute_script_hash(str(script))
    assert all(c in "0123456789abcdef" for c in h)


# -- CHANGE_LABELS --

def test_change_labels_complete():
    assert "initial_generation" in CHANGE_LABELS
    assert "modify_params" in CHANGE_LABELS
    assert "modify_code" in CHANGE_LABELS
    assert "section_patch" in CHANGE_LABELS
    assert "technique_switch" in CHANGE_LABELS
    assert "full_rewrite" in CHANGE_LABELS


# -- IterationArtifact --

def test_iteration_artifact_has_manifest_fields():
    artifact = IterationArtifact(
        iteration=1,
        script_path="/tmp/test.py",
        technique_used="mantaflow_fire",
        render_path="/tmp/render.png",
        cache_path=None,
        score=55.0,
        passed=False,
        primary_issue="hero_object too primitive",
        parameter_changes={},
        escape_level=0,
        script_hash="abc123def456",
        issue_kinds=["hero_object", "lighting"],
        change_label="initial_generation",
    )
    assert artifact.script_hash == "abc123def456"
    assert artifact.issue_kinds == ["hero_object", "lighting"]
    assert artifact.change_label == "initial_generation"


def test_iteration_artifact_defaults():
    artifact = IterationArtifact(
        iteration=1,
        script_path="/tmp/test.py",
        technique_used="fire",
        render_path=None,
        cache_path=None,
        score=0.0,
        passed=False,
        primary_issue=None,
        parameter_changes={},
        escape_level=0,
    )
    assert artifact.script_hash == ""
    assert artifact.issue_kinds == []
    assert artifact.change_label == ""


# -- ArtifactManager.write_iteration_from_values --

def test_write_iteration_includes_manifest_fields():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Point artifacts dir to temp
        mgr = ArtifactManager("test_session")
        mgr.artifact_dir = Path(tmpdir)

        path = mgr.write_iteration_from_values(
            iteration=2,
            script_path="/tmp/script.py",
            technique_used="mantaflow_fire",
            score=65.0,
            passed=True,
            script_hash="deadbeef1234",
            issue_kinds=["hero_object"],
            change_label="section_patch",
        )

        data = json.loads(Path(path).read_text())
        assert data["script_hash"] == "deadbeef1234"
        assert data["issue_kinds"] == ["hero_object"]
        assert data["change_label"] == "section_patch"


# -- IterationResult model --

def test_iteration_result_manifest_fields():
    result = IterationResult(
        iteration=1,
        script=ScriptModification(script_path="/tmp/test.py"),
        execution=BlenderExecution(success=True, run_dir="/tmp"),
        quality=QualityMetrics(overall_score=70.0, passed=True),
        passed=True,
        score=70.0,
        script_hash="aabbccddee12",
        issue_kinds=["lighting", "materials"],
        change_label="modify_code",
    )
    assert result.script_hash == "aabbccddee12"
    assert result.issue_kinds == ["lighting", "materials"]
    assert result.change_label == "modify_code"


def test_iteration_result_default_manifest_fields():
    result = IterationResult(
        iteration=1,
        script=ScriptModification(script_path="/tmp/test.py"),
        execution=BlenderExecution(success=True, run_dir="/tmp"),
        quality=QualityMetrics(overall_score=50.0, passed=False),
        passed=False,
        score=50.0,
    )
    assert result.script_hash == ""
    assert result.issue_kinds == []
    assert result.change_label == ""


# -- get_iteration_summary includes manifest fields --

def test_iteration_summary_includes_manifest_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = ArtifactManager("test_session")
        mgr.artifact_dir = Path(tmpdir)

        mgr.write_iteration_from_values(
            iteration=1,
            script_path="/tmp/s.py",
            technique_used="mantaflow_fire",
            score=45.0,
            passed=False,
            script_hash="112233445566",
            issue_kinds=["hero_object", "lookdev"],
            change_label="initial_generation",
        )

        summary = mgr.get_iteration_summary()
        assert "Iter 1" in summary
        assert "45.0" in summary
