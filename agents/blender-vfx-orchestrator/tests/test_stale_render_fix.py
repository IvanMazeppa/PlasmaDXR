"""Tests for stale render reuse fix (Wave 2A P0).

Validates that:
- Render discovery is scoped to the current run_dir only
- Failed executions are never promoted to success
- partial_render_path is set for diagnostic purposes on failure
"""

import os
import tempfile

import pytest

from models.pipeline_models import ExecutionOutput
from phases.execution import _discover_render_current_run, _apply_render_discovery


@pytest.fixture
def tmp_dirs(tmp_path):
    """Create run_dir, sibling dir, and shared output dir with renders."""
    run_dir = tmp_path / "run_20260313_141502"
    run_dir.mkdir()
    sibling = tmp_path / "run_20260313_120000"
    sibling.mkdir()
    shared_output = tmp_path / "vdb_output" / "explosion_001"
    shared_output.mkdir(parents=True)
    return run_dir, sibling, shared_output


class TestDiscoverRenderCurrentRun:
    def test_finds_in_run_dir(self, tmp_dirs):
        run_dir, _, _ = tmp_dirs
        render = run_dir / "render_0001.png"
        render.write_text("fake")
        assert _discover_render_current_run(str(run_dir)) == str(render)

    def test_finds_in_subdir(self, tmp_dirs):
        run_dir, _, _ = tmp_dirs
        subdir = run_dir / "renders"
        subdir.mkdir()
        render = subdir / "frame_0001.exr"
        render.write_text("fake")
        assert _discover_render_current_run(str(run_dir)) == str(render)

    def test_ignores_sibling_dirs(self, tmp_dirs):
        run_dir, sibling, _ = tmp_dirs
        # Stale render in sibling — must NOT be found
        (sibling / "old_render.png").write_text("stale")
        assert _discover_render_current_run(str(run_dir)) is None

    def test_ignores_shared_output_dir(self, tmp_dirs):
        run_dir, _, shared_output = tmp_dirs
        (shared_output / "prev_iter.png").write_text("stale")
        assert _discover_render_current_run(str(run_dir)) is None

    def test_no_run_dir_returns_none(self):
        assert _discover_render_current_run(None) is None

    def test_nonexistent_run_dir_returns_none(self):
        assert _discover_render_current_run("/nonexistent/path") is None


class TestApplyRenderDiscovery:
    def test_success_fills_render_path(self, tmp_dirs):
        run_dir, _, shared_output = tmp_dirs
        render = run_dir / "render.png"
        render.write_text("fake")
        execution = ExecutionOutput(success=True, run_dir=str(run_dir))
        _apply_render_discovery(execution, str(shared_output))
        assert execution.render_path == str(render)
        assert execution.success is True

    def test_success_keeps_existing_render_path(self, tmp_dirs):
        run_dir, _, shared_output = tmp_dirs
        (run_dir / "new_render.png").write_text("fake")
        original = "/original/render.png"
        execution = ExecutionOutput(success=True, render_path=original, run_dir=str(run_dir))
        _apply_render_discovery(execution, str(shared_output))
        assert execution.render_path == original  # Untouched

    def test_failure_never_promotes(self, tmp_dirs):
        run_dir, _, shared_output = tmp_dirs
        (run_dir / "partial.png").write_text("fake")
        execution = ExecutionOutput(
            success=False,
            error_message="Script crashed",
            run_dir=str(run_dir),
        )
        _apply_render_discovery(execution, str(shared_output))
        assert execution.success is False
        assert execution.render_path is None

    def test_failure_sets_partial_render_path(self, tmp_dirs):
        run_dir, _, shared_output = tmp_dirs
        partial = run_dir / "partial.png"
        partial.write_text("fake")
        execution = ExecutionOutput(
            success=False,
            error_message="Script crashed",
            run_dir=str(run_dir),
        )
        _apply_render_discovery(execution, str(shared_output))
        assert execution.partial_render_path == str(partial)

    def test_failure_no_render_no_partial(self, tmp_dirs):
        run_dir, _, shared_output = tmp_dirs
        execution = ExecutionOutput(
            success=False,
            error_message="Script crashed",
            run_dir=str(run_dir),
        )
        _apply_render_discovery(execution, str(shared_output))
        assert execution.partial_render_path is None
        assert execution.render_path is None

    def test_stale_render_in_shared_dir_not_used(self, tmp_dirs):
        """The core bug scenario: stale render in shared dir must NOT be picked up."""
        run_dir, _, shared_output = tmp_dirs
        (shared_output / "iteration_1_render.png").write_text("stale from iter 1")
        execution = ExecutionOutput(
            success=False,
            error_message="Iter 2 crashed",
            run_dir=str(run_dir),
        )
        _apply_render_discovery(execution, str(shared_output))
        assert execution.success is False
        assert execution.render_path is None
        assert execution.partial_render_path is None
