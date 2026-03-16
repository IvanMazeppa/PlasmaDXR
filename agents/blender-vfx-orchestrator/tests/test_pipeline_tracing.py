"""Tests for pipeline_event tracing."""

import json
import tempfile
from pathlib import Path

from tracing.verbose_processor import VerboseTraceProcessor, pipeline_event, _active_processor
import tracing.verbose_processor as _mod


class TestPipelineEvent:
    """Verify pipeline_event writes structured JSONL alongside SDK spans."""

    def test_emits_to_jsonl_when_active(self, tmp_path):
        log_file = tmp_path / "test_trace.jsonl"
        proc = VerboseTraceProcessor(log_file=str(log_file), console_output=False)
        old = _mod._active_processor
        _mod._active_processor = proc
        try:
            pipeline_event("repair_intent", {
                "mode": "modify_code",
                "trigger": "execution_failure",
                "confidence": 0.95,
                "iteration": 2,
            })
            lines = log_file.read_text().strip().splitlines()
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert record["event_type"] == "pipeline_event"
            assert record["event"] == "repair_intent"
            assert record["mode"] == "modify_code"
            assert record["confidence"] == 0.95
            assert "timestamp" in record
        finally:
            _mod._active_processor = old

    def test_noop_when_no_processor(self):
        old = _mod._active_processor
        _mod._active_processor = None
        try:
            # Should not raise
            pipeline_event("test_event", {"key": "value"})
        finally:
            _mod._active_processor = old

    def test_noop_when_no_log_file(self):
        proc = VerboseTraceProcessor(log_file=None, console_output=False)
        old = _mod._active_processor
        _mod._active_processor = proc
        try:
            pipeline_event("test_event", {"key": "value"})
        finally:
            _mod._active_processor = old

    def test_multiple_events_append(self, tmp_path):
        log_file = tmp_path / "test_multi.jsonl"
        proc = VerboseTraceProcessor(log_file=str(log_file), console_output=False)
        old = _mod._active_processor
        _mod._active_processor = proc
        try:
            pipeline_event("iteration_start", {"iteration": 1})
            pipeline_event("truth_pack_fix", {"fixes_count": 5})
            pipeline_event("execution_result", {"success": True})
            pipeline_event("iteration_result", {"score": 42.0, "passed": False})

            lines = log_file.read_text().strip().splitlines()
            assert len(lines) == 4
            events = [json.loads(l)["event"] for l in lines]
            assert events == [
                "iteration_start",
                "truth_pack_fix",
                "execution_result",
                "iteration_result",
            ]
        finally:
            _mod._active_processor = old

    def test_empty_data_allowed(self, tmp_path):
        log_file = tmp_path / "test_empty.jsonl"
        proc = VerboseTraceProcessor(log_file=str(log_file), console_output=False)
        old = _mod._active_processor
        _mod._active_processor = proc
        try:
            pipeline_event("phase_transition")
            lines = log_file.read_text().strip().splitlines()
            record = json.loads(lines[0])
            assert record["event"] == "phase_transition"
            assert record["event_type"] == "pipeline_event"
        finally:
            _mod._active_processor = old
