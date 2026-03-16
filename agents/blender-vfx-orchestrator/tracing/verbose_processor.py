"""
Verbose Trace Processor for OpenAI Agents SDK.

Implements TracingProcessor to capture all trace and span data,
outputting detailed logs similar to the OpenAI dashboard.

Usage:
    from tracing import enable_verbose_tracing

    # Enable before running the pipeline
    enable_verbose_tracing(log_file="traces/verbose_run.jsonl")

    # Or manually:
    from tracing import VerboseTraceProcessor
    from agents import add_trace_processor

    processor = VerboseTraceProcessor(log_file="my_trace.jsonl")
    add_trace_processor(processor)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agents.tracing import Trace, Span

# Module-level processor reference for enable/disable
_active_processor: Optional["VerboseTraceProcessor"] = None


class VerboseTraceProcessor:
    """
    TracingProcessor that outputs detailed logs like the OpenAI dashboard.

    Captures:
    - Trace lifecycle (start/end with workflow name, group_id)
    - Span lifecycle (agent runs, tool calls, generations)
    - Span data export (full structured data)
    - Timing information
    - Nested span hierarchy

    Output format matches dashboard columns:
    - Trace ID
    - Workflow name
    - Agent sequence (Flow column)
    - Duration
    - Status
    """

    def __init__(
        self,
        log_file: Optional[str] = None,
        console_output: bool = True,
        include_span_data: bool = True,
        pretty_print: bool = True,
    ):
        """
        Initialize verbose trace processor.

        Args:
            log_file: Path to JSONL file for persistent logs
            console_output: Whether to print to stderr
            include_span_data: Include full span data export in logs
            pretty_print: Use formatted console output (vs raw JSON)
        """
        self.log_file = Path(log_file) if log_file else None
        self.console_output = console_output
        self.include_span_data = include_span_data
        self.pretty_print = pretty_print

        # Active traces and spans
        self.active_traces: Dict[str, Dict[str, Any]] = {}
        self.active_spans: Dict[str, Dict[str, Any]] = {}

        # Completed traces for summary
        self.completed_traces: List[Dict[str, Any]] = []

        # Span hierarchy tracking
        self._span_stack: List[str] = []

        # Ensure log directory exists
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def _timestamp(self) -> str:
        """Get current timestamp."""
        return datetime.now().isoformat()

    def _log(self, event_type: str, data: Dict[str, Any]):
        """Log an event to file and/or console."""
        record = {
            "timestamp": self._timestamp(),
            "event_type": event_type,
            **data
        }

        # Write to JSONL file
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(record) + "\n")

        # Console output
        if self.console_output:
            if self.pretty_print:
                self._pretty_print(event_type, data)
            else:
                print(f"[TRACE] {json.dumps(record)}", file=sys.stderr)

    def _pretty_print(self, event_type: str, data: Dict[str, Any]):
        """Print formatted console output."""
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        indent = "  " * len(self._span_stack)

        if event_type == "trace_start":
            print(f"\n{'='*70}", file=sys.stderr)
            print(f"[{ts}] TRACE START: {data.get('name', '?')}", file=sys.stderr)
            print(f"         ID: {data.get('trace_id', '?')[:16]}...", file=sys.stderr)
            if data.get('group_id'):
                print(f"         Group: {data.get('group_id')}", file=sys.stderr)
            print(f"{'='*70}", file=sys.stderr)

        elif event_type == "trace_end":
            duration = data.get('duration_ms', 0)
            status = data.get('status', 'unknown')
            print(f"\n{'='*70}", file=sys.stderr)
            print(f"[{ts}] TRACE END: {data.get('name', '?')}", file=sys.stderr)
            print(f"         Duration: {duration:.0f}ms", file=sys.stderr)
            print(f"         Status: {status}", file=sys.stderr)
            print(f"         Spans: {data.get('span_count', 0)}", file=sys.stderr)
            print(f"{'='*70}\n", file=sys.stderr)

        elif event_type == "span_start":
            span_type = data.get('span_type', 'unknown')
            name = data.get('name', '?')

            # Format based on span type
            if span_type == "agent":
                icon = "[AGENT]"
            elif span_type == "function":
                icon = "[TOOL]"
            elif span_type == "generation":
                icon = "[LLM]"
            elif span_type == "handoff":
                icon = "[HANDOFF]"
            elif span_type == "guardrail":
                icon = "[GUARD]"
            else:
                icon = f"[{span_type.upper()}]"

            print(f"[{ts}] {indent}{icon} START: {name}", file=sys.stderr)

        elif event_type == "span_end":
            span_type = data.get('span_type', 'unknown')
            name = data.get('name', '?')
            duration = data.get('duration_ms', 0)

            # Show span data summary if available
            span_data = data.get('span_data', {})
            summary = ""

            if span_type == "agent":
                if 'output' in span_data:
                    output_preview = str(span_data['output'])[:50]
                    summary = f" -> {output_preview}..."
            elif span_type == "function":
                if 'result' in span_data:
                    result_preview = str(span_data['result'])[:50]
                    summary = f" -> {result_preview}..."
            elif span_type == "generation":
                if 'output_tokens' in span_data:
                    summary = f" ({span_data.get('input_tokens', 0)} in / {span_data['output_tokens']} out)"

            print(f"[{ts}] {indent}END ({duration:.0f}ms): {name}{summary}", file=sys.stderr)

    # =========================================================================
    # TracingProcessor Interface Implementation
    # =========================================================================

    def on_trace_start(self, trace: "Trace") -> None:
        """Called when a new trace begins."""
        trace_id = getattr(trace, 'trace_id', str(id(trace)))
        name = getattr(trace, 'name', 'unnamed')
        group_id = getattr(trace, 'group_id', None)

        self.active_traces[trace_id] = {
            "trace_id": trace_id,
            "name": name,
            "group_id": group_id,
            "start_time": datetime.now(),
            "spans": [],
        }

        self._log("trace_start", {
            "trace_id": trace_id,
            "name": name,
            "group_id": group_id,
        })

    def on_trace_end(self, trace: "Trace") -> None:
        """Called when a trace completes."""
        trace_id = getattr(trace, 'trace_id', str(id(trace)))
        name = getattr(trace, 'name', 'unnamed')

        trace_data = self.active_traces.pop(trace_id, {})
        start_time = trace_data.get('start_time', datetime.now())
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        completed = {
            "trace_id": trace_id,
            "name": name,
            "duration_ms": duration_ms,
            "span_count": len(trace_data.get('spans', [])),
            "status": "completed",
        }
        self.completed_traces.append(completed)

        self._log("trace_end", completed)

    def on_span_start(self, span: "Span[Any]") -> None:
        """Called when a new span begins."""
        span_id = getattr(span, 'span_id', str(id(span)))

        # Extract span metadata
        span_data_obj = getattr(span, 'span_data', None)
        span_type = "unknown"
        name = "unnamed"

        if span_data_obj:
            span_type = getattr(span_data_obj, 'type', 'unknown')
            # Try to get name from various attributes
            name = (
                getattr(span_data_obj, 'name', None) or
                getattr(span_data_obj, 'agent_name', None) or
                getattr(span_data_obj, 'tool_name', None) or
                getattr(span_data_obj, 'function_name', None) or
                span_type
            )

        self.active_spans[span_id] = {
            "span_id": span_id,
            "span_type": span_type,
            "name": name,
            "start_time": datetime.now(),
            "parent_span_id": self._span_stack[-1] if self._span_stack else None,
        }

        self._span_stack.append(span_id)

        self._log("span_start", {
            "span_id": span_id,
            "span_type": span_type,
            "name": name,
            "depth": len(self._span_stack),
        })

    def on_span_end(self, span: "Span[Any]") -> None:
        """Called when a span completes."""
        span_id = getattr(span, 'span_id', str(id(span)))

        # Pop from stack
        if self._span_stack and self._span_stack[-1] == span_id:
            self._span_stack.pop()

        span_info = self.active_spans.pop(span_id, {})
        start_time = span_info.get('start_time', datetime.now())
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Extract span data for export
        exported_data = {}
        if self.include_span_data:
            span_data_obj = getattr(span, 'span_data', None)
            if span_data_obj and hasattr(span_data_obj, 'export'):
                try:
                    exported_data = span_data_obj.export()
                except Exception:
                    pass

        self._log("span_end", {
            "span_id": span_id,
            "span_type": span_info.get('span_type', 'unknown'),
            "name": span_info.get('name', 'unnamed'),
            "duration_ms": duration_ms,
            "span_data": exported_data,
        })

    def shutdown(self) -> None:
        """Called on application shutdown."""
        if self.console_output and self.completed_traces:
            print(f"\n{'='*70}", file=sys.stderr)
            print("TRACING SESSION SUMMARY", file=sys.stderr)
            print(f"{'='*70}", file=sys.stderr)
            print(f"Total traces: {len(self.completed_traces)}", file=sys.stderr)
            total_duration = sum(t.get('duration_ms', 0) for t in self.completed_traces)
            print(f"Total duration: {total_duration:.0f}ms", file=sys.stderr)
            print(f"{'='*70}\n", file=sys.stderr)

    def force_flush(self) -> None:
        """Force processing of queued items."""
        # No batching, so nothing to flush
        pass


def enable_verbose_tracing(
    log_file: Optional[str] = None,
    console_output: bool = True,
    include_span_data: bool = True,
) -> VerboseTraceProcessor:
    """
    Enable verbose tracing globally.

    Args:
        log_file: Path to JSONL file for logs (default: traces/verbose_{timestamp}.jsonl)
        console_output: Print to stderr
        include_span_data: Include full span data in logs

    Returns:
        The active VerboseTraceProcessor instance

    Example:
        from tracing import enable_verbose_tracing

        enable_verbose_tracing(log_file="traces/my_run.jsonl")
        # Now all traces will be logged verbosely
    """
    global _active_processor

    from agents import add_trace_processor

    # Default log file with timestamp
    if log_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"traces/verbose_{ts}.jsonl"

    processor = VerboseTraceProcessor(
        log_file=log_file,
        console_output=console_output,
        include_span_data=include_span_data,
    )

    add_trace_processor(processor)
    _active_processor = processor

    print(f"[TRACING] Verbose tracing enabled", file=sys.stderr)
    if log_file:
        print(f"[TRACING] Log file: {log_file}", file=sys.stderr)

    return processor


def disable_verbose_tracing():
    """
    Disable verbose tracing.

    Note: Due to SDK limitations, this only shuts down the processor
    but does not remove it from the processor list. For a fresh start,
    use set_trace_processors([]) from the SDK.
    """
    global _active_processor

    if _active_processor:
        _active_processor.shutdown()
        _active_processor = None
        print(f"[TRACING] Verbose tracing disabled", file=sys.stderr)


def pipeline_event(
    event: str,
    data: Dict[str, Any] | None = None,
) -> None:
    """
    Emit a pipeline-level event to the active trace JSONL file.

    These events capture pipeline decisions that the SDK TracingProcessor
    cannot see (repair intent, truth pack fixes, execution results, phase
    transitions, stuck-state changes). They are written alongside SDK spans
    so that E2E verification and manifest integrity checks have a single
    source of truth.

    Args:
        event: Short event name, e.g. "repair_intent", "truth_pack_fix",
               "execution_result", "phase_transition", "iteration_start".
        data: Arbitrary structured data for the event.

    No-op when verbose tracing is not enabled.
    """
    if _active_processor is None or _active_processor.log_file is None:
        return

    record = {
        "timestamp": datetime.now().isoformat(),
        "event_type": "pipeline_event",
        "event": event,
        **(data or {}),
    }
    with open(_active_processor.log_file, "a") as f:
        f.write(json.dumps(record) + "\n")
