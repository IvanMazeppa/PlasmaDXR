#!/usr/bin/env python3
"""
Trace Analyzer for Blender VFX Orchestrator

Features:
- Parse JSONL trace files and extract key metrics
- Build span tree with proper parent-child relationships
- Fix tool attribution (assigns tools to correct parent agent)
- Extract guardrail, quality, Blender execution, and LLM metrics
- Auto-generate compact summaries (debloat)
- Watch folder for new traces
- Compare traces for performance regression
- Export data for charting/visualization

Usage:
    # Analyze single trace
    python trace_analyzer.py traces/my_trace.jsonl

    # Analyze all traces in folder
    python trace_analyzer.py traces/

    # Watch for new traces
    python trace_analyzer.py traces/ --watch

    # Compare two traces
    python trace_analyzer.py traces/old.jsonl traces/new.jsonl --compare

    # Export CSV for charting
    python trace_analyzer.py traces/ --export-csv metrics.csv
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import csv


# --- Dataclasses ---

@dataclass
class SpanNode:
    """A single span in the trace tree."""
    span_id: str
    span_type: str  # "agent", "function", "response", "guardrail"
    name: str
    depth: int
    start_ts: datetime
    end_ts: Optional[datetime] = None
    duration_ms: float = 0.0
    span_data: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)


@dataclass
class AgentMetrics:
    """Metrics for a single agent."""
    name: str
    call_count: int = 0
    total_duration_s: float = 0.0
    tool_calls: Dict[str, int] = field(default_factory=dict)
    doc_searches: int = 0
    errors: int = 0

    def add_tool_call(self, tool_name: str):
        self.tool_calls[tool_name] = self.tool_calls.get(tool_name, 0) + 1
        if any(doc in tool_name.lower() for doc in ['doc', 'search', 'semantic']):
            self.doc_searches += 1


@dataclass
class ToolCallMetrics:
    """Metrics for a single tool across the trace."""
    name: str
    call_count: int = 0
    total_duration_ms: float = 0.0


@dataclass
class GuardrailMetrics:
    """Metrics for a single guardrail across the trace."""
    name: str
    check_count: int = 0
    trigger_count: int = 0


@dataclass
class TraceMetrics:
    """Aggregated metrics for a trace file."""
    trace_file: str
    trace_name: str = ""
    timestamp: str = ""
    total_duration_s: float = 0.0
    agent_metrics: Dict[str, AgentMetrics] = field(default_factory=dict)
    total_tool_calls: int = 0
    total_doc_searches: int = 0
    total_errors: int = 0
    iteration_count: int = 0
    effect_type: str = ""
    quality_score: Optional[float] = None
    status: str = "unknown"

    # Timeline data for charting
    timeline: List[Dict[str, Any]] = field(default_factory=list)

    # --- New fields (backward compatible) ---
    agent_order: List[str] = field(default_factory=list)
    guardrail_metrics: Dict[str, GuardrailMetrics] = field(default_factory=dict)
    tool_metrics: Dict[str, ToolCallMetrics] = field(default_factory=dict)
    llm_call_count: int = 0
    llm_total_duration_ms: float = 0.0
    quality_scores: List[float] = field(default_factory=list)
    quality_passed: Optional[bool] = None
    blender_runs: int = 0
    blender_success_count: int = 0
    blender_total_duration_s: float = 0.0
    score_trajectory: List[float] = field(default_factory=list)
    max_depth: int = 0

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to compact summary dictionary."""
        guardrail_checks = sum(g.check_count for g in self.guardrail_metrics.values())
        guardrail_triggers = sum(g.trigger_count for g in self.guardrail_metrics.values())

        result = {
            "trace_file": Path(self.trace_file).name,
            "trace_name": self.trace_name,
            "timestamp": self.timestamp,
            "total_duration_s": round(self.total_duration_s, 1),
            "total_duration_min": round(self.total_duration_s / 60, 2),
            "effect_type": self.effect_type,
            "status": self.status,
            "quality_score": self.quality_score,
            "iteration_count": self.iteration_count,
            "total_tool_calls": self.total_tool_calls,
            "total_doc_searches": self.total_doc_searches,
            "total_errors": self.total_errors,
            "agents": {
                name: {
                    "calls": m.call_count,
                    "duration_s": round(m.total_duration_s, 1),
                    "pct_of_total": round(m.total_duration_s / self.total_duration_s * 100, 1) if self.total_duration_s > 0 else 0,
                    "doc_searches": m.doc_searches,
                    "top_tools": dict(sorted(m.tool_calls.items(), key=lambda x: -x[1])[:5])
                }
                for name, m in sorted(self.agent_metrics.items(), key=lambda x: -x[1].total_duration_s)
            },
            "efficiency": {
                "doc_searches_per_minute": round(self.total_doc_searches / (self.total_duration_s / 60), 2) if self.total_duration_s > 0 else 0,
                "seconds_per_iteration": round(self.total_duration_s / self.iteration_count, 1) if self.iteration_count > 0 else self.total_duration_s,
            },
            # New fields
            "agent_order": self.agent_order,
            "guardrail_checks": guardrail_checks,
            "guardrail_triggers": guardrail_triggers,
            "guardrail_trigger_rate": round(guardrail_triggers / guardrail_checks, 3) if guardrail_checks > 0 else 0.0,
            "llm_calls": self.llm_call_count,
            "llm_duration_total_s": round(self.llm_total_duration_ms / 1000, 1),
            "llm_duration_mean_s": round((self.llm_total_duration_ms / 1000) / self.llm_call_count, 2) if self.llm_call_count > 0 else 0.0,
            "quality_score_final": self.quality_scores[-1] if self.quality_scores else None,
            "quality_passed": self.quality_passed,
            "blender_runs": self.blender_runs,
            "blender_success_count": self.blender_success_count,
            "blender_total_duration_s": round(self.blender_total_duration_s, 1),
            "score_trajectory": self.score_trajectory,
            "max_depth": self.max_depth,
            # Detail dicts for visualizer JSON consumption
            "tool_details": {
                name: {
                    "call_count": tm.call_count,
                    "total_duration_ms": round(tm.total_duration_ms, 1),
                    "avg_duration_ms": round(tm.total_duration_ms / tm.call_count, 1) if tm.call_count > 0 else 0,
                }
                for name, tm in sorted(self.tool_metrics.items(), key=lambda x: -x[1].total_duration_ms)
            },
            "guardrail_details": {
                name: {
                    "check_count": gm.check_count,
                    "trigger_count": gm.trigger_count,
                    "trigger_rate": round(gm.trigger_count / gm.check_count, 3) if gm.check_count > 0 else 0.0,
                }
                for name, gm in sorted(self.guardrail_metrics.items(), key=lambda x: -x[1].check_count)
            },
        }
        return result


# --- Parsing Helpers ---

def parse_timestamp(ts: str) -> datetime:
    """Parse ISO timestamp."""
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return datetime.now()


def _parse_tool_output(output_str: str) -> Optional[Dict[str, Any]]:
    """Safely parse a tool output JSON string. Returns None if not valid JSON."""
    if not output_str:
        return None
    try:
        parsed = json.loads(output_str)
        if isinstance(parsed, dict):
            return parsed
        return None
    except (json.JSONDecodeError, TypeError):
        return None


def build_span_tree(events: List[Dict[str, Any]]) -> Dict[str, SpanNode]:
    """Build a span tree from trace events using depth-based parent inference.

    Uses depth field to infer parent-child: when a span starts at depth N,
    its parent is the most recent open span at depth N-1.
    """
    nodes: Dict[str, SpanNode] = {}
    # depth_stack[depth] = span_id of the most recent open span at that depth
    depth_stack: Dict[int, str] = {}

    for e in events:
        event_type = e.get("event_type", "")
        span_id = e.get("span_id", "")
        if not span_id:
            continue

        if event_type == "span_start":
            name = e.get("name", "")
            span_type = e.get("span_type", "")
            depth = e.get("depth", 0)
            ts_str = e.get("timestamp", "")
            ts = parse_timestamp(ts_str)

            node = SpanNode(
                span_id=span_id,
                span_type=span_type,
                name=name,
                depth=depth,
                start_ts=ts,
            )

            # Find parent: look for open span at depth-1
            parent_depth = depth - 1
            if parent_depth >= 0 and parent_depth in depth_stack:
                parent_id = depth_stack[parent_depth]
                node.parent_id = parent_id
                if parent_id in nodes:
                    nodes[parent_id].children.append(span_id)

            # Guardrails often start+end instantly at same depth as other spans.
            # Only update depth_stack for non-guardrail spans to avoid displacing.
            if span_type != "guardrail":
                depth_stack[depth] = span_id

            nodes[span_id] = node

        elif event_type == "span_end" and span_id in nodes:
            node = nodes[span_id]
            ts_str = e.get("timestamp", "")
            node.end_ts = parse_timestamp(ts_str)
            node.duration_ms = e.get("duration_ms", 0.0)
            node.span_data = e.get("span_data", {})

    return nodes


def _find_parent_agent(nodes: Dict[str, SpanNode], span_id: str) -> Optional[str]:
    """Walk the parent chain from a span to find its owning agent name."""
    visited = set()
    current = span_id
    while current and current not in visited:
        visited.add(current)
        node = nodes.get(current)
        if not node:
            break
        if node.span_type == "agent":
            return node.name
        current = node.parent_id
    return None


# --- Main Analysis ---

def analyze_trace(trace_path: str) -> TraceMetrics:
    """Analyze a single trace file and extract metrics.

    Two-pass approach:
    1. Build span tree with parent-child relationships
    2. Iterate nodes by type to extract all metrics
    """
    metrics = TraceMetrics(trace_file=trace_path)

    events = []
    with open(trace_path, 'r') as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not events:
        return metrics

    # --- Extract trace-level info from trace_start/trace_end events ---
    first_ts = None
    last_ts = None
    seen_agent_names = set()

    for e in events:
        ts_str = e.get("timestamp")
        if ts_str:
            parsed = parse_timestamp(ts_str)
            if first_ts is None or parsed < first_ts:
                first_ts = parsed
            if last_ts is None or parsed > last_ts:
                last_ts = parsed

        event_type = e.get("event_type", "")

        # Prefer "VFX Pipeline:" trace_start for group_id/effect_type
        if event_type == "trace_start":
            name = e.get("name", "")
            if name.startswith("VFX Pipeline:"):
                metrics.trace_name = name
                metrics.timestamp = e.get("timestamp", "")
                group_id = e.get("group_id", "")
                if group_id:
                    # Extract effect type from group_id (strip timestamp suffix)
                    parts = group_id.split("_")
                    # Try to find the effect name (non-numeric prefix)
                    effect_parts = []
                    for p in parts:
                        if p.isdigit() and len(p) >= 8:
                            break
                        effect_parts.append(p)
                    if effect_parts:
                        metrics.effect_type = "_".join(effect_parts)
            elif not metrics.trace_name:
                metrics.trace_name = name
                metrics.timestamp = e.get("timestamp", "")
                group_id = e.get("group_id", "")
                if group_id and not metrics.effect_type:
                    parts = group_id.split("_")
                    effect_parts = []
                    for p in parts:
                        if p.isdigit() and len(p) >= 8:
                            break
                        effect_parts.append(p)
                    if effect_parts:
                        metrics.effect_type = "_".join(effect_parts)

        if event_type == "trace_end":
            status = e.get("status", "")
            if status:
                metrics.status = status

    # Calculate total duration
    if first_ts and last_ts:
        metrics.total_duration_s = (last_ts - first_ts).total_seconds()

    # --- Pass 1: Build span tree ---
    nodes = build_span_tree(events)

    # --- Pass 2: Iterate nodes by type and extract metrics ---
    for span_id, node in nodes.items():
        # Track max depth
        if node.depth > metrics.max_depth:
            metrics.max_depth = node.depth

        if node.span_type == "agent":
            agent_name = node.name
            if agent_name not in metrics.agent_metrics:
                metrics.agent_metrics[agent_name] = AgentMetrics(name=agent_name)
            am = metrics.agent_metrics[agent_name]
            am.call_count += 1

            # Duration (only from span_end data)
            if node.duration_ms > 0:
                am.total_duration_s += node.duration_ms / 1000.0

            # Track agent order (first appearance)
            if agent_name not in seen_agent_names:
                seen_agent_names.add(agent_name)
                metrics.agent_order.append(agent_name)

            # Add to timeline
            if first_ts and node.duration_ms > 0:
                metrics.timeline.append({
                    "agent": agent_name,
                    "start_offset_s": (node.start_ts - first_ts).total_seconds(),
                    "duration_s": node.duration_ms / 1000.0,
                })

        elif node.span_type == "function":
            tool_name = node.name
            duration_ms = node.duration_ms

            metrics.total_tool_calls += 1

            # Tool metrics (global)
            if tool_name not in metrics.tool_metrics:
                metrics.tool_metrics[tool_name] = ToolCallMetrics(name=tool_name)
            tm = metrics.tool_metrics[tool_name]
            tm.call_count += 1
            tm.total_duration_ms += duration_ms

            # Attribute tool to correct parent agent
            parent_agent = _find_parent_agent(nodes, span_id)
            if parent_agent:
                if parent_agent not in metrics.agent_metrics:
                    metrics.agent_metrics[parent_agent] = AgentMetrics(name=parent_agent)
                metrics.agent_metrics[parent_agent].add_tool_call(tool_name)

            # Parse tool output for specific tools
            output_str = node.span_data.get("output", "")
            output_data = _parse_tool_output(output_str)

            if output_data:
                if tool_name == "evaluate_render":
                    score = output_data.get("overall_score")
                    if score is not None:
                        try:
                            score = float(score)
                            metrics.quality_scores.append(score)
                            metrics.quality_score = score
                        except (ValueError, TypeError):
                            pass
                    passed = output_data.get("passed")
                    if passed is not None:
                        metrics.quality_passed = bool(passed)

                elif tool_name == "analyze_with_vision":
                    score = output_data.get("score")
                    if score is not None:
                        try:
                            score = float(score)
                            metrics.quality_scores.append(score)
                            metrics.quality_score = score
                        except (ValueError, TypeError):
                            pass
                    # Also check overall_score (some versions use this)
                    if score is None:
                        score = output_data.get("overall_score")
                        if score is not None:
                            try:
                                score = float(score)
                                metrics.quality_scores.append(score)
                                metrics.quality_score = score
                            except (ValueError, TypeError):
                                pass

                elif tool_name == "execute_blender_script":
                    metrics.blender_runs += 1
                    success = output_data.get("success", False)
                    if success:
                        metrics.blender_success_count += 1
                    dur_s = output_data.get("duration_seconds", 0)
                    if dur_s:
                        try:
                            metrics.blender_total_duration_s += float(dur_s)
                        except (ValueError, TypeError):
                            pass
                    # Count errors from execution failures
                    if not success:
                        metrics.total_errors += 1
                        if parent_agent and parent_agent in metrics.agent_metrics:
                            metrics.agent_metrics[parent_agent].errors += 1

                elif tool_name == "record_experiment_result":
                    score_changes = output_data.get("score_changes", {})
                    score_after = score_changes.get("score_after")
                    if score_after is None:
                        score_after = score_changes.get("quality_score")
                    if score_after is not None:
                        try:
                            metrics.score_trajectory.append(float(score_after))
                        except (ValueError, TypeError):
                            pass

        elif node.span_type == "response":
            metrics.llm_call_count += 1
            metrics.llm_total_duration_ms += node.duration_ms

        elif node.span_type == "guardrail":
            gr_name = node.name
            if gr_name not in metrics.guardrail_metrics:
                metrics.guardrail_metrics[gr_name] = GuardrailMetrics(name=gr_name)
            gm = metrics.guardrail_metrics[gr_name]
            gm.check_count += 1
            triggered = node.span_data.get("triggered", False)
            if triggered:
                gm.trigger_count += 1

    # Count doc searches from agent metrics
    for am in metrics.agent_metrics.values():
        metrics.total_doc_searches += am.doc_searches

    # Estimate iterations (count Script Writer / Code Writer calls or Executor calls)
    writer_calls = 0
    executor_calls = 0
    for name, am in metrics.agent_metrics.items():
        name_lower = name.lower()
        if "writer" in name_lower or "script" in name_lower:
            writer_calls += am.call_count
        if "executor" in name_lower:
            executor_calls += am.call_count
    metrics.iteration_count = max(writer_calls, executor_calls, 1)

    return metrics


# --- Output ---

def print_summary(metrics: TraceMetrics, verbose: bool = False):
    """Print a formatted summary of trace metrics."""
    summary = metrics.to_summary_dict()

    print(f"\n{'=' * 70}")
    print(f"TRACE: {summary['trace_file']}")
    print(f"{'=' * 70}")
    print(f"Name: {summary['trace_name']}")
    print(f"Time: {summary['timestamp']}")
    print(f"Duration: {summary['total_duration_s']:.1f}s ({summary['total_duration_min']:.2f} min)")
    print(f"Effect: {summary['effect_type'] or 'unknown'}")
    print(f"Status: {summary['status']}")
    print(f"Iterations: {summary['iteration_count']}")
    print(f"Max Depth: {summary['max_depth']}")

    print(f"\n--- Agent Breakdown (order: {' -> '.join(summary['agent_order'][:6])}) ---")
    for agent_name, agent_data in summary['agents'].items():
        print(f"  {agent_name}:")
        print(f"    Duration: {agent_data['duration_s']:.1f}s ({agent_data['pct_of_total']:.1f}%)")
        print(f"    Calls: {agent_data['calls']}, Doc Searches: {agent_data['doc_searches']}")
        if verbose and agent_data['top_tools']:
            print(f"    Top Tools: {agent_data['top_tools']}")

    print(f"\n--- Tool Calls ---")
    print(f"  Total: {summary['total_tool_calls']}")
    print(f"  Doc Searches: {summary['total_doc_searches']}")
    if verbose and summary.get('tool_details'):
        top_tools = list(summary['tool_details'].items())[:10]
        for name, td in top_tools:
            print(f"    {name}: {td['call_count']}x, {td['total_duration_ms']:.0f}ms total, {td['avg_duration_ms']:.0f}ms avg")

    print(f"\n--- LLM Calls ---")
    print(f"  Count: {summary['llm_calls']}")
    print(f"  Total Duration: {summary['llm_duration_total_s']:.1f}s")
    print(f"  Mean Duration: {summary['llm_duration_mean_s']:.2f}s")

    print(f"\n--- Guardrails ---")
    print(f"  Checks: {summary['guardrail_checks']}")
    print(f"  Triggers: {summary['guardrail_triggers']}")
    print(f"  Trigger Rate: {summary['guardrail_trigger_rate']:.1%}")
    if verbose and summary.get('guardrail_details'):
        for name, gd in summary['guardrail_details'].items():
            trig = "*TRIGGERED*" if gd['trigger_count'] > 0 else ""
            print(f"    {name}: {gd['check_count']}x checked, {gd['trigger_count']}x triggered {trig}")

    print(f"\n--- Quality ---")
    final = summary.get('quality_score_final')
    print(f"  Final Score: {final if final is not None else 'N/A'}")
    print(f"  Passed: {summary['quality_passed'] if summary['quality_passed'] is not None else 'N/A'}")
    if metrics.score_trajectory:
        traj_str = " -> ".join(str(int(s)) for s in metrics.score_trajectory)
        print(f"  Score Trajectory: {traj_str}")

    print(f"\n--- Blender Execution ---")
    print(f"  Runs: {summary['blender_runs']}")
    print(f"  Successes: {summary['blender_success_count']}")
    print(f"  Total Bake Time: {summary['blender_total_duration_s']:.1f}s")

    print(f"\n--- Efficiency Metrics ---")
    print(f"  Doc Searches/min: {summary['efficiency']['doc_searches_per_minute']:.2f}")
    print(f"  Seconds/iteration: {summary['efficiency']['seconds_per_iteration']:.1f}")
    if summary['total_errors'] > 0:
        print(f"  Errors: {summary['total_errors']}")


def save_summary(metrics: TraceMetrics, output_path: str):
    """Save compact summary to JSON file."""
    summary = metrics.to_summary_dict()
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {output_path}")


def compare_traces(metrics1: TraceMetrics, metrics2: TraceMetrics):
    """Compare two traces and show differences."""
    print(f"\n{'=' * 70}")
    print(f"COMPARISON: {Path(metrics1.trace_file).name} vs {Path(metrics2.trace_file).name}")
    print(f"{'=' * 70}")

    def fmt_delta(old: float, new: float) -> str:
        if old == 0:
            return f"{new:.1f} (new)"
        delta = new - old
        pct = (delta / old) * 100
        sign = "+" if delta > 0 else ""
        color = "worse" if delta > 0 else "better"
        return f"{new:.1f} ({sign}{delta:.1f}, {sign}{pct:.1f}% {color})"

    print(f"\nTotal Duration:")
    print(f"  OLD: {metrics1.total_duration_s:.1f}s")
    print(f"  NEW: {fmt_delta(metrics1.total_duration_s, metrics2.total_duration_s)}")

    print(f"\nDoc Searches:")
    print(f"  OLD: {metrics1.total_doc_searches}")
    print(f"  NEW: {fmt_delta(metrics1.total_doc_searches, metrics2.total_doc_searches)}")

    print(f"\nLLM Calls:")
    print(f"  OLD: {metrics1.llm_call_count}")
    print(f"  NEW: {fmt_delta(metrics1.llm_call_count, metrics2.llm_call_count)}")

    print(f"\nGuardrail Triggers:")
    g1 = sum(g.trigger_count for g in metrics1.guardrail_metrics.values())
    g2 = sum(g.trigger_count for g in metrics2.guardrail_metrics.values())
    print(f"  OLD: {g1}")
    print(f"  NEW: {fmt_delta(g1, g2)}")

    q1 = metrics1.quality_scores[-1] if metrics1.quality_scores else 0
    q2 = metrics2.quality_scores[-1] if metrics2.quality_scores else 0
    print(f"\nQuality Score (final):")
    print(f"  OLD: {q1}")
    print(f"  NEW: {fmt_delta(q1, q2)}")

    print(f"\nAgent Comparison:")
    all_agents = set(metrics1.agent_metrics.keys()) | set(metrics2.agent_metrics.keys())
    for agent in sorted(all_agents):
        old = metrics1.agent_metrics.get(agent, AgentMetrics(name=agent))
        new = metrics2.agent_metrics.get(agent, AgentMetrics(name=agent))
        print(f"  {agent}:")
        print(f"    Duration: {fmt_delta(old.total_duration_s, new.total_duration_s)}")


def export_csv(all_metrics: List[TraceMetrics], output_path: str):
    """Export metrics to CSV for charting."""
    if not all_metrics:
        print("No metrics to export")
        return

    rows = []
    for m in all_metrics:
        summary = m.to_summary_dict()
        row = {
            "trace_file": summary["trace_file"],
            "timestamp": summary["timestamp"],
            "duration_s": summary["total_duration_s"],
            "duration_min": summary["total_duration_min"],
            "iterations": summary["iteration_count"],
            "tool_calls": summary["total_tool_calls"],
            "doc_searches": summary["total_doc_searches"],
            "doc_searches_per_min": summary["efficiency"]["doc_searches_per_minute"],
            "seconds_per_iteration": summary["efficiency"]["seconds_per_iteration"],
            # New columns
            "agent_order": ",".join(summary["agent_order"]),
            "guardrail_checks": summary["guardrail_checks"],
            "guardrail_triggers": summary["guardrail_triggers"],
            "guardrail_trigger_rate": summary["guardrail_trigger_rate"],
            "tool_calls_total": summary["total_tool_calls"],
            "tool_call_duration_total_s": round(sum(
                td["total_duration_ms"] for td in summary.get("tool_details", {}).values()
            ) / 1000, 1),
            "llm_calls": summary["llm_calls"],
            "llm_duration_total_s": summary["llm_duration_total_s"],
            "llm_duration_mean_s": summary["llm_duration_mean_s"],
            "quality_score_final": summary["quality_score_final"],
            "quality_passed": summary["quality_passed"],
            "blender_runs": summary["blender_runs"],
            "blender_success_count": summary["blender_success_count"],
            "blender_total_duration_s": summary["blender_total_duration_s"],
            "score_trajectory": ",".join(str(int(s)) for s in summary["score_trajectory"]) if summary["score_trajectory"] else "",
            "max_depth": summary["max_depth"],
            "effect_type": summary["effect_type"],
            "status": summary["status"],
            "total_errors": summary["total_errors"],
        }
        # Add agent durations
        for agent_name, agent_data in summary["agents"].items():
            safe_name = agent_name.replace(" ", "_").lower()
            row[f"{safe_name}_duration_s"] = agent_data["duration_s"]
            row[f"{safe_name}_pct"] = agent_data["pct_of_total"]
        rows.append(row)

    # Write CSV
    if rows:
        # Ensure consistent columns across all rows
        all_fields = set()
        for row in rows:
            all_fields.update(row.keys())
        fieldnames = sorted(all_fields)

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        print(f"CSV exported to: {output_path}")


def watch_folder(folder_path: str, interval: float = 5.0):
    """Watch folder for new trace files and analyze them."""
    folder = Path(folder_path)
    seen_files = set()

    # Initial scan
    for f in folder.glob("*.jsonl"):
        seen_files.add(str(f))

    print(f"Watching {folder} for new traces (Ctrl+C to stop)...")
    print(f"Already seen: {len(seen_files)} files")

    try:
        while True:
            time.sleep(interval)

            for f in folder.glob("*.jsonl"):
                fpath = str(f)
                if fpath not in seen_files:
                    seen_files.add(fpath)
                    print(f"\n[NEW TRACE DETECTED] {f.name}")

                    # Wait a moment for file to finish writing
                    time.sleep(1)

                    try:
                        metrics = analyze_trace(fpath)
                        print_summary(metrics)

                        # Auto-save summary
                        summary_path = fpath.replace(".jsonl", "_summary.json")
                        save_summary(metrics, summary_path)
                    except Exception as e:
                        print(f"Error analyzing {f.name}: {e}")

    except KeyboardInterrupt:
        print("\nStopped watching.")


def debloat_folder(folder_path: str):
    """Analyze all traces and create summaries."""
    folder = Path(folder_path)
    all_metrics = []

    print(f"Analyzing all traces in {folder}...")

    for f in sorted(folder.glob("*.jsonl")):
        # Skip summary files
        if "_summary" in f.name:
            continue

        print(f"  Processing: {f.name}...")
        try:
            metrics = analyze_trace(str(f))
            all_metrics.append(metrics)

            # Save individual summary
            summary_path = str(f).replace(".jsonl", "_summary.json")
            save_summary(metrics, summary_path)
        except Exception as e:
            print(f"    Error: {e}")

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Analyze VFX Orchestrator traces")
    parser.add_argument("paths", nargs="+", help="Trace file(s) or folder to analyze")
    parser.add_argument("--watch", action="store_true", help="Watch folder for new traces")
    parser.add_argument("--compare", action="store_true", help="Compare two traces")
    parser.add_argument("--export-csv", metavar="FILE", help="Export metrics to CSV")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed tool usage")
    parser.add_argument("--debloat", action="store_true", help="Generate summaries for all traces")

    args = parser.parse_args()

    if args.watch:
        # Watch mode
        if len(args.paths) != 1 or not Path(args.paths[0]).is_dir():
            print("Watch mode requires a single folder path")
            sys.exit(1)
        watch_folder(args.paths[0])

    elif args.compare:
        # Compare mode
        if len(args.paths) != 2:
            print("Compare mode requires exactly 2 trace files")
            sys.exit(1)
        metrics1 = analyze_trace(args.paths[0])
        metrics2 = analyze_trace(args.paths[1])
        compare_traces(metrics1, metrics2)

    elif args.debloat or (len(args.paths) == 1 and Path(args.paths[0]).is_dir()):
        # Folder mode - analyze all and optionally export
        folder = args.paths[0]
        all_metrics = debloat_folder(folder)

        if args.export_csv:
            export_csv(all_metrics, args.export_csv)

        # Print summary table
        print(f"\n{'=' * 120}")
        print(f"SUMMARY: {len(all_metrics)} traces analyzed")
        print(f"{'=' * 120}")
        print(f"{'Trace':<40} {'Duration':>10} {'Iters':>6} {'Quality':>8} {'LLM#':>6} {'Guard':>6} {'BlndrT':>8} {'Status':>10}")
        print(f"{'-' * 120}")
        for m in sorted(all_metrics, key=lambda x: x.timestamp):
            name = Path(m.trace_file).name[:38]
            q_score = f"{m.quality_scores[-1]:.0f}" if m.quality_scores else "-"
            gr_trig = sum(g.trigger_count for g in m.guardrail_metrics.values())
            gr_check = sum(g.check_count for g in m.guardrail_metrics.values())
            gr_str = f"{gr_trig}/{gr_check}"
            blender_t = f"{m.blender_total_duration_s:.0f}s" if m.blender_total_duration_s > 0 else "-"
            print(f"{name:<40} {m.total_duration_s:>8.1f}s {m.iteration_count:>6} {q_score:>8} {m.llm_call_count:>6} {gr_str:>6} {blender_t:>8} {m.status:>10}")

    else:
        # Single file mode
        for path in args.paths:
            metrics = analyze_trace(path)
            print_summary(metrics, verbose=args.verbose)

            # Auto-save summary
            summary_path = path.replace(".jsonl", "_summary.json")
            save_summary(metrics, summary_path)


if __name__ == "__main__":
    main()
