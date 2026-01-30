#!/usr/bin/env python3
"""
Trace Analyzer for Blender VFX Orchestrator

Features:
- Parse JSONL trace files and extract key metrics
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
from typing import Dict, List, Optional, Any
import csv


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
        # Track doc searches
        if any(doc in tool_name.lower() for doc in ['doc', 'search', 'semantic']):
            self.doc_searches += 1


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

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to compact summary dictionary."""
        return {
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
            }
        }


def parse_timestamp(ts: str) -> datetime:
    """Parse ISO timestamp."""
    try:
        return datetime.fromisoformat(ts)
    except:
        return datetime.now()


def analyze_trace(trace_path: str) -> TraceMetrics:
    """Analyze a single trace file and extract metrics."""
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

    # Extract basic info from first/last events
    first_ts = None
    last_ts = None

    for e in events:
        ts = e.get("timestamp")
        if ts:
            parsed = parse_timestamp(ts)
            if first_ts is None or parsed < first_ts:
                first_ts = parsed
            if last_ts is None or parsed > last_ts:
                last_ts = parsed

        # Get trace name
        if e.get("event_type") == "trace_start":
            metrics.trace_name = e.get("name", "")
            metrics.timestamp = e.get("timestamp", "")

    # Calculate total duration
    if first_ts and last_ts:
        metrics.total_duration_s = (last_ts - first_ts).total_seconds()

    # Track spans for agent timing
    spans: Dict[str, Dict[str, Any]] = {}
    agent_names = [
        "Research Agent", "Technique Selector", "API Spec Agent", "Code Writer Agent",
        "Script Writer", "Executor", "Quality Analyst", "Learning Agent",
        "Modification Strategist", "Quality Gate", "Documentation Expert"
    ]

    for e in events:
        event_type = e.get("event_type", "")
        span_id = e.get("span_id", "")
        name = e.get("name", "")
        ts = e.get("timestamp", "")

        if event_type == "span_start":
            spans[span_id] = {"name": name, "start": ts}

            # Track if it's an agent
            if any(agent in name for agent in agent_names):
                if name not in metrics.agent_metrics:
                    metrics.agent_metrics[name] = AgentMetrics(name=name)
                metrics.agent_metrics[name].call_count += 1

        elif event_type == "span_end" and span_id in spans:
            span_data = spans[span_id]
            span_data["end"] = ts

            # Calculate duration
            if "start" in span_data and "end" in span_data:
                start = parse_timestamp(span_data["start"])
                end = parse_timestamp(span_data["end"])
                duration = (end - start).total_seconds()

                span_name = span_data["name"]

                # Add to agent metrics
                for agent_name in agent_names:
                    if agent_name in span_name:
                        if agent_name not in metrics.agent_metrics:
                            metrics.agent_metrics[agent_name] = AgentMetrics(name=agent_name)
                        metrics.agent_metrics[agent_name].total_duration_s += duration

                        # Add to timeline
                        metrics.timeline.append({
                            "agent": agent_name,
                            "start_offset_s": (start - first_ts).total_seconds() if first_ts else 0,
                            "duration_s": duration,
                        })
                        break

                # Check for tool calls (lowercase snake_case names)
                if span_name and span_name[0].islower() and ("_" in span_name or span_name in ["response"]):
                    metrics.total_tool_calls += 1

                    # Find parent agent
                    for agent_name, agent_metrics in metrics.agent_metrics.items():
                        # Simple heuristic: tool is part of most recent agent
                        agent_metrics.add_tool_call(span_name)
                        break

    # Count doc searches
    for agent_metrics in metrics.agent_metrics.values():
        metrics.total_doc_searches += agent_metrics.doc_searches

    # Estimate iterations (count Script Writer calls or Executor calls)
    script_calls = metrics.agent_metrics.get("Script Writer", AgentMetrics(name="")).call_count
    executor_calls = metrics.agent_metrics.get("Executor", AgentMetrics(name="")).call_count
    metrics.iteration_count = max(script_calls, executor_calls, 1)

    return metrics


def print_summary(metrics: TraceMetrics, verbose: bool = False):
    """Print a formatted summary of trace metrics."""
    summary = metrics.to_summary_dict()

    print(f"\n{'=' * 70}")
    print(f"TRACE: {summary['trace_file']}")
    print(f"{'=' * 70}")
    print(f"Name: {summary['trace_name']}")
    print(f"Time: {summary['timestamp']}")
    print(f"Duration: {summary['total_duration_s']:.1f}s ({summary['total_duration_min']:.2f} min)")
    print(f"Iterations: {summary['iteration_count']}")
    print(f"Tool Calls: {summary['total_tool_calls']}")
    print(f"Doc Searches: {summary['total_doc_searches']}")

    print(f"\n--- Agent Breakdown ---")
    for agent_name, agent_data in summary['agents'].items():
        print(f"  {agent_name}:")
        print(f"    Duration: {agent_data['duration_s']:.1f}s ({agent_data['pct_of_total']:.1f}%)")
        print(f"    Calls: {agent_data['calls']}, Doc Searches: {agent_data['doc_searches']}")
        if verbose and agent_data['top_tools']:
            print(f"    Top Tools: {agent_data['top_tools']}")

    print(f"\n--- Efficiency Metrics ---")
    print(f"  Doc Searches/min: {summary['efficiency']['doc_searches_per_minute']:.2f}")
    print(f"  Seconds/iteration: {summary['efficiency']['seconds_per_iteration']:.1f}")


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
        }
        # Add agent durations
        for agent_name, agent_data in summary["agents"].items():
            safe_name = agent_name.replace(" ", "_").lower()
            row[f"{safe_name}_duration_s"] = agent_data["duration_s"]
            row[f"{safe_name}_pct"] = agent_data["pct_of_total"]
        rows.append(row)

    # Write CSV
    if rows:
        fieldnames = list(rows[0].keys())
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
        print(f"\n{'=' * 90}")
        print(f"SUMMARY: {len(all_metrics)} traces analyzed")
        print(f"{'=' * 90}")
        print(f"{'Trace':<40} {'Duration':>10} {'Iters':>6} {'DocSearch':>10} {'Efficiency':>12}")
        print(f"{'-' * 90}")
        for m in sorted(all_metrics, key=lambda x: x.timestamp):
            name = Path(m.trace_file).name[:38]
            eff = m.total_doc_searches / (m.total_duration_s / 60) if m.total_duration_s > 0 else 0
            print(f"{name:<40} {m.total_duration_s:>8.1f}s {m.iteration_count:>6} {m.total_doc_searches:>10} {eff:>10.2f}/min")

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
