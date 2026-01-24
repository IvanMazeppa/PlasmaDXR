"""
Diagnostic Hooks for Analyzing Agent Behavior Patterns.

This module creates detailed traces of what agents are doing,
what prompts they receive, and what outputs they generate.

Use this to diagnose:
- Why agents produce similar outputs
- Whether legacy code is being triggered
- If training data patterns are overriding instructions

Usage:
    from hooks.diagnostic_hooks import DiagnosticHooks

    hooks = DiagnosticHooks(log_file="agent_trace.jsonl")
    result = await Runner.run(agent, prompt, hooks=hooks)

    # After run, analyze the trace
    hooks.print_summary()
"""

from __future__ import annotations

import json
import hashlib
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from agents import RunHooks, Tool

if TYPE_CHECKING:
    from agents import RunContextWrapper, Agent


@dataclass
class PromptFingerprint:
    """Track unique prompt patterns."""
    hash: str
    prompt_preview: str  # First 200 chars
    full_prompt: str
    count: int = 1
    timestamps: List[str] = field(default_factory=list)


@dataclass
class ToolCallPattern:
    """Track tool call patterns."""
    tool_name: str
    args_hash: str
    args_preview: str
    count: int = 1
    timestamps: List[str] = field(default_factory=list)


@dataclass
class OutputPattern:
    """Track output patterns."""
    output_hash: str
    output_preview: str
    output_type: str
    count: int = 1
    agent_name: str = ""


class DiagnosticHooks(RunHooks):
    """
    Hooks that trace agent behavior for pattern analysis.

    Captures:
    - All prompts sent to agents (with deduplication)
    - All tool calls (with arg fingerprinting)
    - All agent outputs (with similarity detection)
    - Timing information
    """

    def __init__(
        self,
        log_file: Optional[str] = None,
        verbose: bool = True,
        track_patterns: bool = True
    ):
        self.log_file = Path(log_file) if log_file else None
        self.verbose = verbose
        self.track_patterns = track_patterns

        # Pattern tracking
        self.prompt_fingerprints: Dict[str, PromptFingerprint] = {}
        self.tool_patterns: Dict[str, ToolCallPattern] = {}
        self.output_patterns: Dict[str, OutputPattern] = {}

        # Event log
        self.events: List[Dict[str, Any]] = []

        # Current context
        self._current_agent: Optional[str] = None
        self._turn_count = 0

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def _hash_content(self, content: str) -> str:
        """Create a short hash for fingerprinting."""
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _log_event(self, event_type: str, data: Dict[str, Any]):
        """Log an event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "agent": self._current_agent,
            "turn": self._turn_count,
            **data
        }
        self.events.append(event)

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(event) + "\n")

        if self.verbose:
            self._print_event(event)

    def _print_event(self, event: Dict[str, Any]):
        """Print event to stderr."""
        ts = event["timestamp"].split("T")[1][:8]
        agent = event.get("agent", "?")[:15].ljust(15)
        etype = event["type"][:12].ljust(12)

        if event["type"] == "tool_call":
            tool = event.get("tool_name", "?")
            print(f"[DIAG {ts}] {agent} | {etype} | {tool}", file=sys.stderr)
        elif event["type"] == "prompt":
            preview = event.get("preview", "")[:50]
            print(f"[DIAG {ts}] {agent} | {etype} | {preview}...", file=sys.stderr)
        elif event["type"] == "output":
            output_type = event.get("output_type", "?")
            print(f"[DIAG {ts}] {agent} | {etype} | {output_type}", file=sys.stderr)
        elif event["type"] == "pattern_detected":
            pattern = event.get("pattern_type", "?")
            count = event.get("count", 0)
            print(f"[DIAG {ts}] {agent} | ⚠️ PATTERN | {pattern} seen {count}x", file=sys.stderr)

    async def on_agent_start(
        self,
        context: RunContextWrapper,
        agent: Agent,
    ) -> None:
        """Called when an agent starts."""
        self._current_agent = agent.name
        self._turn_count = 0

        self._log_event("agent_start", {
            "agent_name": agent.name,
            "model": getattr(agent, "model", "unknown"),
        })

    async def on_agent_end(
        self,
        context: RunContextWrapper,
        agent: Agent,
        output: Any,
    ) -> None:
        """Called when an agent ends - capture output patterns."""
        output_str = str(output) if output else ""
        output_hash = self._hash_content(output_str)
        output_type = type(output).__name__

        # Track output patterns
        if self.track_patterns:
            if output_hash in self.output_patterns:
                pattern = self.output_patterns[output_hash]
                pattern.count += 1

                # Warn about repeated outputs
                if pattern.count >= 2:
                    self._log_event("pattern_detected", {
                        "pattern_type": "repeated_output",
                        "output_type": output_type,
                        "count": pattern.count,
                        "preview": output_str[:100],
                    })
            else:
                self.output_patterns[output_hash] = OutputPattern(
                    output_hash=output_hash,
                    output_preview=output_str[:200],
                    output_type=output_type,
                    agent_name=agent.name,
                )

        self._log_event("agent_end", {
            "agent_name": agent.name,
            "output_type": output_type,
            "output_hash": output_hash,
            "turns_used": self._turn_count,
        })

    async def on_llm_start(
        self,
        context: RunContextWrapper,
        agent: Agent,
        **kwargs,
    ) -> None:
        """Called before LLM call - capture prompt patterns."""
        self._turn_count += 1

        # Extract prompt from context if available
        prompt = ""
        if hasattr(context, 'input') and context.input:
            if isinstance(context.input, str):
                prompt = context.input
            elif isinstance(context.input, list):
                prompt = str(context.input)

        if prompt and self.track_patterns:
            prompt_hash = self._hash_content(prompt)

            if prompt_hash in self.prompt_fingerprints:
                fp = self.prompt_fingerprints[prompt_hash]
                fp.count += 1
                fp.timestamps.append(datetime.now().isoformat())

                # Warn about repeated prompts
                if fp.count >= 2:
                    self._log_event("pattern_detected", {
                        "pattern_type": "repeated_prompt",
                        "count": fp.count,
                        "preview": prompt[:100],
                    })
            else:
                self.prompt_fingerprints[prompt_hash] = PromptFingerprint(
                    hash=prompt_hash,
                    prompt_preview=prompt[:200],
                    full_prompt=prompt,
                    timestamps=[datetime.now().isoformat()],
                )

        self._log_event("llm_start", {
            "turn": self._turn_count,
        })

    async def on_tool_start(
        self,
        context: RunContextWrapper,
        agent: Agent,
        tool: Tool,
    ) -> None:
        """Called before tool execution - capture tool call patterns."""
        tool_name = tool.name if hasattr(tool, 'name') else str(tool)

        # Try to extract args
        args_str = ""
        if hasattr(tool, '_last_args'):
            args_str = str(tool._last_args)

        args_hash = self._hash_content(f"{tool_name}:{args_str}")

        if self.track_patterns:
            if args_hash in self.tool_patterns:
                pattern = self.tool_patterns[args_hash]
                pattern.count += 1
                pattern.timestamps.append(datetime.now().isoformat())

                # Warn about repeated identical tool calls
                if pattern.count >= 3:
                    self._log_event("pattern_detected", {
                        "pattern_type": "repeated_tool_call",
                        "tool_name": tool_name,
                        "count": pattern.count,
                    })
            else:
                self.tool_patterns[args_hash] = ToolCallPattern(
                    tool_name=tool_name,
                    args_hash=args_hash,
                    args_preview=args_str[:100] if args_str else "",
                    timestamps=[datetime.now().isoformat()],
                )

        self._log_event("tool_call", {
            "tool_name": tool_name,
            "args_hash": args_hash,
        })

    def print_summary(self):
        """Print a summary of detected patterns."""
        print("\n" + "="*60, file=sys.stderr)
        print("DIAGNOSTIC SUMMARY", file=sys.stderr)
        print("="*60, file=sys.stderr)

        print(f"\nTotal events: {len(self.events)}", file=sys.stderr)
        print(f"Unique prompts: {len(self.prompt_fingerprints)}", file=sys.stderr)
        print(f"Unique tool patterns: {len(self.tool_patterns)}", file=sys.stderr)
        print(f"Unique outputs: {len(self.output_patterns)}", file=sys.stderr)

        # Find repeated patterns
        repeated_prompts = [p for p in self.prompt_fingerprints.values() if p.count > 1]
        repeated_tools = [t for t in self.tool_patterns.values() if t.count > 2]
        repeated_outputs = [o for o in self.output_patterns.values() if o.count > 1]

        if repeated_prompts:
            print(f"\n⚠️ REPEATED PROMPTS ({len(repeated_prompts)}):", file=sys.stderr)
            for p in sorted(repeated_prompts, key=lambda x: -x.count)[:5]:
                print(f"  [{p.count}x] {p.prompt_preview[:60]}...", file=sys.stderr)

        if repeated_tools:
            print(f"\n⚠️ REPEATED TOOL CALLS ({len(repeated_tools)}):", file=sys.stderr)
            for t in sorted(repeated_tools, key=lambda x: -x.count)[:5]:
                print(f"  [{t.count}x] {t.tool_name}", file=sys.stderr)

        if repeated_outputs:
            print(f"\n⚠️ REPEATED OUTPUTS ({len(repeated_outputs)}):", file=sys.stderr)
            for o in sorted(repeated_outputs, key=lambda x: -x.count)[:5]:
                print(f"  [{o.count}x] {o.output_type}: {o.output_preview[:40]}...", file=sys.stderr)

        print("\n" + "="*60, file=sys.stderr)

    def get_pattern_report(self) -> Dict[str, Any]:
        """Get a structured report of detected patterns."""
        return {
            "total_events": len(self.events),
            "unique_prompts": len(self.prompt_fingerprints),
            "unique_tool_patterns": len(self.tool_patterns),
            "unique_outputs": len(self.output_patterns),
            "repeated_prompts": [
                {"count": p.count, "preview": p.prompt_preview[:100]}
                for p in self.prompt_fingerprints.values() if p.count > 1
            ],
            "repeated_tool_calls": [
                {"tool": t.tool_name, "count": t.count}
                for t in self.tool_patterns.values() if t.count > 2
            ],
            "repeated_outputs": [
                {"type": o.output_type, "count": o.count, "preview": o.output_preview[:100]}
                for o in self.output_patterns.values() if o.count > 1
            ],
        }

    def analyze_coordinator_flow(self) -> Dict[str, Any]:
        """
        Analyze communication between Coordinators and modify_script.

        This specifically looks for the pattern:
        1. Coordinator outputs a decision with parameter_changes
        2. modify_script is called with those parameters
        3. modify_script returns zero changes_made

        This indicates the communication breakdown where Coordinator
        diagnoses API fixes that modify_script can't apply.
        """
        coordinator_decisions = []
        modify_calls = []

        for event in self.events:
            # Track Coordinator outputs
            if event.get("type") == "agent_end" and "Coordinator" in event.get("agent_name", ""):
                coordinator_decisions.append({
                    "timestamp": event["timestamp"],
                    "agent": event["agent_name"],
                    "output_type": event.get("output_type", "?"),
                    "output_preview": event.get("output_hash", "")[:8],
                })

            # Track modify_script calls
            if event.get("type") == "tool_call" and "modify" in event.get("tool_name", "").lower():
                modify_calls.append({
                    "timestamp": event["timestamp"],
                    "tool": event.get("tool_name"),
                    "args_hash": event.get("args_hash", ""),
                })

        # Detect potential breakdown: Coordinator decision followed by modify with no success
        breakdown_candidates = []
        for i, coord in enumerate(coordinator_decisions):
            # Look for modify_script calls shortly after
            for mod in modify_calls:
                if mod["timestamp"] > coord["timestamp"]:
                    breakdown_candidates.append({
                        "coordinator": coord["agent"],
                        "coord_time": coord["timestamp"],
                        "modify_time": mod["timestamp"],
                        "tool": mod["tool"],
                    })
                    break

        return {
            "coordinator_decisions": len(coordinator_decisions),
            "modify_script_calls": len(modify_calls),
            "potential_breakdowns": breakdown_candidates,
            "recommendation": (
                "Check if Coordinator outputs contain 'replace' or 'fix' language "
                "that modify_script cannot handle (it only modifies Config class params)."
                if breakdown_candidates else
                "No obvious Coordinator → modify_script flow detected."
            )
        }


class CommunicationFlowTracker:
    """
    Tracks the flow of information between agents to detect breakdowns.

    Usage:
        tracker = CommunicationFlowTracker()

        # In orchestrator, after Coordinator decision:
        tracker.record_coordinator_output(
            coordinator="ModificationCoordinator",
            decision="modify_params",
            parameters={"fix_clear": "while_loop"}
        )

        # After modify_script call:
        tracker.record_modify_result(
            changes_made=["TURBULENCE: 2.5 -> 3.0"],
            success=True
        )

        # Analyze
        report = tracker.analyze()
    """

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = Path(log_file) if log_file else None
        self.flows: List[Dict[str, Any]] = []
        self._current_flow: Optional[Dict[str, Any]] = None

    def record_coordinator_output(
        self,
        coordinator: str,
        decision: str,
        parameters: Dict[str, Any],
        reasoning: str = ""
    ):
        """Record a Coordinator's output."""
        self._current_flow = {
            "timestamp": datetime.now().isoformat(),
            "coordinator": coordinator,
            "decision": decision,
            "parameters": parameters,
            "reasoning": reasoning[:200],
            "modify_result": None,
            "success": None,
        }

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(json.dumps({
                    "event": "coordinator_output",
                    **self._current_flow
                }) + "\n")

    def record_modify_result(
        self,
        changes_made: List[str],
        success: bool,
        error: str = ""
    ):
        """Record the result of modify_script."""
        if self._current_flow:
            self._current_flow["modify_result"] = {
                "changes_made": changes_made,
                "success": success,
                "error": error,
            }
            self._current_flow["success"] = success and len(changes_made) > 0

            # Detect breakdown
            if self._current_flow["parameters"] and len(changes_made) == 0:
                print(f"[FLOW TRACKER] ⚠️ BREAKDOWN DETECTED:", file=sys.stderr)
                print(f"  Coordinator: {self._current_flow['coordinator']}", file=sys.stderr)
                print(f"  Decision: {self._current_flow['decision']}", file=sys.stderr)
                print(f"  Parameters: {self._current_flow['parameters']}", file=sys.stderr)
                print(f"  Changes Made: NONE", file=sys.stderr)
                print(f"  -> Coordinator output likely contains API fixes that modify_script cannot apply", file=sys.stderr)

            self.flows.append(self._current_flow)
            self._current_flow = None

            if self.log_file:
                with open(self.log_file, "a") as f:
                    f.write(json.dumps({
                        "event": "flow_complete",
                        **self.flows[-1]
                    }) + "\n")

    def analyze(self) -> Dict[str, Any]:
        """Analyze all flows for patterns."""
        total = len(self.flows)
        successful = sum(1 for f in self.flows if f.get("success"))
        breakdowns = [f for f in self.flows if f.get("parameters") and not f.get("success")]

        return {
            "total_flows": total,
            "successful": successful,
            "breakdowns": len(breakdowns),
            "success_rate": f"{successful/total*100:.1f}%" if total > 0 else "N/A",
            "breakdown_details": [
                {
                    "coordinator": b["coordinator"],
                    "decision": b["decision"],
                    "parameters": b["parameters"],
                }
                for b in breakdowns
            ],
            "recommendation": (
                "HIGH BREAKDOWN RATE - Coordinator outputs are not being applied. "
                "Check if parameters contain code patterns vs Config values."
                if len(breakdowns) > total * 0.5 and total > 0 else
                "Flow appears healthy."
            )
        }
