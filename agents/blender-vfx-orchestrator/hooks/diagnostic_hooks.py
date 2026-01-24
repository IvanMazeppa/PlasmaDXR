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
