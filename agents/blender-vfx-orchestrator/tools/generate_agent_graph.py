#!/usr/bin/env python3
"""
Generate visual graphs of the VFX Orchestrator agent topology.

Usage:
    python tools/generate_agent_graph.py [--output-dir DIR]

This uses the SDK's built-in visualization to create Graphviz diagrams showing:
- Agents (yellow rectangles)
- Tools (green ellipses)
- Handoffs (solid arrows)
- Tool invocations (dotted arrows)
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def patch_isinstance_for_generics():
    """
    Patch isinstance to handle subscripted generics (Agent[SharedContext]).
    This is needed because SDK visualization uses isinstance(x, Agent) which
    fails with subscripted generics in Python 3.9+.
    """
    import builtins
    from typing import get_origin

    original_isinstance = builtins.isinstance

    def patched_isinstance(obj, classinfo):
        try:
            return original_isinstance(obj, classinfo)
        except TypeError:
            # Handle subscripted generics by checking the origin
            origin = get_origin(type(obj))
            if origin is not None:
                return original_isinstance(obj, origin)
            # Try checking against the class name
            if hasattr(classinfo, '__name__'):
                return type(obj).__name__ == classinfo.__name__
            return False

    builtins.isinstance = patched_isinstance
    return original_isinstance


def restore_isinstance(original):
    """Restore the original isinstance function."""
    import builtins
    builtins.isinstance = original


def generate_simple_graphs(output_dir: Path) -> list[str]:
    """Generate simple coordinator graphs (no dependencies)."""
    try:
        from agents.extensions.visualization import draw_graph
    except ImportError:
        print("ERROR: Visualization not available. Install with:")
        print('  pip install "openai-agents[viz]"')
        sys.exit(1)

    from orchestrator import (
        create_modification_coordinator,
        create_quality_gate_coordinator,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    # These coordinators don't require agent dependencies
    coordinators = [
        ("modification_coordinator", create_modification_coordinator, "Modification Strategy Coordinator"),
        ("quality_gate_coordinator", create_quality_gate_coordinator, "Quality Gate Coordinator"),
    ]

    for filename, factory, description in coordinators:
        print(f"Generating: {description}...")
        try:
            agent = factory()
            output_path = output_dir / filename
            draw_graph(agent, filename=str(output_path))
            generated_files.append(f"{output_path}.png")
            print(f"  -> {output_path}.png")
        except Exception as e:
            print(f"  ERROR: {e}")

    return generated_files


async def generate_full_orchestrator_graph(output_dir: Path) -> list[str]:
    """Generate graph for the full orchestrator with all agents."""
    try:
        from agents.extensions.visualization import draw_graph
    except ImportError:
        print("ERROR: Visualization not available.")
        return []

    from orchestrator import BlenderVFXOrchestrator

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    # Patch isinstance to handle subscripted generics (Agent[SharedContext])
    original_isinstance = patch_isinstance_for_generics()

    print("Initializing orchestrator (this may take a moment)...")
    orchestrator = BlenderVFXOrchestrator()
    await orchestrator.initialize()

    # Generate graphs for standalone agents
    standalone_agents = [
        ("research_agent", orchestrator._research_agent, "Research Agent"),
        ("script_writer", orchestrator._script_agent_standalone, "Script Writer"),
        ("executor", orchestrator._executor_agent_standalone, "Executor"),
        ("quality_analyst", orchestrator._quality_agent_standalone, "Quality Analyst"),
        ("learning_agent", orchestrator._learning_agent_standalone, "Learning Agent"),
    ]

    for filename, agent, description in standalone_agents:
        print(f"Generating: {description}...")
        try:
            output_path = output_dir / filename
            draw_graph(agent, filename=str(output_path))
            generated_files.append(f"{output_path}.png")
            print(f"  -> {output_path}.png")
        except Exception as e:
            print(f"  ERROR: {e}")

    # Generate graphs for coordinators
    coordinator_agents = [
        ("technique_coordinator", orchestrator._technique_coordinator, "Technique Coordinator"),
        ("modification_coordinator", orchestrator._modification_coordinator, "Modification Coordinator"),
        ("quality_gate_coordinator", orchestrator._quality_gate_coordinator, "Quality Gate Coordinator"),
    ]

    for filename, agent, description in coordinator_agents:
        print(f"Generating: {description}...")
        try:
            output_path = output_dir / filename
            draw_graph(agent, filename=str(output_path))
            generated_files.append(f"{output_path}.png")
            print(f"  -> {output_path}.png")
        except Exception as e:
            print(f"  ERROR: {e}")

    # Restore original isinstance
    restore_isinstance(original_isinstance)

    return generated_files


def check_graphviz_installed() -> bool:
    """Check if graphviz system package is installed."""
    import shutil
    if shutil.which("dot") is None:
        print("ERROR: Graphviz system package not installed.")
        print()
        print("Install with:")
        print("  Ubuntu/Debian: sudo apt-get install graphviz")
        print("  macOS:         brew install graphviz")
        print("  Windows:       choco install graphviz")
        print()
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate visual graphs of VFX Orchestrator agent topology"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "docs" / "graphs",
        help="Output directory for generated graphs (default: docs/graphs/)",
    )
    parser.add_argument(
        "--coordinators-only",
        action="store_true",
        help="Only generate simple coordinator graphs (faster, no initialization)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("VFX Orchestrator Agent Graph Generator")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print()

    if not check_graphviz_installed():
        sys.exit(1)

    generated = []

    if args.coordinators_only:
        generated = generate_simple_graphs(args.output_dir)
    else:
        # Default: generate full orchestrator graphs
        generated = asyncio.run(generate_full_orchestrator_graph(args.output_dir))

    print()
    print("=" * 60)
    print(f"Generated {len(generated)} graph(s)")
    for f in generated:
        print(f"  - {f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
