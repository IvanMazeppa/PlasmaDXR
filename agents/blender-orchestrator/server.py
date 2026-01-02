#!/usr/bin/env python3
"""
Blender VFX Orchestrator - MCP Server + CLI Entry Point

This is a HYBRID agent: both an MCP server (exposing tools to Claude Code)
AND a Claude Agent SDK client (for autonomous reasoning).

Usage:
    # MCP Server mode (for Claude Code integration)
    python server.py --mcp

    # Interactive mode
    python server.py

    # Single asset generation
    python server.py create "explosion_01" "pyro" "A dramatic fireball explosion"

    # Resume session
    python server.py resume "explosion_01_20250101_120000"

    # List sessions
    python server.py list [--status in_progress|completed|failed]

    # Status
    python server.py status
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Import MCP server factory
try:
    from claude_agent_sdk import create_sdk_mcp_server
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Load environment variables
load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("blender-orchestrator")

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Blender VFX Orchestrator - Autonomous asset generation agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python server.py

  # Create a new asset
  python server.py create explosion_01 pyro "A dramatic fireball explosion"

  # Create with options
  python server.py create sun_surface pyro "Realistic sun surface" \\
      --resolution 128 --frames 100 --technique rising_mushroom

  # Resume an interrupted session
  python server.py resume explosion_01_20250101_120000

  # List all sessions
  python server.py list

  # List only in-progress sessions
  python server.py list --status in_progress

  # Check orchestrator status
  python server.py status

  # Set trust score manually
  python server.py trust 0.8

  # Override autonomy level
  python server.py autonomy supervised
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Interactive mode (default)
    interactive_parser = subparsers.add_parser(
        "interactive", help="Interactive mode (default)"
    )

    # Create asset
    create_parser = subparsers.add_parser("create", help="Create a new VFX asset")
    create_parser.add_argument("asset_name", help="Name for the asset")
    create_parser.add_argument(
        "effect_type",
        choices=["pyro", "explosion", "fire", "smoke", "nebula", "sun", "star"],
        help="Type of VFX effect",
    )
    create_parser.add_argument("description", help="Description of what to create")
    create_parser.add_argument(
        "--reference", "-r", help="Path to reference image"
    )
    create_parser.add_argument(
        "--query", "-q", help="Semantic query for evaluation"
    )
    create_parser.add_argument(
        "--resolution", "-R", type=int, default=96, help="Simulation resolution"
    )
    create_parser.add_argument(
        "--frames", "-f", type=int, default=50, help="Animation end frame"
    )
    create_parser.add_argument(
        "--technique", "-t", help="Specific technique to use"
    )

    # Resume session
    resume_parser = subparsers.add_parser("resume", help="Resume an interrupted session")
    resume_parser.add_argument("session_id", help="Session ID to resume")

    # List sessions
    list_parser = subparsers.add_parser("list", help="List available sessions")
    list_parser.add_argument(
        "--status", "-s",
        choices=["in_progress", "completed", "failed", "paused"],
        help="Filter by status",
    )
    list_parser.add_argument(
        "--limit", "-l", type=int, default=20, help="Maximum sessions to list"
    )

    # Status
    status_parser = subparsers.add_parser("status", help="Show orchestrator status")

    # Trust score
    trust_parser = subparsers.add_parser("trust", help="Set trust score")
    trust_parser.add_argument(
        "score", type=float, help="Trust score (0.0 - 1.0)"
    )

    # Autonomy override
    autonomy_parser = subparsers.add_parser("autonomy", help="Set autonomy level override")
    autonomy_parser.add_argument(
        "level",
        choices=["supervised", "guided", "autonomous", "trusted", "auto"],
        help="Autonomy level (auto = use trust score)",
    )

    return parser


async def cmd_interactive(orchestrator):
    """Run interactive mode."""
    print("\n" + "=" * 80)
    print("Blender VFX Orchestrator - Interactive Mode")
    print("=" * 80)
    print(f"\nTrust Score: {orchestrator.autonomy.get_trust_score():.2f}")
    print(f"Autonomy Level: {orchestrator.autonomy.get_level().value}")
    print("\nCommands:")
    print("  create <name> <type> <description>  - Create new asset")
    print("  resume <session_id>                  - Resume session")
    print("  list                                 - List sessions")
    print("  status                               - Show status")
    print("  trust <score>                        - Set trust score")
    print("  autonomy <level>                     - Set autonomy level")
    print("  exit                                 - Exit")
    print()

    while True:
        try:
            user_input = input("> ").strip()

            if not user_input:
                continue

            parts = user_input.split(maxsplit=3)
            cmd = parts[0].lower()

            if cmd in ["exit", "quit"]:
                print("Goodbye!")
                break

            elif cmd == "create" and len(parts) >= 4:
                result = await orchestrator.create_asset(
                    asset_name=parts[1],
                    effect_type=parts[2],
                    description=parts[3],
                )
                print(f"\nResult: {json.dumps(result, indent=2)}")

            elif cmd == "resume" and len(parts) >= 2:
                result = await orchestrator.resume_session(parts[1])
                print(f"\nResult: {json.dumps(result, indent=2)}")

            elif cmd == "list":
                sessions = orchestrator.list_sessions()
                if sessions:
                    print("\nSessions:")
                    for s in sessions:
                        print(f"  {s['session_id']}: {s['status']} "
                              f"(score: {s['best_score']:.1f}, iter: {s['iterations']})")
                else:
                    print("\nNo sessions found.")

            elif cmd == "status":
                status = orchestrator.get_status()
                print(f"\nAutonomy:")
                print(f"  Trust: {status['autonomy']['trust_score']:.2f}")
                print(f"  Level: {status['autonomy']['autonomy_level']}")
                if status['session']:
                    print(f"\nActive Session:")
                    print(f"  ID: {status['session']['session_id']}")
                    print(f"  Status: {status['session']['status']}")

            elif cmd == "trust" and len(parts) >= 2:
                score = float(parts[1])
                orchestrator.set_trust_score(score)
                print(f"Trust score set to {score:.2f}")

            elif cmd == "autonomy" and len(parts) >= 2:
                level = parts[1] if parts[1] != "auto" else None
                orchestrator.set_autonomy_override(level)
                print(f"Autonomy level: {orchestrator.autonomy.get_level().value}")

            else:
                print("Unknown command or missing arguments. Type 'help' for usage.")

        except KeyboardInterrupt:
            print("\nInterrupted. Type 'exit' to quit.")
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            print(f"Error: {e}")


async def cmd_create(args, orchestrator):
    """Create a new asset."""
    print(f"\nCreating asset: {args.asset_name}")
    print(f"  Effect: {args.effect_type}")
    print(f"  Description: {args.description}")
    print(f"  Resolution: {args.resolution}")
    print(f"  Frames: 1-{args.frames}")

    if args.technique:
        print(f"  Technique: {args.technique}")

    print()

    result = await orchestrator.create_asset(
        asset_name=args.asset_name,
        effect_type=args.effect_type,
        description=args.description,
        reference_path=args.reference,
        semantic_query=args.query,
        resolution=args.resolution,
        frame_end=args.frames,
        technique_name=args.technique,
    )

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    print(json.dumps(result, indent=2))


async def cmd_resume(args, orchestrator):
    """Resume an interrupted session."""
    print(f"\nResuming session: {args.session_id}")

    result = await orchestrator.resume_session(args.session_id)

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    print(json.dumps(result, indent=2))


def cmd_list(args, orchestrator):
    """List sessions."""
    sessions = orchestrator.list_sessions(
        status_filter=args.status,
        limit=args.limit,
    )

    if not sessions:
        print("No sessions found.")
        return

    print(f"\n{'Session ID':<40} {'Status':<12} {'Score':<8} {'Iter':<6} {'Updated'}")
    print("-" * 80)

    for s in sessions:
        updated = s.get('updated_at', '')[:19] if s.get('updated_at') else ''
        print(f"{s['session_id']:<40} {s['status']:<12} "
              f"{s['best_score']:<8.1f} {s['iterations']:<6} {updated}")


def cmd_status(orchestrator):
    """Show orchestrator status."""
    status = orchestrator.get_status()

    print("\n" + "=" * 80)
    print("ORCHESTRATOR STATUS")
    print("=" * 80)

    print("\nAutonomy:")
    print(f"  Trust Score: {status['autonomy']['trust_score']:.2f}")
    print(f"  Level: {status['autonomy']['autonomy_level']}")
    if status['autonomy']['override_active']:
        print(f"  Override: {status['autonomy']['override_level']}")

    if status['autonomy']['recent_events']:
        print("\n  Recent Events:")
        for event in status['autonomy']['recent_events'][-5:]:
            sign = "+" if event['adjustment'] > 0 else ""
            print(f"    {event['event']}: {sign}{event['adjustment']:.2f}")

    if status['guardrails']:
        print("\nGuardrails:")
        s = status['guardrails']['session']
        d = status['guardrails']['daily']
        print(f"  Session Tokens: {s['tokens_used']:,}/{s['tokens_limit']:,} "
              f"({s['tokens_percent']:.0f}%)")
        print(f"  Session Cost: ${s['cost_usd']:.2f}/${s['cost_limit']:.2f}")
        print(f"  Daily Cost: ${d['cost_usd']:.2f}/${d['cost_limit']:.2f}")

    if status['workflow']:
        w = status['workflow']
        print("\nWorkflow:")
        print(f"  Stage: {w['stage']}")
        print(f"  Iteration: {w['iteration']}/{w['max_iterations']}")
        print(f"  Best Score: {w['best_score']:.1f} (iter {w['best_iteration']})")
        if w['pending_decision']:
            print(f"  Pending: {w['pending_decision']}")

    # Resumable sessions
    resumable = orchestrator.get_resumable_sessions()
    if resumable:
        print(f"\nResumable Sessions ({len(resumable)}):")
        for s in resumable[:3]:
            print(f"  {s['session_id']}: score {s['best_score']:.1f}")


def cmd_trust(args, orchestrator):
    """Set trust score."""
    orchestrator.set_trust_score(args.score)
    print(f"Trust score set to {args.score:.2f}")
    print(f"Autonomy level: {orchestrator.autonomy.get_level().value}")


def cmd_autonomy(args, orchestrator):
    """Set autonomy level override."""
    level = args.level if args.level != "auto" else None
    orchestrator.set_autonomy_override(level)
    print(f"Autonomy level: {orchestrator.autonomy.get_level().value}")
    if level:
        print(f"(Override active: {level})")
    else:
        print("(Using trust score)")


async def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Import here to avoid import errors if dependencies missing
    try:
        from .orchestrator import BlenderOrchestratorAgent
    except ImportError:
        # Running as script
        sys.path.insert(0, str(Path(__file__).parent))
        from orchestrator import BlenderOrchestratorAgent

    # Create orchestrator
    orchestrator = BlenderOrchestratorAgent(project_root=PROJECT_ROOT)

    # Handle commands that don't need agent SDK
    if args.command == "list":
        cmd_list(args, orchestrator)
        return

    if args.command == "status":
        cmd_status(orchestrator)
        return

    if args.command == "trust":
        cmd_trust(args, orchestrator)
        return

    if args.command == "autonomy":
        cmd_autonomy(args, orchestrator)
        return

    # Commands that need agent SDK
    try:
        await orchestrator.start()

        if args.command == "create":
            await cmd_create(args, orchestrator)

        elif args.command == "resume":
            await cmd_resume(args, orchestrator)

        else:
            # Default to interactive
            await cmd_interactive(orchestrator)

    finally:
        await orchestrator.stop()


def create_mcp_server():
    """Create MCP server exposing orchestrator tools."""
    if not MCP_AVAILABLE:
        raise RuntimeError("claude_agent_sdk not available - cannot create MCP server")

    # Import tools
    from tools import create_asset, get_status, list_sessions, resume_session

    # Create MCP server with all tools
    server = create_sdk_mcp_server(
        name="blender-orchestrator",
        version="0.2.0",
        tools=[
            create_asset,
            get_status,
            list_sessions,
            resume_session,
        ],
    )

    return server


async def run_mcp_server():
    """Run as MCP server (for Claude Code integration)."""
    server = create_mcp_server()
    logger.info("Starting Blender VFX Orchestrator MCP server...")

    # Run the MCP server
    await server.run()


if __name__ == "__main__":
    # Check for MCP mode flag
    if "--mcp" in sys.argv:
        if not MCP_AVAILABLE:
            print("Error: claude_agent_sdk not available for MCP mode")
            sys.exit(1)
        try:
            asyncio.run(run_mcp_server())
        except KeyboardInterrupt:
            print("\nMCP server shutdown.")
            sys.exit(0)
    else:
        # CLI mode
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\nShutdown requested. Goodbye!")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            sys.exit(1)
