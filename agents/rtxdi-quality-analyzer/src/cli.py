"""
RTXDI Quality Analyzer - Standalone CLI

Command-line interface for testing diagnostic tools independently.

Usage:
    # Performance comparison
    python -m src.cli performance --legacy logs/legacy.log --rtxdi-m4 logs/m4.log

    # PIX analysis
    python -m src.cli pix --capture PIX/Captures/latest.wpix

    # Interactive mode
    python -m src.cli interactive
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from .agent import RTXDIAnalyzer


# Load environment variables
load_dotenv()

# Rich console for beautiful output
console = Console()


async def cmd_performance(args):
    """Run performance comparison"""
    console.print("\n[bold cyan]Running Performance Comparison...[/bold cyan]\n")

    analyzer = RTXDIAnalyzer(project_root=args.project_root)

    try:
        report = await analyzer.run_performance_comparison(
            legacy_log=args.legacy,
            rtxdi_m4_log=args.rtxdi_m4,
            rtxdi_m5_log=args.rtxdi_m5
        )

        console.print(Panel(report, title="Performance Comparison Report", border_style="cyan"))

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


async def cmd_pix(args):
    """Run PIX capture analysis"""
    console.print("\n[bold cyan]Running PIX Capture Analysis...[/bold cyan]\n")

    analyzer = RTXDIAnalyzer(project_root=args.project_root)

    try:
        report = await analyzer.run_pix_analysis(
            capture_path=args.capture,
            analyze_buffers=args.buffers
        )

        console.print(Panel(report, title="PIX Analysis Report", border_style="cyan"))

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


async def cmd_interactive(args):
    """Interactive diagnostic mode"""
    console.print(Panel.fit(
        "[bold cyan]RTXDI Quality Analyzer[/bold cyan]\n"
        "Interactive Diagnostic Mode",
        border_style="cyan"
    ))

    analyzer = RTXDIAnalyzer(project_root=args.project_root)

    console.print("\n[dim]Enter your diagnostic query (or 'quit' to exit)[/dim]\n")

    while True:
        try:
            query = console.input("[bold green]Query:[/bold green] ")

            if query.lower() in ["quit", "exit", "q"]:
                console.print("\n[cyan]Goodbye![/cyan]")
                break

            if not query.strip():
                continue

            console.print()
            response = await analyzer.diagnose(query)
            console.print(Panel(response, border_style="cyan"))
            console.print()

        except KeyboardInterrupt:
            console.print("\n\n[cyan]Goodbye![/cyan]")
            break
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}\n")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="RTXDI Quality Analyzer - Diagnostic CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--project-root",
        type=str,
        help="Path to PlasmaDX-Clean project root (defaults to env var)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Performance comparison command
    perf_parser = subparsers.add_parser(
        "performance",
        help="Compare performance between renderer modes"
    )
    perf_parser.add_argument("--legacy", type=str, help="Path to legacy renderer log")
    perf_parser.add_argument("--rtxdi-m4", type=str, help="Path to RTXDI M4 log")
    perf_parser.add_argument("--rtxdi-m5", type=str, help="Path to RTXDI M5 log")
    perf_parser.set_defaults(func=cmd_performance)

    # PIX analysis command
    pix_parser = subparsers.add_parser(
        "pix",
        help="Analyze PIX capture for bottlenecks"
    )
    pix_parser.add_argument("--capture", type=str, help="Path to .wpix capture file")
    pix_parser.add_argument(
        "--no-buffers",
        dest="buffers",
        action="store_false",
        help="Skip buffer dump analysis"
    )
    pix_parser.set_defaults(func=cmd_pix, buffers=True)

    # Interactive mode
    interactive_parser = subparsers.add_parser(
        "interactive",
        help="Interactive diagnostic mode"
    )
    interactive_parser.set_defaults(func=cmd_interactive)

    # Parse arguments
    args = parser.parse_args()

    # Show help if no command specified
    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Run the selected command
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()
