#!/usr/bin/env python3
"""
Quick test of ML visual comparison tool

Tests LPIPS comparison on two recent screenshots
"""

import asyncio
import sys
import os
from pathlib import Path

# Change to project directory for proper module imports
os.chdir(Path(__file__).parent)

# Import as module
from src.tools.ml_visual_comparison import compare_screenshots_ml, format_comparison_report


async def main():
    print("=" * 80)
    print("ML VISUAL COMPARISON - TEST")
    print("=" * 80)
    print()

    # Use two recent screenshots from Pictures
    before_path = "/mnt/c/Users/dilli/Pictures/Screenshots/Screenshot 2025-10-20 194805.png"
    after_path = "/mnt/c/Users/dilli/Pictures/Screenshots/Screenshot 2025-10-21 185625.png"

    print(f"Before: {before_path}")
    print(f"After:  {after_path}")
    print()
    print("Running ML comparison (LPIPS + traditional metrics)...")
    print("Note: First run downloads LPIPS weights (~50MB), may take 30 seconds")
    print()

    # Run comparison
    results = await compare_screenshots_ml(
        before_path=before_path,
        after_path=after_path,
        save_heatmap=True,
        project_root="/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean"
    )

    # Check for errors
    if "error" in results:
        print("ERROR:", results["error"])
        if "suggestion" in results:
            print("Suggestion:", results["suggestion"])
        return

    # Format and print report
    report = format_comparison_report(results)
    print(report)

    print()
    print("=" * 80)
    print("TEST COMPLETE ✅")
    print("=" * 80)

    if "heatmap_path" in results:
        print()
        print("Next steps:")
        print("1. View difference heatmap:", results["heatmap_path"])
        print("2. Reconnect MCP server: claude mcp server reconnect rtxdi-quality-analyzer")
        print("3. Use compare_screenshots_ml tool in Claude Code session")


if __name__ == "__main__":
    asyncio.run(main())
