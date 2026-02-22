#!/usr/bin/env python3
"""
Seed the knowledge base with technique entries from Blender documentation.

Queries the semantic doc vector store for each effect type, extracts
technique names and descriptions, then seeds them into the KB via
seed_technique_entry. Entries start with trust_level="emerging" and
must earn trust through successful production use.

Usage:
    # Dry run — show what would be seeded
    python scripts/seed_kb.py --dry-run

    # Seed all 6 effect types
    python scripts/seed_kb.py

    # Seed a single effect type
    python scripts/seed_kb.py --effect-type fire

    # Verbose output
    python scripts/seed_kb.py --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Path setup — match the pattern used by other orchestrator scripts
SCRIPT_DIR = Path(__file__).parent
ORCHESTRATOR_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ORCHESTRATOR_ROOT))

from tools.semantic_docs_tools import _search_vector_store
from tools.experiment_tracker_tools import _seed_technique_entry_impl

# ============================================================
# EFFECT TYPES AND QUERIES
# ============================================================

EFFECT_TYPES = {
    "fire": {
        "query": "techniques for creating fire effects in Blender mantaflow gas burning flames",
        "domain": "mantaflow gas — burning",
    },
    "smoke": {
        "query": "techniques for creating smoke effects in Blender mantaflow gas non-burning dissipation",
        "domain": "mantaflow gas — non-burning",
    },
    "liquid": {
        "query": "techniques for creating liquid water pour splash effects in Blender mantaflow",
        "domain": "mantaflow liquid — pour/splash",
    },
    "rigid_body": {
        "query": "techniques for creating rigid body destruction shattering effects in Blender",
        "domain": "rigid body — destruction/shattering",
    },
    "particles": {
        "query": "techniques for creating particle effects debris sparks rain in Blender",
        "domain": "particle system — debris/sparks/rain",
    },
    "cloth": {
        "query": "techniques for creating cloth fabric flag curtain simulation in Blender",
        "domain": "cloth simulation — fabric/flag/curtain",
    },
}


# ============================================================
# TECHNIQUE EXTRACTION
# ============================================================

def extract_techniques(results: list[dict], effect_type: str) -> list[dict]:
    """
    Extract technique names and descriptions from vector store results.

    Parses doc content to find distinct techniques. Each result chunk
    is treated as a potential technique source.
    """
    techniques = []
    seen_names = set()

    for result in results:
        content = result.get("content", "")
        if not content or len(content) < 50:
            continue

        # Extract a technique name from headings or filename
        name = _extract_technique_name(content, result, effect_type)
        if not name or name.lower() in seen_names:
            continue
        seen_names.add(name.lower())

        # Build description from first meaningful paragraph
        description = _extract_description(content, effect_type)
        if not description:
            continue

        techniques.append({
            "technique_name": name,
            "description": description,
            "source_file": result.get("filename", "unknown"),
            "score": result.get("score", 0),
        })

    return techniques


def _extract_technique_name(content: str, result: dict, effect_type: str) -> str:
    """Extract a short technique name from doc content."""
    # Try heading extraction first
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            # Skip generic headings
            if heading and len(heading) > 3 and heading.lower() not in (
                "settings", "properties", "options", "description", "usage",
                "parameters", "example", "examples", "notes", "reference",
            ):
                # Normalize to snake_case identifier
                return _to_snake_case(heading)[:60]

    # Fallback: derive from filename
    filename = result.get("filename", "")
    if filename and filename != "unknown":
        stem = Path(filename).stem
        if stem and len(stem) > 3:
            return f"{effect_type}_{_to_snake_case(stem)}"[:60]

    return ""


def _extract_description(content: str, effect_type: str) -> str:
    """Extract a concise description from content."""
    lines = []
    for line in content.split("\n"):
        stripped = line.strip()
        # Skip headers, metadata, and short lines
        if stripped.startswith("#") or stripped.startswith("DocType:"):
            continue
        if stripped.startswith("DocPath:") or stripped.startswith("DocVersion:"):
            continue
        if stripped.startswith("ChunkId:"):
            continue
        if len(stripped) > 20:
            lines.append(stripped)
        if len(lines) >= 3:
            break

    if not lines:
        return ""

    desc = " ".join(lines)
    # Truncate to reasonable length
    if len(desc) > 300:
        desc = desc[:297] + "..."
    return desc


def _to_snake_case(text: str) -> str:
    """Convert a heading to a snake_case identifier."""
    # Replace non-alphanumeric with underscore
    result = []
    for ch in text.lower():
        if ch.isalnum():
            result.append(ch)
        elif result and result[-1] != "_":
            result.append("_")
    return "".join(result).strip("_")


# ============================================================
# MAIN
# ============================================================

def seed_effect_type(
    effect_type: str,
    config: dict,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict:
    """Seed techniques for a single effect type. Returns summary stats."""
    query = config["query"]

    if verbose:
        print(f"\n  Querying: {query}")

    results = _search_vector_store(query, max_results=8, intent="manual")
    techniques = extract_techniques(results, effect_type)

    if verbose:
        print(f"  Results: {len(results)} docs -> {len(techniques)} techniques")

    seeded = 0
    for tech in techniques:
        if verbose:
            print(f"    {tech['technique_name']}: {tech['description'][:80]}...")

        if dry_run:
            seeded += 1
            continue

        result_json = _seed_technique_entry_impl(
            effect_type=effect_type,
            technique_name=tech["technique_name"],
            description=tech["description"],
            source="manual_seed",
        )
        result = json.loads(result_json)
        if result.get("success"):
            seeded += 1
        elif verbose:
            print(f"    FAILED: {result.get('error', 'unknown')}")

    return {
        "effect_type": effect_type,
        "domain": config["domain"],
        "docs_found": len(results),
        "techniques_extracted": len(techniques),
        "entries_seeded": seeded,
        "techniques": [t["technique_name"] for t in techniques],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Seed KB with technique entries from Blender docs"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be seeded without writing to KB"
    )
    parser.add_argument(
        "--effect-type", type=str, choices=list(EFFECT_TYPES.keys()),
        help="Seed only this effect type"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed output for each technique"
    )
    args = parser.parse_args()

    if args.dry_run:
        print("=== DRY RUN — no entries will be written ===\n")

    # Select effect types
    if args.effect_type:
        types_to_seed = {args.effect_type: EFFECT_TYPES[args.effect_type]}
    else:
        types_to_seed = EFFECT_TYPES

    # Seed each effect type
    summaries = []
    total_techniques = 0
    total_seeded = 0

    for effect_type, config in types_to_seed.items():
        print(f"[{effect_type}] {config['domain']}")
        summary = seed_effect_type(
            effect_type, config,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        summaries.append(summary)
        total_techniques += summary["techniques_extracted"]
        total_seeded += summary["entries_seeded"]
        print(f"  -> {summary['techniques_extracted']} techniques, {summary['entries_seeded']} seeded")

    # Summary
    print(f"\n{'=' * 50}")
    action = "would seed" if args.dry_run else "seeded"
    print(f"SUMMARY: {len(summaries)} effect types processed")
    print(f"  Techniques found: {total_techniques}")
    print(f"  Entries {action}: {total_seeded}")
    for s in summaries:
        if s["techniques"]:
            print(f"  [{s['effect_type']}]: {', '.join(s['techniques'])}")


if __name__ == "__main__":
    main()
