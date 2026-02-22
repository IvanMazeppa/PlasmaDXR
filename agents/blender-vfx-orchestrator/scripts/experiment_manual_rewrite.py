#!/usr/bin/env python3
"""
Experiment: Rewrite Blender manual pages for LLM consumption.

Takes raw HTML manual pages and produces dense, LLM-optimized text that
maximizes information per token. Strips UI navigation, formatting artifacts,
and verbose human-oriented explanations while preserving:
- Parameter names, types, and valid ranges
- Behavioral relationships between settings
- Technique comparisons and workflow knowledge
- Conceptual understanding of how features work

This is an EXPERIMENT — processes a small batch of pages to evaluate quality
before committing to a full manual rewrite.

Usage:
    # Process 5 test pages, print results
    python scripts/experiment_manual_rewrite.py --preview

    # Process test pages, save to output dir
    python scripts/experiment_manual_rewrite.py --output ./rewritten_manual_test

    # Process with a specific model
    python scripts/experiment_manual_rewrite.py --model gpt-5-mini --output ./rewritten_manual_test

    # Process ALL physics pages (still a subset of the full manual)
    python scripts/experiment_manual_rewrite.py --all-physics --output ./rewritten_manual_test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("Error: beautifulsoup4 required. Run: pip install beautifulsoup4 lxml", file=sys.stderr)
    sys.exit(1)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ============================================================
# CONFIGURATION
# ============================================================

MANUAL_ROOT = Path(__file__).parent.parent / "assets" / "blender_manual_html" / "blender_manual_v500_en.html"

# Test pages: high-value physics/fluid pages most relevant to VFX orchestrator
TEST_PAGES = [
    "physics/fluid/type/domain/settings.html",
    "physics/fluid/type/flow.html",
    "physics/fluid/type/domain/gas/index.html",
    "physics/fluid/type/domain/gas/noise.html",
    "physics/fluid/type/domain/cache.html",
]

# Extended set: all physics pages
PHYSICS_GLOBS = [
    "physics/**/*.html",
    "render/shader_nodes/shader/principled_volume.html",
    "render/shader_nodes/shader/volume_absorption.html",
    "render/shader_nodes/shader/volume_scatter.html",
    "render/shader_nodes/shader/principled_bsdf.html",
    "render/lights/**/*.html",
    "render/cameras/**/*.html",
    "modeling/modifiers/physics/*.html",
]

# The system prompt for rewriting
REWRITE_SYSTEM_PROMPT = """You are a technical documentation rewriter. Your job is to transform
Blender manual content from human-oriented documentation into dense, LLM-optimized reference text.

GOALS:
- Maximize information density per token
- Preserve ALL technical facts: parameter names, value ranges, defaults, units
- Preserve behavioral relationships: "increasing X causes Y to Z"
- Preserve technique knowledge: when to use what, tradeoffs, alternatives
- Preserve workflow sequences: "do A before B because C"

REMOVE:
- UI navigation instructions ("click X in Properties > Y > Z")
- References to visual UI elements, screenshots, icons, menus
- Verbose introductions and transitions ("In this section we will...")
- Redundant explanations of the same concept
- Generic Blender basics the LLM already knows

FORMAT:
- Use flat markdown: ## for sections, bullet lists for parameters
- Parameter format: `parameter_name` (type, range [min, max], default: X) — description
- Use "→" for cause/effect: "Higher resolution → more detail but slower bake"
- Use "NOTE:" prefix for gotchas and warnings
- Use "TECHNIQUE:" prefix for workflow/approach knowledge
- Keep total output under 60% of input token count

CRITICAL: Do NOT invent information. Only include facts present in the source text.
If the source is vague about a value range, say "range not specified" rather than guessing."""

REWRITE_USER_TEMPLATE = """Rewrite this Blender manual page for LLM consumption.

Source page: {doc_path}

---RAW CONTENT---
{content}
---END RAW CONTENT---

Produce a dense, structured rewrite following the system instructions.
Start with a one-line summary of what this page covers."""


# ============================================================
# HTML EXTRACTION
# ============================================================

@dataclass
class ExtractedPage:
    """A manual page with extracted text content."""
    doc_path: str           # Relative path like "physics/fluid/type/domain/settings.html"
    title: str
    raw_text: str           # Cleaned text from HTML
    word_count: int
    sections: list          # List of (heading_level, heading_text, content)


def extract_page_content(html_path: Path, manual_root: Path) -> Optional[ExtractedPage]:
    """Extract meaningful content from a Blender manual HTML page."""
    try:
        html = html_path.read_text(errors="replace")
    except Exception as e:
        print(f"  Error reading {html_path}: {e}", file=sys.stderr)
        return None

    soup = BeautifulSoup(html, "html.parser")

    # Extract page title
    title_el = soup.find("h1")
    title = title_el.get_text(strip=True).replace("¶", "").strip() if title_el else html_path.stem

    # Find content sections — try multiple strategies
    content_el = None
    for selector in ["section", "div.body-content", "div.document", "article", "main"]:
        content_el = soup.select_one(selector)
        if content_el:
            text = content_el.get_text(separator="\n", strip=True)
            if len(text) > 100:
                break
            content_el = None

    if not content_el:
        return None

    # Extract sections with headings
    sections = []
    current_heading = ("h1", title)
    current_content = []

    for el in content_el.find_all(["h1", "h2", "h3", "h4", "p", "dl", "ul", "ol", "div", "table", "blockquote"]):
        tag = el.name
        if tag in ("h1", "h2", "h3", "h4"):
            # Save previous section
            if current_content:
                content_text = "\n".join(current_content)
                sections.append((*current_heading, content_text))
            current_heading = (tag, el.get_text(strip=True).replace("¶", "").strip())
            current_content = []
        else:
            text = el.get_text(separator=" ", strip=True)
            # Skip very short or navigation-only text
            if len(text) > 10:
                # Clean artifacts
                text = text.replace("¶", "").strip()
                # Skip "Reference Panel:" lines
                if text.startswith("Reference") and "Panel" in text[:30]:
                    continue
                current_content.append(text)

    # Save final section
    if current_content:
        content_text = "\n".join(current_content)
        sections.append((*current_heading, content_text))

    # Build full text
    raw_text = "\n\n".join(
        f"{'#' * (1 if level == 'h1' else 2 if level == 'h2' else 3)} {heading}\n{content}"
        for level, heading, content in sections
    )

    # Calculate relative path
    try:
        rel_path = html_path.relative_to(manual_root)
    except ValueError:
        rel_path = html_path.name

    word_count = len(raw_text.split())

    if word_count < 30:
        return None  # Skip near-empty pages (index pages, etc.)

    return ExtractedPage(
        doc_path=str(rel_path),
        title=title,
        raw_text=raw_text,
        word_count=word_count,
        sections=sections,
    )


# ============================================================
# LLM REWRITE
# ============================================================

def rewrite_page(page: ExtractedPage, client: OpenAI, model: str = "gpt-5-mini") -> dict:
    """Rewrite a manual page using an LLM for information density."""
    user_prompt = REWRITE_USER_TEMPLATE.format(
        doc_path=page.doc_path,
        content=page.raw_text[:12000],  # Cap input to avoid huge prompts
    )

    start = time.time()
    try:
        # Build API params — some models don't support temperature
        api_params = {
            "model": model,
            "messages": [
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_completion_tokens": 4000,
        }
        # Only set temperature for models that support it
        if model not in ("gpt-5-mini", "o3", "o4-mini"):
            api_params["temperature"] = 0.2
        response = client.chat.completions.create(**api_params)
        elapsed = time.time() - start
        rewritten = response.choices[0].message.content
        usage = response.usage

        return {
            "success": True,
            "doc_path": page.doc_path,
            "title": page.title,
            "original_words": page.word_count,
            "rewritten_words": len(rewritten.split()),
            "compression_ratio": len(rewritten.split()) / max(page.word_count, 1),
            "rewritten_text": rewritten,
            "model": model,
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
            "cost_estimate": _estimate_cost(model, usage),
            "elapsed_seconds": round(elapsed, 1),
        }
    except Exception as e:
        return {
            "success": False,
            "doc_path": page.doc_path,
            "title": page.title,
            "error": str(e),
        }


def _estimate_cost(model: str, usage) -> float:
    """Rough cost estimate based on model pricing."""
    if not usage:
        return 0.0
    # Approximate pricing (per 1M tokens) — update as needed
    pricing = {
        "gpt-5-mini": {"input": 0.15, "output": 0.60},
        "gpt-5.2": {"input": 2.50, "output": 10.00},
        "o4-mini": {"input": 1.10, "output": 4.40},
    }
    rates = pricing.get(model, pricing["gpt-5-mini"])
    input_cost = (usage.prompt_tokens / 1_000_000) * rates["input"]
    output_cost = (usage.completion_tokens / 1_000_000) * rates["output"]
    return round(input_cost + output_cost, 6)


# ============================================================
# LOCAL-ONLY MODE (no API calls)
# ============================================================

def extract_only(page: ExtractedPage) -> dict:
    """Extract and clean content without LLM rewrite — for preview/debugging."""
    return {
        "doc_path": page.doc_path,
        "title": page.title,
        "word_count": page.word_count,
        "sections": len(page.sections),
        "section_headings": [f"{level}: {heading}" for level, heading, _ in page.sections],
        "preview": page.raw_text[:2000],
    }


# ============================================================
# MAIN
# ============================================================

def find_pages(manual_root: Path, page_paths: list[str]) -> list[Path]:
    """Resolve page paths relative to manual root."""
    found = []
    for rel in page_paths:
        full = manual_root / rel
        if full.exists():
            found.append(full)
        else:
            print(f"  Warning: page not found: {rel}", file=sys.stderr)
    return found


def find_physics_pages(manual_root: Path) -> list[Path]:
    """Find all physics-relevant pages."""
    pages = set()
    for pattern in PHYSICS_GLOBS:
        for p in manual_root.glob(pattern):
            if p.is_file() and p.suffix == ".html":
                pages.add(p)
    return sorted(pages)


def main():
    parser = argparse.ArgumentParser(description="Experiment: rewrite Blender manual for LLM consumption")
    parser.add_argument("--preview", action="store_true", help="Extract only (no LLM calls) — show what would be processed")
    parser.add_argument("--output", type=str, help="Output directory for rewritten pages")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="Model for rewriting (default: gpt-5-mini)")
    parser.add_argument("--all-physics", action="store_true", help="Process all physics pages (not just test set)")
    parser.add_argument("--dry-run", action="store_true", help="Show pages that would be processed without doing anything")
    args = parser.parse_args()

    manual_root = MANUAL_ROOT
    if not manual_root.exists():
        print(f"Error: Manual root not found: {manual_root}", file=sys.stderr)
        print("Expected HTML manual at: assets/blender_manual_html/blender_manual_v500_en.html/", file=sys.stderr)
        sys.exit(1)

    # Select pages
    if args.all_physics:
        html_paths = find_physics_pages(manual_root)
        print(f"Found {len(html_paths)} physics-relevant pages")
    else:
        html_paths = find_pages(manual_root, TEST_PAGES)
        print(f"Using {len(html_paths)} test pages")

    if args.dry_run:
        for p in html_paths:
            try:
                rel = p.relative_to(manual_root)
            except ValueError:
                rel = p.name
            print(f"  {rel}")
        return

    # Extract content from all pages
    pages = []
    for hp in html_paths:
        page = extract_page_content(hp, manual_root)
        if page:
            pages.append(page)
            print(f"  Extracted: {page.doc_path} ({page.word_count} words, {len(page.sections)} sections)")
        else:
            print(f"  Skipped (empty/unparseable): {hp.name}")

    if not pages:
        print("No pages extracted. Check manual path.", file=sys.stderr)
        sys.exit(1)

    print(f"\nTotal: {len(pages)} pages, {sum(p.word_count for p in pages)} words")

    # Preview mode — extract only
    if args.preview:
        print("\n" + "=" * 60)
        print("PREVIEW MODE — no LLM calls, showing extracted content")
        print("=" * 60)
        for page in pages:
            info = extract_only(page)
            print(f"\n--- {info['doc_path']} ({info['word_count']} words, {info['sections']} sections) ---")
            for h in info["section_headings"]:
                print(f"  {h}")
            print(f"\nPreview:\n{info['preview'][:1000]}...")
        return

    # Rewrite mode — requires OpenAI API
    if not OPENAI_AVAILABLE:
        print("Error: openai package required for rewrite mode. Run: pip install openai", file=sys.stderr)
        sys.exit(1)

    client = OpenAI()
    results = []
    total_cost = 0.0

    print(f"\nRewriting {len(pages)} pages with {args.model}...")
    print("-" * 60)

    for i, page in enumerate(pages):
        print(f"[{i+1}/{len(pages)}] {page.doc_path} ({page.word_count} words)...", end=" ", flush=True)
        result = rewrite_page(page, client, model=args.model)
        results.append(result)

        if result["success"]:
            cost = result["cost_estimate"]
            total_cost += cost
            ratio = result["compression_ratio"]
            print(f"OK ({result['rewritten_words']} words, {ratio:.0%} of original, ${cost:.4f}, {result['elapsed_seconds']}s)")
        else:
            print(f"FAILED: {result.get('error', 'unknown')}")

    # Summary
    successful = [r for r in results if r.get("success")]
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {len(successful)}/{len(results)} pages rewritten")
    print(f"Total cost: ${total_cost:.4f}")
    if successful:
        avg_compression = sum(r["compression_ratio"] for r in successful) / len(successful)
        avg_input = sum(r["original_words"] for r in successful) / len(successful)
        avg_output = sum(r["rewritten_words"] for r in successful) / len(successful)
        print(f"Avg compression: {avg_compression:.0%} (input: {avg_input:.0f} words → output: {avg_output:.0f} words)")
        print(f"Avg elapsed: {sum(r['elapsed_seconds'] for r in successful) / len(successful):.1f}s per page")

        # Estimate full manual cost
        total_manual_pages = 2197  # from earlier count
        physics_pages = len(find_physics_pages(manual_root)) if manual_root.exists() else 200
        cost_per_page = total_cost / len(successful)
        print(f"\nCost projections:")
        print(f"  Physics pages only (~{physics_pages} pages): ${cost_per_page * physics_pages:.2f}")
        print(f"  Full manual ({total_manual_pages} pages): ${cost_per_page * total_manual_pages:.2f}")

    # Save output
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        for result in successful:
            # Save rewritten text
            doc_path = result["doc_path"]
            out_path = output_dir / doc_path.replace(".html", ".md")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Add metadata header
            header = f"""# {result['title']}
DocType: manual-rewritten
DocPath: {doc_path}
DocVersion: 5.0.1
Model: {result['model']}
OriginalWords: {result['original_words']}
RewrittenWords: {result['rewritten_words']}
CompressionRatio: {result['compression_ratio']:.2f}
---

"""
            out_path.write_text(header + result["rewritten_text"])
            print(f"  Saved: {out_path}")

        # Save summary JSON
        summary_path = output_dir / "_rewrite_summary.json"
        summary_path.write_text(json.dumps({
            "model": args.model,
            "pages_processed": len(results),
            "pages_successful": len(successful),
            "total_cost": total_cost,
            "results": [
                {k: v for k, v in r.items() if k != "rewritten_text"}
                for r in results
            ],
        }, indent=2))
        print(f"  Summary: {summary_path}")

    # Print sample output for quality review
    if successful:
        print(f"\n{'=' * 60}")
        print("SAMPLE OUTPUT — first rewritten page for quality review:")
        print(f"{'=' * 60}")
        sample = successful[0]
        print(f"Page: {sample['doc_path']}")
        print(f"Original: {sample['original_words']} words → Rewritten: {sample['rewritten_words']} words")
        print(f"---")
        print(sample["rewritten_text"][:3000])
        if len(sample["rewritten_text"]) > 3000:
            print(f"... ({len(sample['rewritten_text'])} total chars)")


if __name__ == "__main__":
    main()
