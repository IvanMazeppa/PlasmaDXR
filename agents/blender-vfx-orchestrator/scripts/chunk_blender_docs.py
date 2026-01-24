#!/usr/bin/env python3
"""
Blender Documentation Chunker for Vector Store Upload.

Creates header-injected chunks with stable DocPath references for reliable
doc-grounding in the Blender VFX multi-agent system.

Chunking Strategy:
- Manual docs: Chunk by h2/h3 sections, target 500-1200 words
- API docs: Chunk by class/function definitions, target 300-800 words

Each chunk includes a standardized header:
    DocType: manual|api
    DocPath: relative/path.html#section
    DocVersion: 5.0.1
    ChunkId: unique-identifier

Usage:
    # Preview chunks without uploading
    python scripts/chunk_blender_docs.py --preview

    # Generate chunks to output directory
    python scripts/chunk_blender_docs.py --output ./chunked_docs

    # Generate stats only
    python scripts/chunk_blender_docs.py --stats-only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any
from collections import defaultdict

try:
    from bs4 import BeautifulSoup, Tag
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("Warning: beautifulsoup4 not installed. Run: pip install beautifulsoup4 lxml")

# Configuration
DOC_VERSION = "5.0.1"
SOURCE_REPO = "blender-documentation"

# Default paths (can be overridden via args)
DEFAULT_DOCS_BASE = Path("/home/maz3ppa/blender_documentation")
DEFAULT_MANUAL_PATH = DEFAULT_DOCS_BASE / "blender_manual_html"
DEFAULT_API_PATH = DEFAULT_DOCS_BASE / "blender_python_reference_5_0"

# Chunking parameters
MANUAL_MIN_WORDS = 50
MANUAL_TARGET_MIN = 500
MANUAL_TARGET_MAX = 1200
MANUAL_SPLIT_SIZE = 800

API_MIN_WORDS = 20
API_TARGET_MIN = 300
API_TARGET_MAX = 800

# Skip patterns
SKIP_PATTERNS = [
    "Zone.Identifier",
    "_static",
    "_sources",
    "genindex",
    "py-modindex",
    "search.html",
    "404.html",
    "_modules",
]


@dataclass
class DocChunk:
    """A single documentation chunk with metadata."""
    title: str
    doc_type: str  # "manual" or "api"
    doc_path: str  # Relative path like "physics/fluid/domain.html#settings"
    doc_version: str
    chunk_id: str  # Unique ID like "physics/fluid/domain.html#settings#0"
    content: str
    word_count: int
    source_file: str = ""
    section_type: str = ""  # "h2", "h3", "class", "function", etc.

    def to_upload_text(self) -> str:
        """Format chunk for vector store upload with standardized header."""
        return f"""# {self.title}
DocType: {self.doc_type}
DocPath: {self.doc_path}
DocVersion: {self.doc_version}
SourceRepo: {SOURCE_REPO}
ChunkId: {self.chunk_id}
---

{self.content}
"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return asdict(self)


@dataclass
class ChunkingStats:
    """Statistics from a chunking run."""
    total_files: int = 0
    files_processed: int = 0
    files_skipped: int = 0
    files_errored: int = 0
    total_chunks: int = 0
    chunks_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    word_count_distribution: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors: List[str] = field(default_factory=list)

    def add_chunk(self, chunk: DocChunk):
        self.total_chunks += 1
        self.chunks_by_type[chunk.doc_type] += 1

        # Bucket word counts
        wc = chunk.word_count
        if wc < 100:
            self.word_count_distribution["<100"] += 1
        elif wc < 300:
            self.word_count_distribution["100-299"] += 1
        elif wc < 500:
            self.word_count_distribution["300-499"] += 1
        elif wc < 800:
            self.word_count_distribution["500-799"] += 1
        elif wc < 1200:
            self.word_count_distribution["800-1199"] += 1
        else:
            self.word_count_distribution["1200+"] += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_files": self.total_files,
            "files_processed": self.files_processed,
            "files_skipped": self.files_skipped,
            "files_errored": self.files_errored,
            "total_chunks": self.total_chunks,
            "chunks_by_type": dict(self.chunks_by_type),
            "word_count_distribution": dict(self.word_count_distribution),
            "errors": self.errors[:20],  # Limit errors in output
        }


def should_skip_file(file_path: Path) -> bool:
    """Check if file should be skipped based on patterns."""
    path_str = str(file_path)
    for pattern in SKIP_PATTERNS:
        if pattern in path_str:
            return True
    return False


def clean_text(text: str) -> str:
    """Clean extracted text: normalize whitespace, remove artifacts."""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove common artifacts
    text = re.sub(r'\[source\]', '', text)
    text = re.sub(r'¶', '', text)
    # Clean up multiple spaces
    text = re.sub(r' +', ' ', text)
    return text.strip()


def extract_section_content(element: Tag, stop_tags: List[str]) -> str:
    """
    Extract text content from an element until hitting a stop tag.

    Args:
        element: Starting element (usually a header)
        stop_tags: Tag names that indicate section boundary

    Returns:
        Extracted and cleaned text
    """
    content_parts = []

    for sibling in element.find_next_siblings():
        if sibling.name in stop_tags:
            break

        # Skip navigation and sidebars
        if sibling.get('class') and any(
            cls in str(sibling.get('class', []))
            for cls in ['toctree', 'sidebar', 'nav', 'footer']
        ):
            continue

        text = sibling.get_text(separator=' ', strip=True)
        if text:
            content_parts.append(text)

    return clean_text(' '.join(content_parts))


def split_large_content(content: str, title: str, max_words: int = MANUAL_SPLIT_SIZE) -> List[tuple]:
    """
    Split large content into smaller chunks.

    Returns list of (suffix, content) tuples.
    """
    words = content.split()
    if len(words) <= max_words:
        return [("", content)]

    chunks = []
    for i, start in enumerate(range(0, len(words), max_words)):
        chunk_words = words[start:start + max_words]
        chunk_content = ' '.join(chunk_words)
        suffix = f" (Part {i + 1})" if i > 0 or start + max_words < len(words) else ""
        chunks.append((suffix, chunk_content))

    return chunks


def chunk_manual_page(html_path: Path, base_path: Path) -> List[DocChunk]:
    """
    Chunk a manual documentation page by h2/h3 sections.

    Target: 500-1200 words per chunk.
    Strategy: Split on h2/h3 headers, combine small sections, split large ones.
    """
    if not BS4_AVAILABLE:
        return []

    try:
        with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f.read(), 'lxml')
    except Exception as e:
        print(f"  Error parsing {html_path.name}: {e}", file=sys.stderr)
        return []

    relative_path = str(html_path.relative_to(base_path))
    chunks = []

    # Remove script, style, nav elements
    for tag in soup.find_all(['script', 'style', 'nav', 'footer']):
        tag.decompose()

    # Get page title
    page_title = ""
    if soup.title:
        page_title = soup.title.string or ""
    elif soup.h1:
        page_title = soup.h1.get_text(strip=True)
    else:
        page_title = html_path.stem.replace('_', ' ').replace('-', ' ').title()

    # Find section headers (h2 and h3)
    headers = soup.find_all(['h2', 'h3'])

    if not headers:
        # No section headers - treat entire page as one chunk
        main_content = soup.find(['main', 'article', 'div'], class_=re.compile(r'body|content|document'))
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
        else:
            text = soup.get_text(separator='\n', strip=True)

        text = clean_text(text)
        word_count = len(text.split())

        if word_count >= MANUAL_MIN_WORDS:
            # Split if too large
            for suffix, chunk_content in split_large_content(text, page_title, MANUAL_SPLIT_SIZE):
                chunk_word_count = len(chunk_content.split())
                chunk_num = len(chunks)
                chunks.append(DocChunk(
                    title=f"{page_title}{suffix}",
                    doc_type="manual",
                    doc_path=relative_path,
                    doc_version=DOC_VERSION,
                    chunk_id=f"{relative_path}#full#{chunk_num}",
                    content=chunk_content,
                    word_count=chunk_word_count,
                    source_file=str(html_path),
                    section_type="full_page"
                ))
        return chunks

    # Process each section
    for i, header in enumerate(headers):
        section_id = header.get('id', '')
        if not section_id:
            # Generate ID from text
            section_id = re.sub(r'[^a-z0-9]+', '-', header.get_text(strip=True).lower())[:50]

        section_title = header.get_text(strip=True)
        if not section_title:
            continue

        # Determine stop tags based on header level
        if header.name == 'h2':
            stop_tags = ['h2']
        else:  # h3
            stop_tags = ['h2', 'h3']

        # Extract content for this section
        content = extract_section_content(header, stop_tags)
        word_count = len(content.split())

        # Skip very small sections
        if word_count < MANUAL_MIN_WORDS:
            continue

        # Build doc path with anchor
        doc_path = f"{relative_path}#{section_id}" if section_id else relative_path

        # Split large sections
        for suffix, chunk_content in split_large_content(content, section_title, MANUAL_SPLIT_SIZE):
            chunk_word_count = len(chunk_content.split())
            chunk_num = len([c for c in chunks if c.doc_path.startswith(doc_path)])

            chunks.append(DocChunk(
                title=f"{section_title}{suffix}",
                doc_type="manual",
                doc_path=doc_path,
                doc_version=DOC_VERSION,
                chunk_id=f"{doc_path}#{chunk_num}",
                content=chunk_content,
                word_count=chunk_word_count,
                source_file=str(html_path),
                section_type=header.name
            ))

    # Fallback: if headers exist but all sections were too small, try full-page chunking
    if not chunks and headers:
        main_content = soup.find(['main', 'article', 'div'], class_=re.compile(r'body|content|document'))
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
        else:
            text = soup.get_text(separator='\n', strip=True)

        text = clean_text(text)
        word_count = len(text.split())

        # Use a lower threshold for fallback (30 words instead of 50)
        if word_count >= 30:
            for suffix, chunk_content in split_large_content(text, page_title, MANUAL_SPLIT_SIZE):
                chunk_word_count = len(chunk_content.split())
                chunk_num = len(chunks)
                chunks.append(DocChunk(
                    title=f"{page_title}{suffix}",
                    doc_type="manual",
                    doc_path=relative_path,
                    doc_version=DOC_VERSION,
                    chunk_id=f"{relative_path}#fallback#{chunk_num}",
                    content=chunk_content,
                    word_count=chunk_word_count,
                    source_file=str(html_path),
                    section_type="full_page_fallback"
                ))

    return chunks


def chunk_api_page(html_path: Path, base_path: Path) -> List[DocChunk]:
    """
    Chunk an API reference page by class/function definitions.

    Target: 300-800 words per chunk.
    Preserves: signatures, property lists, method documentation.
    """
    if not BS4_AVAILABLE:
        return []

    try:
        with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f.read(), 'lxml')
    except Exception as e:
        print(f"  Error parsing {html_path.name}: {e}", file=sys.stderr)
        return []

    relative_path = str(html_path.relative_to(base_path))
    chunks = []

    # Remove script, style elements
    for tag in soup.find_all(['script', 'style']):
        tag.decompose()

    # Find definition lists (dl elements with class 'py class', 'py function', etc.)
    definitions = soup.find_all('dl', class_=re.compile(r'py\s+(class|function|method|attribute|data)'))

    if not definitions:
        # Fallback: try to find any dl elements that look like API docs
        definitions = soup.find_all('dl')
        definitions = [d for d in definitions if d.find('dt', id=True)]

    if not definitions:
        # No API definitions found - use manual chunking as fallback
        return chunk_manual_page(html_path, base_path)

    for defn in definitions:
        # Get the definition term (class/function name and signature)
        dt = defn.find('dt')
        if not dt:
            continue

        # Extract ID for anchor
        api_id = dt.get('id', '')

        # Extract name and signature
        sig_text = dt.get_text(strip=True)
        if not sig_text:
            continue

        # Clean up signature
        sig_text = clean_text(sig_text)

        # Extract a clean name for the title
        name_match = re.match(r'^([\w\.]+)', sig_text)
        name = name_match.group(1) if name_match else sig_text[:80]

        # Determine definition type
        defn_classes = defn.get('class', [])
        if 'class' in str(defn_classes):
            def_type = "class"
        elif 'function' in str(defn_classes):
            def_type = "function"
        elif 'method' in str(defn_classes):
            def_type = "method"
        elif 'attribute' in str(defn_classes) or 'data' in str(defn_classes):
            def_type = "attribute"
        else:
            def_type = "definition"

        # Get the definition description
        dd = defn.find('dd')
        description = ""
        if dd:
            description = dd.get_text(separator='\n', strip=True)
            description = clean_text(description)

        # Build full content with signature
        if len(sig_text) > 200:
            # Very long signature - truncate
            sig_display = sig_text[:200] + "..."
        else:
            sig_display = sig_text

        full_content = f"```python\n{sig_display}\n```\n\n{description}"
        word_count = len(full_content.split())

        # Skip very small entries
        if word_count < API_MIN_WORDS:
            continue

        # Truncate if too large
        if word_count > API_TARGET_MAX:
            words = full_content.split()
            full_content = ' '.join(words[:API_TARGET_MAX]) + "\n\n[Content truncated]"
            word_count = API_TARGET_MAX

        # Build doc path with anchor
        doc_path = f"{relative_path}#{api_id}" if api_id else relative_path

        chunks.append(DocChunk(
            title=name,
            doc_type="api",
            doc_path=doc_path,
            doc_version=DOC_VERSION,
            chunk_id=f"{doc_path}#0",
            content=full_content,
            word_count=word_count,
            source_file=str(html_path),
            section_type=def_type
        ))

    return chunks


def process_directory(
    path: Path,
    base_path: Path,
    doc_type: str,
    stats: ChunkingStats
) -> List[DocChunk]:
    """
    Process all HTML files in a directory.

    Args:
        path: Directory to process
        base_path: Base path for relative paths
        doc_type: "manual" or "api"
        stats: Statistics tracker

    Returns:
        List of all chunks
    """
    all_chunks = []

    if not path.exists():
        print(f"Warning: Path does not exist: {path}", file=sys.stderr)
        return all_chunks

    html_files = list(path.rglob("*.html"))
    stats.total_files += len(html_files)

    print(f"Processing {len(html_files)} {doc_type} files from {path}...")

    for i, html_file in enumerate(html_files):
        if should_skip_file(html_file):
            stats.files_skipped += 1
            continue

        try:
            if doc_type == "api":
                chunks = chunk_api_page(html_file, base_path)
            else:
                chunks = chunk_manual_page(html_file, base_path)

            for chunk in chunks:
                stats.add_chunk(chunk)
                all_chunks.append(chunk)

            stats.files_processed += 1

            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(html_files)} files ({len(all_chunks)} chunks)...")

        except Exception as e:
            stats.files_errored += 1
            stats.errors.append(f"{html_file.name}: {str(e)[:100]}")

    return all_chunks


def main():
    parser = argparse.ArgumentParser(
        description="Chunk Blender documentation for vector store upload"
    )
    parser.add_argument(
        "--docs-base",
        type=Path,
        default=DEFAULT_DOCS_BASE,
        help="Base path for Blender documentation"
    )
    parser.add_argument(
        "--manual-path",
        type=Path,
        default=None,
        help="Path to manual HTML docs (default: {docs-base}/blender_manual_html)"
    )
    parser.add_argument(
        "--api-path",
        type=Path,
        default=None,
        help="Path to API reference HTML docs (default: {docs-base}/blender_python_reference_5_0)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for chunked files (if not set, just prints stats)"
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Preview N chunks to stdout"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only output statistics, no chunks"
    )
    parser.add_argument(
        "--json-stats",
        type=Path,
        default=None,
        help="Output statistics to JSON file"
    )
    parser.add_argument(
        "--manual-only",
        action="store_true",
        help="Only process manual documentation"
    )
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Only process API documentation"
    )

    args = parser.parse_args()

    if not BS4_AVAILABLE:
        print("Error: beautifulsoup4 is required. Install with: pip install beautifulsoup4 lxml")
        sys.exit(1)

    # Resolve paths
    docs_base = args.docs_base
    manual_path = args.manual_path or docs_base / "blender_manual_html"
    api_path = args.api_path or docs_base / "blender_python_reference_5_0"

    print("=" * 60)
    print("Blender Documentation Chunker")
    print("=" * 60)
    print(f"Docs base: {docs_base}")
    print(f"Manual path: {manual_path}")
    print(f"API path: {api_path}")
    print(f"Doc version: {DOC_VERSION}")
    print()

    stats = ChunkingStats()
    all_chunks = []

    # Process manual docs
    if not args.api_only:
        manual_chunks = process_directory(manual_path, docs_base, "manual", stats)
        all_chunks.extend(manual_chunks)
        print(f"  Manual chunks: {len(manual_chunks)}")

    # Process API docs
    if not args.manual_only:
        api_chunks = process_directory(api_path, docs_base, "api", stats)
        all_chunks.extend(api_chunks)
        print(f"  API chunks: {len(api_chunks)}")

    # Print summary
    print()
    print("=" * 60)
    print("CHUNKING SUMMARY")
    print("=" * 60)
    print(f"Total files: {stats.total_files}")
    print(f"  Processed: {stats.files_processed}")
    print(f"  Skipped: {stats.files_skipped}")
    print(f"  Errors: {stats.files_errored}")
    print()
    print(f"Total chunks: {stats.total_chunks}")
    print(f"  Manual: {stats.chunks_by_type.get('manual', 0)}")
    print(f"  API: {stats.chunks_by_type.get('api', 0)}")
    print()
    print("Word count distribution:")
    for bucket, count in sorted(stats.word_count_distribution.items()):
        print(f"  {bucket}: {count}")

    if stats.errors:
        print()
        print(f"Errors ({len(stats.errors)} total, showing first 10):")
        for err in stats.errors[:10]:
            print(f"  - {err}")

    # Preview chunks
    if args.preview > 0:
        print()
        print("=" * 60)
        print(f"PREVIEW (first {args.preview} chunks)")
        print("=" * 60)
        for chunk in all_chunks[:args.preview]:
            print()
            print("-" * 40)
            print(chunk.to_upload_text()[:1000])
            if len(chunk.content) > 800:
                print("...")

    # Save stats to JSON
    if args.json_stats:
        args.json_stats.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_stats, 'w') as f:
            json.dump(stats.to_dict(), f, indent=2)
        print(f"\nStats saved to: {args.json_stats}")

    # Save chunks to output directory
    if args.output and not args.stats_only:
        args.output.mkdir(parents=True, exist_ok=True)

        # Save chunks as individual files
        manual_dir = args.output / "manual"
        api_dir = args.output / "api"
        manual_dir.mkdir(exist_ok=True)
        api_dir.mkdir(exist_ok=True)

        print()
        print(f"Saving chunks to: {args.output}")

        for i, chunk in enumerate(all_chunks):
            # Create safe filename from chunk_id
            safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', chunk.chunk_id)[:100]
            filename = f"{safe_name}.md"

            if chunk.doc_type == "api":
                out_path = api_dir / filename
            else:
                out_path = manual_dir / filename

            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(chunk.to_upload_text())

            if (i + 1) % 500 == 0:
                print(f"  Saved {i + 1}/{len(all_chunks)} chunks...")

        # Save manifest
        manifest = {
            "doc_version": DOC_VERSION,
            "total_chunks": len(all_chunks),
            "manual_chunks": stats.chunks_by_type.get("manual", 0),
            "api_chunks": stats.chunks_by_type.get("api", 0),
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "doc_type": c.doc_type,
                    "doc_path": c.doc_path,
                    "title": c.title,
                    "word_count": c.word_count,
                }
                for c in all_chunks
            ]
        }

        with open(args.output / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"  Saved manifest.json")
        print(f"\nDone! {len(all_chunks)} chunks saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
