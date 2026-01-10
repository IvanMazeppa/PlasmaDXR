"""
Blender Documentation Search Tools - OpenAI Agents SDK Function Tools.

This module contains the search logic extracted from blender-manual/blender_server.py
as standalone functions wrapped with @function_tool decorator.

This solves the MCP nesting architectural issue where spawning child MCP clients
from within MCP tool handlers causes anyio TaskGroup conflicts:
    RuntimeError: Attempted to exit cancel scope in a different task than it was entered in

By importing these functions directly, we avoid MCP transport entirely while
maintaining the same search capabilities.

Usage:
    from shared import search_manual, search_vdb_workflow, read_page

    # As function_tool for Agents SDK
    docs_agent = Agent(
        name="DocsExpert",
        tools=[search_manual, search_vdb_workflow, read_page],
    )

    # Or call directly
    results = search_manual("volume rendering")
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# OpenAI Agents SDK
try:
    from agents import function_tool
except ImportError:
    # Fallback: create a no-op decorator if agents SDK not available
    def function_tool(fn):
        """No-op decorator when agents SDK not available."""
        return fn

# BeautifulSoup for HTML parsing
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # Will fail gracefully on first use


# =============================================================================
# CONFIGURATION
# =============================================================================

# Server/cache location (same directory as blender-manual for shared cache)
_BLENDER_MANUAL_DIR = Path(__file__).parent.parent.parent / "blender-manual"

# Manual HTML files location
MANUAL_DIR = Path("/mnt/d/Users/dilli/AndroidStudioProjects/blender_manual_mcp")

# Python API reference location
PYTHON_API_DIR = Path("/mnt/d/Users/dilli/AndroidStudioProjects/blender_python_reference_mcp")

# Cache file location (shared with blender-manual MCP server)
CACHE_FILE = _BLENDER_MANUAL_DIR / "manual_index.json"
EMBEDDINGS_FILE = _BLENDER_MANUAL_DIR / "embeddings.npy"
CACHE_VERSION = "4.0"

# Token optimization defaults
DEFAULT_LIMIT = 5
DEFAULT_SNIPPET_LENGTH = 100
MAX_PAGE_LENGTH = 4000

# Semantic search settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32

# VDB/Volume-specific keywords for specialized searches
VDB_KEYWORDS = {
    "export": ["export", "save", "write", "output", ".vdb", "openvdb", "nanovdb"],
    "bake": ["bake", "cache", "simulation", "mantaflow", "fluid", "smoke", "fire"],
    "python": ["bpy.ops", "bpy.types", "bpy.data", "script", "python", "api"],
    "volume_render": ["principled volume", "volume scatter", "density", "emission", "volumetric"],
    "coordinate": ["coordinate", "transform", "axis", "space", "rotation", "scale"],
    "lod": ["lod", "level of detail", "resolution", "subdivision", "voxel"]
}


# =============================================================================
# LOGGING
# =============================================================================

logger = logging.getLogger("blender_docs_tools")
logger.setLevel(logging.INFO)

# Add handler if none exists
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(handler)


# =============================================================================
# IN-MEMORY DATA (Lazy-loaded)
# =============================================================================

_search_index: List[Dict[str, Any]] = []
_index_loaded: bool = False
_embeddings: Optional[Any] = None  # numpy array
_embedding_model: Optional[Any] = None  # SentenceTransformer

# Semantic search availability (lazy-checked)
_semantic_available: Optional[bool] = None
_numpy = None
_SentenceTransformer = None


# =============================================================================
# SEMANTIC SEARCH SUPPORT (Lazy-loaded)
# =============================================================================

def _check_semantic_available() -> bool:
    """Lazy-load semantic search dependencies."""
    global _semantic_available, _numpy, _SentenceTransformer

    if _semantic_available is not None:
        return _semantic_available

    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        _numpy = np
        _SentenceTransformer = SentenceTransformer
        _semantic_available = True
        logger.debug("Semantic search dependencies loaded successfully")
    except ImportError as e:
        logger.debug(f"Semantic search not available: {e}")
        _semantic_available = False

    return _semantic_available


def _load_embedding_model() -> bool:
    """Load the sentence transformer model for semantic search."""
    global _embedding_model

    if not _check_semantic_available():
        return False

    if _embedding_model is not None:
        return True

    try:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}...")
        _embedding_model = _SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded")
        return True
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        return False


def _load_embeddings() -> bool:
    """Load embeddings from file."""
    global _embeddings

    if not _check_semantic_available():
        return False

    if not EMBEDDINGS_FILE.exists():
        logger.debug("No embeddings file found")
        return False

    try:
        _embeddings = _numpy.load(EMBEDDINGS_FILE)

        # Verify size matches index
        if len(_embeddings) != len(_search_index):
            logger.warning(
                f"Embeddings size mismatch: {len(_embeddings)} vs {len(_search_index)} pages"
            )
            _embeddings = None
            return False

        logger.info(f"Loaded {len(_embeddings)} embeddings")
        return True

    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        _embeddings = None
        return False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    lines = [line.strip() for line in text.splitlines()]
    return " ".join(filter(None, lines))


def normalize_path(path: str) -> str:
    """Normalize path separators to forward slashes."""
    return path.replace('\\', '/')


def extract_category_from_path(rel_path: str, source: str = "manual") -> Tuple[str, str, str]:
    """
    Extract category, subcategory, and page type from file path.

    Args:
        rel_path: Relative path to HTML file
        source: "manual" or "python_api"

    Returns:
        Tuple of (category, subcategory, page_type)
    """
    rel_path = normalize_path(rel_path)

    # Handle Python API docs (flat structure with module naming)
    if source == "python_api":
        filename = Path(rel_path).stem
        parts = filename.split('.')
        if len(parts) >= 2:
            return (parts[0], parts[1] if len(parts) > 1 else "", "api")
        return (filename, "", "api")

    # Handle manual docs (hierarchical structure)
    parts = rel_path.split('/')
    if not parts:
        return ("general", "", "reference")

    category = parts[0] if parts else "general"
    subcategory = parts[1] if len(parts) > 1 else ""

    # Determine page type
    page_type = "reference"
    if "getting_started" in rel_path or "tutorial" in rel_path.lower():
        page_type = "tutorial"
    elif "advanced/scripting" in rel_path or "python" in rel_path:
        page_type = "api"

    return (category, subcategory, page_type)


def extract_module_path(filename: str) -> str:
    """Extract Python module path from API doc filename."""
    return Path(filename).stem


def extract_keywords(content: str, title: str, category: str, subcategory: str) -> List[str]:
    """Extract relevant keywords from content and metadata."""
    keywords = set()

    keywords.add(category)
    if subcategory:
        keywords.add(subcategory)

    # Extract from title
    title_words = re.findall(r'\b\w{4,}\b', title.lower())
    keywords.update(title_words)

    # Look for VDB/volume-related terms
    content_lower = content.lower()
    vdb_terms = [
        "vdb", "openvdb", "nanovdb", "volume", "volumetric", "fluid", "smoke",
        "mantaflow", "pyro", "density", "voxel", "export", "cache", "bake"
    ]
    for term in vdb_terms:
        if term in content_lower:
            keywords.add(term)

    # Look for Python API mentions
    if "bpy." in content:
        keywords.add("python_api")

    # Look for node mentions
    if "node" in content_lower or "shader" in content_lower:
        keywords.add("nodes")

    return list(keywords)


def extract_headers(soup: "BeautifulSoup") -> List[str]:
    """Extract all headers from the page."""
    headers = []
    article = soup.find("article", role="main")
    if article:
        for header in article.find_all(['h1', 'h2', 'h3', 'h4']):
            header_text = header.get_text().strip().replace('¶', '').strip()
            if header_text:
                headers.append(header_text)
    return headers


def extract_code_blocks(soup: "BeautifulSoup") -> List[str]:
    """Extract code blocks from the page."""
    code_blocks = []
    article = soup.find("article", role="main")
    if article:
        for pre in article.find_all('pre'):
            code = pre.get_text().strip()
            if code and len(code) < 500:
                code_blocks.append(code)
    return code_blocks[:5]


def compute_score(
    item: Dict[str, Any],
    query_terms: List[str],
    boost_categories: Optional[List[str]] = None,
    boost_keywords: Optional[List[str]] = None,
    original_query: Optional[str] = None
) -> int:
    """Compute relevance score for a search item with improved matching."""
    score = 0
    content_lower = item["content"].lower()
    title_lower = item["title"].lower()

    # Exact phrase match bonus
    if original_query and len(query_terms) > 1:
        query_phrase = original_query.lower()
        if query_phrase in title_lower:
            score += 50
        if query_phrase in content_lower:
            score += 25

    # Title matches (highest weight)
    title_match_count = 0
    for term in query_terms:
        if term in title_lower:
            score += 20
            title_match_count += 1

    # All terms match in title bonus
    if title_match_count == len(query_terms) and len(query_terms) > 1:
        score += 30

    # Header matches
    for header in item.get("headers", []):
        header_lower = header.lower()
        for term in query_terms:
            if term in header_lower:
                score += 10

    # Keyword matches
    item_keywords = [k.lower() for k in item.get("keywords", [])]
    for term in query_terms:
        if term in item_keywords:
            score += 15

    # Content matches (capped)
    for term in query_terms:
        count = content_lower.count(term)
        score += min(count, 10)

    # Proximity bonus
    if len(query_terms) > 1:
        for i, term1 in enumerate(query_terms):
            pos1 = content_lower.find(term1)
            if pos1 != -1:
                for term2 in query_terms[i+1:]:
                    pos2 = content_lower.find(term2)
                    if pos2 != -1 and abs(pos1 - pos2) < 50:
                        score += 5

    # Category boost
    if boost_categories:
        if item.get("category") in boost_categories or item.get("subcategory") in boost_categories:
            score += 25

    # Keyword boost
    if boost_keywords:
        for keyword in boost_keywords:
            if keyword in content_lower or keyword in item_keywords:
                score += 10

    # Path depth penalty
    depth = item["path"].count('/')
    score -= depth

    # Index/overview page boost
    if "index.html" in item["path"] or "introduction" in title_lower:
        score += 5

    return max(0, score)


def format_results(
    results: List[Dict],
    query: str,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
    compact: bool = False
) -> str:
    """Format search results with token optimization options."""
    if not results:
        return (
            f"No results found for '{query}'.\n\n"
            "**Suggestions:**\n"
            "- Try different keywords\n"
            "- Use browse_hierarchy() to explore available topics"
        )

    total = len(results)
    paginated = results[offset:offset + limit]

    # Count sources
    api_count = sum(1 for r in paginated if r.get('source') == 'python_api')
    manual_count = len(paginated) - api_count

    if compact:
        output = f"Found {total} results for '{query}' (showing {offset+1}-{offset+len(paginated)}"
        if api_count > 0 and manual_count > 0:
            output += f" | {manual_count} manual, {api_count} API"
        output += "):\n\n"
        for i, r in enumerate(paginated, offset + 1):
            source_tag = "[API]" if r.get('source') == 'python_api' else ""
            output += f"{i}. {source_tag}**{r['title']}** | `{r['path']}` | {r['score']}pts\n"
        if offset + limit < total:
            output += f"\n[{total - offset - limit} more - use offset={offset + limit}]"
        return output

    # Standard mode
    output = f"Found {total} results for '{query}' (showing {offset+1}-{offset+len(paginated)}"
    if api_count > 0 and manual_count > 0:
        output += f" | {manual_count} manual, {api_count} API"
    output += "):\n\n"

    for i, r in enumerate(paginated, offset + 1):
        source_tag = " [API]" if r.get('source') == 'python_api' else ""
        output += f"## {i}. {r['title']}{source_tag}\n"
        output += f"**Path:** `{r['path']}` | **Score:** {r['score']}\n"
        if r.get('headers'):
            output += f"**Sections:** {', '.join(r['headers'][:3])}\n"
        output += f"{r.get('snippet', '')}\n\n"

    if offset + limit < total:
        output += f"[{total - offset - limit} more results - use offset={offset + limit}]"

    return output


def create_snippet(
    content: str,
    query_terms: List[str],
    length: int = DEFAULT_SNIPPET_LENGTH
) -> str:
    """Create a relevant snippet from content based on query terms."""
    content_lower = content.lower()

    for term in query_terms:
        idx = content_lower.find(term)
        if idx != -1:
            start = max(0, idx - 30)
            end = min(len(content), idx + length)
            snippet = content[start:end]
            if start > 0:
                snippet = "..." + snippet
            if end < len(content):
                snippet = snippet + "..."
            return snippet

    return content[:length] + ("..." if len(content) > length else "")


def _semantic_search(query: str, top_k: int = 10) -> List[Dict]:
    """Perform semantic similarity search."""
    global _embeddings, _embedding_model

    if _embeddings is None or _embedding_model is None:
        return []

    try:
        query_embedding = _embedding_model.encode([query], convert_to_numpy=True)[0]

        similarities = _numpy.dot(_embeddings, query_embedding) / (
            _numpy.linalg.norm(_embeddings, axis=1) * _numpy.linalg.norm(query_embedding)
        )

        top_indices = _numpy.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            item = _search_index[idx]
            results.append({
                "title": item["title"],
                "path": item["path"],
                "headers": item.get("headers", []),
                "similarity": float(similarities[idx]),
                "snippet": item["content"][:DEFAULT_SNIPPET_LENGTH] + "..."
            })

        return results

    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return []


# =============================================================================
# INDEX MANAGEMENT
# =============================================================================

def _index_directory(base_dir: Path, source: str) -> int:
    """Index a single directory and return count of indexed pages."""
    global _search_index
    count = 0

    if BeautifulSoup is None:
        logger.error("BeautifulSoup not available - cannot index HTML files")
        return 0

    if not base_dir.exists():
        logger.warning(f"Directory not found: {base_dir}")
        return 0

    if not base_dir.is_dir():
        logger.warning(f"Path is not a directory: {base_dir}")
        return 0

    html_count = sum(1 for _ in base_dir.rglob("*.html"))
    if html_count == 0:
        logger.warning(f"No HTML files found in {base_dir}")
        return 0

    logger.info(f"Indexing {source} from {base_dir} ({html_count} HTML files)...")

    for root, _, files in os.walk(base_dir):
        for file in files:
            if not file.endswith(".html"):
                continue
            if file.startswith("genindex") or file.startswith("search"):
                continue

            path = Path(root) / file
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    html_content = f.read()
                    soup = BeautifulSoup(html_content, "html.parser")

                    # Extract title
                    title = "Untitled"
                    if soup.title and soup.title.string:
                        title_text = soup.title.string
                        if source == "python_api":
                            title = title_text.replace("— Blender Python API", "").strip()
                        else:
                            title = title_text.replace("— Blender Manual", "").replace("— Blender 5.0 Manual", "").strip()

                    # Extract main content
                    content_div = soup.find("article", role="main")
                    if not content_div:
                        content_div = soup.find("div", class_="document")
                    if not content_div:
                        content_div = soup.body
                    if not content_div:
                        continue

                    text = clean_text(content_div.get_text())
                    rel_path = normalize_path(str(path.relative_to(base_dir)))

                    category, subcategory, page_type = extract_category_from_path(rel_path, source)
                    keywords = extract_keywords(text, title, category, subcategory)
                    headers = extract_headers(soup)[:5]
                    code_blocks = extract_code_blocks(soup)

                    if source == "python_api":
                        module_path = extract_module_path(file)
                        keywords.append(module_path)
                        keywords.extend(module_path.split('.'))

                    index_entry = {
                        "path": rel_path,
                        "title": title,
                        "content": text[:10000],
                        "category": category,
                        "subcategory": subcategory,
                        "page_type": page_type,
                        "keywords": keywords[:15],
                        "headers": headers,
                        "code_count": len(code_blocks),
                        "source": source,
                        "module_path": extract_module_path(file) if source == "python_api" else None,
                    }

                    _search_index.append(index_entry)
                    count += 1

            except Exception as e:
                logger.error(f"Failed to index {path}: {e}")

    return count


def _load_cache() -> bool:
    """Load index from cache file."""
    global _search_index

    if not CACHE_FILE.exists():
        logger.info("No cache file found")
        return False

    try:
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)

        if cache_data.get("version") != CACHE_VERSION:
            logger.info("Cache version mismatch, rebuilding required")
            return False

        _search_index = cache_data.get("index", [])
        logger.info(
            f"Loaded {len(_search_index)} pages from cache "
            f"(built at {cache_data.get('built_at')})"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to load cache: {e}")
        return False


def _save_cache():
    """Save index to cache file."""
    try:
        cache_data = {
            "version": CACHE_VERSION,
            "built_at": datetime.now().isoformat(),
            "blender_version": "5.0",
            "page_count": len(_search_index),
            "index": _search_index
        }

        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f)

        logger.info(f"Saved cache ({len(_search_index)} pages) to {CACHE_FILE}")

    except Exception as e:
        logger.error(f"Failed to save cache: {e}")


def _build_index():
    """Build the search index from HTML files."""
    global _search_index
    _search_index = []

    manual_count = _index_directory(MANUAL_DIR, "manual")
    logger.info(f"Indexed {manual_count} manual pages")

    api_count = _index_directory(PYTHON_API_DIR, "python_api")
    logger.info(f"Indexed {api_count} Python API pages")

    total = manual_count + api_count
    logger.info(f"Total indexed: {total} pages")

    _save_cache()


def ensure_index_loaded() -> bool:
    """
    Ensure the search index is loaded (lazy initialization).

    Call this before performing searches. Returns True if index is ready.
    """
    global _index_loaded

    if _index_loaded and _search_index:
        return True

    logger.info("Loading Blender documentation index...")

    if _load_cache():
        _index_loaded = True
        return True

    logger.info("Cache not available, building index (this may take 60-90 seconds)...")
    _build_index()

    if _search_index:
        _index_loaded = True
        return True

    logger.error("Failed to load or build index")
    return False


def get_index_stats() -> Dict[str, Any]:
    """Get statistics about the loaded index."""
    if not _search_index:
        return {"loaded": False, "page_count": 0}

    manual_count = sum(1 for item in _search_index if item.get("source") == "manual")
    api_count = sum(1 for item in _search_index if item.get("source") == "python_api")

    return {
        "loaded": True,
        "page_count": len(_search_index),
        "manual_pages": manual_count,
        "api_pages": api_count,
        "cache_file": str(CACHE_FILE),
        "semantic_available": _check_semantic_available(),
        "embeddings_loaded": _embeddings is not None,
    }


# =============================================================================
# FUNCTION TOOLS (12 tools wrapped for OpenAI Agents SDK)
# =============================================================================

@function_tool
def search_manual(
    query: str,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
    compact: bool = False
) -> str:
    """
    Enhanced general search across the entire Blender Manual.

    Args:
        query: Search query (keywords, phrases)
        limit: Maximum number of results to return (default: 5)
        offset: Skip first N results for pagination (default: 0)
        compact: If True, returns minimal output (titles/paths only)

    Returns:
        Formatted search results with scores, categories, and snippets.

    Examples:
        - "volume rendering"
        - "export openvdb"
        - "python api fluid simulation"
    """
    if not ensure_index_loaded():
        return "Error: Failed to load documentation index"

    query_terms = query.lower().split()
    results = []

    for item in _search_index:
        score = compute_score(item, query_terms, original_query=query)
        if score > 0:
            results.append({
                "title": item["title"],
                "path": item["path"],
                "headers": item.get("headers", []),
                "score": score,
                "source": item.get("source", "manual"),
                "snippet": create_snippet(item["content"], query_terms) if not compact else ""
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return format_results(results, query, limit, offset, compact)


@function_tool
def search_tutorials(
    topic: str,
    technique: Optional[str] = None,
    limit: int = DEFAULT_LIMIT,
    compact: bool = False
) -> str:
    """
    Search for tutorials and getting started guides.
    Optimized for finding learning resources, especially for volumetrics and VDB workflows.

    Args:
        topic: Main topic (e.g., "volumetrics", "fluid simulation", "rendering")
        technique: Optional specific technique (e.g., "mantaflow", "pyro", "smoke")
        limit: Maximum results (default: 5)
        compact: Minimal output mode

    Returns:
        Tutorial pages with step-by-step guides.

    Examples:
        - search_tutorials("volumetrics")
        - search_tutorials("fluid simulation", "smoke")
        - search_tutorials("rendering", "cycles")
    """
    if not ensure_index_loaded():
        return "Error: Failed to load documentation index"

    query = topic + (f" {technique}" if technique else "")
    query_terms = query.lower().split()
    results = []

    for item in _search_index:
        is_tutorial = (
            item.get("page_type") == "tutorial" or
            "getting_started" in item["path"] or
            "introduction" in item["title"].lower() or
            "tutorial" in item["content"].lower()[:500]
        )

        if not is_tutorial:
            continue

        score = compute_score(
            item, query_terms,
            boost_categories=["getting_started", "introduction"],
            original_query=query
        )

        if score > 0:
            results.append({
                "title": item["title"],
                "path": item["path"],
                "headers": item.get("headers", []),
                "score": score,
                "source": item.get("source", "manual"),
                "snippet": create_snippet(item["content"], query_terms) if not compact else ""
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return format_results(results, query, limit, 0, compact)


@function_tool
def browse_hierarchy(path: Optional[str] = None) -> str:
    """
    Browse the Blender Manual's hierarchical structure like a file tree.
    Navigate through categories and sections to discover content.

    Args:
        path: Optional path prefix to browse (e.g., "render", "physics/fluid")
              If None, shows top-level categories only.

    Returns:
        Hierarchical listing of subcategories and pages.

    Examples:
        - browse_hierarchy()  # Shows top-level categories
        - browse_hierarchy("render")
        - browse_hierarchy("physics/fluid")
    """
    if not ensure_index_loaded():
        return "Error: Failed to load documentation index"

    # Normalize path
    search_path = ""
    if path:
        clean_path = path.strip().strip('"\'')
        clean_path = normalize_path(clean_path).rstrip('/')
        if clean_path:
            search_path = clean_path + '/'

    subdirs = set()
    pages = []

    for item in _search_index:
        item_path = normalize_path(item["path"])

        if search_path:
            if not item_path.lower().startswith(search_path.lower()):
                continue
            rel_path = item_path[len(search_path):]
        else:
            rel_path = item_path

        if not rel_path:
            continue

        if '/' in rel_path:
            subdir = rel_path.split('/')[0]
            if subdir:
                subdirs.add(subdir)
        else:
            pages.append({
                "title": item["title"],
                "path": item_path,
                "source": item.get("source", "manual")
            })

    # Format output
    if search_path:
        output = f"# {search_path[:-1]}/\n\n"
    else:
        output = "# Blender Manual - Categories\n\n"

    if subdirs:
        output += "**Subcategories:**\n"
        for subdir in sorted(subdirs)[:20]:
            full = f"{search_path}{subdir}" if search_path else subdir
            output += f"- `{subdir}/` → browse_hierarchy(\"{full}\")\n"
        if len(subdirs) > 20:
            output += f"- ... and {len(subdirs) - 20} more\n"
        output += "\n"

    if pages:
        output += "**Pages:**\n"
        for p in sorted(pages, key=lambda x: x["title"])[:10]:
            source_tag = "[API]" if p.get("source") == "python_api" else ""
            output += f"- {source_tag}{p['title']} (`{p['path']}`)\n"
        if len(pages) > 10:
            output += f"- ... and {len(pages) - 10} more pages\n"

    if not subdirs and not pages:
        output += "No content found at this path.\n\n"
        output += "**Suggestions:**\n"
        output += "- Try `browse_hierarchy()` to see top-level categories\n"
        output += "- Use forward slashes: `physics/fluid` not `physics\\fluid`\n"

    return output


@function_tool
def search_vdb_workflow(
    query: str,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
    compact: bool = False
) -> str:
    """
    Specialized search for VDB/OpenVDB/NanoVDB workflows.
    Optimized for finding export settings, baking simulations, and volume handling.

    Args:
        query: VDB-related query (e.g., "export vdb", "bake mantaflow", "cache smoke")
        limit: Maximum results (default: 5)
        offset: Pagination offset
        compact: Minimal output mode

    Returns:
        VDB-specific documentation with high relevance.

    Examples:
        - "export openvdb"
        - "bake mantaflow smoke"
        - "cache fluid simulation"
        - "vdb format settings"
    """
    if not ensure_index_loaded():
        return "Error: Failed to load documentation index"

    query_terms = query.lower().split()

    # Expand query with VDB keywords
    expanded = []
    for term in query_terms:
        if term in VDB_KEYWORDS:
            expanded.extend(VDB_KEYWORDS[term])

    all_terms = list(set(query_terms + expanded))
    results = []

    for item in _search_index:
        score = compute_score(
            item, all_terms,
            boost_categories=["physics", "render", "files"],
            boost_keywords=["vdb", "openvdb", "volume", "fluid", "cache"]
        )

        if score > 5:
            results.append({
                "title": item["title"],
                "path": item["path"],
                "headers": item.get("headers", []),
                "score": score,
                "source": item.get("source", "manual"),
                "snippet": create_snippet(item["content"], all_terms) if not compact else ""
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return format_results(results, query, limit, offset, compact)


@function_tool
def search_python_api(
    operation: str,
    limit: int = DEFAULT_LIMIT,
    compact: bool = False,
    api_only: bool = False
) -> str:
    """
    Search for Python API documentation (bpy.ops, bpy.types, bpy.data).
    Searches BOTH the official Python API reference AND manual scripting pages.

    Args:
        operation: API operation or concept (e.g., "bpy.ops.fluid", "export script", "volume")
        limit: Maximum results (default: 5)
        compact: Minimal output mode
        api_only: If True, only search official API docs (not manual)

    Returns:
        Python API documentation and code examples.

    Examples:
        - "bpy.ops.fluid.bake"
        - "export volume script"
        - "bpy.data.objects"
        - "bmesh.types"
    """
    if not ensure_index_loaded():
        return "Error: Failed to load documentation index"

    query_terms = operation.lower().split()
    results = []

    for item in _search_index:
        is_api_doc = item.get("source") == "python_api"
        has_code = item.get("code_count", 0) > 0
        has_python_api = "python_api" in item.get("keywords", [])

        if api_only and not is_api_doc:
            continue

        if not api_only and not (
            is_api_doc or has_code or has_python_api or
            "advanced/scripting" in item["path"]
        ):
            continue

        score = compute_score(
            item, query_terms,
            boost_categories=["bpy", "bmesh", "aud", "advanced", "scripting"],
            boost_keywords=["python", "bpy", "script", "api", "ops", "types", "data"]
        )

        if is_api_doc:
            score += 50
        if has_code:
            score += 30

        if is_api_doc and item.get("module_path"):
            module_lower = item["module_path"].lower()
            for term in query_terms:
                if term in module_lower:
                    score += 25

        if score > 0:
            snippet = ""
            source_tag = "[API]" if is_api_doc else "[Manual]"
            if not compact:
                snippet = create_snippet(item["content"], query_terms)
                if item.get("code_count"):
                    snippet += f" [{item['code_count']} code block(s)]"

            results.append({
                "title": f"{source_tag} {item['title']}",
                "path": item["path"],
                "headers": item.get("headers", []),
                "score": score,
                "snippet": snippet,
                "source": item.get("source", "manual"),
                "module_path": item.get("module_path")
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return format_results(results, operation, limit, 0, compact)


@function_tool
def search_nodes(
    node_type: str,
    category: Optional[str] = None,
    limit: int = DEFAULT_LIMIT,
    compact: bool = False
) -> str:
    """
    Search for shader nodes, compositor nodes, and geometry nodes.
    Optimized for finding node documentation, especially volume-related nodes.

    Args:
        node_type: Node name or type (e.g., "Principled Volume", "Volume Scatter", "Math")
        category: Optional node category ("shader", "compositor", "geometry")
        limit: Maximum results (default: 5)
        compact: Minimal output mode

    Returns:
        Node documentation with inputs, outputs, and usage examples.

    Examples:
        - search_nodes("Principled Volume")
        - search_nodes("Volume Scatter", "shader")
        - search_nodes("blur", "compositor")
    """
    if not ensure_index_loaded():
        return "Error: Failed to load documentation index"

    query_terms = node_type.lower().split()
    if category:
        query_terms.append(category.lower())

    results = []

    for item in _search_index:
        is_node_page = (
            "nodes" in item.get("keywords", []) or
            "node" in item["path"] or
            "shader" in item["path"] or
            "compositing" in item["path"]
        )

        if not is_node_page:
            continue

        score = compute_score(
            item, query_terms,
            boost_categories=["render", "compositing"],
            boost_keywords=["node", "shader", "input", "output"]
        )

        if score > 0:
            results.append({
                "title": item["title"],
                "path": item["path"],
                "headers": item.get("headers", []),
                "score": score,
                "source": item.get("source", "manual"),
                "snippet": create_snippet(item["content"], query_terms) if not compact else ""
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return format_results(results, node_type, limit, 0, compact)


@function_tool
def search_modifiers(
    modifier_name: Optional[str] = None,
    limit: int = DEFAULT_LIMIT,
    compact: bool = False
) -> str:
    """
    Search for modifier documentation.
    Find information about mesh, volume, and simulation modifiers.

    Args:
        modifier_name: Optional specific modifier (e.g., "Volume Displace", "Fluid")
                      If None, returns overview of all modifiers.
        limit: Maximum results (default: 5)
        compact: Minimal output mode

    Returns:
        Modifier documentation with settings and usage.

    Examples:
        - search_modifiers("Volume Displace")
        - search_modifiers("Fluid")
        - search_modifiers()  # List all modifiers
    """
    if not ensure_index_loaded():
        return "Error: Failed to load documentation index"

    query_terms = modifier_name.lower().split() if modifier_name else ["modifier"]
    results = []

    for item in _search_index:
        is_modifier_page = (
            "modifier" in item["path"] or
            "modifier" in item["title"].lower()
        )

        if not is_modifier_page:
            continue

        score = compute_score(
            item, query_terms,
            boost_categories=["modeling", "physics"],
            boost_keywords=["modifier", "generate", "deform"]
        )

        if score > 0:
            results.append({
                "title": item["title"],
                "path": item["path"],
                "headers": item.get("headers", []),
                "score": score,
                "source": item.get("source", "manual"),
                "snippet": create_snippet(item["content"], query_terms) if not compact else ""
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    query_str = modifier_name if modifier_name else "modifiers"
    return format_results(results, query_str, limit, 0, compact)


@function_tool
def read_page(
    path: str,
    max_length: int = MAX_PAGE_LENGTH,
    source: str = "auto"
) -> str:
    """
    Read the full content of a manual or API page with enhanced formatting.
    Use paths returned from search results.

    Args:
        path: Relative path to HTML file (e.g., 'render/volumetrics.html' or 'bpy.ops.mesh.html')
        max_length: Maximum characters to return (default: 4000, use 0 for unlimited)
        source: "manual", "python_api", or "auto" (tries both, default)

    Returns:
        Full page content with headers, paragraphs, code blocks, and lists.

    Examples:
        - read_page("render/cycles/world_settings.html")
        - read_page("bpy.ops.mesh.html")  # Python API doc
        - read_page("physics/fluid/type/domain/cache.html")
    """
    if BeautifulSoup is None:
        return "Error: BeautifulSoup not available - cannot parse HTML files"

    path = normalize_path(path)

    # Determine which directory to look in
    full_path = None
    detected_source = None

    if source == "python_api":
        full_path = PYTHON_API_DIR / path
        detected_source = "python_api"
    elif source == "manual":
        full_path = MANUAL_DIR / path
        detected_source = "manual"
    else:
        # Auto-detect
        if path.startswith("bpy.") or path.startswith("bmesh.") or path.startswith("aud.") or path.startswith("bl_"):
            full_path = PYTHON_API_DIR / path
            if full_path.exists():
                detected_source = "python_api"

        if full_path is None or not full_path.exists():
            full_path = MANUAL_DIR / path
            if full_path.exists():
                detected_source = "manual"

        if not full_path.exists():
            full_path = PYTHON_API_DIR / path
            if full_path.exists():
                detected_source = "python_api"

    # Try with backslashes
    if not full_path.exists():
        alt_path = path.replace('/', '\\')
        for base_dir, src in [(MANUAL_DIR, "manual"), (PYTHON_API_DIR, "python_api")]:
            test_path = base_dir / alt_path
            if test_path.exists():
                full_path = test_path
                detected_source = src
                break

    if not full_path.exists():
        return f"Error: File not found: {path}\n\nTip: Use search tools to find the correct path."

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        # Extract title
        title = "Untitled"
        if soup.title and soup.title.string:
            title_text = soup.title.string
            if detected_source == "python_api":
                title = title_text.replace("— Blender Python API", "").strip()
            else:
                title = title_text.replace("— Blender Manual", "").replace("— Blender 5.0 Manual", "").strip()

        # Find main content
        content = soup.find("article", role="main")
        if not content:
            content = soup.find("div", class_="document")
        if not content:
            content = soup.body

        if not content:
            return "Error: Could not parse page content."

        # Build formatted output
        source_label = "[Python API]" if detected_source == "python_api" else "[Manual]"
        output = f"# {title}\n**Source:** {source_label} | **Path:** `{path}`\n---\n\n"

        for element in content.children:
            if element.name == 'h1':
                output += f"# {element.get_text().strip().replace('¶', '')}\n\n"
            elif element.name == 'h2':
                output += f"## {element.get_text().strip().replace('¶', '')}\n\n"
            elif element.name == 'h3':
                output += f"### {element.get_text().strip().replace('¶', '')}\n\n"
            elif element.name == 'h4':
                output += f"#### {element.get_text().strip().replace('¶', '')}\n\n"
            elif element.name == 'p':
                text = element.get_text().strip()
                if text:
                    output += f"{text}\n\n"
            elif element.name == 'pre':
                code = element.get_text().strip()
                if code:
                    output += f"```python\n{code}\n```\n\n"
            elif element.name in ['ul', 'ol']:
                for li in element.find_all('li', recursive=False):
                    output += f"- {li.get_text().strip()}\n"
                output += "\n"
            elif element.name == 'dl':
                for dt in element.find_all('dt', recursive=False):
                    output += f"**{dt.get_text().strip()}**\n"
                for dd in element.find_all('dd', recursive=False):
                    output += f"  {dd.get_text().strip()}\n"
                output += "\n"

            if max_length > 0 and len(output) > max_length:
                output = output[:max_length]
                output += f"\n\n... [truncated at {max_length} chars - use max_length=0 for full content]"
                break

        # Fallback
        if len(output) < 200:
            output = f"# {title}\n\n"
            text = content.get_text("\n", strip=True)
            if max_length > 0:
                text = text[:max_length]
            output += text

        return output

    except Exception as e:
        return f"Error reading file: {e}"


@function_tool
def list_api_modules(
    category: Optional[str] = None,
    limit: int = 30
) -> str:
    """
    List available Python API modules from the official Blender Python API reference.

    Args:
        category: Optional filter by module category (e.g., "bpy", "bmesh", "aud")
        limit: Maximum modules to list (default: 30)

    Returns:
        List of available API modules with brief descriptions.

    Examples:
        - list_api_modules()  # List all top-level modules
        - list_api_modules("bpy")  # List all bpy.* modules
        - list_api_modules("bmesh")  # List all bmesh.* modules
    """
    if not ensure_index_loaded():
        return "Error: Failed to load documentation index"

    modules = []
    seen = set()

    for item in _search_index:
        if item.get("source") != "python_api":
            continue

        module_path = item.get("module_path", "")
        if not module_path:
            continue

        if category and not module_path.startswith(category):
            continue

        if module_path in seen:
            continue
        seen.add(module_path)

        modules.append({
            "module": module_path,
            "title": item["title"],
            "path": item["path"]
        })

    modules.sort(key=lambda x: x["module"])

    if not modules:
        return f"No API modules found" + (f" matching '{category}'" if category else "")

    output = "# Blender Python API Modules"
    if category:
        output += f" ({category}.*)"
    output += f"\n\nFound {len(modules)} modules (showing up to {limit}):\n\n"

    for mod in modules[:limit]:
        output += f"- **{mod['module']}** → `{mod['path']}`\n"

    if len(modules) > limit:
        output += f"\n[{len(modules) - limit} more modules - use higher limit or filter by category]"

    return output


@function_tool
def search_bpy_operators(
    category: str,
    operation: Optional[str] = None,
    limit: int = DEFAULT_LIMIT
) -> str:
    """
    Search for Blender Python operators (bpy.ops.*) by category.

    Args:
        category: Operator category (e.g., "mesh", "object", "fluid", "curve", "anim")
        operation: Optional specific operation name to search for
        limit: Maximum results (default: 5)

    Returns:
        List of operators in the category with descriptions.

    Examples:
        - search_bpy_operators("mesh")  # All mesh operators
        - search_bpy_operators("fluid", "bake")  # Fluid bake operators
        - search_bpy_operators("object", "select")  # Object selection operators
    """
    if not ensure_index_loaded():
        return "Error: Failed to load documentation index"

    target_module = f"bpy.ops.{category}"
    query_terms = [category.lower()]
    if operation:
        query_terms.append(operation.lower())

    results = []

    for item in _search_index:
        if item.get("source") != "python_api":
            continue

        module_path = item.get("module_path", "")
        if not module_path.startswith(target_module):
            continue

        score = 100

        if operation:
            if operation.lower() in item["content"].lower():
                score += 50
            if operation.lower() in item["title"].lower():
                score += 30

        results.append({
            "title": item["title"],
            "path": item["path"],
            "module": module_path,
            "headers": item.get("headers", [])[:3],
            "score": score,
            "snippet": item["content"][:150] + "..."
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    if not results:
        return f"No operators found for bpy.ops.{category}" + (f" matching '{operation}'" if operation else "")

    output = f"# bpy.ops.{category} Operators\n\n"
    output += f"Found {len(results)} operator modules:\n\n"

    for i, r in enumerate(results[:limit], 1):
        output += f"## {i}. {r['title']}\n"
        output += f"**Module:** `{r['module']}` | **Path:** `{r['path']}`\n"
        if r['headers']:
            output += f"**Contains:** {', '.join(r['headers'])}\n"
        output += f"{r['snippet']}\n\n"

    if len(results) > limit:
        output += f"[{len(results) - limit} more - increase limit to see all]"

    return output


@function_tool
def search_bpy_types(
    typename: str,
    limit: int = DEFAULT_LIMIT
) -> str:
    """
    Search for Blender Python types (bpy.types.*) by name.

    Args:
        typename: Type name to search for (e.g., "Object", "Mesh", "FluidModifier", "Volume")
        limit: Maximum results (default: 5)

    Returns:
        Matching types with their properties and methods.

    Examples:
        - search_bpy_types("Object")
        - search_bpy_types("FluidModifier")
        - search_bpy_types("Volume")
        - search_bpy_types("Particle")
    """
    if not ensure_index_loaded():
        return "Error: Failed to load documentation index"

    query_terms = typename.lower().split()
    results = []

    for item in _search_index:
        if item.get("source") != "python_api":
            continue

        module_path = item.get("module_path", "")
        if not (module_path.startswith("bpy.types") or "types" in module_path):
            continue

        score = 0
        title_lower = item["title"].lower()
        content_lower = item["content"].lower()

        for term in query_terms:
            if term in title_lower:
                score += 30
            if term in content_lower:
                score += min(content_lower.count(term), 10)

        if score > 0:
            results.append({
                "title": item["title"],
                "path": item["path"],
                "module": module_path,
                "headers": item.get("headers", [])[:5],
                "score": score,
                "snippet": create_snippet(item["content"], query_terms, 200)
            })

    results.sort(key=lambda x: x["score"], reverse=True)

    if not results:
        return f"No types found matching '{typename}'"

    output = f"# Blender Python Types matching '{typename}'\n\n"
    output += f"Found {len(results)} matching types:\n\n"

    for i, r in enumerate(results[:limit], 1):
        output += f"## {i}. {r['title']}\n"
        output += f"**Path:** `{r['path']}`\n"
        if r['headers']:
            output += f"**Properties/Methods:** {', '.join(r['headers'])}\n"
        output += f"{r['snippet']}\n\n"

    if len(results) > limit:
        output += f"[{len(results) - limit} more - increase limit to see all]"

    return output


@function_tool
def search_semantic(
    query: str,
    limit: int = DEFAULT_LIMIT,
    compact: bool = False
) -> str:
    """
    Semantic similarity search using AI embeddings.
    Finds conceptually related content even without exact keyword matches.

    Args:
        query: Natural language query describing what you're looking for
        limit: Maximum number of results (default: 5)
        compact: If True, returns minimal output

    Returns:
        Results ranked by semantic similarity to the query.

    Examples:
        - "how to make realistic smoke effects"
        - "rendering transparent volumetric clouds"
        - "baking fluid simulations for game engines"
        - "python script to automate VDB export"

    Note: Requires sentence-transformers package for semantic search.
    Falls back to keyword search if embeddings are not available.
    """
    if not ensure_index_loaded():
        return "Error: Failed to load documentation index"

    # Try to load embeddings if available
    global _embeddings, _embedding_model
    if _check_semantic_available() and _embeddings is None:
        if _load_embedding_model():
            _load_embeddings()

    # Try semantic search
    if _check_semantic_available() and _embeddings is not None and _embedding_model is not None:
        results = _semantic_search(query, top_k=limit * 2)

        if results:
            if compact:
                output = f"Found {len(results)} semantic matches for '{query}':\n\n"
                for i, r in enumerate(results[:limit], 1):
                    output += f"{i}. **{r['title']}** | `{r['path']}` | Sim: {r['similarity']:.3f}\n"
                return output

            output = f"Found {len(results)} semantic matches for '{query}':\n\n"
            for i, r in enumerate(results[:limit], 1):
                output += f"## {i}. {r['title']}\n"
                output += f"**Path:** `{r['path']}` | **Similarity:** {r['similarity']:.3f}\n"
                if r.get('headers'):
                    output += f"**Sections:** {', '.join(r['headers'][:3])}\n"
                output += f"{r.get('snippet', '')}\n\n"

            return output

    # Fallback to keyword search
    fallback_msg = ""
    if not _check_semantic_available():
        fallback_msg = "\n[Semantic search unavailable: install sentence-transformers]\n"
    elif _embeddings is None:
        fallback_msg = "\n[Embeddings not loaded - using keyword search]\n"

    query_terms = query.lower().split()
    results = []

    for item in _search_index:
        score = compute_score(item, query_terms)
        if score > 0:
            results.append({
                "title": item["title"],
                "path": item["path"],
                "headers": item.get("headers", []),
                "score": score,
                "snippet": create_snippet(item["content"], query_terms) if not compact else ""
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return fallback_msg + format_results(results, query, limit, 0, compact)
