"""
Semantic Documentation Search Tools for Blender VFX Orchestrator.

Implements Strategy 1: Vector Store for Blender Documentation.

These tools provide semantic search over Blender 5.0 documentation using
OpenAI vector stores, enabling discovery of conceptually related content
even without exact keyword matches.

Key capabilities:
- Semantic search over Blender Manual and Python API Reference
- Intent-based routing between Manual and API stores
- Alternative approach discovery for stuck situations
- Code example extraction from documentation
- API function discovery based on intent
- Fallback to local blender-manual MCP when vector store fails

Architecture:
- MANUAL store: Conceptual docs (physics, fluid, render, tutorials)
- API store: Python API reference (bpy.types, bpy.ops, functions)

Each chunk has standardized headers:
    DocType: manual|api
    DocPath: relative/path/to/doc.html#section
    DocVersion: 5.0.1
    ChunkId: unique-chunk-identifier
"""

from __future__ import annotations

import os
import json
import re
import sys
from typing import List, Optional, Tuple

from agents import function_tool, RunContextWrapper

# Import SharedContext directly (not under TYPE_CHECKING) because
# @function_tool decorator evaluates type hints at runtime
from models.shared_context import SharedContext

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Two-store architecture: Separate Manual and API stores
MANUAL_STORE_ID = os.getenv(
    "BLENDER_MANUAL_VECTOR_STORE_ID",
    "vs_697564fbfc0c8191b2e44aa47dbaf482"  # New individual-file store (2026-01-25)
)
API_STORE_ID = os.getenv(
    "BLENDER_API_VECTOR_STORE_ID",
    "vs_697571f0275c8191910fea0f2c8bdd3a"  # New clean individual-file store (2026-01-25)
)

# Deprecated: Single store ID (kept for backwards compatibility)
VECTOR_STORE_ID = os.getenv(
    "BLENDER_DOCS_VECTOR_STORE_ID",
    API_STORE_ID  # Default to API store if old env var used
)

# Cache for client
_client: Optional[OpenAI] = None

# File search fallback controls (SDK tool path)
FILE_SEARCH_FALLBACK_ENABLED = os.getenv("DOCS_FILE_SEARCH_FALLBACK", "true").lower() in ("1", "true", "yes")
FILE_SEARCH_FALLBACK_MIN_SCORE = float(os.getenv("DOCS_FILE_SEARCH_MIN_SCORE", "0.15"))
DOCS_SEARCH_MODEL = os.getenv("DOCS_SEARCH_MODEL", "gpt-5.2")

# Keywords that indicate API vs Manual intent
API_KEYWORDS = {
    'bpy.', 'bpy.types', 'bpy.ops', 'bpy.data', 'bpy.context',
    'python', 'script', 'function', 'method', 'property', 'class',
    'api', 'reference', 'parameter', 'return', 'type:', 'example code',
    'FluidDomainSettings', 'FluidModifier', 'MaterialSlot', 'Object',
    'Operator', 'Panel', 'bl_', '__init__', 'execute(', 'invoke(',
}
MANUAL_KEYWORDS = {
    'how to', 'tutorial', 'guide', 'workflow', 'physics',
    'simulation', 'render', 'compositor', 'modeling', 'sculpt',
    'animation', 'rigging', 'texture', 'material', 'node',
    'smoke', 'fire', 'fluid', 'domain', 'effect', 'volumetric',
    'bake', 'cache', 'resolution', 'subdivisions', 'settings',
}


def _get_client() -> Optional[OpenAI]:
    """Get or create OpenAI client."""
    global _client
    if not OPENAI_AVAILABLE:
        return None
    if _client is None:
        _client = OpenAI()
    return _client


def _classify_query_intent(query: str) -> Tuple[str, float]:
    """
    Classify query as 'api', 'manual', or 'both' based on keywords.

    Returns:
        Tuple of (intent, confidence) where confidence is 0.0-1.0
    """
    query_lower = query.lower()

    api_score = sum(1 for kw in API_KEYWORDS if kw.lower() in query_lower)
    manual_score = sum(1 for kw in MANUAL_KEYWORDS if kw.lower() in query_lower)

    # Explicit API references get strong boost
    if 'bpy.' in query_lower or 'bpy.types.' in query_lower:
        api_score += 5

    total = api_score + manual_score
    if total == 0:
        return 'both', 0.5

    api_ratio = api_score / total
    if api_ratio > 0.7:
        return 'api', api_ratio
    elif api_ratio < 0.3:
        return 'manual', 1.0 - api_ratio
    else:
        return 'both', 0.5


def _extract_doc_path(content: str) -> Optional[str]:
    """Extract DocPath from chunk content header."""
    for line in content.split('\n')[:15]:  # Check first 15 lines
        if line.startswith('DocPath:'):
            return line.replace('DocPath:', '').strip()
    return None


def _extract_doc_type(content: str) -> Optional[str]:
    """Extract DocType from chunk content header."""
    for line in content.split('\n')[:15]:
        if line.startswith('DocType:'):
            return line.replace('DocType:', '').strip()
    return None


def _search_vector_store(
    query: str,
    max_results: int = 5,
    filter_category: Optional[str] = None,
    store_id: Optional[str] = None,
    intent: Optional[str] = None
) -> List[dict]:
    """
    Internal function to search the vector store.

    Args:
        query: Search query
        max_results: Maximum results to return
        filter_category: Optional category filter (e.g., "physics", "render")
        store_id: Specific store ID to search (overrides intent routing)
        intent: Force 'api', 'manual', or 'both' (auto-detected if None)

    Returns:
        List of search results with content, metadata, and doc_path
    """
    client = _get_client()
    if not client:
        return []

    # Determine which store(s) to search
    if store_id:
        store_ids = [store_id]
    elif intent == 'api':
        store_ids = [API_STORE_ID]
    elif intent == 'manual':
        store_ids = [MANUAL_STORE_ID]
    elif intent == 'both':
        store_ids = [API_STORE_ID, MANUAL_STORE_ID]
    else:
        # Auto-detect intent
        detected_intent, confidence = _classify_query_intent(query)
        if detected_intent == 'api':
            store_ids = [API_STORE_ID]
        elif detected_intent == 'manual':
            store_ids = [MANUAL_STORE_ID]
        else:
            # Search both stores and merge results
            store_ids = [API_STORE_ID, MANUAL_STORE_ID]

    all_results = []

    for sid in store_ids:
        try:
            # Use the vector_stores.search API (correct API for direct search)
            response = client.vector_stores.search(
                vector_store_id=sid,
                query=query,
                max_num_results=max_results
            )

            for result in response.data:
                # Extract content from result
                content = ''
                if result.content:
                    content = result.content[0].text if result.content else ''

                # Extract DocPath from content header
                doc_path = _extract_doc_path(content)
                doc_type = _extract_doc_type(content)

                # Apply category filter if specified
                if filter_category:
                    if filter_category.lower() not in content.lower():
                        continue

                all_results.append({
                    'content': content[:2000],  # Limit content size
                    'score': result.score or 0,
                    'file_id': result.file_id or '',
                    'filename': result.filename or 'unknown',
                    'doc_path': doc_path,
                    'doc_type': doc_type,
                    'store_id': sid,
                })

        except Exception as e:
            print(f"[semantic_docs] Search error on {sid}: {e}", file=sys.stderr)
            continue

    # Fallback to SDK file_search if results are empty or weak
    top_score = all_results[0].get("score", 0) if all_results else 0
    if FILE_SEARCH_FALLBACK_ENABLED and (not all_results or top_score < FILE_SEARCH_FALLBACK_MIN_SCORE):
        file_search_results = _file_search_vector_store(
            query=query,
            store_ids=store_ids,
            max_results=max_results,
            filter_category=filter_category
        )
        if file_search_results:
            all_results = _merge_results(all_results, file_search_results, max_results)

    # Sort by score and limit
    all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
    return all_results[:max_results]


def _file_search_vector_store(
    query: str,
    store_ids: List[str],
    max_results: int,
    filter_category: Optional[str]
) -> List[dict]:
    """
    Use the SDK file_search tool as a fallback retrieval path.
    """
    client = _get_client()
    if not client:
        return []

    try:
        response = client.responses.create(
            model=DOCS_SEARCH_MODEL,
            input=query,
            tools=[{
                "type": "file_search",
                "vector_store_ids": store_ids,
                "max_num_results": max_results * 2
            }],
            include=["file_search_call.results"]
        )
    except Exception as e:
        print(f"[semantic_docs] file_search error: {e}", file=sys.stderr)
        return []

    results: List[dict] = []
    for output in response.output:
        if getattr(output, "type", "") != "file_search_call":
            continue
        for result in getattr(output, "results", []) or []:
            content = getattr(result, "text", "") or ""
            if filter_category and filter_category.lower() not in content.lower():
                continue
            doc_path = _extract_doc_path(content)
            doc_type = _extract_doc_type(content)
            results.append({
                "content": content[:2000],
                "score": getattr(result, "score", 0) or 0,
                "file_id": getattr(result, "file_id", "") or "",
                "filename": getattr(result, "filename", "unknown") or "unknown",
                "doc_path": doc_path,
                "doc_type": doc_type,
                "store_id": "file_search",
            })

    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return results[:max_results]


def _merge_results(primary: List[dict], secondary: List[dict], max_results: int) -> List[dict]:
    """
    Merge and de-duplicate results from multiple retrieval paths.
    """
    merged = []
    seen = set()

    for item in primary + secondary:
        key = (
            item.get("file_id", ""),
            item.get("filename", ""),
            (item.get("content", "") or "")[:120],
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)

    merged.sort(key=lambda x: x.get("score", 0), reverse=True)
    return merged[:max_results]


async def _fallback_to_blender_manual(query: str, max_results: int = 5) -> List[dict]:
    """
    Fallback to local blender-manual MCP when vector store fails.

    This provides resilience when:
    - Vector store is unavailable
    - Search returns zero results
    - Rate limits are hit
    """
    # Note: This requires the blender-manual MCP to be available
    # For now, return empty list - will be implemented when MCP is integrated
    return []


# =============================================================================
# FUNCTION TOOLS
# =============================================================================

@function_tool
def semantic_search_blender_docs(
    query: str,
    max_results: int = 5,
    include_code_examples: bool = True,
    intent: str = ""
) -> str:
    """
    Search Blender documentation semantically using AI embeddings.

    Unlike keyword search, this finds conceptually related documentation
    even when exact terms don't match. For example, searching "turbulence"
    will also find documentation about "vorticity", "noise_strength", etc.

    WHEN TO USE:
    - Finding alternative approaches to a problem
    - Discovering related API functions you didn't know existed
    - Finding code examples for specific effects
    - Exploring documentation when keyword search fails

    Args:
        query: Natural language description of what you're looking for
               Examples:
               - "how to make smoke rise faster and dissipate"
               - "controlling fluid simulation turbulence"
               - "exporting volumetric data to game engines"
        max_results: Maximum number of documentation chunks to return (1-10)
        include_code_examples: If True, prioritizes results with Python code
        intent: Force routing to 'api', 'manual', or 'both' (auto-detected if empty)

    Returns:
        JSON with:
        - results_found: Number of results
        - results: List of relevant documentation chunks, each with:
            - content: The documentation text
            - score: Relevance score (0-1)
            - source: Source file name
            - doc_path: Stable document path (e.g., physics/fluid/domain.html#settings)
            - doc_type: manual or api
        - code_snippets: Extracted Python code examples (if any found)
        - related_apis: List of bpy.types/bpy.ops references discovered
        - doc_refs: List of stable document paths for citations
    """
    if not OPENAI_AVAILABLE:
        return json.dumps({
            "error": "OpenAI package not available",
            "results_found": 0,
            "results": [],
            "doc_refs": [],
            "fallback": "Use keyword search via search_manual() instead"
        })

    # Enhance query for code if requested
    search_query = query
    if include_code_examples:
        search_query = f"{query} python code example bpy"

    # Search vector store with intent routing
    search_intent = intent if intent in ('api', 'manual', 'both') else None
    results = _search_vector_store(search_query, max_results, intent=search_intent)

    if not results:
        return json.dumps({
            "error": "No results found",
            "results_found": 0,
            "results": [],
            "doc_refs": [],
            "suggestion": f"Try broader terms or use keyword search for: {query}"
        })

    # Extract code snippets and API references
    code_snippets = []
    related_apis = set()
    doc_refs = []

    for result in results:
        content = result.get('content', '')
        doc_path = result.get('doc_path')
        if doc_path and doc_path not in doc_refs:
            doc_refs.append(doc_path)

        # Extract code blocks
        if '```' in content:
            parts = content.split('```')
            for i, part in enumerate(parts):
                if i % 2 == 1:  # Odd indices are code blocks
                    code_snippets.append(part.strip()[:500])  # Limit size

        # Extract API references
        for api_type in ['bpy.types.', 'bpy.ops.', 'bpy.data.', 'bpy.context.']:
            idx = 0
            while True:
                idx = content.find(api_type, idx)
                if idx == -1:
                    break
                # Extract the full API path
                end_idx = idx + len(api_type)
                while end_idx < len(content) and (content[end_idx].isalnum() or content[end_idx] in '._'):
                    end_idx += 1
                api_ref = content[idx:end_idx]
                if len(api_ref) > len(api_type) + 2:  # Has actual content
                    related_apis.add(api_ref)
                idx = end_idx

    return json.dumps({
        "query": query,
        "results_found": len(results),
        "results": [
            {
                "content": r['content'],
                "score": r['score'],
                "source": r['filename'],
                "doc_path": r.get('doc_path', ''),
                "doc_type": r.get('doc_type', 'unknown'),
            }
            for r in results
        ],
        "code_snippets": code_snippets[:5],  # Limit to 5 snippets
        "related_apis": list(related_apis)[:20],  # Limit to 20 APIs
        "doc_refs": doc_refs[:10],  # Stable document paths for citations
    })


@function_tool
def blender_doc_search_bundle(
    effect_type: str,
    description: str = "",
    intent: str = "",
    domain: str = "Mantaflow",
    max_results: int = 6
) -> str:
    """
    Multi-query Blender 5.0 doc search with internal fallbacks.

    This reduces tool-call count by batching multiple queries into ONE tool call.
    Use when you need robust doc grounding without hitting turn limits.

    Args:
        effect_type: Effect type (fire, explosion, smoke, etc.)
        description: Natural language request/goal
        intent: Specific intent to search (optional)
        domain: Domain focus (default: Mantaflow)
        max_results: Max results to return (1-10)

    Returns:
        JSON with:
        - queries_used: list of queries executed
        - results_found: number of results
        - results: list of doc chunks (content, score, source, query)
        - code_snippets: extracted Python snippets (if any)
        - related_apis: bpy.* references discovered
        - doc_refs: list of source filenames
        - warnings: list of warnings (if any)
        - diagnostics: status info
    """
    if not OPENAI_AVAILABLE:
        return json.dumps({
            "error": "OpenAI package not available",
            "results_found": 0,
            "results": [],
            "doc_refs": ["doc_search_unavailable"],
            "warnings": ["OpenAI package missing - cannot access vector store"],
            "diagnostics": {
                "openai_available": False,
                "manual_store_id": MANUAL_STORE_ID,
                "api_store_id": API_STORE_ID,
            },
        })

    queries = []
    if description:
        queries.append(description)
    if effect_type:
        queries.append(f"{effect_type} {domain} simulation Blender 5.0")
        queries.append(f"{effect_type} smoke fire pyro mantaflow settings")
        queries.append(f"{effect_type} volumetric shader principled volume")
    if intent:
        queries.append(f"{intent} bpy python")

    queries.append(f"{domain} cache settings bpy.types.FluidDomainSettings")
    queries.append("Blender 5.0 Mantaflow domain settings")

    # De-duplicate while preserving order
    queries = [q.strip() for q in queries if q and q.strip()]
    seen_queries = set()
    ordered_queries = []
    for q in queries:
        if q.lower() in seen_queries:
            continue
        seen_queries.add(q.lower())
        ordered_queries.append(q)

    results = []
    seen_results = set()
    queries_used = []

    for q in ordered_queries:
        batch = _search_vector_store(q, max_results=max_results)
        queries_used.append(q)
        for r in batch:
            key = (
                r.get("file_id", ""),
                r.get("filename", ""),
                (r.get("content", "") or "")[:120],
            )
            if key in seen_results:
                continue
            seen_results.add(key)
            r["query"] = q
            results.append(r)
        if len(results) >= max_results:
            break

    # Final fallback if nothing found
    warnings = []
    if not results:
        warnings.append("No doc results from bundled queries. Check vector store ID or coverage.")
        fallback_queries = [
            "bpy.types.FluidDomainSettings",
            "bpy.ops.fluid.bake",
            "Blender 5.0 manual fluid simulation",
        ]
        for q in fallback_queries:
            batch = _search_vector_store(q, max_results=max_results)
            queries_used.append(q)
            for r in batch:
                key = (
                    r.get("file_id", ""),
                    r.get("filename", ""),
                    (r.get("content", "") or "")[:120],
                )
                if key in seen_results:
                    continue
                seen_results.add(key)
                r["query"] = q
                results.append(r)
            if results:
                break

    # Extract code snippets and API references
    code_snippets = []
    related_apis = set()
    extracted_doc_paths = []
    for result in results:
        content = result.get("content", "") or ""
        if "```" in content:
            parts = content.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    code_snippets.append(part.strip()[:500])
        for api_type in ["bpy.types.", "bpy.ops.", "bpy.data.", "bpy.context."]:
            idx = 0
            while True:
                idx = content.find(api_type, idx)
                if idx == -1:
                    break
                end_idx = idx + len(api_type)
                while end_idx < len(content) and (content[end_idx].isalnum() or content[end_idx] in "._"):
                    end_idx += 1
                api_ref = content[idx:end_idx]
                if len(api_ref) > len(api_type) + 2:
                    related_apis.add(api_ref)
                idx = end_idx
        # Extract DocPath from result (already parsed in search)
        doc_path = result.get("doc_path") or ""
        if doc_path and doc_path not in extracted_doc_paths:
            extracted_doc_paths.append(doc_path)
        # Fallback: try to extract from content header
        if not doc_path:
            for line in content.splitlines()[:15]:
                if line.startswith("DocPath:"):
                    doc_path = line.replace("DocPath:", "").strip()
                    if doc_path and doc_path not in extracted_doc_paths:
                        extracted_doc_paths.append(doc_path)
                    break

    doc_refs = []
    for doc_path in extracted_doc_paths:
        if doc_path not in doc_refs:
            doc_refs.append(doc_path)
    for r in results:
        filename = r.get("filename", "") or ""
        if filename and filename not in doc_refs:
            doc_refs.append(filename)

    if not doc_refs:
        doc_refs = ["doc_search_empty"]

    return json.dumps({
        "effect_type": effect_type,
        "queries_used": queries_used,
        "results_found": len(results),
        "results": [
            {
                "content": r.get("content", ""),
                "score": r.get("score", 0),
                "source": r.get("filename", "unknown"),
                "doc_path": next(
                    (p for p in extracted_doc_paths if p in (r.get("content", "") or "")),
                    ""
                ),
                "query": r.get("query", ""),
            }
            for r in results[:max_results]
        ],
        "code_snippets": code_snippets[:5],
        "related_apis": list(related_apis)[:20],
        "doc_refs": doc_refs[:10],
        "warnings": warnings,
        "diagnostics": {
            "openai_available": True,
            "manual_store_id": MANUAL_STORE_ID,
                "api_store_id": API_STORE_ID,
            "queries_attempted": len(queries_used),
        },
    })


@function_tool
def find_alternative_approaches(
    wrapper: RunContextWrapper[SharedContext],
    current_approach: str,
    issue: str,
    effect_type: str = "",
    exclude_techniques: str = "[]"
) -> str:
    """
    Search for fundamentally DIFFERENT approaches to solve a VFX issue.

    Use when the current approach has plateaued and you need something
    genuinely different, not just parameter variations.

    WHEN TO USE:
    - At escape_level >= 2 (SWITCH_TECHNIQUE or higher)
    - When same issue persists for 2+ iterations
    - When quality score has plateaued

    The search explicitly excludes the current approach and any
    techniques you've already tried.

    NOTE: With RunContextWrapper, effect_type and exclude_techniques are auto-populated from context.

    Args:
        wrapper: RunContextWrapper with SharedContext (auto-injected by SDK)
        current_approach: What you're currently doing (will be excluded)
                         Example: "increasing flame_smoke parameter"
        issue: The problem you're trying to solve
               Example: "smoke lacks density"
        effect_type: Type of effect (auto-populated from context if empty)
        exclude_techniques: JSON array of techniques to exclude (auto-populated from context)

    Returns:
        JSON with:
        - alternatives_found: Number of alternatives discovered
        - alternatives: List of alternative approaches, each with:
            - approach_name: Short description
            - description: How this approach works
            - code_hints: Relevant API functions or code snippets
            - confidence: Estimated likelihood of helping (0-100)
            - trade_offs: Pros/cons vs current approach
    """
    # Auto-populate from context if not provided
    context = wrapper.context
    if context and hasattr(context, 'session'):
        session = context.session
        if not effect_type and session.request:
            effect_type = session.request.effect_type.value
        if exclude_techniques == "[]" and session.stuck_state:
            exclude = list(session.stuck_state.techniques_tried)
        else:
            try:
                exclude = json.loads(exclude_techniques) if exclude_techniques else []
            except json.JSONDecodeError:
                exclude = []
        print(f"[find_alternative_approaches] Using context: effect_type={effect_type}, exclude={exclude}", file=sys.stderr)
    else:
        try:
            exclude = json.loads(exclude_techniques) if exclude_techniques else []
        except json.JSONDecodeError:
            exclude = []

    if not OPENAI_AVAILABLE:
        return json.dumps({
            "error": "OpenAI package not available",
            "alternatives_found": 0,
            "alternatives": [],
            "fallback": "Use search_alternative_approaches() from proactive_research_tools"
        })

    # Build a query that explicitly seeks alternatives
    exclusion_text = f"NOT {current_approach}"
    if exclude:
        exclusion_text += " NOT " + " NOT ".join(exclude)

    query = f"""
    Blender 5.0 {effect_type} effect alternative solution:

    Problem to solve: {issue}

    Current approach (NOT working): {current_approach}

    I need a DIFFERENT approach that:
    - Uses different Blender API functions
    - Has different parameters to adjust
    - Takes a fundamentally different strategy

    {exclusion_text}
    """

    # Search for alternatives
    results = _search_vector_store(query, max_results=8)

    if not results:
        return json.dumps({
            "issue": issue,
            "alternatives_found": 0,
            "alternatives": [],
            "suggestion": "No semantic matches. Try mining specific Blender API docs."
        })

    # Process results into alternatives
    alternatives = []
    seen_approaches = set()

    for result in results:
        content = result.get('content', '')

        # Skip if matches current approach or excluded techniques
        content_lower = content.lower()
        if current_approach.lower() in content_lower:
            continue
        if any(exc.lower() in content_lower for exc in exclude):
            continue

        # Extract a meaningful approach name
        lines = content.split('\n')
        approach_name = "Unknown approach"
        for line in lines:
            if line.startswith('## ') or line.startswith('# '):
                approach_name = line.lstrip('#').strip()[:100]
                break

        # Skip duplicates
        if approach_name.lower() in seen_approaches:
            continue
        seen_approaches.add(approach_name.lower())

        # Extract code hints (bpy references)
        code_hints = []
        for api_type in ['bpy.types.', 'bpy.ops.', 'bpy.data.']:
            idx = content.find(api_type)
            if idx != -1:
                end_idx = idx
                while end_idx < len(content) and (content[end_idx].isalnum() or content[end_idx] in '._'):
                    end_idx += 1
                code_hints.append(content[idx:end_idx])

        # Estimate confidence based on score and relevance
        base_confidence = int(result.get('score', 0.5) * 100)
        # Boost if contains code examples
        if '```' in content or 'bpy.' in content:
            base_confidence = min(100, base_confidence + 15)

        alternatives.append({
            "approach_name": approach_name,
            "description": content[:500],  # Truncate description
            "code_hints": list(set(code_hints))[:5],
            "confidence": base_confidence,
            "trade_offs": "Requires testing; may have different performance characteristics",
            "source": result.get('filename', 'unknown')
        })

        if len(alternatives) >= 5:
            break

    return json.dumps({
        "issue": issue,
        "current_approach": current_approach,
        "alternatives_found": len(alternatives),
        "alternatives": alternatives,
        "techniques_excluded": exclude
    })


@function_tool
def search_blender_api_by_intent(
    intent: str,
    domain: str = "fluid"
) -> str:
    """
    Find Blender Python API functions based on what you want to accomplish.

    Use when you know WHAT you want to do but don't know WHICH API to use.

    Args:
        intent: What you want to accomplish
                Examples:
                - "increase smoke density over time"
                - "export simulation as VDB"
                - "add turbulence to fluid"
        domain: Domain to focus search (fluid, render, physics, animation, etc.)

    Returns:
        JSON with:
        - apis_found: Number of relevant APIs discovered
        - apis: List of API references with:
            - api_path: Full API path (e.g., bpy.types.FluidDomainSettings)
            - purpose: What this API does
            - relevant_properties: Key properties/methods
            - usage_example: Code example if available
    """
    if not OPENAI_AVAILABLE:
        return json.dumps({
            "error": "OpenAI package not available",
            "apis_found": 0,
            "apis": []
        })

    query = f"""
    Blender Python API for {domain}:
    Intent: {intent}

    bpy.types bpy.ops API reference properties methods
    """

    results = _search_vector_store(query, max_results=6)

    if not results:
        return json.dumps({
            "intent": intent,
            "apis_found": 0,
            "apis": [],
            "suggestion": f"Try search_bpy_types or search_bpy_operators for {domain}"
        })

    # Extract API information
    apis = []
    seen_apis = set()

    for result in results:
        content = result.get('content', '')

        # Find all API references
        for api_type in ['bpy.types.', 'bpy.ops.']:
            idx = 0
            while True:
                idx = content.find(api_type, idx)
                if idx == -1:
                    break

                # Extract full API path
                end_idx = idx
                while end_idx < len(content) and (content[end_idx].isalnum() or content[end_idx] in '._'):
                    end_idx += 1
                api_path = content[idx:end_idx]

                if api_path in seen_apis or len(api_path) < 15:
                    idx = end_idx
                    continue
                seen_apis.add(api_path)

                # Extract surrounding context
                context_start = max(0, idx - 200)
                context_end = min(len(content), end_idx + 300)
                context = content[context_start:context_end]

                # Try to find properties mentioned nearby
                properties = []
                for prop_indicator in ['property', 'attribute', '= ', ': ']:
                    prop_idx = context.find(prop_indicator)
                    if prop_idx != -1:
                        # Extract property name
                        word_start = prop_idx + len(prop_indicator)
                        word_end = word_start
                        while word_end < len(context) and (context[word_end].isalnum() or context[word_end] == '_'):
                            word_end += 1
                        prop = context[word_start:word_end]
                        if prop and len(prop) > 2:
                            properties.append(prop)

                apis.append({
                    "api_path": api_path,
                    "purpose": context[:200].replace('\n', ' '),
                    "relevant_properties": list(set(properties))[:5],
                    "source": result.get('filename', 'unknown')
                })

                idx = end_idx

                if len(apis) >= 10:
                    break

        if len(apis) >= 10:
            break

    return json.dumps({
        "intent": intent,
        "domain": domain,
        "apis_found": len(apis),
        "apis": apis
    })


# =============================================================================
# INTERNAL CALLABLE VERSIONS (for non-agent code)
# =============================================================================

def semantic_search_impl(query: str, max_results: int = 5) -> str:
    """
    Direct callable version of semantic search.

    Use this from non-agent code (e.g., proactive_research_tools).
    """
    if not OPENAI_AVAILABLE:
        return json.dumps({
            "error": "OpenAI not available",
            "results_found": 0,
            "results": []
        })

    results = _search_vector_store(query, max_results)

    return json.dumps({
        "query": query,
        "results_found": len(results),
        "results": [
            {
                "content": r['content'],
                "score": r['score'],
                "source": r['filename']
            }
            for r in results
        ]
    })


def find_alternatives_impl(
    issue: str,
    current_approach: str,
    effect_type: str,
    exclude: List[str] = None
) -> str:
    """
    Direct callable version of find_alternative_approaches.

    Use this from non-agent code.
    """
    exclude = exclude or []
    exclude_json = json.dumps(exclude)

    # Call the internal search logic directly
    if not OPENAI_AVAILABLE:
        return json.dumps({
            "error": "OpenAI not available",
            "alternatives_found": 0,
            "alternatives": []
        })

    # Build query
    query = f"""
    Blender {effect_type} alternative for {issue}
    NOT {current_approach}
    different approach solution
    """

    results = _search_vector_store(query, max_results=5)

    alternatives = []
    for r in results:
        content = r.get('content', '')
        if current_approach.lower() not in content.lower():
            alternatives.append({
                "approach_name": r.get('filename', 'unknown'),
                "description": content[:300],
                "confidence": int(r.get('score', 0.5) * 100)
            })

    return json.dumps({
        "issue": issue,
        "alternatives_found": len(alternatives),
        "alternatives": alternatives
    })
