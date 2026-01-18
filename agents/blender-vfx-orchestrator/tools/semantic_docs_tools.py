"""
Semantic Documentation Search Tools for Blender VFX Orchestrator.

Implements Strategy 1: Vector Store for Blender Documentation.

These tools provide semantic search over Blender 5.0 documentation using
OpenAI vector stores, enabling discovery of conceptually related content
even without exact keyword matches.

Key capabilities:
- Semantic search over Blender Manual and Python API Reference
- Alternative approach discovery for stuck situations
- Code example extraction from documentation
- API function discovery based on intent
"""

from __future__ import annotations

import os
import json
import sys
from typing import List, Optional

from agents import function_tool, RunContextWrapper

# Import SharedContext directly (not under TYPE_CHECKING) because
# @function_tool decorator evaluates type hints at runtime
from models.shared_context import SharedContext

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Vector store configuration
VECTOR_STORE_ID = os.getenv(
    "BLENDER_DOCS_VECTOR_STORE_ID",
    "vs_696acc41b74c8191a8d6f614c0223923"
)

# Cache for client
_client: Optional[OpenAI] = None


def _get_client() -> Optional[OpenAI]:
    """Get or create OpenAI client."""
    global _client
    if not OPENAI_AVAILABLE:
        return None
    if _client is None:
        _client = OpenAI()
    return _client


def _search_vector_store(
    query: str,
    max_results: int = 5,
    filter_category: Optional[str] = None
) -> List[dict]:
    """
    Internal function to search the vector store.

    Args:
        query: Search query
        max_results: Maximum results to return
        filter_category: Optional category filter (e.g., "physics", "render")

    Returns:
        List of search results with content and metadata
    """
    client = _get_client()
    if not client:
        return []

    try:
        # Use the file search tool with the vector store
        # Note: This uses the Responses API with file_search tool
        response = client.responses.create(
            model="gpt-5.2",  # Full reasoning capabilities
            input=query,
            tools=[{
                "type": "file_search",
                "vector_store_ids": [VECTOR_STORE_ID],
                "max_num_results": max_results * 2  # Get extra for filtering
            }],
            include=["file_search_call.results"]
        )

        results = []

        # Extract file search results
        for output in response.output:
            if hasattr(output, 'type') and output.type == 'file_search_call':
                if hasattr(output, 'results'):
                    for result in output.results:
                        # Result is a Pydantic model with attributes: text, score, file_id, filename
                        content = getattr(result, 'text', '') or ''
                        score = getattr(result, 'score', 0) or 0
                        file_id = getattr(result, 'file_id', '') or ''
                        filename = getattr(result, 'filename', 'unknown') or 'unknown'

                        # Apply category filter if specified
                        if filter_category:
                            if filter_category.lower() not in content.lower():
                                continue

                        results.append({
                            'content': content[:2000],  # Limit content size
                            'score': score,
                            'file_id': file_id,
                            'filename': filename,
                        })

                        if len(results) >= max_results:
                            break

        return results

    except Exception as e:
        print(f"[semantic_docs] Search error: {e}")
        return []


# =============================================================================
# FUNCTION TOOLS
# =============================================================================

@function_tool
def semantic_search_blender_docs(
    query: str,
    max_results: int = 5,
    include_code_examples: bool = True
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

    Returns:
        JSON with:
        - results_found: Number of results
        - results: List of relevant documentation chunks, each with:
            - content: The documentation text
            - score: Relevance score (0-1)
            - source: Source file name
        - code_snippets: Extracted Python code examples (if any found)
        - related_apis: List of bpy.types/bpy.ops references discovered
    """
    if not OPENAI_AVAILABLE:
        return json.dumps({
            "error": "OpenAI package not available",
            "results_found": 0,
            "results": [],
            "fallback": "Use keyword search via search_manual() instead"
        })

    # Enhance query for code if requested
    search_query = query
    if include_code_examples:
        search_query = f"{query} python code example bpy"

    # Search vector store
    results = _search_vector_store(search_query, max_results)

    if not results:
        return json.dumps({
            "error": "No results found",
            "results_found": 0,
            "results": [],
            "suggestion": f"Try broader terms or use keyword search for: {query}"
        })

    # Extract code snippets and API references
    code_snippets = []
    related_apis = set()

    for result in results:
        content = result.get('content', '')

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
                "source": r['filename']
            }
            for r in results
        ],
        "code_snippets": code_snippets[:5],  # Limit to 5 snippets
        "related_apis": list(related_apis)[:20]  # Limit to 20 APIs
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
