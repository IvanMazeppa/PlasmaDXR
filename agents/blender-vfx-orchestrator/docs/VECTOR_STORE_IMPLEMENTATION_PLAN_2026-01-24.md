# Vector Store Revamp - Implementation Plan

**Date:** 2026-01-24
**Status:** READY FOR IMPLEMENTATION
**Based On:** VECTOR_STORE_REVAMP_SPEC_2026-01-24.md
**SDK Version:** OpenAI Agents SDK v0.6.9+

---

## Executive Summary

This plan implements the vector store redesign for stable doc-grounding and guardrails in the Blender VFX multi-agent system. The key changes are:

1. **Split Stores** - Two separate vector stores (Manual + API)
2. **Stable Doc Refs** - Header-injected chunks with canonical `DocPath`
3. **Intent-Based Routing** - Tool router selects store based on query intent
4. **Local Fallback** - Offline search via `agents/blender-manual` when vector store fails
5. **Health Checks** - Verification script for coverage validation

---

## Phase 1: New Chunker Script

### File: `scripts/chunk_blender_docs.py`

**Purpose:** Replace `upload_blender_docs_to_vectorstore.py` with a section-aware chunker.

### Chunking Strategy

| Doc Type | Chunk Boundary | Target Size | Header Fields |
|----------|---------------|-------------|---------------|
| Manual | h2/h3 sections | 500-1200 words | DocType, DocPath, DocVersion, ChunkId |
| API | class/function defs | 300-800 words | DocType, DocPath, DocVersion, ChunkId |

### Header Format (Injected per chunk)

```markdown
# {Title}
DocType: manual|api
DocPath: {relative/path.html}#{section-anchor}
DocVersion: 5.0.1
SourceRepo: agents/blender-manual
ChunkId: {path}#{section}#{n}
---

{chunk content}
```

### Implementation

```python
"""
Blender Documentation Chunker for Vector Store Upload.

Creates header-injected chunks with stable DocPath references.
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from bs4 import BeautifulSoup
import json

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

    def to_upload_text(self) -> str:
        """Format chunk for vector store upload."""
        return f"""# {self.title}
DocType: {self.doc_type}
DocPath: {self.doc_path}
DocVersion: {self.doc_version}
ChunkId: {self.chunk_id}
---

{self.content}
"""

def chunk_manual_page(html_path: Path, base_path: Path) -> List[DocChunk]:
    """
    Chunk a manual page by h2/h3 sections.

    Target: 500-1200 words per chunk.
    """
    with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    relative_path = str(html_path.relative_to(base_path))
    chunks = []

    # Find all section headers
    headers = soup.find_all(['h2', 'h3'])

    if not headers:
        # No headers - treat entire page as one chunk
        text = soup.get_text(separator='\n', strip=True)
        if len(text.split()) >= 50:  # Minimum viable content
            chunks.append(DocChunk(
                title=soup.title.string if soup.title else html_path.stem,
                doc_type="manual",
                doc_path=relative_path,
                doc_version="5.0.1",
                chunk_id=f"{relative_path}#0",
                content=text[:6000],  # ~1200 words max
                word_count=len(text.split())
            ))
        return chunks

    # Chunk by section
    for i, header in enumerate(headers):
        section_id = header.get('id', f'section-{i}')
        title = header.get_text(strip=True)

        # Gather content until next header
        content_parts = []
        for sibling in header.find_next_siblings():
            if sibling.name in ['h2', 'h3']:
                break
            text = sibling.get_text(separator=' ', strip=True)
            if text:
                content_parts.append(text)

        content = '\n'.join(content_parts)
        word_count = len(content.split())

        # Skip very small sections
        if word_count < 30:
            continue

        # Split large sections
        if word_count > 1200:
            # Split into ~800 word chunks
            words = content.split()
            for j, start in enumerate(range(0, len(words), 800)):
                sub_content = ' '.join(words[start:start+800])
                chunks.append(DocChunk(
                    title=f"{title} (Part {j+1})",
                    doc_type="manual",
                    doc_path=f"{relative_path}#{section_id}",
                    doc_version="5.0.1",
                    chunk_id=f"{relative_path}#{section_id}#{j}",
                    content=sub_content,
                    word_count=len(sub_content.split())
                ))
        else:
            chunks.append(DocChunk(
                title=title,
                doc_type="manual",
                doc_path=f"{relative_path}#{section_id}",
                doc_version="5.0.1",
                chunk_id=f"{relative_path}#{section_id}#0",
                content=content,
                word_count=word_count
            ))

    return chunks

def chunk_api_page(html_path: Path, base_path: Path) -> List[DocChunk]:
    """
    Chunk an API reference page by class/function definitions.

    Target: 300-800 words per chunk.
    Preserves signatures and property lists.
    """
    with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    relative_path = str(html_path.relative_to(base_path))
    chunks = []

    # Find class and function definitions
    definitions = soup.find_all('dl', class_=['class', 'function', 'method', 'attribute'])

    if not definitions:
        # Fallback to h2 sections
        return chunk_manual_page(html_path, base_path)  # Reuse manual chunking

    for i, defn in enumerate(definitions):
        # Get the definition term (class/function name)
        dt = defn.find('dt')
        if not dt:
            continue

        name = dt.get_text(strip=True)[:100]  # Truncate long names
        api_id = dt.get('id', f'def-{i}')

        # Get the definition description
        dd = defn.find('dd')
        content = dd.get_text(separator='\n', strip=True) if dd else ""

        # Include the signature
        sig = dt.get_text(strip=True)
        full_content = f"```python\n{sig}\n```\n\n{content}"

        word_count = len(full_content.split())

        if word_count < 20:
            continue

        # Truncate very large definitions
        if word_count > 800:
            full_content = ' '.join(full_content.split()[:800]) + "\n[truncated]"
            word_count = 800

        chunks.append(DocChunk(
            title=name,
            doc_type="api",
            doc_path=f"{relative_path}#{api_id}",
            doc_version="5.0.1",
            chunk_id=f"{relative_path}#{api_id}#0",
            content=full_content,
            word_count=word_count
        ))

    return chunks
```

---

## Phase 2: Two-Store Upload Script

### File: `scripts/upload_chunked_docs.py`

**Environment Variables:**
```bash
BLENDER_MANUAL_VECTOR_STORE_ID=vs_manual_xxx  # Manual store
BLENDER_API_VECTOR_STORE_ID=vs_api_xxx        # API store
```

### Implementation

```python
"""
Upload chunked Blender docs to TWO separate vector stores.
"""

import os
import tempfile
from pathlib import Path
from openai import OpenAI
from chunk_blender_docs import chunk_manual_page, chunk_api_page, DocChunk

MANUAL_STORE_ID = os.getenv("BLENDER_MANUAL_VECTOR_STORE_ID")
API_STORE_ID = os.getenv("BLENDER_API_VECTOR_STORE_ID")

DOCS_BASE = Path("/home/maz3ppa/blender_documentation")
MANUAL_PATH = DOCS_BASE / "blender_manual_html"
API_PATH = DOCS_BASE / "blender_python_reference_5_0"

def upload_chunk(client: OpenAI, vector_store_id: str, chunk: DocChunk) -> bool:
    """Upload a single chunk to the specified vector store."""
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', delete=False, encoding='utf-8'
        ) as tmp:
            tmp.write(chunk.to_upload_text())
            tmp_path = tmp.name

        try:
            with open(tmp_path, 'rb') as f:
                file_response = client.files.create(file=f, purpose="assistants")

            client.vector_stores.files.create(
                vector_store_id=vector_store_id,
                file_id=file_response.id
            )
            return True
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        print(f"Error uploading {chunk.chunk_id}: {e}")
        return False

def main():
    client = OpenAI()

    # Chunk and upload Manual docs
    print("Processing Manual documentation...")
    manual_chunks = []
    for html_file in MANUAL_PATH.rglob("*.html"):
        chunks = chunk_manual_page(html_file, DOCS_BASE)
        manual_chunks.extend(chunks)

    print(f"Uploading {len(manual_chunks)} Manual chunks...")
    for i, chunk in enumerate(manual_chunks):
        upload_chunk(client, MANUAL_STORE_ID, chunk)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(manual_chunks)} uploaded")

    # Chunk and upload API docs
    print("\nProcessing API documentation...")
    api_chunks = []
    for html_file in API_PATH.rglob("*.html"):
        chunks = chunk_api_page(html_file, DOCS_BASE)
        api_chunks.extend(chunks)

    print(f"Uploading {len(api_chunks)} API chunks...")
    for i, chunk in enumerate(api_chunks):
        upload_chunk(client, API_STORE_ID, chunk)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(api_chunks)} uploaded")

    print(f"\nDone! Manual: {len(manual_chunks)}, API: {len(api_chunks)}")
```

---

## Phase 3: Intent-Based Tool Router

### File: `tools/semantic_docs_tools.py` (Updated)

**Key Changes:**
1. Add store selection logic based on query intent
2. Extract `DocPath` from chunk headers
3. Implement local fallback

### Store Selection Rules

| Query Pattern | Store |
|---------------|-------|
| Contains `bpy.` | API |
| Contains `bpy.types.` | API |
| Contains `bpy.ops.` | API |
| Contains class/function name | API |
| "Mantaflow", "volume", "shader", "workflow" | Manual |
| Effect descriptions | Manual |
| Default fallback | Manual first, then API |

### Implementation

```python
"""
Updated semantic_docs_tools.py with intent-based routing.
"""

import os
import json
from typing import List, Optional, Tuple

MANUAL_STORE_ID = os.getenv("BLENDER_MANUAL_VECTOR_STORE_ID")
API_STORE_ID = os.getenv("BLENDER_API_VECTOR_STORE_ID")
# Legacy single store (fallback)
LEGACY_STORE_ID = os.getenv("BLENDER_DOCS_VECTOR_STORE_ID")

def _select_store(query: str) -> Tuple[str, str]:
    """
    Select vector store based on query intent.

    Returns: (store_id, store_name)
    """
    query_lower = query.lower()

    # API indicators
    api_patterns = [
        'bpy.', 'bpy.types.', 'bpy.ops.', 'bpy.data.', 'bpy.context.',
        'class ', 'def ', 'method', 'attribute', 'property',
        'fluiddomain', 'fluidflow', 'fluideffector',
        '__init__', 'return type', 'parameters'
    ]

    # Manual indicators
    manual_patterns = [
        'mantaflow', 'volume', 'shader', 'material', 'workflow',
        'tutorial', 'how to', 'guide', 'step by step',
        'physics', 'simulation', 'render', 'animation',
        'principled', 'node', 'compositor', 'eevee', 'cycles'
    ]

    api_score = sum(1 for p in api_patterns if p in query_lower)
    manual_score = sum(1 for p in manual_patterns if p in query_lower)

    if api_score > manual_score and API_STORE_ID:
        return API_STORE_ID, "api"
    elif MANUAL_STORE_ID:
        return MANUAL_STORE_ID, "manual"
    else:
        # Legacy fallback
        return LEGACY_STORE_ID or "", "legacy"

def _extract_doc_path(content: str) -> Optional[str]:
    """
    Extract DocPath from chunk header.

    Looks for: DocPath: path/to/file.html#section
    """
    for line in content.split('\n')[:15]:  # Check first 15 lines
        if line.startswith('DocPath:'):
            return line.replace('DocPath:', '').strip()
    return None

def _search_with_fallback(
    query: str,
    max_results: int = 5,
    store_id: Optional[str] = None
) -> List[dict]:
    """
    Search vector store with local fallback.

    If vector store returns 0 results, falls back to local index.
    """
    client = _get_client()
    if not client:
        return _local_search(query, max_results)

    # Determine store
    if not store_id:
        store_id, _ = _select_store(query)

    if not store_id:
        return _local_search(query, max_results)

    # Vector store search
    results = _search_vector_store_impl(query, max_results, store_id)

    # Fallback if empty
    if not results:
        print(f"[semantic_docs] Vector store empty for '{query[:50]}...', using local fallback")
        return _local_search(query, max_results)

    # Extract stable doc_refs
    for r in results:
        doc_path = _extract_doc_path(r.get('content', ''))
        if doc_path:
            r['doc_ref'] = doc_path
        else:
            r['doc_ref'] = r.get('filename', 'unknown')

    return results

def _local_search(query: str, max_results: int = 5) -> List[dict]:
    """
    Fallback search using local blender-manual index.

    Uses simple keyword matching on local HTML files.
    """
    local_path = Path(__file__).parent.parent.parent / "blender-manual"
    if not local_path.exists():
        return []

    # Simple keyword search (could be enhanced with whoosh or similar)
    results = []
    keywords = query.lower().split()

    for html_file in local_path.rglob("*.html"):
        try:
            content = html_file.read_text(encoding='utf-8', errors='ignore')
            content_lower = content.lower()

            # Score by keyword matches
            score = sum(1 for kw in keywords if kw in content_lower)
            if score > 0:
                results.append({
                    'content': content[:2000],
                    'score': score / len(keywords),
                    'filename': str(html_file.relative_to(local_path)),
                    'doc_ref': str(html_file.relative_to(local_path)),
                    'source': 'local_fallback'
                })
        except Exception:
            continue

    # Sort by score and return top results
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:max_results]
```

---

## Phase 4: FileSearchTool Integration (SDK Native)

### Alternative: Use SDK's FileSearchTool directly

The OpenAI Agents SDK provides `FileSearchTool` for native vector store search. This can be used as an alternative to custom `_search_vector_store()` for agents that need direct access.

### Implementation for Research Agent

```python
from agents import Agent, FileSearchTool

def create_research_agent_with_file_search():
    """
    Create Research Agent with native FileSearchTool.

    Agents can directly search vector stores without custom tools.
    """
    return Agent(
        name="ResearchAgent",
        instructions=RESEARCH_INSTRUCTIONS,
        tools=[
            # Manual store for workflow/concept queries
            FileSearchTool(
                vector_store_ids=[MANUAL_STORE_ID],
                max_num_results=5,
                include_search_results=True,  # Include results in context
            ),
            # API store for code/function queries
            FileSearchTool(
                vector_store_ids=[API_STORE_ID],
                max_num_results=5,
                include_search_results=True,
            ),
            # Existing function tools
            semantic_search_blender_docs,
            find_alternative_approaches,
        ],
        output_type=ResearchOutput,
    )
```

### Trade-offs

| Approach | Pros | Cons |
|----------|------|------|
| **Custom function_tool** | Full control over doc_ref extraction, fallback logic | More code, manual maintenance |
| **SDK FileSearchTool** | Native, automatic, less code | Less control over result format, no fallback |

**Recommendation:** Use both - FileSearchTool for direct agent access, custom tools for orchestration code that needs stable doc_refs.

---

## Phase 5: Health Check Script

### File: `scripts/verify_vector_stores.py`

**Purpose:** Run verification queries and report coverage gaps.

### Implementation

```python
"""
Vector Store Health Check.

Runs standard queries and verifies non-zero results.
"""

import os
import json
from openai import OpenAI

MANUAL_STORE = os.getenv("BLENDER_MANUAL_VECTOR_STORE_ID")
API_STORE = os.getenv("BLENDER_API_VECTOR_STORE_ID")

# Test queries that MUST return results
MANUAL_TEST_QUERIES = [
    "Mantaflow domain settings",
    "Principled Volume temperature",
    "volume rendering blackbody",
    "fluid simulation cache",
    "smoke density dissolve",
]

API_TEST_QUERIES = [
    "bpy.types.FluidDomainSettings",
    "bpy.types.FluidFlowSettings",
    "bpy.ops.fluid.bake_all",
    "bpy.types.ShaderNodeVolumeAbsorption",
    "bpy.types.Object.modifiers",
]

def verify_store(client: OpenAI, store_id: str, queries: list, store_name: str) -> dict:
    """Run verification queries against a store."""
    results = {
        "store": store_name,
        "store_id": store_id,
        "queries_tested": len(queries),
        "queries_passed": 0,
        "queries_failed": [],
        "chunks_with_docpath": 0,
        "chunks_without_docpath": 0,
    }

    for query in queries:
        try:
            response = client.responses.create(
                model="gpt-4o-mini",  # Cheap model for verification
                input=query,
                tools=[{
                    "type": "file_search",
                    "vector_store_ids": [store_id],
                    "max_num_results": 3
                }],
                include=["file_search_call.results"]
            )

            found_results = False
            for output in response.output:
                if hasattr(output, 'type') and output.type == 'file_search_call':
                    if hasattr(output, 'results') and output.results:
                        found_results = True
                        for r in output.results:
                            content = getattr(r, 'text', '') or ''
                            if 'DocPath:' in content:
                                results["chunks_with_docpath"] += 1
                            else:
                                results["chunks_without_docpath"] += 1

            if found_results:
                results["queries_passed"] += 1
            else:
                results["queries_failed"].append(query)

        except Exception as e:
            results["queries_failed"].append(f"{query} (error: {e})")

    return results

def main():
    client = OpenAI()

    print("=" * 60)
    print("Vector Store Health Check")
    print("=" * 60)

    all_passed = True

    if MANUAL_STORE:
        print(f"\nManual Store: {MANUAL_STORE}")
        manual_results = verify_store(client, MANUAL_STORE, MANUAL_TEST_QUERIES, "manual")
        print(f"  Passed: {manual_results['queries_passed']}/{manual_results['queries_tested']}")
        print(f"  DocPath coverage: {manual_results['chunks_with_docpath']}/{manual_results['chunks_with_docpath'] + manual_results['chunks_without_docpath']}")
        if manual_results['queries_failed']:
            print(f"  FAILED queries: {manual_results['queries_failed']}")
            all_passed = False
    else:
        print("\nManual Store: NOT CONFIGURED")
        all_passed = False

    if API_STORE:
        print(f"\nAPI Store: {API_STORE}")
        api_results = verify_store(client, API_STORE, API_TEST_QUERIES, "api")
        print(f"  Passed: {api_results['queries_passed']}/{api_results['queries_tested']}")
        print(f"  DocPath coverage: {api_results['chunks_with_docpath']}/{api_results['chunks_with_docpath'] + api_results['chunks_without_docpath']}")
        if api_results['queries_failed']:
            print(f"  FAILED queries: {api_results['queries_failed']}")
            all_passed = False
    else:
        print("\nAPI Store: NOT CONFIGURED")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("HEALTH CHECK PASSED")
    else:
        print("HEALTH CHECK FAILED - Review failed queries above")
    print("=" * 60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
```

---

## Phase 6: Guardrail Integration

### Update: `guardrails/doc_grounding_guardrail.py`

**Purpose:** Validate that agent outputs include stable `doc_refs`.

```python
"""
Doc Grounding Guardrail.

Validates that outputs include stable doc_refs (not temp filenames).
"""

from agents import output_guardrail, GuardrailFunctionOutput, RunContextWrapper

@output_guardrail
async def validate_doc_refs(
    ctx: RunContextWrapper,
    agent,
    output
) -> GuardrailFunctionOutput:
    """
    Validate that output doc_refs are stable paths, not temp filenames.

    FAILS if:
    - doc_refs is empty when required
    - doc_refs contains temp filenames (tmp_*, *.md without path)
    - doc_refs contains 'doc_search_empty' or 'doc_search_unavailable'
    """
    doc_refs = getattr(output, 'doc_refs', [])

    if not doc_refs:
        return GuardrailFunctionOutput(
            output_info={"reason": "doc_refs is empty"},
            tripwire_triggered=True
        )

    invalid_refs = []
    for ref in doc_refs:
        # Check for temp filenames
        if ref.startswith('tmp_') or ref.startswith('/tmp/'):
            invalid_refs.append(ref)
        # Check for placeholder values
        if ref in ['doc_search_empty', 'doc_search_unavailable', 'unknown']:
            invalid_refs.append(ref)
        # Check for bare .md files (not chunked properly)
        if ref.endswith('.md') and '/' not in ref:
            invalid_refs.append(ref)

    if invalid_refs:
        return GuardrailFunctionOutput(
            output_info={
                "reason": "Invalid doc_refs detected",
                "invalid_refs": invalid_refs,
                "suggestion": "Ensure chunks have DocPath headers"
            },
            tripwire_triggered=True
        )

    return GuardrailFunctionOutput(
        output_info={"doc_refs_valid": True},
        tripwire_triggered=False
    )
```

---

## Migration Plan

### Step 1: Create New Vector Stores (One-time)

```bash
# Via OpenAI API or Dashboard
# Create: blender_manual_5_0 (Manual store)
# Create: blender_api_5_0 (API store)
```

### Step 2: Run Chunker and Upload

```bash
cd agents/blender-vfx-orchestrator

# Set environment
export BLENDER_MANUAL_VECTOR_STORE_ID=vs_manual_xxx
export BLENDER_API_VECTOR_STORE_ID=vs_api_xxx

# Run chunker + upload
python scripts/upload_chunked_docs.py
```

### Step 3: Verify Health

```bash
python scripts/verify_vector_stores.py
```

### Step 4: Update Environment

```bash
# .env update
BLENDER_MANUAL_VECTOR_STORE_ID=vs_manual_xxx
BLENDER_API_VECTOR_STORE_ID=vs_api_xxx
# Keep legacy for fallback
BLENDER_DOCS_VECTOR_STORE_ID=vs_696acc41b74c8191a8d6f614c0223923
```

### Step 5: Update Tools

1. Update `tools/semantic_docs_tools.py` with intent router
2. Add `validate_doc_refs` guardrail to Research Agent
3. Run 2-iteration shakedown test

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Doc hit rate | ~60% (estimated) | >95% |
| doc_refs stability | Temp filenames | Canonical paths |
| Manual vs API separation | Mixed | Clean |
| Guardrail false negatives | High | Low |
| Debugging ease | Hard (temp files) | Easy (real paths) |

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `scripts/chunk_blender_docs.py` | **CREATE** | Section-aware chunker |
| `scripts/upload_chunked_docs.py` | **CREATE** | Two-store uploader |
| `scripts/verify_vector_stores.py` | **CREATE** | Health check |
| `tools/semantic_docs_tools.py` | **MODIFY** | Intent router + fallback |
| `guardrails/doc_grounding_guardrail.py` | **CREATE** | doc_refs validation |
| `.env.example` | **MODIFY** | New env vars |

---

## Rollback Plan

If issues arise:
1. Revert to `BLENDER_DOCS_VECTOR_STORE_ID` (legacy single store)
2. Comment out intent router in `semantic_docs_tools.py`
3. Remove `validate_doc_refs` guardrail

Legacy store remains available as fallback.

---

*End of Implementation Plan*
