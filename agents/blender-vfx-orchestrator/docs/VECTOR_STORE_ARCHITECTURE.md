# Blender Documentation Vector Store Architecture

**Version:** 2.0 (2026-01-25)
**Status:** Production

This document describes the optimized vector store architecture for semantic search over Blender 5.0 documentation.

---

## Overview

The VFX Orchestrator uses OpenAI Vector Stores to provide semantic search capabilities over Blender documentation. This enables agents to find relevant documentation for script generation, error resolution, and technique research.

### Architecture Comparison

| Aspect | Original (v1) | Optimized (v2) |
|--------|---------------|----------------|
| **Store Count** | 1 combined | 2 split (Manual + API) |
| **Total Size** | 819.5 MB | 36.7 MB |
| **File Count** | 4,213 | 17,634 |
| **Avg File Size** | ~68 KB | ~2 KB |
| **Chunking** | None (full pages) | Section-aware (~500-1000 tokens) |
| **Query Routing** | None | Intent-based auto-routing |

---

## Store Configuration

### Production Store IDs

```bash
# Environment variables
BLENDER_MANUAL_VECTOR_STORE_ID=vs_697564fbfc0c8191b2e44aa47dbaf482
BLENDER_API_VECTOR_STORE_ID=vs_697571f0275c8191910fea0f2c8bdd3a
```

### Store Details

| Store | ID | Files | Size | Content |
|-------|-----|-------|------|---------|
| **Manual** | `vs_697564fbfc0c8191b2e44aa47dbaf482` | 4,400 | 10.1 MB | User guide, tutorials, UI reference |
| **API** | `vs_697571f0275c8191910fea0f2c8bdd3a` | 13,234 | 26.6 MB | Python API reference (bpy.types, bpy.ops, etc.) |

### Deprecated Stores

| Store | ID | Status | Notes |
|-------|-----|--------|-------|
| Original Combined | `vs_696acc41b74c8191a8d6f614c0223923` | Deprecated | 819 MB, monolithic files |
| Bundled Manual | `vs_6975104199c08191acb1495c86d581ce` | Deprecated | 100-chunk bundles, less precise |
| Incomplete API | `vs_697512bf81c481919ae3b7a8ffb8223a` | Deprecated | Partial upload (crashed at 34%) |

---

## Chunking Strategy

### Why Chunk?

The original approach uploaded entire HTML pages as single files (~68KB average). This caused:
- **Excessive token usage** - Searches returned large irrelevant sections
- **Poor precision** - Hard to cite specific documentation sections
- **Bloated storage** - 22x more storage than necessary

### Section-Aware Chunking

The optimized approach uses semantic chunking based on document structure:

**Manual Documents:**
- Split on `<h2>` and `<h3>` headings
- Each section becomes a separate chunk
- Preserves heading hierarchy for context

**API Documents:**
- Split on class/function definitions
- Each `bpy.types.*` or `bpy.ops.*` entry is a chunk
- Signature + description + parameters kept together

### Chunk Format

Each chunk includes a metadata header:

```markdown
# Section Title
DocType: api|manual
DocPath: blender_python_reference_5_0/bpy.types.Material.html#bpy.types.Material
DocVersion: 5.0.1
SourceRepo: blender-documentation
ChunkId: blender_python_reference_5_0/bpy.types.Material.html#bpy.types.Material#0
---

[Extracted content here]
```

**Metadata Fields:**
- `DocType` - Enables filtering by documentation type
- `DocPath` - Stable reference path for citations
- `DocVersion` - Blender version for compatibility checks
- `ChunkId` - Unique identifier for deduplication

---

## Query Routing

### Intent Classification

The search system automatically routes queries to the appropriate store:

```python
# API indicators
API_KEYWORDS = {
    'bpy.', 'bpy.types', 'bpy.ops', 'bpy.data', 'bpy.context',
    'python', 'script', 'function', 'method', 'property', 'class',
    'api', 'reference', 'parameter', 'return', 'type:',
}

# Manual indicators
MANUAL_KEYWORDS = {
    'how to', 'tutorial', 'guide', 'workflow', 'settings',
    'panel', 'menu', 'button', 'option', 'interface',
    'render', 'model', 'animate', 'texture', 'material',
}
```

### Routing Logic

1. **Explicit intent** - If query contains `bpy.` or API patterns → API store
2. **Manual indicators** - If query is about workflows/UI → Manual store
3. **Ambiguous** - Search both stores, merge results by score

### Usage

```python
from tools.semantic_docs_tools import search_blender_docs

# Auto-routed based on query content
results = await search_blender_docs(
    query="bpy.types.FluidDomainSettings resolution",
    max_results=5
)

# Explicit store selection
results = await search_blender_docs(
    query="how to bake fluid simulation",
    intent="manual",  # Force manual store
    max_results=5
)
```

---

## Performance Characteristics

### Search Quality (25-query benchmark)

| Metric | Original | Optimized |
|--------|----------|-----------|
| Pass Rate | 25/25 (100%) | 25/25 (100%) |
| Avg Score | 0.73 | 0.82 |
| Better Scores | - | 13 queries |
| Equal Scores | - | 11 queries |
| Worse Scores | - | 1 query |

### Resource Usage

| Metric | Original | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Storage | 819.5 MB | 36.7 MB | **22x smaller** |
| Files | 4,213 | 17,634 | 4x more granular |
| Avg Response | ~2000 tokens | ~500 tokens | **4x fewer tokens** |

### Cost Implications

- **Storage cost**: 22x reduction
- **Embedding cost**: Similar (more files but smaller)
- **Query cost**: 4x reduction (smaller responses)
- **Overall**: Significant cost savings

---

## Upload Process

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/chunk_blender_docs.py` | HTML → chunked markdown conversion |
| `scripts/fast_individual_upload.py` | Parallel upload with max throughput |
| `scripts/verify_vector_stores.py` | Health checks and content verification |

### Upload Command

```bash
cd agents/blender-vfx-orchestrator
source venv/bin/activate

# Upload Manual docs
python scripts/fast_individual_upload.py --manual-only \
    --store-id vs_697564fbfc0c8191b2e44aa47dbaf482

# Upload API docs
python scripts/fast_individual_upload.py --api-only \
    --store-id vs_697571f0275c8191910fea0f2c8bdd3a
```

### Upload Settings

```python
BATCH_SIZE = 500          # OpenAI max per batch
PARALLEL_BATCHES = 6      # Concurrent batch uploads
FILE_UPLOAD_WORKERS = 20  # Parallel file uploads per batch
```

### Typical Upload Times

| Store | Files | Time | Rate |
|-------|-------|------|------|
| Manual | 4,400 | ~28 min | 2.6 files/sec |
| API | 13,234 | ~55 min | 4.0 files/sec |

---

## OpenAI Vector Store Limits

Reference from official documentation:

| Parameter | Limit |
|-----------|-------|
| Max files per store | 10,000 |
| Max file size | 512 MB |
| Max tokens per file | 5,000,000 |
| Batch upload limit | 500 files |
| Search results | 1-50 (default: 10) |
| Max context chunks | 20 per query |

### Chunking Defaults (if not pre-chunked)

| Setting | Default | Range |
|---------|---------|-------|
| `max_chunk_size_tokens` | 800 | 100-4096 |
| `chunk_overlap_tokens` | 400 | 0 to half of max |

Since we pre-chunk files to ~500-1000 tokens, OpenAI keeps them as single chunks.

---

## Maintenance

### Health Check

```bash
python scripts/verify_vector_stores.py
```

Checks:
- Store accessibility
- File counts match expected
- Sample queries return expected content
- No files stuck in "in_progress"

### Content Verification Queries

The verification script tests these categories:

**Manual Store:**
- Fluid/smoke simulation settings
- Rendering (Cycles, EEVEE)
- Particle systems
- Compositing nodes

**API Store:**
- `bpy.types.*` (FluidDomainSettings, Material, Object, Scene)
- `bpy.ops.*` (mesh, object, render, transform)
- `bpy.context` and `bpy.data` access
- Utility modules (bmesh, mathutils)

### Re-indexing

To re-index after Blender version updates:

1. Download new documentation HTML
2. Create new vector stores (don't overwrite existing)
3. Run upload scripts
4. Verify with health checks
5. Update store IDs in `semantic_docs_tools.py`
6. Delete old stores after verification

---

## Troubleshooting

### Query Returns Wrong Content

**Symptom:** Search returns unrelated results

**Causes:**
1. Query too generic - add specific terms
2. Wrong store selected - check intent routing
3. Content not indexed - verify with health check

**Solution:**
```python
# Be specific
results = search_blender_docs("bpy.types.FluidDomainSettings resolution_max")

# Force store selection
results = search_blender_docs(query, intent="api")
```

### Low Relevance Scores

**Symptom:** Results have scores < 0.5

**Causes:**
1. Query doesn't match documentation terminology
2. Content split across multiple chunks

**Solution:**
- Use exact API names when known
- Try alternate phrasings
- Increase `max_results` to find relevant content lower in rankings

### Missing Content

**Symptom:** Known content not found

**Causes:**
1. Upload failed for that file
2. Chunker skipped the section
3. Store ID mismatch

**Verification:**
```python
# Check if content exists
results = client.vector_stores.search(
    vector_store_id=store_id,
    query="exact text from documentation",
    max_num_results=10
)
```

---

## File Structure

```
agents/blender-vfx-orchestrator/
├── tools/
│   └── semantic_docs_tools.py    # Search interface with routing
├── scripts/
│   ├── chunk_blender_docs.py     # HTML chunking logic
│   ├── fast_individual_upload.py # Parallel upload script
│   └── verify_vector_stores.py   # Health checks
└── docs/
    └── VECTOR_STORE_ARCHITECTURE.md  # This document
```

---

## Changelog

### v2.0 (2026-01-25)
- Split into Manual + API stores
- Section-aware chunking (22x size reduction)
- Intent-based query routing
- Stable DocPath references for citations
- Parallel upload with 500-file batches

### v1.0 (2026-01-18)
- Initial combined store
- Full-page uploads (no chunking)
- Single store for all content

---

## References

- [OpenAI Vector Stores API](https://platform.openai.com/docs/api-reference/vector-stores)
- [OpenAI File Search Guide](https://platform.openai.com/docs/assistants/tools/file-search)
- [Blender 5.0 Documentation](https://docs.blender.org/manual/en/5.0/)
- [Blender Python API Reference](https://docs.blender.org/api/current/)
