# Blender Manual MCP Server (Enhanced Edition v4.0)

An advanced Model Context Protocol (MCP) server providing comprehensive search and read access to the **Blender 5.0 Manual** and **Blender Python API Reference**. Optimized for NanoVDB/volumetrics workflows with 12 specialized search tools, persistent caching, and rich metadata extraction.

## Features

🚀 **Fast Performance**
- Persistent caching system (~2 second startup after first run)
- First-time index build: 60-90 seconds for 2,500+ pages
- Search queries: <100ms with improved phrase matching

🔍 **12 Specialized Search Tools**

### Manual Search (7 tools)
1. `search_manual` - General keyword search with enhanced scoring
2. `search_tutorials` - Find tutorials and getting started guides
3. `browse_hierarchy` - Navigate manual structure like a file tree
4. `search_vdb_workflow` - VDB/OpenVDB/NanoVDB specialized search
5. `search_nodes` - Shader/compositor/geometry nodes
6. `search_modifiers` - Modifier documentation
7. `read_page` - Read full page content with enhanced formatting

### Python API Reference (4 tools)
8. `search_python_api` - Unified API + manual scripting search
9. `list_api_modules` - Browse bpy/bmesh/aud modules
10. `search_bpy_operators` - Search bpy.ops.* by category
11. `search_bpy_types` - Search bpy.types.* documentation

### AI-Powered Search (1 tool)
12. `search_semantic` - Semantic similarity with embeddings (requires setup)

🤖 **Semantic Search**
- AI-powered natural language queries
- Finds conceptually related content without exact keyword matches
- Uses sentence-transformers (all-MiniLM-L6-v2, 384-dim embeddings)
- Pre-computed embeddings for 4,227 pages

🎯 **Rich Metadata**
- Category/subcategory auto-detection
- Page type classification (tutorial, API, reference)
- VDB/volume keyword extraction
- Header and code block extraction
- Enhanced relevance scoring

📦 **Extensible Architecture**
- Ready for semantic search integration
- Modular search tool design
- Easy to add new specialized searches

## Quick Start

### Prerequisites

- Python 3.10 or higher
- `pip` (Python package installer)

### Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the server (first run will build index):
   ```bash
   python blender_server.py
   ```

   Expected output:
   ```
   ============================================================
   Blender Manual MCP Server - Enhanced Edition
   Optimized for NanoVDB/Volumetrics Workflow
   ============================================================
   Building fresh index... (this may take 30-60 seconds)
   Indexed 2196 pages.
   Saved index cache (2196 pages) to manual_index.json
   ✓ Index built and cached. Server ready!
   ============================================================
   ```

3. Subsequent runs load from cache (<2 seconds):
   ```
   Loaded 2196 pages from cache (built at ...)
   ✓ Index loaded from cache. Server ready!
   ```

### Semantic Search Setup (Optional but Recommended)

To enable AI-powered semantic search:

1. Create and activate virtual environment:
   ```bash
   cd agents/blender-manual
   python3 -m venv venv
   ```

2. Install semantic search dependencies:
   ```bash
   ./venv/bin/pip install sentence-transformers numpy
   ```

3. Generate embeddings (one-time, ~5 seconds):
   ```bash
   ./venv/bin/python generate_embeddings.py
   ```

   Expected output:
   ```
   Loading index from manual_index.json...
   Loaded 4227 pages from cache.
   Loading embedding model: all-MiniLM-L6-v2...
   Generating embeddings for 4227 pages...
   Saved 4227 embeddings to embeddings.npy
   File size: 6.2 MB
   SUCCESS! Semantic search is now available.
   ```

4. Run server using venv (with semantic search):
   ```bash
   ./run_server.sh
   # Or: ./venv/bin/python blender_server.py
   ```

### Configuration for Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "blender-manual": {
      "command": "python",
      "args": [
        "/absolute/path/to/blender_manual_mcp/blender_server.py"
      ]
    }
  }
}
```

Replace `/absolute/path/to/...` with the actual absolute path to this directory.

## Usage Examples

### NanoVDB Workflow

```python
# Find VDB export documentation
search_vdb_workflow("export openvdb")

# Bake Mantaflow smoke simulation
search_vdb_workflow("bake mantaflow smoke")

# Find Python API for fluid operations
search_python_api("bpy.ops.fluid.bake")

# Volume rendering setup
search_nodes("Principled Volume")
search_manual("volume rendering cycles")
```

### General Usage

```python
# General search
search_manual("volume rendering")

# Find tutorials
search_tutorials("fluid simulation", "smoke")

# Browse manual structure
browse_hierarchy()
browse_hierarchy("physics/fluid")

# Read full page
read_page("physics/fluid/type/domain/cache.html")

# Find modifiers
search_modifiers("Volume Displace")
```

## Tools Reference

### 1. `search_manual(query, limit=10)`
General keyword search with enhanced scoring.

**Parameters:**
- `query` (str): Search query
- `limit` (int, optional): Max results (default: 10)

**Example:**
```python
search_manual("volume rendering", limit=20)
```

### 2. `search_tutorials(topic, technique=None)`
Find tutorials and learning resources.

**Parameters:**
- `topic` (str): Main topic
- `technique` (str, optional): Specific technique

**Example:**
```python
search_tutorials("volumetrics", "smoke")
```

### 3. `browse_hierarchy(path=None)`
Navigate manual structure.

**Parameters:**
- `path` (str, optional): Path to browse (None = top-level)

**Example:**
```python
browse_hierarchy("render")
```

### 4. `search_vdb_workflow(query)`
Specialized VDB/OpenVDB/NanoVDB search.

**Parameters:**
- `query` (str): VDB-related query

**Example:**
```python
search_vdb_workflow("export vdb")
```

### 5. `search_python_api(operation)`
Python API documentation search.

**Parameters:**
- `operation` (str): API operation or concept

**Example:**
```python
search_python_api("bpy.ops.fluid.bake")
```

### 6. `search_nodes(node_type, category=None)`
Search for nodes.

**Parameters:**
- `node_type` (str): Node name or type
- `category` (str, optional): Node category ("shader", "compositor", "geometry")

**Example:**
```python
search_nodes("Principled Volume", "shader")
```

### 7. `search_modifiers(modifier_name=None)`
Modifier documentation search.

**Parameters:**
- `modifier_name` (str, optional): Specific modifier (None = list all)

**Example:**
```python
search_modifiers("Fluid")
```

### 8. `read_page(path)`
Read full page content.

**Parameters:**

- `path` (str): Relative path to HTML file

**Example:**

```python
read_page("render/cycles/world_settings.html")
```

### 9-11. Python API Tools

- `list_api_modules(category=None)` - List bpy/bmesh/aud modules
- `search_bpy_operators(category, operation=None)` - Search operators by category
- `search_bpy_types(typename)` - Search type definitions

**Example:**

```python
list_api_modules("bpy")
search_bpy_operators("mesh", "select")
search_bpy_types("FluidModifier")
```

### 12. `search_semantic(query, limit=5, compact=False)`
AI-powered semantic similarity search using embeddings.

**Parameters:**

- `query` (str): Natural language query
- `limit` (int, optional): Max results (default: 5)
- `compact` (bool, optional): Minimal output mode

**When to Use:**

| Use Semantic Search | Use Keyword Search |
|---------------------|-------------------|
| Natural language questions | Exact term lookup |
| Conceptual queries | API/function names |
| "How do I..." questions | Specific settings |
| Exploring related topics | Known page paths |

**Example:**

```python
search_semantic("how to create realistic volumetric smoke effects")
search_semantic("baking simulations for game engines")
```

**Note:** Requires venv setup. Falls back to keyword search if embeddings unavailable.

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Index build time | 30-60 seconds (first run) |
| Cache load time | <2 seconds |
| Search time | <100ms |
| Total pages | 2,196 HTML files |
| Cache file size | ~50-100MB |
| Memory usage | ~200-300MB |

## Architecture

### Search Scoring (v4.0 Enhanced)

Multi-factor relevance scoring with phrase and proximity matching:
- **Exact phrase match in title**: 50 points
- **Exact phrase match in content**: 25 points
- **All query terms in title**: 30 bonus points
- **Title match**: 20 points per term
- **Header match**: 10 points per term
- **Keyword match**: 15 points per term
- **Content match**: 1 point per occurrence (capped at 10/term)
- **Term proximity bonus**: 5 points (terms within 50 chars)
- **Category boost**: +25 points (specialized searches)
- **Keyword boost**: +10 points (VDB terms, etc.)
- **Index/overview page boost**: +5 points
- **Path depth penalty**: -1 point per level

### VDB Keyword Expansion

Queries are automatically expanded with relevant keywords:
- **export** → export, save, write, .vdb, openvdb, nanovdb
- **bake** → bake, cache, simulation, mantaflow, fluid, smoke
- **python** → bpy.ops, bpy.types, bpy.data, script, api
- **volume_render** → principled volume, volume scatter, density
- **coordinate** → coordinate, transform, axis, space
- **lod** → lod, level of detail, resolution, voxel

### Cache System

- **File**: `manual_index.json`
- **Format**: JSON with metadata
- **Version**: 1.0 (incremental updates)
- **Invalidation**: Automatic on version mismatch
- **Rebuild**: Delete cache file to force rebuild

## Future Enhancements

The architecture is designed to support:

1. **Semantic Search** - Add embeddings for vector similarity
2. **Query Expansion** - Synonym detection, fuzzy matching
3. **Search History** - Track popular queries
4. **Related Pages** - Graph-based suggestions
5. **Image Extraction** - Index figure references
6. **Multi-language** - Support multiple language variants

## Development

### Adding New Search Tools

```python
@mcp.tool()
def search_new_feature(query: str) -> str:
    """Documentation here."""
    if not search_index:
        if not load_cache():
            build_index()

    query_terms = query.lower().split()
    results = []

    for item in search_index:
        score = compute_score(item, query_terms,
                            boost_categories=["category"],
                            boost_keywords=["keyword"])
        if score > 0:
            results.append({**item, "score": score, "snippet": create_snippet(item["content"], query_terms)})

    results.sort(key=lambda x: x["score"], reverse=True)
    return format_results(results, query, limit=10)
```

### Modifying Search Ranking

Edit `compute_score()` function in `blender_server.py` to adjust scoring weights.

### Cache Management

```bash
# Force rebuild
rm manual_index.json
python blender_server.py

# Check cache info
python -c "import json; cache = json.load(open('manual_index.json')); print(f'Pages: {cache[\"page_count\"]}, Built: {cache[\"built_at\"]}')"
```

## Troubleshooting

### Index not building
- Ensure HTML files exist in the directory
- Check Python version (3.10+)
- Verify dependencies: `pip install -r requirements.txt`

### Slow searches
- First search after startup may be slower
- Subsequent searches use warm cache
- Consider increasing memory if swapping occurs

### Cache issues
- Delete `manual_index.json` to force rebuild
- Check disk space (~100MB needed)
- Ensure write permissions

### MCP not connecting
- Verify absolute path in `claude_desktop_config.json`
- Check server starts without errors
- Review Claude Desktop logs for connection issues

## Contributing

This is currently a standalone project. Future contributions may include:
- Additional specialized search tools
- Semantic search implementation
- Multi-language support
- Enhanced result formatting
- Performance optimizations

## License

This project indexes and provides access to the Blender Manual, which is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).

## Credits

- **Blender Manual**: Official Blender documentation
- **MCP Protocol**: Anthropic's Model Context Protocol
- **FastMCP**: Python MCP server framework
- **BeautifulSoup**: HTML parsing library

---

**Optimized for NanoVDB/Volumetrics Workflow** | **12 Specialized Tools** | **<2s Startup** | **2,500+ Pages Indexed**

*Version 4.0 - December 2024*
