# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an enhanced Model Context Protocol (MCP) server that provides comprehensive search and read access to the Blender Manual documentation. The server is optimized for NanoVDB/volumetrics workflows with 8 specialized search tools, persistent caching, and rich metadata extraction.

**Version**: Blender 5.0 Manual
**Total Pages**: ~2,196 HTML files (1.2GB)
**Optimization**: Fast startup with cached indexing (<2 seconds after first run)

## Prerequisites

- Python 3.10 or higher
- Dependencies: `mcp`, `beautifulsoup4`

## Installation

```bash
pip install -r requirements.txt
```

## Running the Server

The server is designed to be used as an MCP server:

```bash
python blender_server.py
```

**First run**: Builds index (~30-60 seconds) and saves cache to `manual_index.json`
**Subsequent runs**: Loads from cache (<2 seconds)

## MCP Configuration

To use this server with Claude Desktop, add to `claude_desktop_config.json`:

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

## Available Tools

The server exposes 8 MCP tools, each optimized for different use cases:

### 1. `search_manual(query, limit=10)`
**General keyword search** across the entire Blender Manual.
- Enhanced scoring: title (20pts), headers (10pts), keywords (15pts), content (1pt each)
- Category metadata and snippet generation
- Best for: General queries, exploratory searches

**Examples**:
```python
search_manual("volume rendering")
search_manual("export openvdb", limit=20)
```

### 2. `search_tutorials(topic, technique=None)`
**Find tutorials and getting started guides**.
- Filters for tutorial/introduction pages
- Optimized for learning resources
- Best for: Learning new workflows, step-by-step guides

**Examples**:
```python
search_tutorials("volumetrics")
search_tutorials("fluid simulation", "smoke")
```

### 3. `browse_hierarchy(path=None)`
**Navigate the manual structure** like a file tree.
- Shows subcategories and pages at any path level
- Displays page sections for quick overview
- Best for: Discovering content, understanding manual organization

**Examples**:
```python
browse_hierarchy()  # Top-level categories
browse_hierarchy("render")
browse_hierarchy("physics/fluid")
```

### 4. `search_vdb_workflow(query)`
**Specialized VDB/OpenVDB/NanoVDB search**.
- Auto-expands queries with VDB-related keywords
- Boosts: physics, render, files categories
- Filters for volume, fluid, cache content
- Best for: VDB export, Mantaflow baking, volume workflows

**Examples**:
```python
search_vdb_workflow("export openvdb")
search_vdb_workflow("bake mantaflow smoke")
search_vdb_workflow("cache fluid simulation")
```

### 5. `search_python_api(operation)`
**Find Python API documentation** (bpy.ops, bpy.types, bpy.data).
- Filters for pages with code blocks
- Boosts scripting/API content
- Shows code example count
- Best for: Automation scripts, API calls, Python workflows

**Examples**:
```python
search_python_api("bpy.ops.fluid.bake")
search_python_api("export volume script")
```

### 6. `search_nodes(node_type, category=None)`
**Search for shader/compositor/geometry nodes**.
- Filters for node documentation
- Optimized for volume-related nodes
- Best for: Principled Volume, Volume Scatter, compositor nodes

**Examples**:
```python
search_nodes("Principled Volume")
search_nodes("Volume Scatter", "shader")
search_nodes("blur", "compositor")
```

### 7. `search_modifiers(modifier_name=None)`
**Find modifier documentation**.
- Filters for modifier pages
- Covers mesh, volume, and simulation modifiers
- Best for: Volume Displace, Fluid, mesh modifiers

**Examples**:
```python
search_modifiers("Volume Displace")
search_modifiers("Fluid")
search_modifiers()  # List all modifiers
```

### 8. `read_page(path)`
**Read full page content** with enhanced formatting.
- Extracts headers, paragraphs, code blocks, lists
- Preserves document structure
- Best for: Deep-diving into specific documentation

**Examples**:
```python
read_page("render/cycles/world_settings.html")
read_page("physics/fluid/type/domain/cache.html")
```

## Architecture

### Core Components

**`blender_server.py`** - Main server implementation using FastMCP

#### Index Building (`build_index()`)
- Walks manual directory recursively
- Parses 2,196 HTML files with BeautifulSoup
- Extracts rich metadata:
  - **Title**: Page title
  - **Content**: Full text content
  - **Category/Subcategory**: Auto-detected from path
  - **Page Type**: tutorial, api, or reference
  - **Keywords**: VDB terms, API mentions, node references
  - **Headers**: All h1-h4 headings
  - **Code Blocks**: Python/command examples
- Saves to `manual_index.json` cache

#### Caching System (`load_cache()`, `save_cache()`)
- **Cache file**: `manual_index.json` (JSON format)
- **Cache version**: 1.0
- **Cache invalidation**: Version mismatch triggers rebuild
- **Performance**: Loads ~2,196 pages in <2 seconds

#### Search Scoring (`compute_score()`)
Enhanced multi-factor scoring:
- **Title match**: 20 points
- **Header match**: 10 points
- **Keyword match**: 15 points
- **Content match**: 1 point per occurrence
- **Category boost**: +25 points (specialized searches)
- **Keyword boost**: +10 points (VDB terms, etc.)
- **Path depth penalty**: -2 points per level

#### VDB Keyword Expansion
Specialized keywords for volumetrics workflow:
```python
VDB_KEYWORDS = {
    "export": ["export", "save", "write", ".vdb", "openvdb", "nanovdb"],
    "bake": ["bake", "cache", "simulation", "mantaflow", "fluid", "smoke"],
    "python": ["bpy.ops", "bpy.types", "bpy.data", "script", "api"],
    "volume_render": ["principled volume", "volume scatter", "density"],
    "coordinate": ["coordinate", "transform", "axis", "space"],
    "lod": ["lod", "level of detail", "resolution", "voxel"]
}
```

### Documentation Structure

The Blender Manual HTML files are organized into topic directories:
- `physics/` - Physics simulation (fluid, smoke, Mantaflow)
- `render/` - Rendering engines (Cycles, EEVEE, volume rendering)
- `files/` - Import/Export formats (VDB, Alembic, FBX)
- `animation/` - Animation features and workflows
- `compositing/` - Compositor nodes and techniques
- `modeling/` - Modeling tools, modifiers, techniques
- `editors/` - UI editor documentation
- `sculpt_paint/` - Sculpting and painting tools
- `grease_pencil/` - 2D drawing and animation
- `getting_started/` - Tutorials and introductions
- `advanced/scripting/` - Python API documentation
- Plus additional directories for interface, scene layout, troubleshooting

### Key Implementation Details

1. **HTML Parsing**: Uses BeautifulSoup with Furo theme conventions (Sphinx documentation). Extracts from `<article role="main">` or fallback selectors.

2. **Text Cleaning**: The `clean_text()` function normalizes whitespace by stripping lines and joining with spaces.

3. **Category Detection**: Auto-tags pages based on file path (e.g., `physics/fluid/` → category: "physics", subcategory: "fluid").

4. **Structured Extraction**: Preserves document structure (headers, paragraphs, code blocks, lists) for better readability.

5. **Extensible Architecture**: Index structure includes comment `# FUTURE: "embeddings": []` for adding semantic search later without breaking changes.

## Use Cases

### NanoVDB/Volumetrics Workflow

The server is specifically optimized for these common tasks:

1. **Export VDB from Blender**
   ```python
   search_vdb_workflow("export openvdb")
   search_python_api("bpy.ops.fluid.bake")
   ```

2. **Bake Mantaflow Simulations**
   ```python
   search_vdb_workflow("bake mantaflow smoke")
   browse_hierarchy("physics/fluid")
   ```

3. **Volume Rendering Setup**
   ```python
   search_nodes("Principled Volume")
   search_manual("volume rendering cycles")
   ```

4. **Python Automation Scripts**
   ```python
   search_python_api("export volume")
   search_python_api("bpy.data.objects")
   ```

5. **Coordinate System Conversion**
   ```python
   search_manual("coordinate system")
   search_manual("axis conversion export")
   ```

6. **LOD Generation**
   ```python
   search_manual("level of detail")
   search_modifiers()
   ```

## Testing

To verify the server is working:

1. **Run the server**: `python blender_server.py`
2. **First run output**:
   ```
   Building fresh index... (this may take 30-60 seconds)
   Indexed 2196 pages.
   Saved index cache (2196 pages) to manual_index.json
   ✓ Index built and cached. Server ready!
   ```
3. **Subsequent runs**:
   ```
   Loaded 2196 pages from cache (built at ...)
   ✓ Index loaded from cache. Server ready!
   ```
4. **If using with Claude Desktop**, the 8 tools should appear in the MCP tools list.

## Performance Characteristics

- **Index build time**: 30-60 seconds (first run only)
- **Cache load time**: <2 seconds
- **Search time**: <100ms for most queries
- **Cache file size**: ~50-100MB (JSON)
- **Memory usage**: ~200-300MB (in-memory index)

## Future Enhancements

The architecture is designed to support these future additions:

1. **Semantic Search**: Add embeddings field to index, implement vector similarity
2. **Query Expansion**: Synonym detection, fuzzy matching
3. **Search History**: Track popular queries and suggest related searches
4. **Related Pages**: Graph-based page relationship suggestions
5. **Image Extraction**: Extract and index figure references
6. **Multi-language Support**: Index multiple language variants

## Common Issues

- **Missing packages**: If import errors occur, ensure `mcp` and `beautifulsoup4` are installed via pip.
- **Empty index**: If no pages are indexed, verify the HTML files exist in the expected directories alongside `blender_server.py`.
- **Encoding errors**: HTML parsing uses `errors="ignore"` to handle encoding issues gracefully.
- **Cache stale**: Delete `manual_index.json` to force a rebuild.
- **Slow search**: First search after startup may be slower; subsequent searches use warm cache.

## Development Notes

### Adding New Search Tools

To add a new specialized search tool:

1. Define a new `@mcp.tool()` function
2. Filter `search_index` for relevant pages
3. Use `compute_score()` with appropriate boosts
4. Format results with `format_results()`

### Modifying Search Ranking

Edit the `compute_score()` function to adjust scoring weights. Current weights:
- Title: 20pts
- Header: 10pts
- Keyword: 15pts
- Content: 1pt each
- Category boost: 25pts
- Keyword boost: 10pts

### Cache Management

- Cache version in `CACHE_VERSION` constant
- Increment version to force rebuild on next run
- Cache stored in `manual_index.json` alongside server file
