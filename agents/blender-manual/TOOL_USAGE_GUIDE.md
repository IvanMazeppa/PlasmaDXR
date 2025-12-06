# Blender Manual MCP Server - Tool Usage Guide

This guide provides detailed instructions on how to effectively use the 9 search tools provided by the Blender Manual MCP server, optimized for NanoVDB/volumetrics workflows.

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Tool Descriptions](#tool-descriptions)
3. [Query Optimization Tips](#query-optimization-tips)
4. [Best Practices by Use Case](#best-practices-by-use-case)
5. [Troubleshooting](#troubleshooting)

---

## Quick Reference

| Tool | Best For | Example Query |
|------|----------|---------------|
| `search_manual` | General questions, broad topics | `"volume rendering cycles"` |
| `search_tutorials` | Learning, step-by-step guides | `topic="fluid simulation"` |
| `browse_hierarchy` | Exploring structure, discovering content | `path="physics"` |
| `search_vdb_workflow` | VDB export, caching, baking | `"bake mantaflow smoke"` |
| `search_python_api` | Python scripts, bpy.ops | `"bpy.ops.fluid"` |
| `search_nodes` | Shader/compositor nodes | `"Principled Volume"` |
| `search_modifiers` | Modifier documentation | `"Fluid"` |
| `read_page` | Full page content | `"physics/fluid/type/domain/cache.html"` |
| `search_semantic` | Natural language, conceptual queries | `"how to make realistic smoke"` |

---

## Tool Descriptions

### 1. `search_manual(query, limit=10)`

**Purpose**: General-purpose search across the entire Blender Manual.

**Parameters**:
- `query` (required): Keywords or phrases to search
- `limit` (optional): Maximum results to return (default: 10, max recommended: 20)

**How It Works**:
- Searches titles, headers, keywords, and content
- Scores results by relevance (title matches > header matches > content matches)
- Returns snippets showing context around matches

**Best Query Strategies**:
```
GOOD: "volume rendering cycles"     # Specific, multiple keywords
GOOD: "smoke density attribute"     # Technical terms
GOOD: "export animation fbx"        # Action + format

AVOID: "how do I render"            # Too vague
AVOID: "volume"                     # Too broad, thousands of results
```

**Example**:
```python
search_manual("coordinate system export", limit=5)
```

---

### 2. `search_tutorials(topic, technique=None)`

**Purpose**: Find tutorials, getting started guides, and learning resources.

**Parameters**:
- `topic` (required): Main subject area
- `technique` (optional): Specific technique or method

**How It Works**:
- Filters for pages in `getting_started/` or marked as tutorials
- Boosts introduction pages and step-by-step content
- Returns beginner-friendly documentation

**Best Query Strategies**:
```
GOOD: topic="fluid simulation", technique="smoke"
GOOD: topic="rendering"
GOOD: topic="animation", technique="keyframes"

AVOID: topic="bpy.ops.fluid"        # Too technical for tutorials
```

**Example**:
```python
search_tutorials("volumetrics", "mantaflow")
```

---

### 3. `browse_hierarchy(path=None)`

**Purpose**: Navigate the manual structure like a file browser.

**Parameters**:
- `path` (optional): Directory path to browse (None = top level)

**How It Works**:
- Returns list of subdirectories and pages at the specified path
- Shows section headers for each page
- Useful for discovering content you didn't know existed

**Best Usage**:
```python
# Start at top level
browse_hierarchy()

# Drill down
browse_hierarchy("physics")
browse_hierarchy("physics/fluid")
browse_hierarchy("render/cycles")
```

**Note**: Currently has a path separator issue on Windows. Use search tools as alternative.

---

### 4. `search_vdb_workflow(query)`

**Purpose**: Specialized search for VDB, OpenVDB, and NanoVDB workflows.

**Parameters**:
- `query` (required): VDB-related search terms

**How It Works**:
- Automatically expands queries with related keywords:
  - "export" → also searches: save, write, .vdb, openvdb, nanovdb
  - "bake" → also searches: cache, simulation, mantaflow, fluid, smoke
  - "python" → also searches: bpy.ops, bpy.types, script, api
- Boosts physics, render, and files categories
- Higher relevance threshold (filters out noise)

**Best Query Strategies**:
```
GOOD: "export openvdb"              # Finds VDB export documentation
GOOD: "bake mantaflow smoke"        # Finds simulation baking
GOOD: "cache fluid simulation"      # Finds caching settings
GOOD: "vdb format settings"         # Finds format options

AVOID: "how to export"              # Missing VDB context
```

**Example Queries for NanoVDB Workflow**:
```python
# Finding export settings
search_vdb_workflow("export openvdb")
search_vdb_workflow("vdb cache format")

# Baking simulations
search_vdb_workflow("bake mantaflow smoke")
search_vdb_workflow("fluid domain cache")

# Volume rendering
search_vdb_workflow("volume density attribute")
```

---

### 5. `search_python_api(operation)`

**Purpose**: Find Python API documentation, scripts, and code examples.

**Parameters**:
- `operation` (required): API call, function name, or scripting concept

**How It Works**:
- Filters for pages containing code blocks
- Boosts `advanced/scripting/` section
- Shows count of code examples per page
- Searches for bpy.ops, bpy.types, bpy.data references

**Best Query Strategies**:
```
GOOD: "bpy.ops.fluid.bake"          # Specific API call
GOOD: "bpy.data.objects"            # Data access
GOOD: "export script volume"        # Scripting task
GOOD: "operator fluid"              # Finding operators

AVOID: "python"                     # Too broad
AVOID: "script"                     # Too generic
```

**Example**:
```python
# Find fluid baking API
search_python_api("bpy.ops.fluid.bake")

# Find volume export scripting
search_python_api("export volume script")

# Find object manipulation
search_python_api("bpy.data.objects modifier")
```

---

### 6. `search_nodes(node_type, category=None)`

**Purpose**: Find shader nodes, compositor nodes, and geometry nodes.

**Parameters**:
- `node_type` (required): Node name or type
- `category` (optional): "shader", "compositor", or "geometry"

**How It Works**:
- Filters for node-related documentation
- Searches in render/, compositing/, and modeling/geometry_nodes/
- Returns node inputs, outputs, and usage

**Best Query Strategies**:
```
GOOD: "Principled Volume"           # Exact node name
GOOD: "Volume Scatter", "shader"    # Node + category
GOOD: "blur", "compositor"          # Effect + category
GOOD: "density"                     # Attribute/property

AVOID: "node"                       # Too generic
```

**Volume-Related Nodes**:
```python
# Primary volume shaders
search_nodes("Principled Volume")
search_nodes("Volume Scatter")
search_nodes("Volume Absorption")

# Volume inputs
search_nodes("Volume Info")
search_nodes("Attribute", "shader")

# Geometry nodes for volumes
search_nodes("distribute points volume", "geometry")
```

---

### 7. `search_modifiers(modifier_name=None)`

**Purpose**: Find modifier documentation.

**Parameters**:
- `modifier_name` (optional): Specific modifier name (None = list all)

**How It Works**:
- Filters for modifier-related pages
- Covers mesh, volume, and simulation modifiers
- Returns settings and usage information

**Best Query Strategies**:
```
GOOD: "Fluid"                       # Specific modifier
GOOD: "Volume Displace"             # Exact name
GOOD: "Subdivision"                 # Partial match
GOOD: None                          # List all modifiers

AVOID: "modify mesh"                # Use search_manual instead
```

**Example**:
```python
# Find specific modifier
search_modifiers("Fluid")
search_modifiers("Ocean")

# List all modifiers
search_modifiers()
```

---

### 8. `read_page(path)`

**Purpose**: Read the full content of a specific manual page.

**Parameters**:
- `path` (required): Relative path to HTML file (from search results)

**How It Works**:
- Extracts headers, paragraphs, code blocks, and lists
- Formats output as readable markdown
- Preserves document structure

**Usage Tips**:
1. First use a search tool to find the correct path
2. Then use `read_page` with the path from results
3. Paths use forward slashes: `render/materials/components/volume.html`

**Example Workflow**:
```python
# Step 1: Search for content
search_vdb_workflow("fluid cache")
# Returns: Path: `physics/fluid/type/domain/cache.html`

# Step 2: Read the full page
read_page("physics/fluid/type/domain/cache.html")
```

**Important Pages for NanoVDB Workflow**:
```python
# Volume rendering fundamentals
read_page("render/materials/components/volume.html")

# Fluid simulation caching
read_page("physics/fluid/type/domain/cache.html")

# Principled Volume shader
read_page("render/shader_nodes/shader/volume_principled.html")

# USD import/export (includes VDB)
read_page("files/import_export/usd.html")
```

---

### 9. `search_semantic(query, limit=5, compact=False)` 🆕

**Purpose**: AI-powered semantic similarity search using embeddings.

**Parameters**:
- `query` (required): Natural language query describing what you're looking for
- `limit` (optional): Maximum results (default: 5)
- `compact` (optional): Minimal output mode

**How It Works**:
- Uses sentence-transformers to generate query embeddings
- Computes cosine similarity against pre-computed document embeddings
- Finds conceptually related content even without exact keyword matches
- Falls back to keyword search if embeddings not available

**Best Query Strategies**:
```
GOOD: "how to create realistic smoke effects"    # Natural language
GOOD: "rendering transparent volumetric clouds"  # Conceptual description
GOOD: "baking simulations for game engines"      # Use case focused

AVOID: "smoke"                                   # Too short/generic
AVOID: "bpy.ops.fluid.bake"                      # Use search_python_api instead
```

**When to Use Semantic vs Keyword Search**:
| Use Semantic Search | Use Keyword Search |
|--------------------|--------------------|
| Natural language questions | Exact term lookup |
| Conceptual queries | API/function names |
| "How do I..." questions | Specific settings |
| Exploring related topics | Known page paths |

**Example Queries**:
```python
# Natural language questions
search_semantic("how to make realistic fire effects")
search_semantic("best practices for baking large simulations")

# Conceptual searches
search_semantic("rendering transparent volumetric materials")
search_semantic("exporting simulations for real-time applications")

# Use case focused
search_semantic("python automation for batch VDB export")
```

**Note**: Requires `sentence-transformers` package. Install with:
```bash
pip install sentence-transformers
```

First run after installation will generate embeddings (~2-3 minutes).

---

## Query Optimization Tips

### 1. Use Specific Keywords

```
# Instead of:
"how to make smoke"

# Use:
"smoke simulation mantaflow"
"domain smoke settings"
```

### 2. Combine Technical Terms

```
# Better results with:
"principled volume density emission"
"mantaflow cache openvdb format"
```

### 3. Use the Right Tool

| Question Type | Best Tool |
|--------------|-----------|
| "What is X?" | `search_manual` |
| "How do I learn X?" | `search_tutorials` |
| "How do I export VDB?" | `search_vdb_workflow` |
| "What Python code for X?" | `search_python_api` |
| "What does X node do?" | `search_nodes` |
| "What does X modifier do?" | `search_modifiers` |
| "Show me the full docs for X" | `read_page` |

### 4. Limit Results to Reduce Noise

```python
# For focused results:
search_manual("volume rendering", limit=5)

# For comprehensive research:
search_manual("volume rendering", limit=20)
```

### 5. Chain Tools Together

```python
# 1. Broad search
search_vdb_workflow("export smoke")

# 2. Find relevant path
# Result shows: physics/fluid/type/domain/cache.html

# 3. Read full documentation
read_page("physics/fluid/type/domain/cache.html")
```

---

## Best Practices by Use Case

### NanoVDB Export Workflow

```python
# 1. Find export documentation
search_vdb_workflow("export openvdb")

# 2. Find cache settings
search_vdb_workflow("domain cache format")

# 3. Find Python API for automation
search_python_api("bpy.ops.fluid.bake")

# 4. Read the cache page fully
read_page("physics/fluid/type/domain/cache.html")
```

### Volume Rendering Setup

```python
# 1. Find shader nodes
search_nodes("Principled Volume")

# 2. Understand volume materials
search_manual("volume rendering cycles")

# 3. Read full volume documentation
read_page("render/materials/components/volume.html")
```

### Mantaflow Smoke Simulation

```python
# 1. Find tutorials
search_tutorials("fluid simulation", "smoke")

# 2. Find specific settings
search_vdb_workflow("mantaflow smoke domain")

# 3. Find caching options
search_vdb_workflow("bake simulation cache")
```

### Coordinate System Conversion

```python
# 1. Find export settings
search_manual("coordinate system export")
search_manual("axis conversion fbx")

# 2. Read relevant export pages
read_page("files/import_export/usd.html")
read_page("files/import_export/alembic.html")
```

---

## Troubleshooting

### "No results found"

- Try broader keywords
- Remove specific version numbers
- Use different synonyms (e.g., "smoke" vs "gas", "cache" vs "bake")

### "Too many irrelevant results"

- Add more specific keywords
- Use the specialized tool (e.g., `search_vdb_workflow` instead of `search_manual`)
- Reduce the `limit` parameter

### "Can't find the path for read_page"

- Use search tools first to find the correct path
- Paths are shown in the `Path:` field of search results
- Use forward slashes: `render/cycles/index.html`

### "browse_hierarchy returns empty"

- Known issue with path separators on Windows
- Use search tools as alternative to discover content
- Try without trailing slashes

---

## Scoring System Explained

Understanding how results are ranked helps you craft better queries:

| Match Type | Points | Example |
|------------|--------|---------|
| Title match | 20 pts | Query "volume" matches title "Volumes" |
| Header match | 10 pts | Query "density" matches section "Density" |
| Keyword match | 15 pts | Query "vdb" matches extracted keyword |
| Content match | 1 pt each | Each occurrence in body text |
| Category boost | +25 pts | Specialized search matches category |
| Keyword boost | +10 pts | VDB-related terms in VDB search |
| Depth penalty | -2 pts/level | Deeper paths slightly penalized |

**Tip**: Use words that appear in titles and headers for better ranking.

---

## Quick Command Reference

```python
# General search
search_manual("volume rendering", limit=10)

# Tutorial search
search_tutorials("fluid simulation", "smoke")

# Browse structure
browse_hierarchy("physics")

# VDB-specific search
search_vdb_workflow("export openvdb cache")

# Python API search
search_python_api("bpy.ops.fluid.bake")

# Node search
search_nodes("Principled Volume", "shader")

# Modifier search
search_modifiers("Fluid")

# Read full page
read_page("physics/fluid/type/domain/cache.html")
```

---

## Version Information

- **Blender Manual Version**: 5.0
- **Total Pages Indexed**: ~2,196
- **Server Version**: 3.0 - Semantic Search Edition
- **Tools Available**: 9 (including semantic search)
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Last Updated**: December 2024
