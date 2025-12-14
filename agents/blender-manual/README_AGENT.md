# Blender Manual MCP Server Agent

This agent provides access to the Blender 5.0 Manual and Python API Reference through an MCP server.

## File Structure

- **Server Files** (this directory):
  - `blender_server.py` - MCP server implementation (v4.0)
  - `requirements.txt` - Python dependencies
  - `manual_index.json` - Cached index (auto-generated)
  - `embeddings.npy` - Semantic search embeddings (auto-generated)
  - `CLAUDE.md` - Project documentation
  - `README.md` - Full usage guide
  - `TOOL_USAGE_GUIDE.md` - Detailed tool documentation

- **Manual HTML Files** (separate location):
  - Blender Manual: `/mnt/d/Users/dilli/AndroidStudioProjects/blender_manual_mcp/`
  - Python API Reference: `/mnt/d/Users/dilli/AndroidStudioProjects/blender_python_reference_mcp/`
  - Combined: ~2,500+ HTML documentation pages

## Configuration

The server is configured in `/home/maz3ppa/.cursor/mcp.json`:

```json
"blender-5-docs-repo": {
  "type": "stdio",
  "command": "python3",
  "args": [
    "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/blender-manual/blender_server.py"
  ]
}
```

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **For semantic search** (optional):
   ```bash
   pip install sentence-transformers numpy
   ```

3. **Connect in Claude Code**:
   - Run `/mcp` command
   - Server starts in ~1-2 seconds
   - 12 tools available

## Tools Available

### Manual Search Tools
1. `search_manual` - General keyword search across entire manual
2. `search_tutorials` - Learning resources and getting started guides
3. `browse_hierarchy` - Navigate structure like a file tree
4. `search_vdb_workflow` - VDB/OpenVDB/NanoVDB specialized search
5. `search_nodes` - Shader/geometry/compositor nodes
6. `search_modifiers` - Modifier documentation
7. `read_page` - Full page content with formatting

### Python API Tools
8. `search_python_api` - Python/bpy documentation (both API reference and manual)
9. `list_api_modules` - Browse available API modules (bpy, bmesh, aud, etc.)
10. `search_bpy_operators` - Search bpy.ops.* operators by category
11. `search_bpy_types` - Search bpy.types.* type documentation

### AI-Powered Search
12. `search_semantic` - AI similarity search (requires sentence-transformers)

See `TOOL_USAGE_GUIDE.md` for detailed usage instructions.
