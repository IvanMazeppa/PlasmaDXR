# Blender Manual MCP Server Agent

This agent provides access to the Blender 5.0 Manual documentation through an MCP server.

## File Structure

- **Server Files** (this directory):
  - `blender_server.py` - MCP server implementation
  - `requirements.txt` - Python dependencies
  - `manual_index.json` - Cached index (auto-generated)
  - `embeddings.npy` - Semantic search embeddings (auto-generated)
  - `CLAUDE.md` - Project documentation
  - `README.md` - Full usage guide
  - `TOOL_USAGE_GUIDE.md` - Detailed tool documentation

- **Manual HTML Files** (separate location):
  - Location: `/mnt/d/Users/dilli/AndroidStudioProjects/blender_manual_mcp/`
  - Contains ~2,196 HTML documentation pages

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
   - 9 tools available

## Tools Available

1. `search_manual` - General keyword search
2. `search_tutorials` - Learning resources
3. `browse_hierarchy` - Navigate structure
4. `search_vdb_workflow` - VDB/OpenVDB/NanoVDB
5. `search_python_api` - Python/bpy documentation
6. `search_nodes` - Shader/geometry nodes
7. `search_modifiers` - Modifier docs
8. `read_page` - Full page content
9. `search_semantic` - AI similarity search (requires sentence-transformers)

See `TOOL_USAGE_GUIDE.md` for detailed usage instructions.
