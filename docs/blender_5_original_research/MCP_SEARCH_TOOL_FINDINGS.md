# MCP Blender Manual Search Tool - Critical Findings

**Date:** 2025-12-13
**Investigator:** Claude Code (Multi-Agent Optimization Session)
**MCP Server:** blender-manual (9+ search tools)

---

## Executive Summary

During agent optimization research, I discovered **significant gaps** in the MCP blender-manual search tools' Python API coverage. These gaps can cause agents to generate incorrect code or miss critical API information.

---

## Critical Finding 1: `openvdb_cache_compress_type` Not Indexed

### Problem
The property `openvdb_cache_compress_type` (FluidDomainSettings) is **NOT directly searchable** via the MCP tools, despite being a critical property for VDB export.

### Evidence
```
Query: search_python_api("openvdb_cache_compress_type")
Result: Generic results (BMesh, bpy_struct) - NO direct match
```

### Impact
- Agents cannot verify the correct enum values ('ZIP', 'BLOSC', 'NONE')
- The HARDWARE_OPTIMIZATION_RESEARCH.md incorrectly states BLOSC is "removed"
- The Blender 5.0 Manual actually shows BLOSC IS available in the UI

### Recommendation
1. Add full indexing of `FluidDomainSettings` properties in the MCP server
2. Include enum value lists in the index for searchability
3. Consider adding a `search_enum_values` tool

---

## Critical Finding 2: Conflicting Information on BLOSC Compression

### Problem
There is conflicting information about BLOSC availability:

| Source | Says |
|--------|------|
| Blender 5.0 Manual (MCP `read_page`) | BLOSC available: "Cache files will be written with Blosc compression" |
| Web research (prior session) | "BLOSC removed from Blender 5.0" |
| Agent prompts (before fix) | "BLOSC (fastest) recommended" |

### Evidence
From MCP `read_page("physics/fluid/type/domain/cache.html")`:
```
Compression
OpenVDB Only
Compression method that is used when writing OpenVDB cache files.
Zip: Cache files will be written with Zip compression.
Blosc: Cache files will be written with Blosc compression. Multithreaded compression.
None: Cache files will be written without any compression.
```

### Resolution
The **Blender 5.0 Manual is authoritative**. BLOSC IS available. The web sources were either:
- Referring to a different Blender version
- Referring to Python API enum vs UI label discrepancy
- Incorrect

### Action Taken
Updated agent prompts to use try/except pattern to handle both cases:
```python
try:
    settings.openvdb_cache_compress_type = 'BLOSC'
except TypeError:
    settings.openvdb_cache_compress_type = 'ZIP'
```

---

## Critical Finding 3: FluidDomainSettings Page Truncation

### Problem
The `read_page("bpy.types.FluidDomainSettings.html")` output is truncated at ~8000 characters, cutting off critical properties including:
- `openvdb_cache_compress_type`
- `openvdb_data_depth` (precision setting)
- `resolution_max`

### Evidence
Page read ends mid-property at `noise_pos_scale` - does not reach `openvdb_*` properties alphabetically.

### Impact
Agents using `read_page` to verify API cannot see all properties, leading to incomplete verification.

### Recommendation
1. Increase `max_length` parameter beyond 8000 for Python API pages
2. Add property-specific search tool: `search_property("FluidDomainSettings", "openvdb")`
3. Or index properties separately from full page content

---

## Critical Finding 4: Python API Index Incomplete

### Problem
The MCP server appears to index manual pages well but has gaps in Python API coverage.

### Evidence
```
Query: search_bpy_types("FluidDomainSettings")
Result: Found 6 matching types - good

Query: search_python_api("openvdb_cache_compress_type")
Result: 2082 generic results, NO specific match to FluidDomainSettings.openvdb_cache_compress_type
```

The search found the type but not its specific properties.

### Recommendation
Consider creating a dedicated property index:
```json
{
  "FluidDomainSettings": {
    "openvdb_cache_compress_type": {
      "type": "enum",
      "values": ["ZIP", "BLOSC", "NONE"],
      "default": "BLOSC"
    },
    ...
  }
}
```

---

## Critical Finding 5: Semantic Search May Hallucinate

### Problem
The `search_semantic` tool uses embeddings which can return "conceptually related" content that doesn't actually contain the search terms.

### Impact
For API verification (where exact property names matter), semantic search can mislead agents into thinking they found verification when they didn't.

### Recommendation
For API verification tasks, agents should:
1. Use `search_bpy_types` first (exact match)
2. Use `read_page` to get full property list
3. Only use `search_semantic` for conceptual discovery, NOT verification

---

## Recommendations for MCP Server Enhancement

### Priority 1 (Critical for Agent Reliability)
1. **Full Property Indexing**: Index all properties of bpy.types.* with their types and enum values
2. **Increase `read_page` limit**: Allow up to 20,000 characters for Python API pages
3. **Add `search_property` tool**: Search for specific properties within types

### Priority 2 (Useful Improvements)
4. **Add `get_enum_values` tool**: Return all valid values for an enum property
5. **Add `verify_property_exists` tool**: Quick existence check for property name
6. **Add API version tracking**: Include Blender version in search results

### Priority 3 (Nice to Have)
7. **Property cross-reference**: Link properties to their manual page equivalents
8. **Deprecation tracking**: Flag deprecated properties with migration notes
9. **Code example extraction**: Index code examples separately for quick access

---

## Impact on Agent Prompts

Based on these findings, I updated the following files:

1. **`agents/blender-scripting/AGENT_PROMPT.md`**
   - Added MCP-First Verification Protocol section
   - Fixed all BLOSC references to ZIP with fallback
   - Added collaboration guidelines with curator

2. **`agents/celestial-body-curator/AGENT_PROMPT.md`**
   - Added MCP-First Verification Protocol section
   - Fixed BLOSC reference in recipe template
   - Added clear handoff protocol with scripting agent

3. **`docs/blender_recipes/README.md`**
   - Fixed BLOSC reference in troubleshooting section

---

## Conclusion

The MCP blender-manual server is valuable but has significant gaps in Python API coverage. Agents relying solely on MCP search results for API verification may generate incorrect code. The mitigation strategy is:

1. Always use try/except for API calls that may vary by Blender version
2. Verify against multiple sources (MCP + web research + actual testing)
3. Use `read_page` with higher limits for complete API documentation
4. Flag uncertain API calls in generated scripts

---

*Document created as part of Multi-Agent Optimization session*
*Last Updated: 2025-12-13*
