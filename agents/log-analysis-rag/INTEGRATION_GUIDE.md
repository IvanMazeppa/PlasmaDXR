# MCP Server Integration Guide

**Date:** 2025-11-15
**Status:** SDK plugin generation complete, integration in progress

---

## ✅ What the SDK Plugin Generated

The Claude Agent SDK plugin created a **production-ready MCP server** with:

1. **main.py** - Clean agent entry point with Desktop API auth
2. **mcp_server/tools.py** - All 6 MCP tools with placeholder TODOs
3. **Tool schemas** - Correct signatures matching our architecture
4. **CLI interface** - Interactive mode + single query mode
5. **Error handling** - Try/catch blocks around all tool calls
6. **System prompt** - Tier 4 diagnostic agent context

### Generated Tools (All Present ✅)

1. `ingest_logs` - Index logs/PIX/buffers
2. `diagnose_issue` - Run LangGraph workflow (TODO at line 140)
3. `query_logs` - Direct retrieval (TODO at line 223)
4. `analyze_pix_capture` - PIX metadata extraction (TODO at line 292)
5. `read_buffer_dump` - Binary buffer parsing (TODO at line 360)
6. `route_to_specialist` - Specialist routing (✅ FULLY IMPLEMENTED!)

---

## 🔧 Integration Tasks

### PRIORITY 1: Connect diagnose_issue to LangGraph Workflow

**File:** `mcp_server/tools.py` (lines 100-192)

**Current (placeholder):**
```python
# Line 140: TODO: Execute workflow and collect results
# result = workflow.invoke(initial_state)
```

**What to implement:**
```python
# Run workflow
result = workflow.invoke(initial_state)

# Extract results from final state
return {
    "content": [{
        "type": "text",
        "text": f"""
🔍 **Diagnostic Report**

**Root Cause:** {result.get('generation', 'No diagnosis')}

**Evidence:**
{chr(10).join(f'  • {ref}' for ref in result.get('file_line_refs', []))}

**Confidence:** {result.get('confidence', 0.0):.2f}

**Recommended Specialist:** {result.get('recommended_specialist') or 'None'}

**Artifacts:**
{chr(10).join(f'  • {art}' for art in result.get('artifact_paths', []))}
"""
    }]
}
```

**Estimated time:** 5 minutes

---

### PRIORITY 2: Implement query_logs Tool

**File:** `mcp_server/tools.py` (lines 194-257)

**Current (placeholder):**
```python
# Line 223: TODO: Actual retrieval logic
# results = retriever.search(semantic_query, top_k=top_k, filters=filters)
```

**What to implement:**
```python
retriever = get_retriever()

# Ensure retriever is loaded
if not retriever.documents:
    retriever.load_all()

# Query using ensemble retriever
results = retriever.query(semantic_query, top_k=top_k)

# Format results
formatted = f"🔎 **Query:** {semantic_query}\n\n"
formatted += f"**Results:** {len(results)} / {top_k}\n\n"

for i, doc in enumerate(results, 1):
    metadata = doc.metadata
    formatted += f"{i}. **{metadata.get('source', 'unknown')}:{metadata.get('line', '?')}**\n"
    formatted += f"   {doc.page_content[:200]}...\n\n"
```

**Estimated time:** 10 minutes

---

### PRIORITY 3: Implement ingest_logs Tool

**File:** `mcp_server/tools.py` (lines 45-98)

**Current (placeholder):**
```python
# Line 78: TODO: Actual ingestion logic
# retriever.ingest_logs(log_files, pix_files)
```

**What to implement:**
```python
retriever = get_retriever()

# Re-initialize retriever with new path
retriever.log_dirs = [path]
if include_pix:
    retriever.log_dirs.extend([f"{PIX_DIR}/buffer_dumps", f"{PIX_DIR}/Captures"])

# Load documents
retriever.load_all()

return {
    "content": [{
        "type": "text",
        "text": f"✅ Ingested {len(retriever.documents)} documents\n" +
                f"  • Logs: {len([d for d in retriever.documents if 'log' in d.metadata.get('type', '')])}\n" +
                f"  • PIX: {len([d for d in retriever.documents if 'pix' in d.metadata.get('type', '')])}"
    }]
}
```

**Estimated time:** 5 minutes

---

## 🎯 Quick Integration Script

Here's the minimal changes needed to make it work:

### Step 1: Update diagnose_issue_tool (line 140)

Replace the TODO with:
```python
# Run workflow
result = workflow.invoke(initial_state)

# Check confidence threshold
if result.get('confidence', 0.0) < confidence_threshold:
    return {
        "content": [{
            "type": "text",
            "text": f"⚠️ Low confidence ({result.get('confidence', 0.0):.2f} < {confidence_threshold})\n" +
                    "Insufficient evidence for reliable diagnosis."
        }]
    }

# Format diagnostic report
report = f"""
🔍 **Diagnostic Report**

**Diagnosis:** {result.get('generation', 'No diagnosis produced')}

**Confidence:** {result.get('confidence', 0.0):.2f}

**Evidence (File:Line References):**
{chr(10).join(f'  • {ref}' for ref in result.get('file_line_refs', ['No references']))}

**Recommended Specialist:** {result.get('recommended_specialist') or 'None (handle locally)'}

**Related Artifacts:**
{chr(10).join(f'  • {art}' for art in result.get('artifact_paths', ['No artifacts']))}
"""

return {
    "content": [{
        "type": "text",
        "text": report.strip()
    }]
}
```

### Step 2: Update query_logs_tool (line 223)

Replace the TODO with:
```python
retriever = get_retriever()

# Ensure documents loaded
if not retriever.documents:
    retriever.load_all()

# Query
docs = retriever.query(semantic_query, top_k=top_k)

# Format results
formatted = f"🔎 **Query:** {semantic_query}\n\n**Results:** {len(docs)}\n\n"
for i, doc in enumerate(docs, 1):
    meta = doc.metadata
    formatted += f"{i}. **{meta.get('source', '?')}:{meta.get('line', '?')}**\n"
    formatted += f"   {doc.page_content[:150]}...\n\n"

return {
    "content": [{
        "type": "text",
        "text": formatted
    }]
}
```

### Step 3: Update ingest_logs_tool (line 78)

Replace the TODO with:
```python
retriever = get_retriever()
retriever.log_dirs = [path]
if include_pix:
    retriever.log_dirs.extend([f"{PIX_DIR}/buffer_dumps"])
retriever.load_all()

return {
    "content": [{
        "type": "text",
        "text": f"✅ Ingested {len(retriever.documents)} documents from {path}"
    }]
}
```

---

## 📋 Testing Checklist

After integration, test:

1. **diagnose_issue**:
   ```bash
   python main.py "Why is DLSS Super Resolution not working?"
   ```
   Expected: Full LangGraph workflow runs, returns diagnosis with confidence

2. **query_logs**:
   ```bash
   python main.py "Find all Map() failures"
   ```
   Expected: Direct retrieval returns log excerpts

3. **ingest_logs**:
   ```bash
   python main.py --ingest /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/build/bin/Debug/logs
   ```
   Expected: Documents loaded into RAG DB

4. **Interactive mode**:
   ```bash
   python main.py --interactive
   ```
   Expected: REPL loop with all tools available

---

## 🚀 Registration with Claude Code

After testing works locally, register the MCP server:

**Add to `.claude/mcp_settings.json`:**
```json
{
  "mcpServers": {
    "log-analysis-rag": {
      "type": "stdio",
      "command": "bash",
      "args": [
        "-c",
        "cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/log-analysis-rag && source venv/bin/activate && python main.py"
      ],
      "cwd": "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/log-analysis-rag",
      "env": {
        "PROJECT_ROOT": "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean",
        "NVIDIA_API_KEY": "${env:NVIDIA_API_KEY}"
      }
    }
  }
}
```

Then test with:
```bash
claude mcp list
claude mcp test log-analysis-rag
```

---

## ⚠️ Known Issues

1. **NVIDIA API 403** - Resolved (key works, was transient error)
2. **Hybrid retriever singleton** - May need thread safety for concurrent queries
3. **PIX/buffer tools** - Still placeholders (low priority)

---

**Total Integration Time:** ~20 minutes
**Priority:** diagnose_issue > query_logs > ingest_logs > PIX/buffer tools
**Status:** Ready to integrate!
