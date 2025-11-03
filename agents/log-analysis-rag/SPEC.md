# PlasmaDX Log Analysis RAG MCP Server

**Purpose:** NVIDIA-style multi-agent RAG system for automated DirectX 12 log analysis

**Based on:** https://github.com/NVIDIA/GenerativeAIExamples/tree/main/community/log_analysis_multi_agent_rag

---

## Architecture

```
PlasmaDX Logs → Embedding (ChromaDB) → RAG Query → Self-Correcting Agent
     ↓                                      ↓
PIX Events                            Diagnostic Report
```

### Components

1. **Log Ingestion**
   - Parse PlasmaDX logs (`.log` files)
   - Parse PIX event timelines (`.csv` from pixtool)
   - Parse GPU buffer dumps (`.bin` files)
   - Embed using sentence-transformers

2. **Vector Database**
   - ChromaDB (local, no API keys needed)
   - Collections:
     - `plasmadx_logs` - Application logs
     - `pix_events` - PIX capture events
     - `gpu_buffers` - Buffer analysis results
   - Metadata: timestamp, severity, component, particle_count

3. **Query Agent**
   - Semantic search: "Find TDR events correlating with particle count"
   - Time-based: "Show logs between WaitForGPU and Map() call"
   - Pattern matching: "Detect Map() failures"
   - Context extraction: Extract surrounding 10 lines for context

4. **Diagnostic Agent**
   - Analyzes query results
   - Generates hypothesis (e.g., "GPU work incomplete before CPU readback")
   - Suggests fixes (e.g., "Add WaitForGPU before Map()")
   - Provides file:line references

5. **Self-Correction Loop**
   - Verifies suggestions against known patterns
   - Cross-references with PIX event data
   - Confidence scoring (0.0-1.0)
   - Only presents high-confidence diagnostics (>0.7)

---

## MCP Tools

### `ingest_logs`
```python
{
  "log_dir": "/path/to/logs",
  "include_pix": true,  # Also parse PIX captures
  "max_files": 10       # Latest 10 log files
}
```
**Returns:** Ingestion summary (X logs, Y events embedded)

### `query_logs`
```python
{
  "query": "Find Map() failures after frame 0",
  "time_range": ["2025-11-03T18:17:30", "2025-11-03T18:17:40"],
  "particle_count": 2045,  # Optional filter
  "max_results": 20
}
```
**Returns:** Relevant log excerpts with context

### `diagnose_issue`
```python
{
  "symptom": "GPU hang at 2045 particles",
  "context": {
    "particle_count": 2045,
    "frame": 1,
    "shader": "PopulateVolumeMip2"
  }
}
```
**Returns:**
- Root cause hypothesis
- Evidence from logs/PIX
- Suggested fix with file:line
- Confidence score

### `analyze_pattern`
```python
{
  "pattern_type": "tdr_correlation",
  "variable": "particle_count",
  "threshold": 2044  # Find pattern around this threshold
}
```
**Returns:** Pattern analysis (e.g., "TDR occurs at exactly ≥2045 particles")

### `compare_runs`
```python
{
  "baseline_log": "PlasmaDX_2044_particles.log",
  "test_log": "PlasmaDX_2045_particles.log"
}
```
**Returns:** Diff showing what changed between successful and failing runs

---

## Implementation Plan

### Phase 1: Basic RAG (Week 1)
- [x] Log parsing (regex-based)
- [ ] ChromaDB setup
- [ ] Basic semantic search
- [ ] MCP server skeleton

### Phase 2: PIX Integration (Week 2)
- [ ] PIX event CSV parsing
- [ ] Timeline correlation (log timestamp → PIX event)
- [ ] Buffer dump integration

### Phase 3: Diagnostic Agent (Week 3)
- [ ] Pattern recognition (TDR, Map() failure, etc.)
- [ ] Hypothesis generation
- [ ] Fix suggestions with file:line

### Phase 4: Self-Correction (Week 4)
- [ ] Confidence scoring
- [ ] Cross-validation with PIX data
- [ ] Historical learning (save successful diagnoses)

---

## Example Workflow

### User Query
```
"Why does the GPU hang at 2045 particles?"
```

### Agent Workflow
1. **Query logs**: Search for "2045" + "hang" + "TDR" + "WaitForGPU"
2. **Find pattern**:
   ```
   [18:17:37] WaitForGPU - frame 0: 0ms
   [18:17:40] WaitForGPU - frame 1: 3000ms  ← TDR!
   ```
3. **Cross-reference PIX**: Check PopulateVolumeMip2 dispatch duration
4. **Hypothesis**: "Shader takes >2 seconds → Windows TDR kills GPU"
5. **Evidence**: 175k voxel writes × (exp() + atomic) = excessive computation
6. **Suggestion**: "Reduce volume resolution from 64³ to 32³"
   - File: `VolumetricReSTIRSystem.cpp:172`
   - Change: `const uint32_t volumeSize = 32;`
7. **Confidence**: 0.85 (high - pattern matches known TDR behavior)

---

## Technical Stack

- **Python 3.10+**
- **LangChain** - Agent orchestration
- **ChromaDB** - Vector database (local, no API)
- **sentence-transformers** - Embedding model (all-MiniLM-L6-v2)
- **Claude Agent SDK 0.1.4** - MCP server implementation
- **Pandas** - Log/CSV parsing
- **Regex** - Pattern matching

---

## Advantages Over Manual Debugging

| Manual | Agent-Based |
|--------|-------------|
| Read 10k line log manually | Semantic search in <1s |
| Guess which shader hangs | PIX timeline analysis |
| Trial-and-error fixes | Evidence-based suggestions |
| Hours of debugging | Minutes to diagnosis |
| Lose context between sessions | Historical learning |

---

## Integration with Existing Agents

This RAG system **complements** existing agents:

- **pix-debug v4**: Provides buffer dumps → RAG ingests them
- **rtxdi-quality-analyzer**: Provides visual analysis → RAG correlates with logs
- **log-analysis-rag**: Provides diagnostic hypotheses → Other agents verify

**Workflow:**
```
1. pix-debug captures frame → Dumps buffers
2. log-analysis-rag analyzes logs → Finds anomaly
3. rtxdi-quality-analyzer checks visuals → Confirms artifact
4. log-analysis-rag generates fix → Suggests code change
```

---

## Next Steps

1. **Install dependencies**:
   ```bash
   pip install langchain chromadb sentence-transformers pandas
   ```

2. **Create MCP server skeleton** using Agent SDK template

3. **Test with existing logs**:
   - Ingest last 10 PlasmaDX logs
   - Query: "Find Map() failures"
   - Verify results match manual inspection

4. **Add PIX integration** once basic RAG works

---

## Success Metrics

- **Query speed**: <1s for semantic search over 100 log files
- **Diagnosis accuracy**: >80% match with manual root cause analysis
- **Time savings**: 10× faster than manual log reading
- **Confidence threshold**: Only show suggestions with >70% confidence

---

**Status:** Specification complete, ready for implementation
**Estimated effort:** 2-3 weeks part-time
**Dependencies:** Python, Claude Agent SDK, ChromaDB
