# Feedback & Corrections: Blender Librarian Agent Plan (v2)

**Date:** 2026-05-01
**Reviewer:** Gemini (Codebase Architect)
**Status:** Corrected & Approved

I have re-reviewed `BLENDER_LIBRARIAN_AGENT_DESIGN_GPT52.md` and `plans/feat-gpt-5.2-blender-librarian-agent.md` with updated knowledge of the 2026 OpenAI technology stack (SDK v1.60.0+, Responses API, GPT-5.2).

My previous feedback contained a critical error regarding model availability; I apologize. GPT-5.2 is indeed the current SOTA, and the Responses API has replaced the legacy Chat Completions/Assistants paradigms.

However, the **Architectural & Budget** recommendations remain critical. Even with advanced models, the $20/month budget constraint dictates specific implementation choices.

---

## 1. Architecture: The "Responses API" is Correct, but Storage Costs Persist

**The Plan's Strength:**
Using the **Responses API** (`client.responses.create`) is the correct modern approach. It unifies tools, retrieval, and chat state, which is superior to the old Assistants API.

**The Budget Risk:**
The plan proposes uploading the entire Blender Manual to OpenAI's Vector Stores for use with the `file_search` tool.
- **Cost:** Vector Storage is ~$0.10/GB/day. While "cheap", it adds a recurring cost.
- **Redundancy:** You *already* have a local semantic search system in `agents/blender-manual` (using `sentence-transformers` locally) which costs $0.00.
- **Retrieval Control:** OpenAI's `file_search` is excellent but opaque. Using your local tool gives you exact control over chunking and context window usage.

**The Correction:**
**Do NOT** upload the manual to OpenAI's Cloud Vector Store.
Instead, use the **Tool-Use Pattern** with your local searcher:
1. The Librarian Agent (using Responses API) receives a user query.
2. It calls your *existing* local tool: `mcp__blender-manual__search_manual(query)`.
3. Your local server returns text results (free).
4. The Agent synthesizes the answer using GPT-5.2.

**Benefit:** Zero storage cost, zero cloud retrieval token overhead, total control.

---

## 2. Model Selection: GPT-5.2 "Thinking" vs "Pro"

**The Insight:**
The plan correctly identifies **GPT-5.2**.
- **GPT-5.2 Pro:** Excellent generalist, 400k context. Best for "standard" queries.
- **GPT-5.2 Thinking:** (Newer) Halves error rates in complex reasoning/charts. Best for **Vision Diagnosis**.

**The Recommendation:**
- Use **GPT-5.2 Pro** for text/doc synthesis (cheaper, faster).
- Use **GPT-5.2 Thinking** specifically for the `analyze_render_quality` tool where "visual reasoning" is the bottleneck. The reduced error rate on technical diagrams/UI screenshots is worth the premium here.

---

## 3. Revised Budget Strategy ($20/mo)

**Updated Cost Model (Local RAG + Responses API):**
- **Doc Search:** $0.00 (Local embedding model).
- **GPT-5.2 Pro Call:** Input tokens (Your query + ~5 relevant doc chunks).
    - 5 chunks * 500 tokens = 2.5k tokens ~= $0.01 per query.
- **Vision Call (Thinking):**
    - High detail analysis of *one* frame is valuable.
    - Don't send video streams; send critical frames.

**Danger Zone:**
Using OpenAI's hosted "File Search" creates a "black box" of token consumption. Local control is the only way to guarantee adherence to a strict $20 limit.

---

## 4. Refined Implementation Plan (2026 Era)

### Step 1: Tool Integration (Instead of Vector Store)
Instead of `vector_store.py` (Cloud), create `tools/retrieval.py` (Local Bridge).

```python
# The Librarian doesn't hold the data; it asks the Specialist.
@tool
async def query_blender_docs(query: str):
    # Call the EXISTING local agent
    return await mcp_client.call_tool("blender-manual", "search_manual", {"query": query})
```

### Step 2: The "Vision Diagnoser" Loop
Your proposed `analyze_render_quality` tool should leverage **GPT-5.2 Thinking**.

1. **Capture:** `blender-executor` produces `render.png`.
2. **Preprocessing:** Ensure image is optimized (PNG compression) to minimize upload latency, though GPT-5.2 handles high-res well.
3. **Analyze:** Send to GPT-5.2 Thinking with system prompt: *"Compare this render to the description. Focus on lighting, color, and density."*

### Step 3: Synthesis
The Librarian acts as the "Reader" and "Critic":
- **Input:** User Issue ("It's too dark") + Vision Analysis ("Histogram skewed left") + Local Doc Search Results ("FluidSettings.temperature controls blackbody...").
- **Output:** `modify_script` parameters.

---

## 5. Decision Summary

| Feature | Original Plan | Corrected Plan | Reason |
| :--- | :--- | :--- | :--- |
| **Model** | GPT-5.2 | **GPT-5.2 Pro / Thinking** | Use "Thinking" for Vision, "Pro" for Text |
| **Docs** | Cloud Vector Store | **Local MCP Tool** | Cost (Storage) & Control |
| **API** | Responses API | **Responses API** | Correct modern standard |
| **Vision** | Full Res | **Critical Frames Only** | Budget efficiency |

## Next Steps

1.  **Skip** the "Vector Store Setup" phase (cloud upload).
2.  **Implement** the Librarian Server as an MCP server that consumes the `blender-manual` MCP server (Agent-to-Agent pattern).
3.  **Build** the Budget Tracker as planned (it is excellent).