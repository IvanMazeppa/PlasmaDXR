"""
MCP Tools for PlasmaDX Log Analysis RAG
Aligned with RUNBOOK_MULTI_AGENT_RAG.md architecture
"""

import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.workflow import get_workflow
from src.tools.hybrid_retriever import PlasmaDXHybridRetriever

# Load environment variables
load_dotenv()

# Initialize paths from environment
PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean")
LOG_DIR = os.getenv("LOG_DIR", f"{PROJECT_ROOT}/build/bin/Debug/logs")
PIX_DIR = os.getenv("PIX_DIR", f"{PROJECT_ROOT}/PIX")

# Initialize retriever (lazy loading)
_retriever: Optional[PlasmaDXHybridRetriever] = None


def get_retriever() -> PlasmaDXHybridRetriever:
    """Get or initialize hybrid retriever singleton"""
    global _retriever
    if _retriever is None:
        # Initialize with default log directories
        log_dirs = [LOG_DIR, f"{PIX_DIR}/buffer_dumps", f"{PIX_DIR}/Captures"]
        _retriever = PlasmaDXHybridRetriever(log_dirs=log_dirs)
    return _retriever


# ============================================================================
# PRIMARY TOOLS (from runbook)
# ============================================================================

async def ingest_logs_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Index logs/PIX/buffers into RAG database

    Args:
        path: Path to log directory (default: LOG_DIR from env)
        include_pix: Also parse PIX captures (default: True)
        max_files: Maximum number of log files to ingest (default: 10)

    Returns:
        Ingestion summary with counts and status
    """
    path = args.get("path", LOG_DIR)
    include_pix = args.get("include_pix", True)
    max_files = args.get("max_files", 10)

    try:
        retriever = get_retriever()

        # Count files to ingest
        log_files = list(Path(path).glob("*.log"))[:max_files]
        pix_files = list(Path(f"{PIX_DIR}/Captures").glob("*.csv")) if include_pix else []

        # Re-initialize retriever with new path
        retriever.log_dirs = [Path(path)]
        if include_pix:
            retriever.log_dirs.extend([Path(f"{PIX_DIR}/buffer_dumps")])

        # Load documents
        retriever.load_documents()

        # Count document types
        logs_count = len([d for d in retriever.documents if 'log' in d.metadata.get('type', '')])
        pix_count = len([d for d in retriever.documents if 'pix' in d.metadata.get('type', '')])

        return {
            "content": [{
                "type": "text",
                "text": f"✅ Ingested {len(retriever.documents)} documents from {path}\n" +
                        f"  • Logs: {logs_count}\n" +
                        f"  • PIX: {pix_count}"
            }]
        }

    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"❌ Ingestion failed: {str(e)}"
            }],
            "isError": True
        }


async def diagnose_issue_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run full LangGraph self-correcting diagnostic workflow

    Args:
        question: Diagnostic question (e.g., "Why does GPU hang at 2045 particles?")
        confidence_threshold: Minimum confidence to report (default: 0.7)
        context: Optional context dict (particle_count, frame, shader, etc.)

    Returns:
        Diagnostic report with root cause, evidence, fix suggestions, and confidence
    """
    question = args.get("question", "")
    confidence_threshold = args.get("confidence_threshold", 0.7)
    context = args.get("context", {})

    if not question:
        return {
            "content": [{
                "type": "text",
                "text": "❌ Error: 'question' parameter is required"
            }],
            "isError": True
        }

    try:
        # Get compiled workflow
        workflow = get_workflow()

        # Prepare initial state
        initial_state = {
            "question": question,
            "documents": [],
            "generation": "",
            "hallucination_score": 0.0,
            "context": context
        }

        # Run workflow
        result = workflow.invoke(initial_state)

        # Filter by confidence threshold
        if result.get('confidence', 0.0) < confidence_threshold:
            return {
                "content": [{
                    "type": "text",
                    "text": f"⚠️ Low confidence ({result['confidence']:.2f} < {confidence_threshold})\n" +
                            "Insufficient evidence to provide reliable diagnosis."
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

    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"❌ Diagnosis failed: {str(e)}"
            }],
            "isError": True
        }


async def query_logs_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Direct hybrid retrieval (BM25 + FAISS) bypassing full workflow

    Args:
        semantic_query: Natural language query
        top_k: Number of results to return (default: 10)
        filters: Optional metadata filters (particle_count, severity, etc.)

    Returns:
        Relevant log excerpts with context and metadata
    """
    semantic_query = args.get("semantic_query", "")
    top_k = args.get("top_k", 10)
    filters = args.get("filters", {})

    if not semantic_query:
        return {
            "content": [{
                "type": "text",
                "text": "❌ Error: 'semantic_query' parameter is required"
            }],
            "isError": True
        }

    try:
        retriever = get_retriever()

        # Ensure documents loaded
        if not retriever.documents:
            retriever.load_documents()

        # Query using hybrid retriever
        docs = retriever.retrieve(semantic_query)[:top_k]

        # Format results
        formatted = f"🔎 **Query:** {semantic_query}\n\n**Results:** {len(docs)}\n\n"
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            formatted += f"{i}. **{meta.get('source', '?')}:{meta.get('line', '?')}**\n"
            formatted += f"   {doc.page_content[:150]}...\n\n"

        return {
            "content": [{
                "type": "text",
                "text": formatted.strip()
            }]
        }

    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"❌ Query failed: {str(e)}"
            }],
            "isError": True
        }


# ============================================================================
# ADDITIONAL TOOLS (supplementary)
# ============================================================================

async def analyze_pix_capture_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract PIX GPU capture metadata and timeline events

    Args:
        capture_path: Path to .wpix file (optional, auto-detects latest)
        extract_events: Export event timeline to CSV (default: True)

    Returns:
        PIX capture analysis with event timeline and performance metrics
    """
    capture_path = args.get("capture_path", None)
    extract_events = args.get("extract_events", True)

    try:
        # Auto-detect latest capture if not specified
        if not capture_path:
            pix_captures = list(Path(f"{PIX_DIR}/Captures").glob("*.wpix"))
            if not pix_captures:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"❌ No PIX captures found in {PIX_DIR}/Captures"
                    }],
                    "isError": True
                }
            capture_path = str(sorted(pix_captures, key=lambda p: p.stat().st_mtime)[-1])

        # TODO: Actual PIX analysis using pixtool
        # pixtool open-capture {capture_path} save-event-list events.csv

        result = {
            "capture_file": Path(capture_path).name,
            "total_events": "N/A (not implemented)",
            "duration_ms": "N/A",
            "draw_calls": "N/A",
            "dispatch_calls": "N/A"
        }

        return {
            "content": [{
                "type": "text",
                "text": f"📊 **PIX Capture Analysis**\n\n" +
                        f"**File:** {result['capture_file']}\n" +
                        f"**Total Events:** {result['total_events']}\n" +
                        f"**Duration:** {result['duration_ms']}\n" +
                        f"**Draw Calls:** {result['draw_calls']}\n" +
                        f"**Dispatch Calls:** {result['dispatch_calls']}"
            }]
        }

    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"❌ PIX analysis failed: {str(e)}"
            }],
            "isError": True
        }


async def read_buffer_dump_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse binary GPU buffer dumps

    Args:
        buffer_path: Path to .bin file (e.g., g_particles.bin)
        buffer_type: Type of buffer (particles, reservoirs, rtLighting)
        max_entries: Maximum number of entries to show (default: 10)

    Returns:
        Parsed buffer data with statistics
    """
    buffer_path = args.get("buffer_path", "")
    buffer_type = args.get("buffer_type", "particles")
    max_entries = args.get("max_entries", 10)

    if not buffer_path:
        return {
            "content": [{
                "type": "text",
                "text": "❌ Error: 'buffer_path' parameter is required"
            }],
            "isError": True
        }

    try:
        if not Path(buffer_path).exists():
            return {
                "content": [{
                    "type": "text",
                    "text": f"❌ Buffer file not found: {buffer_path}"
                }],
                "isError": True
            }

        # TODO: Actual buffer parsing based on type
        # if buffer_type == "particles":
        #     parse_particle_buffer(buffer_path)

        file_size = Path(buffer_path).stat().st_size

        result = {
            "file": Path(buffer_path).name,
            "size_bytes": file_size,
            "buffer_type": buffer_type,
            "entries": "N/A (parsing not implemented)"
        }

        return {
            "content": [{
                "type": "text",
                "text": f"📦 **Buffer Dump Analysis**\n\n" +
                        f"**File:** {result['file']}\n" +
                        f"**Type:** {result['buffer_type']}\n" +
                        f"**Size:** {result['size_bytes']:,} bytes\n" +
                        f"**Entries:** {result['entries']}"
            }]
        }

    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"❌ Buffer read failed: {str(e)}"
            }],
            "isError": True
        }


async def route_to_specialist_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recommend which specialist agent should handle the issue

    Args:
        issue_description: Description of the rendering issue
        symptoms: List of observed symptoms
        context: Optional context (particle_count, FPS, shader, etc.)

    Returns:
        Recommended specialist agent(s) and escalation reasoning
    """
    issue_description = args.get("issue_description", "")
    symptoms = args.get("symptoms", [])
    context = args.get("context", {})

    if not issue_description:
        return {
            "content": [{
                "type": "text",
                "text": "❌ Error: 'issue_description' parameter is required"
            }],
            "isError": True
        }

    try:
        # Simple keyword-based routing (can be enhanced with ML)
        routing_rules = {
            "pix-debugger": ["gpu hang", "tdr", "crash", "freeze", "timeout"],
            "rtxdi-integration-specialist": ["rtxdi", "lighting", "shadows", "reservoir"],
            "dxr-image-quality-analyst": ["visual", "artifacts", "quality", "screenshot", "lpips"],
            "performance-analyzer": ["fps", "performance", "slow", "bottleneck"],
            "buffer-validator": ["buffer", "corruption", "data", "validation"]
        }

        # Score each specialist
        scores = {}
        issue_lower = issue_description.lower()
        symptoms_lower = " ".join(symptoms).lower()
        combined_text = f"{issue_lower} {symptoms_lower}"

        for specialist, keywords in routing_rules.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            if score > 0:
                scores[specialist] = score

        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if not ranked:
            recommendation = "mission-control (general diagnostic agent)"
            reasoning = "No specific specialist matched. Mission control will coordinate."
        else:
            recommendation = ranked[0][0]
            reasoning = f"Matched {ranked[0][1]} keywords: {', '.join(routing_rules[recommendation])}"

        # Format response
        response = f"""
🎯 **Specialist Routing Recommendation**

**Issue:** {issue_description}

**Recommended Agent:** `{recommendation}`

**Reasoning:** {reasoning}
"""

        if len(ranked) > 1:
            response += "\n**Alternative Specialists:**\n"
            for agent, score in ranked[1:3]:  # Top 2 alternatives
                response += f"  • {agent} (score: {score})\n"

        return {
            "content": [{
                "type": "text",
                "text": response.strip()
            }]
        }

    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"❌ Routing failed: {str(e)}"
            }],
            "isError": True
        }
