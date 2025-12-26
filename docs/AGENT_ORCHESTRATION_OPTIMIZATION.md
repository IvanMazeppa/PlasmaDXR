# Agent Orchestration Optimization Tasks

**Purpose:** Optimize multi-agent coordination, context sharing, and parallel execution for the PlasmaDX NanoVDB pipeline.

**Priority:** AFTER core pipeline is functional. These optimizations improve efficiency but aren't blocking.

**Created:** 2025-12-24

---

## Current Agent Inventory

### Active Agents (Post-Cleanup)

| Agent | Purpose | Status |
|-------|---------|--------|
| `blender-manual` | Blender docs, API lookup (12 tools) | ✅ Active |
| `dxr-image-quality-analyst` | LPIPS ML comparison, visual quality | ✅ Active (repurpose for VDB eval) |
| `mission-control` | Strategic orchestration | ✅ Active (repurpose as iteration controller) |
| `log-analysis-rag` | LangGraph RAG diagnostics | ✅ Active (useful for error analysis) |
| `material-system-engineer` | Material properties, shader gen | ⚠️ Evaluate (may be useful for VDB shaders) |
| `pix-debug` | GPU debugging, buffer analysis | ✅ Active (for RT renderer debugging) |

### Deprecated Agents (Disabled 2025-12-24)

| Agent | Reason | Action |
|-------|--------|--------|
| `gaussian-analyzer` | 3D Gaussian particles no longer used | ❌ Disabled |
| `path-and-probe` | Probe grid lighting for particles | ❌ Disabled |
| `dxr-shadow-engineer` | Particle shadow research | ❌ Disabled |
| `dxr-volumetric-pyro-specialist` | Particle pyro effects | ❌ Disabled |

### New Agents (To Be Created)

| Agent | Purpose | Priority |
|-------|---------|----------|
| `blender-executor` | Execute Blender scripts, capture output | HIGH |
| `asset-evaluator` | Render VDB, compare to references | HIGH |
| `technique-researcher` | Web search + docs aggregation | MEDIUM |

---

## Optimization Tasks

### HIGH PRIORITY (After Pipeline Works)

#### 1. Parallel Tool Execution
**Problem:** Tools often called sequentially when they could run in parallel.
**Impact:** 40-60% longer execution times.
**Solution:** Batch independent tool calls using `asyncio.gather()`.

```python
# BEFORE: Sequential (slow)
result1 = await tool1()
result2 = await tool2()
result3 = await tool3()

# AFTER: Parallel (fast)
result1, result2, result3 = await asyncio.gather(
    tool1(),
    tool2(),
    tool3()
)
```

**Implementation:**
- Identify independent tool calls in common workflows
- Wrap in gather() where applicable
- Add parallel execution tracking metrics

---

#### 2. Permission Consolidation
**Problem:** `settings.local.json` has 206+ permission entries, many redundant.
**Impact:** Slower permission checks, harder maintenance.
**Solution:** Consolidate with wildcards and glob patterns.

**Current (verbose):**
```json
"Bash(git add:*)",
"Bash(git commit:*)",
"Bash(git push)",
"Bash(git rm:*)",
"Bash(git checkout:*)",
"Bash(git merge:*)",
"Bash(git branch:*)",
"Bash(git worktree:*)",
...
```

**Optimized (consolidated):**
```json
"Bash(git:*)",
"Bash(cmake:*)",
"Bash(timeout:*)",
"Bash(\"/mnt/c/Program Files/Microsoft Visual Studio\":*)"
```

**Estimated Reduction:** 206 → ~50 entries

---

#### 3. Model Pre-Warming
**Problem:** LPIPS model (528MB) loads lazily, causing 8-12 second delay on first ML comparison.
**Impact:** Poor user experience on first evaluation.
**Solution:** Pre-warm models at server startup.

```python
# agents/asset-evaluator/server.py

import asyncio
from contextlib import asynccontextmanager

# Global model cache
_lpips_model = None
_clip_model = None

async def preload_models():
    """Load ML models at startup."""
    global _lpips_model, _clip_model

    import lpips
    import torch
    from transformers import CLIPProcessor, CLIPModel

    # Load LPIPS
    _lpips_model = lpips.LPIPS(net='alex')
    if torch.cuda.is_available():
        _lpips_model = _lpips_model.cuda()

    # Load CLIP
    _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    print("[asset-evaluator] Models preloaded")

@asynccontextmanager
async def lifespan(app):
    """Server lifespan - preload on startup."""
    await preload_models()
    yield

# Use in FastMCP
mcp = FastMCP("asset-evaluator", lifespan=lifespan)
```

---

### MEDIUM PRIORITY (Optimization Phase)

#### 4. Shared Context Cache
**Problem:** Each MCP server operates in isolation - no shared context.
**Impact:** Redundant file reads, duplicate analysis.
**Solution:** Implement a shared context cache using Redis or SQLite.

```python
# shared/context_cache.py

from diskcache import Cache
from functools import wraps
import hashlib

cache = Cache('/tmp/plasmadx-agent-cache')

def cached(ttl_seconds: int = 300):
    """Cache decorator for expensive operations."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from function name + args
            key = hashlib.md5(
                f"{func.__name__}:{args}:{kwargs}".encode()
            ).hexdigest()

            # Check cache
            if key in cache:
                return cache[key]

            # Execute and cache
            result = await func(*args, **kwargs)
            cache.set(key, result, expire=ttl_seconds)
            return result
        return wrapper
    return decorator

# Usage in any agent
@cached(ttl_seconds=300)
async def read_codebase_file(path: str) -> str:
    with open(path) as f:
        return f.read()
```

**Cache Targets:**
- File reads (5 min TTL)
- Blender docs lookups (1 hour TTL)
- Research results (30 min TTL)
- Technique patterns (permanent)

---

#### 5. Agent Connection Pooling
**Problem:** `mission-control` uses Claude Agent SDK which creates new session per query.
**Impact:** 2-3 second startup latency per coordination task.
**Solution:** Implement persistent connections with connection pooling.

```python
# shared/agent_pool.py

class AgentPool:
    """Pool of persistent agent connections."""

    def __init__(self, max_connections: int = 5):
        self.pool = asyncio.Queue(maxsize=max_connections)
        self.active = 0
        self.max = max_connections

    async def acquire(self, agent_name: str):
        """Get or create agent connection."""
        try:
            return self.pool.get_nowait()
        except asyncio.QueueEmpty:
            if self.active < self.max:
                conn = await self._create_connection(agent_name)
                self.active += 1
                return conn
            # Wait for available connection
            return await self.pool.get()

    async def release(self, conn):
        """Return connection to pool."""
        await self.pool.put(conn)
```

---

#### 6. Context Window Optimization
**Problem:** High token usage per agent due to large prompts and context.
**Impact:** Higher costs, slower responses.
**Solution:** Semantic compression and smart truncation.

**Current Usage:**
| Agent | Tokens | Notes |
|-------|--------|-------|
| mission-control | ~4,000 | System prompt + context |
| log-analysis-rag | ~2,000 | RAG retrieval |
| blender-manual | ~1,500 | Docs snippets |
| **Total overhead** | **~7,500** | Per query |

**Optimization Strategies:**

1. **Semantic Compression**
   ```python
   async def compress_context(text: str, max_tokens: int = 2000) -> str:
       """Compress long text while preserving meaning."""
       # Use embeddings to identify key sentences
       # Return most relevant portions
   ```

2. **Incremental Context**
   ```python
   # Only send changed portions
   diff = compute_diff(previous_context, new_context)
   send_to_agent(diff)
   ```

3. **Smart Truncation**
   ```python
   # Prioritize recent/relevant context
   context = prioritize_by_relevance(full_context, query)[:max_tokens]
   ```

**Expected Savings:** 40-50% token reduction

---

### LOW PRIORITY (Nice to Have)

#### 7. Performance Monitoring Dashboard
**Problem:** No visibility into agent performance metrics.
**Solution:** Track latency, success rates, cache hits.

```python
# monitoring/metrics.py

from prometheus_client import Histogram, Counter, Gauge

# Metrics
tool_latency = Histogram(
    'tool_call_latency_seconds',
    'Time spent in tool calls',
    ['tool_name', 'agent_name']
)

tool_success = Counter(
    'tool_calls_total',
    'Total tool calls',
    ['tool_name', 'agent_name', 'status']
)

active_agents = Gauge(
    'active_agents',
    'Number of active agent connections'
)

cache_hits = Counter(
    'cache_hits_total',
    'Cache hit count',
    ['cache_name']
)

# Usage
with tool_latency.labels(tool='evaluate_vdb', agent='asset-evaluator').time():
    result = await evaluate_vdb()

tool_success.labels(tool='evaluate_vdb', agent='asset-evaluator', status='success').inc()
```

**Dashboard:** Grafana + Prometheus (if needed)

---

#### 8. Error Pattern Learning
**Problem:** Same errors occur repeatedly across sessions.
**Solution:** Build error → fix database.

```python
# learning/error_patterns.py

class ErrorPatternDB:
    def __init__(self, db_path: str):
        self.db = sqlite3.connect(db_path)
        self._init_schema()

    def record_error(
        self,
        error_type: str,
        error_message: str,
        context: dict,
        fix_applied: str,
        success: bool
    ):
        """Record an error and its fix."""
        self.db.execute("""
            INSERT INTO error_patterns
            (error_type, error_message, context, fix_applied, success, timestamp)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
        """, (error_type, error_message, json.dumps(context), fix_applied, success))
        self.db.commit()

    def suggest_fix(self, error_message: str) -> Optional[str]:
        """Find a fix that worked for similar errors."""
        # Use fuzzy matching or embeddings
        cursor = self.db.execute("""
            SELECT fix_applied FROM error_patterns
            WHERE success = 1
            ORDER BY similarity(error_message, ?) DESC
            LIMIT 1
        """, (error_message,))
        result = cursor.fetchone()
        return result[0] if result else None
```

---

## Implementation Order

### Phase 1: Core Pipeline (Current Focus)
1. Create `blender-executor` MCP server
2. Create `asset-evaluator` MCP server
3. Wire up basic iteration loop
4. Test end-to-end with one asset type

### Phase 2: Orchestration (After Pipeline Works)
1. **Parallel tool execution** (HIGH - biggest ROI)
2. **Permission consolidation** (HIGH - cleanup)
3. **Model pre-warming** (HIGH - UX improvement)

### Phase 3: Optimization (Performance Tuning)
4. **Shared context cache** (MEDIUM)
5. **Agent connection pooling** (MEDIUM)
6. **Context window optimization** (MEDIUM)

### Phase 4: Intelligence (Long-term)
7. **Performance monitoring** (LOW)
8. **Error pattern learning** (LOW)

---

## Metrics to Track

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Tool call latency | Unknown | <500ms | Prometheus histogram |
| Parallel execution ratio | ~10% | >60% | % of calls batched |
| Cache hit ratio | 0% | >40% | Cache stats |
| Permission entries | 206 | <50 | Line count |
| First ML call latency | 8-12s | <2s | With pre-warming |
| Context tokens/query | ~7,500 | ~4,000 | Token counter |

---

## Dependencies

For full optimization implementation:

```
# requirements.txt additions
diskcache>=5.6.0          # Shared context cache
prometheus-client>=0.19   # Metrics (optional)
redis>=5.0.0              # Alternative cache (optional)
```

---

**Status:** Design complete, implementation deferred until core pipeline works
**Next Steps:** Focus on `blender-executor` and `asset-evaluator` first
**Owner:** Ben + Claude Code
