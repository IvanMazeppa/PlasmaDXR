# Phase 2B-6: Ebbinghaus Memory Decay — Implementation Plan

**Version:** 1.0
**Date:** 2026-02-24
**Branch:** `0.34.12/phase-2b6-memory-decay` (from `main`)
**Estimated effort:** ~180 lines (spec said ~121, but schema migration + backward compat adds ~60)
**Risk:** Low
**SDK version:** 0.10.1 (upgraded from 0.9.3 — non-breaking, 269 tests pass)

---

## What This Does

Knowledge base entries decay exponentially if not reinforced by successful outcomes. This prevents the KB from accumulating stale patterns that mislead agents.

**Core formula (Ebbinghaus-inspired):**
```python
R(t) = e^(-t / S)
```
Where:
- `t` = days since last reinforcement
- `S` = strength = `reinforcement_count * avg_quality_factor` (clamped to `max(S, 0.1)`)
- `R` = retention score (0.0 to 1.0)

**Thresholds:**
- `R >= 0.3` → included in queries (active)
- `0.1 <= R < 0.3` → excluded from queries but still in storage (dormant)
- `R < 0.1` → archived (moved to archive, not deleted)

**Reinforcement:** Each successful use of a pattern resets `last_reinforced` to now and increments `reinforcement_count`, increasing `S` and resetting the decay clock.

---

## Two Memory Systems

The codebase has **two separate memory systems** that both need decay:

| System | Storage | Entries | Used By |
|--------|---------|---------|---------|
| **Code Pattern Memory** | JSON files in `data/code_patterns/` | `CodePattern` dataclass | `search_code_patterns()`, dynamic instructions (indirectly) |
| **Knowledge Base (KB)** | SQLite `parameter_knowledge` table | Dict rows from `query_knowledge()` | `query_validated_learnings()`, dynamic instructions |

Both must implement the same decay function, but the reinforcement mechanism differs:
- **Code patterns:** Updated via `record_pattern_outcome()` in `code_pattern_memory.py`
- **KB entries:** Updated via new `reinforce_entry()` in `experiment_tracker_tools.py`

---

## Implementation Order

### Step 1: Feature Flag + Config (~15 lines)

**File:** `config/agent_config.py`

Add to `PresetConfig` dataclass (after `hitl_interactive`):
```python
memory_decay_enabled: bool = True
```

Add to `from_dict()`:
```python
memory_decay_enabled=data.get("memory_decay_enabled", True),
```

Add accessor to `AgentConfigManager`:
```python
def use_memory_decay(self) -> bool:
    return self._preset.memory_decay_enabled
```

**File:** `config/presets.yaml`

Add `memory_decay_enabled` to each preset:
- `quick_test`: `true` (still want decay in quick tests)
- `development`: `true`
- `production`: `true`
- `budget_saver`: `true`
- `debug`: `false` (disable for debugging — see all patterns)
- `reasoning`: `true`
- `creative`: `true`

---

### Step 2: Schema Changes for CodePattern (~20 lines)

**File:** `utils/code_pattern_memory.py`

Add two fields to `CodePattern` dataclass (after `last_used_at: str = ""`):
```python
last_reinforced: str = ""       # ISO timestamp of last positive reinforcement
reinforcement_count: int = 0    # Number of positive reinforcement events
```

Update `from_dict()` for backward compatibility with existing JSON files:
```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "CodePattern":
    data.setdefault("last_reinforced", "")
    data.setdefault("reinforcement_count", 0)
    # ... existing logic ...
    return cls(**data)
```

Update `record_pattern_outcome()` — when `success=True`, set reinforcement fields:
```python
if success:
    pattern.last_reinforced = datetime.now().isoformat()
    pattern.reinforcement_count += 1
```

---

### Step 3: `compute_retention()` Function (~30 lines)

**File:** `utils/code_pattern_memory.py`

Add as a standalone module-level function (before the `_pattern_memory` global singleton). This is the core Ebbinghaus decay implementation.

```python
import math
from datetime import datetime
from typing import Optional, Dict, Any, Union

# Retention thresholds
RETENTION_ACTIVE = 0.3     # Above this: pattern is active, included in queries
RETENTION_ARCHIVE = 0.1    # Below this: pattern should be archived

def compute_retention(
    entry: Union["CodePattern", Dict[str, Any]],
    now: Optional[datetime] = None,
) -> float:
    """Compute Ebbinghaus retention score for a memory entry.

    Accepts either a CodePattern (for code pattern memory) or a dict
    (for KB entries from parameter_knowledge).

    Returns float in [0.0, 1.0]. Higher = more retained.
    """
    if now is None:
        now = datetime.now()

    # Extract fields — handle both CodePattern and dict
    if isinstance(entry, dict):
        last_reinforced_str = entry.get("last_reinforced", "")
        reinforcement_count = entry.get("reinforcement_count", 0)
        success_count = entry.get("success_count", reinforcement_count)
        avg_score = entry.get("confidence", 50.0)  # KB uses 'confidence'
    else:
        last_reinforced_str = entry.last_reinforced
        reinforcement_count = entry.reinforcement_count
        success_count = entry.success_count
        avg_score = entry.average_improvement if entry.average_improvement > 0 else 50.0

    # If never reinforced, fall back to created_at or return low retention
    if not last_reinforced_str:
        created_str = entry.get("created_at", "") if isinstance(entry, dict) else entry.created_at
        if not created_str:
            return 0.5  # No timestamps at all — neutral retention
        last_reinforced_str = created_str

    try:
        last_reinforced = datetime.fromisoformat(last_reinforced_str)
    except (ValueError, TypeError):
        return 0.5  # Unparseable timestamp — neutral

    days_since = max((now - last_reinforced).total_seconds() / 86400, 0)

    # Strength: higher reinforcement count + quality → slower decay
    strength = max(reinforcement_count * avg_score / 100.0, 0.1)

    return math.exp(-days_since / strength)
```

**Design decisions:**
- Accepts both `CodePattern` and `dict` — avoids needing separate implementations for the two memory systems
- Falls back to `created_at` when `last_reinforced` is empty (new entries start with full retention)
- Uses `total_seconds() / 86400` instead of `.days` to avoid rounding — a 12-hour-old entry should not show as 0 days
- Clamps strength minimum to 0.1 to prevent division by zero in `exp()`

---

### Step 4: DB Schema Migration for parameter_knowledge (~15 lines)

**File:** `agents/experiment-tracker/database.py`

Add migration logic in `_ensure_tables()` after the existing `CREATE TABLE IF NOT EXISTS` block:

```python
# Migration: add Ebbinghaus decay columns if missing
cols = [r[1] for r in conn.execute("PRAGMA table_info(parameter_knowledge)").fetchall()]
if "last_reinforced" not in cols:
    conn.execute("ALTER TABLE parameter_knowledge ADD COLUMN last_reinforced TEXT DEFAULT ''")
if "reinforcement_count" not in cols:
    conn.execute("ALTER TABLE parameter_knowledge ADD COLUMN reinforcement_count INTEGER DEFAULT 0")
```

Update `query_knowledge()` to include the new columns in its return dicts.

**Risk mitigation:** SQLite `ALTER TABLE ADD COLUMN` is safe — it adds a nullable column with a default. Existing rows get the default value. No data loss.

---

### Step 5: `reinforce_entry()` in experiment_tracker_tools.py (~30 lines)

**File:** `tools/experiment_tracker_tools.py`

Following the file's established two-layer pattern:

**Internal implementation:**
```python
def _reinforce_entry_impl(entry_id: str, quality_score: float = 60.0) -> str:
    """Reinforce a KB entry — resets decay clock and increments reinforcement count.

    Called when a pattern/technique from the KB is used successfully.
    entry_id: the 'parameter' key in parameter_knowledge (e.g., 'technique:fire:mantaflow_gas')
    quality_score: the quality score achieved (0-100), updates confidence via EMA
    """
    tracker = _get_tracker_instance()
    db = tracker.db

    now_iso = datetime.now().isoformat()

    with db._connection() as conn:
        row = conn.execute(
            "SELECT confidence, reinforcement_count FROM parameter_knowledge WHERE parameter = ?",
            (entry_id,)
        ).fetchone()

        if not row:
            return f"Entry '{entry_id}' not found in knowledge base"

        old_confidence = row[0] or 50.0
        old_count = row[1] or 0

        # EMA update for confidence
        new_confidence = old_confidence + 0.1 * (quality_score / 100.0 * 100 - old_confidence)
        new_count = old_count + 1

        conn.execute(
            """UPDATE parameter_knowledge
               SET last_reinforced = ?, reinforcement_count = ?,
                   confidence = ?, last_updated = ?
               WHERE parameter = ?""",
            (now_iso, new_count, new_confidence, now_iso, entry_id)
        )

    return f"Reinforced '{entry_id}': count={new_count}, confidence={new_confidence:.1f}"
```

**Agent-facing wrapper:**
```python
@function_tool
async def reinforce_entry(
    ctx: RunContextWrapper[Any],
    entry_id: str,
    quality_score: float = 60.0,
) -> str:
    """Reinforce a knowledge base entry after successful use.
    Resets the decay clock and increments reinforcement count."""
    return _reinforce_entry_impl(entry_id, quality_score)
```

**Also update `_seed_technique_entry_impl()`** to initialize seed entries with `reinforcement_count=1` and `last_reinforced=now` so they don't immediately decay.

---

### Step 6: Filter Patterns by Retention in code_pattern_tools.py (~20 lines)

**File:** `tools/code_pattern_tools.py`

In `search_code_patterns()`, after `memory.retrieve_patterns_for_issue()` returns:

```python
from utils.code_pattern_memory import compute_retention, RETENTION_ACTIVE

# Filter by Ebbinghaus retention
if config_manager.use_memory_decay():
    now = datetime.now()
    patterns = [p for p in patterns if compute_retention(p, now) >= RETENTION_ACTIVE]
```

Add `retention_score` to each pattern's output dict:
```python
"retention_score": round(compute_retention(p, now), 2),
```

Apply same filtering in `search_patterns_impl()` (the non-agent `_impl` variant).

---

### Step 7: Filter KB Entries by Retention in dynamic_instructions.py (~20 lines)

**File:** `tools/dynamic_instructions.py`

In `query_validated_learnings()`, after KB entries are retrieved and before they're returned:

```python
from utils.code_pattern_memory import compute_retention, RETENTION_ACTIVE

def query_validated_learnings(effect_type, category, min_success_rate):
    # ... existing KB query logic ...

    # Apply Ebbinghaus decay filter
    if _should_apply_decay():
        now = datetime.now()
        entries = [e for e in entries if compute_retention(e, now) >= RETENTION_ACTIVE]

    return entries
```

The `_should_apply_decay()` helper checks the `ENABLE_MEMORY_DECAY` config flag. Uses the same lazy-import pattern the file already uses for KB functions.

Update `format_learnings_as_instructions()` to include retention score in output:
```
- [fire] Use burning_rate 0.3-0.8 for candle flames (success: 82%, retention: 67%)
```

---

### Step 8: Exports (~5 lines)

**File:** `utils/__init__.py`

Add exports: `compute_retention`, `RETENTION_ACTIVE`, `RETENTION_ARCHIVE`

---

### Step 9: Tests (~100 lines)

**File:** `tests/test_memory_decay.py`

| Test Class | Tests | What It Validates |
|-----------|-------|-------------------|
| `TestComputeRetention` | 6 | Fresh entry = 1.0, stale entry < 0.3, very stale < 0.1, reinforcement resets, strength scaling, edge cases (empty strings, missing fields) |
| `TestRetentionFiltering` | 3 | search_code_patterns filters stale, dynamic_instructions filters stale, feature flag disable bypasses filter |
| `TestReinforceEntry` | 3 | Reinforcement updates count + timestamp, EMA updates confidence, entry not found returns error |
| `TestBackwardCompat` | 2 | Old CodePattern JSON without new fields loads fine, old KB entries without new columns work |
| `TestArchiveThreshold` | 2 | Entries below 0.1 flagged for archive, entries above 0.1 kept |

**Total: ~16 tests**

---

## File Change Summary

| File | Change | Lines |
|------|--------|-------|
| `config/agent_config.py` | Add `memory_decay_enabled` field + accessor | ~8 |
| `config/presets.yaml` | Add flag to all 7 presets | ~7 |
| `utils/code_pattern_memory.py` | New fields on CodePattern, `compute_retention()`, update `from_dict()`, update `record_pattern_outcome()` | ~55 |
| `tools/code_pattern_tools.py` | Filter by retention in `search_code_patterns()` + impl variant | ~20 |
| `tools/dynamic_instructions.py` | Filter by retention in `query_validated_learnings()` | ~20 |
| `tools/experiment_tracker_tools.py` | `reinforce_entry()` impl + wrapper, update seed entries | ~35 |
| `agents/experiment-tracker/database.py` | Schema migration + include new columns in query results | ~15 |
| `utils/__init__.py` | Exports | ~5 |
| **New:** `tests/test_memory_decay.py` | 16 tests across 5 test classes | ~100 |
| **Total** | | **~265** |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Old CodePattern JSON files missing new fields | `from_dict()` uses `setdefault()` — backward compatible |
| Old SQLite DB missing new columns | `ALTER TABLE` migration with `IF NOT EXISTS` check |
| Seed entries immediately decay | Initialize with `reinforcement_count=1`, `last_reinforced=now` |
| `last_used_at` is empty string (not None) | Guard with falsy check `if not last_reinforced_str:` |
| Feature breaks existing behavior | `ENABLE_MEMORY_DECAY` flag — set `false` to revert |
| `query_knowledge()` doesn't return new columns | Update return dict to include them |

---

## What This Does NOT Do

- **Does not archive entries automatically.** Archival (moving entries with `R < 0.1` to a separate store) is deferred. For now, they're simply excluded from queries. Archival can be a follow-up task or a periodic cleanup script.
- **Does not change the existing `confidence` property** on CodePattern. The existing linear decay stub in `confidence` remains as-is. `compute_retention()` is a separate, independent score used for filtering.
- **Does not add decay to session state.** Session history and iteration state are ephemeral and don't need decay.

---

## Rollback

Set `memory_decay_enabled: false` in `config/presets.yaml` (or `ENABLE_MEMORY_DECAY = False` in agent_config.py). When disabled:
- `compute_retention()` is never called
- All patterns/entries have implicit retention = 1.0
- No filtering applied

---

## Dependencies

| Dependency | Status |
|-----------|--------|
| KB wipe (Phase 1) | DONE |
| Phase 2B-5 HITL | DONE (current branch) |
| SDK 0.10.1 | DONE (just upgraded, 269 tests pass) |

## Downstream

- **2B-7 (Effect-Type-Scoped Evidence Gating)** depends on 2B-6 — adds `effect_type` scoping to the same filtering mechanism
- **2C-6 (Knowledge Distillation)** will call `reinforce_entry()` when extracting patterns from successful runs
