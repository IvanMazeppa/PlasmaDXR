#!/usr/bin/env python3
"""
Wipe and rebuild the knowledge base for Phase 1 reliability.

Backs up existing data before wiping to prevent data loss.
Run this once before starting Phase 1 testing.

Usage:
    python scripts/wipe_kb.py [--dry-run]
"""

import argparse
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

ORCHESTRATOR_ROOT = Path(__file__).parent.parent
DATA_DIR = ORCHESTRATOR_ROOT / "data"
SESSIONS_DIR = ORCHESTRATOR_ROOT / "sessions"


def wipe_kb(dry_run: bool = False) -> None:
    """Wipe knowledge base with backup."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = DATA_DIR / "backups" / f"pre_phase1_{timestamp}"

    print(f"[KB Wipe] Backup directory: {backup_dir}")
    print(f"[KB Wipe] Dry run: {dry_run}")

    # Collect items to back up and wipe
    targets = []

    # experiments.db
    experiments_db = DATA_DIR / "experiments.db"
    if experiments_db.exists():
        targets.append(("experiments.db", experiments_db))

    # Code patterns
    code_patterns_dir = DATA_DIR / "code_patterns"
    if code_patterns_dir.exists():
        for f in code_patterns_dir.glob("*.json"):
            targets.append((f"code_patterns/{f.name}", f))

    # Physics observations
    physics_dir = DATA_DIR / "physics_observations"
    if physics_dir.exists():
        for f in physics_dir.glob("*.json"):
            targets.append((f"physics_observations/{f.name}", f))

    # SDK conversation DB
    sdk_db = SESSIONS_DIR / "sdk" / "vfx_conversations.db"
    if sdk_db.exists():
        targets.append(("sdk/vfx_conversations.db", sdk_db))

    if not targets:
        print("[KB Wipe] Nothing to wipe — KB is already clean.")
        return

    print(f"[KB Wipe] Found {len(targets)} items to back up and wipe:")
    for label, path in targets:
        size = path.stat().st_size if path.exists() else 0
        print(f"  - {label} ({size:,} bytes)")

    if dry_run:
        print("[KB Wipe] DRY RUN — no changes made.")
        return

    # Create backup directory
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Back up each target
    for label, path in targets:
        dest = backup_dir / label
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        print(f"  Backed up: {label}")

    # Wipe targets
    for label, path in targets:
        if path.suffix == ".db":
            # For SQLite DBs, recreate empty
            path.unlink()
            conn = sqlite3.connect(str(path))
            conn.close()
            print(f"  Wiped (recreated empty): {label}")
        else:
            path.unlink()
            print(f"  Wiped: {label}")

    print(f"\n[KB Wipe] Complete. Backup at: {backup_dir}")
    print("[KB Wipe] KB is now clean for Phase 1 testing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wipe and rebuild VFX knowledge base")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be wiped without doing it")
    args = parser.parse_args()
    wipe_kb(dry_run=args.dry_run)
