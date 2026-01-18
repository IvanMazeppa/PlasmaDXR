#!/usr/bin/env python3
"""
Secret guardrail (fast, zero-deps).

Goal:
- Prevent accidental commits of real API keys inside tracked *.env.example templates.

Design:
- Only scans files tracked by git (git ls-files)
- Only targets files ending with /.env.example
- Never prints secret values (only file/line/var and pattern type)
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class Finding:
    path: str
    line_no: int
    kind: str
    detail: str


def _run(cmd: List[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, errors="replace")


def git_root() -> Path:
    return Path(_run(["git", "rev-parse", "--show-toplevel"]).strip())


def tracked_env_example_files() -> List[str]:
    out = _run(["git", "ls-files"]).splitlines()
    return [p for p in out if p.endswith("/.env.example") or p == ".env.example"]


_KNOWN_SECRET_PATTERNS = {
    # OpenAI keys (older and newer formats tend to start with sk-)
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9]{10,}\b"),
    # GitHub tokens
    "github_token": re.compile(r"\bghp_[A-Za-z0-9]{20,}\b|\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    # AWS access keys
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    # Google API keys
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z\-_]{20,}\b"),
    # Slack tokens
    "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
}

# Variables where non-empty values are very likely to be secrets.
_SENSITIVE_VAR_NAME = re.compile(r"(API_KEY|TOKEN|SECRET|PASSWORD)", re.IGNORECASE)

# Safe placeholder values we allow in templates.
_PLACEHOLDER_VALUES = {
    "",
    "YOUR_KEY_HERE",
    "YOUR_API_KEY_HERE",
    "YOUR_TOKEN_HERE",
    "CHANGEME",
    "REPLACE_ME",
    "EXAMPLE",
    "EXAMPLE_KEY",
    "XXX",
    "<YOUR_KEY_HERE>",
    "<YOUR_API_KEY_HERE>",
    "<YOUR_TOKEN_HERE>",
    "TODO",
}


def _looks_like_real_secret_value(value: str) -> bool:
    v = value.strip().strip('"').strip("'").strip()
    if v in _PLACEHOLDER_VALUES:
        return False
    # Too short to be a real key; allow.
    if len(v) < 12:
        return False
    # If it matches a known secret pattern, treat as real.
    for pat in _KNOWN_SECRET_PATTERNS.values():
        if pat.search(v):
            return True
    # Heuristic: long-ish non-placeholder values assigned to sensitive vars are likely real.
    return True


def scan_env_example(path: str) -> List[Finding]:
    findings: List[Finding] = []
    text = Path(path).read_text(errors="replace").splitlines()

    kv = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$")

    for i, line in enumerate(text, start=1):
        if not line or line.lstrip().startswith("#"):
            continue

        # Pattern scan (line-level). Do NOT print the line.
        for name, pat in _KNOWN_SECRET_PATTERNS.items():
            if pat.search(line):
                findings.append(Finding(path=path, line_no=i, kind="pattern", detail=name))

        m = kv.match(line)
        if not m:
            continue

        var, value = m.group(1), m.group(2)
        if _SENSITIVE_VAR_NAME.search(var) and _looks_like_real_secret_value(value):
            findings.append(Finding(path=path, line_no=i, kind="assignment", detail=var))

    return findings


def main(argv: List[str]) -> int:
    if "--help" in argv or "-h" in argv:
        print("Usage: python3 scripts/secret_guardrail.py")
        return 0

    try:
        root = git_root()
    except Exception as e:
        print(f"[secret-guardrail] ERROR: not a git repo? {e}", file=sys.stderr)
        return 2

    files = tracked_env_example_files()
    if not files:
        print("[secret-guardrail] No tracked .env.example files found (OK).")
        return 0

    all_findings: List[Finding] = []
    for p in files:
        all_findings.extend(scan_env_example(str(root / p)))

    if not all_findings:
        print(f"[secret-guardrail] OK: scanned {len(files)} tracked .env.example file(s), no secrets detected.")
        return 0

    print("[secret-guardrail] BLOCKED: potential secret(s) detected in tracked .env.example template(s):")
    for f in all_findings:
        # Never print values; only metadata.
        print(f"  - {Path(f.path).relative_to(root)}:{f.line_no}  {f.kind} ({f.detail})")
    print("")
    print("Fix: keep .env.example templates empty/placeholders (e.g. OPENAI_API_KEY=) and put real keys in .env (gitignored).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

