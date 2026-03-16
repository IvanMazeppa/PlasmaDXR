"""Deterministic repair intent routing.

Runs BEFORE any modification path and authoritatively decides between
modify_params, modify_code, switch_technique, or request_guidance.
Zero LLM cost — pure keyword/heuristic logic.
"""

from __future__ import annotations

import re
from typing import List

from models.pipeline_models import QualityIssue, QualityOutput, RepairIntent


# ---------------------------------------------------------------------------
# Keyword sets
# ---------------------------------------------------------------------------

STRUCTURAL_KEYWORDS: List[str] = [
    "wrong order", "wrong color", "missing", "geometry", "shape",
    "stripe", "structure", "arrangement", "layout", "topology",
    "wrong material", "wrong object", "missing object", "scene setup",
    "incorrect mesh", "uv", "orientation", "reversed", "flipped",
    "wrong position", "wrong scale", "wrong rotation",
    "scene composition", "object placement", "mesh topology",
]

PARAMETRIC_KEYWORDS: List[str] = [
    "too dark", "too bright", "too fast", "too slow",
    "density", "intensity", "opacity", "resolution",
    "temperature", "strength", "scale", "speed",
    "noise", "turbulence", "damping", "viscosity",
    "exposure", "lighting", "brightness", "contrast",
    "saturation", "color balance",
]


def _count_keyword_hits(text: str, keywords: List[str]) -> int:
    """Count how many keywords appear in text (case-insensitive)."""
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def _collect_issue_text(quality: QualityOutput) -> str:
    """Gather all issue text from a QualityOutput into a single string."""
    parts = []
    if quality.primary_issue:
        parts.append(quality.primary_issue)
    parts.extend(quality.issues)
    parts.extend(quality.suggestions)
    if quality.vision_assessment:
        parts.append(quality.vision_assessment)
    return " ".join(parts)


def _classify_structured_issues(
    issues: List[QualityIssue],
) -> tuple[int, int, List[str]]:
    """Count structural vs parametric structured issues.

    Returns (structural_count, parametric_count, target_sections).
    """
    structural = 0
    parametric = 0
    sections: List[str] = []
    for issue in issues:
        kind = issue.kind.lower() if issue.kind else ""
        if kind in ("structural", "camera", "lighting", "materials", "environment", "composition"):
            # These all require code changes (adding/moving objects, shader edits, scene setup)
            structural += 1
            if issue.target:
                sections.append(issue.target)
        elif kind == "parameter":
            parametric += 1
    return structural, parametric, sections


def choose_repair_intent(
    quality: QualityOutput,
    code_grounded_feedback: str,
    plateau_count: int,
    same_issue_count: int,
    iteration: int,
    escape_level: int = 0,
) -> RepairIntent:
    """Choose a repair strategy deterministically based on quality feedback.

    Priority chain (first match wins):
    1. Execution failure → modify_code
    2. Structured issues (authoritative when present):
       - structural majority → modify_code
       - technique majority → switch_technique
       - parametric majority/equal → modify_params
    3. Keyword scan (fallback when no structured_issues):
       structural > parametric → modify_code
    4. Plateau (>=2 consecutive stagnant iterations) → modify_code
    5. Same issue repeated >=3 times → switch_technique
    6. Escape level >=4 → request_guidance; >=2 → switch_technique
    7. Default → modify_params
    """
    # Combine all text for keyword scanning
    all_text = _collect_issue_text(quality)
    if code_grounded_feedback:
        all_text += " " + code_grounded_feedback

    # --- 1. Execution failure ---
    exec_failure = (
        quality.primary_issue is not None
        and quality.primary_issue.startswith("Execution failed:")
    )
    if exec_failure:
        return RepairIntent(
            mode="modify_code",
            trigger="execution_failure",
            confidence=0.95,
            reasoning=f"Execution failed: {quality.primary_issue}",
        )

    # --- 2. Structured issues from QA (authoritative when present) ---
    if quality.structured_issues:
        s_count, p_count, sections = _classify_structured_issues(
            quality.structured_issues
        )
        # Count technique-class issues
        t_count = sum(
            1 for i in quality.structured_issues
            if (i.kind or "").lower() == "technique"
        )
        if s_count > p_count:
            return RepairIntent(
                mode="modify_code",
                trigger="structured_issues_structural_majority",
                confidence=0.85,
                target_sections=sections,
                reasoning=f"{s_count} structural vs {p_count} parametric issues",
            )
        if t_count > 0 and t_count >= s_count and t_count >= p_count:
            return RepairIntent(
                mode="switch_technique",
                trigger="structured_issues_technique",
                confidence=0.80,
                reasoning=f"{t_count} technique issues — technique change needed",
            )
        # Parametric majority or equal — structured signal says modify_params
        return RepairIntent(
            mode="modify_params",
            trigger="structured_issues_parametric",
            confidence=0.80,
            reasoning=f"{p_count} parametric vs {s_count} structural issues",
        )

    # --- 3. Keyword scan (fallback when structured_issues absent) ---
    structural_hits = _count_keyword_hits(all_text, STRUCTURAL_KEYWORDS)
    parametric_hits = _count_keyword_hits(all_text, PARAMETRIC_KEYWORDS)
    if structural_hits > 0 and structural_hits >= parametric_hits:
        return RepairIntent(
            mode="modify_code",
            trigger="keyword_structural_majority",
            confidence=min(0.8, 0.5 + structural_hits * 0.1),
            reasoning=f"{structural_hits} structural vs {parametric_hits} parametric keyword hits",
        )

    # --- 4. Plateau ---
    if plateau_count >= 2:
        return RepairIntent(
            mode="modify_code",
            trigger="plateau",
            confidence=0.75,
            reasoning=f"Score plateaued for {plateau_count} consecutive iterations",
        )

    # --- 5. Same issue repeated ---
    if same_issue_count >= 3:
        return RepairIntent(
            mode="switch_technique",
            trigger="same_issue_repeated",
            confidence=0.8,
            reasoning=f"Same issue repeated {same_issue_count} times",
        )

    # --- 6. Escape level ---
    if escape_level >= 4:
        return RepairIntent(
            mode="request_guidance",
            trigger="escape_level_4",
            confidence=0.9,
            reasoning=f"Escape level {escape_level} — requesting human guidance",
        )
    if escape_level >= 2:
        return RepairIntent(
            mode="switch_technique",
            trigger="escape_level_2",
            confidence=0.7,
            reasoning=f"Escape level {escape_level} — switching technique",
        )

    # --- 7. Default ---
    return RepairIntent(
        mode="modify_params",
        trigger="default",
        confidence=0.5,
        reasoning="No structural issues detected; parameter tuning appropriate",
    )
