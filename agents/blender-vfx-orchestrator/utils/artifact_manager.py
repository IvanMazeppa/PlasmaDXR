"""
Artifact Manager: File-based handoffs between agents.

Instead of passing full data inline in prompts, agents write artifacts to disk
and pass file references. This reduces context bloat and clarifies information flow.

Key principle: Agents hand off FILE REFERENCES, not inline dumps.

Artifact Types:
- research.json: Research findings from Phase 0
- quality.json: Quality evaluation from Phase 3
- iteration_{n}.json: Per-iteration state snapshot
- scorecard.json: Quality scorecard with deterministic metrics
- manifest.json: Run manifest with inputs/outputs/decisions

Usage:
    manager = ArtifactManager(session_id)

    # Write artifact
    path = manager.write_research(research_output)

    # Reference in prompt
    prompt = f"Research findings are at: {path}"

    # Agent reads with Read tool when needed
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Base directory for artifacts
ARTIFACTS_DIR = Path(__file__).parent.parent / "sessions" / "artifacts"


@dataclass
class ResearchArtifact:
    """Research findings from Phase 0."""
    recommended_approach: str
    key_parameters: Dict[str, Any]
    api_modules: List[str]
    warnings: List[str]
    alternative_approaches: List[str]
    doc_refs: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class QualityArtifact:
    """Quality evaluation from Phase 3."""
    overall_score: float
    passed: bool
    primary_issue: Optional[str]
    issues: List[str]
    suggestions: List[str]
    vision_assessment: str
    reference_similarity: Optional[float]
    render_path: Optional[str]
    iteration: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class IterationArtifact:
    """Per-iteration state snapshot."""
    iteration: int
    script_path: str
    technique_used: str
    render_path: Optional[str]
    cache_path: Optional[str]
    score: float
    passed: bool
    primary_issue: Optional[str]
    parameter_changes: Dict[str, Any]
    escape_level: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ScorecardArtifact:
    """Quality scorecard with deterministic metrics."""
    session_id: str
    iteration: int

    # Quality metrics
    overall_score: float
    passed: bool
    critical_issues: List[str]
    warnings: List[str]

    # Artifact metrics (from gates)
    cache_size_mb: float
    render_count: int
    vdb_count: int

    # Cost/time metrics
    iteration_cost_usd: float
    iteration_time_seconds: float
    utility_score: float  # SICA utility

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ManifestArtifact:
    """Run manifest with inputs/outputs/decisions."""
    session_id: str
    iteration: int

    # Inputs
    effect_type: str
    description: str
    technique: str
    previous_score: float

    # Outputs
    script_path: str
    render_path: Optional[str]
    cache_path: Optional[str]

    # Decisions
    coordinator_decisions: List[Dict[str, Any]]

    # Environment
    blender_version: str = "5.0"
    sdk_version: str = "0.7.0"

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ArtifactManager:
    """
    Manages artifact persistence for a VFX session.

    Artifacts are written to: sessions/artifacts/{session_id}/
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.artifact_dir = ARTIFACTS_DIR / session_id
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def _write_json(self, filename: str, data: Union[dict, dataclass]) -> str:
        """Write JSON artifact and return path."""
        path = self.artifact_dir / filename

        if hasattr(data, '__dataclass_fields__'):
            data = asdict(data)

        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"[Artifacts] Wrote: {path}", file=sys.stderr)
        return str(path)

    def _read_json(self, filename: str) -> Optional[dict]:
        """Read JSON artifact if exists."""
        path = self.artifact_dir / filename
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    # =========================================================================
    # WRITE METHODS
    # =========================================================================

    def write_research(self, research: ResearchArtifact) -> str:
        """Write research findings artifact."""
        return self._write_json("research.json", research)

    def write_research_from_output(
        self,
        recommended_approach: str,
        key_parameters: Dict[str, Any],
        api_modules: List[str],
        warnings: List[str],
        alternative_approaches: List[str],
        doc_refs: List[str],
    ) -> str:
        """Write research from raw values."""
        artifact = ResearchArtifact(
            recommended_approach=recommended_approach,
            key_parameters=key_parameters,
            api_modules=api_modules,
            warnings=warnings,
            alternative_approaches=alternative_approaches,
            doc_refs=doc_refs,
        )
        return self.write_research(artifact)

    def write_quality(self, quality: QualityArtifact) -> str:
        """Write quality evaluation artifact."""
        return self._write_json(f"quality_iter{quality.iteration}.json", quality)

    def write_quality_from_output(
        self,
        iteration: int,
        overall_score: float,
        passed: bool,
        primary_issue: Optional[str],
        issues: List[str],
        suggestions: List[str],
        vision_assessment: str = "",
        reference_similarity: Optional[float] = None,
        render_path: Optional[str] = None,
    ) -> str:
        """Write quality from raw values."""
        artifact = QualityArtifact(
            iteration=iteration,
            overall_score=overall_score,
            passed=passed,
            primary_issue=primary_issue,
            issues=issues,
            suggestions=suggestions,
            vision_assessment=vision_assessment,
            reference_similarity=reference_similarity,
            render_path=render_path,
        )
        return self.write_quality(artifact)

    def write_iteration(self, iteration: IterationArtifact) -> str:
        """Write iteration state artifact."""
        return self._write_json(f"iteration_{iteration.iteration}.json", iteration)

    def write_iteration_from_values(
        self,
        iteration: int,
        script_path: str,
        technique_used: str,
        score: float,
        passed: bool,
        primary_issue: Optional[str] = None,
        render_path: Optional[str] = None,
        cache_path: Optional[str] = None,
        parameter_changes: Optional[Dict[str, Any]] = None,
        escape_level: int = 0,
    ) -> str:
        """Write iteration from raw values."""
        artifact = IterationArtifact(
            iteration=iteration,
            script_path=script_path,
            technique_used=technique_used,
            render_path=render_path,
            cache_path=cache_path,
            score=score,
            passed=passed,
            primary_issue=primary_issue,
            parameter_changes=parameter_changes or {},
            escape_level=escape_level,
        )
        return self.write_iteration(artifact)

    def write_scorecard(self, scorecard: ScorecardArtifact) -> str:
        """Write quality scorecard artifact."""
        return self._write_json(f"scorecard_iter{scorecard.iteration}.json", scorecard)

    def write_scorecard_from_values(
        self,
        iteration: int,
        overall_score: float,
        passed: bool,
        critical_issues: List[str],
        warnings: List[str],
        cache_size_mb: float,
        render_count: int,
        vdb_count: int = 0,
        iteration_cost_usd: float = 0.0,
        iteration_time_seconds: float = 0.0,
        utility_score: float = 0.0,
    ) -> str:
        """Write scorecard from raw values."""
        artifact = ScorecardArtifact(
            session_id=self.session_id,
            iteration=iteration,
            overall_score=overall_score,
            passed=passed,
            critical_issues=critical_issues,
            warnings=warnings,
            cache_size_mb=cache_size_mb,
            render_count=render_count,
            vdb_count=vdb_count,
            iteration_cost_usd=iteration_cost_usd,
            iteration_time_seconds=iteration_time_seconds,
            utility_score=utility_score,
        )
        return self.write_scorecard(artifact)

    def write_manifest(self, manifest: ManifestArtifact) -> str:
        """Write run manifest artifact."""
        return self._write_json(f"manifest_iter{manifest.iteration}.json", manifest)

    def write_manifest_from_values(
        self,
        iteration: int,
        effect_type: str,
        description: str,
        technique: str,
        previous_score: float,
        script_path: str,
        render_path: Optional[str] = None,
        cache_path: Optional[str] = None,
        coordinator_decisions: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Write manifest from raw values."""
        artifact = ManifestArtifact(
            session_id=self.session_id,
            iteration=iteration,
            effect_type=effect_type,
            description=description,
            technique=technique,
            previous_score=previous_score,
            script_path=script_path,
            render_path=render_path,
            cache_path=cache_path,
            coordinator_decisions=coordinator_decisions or [],
        )
        return self.write_manifest(artifact)

    # =========================================================================
    # READ METHODS
    # =========================================================================

    def get_research_path(self) -> Optional[str]:
        """Get path to research artifact if exists."""
        path = self.artifact_dir / "research.json"
        return str(path) if path.exists() else None

    def get_quality_path(self, iteration: int) -> Optional[str]:
        """Get path to quality artifact for iteration."""
        path = self.artifact_dir / f"quality_iter{iteration}.json"
        return str(path) if path.exists() else None

    def get_latest_quality_path(self) -> Optional[str]:
        """Get path to most recent quality artifact."""
        quality_files = sorted(self.artifact_dir.glob("quality_iter*.json"))
        return str(quality_files[-1]) if quality_files else None

    def get_iteration_path(self, iteration: int) -> Optional[str]:
        """Get path to iteration artifact."""
        path = self.artifact_dir / f"iteration_{iteration}.json"
        return str(path) if path.exists() else None

    def get_scorecard_path(self, iteration: int) -> Optional[str]:
        """Get path to scorecard artifact."""
        path = self.artifact_dir / f"scorecard_iter{iteration}.json"
        return str(path) if path.exists() else None

    def get_manifest_path(self, iteration: int) -> Optional[str]:
        """Get path to manifest artifact."""
        path = self.artifact_dir / f"manifest_iter{iteration}.json"
        return str(path) if path.exists() else None

    def read_research(self) -> Optional[dict]:
        """Read research artifact."""
        return self._read_json("research.json")

    def read_quality(self, iteration: int) -> Optional[dict]:
        """Read quality artifact for iteration."""
        return self._read_json(f"quality_iter{iteration}.json")

    def read_iteration(self, iteration: int) -> Optional[dict]:
        """Read iteration artifact."""
        return self._read_json(f"iteration_{iteration}.json")

    # =========================================================================
    # SUMMARY METHODS
    # =========================================================================

    def get_iteration_summary(self, max_iterations: int = 5) -> str:
        """
        Get a summary of recent iterations for context.

        Returns a compact text summary suitable for prompts.
        """
        lines = ["## Recent Iterations"]

        iteration_files = sorted(self.artifact_dir.glob("iteration_*.json"))[-max_iterations:]

        for f in iteration_files:
            data = json.loads(f.read_text())
            issue = data.get('primary_issue') or 'None'
            lines.append(
                f"- Iter {data['iteration']}: score={data['score']:.1f}, "
                f"technique={data['technique_used']}, "
                f"issue={issue[:40] if issue else 'None'}"
            )

        if not iteration_files:
            lines.append("- No previous iterations")

        return "\n".join(lines)

    def get_artifact_paths_summary(self) -> str:
        """
        Get a summary of available artifact paths.

        Useful for including in prompts so agents know what's available.
        """
        lines = ["## Available Artifacts"]

        research_path = self.get_research_path()
        if research_path:
            lines.append(f"- Research: {research_path}")

        quality_path = self.get_latest_quality_path()
        if quality_path:
            lines.append(f"- Latest Quality: {quality_path}")

        iteration_files = sorted(self.artifact_dir.glob("iteration_*.json"))
        if iteration_files:
            lines.append(f"- Iterations: {len(iteration_files)} available")
            lines.append(f"  Latest: {iteration_files[-1]}")

        scorecard_files = sorted(self.artifact_dir.glob("scorecard_*.json"))
        if scorecard_files:
            lines.append(f"- Scorecards: {len(scorecard_files)} available")

        return "\n".join(lines)


def get_artifact_manager(session_id: str) -> ArtifactManager:
    """Get or create artifact manager for session."""
    return ArtifactManager(session_id)
