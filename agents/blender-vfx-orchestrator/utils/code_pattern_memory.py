"""
Code Pattern Memory for Blender VFX Orchestrator.

Implements Strategy 4: Code Pattern Memory.

Unlike parameter-based knowledge that tracks metadata like "increased flame_smoke
from 2.0 to 3.5", this stores actual working Python code snippets that can be
retrieved semantically and applied to new scripts.

Key capabilities:
- Store successful code patterns with context
- Semantic retrieval based on issue description
- Track pattern effectiveness (usage count, success rate)
- Deduplicate similar patterns
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class CodePattern:
    """A successful code pattern extracted from an experiment."""

    # Required fields (no defaults) - must come first
    pattern_id: str
    name: str
    issue_category: str
    code_snippet: str

    # Optional fields with defaults
    effect_types: List[str] = field(default_factory=list)
    context_before: str = ""  # Lines before the change for context
    context_after: str = ""   # Lines after the change for context
    parameters_affected: List[str] = field(default_factory=list)
    blender_apis_used: List[str] = field(default_factory=list)

    # Effectiveness tracking
    average_improvement: float = 0.0
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0

    # Provenance
    source_experiments: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used_at: str = ""

    @property
    def success_rate(self) -> float:
        """Calculate success rate from usage history."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    @property
    def confidence(self) -> float:
        """
        Calculate confidence score (0-100) based on:
        - Success rate
        - Usage count (more data = more confidence)
        - Average improvement
        """
        if self.usage_count == 0:
            return 50.0  # Neutral confidence for new patterns

        # Base confidence from success rate
        base = self.success_rate * 60

        # Bonus for more data (up to +20 for 10+ uses)
        usage_bonus = min(20, self.usage_count * 2)

        # Bonus for high improvement (up to +20 for 20+ point improvement)
        improvement_bonus = min(20, self.average_improvement)

        score = min(100, base + usage_bonus + improvement_bonus)

        # Apply 20% decay if unused for 30+ days
        if self.last_used_at:
            try:
                last_used = datetime.fromisoformat(self.last_used_at)
                days_idle = (datetime.now() - last_used).days
                if days_idle > 30:
                    decay = 0.2 * min(days_idle / 30, 3)  # Max 60% decay at 90+ days
                    score *= (1.0 - decay)
            except (ValueError, TypeError):
                pass  # Invalid date, skip decay

        return score

    def to_embedding_text(self) -> str:
        """Generate text representation for vector embedding."""
        return f"""
Issue Category: {self.issue_category}
Effect Types: {', '.join(self.effect_types)}
Parameters Affected: {', '.join(self.parameters_affected)}
Blender APIs: {', '.join(self.blender_apis_used)}

Code Pattern:
{self.code_snippet}

Context:
{self.context_before}
[CHANGE]
{self.context_after}
"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodePattern":
        """Create CodePattern from dictionary."""
        return cls(**data)


# Issue category mappings for normalization
ISSUE_CATEGORIES = {
    # Density issues
    "smoke too thin": "density_insufficient",
    "not enough smoke": "density_insufficient",
    "smoke dissipates too fast": "density_insufficient",
    "low density": "density_insufficient",
    "smoke too thick": "density_excessive",
    "too much smoke": "density_excessive",

    # Movement issues
    "smoke rises too fast": "velocity_excessive",
    "smoke moves too slow": "velocity_insufficient",
    "no turbulence": "turbulence_insufficient",
    "turbulence too strong": "turbulence_excessive",

    # Shape issues
    "smoke blob": "shape_too_uniform",
    "lacks detail": "detail_insufficient",
    "too noisy": "detail_excessive",

    # Color/emission issues
    "color wrong": "color_incorrect",
    "too bright": "emission_excessive",
    "too dark": "emission_insufficient",

    # Temporal issues
    "flickering": "temporal_instability",
    "pops": "temporal_instability",
    "jerky motion": "temporal_instability",
}


class CodePatternMemory:
    """
    Persistent storage for successful code patterns.

    Unlike parameter-based knowledge, this stores actual working code
    that can be retrieved semantically and applied to new scripts.

    Storage structure:
    - data/code_patterns/patterns.json: Pattern index
    - data/code_patterns/{pattern_id}.json: Individual pattern files
    """

    def __init__(self, storage_dir: str = "data/code_patterns"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.patterns: Dict[str, CodePattern] = {}
        self._load_patterns()

    def _load_patterns(self) -> None:
        """Load all patterns from storage."""
        index_file = self.storage_dir / "patterns_index.json"
        if not index_file.exists():
            return

        try:
            with open(index_file) as f:
                index = json.load(f)

            for pattern_id in index.get("patterns", []):
                pattern_file = self.storage_dir / f"{pattern_id}.json"
                if pattern_file.exists():
                    with open(pattern_file) as f:
                        data = json.load(f)
                    self.patterns[pattern_id] = CodePattern.from_dict(data)
        except Exception as e:
            print(f"[CodePatternMemory] Error loading patterns: {e}")

    def _save_pattern(self, pattern: CodePattern) -> None:
        """Save a single pattern to storage."""
        pattern.updated_at = datetime.now().isoformat()

        # Save individual pattern file
        pattern_file = self.storage_dir / f"{pattern.pattern_id}.json"
        with open(pattern_file, "w") as f:
            json.dump(pattern.to_dict(), f, indent=2)

        # Update index
        self._save_index()

    def _save_index(self) -> None:
        """Save the pattern index."""
        index_file = self.storage_dir / "patterns_index.json"
        index = {
            "patterns": list(self.patterns.keys()),
            "count": len(self.patterns),
            "updated_at": datetime.now().isoformat()
        }
        with open(index_file, "w") as f:
            json.dump(index, f, indent=2)

    def _generate_id(self, code_snippet: str, issue: str) -> str:
        """Generate a unique pattern ID from content."""
        content = f"{code_snippet}:{issue}"
        hash_val = hashlib.sha256(content.encode()).hexdigest()[:12]
        return f"pat_{hash_val}"

    def _categorize_issue(self, issue: str) -> str:
        """Normalize issue description to standard category."""
        issue_lower = issue.lower()

        # Check for exact or partial matches
        for pattern, category in ISSUE_CATEGORIES.items():
            if pattern in issue_lower:
                return category

        # Default: use the issue as-is but sanitized
        sanitized = re.sub(r'[^a-z0-9_]', '_', issue_lower)
        return sanitized[:50]

    def _extract_blender_apis(self, code: str) -> List[str]:
        """Extract Blender API references from code."""
        apis = set()

        # Match bpy.types.*, bpy.ops.*, bpy.data.*, bpy.context.*
        patterns = [
            r'bpy\.types\.\w+',
            r'bpy\.ops\.\w+\.\w+',
            r'bpy\.data\.\w+',
            r'bpy\.context\.\w+',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, code)
            apis.update(matches)

        return list(apis)

    def _extract_parameters(self, code: str) -> List[str]:
        """Extract parameter names from code (assignments to properties)."""
        params = set()

        # Match property assignments like: obj.flame_smoke = 2.5
        # or domain.dissolve_speed = 10
        pattern = r'\.(\w+)\s*='
        matches = re.findall(pattern, code)

        # Filter out common non-parameter assignments
        excluded = {'name', 'data', 'type', 'location', 'rotation', 'scale'}
        params.update(m for m in matches if m not in excluded)

        return list(params)

    def _find_similar(self, code_snippet: str, threshold: float = 0.8) -> Optional[CodePattern]:
        """
        Find an existing pattern with similar code.

        Uses simple similarity based on:
        - Same Blender APIs used
        - Same parameters affected
        - Similar code structure
        """
        new_apis = set(self._extract_blender_apis(code_snippet))
        new_params = set(self._extract_parameters(code_snippet))

        for pattern in self.patterns.values():
            existing_apis = set(pattern.blender_apis_used)
            existing_params = set(pattern.parameters_affected)

            # Calculate Jaccard similarity for APIs and params
            api_sim = len(new_apis & existing_apis) / max(1, len(new_apis | existing_apis))
            param_sim = len(new_params & existing_params) / max(1, len(new_params | existing_params))

            # Average similarity
            avg_sim = (api_sim + param_sim) / 2

            if avg_sim >= threshold:
                return pattern

        return None

    def record_successful_pattern(
        self,
        issue: str,
        code_snippet: str,
        effect_type: str,
        improvement: float,
        experiment_id: str,
        name: Optional[str] = None,
        context_before: str = "",
        context_after: str = ""
    ) -> str:
        """
        Record a new successful code pattern.

        If a similar pattern exists, updates it instead of creating a duplicate.

        Args:
            issue: Description of the issue this pattern fixes
            code_snippet: The actual Python code
            effect_type: Type of effect (pyro, explosion, fire, etc.)
            improvement: Score improvement achieved
            experiment_id: ID of the experiment this came from
            name: Optional human-readable name for the pattern
            context_before: Code context before the change
            context_after: Code context after the change

        Returns:
            pattern_id of the stored/updated pattern
        """
        # Check for similar existing pattern
        existing = self._find_similar(code_snippet)

        if existing:
            # Update existing pattern
            existing.usage_count += 1
            existing.success_count += 1
            existing.source_experiments.append(experiment_id)

            # Update running average improvement
            n = existing.usage_count
            existing.average_improvement = (
                (existing.average_improvement * (n - 1) + improvement) / n
            )

            # Add effect type if new
            if effect_type not in existing.effect_types:
                existing.effect_types.append(effect_type)

            self._save_pattern(existing)
            return existing.pattern_id

        # Create new pattern
        pattern_id = self._generate_id(code_snippet, issue)
        issue_category = self._categorize_issue(issue)

        # Auto-generate name if not provided
        if not name:
            name = f"{issue_category}_{effect_type}_fix"

        pattern = CodePattern(
            pattern_id=pattern_id,
            name=name,
            issue_category=issue_category,
            code_snippet=code_snippet,
            effect_types=[effect_type],
            context_before=context_before,
            context_after=context_after,
            parameters_affected=self._extract_parameters(code_snippet),
            blender_apis_used=self._extract_blender_apis(code_snippet),
            average_improvement=improvement,
            usage_count=1,
            success_count=1,
            failure_count=0,
            source_experiments=[experiment_id]
        )

        self.patterns[pattern_id] = pattern
        self._save_pattern(pattern)

        return pattern_id

    def record_pattern_outcome(
        self,
        pattern_id: str,
        success: bool,
        improvement: float = 0.0
    ) -> bool:
        """
        Record the outcome of applying a pattern.

        Args:
            pattern_id: ID of the pattern that was applied
            success: Whether the application was successful
            improvement: Score improvement achieved (if successful)

        Returns:
            True if pattern was found and updated
        """
        if pattern_id not in self.patterns:
            return False

        pattern = self.patterns[pattern_id]
        pattern.usage_count += 1

        if success:
            pattern.success_count += 1
            # Update running average improvement
            n = pattern.success_count
            pattern.average_improvement = (
                (pattern.average_improvement * (n - 1) + improvement) / n
            )
        else:
            pattern.failure_count += 1

        self._save_pattern(pattern)
        return True

    def retrieve_patterns_for_issue(
        self,
        issue: str,
        effect_type: Optional[str] = None,
        min_confidence: float = 50.0,
        max_results: int = 5
    ) -> List[CodePattern]:
        """
        Retrieve patterns that might fix a given issue.

        Args:
            issue: Description of the issue to fix
            effect_type: Optional filter by effect type
            min_confidence: Minimum confidence threshold (0-100)
            max_results: Maximum number of patterns to return

        Returns:
            List of matching patterns, sorted by confidence
        """
        issue_category = self._categorize_issue(issue)
        issue_lower = issue.lower()

        candidates = []

        for pattern in self.patterns.values():
            # Filter by confidence
            if pattern.confidence < min_confidence:
                continue

            # Filter by effect type if specified
            if effect_type and effect_type not in pattern.effect_types:
                continue

            # Score based on issue match
            score = 0

            # Exact category match
            if pattern.issue_category == issue_category:
                score += 50

            # Partial issue text match
            if any(word in pattern.issue_category for word in issue_lower.split()):
                score += 20

            # Add confidence as part of score
            score += pattern.confidence * 0.3

            if score > 0:
                candidates.append((score, pattern))

        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)

        return [p for _, p in candidates[:max_results]]

    def retrieve_patterns_by_api(
        self,
        api_name: str,
        min_confidence: float = 50.0
    ) -> List[CodePattern]:
        """
        Retrieve patterns that use a specific Blender API.

        Args:
            api_name: Blender API to search for (e.g., "bpy.types.FluidDomainSettings")
            min_confidence: Minimum confidence threshold

        Returns:
            List of matching patterns
        """
        api_lower = api_name.lower()

        matches = []
        for pattern in self.patterns.values():
            if pattern.confidence < min_confidence:
                continue

            if any(api_lower in api.lower() for api in pattern.blender_apis_used):
                matches.append(pattern)

        # Sort by confidence
        matches.sort(key=lambda p: p.confidence, reverse=True)
        return matches

    def get_pattern(self, pattern_id: str) -> Optional[CodePattern]:
        """Get a specific pattern by ID."""
        return self.patterns.get(pattern_id)

    def list_patterns(
        self,
        effect_type: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> List[CodePattern]:
        """
        List all patterns, optionally filtered.

        Args:
            effect_type: Optional filter by effect type
            min_confidence: Minimum confidence threshold

        Returns:
            List of patterns sorted by confidence
        """
        patterns = []

        for pattern in self.patterns.values():
            if pattern.confidence < min_confidence:
                continue
            if effect_type and effect_type not in pattern.effect_types:
                continue
            patterns.append(pattern)

        patterns.sort(key=lambda p: p.confidence, reverse=True)
        return patterns

    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics about the pattern library."""
        if not self.patterns:
            return {
                "total_patterns": 0,
                "categories": {},
                "effect_types": {},
                "average_confidence": 0,
                "total_usage": 0
            }

        categories: Dict[str, int] = {}
        effect_types: Dict[str, int] = {}
        total_confidence = 0
        total_usage = 0

        for pattern in self.patterns.values():
            # Count categories
            cat = pattern.issue_category
            categories[cat] = categories.get(cat, 0) + 1

            # Count effect types
            for et in pattern.effect_types:
                effect_types[et] = effect_types.get(et, 0) + 1

            total_confidence += pattern.confidence
            total_usage += pattern.usage_count

        return {
            "total_patterns": len(self.patterns),
            "categories": categories,
            "effect_types": effect_types,
            "average_confidence": total_confidence / len(self.patterns),
            "total_usage": total_usage,
            "high_confidence_patterns": sum(1 for p in self.patterns.values() if p.confidence >= 70)
        }


# Global instance (lazy initialization)
_pattern_memory: Optional[CodePatternMemory] = None


def get_pattern_memory(storage_dir: str = "data/code_patterns") -> CodePatternMemory:
    """Get or create the global CodePatternMemory instance."""
    global _pattern_memory
    if _pattern_memory is None:
        _pattern_memory = CodePatternMemory(storage_dir)
    return _pattern_memory
