"""
Pydantic Models for Structured Agent Outputs.

Provides type-safe, validated output types for the blender-librarian agents,
replacing manual JSON string parsing with automatic validation and IDE support.

Key Benefits:
- Automatic validation (confidence 0-1, severity enum, etc.)
- IDE autocomplete and type hints
- Eliminates JSON parse errors
- Self-documenting schemas via Field descriptions
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ModificationAdvice(BaseModel):
    """
    Structured output for documentation expert agent.

    Used by get_modification_advice tool and search_docs_with_agents_sdk
    to provide type-safe parameter recommendations.
    """
    answer: str = Field(
        description="1-2 sentence summary of what was found in documentation"
    )
    modifications_json: str = Field(
        default="{}",
        description="JSON string of parameter changes, e.g. '{\"flame_smoke\": 2.5, \"turbulence\": 0.4}'"
    )
    rationale: str = Field(
        description="Why these changes should help based on documentation"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in recommendation (0.0-1.0)"
    )
    citations: List[str] = Field(
        default_factory=list,
        description="Documentation paths used as sources"
    )

    @property
    def modifications(self) -> Dict[str, float]:
        """Parse modifications_json to dict for backward compatibility."""
        import json
        try:
            return json.loads(self.modifications_json)
        except (json.JSONDecodeError, TypeError):
            return {}

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Increasing flame_smoke ratio adds more smoke to the simulation.",
                "modifications_json": "{\"flame_smoke\": 2.5, \"turbulence\": 0.4}",
                "rationale": "Higher flame_smoke creates denser smoke trails which improves solar prominence appearance.",
                "confidence": 0.85,
                "citations": ["physics/fluid/type/flow.html", "physics/fluid/settings.html"]
            }
        }


class RenderDiagnosis(BaseModel):
    """
    Structured output for vision expert agent.

    Used by diagnose_render_issue tool to provide type-safe
    visual quality assessments.
    """
    diagnosis: str = Field(
        description="Primary issue description (1-2 sentences)"
    )
    primary_issue: str = Field(
        description="Issue category name (e.g., 'color_too_cool', 'missing_limb_darkening')"
    )
    severity: str = Field(
        pattern="^(critical|high|medium|low)$",
        description="Severity level: critical, high, medium, or low"
    )
    secondary_issues: List[str] = Field(
        default_factory=list,
        description="Additional issues found during analysis"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Actionable recommendations to fix the issues"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in diagnosis (0.0-1.0)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "diagnosis": "The render shows insufficient limb darkening, causing the sun to appear flat.",
                "primary_issue": "missing_limb_darkening",
                "severity": "high",
                "secondary_issues": ["color_too_uniform", "lacks_surface_texture"],
                "recommendations": [
                    "Increase limb darkening factor",
                    "Add granulation texture",
                    "Adjust color temperature gradient"
                ],
                "confidence": 0.9
            }
        }


class AgentSearchResult(BaseModel):
    """
    Structured output for Agents SDK documentation search.

    Used by search_docs_with_agents_sdk to combine documentation
    findings with optional vision diagnosis.
    """
    answer: str = Field(
        description="Synthesized answer from documentation search"
    )
    modifications_json: str = Field(
        default="{}",
        description="JSON string of parameter modifications, e.g. '{\"domain_scale_z\": 1.5}'"
    )
    rationale: str = Field(
        description="Why these modifications should help"
    )
    citations: List[str] = Field(
        default_factory=list,
        description="Documentation paths used"
    )
    diagnosis: Optional[RenderDiagnosis] = Field(
        default=None,
        description="Vision-based diagnosis if include_vision was True"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for cross-session learning"
    )
    cost_estimate: Optional[float] = Field(
        default=None,
        description="Estimated API cost for this query"
    )

    @property
    def modifications(self) -> Dict[str, float]:
        """Parse modifications_json to dict for backward compatibility."""
        import json
        try:
            return json.loads(self.modifications_json)
        except (json.JSONDecodeError, TypeError):
            return {}

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "To fix the missing prominences, increase domain height and add more turbulence.",
                "modifications_json": "{\"domain_scale_z\": 1.5, \"turbulence\": 0.6}",
                "rationale": "Prominences need vertical space and chaotic motion to form properly.",
                "citations": ["physics/fluid/type/domain.html"],
                "diagnosis": None,
                "session_id": "session_abc123",
                "cost_estimate": 0.03
            }
        }


class PlaybookEntry(BaseModel):
    """
    Structured entry for the playbook knowledge base.

    Records learned fixes that work for specific symptoms,
    enabling FREE lookups of known solutions.
    """
    effect_type: str = Field(
        description="Effect type this fix applies to (sun, explosion, etc.)"
    )
    symptom: str = Field(
        description="The issue this fix addresses"
    )
    fix: Dict[str, float] = Field(
        description="Parameter modifications that resolved the issue"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        default=0.7,
        description="Confidence that this fix works"
    )
    times_used: int = Field(
        default=0,
        description="Number of times this fix has been applied"
    )
    success_rate: float = Field(
        ge=0.0,
        le=1.0,
        default=1.0,
        description="Success rate when applied"
    )


class BudgetStatus(BaseModel):
    """
    Structured output for budget tracking.

    Reports current API spending against monthly budget limits.
    """
    vision_spent: float = Field(
        ge=0.0,
        description="Amount spent on vision API calls this month"
    )
    doc_spent: float = Field(
        ge=0.0,
        description="Amount spent on documentation search calls this month"
    )
    total_spent: float = Field(
        ge=0.0,
        description="Total amount spent this month"
    )
    monthly_budget: float = Field(
        ge=0.0,
        description="Monthly budget limit"
    )
    remaining: float = Field(
        description="Remaining budget"
    )
    can_afford_vision: bool = Field(
        description="Whether budget allows a vision API call"
    )
    can_afford_doc: bool = Field(
        description="Whether budget allows a documentation search call"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "vision_spent": 1.25,
                "doc_spent": 0.50,
                "total_spent": 1.75,
                "monthly_budget": 10.0,
                "remaining": 8.25,
                "can_afford_vision": True,
                "can_afford_doc": True
            }
        }
