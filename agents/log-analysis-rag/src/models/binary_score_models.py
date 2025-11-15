"""
Pydantic models for structured LLM outputs
Based on NVIDIA BAT.AI binary scoring with confidence extensions
"""

from pydantic import BaseModel, Field
from typing import Optional


class GradeDocuments(BaseModel):
    """Binary score for document relevance with confidence"""

    binary_score: str = Field(
        description="Document relevance score: 'yes' or 'no'"
    )
    confidence: float = Field(
        description="Confidence in grading decision (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Why this document is/isn't relevant"
    )


class GradeGeneration(BaseModel):
    """Score for generation quality and hallucination detection"""

    binary_score: str = Field(
        description="Generation quality: 'yes' (grounded) or 'no' (hallucination)"
    )
    confidence: float = Field(
        description="Confidence in diagnosis (0.0-1.0)",
        ge=0.0,
        le=1.0
    )


class DiagnosisOutput(BaseModel):
    """Structured diagnostic output for PlasmaDX"""

    diagnosis: str = Field(
        description="Clear diagnostic explanation of the issue"
    )
    confidence: float = Field(
        description="Confidence in diagnosis (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    recommended_specialist: Optional[str] = Field(
        default=None,
        description="Which specialist agent should handle this (e.g., 'rt-shadow-engineer')"
    )
    file_line_refs: list[str] = Field(
        default_factory=list,
        description="Code locations related to issue (e.g., 'particle_physics.hlsl:42')"
    )
    artifact_paths: list[str] = Field(
        default_factory=list,
        description="Related artifacts (screenshots, PIX captures, buffer dumps)"
    )
