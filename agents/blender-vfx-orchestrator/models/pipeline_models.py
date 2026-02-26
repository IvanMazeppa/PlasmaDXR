"""
Pipeline output models for agent-to-agent data passing.

These Pydantic models define structured outputs for each pipeline phase.
Extracted from orchestrator.py to break circular imports between
orchestrator and phase modules.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


# =============================================================================
# AGENT OUTPUT MODELS
# =============================================================================


class ResearchOutput(BaseModel):
    """Output from Research Agent - provides starting parameters for script generation.

    Phase 3: Structured schema for deterministic research outputs.
    All fields must be populated from actual tool results, not invented.
    """

    @model_validator(mode="before")
    @classmethod
    def normalize_field_names(cls, data: Any) -> Any:
        """Normalize title-case keys to snake_case.

        LLMs sometimes return keys like "Recommended Approach" instead of
        "recommended_approach" despite the JSON schema specifying snake_case.
        """
        if isinstance(data, dict):
            return {
                key.lower().replace(" ", "_").replace("-", "_"): value
                for key, value in data.items()
            }
        return data

    recommended_approach: str = Field(description="Best approach for the effect type (from documentation)")
    key_parameters: Dict[str, Any] = Field(default_factory=dict, description="Recommended parameter values (from patterns or docs)")
    api_modules: List[str] = Field(default_factory=list, description="Blender API modules to use (e.g., bpy.types.FluidDomainSettings)")
    code_patterns: List[Dict[str, Any]] = Field(default_factory=list, description="Proven patterns from library: [{pattern_id, issue, code_snippet}]")
    warnings: List[str] = Field(default_factory=list, description="Potential pitfalls to avoid (from knowledge base)")
    alternative_approaches: List[str] = Field(default_factory=list, description="Backup approaches if primary fails")
    doc_refs: List[str] = Field(default_factory=list, description="Blender 5.0 documentation references used (URLs or section names)")


class ScriptOutput(BaseModel):
    """Output from Script Writer - path to generated/modified script."""
    script_path: str = Field(description="Absolute path to the generated script")
    technique_used: str = Field(description="Technique/approach used in script generation")
    parameters_set: Dict[str, Any] = Field(default_factory=dict, description="Key parameters configured")
    validation_passed: bool = Field(default=True, description="Whether script validation passed")
    validation_errors: List[Any] = Field(default_factory=list, description="Any validation errors (strings or structured dicts)")


class ExecutionOutput(BaseModel):
    """Output from Executor - render path and execution status."""
    success: bool = Field(description="Whether Blender execution succeeded")
    render_path: Optional[str] = Field(default=None, description="Path to rendered output")
    vdb_path: Optional[str] = Field(default=None, description="Path to VDB volume data")
    run_dir: Optional[str] = Field(default=None, description="Executor output directory containing cache and logs")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time_seconds: float = Field(default=0.0, description="Total execution time")


class QualityOutput(BaseModel):
    """Output from Quality Analyst - evaluation results with feedback."""
    overall_score: float = Field(description="Quality score 0-100")
    passed: bool = Field(description="Whether quality threshold was met")
    primary_issue: Optional[str] = Field(default=None, description="Most critical issue to fix")
    issues: List[str] = Field(default_factory=list, description="All identified issues")
    suggestions: List[str] = Field(default_factory=list, description="Specific improvement suggestions")
    vision_assessment: str = Field(default="", description="Detailed visual quality description")
    reference_similarity: Optional[float] = Field(default=None, description="Similarity to reference (if available)")


class LearningOutput(BaseModel):
    """Output from Learning Agent - experiment recorded, next action suggested."""
    experiment_recorded: bool = Field(default=True, description="Whether experiment was logged")
    pattern_extracted: bool = Field(default=False, description="Whether a new pattern was extracted")
    pattern_id: Optional[str] = Field(default=None, description="ID of extracted pattern")
    next_action: str = Field(description="Recommended next action: 'iterate', 'switch_technique', 'complete'")
    suggested_modifications: List[str] = Field(default_factory=list, description="Text descriptions of changes for context")
    parameter_modifications: Dict[str, Any] = Field(
        default_factory=dict,
        description="Concrete parameter changes as {param_name: new_value}, e.g., {'temperature': 3.0, 'density': 5.0}"
    )


# =============================================================================
# COORDINATOR DECISION MODELS (Phase 2: Agents-as-Tools Pattern)
# =============================================================================


class TechniqueDecision(BaseModel):
    """Output from Coordinator for initial technique selection."""
    selected_technique: str = Field(description="The technique to use (e.g., 'mantaflow_fire', 'shader_based_volume')")
    reasoning: str = Field(description="Why this technique was selected")
    key_parameters: Dict[str, Any] = Field(default_factory=dict, description="Starting parameters for the technique")
    alternative_techniques: List[str] = Field(default_factory=list, description="Backup techniques if primary fails")
    research_summary: str = Field(default="", description="Summary of research findings")


class ModificationDecision(BaseModel):
    """Output from Coordinator for parameter modification strategy."""
    action: str = Field(description="Action to take: 'modify_params', 'modify_code', 'switch_technique', 'continue'")
    parameter_changes: Dict[str, Any] = Field(default_factory=dict, description="Concrete parameter changes (for modify_params)")
    code_change_description: Optional[str] = Field(default=None, description="Description of structural code changes needed (for modify_code)")
    new_technique: Optional[str] = Field(default=None, description="New technique if switching")
    reasoning: str = Field(default="See code_change_description or parameter_changes", description="Why this modification strategy was chosen")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in this decision (0.0-1.0)")


class QualityDecision(BaseModel):
    """Output from Coordinator for quality gate interpretation."""
    passed: bool = Field(description="Whether quality gate is passed")
    should_continue: bool = Field(description="Whether to continue iterating")
    next_action: str = Field(description="Next action: 'complete', 'iterate', 'switch_technique', 'request_guidance'")
    escape_level: int = Field(ge=0, le=4, description="Current escape velocity level (0-4)")
    reasoning: str = Field(description="Explanation of the quality decision")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def compute_sica_utility(score: float, cost_usd: float, time_seconds: float) -> float:
    """
    SICA utility function for selecting best iteration.

    U = 0.5 * score_norm + 0.25 * (1 - cost/10) + 0.25 * (1 - time/300)
    score_norm is score / 100.
    """
    score_norm = max(0.0, min(1.0, score / 100.0))
    cost_component = 1.0 - min(1.0, max(0.0, cost_usd) / 10.0)
    time_component = 1.0 - min(1.0, max(0.0, time_seconds) / 300.0)
    utility = 0.5 * score_norm + 0.25 * cost_component + 0.25 * time_component
    return max(0.0, min(1.0, utility))
