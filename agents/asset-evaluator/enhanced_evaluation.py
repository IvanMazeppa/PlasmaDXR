#!/usr/bin/env python3
"""
Enhanced VFX Evaluation Module

Addresses the challenges identified in ML_VISION_EVALUATION_CHALLENGES.md:
1. Domain mismatch (rendered vs. real)
2. CLIP plateau effect
3. No actionable feedback
4. Single-path optimization trap
5. Temporal quality not evaluated
6. Reference image dependency

Solutions implemented:
- Multi-prompt CLIP for gradient signal (fine-grained quality)
- LAION Aesthetics for reference-free scoring
- Moondream VLM for actionable diagnostics
- Attribute-specific analyzers for VFX properties
- ImageReward for text-image alignment
"""

import json
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

# Lazy-loaded models
_aesthetic_model = None
_aesthetic_linear = None
_image_reward_model = None
_moondream_model = None
_moondream_tokenizer = None


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class VFXDiagnostics:
    """Structured diagnostics for VFX quality."""
    color_temperature: str  # "too_cool", "good", "too_warm"
    density: str            # "too_sparse", "good", "too_dense"
    shape: str              # "amorphous", "good", "needs_more_definition"
    brightness: str         # "too_dim", "good", "too_bright"
    smoke_ratio: str        # "too_little", "good", "too_much"
    contrast: str           # "too_low", "good", "too_high"
    detail_level: str       # "too_smooth", "good", "too_noisy"
    raw_analysis: str       # VLM raw output for debugging


@dataclass
class SuggestedParameters:
    """Blender parameter suggestions based on diagnostics."""
    flame_max_temp: Optional[str] = None
    flame_smoke: Optional[str] = None
    vorticity: Optional[str] = None
    resolution: Optional[str] = None
    dissolve_speed: Optional[str] = None
    buoyancy: Optional[str] = None
    turbulence: Optional[str] = None
    heat: Optional[str] = None
    density: Optional[str] = None


@dataclass
class MultiPromptCLIPResult:
    """Fine-grained CLIP scores across quality dimensions."""
    overall_match: float          # Standard CLIP score
    quality_dimension_scores: Dict[str, float]  # e.g., {"intensity": 0.72, "realism": 0.65}
    gradient_signal: float        # Composite score with more range
    best_matching_description: str
    worst_matching_description: str


@dataclass
class EnhancedEvaluationResult:
    """Comprehensive evaluation with diagnostics and suggestions."""
    # Core scores
    passed: bool
    overall_score: float

    # Individual metrics
    lpips_score: Optional[float]
    clip_score: Optional[float]
    aesthetic_score: Optional[float]
    image_reward_score: Optional[float]
    multi_prompt_gradient: Optional[float]

    # Diagnostics
    diagnostics: Optional[VFXDiagnostics]
    suggested_parameters: Optional[SuggestedParameters]

    # Metadata
    render_path: str
    reference_path: Optional[str]
    semantic_query: Optional[str]
    recommendations: List[str]


# =============================================================================
# Model Loaders (Lazy)
# =============================================================================

def get_aesthetic_model():
    """Load LAION Aesthetics V2 predictor."""
    global _aesthetic_model, _aesthetic_linear
    if _aesthetic_model is None:
        try:
            import torch
            import clip
            from PIL import Image

            # Load CLIP for embeddings
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _aesthetic_model, _ = clip.load("ViT-L/14", device=device)

            # Load aesthetic linear head
            # The aesthetic predictor is a linear layer on top of CLIP embeddings
            # Weights from: https://github.com/LAION-AI/aesthetic-predictor
            aesthetic_path = Path(__file__).parent / "weights" / "sa_0_4_vit_l_14_linear.pth"
            if aesthetic_path.exists():
                state_dict = torch.load(aesthetic_path, map_location=device)
                _aesthetic_linear = torch.nn.Linear(768, 1).to(device)
                _aesthetic_linear.load_state_dict(state_dict)
                _aesthetic_linear.eval()
            else:
                # Fallback: use simple heuristic if weights not available
                _aesthetic_linear = None
                print(f"Aesthetic weights not found at {aesthetic_path}, using fallback")

        except ImportError as e:
            raise RuntimeError(f"Failed to load aesthetic model: {e}")
    return _aesthetic_model, _aesthetic_linear


def get_image_reward_model():
    """Load ImageReward model for text-image quality."""
    global _image_reward_model
    if _image_reward_model is None:
        try:
            import ImageReward as RM
            _image_reward_model = RM.load("ImageReward-v1.0")
        except ImportError:
            print("ImageReward not installed. Run: pip install image-reward")
            return None
    return _image_reward_model


def get_moondream_model():
    """Load Moondream VLM for diagnostic analysis."""
    global _moondream_model, _moondream_tokenizer
    if _moondream_model is None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            model_id = "vikhyatk/moondream2"
            revision = "2025-01-09"  # Use specific revision for consistency

            _moondream_tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
            _moondream_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
        except ImportError as e:
            print(f"Moondream dependencies not installed: {e}")
            return None, None
    return _moondream_model, _moondream_tokenizer


# =============================================================================
# Multi-Prompt CLIP Evaluation
# =============================================================================

def evaluate_multi_prompt_clip(
    image_path: str,
    base_description: str,
    effect_type: str = "explosion"
) -> MultiPromptCLIPResult:
    """
    Use graduated prompts to get fine-grained quality signal.

    Instead of single "is this an explosion?" check, we ask:
    - "a faint wisp of smoke" (bad)
    - "a small fire" (poor)
    - "a bright explosion" (good)
    - "a dramatic, intense explosion with flames and debris" (excellent)

    This provides gradient signal for iteration.
    """
    import torch
    import clip
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Load image
    image = preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)

    # Define graduated prompts for different quality levels
    quality_prompts = get_graduated_prompts(effect_type, base_description)

    # Dimension-specific prompts for diagnostic granularity
    dimension_prompts = {
        "intensity": [
            f"a faint, barely visible {effect_type}",
            f"a visible {effect_type}",
            f"a bright, intense {effect_type}",
            f"an extremely bright, dramatic {effect_type}"
        ],
        "realism": [
            f"a fake looking {effect_type}",
            f"a stylized {effect_type}",
            f"a realistic looking {effect_type}",
            f"a photorealistic {effect_type}"
        ],
        "dynamics": [
            f"a static, frozen {effect_type}",
            f"a gently moving {effect_type}",
            f"a dynamic {effect_type} with motion",
            f"an explosive, energetic {effect_type} with rapid motion"
        ],
        "detail": [
            f"a blurry {effect_type}",
            f"a {effect_type} with some detail",
            f"a detailed {effect_type}",
            f"a highly detailed {effect_type} with intricate features"
        ]
    }

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Score each graduated prompt
        quality_scores = []
        for prompt in quality_prompts:
            text_tokens = clip.tokenize([prompt]).to(device)
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).item()
            quality_scores.append((prompt, (similarity + 1) / 2))

        # Score dimension-specific prompts
        dimension_scores = {}
        for dim_name, prompts in dimension_prompts.items():
            dim_score = 0.0
            for i, prompt in enumerate(prompts):
                text_tokens = clip.tokenize([prompt]).to(device)
                text_features = model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (image_features @ text_features.T).item()
                # Weight higher quality prompts more
                weight = (i + 1) / len(prompts)
                dim_score += ((similarity + 1) / 2) * weight
            dimension_scores[dim_name] = round(dim_score / sum((j+1)/len(prompts) for j in range(len(prompts))), 4)

    # Find best/worst matching descriptions
    quality_scores_sorted = sorted(quality_scores, key=lambda x: x[1], reverse=True)
    best_match = quality_scores_sorted[0]
    worst_match = quality_scores_sorted[-1]

    # Calculate gradient signal (weighted average favoring higher quality prompts)
    gradient_signal = sum(
        score * (i + 1) for i, (_, score) in enumerate(sorted(quality_scores, key=lambda x: x[1]))
    ) / sum(i + 1 for i in range(len(quality_scores)))

    # Standard CLIP score with base description
    text_tokens = clip.tokenize([base_description]).to(device)
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    overall_match = (image_features @ text_features.T).item()

    return MultiPromptCLIPResult(
        overall_match=round((overall_match + 1) / 2, 4),
        quality_dimension_scores=dimension_scores,
        gradient_signal=round(gradient_signal, 4),
        best_matching_description=best_match[0],
        worst_matching_description=worst_match[0]
    )


def get_graduated_prompts(effect_type: str, base_description: str) -> List[str]:
    """Generate graduated quality prompts for effect type."""
    prompts = {
        "explosion": [
            "a faint puff of smoke",
            "a small fire with some smoke",
            "a visible explosion with flames",
            "a bright explosion with orange flames and smoke",
            "a dramatic fiery explosion with billowing smoke and debris",
            "an intense, photorealistic explosion with bright flames, dense smoke, and flying debris"
        ],
        "pyro": [
            "a barely visible wisp",
            "a small flame",
            "a visible fire",
            "a bright, active fire",
            "a dramatic fire with smoke and embers",
            "an intense, raging inferno with dynamic flames"
        ],
        "nebula": [
            "a faint smudge of color",
            "a visible cloud of gas",
            "a glowing nebula",
            "a colorful nebula with visible structure",
            "a dramatic nebula with bright cores and tendrils",
            "a stunning, detailed nebula with emission regions and dust lanes"
        ],
        "supernova": [
            "a dim point of light",
            "a bright star",
            "a stellar explosion",
            "a bright supernova with expanding shell",
            "a dramatic supernova with shock waves",
            "an intense supernova explosion with brilliant core and expanding debris"
        ]
    }

    return prompts.get(effect_type, [
        f"a barely visible {base_description}",
        f"a faint {base_description}",
        f"a visible {base_description}",
        f"a {base_description}",
        f"a dramatic {base_description}",
        f"an intense, stunning {base_description}"
    ])


# =============================================================================
# LAION Aesthetics Scoring (Reference-Free)
# =============================================================================

def evaluate_aesthetic_quality(image_path: str) -> float:
    """
    Reference-free aesthetic scoring using LAION Aesthetics V2.

    Returns score from 1-10 (higher = more aesthetic).
    Addresses Challenge 6: Reference Image Dependency
    """
    try:
        import torch
        import clip
        from PIL import Image

        model, linear = get_aesthetic_model()
        if model is None:
            return 5.0  # Neutral fallback

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load and preprocess image
        _, preprocess = clip.load("ViT-L/14", device=device)
        image = preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            if linear is not None:
                # Use trained aesthetic predictor
                aesthetic_score = linear(image_features.float()).item()
            else:
                # Fallback: use simple heuristic based on CLIP similarity
                # to aesthetic concepts
                aesthetic_prompts = [
                    "a beautiful image",
                    "a high quality photograph",
                    "stunning visual effects",
                    "professional CGI rendering"
                ]
                scores = []
                for prompt in aesthetic_prompts:
                    text_tokens = clip.tokenize([prompt]).to(device)
                    text_features = model.encode_text(text_tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    similarity = (image_features @ text_features.T).item()
                    scores.append(similarity)
                aesthetic_score = (sum(scores) / len(scores) + 1) * 5  # Scale to 1-10

        return max(1.0, min(10.0, aesthetic_score))

    except Exception as e:
        print(f"Aesthetic evaluation failed: {e}")
        return 5.0


# =============================================================================
# ImageReward Scoring
# =============================================================================

def evaluate_image_reward(image_path: str, prompt: str) -> float:
    """
    Text-image alignment scoring using ImageReward.

    Trained on human preference data, better than CLIP for quality judgment.
    Returns score typically in range [-2, 2], higher is better.
    """
    try:
        model = get_image_reward_model()
        if model is None:
            return 0.0

        score = model.score(prompt, image_path)
        return float(score)

    except Exception as e:
        print(f"ImageReward evaluation failed: {e}")
        return 0.0


# =============================================================================
# VLM Diagnostic Analysis (Moondream)
# =============================================================================

def analyze_vfx_diagnostics(
    image_path: str,
    effect_type: str = "explosion"
) -> Tuple[VFXDiagnostics, SuggestedParameters]:
    """
    Use Moondream VLM to get actionable diagnostic feedback.

    Addresses Challenge 3: No Actionable Feedback
    Returns structured diagnostics that map to Blender parameters.
    """
    model, tokenizer = get_moondream_model()

    if model is None:
        # Fallback to heuristic analysis
        return _fallback_diagnostics(image_path, effect_type)

    try:
        from PIL import Image

        image = Image.open(image_path).convert('RGB')

        # Structured prompt for diagnostic analysis
        prompt = f"""Analyze this {effect_type} VFX render. For each category, answer with ONLY the specified options:

1. Color temperature (too_cool/good/too_warm):
2. Density (too_sparse/good/too_dense):
3. Shape definition (amorphous/good/well_defined):
4. Brightness (too_dim/good/too_bright):
5. Smoke to fire ratio (too_little_smoke/good/too_much_smoke):
6. Contrast (too_low/good/too_high):
7. Detail level (too_smooth/good/too_noisy):

Format your answer as a simple list."""

        # Encode and generate
        enc_image = model.encode_image(image)
        response = model.answer_question(enc_image, prompt, tokenizer)

        # Parse response
        diagnostics = _parse_vlm_response(response, effect_type)
        suggestions = _generate_parameter_suggestions(diagnostics)

        return diagnostics, suggestions

    except Exception as e:
        print(f"VLM analysis failed: {e}")
        return _fallback_diagnostics(image_path, effect_type)


def _parse_vlm_response(response: str, effect_type: str) -> VFXDiagnostics:
    """Parse VLM response into structured diagnostics."""
    # Default values
    diag = {
        "color_temperature": "good",
        "density": "good",
        "shape": "good",
        "brightness": "good",
        "smoke_ratio": "good",
        "contrast": "good",
        "detail_level": "good",
        "raw_analysis": response
    }

    response_lower = response.lower()

    # Color temperature
    if "too_cool" in response_lower or "too cool" in response_lower:
        diag["color_temperature"] = "too_cool"
    elif "too_warm" in response_lower or "too warm" in response_lower:
        diag["color_temperature"] = "too_warm"

    # Density
    if "too_sparse" in response_lower or "sparse" in response_lower:
        diag["density"] = "too_sparse"
    elif "too_dense" in response_lower or "dense" in response_lower:
        diag["density"] = "too_dense"

    # Shape
    if "amorphous" in response_lower or "undefined" in response_lower:
        diag["shape"] = "amorphous"
    elif "well_defined" in response_lower or "well defined" in response_lower:
        diag["shape"] = "well_defined"

    # Brightness
    if "too_dim" in response_lower or "dim" in response_lower:
        diag["brightness"] = "too_dim"
    elif "too_bright" in response_lower or "overexposed" in response_lower:
        diag["brightness"] = "too_bright"

    # Smoke ratio
    if "too_little" in response_lower or "more smoke" in response_lower:
        diag["smoke_ratio"] = "too_little"
    elif "too_much" in response_lower or "less smoke" in response_lower:
        diag["smoke_ratio"] = "too_much"

    # Contrast
    if "too_low" in response_lower or "flat" in response_lower:
        diag["contrast"] = "too_low"
    elif "too_high" in response_lower or "harsh" in response_lower:
        diag["contrast"] = "too_high"

    # Detail
    if "too_smooth" in response_lower or "blurry" in response_lower:
        diag["detail_level"] = "too_smooth"
    elif "too_noisy" in response_lower or "noisy" in response_lower:
        diag["detail_level"] = "too_noisy"

    return VFXDiagnostics(**diag)


def _generate_parameter_suggestions(diag: VFXDiagnostics) -> SuggestedParameters:
    """Map diagnostics to Blender parameter adjustments."""
    suggestions = SuggestedParameters()

    # Color temperature -> flame temperature
    if diag.color_temperature == "too_cool":
        suggestions.flame_max_temp = "increase 20%"
        suggestions.heat = "increase 30%"
    elif diag.color_temperature == "too_warm":
        suggestions.flame_max_temp = "decrease 15%"
        suggestions.heat = "decrease 20%"

    # Density -> density/dissolve
    if diag.density == "too_sparse":
        suggestions.density = "increase 40%"
        suggestions.dissolve_speed = "decrease 30%"
    elif diag.density == "too_dense":
        suggestions.density = "decrease 30%"
        suggestions.dissolve_speed = "increase 20%"

    # Shape -> vorticity/turbulence
    if diag.shape == "amorphous":
        suggestions.vorticity = "increase 50%"
        suggestions.turbulence = "increase 30%"

    # Brightness -> emission/heat
    if diag.brightness == "too_dim":
        suggestions.heat = "increase 40%"
    elif diag.brightness == "too_bright":
        suggestions.heat = "decrease 30%"

    # Smoke ratio
    if diag.smoke_ratio == "too_little":
        suggestions.flame_smoke = "increase 50%"
    elif diag.smoke_ratio == "too_much":
        suggestions.flame_smoke = "decrease 40%"

    # Detail -> resolution
    if diag.detail_level == "too_smooth":
        suggestions.resolution = "increase 50%"
    elif diag.detail_level == "too_noisy":
        suggestions.resolution = "decrease 25%"
        suggestions.turbulence = "decrease 20%"

    # Buoyancy for dynamics
    if diag.shape == "amorphous":
        suggestions.buoyancy = "increase 30%"

    return suggestions


def _fallback_diagnostics(image_path: str, effect_type: str) -> Tuple[VFXDiagnostics, SuggestedParameters]:
    """Fallback diagnostics using image statistics."""
    from PIL import Image
    import numpy as np

    try:
        img = Image.open(image_path).convert('RGB')
        arr = np.array(img)

        # Analyze color channels
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

        # Color temperature (R/B ratio)
        r_mean, b_mean = r.mean(), b.mean()
        if r_mean > b_mean * 1.3:
            color_temp = "too_warm"
        elif b_mean > r_mean * 1.2:
            color_temp = "too_cool"
        else:
            color_temp = "good"

        # Brightness
        brightness = arr.mean()
        if brightness < 60:
            bright = "too_dim"
        elif brightness > 180:
            bright = "too_bright"
        else:
            bright = "good"

        # Contrast (standard deviation)
        contrast = arr.std()
        if contrast < 40:
            contr = "too_low"
        elif contrast > 90:
            contr = "too_high"
        else:
            contr = "good"

        # Density (non-black pixels)
        non_black = (arr.sum(axis=2) > 30).mean()
        if non_black < 0.15:
            dens = "too_sparse"
        elif non_black > 0.7:
            dens = "too_dense"
        else:
            dens = "good"

        diag = VFXDiagnostics(
            color_temperature=color_temp,
            density=dens,
            shape="good",  # Can't determine without VLM
            brightness=bright,
            smoke_ratio="good",  # Can't determine without VLM
            contrast=contr,
            detail_level="good",
            raw_analysis="Fallback heuristic analysis (VLM unavailable)"
        )

        suggestions = _generate_parameter_suggestions(diag)
        return diag, suggestions

    except Exception as e:
        # Ultimate fallback
        return VFXDiagnostics(
            color_temperature="good",
            density="good",
            shape="good",
            brightness="good",
            smoke_ratio="good",
            contrast="good",
            detail_level="good",
            raw_analysis=f"Analysis failed: {e}"
        ), SuggestedParameters()


# =============================================================================
# Composite Enhanced Evaluation
# =============================================================================

async def enhanced_evaluate_render(
    render_path: str,
    semantic_query: str,
    effect_type: str = "explosion",
    reference_path: Optional[str] = None,
    lpips_threshold: float = 0.35,
    clip_threshold: float = 0.60,
    aesthetic_threshold: float = 5.5,
    require_all: bool = False
) -> str:
    """
    Comprehensive VFX evaluation with multi-modal metrics and diagnostics.

    Combines:
    - LPIPS (if reference provided): Perceptual similarity
    - Multi-prompt CLIP: Fine-grained quality gradient
    - LAION Aesthetics: Reference-free quality
    - ImageReward: Text-image alignment quality
    - Moondream VLM: Actionable diagnostics

    Returns detailed evaluation with parameter suggestions.
    """
    from pathlib import Path

    render = Path(render_path)
    if not render.exists():
        return json.dumps({"error": f"Render not found: {render}"})

    results = {
        "lpips_score": None,
        "clip_score": None,
        "aesthetic_score": None,
        "image_reward_score": None,
        "multi_prompt_gradient": None,
    }

    passed_metrics = []
    failed_metrics = []
    recommendations = []

    # 1. LPIPS (if reference provided)
    if reference_path:
        from server import compare_lpips
        lpips_result = await compare_lpips(str(render), reference_path)
        lpips_data = json.loads(lpips_result)
        if "error" not in lpips_data:
            results["lpips_score"] = lpips_data["similarity_score"]
            if results["lpips_score"] <= lpips_threshold:
                passed_metrics.append("LPIPS")
            else:
                failed_metrics.append("LPIPS")
                recommendations.append(f"LPIPS {results['lpips_score']:.3f} > {lpips_threshold}: structural differences from reference")

    # 2. Multi-prompt CLIP (gradient signal)
    try:
        mp_result = evaluate_multi_prompt_clip(str(render), semantic_query, effect_type)
        results["clip_score"] = mp_result.overall_match
        results["multi_prompt_gradient"] = mp_result.gradient_signal

        if mp_result.overall_match >= clip_threshold:
            passed_metrics.append("CLIP")
        else:
            failed_metrics.append("CLIP")
            recommendations.append(f"CLIP {mp_result.overall_match:.3f} < {clip_threshold}")

        # Use dimension scores for more specific feedback
        for dim, score in mp_result.quality_dimension_scores.items():
            if score < 0.5:
                recommendations.append(f"Improve {dim}: current score {score:.2f}")

    except Exception as e:
        recommendations.append(f"CLIP evaluation failed: {e}")

    # 3. Aesthetic scoring (reference-free)
    try:
        results["aesthetic_score"] = evaluate_aesthetic_quality(str(render))
        if results["aesthetic_score"] >= aesthetic_threshold:
            passed_metrics.append("Aesthetic")
        else:
            failed_metrics.append("Aesthetic")
            recommendations.append(f"Aesthetic score {results['aesthetic_score']:.1f} < {aesthetic_threshold}")
    except Exception as e:
        recommendations.append(f"Aesthetic evaluation failed: {e}")

    # 4. ImageReward (text-image quality)
    try:
        results["image_reward_score"] = evaluate_image_reward(str(render), semantic_query)
        if results["image_reward_score"] > 0:
            passed_metrics.append("ImageReward")
        else:
            failed_metrics.append("ImageReward")
    except Exception as e:
        pass  # Optional metric

    # 5. VLM Diagnostics
    diagnostics = None
    suggestions = None
    try:
        diagnostics, suggestions = analyze_vfx_diagnostics(str(render), effect_type)

        # Generate recommendations from diagnostics
        if diagnostics.color_temperature != "good":
            recommendations.append(f"Color: {diagnostics.color_temperature}")
        if diagnostics.density != "good":
            recommendations.append(f"Density: {diagnostics.density}")
        if diagnostics.brightness != "good":
            recommendations.append(f"Brightness: {diagnostics.brightness}")
        if diagnostics.smoke_ratio != "good":
            recommendations.append(f"Smoke: {diagnostics.smoke_ratio}")
    except Exception as e:
        recommendations.append(f"Diagnostic analysis limited: {e}")

    # Calculate overall pass/fail
    if require_all:
        passed = len(failed_metrics) == 0 and len(passed_metrics) > 0
    else:
        passed = len(passed_metrics) > 0

    # Calculate composite score (0-100)
    scores = []
    if results["lpips_score"] is not None:
        scores.append(max(0, (1 - results["lpips_score"] / 0.5)) * 100)
    if results["clip_score"] is not None:
        scores.append(results["clip_score"] * 100)
    if results["aesthetic_score"] is not None:
        scores.append(results["aesthetic_score"] * 10)  # Scale 1-10 to 0-100
    if results["multi_prompt_gradient"] is not None:
        scores.append(results["multi_prompt_gradient"] * 100)

    overall_score = sum(scores) / len(scores) if scores else 0

    result = EnhancedEvaluationResult(
        passed=passed,
        overall_score=round(overall_score, 1),
        lpips_score=results["lpips_score"],
        clip_score=results["clip_score"],
        aesthetic_score=results["aesthetic_score"],
        image_reward_score=results["image_reward_score"],
        multi_prompt_gradient=results["multi_prompt_gradient"],
        diagnostics=diagnostics,
        suggested_parameters=suggestions,
        render_path=str(render),
        reference_path=reference_path,
        semantic_query=semantic_query,
        recommendations=recommendations if not passed else ["Quality acceptable"] + recommendations[:3]
    )

    return json.dumps({
        "passed": result.passed,
        "overall_score": result.overall_score,
        "scores": {
            "lpips": result.lpips_score,
            "clip": result.clip_score,
            "aesthetic": result.aesthetic_score,
            "image_reward": result.image_reward_score,
            "gradient_signal": result.multi_prompt_gradient
        },
        "diagnostics": asdict(result.diagnostics) if result.diagnostics else None,
        "suggested_parameters": asdict(result.suggested_parameters) if result.suggested_parameters else None,
        "recommendations": result.recommendations,
        "passed_metrics": passed_metrics,
        "failed_metrics": failed_metrics
    }, indent=2)


# =============================================================================
# Exploration Support (Challenge 4)
# =============================================================================

def suggest_alternative_approaches(
    current_approach: str,
    scores_history: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Suggest fundamentally different approaches when stuck in local optima.

    Addresses Challenge 4: Single-Path Optimization Trap
    """
    alternatives = []

    # Analyze score history for plateau detection
    if len(scores_history) >= 3:
        recent_scores = [s.get("overall_score", 0) for s in scores_history[-3:]]
        improvement = max(recent_scores) - min(recent_scores)

        if improvement < 2:  # Less than 2% improvement = plateau
            alternatives.append({
                "reason": "Score plateau detected",
                "suggestion": "Try a fundamentally different emitter type",
                "options": [
                    {"type": "sphere_emitter", "description": "Point source explosion"},
                    {"type": "plane_emitter", "description": "Ground-based fire"},
                    {"type": "mesh_emitter", "description": "Object on fire"},
                    {"type": "animated_inflow", "description": "Dynamic fuel source"}
                ]
            })

    # Suggest based on current approach weaknesses
    if "sphere" in current_approach.lower():
        alternatives.append({
            "reason": "Sphere emitters can lack ground interaction",
            "suggestion": "Try plane emitter with upward buoyancy",
            "expected_benefit": "More realistic grounding and smoke behavior"
        })

    if "plane" in current_approach.lower():
        alternatives.append({
            "reason": "Plane emitters may lack 3D volumetric depth",
            "suggestion": "Try animated point sources for more depth",
            "expected_benefit": "Better parallax and volumetric feel"
        })

    return alternatives


# =============================================================================
# Temporal Quality Analysis (Challenge 5)
# =============================================================================

def analyze_temporal_quality(
    frame_paths: List[str],
    sample_rate: int = 5
) -> Dict[str, Any]:
    """
    Analyze temporal consistency across animation frames.

    Addresses Challenge 5: Temporal Quality Not Evaluated
    """
    from PIL import Image
    import numpy as np

    if len(frame_paths) < 2:
        return {"error": "Need at least 2 frames for temporal analysis"}

    # Sample frames evenly
    indices = list(range(0, len(frame_paths), sample_rate))
    if indices[-1] != len(frame_paths) - 1:
        indices.append(len(frame_paths) - 1)

    sampled_paths = [frame_paths[i] for i in indices]

    # Load frames
    frames = []
    for path in sampled_paths:
        try:
            img = Image.open(path).convert('RGB')
            frames.append(np.array(img).astype(float))
        except:
            continue

    if len(frames) < 2:
        return {"error": "Could not load enough frames"}

    # Calculate frame-to-frame differences
    diffs = []
    for i in range(len(frames) - 1):
        diff = np.abs(frames[i+1] - frames[i]).mean()
        diffs.append(diff)

    # Analyze temporal metrics
    avg_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    max_diff = np.max(diffs)

    # Detect flickering (high variance in differences)
    flicker_score = std_diff / (avg_diff + 1e-6)  # Higher = more flickering

    # Detect temporal consistency
    consistency_score = 1.0 / (1.0 + flicker_score)

    # Detect static frames (no change)
    static_frames = sum(1 for d in diffs if d < 5)

    return {
        "frame_count": len(frame_paths),
        "sampled_count": len(frames),
        "temporal_consistency": round(consistency_score, 3),
        "flicker_risk": "high" if flicker_score > 2 else "medium" if flicker_score > 1 else "low",
        "average_motion": round(avg_diff, 2),
        "motion_variance": round(std_diff, 2),
        "peak_motion_frame": indices[diffs.index(max(diffs))] if diffs else 0,
        "static_frame_count": static_frames,
        "recommendations": _get_temporal_recommendations(consistency_score, flicker_score, static_frames)
    }


def _get_temporal_recommendations(consistency: float, flicker: float, static: int) -> List[str]:
    """Generate temporal quality recommendations."""
    recs = []

    if flicker > 2:
        recs.append("High flicker detected: reduce noise/turbulence or add temporal smoothing")
    if flicker > 1:
        recs.append("Moderate flicker: consider motion blur or longer simulation substeps")
    if static > 3:
        recs.append(f"{static} near-static frames: increase velocity/buoyancy or shorten animation")
    if consistency < 0.5:
        recs.append("Low temporal consistency: check simulation timesteps and cache")

    if not recs:
        recs.append("Temporal quality looks good")

    return recs


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def test():
        # Test with a sample image
        test_image = "/home/maz3ppa/projects/PlasmaDXR/build/vdb_output/supernova_burst/renders/frame_0035.png"

        if Path(test_image).exists():
            result = await enhanced_evaluate_render(
                test_image,
                "a bright supernova explosion in space",
                effect_type="supernova"
            )
            print(result)
        else:
            print("Test image not found, skipping test")

    asyncio.run(test())
