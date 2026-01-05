#!/usr/bin/env python3
"""
Consolidated Evaluation Module - January 2026 Upgrade

Consolidates 32 tools into 6 unified tools with:
- SigLIP 2 (replaces CLIP) for semantic similarity
- pyiqa TOPIQ + QualiCLIP for image quality assessment
- Moondream VLM for actionable diagnostics
- LPIPS for perceptual similarity
- Feature size CV for texture analysis

Key design decisions:
- Lazy loading for all ML models (~5GB total)
- Profile-based evaluation (quick/standard/comprehensive)
- VLM replaces hardcoded artifact detectors
"""

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MetricScores:
    """Individual metric scores."""
    lpips: Optional[float] = None           # Lower = better (0-1)
    siglip: Optional[float] = None          # Higher = better (0-1)
    topiq: Optional[float] = None           # Higher = better (0-1)
    qualiclip: Optional[float] = None       # Higher = better (0-1)
    structural_dino: Optional[float] = None # Higher = better (0-1)
    feature_cv: Optional[float] = None      # Feature size coefficient of variation
    texture_procedural: Optional[float] = None  # Lower = more natural (0-100)


@dataclass
class DiagnosedIssue:
    """A single diagnosed issue from VLM."""
    description: str
    severity: str  # critical, major, minor
    location: Optional[str] = None  # e.g., "top-left prominence", "overall"
    likely_cause: Optional[str] = None
    suggested_fix: Optional[str] = None


@dataclass
class EvaluationResult:
    """Result of evaluate_render tool."""
    overall_score: float  # 0-100
    passed: bool
    metric_scores: MetricScores
    diagnostics: List[DiagnosedIssue] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    profile_used: str = "standard"
    effect_type_detected: str = "unknown"
    evaluation_time_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class ComparisonResult:
    """Result of compare_renders tool."""
    winner: str  # "A", "B", or "similar"
    score_a: float
    score_b: float
    improvements: List[str] = field(default_factory=list)
    regressions: List[str] = field(default_factory=list)
    recommendation: str = ""
    comparison_type: str = "quality"


@dataclass
class DiagnosisResult:
    """Result of diagnose_issues tool."""
    issues: List[DiagnosedIssue]
    primary_issue: Optional[DiagnosedIssue] = None
    overall_assessment: str = ""
    vlm_used: bool = False
    raw_vlm_output: Optional[str] = None


@dataclass
class ReferenceStats:
    """Reference dataset statistics."""
    effect_type: str
    sample_count: int
    brightness_mean: float
    brightness_std: float
    warm_ratio_mean: float
    warm_ratio_std: float
    edge_density_mean: float
    edge_density_std: float
    feature_cv_mean: float
    feature_cv_std: float


@dataclass
class RenderInfo:
    """Information about a render file."""
    path: str
    filename: str
    size_bytes: int
    modified_time: str
    dimensions: Optional[Tuple[int, int]] = None


@dataclass
class TrainingResult:
    """Result of train_model tool."""
    success: bool
    model_path: Optional[str] = None
    epochs_completed: int = 0
    final_accuracy: Optional[float] = None
    error: Optional[str] = None


# =============================================================================
# Lazy-Loaded ML Models
# =============================================================================

_siglip_model = None
_siglip_processor = None
_topiq_model = None
_qualiclip_model = None
_moondream_model = None
_moondream_tokenizer = None
_lpips_model = None
_dino_model = None


def get_device():
    """Get the best available device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    # MPS on Apple Silicon
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_siglip_model():
    """
    Lazy load SigLIP 2 model (replaces OpenAI CLIP).

    SigLIP 2 advantages:
    - Better zero-shot performance on domain-specific images
    - Trained on larger, more diverse dataset
    - Available via transformers (no separate install)
    """
    global _siglip_model, _siglip_processor
    if _siglip_model is None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor

            model_name = "google/siglip-so400m-patch14-384"
            device = get_device()

            _siglip_processor = AutoProcessor.from_pretrained(model_name)
            _siglip_model = AutoModel.from_pretrained(model_name)
            _siglip_model = _siglip_model.to(device)
            _siglip_model.eval()

            print(f"[SigLIP 2] Loaded {model_name} on {device}")
        except Exception as e:
            print(f"[SigLIP 2] Failed to load: {e}")
            raise RuntimeError(f"SigLIP 2 not available: {e}")
    return _siglip_model, _siglip_processor


def get_topiq_model():
    """
    Lazy load TOPIQ model from pyiqa.

    TOPIQ: Top-down Image Quality Assessment
    - Full-reference metric when reference provided
    - No-reference metric otherwise
    """
    global _topiq_model
    if _topiq_model is None:
        try:
            import pyiqa
            import torch
            device = get_device()

            # TOPIQ-FR for full-reference, TOPIQ-NR for no-reference
            _topiq_model = pyiqa.create_metric('topiq_fr', device=device)
            print(f"[TOPIQ] Loaded on {device}")
        except Exception as e:
            print(f"[TOPIQ] Failed to load: {e}")
            return None
    return _topiq_model


def get_qualiclip_model():
    """
    Lazy load QualiCLIP model from pyiqa.

    QualiCLIP: Quality-aware CLIP for no-reference quality assessment
    - Uses CLIP features with quality-aware fine-tuning
    - No reference image needed
    """
    global _qualiclip_model
    if _qualiclip_model is None:
        try:
            import pyiqa
            import torch
            device = get_device()

            _qualiclip_model = pyiqa.create_metric('qalign', device=device)
            print(f"[QualiCLIP] Loaded on {device}")
        except Exception as e:
            # QualiCLIP might not be available, try alternatives
            print(f"[QualiCLIP] Failed to load: {e}, trying qalign")
            try:
                import pyiqa
                _qualiclip_model = pyiqa.create_metric('niqe', device=device)
                print(f"[QualiCLIP fallback] Using NIQE on {device}")
            except Exception as e2:
                print(f"[QualiCLIP] No fallback available: {e2}")
                return None
    return _qualiclip_model


def get_moondream_model():
    """
    Lazy load Moondream 2 VLM for diagnostic analysis.

    Moondream 2: Fast, lightweight VLM (~2B parameters)
    - Runs locally (free, no API costs)
    - Good at image description and comparison
    - ~1.8GB model weights
    """
    global _moondream_model, _moondream_tokenizer
    if _moondream_model is None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_id = "vikhyatk/moondream2"
            revision = "2025-01-09"
            device = get_device()

            _moondream_tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                revision=revision
            )
            _moondream_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )

            if device != "cuda":
                _moondream_model = _moondream_model.to(device)

            print(f"[Moondream 2] Loaded on {device}")
        except Exception as e:
            print(f"[Moondream 2] Failed to load: {e}")
            return None, None
    return _moondream_model, _moondream_tokenizer


def get_lpips_model():
    """Lazy load LPIPS model (~528MB)."""
    global _lpips_model
    if _lpips_model is None:
        try:
            import torch
            import lpips
            device = get_device()
            _lpips_model = lpips.LPIPS(net='alex').to(device)
            print(f"[LPIPS] Loaded on {device}")
        except Exception as e:
            print(f"[LPIPS] Failed to load: {e}")
            raise RuntimeError(f"LPIPS not available: {e}")
    return _lpips_model


def get_dino_model():
    """Lazy load DINOv2 for structural similarity."""
    global _dino_model
    if _dino_model is None:
        try:
            import torch
            device = get_device()
            _dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            _dino_model = _dino_model.to(device)
            _dino_model.eval()
            print(f"[DINOv2] Loaded on {device}")
        except Exception as e:
            print(f"[DINOv2] Failed to load: {e}")
            return None
    return _dino_model


# =============================================================================
# Image Processing Utilities
# =============================================================================

def load_image_pil(path: str):
    """Load image as PIL Image."""
    from PIL import Image
    return Image.open(path).convert('RGB')


def load_image_tensor(path: str, size: int = 256):
    """Load image as normalized tensor for LPIPS."""
    import torch
    from PIL import Image
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0)


def load_image_numpy(path: str) -> np.ndarray:
    """Load image as numpy array (0-255 uint8)."""
    from PIL import Image
    img = Image.open(path).convert('RGB')
    return np.array(img)


# =============================================================================
# Metric Computation Functions
# =============================================================================

def compute_siglip_similarity(image_path: str, query: str) -> float:
    """
    Compute SigLIP 2 similarity between image and text query.

    Returns: Similarity score 0-1 (higher = better match)
    """
    import torch

    model, processor = get_siglip_model()
    device = next(model.parameters()).device

    # Load and preprocess
    image = load_image_pil(image_path)
    inputs = processor(
        text=[query],
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        # Normalize
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # Cosine similarity
        similarity = (image_embeds @ text_embeds.T).squeeze()

        # Sigmoid to convert logits to probability
        similarity = torch.sigmoid(similarity).item()

    return similarity


def compute_siglip_image_similarity(image_a_path: str, image_b_path: str) -> float:
    """
    Compute SigLIP 2 similarity between two images.

    Returns: Similarity score 0-1 (higher = more similar)
    """
    import torch

    model, processor = get_siglip_model()
    device = next(model.parameters()).device

    # Load images
    image_a = load_image_pil(image_a_path)
    image_b = load_image_pil(image_b_path)

    # Process both images
    inputs_a = processor(images=image_a, return_tensors="pt").to(device)
    inputs_b = processor(images=image_b, return_tensors="pt").to(device)

    with torch.no_grad():
        embeds_a = model.get_image_features(**inputs_a)
        embeds_b = model.get_image_features(**inputs_b)

        # Normalize
        embeds_a = embeds_a / embeds_a.norm(dim=-1, keepdim=True)
        embeds_b = embeds_b / embeds_b.norm(dim=-1, keepdim=True)

        # Cosine similarity
        similarity = (embeds_a @ embeds_b.T).squeeze().item()

        # Convert from [-1, 1] to [0, 1]
        similarity = (similarity + 1) / 2

    return similarity


def compute_lpips(image_a_path: str, image_b_path: str) -> float:
    """
    Compute LPIPS perceptual distance.

    Returns: LPIPS score 0-1 (lower = more similar)
    """
    import torch

    model = get_lpips_model()
    device = next(model.parameters()).device

    img_a = load_image_tensor(image_a_path).to(device)
    img_b = load_image_tensor(image_b_path).to(device)

    with torch.no_grad():
        distance = model(img_a, img_b).item()

    return distance


def compute_topiq(render_path: str, reference_path: str = None) -> Optional[float]:
    """
    Compute TOPIQ quality score.

    Returns: Quality score 0-1 (higher = better)
    """
    model = get_topiq_model()
    if model is None:
        return None

    try:
        if reference_path:
            # Full-reference mode
            score = model(render_path, reference_path).item()
        else:
            # No-reference mode
            score = model(render_path).item()
        return score
    except Exception as e:
        print(f"[TOPIQ] Computation failed: {e}")
        return None


def compute_qualiclip(image_path: str) -> Optional[float]:
    """
    Compute QualiCLIP no-reference quality score.

    Returns: Quality score (higher = better)
    """
    model = get_qualiclip_model()
    if model is None:
        return None

    try:
        score = model(image_path).item()
        return score
    except Exception as e:
        print(f"[QualiCLIP] Computation failed: {e}")
        return None


def compute_feature_cv(image_path: str, threshold: int = 30) -> float:
    """
    Compute feature size coefficient of variation.

    This is the SINGLE MOST DISCRIMINATING metric for procedural vs natural textures:
    - Procedural textures: CV = 1-3 (uniform feature sizes)
    - Natural textures: CV = 20-40 (varied feature sizes)

    Returns: CV value (lower = more procedural/uniform)
    """
    import cv2

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0

    # Binary threshold to get feature mask
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels <= 1:  # Only background
        return 0.0

    # Get feature sizes (exclude background at index 0)
    sizes = stats[1:, cv2.CC_STAT_AREA]

    if len(sizes) < 2:
        return 0.0

    # Coefficient of variation
    cv = np.std(sizes) / np.mean(sizes) * 100 if np.mean(sizes) > 0 else 0

    return cv


def compute_dino_structural_similarity(
    render_path: str,
    reference_path: str
) -> Optional[float]:
    """
    Compute DINOv2 structural similarity.

    Returns: Similarity score 0-1 (higher = more similar)
    """
    import torch

    model = get_dino_model()
    if model is None:
        return None

    device = next(model.parameters()).device

    try:
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img_render = transform(load_image_pil(render_path)).unsqueeze(0).to(device)
        img_ref = transform(load_image_pil(reference_path)).unsqueeze(0).to(device)

        with torch.no_grad():
            feat_render = model(img_render)
            feat_ref = model(img_ref)

            # Normalize
            feat_render = feat_render / feat_render.norm(dim=-1, keepdim=True)
            feat_ref = feat_ref / feat_ref.norm(dim=-1, keepdim=True)

            # Cosine similarity
            similarity = (feat_render @ feat_ref.T).squeeze().item()

            # Convert to 0-1
            similarity = (similarity + 1) / 2

        return similarity
    except Exception as e:
        print(f"[DINOv2] Computation failed: {e}")
        return None


# =============================================================================
# VLM Diagnosis Functions
# =============================================================================

def diagnose_with_moondream(
    render_path: str,
    reference_path: str = None,
    effect_type: str = "unknown"
) -> DiagnosisResult:
    """
    Use Moondream VLM to diagnose visual issues.

    This replaces ALL specific artifact detectors with a general-purpose VLM.
    """
    model, tokenizer = get_moondream_model()

    if model is None:
        return DiagnosisResult(
            issues=[],
            overall_assessment="VLM not available, using fallback diagnostics",
            vlm_used=False
        )

    try:
        from PIL import Image

        # Load render image
        render_image = Image.open(render_path).convert('RGB')

        # Construct prompt based on effect type
        if effect_type == "sun" or effect_type == "star":
            effect_context = """This is a VFX render of a sun/star with prominences.
Expected features: Solar disk, limb darkening, prominences/eruptions, granulation texture.
Common issues: Triangular 'cat ear' artifacts, uniform procedural texture, missing corona, wrong colors."""
        elif effect_type == "explosion" or effect_type == "pyro":
            effect_context = """This is a VFX render of an explosion/pyro effect.
Expected features: Fire colors, smoke, dynamic shapes, brightness gradients.
Common issues: Too dark, missing fire colors, flat shapes, uniform density."""
        elif effect_type == "nebula":
            effect_context = """This is a VFX render of a nebula/gas cloud.
Expected features: Colorful gas clouds, stars, gradients, varied opacity.
Common issues: Too uniform, wrong colors, missing depth."""
        else:
            effect_context = "This is a VFX render."

        prompt = f"""{effect_context}

Analyze this render and identify any visual problems. For each issue:
1. Describe the problem specifically
2. Rate severity (critical/major/minor)
3. Suggest what might fix it

If comparing to a reference, note key differences.

Be concise and focus on actionable issues."""

        # Moondream encode
        enc_image = model.encode_image(render_image)

        # Generate response
        response = model.answer_question(enc_image, prompt, tokenizer)

        # Parse VLM response into structured issues
        issues = _parse_vlm_diagnosis(response, effect_type)

        # Determine primary issue
        primary = None
        for issue in issues:
            if issue.severity == "critical":
                primary = issue
                break
        if primary is None and issues:
            primary = issues[0]

        return DiagnosisResult(
            issues=issues,
            primary_issue=primary,
            overall_assessment=response[:500] if len(response) > 500 else response,
            vlm_used=True,
            raw_vlm_output=response
        )

    except Exception as e:
        print(f"[Moondream] Diagnosis failed: {e}")
        return DiagnosisResult(
            issues=[DiagnosedIssue(
                description=f"VLM analysis failed: {str(e)}",
                severity="minor"
            )],
            overall_assessment=f"Error during VLM analysis: {e}",
            vlm_used=False
        )


def _parse_vlm_diagnosis(response: str, effect_type: str) -> List[DiagnosedIssue]:
    """Parse VLM text response into structured issues."""
    issues = []

    # Simple keyword-based parsing
    response_lower = response.lower()

    # Check for common issues based on keywords
    if "too dark" in response_lower or "dark" in response_lower and "too" in response_lower:
        issues.append(DiagnosedIssue(
            description="Render is too dark",
            severity="major",
            likely_cause="Low emission or blackbody temperature",
            suggested_fix="Increase flame_max_temp or emission multiplier"
        ))

    if "triangular" in response_lower or "cat ear" in response_lower or "pointy" in response_lower:
        issues.append(DiagnosedIssue(
            description="Triangular/cat-ear artifacts in prominences",
            severity="critical",
            location="Prominence regions",
            likely_cause="Insufficient turbulence or vorticity",
            suggested_fix="Increase vorticity (0.3-0.8) and turbulence"
        ))

    if "uniform" in response_lower and ("texture" in response_lower or "pattern" in response_lower):
        issues.append(DiagnosedIssue(
            description="Uniform/procedural texture pattern",
            severity="major",
            likely_cause="Noise settings too regular",
            suggested_fix="Add multi-scale noise, increase noise detail"
        ))

    if "color" in response_lower and ("wrong" in response_lower or "incorrect" in response_lower):
        issues.append(DiagnosedIssue(
            description="Incorrect color temperature",
            severity="major",
            likely_cause="Blackbody temperature misconfigured",
            suggested_fix="Adjust temperature for effect type (sun: ~5800K, fire: 1500-3000K)"
        ))

    if "missing" in response_lower and "prominence" in response_lower:
        issues.append(DiagnosedIssue(
            description="Missing or weak prominences",
            severity="major",
            location="Edge of solar disk",
            likely_cause="Flow velocity too low or domain too small",
            suggested_fix="Increase initial velocity, extend domain height"
        ))

    if "blurry" in response_lower or "lack" in response_lower and "detail" in response_lower:
        issues.append(DiagnosedIssue(
            description="Lack of detail/blurry",
            severity="minor",
            likely_cause="Resolution too low",
            suggested_fix="Increase simulation resolution"
        ))

    if not issues:
        # If no specific issues detected, add a general note
        if "good" in response_lower or "well" in response_lower:
            issues.append(DiagnosedIssue(
                description="No major issues detected",
                severity="minor",
                suggested_fix="Consider fine-tuning for additional quality"
            ))
        else:
            issues.append(DiagnosedIssue(
                description="VLM analysis complete - review raw output for details",
                severity="minor"
            ))

    return issues


# =============================================================================
# Effect Type Detection
# =============================================================================

def detect_effect_type(image_path: str) -> str:
    """
    Auto-detect effect type from image characteristics.

    Returns: "sun", "explosion", "nebula", "fire", "smoke", or "unknown"
    """
    img = load_image_numpy(image_path)

    # Compute color statistics
    r_mean = img[:, :, 0].mean()
    g_mean = img[:, :, 1].mean()
    b_mean = img[:, :, 2].mean()

    brightness = (r_mean + g_mean + b_mean) / 3
    warm_ratio = r_mean / (b_mean + 1)

    # Check for circular shape (sun/star)
    gray = np.mean(img, axis=2)
    h, w = gray.shape
    center_brightness = gray[h//3:2*h//3, w//3:2*w//3].mean()
    edge_brightness = np.concatenate([
        gray[:h//3, :].flatten(),
        gray[2*h//3:, :].flatten(),
        gray[:, :w//3].flatten(),
        gray[:, 2*w//3:].flatten()
    ]).mean()

    circular_ratio = center_brightness / (edge_brightness + 1)

    # Heuristic classification
    if circular_ratio > 2.0 and warm_ratio > 1.5:
        return "sun"
    elif warm_ratio > 2.0 and brightness > 80:
        return "explosion"
    elif warm_ratio > 1.5 and brightness < 60:
        return "fire"
    elif b_mean > r_mean and brightness < 100:
        return "nebula"
    elif brightness < 50 and warm_ratio < 1.5:
        return "smoke"

    return "unknown"


# =============================================================================
# Main Evaluation Functions
# =============================================================================

def evaluate_render_unified(
    render_path: str,
    reference_path: str = None,
    effect_type: str = "auto",
    profile: str = "standard",
    include_diagnostics: bool = True,
    include_suggestions: bool = True
) -> EvaluationResult:
    """
    Unified render evaluation - the main entry point.

    Profiles:
    - quick: LPIPS + SigLIP only (~2 seconds)
    - standard: + TOPIQ, feature_cv (~10 seconds)
    - comprehensive: + DINOv2, VLM diagnosis (~30 seconds)
    """
    import time
    start_time = time.time()

    # Validate paths
    render_path = str(Path(render_path).resolve())
    if not Path(render_path).exists():
        return EvaluationResult(
            overall_score=0,
            passed=False,
            metric_scores=MetricScores(),
            error=f"Render not found: {render_path}"
        )

    if reference_path:
        reference_path = str(Path(reference_path).resolve())
        if not Path(reference_path).exists():
            return EvaluationResult(
                overall_score=0,
                passed=False,
                metric_scores=MetricScores(),
                error=f"Reference not found: {reference_path}"
            )

    # Auto-detect effect type
    if effect_type == "auto":
        effect_type = detect_effect_type(render_path)

    metrics = MetricScores()
    diagnostics = []
    suggestions = []

    # Profile: quick - basic metrics only
    if profile in ["quick", "standard", "comprehensive"]:
        # LPIPS (if reference provided)
        if reference_path:
            try:
                metrics.lpips = compute_lpips(render_path, reference_path)
            except Exception as e:
                print(f"LPIPS failed: {e}")

        # SigLIP semantic similarity
        try:
            effect_queries = {
                "sun": "a realistic sun with solar prominences and corona",
                "explosion": "a bright fiery explosion with flames and smoke",
                "nebula": "a colorful space nebula with gas clouds and stars",
                "fire": "realistic flames and fire",
                "smoke": "volumetric smoke simulation",
                "unknown": "a high quality VFX render"
            }
            query = effect_queries.get(effect_type, effect_queries["unknown"])
            metrics.siglip = compute_siglip_similarity(render_path, query)
        except Exception as e:
            print(f"SigLIP failed: {e}")

    # Profile: standard - add quality metrics
    if profile in ["standard", "comprehensive"]:
        # TOPIQ
        try:
            metrics.topiq = compute_topiq(render_path, reference_path)
        except Exception as e:
            print(f"TOPIQ failed: {e}")

        # QualiCLIP (no-reference)
        try:
            metrics.qualiclip = compute_qualiclip(render_path)
        except Exception as e:
            print(f"QualiCLIP failed: {e}")

        # Feature CV for texture analysis
        try:
            metrics.feature_cv = compute_feature_cv(render_path)
        except Exception as e:
            print(f"Feature CV failed: {e}")

    # Profile: comprehensive - add structural and VLM
    if profile == "comprehensive":
        # DINOv2 structural similarity
        if reference_path:
            try:
                metrics.structural_dino = compute_dino_structural_similarity(
                    render_path, reference_path
                )
            except Exception as e:
                print(f"DINOv2 failed: {e}")

        # VLM diagnostics
        if include_diagnostics:
            try:
                diagnosis = diagnose_with_moondream(
                    render_path, reference_path, effect_type
                )
                diagnostics = diagnosis.issues

                # Generate suggestions from diagnostics
                if include_suggestions:
                    for issue in diagnostics:
                        if issue.suggested_fix:
                            suggestions.append(issue.suggested_fix)
            except Exception as e:
                print(f"VLM diagnosis failed: {e}")

    # Compute overall score (weighted combination)
    score_components = []
    weights = []

    if metrics.lpips is not None:
        # LPIPS: lower is better, convert to 0-100 score
        lpips_score = max(0, 100 - metrics.lpips * 200)
        score_components.append(lpips_score)
        weights.append(0.25)

    if metrics.siglip is not None:
        score_components.append(metrics.siglip * 100)
        weights.append(0.20)

    if metrics.topiq is not None:
        score_components.append(metrics.topiq * 100)
        weights.append(0.20)

    if metrics.structural_dino is not None:
        score_components.append(metrics.structural_dino * 100)
        weights.append(0.20)

    if metrics.feature_cv is not None:
        # Feature CV: higher is better for natural textures (up to ~30)
        cv_score = min(100, metrics.feature_cv * 3)
        score_components.append(cv_score)
        weights.append(0.15)

    # Calculate weighted average
    if score_components:
        total_weight = sum(weights[:len(score_components)])
        overall_score = sum(s * w for s, w in zip(score_components, weights)) / total_weight
    else:
        overall_score = 0

    # Determine pass/fail
    passed = overall_score >= 60  # 60% threshold

    evaluation_time = time.time() - start_time

    return EvaluationResult(
        overall_score=round(overall_score, 1),
        passed=passed,
        metric_scores=metrics,
        diagnostics=diagnostics,
        suggestions=suggestions,
        profile_used=profile,
        effect_type_detected=effect_type,
        evaluation_time_seconds=round(evaluation_time, 2)
    )


def compare_renders_unified(
    render_a: str,
    render_b: str,
    reference_path: str = None,
    comparison_type: str = "quality"
) -> ComparisonResult:
    """
    Compare two renders - unified comparison tool.

    comparison_type:
    - quality: Which is better overall?
    - iteration: Did changes improve the render?
    """
    # Evaluate both renders
    eval_a = evaluate_render_unified(render_a, reference_path, profile="standard")
    eval_b = evaluate_render_unified(render_b, reference_path, profile="standard")

    score_a = eval_a.overall_score
    score_b = eval_b.overall_score

    # Determine winner
    score_diff = score_b - score_a
    if abs(score_diff) < 5:
        winner = "similar"
    elif score_diff > 0:
        winner = "B"
    else:
        winner = "A"

    # Analyze improvements/regressions
    improvements = []
    regressions = []

    if eval_a.metric_scores.lpips and eval_b.metric_scores.lpips:
        lpips_diff = eval_a.metric_scores.lpips - eval_b.metric_scores.lpips
        if lpips_diff > 0.05:
            improvements.append(f"LPIPS improved by {lpips_diff:.3f}")
        elif lpips_diff < -0.05:
            regressions.append(f"LPIPS regressed by {-lpips_diff:.3f}")

    if eval_a.metric_scores.siglip and eval_b.metric_scores.siglip:
        siglip_diff = eval_b.metric_scores.siglip - eval_a.metric_scores.siglip
        if siglip_diff > 0.05:
            improvements.append(f"Semantic similarity improved by {siglip_diff:.3f}")
        elif siglip_diff < -0.05:
            regressions.append(f"Semantic similarity regressed by {-siglip_diff:.3f}")

    if eval_a.metric_scores.feature_cv and eval_b.metric_scores.feature_cv:
        cv_diff = eval_b.metric_scores.feature_cv - eval_a.metric_scores.feature_cv
        if cv_diff > 5:
            improvements.append(f"Texture more natural (CV +{cv_diff:.1f})")
        elif cv_diff < -5:
            regressions.append(f"Texture more procedural (CV {cv_diff:.1f})")

    # Generate recommendation
    if winner == "similar":
        recommendation = "Both renders are similar quality. Consider other factors."
    elif winner == "B":
        recommendation = f"Render B is better (score: {score_b:.1f} vs {score_a:.1f}). "
        if improvements:
            recommendation += f"Improvements: {', '.join(improvements[:2])}"
    else:
        recommendation = f"Render A is better (score: {score_a:.1f} vs {score_b:.1f}). "
        if regressions:
            recommendation += f"B regressed in: {', '.join(regressions[:2])}"

    return ComparisonResult(
        winner=winner,
        score_a=score_a,
        score_b=score_b,
        improvements=improvements,
        regressions=regressions,
        recommendation=recommendation,
        comparison_type=comparison_type
    )


# =============================================================================
# JSON Serialization
# =============================================================================

def convert_numpy_types(obj):
    """Recursively convert numpy types for JSON serialization."""
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_numpy_types(v) for v in obj]
    return obj


def result_to_json(result) -> str:
    """Convert dataclass result to JSON string."""
    if hasattr(result, '__dataclass_fields__'):
        data = asdict(result)
    else:
        data = result
    return json.dumps(convert_numpy_types(data), indent=2)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Consolidated Evaluation Module - January 2026")
    print("=" * 50)

    # Test imports
    print("\nTesting model imports...")

    try:
        print("- Testing SigLIP 2...")
        model, proc = get_siglip_model()
        print("  SigLIP 2: OK")
    except Exception as e:
        print(f"  SigLIP 2: FAILED ({e})")

    try:
        print("- Testing TOPIQ...")
        model = get_topiq_model()
        print(f"  TOPIQ: {'OK' if model else 'FAILED'}")
    except Exception as e:
        print(f"  TOPIQ: FAILED ({e})")

    try:
        print("- Testing LPIPS...")
        model = get_lpips_model()
        print("  LPIPS: OK")
    except Exception as e:
        print(f"  LPIPS: FAILED ({e})")

    try:
        print("- Testing Moondream...")
        model, tok = get_moondream_model()
        print(f"  Moondream: {'OK' if model else 'FAILED'}")
    except Exception as e:
        print(f"  Moondream: FAILED ({e})")

    print("\nModule loaded successfully!")
