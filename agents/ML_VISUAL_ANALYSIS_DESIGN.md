# ML-Powered Visual Analysis System - Design Document

**Goal:** Use machine learning to automatically detect visual differences, regressions, and quality issues in RTXDI rendering with superhuman precision and consistency.

**Status:** Design phase
**Effort:** Phase 3 (3-4 hours), Phase 4 (6-8 hours)
**Impact:** Game-changing - Automated quality assurance for ray tracing

---

## Architecture Overview

```
Screenshot Capture (existing tool)
       ↓
  Image Preprocessing
       ↓
  ┌─────────────────────────────────┐
  │   ML Visual Analysis Pipeline   │
  ├─────────────────────────────────┤
  │ 1. Traditional CV Metrics       │ ← Fast baseline
  │ 2. Learned Perceptual Similarity│ ← Pre-trained LPIPS
  │ 3. Feature Extraction (CNN)     │ ← Deep features
  │ 4. RTXDI Quality Classifier     │ ← Custom trained
  │ 5. Anomaly Detection            │ ← Autoencoder
  └─────────────────────────────────┘
       ↓
  Aggregated Quality Score + Heatmaps
       ↓
  Claude Analysis + Recommendations
```

---

## Phase 3: ML-Powered Before/After Comparison

### Multi-Level Analysis Approach

**Level 1: Traditional CV Metrics (Fast, Interpretable)**
```python
def traditional_metrics(img_before: np.ndarray, img_after: np.ndarray) -> dict:
    """
    Fast baseline metrics - always compute these first

    Returns:
        {
            "ssim": 0.95,           # Structural similarity (0-1)
            "mse": 245.3,           # Mean squared error
            "psnr": 34.2,           # Peak signal-to-noise ratio (dB)
            "histogram_corr": 0.98, # Color distribution similarity
            "perceptual_hash": 0.92 # pHash similarity (0-1)
        }
    """
    # SSIM (Structural Similarity Index)
    ssim_score = structural_similarity(img_before, img_after, multichannel=True)

    # MSE and PSNR
    mse = np.mean((img_before - img_after) ** 2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')

    # Histogram correlation (color distribution)
    hist_corr = compare_histograms(img_before, img_after)

    # Perceptual hash (robust to small changes)
    phash_sim = compare_perceptual_hash(img_before, img_after)

    return {
        "ssim": ssim_score,
        "mse": mse,
        "psnr": psnr,
        "histogram_corr": hist_corr,
        "perceptual_hash": phash_sim
    }
```

**Level 2: Learned Perceptual Similarity (LPIPS - State-of-the-art)**
```python
import lpips

def perceptual_similarity(img_before: torch.Tensor, img_after: torch.Tensor) -> dict:
    """
    LPIPS (Learned Perceptual Image Patch Similarity)
    Pre-trained on ImageNet, understands human perception

    Paper: "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"
           Zhang et al., CVPR 2018

    Returns:
        {
            "lpips_alex": 0.12,      # AlexNet backbone
            "lpips_vgg": 0.09,       # VGG backbone (more accurate)
            "lpips_squeeze": 0.11    # SqueezeNet backbone (faster)
        }
    """
    # Load pre-trained LPIPS model (VGG backbone recommended)
    loss_fn = lpips.LPIPS(net='vgg', version='0.1')

    # Compute perceptual distance (lower = more similar)
    distance = loss_fn(img_before, img_after).item()

    # Convert to similarity score (0-1, higher = more similar)
    similarity = 1.0 / (1.0 + distance)

    return {
        "lpips_distance": distance,
        "lpips_similarity": similarity,
        "human_aligned": True  # LPIPS correlates well with human judgments
    }
```

**Level 3: Deep Feature Comparison (CNN Embeddings)**
```python
from torchvision.models import resnet50

def feature_extraction_comparison(img_before: torch.Tensor, img_after: torch.Tensor) -> dict:
    """
    Extract deep features using pre-trained ResNet50
    Compare features at multiple layers (low-level edges → high-level semantics)

    Returns:
        {
            "layer1_similarity": 0.96,  # Low-level features (edges, textures)
            "layer2_similarity": 0.94,  # Mid-level features (patterns)
            "layer3_similarity": 0.91,  # High-level features (objects)
            "layer4_similarity": 0.88,  # Semantic features
            "feature_distance": 234.5   # L2 distance in feature space
        }
    """
    # Load pre-trained ResNet50
    model = resnet50(pretrained=True)
    model.eval()

    # Extract features at multiple layers
    features_before = extract_multiscale_features(model, img_before)
    features_after = extract_multiscale_features(model, img_after)

    # Compute cosine similarity at each layer
    similarities = {}
    for layer_name, (feat_before, feat_after) in zip(
        ["layer1", "layer2", "layer3", "layer4"],
        zip(features_before, features_after)
    ):
        similarity = cosine_similarity(feat_before, feat_after)
        similarities[f"{layer_name}_similarity"] = similarity

    # Overall feature distance
    feature_distance = torch.dist(features_before[-1], features_after[-1], p=2).item()

    return {
        **similarities,
        "feature_distance": feature_distance
    }
```

**Level 4: RTXDI-Specific Quality Classifier (Custom Trained)**
```python
class RTXDIQualityClassifier(nn.Module):
    """
    Custom neural network trained to classify RTXDI rendering quality

    Training data:
    - User-labeled screenshots: "good" vs "bad" RTXDI rendering
    - Known issues: patchwork artifacts, light saturation, temporal noise
    - Comparison pairs: legacy vs RTXDI M4 vs M5

    Output:
    - Quality score (0-1)
    - Issue classifications (patchwork, saturation, noise, shadows)
    - Confidence scores
    """

    def __init__(self):
        super().__init__()
        # ResNet50 backbone (pre-trained on ImageNet)
        self.backbone = resnet50(pretrained=True)

        # Custom classification head
        self.quality_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Quality score 0-1
        )

        # Issue detection head (multi-label classification)
        self.issue_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 5),  # 5 issue types
            nn.Sigmoid()  # Multi-label probabilities
        )

    def forward(self, x):
        features = self.backbone(x)
        quality = self.quality_head(features)
        issues = self.issue_head(features)

        return {
            "quality_score": quality,
            "issues": {
                "patchwork_artifacts": issues[0],
                "light_saturation": issues[1],
                "temporal_noise": issues[2],
                "shadow_artifacts": issues[3],
                "color_bleeding": issues[4]
            }
        }
```

**Level 5: Anomaly Detection (Autoencoder)**
```python
class VisualAnomalyDetector(nn.Module):
    """
    Autoencoder trained on "good" RTXDI screenshots
    High reconstruction error = visual anomaly

    Use case: Detect unexpected visual issues not in training data
    """

    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),   # 64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), # 16x16
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1), # 8x8
            nn.ReLU()
        )

        # Decoder (mirror of encoder)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

    def anomaly_score(self, x):
        """Higher score = more anomalous"""
        reconstructed = self.forward(x)
        mse = F.mse_loss(reconstructed, x, reduction='none')
        # Compute per-pixel anomaly score
        anomaly_map = mse.mean(dim=1)  # Average across RGB
        return anomaly_map.mean().item(), anomaly_map
```

### Aggregated Comparison Report

```python
async def compare_screenshots_ml(
    before_path: str,
    after_path: str,
    use_ml: bool = True
) -> dict:
    """
    Comprehensive ML-powered screenshot comparison

    Args:
        before_path: Path to "before" screenshot
        after_path: Path to "after" screenshot
        use_ml: Enable ML models (requires GPU, slower but more accurate)

    Returns:
        {
            "overall_similarity": 0.92,  # Weighted aggregate score
            "traditional_metrics": {...},
            "perceptual_metrics": {...},
            "feature_metrics": {...},
            "quality_analysis": {...},
            "anomaly_detection": {...},
            "difference_heatmap": np.ndarray,  # Visual diff overlay
            "interpretation": "The images are 92% similar. Primary difference:
                              light intensity increased by 15% in upper-left quadrant.
                              RTXDI temporal accumulation improved (less patchwork).
                              No anomalies detected.",
            "recommendations": [...]
        }
    """
    # Load images
    img_before = load_and_preprocess(before_path)
    img_after = load_and_preprocess(after_path)

    results = {}

    # Level 1: Traditional metrics (always compute - fast)
    results["traditional"] = traditional_metrics(img_before, img_after)

    if use_ml:
        # Level 2: LPIPS (pre-trained)
        results["lpips"] = perceptual_similarity(
            torch.from_numpy(img_before),
            torch.from_numpy(img_after)
        )

        # Level 3: Deep features
        results["features"] = feature_extraction_comparison(
            torch.from_numpy(img_before),
            torch.from_numpy(img_after)
        )

        # Level 4: RTXDI quality (if model trained)
        if rtxdi_classifier_exists():
            results["rtxdi_quality"] = classify_rtxdi_quality(img_after)

        # Level 5: Anomaly detection (if model trained)
        if anomaly_detector_exists():
            results["anomaly"] = detect_anomalies(img_after)

    # Compute difference heatmap
    heatmap = compute_difference_heatmap(img_before, img_after)
    results["difference_heatmap"] = heatmap

    # Aggregate overall similarity score
    results["overall_similarity"] = aggregate_similarity_score(results)

    # Generate interpretation
    results["interpretation"] = interpret_results(results, img_before, img_after)

    return results
```

---

## Phase 4: Automated Visual Regression Detection

### Baseline Management System

**1. Capture Golden Baselines**
```python
async def capture_baseline(
    renderer_mode: str,  # "legacy", "rtxdi_m4", "rtxdi_m5"
    scenario: str,       # "close_distance", "far_distance", "stress_test"
    preset: str          # "performance", "balanced", "quality"
) -> str:
    """
    Capture and store golden baseline screenshot

    Baselines stored in: PIX/baselines/{renderer_mode}/{scenario}/{preset}/

    Returns:
        Path to saved baseline
    """
    # Run PlasmaDX with specific config
    config_path = f"configs/scenarios/{scenario}.json"
    await run_plasmadx(config=config_path, renderer=renderer_mode, preset=preset)

    # Wait for stabilization (60 frames for temporal accumulation)
    await wait_for_stabilization(frames=60)

    # Capture screenshot using existing tool
    screenshot_path = await capture_screenshot(
        description=f"baseline_{renderer_mode}_{scenario}_{preset}"
    )

    # Save to baselines directory
    baseline_dir = Path("PIX/baselines") / renderer_mode / scenario / preset
    baseline_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = baseline_dir / f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    shutil.copy(screenshot_path, baseline_path)

    # Store metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "renderer_mode": renderer_mode,
        "scenario": scenario,
        "preset": preset,
        "config_path": config_path,
        "particle_count": 10000,  # Extract from config
        "light_count": 13,         # Extract from config
        "resolution": "1920x1080"
    }

    with open(baseline_path.with_suffix('.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    return str(baseline_path)
```

**2. ML-Based Regression Detection**
```python
class RegressionDetector:
    """
    Automated visual regression detection using ML

    Training data:
    - Historical baselines (known good)
    - Previous regressions (known bad)
    - User-labeled examples

    Detection modes:
    1. Comparison-based: Current vs baseline
    2. Anomaly-based: Current vs learned "good" distribution
    3. Trend-based: Detect gradual quality degradation over time
    """

    def __init__(self):
        # Load pre-trained models
        self.lpips_model = lpips.LPIPS(net='vgg')
        self.rtxdi_classifier = load_rtxdi_quality_model()
        self.anomaly_detector = load_anomaly_detector_model()

        # Thresholds (learned from validation data)
        self.thresholds = {
            "ssim_min": 0.90,        # SSIM below this = potential regression
            "lpips_max": 0.15,       # LPIPS above this = perceptual difference
            "quality_min": 0.75,     # Quality score below this = issue
            "anomaly_max": 0.30      # Anomaly score above this = anomaly
        }

    async def detect_regression(
        self,
        current_screenshot: str,
        baseline_path: str,
        sensitivity: str = "medium"  # "low", "medium", "high"
    ) -> dict:
        """
        Detect visual regressions using multi-level ML analysis

        Returns:
            {
                "regression_detected": True/False,
                "severity": "critical" | "major" | "minor" | "none",
                "confidence": 0.95,
                "issues": [
                    {
                        "type": "patchwork_artifacts",
                        "severity": 0.82,
                        "location": "upper_left_quadrant",
                        "description": "RTXDI temporal accumulation showing patchwork pattern"
                    }
                ],
                "metrics": {...},
                "heatmap": np.ndarray,
                "recommendation": "..."
            }
        """
        # Load images
        current = load_image(current_screenshot)
        baseline = load_image(baseline_path)

        # Run all analysis levels
        traditional = traditional_metrics(baseline, current)
        lpips_result = perceptual_similarity(baseline, current)
        features = feature_extraction_comparison(baseline, current)
        quality = self.rtxdi_classifier(current)
        anomaly_score, anomaly_map = self.anomaly_detector.anomaly_score(current)

        # Aggregate results
        results = {
            "traditional": traditional,
            "lpips": lpips_result,
            "features": features,
            "quality": quality,
            "anomaly": {
                "score": anomaly_score,
                "map": anomaly_map
            }
        }

        # Adjust thresholds based on sensitivity
        thresholds = self.adjust_thresholds(self.thresholds, sensitivity)

        # Detect regressions
        issues = []

        # Check traditional metrics
        if traditional["ssim"] < thresholds["ssim_min"]:
            issues.append({
                "type": "structural_difference",
                "severity": 1.0 - traditional["ssim"],
                "metric": "SSIM",
                "value": traditional["ssim"],
                "threshold": thresholds["ssim_min"]
            })

        # Check perceptual similarity
        if lpips_result["lpips_distance"] > thresholds["lpips_max"]:
            issues.append({
                "type": "perceptual_difference",
                "severity": lpips_result["lpips_distance"],
                "metric": "LPIPS",
                "value": lpips_result["lpips_distance"],
                "threshold": thresholds["lpips_max"]
            })

        # Check RTXDI quality
        if quality["quality_score"] < thresholds["quality_min"]:
            issues.append({
                "type": "quality_degradation",
                "severity": 1.0 - quality["quality_score"],
                "metric": "RTXDI Quality Score",
                "value": quality["quality_score"],
                "threshold": thresholds["quality_min"]
            })

        # Check for anomalies
        if anomaly_score > thresholds["anomaly_max"]:
            issues.append({
                "type": "visual_anomaly",
                "severity": anomaly_score,
                "metric": "Anomaly Score",
                "value": anomaly_score,
                "threshold": thresholds["anomaly_max"]
            })

        # Classify severity
        regression_detected = len(issues) > 0
        severity = "none"
        confidence = 0.0

        if regression_detected:
            max_severity = max(issue["severity"] for issue in issues)

            if max_severity > 0.75:
                severity = "critical"
                confidence = 0.95
            elif max_severity > 0.50:
                severity = "major"
                confidence = 0.85
            elif max_severity > 0.25:
                severity = "minor"
                confidence = 0.70
            else:
                severity = "negligible"
                confidence = 0.60

        # Generate difference heatmap
        heatmap = compute_difference_heatmap(baseline, current)

        # Generate recommendation
        recommendation = self.generate_recommendation(issues, results)

        return {
            "regression_detected": regression_detected,
            "severity": severity,
            "confidence": confidence,
            "issues": issues,
            "metrics": results,
            "heatmap": heatmap,
            "recommendation": recommendation
        }
```

**3. Continuous Monitoring System**
```python
async def continuous_regression_monitoring(
    watch_directory: str = "PIX/screenshots",
    baseline_directory: str = "PIX/baselines",
    interval_seconds: int = 300  # Check every 5 minutes
):
    """
    Continuously monitor for new screenshots and auto-detect regressions

    Workflow:
    1. Watch PIX/screenshots/ for new files
    2. Identify matching baseline (by renderer mode, scenario, preset)
    3. Run ML regression detection
    4. If regression detected, notify user and create report
    5. Store results in regression_reports/ directory
    """
    detector = RegressionDetector()

    while True:
        # Get new screenshots (not yet analyzed)
        new_screenshots = get_unanalyzed_screenshots(watch_directory)

        for screenshot_path in new_screenshots:
            # Parse metadata from filename
            metadata = parse_screenshot_metadata(screenshot_path)

            # Find matching baseline
            baseline_path = find_matching_baseline(
                baseline_directory,
                renderer_mode=metadata["renderer_mode"],
                scenario=metadata["scenario"],
                preset=metadata["preset"]
            )

            if not baseline_path:
                log_warning(f"No baseline found for {screenshot_path}")
                continue

            # Run regression detection
            result = await detector.detect_regression(
                screenshot_path,
                baseline_path,
                sensitivity="medium"
            )

            # If regression detected, create report
            if result["regression_detected"]:
                report_path = create_regression_report(
                    screenshot_path,
                    baseline_path,
                    result
                )

                # Notify user
                notify_user_regression_detected(
                    severity=result["severity"],
                    confidence=result["confidence"],
                    report_path=report_path
                )

            # Mark as analyzed
            mark_screenshot_analyzed(screenshot_path)

        # Wait for next check
        await asyncio.sleep(interval_seconds)
```

---

## Neural Network Builder Integration

### Use Cases for neural-network-builder Plugin

**1. Train RTXDI Quality Classifier**
```
@neural-network-builder train a CNN classifier to detect RTXDI rendering quality

Task: Binary classification (good vs bad RTXDI rendering)
Architecture: ResNet50 backbone + custom head
Training data:
  - Good: PIX/baselines/rtxdi_m5/ (all baselines are good by definition)
  - Bad: PIX/regressions/known_issues/ (collect known bad screenshots)
  - Augmentation: Random brightness, contrast, saturation, rotation

Dataset size: Start with 100 good + 100 bad, augment to 1000 each
Validation split: 80/20
Epochs: 50
Batch size: 16
Loss: Binary cross-entropy
Optimizer: Adam (lr=1e-4)

Output: rtxdi_quality_classifier.pth
```

**2. Train Visual Anomaly Detector**
```
@neural-network-builder train an autoencoder for visual anomaly detection

Task: Unsupervised anomaly detection (reconstruction-based)
Architecture: Convolutional autoencoder (encoder + decoder)
Training data:
  - Only "good" screenshots from PIX/baselines/
  - Train to reconstruct normal rendering
  - High reconstruction error = anomaly

Dataset size: 500 good screenshots (all baselines)
Validation split: 80/20
Epochs: 100
Batch size: 32
Loss: MSE (reconstruction error)
Optimizer: Adam (lr=1e-3)

Output: visual_anomaly_detector.pth
```

**3. Train Feature Extractor for Similarity**
```
@neural-network-builder train a Siamese network for learned visual similarity

Task: Learn embedding space where similar renderings are close
Architecture: Siamese ResNet (twin networks + contrastive loss)
Training data:
  - Positive pairs: Same scene, different frames (temporal consistency)
  - Negative pairs: Different scenes or renderer modes
  - Triplet loss: anchor, positive, negative

Dataset size: 1000 pairs
Validation split: 80/20
Epochs: 100
Batch size: 16
Loss: Triplet loss (margin=1.0)
Optimizer: Adam (lr=1e-4)

Output: visual_similarity_embedder.pth
```

---

## Training Data Collection Strategy

### Phase 1: Baseline Collection (Week 1)

**Automated baseline capture for all configurations:**
```bash
# Script: tools/collect_baselines.sh
for renderer in legacy rtxdi_m4 rtxdi_m5; do
  for scenario in close_distance far_distance stress_test; do
    for preset in performance balanced quality; do
      # Run PlasmaDX with config
      ./build/Debug/PlasmaDX-Clean.exe \
        --config=configs/scenarios/${scenario}.json \
        --renderer=${renderer} \
        --preset=${preset}

      # Wait for stabilization (60 frames)
      sleep 5

      # Capture screenshot
      ./tools/screenshot.sh "baseline_${renderer}_${scenario}_${preset}"

      # Close PlasmaDX
      pkill PlasmaDX-Clean
    done
  done
done
```

**Expected output:**
- 3 renderer modes × 3 scenarios × 3 presets = **27 baseline screenshots**
- Stored in `PIX/baselines/{renderer}/{scenario}/{preset}/baseline_YYYYMMDD_HHMMSS.png`

### Phase 2: Known Issue Collection (Week 1-2)

**Collect screenshots of known issues:**
```
PIX/training_data/known_issues/
├── patchwork_artifacts/         # RTXDI M4 temporal patchwork
├── light_saturation/            # Grid cell saturation (>10 lights/cell)
├── temporal_noise/              # Unstable temporal accumulation
├── shadow_artifacts/            # PCSS hard edges, incorrect shadows
├── color_bleeding/              # Incorrect light scattering
└── black_frames/                # Complete render failure
```

**Manual labeling:**
- User runs tests, captures screenshots when issues occur
- Label with issue type + severity (1-5 scale)
- Store metadata in JSON sidecar files

### Phase 3: Data Augmentation (Automated)

**Augmentation for classifier training:**
```python
import albumentations as A

transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.RandomRotate90(p=0.2),
    A.HorizontalFlip(p=0.5)
])

# Augment dataset: 100 original → 1000 augmented
```

---

## Implementation Phases

### Phase 3: Before/After ML Comparison (3-4 hours)

**Deliverables:**
1. `src/tools/ml_visual_comparison.py` - ML comparison pipeline
2. Integration with rtxdi-quality-analyzer MCP server
3. New tool: `compare_screenshots_ml`
4. Requirements: LPIPS (pre-trained, no training needed)

**Quick start:**
```bash
# Install ML dependencies
pip install torch torchvision lpips scikit-image opencv-python albumentations

# Add to rtxdi-quality-analyzer
claude mcp server reconnect rtxdi-quality-analyzer

# Test ML comparison
"Compare screenshots using ML:
  before=/mnt/c/Users/dilli/Pictures/PlasmaDX-Screenshots/rtxdi_m4_before.png
  after=/mnt/c/Users/dilli/Pictures/PlasmaDX-Screenshots/rtxdi_m5_after.png"
```

### Phase 4: Automated Regression Detection (6-8 hours)

**Deliverables:**
1. Baseline capture system
2. Continuous monitoring service
3. Regression report generation
4. Claude Code integration (auto-notify on regression)

**Workflow:**
```bash
# 1. Collect baselines (one-time, ~30 minutes)
./tools/collect_baselines.sh

# 2. Start continuous monitoring
python -m agents.rtxdi-quality-analyzer.src.tools.regression_monitor

# 3. Make code changes, test
./build/Debug/PlasmaDX-Clean.exe

# 4. Auto-detect regressions
# (monitoring service captures screenshot, compares to baseline, alerts if regression)
```

### Phase 5: Train Custom Models (2-4 weeks)

**Using neural-network-builder plugin:**
```
Week 1: Collect 200+ labeled screenshots (100 good, 100 bad)
Week 2: Train RTXDI quality classifier using @neural-network-builder
Week 3: Train visual anomaly detector (autoencoder)
Week 4: Train similarity embedder (Siamese network)
```

---

## Expected Performance

### Accuracy Metrics (Target)

**Traditional CV (baseline):**
- SSIM correlation with human judgment: ~0.70
- Fast (10ms per comparison)
- Good for large structural changes
- Poor for perceptual differences

**LPIPS (pre-trained):**
- Correlation with human judgment: ~0.92
- Medium speed (50ms per comparison on GPU)
- Excellent for perceptual differences
- No training required

**Custom RTXDI Classifier (after training):**
- RTXDI-specific issue detection: >95% accuracy
- Real-time inference (10ms on GPU)
- Understands domain-specific issues
- Requires training data collection (2 weeks)

**Anomaly Detector (after training):**
- Novel issue detection: ~85% recall
- Low false positive rate: <5%
- Detects unexpected issues not in training data
- Requires only "good" screenshots for training

### Regression Detection Performance

**Sensitivity settings:**
- **Low:** Only critical regressions (SSIM < 0.80, LPIPS > 0.30)
- **Medium:** Major + critical (SSIM < 0.90, LPIPS > 0.15)
- **High:** All detectable issues (SSIM < 0.95, LPIPS > 0.10)

**False positive rate:**
- Low sensitivity: <1% (only obvious regressions)
- Medium sensitivity: ~5% (recommended for CI/CD)
- High sensitivity: ~15% (manual review needed)

---

## Integration with Existing Tools

### rtxdi-quality-analyzer Tool Integration

**New tools to add:**
```python
Tool(
    name="compare_screenshots_ml",
    description="ML-powered before/after screenshot comparison with perceptual similarity",
    inputSchema={
        "type": "object",
        "properties": {
            "before_path": {
                "type": "string",
                "description": "Path to 'before' screenshot"
            },
            "after_path": {
                "type": "string",
                "description": "Path to 'after' screenshot"
            },
            "use_ml": {
                "type": "boolean",
                "description": "Enable ML models (requires GPU, slower but more accurate)",
                "default": True
            }
        },
        "required": ["before_path", "after_path"]
    }
)

Tool(
    name="detect_regression",
    description="Detect visual regressions by comparing screenshot against baseline",
    inputSchema={
        "type": "object",
        "properties": {
            "screenshot_path": {
                "type": "string",
                "description": "Path to current screenshot"
            },
            "baseline_path": {
                "type": "string",
                "description": "Path to baseline screenshot (optional, auto-detect if not provided)"
            },
            "sensitivity": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "Detection sensitivity",
                "default": "medium"
            }
        },
        "required": ["screenshot_path"]
    }
)

Tool(
    name="capture_baseline",
    description="Capture golden baseline screenshot for regression testing",
    inputSchema={
        "type": "object",
        "properties": {
            "renderer_mode": {
                "type": "string",
                "enum": ["legacy", "rtxdi_m4", "rtxdi_m5"],
                "description": "Renderer mode"
            },
            "scenario": {
                "type": "string",
                "description": "Test scenario (e.g., 'close_distance', 'far_distance')"
            },
            "preset": {
                "type": "string",
                "enum": ["performance", "balanced", "quality"],
                "description": "Shadow quality preset"
            }
        },
        "required": ["renderer_mode", "scenario", "preset"]
    }
)
```

---

## Token Budget for Claude Analysis

**Traditional comparison (no ML):**
- 2 images × 500 tokens (downscaled) = 1,000 tokens
- Report text: ~200 tokens
- **Total: ~1,200 tokens per comparison**

**ML-powered comparison:**
- 2 images × 500 tokens = 1,000 tokens
- LPIPS results: ~100 tokens
- Feature analysis: ~150 tokens
- Quality classifier results: ~100 tokens
- Anomaly detection: ~150 tokens
- Difference heatmap: ~300 tokens
- Report text: ~400 tokens
- **Total: ~2,200 tokens per comparison**

**Regression monitoring (automated):**
- Only notify user if regression detected
- Token cost: 0 tokens (monitoring runs locally)
- Only sends report to Claude when issue found
- **Budget-friendly for continuous monitoring**

---

## Success Metrics

**Phase 3 Success Criteria:**
- ✅ ML comparison tool working with LPIPS
- ✅ Correlation with human judgment >0.90
- ✅ Comparison time <1 second (on GPU)
- ✅ Difference heatmap highlights actual changes
- ✅ Claude provides actionable recommendations

**Phase 4 Success Criteria:**
- ✅ Baseline capture system automated
- ✅ Regression detection catches >95% of known issues
- ✅ False positive rate <5% (medium sensitivity)
- ✅ Continuous monitoring service stable (24/7)
- ✅ Integration with Claude Code (auto-alerts)

**Long-term Success (with custom models):**
- ✅ RTXDI quality classifier >95% accuracy
- ✅ Anomaly detector catches novel issues (85% recall)
- ✅ Training data collection automated
- ✅ Models retrained monthly with new data

---

## Dependencies

**Python packages:**
```txt
# Core ML
torch==2.0.1
torchvision==0.15.2
lpips==0.1.4

# Computer vision
opencv-python==4.8.1
scikit-image==0.21.0
albumentations==1.3.1

# Utilities
numpy==1.24.3
pillow==10.0.0

# Optional (for training)
tensorboard==2.14.0
pytorch-lightning==2.0.7
```

**Hardware:**
- GPU recommended (10× faster than CPU for ML models)
- 4GB VRAM minimum (ResNet50 + LPIPS)
- 8GB VRAM recommended (training custom models)
- CPU fallback available (50× slower)

---

## Next Steps

1. **Install ML dependencies** (5 minutes)
   ```bash
   cd agents/rtxdi-quality-analyzer
   pip install torch torchvision lpips scikit-image opencv-python
   ```

2. **Add ML comparison tool** (2 hours)
   - Create `src/tools/ml_visual_comparison.py`
   - Integrate with agent.py
   - Test with existing screenshots

3. **Collect baselines** (30 minutes)
   - Run `tools/collect_baselines.sh`
   - Store in `PIX/baselines/`

4. **Implement regression detection** (4 hours)
   - Create `src/tools/regression_detector.py`
   - Add continuous monitoring service
   - Test with known regressions

5. **Train custom models** (2-4 weeks)
   - Collect labeled training data
   - Use @neural-network-builder plugin
   - Fine-tune on RTXDI-specific issues

---

**Last Updated:** 2025-10-23
**Status:** Design complete, ready for Phase 3 implementation
**Estimated Total Time:** Phase 3 (3-4 hours) + Phase 4 (6-8 hours) = 10-12 hours
**Owner:** Ben + Claude Code + @neural-network-builder

---

## References

- **LPIPS Paper:** "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" (Zhang et al., CVPR 2018)
- **SSIM:** "Image Quality Assessment: From Error Visibility to Structural Similarity" (Wang et al., IEEE TIP 2004)
- **Neural Network Builder Plugin:** Claude Code Agents SDK
- **RTXDI Documentation:** NVIDIA RTX Direct Illumination SDK
- **PyTorch:** https://pytorch.org/
- **Albumentations:** https://albumentations.ai/
