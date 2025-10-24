# Visual Analysis & ML Enhancement Roadmap

**Created:** 2025-10-24
**Status:** Phase 1 In Progress

This document outlines the roadmap for enhancing PlasmaDX's visual quality analysis system with ML-powered assessment, reference image training, and advanced Agent SDK visualizations.

---

## Overview

**Goal:** Transform visual quality analysis from qualitative human assessment to quantitative ML-powered feedback with actionable, config-specific recommendations.

**Key Innovations:**
1. **Screenshot Metadata Embedding** - Capture full program state alongside images
2. **ML Quality Assessment** - Train models on reference images (Hubble, Interstellar VFX)
3. **Agent SDK Image Return** - Generate annotated screenshots, previews, comparisons
4. **Aggregate Visualizations** - Temporal stability analysis, quality timelines

---

## Phase 1: Screenshot Metadata Embedding ✅ HIGH VALUE

**Timeline:** 1-2 hours
**Status:** 🔄 IN PROGRESS

### Implementation Tasks

**1.1 Add Metadata Schema (Application.h)**
```cpp
struct ScreenshotMetadata {
    // Rendering config
    bool rtxdiEnabled;
    bool rtxdiM5Enabled;
    float temporalBlendFactor;
    int shadowRaysPerLight;
    int lightCount;

    // Particle config
    int particleCount;
    float particleRadius;
    float gravityStrength;

    // Performance
    float fps;
    float frameTime;

    // Camera
    XMFLOAT3 cameraPos;
    XMFLOAT3 cameraLookAt;

    // PINN config
    bool pinnEnabled;
    std::string modelPath;

    // Timestamp
    std::string timestamp;
};
```

**1.2 Capture Metadata in F2 Handler (Application.cpp)**
- Hook into existing `CaptureScreenshot()` function (line ~976-1014)
- Populate metadata struct from current program state
- Serialize to JSON and save as sidecar file

**1.3 JSON Serialization**
- Use simple stringstream-based JSON serialization (no external libs needed)
- Format: `screenshot_YYYY-MM-DD_HH-MM-SS.bmp.json`

**1.4 Update MCP Tool (rtxdi_server.py)**
- Modify `list_recent_screenshots()` to also read metadata
- Add new tool: `get_screenshot_metadata(path) → JSON`
- Enhance `assess_visual_quality()` to load metadata automatically

**1.5 Config-Specific Recommendations**
Update visual quality assessment to provide specific advice:

**Before (generic):**
> "Enable RTXDI M5 temporal accumulation to eliminate patchwork"

**After (specific):**
> "Your RTXDI M5 is disabled (`rtxdi_m5_enabled: false` in metadata). Enable via:
> - ImGui: Check 'RTXDI M5 Temporal Accumulation'
> - Config: Set `rtxdi_temporal_accumulation: true` in configs/builds/Debug.json
> - Expected improvement: Patchwork disappears in ~67ms (8 frames @ 120 FPS)"

### Benefits
- **Precise debugging** - Agent knows exact config causing issues
- **Reproducibility** - Can recreate exact render conditions
- **Regression detection** - Track FPS/quality changes over time
- **A/B testing** - Compare different config settings scientifically

### Example Metadata File
```json
{
  "timestamp": "2025-10-24T04:41:14Z",
  "rendering": {
    "rtxdi_enabled": true,
    "rtxdi_m5_enabled": false,
    "temporal_blend_factor": 0.0,
    "shadow_rays_per_light": 1,
    "light_count": 13
  },
  "particles": {
    "count": 10000,
    "radius": 15.0,
    "gravity_strength": 1.0
  },
  "performance": {
    "fps": 118.4,
    "frame_time_ms": 8.45
  },
  "camera": {
    "position": [0, 500, 1200],
    "look_at": [0, 0, 0]
  },
  "pinn": {
    "enabled": false,
    "model_path": ""
  }
}
```

---

## Phase 2: Basic Image Return Tools 🎨

**Timeline:** 2-3 hours
**Status:** ⏳ PLANNED

### New MCP Tools

**2.1 Annotated Screenshot Analysis**
```python
@server.call_tool()
async def annotate_screenshot(screenshot_path: str) -> ImageContent:
    """Return screenshot with visual annotations highlighting issues"""
    # Load image
    img = Image.open(screenshot_path)
    draw = ImageDraw.Draw(img)

    # Detect issues programmatically
    if detect_rtxdi_patchwork(img):
        # Draw red circles around patchy regions
        for region in patchy_regions:
            draw.ellipse(region, outline='red', width=3)
            draw.text(region[:2], "RTXDI M5 needed", fill='red')

    if detect_weak_rim_lighting(img):
        # Highlight backlit particles with weak glow
        for region in backlit_regions:
            draw.rectangle(region, outline='yellow', width=2)
            draw.text(region[:2], "Weak rim lighting", fill='yellow')

    # Return annotated image to Claude
    return encode_image_as_base64(img)
```

**Example Output:** Screenshot with colored boxes, arrows, and text labels pointing to specific visual issues.

**2.2 Preview Suggestion Tool**
```python
@server.call_tool()
async def preview_suggestion(screenshot_path: str,
                             suggestion: str) -> ImageContent:
    """Generate visual preview of what a suggestion would look like"""
    img = Image.open(screenshot_path)

    if suggestion == "increase_rim_lighting":
        # Apply heuristic enhancement (brighten edges near lights)
        enhanced = apply_rim_lighting_boost(img, intensity=1.5)

    elif suggestion == "smooth_rtxdi_m5":
        # Simulate temporal smoothing with median blur
        enhanced = apply_median_blur(img, kernel_size=5)

    elif suggestion == "enhance_temperature_gradient":
        # Apply color grading toward blackbody colors
        enhanced = apply_color_curve(img, curve=blackbody_gradient)

    # Return side-by-side: original | preview
    comparison = create_side_by_side(img, enhanced)
    add_label(comparison, "Current", position="left")
    add_label(comparison, f"With {suggestion}", position="right")

    return encode_image(comparison)
```

**Use Case:** Agent can show: *"If you implement my rim lighting suggestion, it would look approximately like this →"*

**2.3 Side-by-Side Comparison**
```python
@server.call_tool()
async def create_comparison(before_path: str,
                           after_path: str) -> ImageContent:
    """Create side-by-side comparison with diff heatmap"""
    before = Image.open(before_path)
    after = Image.open(after_path)

    # Create 3-panel comparison
    comparison = create_triptych(
        left=before,
        center=create_diff_heatmap(before, after),
        right=after
    )

    # Add labels and metadata
    add_text(comparison, "Before", position="top-left")
    add_text(comparison, "Difference (LPIPS)", position="top-center")
    add_text(comparison, "After", position="top-right")

    # Add quality scores
    before_meta = load_metadata(before_path)
    after_meta = load_metadata(after_path)
    add_text(comparison, f"FPS: {before_meta['fps']:.1f}", position="bottom-left")
    add_text(comparison, f"FPS: {after_meta['fps']:.1f}", position="bottom-right")

    return encode_image(comparison)
```

### Implementation Details

**Heuristic Image Analysis:**
- **RTXDI patchwork detection:** Color variance analysis on outer disk regions
- **Weak rim lighting:** Brightness analysis of backlit particle edges
- **Temperature gradient:** Color histogram analysis (expect hot→cool progression)

**Image Manipulation Libraries:**
- Pillow (already installed) for basic operations
- OpenCV (optional) for advanced computer vision
- NumPy for pixel-level operations

---

## Phase 3: Reference Image Library 📚

**Timeline:** Ongoing (1-2 hours initial setup)
**Status:** ⏳ PLANNED

### Directory Structure
```
screenshots/
├── reference/
│   ├── golden_standard/        # A+ grade (95-100)
│   │   ├── hubble_accretion_disk_01.png
│   │   ├── interstellar_gargantua_still.png
│   │   ├── nvidia_volumetric_demo.png
│   │   └── plasmadx_best_render_2025-10-17.bmp
│   ├── good/                   # B grade (80-90)
│   │   ├── plasmadx_multi_light_v1.bmp
│   │   └── ...
│   ├── issues/                 # C-D grade (60-75)
│   │   ├── rtxdi_patchwork_visible.bmp
│   │   └── weak_rim_lighting.bmp
│   └── failures/               # F grade (<60)
│       ├── black_screen_bug.bmp
│       └── no_rt_lighting.bmp
└── annotations/                # JSON quality scores
    ├── hubble_accretion_disk_01.json
    ├── plasmadx_best_render_2025-10-17.json
    └── ...
```

### Data Sources

**Astrophysics References:**
- [NASA Hubble Space Telescope Gallery](https://hubblesite.org/images/gallery)
  - Search: "accretion disk", "black hole", "nebula"
  - 4K resolution images, public domain
- [ESO (European Southern Observatory)](https://www.eso.org/public/images/)
  - VLT (Very Large Telescope) accretion disk observations
  - High-resolution, scientifically accurate

**VFX References:**
- Interstellar (2014) - Gargantua black hole stills
  - Kip Thorne's scientifically accurate visualization
  - Available in Blu-ray art books, VFX breakdowns
- NVIDIA/AMD volumetric rendering tech demos
  - RTX tech demos, ray tracing showcases
- Blender volumetric rendering examples

**Academic Simulations:**
- ArXiv papers with accretion disk visualizations
- University research labs (MIT, Caltech) black hole simulations

### Annotation Format
```json
{
  "screenshot": "hubble_accretion_disk_01.png",
  "source": "NASA Hubble Space Telescope",
  "date": "2023-04-15",
  "grade": "A+",
  "overall_score": 98,
  "scores": {
    "volumetric_depth": 100,
    "lighting_quality": 98,
    "temperature_gradient": 95,
    "rtxdi_quality": "N/A (real photo)",
    "shadow_quality": 97,
    "scattering": 100,
    "temporal_stability": "N/A (static image)"
  },
  "notes": "Golden standard for volumetric depth. Perfect atmospheric scattering. Temperature gradient shows hot inner disk (white) to cool outer (orange/red).",
  "key_features": [
    "Strong rim lighting on dust lanes",
    "Volumetric fog-like quality",
    "Realistic blackbody emission colors",
    "Multi-scale detail (small particles to large structures)"
  ]
}
```

### Manual Annotation Process
1. Collect 20-30 reference images (10 golden, 5 good, 5 issues, 5 failures)
2. For each, manually score 7 quality dimensions (0-100)
3. Write detailed notes on what makes it good/bad
4. Save JSON annotation files

---

## Phase 4: ML Quality Assessment Model 🤖

**Timeline:** 1-2 weeks
**Status:** 🔬 RESEARCH

### Model Architecture

**Option A: Quality Regression Network (Recommended)**
```python
class QualityAssessmentNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Pretrained ResNet50 feature extractor
        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove classification head

        # Quality score heads (one per rubric dimension)
        self.depth_head = nn.Linear(2048, 1)        # Volumetric depth
        self.lighting_head = nn.Linear(2048, 1)     # Lighting quality
        self.temp_head = nn.Linear(2048, 1)         # Temperature gradient
        self.rtxdi_head = nn.Linear(2048, 1)        # RTXDI quality
        self.shadow_head = nn.Linear(2048, 1)       # Shadow quality
        self.scatter_head = nn.Linear(2048, 1)      # Scattering
        self.temporal_head = nn.Linear(2048, 1)     # Temporal stability

    def forward(self, x):
        features = self.backbone(x)
        return {
            'volumetric_depth': self.depth_head(features),
            'lighting_quality': self.lighting_head(features),
            'temperature_gradient': self.temp_head(features),
            'rtxdi_quality': self.rtxdi_head(features),
            'shadow_quality': self.shadow_head(features),
            'scattering': self.scatter_head(features),
            'temporal_stability': self.temporal_head(features)
        }

# Training
loss = MSE(predicted_scores, human_annotated_scores)
```

**Model Size:** ~100 MB (ResNet50 backbone)
**Inference Time:** ~50ms on CPU, ~5ms on GPU
**Accuracy Target:** 85-90% correlation with human judgment

**Option B: Contrastive Learning (Advanced)**
```python
# Learn embedding space where similar-quality images cluster together
# Use triplet loss: (anchor, positive, negative)
model = ContrastiveLearningNet()
loss = TripletLoss(anchor=good_render,
                   positive=reference_image,
                   negative=poor_render)
```

### Training Dataset

**Minimum viable dataset:**
- 100 annotated images (mix of PlasmaDX + references)
- 20 golden standard
- 30 good quality
- 30 fair quality
- 20 poor quality

**Ideal dataset:**
- 500+ annotated images
- 100 reference images (Hubble, VFX)
- 400 PlasmaDX renders (across development timeline)

### Data Collection Strategy

**Automated PlasmaDX screenshots:**
```python
# Test harness to generate diverse training data
for config in test_configs:
    # Vary RTXDI settings
    for rtxdi_m5 in [True, False]:
        for shadow_rays in [1, 4, 8, 16]:
            # Vary camera angles
            for angle in camera_angles:
                # Capture screenshot
                screenshot = capture_with_config(config, angle)
                # Manually annotate quality
                annotate_screenshot(screenshot)
```

**Goal:** Cover full configuration space to train robust model

### Deployment as MCP Tool

```python
@server.call_tool()
async def assess_quality_ml(screenshot_path: str) -> dict:
    """ML-powered quality assessment (objective scores)"""
    # Load model (lazy loading)
    model = load_quality_model()

    # Preprocess image
    img = preprocess(screenshot_path)

    # Inference
    with torch.no_grad():
        scores = model(img)

    # Return scores
    return {
        "overall_score": np.mean(list(scores.values())),
        "dimension_scores": scores,
        "grade": calculate_letter_grade(scores),
        "confidence": calculate_confidence(scores)
    }
```

**Benefits:**
- Objective quality metric (not just human judgment)
- Fast assessment (50ms vs 30s for human)
- Consistent scoring (no human bias variation)
- Tracks improvement trends automatically

---

## Phase 5: Style Transfer & Enhancement 🎨

**Timeline:** Research project (2-4 weeks)
**Status:** 🔬 EXPERIMENTAL

### Concept

Learn the "style" of photorealistic accretion disks and transfer to PlasmaDX renders.

**Goal:** Generate "target aesthetic" previews showing what golden standard quality looks like for a given scene.

### Approach A: CycleGAN Domain Transfer

```python
# Train unpaired image-to-image translation
CycleGAN(
    domain_A=plasmadx_renders,  # Source domain
    domain_B=hubble_references   # Target domain
) → enhanced_plasmadx_renders

# Loss functions:
# - Adversarial loss (fool discriminator)
# - Cycle consistency loss (A→B→A ≈ A)
# - Identity loss (preserve content)
```

**Use Case:** Transform current PlasmaDX render to "Hubble-like" quality automatically.

**Challenges:**
- Requires large dataset (hundreds of images)
- May hallucinate unrealistic features
- Computationally expensive training

### Approach B: AdaIN Style Transfer

```python
# Adaptive Instance Normalization (real-time style transfer)
StyleTransferNet(
    content=plasmadx_screenshot,
    style=hubble_reference
) → stylized_screenshot

# Preserves content structure, applies reference style
```

**Benefits:**
- Works with single reference image
- Real-time inference (~100ms)
- Preserves scene content (just changes "look")

**Use Case:** Show "this is what your scene would look like with Hubble-quality rendering"

### Approach C: Perceptual Loss for Rendering

```python
# Use LPIPS perceptual loss during shader development
class PerceptualRenderingLoss:
    def __init__(self):
        self.lpips = lpips.LPIPS(net='vgg')

    def compute_loss(self, rendered, reference):
        # Perceptual distance (how different do they look?)
        perceptual_loss = self.lpips(rendered, reference)

        # Physics loss (preserve physical accuracy)
        physics_loss = check_physics_constraints(rendered)

        return perceptual_loss + λ * physics_loss

# Use to guide PINN training or shader parameter tuning
```

**Use Case:** Automatically tune shader parameters to match reference aesthetic while preserving physics.

---

## Advanced Visualization Tools 📊

### Tool 1: Temporal Stability Analysis

```python
@server.call_tool()
async def analyze_temporal_stability(screenshot_dir: str,
                                     frame_count: int = 30) -> ImageContent:
    """Analyze flickering by comparing consecutive frames"""
    # Load 30 consecutive screenshots
    frames = load_consecutive_frames(screenshot_dir, count=30)

    # Compute per-pixel variance over time
    variance_map = np.var(frames, axis=0)

    # Normalize to heatmap (blue=stable, red=flickering)
    heatmap = create_heatmap(variance_map, colormap='jet')

    # Overlay on first frame
    result = overlay_heatmap(frames[0], heatmap, alpha=0.6)

    # Add annotation
    add_text(result, "Red regions = high temporal instability",
             position="bottom", color="white")

    # Add statistics
    mean_variance = np.mean(variance_map)
    max_variance = np.max(variance_map)
    add_text(result, f"Mean variance: {mean_variance:.3f}",
             position="top-left")
    add_text(result, f"Max variance: {max_variance:.3f}",
             position="top-right")

    return encode_image(result)
```

**Use Case:** Capture 30 frames with F2, run tool, agent shows exactly where flickering occurs (e.g., RTXDI M5 instability zones).

### Tool 2: Quality Progression Timeline

```python
@server.call_tool()
async def create_quality_timeline(date_range: str = "last_7_days") -> ImageContent:
    """Show quality improvement over time as visual timeline"""
    # Get all screenshots in date range
    screenshots = get_screenshots_in_date_range(date_range)

    # Assess quality for each
    quality_scores = []
    for screenshot in screenshots:
        scores = assess_quality_ml(screenshot)
        quality_scores.append(scores['overall_score'])

    # Create filmstrip-style visualization
    filmstrip = create_horizontal_filmstrip(screenshots, max_width=1920)

    # Overlay quality score graph
    graph = create_line_graph(quality_scores,
                              color='green',
                              ylabel='Quality Score',
                              ylim=[0, 100])
    overlay_graph(filmstrip, graph, position='bottom')

    # Add annotations for major changes
    for i, (score, prev_score) in enumerate(zip(quality_scores[1:], quality_scores)):
        if abs(score - prev_score) > 10:  # Significant change
            add_marker(filmstrip, position=i, text=f"+{score-prev_score:.1f}")

    return encode_image(filmstrip)
```

**Use Case:** Track quality improvement over development timeline, visualize impact of major changes (e.g., RTXDI M5 implementation spike).

### Tool 3: Reference Comparison Grid

```python
@server.call_tool()
async def compare_to_references(current_screenshot: str,
                                num_references: int = 4) -> ImageContent:
    """Compare current render to top reference images in grid"""
    # Load current screenshot
    current = Image.open(current_screenshot)

    # Find most similar golden standard references
    refs = find_similar_references(current, count=num_references)

    # Create 2x3 grid layout:
    # [ref1] [ref2]
    # [current]  ← center position
    # [ref3] [ref4]
    grid = create_grid_layout([
        refs[0], refs[1],
        None, current, None,
        refs[2], refs[3]
    ], cols=3)

    # Add similarity scores to each reference
    for i, ref in enumerate(refs):
        lpips_score = compute_lpips_similarity(current, ref)
        position = get_grid_position(i, center_empty=True)
        add_label(grid, f"LPIPS: {lpips_score:.3f}", position=position)
        add_label(grid, ref['name'], position=position, offset_y=20)

    # Label current screenshot
    add_label(grid, "Current Render", position='center',
              color='yellow', font_size=20)

    # Add overall assessment
    avg_similarity = np.mean([s['lpips'] for s in refs])
    assessment = "Excellent!" if avg_similarity < 0.1 else \
                 "Good" if avg_similarity < 0.2 else \
                 "Needs improvement"
    add_text(grid, f"Overall similarity: {assessment} ({avg_similarity:.3f})",
             position='bottom-center', color='white')

    return encode_image(grid)
```

**Use Case:** Instantly see how current render compares to 4 best reference images, with similarity scores.

---

## Integration with Existing Systems

### Screenshot Capture Workflow

**Before Phase 1:**
```
User presses F2 → Screenshot saved → Done
```

**After Phase 1:**
```
User presses F2 →
  1. Screenshot saved (screenshot_XXX.bmp)
  2. Metadata captured (screenshot_XXX.bmp.json)
  3. MCP tool can now provide config-specific advice
```

**After Phase 2:**
```
User presses F2 →
  1. Screenshot + metadata saved
  2. Agent can assess quality
  3. Agent can return annotated image showing issues
  4. Agent can preview what suggestions would look like
```

**After Phase 4:**
```
User presses F2 →
  1. Screenshot + metadata saved
  2. ML model instantly scores quality (50ms)
  3. Agent compares to reference library
  4. Agent provides specific recommendations
  5. Agent tracks quality trend over time
```

### MCP Server Tool Evolution

**Current (4 tools):**
- `list_recent_screenshots`
- `compare_performance`
- `analyze_pix_capture`
- `compare_screenshots_ml`
- `assess_visual_quality`

**Phase 1 (+2 tools):**
- `get_screenshot_metadata` ← NEW
- `assess_visual_quality` ← ENHANCED (reads metadata)

**Phase 2 (+3 tools):**
- `annotate_screenshot` ← NEW
- `preview_suggestion` ← NEW
- `create_comparison` ← NEW (enhanced from existing)

**Phase 4 (+1 tool):**
- `assess_quality_ml` ← NEW (ML-powered)

**Advanced (+3 tools):**
- `analyze_temporal_stability` ← NEW
- `create_quality_timeline` ← NEW
- `compare_to_references` ← NEW

**Total: 13 tools** for comprehensive visual analysis.

---

## Success Metrics

### Phase 1 Success Criteria
- ✅ Metadata JSON files saved alongside every screenshot
- ✅ MCP tool can read and parse metadata
- ✅ Agent recommendations cite specific config values
- ✅ Example: "Your `shadow_rays_per_light: 1` explains temporal noise"

### Phase 2 Success Criteria
- ✅ Agent can return annotated screenshots with issue highlights
- ✅ Agent can preview visual improvements before implementation
- ✅ Annotations are visually clear and actionable

### Phase 3 Success Criteria
- ✅ 20+ reference images collected and annotated
- ✅ Golden standard library established
- ✅ Agent can compare current render to references

### Phase 4 Success Criteria
- ✅ ML model trained with 85%+ correlation to human judgment
- ✅ Inference time <100ms on CPU
- ✅ Model deployed as MCP tool
- ✅ Quality tracking dashboard operational

### Overall Success Criteria
- ✅ Agent provides **specific, actionable** recommendations (not generic)
- ✅ Users can see **visual previews** of suggestions before implementing
- ✅ Quality **improves measurably** over time (tracked by ML model)
- ✅ Development velocity increases (faster iteration with immediate feedback)

---

## Future Extensions

### Real-Time Quality Feedback
Integrate ML model into PlasmaDX application:
```cpp
// In Application::Render() after swapchain present
if (ImGui::IsKeyPressed(ImGuiKey_F3)) {  // F3 = quick quality check
    auto screenshot = CaptureFramebuffer();
    auto scores = m_qualityModel->Assess(screenshot);

    // Display overlay
    DrawQualityOverlay(scores);  // Shows 7 dimension scores
}
```

**Use Case:** Instant quality feedback during shader development.

### Automated Regression Testing
```python
# CI/CD pipeline quality gate
def test_visual_quality():
    screenshot = render_test_scene()
    quality = assess_quality_ml(screenshot)

    assert quality['overall_score'] > 75, "Quality regression detected!"
    assert quality['rtxdi_quality'] > 70, "RTXDI quality below threshold"
```

### Parameter Optimization
```python
# Bayesian optimization to find best shader parameters
from skopt import gp_minimize

def objective(params):
    rim_lighting_intensity, scatter_g = params

    # Render with these params
    screenshot = render_with_params(rim_lighting_intensity, scatter_g)

    # Assess quality
    quality = assess_quality_ml(screenshot)

    # Maximize quality score
    return -quality['overall_score']

# Find optimal parameters
result = gp_minimize(objective,
                     dimensions=[(0.5, 2.5), (0.3, 0.9)],
                     n_calls=50)

print(f"Optimal rim lighting: {result.x[0]}")
print(f"Optimal scatter g: {result.x[1]}")
```

**Use Case:** Automatically find shader parameters that maximize visual quality.

---

## Resource Requirements

### Phase 1 (Metadata)
- **Development time:** 1-2 hours
- **Storage:** ~5 KB per screenshot (JSON metadata)
- **Dependencies:** None (uses standard C++/Python JSON)

### Phase 2 (Image Tools)
- **Development time:** 2-3 hours
- **Storage:** Negligible (images generated on-demand)
- **Dependencies:** Pillow, NumPy (already installed)

### Phase 3 (Reference Library)
- **Development time:** 1-2 hours (initial setup) + ongoing curation
- **Storage:** ~1-2 GB (100-200 reference images @ 4K)
- **Dependencies:** None

### Phase 4 (ML Model)
- **Development time:** 1-2 weeks (training + testing)
- **Storage:** ~100 MB (ResNet50 model weights)
- **Compute:** GPU recommended for training (RTX 4060 Ti sufficient)
- **Dependencies:** PyTorch, torchvision, scikit-learn

### Phase 5 (Style Transfer)
- **Development time:** 2-4 weeks (research project)
- **Storage:** ~500 MB (CycleGAN models)
- **Compute:** GPU required (RTX 4060 Ti minimum)
- **Dependencies:** PyTorch, torchvision

---

## Technical Notes

### Image Size Optimization
All MCP tool image returns should:
- Resize to max 1280x720 (preserve aspect ratio)
- Convert to PNG with optimization
- Target: <500 KB per image (avoid API 400 errors)

### Lazy Loading Pattern
ML models should use lazy loading to avoid MCP timeout:
```python
class QualityModel:
    def __init__(self):
        self.model = None  # Lazy load on first use

    def _ensure_loaded(self):
        if self.model is None:
            self.model = torch.load('quality_model.pt')
            self.model.eval()
```

### Metadata Schema Versioning
```json
{
  "schema_version": "1.0",
  "timestamp": "...",
  "rendering": {...}
}
```

Allows future schema updates without breaking compatibility.

---

## Documentation Updates

After each phase, update:
- `CLAUDE.md` - Add new MCP tools to tool list
- `agents/rtxdi-quality-analyzer/README.md` - Document new features
- `screenshots/README.md` - Explain metadata format and reference library

---

## Conclusion

This roadmap transforms PlasmaDX's visual analysis from manual, qualitative assessment to automated, quantitative feedback with ML-powered insights. The progression is carefully designed:

1. **Phase 1** - Foundation (metadata capture)
2. **Phase 2** - Immediate visual feedback (annotations, previews)
3. **Phase 3** - Reference library (objective quality targets)
4. **Phase 4** - ML automation (objective scoring)
5. **Phase 5** - Advanced research (style transfer)

Each phase builds on the previous, delivering incremental value while maintaining a clear path to the ultimate goal: **automated, actionable visual quality feedback that accelerates development**.

**Next Step:** Implement Phase 1 (estimated 1-2 hours).

---

**Last Updated:** 2025-10-24
**Maintained by:** Ben + Claude Code sessions
