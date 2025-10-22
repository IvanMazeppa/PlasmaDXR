# Adaptive Quality System - ML-Based Performance Prediction

## Overview

The Adaptive Quality System uses machine learning to predict frame time based on scene complexity and automatically adjusts rendering quality settings to maintain your target FPS (60/120/144).

**Key Features:**
- Real-time frame time prediction
- Automatic quality adjustment (Ultra → High → Medium → Low → Minimal)
- Hysteresis to prevent quality oscillation
- Performance data collection for model training
- Lightweight decision tree inference (no external ML libs in C++)

## Quick Start

### 1. Generate Initial Training Data

```bash
# Generate synthetic performance data
cd ml
python generate_training_data.py --output training_data/synthetic_performance_data.csv --samples 10000
```

### 2. Train the ML Model

```bash
# Train model with synthetic data
python train_adaptive_quality.py --data training_data/synthetic_performance_data.csv --output models/adaptive_quality.model

# Or use custom model types
python train_adaptive_quality.py --data training_data/synthetic_performance_data.csv --model gradient_boosting
python train_adaptive_quality.py --data training_data/synthetic_performance_data.csv --model decision_tree
```

### 3. Enable in Application

1. Launch PlasmaDX-Clean
2. Open ImGui panel (F1 if hidden)
3. Expand "Adaptive Quality (ML)" section
4. Check "Enable Adaptive Quality"
5. Set your target FPS (60/120/144)
6. Watch quality auto-adjust!

## Collecting Real Performance Data

For best results, collect real performance data from your hardware:

### Step 1: Enable Data Collection

1. Launch PlasmaDX-Clean
2. Open "Adaptive Quality (ML)" section
3. Check "Collect Performance Data"
4. Recording indicator appears

### Step 2: Test Different Scenarios

Run through various scenarios to build a comprehensive dataset:

**Particle Count Variations:**
- 10K particles
- 25K particles
- 50K particles
- 100K particles

**Feature Combinations:**
- Shadow rays: 1, 4, 8, 16
- With/without in-scattering
- With/without phase function
- With/without RT lighting
- With/without god rays

**Camera Distances:**
- Close (100-400 units)
- Medium (400-800 units)
- Far (800-2000 units)

**Light Counts:**
- Few lights (1-5)
- Medium (6-10)
- Many (11-16)

### Step 3: Train with Real Data

```bash
# Train model with collected data
python train_adaptive_quality.py --data training_data/performance_data.csv --output models/adaptive_quality.model

# Generate analysis plots
python train_adaptive_quality.py --data training_data/performance_data.csv --plots analysis
```

Check `analysis/` directory for:
- `prediction_vs_actual.png` - Model accuracy
- `residuals.png` - Prediction errors
- `feature_importance.png` - Which features matter most

## Quality Levels

### Ultra (Best Visual Quality)
- 16 shadow rays per light
- All features enabled
- God rays: 1.0 density
- Target: 30-60 FPS

### High (Recommended)
- 8 shadow rays per light
- Most features enabled
- No in-scattering (too expensive)
- God rays: 0.5 density
- Target: 90-120 FPS

### Medium (Balanced)
- 4 shadow rays per light
- Essential features only
- God rays: 0.2 density
- Target: 120-144 FPS

### Low (Performance)
- 1 shadow ray + temporal filtering
- Phase function only
- No god rays
- Target: 144-165 FPS

### Minimal (Maximum Performance)
- 1 shadow ray
- No phase function
- RT lighting disabled
- No god rays
- Target: 165+ FPS

## How It Works

### Input Features (12 total)

**Scene Complexity:**
1. Particle count (10K-100K)
2. Light count (1-16)
3. Camera distance (100-2000)

**Quality Settings:**
4. Shadow rays per light (1-16)
5. Use shadow rays (0/1)
6. Use in-scattering (0/1)
7. Use phase function (0/1)
8. Use anisotropic Gaussians (0/1)
9. Enable temporal filtering (0/1)
10. Use RTXDI (0/1)
11. Enable RT lighting (0/1)
12. God ray density (0.0-1.0)

### Output

**Predicted frame time** (milliseconds)

### Quality Adjustment Algorithm

1. **Measure** current frame time
2. **Predict** frame time for each quality level
3. **Select** highest quality that meets target FPS
4. **Apply** hysteresis (requires 30 sustained frames before change)
5. **Smooth** transitions (minimum 2 seconds between changes)

### Hysteresis Prevents Oscillation

- **Reduce quality:** After 30 frames below target (0.5s @ 60fps)
- **Increase quality:** After 60 frames above target (1.0s @ 60fps)
- **Minimum delay:** 2 seconds between any quality change

## Performance Impact

**Overhead:** < 0.1ms per frame
- Decision tree inference: ~0.01ms
- Feature extraction: ~0.05ms
- Quality application: ~0.02ms

**Total:** Negligible compared to rendering (5-20ms)

## Model Architecture

### Default: Gradient Boosting Regressor

**Training:**
- 100 estimators
- Max depth: 5
- Learning rate: 0.1
- Min samples split: 10

**Inference:**
- Exports single decision tree
- ~100-500 nodes typical
- Binary format: ~5KB
- No external dependencies

### Alternative: Random Forest

Better for sparse data:
```bash
python train_adaptive_quality.py --model random_forest
```

### Alternative: Simple Decision Tree

Fastest inference:
```bash
python train_adaptive_quality.py --model decision_tree
```

## File Structure

```
ml/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── generate_training_data.py           # Synthetic data generator
├── train_adaptive_quality.py           # Training pipeline
├── training_data/
│   ├── synthetic_performance_data.csv  # Generated data
│   └── performance_data.csv            # Real collected data
├── models/
│   └── adaptive_quality.model          # Trained model (binary)
└── analysis/
    ├── prediction_vs_actual.png        # Accuracy plot
    ├── residuals.png                   # Error analysis
    └── feature_importance.png          # Feature ranking
```

## Troubleshooting

### Model not loading

**Symptom:** "No trained model found, using default heuristics"
**Solution:** Train a model first with `train_adaptive_quality.py`

### Poor predictions

**Symptom:** Quality keeps changing or predictions are off
**Solution:**
1. Collect more real data (at least 1000 samples)
2. Cover full range of scenarios
3. Retrain with `--model gradient_boosting`

### Quality oscillates

**Symptom:** Quality level bounces between settings
**Solution:** System uses hysteresis by default. If still oscillating:
1. Check if FPS target is realistic for your hardware
2. Collect more training data near target FPS
3. Increase hysteresis threshold in AdaptiveQualitySystem.cpp

### Data collection not working

**Symptom:** CSV file is empty or not created
**Solution:**
1. Check write permissions for `ml/training_data/`
2. Ensure "Collect Performance Data" is checked in ImGui
3. Run application for at least 1 minute to collect samples

## Advanced Usage

### Custom Cost Models

Edit `generate_training_data.py` to match your hardware:

```python
# Adjust cost multipliers for your GPU
shadowCost = shadowRaysPerLight * lightCount * 0.35  # Change 0.35
particleCost = (particleCount / 10000.0) * 0.8       # Change 0.8
```

### Multi-GPU Training Data

Collect data from different GPUs and combine:

```bash
# RTX 4060 Ti data
python train_adaptive_quality.py --data training_data/rtx4060ti_data.csv

# RTX 3080 data
python train_adaptive_quality.py --data training_data/rtx3080_data.csv

# Combine datasets
cat training_data/rtx4060ti_data.csv training_data/rtx3080_data.csv > combined.csv
python train_adaptive_quality.py --data combined.csv
```

### Export Model Metrics

```bash
python train_adaptive_quality.py --data training_data/performance_data.csv --plots analysis

# Check R² score in output
# R² > 0.90 = Excellent
# R² > 0.80 = Good
# R² < 0.70 = Need more data
```

## Dependencies

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

## Future Enhancements

**Planned Features:**
- Per-GPU model profiles (RTX 4060 Ti, RTX 3080, etc.)
- Online learning (update model during runtime)
- Multi-objective optimization (FPS + visual quality score)
- Neural network models (ONNX export for GPU inference)
- Adaptive hysteresis based on frame time variance

## Contributing

To improve the model:

1. Collect diverse performance data
2. Test on different hardware
3. Share training data (anonymized)
4. Report prediction accuracy issues

## License

Part of PlasmaDX-Clean project.
