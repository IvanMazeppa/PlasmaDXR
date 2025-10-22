# Adaptive Quality System - Quick Start Guide

## 5-Minute Setup

### Step 1: Install Python Dependencies

```bash
cd ml
pip install -r requirements.txt
```

### Step 2: Generate Synthetic Training Data

```bash
python generate_training_data.py
```

**Output:** `training_data/synthetic_performance_data.csv` (10,000 samples)

### Step 3: Train the ML Model

```bash
python train_adaptive_quality.py
```

**Output:**
- `models/adaptive_quality.model` (trained model)
- `analysis/prediction_vs_actual.png` (accuracy plot)
- `analysis/feature_importance.png` (feature ranking)

**Expected Results:**
```
Model Performance:
  MAE:  0.8-1.2 ms
  RMSE: 1.0-1.5 ms
  R²:   0.85-0.95
```

### Step 4: Enable in Application

1. Build and run PlasmaDX-Clean:
   ```bash
   MSBuild PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64
   ./build/Debug/PlasmaDX-Clean.exe
   ```

2. Open ImGui panel (F1 if hidden)

3. Expand **"Adaptive Quality (ML)"** section

4. Check **"Enable Adaptive Quality"**

5. Select target FPS: **120 FPS** (recommended)

6. Watch the magic happen! ✨

## What You'll See

### ImGui Panel

```
Adaptive Quality (ML)
  ✓ ML-Based Performance Prediction

  ☑ Enable Adaptive Quality

  Target FPS: [ 120 FPS ▼ ]

  Current Quality Level:
    High

  Predicted Frame Time: 7.45 ms
  Target Frame Time: 8.33 ms

  [████████░░] Performance Bar (Green = Good)

  Training Data Collection
  ☐ Collect Performance Data
```

### Quality Levels

The system will automatically adjust between:

| Level | Shadow Rays | Features | Target FPS |
|-------|------------|----------|-----------|
| **Ultra** | 16 | All enabled | 30-60 |
| **High** | 8 | Most enabled | 90-120 |
| **Medium** | 4 | Essential | 120-144 |
| **Low** | 1+temporal | Minimal | 144-165 |
| **Minimal** | 1 | Bare minimum | 165+ |

### Color Coding

- **Magenta** = Ultra (max quality)
- **Green** = High (recommended)
- **Yellow** = Medium (balanced)
- **Orange** = Low (performance)
- **Red** = Minimal (maximum FPS)

## Testing the System

### Test 1: Particle Count Scaling

1. Enable Adaptive Quality
2. Set target: 120 FPS
3. Adjust particle count in ImGui:
   - Start: 10K particles → Should stay at **High**
   - Increase: 50K particles → May drop to **Medium**
   - Increase: 100K particles → May drop to **Low**

### Test 2: Feature Toggles

1. Enable expensive features:
   - In-Scattering (F6) → Quality should drop
   - God Rays → Quality should drop

2. Disable features:
   - Quality should increase back up

### Test 3: Camera Distance

1. Move camera close (100-400 units)
   - More particles on screen
   - Quality may drop

2. Move camera far (1000+ units)
   - Fewer particles visible
   - Quality should increase

## Collecting Real Data (Optional)

For best results on your hardware:

### 1. Enable Data Collection

ImGui → Adaptive Quality (ML) → ☑ Collect Performance Data

### 2. Run Test Scenarios (5-10 minutes)

**Particle Counts:**
- 10K, 25K, 50K, 100K

**Features:**
- Shadow rays: 1, 4, 8, 16
- Toggle in-scattering
- Toggle phase function
- Toggle RT lighting

**Camera Distances:**
- Close (200 units)
- Medium (600 units)
- Far (1500 units)

### 3. Stop Collection

Uncheck "Collect Performance Data"

### 4. Retrain Model

```bash
python train_adaptive_quality.py --data training_data/performance_data.csv
```

### 5. Restart Application

New model will be loaded automatically!

## Troubleshooting

### "No trained model found"

**Problem:** Model file missing
**Solution:** Run `python train_adaptive_quality.py`

### Quality keeps changing

**Problem:** FPS near threshold
**Solution:** System uses hysteresis (30 frames). This is normal.

### Predictions seem off

**Problem:** Model not trained for your hardware
**Solution:** Collect real data and retrain

### Data file is empty

**Problem:** Write permissions or not running long enough
**Solution:**
1. Check `ml/training_data/` exists
2. Run for at least 1 minute with collection enabled

## Next Steps

### Advanced Usage

- **Multi-GPU Profiling:** Collect data from different GPUs
- **Custom Cost Models:** Edit `generate_training_data.py`
- **Model Comparison:** Try `--model random_forest` vs `--model gradient_boosting`

### Performance Tuning

- **Hysteresis:** Adjust in `AdaptiveQualitySystem.cpp` (default: 30 frames)
- **Quality Change Delay:** Adjust in `AdaptiveQualitySystem.cpp` (default: 2 seconds)
- **Target FPS:** Match your monitor refresh rate (60/120/144/165)

### Share Your Results

Help improve the model:
1. Collect diverse data
2. Test on different hardware
3. Report accuracy (R² score from training output)

## FAQ

**Q: Does this work without a trained model?**
A: Yes! Falls back to heuristic cost model. Less accurate but functional.

**Q: What's the performance overhead?**
A: < 0.1ms per frame. Negligible.

**Q: Can I disable it temporarily?**
A: Yes, uncheck "Enable Adaptive Quality" in ImGui.

**Q: Does it work with RTXDI?**
A: Yes! System detects RTXDI vs multi-light automatically.

**Q: Can I use this for other games/engines?**
A: Yes! Port `AdaptiveQualitySystem.h/cpp` and adjust features.

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review `ml/README.md` for detailed docs
3. File issue in project repository

---

**Enjoy automatic performance optimization!** 🚀
