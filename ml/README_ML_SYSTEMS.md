# ML Systems Directory

This directory contains all machine learning systems for PlasmaDX-Clean, including PINN physics acceleration and adaptive quality prediction.

---

## Directory Structure

```
ml/
├── pinn_accretion_disk.py          # PINN training script (530 lines)
├── collect_physics_data.py         # GPU buffer → training data (300 lines)
├── test_pinn.py                     # PINN model validation
├── validate_onnx_model.py           # ONNX C++ compatibility check
├── train_adaptive_quality.py       # Adaptive quality system training
├── test_model.py                    # Adaptive quality model validation
├── generate_training_data.py       # Training data generator
├── requirements_pinn.txt            # Python dependencies for PINN
├── models/
│   ├── pinn_accretion_disk.onnx     # Trained PINN model (264 KB)
│   ├── pinn_config.json             # PINN model configuration
│   └── adaptive_quality.model       # Adaptive quality model (16 KB)
└── training_data/
    ├── physics_data.npz             # PINN training dataset
    └── performance_data.csv         # Quality prediction dataset
```

---

## ML System 1: PINN Physics Acceleration

**Status:** ✅ Python training complete, ✅ C++ integration complete

**Purpose:** Accelerate particle physics simulation at high particle counts (50K+) using physics-informed neural networks.

**Files:**
- `pinn_accretion_disk.py` - Main training script
- `collect_physics_data.py` - Convert GPU buffer dumps to training data
- `test_pinn.py` - Validate trained PINN model
- `validate_onnx_model.py` - ✅ **NEW** - Verify C++ compatibility
- `models/pinn_accretion_disk.onnx` - Trained model for C++ inference

### Training Workflow

```bash
# 1. Collect training data from GPU
../build/Debug/PlasmaDX-Clean.exe --dump-buffers 120
python collect_physics_data.py --input ../PIX/buffer_dumps

# 2. Train PINN (~20 minutes on GPU)
python pinn_accretion_disk.py

# 3. Test PINN model
python test_pinn.py --model models/pinn_accretion_disk.onnx

# 4. Validate ONNX compatibility
python validate_onnx_model.py
```

### Expected Training Output

```
Epoch 2000/2000
  Total Loss: 0.000234
  Data Loss: 0.000156
  Physics Losses:
    keplerian: 0.000012
    angular_momentum: 0.000034
    energy: 0.000008
    gr: 0.000024

Model exported to ml/models/pinn_accretion_disk.onnx ✅
```

### C++ Integration

**Header:** `src/ml/PINNPhysicsSystem.h` (152 lines)
**Implementation:** `src/ml/PINNPhysicsSystem.cpp` (415 lines)

**Usage:**
```cpp
#include "ml/PINNPhysicsSystem.h"

PINNPhysicsSystem pinn;
pinn.Initialize("ml/models/pinn_accretion_disk.onnx");
pinn.SetEnabled(true);
pinn.SetHybridMode(true);

// Batch inference
pinn.PredictForcesBatch(positions, velocities, forces, count, time);
```

**Documentation:**
- `PINN_CPP_INTEGRATION_GUIDE.md` - Comprehensive integration guide
- `PINN_CPP_IMPLEMENTATION_SUMMARY.md` - Implementation summary and next steps

---

## ML System 2: Adaptive Quality Prediction

**Status:** ✅ Complete and operational

**Purpose:** Predict frame time based on scene complexity and automatically adjust quality settings to maintain target FPS.

**Files:**
- `train_adaptive_quality.py` - Train decision tree model
- `test_model.py` - Validate adaptive quality model
- `generate_training_data.py` - Generate synthetic training data
- `models/adaptive_quality.model` - Trained decision tree model

### Training Workflow

```bash
# 1. Collect performance data (run PlasmaDX with various settings)
../build/Debug/PlasmaDX-Clean.exe --collect-data

# 2. Train adaptive quality model
python train_adaptive_quality.py

# 3. Test model
python test_model.py
```

### C++ Integration

**Header:** `src/ml/AdaptiveQualitySystem.h` (152 lines)
**Implementation:** `src/ml/AdaptiveQualitySystem.cpp` (415 lines)

**Already integrated** into Application class.

---

## Python Dependencies

### PINN Dependencies (`requirements_pinn.txt`)

```
torch>=2.0.0
onnx>=1.14.0
onnxruntime>=1.16.0
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.10.0
```

Install:
```bash
pip install -r requirements_pinn.txt
```

### System Requirements

- **Python:** 3.8+
- **CUDA:** 11.8+ (for GPU training, optional)
- **RAM:** 8 GB minimum, 16 GB recommended
- **Disk:** ~500 MB for models and training data

---

## Model Information

### PINN Model (`pinn_accretion_disk.onnx`)

- **Architecture:** 5 layers × 128 neurons (Tanh activation)
- **Parameters:** ~50,000 trainable weights
- **Input:** (r, θ, φ, v_r, v_θ, v_φ, t) - 7D phase space + time
- **Output:** (F_r, F_θ, F_φ) - 3D force vector in spherical coordinates
- **Training:** Combined data + physics loss function
- **Constraints:** GR, Keplerian motion, L conservation, E conservation, α-viscosity
- **File Size:** 264 KB (model) + 19 KB (metadata)

### Adaptive Quality Model (`adaptive_quality.model`)

- **Architecture:** Decision tree (binary format)
- **Features:** 12 (particle count, light count, camera distance, shadow rays, etc.)
- **Output:** Predicted frame time (ms)
- **Training:** Supervised learning on performance data
- **File Size:** 16 KB

---

## Performance Metrics

### PINN Speedup (Expected)

| Particle Count | GPU Only | PINN Hybrid | Speedup |
|----------------|----------|-------------|---------|
| 10,000         | 120 FPS  | 120 FPS     | 1.0×    |
| 50,000         | 45 FPS   | 180 FPS     | **4.0×** |
| 100,000        | 18 FPS   | 110 FPS     | **6.1×** |
| 200,000        | 7 FPS    | 60 FPS      | **8.6×** |

### Adaptive Quality Impact

- **Automatic quality adjustment:** Maintains target FPS (60, 120, 144)
- **Hysteresis:** Prevents rapid quality oscillation (2-second delay between changes)
- **Presets:** Ultra, High, Medium, Low, Minimal

---

## Troubleshooting

### PINN Training Issues

**Problem:** "No training data found"
```bash
# Solution: Collect data from GPU
../build/Debug/PlasmaDX-Clean.exe --dump-buffers 120
python collect_physics_data.py --input ../PIX/buffer_dumps
```

**Problem:** "CUDA out of memory"
```bash
# Solution: Reduce batch size in pinn_accretion_disk.py
batch_size = 512  # Default: 1024
```

**Problem:** "Physics loss not decreasing"
```bash
# Solution: Increase physics loss weights
lambda_keplerian = 0.5  # Default: 0.1
lambda_L = 0.5          # Default: 0.1
lambda_E = 0.5          # Default: 0.1
lambda_gr = 0.5         # Default: 0.1
```

### ONNX Export Issues

**Problem:** "ONNX export failed"
```bash
# Solution: Update ONNX and PyTorch
pip install --upgrade onnx onnxruntime torch
```

**Problem:** "Model validation failed"
```bash
# Solution: Run validation script
python validate_onnx_model.py

# Check for specific errors
```

### C++ Integration Issues

**Problem:** "ONNX Runtime not found"
```bash
# Solution: Install ONNX Runtime
# Download from: https://github.com/microsoft/onnxruntime/releases
# Extract to: external/onnxruntime/
# Rebuild project
```

**Problem:** "Failed to load PINN model"
```bash
# Solution: Verify model file exists
ls -lh models/pinn_accretion_disk.onnx

# Check working directory when running executable
cd build/Debug
./PlasmaDX-Clean.exe  # Should find ../../ml/models/pinn_accretion_disk.onnx
```

---

## References

### Documentation
- `PINN_README.md` - PINN Python training guide (500+ lines)
- `PINN_CPP_INTEGRATION_GUIDE.md` - C++ integration guide
- `PINN_CPP_IMPLEMENTATION_SUMMARY.md` - Implementation summary
- `PINN_IMPLEMENTATION_SUMMARY.md` - Overall PINN project summary

### Papers
- [PINN Paper](https://doi.org/10.1016/j.jcp.2018.10.045) - Raissi et al. (2019)
- [Shakura-Sunyaev](https://ui.adsabs.harvard.edu/abs/1973A%26A....24..337S) - α-disk model

### External Tools
- [ONNX Runtime](https://onnxruntime.ai/)
- [PyTorch](https://pytorch.org/)
- [ONNX](https://onnx.ai/)

---

**Last Updated:** 2025-10-24
**Status:** PINN C++ integration complete ✅
**Next:** ParticleSystem integration for hybrid physics mode
