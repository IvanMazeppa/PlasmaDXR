# PINN Model Training Guide - Complete Reference

**Last Updated:** 2025-11-27
**Model Version:** v3 (Total Forces, 10D Cartesian)
**Status:** Production-ready training pipeline

---

## 📁 Project Structure - Where Everything Lives

```
PlasmaDX-Clean/
│
├── ml/                                    # ML/PINN root directory
│   ├── venv/                             # Python virtual environment
│   │   ├── bin/python3                   # Python interpreter
│   │   └── lib/python3.12/site-packages/ # PyTorch, ONNX, etc.
│   │
│   ├── models/                           # Trained ONNX models (OUTPUT)
│   │   ├── pinn_v3_total_forces.onnx           # Model graph (3KB)
│   │   ├── pinn_v3_total_forces.onnx.data      # Weights (265KB)
│   │   ├── pinn_v3_training_loss.png           # Training curve
│   │   ├── pinn_v2_turbulent.onnx              # Legacy v2 model
│   │   └── pinn_accretion_disk.onnx            # Legacy v1 model
│   │
│   ├── training_data/                    # Generated training datasets
│   │   ├── pinn_v3_total_forces.npz            # v3 data (100K samples)
│   │   ├── pinn_v2_turbulent.npz               # v2 data (legacy)
│   │   └── accretion_disk_physics_data.npz     # v1 data (legacy)
│   │
│   ├── training_log_v3.txt               # Training progress log
│   │
│   ├── pinn_v3_total_forces.py           # 🔵 v3 TRAINING SCRIPT (CURRENT)
│   ├── pinn_v2_turbulent.py              # v2 training script (legacy)
│   ├── pinn_accretion_disk.py            # v1 training script (legacy)
│   │
│   ├── test_pinn.py                      # Model inference testing
│   ├── collect_physics_data.py           # GPU buffer dump collector
│   │
│   └── requirements_pinn.txt             # Python dependencies
│
├── build/bin/Debug/ml/models/            # Deployed models (RUNTIME)
│   ├── pinn_v3_total_forces.onnx         # Copy of trained model
│   └── pinn_v3_total_forces.onnx.data    # Copy of weights
│
├── src/ml/                               # C++ ONNX Runtime integration
│   ├── PINNPhysicsSystem.h               # PINN inference class header
│   └── PINNPhysicsSystem.cpp             # PINN inference implementation
│
└── src/particles/                        # Particle system integration
    ├── ParticleSystem.h                  # Physics loop
    └── ParticleSystem.cpp                # Calls PINN inference
```

---

## 🧬 Model Architecture - PINN v3 Specification

### Network Design:

```python
class AccretionDiskPINN_v3(nn.Module):
    """
    Physics-Informed Neural Network for Accretion Disk Dynamics

    Architecture: 10 → 128 → 128 → 128 → 128 → 128 → 3
    Activation: Tanh (smooth, differentiable for physics)
    Parameters: 67,843 total
    """

    def __init__(self, hidden_dim=128, num_layers=5):
        super().__init__()

        # Input layer: 10D → 128
        layers = [nn.Linear(10, hidden_dim), nn.Tanh()]

        # Hidden layers: 4× (128 → 128)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        # Output layer: 128 → 3
        layers.append(nn.Linear(hidden_dim, 3))

        self.network = nn.Sequential(*layers)
```

### Input Features (10D Cartesian):

| Index | Feature | Range | Units | Description |
|-------|---------|-------|-------|-------------|
| 0 | `x` | -300 to +300 | normalized | Cartesian position X |
| 1 | `y` | -300 to +300 | normalized | Cartesian position Y (vertical) |
| 2 | `z` | -300 to +300 | normalized | Cartesian position Z |
| 3 | `vx` | -5 to +5 | normalized | Velocity X component |
| 4 | `vy` | -5 to +5 | normalized | Velocity Y component |
| 5 | `vz` | -5 to +5 | normalized | Velocity Z component |
| 6 | `t` | 0 to 100 | normalized | Simulation time |
| 7 | `M_bh` | 0.8 to 1.2 | multiplier | Black hole mass (1.0 = default) |
| 8 | `α` | 0.05 to 0.15 | unitless | Shakura-Sunyaev viscosity |
| 9 | `H/R` | 0.05 to 0.15 | ratio | Disk thickness ratio |

### Output Forces (3D Cartesian):

| Index | Feature | Range | Units | Description |
|-------|---------|-------|-------|-------------|
| 0 | `Fx` | -2 to +2 | normalized | Total force X component |
| 1 | `Fy` | -0.5 to +0.5 | normalized | Total force Y component |
| 2 | `Fz` | -2 to +2 | normalized | Total force Z component |

**Force Decomposition:**
```
F_total = F_gravity + F_viscosity + F_MRI

F_gravity:   -GM * M_bh * r_hat / r²  (radially inward)
F_viscosity: -ν_eff * v_phi * phi_hat (azimuthal damping)
F_MRI:       random(0, α*0.001)       (turbulent fluctuations)
```

---

## 🔧 Training Script - `pinn_v3_total_forces.py`

### Key Constants (CRITICAL - Must Match Physics):

```python
# Line 28-30 in pinn_v3_total_forces.py

GM = 100.0      # ← Gravitational parameter (CRITICAL!)
                # GM=1.0 → forces too weak (0.0001 mag)
                # GM=100.0 → visible orbital forces (0.01 mag)

R_ISCO = 6.0    # Innermost stable circular orbit
                # (3× Schwarzschild radius for non-rotating black hole)
```

**⚠️ WARNING:** Changing `GM` requires full retraining (data generation + model training).

### Training Data Generation:

**Function:** `generate_training_data_v3(num_samples=100000)`

**Physics Implemented:**

1. **Keplerian Orbits:**
   ```python
   # Orbital radius: r ∈ [10, 300] normalized units
   r = np.random.uniform(10.0, 300.0)

   # Circular velocity: v = sqrt(GM/r)
   v_kepler = np.sqrt(GM / r)

   # Azimuthal velocity (tangent to orbit)
   vx = -v_kepler * np.sin(theta)
   vz = +v_kepler * np.cos(theta)
   ```

2. **Disk Thickness (Gaussian):**
   ```python
   # Vertical position: y ~ N(0, 0.1*r)
   height = np.random.normal(0, 0.1 * r)
   y = height
   ```

3. **Gravitational Force:**
   ```python
   # Newton's law of gravitation
   r_vec = np.array([x, y, z])
   r_mag = np.sqrt(x**2 + y**2 + z**2)
   r_hat = r_vec / r_mag

   F_grav = -GM * M_bh * r_hat / (r_mag**2)
   ```

4. **Shakura-Sunyaev Viscosity:**
   ```python
   # Effective viscosity: ν_eff = α * H * c_s
   # c_s ≈ H * Ω (sound speed)
   nu_eff = alpha * H_R * 0.01

   # Azimuthal velocity component
   r_cyl = np.sqrt(x**2 + z**2)
   phi_hat = np.array([-z, 0, x]) / r_cyl
   v_phi = vx * phi_hat[0] + vz * phi_hat[2]

   # Viscous drag (opposes rotation)
   F_visc_mag = -nu_eff * v_phi / r_cyl
   F_visc = F_visc_mag * phi_hat
   ```

5. **Magneto-Rotational Instability (MRI):**
   ```python
   # Random turbulent forces
   F_mri = np.random.normal(0, alpha * 0.001, size=3)
   F_mri[1] *= 0.2  # Less vertical turbulence
   ```

**Output Statistics (GM=100):**
```
Force ranges:
  Fx: [-1.16, +1.13]  (strong radial forces)
  Fy: [-0.27, +0.25]  (weak vertical forces)
  Fz: [-1.19, +1.19]  (strong radial forces)

Average force magnitude: 0.0329
(At r=100: F ≈ -GM/r² = -100/10000 = -0.01)
```

### Training Hyperparameters:

```python
# Optimization
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# Training
epochs = 200                # Total training epochs
batch_size = 1024           # Samples per batch
device = 'cuda'             # GPU acceleration (RTX 4060 Ti)

# Loss function (multi-component)
loss = (
    MSE(F_pred, F_true) +                      # Force matching
    0.1 * MSE(|F_pred| - |F_grav|) +          # Gravity dominance
    0.1 * MSE(r × F_pred - r × F_true)        # Angular momentum
)
```

**Training Time:** ~9 minutes (200 epochs on CUDA)

---

## 📚 Complete Training Workflow

### Step 1: Environment Setup (One-Time)

**Create virtual environment:**
```bash
cd ml
python3 -m venv venv
source venv/bin/activate  # Linux/WSL
# OR: venv\Scripts\activate  # Windows

pip install -r requirements_pinn.txt
```

**Dependencies (`requirements_pinn.txt`):**
```
torch>=2.0.0
onnx>=1.14.0
onnxruntime-gpu>=1.17.0  # For GPU inference in C++
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.10.0
```

### Step 2: Generate Training Data

**Command:**
```bash
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean
ml/venv/bin/python3 ./pinn_v3_total_forces.py --generate-data
```

**Output:**
```
[v3] Generating 100000 training samples with TOTAL forces...
  Generated 10000/100000 samples...
  Generated 20000/100000 samples...
  ...
  Generated 100000/100000 samples...
[v3] Training data saved to ml/training_data/pinn_v3_total_forces.npz
  States shape: (100000, 10)
  Forces shape: (100000, 3)
  Force ranges: Fx=[-1.1573, 1.1268]
                Fy=[-0.2742, 0.2549]
                Fz=[-1.1864, 1.1902]
  Average force magnitude: 0.0329
```

**Generated File:** `ml/training_data/pinn_v3_total_forces.npz` (~76 MB)

**Data Format (NumPy):**
```python
data = np.load('ml/training_data/pinn_v3_total_forces.npz')
states = data['states']    # [100000, 10] - input features
forces = data['forces']    # [100000, 3]  - target forces
```

### Step 3: Train Model

**Command:**
```bash
ml/venv/bin/python3 ./pinn_v3_total_forces.py --epochs 200
```

**Training Progress (stdout):**
```
================================================================================
PINN v3 Training - Total Force Output
================================================================================

Loading training data from ml/training_data/pinn_v3_total_forces.npz...
  Loaded 100000 training samples
  State shape: torch.Size([100000, 10])
  Force shape: torch.Size([100000, 3])

Using device: cuda
Model parameters: 67,843

Training for 200 epochs...
Epoch 10/200 - Loss: 0.001603 - LR: 0.001000
Epoch 20/200 - Loss: 0.000754 - LR: 0.001000
...
Epoch 200/200 - Loss: 0.000073 - LR: 0.000125

Exporting to ONNX: ml/models/pinn_v3_total_forces.onnx
[torch.onnx] Translate the graph into ONNX... ✅
✅ Model saved to ml/models/pinn_v3_total_forces.onnx
   Final loss: 0.000073
   Training curve saved to ml/models/pinn_v3_training_loss.png
```

**Training Log:** `ml/training_log_v3.txt` (append-only, tracks all training runs)

**Generated Files:**
- `ml/models/pinn_v3_total_forces.onnx` (3 KB) - Model graph
- `ml/models/pinn_v3_total_forces.onnx.data` (265 KB) - Network weights
- `ml/models/pinn_v3_training_loss.png` - Training curve visualization

**Loss Curve Interpretation:**
- **Good:** Monotonic decrease, converges to ~0.0001 or lower
- **Bad:** Oscillating, increasing, or plateau at >0.001
- **Excellent:** Final loss <0.0001 (current: 0.000073)

### Step 4: Deploy Model to Application

**Command:**
```bash
cp ml/models/pinn_v3_total_forces.onnx* build/bin/Debug/ml/models/
```

**Verify Deployment:**
```bash
ls -lh build/bin/Debug/ml/models/pinn_v3_total_forces.*

# Expected output:
# -rwxrwxrwx ... 3.0K ... pinn_v3_total_forces.onnx
# -rwxrwxrwx ... 265K ... pinn_v3_total_forces.onnx.data
```

**Model Loading (C++):**
```cpp
// In ParticleSystem::InitializePINN() - src/particles/ParticleSystem.cpp
// Tries v3 first, falls back to v2, then v1
bool pinnLoaded = false;
if (m_pinnPhysics->Initialize("ml/models/pinn_v3_total_forces.onnx")) {
    LOG_INFO("[PINN] Loaded v3 TOTAL FORCES model");
    pinnLoaded = true;
}
```

### Step 5: Test Model (Optional)

**Inference Test Script:**
```bash
ml/venv/bin/python3 ./test_pinn.py --model ml/models/pinn_v3_total_forces.onnx
```

**Outputs:**
```
Testing PINN model: ml/models/pinn_v3_total_forces.onnx
Input shape: [1, 10]
Output shape: [1, 3]

Test particle at r=100:
  Position: (100.0, 0.0, 0.0)
  Velocity: (0.0, 0.0, 1.0)  # Keplerian v=sqrt(GM/r)=1.0

  Predicted force: (-0.0098, 0.0002, -0.0001)
  Force magnitude: 0.0099
  Radial component: -0.0098 (attractive ✓)

Expected: F_radial ≈ -GM/r² = -100/10000 = -0.01
Error: 2.0% (excellent!)
```

---

## 🔄 Retraining Workflow - When and How

### When to Retrain:

1. **Physics Parameter Changes:**
   - Changed `GM` (gravitational parameter)
   - Changed `R_ISCO` (innermost stable orbit)
   - Modified force calculation (viscosity, MRI)

2. **Training Data Issues:**
   - Forces too weak/strong (adjust `GM`)
   - Wrong orbital distribution (adjust radius range)
   - Insufficient samples (increase `num_samples`)

3. **Model Architecture Changes:**
   - Changed `hidden_dim` (layer width)
   - Changed `num_layers` (network depth)
   - Changed activation function

4. **Performance Problems:**
   - Model predicts NaN (numeric instability)
   - High inference time (network too large)
   - Poor generalization (overfitting)

### Full Retraining Procedure:

```bash
# 1. Modify training script
nano pinn_v3_total_forces.py
# Change GM, network architecture, etc.

# 2. Delete old data (forces regeneration)
rm ml/training_data/pinn_v3_total_forces.npz

# 3. Generate new training data
ml/venv/bin/python3 ./pinn_v3_total_forces.py --generate-data

# 4. Train model (200 epochs, ~9 min)
ml/venv/bin/python3 ./pinn_v3_total_forces.py --epochs 200

# 5. Backup old model (optional)
cp ml/models/pinn_v3_total_forces.onnx ml/models/pinn_v3_total_forces.onnx.backup

# 6. Deploy new model
cp ml/models/pinn_v3_total_forces.onnx* build/bin/Debug/ml/models/

# 7. Test in application
./build/bin/Debug/PlasmaDX-Clean.exe
# Press 'P' to enable PINN
# Check force diagnostics in log
```

---

## 🐛 Troubleshooting Common Issues

### Issue 1: "Training data not found"

**Error:**
```
FileNotFoundError: ml/training_data/pinn_v3_total_forces.npz
```

**Solution:**
```bash
ml/venv/bin/python3 ./pinn_v3_total_forces.py --generate-data
```

### Issue 2: "CUDA out of memory"

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**
1. Reduce batch size: `batch_size = 512` (line ~355)
2. Use CPU: `device = 'cpu'` (line ~345) - slower but works
3. Close other GPU applications (browsers, games)

### Issue 3: Training loss not decreasing

**Symptoms:**
```
Epoch 10/200 - Loss: 0.523000
Epoch 20/200 - Loss: 0.521000  # Barely changed!
```

**Causes & Solutions:**
1. **Learning rate too low:** Increase to `lr=0.005`
2. **Bad initialization:** Retrain (random seed different)
3. **Data quality:** Check force ranges in training data
4. **Network too shallow:** Increase `num_layers=7`

### Issue 4: Model outputs NaN

**Symptoms:**
```
[PINN] Frame 60: Avg force: (nan, nan, nan) mag=nan
```

**Causes:**
1. **Numeric overflow:** Forces too large (reduce `GM`)
2. **Division by zero:** Check r_mag < epsilon handling
3. **Gradient explosion:** Reduce learning rate

**Solution:**
```python
# Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Issue 5: Forces still too weak after retraining

**Symptoms:**
```
Expected: mag=0.01-0.03
Actual: mag=0.0005-0.001
```

**Debug Steps:**
1. **Verify model file timestamp:**
   ```bash
   ls -l build/bin/Debug/ml/models/pinn_v3_total_forces.onnx
   # Should match recent training time
   ```

2. **Check training data stats:**
   ```bash
   grep "Average force magnitude" ml/training_data/pinn_v3_total_forces.npz
   # Should show ~0.03 (not 0.0003)
   ```

3. **Test raw ONNX output:** Use `test_pinn.py` to verify model predictions

4. **Check C++ integration:** See `PINN_COMPREHENSIVE_ANALYSIS.md` diagnostic plan

---

## 📐 Model Versioning - v1, v2, v3 Comparison

| Feature | v1 (Legacy) | v2 (Turbulent) | v3 (Total Forces) |
|---------|-------------|----------------|-------------------|
| **Input Dim** | 7D spherical | 7D spherical + 3D params | 10D Cartesian |
| **Coordinates** | (r, θ, φ, v_r, v_θ, v_φ, t) | Same as v1 | (x, y, z, vx, vy, vz, t, M_bh, α, H/R) |
| **Output** | Net forces (F_net ≈ 0) | Turbulent forces | **TOTAL forces** ✓ |
| **Physics** | Simplified Keplerian | MRI + Kolmogorov | **Gravity + viscosity + MRI** ✓ |
| **Parameters** | 50K | 67K | **67,843** ✓ |
| **GM value** | 1.0 (weak) | 1.0 (weak) | **100.0 (strong)** ✓ |
| **Status** | Deprecated | Deprecated | **CURRENT** ✓ |

**Migration:** All new work should use v3. v1/v2 kept for reference only.

---

## 🎓 Advanced Topics

### Custom Physics Modifications

**Example: Add magnetic field forces**

1. **Modify data generation** (`pinn_v3_total_forces.py` line ~270):
   ```python
   # Lorentz force: F = q(v × B)
   B_field = np.array([0, 0.1, 0])  # Vertical magnetic field
   F_magnetic = np.cross(v_vec, B_field) * 0.01

   F_total = F_grav + F_visc + F_mri + F_magnetic
   ```

2. **Regenerate data + retrain:**
   ```bash
   rm ml/training_data/pinn_v3_total_forces.npz
   ml/venv/bin/python3 ./pinn_v3_total_forces.py --generate-data
   ml/venv/bin/python3 ./pinn_v3_total_forces.py --epochs 200
   ```

### Hyperparameter Tuning

**Learning Rate Sweep:**
```bash
for lr in 0.0001 0.0005 0.001 0.005 0.01; do
  python pinn_v3_total_forces.py --epochs 50 --lr $lr
  mv ml/models/pinn_v3_total_forces.onnx ml/models/pinn_v3_lr${lr}.onnx
done
```

**Best Practices:**
- Start with `lr=0.001` (default)
- If loss oscillates: reduce to `0.0005`
- If loss plateaus: increase to `0.005`
- Monitor validation loss (add 20% validation split)

### Model Compression

**Current model:** 265 KB (67,843 parameters)

**Smaller model (for low-end GPUs):**
```python
# Reduce hidden dimension
model = AccretionDiskPINN_v3(hidden_dim=64, num_layers=4)
# Results in: 17,731 parameters (~68 KB)
```

**Larger model (for accuracy):**
```python
model = AccretionDiskPINN_v3(hidden_dim=256, num_layers=6)
# Results in: 267,011 parameters (~1 MB)
```

---

## 📖 Reference Documentation

### Python API

**Training Script Arguments:**
```bash
python pinn_v3_total_forces.py [OPTIONS]

Options:
  --generate-data       Generate training data (100K samples)
  --epochs EPOCHS       Number of training epochs (default: 200)
  --batch-size SIZE     Batch size (default: 1024)
  --lr RATE             Learning rate (default: 0.001)
  --device DEVICE       Device (cuda/cpu, default: cuda)
  --help                Show help message
```

### File Formats

**ONNX Model (.onnx + .onnx.data):**
- Binary format for neural network inference
- `.onnx`: Model graph (operators, shapes)
- `.onnx.data`: Network weights (float32)
- Loaded by ONNX Runtime in C++

**NPZ Training Data (.npz):**
```python
import numpy as np
data = np.load('ml/training_data/pinn_v3_total_forces.npz')

# Arrays:
states = data['states']    # [100000, 10] float32
forces = data['forces']    # [100000, 3] float32
```

### C++ Integration Points

**Model Loading:**
- `src/ml/PINNPhysicsSystem.cpp:59-141` - ONNX Runtime initialization
- `src/particles/ParticleSystem.cpp:107-125` - Model file loading

**Inference:**
- `src/ml/PINNPhysicsSystem.cpp:214-260` - Prepare input tensor (10D)
- `src/ml/PINNPhysicsSystem.cpp:292-324` - ONNX Runtime inference
- `src/ml/PINNPhysicsSystem.cpp:330-360` - Extract force predictions

**Integration:**
- `src/particles/ParticleSystem.cpp:549-586` - PINN force prediction batch
- `src/particles/ParticleSystem.cpp:665-720` - Velocity Verlet integration

---

## ✅ Quick Reference Checklist

### Standard Retraining (GM change):
- [ ] Edit `pinn_v3_total_forces.py` line 29 (change `GM`)
- [ ] Delete old training data: `rm ml/training_data/pinn_v3_total_forces.npz`
- [ ] Generate new data: `python pinn_v3_total_forces.py --generate-data`
- [ ] Train model: `python pinn_v3_total_forces.py --epochs 200` (~9 min)
- [ ] Copy to build: `cp ml/models/pinn_v3_total_forces.onnx* build/bin/Debug/ml/models/`
- [ ] Test in app, verify force magnitude in logs

### Quick Test (no retraining):
- [ ] Run app: `./build/bin/Debug/PlasmaDX-Clean.exe`
- [ ] Enable PINN: Press 'P' key
- [ ] Check log: `[PINN] Loaded v3 TOTAL FORCES model`
- [ ] Wait 60 frames, check: `Avg force mag=0.010-0.030` (should be strong!)

### Troubleshooting (forces weak):
- [ ] Verify model timestamp: `ls -l build/bin/Debug/ml/models/pinn_v3*.onnx`
- [ ] Check training data: `grep "Average force" ml/training_log_v3.txt`
- [ ] Test raw ONNX: `python test_pinn.py --model ml/models/pinn_v3_total_forces.onnx`
- [ ] Add C++ debug logging (see `PINN_COMPREHENSIVE_ANALYSIS.md`)

---

**End of Training Guide**
**For physics analysis and debugging:** See `PINN_COMPREHENSIVE_ANALYSIS.md`
**For original diagnostics:** See `PINN_SESSION_SUMMARY.md`
