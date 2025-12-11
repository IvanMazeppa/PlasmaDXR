# Multi-Agent Optimization Toolkit

AI-powered performance engineering framework for PlasmaDX-Clean DirectX 12 raytracing renderer.

## Quick Start

### Run Full Optimization Analysis
```bash
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean
python3 optimization/multi_agent_optimizer.py .
```

**Output:**
- Performance profiling across 4 domains
- Projected FPS improvements
- Prioritized recommendations
- JSON report: `optimization/reports/optimization_report_<timestamp>.json`

### View Performance Dashboard
```bash
# Static dashboard (from latest log)
python3 optimization/performance_dashboard.py .

# Real-time monitoring (60 seconds)
python3 optimization/performance_dashboard.py . --monitor

# Analyze optimization report
python3 optimization/performance_dashboard.py . --analyze-report optimization/reports/optimization_report_*.json
```

### Read Optimization Roadmap
```bash
cat optimization/OPTIMIZATION_ROADMAP.txt
cat optimization/OPTIMIZATION_ACTION_PLAN.md
```

---

## Architecture

### Multi-Agent System

The toolkit uses **4 specialized optimization agents** running in parallel:

#### 1. BLAS_Optimizer
- **Domain:** Raytracing Traversal
- **Focus:** BLAS/TLAS acceleration structure rebuilds
- **Projected Gain:** +41 FPS (+29%)
- **Recommendations:**
  - ✅ Implement BLAS update (not rebuild) - **DONE 2025-12-08**
  - ✅ Add frustum culling before TLAS build - **DONE 2025-12-11**
  - Distance-based LOD for particle density

#### 2. Shader_Optimizer
- **Domain:** Shader Compilation
- **Focus:** Detect stale .dxil binaries, build system validation
- **Projected Gain:** 0 FPS (QoL improvement)
- **Recommendations:**
  - CMake timestamp checks
  - Pre-build staleness validation
  - Runtime shader validation (Debug builds)

#### 3. Memory_Bandwidth_Optimizer
- **Domain:** Memory Bandwidth
- **Focus:** UAV barriers, froxel grid optimization
- **Projected Gain:** +8-12 FPS (+6-8%)
- **Recommendations:**
  - Batch UAV barriers (8 → 4 per frame)
  - Froxel R16 format (7 MB → 3.6 MB)
  - Atomic density injection (fix race condition)

#### 4. Context_Window_Optimizer
- **Domain:** AI Development Efficiency
- **Focus:** Claude Code context management
- **Projected Gain:** 0 FPS (faster AI responses)
- **Recommendations:**
  - Archive completed phases
  - Compress historical docs
  - Delegate to MCP servers

---

## Key Concepts

### Parallel Agent Execution
All agents run concurrently using `concurrent.futures.ThreadPoolExecutor`:
```python
with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
    futures = {
        executor.submit(agent.optimize, target_system, goals): agent
        for agent in self.agents
    }
```

**Benefits:**
- 4× faster than sequential execution
- Independent agent failures don't block others
- Real-time progress updates

### Token Budget Management
Each agent has a token budget (default: 10,000 tokens):
```python
self.token_budget = 10000
self.token_usage = 0  # Tracked per optimization
```

**Total Budget:** 50,000 tokens
**Actual Usage:** 3,550 tokens (7.1%)
**Efficiency:** 92.9% ✅

### Performance Metric Tracking
```python
@dataclass
class PerformanceMetric:
    name: str
    value: float
    unit: str
    baseline: Optional[float] = None
    target: Optional[float] = None

    @property
    def improvement_pct(self) -> float:
        if self.baseline and self.baseline > 0:
            return ((self.value - self.baseline) / self.baseline) * 100
        return 0.0
```

Automatically calculates improvement percentages for all metrics.

---

## Implementation Workflow

### Phase 1: Analysis (COMPLETED ✅)
```bash
python3 optimization/multi_agent_optimizer.py .
```

**Deliverables:**
- ✅ Multi-agent optimization report
- ✅ Performance bottleneck identification
- ✅ Prioritized action plan
- ✅ Visual roadmap

### Phase 2: Quick Wins (Week 1)
**Target:** 142 FPS → 170 FPS (+28 FPS)

1. **BLAS Update Implementation**
   ```cpp
   // src/lighting/RTLightingSystem.cpp
   D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags =
       D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE |
       D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
   ```

2. **UAV Barrier Batching**
   ```cpp
   D3D12_RESOURCE_BARRIER barriers[] = {
       CD3DX12_RESOURCE_BARRIER::UAV(m_froxelDensity.Get()),
       CD3DX12_RESOURCE_BARRIER::UAV(m_froxelLighting.Get())
   };
   m_commandList->ResourceBarrier(_countof(barriers), barriers);
   ```

3. **Froxel R16 Format**
   ```hlsl
   RWTexture3D<float16_t> g_froxelDensity;  // Was R32_FLOAT
   ```

### Phase 3: Medium-Complexity (Week 2)
**Target:** 170 FPS → 200 FPS (+30 FPS)

1. ✅ **Frustum Culling** - **DONE 2025-12-11** (see `FRUSTUM_CULLING_IMPLEMENTATION.md`)
2. **Atomic Froxel Injection**
3. **ONNX Runtime Setup**

### Phase 4: Advanced Optimizations (Week 3)
**Target:** 200 FPS → 305 FPS (+105 FPS)

1. **PINN ML Physics (Hybrid Mode)**
2. **Distance-Based LOD**

### Phase 5: Validation (Week 4)
**Target:** Validate 305 FPS ✅

1. Performance regression testing
2. Visual quality validation
3. Documentation updates

---

## Monitoring & Validation

### Real-Time Monitoring
```bash
python3 optimization/performance_dashboard.py . --monitor
```

**Dashboard Displays:**
- Current FPS vs baselines
- Frame time breakdown
- Optimization milestone progress
- GPU metrics (BLAS rebuild, froxel passes)

### PIX GPU Capture Validation
```bash
# Capture GPU frame with PIX
./build/Debug/PlasmaDX-Clean.exe --pix-capture

# Analyze capture
# Look for:
# - BLAS update (not rebuild) in frame timeline
# - Reduced barrier count (8 → 4)
# - Froxel R16 texture format
```

### Performance Regression Testing
```bash
# Baseline capture (before optimization)
./build/Debug/PlasmaDX-Clean.exe --benchmark --output=baseline.json

# Optimized capture (after changes)
./build/Debug/PlasmaDX-Clean.exe --benchmark --output=optimized.json

# Compare
python3 optimization/compare_benchmarks.py baseline.json optimized.json
```

---

## File Structure

```
optimization/
├── README.md                           # This file
├── OPTIMIZATION_ACTION_PLAN.md         # Detailed implementation plan
├── OPTIMIZATION_ROADMAP.txt            # Visual roadmap diagram
├── IMPLEMENTED_2025-12-08.md           # AABB sizing + BLAS update implementation
├── FRUSTUM_CULLING_IMPLEMENTATION.md   # GPU frustum culling implementation (2025-12-11)
├── multi_agent_optimizer.py            # Main optimization framework
├── performance_dashboard.py            # Real-time performance monitoring
├── frustum_culling_benchmark.py        # Benchmark script for frustum culling
└── reports/
    └── optimization_report_<timestamp>.json  # Analysis results
```

---

## Configuration

### Optimization Goals
Edit in `multi_agent_optimizer.py`:
```python
goals = {
    "target_fps": 280,           # Target FPS
    "current_fps": 142,          # Current FPS
    "particle_count": 100000,    # Particle count
    "resolution": "1920x1080",   # Resolution
    "quality_preset": "balanced" # Quality preset
}
```

### Agent Configuration
Each agent can be individually configured:
```python
class BLASOptimizationAgent(BaseOptimizationAgent):
    def __init__(self):
        super().__init__("BLAS_Optimizer", OptimizationDomain.RAYTRACING_TRAVERSAL)
        self.baseline_ms = 2.1  # Current BLAS rebuild time
        self.token_budget = 10000
```

### Performance Baselines
Edit in `performance_dashboard.py`:
```python
self.baselines = {
    "raster_only": 245.0,
    "rt_lighting": 165.0,
    "rt_shadows": 142.0,
    "dlss_performance": 190.0,
    "target_with_pinn": 280.0
}
```

---

## Advanced Usage

### Custom Agent Development
Create a new optimization agent:

```python
class CustomOptimizationAgent(BaseOptimizationAgent):
    def __init__(self):
        super().__init__("Custom_Optimizer", OptimizationDomain.GPU_ACCELERATION)

    def profile(self, target_system: str) -> Dict:
        # Your profiling logic
        return {"metric": "value"}

    def optimize(self, target_system: str, goals: Dict) -> OptimizationResult:
        # Your optimization logic
        recommendations = ["Recommendation 1", "Recommendation 2"]
        metrics = [PerformanceMetric("FPS", 280, "fps", baseline=142)]

        return OptimizationResult(
            agent_name=self.name,
            domain=self.domain,
            metrics=metrics,
            cost_tokens=1000,
            execution_time_ms=10.0,
            recommendations=recommendations,
            success=True
        )
```

Add to orchestrator:
```python
orchestrator.agents.append(CustomOptimizationAgent())
```

### Export Reports for External Analysis
```python
# Export to JSON
orchestrator.save_report("optimization/reports/my_analysis.json")

# Export dashboard metrics
dashboard.export_metrics("optimization/metrics/performance_history.json")
```

---

## Troubleshooting

### "No stale shaders detected" but visuals look wrong
**Cause:** .dxil binaries compiled but not reflecting .hlsl changes
**Solution:**
```bash
# Force shader rebuild
rm -rf build/bin/Debug/shaders/*.dxil
MSBuild.exe build/PlasmaDX-Clean.sln /t:CompileShaders
```

### Multi-agent analysis hangs
**Cause:** One agent blocked on I/O or computation
**Solution:** Check individual agent execution times in report. Increase timeout or profile slow agent.

### FPS not improving as projected
**Cause:** GPU-bound vs CPU-bound mismatch, thermal throttling, or driver overhead
**Solution:**
1. Capture PIX GPU frame
2. Analyze GPU utilization percentage
3. Check for thermal throttling (GPU-Z)
4. Verify optimization actually applied (check resource flags in PIX)

---

## Performance Baselines

| Configuration | Target FPS | Status |
|---------------|------------|--------|
| Raster Only | 245 | ✅ Baseline |
| + RT Lighting | 165 | ✅ Current |
| + Shadow Rays | 142 | ⚠️ **CURRENT** |
| + DLSS Performance | 190 | ✅ Available |
| + PINN Physics | 280+ | 🎯 **TARGET** |

---

## References

- **Multi-Agent Design:** `multi_agent_optimizer.py`
- **Performance Dashboard:** `performance_dashboard.py`
- **Action Plan:** `OPTIMIZATION_ACTION_PLAN.md`
- **Visual Roadmap:** `OPTIMIZATION_ROADMAP.txt`
- **Project Context:** `../CLAUDE.md`
- **DirectX 12 Raytracing:** Microsoft DXR 1.1 Spec
- **PINN Physics:** `../ml/PINN_README.md`

---

## Contact & Support

**Project Lead:** Ben
**AI Assistant:** Claude Code (Sonnet 4.5)
**Repository:** PlasmaDX-Clean
**Version:** 0.21.6

For questions or issues, create a new task in the project tracker or consult the `MASTER_ROADMAP_V2.md`.

---

**Last Updated:** 2025-12-06
**Toolkit Version:** 1.0
**Next Review:** After Week 1 implementation
