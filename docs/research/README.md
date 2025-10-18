# PlasmaDX Research Library

Comprehensive DXR / ML / Optimization research documentation created by `rt-ml-technique-researcher-v2` agent.

## Overview

**Total Documents:** 53 markdown files
**Total Size:** ~976KB
**Primary Topics:** DirectX 12, DXR 1.1, ML lighting, ReSTIR, optimization techniques
**Created By:** `rt-ml-technique-researcher-v2.md` agent (2025-09-20 to 2025-10-11)
**Location:** `AdvancedTechniqueWebSearches/`

## Document Categories

### DXR Core Techniques
**Location:** `AdvancedTechniqueWebSearches/`

| Document | Topics Covered |
|----------|---------------|
| `DXR_Acceleration_Structures_Guide.md` | BLAS/TLAS building, procedural primitives |
| `DXR_Inline_Ray_Tracing_RayQuery_Guide.md` | RayQuery API, inline RT |
| `DXR_Pipeline_and_SBT_Guide.md` | Shader Binding Tables, hit groups |
| `DXR_Ray_Traced_Reflections_Guide.md` | Reflection techniques |
| `DXR_Ray_Traced_Shadows_Guide.md` | Shadow ray optimization |
| `dxr_12_new_features_2025.md` | Latest DXR features |

### ReSTIR & Many-Light Algorithms
**Location:** `AdvancedTechniqueWebSearches/`

| Document | Topics Covered |
|----------|---------------|
| `DXR_ReSTIR_DI_GI_Guide.md` | ReSTIR Direct/Global Illumination |
| `efficiency_optimizations/ReSTIR_Particle_Integration.md` | ReSTIR for particle systems |

**Note:** Custom ReSTIR implementation deprecated, moving to NVIDIA RTXDI (Phase 4).

### ML & AI Lighting
**Location:** `AdvancedTechniqueWebSearches/ai_lighting/`

| Document | Topics Covered |
|----------|---------------|
| `Secondary_Ray_Inter_Particle_Bouncing.md` | Neural inter-particle lighting |
| `learned_illumination/Plasma_Emission_Models.md` | ML plasma emission prediction |

### Denoising
**Location:** `AdvancedTechniqueWebSearches/`

| Document | Topics Covered |
|----------|---------------|
| `DXR_Denoising_NRD_Integration_Guide.md` | NVIDIA Real-Time Denoiser integration |

### Volumetric Rendering
**Location:** `AdvancedTechniqueWebSearches/dxr_volumetric_integration/`

| Document | Topics Covered |
|----------|---------------|
| `inline_ray_tracing_volumetrics_2024.md` | RayQuery volumetric techniques |

### Performance Optimization
**Location:** `AdvancedTechniqueWebSearches/efficiency_optimizations/`

| Document | Topics Covered |
|----------|---------------|
| `BLAS_PERFORMANCE_GUIDE.md` | BLAS update vs rebuild optimization |
| `ADA_LOVELACE_DXR12_FEATURES.md` | RTX 40-series specific features |
| `gpu_scheduling/GPU_Driven_LOD.md` | GPU-driven LOD for particles |
| `ray_coherence_blas_tlas_2024.md` | Ray coherence optimization |

### Hardware-Specific
**Location:** `AdvancedTechniqueWebSearches/`

| Document | Topics Covered |
|----------|---------------|
| `nvidia_rtx_4060ti_driver_regression_analysis.md` | RTX 4060 Ti driver issues |

### Troubleshooting
**Location:** `AdvancedTechniqueWebSearches/`

| Document | Topics Covered |
|----------|---------------|
| `dxr_shadow_map_troubleshooting_guide.md` | Shadow map debugging |

### Executive Summaries
**Location:** `AdvancedTechniqueWebSearches/`

| Document | Topics Covered |
|----------|---------------|
| `EXECUTIVE_SUMMARY_PARTICLE_RT.md` | High-level RT particle rendering overview |
| `IMPLEMENTATION_QUICKSTART.md` | Quick implementation guide |

## Usage by Agents

### v3 Production Agents

**buffer-validator-v3** - References format specs from:
- `DXR_Acceleration_Structures_Guide.md` (AABB formats)

**pix-debugger-v3** - Consults for root cause analysis:
- `dxr_shadow_map_troubleshooting_guide.md`
- `DXR_Ray_Traced_Shadows_Guide.md`
- `nvidia_rtx_4060ti_driver_regression_analysis.md`

**performance-analyzer-v3** - Uses optimization research:
- `BLAS_PERFORMANCE_GUIDE.md`
- `efficiency_optimizations/ray_coherence_blas_tlas_2024.md`
- `ADA_LOVELACE_DXR12_FEATURES.md`

### v2 Specialized Agents

**rt-ml-technique-researcher-v2** - Created these documents, continues to add new research

**dxr-systems-engineer-v2** - Implementation references:
- `DXR_Acceleration_Structures_Guide.md`
- `DXR_Pipeline_and_SBT_Guide.md`
- `DXR_Inline_Ray_Tracing_RayQuery_Guide.md`

**physics-performance-agent-v2** - Optimization techniques:
- `gpu_scheduling/GPU_Driven_LOD.md`
- `efficiency_optimizations/ReSTIR_Particle_Integration.md`

## Search Tips

### Finding Specific Topics

```bash
# Search all research docs
grep -r "RTXDI" AdvancedTechniqueWebSearches/

# Find denoising references
grep -r "NRD" AdvancedTechniqueWebSearches/

# Search for optimization techniques
grep -r "optimization" AdvancedTechniqueWebSearches/efficiency_optimizations/
```

### Common Queries

- **"How to implement BLAS updates?"** → `BLAS_PERFORMANCE_GUIDE.md`
- **"ReSTIR spatial reuse?"** → `DXR_ReSTIR_DI_GI_Guide.md`
- **"Volumetric ray marching?"** → `dxr_volumetric_integration/inline_ray_tracing_volumetrics_2024.md`
- **"RTX 4060 Ti quirks?"** → `nvidia_rtx_4060ti_driver_regression_analysis.md`
- **"Neural denoising?"** → `DXR_Denoising_NRD_Integration_Guide.md`

## Future MCP Integration

### Planned Features (Phase 2)

1. **Watched Folder Automation**
   - Automatically index new research docs created by agents
   - Add to DXR MCP server search database
   - Separate search tools for "official DX12 docs" vs "research docs"

2. **Custom Search Tools**
   ```
   mcp__dx12-research__search_plasmadx_research
   mcp__dx12-research__get_research_document
   ```

3. **Automatic Categorization**
   - AI-powered topic tagging
   - Cross-reference linking
   - Relevance scoring for agent queries

### Current Workflow (Manual)

1. `rt-ml-technique-researcher-v2` creates new document
2. Manually move to `docs/research/AdvancedTechniqueWebSearches/`
3. Agents use Read tool to access
4. Update this README with new document entry

## Maintenance

### Adding New Research

When `rt-ml-technique-researcher-v2` creates a new document:

1. Save to `docs/research/AdvancedTechniqueWebSearches/[category]/`
2. Update this README with entry in appropriate category table
3. (Future) MCP server auto-indexes the document

### Document Quality Standards

All research documents should include:
- **Date created** and agent responsible
- **Primary sources** (NVIDIA, Microsoft, academic papers)
- **Code examples** where applicable
- **Known limitations** and gotchas
- **Cross-references** to related PlasmaDX code

---

**Last Updated:** 2025-10-17
**Document Count:** 53
**Created By:** rt-ml-technique-researcher-v2
**Total Research Sessions:** ~15 (2025-09-20 to 2025-10-11)
