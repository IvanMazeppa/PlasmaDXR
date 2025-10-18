# DirectX Raytracing 1.2 New Features and API Changes (2025)

## Source
- Article: [Announcing DirectX Raytracing 1.2, PIX, Neural Rendering and more at GDC 2025!](https://devblogs.microsoft.com/directx/announcing-directx-raytracing-1-2-pix-neural-rendering-and-more-at-gdc-2025/)
- Authors: Microsoft DirectX Team
- Date: March 2025
- Conference: GDC 2025

## Summary

Microsoft announced DirectX Raytracing (DXR) 1.2 at GDC 2025, introducing revolutionary performance improvements and new technologies for real-time ray tracing. The update brings two major new features: Opacity Micromaps (OMM) and Shader Execution Reordering (SER), both delivering substantial leaps in ray tracing performance. DXR 1.2 represents a significant advancement, with performance improvements of up to 40% in complex scenes and achieving up to 2.3x performance improvement in path-traced games with OMM.

## Key Innovation

### 1. Opacity Micromaps (OMM)
- Optimizes alpha-tested geometry by providing dedicated hardware access to opacity information during ray traversal
- Reduces necessity for any-hit shader invocations
- Delivers up to 2.3x performance improvement in path-traced games
- Efficiently handles high-detail opacity data such as alpha textures

### 2. Shader Execution Reordering (SER)
- Addresses GPU inefficiency caused by incoherent ray behavior
- Dynamically groups shader invocations with similar execution paths
- Minimizes thread divergence that has historically plagued ray tracing workloads
- Allows application shader code to inform hardware how to find coherency across rays
- Enables up to 2x performance improvement in ray tracing operations

## Implementation Details

### RT Pipeline State Object v1.1 Improvements
- RTPSO represents a full set of shaders reachable by a DispatchRays call
- All configuration options resolved, including local root signatures and other state
- Can be thought of as an executable state object

### HitObject Enhancements
- Improves flexibility of the ray tracing pipeline
- Common code (vertex fetch, interpolation) no longer needs duplication across closesthit shaders
- Common code can be part of the raygeneration shader, executing before closesthit shading
- HitObject decouples existing TraceRay functionality into traversal and shading stages

### GPU Work Creation Enhancements
- Enhanced support for GPU-driven rendering workflows
- Improved dynamic work generation capabilities

## Performance Metrics
- Overall Performance: Up to 40% improvement in complex scenes (demonstrated by Remedy Entertainment with Alan Wake 2)
- OMM: Up to 2.3x performance improvement in path-traced games
- SER: Up to 2x performance boost in ray tracing operations
- Real-world example: SER saves 24% GPU time on TraceMain pass (NVIDIA GeForce RTX 5080 GPU)

## Hardware Requirements
- **Minimum GPU**: RTX 20 Series and newer (SER is no-op on older hardware, no performance downside)
- **Optimal GPU**: RTX 40/50 Series (full SER acceleration), Intel Arc B-Series, Intel Core Ultra Series 2
- **NVIDIA Support**: Committed driver support across GeForce RTX GPUs
- **Multi-vendor Support**: Active collaboration with AMD, Intel, and Qualcomm

## Implementation Complexity
- **Estimated Dev Time**: Moderate (requires Agility SDK preview and DXC updates)
- **Risk Level**: Medium (preview technology, hardware-dependent features)
- **Dependencies**:
  - AgilitySDK 1.717.1-preview
  - Preview Shader Model 6.9 support in DXC
  - Preview Agility SDK (available late April 2025)

## Related Techniques
- Shader Model 6.9 Cooperative Vectors for neural rendering integration
- Inline ray tracing for compute shaders
- Neural Radiance Cache (NRC) for indirect lighting

## Notes for PlasmaDX Integration

### Immediate Opportunities
1. **SER Integration**: Can be implemented now with preview SDK, provides immediate performance benefits for volumetric ray marching
2. **OMM for Alpha Geometry**: Beneficial if using alpha-tested geometry for particle effects or volume boundaries
3. **HitObject Workflow**: Useful for separating traversal from volumetric shading logic

### RTX 4060Ti Specific Considerations
- SER will provide full acceleration on RTX 40 series hardware
- OMM benefits may be limited if not using complex alpha-tested geometry
- Memory bandwidth constraints may limit some advanced features

### Implementation Priority
1. **High Priority**: SER implementation for ray coherence improvement
2. **Medium Priority**: HitObject integration for cleaner shader architecture
3. **Low Priority**: OMM (unless using alpha-tested volumetric boundaries)

### Timeline Considerations
- Preview SDK available late April 2025
- Production-ready release timeline undefined
- Consider implementing with fallback paths for older DXR versions