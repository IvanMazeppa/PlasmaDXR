# Non‑Raytracing Enhancements — Ranked

1. Async compute overlap (AABB, BLAS/TLAS refit, lighting) — Score: 9.2

Value: Overlap compute with graphics using separate queues/fences; hide RT latency.
Where: Scheduling in `src/core/Application.*` and `RTLightingSystem_RayQuery` dispatch.
Web Search: "D3D12 async compute overlapping ray tracing compute fences".

1. GPU-driven culling and compaction of particles — Score: 9.0

Value: Frustum/size/importance culling and stream compaction before AABBs and RT.
Where: New compute passes prior to `GenerateAABBs`.
Web Search: "GPU frustum culling compute shader prefix sum compaction".

1. Framegraph + transient resource allocator — Score: 8.8

Value: Declarative passes, barrier planning, and transient aliasing to cut VRAM.
Where: New framegraph module orchestrating RT and raster passes.
Web Search: "framegraph transient resource aliasing D3D12".

1. DXR timings and ray count telemetry — Score: 8.6

Value: Timestamp passes and count traced rays; log to CSV for tuning.
Where: Instrument RT passes; extend `src/utils/Logger.*` and logs under `logs/`.
MCP Search: "Where does the engine log GPU/CPU timings today? Show timestamp query usage in `src/`".

1. Quality tiers and runtime toggles — Score: 8.4

Value: Expose rays-per-particle, SER enable, denoiser toggles, max distance, etc.
Where: Extend `Application` config and UI; plumb constants to shaders.
MCP Search: "Find config flags influencing RT (enableRT, particleCount, etc.) in `Application` and `RTLightingSystem_RayQuery`; show definitions/uses".

1. Deterministic/repro modes — Score: 8.2

Value: Stable RNG seeds and replay scripts for consistent debugging and PIX captures.
Where: Seed management on CPU + capture/playback harness.
Web Search: "deterministic random seeds graphics debugging reproducibility".

1. CPU/GPU pipelining for AS builds — Score: 8.1

Value: Prepare next-frame instance descs while GPU builds current AS; double-buffer inputs.
Where: TLAS input staging and copy scheduling.
Web Search: "D3D12 TLAS build CPU GPU overlap double buffering".

1. VRAM budget watchdog — Score: 8.0

Value: Track AS + particle buffers; compact or throttle features near budget.
Where: Central allocator using DXGI memory queries.
Web Search: "DXGI QueryVideoMemoryInfo D3D12 memory budget allocator".

1. Visual debugging overlays — Score: 7.9

Value: On-screen overlays for BVH depth, hit rates, ray dir heatmaps.
Where: HUD draw pass; readback counters.
MCP Search: "Where are existing debug overlays/HUD rendering implemented? Show overlay draw code paths".

1. CI artifacts with GPU capture symbols — Score: 7.8

Value: Store PDBs/DXIL for PIX captures; add perf regression gates.
Where: Build scripts and artifact retention.
Web Search: "PIX GPU capture CI pipeline symbols windows".
