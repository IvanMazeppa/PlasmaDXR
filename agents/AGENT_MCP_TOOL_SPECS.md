# MCP Tool Specifications – Agent Registry

Created: 2025‑11‑14

This registry defines agent tools and input schemas for the Claude Agent SDK MCP creator.

---

## mission-control (SDK in‑process)
- `dispatch_plan` { goal: string, constraints?: string[] }
- `record_decision` { title: string, rationale: string, artifacts?: string[] }
- `trigger_review` { topic: string, reviewers?: string[] }
- `publish_summary` { path?: string }

## knowledge-archivist (SDK in‑process)
- `register_artifact` { path: string, kind: "screenshot"|"log"|"pix"|"doc", tags?: string[] }
- `search_artifacts` { query: string, kind?: string, limit?: number }
- `summarize_thread` { thread_id: string }
- `export_memory` { out_path: string }

## pipeline-runner (SDK in‑process)
- `rebuild_shaders` { targets?: string[] }
- `run_plasmadx` { config_path?: string, duration_sec?: number, headless?: boolean }
- `capture_screenshot` { out_dir?: string, downscale?: number, include_metadata?: boolean }
- `collect_logs` { since_minutes?: number }
- `run_pix` { capture_at_frame?: number, preset?: "quick"|"full" }

## imageops-agent (SDK in‑process)
- `set_camera_pose` { x: number, y: number, z: number, yaw: number, pitch: number }
- `sweep_path` { waypoints: {x:number,y:number,z:number}[], fps?: number }
- `capture_series` { count: number, interval_sec: number, tag?: string }
- `tag_metadata` { path: string, metadata: object }

## dxr-image-quality-analyst (external)
- `compare_screenshots_ml` { before_path: string, after_path: string, save_heatmap?: boolean }
- `assess_visual_quality` { screenshot_path: string }
- `list_recent_screenshots` { limit?: number }

## log-analysis-rag (external)
- `ingest_logs` { log_dir: string, include_pix?: boolean, max_files?: number }
- `query_logs` { query: string, time_range?: [string,string], particle_count?: number, max_results?: number }
- `diagnose_issue` { symptom: string, context?: object }
- `compare_runs` { baseline_log: string, test_log: string }

## dxr-shadow-engineer (external)
- `research_shadow_techniques` { query: string, focus?: string, include_papers?: boolean, include_code?: boolean }
- `generate_shadow_shader` { technique: string, quality_preset: string, integration: string, features?: string[] }
- `analyze_shadow_performance` { technique: string, particle_count: number, light_count: number }

## dxr-volumetric-pyro-specialist (external)
- `research_pyro_techniques` { query?: string }
- `design_explosion_effect` { effect_type: string, duration_sec?: number, max_radius?: number, peak_temperature?: number, particle_budget?: number }
- `estimate_pyro_performance` { particle_count: number, noise_octaves: number }

## material-system-engineer (external)
- `generate_material_shader` { material_type: string, properties: object, base_shader_template?: string }
- `generate_particle_struct` { base_struct: string, new_fields: object[], target_alignment?: number }
- `generate_imgui_controls` { material_types: string[] }

## gaussian-analyzer (external)
- `analyze_gaussian_parameters` { analysis_depth: string, focus_area: string }
- `simulate_material_properties` { material_type: string, properties: object, render_mode?: string }
- `estimate_performance_impact` { particle_struct_bytes: number, material_types_count: number, shader_complexity: string, particle_counts: number[] }
