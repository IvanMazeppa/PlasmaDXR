Multi-Agent Pipeline Analysis: Findings and Recommendations

  Current Architecture Strengths

  Your pipeline has a solid foundation:
  - Technique Catalog (8 pyro techniques with distinct parameter ranges)
  - ML-based evaluation (LPIPS + CLIP + enhanced multi-modal)
  - Experiment tracking with knowledge base learning
  - Blender Manual MCP for API documentation (underutilized)

  ---
  Critical Improvements for More Realistic Simulations

  1. Missing: Noise/Upres Pass for Detail (HIGH IMPACT)

  Your technique catalog defines ADVANCED_TECHNIQUES["noise_upscaling"] but it's never used in generate_script().

  The Blender Manual confirms:
  "Adding noise creates finer detailed simulation on top of the base. This makes it possible to add more details to gases without changing overall fluid motion."

  Recommendation: Add a second pass to generated scripts:
  # In generated script, after base simulation:
  settings.use_noise = True
  settings.noise_scale = 2  # 2-4x resolution multiplier
  settings.noise_strength = 1.5  # Turbulence intensity

  Expected Improvement: 2-4× visual detail at lower computation cost than increasing base resolution.

  ---
  2. Limited Emission Dynamics (HIGH IMPACT)

  Current scripts use static emitters. Real explosions have:
  - Animated fuel emission (burst → decay)
  - Temperature ramping (hot ignition → cooling)
  - Variable velocity (explosive initial → settling)

  Recommendation: Add keyframe animation to flow settings:
  # Frame 1-10: Explosive ignition
  flow.fuel_amount = 8.0
  flow.temperature = 6.0
  flow.velocity_normal = 15.0

  # Keyframe at frame 10: Decay
  flow.keyframe_insert(data_path="fuel_amount", frame=10)
  flow.fuel_amount = 0.5  # Fuel runs out
  flow.keyframe_insert(data_path="fuel_amount", frame=25)

  Add technique parameter: emission_dynamics: "burst", "sustained", "pulsing"

  ---
  3. No Effector/Guide Integration (MEDIUM IMPACT)

  Your ADVANCED_TECHNIQUES defines effector guides but they're never implemented:
  - effector_collision - Invisible deflection surfaces
  - effector_guide - Direct fluid along paths
  - domain_guides - Global force fields

  Use Cases:
  - Mushroom cloud with ground deflection
  - Directed plasma streams
  - Smoke flowing around invisible obstacles

  Recommendation: Add an optional effector parameter to techniques:
  "rising_mushroom": {
      ...
      "effectors": {
          "ground_deflector": {
              "type": "COLLISION",
              "shape": "plane",
              "position_z": -3.0,
          }
      }
  }

  ---
  4. Fixed Buoyancy → Unrealistic Physics (MEDIUM IMPACT)

  Current techniques use static alpha/beta buoyancy values. Real combustion:
  - Hot gas rises fast initially
  - Cools and slows as it expands
  - May descend when cooler than ambient

  Recommendation: Add temperature-dependent buoyancy animation or use the Buoyancy Heat parameter more aggressively:
  - Increase beta (heat buoyancy) for more reactive temperature effects
  - Your current ground_hugger uses negative beta (-0.5 to 0.3), but more techniques could benefit

  ---
  5. Missing Adaptive Domain (LOW-MEDIUM IMPACT)

  The manual confirms:
  "Domain shrinks/grows to fit gas, saving computation. 50-80% reduction in empty voxel computation."

  Your scripts set use_adaptive_domain = True in the template but don't configure:
  - add_resolution (extra voxels at boundary)
  - margin (padding around fluid)
  - threshold (density cutoff)

  Recommendation: Add to domain setup:
  settings.use_adaptive_domain = True
  settings.adaptive_margin = 4
  settings.adaptive_threshold = 0.01

  ---
  6. Evaluation Doesn't Use Temporal Analysis (HIGH IMPACT)

  You have analyze_temporal_quality() in asset-evaluator but it's not integrated into the create_asset loop. Fire/smoke simulations have temporal artifacts (flickering, popping) that single-frame CLIP/LPIPS misses.

  Recommendation: After CLIP/LPIPS pass, run temporal check:
  # In iteration loop, after spatial quality check:
  if iteration.clip_score > 0.55:  # Only if close to passing
      temporal_result = await analyze_temporal_quality(
          frame_directory=output_dir,
          sample_rate=5
      )
      if temporal_result["flicker_risk"] == "high":
          iteration.recommendations.append("Increase dissolve_speed to reduce flickering")

  ---
  7. Script Generator Doesn't Consult Blender Manual (CRITICAL)

  The biggest gap: Blender Manual MCP is never queried during generation. This means:
  - No validation that parameters are within valid ranges
  - No check for deprecated Blender 5.0 APIs
  - No discovery of new techniques from manual

  Recommendation: Add a validation step to generate_script():
  # Before generating, validate key parameters
  if effect_type == "pyro":
      # Query manual for current valid ranges
      burning_rate_info = await mcp__blender-manual__search_python_api(
          "burning_rate FluidDomainSettings"
      )
      # Parse range: [0.01, 4.0]
      # Validate: domain_params["burning_rate"] is within range

  ---
  8. Missing Reference Image Library (MEDIUM IMPACT)

  find_reference_images() searches assets/reference_images/ but it's sparse. LPIPS works best with good references.

  Recommendation: Curate a reference library:
  assets/reference_images/
  ├── explosions/
  │   ├── nuclear_mushroom.png
  │   ├── grenade_burst.png
  │   └── fireball_spherical.png
  ├── fire_smoke/
  │   ├── campfire_embers.png
  │   └── wildfire_plume.png
  └── celestial/
      ├── supernova.png
      └── stellar_flare.png

  ---
  Implementation Prioritization

  | Priority | Improvement                   | Effort  | Impact                   |
  |----------|-------------------------------|---------|--------------------------|
  | 1        | Add Noise/Upres pass          | 2 hours | HIGH (2-4× detail)       |
  | 2        | Integrate temporal evaluation | 1 hour  | HIGH (catch flickering)  |
  | 3        | Animated emission dynamics    | 3 hours | HIGH (realistic decay)   |
  | 4        | Blender Manual validation     | 2 hours | MEDIUM (prevent errors)  |
  | 5        | Effector/guide support        | 3 hours | MEDIUM (complex effects) |
  | 6        | Reference image library       | 1 hour  | MEDIUM (better LPIPS)    |
  | 7        | Adaptive domain tuning        | 30 min  | LOW (performance)        |

  ---
  Quick Win: Noise Upres Implementation

  Here's the specific change to immediately improve realism:

  # In server.py generate_script(), add after gas_settings:

  # Noise upres for fine detail (from ADVANCED_TECHNIQUES)
  noise_settings = """
      # Noise upres for fine detail
      settings.use_noise = True
      settings.noise_scale = 2  # 2x resolution boost
      settings.noise_strength = 1.2  # Turbulence intensity
      settings.noise_pos_scale = 2.0  # Vortex size
  """

  # Append to gas_settings
  gas_settings += noise_settings

● ---
  Summary

  Your pipeline has strong foundations but is leaving significant realism potential on the table. The top 3 changes for immediate improvement:

  1. Enable noise/upres in generated scripts (2-4× detail improvement)
  2. Add animated emission dynamics (burst→decay patterns)
  3. Integrate temporal quality check in the iteration loop