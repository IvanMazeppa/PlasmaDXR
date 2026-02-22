# Settings
DocType: manual-rewritten
DocPath: physics/fluid/type/domain/settings.html
DocVersion: 5.0.1
Model: gpt-5-mini
OriginalWords: 4041
RewrittenWords: 898
CompressionRatio: 0.22
---

One-line summary: Reference for Domain settings controlling fluid (liquid/gas) simulation bounds, resolution, timestepping, gas/fire behavior, and related tradeoffs.

## Domain (general)
- `purpose` (type not specified, range [not specified], default: not specified) — Domain object contains entire simulation; fluids cannot leave the domain: they will either collide with domain edges or disappear depending on settings. Translation, rotation, scaling of the domain are allowed.
- `domain_shape_behavior` (type not specified, range [not specified], default: not specified) — Any mesh can be used as domain, but simulator uses the mesh’s bounding box as the domain bounds → actual simulation domain is rectangular.

TECHNIQUE: Create a domain just large enough to enclose the simulated region. Larger domains → need proportionally more Resolution Divisions and longer bake times.

## Domain parameters
- `Domain Type` (type not specified, range [Liquid, Gas], default: not specified) — Selects whether domain simulates liquid or gas. Liquid domains include all liquid flow objects intersecting the domain. Gas domains consider Smoke, Fire, and Smoke & Fire flow objects. Domain type cannot be changed dynamically.
- `Resolution Divisions` (type not specified, range [not specified], default: not specified) — Controls number of subdivisions (voxels) in the domain. Higher values → more detail but longer compute/bake time. Resolution is defined in subdivisions → larger physical domains require proportionally more divisions to maintain equivalent detail (example: 1 m cube at 64 divisions → 2 m cube needs 128 divisions). The longest bounding-box dimension is used as base for division count. A small preview cube can visualize voxel size.
- `Time Scale` (type not specified, range [not specified], default: not specified) — Controls simulation speed. Lower → slow-motion; higher → advance simulation faster → useful for generating fluids for still renders.
- `Adaptive Time Steps` (type not specified, range [not specified], default: not specified) — When enabled, solver chooses multiple simulation sub-steps per frame automatically based on CFL Number, Timesteps Minimum/Maximum, current Frame Rate, and Time Scale.
- `CFL Number` (type not specified, range [not specified], default: not specified) — Maximum allowed fluid velocity per grid cell measured in grid cells per time step. If fluid would exceed this, solver subdivides the timestep. Higher CFL → fewer sub-steps and less computation time → reduced physical accuracy for fast flows. Lower CFL → more sub-steps, longer compute time → better accuracy at high velocities.
  - NOTE: When lowering CFL, increase Timesteps Maximum. When increasing CFL, adjust Timesteps Minimum accordingly.
- `Timesteps Maximum` (type not specified, range [not specified], default: not specified) — Maximum allowed sub-steps per frame; solver may divide a frame’s step up to this number.
- `Timesteps Minimum` (type not specified, range [not specified], default: not specified) — Minimum number of simulation sub-steps performed per frame.
- `Gravity` (type not specified, range [not specified], default: uses global scene gravity) — By default solver uses global scene gravity. Disabling use of global gravity (scene setting) enables fluid-specific gravity options.
- `Empty Space` (Gas only) (type not specified, range [not specified], default: not specified) — Voxels with values under this threshold are considered empty space. More empty space → optimized rendering and smaller OpenVDB cache sizes.
- `Delete in Obstacle` (type not specified, range [not specified], default: not specified) — Remove any volume of fluid that intersects an obstacle inside the domain.

## Border Collisions (Domain: Gas)
- `Border Collisions` (type not specified, range [not specified], default: not specified) — Controls which domain faces allow fluid to pass through and disappear without influencing the rest of the simulation versus which faces reflect/deflect fluid.

## Gas
- `Buoyancy Density` (type not specified, range [not specified], default: not specified) — Buoyant force based on gas density. Values > 0 → gas rises (lighter than ambient); values < 0 → gas sinks (heavier than ambient).
- `Buoyancy Heat` (type not specified, range [not specified], default: not specified) — Controls how temperature affects gas buoyancy. Dependent on each flow object’s Initial Temperature:
  - Values > 0 → gas from flows with positive Initial Temperature rises; gas from negative Initial Temperature sinks.
  - Values < 0 → behavior inverts (positive Initial Temperature produces sinking gas).
  - NOTE: Gas from multiple flow objects mixes and approaches thermal equilibrium.
- `Vorticity` (type not specified, range [not specified], default: not specified) — Controls turbulence amount. Higher → many small swirls; lower → smoother shapes.

### Dissolve (Gas)
- `Dissolve` (type not specified, range [not specified], default: not specified) — Allow gas to dissipate over time.
- `Time` (type not specified, range [not specified], default: not specified) — Speed of gas dissipation measured in frames.
- `Slow` (type not specified, range [not specified], default: not specified) — When enabled, dissolve uses a logarithmic curve: quick dissipation initially, then slower lingering decay.

## Fire (Domain: Gas)
- `Reaction Speed` (type not specified, range [not specified], default: not specified) — How fast fuel burns. Larger values → fuel burns faster → smaller flames (fuel consumed before traveling far). Smaller values → slower burning → larger flames.
- `Flame Smoke` (type not specified, range [not specified], default: not specified) — Amount of extra smoke generated by burning fuel; most visible when using Fire+Smoke flow objects.
- `Vorticity` (fire-specific) (type not specified, range [not specified], default: not specified) — Additional vorticity applied to flames in addition to the global fluid vorticity.

NOTE: Parameter types, explicit numeric ranges, and defaults are not specified on the source page where omitted.