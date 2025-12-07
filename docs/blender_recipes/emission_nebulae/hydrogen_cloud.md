# Hydrogen Cloud (Emission Nebula) - Blender VDB Recipe

**Difficulty:** Beginner
**Method:** Mantaflow Smoke Simulation
**Blender Version:** 5.0+
**Export Format:** OpenVDB (density, temperature)
**Estimated File Size:** ~5-15 MB per frame at resolution 128

---

## Overview

This recipe creates a wispy, glowing hydrogen cloud similar to those found in emission nebulae like the Orion Nebula. The result is a volumetric VDB that can be imported into PlasmaDX-Clean for real-time rendering with proper emission and scattering.

Hydrogen clouds are the simplest celestial volume to create - they're essentially colorized smoke with emission. This makes them an excellent starting point for learning the Blender-to-PlasmaDX workflow.

## Visual Reference

### Real-World Examples
- **Orion Nebula (M42)** - Classic red/pink emission nebula
- **Carina Nebula** - Complex gas pillars with emission
- **Eagle Nebula** - "Pillars of Creation" structure

### Target Appearance
- **Color:** Pink/red (H-alpha emission) with blue/purple accents (reflection)
- **Structure:** Wispy, filamentary, cloud-like with varying density
- **Motion:** Slow turbulent drift, not static

---

## Astrophysical Properties

| Property | Value | Notes |
|----------|-------|-------|
| Temperature | 8,000 - 12,000 K | Ionized hydrogen emission |
| Density | 10-1000 atoms/cm³ | Extremely diffuse by Earth standards |
| Scale | 1-100 parsecs | In Blender: 1 parsec = 100 units |
| Composition | 90% H, 10% He | Trace metals give color variation |

### Emission/Absorption
- **Emission:** Strong H-alpha (656.3nm, red), H-beta (486.1nm, cyan)
- **Absorption:** Minimal - these are emission nebulae
- **Scattering:** Moderate backward scattering from dust inclusions

### Color Science
The iconic pink/red color comes from ionized hydrogen recombination:
- Electrons captured by protons emit photons at specific wavelengths
- H-alpha (red) is the brightest emission line
- H-beta (blue-green) provides color variation
- Dust reflection adds blue haze at edges

---

## Blender Workflow

### Prerequisites
- [ ] Blender 5.0 or later installed
- [ ] Basic understanding of Blender's 3D viewport navigation
- [ ] ~500MB free disk space for cache

### Step 1: Create the Domain

1. **Open Blender** with default scene
2. **Delete default cube** (select, press X, confirm)
3. **Add domain cube:**
   - Press Shift+A → Mesh → Cube
   - Scale to 4m × 4m × 4m (press S, type 4, Enter)
   - Move up so base is at origin (press G, Z, type 2, Enter)
4. **Name it** "NebulaDomain" in the Outliner

### Step 2: Add Fluid Physics

1. **Select NebulaDomain**
2. **Go to Physics tab** (bouncing ball icon in Properties panel)
3. **Click "Fluid"** to add fluid physics
4. **Set Type to "Domain"**
5. **Set Domain Type to "Gas"** (smoke, not liquid)

### Step 3: Configure Domain Settings

In the Domain Settings panel:

**Resolution:**
- Resolution Divisions: **64** (start low, increase later)
- Enable "Adaptive Domain": **Yes** (saves memory)

**Gas:**
- Buoyancy Density: **0.5** (gentle rise)
- Heat: **1.0**
- Vorticity: **0.3** (creates swirling)

**Dissolve:**
- Enable Dissolve Smoke: **Yes**
- Dissolve Time: **80** frames (slow fade)

### Step 4: Create the Emitter

1. **Add a sphere:** Shift+A → Mesh → UV Sphere
2. **Scale to 0.3m:** S, 0.3, Enter
3. **Position inside domain:** G, Z, 1.5, Enter
4. **Name it** "NebulaEmitter"
5. **Add Fluid physics** (Physics tab → Fluid)
6. **Set Type to "Flow"**

**Flow Settings:**
- Flow Type: **Smoke**
- Flow Behavior: **Inflow** (continuous emission)
- Smoke Color: (0.8, 0.3, 0.4) - pinkish
- Temperature: **2.0** (warm, for color variation)
- Density: **1.0**

### Step 5: Set Up the Material

1. **Select NebulaDomain**
2. **Go to Material tab** (checkered sphere icon)
3. **Click "New"** to create material
4. **Switch to Shader Editor** (or use Material Properties)
5. **Delete default Principled BSDF**
6. **Add Principled Volume node:** Shift+A → Shader → Principled Volume
7. **Connect to Volume output** of Material Output

**Principled Volume Settings:**
| Setting | Value | Why |
|---------|-------|-----|
| Color | (0.8, 0.4, 0.5) | Pink/red base |
| Density | 0.8 | Visible but not opaque |
| Anisotropy | -0.3 | Backward scattering (wispy look) |
| Absorption Color | (0.1, 0.1, 0.2) | Slight blue absorption |
| Emission Strength | 2.0 | Glowing effect |
| Emission Color | (0.9, 0.4, 0.5) | Pink emission |
| Blackbody Intensity | 0.0 | Using direct color, not temperature |
| Temperature | 10000 | For reference (not affecting color here) |

### Step 6: Configure VDB Export

1. **Select NebulaDomain**
2. **Go to Physics tab → Fluid → Cache**
3. **Configure cache settings:**

| Setting | Value | Why |
|---------|-------|-----|
| Cache Directory | `//vdb_cache/` | Relative to .blend file |
| Frame Start | 1 | |
| Frame End | 100 | Adjust as needed |
| Type | **All** | Single bake operation |
| Data Format | **OpenVDB** | Required for PlasmaDX |
| Compression | **BLOSC** | Fast + good compression |
| Precision | **Half** | 16-bit, good balance |

### Step 7: Bake the Simulation

1. **Save your .blend file** (Ctrl+S) - IMPORTANT!
2. **In Cache panel, click "Bake All"**
3. **Wait...** (1-10 minutes depending on resolution)
4. **Check cache folder** for .vdb files

---

## Key Settings Reference

### Domain Settings Summary
| Setting | Value | Why |
|---------|-------|-----|
| Resolution Divisions | 64-128 | Balance detail/speed |
| Adaptive Domain | On | Memory optimization |
| Buoyancy Density | 0.5 | Gentle upward drift |
| Vorticity | 0.3 | Swirling turbulence |
| Dissolve Time | 80 | Slow fade for nebula |

### Principled Volume Summary
| Setting | Value | Why |
|---------|-------|-----|
| Color | (0.8, 0.4, 0.5) | H-alpha pink |
| Density | 0.8 | Semi-transparent |
| Anisotropy | -0.3 | Backward scatter |
| Emission Strength | 2.0 | Self-illumination |

---

## Python Automation

```python
"""
Hydrogen Cloud (Emission Nebula) - Quick Setup Script
Creates a complete nebula simulation ready for VDB export.

Usage:
1. Open Blender
2. Switch to Scripting workspace
3. Create new text file, paste this script
4. Run Script (Alt+P)
"""
import bpy
import math

def clear_scene():
    """Remove default objects."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def create_hydrogen_cloud(
    name: str = "HydrogenCloud",
    domain_size: float = 4.0,
    resolution: int = 64,
    frame_end: int = 100,
    cache_dir: str = "//vdb_cache/"
):
    """
    Create a complete hydrogen cloud nebula simulation.

    Args:
        name: Base name for objects
        domain_size: Size of domain cube in meters
        resolution: Voxel resolution (higher = more detail)
        frame_end: Last frame to simulate
        cache_dir: Directory for VDB cache files
    """
    print(f"[Nebula] Creating hydrogen cloud '{name}'...")

    # Create domain
    bpy.ops.mesh.primitive_cube_add(
        size=domain_size,
        location=(0, 0, domain_size / 2)
    )
    domain = bpy.context.active_object
    domain.name = f"{name}_Domain"

    # Add fluid modifier
    bpy.ops.object.modifier_add(type='FLUID')
    domain.modifiers['Fluid'].fluid_type = 'DOMAIN'
    settings = domain.modifiers['Fluid'].domain_settings

    # Domain settings
    settings.domain_type = 'GAS'
    settings.resolution_max = resolution
    settings.use_adaptive_domain = True

    # Gas settings
    settings.alpha = 0.5  # Buoyancy
    settings.beta = 1.0   # Heat buoyancy
    settings.vorticity = 0.3

    # Dissolve
    settings.use_dissolve_smoke = True
    settings.dissolve_speed = 80

    # VDB Export settings
    settings.cache_data_format = 'OPENVDB'
    settings.openvdb_cache_compress_type = 'BLOSC'
    settings.cache_precision = 'HALF'
    settings.cache_directory = cache_dir
    settings.cache_frame_start = 1
    settings.cache_frame_end = frame_end

    print(f"  Domain created: {resolution}^3 resolution")

    # Create emitter
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=domain_size * 0.1,
        location=(0, 0, domain_size * 0.4)
    )
    emitter = bpy.context.active_object
    emitter.name = f"{name}_Emitter"

    # Add flow modifier
    bpy.ops.object.modifier_add(type='FLUID')
    emitter.modifiers['Fluid'].fluid_type = 'FLOW'
    flow = emitter.modifiers['Fluid'].flow_settings

    flow.flow_type = 'SMOKE'
    flow.flow_behavior = 'INFLOW'
    flow.smoke_color = (0.8, 0.3, 0.4)  # Pink
    flow.temperature = 2.0
    flow.density = 1.0

    print(f"  Emitter created")

    # Create material
    mat = bpy.data.materials.new(name=f"{name}_Material")
    mat.use_nodes = True
    domain.data.materials.append(mat)

    # Set up volume shader
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    nodes.clear()

    # Add Principled Volume
    volume = nodes.new(type='ShaderNodeVolumePrincipled')
    volume.location = (0, 0)
    volume.inputs['Color'].default_value = (0.8, 0.4, 0.5, 1.0)
    volume.inputs['Density'].default_value = 0.8
    volume.inputs['Anisotropy'].default_value = -0.3
    volume.inputs['Absorption Color'].default_value = (0.1, 0.1, 0.2, 1.0)
    volume.inputs['Emission Strength'].default_value = 2.0
    volume.inputs['Emission Color'].default_value = (0.9, 0.4, 0.5, 1.0)

    # Add output
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (300, 0)

    # Connect
    links.new(volume.outputs['Volume'], output.inputs['Volume'])

    print(f"  Material created")

    print(f"\n[Nebula] Setup complete!")
    print(f"  Domain: {domain.name}")
    print(f"  Emitter: {emitter.name}")
    print(f"  Cache: {cache_dir}")
    print(f"  Frames: 1-{frame_end}")
    print(f"\nNext steps:")
    print(f"  1. Save your .blend file (Ctrl+S)")
    print(f"  2. Select '{domain.name}'")
    print(f"  3. Go to Physics > Fluid > Cache")
    print(f"  4. Click 'Bake All'")
    print(f"  5. Find VDB files in {cache_dir}")

    return domain, emitter

# Run if executed directly
if __name__ == "__main__":
    clear_scene()
    domain, emitter = create_hydrogen_cloud(
        name="HydrogenCloud",
        resolution=64,
        frame_end=100
    )
```

---

## VDB Export Configuration

### Required Grids
- **density**: Main visible structure (required)
- **temperature**: For emission color variation (optional but recommended)

### Grid Contents After Bake
Your VDB files will contain:
- `density` - Smoke density (0.0 to ~1.0)
- `temperature` - Heat values (normalized)
- `velocity` - Motion vectors (if motion blur needed)

### File Naming
Blender creates files like:
```
vdb_cache/
├── fluid_data_0001.vdb
├── fluid_data_0002.vdb
├── ...
└── fluid_data_0100.vdb
```

---

## PlasmaDX Integration

### Material Type Mapping
This hydrogen cloud should use material type: `GAS_CLOUD`

| Property | Blender Value | PlasmaDX Value | Conversion |
|----------|--------------|----------------|------------|
| Opacity | Density 0.8 | opacity 0.3 | Divide by 2.5 |
| Scattering | Anisotropy -0.3 | phase_g -0.3 | Direct |
| Emission | Strength 2.0 | emission 0.5 | Divide by 4 |
| Albedo | Color (0.8,0.4,0.5) | albedo (0.8,0.4,0.5) | Direct |

### Expected Visual Result
In PlasmaDX-Clean, this nebula should appear as:
- Soft, glowing pink/red cloud
- Visible rim lighting from directional lights
- Gentle internal glow (emission)
- Wispy edges due to backward scattering
- Semi-transparent with visible depth

### Loading in PlasmaDX
```cpp
// Pseudo-code for VDB loading (NanoVDB system)
NanoVDBVolume nebula = LoadVDB("vdb_cache/fluid_data_0050.vdb");
nebula.SetMaterialType(MaterialType::GAS_CLOUD);
nebula.SetEmissionMultiplier(0.5f);
nebula.SetScatteringCoefficient(1.2f);
nebula.SetPhaseG(-0.3f);
```

---

## Variations

### Variation 1: Blue Reflection Nebula
Change settings for scattered starlight appearance:
- Principled Volume Color: (0.4, 0.5, 0.9)
- Anisotropy: +0.3 (forward scattering)
- Emission Strength: 0.5 (mostly reflected, not emitting)

### Variation 2: Dense Star-Forming Core
More opaque, hotter central region:
- Flow Density: 2.0
- Temperature: 4.0
- Dissolve Time: 150 (slower fade)
- Principled Volume Density: 1.5

### Variation 3: Animated Turbulence
Add motion for dynamic nebula:
- Enable "Noise" in domain settings
- Noise Strength: 2.0
- Noise Scale: 4.0
- Animate emitter position along a path

---

## Troubleshooting

### Issue: Simulation is invisible in viewport
**Solution:**
1. Check "Display As" in Fluid Domain settings is set to "Final"
2. Ensure frame is within cache range
3. Check cache directory has .vdb files

### Issue: VDB file is 0 bytes
**Solution:**
1. Save .blend file before baking
2. Check cache directory is writable
3. Ensure simulation has density (emitter is inside domain)

### Issue: Simulation looks blocky
**Solution:**
1. Increase Resolution Divisions (try 128)
2. Enable "High Resolution" in domain if available
3. Increase Noise divisions for detail

### Issue: Smoke disappears too quickly
**Solution:**
1. Increase Dissolve Time (try 150+)
2. Decrease Buoyancy to keep smoke in domain longer
3. Make domain taller

---

## Related Recipes
- [Emission Pillar](emission_pillar.md) - More structured, column-like nebula
- [Dark Nebula](../dark_structures/dark_nebula.md) - Absorption instead of emission
- [Supernova Remnant](../explosions/supernova_remnant.md) - Explosive shell structure

---

*Recipe Version: 1.0.0*
*Last Updated: 2025-12-07*
*Tested with: Blender 5.0*
*Author: celestial-body-curator agent*
