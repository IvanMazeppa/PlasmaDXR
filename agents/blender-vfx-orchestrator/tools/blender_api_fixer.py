"""
Blender 5.0 API Fixer.

Pre-execution validation and auto-fix for known-bad API patterns.
This breaks the infinite loop caused by Coordinator diagnoses that
_modify_script_impl cannot handle.

The pattern:
1. Script fails with deprecated/removed API call
2. Coordinator correctly diagnoses the issue
3. _modify_script_impl can only change Config class params
4. Fix never applied → same error → infinite loop

This module fixes known-bad patterns BEFORE execution.

VECTOR STORE VALIDATION (Option B):
Additionally validates ALL bpy.*/bmesh.* calls against the Blender 5.0
vector store to catch API hallucinations that aren't in the static fix list.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

# Known-bad patterns and their fixes for Blender 5.0
# Format: (pattern_regex, replacement, description)
BLENDER_50_FIXES: List[Tuple[str, str, str]] = [
    # ShaderNodeTexMusgrave removed in Blender 4.0+ → use ShaderNodeTexNoise
    (
        r"ShaderNodeTexMusgrave",
        r"ShaderNodeTexNoise",
        "ShaderNodeTexMusgrave → ShaderNodeTexNoise (Musgrave removed in 4.0)"
    ),

    # ColorRamp.elements.clear() was removed in Blender 5.0
    (
        r"(\w+\.color_ramp\.elements)\.clear\(\)",
        r"""# Blender 5.0: elements.clear() removed - remove all but first element
while len(\1) > 1:
    \1.remove(\1[0])""",
        "ColorRamp.elements.clear() → while loop removal"
    ),

    # use_nodes is always True in Blender 5.0 — remove line entirely (no-op that triggers
    # DeprecationWarning → exit_code=1 in headless mode)
    (
        r"[^\n]*\.use_nodes\s*=\s*True[^\n]*\n?",
        r"",
        "use_nodes removal (always True in 5.0)"
    ),

    # lamp → light (old API)
    (
        r"bpy\.types\.Lamp",
        r"bpy.types.Light",
        "bpy.types.Lamp → bpy.types.Light"
    ),
    (
        r"\.lamp\b",
        r".light",
        ".lamp → .light"
    ),

    # render.image_settings.color_mode changes
    (
        r"color_mode\s*=\s*['\"]BW['\"]",
        r"color_mode = 'RGBA'  # BW removed in 5.0, use RGBA with compositor",
        "color_mode BW → RGBA"
    ),

    # bpy.ops.import_scene.obj renamed
    (
        r"bpy\.ops\.import_scene\.obj\(",
        r"bpy.ops.wm.obj_import(",
        "import_scene.obj → wm.obj_import"
    ),

    # bpy.ops.export_scene.obj renamed
    (
        r"bpy\.ops\.export_scene\.obj\(",
        r"bpy.ops.wm.obj_export(",
        "export_scene.obj → wm.obj_export"
    ),

    # Principled BSDF input name changes in 4.0+
    (
        r"\.inputs\[['\"]Subsurface['\"]",
        r".inputs['Subsurface Weight'",
        "BSDF Subsurface → Subsurface Weight"
    ),
    (
        r"\.inputs\[['\"]Transmission['\"]",
        r".inputs['Transmission Weight'",
        "BSDF Transmission → Transmission Weight"
    ),
    (
        r"\.inputs\[['\"]Specular['\"]",
        r".inputs['Specular IOR Level'",
        "BSDF Specular → Specular IOR Level"
    ),
    (
        r"\.inputs\[['\"]Clearcoat['\"]",
        r".inputs['Coat Weight'",
        "BSDF Clearcoat → Coat Weight"
    ),
    (
        r"\.inputs\[['\"]Clearcoat Roughness['\"]",
        r".inputs['Coat Roughness'",
        "BSDF Clearcoat Roughness → Coat Roughness"
    ),

    # World Output node input renamed in Blender 5.0
    # The input socket is named 'Surface' not 'World'
    (
        r"\.inputs\[['\"]World['\"]",
        r".inputs['Surface'",
        "World Output inputs['World'] → inputs['Surface'] (Blender 5.0)"
    ),

    # Brick Texture node: LLMs hallucinate 'Mortar Color' but correct name is 'Mortar'
    (
        r"\.inputs\[['\"]Mortar Color['\"]",
        r".inputs['Mortar'",
        "Brick Texture inputs['Mortar Color'] → inputs['Mortar']"
    ),
    # Brick Texture node: LLMs hallucinate 'Brick Color 1/2' but correct is 'Color 1/2'
    (
        r"\.inputs\[['\"]Brick Color 1['\"]",
        r".inputs['Color 1'",
        "Brick Texture inputs['Brick Color 1'] → inputs['Color 1']"
    ),
    (
        r"\.inputs\[['\"]Brick Color 2['\"]",
        r".inputs['Color 2'",
        "Brick Texture inputs['Brick Color 2'] → inputs['Color 2']"
    ),

    # Cycles samples path changes
    (
        r"scene\.cycles\.progressive\s*=",
        r"# progressive removed in 4.0+\n# scene.cycles.progressive =",
        "cycles.progressive removed"
    ),

    # Object.select deprecated → Object.select_set()
    (
        r"(\w+)\.select\s*=\s*True",
        r"\1.select_set(True)",
        "obj.select = True → obj.select_set(True)"
    ),
    (
        r"(\w+)\.select\s*=\s*False",
        r"\1.select_set(False)",
        "obj.select = False → obj.select_set(False)"
    ),

    # scene.update() removed → depsgraph.update()
    (
        r"bpy\.context\.scene\.update\(\)",
        r"bpy.context.view_layer.depsgraph.update()",
        "scene.update() → depsgraph.update()"
    ),

    # use_auto_smooth removed in Blender 4.1+ (now handled automatically)
    (
        r"(?m)^.*\.use_auto_smooth\s*=\s*.*$",
        r"# Blender 4.1+: use_auto_smooth removed (handled automatically)",
        "use_auto_smooth removed"
    ),

    # FluidFlowSettings.sampling_substeps does NOT EXIST - correct is subframes
    (
        r"\.sampling_substeps\s*=",
        r".subframes =",
        "sampling_substeps → subframes (FluidFlowSettings)"
    ),

    # FluidDomainSettings.use_dissolve does NOT EXIST - correct is use_dissolve_smoke
    (
        r"\.use_dissolve\s*=",
        r".use_dissolve_smoke =",
        "use_dissolve → use_dissolve_smoke (typo fix)"
    ),
    # Fluid domain adaptive flag renamed
    (
        r"\.adaptive_domain\s*=",
        r".use_adaptive_domain =",
        "adaptive_domain → use_adaptive_domain"
    ),
    # Fix accidental double-prefix from corrections
    (
        r"\.use_use_adaptive_domain\s*=",
        r".use_adaptive_domain =",
        "use_use_adaptive_domain → use_adaptive_domain"
    ),
    # Invalid flow behavior enum (GAS -> INFLOW)
    (
        r"flow_behavior\s*=\s*['\"]GAS['\"]",
        r"flow_behavior = 'INFLOW'",
        "flow_behavior GAS → INFLOW"
    ),
    # Invalid flow type enum (GAS -> FIRE)
    (
        r"flow_type\s*=\s*['\"]GAS['\"]",
        r"flow_type = 'FIRE'",
        "flow_type GAS → FIRE"
    ),

    # Fluid behavior constants changed
    (
        r"flow_behavior\s*=\s*['\"]INFLOW_OUTFLOW['\"]",
        r"flow_behavior = 'INFLOW'",
        "INFLOW_OUTFLOW → INFLOW"
    ),
    # FluidDomainSettings.use_caching removed in Blender 5.0
    (
        r"(?m)^.*\.use_caching\s*=\s*.*$",
        r"# Blender 5.0: use_caching removed (line deleted)",
        "use_caching removed"
    ),
    # FluidDomainSettings.noise_res_factor removed in Blender 5.0
    (
        r"(?m)^.*\.noise_res_factor\s*=\s*.*$",
        r"# Blender 5.0: noise_res_factor removed (line deleted)",
        "noise_res_factor removed"
    ),
    # bpy.ops.fluid.free_all() fails on fresh scenes with "grids still in use"
    # Replace with pass (not just comment) to preserve try/except structure
    (
        r"bpy\.ops\.fluid\.free_all\(\)",
        r"pass  # Blender 5.0: free_all() removed - causes 'grids still in use'",
        "free_all() removed (grids in use error)"
    ),
    # Also catch the with context override version - replace entire with block
    (
        r"(?m)^(\s*)with\s+bpy\.context\.temp_override[^:]*:\s*\n\s*bpy\.ops\.fluid\.free_all\(\)[^\n]*\n",
        r"\1pass  # Blender 5.0: free_all() with temp_override removed\n",
        "free_all() with override removed"
    ),
    # CRITICAL BUG FIX: temp_override + quick_smoke corrupts the scene in Blender 5.0
    # The emitter object is deleted and replaced with a Cube, and the FLOW modifier is lost.
    # Fix: Replace temp_override block with direct call (simple selection is sufficient)
    (
        r"(?m)^(\s*)with\s+bpy\.context\.temp_override\([^)]*\):\s*\n\s*bpy\.ops\.object\.quick_smoke\(\)",
        r"\1# API Fixer: temp_override removed (corrupts quick_smoke in Blender 5.0)\n\1bpy.ops.object.quick_smoke()",
        "temp_override removed from quick_smoke (Blender 5.0 bug)"
    ),
    # FluidDomainSettings.resolution_divisions does NOT exist - use resolution_max
    (
        r"\.resolution_divisions\b",
        r".resolution_max",
        "resolution_divisions → resolution_max"
    ),
    (
        r"['\"]resolution_divisions['\"]",
        r"'resolution_max'",
        "resolution_divisions key → resolution_max"
    ),
    # LLM TYPO: use_adaptive_time_steps → use_adaptive_timesteps (no underscore between time/steps)
    (
        r"\.use_adaptive_time_steps\b",
        r".use_adaptive_timesteps",
        "use_adaptive_time_steps → use_adaptive_timesteps (typo fix)"
    ),
    (
        r"['\"]use_adaptive_time_steps['\"]",
        r"'use_adaptive_timesteps'",
        "use_adaptive_time_steps key → use_adaptive_timesteps (typo fix)"
    ),
    # LLM TRUNCATION: cache_format → cache_data_format (missing 'data' part)
    (
        r"\.cache_format\b",
        r".cache_data_format",
        "cache_format → cache_data_format (typo fix)"
    ),
    (
        r"['\"]cache_format['\"]",
        r"'cache_data_format'",
        "cache_format key → cache_data_format (typo fix)"
    ),
    # LLM HALLUCINATION: FluidFlowSettings.velocity does NOT exist
    # Correct attributes: velocity_factor, velocity_normal, velocity_random, use_initial_velocity
    # Note: Use negative lookbehind to avoid double-applying (velocity_factor → velocity_factor_factor)
    (
        r"(flow_settings|flow)\.velocity(?!_)\s*=",
        r"\1.velocity_factor =",
        "velocity → velocity_factor (FluidFlowSettings)"
    ),
    # LLM HALLUCINATION: velocity_multi does NOT exist
    (
        r"\.velocity_multi\s*=",
        r".velocity_factor =",
        "velocity_multi → velocity_factor (FluidFlowSettings)"
    ),
    # LLM HALLUCINATION: FluidDomainSettings.reaction_speed does NOT exist in Blender 5.0
    # The correct attribute is burning_rate (float 0.0-4.0, default 0.75)
    (
        r"\.reaction_speed\s*=",
        r".burning_rate =",
        "reaction_speed → burning_rate (FluidDomainSettings)"
    ),
    # LLM HALLUCINATION: fire_reaction_speed also does NOT exist
    (
        r"\.fire_reaction_speed\s*=",
        r".burning_rate =",
        "fire_reaction_speed → burning_rate (FluidDomainSettings)"
    ),
    # String key variants for rna_set / set_if_exists helpers
    (
        r"['\"]reaction_speed['\"]",
        r"'burning_rate'",
        "reaction_speed key → burning_rate"
    ),
    (
        r"['\"]fire_reaction_speed['\"]",
        r"'burning_rate'",
        "fire_reaction_speed key → burning_rate"
    ),
    # Type fix: noise_scale must be int, not float
    # Match patterns like: noise_scale = 1.0, noise_scale = 2.0, etc.
    (
        r"(\.noise_scale\s*=\s*)(\d+)\.0\b",
        r"\1\2  # Must be int, not float",
        "noise_scale float → int"
    ),
    # P0 FIX: Volume shader density too low - boost from 3.0 to 10.0
    # This fixes "grey sphere" renders where the volume is too transparent
    (
        r"(multiply\.inputs\[1\]\.default_value\s*=\s*)([0-5]\.0)",
        r"\g<1>10.0  # Boosted from \2 for visibility",
        "Volume density multiplier boosted (grey sphere fix)"
    ),
    # P0 FIX: Emission strength too low for fire visibility
    (
        r"(bb_emission\.inputs\['Strength'\]\.default_value\s*=\s*)([0-5]\.0)",
        r"\g<1>20.0  # Boosted from \2 for fire visibility",
        "Emission strength boosted (fire visibility fix)"
    ),
    # P0 FIX: Blackbody intensity too low
    (
        r"(inputs\['Blackbody Intensity'\]\.default_value\s*=\s*)([0-3]\.0)",
        r"\g<1>8.0  # Boosted from \2 for fire glow",
        "Blackbody intensity boosted (fire glow fix)"
    ),

    # Render only representative frames for quality evaluation, not full animation
    # Pattern 1: FRAME_START/FRAME_END constants
    (
        r"(frames\s*=\s*)range\(\s*FRAME_START\s*,\s*FRAME_END\s*\+\s*1\s*\)",
        r"\1[min(FRAME_START + 5, FRAME_END), (FRAME_START + FRAME_END) // 2, FRAME_END]  # Representative frames for eval",
        "Render 3 representative frames instead of full animation"
    ),
    # Pattern 1b: Config.FRAME_START/FRAME_END in frames range
    (
        r"(frames\s*=\s*)range\(\s*Config\.FRAME_START\s*,\s*Config\.FRAME_END\s*\+\s*1\s*(?:,\s*[^)]*)?\)",
        r"\1[min(Config.FRAME_START + 5, Config.FRAME_END), (Config.FRAME_START + Config.FRAME_END) // 2, Config.FRAME_END]  # Representative frames for eval",
        "Render 3 representative frames (Config.FRAME_* range)"
    ),
    # Pattern 1b-alt: frames = list(range(...)) (Config)
    (
        r"(frames\s*=\s*)list\(range\(\s*Config\.FRAME_START\s*,\s*Config\.FRAME_END\s*\+\s*1\s*(?:,\s*[^)]*)?\)\)",
        r"\1[min(Config.FRAME_START + 5, Config.FRAME_END), (Config.FRAME_START + Config.FRAME_END) // 2, Config.FRAME_END]  # Representative frames for eval",
        "Render 3 representative frames (Config.FRAME_* list(range))"
    ),
    # Pattern 1b-alt: frames = list(range(...)) (FRAME_START)
    (
        r"(frames\s*=\s*)list\(range\(\s*FRAME_START\s*,\s*FRAME_END\s*\+\s*1\s*(?:,\s*[^)]*)?\)\)",
        r"\1[min(FRAME_START + 5, FRAME_END), (FRAME_START + FRAME_END) // 2, FRAME_END]  # Representative frames for eval",
        "Render 3 representative frames (FRAME_START list(range))"
    ),
    # Pattern 1b-alt: frames = list(range(...)) (scene.frame_start/end)
    (
        r"(frames\s*=\s*)list\(range\(\s*scene\.frame_start\s*,\s*scene\.frame_end\s*\+\s*1\s*(?:,\s*[^)]*)?\)\)",
        r"\1[min(scene.frame_start + 5, scene.frame_end), (scene.frame_start + scene.frame_end) // 2, scene.frame_end]  # Representative frames for eval",
        "Render 3 representative frames (scene.frame_start list(range))"
    ),
    # Pattern 1c: return list(range(...)) variants (Config)
    (
        r"return\s+list\(range\(\s*Config\.FRAME_START\s*,\s*Config\.FRAME_END\s*\+\s*1\s*(?:,\s*[^)]*)?\)\)",
        "return [min(Config.FRAME_START + 5, Config.FRAME_END), (Config.FRAME_START + Config.FRAME_END) // 2, Config.FRAME_END]  # Representative frames for eval",
        "Render 3 representative frames (Config.FRAME_* return list(range))"
    ),
    # Pattern 1d: return list(range(...)) variants (FRAME_START)
    (
        r"return\s+list\(range\(\s*FRAME_START\s*,\s*FRAME_END\s*\+\s*1\s*(?:,\s*[^)]*)?\)\)",
        "return [min(FRAME_START + 5, FRAME_END), (FRAME_START + FRAME_END) // 2, FRAME_END]  # Representative frames for eval",
        "Render 3 representative frames (FRAME_START return list(range))"
    ),
    # Pattern 2: scene.frame_start/frame_end (most common in generated scripts)
    (
        r"for\s+(\w+)\s+in\s+range\(\s*scene\.frame_start\s*,\s*scene\.frame_end\s*\+\s*1\s*\)\s*:",
        r"for \1 in [min(scene.frame_start + 5, scene.frame_end), (scene.frame_start + scene.frame_end) // 2, scene.frame_end]:  # Representative frames",
        "Render 3 representative frames (scene.frame_start/end pattern)"
    ),
    # Pattern 2b: Config.FRAME_START/FRAME_END in for-loop
    (
        r"for\s+(\w+)\s+in\s+range\(\s*Config\.FRAME_START\s*,\s*Config\.FRAME_END\s*\+\s*1\s*(?:,\s*[^)]*)?\)\s*:",
        r"for \1 in [min(Config.FRAME_START + 5, Config.FRAME_END), (Config.FRAME_START + Config.FRAME_END) // 2, Config.FRAME_END]:  # Representative frames",
        "Render 3 representative frames (Config.FRAME_* loop)"
    ),
    # Pattern 2c: FRAME_START/FRAME_END in for-loop
    (
        r"for\s+(\w+)\s+in\s+range\(\s*FRAME_START\s*,\s*FRAME_END\s*\+\s*1\s*(?:,\s*[^)]*)?\)\s*:",
        r"for \1 in [min(FRAME_START + 5, FRAME_END), (FRAME_START + FRAME_END) // 2, FRAME_END]:  # Representative frames",
        "Render 3 representative frames (FRAME_START loop)"
    ),
    # Pattern 3: frame_start/frame_end variables (lowercase)
    (
        r"for\s+(\w+)\s+in\s+range\(\s*frame_start\s*,\s*frame_end\s*\+\s*1\s*\)\s*:",
        r"for \1 in [min(frame_start + 5, frame_end), (frame_start + frame_end) // 2, frame_end]:  # Representative frames",
        "Render 3 representative frames (frame_start/end pattern)"
    ),

    # Mantaflow cache_type REPLAY doesn't produce full bake data - must be ALL
    (
        r"cache_type\s*=\s*['\"]REPLAY['\"]",
        r"cache_type = 'ALL'  # API Fixer: REPLAY doesn't produce full bake data",
        "cache_type REPLAY → ALL (full bake data)"
    ),

    # Mantaflow cache_type MODULAR also doesn't work correctly in headless mode
    # User reported physics simulation fails with MODULAR, works with ALL
    (
        r"cache_type\s*=\s*['\"]MODULAR['\"]",
        r"cache_type = 'ALL'  # API Fixer: MODULAR fails in headless, use ALL",
        "cache_type MODULAR → ALL (headless fix)"
    ),

    # Blender-relative cache paths (//) don't work in headless mode without .blend file
    # Convert to absolute path using project_root or os.getcwd()
    (
        r"cache_dir\s*=\s*['\"]//([^'\"]+)['\"]",
        r"cache_dir = os.environ.get('BLENDER_CACHE_DIR') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '\1')  # API Fixer: absolute path for headless",
        "Blender-relative cache path → env-aware absolute path (headless fix)"
    ),
    # Cache directory defined from script dir (allow env override for consistency)
    (
        r"cache_dir\s*=\s*os\.path\.join\(\s*os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\s*,\s*['\"]([^'\"]+)['\"]\s*\)",
        r"cache_dir = os.environ.get('BLENDER_CACHE_DIR') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '\1')  # API Fixer: env override",
        "cache_dir uses BLENDER_CACHE_DIR when set"
    ),
    # Project root from // (headless-safe, env-aware)
    (
        r"(\w+)\s*=\s*bpy\.path\.abspath\(['\"]//['\"]\)",
        r"\1 = os.environ.get('BLENDER_OUTPUT_DIR') or bpy.path.abspath('//') or os.path.dirname(os.path.abspath(__file__))  # API Fixer: output root",
        "bpy.path.abspath('//') → env-aware output root"
    ),

    # =============================================================================
    # BMESH OPERATORS API CHANGES (Blender 5.0)
    # =============================================================================
    # bmesh.ops.create_cone: diameter1/diameter2 renamed to radius1/radius2 in Blender 5.0
    # Note: The values also changed semantically (diameter was actually treated as radius
    # in older versions, so the new naming is more accurate)
    (
        r"bmesh\.ops\.create_cone\(([^)]*)\bdiameter1\s*=",
        r"bmesh.ops.create_cone(\1radius1 =",
        "bmesh.ops.create_cone: diameter1 → radius1 (Blender 5.0)"
    ),
    (
        r"bmesh\.ops\.create_cone\(([^)]*)\bdiameter2\s*=",
        r"bmesh.ops.create_cone(\1radius2 =",
        "bmesh.ops.create_cone: diameter2 → radius2 (Blender 5.0)"
    ),
    # Also fix standalone diameter references in bmesh context (less common)
    (
        r",\s*diameter1\s*=",
        r", radius1 =",
        "diameter1 → radius1 (bmesh cone)"
    ),
    (
        r",\s*diameter2\s*=",
        r", radius2 =",
        "diameter2 → radius2 (bmesh cone)"
    ),

    # =============================================================================
    # CRASH PATTERN FIXES (Blender 5.0 Cycles Segfaults)
    # =============================================================================
    # ShaderNodeMixRGB was renamed to ShaderNodeMix in Blender 3.4+.
    # Using the old name in volume shader chains causes Cycles to crash with
    # EXCEPTION_ACCESS_VIOLATION in ccl::MixNode::is_linear_operation.
    # This is the #1 cause of exit code 139 (segfault) during renders.
    (
        r"nodes\.new\(['\"]ShaderNodeMixRGB['\"]\)",
        r"nodes.new('ShaderNodeMix')  # API Fixer: ShaderNodeMixRGB → ShaderNodeMix (crash fix)",
        "ShaderNodeMixRGB → ShaderNodeMix (Cycles volume crash fix)"
    ),
    # ShaderNodeSeparateRGB → ShaderNodeSeparateColor (renamed in Blender 3.4+)
    (
        r"nodes\.new\(['\"]ShaderNodeSeparateRGB['\"]\)",
        r"nodes.new('ShaderNodeSeparateColor')  # API Fixer: SeparateRGB → SeparateColor",
        "ShaderNodeSeparateRGB → ShaderNodeSeparateColor"
    ),
    # ShaderNodeCombineRGB → ShaderNodeCombineColor (renamed in Blender 3.4+)
    (
        r"nodes\.new\(['\"]ShaderNodeCombineRGB['\"]\)",
        r"nodes.new('ShaderNodeCombineColor')  # API Fixer: CombineRGB → CombineColor",
        "ShaderNodeCombineRGB → ShaderNodeCombineColor"
    ),
    # BLENDER_EEVEE_NEXT → BLENDER_EEVEE (renamed in Blender 4.2+)
    (
        r"['\"]BLENDER_EEVEE_NEXT['\"]",
        r"'BLENDER_EEVEE'  # API Fixer: EEVEE_NEXT → EEVEE (renamed in 4.2+)",
        "BLENDER_EEVEE_NEXT → BLENDER_EEVEE"
    ),
    # ShaderNodeSeparateColor uses 'Red', 'Green', 'Blue' instead of 'R', 'G', 'B'
    (
        r"\.outputs\[['\"]R['\"]\]",
        r".outputs['Red']  # API Fixer: R → Red (ShaderNodeSeparateColor)",
        "SeparateColor output R → Red"
    ),
    (
        r"\.outputs\[['\"]G['\"]\]",
        r".outputs['Green']  # API Fixer: G → Green (ShaderNodeSeparateColor)",
        "SeparateColor output G → Green"
    ),
    (
        r"\.outputs\[['\"]B['\"]\]",
        r".outputs['Blue']  # API Fixer: B → Blue (ShaderNodeSeparateColor)",
        "SeparateColor output B → Blue"
    ),

    # =============================================================================
    # NODE LINKING DIRECTION FIX (LLM Hallucination)
    # =============================================================================
    # LLMs sometimes try to link INPUT → INPUT which is invalid.
    # nt.links.new(source, dest) requires: source = output socket, dest = input socket.
    # Pattern: nt.links.new(xxx.inputs[N], yyy.inputs[M]) is ALWAYS wrong.
    # Fix: Comment out the invalid line since we can't infer correct behavior.
    (
        r"(?m)^(\s*)((?:nt|node_tree|mat\.node_tree)\.links\.new\(\s*\w+\.inputs\[\d+\]\s*,\s*\w+\.inputs\[\d+\]\s*\))",
        r"\1# API Fixer: REMOVED - input→input link invalid (must be output→input)\n\1# \2",
        "Removed invalid input→input node link (must be output→input)"
    ),
    # Also catch named input sockets: .inputs['Name']
    (
        r"(?m)^(\s*)((?:nt|node_tree|mat\.node_tree)\.links\.new\(\s*\w+\.inputs\[['\"][^'\"]+['\"]\]\s*,\s*\w+\.inputs)",
        r"\1# API Fixer: REMOVED - input→input link invalid (must be output→input)\n\1# \2",
        "Removed invalid input→input node link (named sockets)"
    ),
    # Catch output→output links too (equally invalid)
    (
        r"(?m)^(\s*)((?:nt|node_tree|mat\.node_tree)\.links\.new\(\s*\w+\.outputs\[\d+\]\s*,\s*\w+\.outputs\[\d+\]\s*\))",
        r"\1# API Fixer: REMOVED - output→output link invalid (must be output→input)\n\1# \2",
        "Removed invalid output→output node link (must be output→input)"
    ),
]


# =============================================================================
# VOLUME MATERIAL INJECTION (P0 FIX - Grey Sphere Problem)
# =============================================================================
# Without a volume shader on the domain, Cycles renders the mesh geometry
# instead of the smoke/fire volume, resulting in a grey sphere.

VOLUME_MATERIAL_SNIPPET = '''
# ==== API FIXER: Volume Material Setup (Mantaflow) ====
# Without this, renders show grey mesh instead of smoke/fire
def _api_fixer_setup_volume_material(domain_obj, effect_type="SMOKE"):
    """Ensure domain has a volume shader for rendering smoke/fire."""
    mat_name = f"{domain_obj.name}_VolumeMaterial"

    # Check if material already exists
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(name=mat_name)
        # use_nodes is always True in Blender 5.0 — no need to set it

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Clear default nodes
        nodes.clear()

        # Create nodes based on effect type
        output = nodes.new('ShaderNodeOutputMaterial')
        output.location = (300, 0)

        if effect_type in ("FIRE", "BOTH"):
            # Fire/smoke: Principled Volume with blackbody emission
            volume = nodes.new('ShaderNodeVolumePrincipled')
            volume.location = (0, 0)
            volume.inputs['Density'].default_value = 5.0
            volume.inputs['Anisotropy'].default_value = 0.3
            volume.inputs['Blackbody Intensity'].default_value = 1.0
            volume.inputs['Blackbody Tint'].default_value = (1.0, 0.8, 0.5, 1.0)

            # Connect density and flame attributes
            attr_density = nodes.new('ShaderNodeAttribute')
            attr_density.location = (-400, 100)
            attr_density.attribute_name = 'density'
            attr_density.attribute_type = 'GEOMETRY'

            attr_flame = nodes.new('ShaderNodeAttribute')
            attr_flame.location = (-400, -100)
            attr_flame.attribute_name = 'flame'
            attr_flame.attribute_type = 'GEOMETRY'

            # Multiply density
            multiply = nodes.new('ShaderNodeMath')
            multiply.location = (-200, 100)
            multiply.operation = 'MULTIPLY'
            multiply.inputs[1].default_value = 5.0

            links.new(attr_density.outputs['Fac'], multiply.inputs[0])
            links.new(multiply.outputs['Value'], volume.inputs['Density'])
            links.new(attr_flame.outputs['Fac'], volume.inputs['Blackbody Intensity'])
            links.new(volume.outputs['Volume'], output.inputs['Volume'])

        else:
            # Smoke only: simpler volume scatter
            volume = nodes.new('ShaderNodeVolumePrincipled')
            volume.location = (0, 0)
            volume.inputs['Density'].default_value = 5.0
            volume.inputs['Anisotropy'].default_value = 0.3
            volume.inputs['Blackbody Intensity'].default_value = 0.0

            attr_density = nodes.new('ShaderNodeAttribute')
            attr_density.location = (-400, 100)
            attr_density.attribute_name = 'density'
            attr_density.attribute_type = 'GEOMETRY'

            multiply = nodes.new('ShaderNodeMath')
            multiply.location = (-200, 100)
            multiply.operation = 'MULTIPLY'
            multiply.inputs[1].default_value = 5.0

            links.new(attr_density.outputs['Fac'], multiply.inputs[0])
            links.new(multiply.outputs['Value'], volume.inputs['Density'])
            links.new(volume.outputs['Volume'], output.inputs['Volume'])

    # Assign material to domain
    if domain_obj.data.materials:
        domain_obj.data.materials[0] = mat
    else:
        domain_obj.data.materials.append(mat)

    return mat

# Find domain object and apply volume material
_api_fixer_domain = None
for obj in bpy.data.objects:
    for mod in obj.modifiers:
        if mod.type == 'FLUID' and hasattr(mod, 'fluid_type') and mod.fluid_type == 'DOMAIN':
            _api_fixer_domain = obj
            break
    if _api_fixer_domain:
        break

if _api_fixer_domain:
    # Determine effect type from flow settings
    _api_fixer_effect_type = "SMOKE"
    for obj in bpy.data.objects:
        for mod in obj.modifiers:
            if mod.type == 'FLUID' and hasattr(mod, 'fluid_type') and mod.fluid_type == 'FLOW':
                if hasattr(mod, 'flow_settings') and mod.flow_settings:
                    ft = getattr(mod.flow_settings, 'flow_type', 'SMOKE')
                    if ft in ('FIRE', 'BOTH'):
                        _api_fixer_effect_type = ft
                        break
    _api_fixer_setup_volume_material(_api_fixer_domain, _api_fixer_effect_type)
    print(f"[API Fixer] Applied volume material to {_api_fixer_domain.name} ({_api_fixer_effect_type})")
# ==== END API FIXER: Volume Material Setup ====
'''


# =============================================================================
# LIQUID MATERIAL INJECTION (S1.5-B FIX - White Water Problem)
# =============================================================================
# Liquid domains render via the generated mesh surface, not a volume shader.
# Without a material on the liquid mesh, it renders as default white —
# indistinguishable from white scene objects (cabinets, walls, etc.).

LIQUID_MATERIAL_SNIPPET = '''
# ==== API FIXER: Liquid Surface Material Setup (Mantaflow LIQUID) ====
# Without this, liquid mesh renders as default white — invisible against
# white scene objects. Assigns a water-like Principled BSDF.
def _api_fixer_setup_liquid_material(domain_obj):
    """Ensure liquid domain mesh gets a water-like surface material."""
    # Liquid mesh is generated by bake — it's the domain object itself
    # or a child mesh. Check the domain first.
    target_obj = domain_obj

    # If domain already has a user-authored water/liquid material, skip.
    if target_obj.data.materials:
        existing_mat = target_obj.data.materials[0]
        if existing_mat and existing_mat.use_nodes:
            # Check if it already has transmission or glass-like setup
            for node in existing_mat.node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    # Check Transmission Weight — if > 0.1, user set up water material
                    tw_input = node.inputs.get('Transmission Weight')
                    if tw_input and tw_input.default_value > 0.1:
                        print(f"[API Fixer] Liquid domain {domain_obj.name} already has water material — skipping")
                        return None
                elif node.type == 'BSDF_GLASS':
                    print(f"[API Fixer] Liquid domain {domain_obj.name} already has glass/water material — skipping")
                    return None

    mat_name = f"{domain_obj.name}_WaterMaterial"
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(name=mat_name)
        # use_nodes is always True in Blender 5.0 — no need to set it

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Clear default nodes
        nodes.clear()

        # Output
        output = nodes.new('ShaderNodeOutputMaterial')
        output.location = (400, 0)

        # Principled BSDF — water-like surface
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.location = (0, 0)

        # Water properties
        bsdf.inputs['Base Color'].default_value = (0.6, 0.75, 0.85, 1.0)  # Slight blue tint
        bsdf.inputs['Roughness'].default_value = 0.02  # Nearly mirror-smooth
        bsdf.inputs['IOR'].default_value = 1.333  # Water IOR
        bsdf.inputs['Transmission Weight'].default_value = 0.95  # Mostly transparent
        bsdf.inputs['Specular IOR Level'].default_value = 0.5  # Standard specular

        # Surface output
        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

        # Optional: Volume absorption for depth coloring
        absorb = nodes.new('ShaderNodeVolumeAbsorption')
        absorb.location = (0, -200)
        absorb.inputs['Color'].default_value = (0.4, 0.65, 0.8, 1.0)  # Blue-tinted absorption
        absorb.inputs['Density'].default_value = 0.3  # Subtle depth coloring
        links.new(absorb.outputs['Volume'], output.inputs['Volume'])

    # Assign material to domain object
    if target_obj.data.materials:
        # Replace first (default) material
        target_obj.data.materials[0] = mat
    else:
        target_obj.data.materials.append(mat)

    print(f"[API Fixer] Applied water material to liquid domain {domain_obj.name}")
    return mat

# Find liquid domain and apply water material
_api_fixer_liquid_domain = None
for obj in bpy.data.objects:
    for mod in obj.modifiers:
        if mod.type == 'FLUID' and hasattr(mod, 'fluid_type') and mod.fluid_type == 'DOMAIN':
            if hasattr(mod, 'domain_settings') and mod.domain_settings:
                if mod.domain_settings.domain_type == 'LIQUID':
                    _api_fixer_liquid_domain = obj
                    break
    if _api_fixer_liquid_domain:
        break

if _api_fixer_liquid_domain:
    # Also ensure mesh visualization is enabled
    if hasattr(_api_fixer_liquid_domain.modifiers[0].domain_settings, 'use_mesh'):
        _api_fixer_liquid_domain.modifiers[0].domain_settings.use_mesh = True
        print(f"[API Fixer] Enabled use_mesh on liquid domain {_api_fixer_liquid_domain.name}")
    _api_fixer_setup_liquid_material(_api_fixer_liquid_domain)
else:
    print("[API Fixer] No LIQUID domain found — liquid material injection skipped")
# ==== END API FIXER: Liquid Surface Material Setup ====
'''


ANIMATION_TO_STILLS_SNIPPET = '''
# ==== API FIXER: Animation → Representative Still Renders ====
# bpy.ops.render.render(animation=True) does NOT write individual frame files in
# headless mode. Replace with per-frame still renders of 3 representative frames.
_api_fixer_scene = bpy.context.scene
_api_fixer_fs = _api_fixer_scene.frame_start
_api_fixer_fe = _api_fixer_scene.frame_end
_api_fixer_base_path = _api_fixer_scene.render.filepath
_api_fixer_frames = [min(_api_fixer_fs + 5, _api_fixer_fe), (_api_fixer_fs + _api_fixer_fe) // 2, _api_fixer_fe]
for _api_fixer_f in _api_fixer_frames:
    _api_fixer_scene.frame_set(_api_fixer_f)
    _api_fixer_scene.render.filepath = f"{_api_fixer_base_path}{_api_fixer_f:04d}"
    bpy.ops.render.render(write_still=True)
    print(f"[API Fixer] Rendered frame {_api_fixer_f}")
_api_fixer_scene.render.filepath = _api_fixer_base_path
# ==== END API FIXER: Animation → Representative Still Renders ====
'''


BAKE_FRAME_ALIGNMENT_SNIPPET = '''
# ==== API FIXER: Bake Frame Range Alignment ====
# Mantaflow cache_frame_start/cache_frame_end default to 1-120 independent of scene.frame_end.
# Mismatched ranges waste bake time or cause missing frames.
_api_fixer_scene = bpy.context.scene
for _api_fixer_obj in bpy.data.objects:
    for _api_fixer_mod in _api_fixer_obj.modifiers:
        if _api_fixer_mod.type == 'FLUID' and hasattr(_api_fixer_mod, 'fluid_type') and _api_fixer_mod.fluid_type == 'DOMAIN':
            _ds = _api_fixer_mod.domain_settings
            _ds.cache_frame_start = _api_fixer_scene.frame_start
            _ds.cache_frame_end = _api_fixer_scene.frame_end
            print(f"[API Fixer] Aligned bake range: {_ds.cache_frame_start}-{_ds.cache_frame_end} (scene: {_api_fixer_scene.frame_start}-{_api_fixer_scene.frame_end})")
            break
# ==== END API FIXER: Bake Frame Range Alignment ====
'''


CAMERA_DISTANCE_FIX_SNIPPET = '''
# ==== API FIXER: Camera Distance Validation ====
# Validates camera is not inside SUBJECT geometry (domains, small objects).
# Excludes room-scale geometry (walls, floors, ceilings) from bounding box —
# a camera INSIDE a room is normal for interior shots, not an error.
# Also fixes scale=(0,0,0) corruption from matrix_world look_at() functions.
import math as _cam_math
for _cam_obj in bpy.data.objects:
    if _cam_obj.type != 'CAMERA':
        continue
    # Fix scale first (matrix_world corruption causes black renders)
    _cam_obj.scale = (1.0, 1.0, 1.0)
    # Collect visible mesh objects, separating subject from environment
    _all_mesh = [o for o in bpy.data.objects if o.type == 'MESH' and o.visible_get()]
    if not _all_mesh:
        continue
    from mathutils import Vector as _V
    # Filter out room-scale geometry: objects with any world-space dimension > 2.5m
    # are likely walls, floors, ceilings — not subjects the camera should avoid
    _env_names = {'floor', 'wall', 'ceiling', 'ground', 'room', 'backdrop'}
    _subject_objs = []
    for _mo in _all_mesh:
        _bb = [_mo.matrix_world @ _V(_c) for _c in _mo.bound_box]
        _omin = _V((min(v[i] for v in _bb) for i in range(3)))
        _omax = _V((max(v[i] for v in _bb) for i in range(3)))
        _dims = _omax - _omin
        _max_dim = max(_dims[0], _dims[1], _dims[2])
        _is_env = _max_dim > 2.5 or any(n in _mo.name.lower() for n in _env_names)
        if not _is_env:
            _subject_objs.append(_mo)
    # Use subject objects for bounding box; fall back to all if none qualify
    _target_objs = _subject_objs if _subject_objs else _all_mesh
    _smin = _V((1e9, 1e9, 1e9))
    _smax = _V((-1e9, -1e9, -1e9))
    for _mo in _target_objs:
        for _c in _mo.bound_box:
            _wc = _mo.matrix_world @ _V(_c)
            _smin = _V((min(_smin[i], _wc[i]) for i in range(3)))
            _smax = _V((max(_smax[i], _wc[i]) for i in range(3)))
    _center = (_smin + _smax) / 2
    _diag = (_smax - _smin).length
    if _diag < 0.01:
        continue
    # Compute minimum distance from focal length and SUBJECT size (not room size)
    _focal = _cam_obj.data.lens
    _sensor = _cam_obj.data.sensor_width
    _hfov = 2 * _cam_math.atan(_sensor / (2 * _focal))
    _min_dist = max((_diag / (2 * _cam_math.tan(_hfov / 2))) * 0.6, 0.3)
    # Only reposition if camera is inside the SUBJECT bounding box or very close
    _loc = _cam_obj.location.copy()
    _inside = all(_smin[i] - 0.05 <= _loc[i] <= _smax[i] + 0.05 for i in range(3))
    _dist = (_loc - _center).length
    if _inside or _dist < _min_dist:
        _dir = _loc - _center
        if _dir.length < 0.01:
            _dir = _V((0, -1, 0.3))
        _dir.normalize()
        _new_dist = max(_min_dist * 1.5, 0.5)
        _cam_obj.location = _center + _dir * _new_dist
        _look = (_center - _cam_obj.location).normalized()
        _cam_obj.rotation_euler = _look.to_track_quat('-Z', 'Y').to_euler()
        print(f"[API Fixer] Camera repositioned: {_loc} -> {_cam_obj.location} (was {'inside' if _inside else 'too close'}, min_dist={_min_dist:.2f})")
    else:
        print(f"[API Fixer] Camera OK: dist={_dist:.2f}, min_dist={_min_dist:.2f}, subject_diag={_diag:.2f}")
# ==== END API FIXER: Camera Distance Validation ====
'''


BAKE_BEFORE_RENDER_SNIPPET = '''
# ==== API FIXER: Bake Fluid Simulation Before Rendering ====
# Without baking, Mantaflow shows green FLIP particles instead of fluid mesh.
# Must run bpy.ops.fluid.bake_all() BEFORE any render calls.
_api_fixer_domain_obj = None
for _api_fixer_obj in bpy.data.objects:
    for _api_fixer_mod in _api_fixer_obj.modifiers:
        if _api_fixer_mod.type == 'FLUID' and hasattr(_api_fixer_mod, 'fluid_type') and _api_fixer_mod.fluid_type == 'DOMAIN':
            _api_fixer_domain_obj = _api_fixer_obj
            break
    if _api_fixer_domain_obj:
        break

if _api_fixer_domain_obj:
    # Ensure domain is selected for bake operator
    bpy.ops.object.select_all(action='DESELECT')
    _api_fixer_domain_obj.select_set(True)
    bpy.context.view_layer.objects.active = _api_fixer_domain_obj

    # Align cache frame range with scene
    _api_fixer_ds = _api_fixer_domain_obj.modifiers['Fluid'].domain_settings
    _api_fixer_ds.cache_frame_start = bpy.context.scene.frame_start
    _api_fixer_ds.cache_frame_end = bpy.context.scene.frame_end

    # Bake all fluid data
    print(f"[API Fixer] Baking fluid simulation for frames {_api_fixer_ds.cache_frame_start}-{_api_fixer_ds.cache_frame_end}...")
    bpy.ops.fluid.bake_all()
    print("[API Fixer] Fluid bake complete.")
else:
    print("[API Fixer] No fluid domain found - skipping bake.")
# ==== END API FIXER: Bake Fluid Simulation ====
'''


def _fix_mix_node_sockets(content: str) -> tuple[str, bool]:
    """
    Fix ShaderNodeMix socket names after MixRGB→Mix conversion.

    ShaderNodeMixRGB used: inputs['Fac'], inputs['Color1'], inputs['Color2'], outputs['Color']
    ShaderNodeMix uses:    inputs['Factor'], inputs['A'], inputs['B'], outputs['Result']

    Also injects data_type='RGBA' since MixRGB was always color mode.

    Returns:
        Tuple of (modified_content, was_modified)
    """
    # Only run if we converted MixRGB → Mix
    if "ShaderNodeMix" not in content:
        return content, False

    # Find all variable names assigned as ShaderNodeMix
    # Match any node tree variable (nodes, mat_nodes, copper_nodes, etc.)
    mix_vars = set()
    for match in re.finditer(
        r"(\w+)\s*=\s*\w+\.new\(['\"]ShaderNodeMix['\"]\)",
        content
    ):
        mix_vars.add(match.group(1))

    if not mix_vars:
        return content, False

    modified = False
    for var in mix_vars:
        escaped = re.escape(var)

        # Inject data_type='RGBA' after EVERY node creation line for this var
        # (scripts may create multiple mix nodes with the same variable name in different functions)
        dtype_pattern = rf"([ \t]*{escaped}\s*=\s*\w+\.new\(['\"]ShaderNodeMix['\"]\)[^\n]*\n)"
        if re.search(dtype_pattern, content):
            def _add_dtype(m):
                # Match the indent of the creation line
                line_text = m.group(1)
                indent = re.match(r'^(\s*)', line_text).group(1)
                return line_text + f"{indent}{var}.data_type = 'RGBA'  # MixRGB was always color mode\n"
            # Replace ALL occurrences (count=0), not just the first
            new_content = re.sub(dtype_pattern, _add_dtype, content)
            if new_content != content:
                content = new_content
                modified = True

        # Fac → Factor
        pattern_fac = rf"({escaped}\.inputs\[)['\"]Fac['\"]\]"
        if re.search(pattern_fac, content):
            content = re.sub(pattern_fac, r"\g<1>'Factor']", content)
            modified = True
        # Color1 → A
        pattern_c1 = rf"({escaped}\.inputs\[)['\"]Color1['\"]\]"
        if re.search(pattern_c1, content):
            content = re.sub(pattern_c1, r"\g<1>'A']", content)
            modified = True
        # Color2 → B
        pattern_c2 = rf"({escaped}\.inputs\[)['\"]Color2['\"]\]"
        if re.search(pattern_c2, content):
            content = re.sub(pattern_c2, r"\g<1>'B']", content)
            modified = True
        # outputs['Color'] → outputs['Result']
        pattern_out = rf"({escaped}\.outputs\[)['\"]Color['\"]\]"
        if re.search(pattern_out, content):
            content = re.sub(pattern_out, r"\g<1>'Result']", content)
            modified = True

    return content, modified


def _inject_animation_to_stills(content: str) -> tuple[str, bool]:
    """
    Replace bpy.ops.render.render(animation=True) with per-frame still renders.

    In headless Blender, animation=True renders frames internally but does NOT
    write individual image files that the pipeline can find for quality evaluation.
    This replaces the call with a loop that renders 3 representative frames using
    write_still=True.

    Args:
        content: Script content

    Returns:
        Tuple of (modified_content, was_modified)
    """
    # Match render(animation=True) with optional extra kwargs
    animation_match = re.search(
        r"^(\s*)bpy\.ops\.render\.render\([^)]*animation\s*=\s*True[^)]*\).*$",
        content,
        re.MULTILINE
    )

    if animation_match and "API FIXER: Animation" not in content:
        indent = animation_match.group(1)
        indented_snippet = "\n".join(
            indent + line if line.strip() else line
            for line in ANIMATION_TO_STILLS_SNIPPET.split("\n")
        )
        content = (
            content[:animation_match.start()]
            + indented_snippet
            + content[animation_match.end():]
        )
        return content, True

    return content, False


def _inject_bake_frame_alignment(content: str) -> tuple[str, bool]:
    """
    Inject bake frame range alignment if script uses Mantaflow and baking.

    Ensures cache_frame_start/cache_frame_end match scene.frame_start/frame_end
    so the bake doesn't waste time on 120 default frames when scene is 25 frames.

    Args:
        content: Script content

    Returns:
        Tuple of (modified_content, was_modified)
    """
    # Check if script uses Mantaflow domain (more robust pattern matching)
    # Matches: fluid_type = 'DOMAIN', fluid_type == 'DOMAIN', getattr(...) == 'DOMAIN', etc.
    has_fluid_domain = re.search(
        r"fluid_type\s*[=!]=?\s*['\"]DOMAIN['\"]|"
        r"type\s*=\s*['\"]FLUID['\"]|"
        r"FLUID\s+DOMAIN|"
        r"domain_settings",
        content,
        re.IGNORECASE
    ) is not None

    # Check if script has a bake call
    has_bake = re.search(
        r"bpy\.ops\.fluid\.bake_data|bpy\.ops\.fluid\.bake_all|bpy\.ops\.fluid\.bake_noise",
        content
    ) is not None

    # Check if already aligned
    already_aligned = re.search(
        r"cache_frame_start|cache_frame_end|Bake Frame Range Alignment",
        content
    ) is not None

    if has_fluid_domain and has_bake and not already_aligned:
        # Inject before the first bake call
        bake_match = re.search(
            r"^(\s*)(bpy\.ops\.fluid\.bake_data|bpy\.ops\.fluid\.bake_all)",
            content,
            re.MULTILINE
        )
        if bake_match:
            insert_pos = bake_match.start()
            indent = bake_match.group(1)
            indented_snippet = "\n".join(
                indent + line if line.strip() else line
                for line in BAKE_FRAME_ALIGNMENT_SNIPPET.split("\n")
            )
            content = content[:insert_pos] + indented_snippet + "\n\n" + content[insert_pos:]
            return content, True

    return content, False


def _inject_bake_before_render(content: str) -> tuple[str, bool]:
    """
    Inject fluid bake call if script has fluid domain but no bake operation.

    Without baking, Mantaflow shows green FLIP particles (debug visualization)
    instead of the actual fluid mesh/surface.

    Args:
        content: Script content

    Returns:
        Tuple of (modified_content, was_modified)
    """
    # Check if script has a fluid domain (robust pattern matching)
    has_fluid_domain = re.search(
        r"fluid_type\s*[=!]=?\s*['\"]DOMAIN['\"]|"
        r"type\s*=\s*['\"]FLUID['\"]|"
        r"FLUID\s+DOMAIN|"
        r"domain_settings",
        content,
        re.IGNORECASE
    ) is not None

    # Check if script already has bake call
    has_bake = re.search(
        r"bpy\.ops\.fluid\.bake_data|bpy\.ops\.fluid\.bake_all|bpy\.ops\.fluid\.bake_noise",
        content
    ) is not None

    # Check if already injected
    already_injected = "API FIXER: Bake Fluid Simulation" in content

    # Check if there's a render call
    has_render = re.search(r"bpy\.ops\.render\.render", content) is not None

    if has_fluid_domain and not has_bake and not already_injected and has_render:
        # Find the render section - inject before it
        render_match = re.search(
            r"^(\s*)(# =+\s*Render|# Render|for\s+\w+\s+in\s+.*frame|bpy\.ops\.render\.render)",
            content,
            re.MULTILINE
        )

        if render_match:
            insert_pos = render_match.start()
            indent = render_match.group(1)
            indented_snippet = "\n".join(
                indent + line if line.strip() else line
                for line in BAKE_BEFORE_RENDER_SNIPPET.split("\n")
            )
            content = content[:insert_pos] + indented_snippet + "\n\n" + content[insert_pos:]
            return content, True

    return content, False


def _inject_volume_material_setup(content: str) -> tuple[str, bool]:
    """
    Inject volume material setup if script uses Mantaflow but lacks volume shader.

    Args:
        content: Script content

    Returns:
        Tuple of (modified_content, was_modified)
    """
    # Check if script uses Mantaflow domain (robust pattern matching)
    has_fluid_domain = re.search(
        r"fluid_type\s*[=!]=?\s*['\"]DOMAIN['\"]|"
        r"type\s*=\s*['\"]FLUID['\"]|"
        r"FLUID\s+DOMAIN|"
        r"domain_settings",
        content,
        re.IGNORECASE
    ) is not None

    # Check if domain is LIQUID — liquid domains render via mesh surface, not volume shader.
    # Volume material injection would overwrite the correct water/glass material.
    # Match both direct assignment (ds.domain_type = 'LIQUID') and safe_set patterns
    # (safe_set(ds, 'domain_type', 'LIQUID', ...)).
    is_liquid_domain = re.search(
        r"domain_type\s*[=,]\s*['\"]LIQUID['\"]|"
        r"['\"]domain_type['\"]\s*,\s*['\"]LIQUID['\"]|"
        r"Domain\s*Type['\"]?\s*,\s*['\"]Liquid['\"]|"
        r"set_enum_by_predicate\([^)]*['\"]LIQUID['\"]|"
        r"domain_settings[^)]*['\"]LIQUID['\"]",
        content,
        re.IGNORECASE
    ) is not None

    # Check if script already has volume material setup
    has_volume_material = re.search(
        r"ShaderNodeVolumePrincipled|ShaderNodeVolumeScatter|ShaderNodeVolumeAbsorption|"
        r"Volume\s*Scatter|Volume\s*Absorption|VolumeMaterial|_setup_volume_material",
        content,
        re.IGNORECASE
    ) is not None

    # Check if there's a render call (otherwise no point adding material)
    has_render = re.search(r"bpy\.ops\.render\.render", content) is not None

    if has_fluid_domain and not is_liquid_domain and not has_volume_material and has_render:
        # Find injection point: after bpy.ops.fluid.bake or before bpy.ops.render.render
        # Prefer injecting before render call
        render_match = re.search(
            r"^(\s*)(scene\.render\.filepath\s*=|bpy\.ops\.render\.render)",
            content,
            re.MULTILINE
        )

        if render_match:
            insert_pos = render_match.start()
            indent = render_match.group(1)
            # Indent the snippet
            indented_snippet = "\n".join(
                indent + line if line.strip() else line
                for line in VOLUME_MATERIAL_SNIPPET.split("\n")
            )
            content = content[:insert_pos] + indented_snippet + "\n\n" + content[insert_pos:]
            return content, True
        else:
            # Fallback: append at the end before any final print statements
            content = content.rstrip() + "\n\n" + VOLUME_MATERIAL_SNIPPET
            return content, True

    return content, False


def _inject_liquid_material_setup(content: str) -> tuple[str, bool]:
    """
    Inject water-like surface material for LIQUID domain scripts.

    Liquid domains render via the generated mesh surface, not a volume shader.
    Without a material, the liquid mesh renders as default white — invisible
    against white scene objects. This injects a Principled BSDF with water
    properties (IOR 1.333, high transmission, low roughness).

    Skips injection if:
    - Script is not a LIQUID domain script
    - Script already has a water/glass material setup
    - No render call exists
    """
    # Only inject for LIQUID domain scripts
    is_liquid_domain = re.search(
        r"domain_type\s*[=,]\s*['\"]LIQUID['\"]|"
        r"['\"]domain_type['\"]\s*,\s*['\"]LIQUID['\"]|"
        r"set_enum_by_predicate\([^)]*['\"]LIQUID['\"]|"
        r"domain_settings[^)]*['\"]LIQUID['\"]",
        content,
        re.IGNORECASE
    ) is not None

    if not is_liquid_domain:
        return content, False

    # Check if script already has a water/glass/transmission material setup
    has_water_material = re.search(
        r"Transmission\s*Weight|"
        r"BSDF_GLASS|BsdfGlass|"
        r"WaterMaterial|water_material|liquid_material|"
        r"_api_fixer_setup_liquid_material|"
        r"IOR.*1\.33",
        content,
        re.IGNORECASE
    ) is not None

    if has_water_material:
        return content, False

    # Check if there's a render call
    has_render = re.search(r"bpy\.ops\.render\.render", content) is not None
    if not has_render:
        return content, False

    # Find injection point: before bpy.ops.render.render
    render_match = re.search(
        r"^(\s*)(scene\.render\.filepath\s*=|bpy\.ops\.render\.render)",
        content,
        re.MULTILINE
    )

    if render_match:
        insert_pos = render_match.start()
        indent = render_match.group(1)
        indented_snippet = "\n".join(
            indent + line if line.strip() else line
            for line in LIQUID_MATERIAL_SNIPPET.split("\n")
        )
        content = content[:insert_pos] + indented_snippet + "\n\n" + content[insert_pos:]
        return content, True
    else:
        # Fallback: append at end
        content = content.rstrip() + "\n\n" + LIQUID_MATERIAL_SNIPPET
        return content, True


def _inject_plane_init_for_liquid_flow(content: str) -> tuple[str, bool]:
    """
    Ensure liquid emitters use plane initialization.

    For liquid inflows using a plane emitter, Blender expects
    `use_plane_init=True` to emit volume correctly. Without it, bakes can
    complete but produce near-empty cache data.
    """
    # Skip if already set.
    if re.search(r"use_plane_init\s*=\s*(True|False)", content):
        return content, False

    # Find the liquid flow_type assignment and insert plane init after it.
    match = re.search(
        r"^(\s*)(\w+)\.flow_type\s*=\s*['\"]LIQUID['\"]\s*$",
        content,
        re.MULTILINE
    )
    if not match:
        return content, False

    indent = match.group(1)
    var_name = match.group(2)
    insert = f"\n{indent}{var_name}.use_plane_init = True  # Planar liquid emitter\n"
    content = content[:match.end()] + insert + content[match.end():]
    return content, True


def _fix_particle_radius_value(content: str) -> tuple[str, bool]:
    """
    Fix absurdly small particle_radius values on Mantaflow liquid domains.

    The LLM often hallucinates real-world units (e.g. 0.0025 for 2.5mm) when
    particle_radius is measured in cell sizes (default 1.0, range 0-10).
    Values < 0.1 produce empty mesh output. Fix to 1.0 (default).
    """
    # Match: safe_set(ds, 'particle_radius', 0.0025, ...) or ds.particle_radius = 0.0025
    pattern = r"((?:safe_set\s*\([^,]+,\s*['\"]particle_radius['\"],\s*)(\d*\.?\d+)|(\.particle_radius\s*=\s*)(\d*\.?\d+))"
    modified = False

    def _check_and_fix(m):
        nonlocal modified
        full = m.group(0)
        # Extract numeric value from either pattern
        val_str = m.group(2) or m.group(4)
        try:
            val = float(val_str)
        except (TypeError, ValueError):
            return full
        if val < 0.1:
            modified = True
            return full.replace(val_str, "1.0")
        return full

    new_content = re.sub(pattern, _check_and_fix, content)
    if modified:
        # Add a comment explaining the fix
        new_content = new_content.replace(
            "particle_radius", "particle_radius"  # No-op to avoid double comments
        )
    return new_content, modified


def _fix_cube_scale_halving(content: str) -> tuple[str, bool]:
    """
    Fix the systematic '* 0.5' error on obj.scale after primitive_cube_add(size=1.0).

    LLMs are deeply trained to treat scale as half-extents, systematically writing:
        obj.scale = (W * 0.5, D * 0.5, H * 0.5)
    when the correct code is:
        obj.scale = (W, D, H)

    With primitive_cube_add(size=1.0), vertices are at ±0.5, and obj.scale=(s,s,s)
    gives visual dimensions (s, s, s), NOT half-extents. The * 0.5 halves everything,
    creating gaps between objects ("exploded geometry") and emitters landing outside
    undersized domains.

    This fixer strips '* 0.5' from scale tuples where ALL three components are halved.
    It preserves location math (where * 0.5 centering is often correct).
    """
    modified = False

    # Match: .scale = (expr1 * 0.5, expr2 * 0.5, expr3 * 0.5)
    # Only fix when ALL three components have * 0.5 (the systematic halving pattern)
    # Each expr can be: a variable, a number, or a sub-expression like (A + B)
    scale_pattern = re.compile(
        r'(\.scale\s*=\s*\()'           # .scale = (
        r'([^,]+?)\s*\*\s*0\.5'          # expr1 * 0.5
        r'\s*,\s*'                        # ,
        r'([^,]+?)\s*\*\s*0\.5'          # expr2 * 0.5
        r'\s*,\s*'                        # ,
        r'([^)]+?)\s*\*\s*0\.5'          # expr3 * 0.5
        r'(\s*\))'                        # )
    )

    def _strip_halving(m):
        nonlocal modified
        prefix = m.group(1)     # .scale = (
        expr1 = m.group(2).strip()
        expr2 = m.group(3).strip()
        expr3 = m.group(4).strip()
        suffix = m.group(5)     # )
        modified = True
        return f"{prefix}{expr1}, {expr2}, {expr3}{suffix}"

    new_content = scale_pattern.sub(_strip_halving, content)
    return new_content, modified


def _inject_camera_distance_fix(content: str) -> tuple[str, bool]:
    """
    Inject camera distance validation before render calls.

    Validates that the camera is not inside the scene bounding box or too close
    for its focal length. Also fixes scale=(0,0,0) corruption from matrix_world
    look_at() functions (superset of the old camera scale fix).

    Args:
        content: Script content

    Returns:
        Tuple of (modified_content, was_modified)
    """
    # Check if script has a render call
    has_render = re.search(r"bpy\.ops\.render\.render", content) is not None

    # Check if script has a camera setup
    has_camera = re.search(
        r"(camera_add|bpy\.data\.cameras\.new|Camera_Data)",
        content
    ) is not None

    # Check if already has either version of the camera fix
    already_fixed = (
        "Camera Distance Validation" in content
        or "Camera Scale Normalization" in content
    )

    if has_render and has_camera and not already_fixed:
        # Find the FIRST actual render call (not comment-like "# Render engine")
        # Priority: bpy.ops.render.render > section header > comment
        render_match = None
        # 1st priority: actual render API call
        render_match = re.search(
            r"^(\s*)(bpy\.ops\.render\.render)",
            content,
            re.MULTILINE
        )
        # 2nd priority: render section header with === dividers
        if not render_match:
            render_match = re.search(
                r"^(\s*)(# =+\s*Render\s*=+)",
                content,
                re.MULTILINE
            )

        if render_match:
            insert_pos = render_match.start()
            indent = render_match.group(1)
            indented_snippet = "\n".join(
                indent + line if line.strip() else line
                for line in CAMERA_DISTANCE_FIX_SNIPPET.split("\n")
            )
            content = content[:insert_pos] + indented_snippet + "\n\n" + content[insert_pos:]
            return content, True

    return content, False


# =============================================================================
# VECTOR STORE VALIDATION (Option B - Runtime API Verification)
# =============================================================================
# This validates ALL bpy.*/bmesh.* API calls against the Blender 5.0 vector
# store to catch hallucinations not covered by the static BLENDER_50_FIXES list.

# Enable/disable vector store validation
VECTOR_STORE_VALIDATION_ENABLED = os.getenv("API_FIXER_VECTOR_VALIDATION", "true").lower() in ("1", "true", "yes")

# Cache for vector store lookups to avoid repeated queries
_api_signature_cache: Dict[str, Optional[Dict]] = {}


def _get_semantic_search():
    """Lazy import semantic search to avoid circular imports."""
    try:
        from tools.semantic_docs_tools import semantic_search_impl
        return semantic_search_impl
    except ImportError:
        return None


def _extract_api_calls(content: str) -> List[Tuple[str, str, str, int]]:
    """
    Extract all bmesh.ops.* and bpy.ops.* function calls with their parameters.

    Args:
        content: Script content

    Returns:
        List of tuples: (full_call, module, function_name, line_number)
        e.g., ("bmesh.ops.create_cone", "bmesh.ops", "create_cone", 225)
    """
    calls = []

    # Pattern for bmesh.ops.function_name( ... )
    bmesh_pattern = r'(bmesh\.ops)\.(\w+)\s*\('
    for match in re.finditer(bmesh_pattern, content):
        full_call = f"{match.group(1)}.{match.group(2)}"
        line_num = content[:match.start()].count('\n') + 1
        calls.append((full_call, match.group(1), match.group(2), line_num))

    # Pattern for bpy.ops.module.function_name( ... )
    bpy_ops_pattern = r'(bpy\.ops)\.(\w+)\.(\w+)\s*\('
    for match in re.finditer(bpy_ops_pattern, content):
        full_call = f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
        line_num = content[:match.start()].count('\n') + 1
        calls.append((full_call, match.group(1), f"{match.group(2)}.{match.group(3)}", line_num))

    return calls


def _extract_call_params(content: str, call_pattern: str) -> List[Tuple[str, str, int]]:
    """
    Extract parameter names from a specific API call in the content.

    Args:
        content: Script content
        call_pattern: The API call to find (e.g., "bmesh.ops.create_cone")

    Returns:
        List of tuples: (param_name, param_value, line_number)
    """
    params = []

    # Find the call and extract everything until the closing paren
    # This is complex because calls can span multiple lines
    escaped_pattern = re.escape(call_pattern)
    call_match = re.search(rf'{escaped_pattern}\s*\(([^)]+)\)', content, re.DOTALL)

    if call_match:
        param_str = call_match.group(1)
        line_num = content[:call_match.start()].count('\n') + 1

        # Extract keyword arguments (name=value)
        kwarg_pattern = r'(\w+)\s*='
        for match in re.finditer(kwarg_pattern, param_str):
            params.append((match.group(1), "", line_num))

    return params


def _query_vector_store_for_signature(api_call: str) -> Optional[Dict]:
    """
    Query the vector store to get the correct function signature.

    Args:
        api_call: Full API call (e.g., "bmesh.ops.create_cone")

    Returns:
        Dict with signature info or None if not found
    """
    # Check cache first
    if api_call in _api_signature_cache:
        return _api_signature_cache[api_call]

    semantic_search = _get_semantic_search()
    if semantic_search is None:
        return None

    try:
        # Build a more specific query for better vector store matches
        # Vector store semantic search works better with descriptive terms
        func_name = api_call.split(".")[-1]

        # For bmesh.ops, include common parameter-like terms to improve match
        if api_call.startswith("bmesh.ops"):
            # These terms help find the actual API doc rather than index pages
            query = f"{api_call} bm {func_name}"
        elif api_call.startswith("bpy.ops"):
            query = f"{api_call} operator"
        else:
            query = api_call

        # Query the vector store with enhanced query
        result_str = semantic_search(query, 5)
        result = json.loads(result_str)

        # If first query doesn't find exact match, try fallback query
        found_exact = any(api_call in item.get("content", "") for item in result.get("results", []))
        if not found_exact:
            # Try alternative query with different context
            alt_query = f"{api_call} Parameters"
            result_str = semantic_search(alt_query, 5)
            result = json.loads(result_str)

        if not result.get("results"):
            _api_signature_cache[api_call] = None
            return None

        # Look for exact match in results
        for item in result["results"]:
            doc_content = item.get("content", "")

            # Check if this result is for the exact function we're looking for
            if api_call in doc_content:
                # Extract the function signature from the doc
                # Pattern: function_name(param1=default, param2=default, ...)
                func_name = api_call.split(".")[-1]
                sig_pattern = rf'{re.escape(func_name)}\s*\(([^)]+)\)'
                sig_match = re.search(sig_pattern, doc_content)

                if sig_match:
                    sig_str = sig_match.group(1)
                    # Extract parameter names
                    params = []
                    for param in sig_str.split(","):
                        param = param.strip()
                        if "=" in param:
                            param_name = param.split("=")[0].strip()
                            params.append(param_name)
                        elif param and not param.startswith("*"):
                            params.append(param.strip())

                    signature_info = {
                        "api_call": api_call,
                        "parameters": params,
                        "raw_signature": sig_match.group(0),
                        "doc_content": doc_content[:500]
                    }
                    _api_signature_cache[api_call] = signature_info
                    return signature_info

        _api_signature_cache[api_call] = None
        return None

    except Exception as e:
        print(f"[API Fixer] Vector store query failed for {api_call}: {e}", file=sys.stderr)
        _api_signature_cache[api_call] = None
        return None


def _generate_param_fix(wrong_param: str, correct_params: List[str]) -> Optional[Tuple[str, str]]:
    """
    Generate a fix for a wrong parameter name.

    Args:
        wrong_param: The parameter name used in the script
        correct_params: List of correct parameter names from the API

    Returns:
        Tuple of (wrong_param, correct_param) or None if no fix found
    """
    # Common parameter renames (add more as discovered)
    PARAM_RENAMES = {
        "diameter1": "radius1",
        "diameter2": "radius2",
        "diameter": "radius",
        "subdivisions": "segments",
        "resolution_divisions": "resolution_max",
    }

    # Check known renames
    if wrong_param in PARAM_RENAMES:
        correct = PARAM_RENAMES[wrong_param]
        if correct in correct_params:
            return (wrong_param, correct)

    # Try fuzzy matching (simple prefix/suffix matching)
    for correct_param in correct_params:
        # Check if wrong param is a substring of correct param
        if wrong_param in correct_param or correct_param in wrong_param:
            return (wrong_param, correct_param)

    return None


def _validate_and_fix_api_calls(content: str) -> Tuple[str, List[str]]:
    """
    Validate ALL Blender API calls against the vector store and fix mismatches.

    This is the main entry point for Option B validation.

    Args:
        content: Script content

    Returns:
        Tuple of (modified_content, list_of_fixes_applied)
    """
    if not VECTOR_STORE_VALIDATION_ENABLED:
        return content, []

    fixes_applied = []

    # Extract all API calls
    api_calls = _extract_api_calls(content)

    # Track which APIs we've already validated to avoid duplicates
    validated_apis: Set[str] = set()

    for full_call, module, func_name, line_num in api_calls:
        if full_call in validated_apis:
            continue
        validated_apis.add(full_call)

        # Skip common, stable APIs that don't need validation
        SKIP_APIS = {
            "bpy.ops.object.select_all",
            "bpy.ops.object.delete",
            "bpy.ops.object.camera_add",
            "bpy.ops.object.light_add",
            "bpy.ops.mesh.primitive_cube_add",
            "bpy.ops.mesh.primitive_plane_add",
            "bpy.ops.mesh.primitive_uv_sphere_add",
            "bpy.ops.mesh.primitive_cylinder_add",
            "bpy.ops.render.render",
            "bpy.ops.wm.save_as_mainfile",
            "bpy.ops.fluid.bake_all",
            "bpy.ops.fluid.bake_data",
        }
        if full_call in SKIP_APIS:
            continue

        # Query vector store for correct signature
        signature = _query_vector_store_for_signature(full_call)

        if signature is None:
            # API not found in vector store - could be valid or could be hallucinated
            # For now, log a warning but don't block
            continue

        # Extract parameters used in the script for this call
        script_params = _extract_call_params(content, full_call)
        correct_params = signature.get("parameters", [])

        # Check each parameter
        for param_name, _, param_line in script_params:
            if param_name not in correct_params:
                # Parameter not in expected list - try to find a fix
                fix = _generate_param_fix(param_name, correct_params)

                if fix:
                    wrong, correct = fix
                    # Apply the fix to the content
                    # Use word boundary and capture whitespace to preserve formatting
                    pattern = rf'\b{re.escape(wrong)}(\s*)='
                    replacement = rf'{correct}\1='

                    if re.search(pattern, content):
                        content = re.sub(pattern, replacement, content)
                        fixes_applied.append(
                            f"[Vector Store] {full_call}: {wrong} → {correct} "
                            f"(line ~{param_line})"
                        )
                else:
                    # Unknown parameter - log warning
                    print(
                        f"[API Fixer] WARNING: Unknown parameter '{param_name}' in {full_call} "
                        f"(line ~{param_line}). Expected: {correct_params}",
                        file=sys.stderr
                    )

    return content, fixes_applied


def validate_and_fix_script(script_path: str) -> Dict:
    """
    Validate script for known-bad Blender 5.0 API patterns and auto-fix.

    Args:
        script_path: Path to the Blender Python script

    Returns:
        Dict with:
            - fixed: bool - whether any fixes were applied
            - fixes_applied: List[str] - descriptions of fixes
            - original_path: str - original script path
            - fixed_path: str - path to fixed script (same if no fixes)
    """
    path = Path(script_path)
    if not path.exists():
        return {
            "fixed": False,
            "fixes_applied": [],
            "original_path": str(path),
            "fixed_path": str(path),
            "error": f"Script not found: {path}"
        }

    content = path.read_text()
    original_content = content
    fixes_applied = []

    # Fix mixed tabs/spaces globally (common LLM issue)
    if '\t' in content:
        content = content.replace('\t', '    ')
        fixes_applied.append("Replaced tabs with spaces")

    for pattern, replacement, description in BLENDER_50_FIXES:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            fixes_applied.append(description)

    # Fix ShaderNodeMix socket names (Color1→A, Color2→B) after MixRGB conversion
    content, mix_sockets_fixed = _fix_mix_node_sockets(content)
    if mix_sockets_fixed:
        fixes_applied.append("Fixed ShaderNodeMix socket names (Color1→A, Color2→B)")

    # VECTOR STORE VALIDATION (Option B): Validate ALL API calls against Blender 5.0 docs
    # This catches hallucinated parameters not in the static BLENDER_50_FIXES list
    content, vector_store_fixes = _validate_and_fix_api_calls(content)
    fixes_applied.extend(vector_store_fixes)

    # Camera safety fix: add camera if rendering but no camera setup exists
    has_render_call = re.search(r"bpy\.ops\.render\.render", content) is not None
    has_camera_setup = re.search(
        r"(scene\.camera|bpy\.context\.scene\.camera|camera_add)",
        content
    ) is not None

    if has_render_call and not has_camera_setup:
        camera_snippet = (
            "\n# API Fixer: Ensure camera exists for rendering\n"
            "scene = bpy.context.scene\n"
            "if scene.camera is None:\n"
            "    bpy.ops.object.camera_add(location=(6, -6, 4))\n"
            "    cam = bpy.context.active_object\n"
            "    cam.rotation_euler = (1.1, 0, 0.8)\n"
            "    scene.camera = cam\n"
        )

        insert_pos = None
        scene_match = re.search(r"^scene\s*=\s*bpy\.context\.scene.*$", content, re.MULTILINE)
        if scene_match:
            insert_pos = scene_match.end()
        else:
            import_match = re.search(r"^import bpy.*$", content, re.MULTILINE)
            if import_match:
                insert_pos = import_match.end()

        if insert_pos is not None:
            content = content[:insert_pos] + camera_snippet + content[insert_pos:]
        else:
            content = camera_snippet + "\n" + content

        fixes_applied.append("Added camera setup before render (missing scene.camera)")

    # P0 FIX: Inject volume material for Mantaflow domains (fixes grey sphere)
    content, volume_fixed = _inject_volume_material_setup(content)
    if volume_fixed:
        fixes_applied.append("Injected volume material for Mantaflow domain (fixes grey sphere)")

    # S1.5-B FIX: Inject water material for LIQUID domain scripts (fixes white water)
    content, liquid_mat_fixed = _inject_liquid_material_setup(content)
    if liquid_mat_fixed:
        fixes_applied.append("Injected water material for liquid domain (fixes white-on-white water)")

    # P0 FIX: Inject fluid bake call if missing (fixes green FLIP particle display)
    content, bake_injected = _inject_bake_before_render(content)
    if bake_injected:
        fixes_applied.append("Injected fluid bake call before render (fixes green particle display)")

    # P1 FIX: Align bake frame range with scene frame range
    content, bake_fixed = _inject_bake_frame_alignment(content)
    if bake_fixed:
        fixes_applied.append("Aligned bake cache frame range with scene frame range")

    # P1 FIX: Planar liquid inflow emitters need plane init
    content, plane_init_fixed = _inject_plane_init_for_liquid_flow(content)
    if plane_init_fixed:
        fixes_applied.append("Enabled use_plane_init for liquid flow emitter")

    # P2 FIX: Fix hallucinated particle_radius values (must be in cell sizes, not real-world)
    content, particle_radius_fixed = _fix_particle_radius_value(content)
    if particle_radius_fixed:
        fixes_applied.append("Fixed particle_radius: value < 0.1 is nonsensical (reset to 1.0)")

    # P0 FIX: Strip * 0.5 from scale tuples (LLM halves all dimensions, causing exploded geometry)
    content, scale_fixed = _fix_cube_scale_halving(content)
    if scale_fixed:
        fixes_applied.append("Removed * 0.5 from obj.scale tuples (fixes exploded geometry)")

    # P0 FIX: Camera distance validation (fixes too-close/inside-geometry cameras + scale corruption)
    content, camera_fixed = _inject_camera_distance_fix(content)
    if camera_fixed:
        fixes_applied.append("Injected camera distance validation (fixes too-close/inside-geometry cameras)")

    # P0 FIX: Replace animation=True with per-frame still renders (headless compat)
    content, stills_fixed = _inject_animation_to_stills(content)
    if stills_fixed:
        fixes_applied.append("Replaced animation=True with representative still renders (headless fix)")

    # Post-fix: Python syntax validation (catches errors introduced by fixers)
    # Loop to handle cascading indentation errors (fix one → new error on next line → repeat)
    for _syntax_pass in range(10):  # max 10 passes
        try:
            compile(content, str(path), 'exec')
            break  # All good
        except SyntaxError as e:
            if 'unexpected indent' in str(e.msg) and e.lineno:
                lines = content.split('\n')
                idx = e.lineno - 1  # 0-indexed
                if 0 <= idx < len(lines):
                    stripped = lines[idx].lstrip()
                    # Find indent of previous non-blank line
                    prev_indent = 0
                    for j in range(idx - 1, -1, -1):
                        if lines[j].strip():
                            prev_indent = len(lines[j]) - len(lines[j].lstrip())
                            break
                    lines[idx] = ' ' * prev_indent + stripped
                    content = '\n'.join(lines)
                    if _syntax_pass == 0:
                        fixes_applied.append(f"Fixed indentation error at line {e.lineno}")
                    continue
            # Non-indentation error or can't fix — warn and stop
            fixes_applied.append(f"WARNING: Unfixable syntax error at line {e.lineno}: {e.msg}")
            break

    if fixes_applied:
        # Write fixed content back
        path.write_text(content)
        print(f"[API Fixer] Applied {len(fixes_applied)} fixes to {path.name}")
        for fix in fixes_applied:
            print(f"  - {fix}")

        return {
            "fixed": True,
            "fixes_applied": fixes_applied,
            "original_path": str(path),
            "fixed_path": str(path),
            "content_changed": True
        }
    else:
        return {
            "fixed": False,
            "fixes_applied": [],
            "original_path": str(path),
            "fixed_path": str(path),
            "content_changed": False
        }


def check_for_known_issues(script_path: str) -> List[str]:
    """
    Check script for known-bad patterns WITHOUT fixing.

    Args:
        script_path: Path to the Blender Python script

    Returns:
        List of issue descriptions found
    """
    path = Path(script_path)
    if not path.exists():
        return [f"Script not found: {path}"]

    content = path.read_text()
    issues = []

    for pattern, _, description in BLENDER_50_FIXES:
        if re.search(pattern, content):
            issues.append(description)

    return issues


def add_custom_fix(pattern: str, replacement: str, description: str) -> None:
    """
    Add a custom fix pattern at runtime.

    Useful when the Coordinator discovers a new issue that should be
    automatically fixed in future runs.

    Args:
        pattern: Regex pattern to match
        replacement: Replacement string (can use \\1, \\2 for groups)
        description: Human-readable description
    """
    BLENDER_50_FIXES.append((pattern, replacement, description))
    print(f"[API Fixer] Added custom fix: {description}")


# CLI for testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python blender_api_fixer.py <script_path> [--check-only]")
        sys.exit(1)

    script_path = sys.argv[1]
    check_only = "--check-only" in sys.argv

    if check_only:
        issues = check_for_known_issues(script_path)
        if issues:
            print(f"Found {len(issues)} issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("No known issues found")
    else:
        result = validate_and_fix_script(script_path)
        if result["fixed"]:
            print(f"Fixed {len(result['fixes_applied'])} issues")
        else:
            print("No fixes needed")
