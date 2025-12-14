"""
GPT-5.2 — Blender TDR-Safe Configuration Script (Blender 5.0+)

Purpose:
This is a *one-file* helper you can run before heavy volumetric work to reduce the odds of a
Windows GPU driver timeout (TDR) when using Cycles/volumes.

Rationale (Blender 5.0 Manual):
- Cycles GPU rendering can hit a Windows driver timeout on heavy scenes/volumes
  (manual mentions "display driver lost connection" and suggests reducing tile size or
  increasing the time-out): `render/cycles/gpu_rendering.html`
- Cycles Volumes has safeguards like Step Rate and Max Steps to cap extremely long renders:
  `render/cycles/render_settings/volumes.html`
- The Render Device can be CPU or GPU Compute:
  `render/cycles/render_settings/index.html`

What this script does (safe defaults):
- Prefer EEVEE for the active scene (reduces accidental Cycles viewport TDR).
- If Cycles is used, default to CPU device (no Windows GPU timeout risk).
- Apply conservative Cycles sample/bounce/volume-step limits when those properties exist.
- Print the set values so you can confirm in the console.

Run (headless):
  blender -b -P docs/blender_recipes/GPT-5-2_Scripts_Docs_Advice/blender_tdr_safe_config.py

Run (interactive):
  Open Blender → Scripting workspace → open this file → Run Script (Alt+P)
"""

from __future__ import annotations

import sys

import bpy


def _safe_enum_set(obj, prop_name: str, desired: str) -> bool:
    try:
        prop = obj.bl_rna.properties[prop_name]
        valid = {it.identifier for it in prop.enum_items}
        if desired in valid:
            setattr(obj, prop_name, desired)
            return True
        return False
    except Exception:
        return False


def _safe_rna_set(obj, prop_name: str, value) -> bool:
    try:
        if not hasattr(obj, "bl_rna"):
            return False
        if prop_name not in obj.bl_rna.properties:
            return False
        setattr(obj, prop_name, value)
        return True
    except Exception:
        return False


def _available_render_engines() -> set[str]:
    try:
        return {it.identifier for it in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    except Exception:
        return set()


def apply(scene: bpy.types.Scene) -> None:
    engines = _available_render_engines()

    # Prefer EEVEE for day-to-day interaction (reduces accidental Cycles viewport renders).
    if "BLENDER_EEVEE_NEXT" in engines:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    elif "BLENDER_EEVEE" in engines:
        scene.render.engine = "BLENDER_EEVEE"

    print("[GPT-5.2 TDR-SAFE] Render engine set to:", scene.render.engine)

    # Cycles safety (only if cycles settings exist)
    try:
        cyc = scene.cycles
    except Exception:
        print("[GPT-5.2 TDR-SAFE] Scene has no Cycles settings; done.")
        return

    # Force CPU to avoid Windows GPU driver timeouts (user can override later).
    if _safe_enum_set(cyc, "device", "CPU"):
        print("[GPT-5.2 TDR-SAFE] Cycles device:", getattr(cyc, "device", "<unknown>"))

    # Conservative sampling / bounces.
    _safe_rna_set(cyc, "samples", 16)
    _safe_rna_set(cyc, "max_bounces", 2)
    _safe_rna_set(cyc, "transparent_max_bounces", 2)
    _safe_rna_set(cyc, "volume_bounces", 0)

    # Volume ray-marching guards (names can vary slightly; try common ones).
    for step_rate_name in ("volume_step_rate", "volume_step_rate_render"):
        if _safe_rna_set(cyc, step_rate_name, 2.0):
            print(f"[GPT-5.2 TDR-SAFE] {step_rate_name}: {getattr(cyc, step_rate_name)}")
            break
    for step_rate_vp_name in ("volume_step_rate_viewport",):
        if _safe_rna_set(cyc, step_rate_vp_name, 4.0):
            print(f"[GPT-5.2 TDR-SAFE] {step_rate_vp_name}: {getattr(cyc, step_rate_vp_name)}")
            break
    for max_steps_name in ("volume_max_steps",):
        if _safe_rna_set(cyc, max_steps_name, 128):
            print(f"[GPT-5.2 TDR-SAFE] {max_steps_name}: {getattr(cyc, max_steps_name)}")
            break

    # Print a quick "volume-related properties" summary to help future debugging.
    try:
        vol_props = [p.identifier for p in cyc.bl_rna.properties if "volume" in p.identifier]
        print("[GPT-5.2 TDR-SAFE] Cycles volume-related RNA properties:", ", ".join(sorted(vol_props)) or "(none)")
    except Exception:
        pass


def main() -> None:
    apply(bpy.context.scene)

    # Note: preferences changes (Cycles Render Devices) live in Preferences UI.
    # This script intentionally avoids forcing Preferences changes, since those are global and
    # can surprise users. The scene-level CPU device is typically enough to avoid TDR.
    print("[GPT-5.2 TDR-SAFE] Done.")


if __name__ == "__main__":
    main()


