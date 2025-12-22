  Continue Luminous Star Particles tweaking in PlasmaDX-LuminousStars worktree.

  Status: Core implementation COMPLETE and committed (edb3f11). Stars are orbiting correctly (coordinates updating), but ImGui controls need fixes.

  Known Issues to Fix:
  1. Position controls locked - Can see stars moving but can't manually adjust positions
  2. Color controls locked - Can't change star light colors via ImGui
  3. Brightness controls locked - Intensity sliders not affecting star lights

  What's Working:
  - 16 star particles with Keplerian orbital motion ✅
  - CPU position prediction syncing to GPU ✅
  - 29 total lights (16 star + 13 static) ✅
  - SUPERGIANT_STAR material (0.15 opacity, 15× emission) ✅

  Key Files:
  - src/particles/LuminousParticleSystem.cpp - Star management, Update(), GetStarLights()
  - src/core/Application.cpp:4856-4931 - ImGui controls for Luminous Stars
  - src/core/Application.cpp:814-827 - Light merging in render loop
