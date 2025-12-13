===============================================
   PlasmaDX Worktree Scripts
===============================================

Location: D:\Users\dilli\Scripts\

-----------------------------------------------
FIRST TIME SETUP (run once!)
-----------------------------------------------

1. Double-click: setup_ssh_key.bat
2. Enter your password ONE LAST TIME
3. You'll never need to type it again!

-----------------------------------------------
SCRIPTS
-----------------------------------------------

setup_ssh_key.bat
  - Run once to set up passwordless SSH
  - Never type your password again after this

connect_worktree.bat
  - Interactive menu to SSH into a worktree
  - Opens a terminal session

open_worktrees.bat
  - Opens worktrees in Cursor IDE
  - No arguments = opens ALL worktrees
  - With arguments = opens only those specified

  Examples:
    open_worktrees.bat              (opens all 6)
    open_worktrees.bat nanovdb      (opens just nanovdb)
    open_worktrees.bat nanovdb blender   (opens two)

open_worktrees.ps1
  - PowerShell version with more features
  - Same usage as .bat version plus:

  Examples:
    .\open_worktrees.ps1 -List      (show available worktrees)
    .\open_worktrees.ps1 -Help      (show help)

-----------------------------------------------
AVAILABLE WORKTREE NAMES
-----------------------------------------------

  main       - PlasmaDX-Clean (main development)
  nanovdb    - PlasmaDX-NanoVDB (volumetric rendering)
  blender    - PlasmaDX-Blender (VDB asset creation)
  multiagent - PlasmaDX-MultiAgent (multi-agent dev)
  pinn       - PlasmaDX-PINN-v4 (neural networks)
  gaussianq  - PlasmaDX-GaussianImageQ (image quality)

-----------------------------------------------
TIPS
-----------------------------------------------

- Create a desktop shortcut to open_worktrees.bat
- Pin to taskbar for quick access
- Run from PowerShell for the .ps1 version

===============================================
