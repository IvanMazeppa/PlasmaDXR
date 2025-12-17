@echo off
echo Creating shared docs symlinks...

REM LuminousStars
if not exist "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-LuminousStars\docs\shared" (
    mklink /J "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-LuminousStars\docs\shared" "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\docs"
    echo LuminousStars: Created
) else (
    echo LuminousStars: Already exists
)

REM Blender
if not exist "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Blender\docs\shared" (
    mklink /J "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Blender\docs\shared" "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\docs"
    echo Blender: Created
) else (
    echo Blender: Already exists
)

REM NanoVDB
if not exist "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-NanoVDB\docs\shared" (
    mklink /J "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-NanoVDB\docs\shared" "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\docs"
    echo NanoVDB: Created
) else (
    echo NanoVDB: Already exists
)

REM MultiAgent
if not exist "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-MultiAgent\docs\shared" (
    mklink /J "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-MultiAgent\docs\shared" "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\docs"
    echo MultiAgent: Created
) else (
    echo MultiAgent: Already exists
)

echo.
echo Done! Each worktree now has docs/shared/ pointing to main docs.
pause
