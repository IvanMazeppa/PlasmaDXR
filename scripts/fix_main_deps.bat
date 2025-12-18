@echo off
echo Fixing main repo dependencies...

REM Create DLSS junction
if not exist "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\dlss" (
    mklink /J "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\dlss" "D:\Users\dilli\AndroidStudioProjects\DLSS"
    echo DLSS: Created junction
) else (
    echo DLSS: Already exists
)

echo.
echo Dependencies fixed!
pause
