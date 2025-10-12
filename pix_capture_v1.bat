@echo off
echo ======================================
echo PIX Capture v1 - Simplified Launch
echo ======================================
echo.
echo Configuration:
echo   Particles: 10000 (default)
echo   Renderer: Gaussian (default)
echo   Frames captured: 1
echo   No command-line args needed!
echo.

cd /d D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean

REM Delete old capture
if exist "pix\Captures\v1_test.wpix" del "pix\Captures\v1_test.wpix"

echo Launching app and taking capture...
echo.

REM Simplified command - no --command-line needed since defaults are correct!
"C:\Program Files\Microsoft PIX\2509.25\pixtool.exe" launch "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\build\Debug\PlasmaDX-Clean.exe" --working-directory="D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean" take-capture --frames=1 --open save-capture "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\pix\Captures\v1_test.wpix"

echo.
echo ======================================
echo Results
echo ======================================
echo.

if exist "pix\Captures\v1_test.wpix" (
    for %%A in ("pix\Captures\v1_test.wpix") do (
        echo SUCCESS! PIX Capture v1 Working!
        echo   File: pix\Captures\v1_test.wpix
        echo   Size: %%~zA bytes (~%%~zAMB)
        echo.
        if %%~zA LSS 100000 (
            echo   WARNING: File is small (^<100KB^) - may have issues
        ) else (
            echo   File size looks good - capture successful!
        )
    )
    echo.
    echo This is v1 of the PIX automation agent!
    echo   - Captures frame 0-1 automatically
    echo   - Gaussian renderer active by default
    echo   - 10K particles by default
    echo   - Simplified command syntax
    echo.
    echo Open in PIX:
    echo   "C:\Program Files\Microsoft PIX\2509.25\WinPix.exe" pix\Captures\v1_test.wpix
    echo.
    echo Next: Config file for runtime control
) else (
    echo FAILED: No capture file created
    echo Check logs\ folder for errors
)

echo.
echo Latest log file:
dir /b /o-d logs\*.log | findstr /n "^" | findstr "^1:"

echo.
pause
