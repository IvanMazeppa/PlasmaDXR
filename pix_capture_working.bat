@echo off
echo =====================================
echo PIX Capture - WORKING VERSION
echo =====================================
echo.
echo Using full paths and proper flags
echo.
pause

REM Set environment variables
set PIX_AUTO_CAPTURE=1
set PIX_CAPTURE_FRAME=120

echo Environment variables set:
echo   PIX_AUTO_CAPTURE=%PIX_AUTO_CAPTURE%
echo   PIX_CAPTURE_FRAME=%PIX_CAPTURE_FRAME%
echo.

REM Change to project directory
cd /d D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean

echo Launching via pixtool from its install folder...
echo App will use Gaussian renderer with 10K particles
echo Capture will trigger at frame 120 and save to pix\Captures\
echo.

REM Run pixtool from its install folder with CORRECT syntax
REM Strategy: Launch app, wait a bit for warmup, then take-capture
REM Key changes:
REM   1. Full path to pixtool.exe
REM   2. Full path to app exe (just the exe, no args)
REM   3. App arguments passed via --command-line parameter (CRITICAL!)
REM   4. Working directory set to project root (CRITICAL for shader/log paths!)
REM   5. Removed PIX_AUTO_CAPTURE env vars (conflicts with programmatic-capture)
REM   6. Using take-capture after 3 second delay for warmup
timeout /t 1 /nobreak >nul
start "PIX Capture" /B "C:\Program Files\Microsoft PIX\2509.25\pixtool.exe" launch "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\build\Debug\PlasmaDX-Clean.exe" --command-line="--particles 10000 --gaussian" --working-directory="D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean"
echo Waiting 3 seconds for app warmup...
timeout /t 3 /nobreak
echo Taking capture...
"C:\Program Files\Microsoft PIX\2509.25\pixtool.exe" take-capture --frames=1 --open save-capture "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\pix\Captures\test.wpix"

echo.
echo =====================================
echo Checking results...
echo =====================================
echo.

REM Check if capture was created
if exist "pix\Captures\test.wpix" (
    for %%A in ("pix\Captures\test.wpix") do (
        echo SUCCESS! Capture file created!
        echo   File: pix\Captures\test.wpix
        echo   Size: %%~zA bytes
        echo.
        if %%~zA LSS 100000 (
            echo   WARNING: File is small ^(^<100KB^)
        ) else (
            echo   SUCCESS! File size looks good!
        )
    )
    echo.
    echo You can now open this capture in PIX:
    echo   "C:\Program Files\Microsoft PIX\2509.25\WinPix.exe" pix\Captures\test.wpix
) else (
    echo FAILED: No capture file created
    echo.
    echo Check logs\ folder for error messages
)

echo.
echo Latest log file:
dir /b /o-d logs\*.log | findstr /n "^" | findstr "^1:"

echo.
pause