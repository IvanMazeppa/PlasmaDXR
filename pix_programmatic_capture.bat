@echo off
echo ======================================
echo PIX Programmatic Capture Test
echo ======================================
echo.

cd /d D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean

echo Configuration:
echo   Warmup frames: 120
echo   Renderer: Gaussian (10K particles)
echo   Working dir: Project root
echo.

REM Launch with PIX using correct syntax
REM - App will call PIXBeginCapture() at frame 120 (via env vars)
REM - pixtool programmatic-capture will wait for the capture
REM - When capture completes, app exits and pixtool saves the .wpix
"C:\Program Files\Microsoft PIX\2509.25\pixtool.exe" launch "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\build\Debug\PlasmaDX-Clean.exe" --command-line="--particles 10000 --gaussian" --working-directory="D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean" --setenv="PIX_AUTO_CAPTURE=1" --setenv="PIX_CAPTURE_FRAME=120" programmatic-capture --open

echo.
echo ======================================
echo Checking results...
echo ======================================
echo.

REM Check if capture was created
if exist "pix\Captures\auto_capture.wpix" (
    for %%A in ("pix\Captures\auto_capture.wpix") do (
        echo SUCCESS! Capture file created!
        echo   File: pix\Captures\auto_capture.wpix
        echo   Size: %%~zA bytes
        echo.
        if %%~zA LSS 100000 (
            echo   WARNING: File is small (^<100KB^)
        ) else (
            echo   SUCCESS! File size looks good!
        )
    )
    echo.
    echo You can now open this capture in PIX:
    echo   "C:\Program Files\Microsoft PIX\2509.25\WinPix.exe" pix\Captures\auto_capture.wpix
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
