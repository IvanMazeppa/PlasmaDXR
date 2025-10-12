@echo off
echo ======================================
echo PIX Take-Capture Test (Immediate)
echo ======================================
echo.

cd /d D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean

echo Strategy: Launch app, take capture of first frame
echo.

REM Use take-capture instead of programmatic-capture
REM Commands execute in order: launch, then take-capture
REM This will capture the first frame after launch
"C:\Program Files\Microsoft PIX\2509.25\pixtool.exe" launch "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\build\Debug\PlasmaDX-Clean.exe" --command-line="--particles 10000 --gaussian" --working-directory="D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean" take-capture --frames=1 save-capture "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\pix\Captures\test_immediate.wpix"

echo.
echo ======================================
echo Checking results...
echo ======================================
echo.

if exist "pix\Captures\test_immediate.wpix" (
    for %%A in ("pix\Captures\test_immediate.wpix") do (
        echo SUCCESS! Capture file created!
        echo   File: pix\Captures\test_immediate.wpix
        echo   Size: %%~zA bytes

        if %%~zA LSS 100000 (
            echo   WARNING: File is small (^<100KB^)
        ) else (
            echo   File size looks good!
        )
    )
    echo.
    echo Open in PIX with:
    echo   "C:\Program Files\Microsoft PIX\2509.25\WinPix.exe" pix\Captures\test_immediate.wpix
) else (
    echo FAILED: No capture file created
)

echo.
pause
