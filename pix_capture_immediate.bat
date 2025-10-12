@echo off
echo ======================================
echo PIX Immediate Capture Test
echo ======================================
echo.
echo Testing: launch + take-capture (no delay)
echo Note: This will capture early frames, not frame 120
echo Goal: Verify the pixtool workflow works at all
echo.

cd /d D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean

REM Delete old capture
if exist "pix\Captures\immediate_test.wpix" del "pix\Captures\immediate_test.wpix"

echo Launching app and taking immediate capture...
echo.

"C:\Program Files\Microsoft PIX\2509.25\pixtool.exe" launch "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\build\Debug\PlasmaDX-Clean.exe" --command-line="--particles 10000 --gaussian" --working-directory="D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean" take-capture --frames=1 --open save-capture "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\pix\Captures\immediate_test.wpix"

echo.
echo ======================================
echo Results
echo ======================================
echo.

if exist "pix\Captures\immediate_test.wpix" (
    for %%A in ("pix\Captures\immediate_test.wpix") do (
        echo SUCCESS! Capture file created!
        echo   File: pix\Captures\immediate_test.wpix
        echo   Size: %%~zA bytes
        echo.
        if %%~zA LSS 100000 (
            echo   WARNING: File is small (^<100KB^)
        ) else (
            echo   File size looks good!
        )
    )
    echo.
    echo This proves the pixtool workflow WORKS!
    echo Now we just need to capture the RIGHT frame (120+)
    echo.
    echo Open in PIX:
    echo   "C:\Program Files\Microsoft PIX\2509.25\WinPix.exe" pix\Captures\immediate_test.wpix
) else (
    echo FAILED: No capture file created
)

echo.
pause