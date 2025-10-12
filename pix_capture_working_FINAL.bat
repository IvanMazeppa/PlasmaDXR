@echo off
REM =========================================================================
REM PIX Autonomous Capture - FINAL WORKING VERSION
REM =========================================================================
REM
REM This script successfully captures GPU frames using pixtool's take-capture.
REM
REM HOW IT WORKS:
REM   1. pixtool launches the app under GPU capture mode
REM   2. take-capture waits for frames to render, then captures them
REM   3. By specifying --frames=N, we can capture after some warmup
REM
REM NOTES:
REM   - Can't specify exact frame number with take-capture
REM   - Can't wait for arbitrary warmup time
REM   - But captures ARE successful and contain valid GPU data!
REM   - For warmup: increase --frames and PIX will capture LATER frames
REM
REM =========================================================================

echo.
echo ==========================================
echo   PIX Autonomous Capture v1.0
echo ==========================================
echo.

cd /d D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean

set FRAMES_TO_CAPTURE=1
set OUTPUT_FILE=autonomous_capture.wpix

echo Configuration:
echo   Frames to capture: %FRAMES_TO_CAPTURE%
echo   Output file: %OUTPUT_FILE%
echo   Renderer: Gaussian (10K particles)
echo   Working directory: Project root
echo.

echo Launching app and capturing...
echo.

REM Single pixtool command with chained operations
REM This is the CORRECT way according to Microsoft documentation
"C:\Program Files\Microsoft PIX\2509.25\pixtool.exe" launch "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\build\Debug\PlasmaDX-Clean.exe" --command-line="--particles 10000 --gaussian" --working-directory="D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean" take-capture --frames=%FRAMES_TO_CAPTURE% save-capture "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\pix\Captures\%OUTPUT_FILE%"

echo.
echo ==========================================
echo   Results
echo ==========================================
echo.

if exist "pix\Captures\%OUTPUT_FILE%" (
    for %%A in ("pix\Captures\%OUTPUT_FILE%") do (
        set SIZE_BYTES=%%~zA
        set /a SIZE_MB=%%~zA/1048576
    )

    echo SUCCESS! Capture created!
    echo.
    echo   File: pix\Captures\%OUTPUT_FILE%
    echo   Size: %SIZE_MB% MB ^(%SIZE_BYTES% bytes^)
    echo.

    if %SIZE_MB% LSS 1 (
        echo   [!] WARNING: File is small ^(^<1MB^) - capture may be incomplete
    ) else (
        echo   [+] File size looks good!
    )

    echo.
    echo ==========================================
    echo   Next Steps
    echo ==========================================
    echo.
    echo 1. Open in PIX GUI:
    echo    ^> "C:\Program Files\Microsoft PIX\2509.25\WinPix.exe" pix\Captures\%OUTPUT_FILE%
    echo.
    echo 2. Analyze with pixtool:
    echo    ^> pixtool open-capture pix\Captures\%OUTPUT_FILE% list-counters
    echo    ^> pixtool open-capture pix\Captures\%OUTPUT_FILE% save-event-list events.csv
    echo    ^> pixtool open-capture pix\Captures\%OUTPUT_FILE% save-screenshot frame.png
    echo.
    echo 3. Export to C++ for debugging:
    echo    ^> pixtool open-capture pix\Captures\%OUTPUT_FILE% export-to-cpp exported_code\
    echo.

) else (
    echo [!] FAILED: No capture file created
    echo.
    echo Troubleshooting:
    echo   - Check logs\ folder for error messages
    echo   - Verify app launches successfully without PIX
    echo   - Ensure D3D12 debug layer doesn't conflict ^(should auto-disable^)
    echo.

    echo Latest log file:
    dir /b /o-d logs\*.log 2>nul | findstr /n "^" | findstr "^1:" 2>nul
    echo.
)

echo ==========================================
echo.
pause
