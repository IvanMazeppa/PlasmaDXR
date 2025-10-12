@echo off
echo =====================================
echo PIX Capture Test - Full Workflow
echo =====================================
echo.
echo This will create an actual .wpix capture file!
echo.
echo What happens:
echo   1. Set PIX_AUTO_CAPTURE=1 and PIX_CAPTURE_FRAME=120
echo   2. Launch app via pixtool (creates PIX session)
echo   3. App runs to frame 120
echo   4. App triggers PIXBeginCapture/PIXEndCapture
echo   5. App exits
echo   6. pixtool saves capture to pix\Captures\test.wpix
echo.
pause

REM Set environment variables
set PIX_AUTO_CAPTURE=1
set PIX_CAPTURE_FRAME=120

echo.
echo Environment set:
echo   PIX_AUTO_CAPTURE=%PIX_AUTO_CAPTURE%
echo   PIX_CAPTURE_FRAME=%PIX_CAPTURE_FRAME%
echo.
echo Launching via pixtool...
echo (This may take 5-10 seconds)
echo.

REM Launch via pixtool with programmatic capture
pix\pixtool.exe launch ".\build\Debug\PlasmaDX-Clean.exe --particles 10000 --gaussian" programmatic-capture --open save-capture "pix\Captures\test.wpix"

echo.
echo =====================================
echo Checking results...
echo =====================================
echo.

if exist "pix\Captures\test.wpix" (
    for %%A in ("pix\Captures\test.wpix") do (
        echo SUCCESS! Capture file created:
        echo   File: pix\Captures\test.wpix
        echo   Size: %%~zA bytes
        echo.
        if %%~zA LSS 100000 (
            echo   WARNING: File is very small ^(^<100KB^)
            echo   May be empty or incomplete
        ) else (
            echo   File size looks good!
        )
    )
) else (
    echo FAILED: No capture file created
    echo.
    echo Possible issues:
    echo   1. App didn't launch
    echo   2. PIXBeginCapture/EndCapture failed
    echo   3. Environment variables not passed correctly
    echo.
    echo Check logs\ folder for PIX messages
)

echo.
pause