@echo off
echo Testing simple PIX launch (no capture)
echo.

cd /d D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean

"C:\Program Files\Microsoft PIX\2509.25\pixtool.exe" launch "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\build\Debug\PlasmaDX-Clean.exe" --command-line="--particles 10000 --gaussian" --working-directory="D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean"

echo.
echo App closed. Check log:
dir /b /o-d logs\*.log | findstr /n "^" | findstr "^1:"
pause
