@echo off
echo Testing simple PIX launch (no capture)
echo.

cd /d D:\Users\dilli\AndroidStudioProjects\PlasmaDXR

"C:\Program Files\Microsoft PIX\2509.25\pixtool.exe" launch "D:\Users\dilli\AndroidStudioProjects\PlasmaDXR\build\Debug\PlasmaDXR.exe" --command-line="--particles 10000 --gaussian" --working-directory="D:\Users\dilli\AndroidStudioProjects\PlasmaDXR"

echo.
echo App closed. Check log:
dir /b /o-d logs\*.log | findstr /n "^" | findstr "^1:"
pause
