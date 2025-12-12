@echo off
echo.
echo === SSH Key Setup for PlasmaDX Server ===
echo.
echo This will copy your SSH key to the server.
echo You'll enter your password ONE LAST TIME, then never again.
echo.
pause

echo.
echo Copying SSH key...
type %USERPROFILE%\.ssh\id_ed25519.pub | ssh maz3ppa@192.168.0.237 "cat >> ~/.ssh/authorized_keys"

echo.
echo Testing connection (should NOT ask for password)...
ssh maz3ppa@192.168.0.237 "echo SUCCESS! SSH key authentication is working!"

echo.
echo If you saw SUCCESS above, you're all set!
echo.
pause
