@echo off
setlocal
set KEY=C:\Users\Usuario\Desktop\servers\Raspberry\Raspberry2\keys\id_ed25519
set DEST=dsantamaria@100.78.53.106:~/parkwatch_receiver
set SRC=C:\Users\Usuario\Desktop\OtherProjects\exit\ParkWatch\pi_receiver

echo === Copying pi_receiver to Raspberry2 (Tailscale) ===
echo Source:      %SRC%
echo Destination: %DEST%
echo.

rem individual files: "scp -r folder" nests pi_receiver/ inside if the destination already exists
scp -i "%KEY%" -o StrictHostKeyChecking=accept-new "%SRC%\app.py" "%SRC%\requirements.txt" "%SRC%\.env" "%SRC%\.env.example" %DEST%/

if %ERRORLEVEL% EQU 0 (
    echo.
    echo === Copy completed successfully ===
) else (
    echo.
    echo === ERROR: the copy failed, check the message above ===
)

echo.
pause
