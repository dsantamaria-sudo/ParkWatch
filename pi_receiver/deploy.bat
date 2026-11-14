@echo off
rem Copies the receiver to the Raspberry Pi. Set before running (e.g. in your user environment):
rem   PW_PI_HOST  user@host of the Pi        PW_PI_KEY  path to the SSH private key
setlocal
if "%PW_PI_HOST%"=="" (echo Set PW_PI_HOST=user@host of the Raspberry Pi & exit /b 1)
if "%PW_PI_KEY%"=="" (echo Set PW_PI_KEY=path to the SSH private key & exit /b 1)
set KEY=%PW_PI_KEY%
set DEST=%PW_PI_HOST%:~/parkwatch_receiver
set SRC=%~dp0.

echo === Copying pi_receiver to the Raspberry Pi ===
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
