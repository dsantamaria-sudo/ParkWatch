@echo off
rem Copies the receiver to the Raspberry Pi and installs its environment. Set before running:
rem   PW_PI_HOST  user@host of the Pi        PW_PI_KEY  path to the SSH private key
setlocal
if "%PW_PI_HOST%"=="" (echo Set PW_PI_HOST=user@host of the Raspberry Pi & exit /b 1)
if "%PW_PI_KEY%"=="" (echo Set PW_PI_KEY=path to the SSH private key & exit /b 1)
set KEY=%PW_PI_KEY%
set HOST=%PW_PI_HOST%
set SSH=ssh -i "%KEY%" -o StrictHostKeyChecking=accept-new %HOST%

echo === 1/5: copying pi_receiver again (includes requirements.txt) ===
rem individual files: "scp -r folder" nests pi_receiver/ inside if the destination already exists
set SRC=%~dp0.
%SSH% "mkdir -p ~/parkwatch_receiver"
if %ERRORLEVEL% NEQ 0 goto :fail
scp -i "%KEY%" -o StrictHostKeyChecking=accept-new "%SRC%\app.py" "%SRC%\requirements.txt" "%SRC%\.env" "%SRC%\.env.example" %HOST%:~/parkwatch_receiver/
if %ERRORLEVEL% NEQ 0 goto :fail

echo.
echo === 2/5: creating the virtual environment (if missing) ===
%SSH% "test -d ~/parkwatch_receiver/venv || python3 -m venv ~/parkwatch_receiver/venv"
if %ERRORLEVEL% NEQ 0 goto :fail

echo.
echo === 3/5: upgrading pip ===
%SSH% "~/parkwatch_receiver/venv/bin/pip install --upgrade pip -q"
if %ERRORLEVEL% NEQ 0 goto :fail

echo.
echo === 4/5: installing dependencies (may take a few minutes on a Pi) ===
%SSH% "~/parkwatch_receiver/venv/bin/pip install -r ~/parkwatch_receiver/requirements.txt -q"
if %ERRORLEVEL% NEQ 0 goto :fail

echo.
echo === 5/5: verifying the installation ===
%SSH% "~/parkwatch_receiver/venv/bin/python -c \"import fastapi, uvicorn, azure.storage.blob, astral, PIL, dotenv; print('IMPORTS_OK')\""
if %ERRORLEVEL% NEQ 0 goto :fail

echo.
echo === Environment ready on the Pi ===
goto :end

:fail
echo.
echo === ERROR in the previous step, check the message above ===

:end
echo.
pause
