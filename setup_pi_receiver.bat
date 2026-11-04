@echo off
setlocal
set KEY=C:\Users\Usuario\Desktop\servers\Raspberry\Raspberry2\keys\id_ed25519
set HOST=dsantamaria@100.78.53.106
set SSH=ssh -i "%KEY%" -o StrictHostKeyChecking=accept-new %HOST%

echo === 1/5: copying pi_receiver again (includes requirements.txt) ===
rem individual files: "scp -r folder" nests pi_receiver/ inside if the destination already exists
set SRC=C:\Users\Usuario\Desktop\OtherProjects\exit\ParkWatch\pi_receiver
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
