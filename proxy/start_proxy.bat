@echo off
echo Starting NSE Options Proxy...

cd /d "%~dp0"

start "NSE Proxy" cmd /k "pip install flask >nul 2>&1 && python app.py"

timeout /t 3 /nobreak >nul

start "ngrok" cmd /k "ngrok http --domain=angelic-dispersed-overall.ngrok-free.app 8000"

echo.
echo Both services started.
echo Proxy: http://localhost:8000
echo Public: https://angelic-dispersed-overall.ngrok-free.app
echo.
echo Keep both windows open while using the dashboard.
pause
