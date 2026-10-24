@echo off
REM Binzy - Smart Waste Segregation Assistant. Opens http://localhost:8000 (all caches stay inside this folder).
REM Use "run_app.bat --lan" to also open it from a phone on the same Wi-Fi.
cd /d "%~dp0"
set HF_HOME=%~dp0.cache\hf
set TORCH_HOME=%~dp0.cache\torch
set HF_HUB_OFFLINE=1
set OMP_NUM_THREADS=1
start "" cmd /c "timeout /t 25 >nul && start http://localhost:8000"
".venv\Scripts\python.exe" server.py %*
pause
