@echo off
REM Full reproduction: split + train, evaluate, hybrid build, demo samples, PPT/report. Caches stay inside this folder.
cd /d "%~dp0"
set HF_HOME=%~dp0.cache\hf
set TORCH_HOME=%~dp0.cache\torch
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set PY=.venv\Scripts\python.exe
%PY% train.py --epochs 5 --workers 2 || goto :err
%PY% evaluate.py || goto :err
%PY% build_hybrid.py || goto :err
%PY% generate_demo_samples.py || goto :err
%PY% build_deliverables.py || goto :err
echo All done.
pause
exit /b 0
:err
echo Pipeline failed.
pause
exit /b 1
