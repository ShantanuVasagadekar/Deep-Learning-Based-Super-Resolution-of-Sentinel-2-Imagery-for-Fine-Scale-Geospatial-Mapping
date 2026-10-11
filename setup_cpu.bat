@echo off
setlocal
cd /d "%~dp0"
echo Creating a CPU-only Python environment for this project...
python -m venv .venv-cpu
if errorlevel 1 goto failed
".venv-cpu\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv-cpu\Scripts\python.exe" -m pip install "torch==2.11.0+cpu" --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
".venv-cpu\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Setup complete. Double-click run_app.bat to start.
pause
exit /b 0
:failed
echo Setup failed. Install 64-bit Python 3.10 or newer with Python on PATH, then retry.
echo The error above contains the specific dependency or network problem.
pause
exit /b 1
