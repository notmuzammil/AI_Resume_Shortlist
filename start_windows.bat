@echo off
setlocal
cd /d "%~dp0"
set "PY=python"
where py >nul 2>nul
if not errorlevel 1 (
    py -3.11 -c "import sys" >nul 2>nul
    if not errorlevel 1 (
        set "PY=py -3.11"
    ) else (
        py -3.12 -c "import sys" >nul 2>nul
        if not errorlevel 1 (set "PY=py -3.12") else (set "PY=py -3")
    )
)
if not exist ".venv\Scripts\python.exe" (
    echo Setting up Python environment...
    %PY% -m venv .venv
    if errorlevel 1 goto :failed
    ".venv\Scripts\python.exe" -m pip install -r requirements-studio.txt
    if errorlevel 1 goto :failed
)
echo Starting AI Resume Shortlist System at http://127.0.0.1:8501
".venv\Scripts\python.exe" -m streamlit run platform_ui.py --server.address=127.0.0.1
if errorlevel 1 goto :failed
exit /b 0
:failed
echo.
echo Startup failed. Check Python, internet access for the first install, and README.md.
pause
exit /b 1
