@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>&1
if %errorlevel%==0 (
    py launch_facial_recognition.py
) else (
    python launch_facial_recognition.py
)

if errorlevel 1 (
    echo.
    echo Launch failed. Make sure Python is installed and available in PATH.
    pause
)

endlocal
