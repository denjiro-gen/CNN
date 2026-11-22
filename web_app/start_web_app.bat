@echo off
title Aedes aegypti & Insect CNN Detector Web App
cd /d "%~dp0"

echo ==========================================================
echo   Starting Aedes aegypti & Insect CNN Detector Web App
echo ==========================================================
echo.

where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ and add it to your system PATH.
    pause
    exit /b 1
)

echo Checking dependencies...
python -c "import numpy, PIL" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Installing required dependencies (numpy, pillow)...
    pip install numpy pillow
)

echo Starting web server at http://localhost:5000 ...
start "" http://localhost:5000
python app.py 5000

pause
