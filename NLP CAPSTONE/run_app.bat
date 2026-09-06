@echo off
title StudyPal - AI Textbook Assistant
echo ========================================================
echo   Starting StudyPal - AI Textbook Assistant (Streamlit)
echo ========================================================
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found in .\venv.
    echo Please make sure the venv is configured.
    pause
    exit /b 1
)

echo Launching app with virtual environment...
.\venv\Scripts\python.exe -m streamlit run app.py

pause
