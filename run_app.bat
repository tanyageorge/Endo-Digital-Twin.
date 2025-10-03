@echo off
REM Endo Digital Twin - Launch Script for Windows
echo 🩸 Starting Endo Digital Twin...
echo 📱 The app will open in your browser at http://localhost:8501
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if requirements are installed
echo 🔍 Checking dependencies...
python -c "import streamlit, pandas, plotly" >nul 2>&1
if %errorlevel% neq 0 (
    echo 📦 Installing dependencies...
    python -m pip install -r requirements.txt
)

REM Launch the app
echo 🚀 Launching Streamlit app...
python -m streamlit run app.py

