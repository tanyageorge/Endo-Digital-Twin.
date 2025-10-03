#!/bin/bash

# Endo Digital Twin - Launch Script
echo "🩸 Starting Endo Digital Twin..."
echo "📱 The app will open in your browser at http://localhost:8501"
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if requirements are installed
echo "🔍 Checking dependencies..."
python3 -c "import streamlit, pandas, plotly" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing dependencies..."
    python3 -m pip install -r requirements.txt
fi

# Launch the app
echo "🚀 Launching Streamlit app..."
python3 -m streamlit run app.py

