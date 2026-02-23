#!/bin/bash
echo "==================================================="
echo "        Welcome to R Talal Sensors (Mac/Linux)    "
echo "        Sensor Pipeline Analysis Engine           "
echo "==================================================="
echo ""
echo "Setting up the mathematical environment..."
echo "This might take a minute on the very first run."

# Move to the script's directory
cd "$(dirname "$0")"

# Check if Virtual Environment exists, if not create it
if [ ! -d "venv" ]; then
    echo "[!] Creating isolated Python environment..."
    python3 -m venv venv
fi

# Activate the environment
source venv/bin/activate

# Install required pip packages securely
echo "[!] Verifying Scientific Packages..."
pip install -r requirements.txt --quiet

# Launch
echo ""
echo "🚀 Booting up the Neural Pipeline on Localhost..."
streamlit run app.py
