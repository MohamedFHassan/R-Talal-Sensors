@echo off
echo ===================================================
echo        Welcome to R Talal Sensors (Windows)       
echo        Sensor Pipeline Analysis Engine           
echo ===================================================
echo.
echo Setting up the mathematical environment...
echo This might take a minute on the very first run.

REM Check if Virtual Environment exists, if not create it
if not exist venv (
    echo [!] Creating isolated Python environment...
    python -m venv venv
)

REM Activate the environment
call venv\Scripts\activate.bat

REM Install required pip packages securely
echo [!] Verifying Scientific Packages...
pip install -r requirements.txt --quiet

REM Launch
echo.
echo 🚀 Booting up the Neural Pipeline on Localhost...
streamlit run app.py
pause
