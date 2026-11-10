@echo off
REM Copy this file to start_all.bat (git ignores that name) and fill in your own values.
cd /d "%~dp0"
call venv\Scripts\activate
set GEE_PROJECT=your-earth-engine-project-id
set AGRISENSE_PASSWORD=choose-a-long-password
set AGRISENSE_API_URL=http://127.0.0.1:8000
start "AgriSense API" cmd /k "uvicorn api:app --host 127.0.0.1 --port 8000"
echo Waiting for the API to start...
timeout /t 8 >nul
streamlit run app.py --server.port 8501
