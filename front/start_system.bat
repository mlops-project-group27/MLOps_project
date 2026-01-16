@echo off
echo Starting Credit Card Fraud Detection System...
echo.

REM Activate the mlops environment
call C:\Users\georg\anaconda3\Scripts\activate.bat C:\Users\georg\anaconda3\envs\mlops

REM Navigate to the project directory
cd /d C:\Users\georg\Desktop\MLOps_project

echo Starting Backend API Server on port 8000...
set PYTHONPATH=src
start "Backend API" cmd /k "C:\Users\georg\anaconda3\envs\mlops\python.exe -m uvicorn src.credit_card_fraud_analysis.api:app --reload --host 0.0.0.0 --port 8000"

echo Waiting for backend to initialize...
timeout /t 5 /nobreak > nul

echo Starting Frontend on port 8501...
set BACKEND_URL=http://localhost:8000
start "Frontend" cmd /k "C:\Users\georg\anaconda3\envs\mlops\python.exe -m streamlit run front/frontend.py --server.port=8501"

echo.
echo ========================================
echo   Credit Card Fraud Detection System
echo ========================================
echo Backend API: http://localhost:8000
echo Frontend:    http://localhost:8501
echo.
echo Both services are starting in separate windows.
echo The frontend will automatically open in your browser.
echo.
echo Press any key to exit this window...
pause > nul