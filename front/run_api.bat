@echo off
call C:\Users\georg\anaconda3\Scripts\activate.bat C:\Users\georg\anaconda3\envs\mlops
cd /d C:\Users\georg\Desktop\MLOps_project
set PYTHONPATH=src
echo Starting Credit Card Fraud Detection API...
C:\Users\georg\anaconda3\envs\mlops\python.exe -m uvicorn src.credit_card_fraud_analysis.api:app --reload --host 0.0.0.0 --port 8000
pause