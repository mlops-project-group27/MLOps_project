@echo off
call C:\Users\georg\anaconda3\Scripts\activate.bat C:\Users\georg\anaconda3\envs\mlops
cd /d C:\Users\georg\Desktop\MLOps_project
set BACKEND_URL=http://localhost:8000
echo Starting Credit Card Fraud Detection Frontend...
C:\Users\georg\anaconda3\envs\mlops\python.exe -m streamlit run frontend.py --server.port=8501
pause