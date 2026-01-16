@echo off
call C:\Users\georg\anaconda3\Scripts\activate.bat C:\Users\georg\anaconda3\envs\mlops
cd /d C:\Users\georg\Desktop\MLOps_project
set PYTHONPATH=src
python src/credit_card_fraud_analysis/train_lightning.py
pause