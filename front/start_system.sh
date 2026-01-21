#!/bin/bash

echo "Starting Credit Card Fraud Detection System..."
echo

# Ensure we're in the project root directory
if [ ! -d "src/credit_card_fraud_analysis" ]; then
    echo "Error: Please run this script from the MLOps_project root directory."
    echo "Make sure you can see the 'src' and 'front' folders from where you're running this."
    exit 1
fi

echo "Using current Python environment..."
python3 --version
echo

echo "Starting Backend API Server on port 8000..."
export PYTHONPATH=src
# Run backend in background
python3 -m uvicorn src.credit_card_fraud_analysis.api:app \
    --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "Waiting for backend to initialize..."
sleep 5

echo "Starting Frontend on port 8501..."
export BACKEND_URL="http://localhost:8000"
python3 -m streamlit run front/frontend.py --server.port=8501 &
FRONTEND_PID=$!

echo
echo "========================================"
echo "  Credit Card Fraud Detection System"
echo "========================================"
echo "Backend API: http://localhost:8000"
echo "Frontend:    http://localhost:8501"
echo
echo "Both services are starting in the background."
echo "Use Ctrl+C to stop this script."
echo

# Keep script alive so background processes don't die
wait $BACKEND_PID $FRONTEND_PID
