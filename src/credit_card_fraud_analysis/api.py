import numpy as np
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List
from pathlib import Path
from credit_card_fraud_analysis.hydra_config_loader import load_config
from credit_card_fraud_analysis.lightning_module import LitAutoEncoder
import onnxruntime as ort
import onnx
from credit_card_fraud_analysis.utils.my_logger import logger

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"


class TransactionRequest(BaseModel):
    features: List[float]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize both state variables
    app.state.model = None
    app.state.ort_session = None
    app.state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load PyTorch Model
    logger.info("Loading PyTorch Model")
    ckpt_files = list(MODELS_DIR.glob("*.ckpt"))
    if ckpt_files:
        latest_ckpt = max(ckpt_files, key=lambda p: p.stat().st_mtime)
        try:
            app.state.model = LitAutoEncoder.load_from_checkpoint(latest_ckpt)
            app.state.model.to(app.state.device)
            app.state.model.eval()
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")

    target_model_path = MODELS_DIR / "optimized_model.onnx"
    if target_model_path.exists():
        logger.info(f"Priority: Found optimized model at {target_model_path.name}")
    else:
        onnx_files = list(MODELS_DIR.glob("*.onnx"))
        if onnx_files:
            target_model_path = max(onnx_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Loading most recent generic ONNX file: {target_model_path.name}")
        else:
            target_model_path = None
            logger.error("No ONNX files found in the models directory.")

    if target_model_path:
        try:
            model_str_path = str(target_model_path)

            # Initialize session
            app.state.ort_session = ort.InferenceSession(model_str_path)

            # Verification logic from class
            onnx_model = onnx.load(model_str_path)
            onnx.checker.check_model(onnx_model)

            logger.info("ONNX Model loaded and verified successfully.")
            # logger.debug(onnx.printer.to_text(onnx_model.graph))
        except Exception as e:
            logger.error(f"Failed to initialize ONNX session or checker: {e}")

    logger.info("Loading configurations")
    try:
        config = load_config()
        app.state.threshold = getattr(config.evaluation, "threshold_percentile", 0.005)
    except Exception:
        app.state.threshold = 0.005

    yield
    # Cleanup
    if app.state.model: del app.state.model
    if app.state.ort_session: del app.state.ort_session


app = FastAPI(title="Credit Card Fraud API", lifespan=lifespan)


@app.get("/")
def root():
    return {"message": "Credit Card Fraud Detection API is active"}


@app.post("/predict")
async def predict(request: Request, data: TransactionRequest, use_onnx: bool = True):
    threshold = request.app.state.threshold
    mse_loss = 0

    # ONNX
    if use_onnx:
        logger.info("Using ONNX session for prediction...")
        session = request.app.state.ort_session
        if not session:
            raise HTTPException(status_code=503, detail="ONNX session not available")

        input_data = np.array([data.features], dtype=np.float32)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_data})
        reconstruction = outputs[0]
        mse_loss = np.mean((input_data - reconstruction) ** 2)

    else:
        logger.info("Using Pytorch saved model for prediction...")
        model = request.app.state.model
        if not model:
            raise HTTPException(status_code=503, detail="PyTorch model not available")

        input_tensor = torch.tensor([data.features], dtype=torch.float32).to(request.app.state.device)
        with torch.no_grad():
            reconstruction_tensor = model(input_tensor)
            mse_loss = torch.mean((input_tensor - reconstruction_tensor) ** 2).item()

    return {
        "is_fraud": bool(mse_loss > threshold),
        "reconstruction_error": float(mse_loss),
        "engine": "onnx" if use_onnx else "pytorch"
    }
