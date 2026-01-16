import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List
from pathlib import Path

from credit_card_fraud_analysis.lightning_module import LitAutoEncoder
from credit_card_fraud_analysis.hydra_config_loader import load_config


BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"


class TransactionRequest(BaseModel):
    features: List[float]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize state defaults
    app.state.model = None
    app.state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Searching for checkpoints in: {MODELS_DIR.absolute()}")

    # 1. Dynamically find the .ckpt file
    # train_lightning.py saves files like 'lit-autoencoder-epoch=XX-train_loss=XX.ckpt'
    ckpt_files = list(MODELS_DIR.glob("*.ckpt"))

    if not ckpt_files:
        print(f"ERROR: No checkpoint found in {MODELS_DIR}")
    else:
        # Use the most recently modified checkpoint
        latest_ckpt = max(ckpt_files, key=lambda p: p.stat().st_mtime)
        print(f"Loading checkpoint: {latest_ckpt.name}")

        try:
            # Load the LitAutoEncoder
            model = LitAutoEncoder.load_from_checkpoint(latest_ckpt)
            model.to(app.state.device)
            model.eval()  # Set to inference mode
            app.state.model = model
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")

    # Load configuration for threshold
    try:
        config = load_config()
        app.state.threshold = getattr(config.evaluation, "threshold_percentile", 0.005)
    except Exception:
        app.state.threshold = 0.005

    yield
    if app.state.model:
        del app.state.model


app = FastAPI(title="Credit Card Fraud API", lifespan=lifespan)


@app.post("/predict")
async def predict(request: Request, data: TransactionRequest):
    # Retrieve from state
    model = request.app.state.model
    device = request.app.state.device
    threshold = request.app.state.threshold

    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded on server.")

    expected_dim = model.encoder[0].in_features
    actual_dim = len(data.features)

    if actual_dim != expected_dim:
        raise HTTPException(
            status_code=400,
            detail=f"Dimension mismatch: Model expects {expected_dim} features, but got {actual_dim}."
        )

    input_tensor = torch.tensor([data.features], dtype=torch.float32).to(request.app.state.device)

    with torch.no_grad():
        reconstruction = model(input_tensor)
        mse_loss = torch.mean((input_tensor - reconstruction) ** 2, dim=1).item()

    return {
        "is_fraud": bool(mse_loss > request.app.state.threshold),
        "reconstruction_error": mse_loss,
        "threshold": request.app.state.threshold
    }