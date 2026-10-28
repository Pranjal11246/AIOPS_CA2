
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Locate the trained model relative to this file.
MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"

app = FastAPI(
    title="Used Car Price Prediction API",
    description="Predict used car prices using a trained machine learning model.",
    version="1.0.0",
)

# Load the model once when the application starts.
model_pipeline = None

if MODEL_PATH.exists():
    try:
        model_pipeline = joblib.load(MODEL_PATH)
    except Exception as exc:
        print(f"Could not load model: {exc}")
else:
    print(f"Model file not found: {MODEL_PATH}")


class CarDetails(BaseModel):
    brand: str = Field(..., examples=["Toyota"])
    model: str = Field(..., examples=["Camry"])
    model_year: int = Field(..., ge=1900, le=2100)
    milage: float = Field(..., ge=0)
    fuel_type: str = Field(..., examples=["Gasoline"])
    engine: str = Field(..., examples=["2.5L I4"])
    transmission: str = Field(..., examples=["Automatic"])
    ext_col: str = Field(..., examples=["White"])
    int_col: str = Field(..., examples=["Black"])
    accident: str = Field(..., examples=["None reported"])
    clean_title: str = Field(..., examples=["Yes"])


@app.get("/")
def home():
    return {
        "message": "Used Car Price Prediction API",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    if model_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check model.joblib and dependencies.",
        )

    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
def predict_price(car: CarDetails):
    if model_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is unavailable.",
        )

    # Use a DataFrame because the trained pipeline expects named columns.
    car_data = pd.DataFrame([car.model_dump()])

    try:
        prediction = model_pipeline.predict(car_data)
        predicted_price = float(prediction[0])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc

    return {
        "predicted_price": round(predicted_price, 2),
        "currency": "USD",
    }
