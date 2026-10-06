"""FastAPI service for churn prediction.

Loads the tracked champion bundle (preprocessor + calibrated model + decision
threshold) produced by ``scripts/run_analysis.py`` and saved to
``models/champion.pkl`` (also logged to MLflow).

Endpoints:
- ``GET /health`` — liveness + model metadata
- ``POST /predict`` — single customer -> churn probability + decision
- ``POST /predict/batch`` — list of customers -> list of predictions
"""

from __future__ import annotations

import os
import pickle
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BUNDLE_PATH = os.environ.get("CHURN_MODEL_PATH", "models/champion.pkl")

app = FastAPI(title="Customer Churn Prediction API", version="0.1.0")

_bundle: dict | None = None


def get_bundle() -> dict:
    """Lazy-load the model bundle (works under TestClient too)."""
    global _bundle
    if _bundle is None:
        if not os.path.exists(BUNDLE_PATH):
            raise HTTPException(
                status_code=503,
                detail=f"Model bundle not found at {BUNDLE_PATH}. "
                       "Run scripts/run_analysis.py first.",
            )
        with open(BUNDLE_PATH, "rb") as f:
            _bundle = pickle.load(f)
    return _bundle


class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=200)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0, le=1000)
    TotalCharges: float | None = Field(default=None, ge=0)


def _predict_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    from .features import clean, engineer  # local import: keeps API import light
    bundle = get_bundle()
    df = engineer(clean(df_raw.assign(Churn="No")))
    X = bundle["preprocessor"].transform(df)
    proba = bundle["model"].predict_proba(X)[:, 1]
    threshold = bundle["threshold"]
    return pd.DataFrame({
        "churn_probability": proba.round(4),
        "churn_prediction": (proba >= threshold).astype(int),
        "threshold": threshold,
    })


@app.get("/health")
def health():
    bundle = get_bundle()
    return {
        "status": "ok",
        "model": bundle.get("model_name"),
        "threshold": bundle.get("threshold"),
        "trained_at": bundle.get("trained_at"),
    }


@app.post("/predict")
def predict(customer: Customer):
    df = pd.DataFrame([customer.model_dump()])
    row = _predict_df(df).iloc[0]
    return {
        "churn_probability": float(row["churn_probability"]),
        "churn_prediction": int(row["churn_prediction"]),
        "threshold": float(row["threshold"]),
    }


@app.post("/predict/batch")
def predict_batch(customers: list[Customer]):
    if len(customers) == 0:
        raise HTTPException(status_code=422, detail="Empty batch.")
    if len(customers) > 1000:
        raise HTTPException(status_code=422, detail="Batch too large (max 1000).")
    df = pd.DataFrame([c.model_dump() for c in customers])
    out = _predict_df(df)
    return {
        "predictions": [
            {
                "churn_probability": float(r.churn_probability),
                "churn_prediction": int(r.churn_prediction),
            }
            for r in out.itertuples()
        ],
        "threshold": float(out["threshold"].iloc[0]),
    }
