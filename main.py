"""
Bank Churn Prediction API
==========================
FastAPI server that:
  1. Loads the trained model bundle from ./model/churn_model.pkl at startup.
  2. Exposes POST /predict – accepts customer features, returns churn probability.
  3. Exposes GET  /model-info – returns which model is loaded and its feature list.

Start:
    cd server
    uvicorn main:app --reload --port 8000
"""

import os
import pickle
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ── Model bundle (loaded once at startup) ──────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "churn_model.pkl")

_bundle: dict = {}   # populated in lifespan


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model bundle when the server starts."""
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}.\n"
            "Run  python ml/train_model.py  first to generate it."
        )

    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)

    _bundle.update(data)
    print(f"[startup] Loaded model: {_bundle['model_name']}")
    print(f"[startup] Features:     {_bundle['feature_cols']}")
    yield
    _bundle.clear()


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Bank Churn Prediction API",
    description="Predict whether a banking customer will churn.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ─────────────────────────────────────────────────
class CustomerInput(BaseModel):
    """Raw customer data exactly as a user would fill in a form."""

    gender:         str   = Field(..., examples=["Male"], description="Male or Female")
    senior_citizen: int   = Field(..., ge=0, le=1, examples=[0], description="1 = senior (65+)")
    tenure:         float = Field(..., gt=0, le=120, examples=[24], description="Months with bank")
    monthly_charges: float = Field(..., gt=0, examples=[65.5], description="Monthly fee (USD)")
    total_charges:   float = Field(..., gt=0, examples=[1572.0], description="Cumulative charges (USD)")
    contract:        str  = Field(..., examples=["Month-to-month"],
                                  description="Month-to-month | One year | Two year")
    payment_method:  str  = Field(..., examples=["Electronic check"],
                                  description="Electronic check | Mailed check | Bank transfer | Credit card")

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v):
        if v not in ("Male", "Female"):
            raise ValueError("gender must be 'Male' or 'Female'")
        return v

    @field_validator("contract")
    @classmethod
    def validate_contract(cls, v):
        valid = ("Month-to-month", "One year", "Two year")
        if v not in valid:
            raise ValueError(f"contract must be one of {valid}")
        return v

    @field_validator("payment_method")
    @classmethod
    def validate_payment(cls, v):
        valid = ("Electronic check", "Mailed check", "Bank transfer", "Credit card")
        if v not in valid:
            raise ValueError(f"payment_method must be one of {valid}")
        return v


class PredictionResponse(BaseModel):
    churn:          bool
    churn_label:    str
    probability:    float          # probability of churning (class 1)
    confidence_pct: float          # human-readable percentage
    model_used:     str
    risk_level:     str            # Low / Medium / High
    top_factors:    list[str]


# ── Helper: build feature vector ───────────────────────────────────────────────
def build_feature_vector(inp: CustomerInput) -> np.ndarray:
    label_maps = _bundle["label_maps"]

    gender         = label_maps["Gender"][inp.gender]
    contract       = label_maps["Contract"][inp.contract]
    payment_method = label_maps["PaymentMethod"][inp.payment_method]

    tenure  = inp.tenure
    monthly = inp.monthly_charges
    total   = inp.total_charges

    # Engineered features (must match train_model.py exactly)
    charges_per_month       = total / tenure
    charges_tenure_interact = monthly * tenure

    feature_vec = np.array([[
        gender,
        inp.senior_citizen,
        tenure,
        monthly,
        total,
        contract,
        payment_method,
        charges_per_month,
        charges_tenure_interact,
    ]])

    return _bundle["scaler"].transform(feature_vec)


def risk_label(prob: float) -> str:
    if prob < 0.35:  return "Low"
    if prob < 0.60:  return "Medium"
    return "High"


def top_factors(inp: CustomerInput, prob: float) -> list[str]:
    """Simple heuristic explanations that mirror the feature-importance ranking."""
    factors = []
    if inp.contract == "Month-to-month":
        factors.append("Month-to-month contract (highest churn risk)")
    if inp.monthly_charges > 80:
        factors.append(f"High monthly charges (${inp.monthly_charges:.0f})")
    if inp.tenure < 12:
        factors.append(f"Short tenure ({inp.tenure:.0f} months)")
    if inp.senior_citizen:
        factors.append("Senior citizen")
    if inp.payment_method == "Electronic check":
        factors.append("Electronic check payment (correlates with churn)")
    if not factors:
        factors.append("No strong individual risk factors identified")
    return factors[:3]


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Bank Churn Prediction API", "docs": "/docs"}


@app.get("/model-info")
def model_info():
    return {
        "model_name":   _bundle.get("model_name"),
        "feature_cols": _bundle.get("feature_cols"),
        "model_path":   MODEL_PATH,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    try:
        X = build_feature_vector(customer)
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Encoding error: {e}")

    model    = _bundle["model"]
    prob     = float(model.predict_proba(X)[0][1])
    churn    = prob >= 0.5

    return PredictionResponse(
        churn          = churn,
        churn_label    = "Will Churn" if churn else "Will NOT Churn",
        probability    = round(prob, 4),
        confidence_pct = round(prob * 100, 1),
        model_used     = _bundle["model_name"],
        risk_level     = risk_label(prob),
        top_factors    = top_factors(customer, prob),
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(_bundle)}
