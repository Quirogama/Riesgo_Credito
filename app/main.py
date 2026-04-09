from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT_DIR / "app" / "static"
MODEL_PATH = ROOT_DIR / "models" / "xgb_credit_pipeline.joblib"
METRICS_PATH = ROOT_DIR / "models" / "metrics.json"


class LoanRequest(BaseModel):
    LIMIT_BAL: float = Field(..., description="Amount of given credit (NT dollar)")
    SEX: int = Field(..., ge=1, le=2, description="1=male, 2=female")
    EDUCATION: int = Field(..., ge=0, le=6)
    MARRIAGE: int = Field(..., ge=0, le=3)
    AGE: int = Field(..., ge=18, le=100)
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float


app = FastAPI(
    title="Credit Risk Scoring API",
    description="XGBoost + SHAP model to estimate loan default probability",
    version="1.0.0",
)

MODEL = None
METRICS = None
EXPLAINER = None


@app.on_event("startup")
def load_artifacts() -> None:
    global MODEL, METRICS, EXPLAINER

    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        raise RuntimeError(
            "Model artifacts not found. Run `python -m src.train` before starting the API."
        )

    MODEL = joblib.load(MODEL_PATH)
    METRICS = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    EXPLAINER = shap.TreeExplainer(MODEL.named_steps["classifier"])


def payload_to_frame(payload: LoanRequest) -> pd.DataFrame:
    sample = payload.model_dump()
    return pd.DataFrame([sample])


@app.get("/")
def health() -> dict:
    return {
        "status": "ok",
        "message": "Credit risk API running",
        "auc_test": METRICS.get("auc") if METRICS else None,
        "demo_url": "/demo",
        "docs_url": "/docs",
    }


@app.get("/demo", include_in_schema=False)
def demo_ui() -> FileResponse:
    demo_file = STATIC_DIR / "demo.html"
    if not demo_file.exists():
        raise HTTPException(status_code=404, detail="Demo UI not found")
    return FileResponse(demo_file)


@app.get("/model-info")
def model_info() -> dict:
    return METRICS


@app.post("/predict")
def predict(payload: LoanRequest) -> dict:
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    sample_df = payload_to_frame(payload)
    probability_default = float(MODEL.predict_proba(sample_df)[0, 1])
    label = int(probability_default >= METRICS.get("threshold", 0.5))

    return {
        "probability_default": probability_default,
        "predicted_label": label,
        "risk_level": "high" if label == 1 else "low",
        "threshold": METRICS.get("threshold", 0.5),
    }


@app.post("/explain")
def explain(payload: LoanRequest, top_k: int = 8) -> dict:
    if MODEL is None or EXPLAINER is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    sample_df = payload_to_frame(payload)
    transformed = MODEL.named_steps["preprocessor"].transform(sample_df)
    transformed_names = MODEL.named_steps["preprocessor"].get_feature_names_out().tolist()

    shap_values = EXPLAINER.shap_values(transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    values = np.array(shap_values).reshape(-1)
    indices = np.argsort(np.abs(values))[::-1][:top_k]

    top_contributors = [
        {
            "feature": transformed_names[i],
            "shap_value": float(values[i]),
            "direction": "increases_risk" if values[i] > 0 else "decreases_risk",
        }
        for i in indices
    ]

    expected_value = EXPLAINER.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(np.array(expected_value).ravel()[-1])
    else:
        expected_value = float(expected_value)

    return {
        "base_value": expected_value,
        "top_contributors": top_contributors,
    }
