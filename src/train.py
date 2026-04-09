from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import shap
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

RANDOM_STATE = 42
DATASET_NAME = "default-of-credit-card-clients"
DATASET_VERSION = 1
TARGET_COLUMN = "default payment next month"
CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]
RAW_TO_BUSINESS_COLUMNS = {
    "x1": "LIMIT_BAL",
    "x2": "SEX",
    "x3": "EDUCATION",
    "x4": "MARRIAGE",
    "x5": "AGE",
    "x6": "PAY_0",
    "x7": "PAY_2",
    "x8": "PAY_3",
    "x9": "PAY_4",
    "x10": "PAY_5",
    "x11": "PAY_6",
    "x12": "BILL_AMT1",
    "x13": "BILL_AMT2",
    "x14": "BILL_AMT3",
    "x15": "BILL_AMT4",
    "x16": "BILL_AMT5",
    "x17": "BILL_AMT6",
    "x18": "PAY_AMT1",
    "x19": "PAY_AMT2",
    "x20": "PAY_AMT3",
    "x21": "PAY_AMT4",
    "x22": "PAY_AMT5",
    "x23": "PAY_AMT6",
}

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "xgb_credit_pipeline.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"


def load_data():
    dataset = fetch_openml(
        name=DATASET_NAME,
        version=DATASET_VERSION,
        as_frame=True,
        parser="auto",
    )
    frame = dataset.frame.copy()

    target_column = TARGET_COLUMN
    if target_column not in frame.columns:
        target_column = dataset.target_names[0]

    y = frame[target_column].astype(int)
    X = frame.drop(columns=[target_column])

    # OpenML can expose this dataset as x1..x23 and y; map to business names.
    maybe_raw_columns = {column.lower() for column in X.columns}
    if all(raw in maybe_raw_columns for raw in RAW_TO_BUSINESS_COLUMNS):
        rename_map = {
            column: RAW_TO_BUSINESS_COLUMNS[column.lower()]
            for column in X.columns
            if column.lower() in RAW_TO_BUSINESS_COLUMNS
        }
        X = X.rename(columns=rename_map)

    # Keep names stable for API payloads and artifact metadata.
    X.columns = [column.strip().upper() for column in X.columns]

    return X, y, target_column


def build_pipeline(feature_columns: list[str]) -> Pipeline:
    numeric_features = [
        column for column in feature_columns if column not in CATEGORICAL_FEATURES
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", numeric_features),
        ]
    )

    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=4,
        subsample=0.85,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def compute_global_shap(pipeline: Pipeline, X_sample):
    transformed = pipeline.named_steps["preprocessor"].transform(X_sample)
    model = pipeline.named_steps["classifier"]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranking_idx = np.argsort(mean_abs_shap)[::-1]

    top_features = [
        {"feature": feature_names[idx], "mean_abs_shap": float(mean_abs_shap[idx])}
        for idx in ranking_idx[:20]
    ]
    return top_features


def train_and_save() -> None:
    X, y, target_column = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = build_pipeline(X.columns.tolist())
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    top_features = compute_global_shap(pipeline, X_test.sample(n=3000, random_state=RANDOM_STATE))

    metrics = {
        "dataset": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "auc": float(auc),
        "target_column": target_column,
        "threshold": 0.5,
        "features": X.columns.tolist(),
        "top_shap_features": top_features,
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"AUC test: {auc:.4f}")


if __name__ == "__main__":
    train_and_save()
