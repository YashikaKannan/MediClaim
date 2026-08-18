from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "processed" / "PROVIDER_ML_READY_KAGGLE.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_kaggle_data() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Kaggle dataset not found: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    logging.info("Dataset shape: %s", df.shape)
    return df


def prepare_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    feature_df = df.drop(columns=["Provider", "Fraud_Label"], errors="ignore")
    feature_df = feature_df.copy()

    for col in feature_df.columns:
        if feature_df[col].dtype == object:
            feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")

    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_df = feature_df.loc[:, feature_df.nunique(dropna=True) > 1]
    return feature_df


def percentile_risk(scores: pd.Series) -> pd.Series:
    risk = scores.rank(pct=True, method="average") * 100.0
    return risk.clip(0, 100).round(4)


def evaluation_row(model_name: str, y_true: pd.Series, risk_scores: pd.Series) -> dict:
    from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

    pred = (risk_scores >= 50).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "Model": model_name,
        "ROC_AUC": roc_auc_score(y_true, risk_scores),
        "PR_AUC": average_precision_score(y_true, risk_scores),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }
