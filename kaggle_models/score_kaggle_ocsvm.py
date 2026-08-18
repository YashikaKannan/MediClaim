from __future__ import annotations

import logging

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from common import BASE_DIR, OUTPUT_DIR, load_kaggle_data, prepare_feature_matrix, percentile_risk

MODEL_PATH = BASE_DIR / "outputs" / "kaggle_ocsvm_model.joblib"
SCALER_PATH = BASE_DIR / "outputs" / "kaggle_ocsvm_scaler.joblib"
OUTPUT_PATH = OUTPUT_DIR / "kaggle_ocsvm_scores.csv"


def main() -> None:
    logging.info("Scoring Kaggle One-Class SVM")
    df = load_kaggle_data()
    X = prepare_feature_matrix(df)
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)

    X_scaled = scaler.transform(X)
    scores = -model.score_samples(X_scaled)
    risk = percentile_risk(pd.Series(scores, index=df.index))

    out = df[["Provider", "Fraud_Label"]].copy()
    out["OCSVM_Score"] = scores
    out["OCSVM_Risk"] = risk.values
    out = out.sort_values("OCSVM_Risk", ascending=False)
    out.to_csv(OUTPUT_PATH, index=False)
    logging.info("Saved %s", OUTPUT_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
