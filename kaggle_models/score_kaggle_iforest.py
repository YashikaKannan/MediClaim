from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd

from common import BASE_DIR, OUTPUT_DIR, load_kaggle_data, prepare_feature_matrix, percentile_risk

MODEL_PATH = BASE_DIR / "outputs" / "kaggle_iforest_model.joblib"
OUTPUT_PATH = OUTPUT_DIR / "kaggle_iforest_scores.csv"


def main() -> None:
    logging.info("Scoring Kaggle Isolation Forest")
    df = load_kaggle_data()
    X = prepare_feature_matrix(df)
    model = joblib.load(MODEL_PATH)

    raw_score = -model.decision_function(X)
    scores = pd.Series(raw_score, index=df.index)
    risk = percentile_risk(scores)

    out = df[["Provider", "Fraud_Label"]].copy()
    out["IForest_Score"] = scores.values
    out["IForest_Risk"] = risk.values
    out = out.sort_values("IForest_Risk", ascending=False)
    out.to_csv(OUTPUT_PATH, index=False)
    logging.info("Saved %s", OUTPUT_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
