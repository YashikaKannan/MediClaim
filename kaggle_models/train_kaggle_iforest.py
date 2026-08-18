from __future__ import annotations

import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from common import BASE_DIR, OUTPUT_DIR, load_kaggle_data, prepare_feature_matrix, percentile_risk


MODEL_PATH = BASE_DIR / "outputs" / "kaggle_iforest_model.joblib"
SCORES_PATH = OUTPUT_DIR / "kaggle_iforest_scores.csv"


def main() -> None:
    start = time.time()
    logging.info("Starting Isolation Forest training for Kaggle features")
    df = load_kaggle_data()
    X = prepare_feature_matrix(df)
    logging.info("Feature count: %s", X.shape[1])

    model = IsolationForest(
        n_estimators=300,
        random_state=42,
        contamination="auto",
    )
    model.fit(X)
    joblib.dump(model, MODEL_PATH)

    raw_score = -model.decision_function(X)
    scores = pd.Series(raw_score, index=df.index)
    risk = percentile_risk(scores)

    out = df[["Provider", "Fraud_Label"]].copy()
    out["IForest_Score"] = scores.values
    out["IForest_Risk"] = risk.values
    out = out.sort_values("IForest_Risk", ascending=False)
    out.to_csv(SCORES_PATH, index=False)
    logging.info("Training completed in %.2f seconds", time.time() - start)
    logging.info("Saved Isolation Forest scores to %s", SCORES_PATH)
    logging.info("Saved model to %s", MODEL_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
