from __future__ import annotations

import logging
import time

import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

from common import BASE_DIR, OUTPUT_DIR, load_kaggle_data, prepare_feature_matrix, percentile_risk

MODEL_PATH = BASE_DIR / "outputs" / "kaggle_lof_model.joblib"
SCORES_PATH = OUTPUT_DIR / "kaggle_lof_scores.csv"


def main() -> None:
    start = time.time()
    logging.info("Starting LOF training for Kaggle features")
    df = load_kaggle_data()
    X = prepare_feature_matrix(df)
    logging.info("Feature count: %s", X.shape[1])

    model = LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.10,
        novelty=False,
    )
    model.fit_predict(X)
    raw_score = -model.negative_outlier_factor_
    scores = pd.Series(raw_score, index=df.index)
    risk = percentile_risk(scores)

    out = df[["Provider", "Fraud_Label"]].copy()
    out["LOF_Score"] = scores.values
    out["LOF_Risk"] = risk.values
    out = out.sort_values("LOF_Risk", ascending=False)
    out.to_csv(SCORES_PATH, index=False)
    joblib = __import__("joblib")
    joblib.dump(model, MODEL_PATH)
    logging.info("Training completed in %.2f seconds", time.time() - start)
    logging.info("Saved LOF scores to %s", SCORES_PATH)
    logging.info("Saved model to %s", MODEL_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
