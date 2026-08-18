from __future__ import annotations

import logging
import time

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from common import BASE_DIR, OUTPUT_DIR, load_kaggle_data, prepare_feature_matrix, percentile_risk

MODEL_PATH = BASE_DIR / "outputs" / "kaggle_ocsvm_model.joblib"
SCORES_PATH = OUTPUT_DIR / "kaggle_ocsvm_scores.csv"


def main() -> None:
    start = time.time()
    logging.info("Starting One-Class SVM training for Kaggle features")
    df = load_kaggle_data()
    X = prepare_feature_matrix(df)
    logging.info("Feature count: %s", X.shape[1])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=0.10,
    )
    model.fit(X_scaled)

    scores = -model.score_samples(X_scaled)
    risk = percentile_risk(pd.Series(scores, index=df.index))

    out = df[["Provider", "Fraud_Label"]].copy()
    out["OCSVM_Score"] = scores
    out["OCSVM_Risk"] = risk.values
    out = out.sort_values("OCSVM_Risk", ascending=False)
    out.to_csv(SCORES_PATH, index=False)
    import joblib
    joblib.dump(scaler, BASE_DIR / "outputs" / "kaggle_ocsvm_scaler.joblib")
    joblib.dump(model, MODEL_PATH)
    logging.info("Training completed in %.2f seconds", time.time() - start)
    logging.info("Saved OCSVM scores to %s", SCORES_PATH)
    logging.info("Saved model to %s", MODEL_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
