from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

from common import BASE_DIR, OUTPUT_DIR, evaluation_row


MODEL_FILES = {
    "IForest": OUTPUT_DIR / "kaggle_iforest_scores.csv",
    "LOF": OUTPUT_DIR / "kaggle_lof_scores.csv",
    "OCSVM": OUTPUT_DIR / "kaggle_ocsvm_scores.csv",
}


def main() -> None:
    logging.info("Evaluating Kaggle anomaly models")
    start = time.time()
    rows = []
    for name, path in MODEL_FILES.items():
        df = pd.read_csv(path)
        score_col = {
            "IForest": "IForest_Risk",
            "LOF": "LOF_Risk",
            "OCSVM": "OCSVM_Risk",
        }[name]
        row = evaluation_row(name, df["Fraud_Label"], df[score_col])
        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    logging.info("Saved model comparison metrics to %s", OUTPUT_DIR / "model_comparison.csv")
    logging.info("Evaluation completed in %.2f seconds", time.time() - start)
    print(metrics_df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
