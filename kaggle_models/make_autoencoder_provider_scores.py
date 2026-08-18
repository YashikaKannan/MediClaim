from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"


def safe_ratio(n, d):
    d = pd.Series(d).replace(0, np.nan)
    return pd.Series(n).divide(d).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.info("Creating provider-level autoencoder summary from existing outputs")

    provider_path = BASE_DIR / "autoencoder" / "results" / "provider_investigation_queue.csv"
    if not provider_path.exists():
        provider_path = BASE_DIR / "claim_autoencoder" / "results" / "provider_investigation_queue.csv"
    if not provider_path.exists():
        logging.warning("No provider autoencoder output found; creating zeroed adapter output")
        provider_df = pd.DataFrame(columns=["Provider", "MeanReconstructionError", "MaxReconstructionError", "AnomalousClaims", "AnomalyRate", "ProviderRiskScore"])
    else:
        provider_df = pd.read_csv(provider_path)
        if "Provider" not in provider_df.columns:
            raise ValueError(f"Autoencoder provider output missing Provider column: {provider_path}")

    if "MeanReconstructionError" not in provider_df.columns and "MeanReconstructionError" not in provider_df.columns:
        provider_df["MeanReconstructionError"] = 0.0
    if "MaxReconstructionError" not in provider_df.columns:
        provider_df["MaxReconstructionError"] = 0.0
    if "AnomalousClaims" not in provider_df.columns:
        provider_df["AnomalousClaims"] = 0
    if "AnomalyRate" not in provider_df.columns:
        provider_df["AnomalyRate"] = 0.0
    if "ProviderRiskScore" not in provider_df.columns and "AverageRiskScore" in provider_df.columns:
        provider_df["ProviderRiskScore"] = provider_df["AverageRiskScore"]
    if "ProviderRiskScore" not in provider_df.columns:
        provider_df["ProviderRiskScore"] = 0.0

    out = provider_df[["Provider", "MeanReconstructionError", "MaxReconstructionError", "AnomalousClaims", "AnomalyRate", "ProviderRiskScore"]].copy()
    out = out.rename(columns={
        "AnomalousClaims": "Anomalous_Claim_Count",
        "AnomalyRate": "Anomalous_Claim_Percentage",
        "ProviderRiskScore": "Autoencoder_Risk",
        "MeanReconstructionError": "Mean_Reconstruction_Error",
        "MaxReconstructionError": "Max_Reconstruction_Error",
    })
    out["Mean_Reconstruction_Error"] = pd.to_numeric(out["Mean_Reconstruction_Error"], errors="coerce").fillna(0.0)
    out["Max_Reconstruction_Error"] = pd.to_numeric(out["Max_Reconstruction_Error"], errors="coerce").fillna(0.0)
    out["Anomalous_Claim_Count"] = pd.to_numeric(out["Anomalous_Claim_Count"], errors="coerce").fillna(0.0)
    out["Anomalous_Claim_Percentage"] = pd.to_numeric(out["Anomalous_Claim_Percentage"], errors="coerce").fillna(0.0)
    out["Autoencoder_Risk"] = pd.to_numeric(out["Autoencoder_Risk"], errors="coerce").fillna(0.0).clip(0, 100)

    output_path = OUTPUT_DIR / "autoencoder_provider_scores.csv"
    out.to_csv(output_path, index=False)
    logging.info("Saved %s", output_path)


if __name__ == "__main__":
    main()
