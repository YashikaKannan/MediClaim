from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"


def build_provider_explanations() -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.info("Generating provider-level explanations for Kaggle risk scoring")

    risk_df = pd.read_csv(OUTPUT_DIR / "final_provider_risk_scores.csv")
    feature_df = pd.read_csv(BASE_DIR / "processed" / "PROVIDER_ML_READY_KAGGLE.csv")

    merged = risk_df.merge(feature_df[["Provider", "Total_Reimbursement", "Total_Claim_Count", "Claims_Per_Beneficiary"]], on="Provider", how="left")
    merged["Total_Reimbursement"] = pd.to_numeric(merged["Total_Reimbursement"], errors="coerce").fillna(0.0)
    merged["Total_Claim_Count"] = pd.to_numeric(merged["Total_Claim_Count"], errors="coerce").fillna(0.0)
    merged["Claims_Per_Beneficiary"] = pd.to_numeric(merged["Claims_Per_Beneficiary"], errors="coerce").fillna(0.0)

    explanations = []
    for _, row in merged.iterrows():
        reasons = []
        if row["IForest_Risk"] >= 90:
            reasons.append(f"Provider is in the {row['IForest_Risk']:.1f}th percentile of Isolation Forest scores.")
        if row["LOF_Risk"] >= 90:
            reasons.append(f"Provider is in the {row['LOF_Risk']:.1f}th percentile of LOF anomaly scores.")
        if row["Autoencoder_Risk"] >= 80:
            reasons.append(f"Provider has {row['Autoencoder_Risk']:.1f} risk from the Autoencoder reconstruction profile.")
        if row["Peer_Risk"] >= 80:
            reasons.append(f"Provider exceeds peer benchmark thresholds with a peer risk of {row['Peer_Risk']:.1f}.")
        if row["Total_Reimbursement"] > 0:
            reimbursement_ratio = row["Total_Reimbursement"] / (merged["Total_Reimbursement"].median() + 1e-8)
            if reimbursement_ratio > 3.5:
                reasons.append(f"Provider reimbursement is {reimbursement_ratio:.1f}x the median reimbursement level.")
        if row["Claims_Per_Beneficiary"] > merged["Claims_Per_Beneficiary"].median() * 2:
            reasons.append(f"Provider claim intensity is {row['Claims_Per_Beneficiary']:.2f} claims per beneficiary, well above the cohort median.")
        if row["Total_Claim_Count"] > merged["Total_Claim_Count"].median() * 2:
            reasons.append(f"Provider has a substantially elevated claim volume ({row['Total_Claim_Count']:.0f} total claims).")

        if not reasons:
            reasons.append("Provider profile remains within the normal provider risk range but is being monitored for trend escalation.")

        top_reasons = reasons[:3]
        explanation_text = " | ".join(top_reasons)
        explanations.append({
            "Provider": row["Provider"],
            "Fraud_Label": row["Fraud_Label"],
            "Final_Risk_Score": round(float(row["Final_Risk_Score"]), 2),
            "Risk_Level": row["Risk_Level"],
            "Priority_Score": round(float(row["Priority_Score"]), 2),
            "Explanation": explanation_text,
            "Top_Reasons": " | ".join(top_reasons),
        })

    out = pd.DataFrame(explanations)
    out = out.sort_values(["Final_Risk_Score", "Priority_Score"], ascending=[False, False]).reset_index(drop=True)
    output_path = OUTPUT_DIR / "provider_explanations.csv"
    out.to_csv(output_path, index=False)
    logging.info("Saved provider explanations to %s", output_path)
    return out


if __name__ == "__main__":
    build_provider_explanations()
