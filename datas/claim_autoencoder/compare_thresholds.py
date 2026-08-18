import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLAIM_PATH = os.path.join(
    BASE_DIR, "results", "claim_anomaly_results.csv"
)

print("=" * 75)
print("AUTOENCODER THRESHOLD COMPARISON")
print("=" * 75)

df = pd.read_csv(CLAIM_PATH)

# Convert provider label to 0/1
df["FraudLabel"] = (
    df["PotentialFraud"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"yes": 1, "no": 0})
)

percentiles = [95, 97, 98, 99, 99.5]

results = []

for percentile in percentiles:

    threshold = np.percentile(
        df["ReconstructionError"],
        percentile
    )

    # Claim anomaly based on this threshold
    df["TempAnomaly"] = (
        df["ReconstructionError"] > threshold
    )

    flagged_claims = int(df["TempAnomaly"].sum())

    flagged_percentage = (
        flagged_claims / len(df) * 100
    )

    # Provider is high-risk if it has >= 1 anomalous claim
    provider = (
        df.groupby("Provider")
        .agg(
            FraudLabel=("FraudLabel", "first"),
            AnomalousClaims=("TempAnomaly", "sum")
        )
        .reset_index()
    )

    provider["PredictedHighRisk"] = (
        provider["AnomalousClaims"] > 0
    ).astype(int)

    y_true = provider["FraudLabel"]
    y_pred = provider["PredictedHighRisk"]

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    fraud_detected = int(
        provider.loc[
            provider["PredictedHighRisk"] == 1,
            "FraudLabel"
        ].sum()
    )

    high_risk_providers = int(
        provider["PredictedHighRisk"].sum()
    )

    results.append({
        "Percentile": percentile,
        "Threshold": threshold,
        "FlaggedClaims": flagged_claims,
        "FlaggedClaimPct": flagged_percentage,
        "HighRiskProviders": high_risk_providers,
        "FraudProvidersDetected": fraud_detected,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })


results_df = pd.DataFrame(results)

print("\nRESULTS\n")

print(
    results_df.to_string(
        index=False,
        formatters={
            "Threshold": "{:.6f}".format,
            "FlaggedClaimPct": "{:.2f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1": "{:.4f}".format
        }
    )
)

# Find highest F1 threshold
best = results_df.loc[
    results_df["F1"].idxmax()
]

print("\n" + "=" * 75)
print("BEST THRESHOLD BY F1")
print("=" * 75)

print("Percentile:", best["Percentile"])
print("Threshold :", round(best["Threshold"], 6))
print("Precision :", round(best["Precision"], 4))
print("Recall    :", round(best["Recall"], 4))
print("F1 Score  :", round(best["F1"], 4))
print(
    "Fraud providers detected:",
    int(best["FraudProvidersDetected"]),
    "/ 506"
)

output_path = os.path.join(
    BASE_DIR,
    "validation",
    "threshold_comparison.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print("\nSaved:")
print(output_path)

print("\n[SUCCESS] THRESHOLD COMPARISON COMPLETE")