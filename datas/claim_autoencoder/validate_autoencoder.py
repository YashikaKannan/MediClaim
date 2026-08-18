import os
import pandas as pd
import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLAIM_RESULTS_PATH = os.path.join(
    BASE_DIR,
    "results",
    "claim_anomaly_results.csv"
)

PROVIDER_RESULTS_PATH = os.path.join(
    BASE_DIR,
    "results",
    "provider_investigation_queue.csv"
)

VALIDATION_DIR = os.path.join(
    BASE_DIR,
    "validation"
)

os.makedirs(VALIDATION_DIR, exist_ok=True)

print("=" * 70)
print("AUTOENCODER VALIDATION")
print("=" * 70)


# ============================================================
# 1. LOAD RESULTS
# ============================================================

print("\n[1] Loading Autoencoder results...")

claims = pd.read_csv(CLAIM_RESULTS_PATH)
providers = pd.read_csv(PROVIDER_RESULTS_PATH)

print("Claim results:", claims.shape)
print("Provider results:", providers.shape)


# ============================================================
# 2. PREPARE FRAUD LABELS
# ============================================================

print("\n[2] Preparing provider fraud labels...")

providers = providers[
    providers["PotentialFraud"].notna()
].copy()

providers["FraudLabel"] = (
    providers["PotentialFraud"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "yes": 1,
        "no": 0
    })
)

providers = providers[
    providers["FraudLabel"].notna()
].copy()

providers["FraudLabel"] = providers["FraudLabel"].astype(int)

print(
    "Known fraud providers:",
    int(providers["FraudLabel"].sum())
)

print(
    "Known non-fraud providers:",
    int((providers["FraudLabel"] == 0).sum())
)


# ============================================================
# 3. COMPARE PROVIDER ANOMALY BEHAVIOR
# ============================================================

print("\n[3] Provider anomaly comparison...")

comparison = (
    providers
    .groupby("FraudLabel")
    .agg(
        Providers=("Provider", "count"),
        MeanAnomalyRate=("AnomalyRate", "mean"),
        MedianAnomalyRate=("AnomalyRate", "median"),
        MeanAverageRisk=("AverageRiskScore", "mean"),
        MeanMaxRisk=("MaximumRiskScore", "mean")
    )
)

print(comparison)


# ============================================================
# 4. PROVIDER RANKING METRICS
# ============================================================

print("\n[4] Calculating provider-level ranking metrics...")

y_true = providers["FraudLabel"].values

# Main continuous provider score:
# anomaly rate + small contribution from average risk
provider_score = (
    providers["AnomalyRate"].values
    +
    0.1 * providers["AverageRiskScore"].values
)

if len(np.unique(y_true)) == 2:

    roc_auc = roc_auc_score(
        y_true,
        provider_score
    )

    pr_auc = average_precision_score(
        y_true,
        provider_score
    )

    print("Provider ROC-AUC:", round(roc_auc, 4))
    print("Provider PR-AUC :", round(pr_auc, 4))

else:

    roc_auc = np.nan
    pr_auc = np.nan

    print("ROC-AUC / PR-AUC cannot be calculated.")


# ============================================================
# 5. TOP-K INVESTIGATION PERFORMANCE
# ============================================================

print("\n[5] Evaluating prioritized provider queue...")

providers = providers.sort_values(
    by=[
        "AnomalyRate",
        "AverageRiskScore"
    ],
    ascending=False
).reset_index(drop=True)

total_fraud = providers["FraudLabel"].sum()

top_percentages = [
    1,
    5,
    10,
    20
]

topk_rows = []

for pct in top_percentages:

    k = max(
        1,
        int(
            np.ceil(
                len(providers) * pct / 100
            )
        )
    )

    top = providers.head(k)

    fraud_found = int(
        top["FraudLabel"].sum()
    )

    precision_at_k = (
        fraud_found / k
        if k > 0
        else 0
    )

    recall_at_k = (
        fraud_found / total_fraud
        if total_fraud > 0
        else 0
    )

    topk_rows.append({
        "TopPercent": pct,
        "ProvidersReviewed": k,
        "FraudProvidersFound": fraud_found,
        "Precision": round(precision_at_k, 4),
        "Recall": round(recall_at_k, 4)
    })

topk_df = pd.DataFrame(topk_rows)

print(topk_df.to_string(index=False))


# ============================================================
# 6. SIMPLE BINARY VALIDATION
# ============================================================

print("\n[6] Binary provider flag validation...")

# A provider is treated as high-risk if it has
# at least one anomalous claim.

providers["PredictedHighRisk"] = (
    providers["AnomalousClaims"] > 0
).astype(int)

precision = precision_score(
    providers["FraudLabel"],
    providers["PredictedHighRisk"],
    zero_division=0
)

recall = recall_score(
    providers["FraudLabel"],
    providers["PredictedHighRisk"],
    zero_division=0
)

f1 = f1_score(
    providers["FraudLabel"],
    providers["PredictedHighRisk"],
    zero_division=0
)

cm = confusion_matrix(
    providers["FraudLabel"],
    providers["PredictedHighRisk"]
)

print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 7. CLAIM-LEVEL SUPPORTING ANALYSIS
# ============================================================

print("\n[7] Comparing anomalous claims by provider label...")

claims["FraudLabel"] = (
    claims["PotentialFraud"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "yes": 1,
        "no": 0
    })
)

claim_summary = (
    claims
    .dropna(subset=["FraudLabel"])
    .groupby("FraudLabel")
    .agg(
        TotalClaims=("ClaimID", "count"),
        FlaggedClaims=("IsAnomaly", "sum"),
        MeanRiskScore=("RiskScore", "mean"),
        MeanReconstructionError=("ReconstructionError", "mean")
    )
)

claim_summary["FlaggedPercentage"] = (
    claim_summary["FlaggedClaims"]
    /
    claim_summary["TotalClaims"]
    * 100
)

print(claim_summary)


# ============================================================
# 8. SAVE VALIDATION RESULTS
# ============================================================

topk_path = os.path.join(
    VALIDATION_DIR,
    "provider_topk_validation.csv"
)

topk_df.to_csv(
    topk_path,
    index=False
)

provider_validation_path = os.path.join(
    VALIDATION_DIR,
    "provider_validation_results.csv"
)

providers.to_csv(
    provider_validation_path,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("[SUCCESS] AUTOENCODER VALIDATION COMPLETE")
print("=" * 70)

print("\nProvider ROC-AUC:", round(roc_auc, 4))
print("Provider PR-AUC :", round(pr_auc, 4))

print("\nBinary metrics:")
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

print("\nValidation outputs saved in:")
print(VALIDATION_DIR)