import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
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

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "results",
    "improved_provider_investigation_queue.csv"
)

VALIDATION_PATH = os.path.join(
    BASE_DIR,
    "validation",
    "improved_ranking_validation.csv"
)

print("=" * 70)
print("IMPROVED AUTOENCODER PROVIDER RANKING")
print("=" * 70)


# ============================================================
# 1. LOAD CLAIM RESULTS
# ============================================================

print("\n[1] Loading claim-level Autoencoder results...")

claims = pd.read_csv(CLAIM_RESULTS_PATH)

print("Total claims:", len(claims))
print("Providers:", claims["Provider"].nunique())


# ============================================================
# 2. PROVIDER-LEVEL AGGREGATION
# ============================================================

print("\n[2] Creating richer provider-level features...")

provider = (
    claims.groupby("Provider")
    .agg(
        TotalClaims=("ClaimID", "count"),
        AnomalousClaims=("IsAnomaly", "sum"),

        MeanReconstructionError=(
            "ReconstructionError",
            "mean"
        ),

        MedianReconstructionError=(
            "ReconstructionError",
            "median"
        ),

        P95ReconstructionError=(
            "ReconstructionError",
            lambda x: np.percentile(x, 95)
        ),

        MaxReconstructionError=(
            "ReconstructionError",
            "max"
        ),

        AverageRiskScore=(
            "RiskScore",
            "mean"
        ),

        P95RiskScore=(
            "RiskScore",
            lambda x: np.percentile(x, 95)
        ),

        MaximumRiskScore=(
            "RiskScore",
            "max"
        ),

        PotentialFraud=(
            "PotentialFraud",
            "first"
        )
    )
    .reset_index()
)


# ============================================================
# 3. ANOMALY RATE
# ============================================================

provider["AnomalyRate"] = (
    provider["AnomalousClaims"]
    /
    provider["TotalClaims"]
) * 100


# ============================================================
# 4. PERCENTILE RANKING
# ============================================================

print("\n[3] Creating normalized provider risk indicators...")

# Percentile ranks make the different measures comparable.
# No fraud label is used here.

provider["AnomalyRateRank"] = (
    provider["AnomalyRate"]
    .rank(pct=True)
)

provider["P95ErrorRank"] = (
    provider["P95ReconstructionError"]
    .rank(pct=True)
)

provider["MeanErrorRank"] = (
    provider["MeanReconstructionError"]
    .rank(pct=True)
)

provider["MaxErrorRank"] = (
    provider["MaxReconstructionError"]
    .rank(pct=True)
)

provider["VolumeRank"] = (
    provider["TotalClaims"]
    .rank(pct=True)
)


# ============================================================
# 5. IMPROVED PROVIDER RISK SCORE
# ============================================================

print("\n[4] Calculating improved provider risk score...")

# Fixed unsupervised weights.
# PotentialFraud is NOT used to choose the score.

provider["ProviderRiskScore"] = 100 * (

    0.40 * provider["P95ErrorRank"]

    + 0.25 * provider["MeanErrorRank"]

    + 0.20 * provider["AnomalyRateRank"]

    + 0.10 * provider["MaxErrorRank"]

    + 0.05 * provider["VolumeRank"]
)

provider["ProviderRiskScore"] = (
    provider["ProviderRiskScore"]
    .clip(0, 100)
    .round(2)
)


# ============================================================
# 6. RISK LEVEL
# ============================================================

provider["RiskLevel"] = pd.cut(
    provider["ProviderRiskScore"],
    bins=[
        -np.inf,
        50,
        70,
        85,
        np.inf
    ],
    labels=[
        "Low",
        "Medium",
        "High",
        "Critical"
    ]
)


# ============================================================
# 7. INVESTIGATION PRIORITY
# ============================================================

provider = provider.sort_values(
    by="ProviderRiskScore",
    ascending=False
).reset_index(drop=True)

provider["InvestigationRank"] = (
    np.arange(1, len(provider) + 1)
)


# ============================================================
# 8. SAVE IMPROVED QUEUE
# ============================================================

provider.to_csv(
    OUTPUT_PATH,
    index=False
)

print("[OK] Improved investigation queue created")


# ============================================================
# 9. VALIDATION
# ============================================================

print("\n[5] Evaluating improved ranking...")

evaluation = provider[
    provider["PotentialFraud"].notna()
].copy()

evaluation["FraudLabel"] = (
    evaluation["PotentialFraud"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "yes": 1,
        "no": 0
    })
)

evaluation = evaluation[
    evaluation["FraudLabel"].notna()
].copy()

evaluation["FraudLabel"] = (
    evaluation["FraudLabel"]
    .astype(int)
)

y_true = evaluation["FraudLabel"]
y_score = evaluation["ProviderRiskScore"]

roc_auc = roc_auc_score(
    y_true,
    y_score
)

pr_auc = average_precision_score(
    y_true,
    y_score
)

print("Improved Provider ROC-AUC:", round(roc_auc, 4))
print("Improved Provider PR-AUC :", round(pr_auc, 4))


# ============================================================
# 10. TOP-K PERFORMANCE
# ============================================================

print("\n[6] Top-K investigation performance...")

evaluation = evaluation.sort_values(
    "ProviderRiskScore",
    ascending=False
).reset_index(drop=True)

total_fraud = int(
    evaluation["FraudLabel"].sum()
)

results = []

for pct in [1, 5, 10, 20]:

    k = max(
        1,
        int(
            np.ceil(
                len(evaluation)
                * pct
                / 100
            )
        )
    )

    top = evaluation.head(k)

    fraud_found = int(
        top["FraudLabel"].sum()
    )

    precision = (
        fraud_found / k
    )

    recall = (
        fraud_found / total_fraud
        if total_fraud > 0
        else 0
    )

    results.append({
        "TopPercent": pct,
        "ProvidersReviewed": k,
        "FraudProvidersFound": fraud_found,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4)
    })


results_df = pd.DataFrame(results)

print(
    results_df.to_string(
        index=False
    )
)

results_df.to_csv(
    VALIDATION_PATH,
    index=False
)


# ============================================================
# 11. TOP 10 PROVIDERS
# ============================================================

print("\n[7] Top 10 providers for investigation:")

display_columns = [
    "InvestigationRank",
    "Provider",
    "TotalClaims",
    "AnomalousClaims",
    "AnomalyRate",
    "ProviderRiskScore",
    "RiskLevel",
    "PotentialFraud"
]

print(
    provider[
        display_columns
    ]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("[SUCCESS] IMPROVED PROVIDER RANKING COMPLETE")
print("=" * 70)

print("\nImproved Provider ROC-AUC:", round(roc_auc, 4))
print("Improved Provider PR-AUC :", round(pr_auc, 4))

print("\nQueue saved:")
print(OUTPUT_PATH)

print("\nValidation saved:")
print(VALIDATION_PATH)

print("\nIMPORTANT:")
print(
    "PotentialFraud was used only for evaluation, "
    "not for calculating ProviderRiskScore."
)