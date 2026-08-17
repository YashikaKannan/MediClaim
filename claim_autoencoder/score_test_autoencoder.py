import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_DATA_PATH = os.path.join(
    BASE_DIR,
    "processed",
    "autoencoder_test_claim_dataset.csv"
)

TRAIN_RESULTS_PATH = os.path.join(
    BASE_DIR,
    "results",
    "claim_anomaly_results.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "claim_autoencoder.keras"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "model",
    "robust_scaler.pkl"
)

FEATURES_PATH = os.path.join(
    BASE_DIR,
    "model",
    "features.json"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(RESULT_DIR, exist_ok=True)

print("=" * 70)
print("UNSEEN TEST DATA - AUTOENCODER SCORING")
print("=" * 70)


# ============================================================
# 1. LOAD TEST DATA
# ============================================================

print("\n[1] Loading unseen Test claims...")

df = pd.read_csv(TEST_DATA_PATH)

print("Test dataset shape:", df.shape)
print("Test claims:", len(df))
print("Test providers:", df["Provider"].nunique())


# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

print("\n[2] Loading trained Autoencoder and scaler...")

model = tf.keras.models.load_model(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

with open(FEATURES_PATH, "r") as f:
    FEATURES = json.load(f)

print("Model loaded successfully")
print("Features:", len(FEATURES))


# ============================================================
# 3. PREPARE TEST FEATURES
# ============================================================

print("\n[3] Preparing Test features...")

X = df[FEATURES].copy()

X = X.replace(
    [np.inf, -np.inf],
    0
)

X = X.fillna(0)

# Same transformation used during training
X_log = np.log1p(X)

# IMPORTANT:
# transform only - DO NOT fit scaler again
X_scaled = scaler.transform(X_log)

print("[OK] Same preprocessing applied")


# ============================================================
# 4. AUTOENCODER PREDICTION
# ============================================================

print("\n[4] Reconstructing unseen Test claims...")

X_pred = model.predict(
    X_scaled,
    batch_size=1024,
    verbose=1
)


# ============================================================
# 5. RECONSTRUCTION ERROR
# ============================================================

print("\n[5] Calculating reconstruction errors...")

reconstruction_error = np.mean(
    np.square(
        X_scaled - X_pred
    ),
    axis=1
)

df["ReconstructionError"] = reconstruction_error


# ============================================================
# 6. GET THRESHOLD FROM TRAIN DATA
# ============================================================

print("\n[6] Loading Train anomaly threshold...")

train_results = pd.read_csv(
    TRAIN_RESULTS_PATH,
    usecols=["ReconstructionError"]
)

TRAIN_THRESHOLD = np.percentile(
    train_results["ReconstructionError"],
    99
)

print(
    "Train 99th percentile threshold:",
    TRAIN_THRESHOLD
)

# Apply TRAIN threshold to unseen Test claims
df["IsAnomaly"] = (
    df["ReconstructionError"]
    > TRAIN_THRESHOLD
)


# ============================================================
# 7. RISK SCORE
# ============================================================

print("\n[7] Generating Test risk scores...")

# Percentile rank gives an interpretable 0-100
# prioritization score within the Test population.
df["RiskScore"] = (
    df["ReconstructionError"]
    .rank(
        method="average",
        pct=True
    )
    * 100
)

df["RiskScore"] = (
    df["RiskScore"]
    .round(2)
)


# ============================================================
# 8. EXPLAINABILITY
# ============================================================

print("\n[8] Generating investigation reasons...")

feature_medians = df[FEATURES].median()


def generate_reason(row):

    reasons = []

    if (
        row["InscClaimAmtReimbursed"]
        > feature_medians["InscClaimAmtReimbursed"] * 3
    ):
        reasons.append(
            "Unusually high reimbursement"
        )

    if (
        row["DeductibleAmtPaid"]
        > feature_medians["DeductibleAmtPaid"] * 3
    ):
        reasons.append(
            "High deductible amount"
        )

    if (
        row["ClaimDuration"]
        > feature_medians["ClaimDuration"] * 3
    ):
        reasons.append(
            "Unusually long claim duration"
        )

    if (
        row["DiagnosisCount"]
        > feature_medians["DiagnosisCount"] * 2
    ):
        reasons.append(
            "High diagnosis count"
        )

    if (
        row["ProcedureCount"]
        > feature_medians["ProcedureCount"] * 2
    ):
        reasons.append(
            "High procedure count"
        )

    if row["HospitalStayDays"] > 15:
        reasons.append(
            "Long hospital stay"
        )

    if not reasons:
        reasons.append(
            "Unusual combination of claim features"
        )

    return "; ".join(reasons)


df["Reason"] = df.apply(
    generate_reason,
    axis=1
)


# ============================================================
# 9. SAVE CLAIM-LEVEL TEST RESULTS
# ============================================================

claim_columns = [
    "ClaimID",
    "BeneID",
    "Provider",
    "ClaimType",
    "ReconstructionError",
    "RiskScore",
    "IsAnomaly",
    "Reason"
]

test_claim_results = df[
    claim_columns
].copy()

CLAIM_OUTPUT_PATH = os.path.join(
    RESULT_DIR,
    "test_claim_anomaly_results.csv"
)

test_claim_results.to_csv(
    CLAIM_OUTPUT_PATH,
    index=False
)


# ============================================================
# 10. PROVIDER AGGREGATION
# ============================================================

print("\n[9] Creating Test provider investigation queue...")

provider = (
    df.groupby("Provider")
    .agg(
        TotalClaims=(
            "ClaimID",
            "count"
        ),

        AnomalousClaims=(
            "IsAnomaly",
            "sum"
        ),

        MeanReconstructionError=(
            "ReconstructionError",
            "mean"
        ),

        P95ReconstructionError=(
            "ReconstructionError",
            lambda x: np.percentile(x, 95)
        ),

        MaxReconstructionError=(
            "ReconstructionError",
            "max"
        ),

        AverageClaimRisk=(
            "RiskScore",
            "mean"
        )
    )
    .reset_index()
)


provider["AnomalyRate"] = (
    provider["AnomalousClaims"]
    /
    provider["TotalClaims"]
    * 100
)


# ============================================================
# 11. IMPROVED PROVIDER SCORE
# ============================================================

# Same UNSUPERVISED weighting logic used for
# improved Train provider ranking.

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
# 12. RISK LEVEL
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
# 13. INVESTIGATION RANK
# ============================================================

provider = provider.sort_values(
    "ProviderRiskScore",
    ascending=False
).reset_index(drop=True)

provider["InvestigationRank"] = (
    np.arange(
        1,
        len(provider) + 1
    )
)


PROVIDER_OUTPUT_PATH = os.path.join(
    RESULT_DIR,
    "test_provider_investigation_queue.csv"
)

provider.to_csv(
    PROVIDER_OUTPUT_PATH,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

total_claims = len(df)

flagged_claims = int(
    df["IsAnomaly"].sum()
)

flagged_percentage = (
    flagged_claims
    /
    total_claims
    * 100
)

providers_with_anomalies = int(
    (provider["AnomalousClaims"] > 0)
    .sum()
)

print("\n" + "=" * 70)
print("[SUCCESS] UNSEEN TEST SCORING COMPLETE")
print("=" * 70)

print("\nTotal Test claims:", total_claims)

print(
    "Flagged Test claims:",
    flagged_claims
)

print(
    "Flagged percentage:",
    round(flagged_percentage, 2),
    "%"
)

print(
    "Train threshold used:",
    TRAIN_THRESHOLD
)

print(
    "Total Test providers:",
    len(provider)
)

print(
    "Providers with anomalous claims:",
    providers_with_anomalies
)

print("\nTop 10 providers for investigation:")

print(
    provider[
        [
            "InvestigationRank",
            "Provider",
            "TotalClaims",
            "AnomalousClaims",
            "AnomalyRate",
            "ProviderRiskScore",
            "RiskLevel"
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\nClaim results saved:")
print(CLAIM_OUTPUT_PATH)

print("\nProvider queue saved:")
print(PROVIDER_OUTPUT_PATH)

print(
    "\nIMPORTANT: Test data was scored using "
    "the already-trained model and Train threshold."
)