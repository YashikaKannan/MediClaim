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

DATA_PATH = os.path.join(
    BASE_DIR,
    "processed",
    "autoencoder_claim_dataset.csv"
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
print("AUTOENCODER RESULT GENERATION")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1] Loading claim dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# 2. LOAD MODEL / SCALER / FEATURES
# ============================================================

print("\n[2] Loading trained Autoencoder...")

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(FEATURES_PATH, "r") as f:
    FEATURES = json.load(f)

print("Features loaded:", len(FEATURES))


# ============================================================
# 3. PREPARE DATA
# ============================================================

print("\n[3] Preparing features...")

X = df[FEATURES].copy()

X = X.replace([np.inf, -np.inf], 0)
X = X.fillna(0)

X_log = np.log1p(X)

X_scaled = scaler.transform(X_log)


# ============================================================
# 4. AUTOENCODER RECONSTRUCTION
# ============================================================

print("\n[4] Generating reconstructions...")

X_pred = model.predict(
    X_scaled,
    batch_size=1024,
    verbose=1
)


# ============================================================
# 5. RECONSTRUCTION ERROR
# ============================================================

print("\n[5] Calculating reconstruction error...")

reconstruction_error = np.mean(
    np.square(X_scaled - X_pred),
    axis=1
)

df["ReconstructionError"] = reconstruction_error


# ============================================================
# 6. ANOMALY THRESHOLD
# ============================================================

threshold = np.percentile(
    reconstruction_error,
    99
)

df["IsAnomaly"] = (
    df["ReconstructionError"] > threshold
)


# ============================================================
# 7. RISK SCORE 0-100
# ============================================================

print("\n[6] Creating risk score...")

error_min = reconstruction_error.min()
error_max = reconstruction_error.max()

df["RiskScore"] = (
    (
        reconstruction_error - error_min
    )
    /
    (
        error_max - error_min + 1e-10
    )
) * 100

df["RiskScore"] = (
    df["RiskScore"]
    .clip(0, 100)
    .round(2)
)


# ============================================================
# 8. SIMPLE EXPLAINABILITY
# ============================================================

print("\n[7] Generating anomaly reasons...")

feature_medians = df[FEATURES].median()

def generate_reason(row):

    reasons = []

    if row["InscClaimAmtReimbursed"] > feature_medians["InscClaimAmtReimbursed"] * 3:
        reasons.append("Unusually high reimbursement")

    if row["DeductibleAmtPaid"] > feature_medians["DeductibleAmtPaid"] * 3:
        reasons.append("High deductible amount")

    if row["ClaimDuration"] > feature_medians["ClaimDuration"] * 3:
        reasons.append("Unusually long claim duration")

    if row["DiagnosisCount"] > feature_medians["DiagnosisCount"] * 2:
        reasons.append("High diagnosis count")

    if row["ProcedureCount"] > feature_medians["ProcedureCount"] * 2:
        reasons.append("High procedure count")

    if row["HospitalStayDays"] > 15:
        reasons.append("Long hospital stay")

    if len(reasons) == 0:
        reasons.append("Unusual combination of claim features")

    return "; ".join(reasons)


df["Reason"] = df.apply(
    generate_reason,
    axis=1
)


# ============================================================
# 9. CLAIM LEVEL OUTPUT
# ============================================================

claim_output_columns = [
    "ClaimID",
    "BeneID",
    "Provider",
    "ClaimType",
    "ReconstructionError",
    "RiskScore",
    "IsAnomaly",
    "Reason",
    "PotentialFraud"
]

claim_results = df[
    claim_output_columns
].copy()

claim_result_path = os.path.join(
    RESULT_DIR,
    "claim_anomaly_results.csv"
)

claim_results.to_csv(
    claim_result_path,
    index=False
)


# ============================================================
# 10. PROVIDER LEVEL AGGREGATION
# ============================================================

print("\n[8] Creating provider investigation queue...")

provider_results = (
    df.groupby("Provider")
    .agg(
        TotalClaims=("ClaimID", "count"),
        AnomalousClaims=("IsAnomaly", "sum"),
        AverageRiskScore=("RiskScore", "mean"),
        MaximumRiskScore=("RiskScore", "max"),
        PotentialFraud=("PotentialFraud", "first")
    )
    .reset_index()
)

provider_results["AnomalyRate"] = (
    provider_results["AnomalousClaims"]
    /
    provider_results["TotalClaims"]
) * 100

provider_results["AnomalyRate"] = (
    provider_results["AnomalyRate"]
    .round(2)
)

provider_results["AverageRiskScore"] = (
    provider_results["AverageRiskScore"]
    .round(2)
)

provider_results = provider_results.sort_values(
    by=[
        "AnomalyRate",
        "AverageRiskScore"
    ],
    ascending=False
)

provider_result_path = os.path.join(
    RESULT_DIR,
    "provider_investigation_queue.csv"
)

provider_results.to_csv(
    provider_result_path,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

total_claims = len(df)
flagged_claims = int(df["IsAnomaly"].sum())

print("\n" + "=" * 70)
print("[SUCCESS] AUTOENCODER RESULTS GENERATED")
print("=" * 70)

print("\nTotal claims:", total_claims)

print("Flagged claims:", flagged_claims)

print(
    "Flagged percentage:",
    round(
        flagged_claims / total_claims * 100,
        2
    ),
    "%"
)

print(
    "99th percentile threshold:",
    threshold
)

print(
    "Minimum risk score:",
    df["RiskScore"].min()
)

print(
    "Maximum risk score:",
    df["RiskScore"].max()
)

print("\nClaim results saved:")
print(claim_result_path)

print("\nProvider investigation queue saved:")
print(provider_result_path)