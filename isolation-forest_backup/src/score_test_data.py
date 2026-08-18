import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    BASE_DIR
    / "processed"
    / "validation_isolation_forest.joblib"
)

TEST_FEATURE_FILE = (
    BASE_DIR
    / "processed"
    / "test_model_features.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "processed"
    / "final_test_scores.csv"
)

THRESHOLD = 0.54

FEATURES = [
    "IP_Claim_Count",
    "OP_Claim_Count",
    "IP_Total_Reimbursement",
    "OP_Total_Reimbursement",
    "IP_Avg_Reimbursement",
    "OP_Avg_Reimbursement",
    "Total_Unique_Beneficiaries",
    "Claims_Per_Beneficiary",
    "Reimbursement_Per_Claim",
    "IP_Claim_Share"
]


# ============================================================
# START
# ============================================================

print("=" * 70)
print("FINAL SEPARATE TEST DATA SCORING")
print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading frozen Isolation Forest...")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully.")

print(
    f"Model: {model}"
)


# ============================================================
# LOAD TEST FEATURES
# ============================================================

print("\nLoading separate test features...")

test = pd.read_csv(TEST_FEATURE_FILE)

print(
    f"Test data shape: {test.shape}"
)


# ============================================================
# CHECK PROVIDER COLUMN
# ============================================================

if "Provider" not in test.columns:
    raise ValueError(
        "Provider column missing from test data."
    )


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [
    col
    for col in FEATURES
    if col not in test.columns
]

if missing_features:
    raise ValueError(
        f"Missing required features: {missing_features}"
    )

print("\nAll required features are present.")


# ============================================================
# DUPLICATE CHECK
# ============================================================

duplicate_providers = (
    test["Provider"].duplicated().sum()
)

print(
    f"\nDuplicate providers: "
    f"{duplicate_providers}"
)

if duplicate_providers > 0:
    raise ValueError(
        "Duplicate Provider IDs found."
    )


# ============================================================
# DATA QUALITY
# ============================================================

X_test = test[FEATURES].copy()

print("\nChecking test feature quality...")

missing_values = X_test.isna().sum().sum()

infinite_values = np.isinf(
    X_test.to_numpy()
).sum()

negative_values = (
    X_test < 0
).sum().sum()


print(
    f"Missing values  : {missing_values}"
)

print(
    f"Infinite values : {infinite_values}"
)

print(
    f"Negative values : {negative_values}"
)


if missing_values > 0:
    raise ValueError(
        "Missing values found in test data."
    )

if infinite_values > 0:
    raise ValueError(
        "Infinite values found in test data."
    )

if negative_values > 0:
    raise ValueError(
        "Negative values found in test data."
    )


# ============================================================
# FEATURE ORDER
# ============================================================

print("\nFeature order:")

for i, feature in enumerate(FEATURES, start=1):
    print(
        f"{i}. {feature}"
    )


# ============================================================
# SCORE TEST DATA
# ============================================================

print("\nScoring separate test providers...")

decision_values = model.decision_function(
    X_test
)

anomaly_scores = (
    0.5 - decision_values
)


# ============================================================
# CREATE OUTPUT
# ============================================================

results = pd.DataFrame(
    {
        "Provider": test["Provider"],
        "Anomaly_Score": anomaly_scores
    }
)


# ============================================================
# APPLY FROZEN THRESHOLD
# ============================================================

results["Anomaly_Flag"] = (
    results["Anomaly_Score"]
    >= THRESHOLD
).astype(int)


# ============================================================
# TEST SUMMARY
# ============================================================

total_providers = len(results)

suspicious_providers = int(
    results["Anomaly_Flag"].sum()
)

suspicious_percentage = (
    suspicious_providers
    / total_providers
    * 100
)


# ============================================================
# SCORE STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST SCORING COMPLETE")
print("=" * 70)

print(
    f"\nTest providers       : "
    f"{total_providers}"
)

print(
    f"Frozen threshold     : "
    f"{THRESHOLD:.2f}"
)

print(
    f"Suspicious providers : "
    f"{suspicious_providers}"
)

print(
    f"Suspicious percentage: "
    f"{suspicious_percentage:.2f}%"
)


print("\nAnomaly score statistics:")

print(
    results["Anomaly_Score"].describe()
)


# ============================================================
# TOP SUSPICIOUS PROVIDERS
# ============================================================

print("\nTop 30 suspicious test providers:")

top_30 = (
    results
    .sort_values(
        "Anomaly_Score",
        ascending=False
    )
    .head(30)
)

print(
    top_30.to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

results.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nSaved final test scores:")

print(
    OUTPUT_FILE
)


# ============================================================
# FINAL QUALITY CHECK
# ============================================================

print("\nFinal checks:")

print(
    f"Rows            : {len(results)}"
)

print(
    f"Missing scores  : "
    f"{results['Anomaly_Score'].isna().sum()}"
)

print(
    f"Infinite scores : "
    f"{np.isinf(results['Anomaly_Score']).sum()}"
)

print(
    f"Flagged         : "
    f"{results['Anomaly_Flag'].sum()}"
)


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)