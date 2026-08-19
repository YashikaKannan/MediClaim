import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "processed" / "test_provider_features.csv"
OUTPUT_FILE = BASE_DIR / "processed" / "test_model_features.csv"


# EXACT SAME 10 FEATURES USED DURING TRAINING
MODEL_FEATURES = [
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
print("PREPARING TEST FEATURES FOR ISOLATION FOREST")
print("=" * 70)

print("\nLoading test provider features...")

df = pd.read_csv(INPUT_FILE)

print(f"Input shape: {df.shape}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

print("\nChecking required features...")

missing_features = [
    feature for feature in MODEL_FEATURES
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing required features: {missing_features}"
    )

print("All required features are present.")


# ============================================================
# SELECT PROVIDER ID + MODEL FEATURES
# ============================================================

test_model = df[["Provider"] + MODEL_FEATURES].copy()

print(f"\nSelected feature count: {len(MODEL_FEATURES)}")


# ============================================================
# CHECK ORIGINAL VALUES
# ============================================================

print("\nChecking original test values...")

numeric_data = test_model[MODEL_FEATURES]

print(f"Missing values: {numeric_data.isna().sum().sum()}")
print(
    f"Infinite values: "
    f"{np.isinf(numeric_data.to_numpy()).sum()}"
)
print(
    f"Negative values: "
    f"(numeric_data < 0).sum().sum()"
)


# ============================================================
# APPLY SAME LOG1P TRANSFORMATION
# ============================================================

print("\nApplying log1p transformation...")

test_model[MODEL_FEATURES] = np.log1p(
    test_model[MODEL_FEATURES]
)


# ============================================================
# CHECK TRANSFORMED DATA
# ============================================================

print("\nChecking transformed test features...")

transformed = test_model[MODEL_FEATURES]

missing_after = transformed.isna().sum().sum()
infinite_after = np.isinf(transformed.to_numpy()).sum()
negative_after = (transformed < 0).sum().sum()

print(f"Missing values after transformation: {missing_after}")
print(f"Infinite values after transformation: {infinite_after}")
print(f"Negative values after transformation: {negative_after}")


if missing_after > 0:
    raise ValueError("Missing values found after transformation.")

if infinite_after > 0:
    raise ValueError("Infinite values found after transformation.")

if negative_after > 0:
    raise ValueError("Negative values found after transformation.")


# ============================================================
# SAVE
# ============================================================

test_model.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TEST MODEL FEATURE PREPARATION COMPLETE")
print("=" * 70)

print(f"\nOutput file:")
print(OUTPUT_FILE)

print(f"\nShape: {test_model.shape}")

print("\nFeatures:")

for i, feature in enumerate(MODEL_FEATURES, 1):
    print(f"{i}. {feature}")

print("\nFirst 5 rows:")
print(test_model.head())

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)