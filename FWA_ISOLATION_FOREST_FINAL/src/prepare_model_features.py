import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = Path("processed/provider_features.csv")
OUTPUT_FILE = Path("processed/model_features.csv")

# --------------------------------------------------
# 1. Load clean provider features
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("PREPARING ISOLATION FOREST FEATURES")
print("=" * 70)

print("\nInput shape:", df.shape)


# --------------------------------------------------
# 2. Select meaningful FWA features
# --------------------------------------------------

selected_features = [
    "Provider",

    # Inpatient behavior
    "IP_Claim_Count",
    "IP_Unique_Beneficiaries",
    "IP_Total_Reimbursement",
    "IP_Avg_Reimbursement",

    # Outpatient behavior
    "OP_Claim_Count",
    "OP_Unique_Beneficiaries",
    "OP_Total_Reimbursement",
    "OP_Avg_Reimbursement",

    # Overall provider behavior
    "Total_Unique_Beneficiaries",
    "Total_Claim_Count",
    "Total_Reimbursement",

    # Behavioral ratios
    "Claims_Per_Beneficiary",
    "Reimbursement_Per_Claim",

    # Inpatient vs outpatient behavior
    "IP_Claim_Share"
]

model_df = df[selected_features].copy()

print("\nSelected feature count:", len(selected_features) - 1)


# --------------------------------------------------
# 3. Log transform highly skewed magnitude features
# --------------------------------------------------

log_features = [
    "IP_Claim_Count",
    "IP_Unique_Beneficiaries",
    "IP_Total_Reimbursement",
    "IP_Avg_Reimbursement",

    "OP_Claim_Count",
    "OP_Unique_Beneficiaries",
    "OP_Total_Reimbursement",
    "OP_Avg_Reimbursement",

    "Total_Unique_Beneficiaries",
    "Total_Claim_Count",
    "Total_Reimbursement",

    "Claims_Per_Beneficiary",
    "Reimbursement_Per_Claim"
]

for col in log_features:
    model_df[col] = np.log1p(model_df[col])


# --------------------------------------------------
# 4. Check data
# --------------------------------------------------

numeric_features = [
    col for col in model_df.columns
    if col != "Provider"
]

print("\nMissing values:")
print(model_df[numeric_features].isnull().sum().sum())

print("\nInfinite values:")
print(
    np.isinf(model_df[numeric_features]).sum().sum()
)

print("\nNegative values:")
print(
    (model_df[numeric_features] < 0).sum().sum()
)


# --------------------------------------------------
# 5. Save
# --------------------------------------------------

model_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("MODEL FEATURE PREPARATION COMPLETE")
print("=" * 70)

print("Output file:", OUTPUT_FILE)
print("Shape:", model_df.shape)

print("\nFeatures:")
for i, col in enumerate(numeric_features, 1):
    print(f"{i:2}. {col}")

print("\nFirst 5 rows:")
print(model_df.head())