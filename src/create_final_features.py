import pandas as pd
from pathlib import Path

INPUT_FILE = Path("processed/model_features.csv")
OUTPUT_FILE = Path("processed/model_features_final.csv")

# Load transformed model features
df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("CREATING FINAL ISOLATION FOREST FEATURE SET")
print("=" * 70)

print("\nInput shape:", df.shape)

# Final selected behavioral features
final_features = [
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

# Provider is retained only for identification
final_df = df[
    ["Provider"] + final_features
].copy()

# Safety checks
print("\nMissing values:", final_df.isnull().sum().sum())
print("Duplicate providers:", final_df["Provider"].duplicated().sum())

# Save
final_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 70)
print("FINAL FEATURE SET CREATED")
print("=" * 70)

print("Output file:", OUTPUT_FILE)
print("Shape:", final_df.shape)

print("\nModel feature count:", len(final_features))

print("\nFinal model features:")

for i, feature in enumerate(final_features, 1):
    print(f"{i:2}. {feature}")

print("\nFirst 5 rows:")
print(final_df.head())0