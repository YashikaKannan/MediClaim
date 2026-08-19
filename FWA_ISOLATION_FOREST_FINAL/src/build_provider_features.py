import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("processed")

OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

labels = pd.read_csv(
    DATA_DIR / "Train-1542865627584.csv"
)

inpatient = pd.read_csv(
    DATA_DIR / "Train_Inpatientdata-1542865627584.csv"
)

outpatient = pd.read_csv(
    DATA_DIR / "Train_Outpatientdata-1542865627584.csv"
)

print("Data loaded successfully.")
print("Labels     :", labels.shape)
print("Inpatient  :", inpatient.shape)
print("Outpatient :", outpatient.shape)


# --------------------------------------------------
# 2. INPATIENT FEATURES
# --------------------------------------------------

ip_features = (
    inpatient
    .groupby("Provider")
    .agg(
        IP_Claim_Count=("ClaimID", "nunique"),
        IP_Unique_Beneficiaries=("BeneID", "nunique"),
        IP_Total_Reimbursement=("InscClaimAmtReimbursed", "sum"),
        IP_Avg_Reimbursement=("InscClaimAmtReimbursed", "mean"),
        IP_Total_Deductible=("DeductibleAmtPaid", "sum"),
        IP_Avg_Deductible=("DeductibleAmtPaid", "mean")
    )
    .reset_index()
)

print("\nInpatient provider features:", ip_features.shape)


# --------------------------------------------------
# 3. OUTPATIENT FEATURES
# --------------------------------------------------

op_features = (
    outpatient
    .groupby("Provider")
    .agg(
        OP_Claim_Count=("ClaimID", "nunique"),
        OP_Unique_Beneficiaries=("BeneID", "nunique"),
        OP_Total_Reimbursement=("InscClaimAmtReimbursed", "sum"),
        OP_Avg_Reimbursement=("InscClaimAmtReimbursed", "mean"),
        OP_Total_Deductible=("DeductibleAmtPaid", "sum"),
        OP_Avg_Deductible=("DeductibleAmtPaid", "mean")
    )
    .reset_index()
)

print("Outpatient provider features:", op_features.shape)


# --------------------------------------------------
# 4. TRUE combined unique beneficiaries
# --------------------------------------------------

ip_beneficiaries = (
    inpatient[["Provider", "BeneID"]]
    .drop_duplicates()
)

op_beneficiaries = (
    outpatient[["Provider", "BeneID"]]
    .drop_duplicates()
)

all_beneficiaries = pd.concat(
    [ip_beneficiaries, op_beneficiaries],
    ignore_index=True
).drop_duplicates()

combined_beneficiaries = (
    all_beneficiaries
    .groupby("Provider")
    .size()
    .reset_index(
        name="Total_Unique_Beneficiaries"
    )
)

print(
    "Combined beneficiary features:",
    combined_beneficiaries.shape
)


# --------------------------------------------------
# 5. Start with ALL labeled providers
# --------------------------------------------------

provider_features = labels[["Provider"]].copy()

provider_features = provider_features.merge(
    ip_features,
    on="Provider",
    how="left"
)

provider_features = provider_features.merge(
    op_features,
    on="Provider",
    how="left"
)

provider_features = provider_features.merge(
    combined_beneficiaries,
    on="Provider",
    how="left"
)


# --------------------------------------------------
# 6. Fill missing feature values with zero
# --------------------------------------------------

feature_columns = [
    col for col in provider_features.columns
    if col != "Provider"
]

provider_features[feature_columns] = (
    provider_features[feature_columns]
    .fillna(0)
)


# --------------------------------------------------
# 7. Combined features
# --------------------------------------------------

provider_features["Total_Claim_Count"] = (
    provider_features["IP_Claim_Count"]
    + provider_features["OP_Claim_Count"]
)

provider_features["Total_Reimbursement"] = (
    provider_features["IP_Total_Reimbursement"]
    + provider_features["OP_Total_Reimbursement"]
)

provider_features["Total_Deductible"] = (
    provider_features["IP_Total_Deductible"]
    + provider_features["OP_Total_Deductible"]
)


# --------------------------------------------------
# 8. Behavioral ratios
# --------------------------------------------------

provider_features["Claims_Per_Beneficiary"] = (
    provider_features["Total_Claim_Count"]
    / provider_features[
        "Total_Unique_Beneficiaries"
    ].replace(0, 1)
)

provider_features["Reimbursement_Per_Beneficiary"] = (
    provider_features["Total_Reimbursement"]
    / provider_features[
        "Total_Unique_Beneficiaries"
    ].replace(0, 1)
)

provider_features["Reimbursement_Per_Claim"] = (
    provider_features["Total_Reimbursement"]
    / provider_features[
        "Total_Claim_Count"
    ].replace(0, 1)
)

provider_features["IP_Claim_Share"] = (
    provider_features["IP_Claim_Count"]
    / provider_features[
        "Total_Claim_Count"
    ].replace(0, 1)
)


# --------------------------------------------------
# 9. Save
# --------------------------------------------------

output_file = OUTPUT_DIR / "provider_features.csv"

provider_features.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 70)
print("CORRECTED FEATURE BUILD COMPLETE")
print("=" * 70)

print("Output file:", output_file)
print("Shape:", provider_features.shape)

print("\nFirst 5 rows:")
print(provider_features.head())

print("\nMissing values:")
print(
    provider_features.isnull().sum().sum()
)

print("\nDuplicate providers:")
print(
    provider_features["Provider"].duplicated().sum()
)

print("\nUnique providers:")
print(
    provider_features["Provider"].nunique()
)

print("\nFeature columns:")
print(
    provider_features.columns.tolist()
)