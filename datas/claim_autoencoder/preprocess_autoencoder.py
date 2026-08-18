import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "autoencoder", "processed")

os.makedirs(OUTPUT_DIR, exist_ok=True)

INPATIENT_PATH = os.path.join(DATA_DIR, "Train_Inpatientdata-1542865627584.csv")
OUTPATIENT_PATH = os.path.join(DATA_DIR, "Train_Outpatientdata-1542865627584.csv")
BENEFICIARY_PATH = os.path.join(DATA_DIR, "Train_Beneficiarydata-1542865627584.csv")
LABEL_PATH = os.path.join(DATA_DIR, "Train-1542865627584.csv")

print("=" * 70)
print("AUTOENCODER PREPROCESSING")
print("=" * 70)

print("\n[1] Loading datasets...")

inpatient = pd.read_csv(INPATIENT_PATH)
outpatient = pd.read_csv(OUTPATIENT_PATH)
beneficiary = pd.read_csv(BENEFICIARY_PATH)
provider_labels = pd.read_csv(LABEL_PATH)

print("Inpatient shape :", inpatient.shape)
print("Outpatient shape:", outpatient.shape)
print("Beneficiary shape:", beneficiary.shape)
print("Provider labels :", provider_labels.shape)

inpatient["ClaimType"] = "Inpatient"
outpatient["ClaimType"] = "Outpatient"

def create_claim_features(df):
    df = df.copy()

    df["ClaimStartDt"] = pd.to_datetime(df["ClaimStartDt"], errors="coerce")
    df["ClaimEndDt"] = pd.to_datetime(df["ClaimEndDt"], errors="coerce")

    df["ClaimDuration"] = (
        df["ClaimEndDt"] - df["ClaimStartDt"]
    ).dt.days + 1

    df["ClaimDuration"] = df["ClaimDuration"].fillna(1).clip(lower=1)

    diagnosis_cols = [
        col for col in df.columns
        if col.startswith("ClmDiagnosisCode_")
    ]

    df["DiagnosisCount"] = df[diagnosis_cols].notna().sum(axis=1)

    procedure_cols = [
        col for col in df.columns
        if col.startswith("ClmProcedureCode_")
    ]

    df["ProcedureCount"] = df[procedure_cols].notna().sum(axis=1)

    physician_cols = [
        col for col in [
            "AttendingPhysician",
            "OperatingPhysician",
            "OtherPhysician"
        ]
        if col in df.columns
    ]

    df["PhysicianCount"] = df[physician_cols].notna().sum(axis=1)

    return df


inpatient = create_claim_features(inpatient)
outpatient = create_claim_features(outpatient)

inpatient["AdmissionDt"] = pd.to_datetime(
    inpatient["AdmissionDt"],
    errors="coerce"
)

inpatient["DischargeDt"] = pd.to_datetime(
    inpatient["DischargeDt"],
    errors="coerce"
)

inpatient["HospitalStayDays"] = (
    inpatient["DischargeDt"] - inpatient["AdmissionDt"]
).dt.days + 1

inpatient["HospitalStayDays"] = (
    inpatient["HospitalStayDays"]
    .fillna(0)
    .clip(lower=0)
)

outpatient["HospitalStayDays"] = 0

print("\n[2] Combining inpatient and outpatient claims...")

claims = pd.concat(
    [inpatient, outpatient],
    ignore_index=True,
    sort=False
)

print("Combined claims:", claims.shape)

chronic_columns = [
    "ChronicCond_Alzheimer",
    "ChronicCond_Heartfailure",
    "ChronicCond_KidneyDisease",
    "ChronicCond_Cancer",
    "ChronicCond_ObstrPulmonary",
    "ChronicCond_Depression",
    "ChronicCond_Diabetes",
    "ChronicCond_IschemicHeart",
    "ChronicCond_Osteoporasis",
    "ChronicCond_rheumatoidarthritis",
    "ChronicCond_stroke"
]

existing_chronic = [
    col for col in chronic_columns
    if col in beneficiary.columns
]

beneficiary["ChronicConditionCount"] = (
    beneficiary[existing_chronic]
    .eq(1)
    .sum(axis=1)
)

beneficiary_features = beneficiary[
    [
        "BeneID",
        "IPAnnualReimbursementAmt",
        "IPAnnualDeductibleAmt",
        "OPAnnualReimbursementAmt",
        "OPAnnualDeductibleAmt",
        "ChronicConditionCount"
    ]
]

claims = claims.merge(
    beneficiary_features,
    on="BeneID",
    how="left"
)

print("After beneficiary merge:", claims.shape)

claims = claims.merge(
    provider_labels,
    on="Provider",
    how="left"
)

feature_columns = [
    "InscClaimAmtReimbursed",
    "DeductibleAmtPaid",
    "ClaimDuration",
    "DiagnosisCount",
    "ProcedureCount",
    "PhysicianCount",
    "HospitalStayDays",
    "IPAnnualReimbursementAmt",
    "IPAnnualDeductibleAmt",
    "OPAnnualReimbursementAmt",
    "OPAnnualDeductibleAmt",
    "ChronicConditionCount"
]

for col in feature_columns:
    claims[col] = pd.to_numeric(claims[col], errors="coerce")
    claims[col] = claims[col].fillna(0)

claims[feature_columns] = claims[feature_columns].replace(
    [np.inf, -np.inf],
    0
)

for col in feature_columns:
    claims[col] = claims[col].clip(lower=0)

final_columns = [
    "ClaimID",
    "BeneID",
    "Provider",
    "ClaimType"
] + feature_columns + [
    "PotentialFraud"
]

final_df = claims[final_columns].copy()

print("\n[3] Final dataset summary")
print("Rows:", len(final_df))
print("Columns:", len(final_df.columns))
print("Unique claims:", final_df["ClaimID"].nunique())
print("Unique providers:", final_df["Provider"].nunique())

print("\nPotentialFraud distribution:")
print(final_df["PotentialFraud"].value_counts(dropna=False))

print("\nMissing feature values:")
print(final_df[feature_columns].isnull().sum())

output_file = os.path.join(
    OUTPUT_DIR,
    "autoencoder_claim_dataset.csv"
)

final_df.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print("[SUCCESS] AUTOENCODER DATASET CREATED")
print("=" * 70)

print("Saved to:")
print(output_file)

print("\nDataset shape:")
print(final_df.shape)

print("\nIMPORTANT:")
print("PotentialFraud is NOT used as Autoencoder input.")
print("It is kept only for final validation.")