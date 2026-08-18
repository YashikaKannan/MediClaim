import os
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "autoencoder", "processed")

os.makedirs(OUTPUT_DIR, exist_ok=True)

INPATIENT_PATH = os.path.join(
    DATA_DIR,
    "Test_Inpatientdata-1542969243754.csv"
)

OUTPATIENT_PATH = os.path.join(
    DATA_DIR,
    "Test_Outpatientdata-1542969243754.csv"
)

BENEFICIARY_PATH = os.path.join(
    DATA_DIR,
    "Test_Beneficiarydata-1542969243754.csv"
)

PROVIDER_PATH = os.path.join(
    DATA_DIR,
    "Test-1542969243754.csv"
)

print("=" * 70)
print("TEST DATA PREPROCESSING")
print("=" * 70)


# ============================================================
# 1. LOAD TEST DATA
# ============================================================

print("\n[1] Loading Test datasets...")

inpatient = pd.read_csv(INPATIENT_PATH)
outpatient = pd.read_csv(OUTPATIENT_PATH)
beneficiary = pd.read_csv(BENEFICIARY_PATH)
test_providers = pd.read_csv(PROVIDER_PATH)

print("Test Inpatient :", inpatient.shape)
print("Test Outpatient:", outpatient.shape)
print("Test Beneficiary:", beneficiary.shape)
print("Test Providers :", test_providers.shape)


# ============================================================
# 2. CLAIM TYPE
# ============================================================

inpatient["ClaimType"] = "Inpatient"
outpatient["ClaimType"] = "Outpatient"


# ============================================================
# 3. CLAIM FEATURES
# ============================================================

def create_claim_features(df):

    df = df.copy()

    df["ClaimStartDt"] = pd.to_datetime(
        df["ClaimStartDt"],
        errors="coerce"
    )

    df["ClaimEndDt"] = pd.to_datetime(
        df["ClaimEndDt"],
        errors="coerce"
    )

    df["ClaimDuration"] = (
        df["ClaimEndDt"] - df["ClaimStartDt"]
    ).dt.days + 1

    df["ClaimDuration"] = (
        df["ClaimDuration"]
        .fillna(1)
        .clip(lower=1)
    )

    diagnosis_cols = [
        col for col in df.columns
        if col.startswith("ClmDiagnosisCode_")
    ]

    df["DiagnosisCount"] = (
        df[diagnosis_cols]
        .notna()
        .sum(axis=1)
    )

    procedure_cols = [
        col for col in df.columns
        if col.startswith("ClmProcedureCode_")
    ]

    df["ProcedureCount"] = (
        df[procedure_cols]
        .notna()
        .sum(axis=1)
    )

    physician_cols = [
        col for col in [
            "AttendingPhysician",
            "OperatingPhysician",
            "OtherPhysician"
        ]
        if col in df.columns
    ]

    df["PhysicianCount"] = (
        df[physician_cols]
        .notna()
        .sum(axis=1)
    )

    return df


inpatient = create_claim_features(inpatient)
outpatient = create_claim_features(outpatient)


# ============================================================
# 4. HOSPITAL STAY
# ============================================================

inpatient["AdmissionDt"] = pd.to_datetime(
    inpatient["AdmissionDt"],
    errors="coerce"
)

inpatient["DischargeDt"] = pd.to_datetime(
    inpatient["DischargeDt"],
    errors="coerce"
)

inpatient["HospitalStayDays"] = (
    inpatient["DischargeDt"]
    - inpatient["AdmissionDt"]
).dt.days + 1

inpatient["HospitalStayDays"] = (
    inpatient["HospitalStayDays"]
    .fillna(0)
    .clip(lower=0)
)

outpatient["HospitalStayDays"] = 0


# ============================================================
# 5. COMBINE CLAIMS
# ============================================================

print("\n[2] Combining Test claims...")

claims = pd.concat(
    [inpatient, outpatient],
    ignore_index=True,
    sort=False
)

print("Combined Test claims:", claims.shape)


# ============================================================
# 6. BENEFICIARY FEATURES
# ============================================================

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


# ============================================================
# 7. AUTOENCODER FEATURES
# ============================================================

FEATURES = [
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

for col in FEATURES:

    claims[col] = pd.to_numeric(
        claims[col],
        errors="coerce"
    )

    claims[col] = claims[col].fillna(0)


claims[FEATURES] = claims[FEATURES].replace(
    [np.inf, -np.inf],
    0
)

for col in FEATURES:
    claims[col] = claims[col].clip(lower=0)


# ============================================================
# 8. KEEP ONLY OFFICIAL TEST PROVIDERS
# ============================================================

if "Provider" in test_providers.columns:

    valid_providers = set(
        test_providers["Provider"]
        .dropna()
        .astype(str)
    )

    claims = claims[
        claims["Provider"]
        .astype(str)
        .isin(valid_providers)
    ].copy()


# ============================================================
# 9. FINAL TEST DATASET
# ============================================================

final_columns = [
    "ClaimID",
    "BeneID",
    "Provider",
    "ClaimType"
] + FEATURES

final_df = claims[
    final_columns
].copy()


# ============================================================
# 10. VALIDATION CHECKS
# ============================================================

print("\n[3] Final Test dataset")

print("Rows:", len(final_df))
print("Columns:", len(final_df.columns))

print(
    "Unique claims:",
    final_df["ClaimID"].nunique()
)

print(
    "Unique providers:",
    final_df["Provider"].nunique()
)

print(
    "Missing feature values:",
    int(final_df[FEATURES].isnull().sum().sum())
)


# ============================================================
# 11. SAVE
# ============================================================

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "autoencoder_test_claim_dataset.csv"
)

final_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 70)
print("[SUCCESS] TEST DATA PREPROCESSING COMPLETE")
print("=" * 70)

print("\nSaved to:")
print(OUTPUT_PATH)

print("\nDataset shape:")
print(final_df.shape)

print(
    "\nSame 12 Autoencoder features used "
    "for Train and Test."
)