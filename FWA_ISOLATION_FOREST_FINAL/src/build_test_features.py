import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TEST FILES
# ============================================================

TEST_PROVIDER_FILE = RAW_DIR / "Test-1542969243754.csv"

TEST_INPATIENT_FILE = (
    RAW_DIR / "Test_Inpatientdata-1542969243754.csv"
)

TEST_OUTPATIENT_FILE = (
    RAW_DIR / "Test_Outpatientdata-1542969243754.csv"
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def safe_numeric(series):
    """
    Convert a column to numeric.
    Invalid values become NaN.
    NaN values are replaced with 0.
    """
    return pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0)


# ============================================================
# INPATIENT FEATURES
# ============================================================

def build_inpatient_features(df):

    print("\nBuilding inpatient provider features...")

    # Convert reimbursement and deductible columns
    # to numeric values.

    df["InscClaimAmtReimbursed"] = safe_numeric(
        df["InscClaimAmtReimbursed"]
    )

    df["DeductibleAmtPaid"] = safe_numeric(
        df["DeductibleAmtPaid"]
    )

    # Aggregate claims by Provider

    features = (
        df.groupby("Provider")
        .agg(
            IP_Claim_Count=(
                "ClaimID",
                "count"
            ),

            IP_Unique_Beneficiaries=(
                "BeneID",
                "nunique"
            ),

            IP_Total_Reimbursement=(
                "InscClaimAmtReimbursed",
                "sum"
            ),

            IP_Avg_Reimbursement=(
                "InscClaimAmtReimbursed",
                "mean"
            ),

            IP_Total_Deductible=(
                "DeductibleAmtPaid",
                "sum"
            ),

            IP_Avg_Deductible=(
                "DeductibleAmtPaid",
                "mean"
            ),
        )
        .reset_index()
    )

    print(
        f"Inpatient provider features: "
        f"{features.shape}"
    )

    return features


# ============================================================
# OUTPATIENT FEATURES
# ============================================================

def build_outpatient_features(df):

    print("\nBuilding outpatient provider features...")

    # Convert reimbursement and deductible columns
    # to numeric values.

    df["InscClaimAmtReimbursed"] = safe_numeric(
        df["InscClaimAmtReimbursed"]
    )

    df["DeductibleAmtPaid"] = safe_numeric(
        df["DeductibleAmtPaid"]
    )

    # Aggregate claims by Provider

    features = (
        df.groupby("Provider")
        .agg(
            OP_Claim_Count=(
                "ClaimID",
                "count"
            ),

            OP_Unique_Beneficiaries=(
                "BeneID",
                "nunique"
            ),

            OP_Total_Reimbursement=(
                "InscClaimAmtReimbursed",
                "sum"
            ),

            OP_Avg_Reimbursement=(
                "InscClaimAmtReimbursed",
                "mean"
            ),

            OP_Total_Deductible=(
                "DeductibleAmtPaid",
                "sum"
            ),

            OP_Avg_Deductible=(
                "DeductibleAmtPaid",
                "mean"
            ),
        )
        .reset_index()
    )

    print(
        f"Outpatient provider features: "
        f"{features.shape}"
    )

    return features


# ============================================================
# START
# ============================================================

print("=" * 70)
print("BUILDING TEST PROVIDER FEATURES")
print("=" * 70)


# ============================================================
# CHECK INPUT FILES
# ============================================================

print("\nChecking input files...")

required_files = [
    TEST_PROVIDER_FILE,
    TEST_INPATIENT_FILE,
    TEST_OUTPATIENT_FILE,
]

for file_path in required_files:

    if not file_path.exists():

        print("\nERROR: File not found:")
        print(file_path)

        raise FileNotFoundError(file_path)


print("All required test files found.")


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test data...")

providers = pd.read_csv(
    TEST_PROVIDER_FILE
)

inpatient = pd.read_csv(
    TEST_INPATIENT_FILE
)

outpatient = pd.read_csv(
    TEST_OUTPATIENT_FILE
)


print("\nLoaded data:")

print(
    f"Test providers    : "
    f"{providers.shape}"
)

print(
    f"Test inpatient    : "
    f"{inpatient.shape}"
)

print(
    f"Test outpatient   : "
    f"{outpatient.shape}"
)


# ============================================================
# VALIDATE COLUMNS
# ============================================================

print("\nValidating required columns...")


# Provider file

required_provider_columns = [
    "Provider"
]


# Claim files

required_claim_columns = [
    "BeneID",
    "ClaimID",
    "Provider",
    "InscClaimAmtReimbursed",
    "DeductibleAmtPaid",
]


# Check provider file

for column in required_provider_columns:

    if column not in providers.columns:

        raise ValueError(
            "Missing required column in "
            f"provider test file: {column}"
        )


# Check inpatient file

for column in required_claim_columns:

    if column not in inpatient.columns:

        raise ValueError(
            "Missing required column in "
            f"inpatient test file: {column}"
        )


# Check outpatient file

for column in required_claim_columns:

    if column not in outpatient.columns:

        raise ValueError(
            "Missing required column in "
            f"outpatient test file: {column}"
        )


print("Required columns are present.")


# ============================================================
# CLEAN PROVIDER IDS
# ============================================================

providers["Provider"] = (
    providers["Provider"]
    .astype(str)
    .str.strip()
)

inpatient["Provider"] = (
    inpatient["Provider"]
    .astype(str)
    .str.strip()
)

outpatient["Provider"] = (
    outpatient["Provider"]
    .astype(str)
    .str.strip()
)


# ============================================================
# REMOVE EMPTY PROVIDER IDS
# ============================================================

providers = providers[
    providers["Provider"].notna()
    & (providers["Provider"] != "")
].copy()


inpatient = inpatient[
    inpatient["Provider"].notna()
    & (inpatient["Provider"] != "")
].copy()


outpatient = outpatient[
    outpatient["Provider"].notna()
    & (outpatient["Provider"] != "")
].copy()


# ============================================================
# CHECK DUPLICATE PROVIDERS
# ============================================================

duplicate_count = (
    providers["Provider"]
    .duplicated()
    .sum()
)

print("\nProvider ID check:")

print(
    "Duplicate provider IDs in test file: "
    f"{duplicate_count}"
)


if duplicate_count > 0:

    print(
        "Removing duplicate provider IDs..."
    )

    providers = providers.drop_duplicates(
        subset=["Provider"]
    ).copy()


# ============================================================
# BUILD CLAIM FEATURES
# ============================================================

ip_features = build_inpatient_features(
    inpatient
)

op_features = build_outpatient_features(
    outpatient
)


# ============================================================
# MERGE PROVIDER FEATURES
# ============================================================

print("\nMerging provider features...")


test_features = providers[
    ["Provider"]
].copy()


test_features = test_features.merge(
    ip_features,
    on="Provider",
    how="left"
)


test_features = test_features.merge(
    op_features,
    on="Provider",
    how="left"
)


# ============================================================
# NUMERICAL FEATURE LIST
# ============================================================

numeric_columns = [

    "IP_Claim_Count",
    "IP_Unique_Beneficiaries",
    "IP_Total_Reimbursement",
    "IP_Avg_Reimbursement",
    "IP_Total_Deductible",
    "IP_Avg_Deductible",

    "OP_Claim_Count",
    "OP_Unique_Beneficiaries",
    "OP_Total_Reimbursement",
    "OP_Avg_Reimbursement",
    "OP_Total_Deductible",
    "OP_Avg_Deductible",
]


# ============================================================
# FILL PROVIDERS WITH NO CLAIMS
# ============================================================

for column in numeric_columns:

    test_features[column] = (
        pd.to_numeric(
            test_features[column],
            errors="coerce"
        )
        .fillna(0)
    )


# ============================================================
# TOTAL FEATURES
# ============================================================

print(
    "\nCreating combined claim features..."
)


# Total claims

test_features["Total_Claim_Count"] = (
    test_features["IP_Claim_Count"]
    + test_features["OP_Claim_Count"]
)


# Total unique beneficiaries
#
# We use MAX rather than addition because
# the same beneficiary can appear in both
# inpatient and outpatient claims.

test_features[
    "Total_Unique_Beneficiaries"
] = (
    test_features[
        [
            "IP_Unique_Beneficiaries",
            "OP_Unique_Beneficiaries"
        ]
    ]
    .max(axis=1)
)


# Total reimbursement

test_features["Total_Reimbursement"] = (
    test_features["IP_Total_Reimbursement"]
    + test_features["OP_Total_Reimbursement"]
)


# Total deductible

test_features["Total_Deductible"] = (
    test_features["IP_Total_Deductible"]
    + test_features["OP_Total_Deductible"]
)


# ============================================================
# RATIO FEATURES
# ============================================================

print(
    "Creating ratio features..."
)


# Claims per beneficiary

test_features[
    "Claims_Per_Beneficiary"
] = np.where(

    test_features[
        "Total_Unique_Beneficiaries"
    ] > 0,

    test_features[
        "Total_Claim_Count"
    ]
    /
    test_features[
        "Total_Unique_Beneficiaries"
    ],

    0
)


# Reimbursement per beneficiary

test_features[
    "Reimbursement_Per_Beneficiary"
] = np.where(

    test_features[
        "Total_Unique_Beneficiaries"
    ] > 0,

    test_features[
        "Total_Reimbursement"
    ]
    /
    test_features[
        "Total_Unique_Beneficiaries"
    ],

    0
)


# Reimbursement per claim

test_features[
    "Reimbursement_Per_Claim"
] = np.where(

    test_features[
        "Total_Claim_Count"
    ] > 0,

    test_features[
        "Total_Reimbursement"
    ]
    /
    test_features[
        "Total_Claim_Count"
    ],

    0
)


# ============================================================
# INPATIENT CLAIM SHARE
# ============================================================

test_features[
    "IP_Claim_Share"
] = np.where(

    test_features[
        "Total_Claim_Count"
    ] > 0,

    test_features[
        "IP_Claim_Count"
    ]
    /
    test_features[
        "Total_Claim_Count"
    ],

    0
)


# ============================================================
# FINAL COLUMN ORDER
# ============================================================

final_columns = [

    "Provider",

    "IP_Claim_Count",
    "IP_Unique_Beneficiaries",
    "IP_Total_Reimbursement",
    "IP_Avg_Reimbursement",
    "IP_Total_Deductible",
    "IP_Avg_Deductible",

    "OP_Claim_Count",
    "OP_Unique_Beneficiaries",
    "OP_Total_Reimbursement",
    "OP_Avg_Reimbursement",
    "OP_Total_Deductible",
    "OP_Avg_Deductible",

    "Total_Unique_Beneficiaries",
    "Total_Claim_Count",
    "Total_Reimbursement",
    "Total_Deductible",

    "Claims_Per_Beneficiary",
    "Reimbursement_Per_Beneficiary",
    "Reimbursement_Per_Claim",

    "IP_Claim_Share",
]


test_features = test_features[
    final_columns
]


# ============================================================
# REMOVE INFINITE VALUES
# ============================================================

print(
    "\nPerforming final quality checks..."
)


test_features = test_features.replace(
    [np.inf, -np.inf],
    0
)


# ============================================================
# FINAL NUMERIC CLEANUP
# ============================================================

for column in final_columns:

    if column != "Provider":

        test_features[column] = (
            pd.to_numeric(
                test_features[column],
                errors="coerce"
            )
            .fillna(0)
        )


# ============================================================
# QUALITY CHECKS
# ============================================================

missing_values = (
    test_features
    .isna()
    .sum()
    .sum()
)


duplicate_providers = (
    test_features["Provider"]
    .duplicated()
    .sum()
)


unique_providers = (
    test_features["Provider"]
    .nunique()
)


# ============================================================
# SAVE TEST FEATURES
# ============================================================

output_file = (
    PROCESSED_DIR
    / "test_provider_features.csv"
)


test_features.to_csv(
    output_file,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("TEST FEATURE BUILD COMPLETE")
print("=" * 70)


print("\nOutput file:")
print(output_file)


print(
    f"\nShape: "
    f"{test_features.shape}"
)


print(
    f"\nMissing values: "
    f"{missing_values}"
)


print(
    f"Duplicate providers: "
    f"{duplicate_providers}"
)


print(
    f"Unique providers: "
    f"{unique_providers}"
)


print("\nFeature columns:")


for i, column in enumerate(
    test_features.columns,
    start=1
):

    print(
        f"{i}. {column}"
    )


print("\nFirst 5 rows:")

print(
    test_features.head()
)


print("\nTest feature summary:")

print(
    test_features
    .drop(columns=["Provider"])
    .describe()
)


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)