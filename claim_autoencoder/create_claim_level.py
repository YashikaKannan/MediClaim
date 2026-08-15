import pandas as pd

DATA_PATH = "../carrier_clean_final_perfect.csv"
OUTPUT_PATH = "CLAIM_LEVEL_AUTOENCODER_DATA.csv"

print("Loading carrier dataset...")

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

print("Original rows:", len(df))
print("Unique claims:", df["CLM_ID"].nunique())


# ----------------------------------
# Claim-level aggregation
# ----------------------------------

print("\nAggregating service lines into claims...")

aggregation = {
    "BENE_ID": "first",

    # Already claim-level values
    "CLM_PMT_AMT": "first",
    "NCH_CARR_CLM_SBMTD_CHRG_AMT": "first",
    "NCH_CARR_CLM_ALOWD_AMT": "first",

    # Line-level amounts
    "LINE_SBMTD_CHRG_AMT": "sum",
    "LINE_ALOWD_CHRG_AMT": "sum",
    "LINE_PRVDR_PMT_AMT": "sum",

    # Repeated claim-level features
    "LINE_SRVC_CNT": "first",
    "DIAGNOSIS_COUNT": "first"
}


claim_df = (
    df.groupby(
        "CLM_ID",
        as_index=False
    )
    .agg(aggregation)
)


# ----------------------------------
# Add number of lines per claim
# ----------------------------------

line_counts = (
    df.groupby("CLM_ID")
    .size()
    .reset_index(name="CLAIM_LINE_COUNT")
)

claim_df = claim_df.merge(
    line_counts,
    on="CLM_ID",
    how="left"
)


# ----------------------------------
# Validation
# ----------------------------------

print("\n==============================")
print("CLAIM LEVEL DATA")
print("==============================")

print(
    "Final rows:",
    len(claim_df)
)

print(
    "Unique CLM_ID:",
    claim_df["CLM_ID"].nunique()
)

print(
    "Duplicate CLM_ID:",
    claim_df["CLM_ID"].duplicated().sum()
)

print(
    "Missing values:",
    claim_df.isna().sum().sum()
)

print(
    "Maximum lines per claim:",
    claim_df["CLAIM_LINE_COUNT"].max()
)


# ----------------------------------
# Save
# ----------------------------------

claim_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    "\nSaved as:",
    OUTPUT_PATH
)

print("\nCLAIM-LEVEL DATASET CREATED!")