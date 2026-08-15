
import pandas as pd
import numpy as np

# =====================================================================
# PATHS -- CHANGE THESE 3 LINES TO MATCH YOUR COMPUTER
# =====================================================================
PROVIDER_PATH = r"F:\Downloads\PROVIDER_ML_READY.csv"
LEIE_PATH = r"F:\Downloads\leie_clean_specialty_filled.csv"   # optional, for Step 15

OUTPUT_MAIN = r"F:\Downloads\PROVIDER_PEER_BENCHMARK_RESULTS.csv"
OUTPUT_MANUAL_REVIEW = r"F:\Downloads\MANUAL_REVIEW_QUEUE.csv"

MIN_PEERS = 30   # minimum providers needed in a specialty to compare fairly


# =====================================================================
# STEP 1: Load Dataset
# =====================================================================
df = pd.read_csv(PROVIDER_PATH, low_memory=False)
print(f"STEP 1 -- Loaded: {df.shape[0]} rows, {df.shape[1]} columns")


# =====================================================================
# STEP 2: Verify Data
# =====================================================================
print(f"STEP 2 -- Unique providers (NPIs): {df['Rndrng_NPI'].nunique()}")
print(f"STEP 2 -- Null values in key columns:")
print(df[["Rndrng_NPI", "Rndrng_Prvdr_Type", "Tot_Mdcr_Pymt_Amt",
          "Tot_Srvcs", "Tot_Benes"]].isnull().sum())

if df["Rndrng_NPI"].duplicated().any():
    print("STEP 2 -- Duplicate NPIs found, removing them")
    df = df.drop_duplicates("Rndrng_NPI")


# =====================================================================
# STEP 3: Select Features
# =====================================================================
features = [
    "Tot_Mdcr_Pymt_Amt",
    "Tot_Srvcs",
    "Tot_Benes",
    "Charge_Per_Service",
    "Payment_to_Allowed_Ratio",
    "Services_Per_Beneficiary",
]
print(f"\nSTEP 3 -- Using features: {features}")


# =====================================================================
# STEP 4: Keep Raw Columns (needed later for real "4.5x higher" numbers)
# =====================================================================
df["RAW_PAYMENT"] = df["Tot_Mdcr_Pymt_Amt"]
df["RAW_SERVICES"] = df["Tot_Srvcs"]
df["RAW_BENES"] = df["Tot_Benes"]


# =====================================================================
# STEP 5: Create Log Features (for statistics only, not for explanations)
# =====================================================================
df["PAYMENT_LOG"] = np.log1p(df["Tot_Mdcr_Pymt_Amt"])
df["SERVICES_LOG"] = np.log1p(df["Tot_Srvcs"])
df["BENES_LOG"] = np.log1p(df["Tot_Benes"])
df["CHARGE_LOG"] = np.log1p(df["Charge_Per_Service"])


# =====================================================================
# STEP 6: Check Specialty Sizes
# =====================================================================
specialty_size = df.groupby("Rndrng_Prvdr_Type").size()
df["SPECIALTY_SIZE"] = df["Rndrng_Prvdr_Type"].map(specialty_size)
print(f"\nSTEP 6 -- Total specialties found: {df['Rndrng_Prvdr_Type'].nunique()}")


# =====================================================================
# STEP 7: Create Valid Peer Groups (>= 30 providers)
# =====================================================================
peer_df = df[df["SPECIALTY_SIZE"] >= MIN_PEERS].copy()
manual_review_df = df[df["SPECIALTY_SIZE"] < MIN_PEERS].copy()
manual_review_df["STATUS"] = "INSUFFICIENT_PEERS"

print(f"STEP 7 -- Providers with enough peers: {len(peer_df)}")
print(f"STEP 7 -- Providers sent to manual review (too few peers): {len(manual_review_df)}")


# =====================================================================
# STEP 8: Compute Peer Medians
# =====================================================================
peer_df["PEER_PAYMENT_MEDIAN"] = peer_df.groupby("Rndrng_Prvdr_Type")["RAW_PAYMENT"].transform("median")
peer_df["PEER_SERVICE_MEDIAN"] = peer_df.groupby("Rndrng_Prvdr_Type")["RAW_SERVICES"].transform("median")
peer_df["PEER_BENE_MEDIAN"] = peer_df.groupby("Rndrng_Prvdr_Type")["RAW_BENES"].transform("median")


# =====================================================================
# STEP 9: Create Peer Ratios
# =====================================================================
peer_df["PAYMENT_RATIO"] = peer_df["RAW_PAYMENT"] / peer_df["PEER_PAYMENT_MEDIAN"]
peer_df["SERVICE_RATIO"] = peer_df["RAW_SERVICES"] / peer_df["PEER_SERVICE_MEDIAN"]
peer_df["BENE_RATIO"] = peer_df["RAW_BENES"] / peer_df["PEER_BENE_MEDIAN"]


# =====================================================================
# STEP 10: Compute Robust Z-Score (median + MAD, resistant to outliers)
# =====================================================================
def robust_z(group):
    median = group.median()
    mad = (group - median).abs().median()
    if mad == 0:
        return pd.Series(0, index=group.index)
    return (group - median) / mad

peer_df["PAYMENT_ROBUST_Z"] = peer_df.groupby("Rndrng_Prvdr_Type")["PAYMENT_LOG"].transform(robust_z)
peer_df["SERVICE_ROBUST_Z"] = peer_df.groupby("Rndrng_Prvdr_Type")["SERVICES_LOG"].transform(robust_z)
peer_df["BENE_ROBUST_Z"] = peer_df.groupby("Rndrng_Prvdr_Type")["BENES_LOG"].transform(robust_z)


# =====================================================================
# STEP 11: Build Risk Score (0-100)
# Only POSITIVE deviations count -- unusually LOW billing isn't a
# fraud risk, unusually HIGH billing is.
# =====================================================================
def clip_positive(s):
    return s.clip(lower=0)

combined_z = (
    clip_positive(peer_df["PAYMENT_ROBUST_Z"]) +
    clip_positive(peer_df["SERVICE_ROBUST_Z"]) +
    clip_positive(peer_df["BENE_ROBUST_Z"])
)

# percentile-rank normalization -- keeps score spread even 0-100
# without one extreme provider compressing everyone else near 0
peer_df["RISK_SCORE"] = (combined_z.rank(pct=True) * 100).round(1)

def risk_level(score):
    if score <= 30:
        return "Low"
    elif score <= 60:
        return "Medium"
    elif score <= 80:
        return "High"
    else:
        return "Critical"

peer_df["RISK_LEVEL"] = peer_df["RISK_SCORE"].apply(risk_level)

print(f"\nSTEP 11 -- Risk level distribution:")
print(peer_df["RISK_LEVEL"].value_counts())


# =====================================================================
# STEP 12: Flag Providers (top 1% / 3% / 5%)
# =====================================================================
for pct, label in [(0.99, "TOP_1PCT"), (0.97, "TOP_3PCT"), (0.95, "TOP_5PCT")]:
    threshold = peer_df["RISK_SCORE"].quantile(pct)
    peer_df[label] = peer_df["RISK_SCORE"] >= threshold

peer_df["FLAG"] = peer_df["TOP_5PCT"]

print(f"\nSTEP 12 -- Top 1% flagged: {peer_df['TOP_1PCT'].sum()}")
print(f"STEP 12 -- Top 3% flagged: {peer_df['TOP_3PCT'].sum()}")
print(f"STEP 12 -- Top 5% flagged: {peer_df['TOP_5PCT'].sum()}")


# =====================================================================
# STEP 13: Generate Explainable Reasons
# =====================================================================
def build_reasons(row):
    reasons = []
    if row["PAYMENT_RATIO"] > 1.5:
        reasons.append(f"Bills {row['PAYMENT_RATIO']:.1f}x higher than specialty median")
    if row["SERVICE_RATIO"] > 1.5:
        reasons.append(f"Provides {row['SERVICE_RATIO']:.1f}x more services than peers")
    if row["BENE_RATIO"] > 1.5:
        reasons.append(f"Serves {row['BENE_RATIO']:.1f}x more beneficiaries than peers")
    if not reasons:
        reasons.append("No significant peer deviation detected")
    return reasons

peer_df["REASONS_LIST"] = peer_df.apply(build_reasons, axis=1)
peer_df["REASON_1"] = peer_df["REASONS_LIST"].apply(lambda x: x[0] if len(x) > 0 else "")
peer_df["REASON_2"] = peer_df["REASONS_LIST"].apply(lambda x: x[1] if len(x) > 1 else "")
peer_df["REASON_3"] = peer_df["REASONS_LIST"].apply(lambda x: x[2] if len(x) > 2 else "")


# =====================================================================
# STEP 14: Small Specialty Queue -- already built in Step 7
# (manual_review_df holds these -- saved separately below, not removed)
# =====================================================================


# =====================================================================
# STEP 15: LEIE Validation (business validation, NOT accuracy/precision/recall)
# =====================================================================
try:
    leie = pd.read_csv(LEIE_PATH, dtype=str, low_memory=False)
    leie_npis = set(leie[leie["HAS_VALID_NPI"].astype(str) == "True"]["NPI"].astype(str))

    peer_df["NPI_STR"] = peer_df["Rndrng_NPI"].astype(str)
    peer_df["IN_LEIE"] = peer_df["NPI_STR"].isin(leie_npis)

    overlap = peer_df[peer_df["TOP_1PCT"]]["IN_LEIE"].sum()
    print(f"\nSTEP 15 -- LEIE overlap: {overlap} of {peer_df['TOP_1PCT'].sum()} "
          f"top-1% flagged providers appear in LEIE")
except FileNotFoundError:
    print("\nSTEP 15 -- LEIE file not found, skipping validation")
    peer_df["IN_LEIE"] = False


# =====================================================================
# FINAL OUTPUT FILES
# =====================================================================
output_cols = [
    "Rndrng_NPI", "Rndrng_Prvdr_Type",
    "PAYMENT_ROBUST_Z", "SERVICE_ROBUST_Z", "BENE_ROBUST_Z",
    "PAYMENT_RATIO", "SERVICE_RATIO", "BENE_RATIO",
    "RISK_SCORE", "RISK_LEVEL",
    "REASON_1", "REASON_2", "REASON_3",
    "TOP_1PCT", "TOP_3PCT", "TOP_5PCT", "FLAG", "IN_LEIE",
]
peer_df[output_cols].to_csv(OUTPUT_MAIN, index=False)
manual_review_df.to_csv(OUTPUT_MANUAL_REVIEW, index=False)

print(f"\nSaved main results to: {OUTPUT_MAIN}")
print(f"Saved manual review queue to: {OUTPUT_MANUAL_REVIEW}")


# =====================================================================
# SUMMARY REPORT (screenshot this for your presentation/report)
# =====================================================================
print("\n" + "=" * 50)
print("SUMMARY REPORT")
print("=" * 50)
print(f"Total Providers Analyzed: {len(peer_df)}")
print(f"Specialties Analyzed: {peer_df['Rndrng_Prvdr_Type'].nunique()}")
print(f"Providers in Manual Review Queue: {len(manual_review_df)}")
print(f"Top 1% Flagged: {peer_df['TOP_1PCT'].sum()}")
print(f"Top 3% Flagged: {peer_df['TOP_3PCT'].sum()}")
print(f"Top 5% Flagged: {peer_df['TOP_5PCT'].sum()}")
print(f"\nTop 5 Highest-Risk Providers:")
print(peer_df.sort_values("RISK_SCORE", ascending=False)
      [["Rndrng_NPI", "Rndrng_Prvdr_Type", "RISK_SCORE", "REASON_1"]]
      .head(5).to_string(index=False))