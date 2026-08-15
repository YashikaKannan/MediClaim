import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(
    "PROVIDER_ML_READY.csv",
    low_memory=False
)

print("Shape:", df.shape)

# =====================================================
# PEER GROUP ANALYSIS
# =====================================================

peer_size = (
    df.groupby("Rndrng_Prvdr_Type")
      .size()
      .reset_index(name="Provider_Count")
      .sort_values(
          by="Provider_Count",
          ascending=False
      )
)

print(peer_size)

MIN_PEERS = 30

valid_groups = peer_size[
    peer_size["Provider_Count"] >= MIN_PEERS
]["Rndrng_Prvdr_Type"]

df["Peer_Status"] = np.where(
    df["Rndrng_Prvdr_Type"].isin(valid_groups),
    "VALID",
    "INSUFFICIENT_PEERS"
)


# =====================================================
# LEIE / OIG VALIDATION
# =====================================================

try:

    leie = pd.read_csv(
        "leie_clean_specialty_filled.csv",
        low_memory=False
    )

    print("LEIE Shape:", leie.shape)


    leie["NPI"] = (
        leie["NPI"]
        .astype(str)
        .str.strip()
    )

    df["Rndrng_NPI"] = (
        df["Rndrng_NPI"]
        .astype(str)
        .str.strip()
    )

    df = df.merge(
        leie[["NPI"]],
        left_on="Rndrng_NPI",
        right_on="NPI",
        how="left"
    )

    df["LEIE_Hit"] = np.where(
        df["NPI"].notna(),
        1,
        0
    )

    df.drop(
        columns=["NPI"],
        inplace=True,
        errors="ignore"
    )

    print(
        "LEIE Matches:",
        df["LEIE_Hit"].sum()
    )

except:

    print(
        "OIG file not found. Skipping OIG validation."
    )

    df["LEIE_Hit"] = 0




print("\nPeer Status Distribution")
print(df["Peer_Status"].value_counts())

# =====================================================
# PRESERVE ORIGINAL VALUES
# =====================================================

df["Orig_Payment"] = df["Tot_Mdcr_Pymt_Amt"]
df["Orig_Service"] = df["Tot_Srvcs"]
df["Orig_Benes"] = df["Tot_Benes"]

# =====================================================
# LOG TRANSFORMATION
# =====================================================

log_features = [
    "Tot_Mdcr_Pymt_Amt",
    "Tot_Srvcs",
    "Tot_Benes"
]

for col in log_features:
    df[col] = np.log1p(df[col])

# =====================================================
# ROBUST Z SCORE FUNCTION
# =====================================================

def robust_z(series):

    median = series.median()

    mad = (series - median).abs().median()

    if mad == 0:
        return pd.Series(
            np.zeros(len(series)),
            index=series.index
        )

    return (
        (series - median)
        /
        (1.4826 * mad)
    )

# =====================================================
# PEER Z SCORES
# =====================================================

df["Payment_ZScore"] = (
    df.groupby("Rndrng_Prvdr_Type")
      ["Tot_Mdcr_Pymt_Amt"]
      .transform(robust_z)
)

df["Service_ZScore"] = (
    df.groupby("Rndrng_Prvdr_Type")
      ["Tot_Srvcs"]
      .transform(robust_z)
)

df["Beneficiary_ZScore"] = (
    df.groupby("Rndrng_Prvdr_Type")
      ["Tot_Benes"]
      .transform(robust_z)
)

print("\nZ Score Summary")
print(
    df[
        [
            "Payment_ZScore",
            "Service_ZScore",
            "Beneficiary_ZScore"
        ]
    ].describe()
)

# =====================================================
# PERCENTILE WITHIN PEER GROUP
# =====================================================

df["Payment_Percentile"] = (
    df.groupby("Rndrng_Prvdr_Type")
      ["Orig_Payment"]
      .rank(pct=True)
)

# =====================================================
# PEER MEDIANS
# =====================================================

peer_medians = (
    df.groupby("Rndrng_Prvdr_Type")
      .agg({
          "Orig_Payment": "median",
          "Orig_Service": "median",
          "Orig_Benes": "median"
      })
      .rename(columns={
          "Orig_Payment": "Peer_Median_Payment",
          "Orig_Service": "Peer_Median_Service",
          "Orig_Benes": "Peer_Median_Benes"
      })
)

df = df.merge(
    peer_medians,
    on="Rndrng_Prvdr_Type",
    how="left"
)

# =====================================================
# GEOGRAPHIC PEER COMPARISON
# =====================================================

geo_peer = (
    df.groupby(
        [
            "Rndrng_Prvdr_Type",
            "Rndrng_Prvdr_State_Abrvtn"
        ]
    )
    .agg({
        "Orig_Payment": "median"
    })
    .rename(columns={
        "Orig_Payment":
        "Geo_Median_Payment"
    })
)

df = df.merge(
    geo_peer,
    on=[
        "Rndrng_Prvdr_Type",
        "Rndrng_Prvdr_State_Abrvtn"
    ],
    how="left"
)

df["Geo_Payment_Ratio"] = (
    df["Orig_Payment"]
    /
    df["Geo_Median_Payment"]
)

df["Geo_Anomaly"] = np.where(
    df["Geo_Payment_Ratio"] > 3,
    1,
    0
)

# =====================================================
# PEER RATIOS
# =====================================================

df["Payment_Ratio"] = (
    df["Orig_Payment"]
    /
    df["Peer_Median_Payment"]
)

df["Service_Ratio"] = (
    df["Orig_Service"]
    /
    df["Peer_Median_Service"]
)

df["Beneficiary_Ratio"] = (
    df["Orig_Benes"]
    /
    df["Peer_Median_Benes"]
)

# =====================================================
# FINANCIAL LEAKAGE
# =====================================================

df["Potential_Leakage"] = np.where(
    df["Payment_Ratio"] > 2,

    df["Orig_Payment"] -
    (2 * df["Peer_Median_Payment"]),

    0
)

df["Potential_Leakage"] = (
    df["Potential_Leakage"]
    .clip(lower=0)
)

total_leakage = (
    df["Potential_Leakage"]
    .sum()
)

print(
    f"\nEstimated Total Leakage: ${total_leakage:,.2f}"
)


print("\nRatio Summary")
print(
    df[
        [
            "Payment_Ratio",
            "Service_Ratio",
            "Beneficiary_Ratio"
        ]
    ].describe()
)




# =====================================================
# POSITIVE ANOMALIES ONLY
# =====================================================

df["Payment_Risk"] = (
    df["Payment_ZScore"]
    .clip(lower=0)
)

df["Service_Risk"] = (
    df["Service_ZScore"]
    .clip(lower=0)
)

df["Beneficiary_Risk"] = (
    df["Beneficiary_ZScore"]
    .clip(lower=0)
)

# =====================================================
# PERCENTILE RISK
# =====================================================

df["Percentile_Risk"] = (
    df["Payment_Percentile"] * 100
)

# =====================================================
# FINAL RAW RISK
# =====================================================

df["Ratio_Risk"] = (
      0.50 * np.log1p(df["Payment_Ratio"])
    + 0.30 * np.log1p(df["Service_Ratio"])
    + 0.20 * np.log1p(df["Beneficiary_Ratio"])
)

# df["Raw_Risk"] = (
#       0.35 * df["Payment_Risk"]
#     + 0.25 * df["Service_Risk"]
#     + 0.25 * df["Beneficiary_Risk"]
#     + 0.15 * (df["Percentile_Risk"] / 100)
#     + 0.15 * df["Ratio_Risk"]
# )


leakage_risk = np.log1p(
    df["Potential_Leakage"]
)

df["Raw_Risk"] = (
      0.30 * df["Payment_Risk"]
    + 0.20 * df["Service_Risk"]
    + 0.20 * df["Beneficiary_Risk"]
    + 0.10 * (df["Percentile_Risk"] / 100)
    + 0.10 * df["Ratio_Risk"]
    + 0.05 * df["Geo_Anomaly"]
    + 0.05 * leakage_risk
)

# =====================================================
# LEIE RISK BOOST
# =====================================================
df["Peer_Risk_Score"] = (
    df["Raw_Risk"]
      .rank(pct=True)
      * 100
)

df.loc[
    df["LEIE_Hit"] == 1,
    "Peer_Risk_Score"
] += 20

df["Peer_Risk_Score"] = (
    df["Peer_Risk_Score"]
    .clip(upper=100)
)




# =====================================================
# SCALE TO 0-100
# =====================================================

# scaler = MinMaxScaler(
#     feature_range=(0, 100)
# )

# df["Peer_Risk_Score"] = scaler.fit_transform(
#     df[["Raw_Risk"]]
# )

# df["Peer_Risk_Score"] = (
#     df["Raw_Risk"]
#       .rank(pct=True)
#       * 100
# )

df.loc[
    df["Peer_Status"] == "INSUFFICIENT_PEERS",
    "Peer_Risk_Score"
] = np.nan

# =====================================================
# RISK LEVELS
# =====================================================

# def risk_level(score):

#     if pd.isna(score):
#         return "INSUFFICIENT_PEERS"

#     elif score >= 80:
#         return "CRITICAL"

#     elif score >= 60:
#         return "HIGH"

#     elif score >= 40:
#         return "MEDIUM"

#     else:
#         return "LOW"

p90 = df["Peer_Risk_Score"].quantile(0.90)
p95 = df["Peer_Risk_Score"].quantile(0.95)
p99 = df["Peer_Risk_Score"].quantile(0.99)


def risk_level(score):

    if pd.isna(score):
        return "INSUFFICIENT_PEERS"
    elif score < p90:
        return "LOW"
    elif score < p95:
        return "MEDIUM"
    elif score < p99:
        return "HIGH"
    else:
        return "CRITICAL"

df["Risk_Level"] = (
    df["Peer_Risk_Score"]
    .apply(risk_level)
)

# =====================================================
# RESULTS
# =====================================================

print("\nRisk Distribution")
print(df["Risk_Level"].value_counts())

print("\nRisk Score Summary")
print(
    df["Peer_Risk_Score"]
    .describe()
)

# =====================================================
# TOP SUSPICIOUS PROVIDERS
# =====================================================

top10 = (
    df.sort_values(
        by="Peer_Risk_Score",
        ascending=False
    )
    [[
        "Rndrng_NPI",
        "Rndrng_Prvdr_Type",
        "Payment_Ratio",
        "Service_Ratio",
        "Beneficiary_Ratio",
        "Payment_Percentile",
        "Peer_Risk_Score",
        "Risk_Level"
    ]]
    .head(10)
)

def fraud_flag(row):

    if row["LEIE_Hit"] == 1:
        return "EXCLUDED_PROVIDER"
    
    if row["Risk_Level"] == "CRITICAL":
        return "REVIEW_IMMEDIATELY"

    elif row["Risk_Level"] == "HIGH":
        return "REVIEW"

    elif row["Risk_Level"] == "MEDIUM":
        return "MONITOR"

    elif row["Risk_Level"] == "LOW":
        return "NORMAL"

    return "INSUFFICIENT_DATA"


df["Fraud_Flag"] = df.apply(
    fraud_flag,
    axis=1
)


print("\nTOP 10 HIGH-RISK PROVIDERS")
print(top10)

# Reason Explainability
def generate_reason(row):

    reasons = []

    if row["Payment_ZScore"] > 2:
        reasons.append(
            f"Payment significantly above peers (Z={row['Payment_ZScore']:.2f})"
        )

    if row["Service_ZScore"] > 2:
        reasons.append(
            f"Service volume significantly above peers (Z={row['Service_ZScore']:.2f})"
        )

    if row["Beneficiary_ZScore"] > 2:
        reasons.append(
            f"Beneficiary count significantly above peers (Z={row['Beneficiary_ZScore']:.2f})"
        )

    if row["Payment_Ratio"] > 5:
        reasons.append(
            f"Payment is {row['Payment_Ratio']:.1f}x peer median"
        )

    if row["Geo_Payment_Ratio"] > 3:
        reasons.append(
            f"Payment is {row['Geo_Payment_Ratio']:.1f}x higher than state peers"
        )

    if row["Payment_Percentile"] > 0.99:
        reasons.append(
            "Top 1% provider by Medicare payments"
        )

    if row["Potential_Leakage"] > 50000:
        reasons.append(
            f"Potential excess payment ${row['Potential_Leakage']:,.0f}"
        )

    if row["LEIE_Hit"] == 1:
        reasons.append(
            "Provider appears in OIG exclusion database"
        )

    if len(reasons) == 0:
        return "Within expected peer range"

    return " | ".join(reasons)

df["Explainability"] = df.apply(
    generate_reason,
    axis=1
)

# DASHBOARD GRAPHS

# Risk Level Distribution
df["Risk_Level"].value_counts().plot(
    kind="bar"
)

plt.title("Provider Risk Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Top 10 High Risk Providers
top10 = df.sort_values(
    "Peer_Risk_Score",
    ascending=False
).head(10)

plt.figure(figsize=(10,5))

plt.bar(
    top10["Rndrng_NPI"].astype(str),
    top10["Peer_Risk_Score"]
)

plt.xticks(rotation=90)

plt.title("Top 10 High Risk Providers")

plt.tight_layout()
plt.show()


# Payment Ratio vs Risk Score
plt.figure(figsize=(8,6))

plt.scatter(
    df["Payment_Ratio"],
    df["Peer_Risk_Score"],
    alpha=0.4
)

plt.xlabel("Payment Ratio")
plt.ylabel("Risk Score")

plt.title(
    "Payment Ratio vs Peer Risk Score"
)

plt.show()

# Leakage Chart
top_leakage = (
    df.sort_values(
        "Potential_Leakage",
        ascending=False
    )
    .head(10)
)

plt.figure(figsize=(10,5))

plt.bar(
    top_leakage["Rndrng_NPI"].astype(str),
    top_leakage["Potential_Leakage"]
)

plt.xticks(rotation=90)

plt.title(
    "Top Potential Financial Leakage"
)

plt.tight_layout()
plt.show()


# Geographic Anomalies
geo_counts = (
    df["Geo_Anomaly"]
    .value_counts()
)

plt.figure(figsize=(5,4))

geo_counts.plot(
    kind="bar"
)

plt.title(
    "Geographic Peer Anomalies"
)

plt.show()

# =====================================================
# SAVE OUTPUT
# =====================================================

final_output = df[
    [
        "Rndrng_NPI",
        "Rndrng_Prvdr_Type",

        "Payment_ZScore",
        "Service_ZScore",
        "Beneficiary_ZScore",

        "Payment_Ratio",
        "Service_Ratio",
        "Beneficiary_Ratio",

        "Payment_Percentile",
        "LEIE_Hit",
        "Geo_Payment_Ratio",
        "Potential_Leakage",

        "Peer_Risk_Score",
        "Risk_Level",

        "Fraud_Flag",
        "Explainability"
    ]
]

df.to_csv(
    "PEER_ANALYSIS_RESULTS.csv",
    index=False
)

final_output.to_csv(
    "PEER_ANALYSIS_RESULTS_col.csv",
    index=False
)

print(
    "\nSaved: PEER_ANALYSIS_RESULTS.csv"
)