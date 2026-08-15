import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import LocalOutlierFactor


# ============================================================
# MEMBER 5 - LOF PROVIDER ANOMALY DETECTION
# FINAL HACKATHON VERSION
# ============================================================

PROVIDER_FILE = "PROVIDER_ML_READY.csv"
LEIE_FILE = "leie_clean_specialty_filled.csv"
OUTPUT_FILE = "member5_lof_results.csv"

MIN_PEERS = 20
N_NEIGHBORS = 20
CONTAMINATION = 0.03


# ============================================================
# 1. LOAD PROVIDER DATA
# ============================================================

print("=" * 70)
print("       MEMBER 5 - LOF PROVIDER ANOMALY DETECTION")
print("=" * 70)

df = pd.read_csv(
    PROVIDER_FILE,
    low_memory=False
)

print(f"\nTotal Providers: {len(df):,}")


# ============================================================
# 2. FEATURES REQUIRED BY TEAM SPECIFICATION
# ============================================================

features = [
    "Tot_Srvcs",
    "Tot_Benes",
    "Tot_Mdcr_Pymt_Amt",
    "Services_Per_Beneficiary",
    "Charge_Per_Service",
    "Payment_to_Allowed_Ratio"
]

log_cols = [
    "Tot_Srvcs",
    "Tot_Benes",
    "Tot_Mdcr_Pymt_Amt",
    "Charge_Per_Service"
]

required_columns = [
    "Rndrng_NPI",
    "Rndrng_Prvdr_Type"
] + features


# ============================================================
# 3. VALIDATE INPUT COLUMNS
# ============================================================

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    print("\nERROR - Missing required columns:")

    for col in missing_columns:
        print(f" - {col}")

    raise SystemExit(1)

print("\nAll required provider columns are available.")


# ============================================================
# 4. PREPARE OUTPUT COLUMNS
# ============================================================

df["LOF_Score"] = np.nan

df["LOF_Anomaly"] = False

df["Risk_Score"] = np.nan

df["Risk_Level"] = "INSUFFICIENT_PEERS"

df["Payment_Ratio"] = np.nan

df["Service_Ratio"] = np.nan

df["Beneficiary_Ratio"] = np.nan

df["LEIE_Match"] = False

df["Explanation"] = ""


# ============================================================
# 5. SPECIALTY GROUPING
# ============================================================

specialty_counts = (
    df["Rndrng_Prvdr_Type"]
    .value_counts()
)

valid_specialties = specialty_counts[
    specialty_counts >= MIN_PEERS
]

small_specialties = specialty_counts[
    specialty_counts < MIN_PEERS
]

print("\n" + "=" * 70)
print("SPECIALTY GROUPING")
print("=" * 70)

print(
    f"\nTotal specialties: "
    f"{len(specialty_counts)}"
)

print(
    f"Specialties with >= {MIN_PEERS} providers: "
    f"{len(valid_specialties)}"
)

print(
    f"Specialties with < {MIN_PEERS} providers: "
    f"{len(small_specialties)}"
)


# ============================================================
# 6. PROCESS EACH SPECIALTY
# ============================================================

processed_count = 0
lof_anomaly_count = 0


for specialty, group in df.groupby(
    "Rndrng_Prvdr_Type"
):

    # --------------------------------------------------------
    # Minimum peer requirement
    # --------------------------------------------------------

    if len(group) < MIN_PEERS:
        continue

    print(
        f"\nProcessing: {specialty} "
        f"({len(group):,} providers)"
    )

    group = group.copy()

    # --------------------------------------------------------
    # Preserve original values for explanations
    # --------------------------------------------------------

    original_values = group[
        [
            "Tot_Srvcs",
            "Tot_Benes",
            "Tot_Mdcr_Pymt_Amt",
            "Charge_Per_Service"
        ]
    ].copy()


    # ========================================================
    # 7. HANDLE MISSING VALUES
    # ========================================================

    for col in features:

        median_value = group[col].median()

        # If the entire column is missing within a specialty,
        # use zero as a defensive fallback.
        if pd.isna(median_value):
            median_value = 0.0

        group[col] = group[col].fillna(
            median_value
        )


    # ========================================================
    # 8. LOG TRANSFORMATION
    # ========================================================

    for col in log_cols:

        # Count/financial values should not be negative
        # before log1p.
        group[col] = group[col].clip(
            lower=0
        )

        group[col] = np.log1p(
            group[col]
        )


    # ========================================================
    # 9. ROBUST SCALING
    # ========================================================

    scaler = RobustScaler()

    X = scaler.fit_transform(
        group[features]
    )


    # ========================================================
    # 10. LOF MODEL
    # ========================================================

    neighbors = min(
        N_NEIGHBORS,
        len(group) - 1
    )

    lof = LocalOutlierFactor(
        n_neighbors=neighbors,
        contamination=CONTAMINATION
    )

    predictions = lof.fit_predict(X)


    # ========================================================
    # 11. LOF SCORE
    # ========================================================

    # sklearn negative_outlier_factor_:
    # more negative = more anomalous
    #
    # Negating it makes:
    # higher score = higher anomaly severity

    lof_scores = (
        -lof.negative_outlier_factor_
    )


    # ========================================================
    # 12. ACTUAL LOF ANOMALY
    # ========================================================

    # -1 = anomaly
    #  1 = normal

    lof_anomalies = (
        predictions == -1
    )


    # ========================================================
    # 13. SPECIALTY-RELATIVE RISK SCORE
    # ========================================================

    # Percentile ranking within the specialty.
    #
    # This avoids making the maximum provider in every
    # specialty automatically equivalent to a global risk.
    #
    # 0-100:
    # higher = more anomalous relative to specialty peers.

    risk_scores = (
        pd.Series(
            lof_scores,
            index=group.index
        )
        .rank(
            method="average",
            pct=True
        )
        * 100
    ).to_numpy()


    # ========================================================
    # 14. RISK LEVEL
    # ========================================================

    risk_levels = np.select(
        [
            risk_scores <= 30,
            risk_scores <= 60,
            risk_scores <= 80,
            risk_scores > 80
        ],
        [
            "Low",
            "Medium",
            "High",
            "Critical"
        ],
        default="Low"
    )


    # ========================================================
    # 15. PEER MEDIANS
    # ========================================================

    peer_payment_median = (
        original_values[
            "Tot_Mdcr_Pymt_Amt"
        ].median()
    )

    peer_service_median = (
        original_values[
            "Tot_Srvcs"
        ].median()
    )

    peer_beneficiary_median = (
        original_values[
            "Tot_Benes"
        ].median()
    )


    # ========================================================
    # 16. PAYMENT RATIO
    # ========================================================

    if (
        pd.notna(peer_payment_median)
        and peer_payment_median != 0
    ):

        payment_ratios = (
            original_values[
                "Tot_Mdcr_Pymt_Amt"
            ]
            / peer_payment_median
        )

    else:

        payment_ratios = pd.Series(
            0.0,
            index=group.index
        )


    # ========================================================
    # 17. SERVICE RATIO
    # ========================================================

    if (
        pd.notna(peer_service_median)
        and peer_service_median != 0
    ):

        service_ratios = (
            original_values[
                "Tot_Srvcs"
            ]
            / peer_service_median
        )

    else:

        service_ratios = pd.Series(
            0.0,
            index=group.index
        )


    # ========================================================
    # 18. BENEFICIARY RATIO
    # ========================================================

    if (
        pd.notna(peer_beneficiary_median)
        and peer_beneficiary_median != 0
    ):

        beneficiary_ratios = (
            original_values[
                "Tot_Benes"
            ]
            / peer_beneficiary_median
        )

    else:

        beneficiary_ratios = pd.Series(
            0.0,
            index=group.index
        )


    # ========================================================
    # 19. EXPLAINABLE REASONS
    # ========================================================

    explanations = []


    for i in range(len(group)):

        reasons = []

        payment_ratio = (
            payment_ratios.iloc[i]
        )

        service_ratio = (
            service_ratios.iloc[i]
        )

        beneficiary_ratio = (
            beneficiary_ratios.iloc[i]
        )


        # ----------------------------------------------------
        # Payment deviation
        # ----------------------------------------------------

        if payment_ratio >= 2:

            reasons.append(
                f"Medicare payment amount is "
                f"{payment_ratio:.1f}x higher than "
                f"specialty peer median"
            )

        elif (
            payment_ratio > 0
            and payment_ratio <= 0.5
        ):

            reasons.append(
                f"Medicare payment amount is "
                f"{1 / payment_ratio:.1f}x lower than "
                f"specialty peer median"
            )


        # ----------------------------------------------------
        # Service deviation
        # ----------------------------------------------------

        if service_ratio >= 2:

            reasons.append(
                f"Services are "
                f"{service_ratio:.1f}x higher than "
                f"specialty peer median"
            )

        elif (
            service_ratio > 0
            and service_ratio <= 0.5
        ):

            reasons.append(
                f"Services are "
                f"{1 / service_ratio:.1f}x lower than "
                f"specialty peer median"
            )


        # ----------------------------------------------------
        # Beneficiary deviation
        # ----------------------------------------------------

        if beneficiary_ratio >= 2:

            reasons.append(
                f"Beneficiaries are "
                f"{beneficiary_ratio:.1f}x higher than "
                f"specialty peer median"
            )

        elif (
            beneficiary_ratio > 0
            and beneficiary_ratio <= 0.5
        ):

            reasons.append(
                f"Beneficiaries are "
                f"{1 / beneficiary_ratio:.1f}x lower than "
                f"specialty peer median"
            )


        # ----------------------------------------------------
        # LOF anomaly explanation
        # ----------------------------------------------------

        if lof_anomalies[i]:

            reasons.append(
                "Provider identified as a local "
                "density outlier by LOF"
            )


        # ----------------------------------------------------
        # Normal provider explanation
        # ----------------------------------------------------

        if not reasons:

            reasons.append(
                "Billing behavior is within the "
                "observed specialty peer range"
            )


        explanations.append(
            " | ".join(reasons)
        )


    # ========================================================
    # 20. SAVE SPECIALTY RESULTS
    # ========================================================

    indices = group.index

    df.loc[
        indices,
        "LOF_Score"
    ] = lof_scores

    df.loc[
        indices,
        "LOF_Anomaly"
    ] = lof_anomalies

    df.loc[
        indices,
        "Risk_Score"
    ] = risk_scores

    df.loc[
        indices,
        "Risk_Level"
    ] = risk_levels

    df.loc[
        indices,
        "Payment_Ratio"
    ] = payment_ratios.values

    df.loc[
        indices,
        "Service_Ratio"
    ] = service_ratios.values

    df.loc[
        indices,
        "Beneficiary_Ratio"
    ] = beneficiary_ratios.values

    df.loc[
        indices,
        "Explanation"
    ] = explanations


    processed_count += len(group)

    lof_anomaly_count += int(
        np.sum(lof_anomalies)
    )


# ============================================================
# 21. ROUND NUMERICAL RESULTS
# ============================================================

df["LOF_Score"] = (
    df["LOF_Score"]
    .round(4)
)

df["Risk_Score"] = (
    df["Risk_Score"]
    .round(2)
)

df["Payment_Ratio"] = (
    df["Payment_Ratio"]
    .round(2)
)

df["Service_Ratio"] = (
    df["Service_Ratio"]
    .round(2)
)

df["Beneficiary_Ratio"] = (
    df["Beneficiary_Ratio"]
    .round(2)
)


# ============================================================
# 22. LEIE OVERLAP ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("LEIE OVERLAP ANALYSIS")
print("=" * 70)


leie = pd.read_csv(
    LEIE_FILE,
    low_memory=False
)

print(
    f"\nTotal LEIE records: "
    f"{len(leie):,}"
)


# ------------------------------------------------------------
# Validate LEIE NPI column
# ------------------------------------------------------------

if "NPI" not in leie.columns:

    print(
        "\nERROR: LEIE dataset does not contain "
        "an NPI column."
    )

    raise SystemExit(1)


# ------------------------------------------------------------
# Clean NPIs
# ------------------------------------------------------------

provider_npi = pd.to_numeric(
    df["Rndrng_NPI"],
    errors="coerce"
)

leie_npi = pd.to_numeric(
    leie["NPI"],
    errors="coerce"
)


# ------------------------------------------------------------
# Valid LEIE NPIs
# ------------------------------------------------------------

valid_leie_npi = (
    leie_npi.notna()
    &
    (leie_npi > 0)
)


leie_npi_set = set(
    leie_npi[
        valid_leie_npi
    ]
    .astype("int64")
)


print(
    f"Valid unique LEIE NPIs: "
    f"{len(leie_npi_set):,}"
)


# ------------------------------------------------------------
# Match provider NPIs against LEIE
# ------------------------------------------------------------

valid_provider_npi = (
    provider_npi.notna()
    &
    (provider_npi > 0)
)

df["LEIE_Match"] = False

df.loc[
    valid_provider_npi,
    "LEIE_Match"
] = (
    provider_npi[
        valid_provider_npi
    ]
    .astype("int64")
    .isin(leie_npi_set)
    .values
)


# ============================================================
# 23. ACTUAL LOF ANOMALY VALIDATION
# ============================================================

lof_flagged = (
    df["LOF_Anomaly"] == True
)

actual_lof_count = int(
    lof_flagged.sum()
)


# ============================================================
# 24. LEIE OVERLAP AMONG ACTUAL LOF ANOMALIES
# ============================================================

lof_leie_matches = (
    lof_flagged
    &
    df["LEIE_Match"]
)

leie_matches_count = int(
    lof_leie_matches.sum()
)


if actual_lof_count > 0:

    leie_overlap_percentage = (
        leie_matches_count
        /
        actual_lof_count
    ) * 100

else:

    leie_overlap_percentage = 0.0


print(
    f"\nActual LOF anomaly providers: "
    f"{actual_lof_count:,}"
)

print(
    f"LEIE matches among LOF anomalies: "
    f"{leie_matches_count:,}"
)

print(
    f"LEIE overlap percentage: "
    f"{leie_overlap_percentage:.2f}%"
)


# ============================================================
# 25. TOP 1%, 3%, 5% RISK PROVIDERS
# ============================================================

valid_results = df[
    df["Risk_Score"].notna()
].copy()

valid_results = (
    valid_results
    .sort_values(
        "Risk_Score",
        ascending=False
    )
)


top_1_count = max(
    1,
    int(len(valid_results) * 0.01)
)

top_3_count = max(
    1,
    int(len(valid_results) * 0.03)
)

top_5_count = max(
    1,
    int(len(valid_results) * 0.05)
)


# ============================================================
# 26. FINAL OUTPUT COLUMNS
# ============================================================

output_columns = [
    "Rndrng_NPI",
    "Rndrng_Prvdr_Type",
    "LOF_Score",
    "LOF_Anomaly",
    "Risk_Score",
    "Risk_Level",
    "Payment_Ratio",
    "Service_Ratio",
    "Beneficiary_Ratio",
    "LEIE_Match",
    "Explanation"
]


results = df[
    output_columns
].copy()


# ============================================================
# 27. SAVE FINAL RESULTS
# ============================================================

results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 28. FINAL VALIDATION SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL LOF VALIDATION SUMMARY")
print("=" * 70)

print(
    f"\nTotal Providers: "
    f"{len(df):,}"
)

print(
    f"Providers Analyzed by LOF: "
    f"{processed_count:,}"
)

print(
    f"Insufficient Peer Providers: "
    f"{len(df) - processed_count:,}"
)

print(
    f"Actual LOF Anomalies: "
    f"{actual_lof_count:,}"
)

print(
    f"Normal Providers: "
    f"{processed_count - actual_lof_count:,}"
)

print(
    f"LEIE Matches among LOF Anomalies: "
    f"{leie_matches_count:,}"
)

print(
    f"LEIE Overlap: "
    f"{leie_overlap_percentage:.2f}%"
)


# ============================================================
# 29. RISK LEVEL DISTRIBUTION
# ============================================================

print("\nRisk Level Distribution:")

print(
    results[
        "Risk_Level"
    ].value_counts()
)


# ============================================================
# 30. TOP 1%, 3%, 5%
# ============================================================

print(
    f"\nTop 1% providers: "
    f"{top_1_count:,}"
)

print(
    f"Top 3% providers: "
    f"{top_3_count:,}"
)

print(
    f"Top 5% providers: "
    f"{top_5_count:,}"
)


# ============================================================
# 31. TOP 10 HIGH-RISK PROVIDERS
# ============================================================

print("\n" + "=" * 70)
print("TOP 10 HIGH-RISK PROVIDERS")
print("=" * 70)

top10 = valid_results[
    [
        "Rndrng_NPI",
        "Rndrng_Prvdr_Type",
        "LOF_Score",
        "LOF_Anomaly",
        "Risk_Score",
        "Risk_Level",
        "LEIE_Match"
    ]
].head(10)

print(
    top10.to_string(
        index=False
    )
)


# ============================================================
# 32. LOF + LEIE MATCHED PROVIDERS
# ============================================================

print("\n" + "=" * 70)
print("LOF + LEIE MATCHED PROVIDERS")
print("=" * 70)

matched = results[
    (
        results["LOF_Anomaly"] == True
    )
    &
    (
        results["LEIE_Match"] == True
    )
]

if len(matched) > 0:

    print(
        matched[
            [
                "Rndrng_NPI",
                "Rndrng_Prvdr_Type",
                "LOF_Score",
                "Risk_Score",
                "Risk_Level",
                "LEIE_Match"
            ]
        ]
        .sort_values(
            "Risk_Score",
            ascending=False
        )
        .head(20)
        .to_string(index=False)
    )

else:

    print(
        "\nNo LOF anomaly providers "
        "matched a valid LEIE NPI."
    )


# ============================================================
# 33. FINAL COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("PROCESS COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    f"\nOutput file created:"
)

print(
    OUTPUT_FILE
)

print(
    "\nMember 5 LOF + LEIE analysis completed."
)

print(
    "\nFinal output columns:"
)

for col in output_columns:
    print(
        f" - {col}"
    )