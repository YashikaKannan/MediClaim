import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model


DATA_PATH = "CLAIM_LEVEL_AUTOENCODER_DATA.csv"
features = [
    "CLM_PMT_AMT",
    "NCH_CARR_CLM_SBMTD_CHRG_AMT",
    "NCH_CARR_CLM_ALOWD_AMT",
    "LINE_SBMTD_CHRG_AMT",
    "LINE_ALOWD_CHRG_AMT",
    "LINE_PRVDR_PMT_AMT",
    "LINE_SRVC_CNT",
    "DIAGNOSIS_COUNT"
]


reason_names = {
    "CLM_PMT_AMT":
        "Unusual total claim payment",

    "NCH_CARR_CLM_SBMTD_CHRG_AMT":
        "Unusual submitted claim charge",

    "NCH_CARR_CLM_ALOWD_AMT":
        "Unusual allowed claim amount",

    "LINE_SBMTD_CHRG_AMT":
        "Unusual total line submitted charges",

    "LINE_ALOWD_CHRG_AMT":
        "Unusual total line allowed charges",

    "LINE_PRVDR_PMT_AMT":
        "Unusual total provider payment",

    "LINE_SRVC_CNT":
        "Unusual service count",

    "DIAGNOSIS_COUNT":
        "Unusual diagnosis count",

    "CLAIM_LINE_COUNT":
        "Unusual number of claim lines"
}


print("Loading claim-level dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("Unique claims:", df["CLM_ID"].nunique())


# ----------------------------------
# Prepare features
# ----------------------------------

print("\nPreparing features...")

X_df = df[features].copy()

X_df = X_df.apply(
    pd.to_numeric,
    errors="coerce"
)

X_df = X_df.replace(
    [np.inf, -np.inf],
    np.nan
)

X_df = X_df.fillna(
    X_df.median()
)

X_log = np.log1p(X_df)


# ----------------------------------
# Load scaler
# ----------------------------------

print("Loading scaler...")

scaler = joblib.load(
    "claim_level_scaler.pkl"
)

X = scaler.transform(X_log)


# ----------------------------------
# Load model
# ----------------------------------

print("Loading claim-level Autoencoder...")

autoencoder = load_model(
    "claim_level_autoencoder.keras"
)


# ----------------------------------
# Reconstruction
# ----------------------------------

print("Reconstructing claims...")

reconstructed = autoencoder.predict(
    X,
    batch_size=1024,
    verbose=1
)


# ----------------------------------
# Reconstruction errors
# ----------------------------------

print("Calculating reconstruction errors...")

feature_errors = np.square(
    X - reconstructed
)

mse = np.mean(
    feature_errors,
    axis=1
)


# ----------------------------------
# Percentile risk score
# ----------------------------------

print("Calculating percentile risk scores...")

risk_score = (
    pd.Series(mse)
    .rank(
        method="average",
        pct=True
    )
    * 100
)

df["RISK_SCORE"] = np.round(
    risk_score.to_numpy(),
    2
)

df["RECONSTRUCTION_ERROR"] = mse


# ----------------------------------
# Top 1% threshold
# ----------------------------------

threshold = np.percentile(
    mse,
    99
)

df["FLAG"] = (
    mse >= threshold
)


# ----------------------------------
# Top 3 reasons
# ----------------------------------

print("Generating explainable reasons...")

top_indices = np.argsort(
    feature_errors,
    axis=1
)[:, -3:][:, ::-1]


df["TOP_REASON_1"] = [
    reason_names[features[row[0]]]
    for row in top_indices
]

df["TOP_REASON_2"] = [
    reason_names[features[row[1]]]
    for row in top_indices
]

df["TOP_REASON_3"] = [
    reason_names[features[row[2]]]
    for row in top_indices
]


# ----------------------------------
# Final investigation output
# ----------------------------------

output_columns = [
    "CLM_ID",
    "BENE_ID",
    "RISK_SCORE",
    "RECONSTRUCTION_ERROR",
    "FLAG",
    "TOP_REASON_1",
    "TOP_REASON_2",
    "TOP_REASON_3"
]

results = df[output_columns].copy()


results = results.sort_values(
    "RISK_SCORE",
    ascending=False
)


OUTPUT_PATH = "CLAIM_LEVEL_AUTOENCODER_RESULTS.csv"

results.to_csv(
    OUTPUT_PATH,
    index=False
)


# ----------------------------------
# Summary
# ----------------------------------

print("\n================================")
print("FINAL CLAIM-LEVEL RESULTS")
print("================================")

print(
    "Total Claims:",
    len(results)
)

print(
    "Unique Claims:",
    results["CLM_ID"].nunique()
)

print(
    "Duplicate CLM_ID:",
    results["CLM_ID"].duplicated().sum()
)

print(
    "Flagged Claims:",
    int(results["FLAG"].sum())
)

print(
    "Flagged Percentage:",
    round(
        results["FLAG"].mean() * 100,
        2
    ),
    "%"
)

print(
    "99th Percentile MSE Threshold:",
    threshold
)

print(
    "Minimum Risk Score:",
    results["RISK_SCORE"].min()
)

print(
    "Maximum Risk Score:",
    results["RISK_SCORE"].max()
)

print(
    "\nOutput saved:",
    OUTPUT_PATH
)

print("================================")
print("FINAL CLAIM-LEVEL PIPELINE COMPLETE!")
print("================================")
