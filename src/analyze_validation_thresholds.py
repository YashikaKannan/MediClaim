import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

VALIDATION_FILE = (
    BASE_DIR
    / "processed"
    / "validation_results.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "processed"
    / "validation_threshold_analysis.csv"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("VALIDATION THRESHOLD ANALYSIS")
print("=" * 70)


# ============================================================
# LOAD VALIDATION RESULTS
# ============================================================

print("\nLoading validation results...")

df = pd.read_csv(VALIDATION_FILE)

print(
    f"Validation data shape: {df.shape}"
)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Provider",
    "Anomaly_Score",
    "Actual"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# DATA QUALITY
# ============================================================

print("\nChecking validation data...")

print(
    f"Missing values : {df[required_columns].isna().sum().sum()}"
)

print(
    f"Infinite scores: "
    f"{np.isinf(df['Anomaly_Score']).sum()}"
)


# ============================================================
# ACTUAL LABEL DISTRIBUTION
# ============================================================

print("\nActual validation distribution:")

print(
    df["Actual"].value_counts()
    .sort_index()
)

fraud_count = int(
    (df["Actual"] == 1).sum()
)

normal_count = int(
    (df["Actual"] == 0).sum()
)

print(
    f"\nNormal providers : {normal_count}"
)

print(
    f"Fraud providers  : {fraud_count}"
)

print(
    f"Fraud percentage : "
    f"{fraud_count / len(df) * 100:.2f}%"
)


# ============================================================
# THRESHOLD RANGE
# ============================================================

thresholds = np.round(
    np.arange(
        0.45,
        0.701,
        0.01
    ),
    2
)


# ============================================================
# ANALYZE THRESHOLDS
# ============================================================

results = []


print("\n")
print("=" * 70)
print("THRESHOLD COMPARISON")
print("=" * 70)

print(
    "\nThreshold  Alerts  Alert%  Precision  Recall  F1"
)

print(
    "-" * 70
)


for threshold in thresholds:

    predicted = (
        df["Anomaly_Score"]
        >= threshold
    ).astype(int)

    alerts = int(
        predicted.sum()
    )

    alert_percentage = (
        alerts / len(df) * 100
    )

    precision = precision_score(
        df["Actual"],
        predicted,
        zero_division=0
    )

    recall = recall_score(
        df["Actual"],
        predicted,
        zero_division=0
    )

    f1 = f1_score(
        df["Actual"],
        predicted,
        zero_division=0
    )

    results.append(
        {
            "Threshold": threshold,
            "Alerts": alerts,
            "Alert_Percentage": alert_percentage,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }
    )

    print(
        f"{threshold:8.2f} "
        f"{alerts:7d} "
        f"{alert_percentage:7.2f}% "
        f"{precision:9.4f} "
        f"{recall:7.4f} "
        f"{f1:7.4f}"
    )


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# BEST F1
# ============================================================

best_f1_row = (
    results_df
    .loc[
        results_df["F1"].idxmax()
    ]
)


# ============================================================
# BEST PRECISION
# ============================================================

best_precision_row = (
    results_df
    .loc[
        results_df["Precision"].idxmax()
    ]
)


# ============================================================
# BEST RECALL
# ============================================================

best_recall_row = (
    results_df
    .loc[
        results_df["Recall"].idxmax()
    ]
)


# ============================================================
# PRINT BEST RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("BEST F1 THRESHOLD")
print("=" * 70)

print(
    f"Threshold : "
    f"{best_f1_row['Threshold']:.2f}"
)

print(
    f"Alerts    : "
    f"{int(best_f1_row['Alerts'])}"
)

print(
    f"Alert %   : "
    f"{best_f1_row['Alert_Percentage']:.2f}%"
)

print(
    f"Precision : "
    f"{best_f1_row['Precision']:.4f}"
)

print(
    f"Recall    : "
    f"{best_f1_row['Recall']:.4f}"
)

print(
    f"F1 Score  : "
    f"{best_f1_row['F1']:.4f}"
)


print("\n")
print("=" * 70)
print("BEST PRECISION THRESHOLD")
print("=" * 70)

print(
    f"Threshold : "
    f"{best_precision_row['Threshold']:.2f}"
)

print(
    f"Alerts    : "
    f"{int(best_precision_row['Alerts'])}"
)

print(
    f"Alert %   : "
    f"{best_precision_row['Alert_Percentage']:.2f}%"
)

print(
    f"Precision : "
    f"{best_precision_row['Precision']:.4f}"
)

print(
    f"Recall    : "
    f"{best_precision_row['Recall']:.4f}"
)

print(
    f"F1 Score  : "
    f"{best_precision_row['F1']:.4f}"
)


print("\n")
print("=" * 70)
print("BEST RECALL THRESHOLD")
print("=" * 70)

print(
    f"Threshold : "
    f"{best_recall_row['Threshold']:.2f}"
)

print(
    f"Alerts    : "
    f"{int(best_recall_row['Alerts'])}"
)

print(
    f"Alert %   : "
    f"{best_recall_row['Alert_Percentage']:.2f}%"
)

print(
    f"Precision : "
    f"{best_recall_row['Precision']:.4f}"
)

print(
    f"Recall    : "
    f"{best_recall_row['Recall']:.4f}"
)

print(
    f"F1 Score  : "
    f"{best_recall_row['F1']:.4f}"
)


# ============================================================
# CURRENT THRESHOLD 0.54
# ============================================================

current_threshold = 0.54

current_row = results_df[
    results_df["Threshold"]
    == current_threshold
]

if not current_row.empty:

    current_row = current_row.iloc[0]

    print("\n")
    print("=" * 70)
    print("CURRENT THRESHOLD COMPARISON")
    print("=" * 70)

    print(
        f"Current threshold : "
        f"{current_threshold:.2f}"
    )

    print(
        f"Alerts            : "
        f"{int(current_row['Alerts'])}"
    )

    print(
        f"Alert %           : "
        f"{current_row['Alert_Percentage']:.2f}%"
    )

    print(
        f"Precision         : "
        f"{current_row['Precision']:.4f}"
    )

    print(
        f"Recall            : "
        f"{current_row['Recall']:.4f}"
    )

    print(
        f"F1 Score          : "
        f"{current_row['F1']:.4f}"
    )


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n")
print("=" * 70)
print("VALIDATION THRESHOLD ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"\nSaved file:\n{OUTPUT_FILE}"
)

print(
    f"\nRows analyzed: {len(df)}"
)

print("\nDONE")

print("=" * 70)