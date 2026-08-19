import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

# --------------------------------------------------
# Load evaluation results
# --------------------------------------------------

FILE = Path("processed/evaluation_results.csv")

df = pd.read_csv(FILE)

print("=" * 70)
print("ISOLATION FOREST THRESHOLD ANALYSIS")
print("=" * 70)

print("\nProviders:", len(df))

# --------------------------------------------------
# Actual labels
# --------------------------------------------------

y_true = df["Fraud_Label"]

scores = df["Anomaly_Score"]

# --------------------------------------------------
# Try multiple score thresholds
# --------------------------------------------------

thresholds = [
    0.50,
    0.51,
    0.52,
    0.53,
    0.54,
    0.55,
    0.56,
    0.57,
    0.58,
    0.59,
    0.60,
    0.61,
    0.62,
    0.63,
    0.64,
    0.65
]

results = []

for threshold in thresholds:

    y_pred = (scores >= threshold).astype(int)

    alerts = y_pred.sum()

    alert_percentage = (
        alerts / len(df)
    ) * 100

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    results.append({
        "Threshold": threshold,
        "Alerts": alerts,
        "Alert_%": alert_percentage,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

# --------------------------------------------------
# Create result table
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\nThreshold comparison:\n")

print(
    results_df.to_string(
        index=False,
        formatters={
            "Alert_%": "{:.2f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1": "{:.4f}".format
        }
    )
)

# --------------------------------------------------
# Best F1 threshold
# --------------------------------------------------

best = results_df.loc[
    results_df["F1"].idxmax()
]

print("\n" + "=" * 70)
print("BEST F1 THRESHOLD")
print("=" * 70)

print(
    f"Threshold : {best['Threshold']:.2f}"
)

print(
    f"Alerts    : {int(best['Alerts'])}"
)

print(
    f"Alert %   : {best['Alert_%']:.2f}%"
)

print(
    f"Precision : {best['Precision']:.4f}"
)

print(
    f"Recall    : {best['Recall']:.4f}"
)

print(
    f"F1 Score  : {best['F1']:.4f}"
)

# --------------------------------------------------
# Save results
# --------------------------------------------------

output = Path(
    "processed/threshold_analysis.csv"
)

results_df.to_csv(
    output,
    index=False
)

print("\nSaved:")
print(output)

print("\nThreshold analysis completed.")