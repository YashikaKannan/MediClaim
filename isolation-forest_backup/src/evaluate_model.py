import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score
)

SCORES_FILE = Path("processed/anomaly_scores.csv")
LABEL_FILE = Path("data/raw/Train-1542865627584.csv")

# --------------------------------------------------
# 1. Load predictions and labels
# --------------------------------------------------

scores = pd.read_csv(SCORES_FILE)

labels = pd.read_csv(LABEL_FILE)

print("=" * 70)
print("ISOLATION FOREST EVALUATION")
print("=" * 70)

print("\nScores:", scores.shape)
print("Labels:", labels.shape)

# --------------------------------------------------
# 2. Convert PotentialFraud to binary
# --------------------------------------------------

labels["Fraud_Label"] = (
    labels["PotentialFraud"]
    .map({
        "Yes": 1,
        "No": 0
    })
)

# --------------------------------------------------
# 3. Merge
# --------------------------------------------------

evaluation = scores.merge(
    labels[["Provider", "PotentialFraud", "Fraud_Label"]],
    on="Provider",
    how="inner"
)

print("\nMerged evaluation data:")
print(evaluation.shape)

# --------------------------------------------------
# 4. Check label distribution
# --------------------------------------------------

print("\nActual label distribution:")

print(
    evaluation["PotentialFraud"]
    .value_counts()
)

print("\nActual fraud percentage:")

fraud_rate = evaluation["Fraud_Label"].mean() * 100

print(
    round(fraud_rate, 2),
    "%"
)

# --------------------------------------------------
# 5. Model predictions
# --------------------------------------------------

y_true = evaluation["Fraud_Label"]

y_pred = evaluation["Anomaly_Flag"]

y_score = evaluation["Anomaly_Score"]

# --------------------------------------------------
# 6. Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

# --------------------------------------------------
# 7. Classification metrics
# --------------------------------------------------

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

print("\nClassification Metrics:")
print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

# --------------------------------------------------
# 8. Ranking metrics
# --------------------------------------------------

pr_auc = average_precision_score(
    y_true,
    y_score
)

roc_auc = roc_auc_score(
    y_true,
    y_score
)

print("\nRanking Metrics:")

print(
    f"PR-AUC    : {pr_auc:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)

# --------------------------------------------------
# 9. Full report
# --------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "Not Fraud",
            "Potential Fraud"
        ],
        zero_division=0
    )
)

# --------------------------------------------------
# 10. Top suspicious providers
# --------------------------------------------------

print("\nTop 20 suspicious providers:")

top = evaluation.sort_values(
    "Anomaly_Score",
    ascending=False
).head(20)

print(
    top[
        [
            "Provider",
            "Anomaly_Score",
            "Anomaly_Flag",
            "PotentialFraud"
        ]
    ].to_string(index=False)
)

# --------------------------------------------------
# 11. Save evaluation data
# --------------------------------------------------

output_file = Path(
    "processed/evaluation_results.csv"
)

evaluation.to_csv(
    output_file,
    index=False
)

print("\nSaved evaluation results:")
print(output_file)

print("\nEvaluation completed.")