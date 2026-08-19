import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SCORES_FILE = (
    BASE_DIR
    / "processed"
    / "test_anomaly_scores.csv"
)

LABEL_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "Test-1542969243754.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "processed"
    / "test_evaluation_results.csv"
)

THRESHOLD = 0.54


# ============================================================
# START
# ============================================================

print("=" * 70)
print("SEPARATE TEST DATA EVALUATION")
print("=" * 70)


# ============================================================
# LOAD MODEL SCORES
# ============================================================

print("\nLoading test anomaly scores...")

scores = pd.read_csv(SCORES_FILE)

print(f"Scores shape: {scores.shape}")

print("\nScore columns:")
print(scores.columns.tolist())


# ============================================================
# LOAD TEST LABELS
# ============================================================

print("\nLoading test labels...")

labels = pd.read_csv(LABEL_FILE)

print(f"Labels shape: {labels.shape}")

print("\nLabel columns:")
print(labels.columns.tolist())


# ============================================================
# FIND PROVIDER COLUMN
# ============================================================

provider_candidates = [
    "Provider",
    "provider",
    "Provider_ID",
    "ProviderID",
    "NPI",
    "npi"
]

score_provider_col = None
label_provider_col = None

for col in provider_candidates:
    if col in scores.columns:
        score_provider_col = col
        break

for col in provider_candidates:
    if col in labels.columns:
        label_provider_col = col
        break


if score_provider_col is None:
    raise ValueError(
        "Provider column not found in score file."
    )

if label_provider_col is None:
    raise ValueError(
        "Provider column not found in label file."
    )


print(
    f"\nScore provider column: {score_provider_col}"
)

print(
    f"Label provider column: {label_provider_col}"
)


# ============================================================
# FIND FRAUD LABEL COLUMN
# ============================================================

label_candidates = [
    "PotentialFraud",
    "potentialfraud",
    "Potential_Fraud",
    "Fraud",
    "fraud",
    "Label",
    "label"
]

fraud_col = None

for col in label_candidates:
    if col in labels.columns:
        fraud_col = col
        break


if fraud_col is None:
    raise ValueError(
        "Fraud label column not found.\n"
        f"Available columns: {labels.columns.tolist()}"
    )


print(
    f"Fraud label column: {fraud_col}"
)


# ============================================================
# PREPARE LABEL DATA
# ============================================================

labels = labels[
    [label_provider_col, fraud_col]
].copy()

labels = labels.rename(
    columns={
        label_provider_col: "Provider",
        fraud_col: "PotentialFraud"
    }
)


# ============================================================
# STANDARDIZE PROVIDER IDs
# ============================================================

scores["Provider"] = (
    scores[score_provider_col]
    .astype(str)
    .str.strip()
)

labels["Provider"] = (
    labels["Provider"]
    .astype(str)
    .str.strip()
)


# ============================================================
# CHECK DUPLICATES
# ============================================================

print("\nChecking duplicate providers...")

score_duplicates = scores["Provider"].duplicated().sum()
label_duplicates = labels["Provider"].duplicated().sum()

print(
    f"Duplicate providers in scores: {score_duplicates}"
)

print(
    f"Duplicate providers in labels: {label_duplicates}"
)

if score_duplicates > 0:
    raise ValueError(
        "Duplicate providers found in score data."
    )

if label_duplicates > 0:
    raise ValueError(
        "Duplicate providers found in label data."
    )


# ============================================================
# MERGE
# ============================================================

print("\nMerging test scores with actual labels...")

evaluation = scores[
    ["Provider", "Anomaly_Score", "Anomaly_Flag"]
].merge(
    labels,
    on="Provider",
    how="inner"
)

print(
    f"Merged evaluation shape: {evaluation.shape}"
)


# ============================================================
# CHECK MERGE COVERAGE
# ============================================================

print("\nMerge coverage:")

print(
    f"Test score providers: {len(scores)}"
)

print(
    f"Test label providers: {len(labels)}"
)

print(
    f"Matched providers: {len(evaluation)}"
)

if len(evaluation) != len(scores):
    print(
        "\nWARNING:"
        " Not every scored test provider has a label."
    )


# ============================================================
# CONVERT LABELS
# ============================================================

evaluation["Actual"] = (
    evaluation["PotentialFraud"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0,
        "true": 1,
        "false": 0
    })
)


if evaluation["Actual"].isna().sum() > 0:

    print("\nUnrecognized label values:")

    print(
        evaluation.loc[
            evaluation["Actual"].isna(),
            "PotentialFraud"
        ].value_counts()
    )

    raise ValueError(
        "Some fraud labels could not be converted."
    )


# ============================================================
# PREDICTIONS
# ============================================================

evaluation["Predicted"] = (
    evaluation["Anomaly_Score"] >= THRESHOLD
).astype(int)


# ============================================================
# ACTUAL LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("ACTUAL TEST LABEL DISTRIBUTION")
print("=" * 70)

actual_counts = (
    evaluation["Actual"]
    .value_counts()
    .sort_index()
)

print(
    f"Not Fraud : {actual_counts.get(0, 0)}"
)

print(
    f"Potential Fraud : {actual_counts.get(1, 0)}"
)

fraud_percentage = (
    evaluation["Actual"].mean() * 100
)

print(
    f"\nActual fraud percentage: "
    f"{fraud_percentage:.2f}%"
)


# ============================================================
# PREDICTED DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("MODEL PREDICTION DISTRIBUTION")
print("=" * 70)

predicted_counts = (
    evaluation["Predicted"]
    .value_counts()
    .sort_index()
)

print(
    f"Predicted Normal : "
    f"{predicted_counts.get(0, 0)}"
)

print(
    f"Predicted Suspicious : "
    f"{predicted_counts.get(1, 0)}"
)

prediction_percentage = (
    evaluation["Predicted"].mean() * 100
)

print(
    f"\nPredicted suspicious percentage: "
    f"{prediction_percentage:.2f}%"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    evaluation["Actual"],
    evaluation["Predicted"]
)

print(cm)

tn, fp, fn, tp = cm.ravel()

print("\nBreakdown:")
print(f"True Negatives  : {tn}")
print(f"False Positives : {fp}")
print(f"False Negatives : {fn}")
print(f"True Positives  : {tp}")


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

precision = precision_score(
    evaluation["Actual"],
    evaluation["Predicted"],
    zero_division=0
)

recall = recall_score(
    evaluation["Actual"],
    evaluation["Predicted"],
    zero_division=0
)

f1 = f1_score(
    evaluation["Actual"],
    evaluation["Predicted"],
    zero_division=0
)


print("\n" + "=" * 70)
print("TEST CLASSIFICATION METRICS")
print("=" * 70)

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# ============================================================
# RANKING METRICS
# ============================================================

roc_auc = roc_auc_score(
    evaluation["Actual"],
    evaluation["Anomaly_Score"]
)

pr_auc = average_precision_score(
    evaluation["Actual"],
    evaluation["Anomaly_Score"]
)


print("\n" + "=" * 70)
print("TEST RANKING METRICS")
print("=" * 70)

print(f"ROC-AUC : {roc_auc:.4f}")
print(f"PR-AUC  : {pr_auc:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("TEST CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        evaluation["Actual"],
        evaluation["Predicted"],
        target_names=[
            "Not Fraud",
            "Potential Fraud"
        ],
        zero_division=0
    )
)


# ============================================================
# TOP SUSPICIOUS TEST PROVIDERS
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 SUSPICIOUS TEST PROVIDERS")
print("=" * 70)

top_20 = (
    evaluation
    .sort_values(
        "Anomaly_Score",
        ascending=False
    )
    .head(20)
)

print(
    top_20[
        [
            "Provider",
            "Anomaly_Score",
            "Anomaly_Flag",
            "PotentialFraud"
        ]
    ].to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

evaluation.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("TEST EVALUATION COMPLETE")
print("=" * 70)

print("\nSaved:")
print(OUTPUT_FILE)

print(
    f"\nFinal evaluation rows: "
    f"{len(evaluation)}"
)

print("\nDONE")
print("=" * 70)