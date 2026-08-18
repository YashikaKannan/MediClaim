import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FEATURE_FILE = (
    BASE_DIR
    / "processed"
    / "model_features_final.csv"
)

LABEL_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "Train-1542865627584.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "processed"
    / "validation_results.csv"
)

MODEL_FILE = (
    BASE_DIR
    / "processed"
    / "validation_isolation_forest.joblib"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Fixed threshold from our previous threshold analysis
THRESHOLD = 0.54


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURES = [
    "IP_Claim_Count",
    "OP_Claim_Count",
    "IP_Total_Reimbursement",
    "OP_Total_Reimbursement",
    "IP_Avg_Reimbursement",
    "OP_Avg_Reimbursement",
    "Total_Unique_Beneficiaries",
    "Claims_Per_Beneficiary",
    "Reimbursement_Per_Claim",
    "IP_Claim_Share"
]


# ============================================================
# START
# ============================================================

print("=" * 70)
print("CLEAN TRAIN / VALIDATION ISOLATION FOREST EXPERIMENT")
print("=" * 70)


# ============================================================
# LOAD FEATURES
# ============================================================

print("\nLoading provider model features...")

features = pd.read_csv(FEATURE_FILE)

print(
    f"Feature data shape: {features.shape}"
)


# ============================================================
# LOAD LABELS
# ============================================================

print("\nLoading training labels...")

labels = pd.read_csv(LABEL_FILE)

print(
    f"Label data shape: {labels.shape}"
)

print("\nLabel columns:")
print(labels.columns.tolist())


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

if "Provider" not in features.columns:
    raise ValueError(
        "Provider column missing from model features."
    )

if "Provider" not in labels.columns:
    raise ValueError(
        "Provider column missing from label file."
    )

if "PotentialFraud" not in labels.columns:
    raise ValueError(
        "PotentialFraud column missing from label file."
    )


# ============================================================
# CHECK FEATURE COLUMNS
# ============================================================

missing_features = [
    col
    for col in FEATURES
    if col not in features.columns
]

if missing_features:
    raise ValueError(
        f"Missing model features: {missing_features}"
    )


print("\nAll required columns are present.")


# ============================================================
# CHECK DUPLICATES
# ============================================================

print("\nChecking Provider IDs...")

feature_duplicates = (
    features["Provider"].duplicated().sum()
)

label_duplicates = (
    labels["Provider"].duplicated().sum()
)

print(
    f"Duplicate providers in features: "
    f"{feature_duplicates}"
)

print(
    f"Duplicate providers in labels: "
    f"{label_duplicates}"
)


if feature_duplicates > 0:
    raise ValueError(
        "Duplicate Provider IDs found in feature data."
    )

if label_duplicates > 0:
    raise ValueError(
        "Duplicate Provider IDs found in label data."
    )


# ============================================================
# MERGE FEATURES + LABELS
# ============================================================

print("\nMerging provider features with fraud labels...")

df = features.merge(
    labels[
        [
            "Provider",
            "PotentialFraud"
        ]
    ],
    on="Provider",
    how="inner"
)


print(
    f"Merged dataset shape: {df.shape}"
)


# ============================================================
# VERIFY MERGE
# ============================================================

if len(df) != len(features):

    print(
        "\nWARNING:"
    )

    print(
        f"Feature providers: {len(features)}"
    )

    print(
        f"Merged providers: {len(df)}"
    )

    missing_providers = set(
        features["Provider"]
    ) - set(
        labels["Provider"]
    )

    print(
        f"Providers without labels: "
        f"{len(missing_providers)}"
    )

    if len(missing_providers) > 0:
        print(
            "Example missing providers:"
        )
        print(
            list(missing_providers)[:10]
        )

        raise ValueError(
            "Some model providers do not have labels."
        )


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\nActual fraud label distribution:")

print(
    df["PotentialFraud"].value_counts()
)


print("\nActual fraud percentage:")

print(
    df["PotentialFraud"]
    .value_counts(normalize=True)
    * 100
)


# ============================================================
# CONVERT LABELS
# ============================================================

df["Actual"] = (
    df["PotentialFraud"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map(
        {
            "yes": 1,
            "no": 0,
            "1": 1,
            "0": 0,
            "true": 1,
            "false": 0
        }
    )
)


if df["Actual"].isna().sum() > 0:

    print(
        "\nUnknown PotentialFraud values:"
    )

    print(
        df.loc[
            df["Actual"].isna(),
            "PotentialFraud"
        ].unique()
    )

    raise ValueError(
        "Unable to convert PotentialFraud labels."
    )


# ============================================================
# PREPARE X AND Y
# ============================================================

X = df[FEATURES].copy()

y = df["Actual"].copy()


# ============================================================
# DATA QUALITY
# ============================================================

print("\nChecking data quality...")

missing_values = X.isna().sum().sum()

infinite_values = np.isinf(
    X.to_numpy()
).sum()

negative_values = (
    X < 0
).sum().sum()


print(
    f"Missing values  : {missing_values}"
)

print(
    f"Infinite values : {infinite_values}"
)

print(
    f"Negative values : {negative_values}"
)


if missing_values > 0:
    raise ValueError(
        "Missing values found."
    )

if infinite_values > 0:
    raise ValueError(
        "Infinite values found."
    )

if negative_values > 0:
    raise ValueError(
        "Negative values found."
    )


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

print("\nCreating train / validation split...")

(
    X_train,
    X_valid,
    y_train,
    y_valid,
    id_train,
    id_valid
) = train_test_split(
    X,
    y,
    df["Provider"],
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


print("\nSplit complete:")

print(
    f"Training providers   : {len(X_train)}"
)

print(
    f"Validation providers : {len(X_valid)}"
)

print(
    f"Training fraud       : {y_train.sum()}"
)

print(
    f"Training normal      : {(y_train == 0).sum()}"
)

print(
    f"Validation fraud     : {y_valid.sum()}"
)

print(
    f"Validation normal    : {(y_valid == 0).sum()}"
)


# ============================================================
# TRAIN ISOLATION FOREST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING ISOLATION FOREST")
print("=" * 70)

model = IsolationForest(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1
)


print("\nTraining model...")

model.fit(X_train)

print("Training completed.")


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)

print(
    f"\nValidation model saved:\n{MODEL_FILE}"
)


# ============================================================
# SCORE VALIDATION DATA
# ============================================================

print("\nScoring validation providers...")

decision_values = model.decision_function(
    X_valid
)

anomaly_scores = (
    0.5 - decision_values
)


# ============================================================
# CREATE RESULTS
# ============================================================

results = pd.DataFrame(
    {
        "Provider": id_valid.values,
        "Anomaly_Score": anomaly_scores,
        "Actual": y_valid.values
    }
)


# ============================================================
# APPLY FIXED THRESHOLD
# ============================================================

results["Predicted"] = (
    results["Anomaly_Score"]
    >= THRESHOLD
).astype(int)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)


cm = confusion_matrix(
    results["Actual"],
    results["Predicted"]
)


print("\nConfusion Matrix:")
print(cm)


tn, fp, fn, tp = cm.ravel()


print("\nBreakdown:")
print(
    f"True Negatives  : {tn}"
)

print(
    f"False Positives : {fp}"
)

print(
    f"False Negatives : {fn}"
)

print(
    f"True Positives   : {tp}"
)


# ============================================================
# METRICS
# ============================================================

precision = precision_score(
    results["Actual"],
    results["Predicted"],
    zero_division=0
)

recall = recall_score(
    results["Actual"],
    results["Predicted"],
    zero_division=0
)

f1 = f1_score(
    results["Actual"],
    results["Predicted"],
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


# ============================================================
# RANKING METRICS
# ============================================================

roc_auc = roc_auc_score(
    results["Actual"],
    results["Anomaly_Score"]
)

pr_auc = average_precision_score(
    results["Actual"],
    results["Anomaly_Score"]
)


print("\nRanking Metrics:")

print(
    f"ROC-AUC : {roc_auc:.4f}"
)

print(
    f"PR-AUC  : {pr_auc:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        results["Actual"],
        results["Predicted"],
        target_names=[
            "Not Fraud",
            "Potential Fraud"
        ],
        zero_division=0
    )
)


# ============================================================
# FRAUD / ALERT DISTRIBUTION
# ============================================================

print("\nValidation actual fraud percentage:")

print(
    f"{results['Actual'].mean() * 100:.2f}%"
)


print("\nValidation predicted suspicious percentage:")

print(
    f"{results['Predicted'].mean() * 100:.2f}%"
)


# ============================================================
# TOP SUSPICIOUS PROVIDERS
# ============================================================

print("\nTop 20 suspicious validation providers:")

top_20 = (
    results
    .sort_values(
        "Anomaly_Score",
        ascending=False
    )
    .head(20)
)


print(
    top_20.to_string(index=False)
)


# ============================================================
# SAVE RESULTS
# ============================================================

results.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 70)
print("VALIDATION EXPERIMENT COMPLETE")
print("=" * 70)


print(
    f"\nSaved validation results:\n{OUTPUT_FILE}"
)

print(
    f"\nValidation rows: {len(results)}"
)

print("\nDONE")

print("=" * 70)