"""
Model training and hyperparameter tuning
for One-Class SVM healthcare provider anomaly detection.
"""

import pandas as pd
import numpy as np

from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from config import (
    PROVIDER_ID_COL,
    FRAUD_LABEL_COL,
    RANDOM_STATE,
    NU_VALUES,
    GAMMA_VALUES,
    KERNEL,
    KERNEL_VALUES,
    TRAIN_SPLIT_RATIO,
    MIN_RECALL,
    get_logger
)

from utils import (
    verify_dataframe,
    check_no_data_leakage,
    create_model_metadata
)

from feature_engineering import (
    verify_features,
    prepare_features_for_training
)

from preprocessing import (
    scale_features,
    create_scaler
)


logger = get_logger(__name__)


# ============================================================================
# TRAIN / VALIDATION SPLIT
# ============================================================================

def train_test_split_providers(
    provider_features,
    test_size=0.2,
    random_state=RANDOM_STATE
):
    """
    Split providers into train and validation sets at provider level.

    No provider should appear in both splits.
    """

    logger.info("=" * 80)
    logger.info("TRAIN/VALIDATION SPLIT (PROVIDER LEVEL)")
    logger.info("=" * 80)

    if FRAUD_LABEL_COL not in provider_features.columns:
        raise ValueError(
            f"Missing {FRAUD_LABEL_COL} column for splitting"
        )

    providers = provider_features[PROVIDER_ID_COL].values
    labels = provider_features[FRAUD_LABEL_COL].values

    train_indices, val_indices = train_test_split(
        np.arange(len(provider_features)),
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    train_providers_list = providers[train_indices]
    val_providers_list = providers[val_indices]

    overlap = set(train_providers_list) & set(val_providers_list)

    if overlap:
        raise ValueError(
            f"Provider overlap in train/validation split: "
            f"{len(overlap)} providers"
        )

    logger.info(
        f"Total providers: {len(providers)}"
    )

    logger.info(
        f"Training providers: {len(train_providers_list)}"
    )

    logger.info(
        f"Validation providers: {len(val_providers_list)}"
    )

    df_train = (
        provider_features
        .iloc[train_indices]
        .reset_index(drop=True)
    )

    df_val = (
        provider_features
        .iloc[val_indices]
        .reset_index(drop=True)
    )

    train_fraud_dist = df_train[FRAUD_LABEL_COL].value_counts()
    val_fraud_dist = df_val[FRAUD_LABEL_COL].value_counts()

    logger.info(
        f"\nTraining set fraud distribution:"
    )
    logger.info(
        f"  {train_fraud_dist.to_dict()}"
    )

    logger.info(
        f"\nValidation set fraud distribution:"
    )
    logger.info(
        f"  {val_fraud_dist.to_dict()}"
    )

    return (
        df_train,
        df_val,
        train_providers_list,
        val_providers_list
    )


# ============================================================================
# TRAINING DATA PREPARATION
# ============================================================================

def prepare_training_data(
    df_train,
    feature_cols,
    scaler=None,
    fit_scaler=True
):
    """
    Prepare training features, handle invalid values and scale.
    """

    X_train = df_train[feature_cols].copy()

    nan_count = X_train.isnull().sum().sum()

    if nan_count > 0:
        logger.warning(
            f"Filling {nan_count} NaN values in training data"
        )
        X_train = X_train.fillna(0)

    inf_count = np.isinf(X_train).sum().sum()

    if inf_count > 0:
        logger.warning(
            f"Replacing {inf_count} infinite values "
            f"in training data"
        )

        X_train = X_train.replace(
            [np.inf, -np.inf],
            0
        )

    if scaler is None:

        X_train_scaled, scaler = scale_features(
            X_train,
            scaler=None,
            fit=fit_scaler
        )

    else:

        X_train_scaled, _ = scale_features(
            X_train,
            scaler=scaler,
            fit=False
        )

    verify_features(X_train_scaled)

    return X_train_scaled, scaler


# ============================================================================
# VALIDATION DATA PREPARATION
# ============================================================================

def prepare_validation_data(
    df_val,
    feature_cols,
    scaler,
    feature_order=None
):
    """
    Prepare validation features using the training scaler.
    """

    if feature_order is not None:

        if feature_cols != feature_order:

            raise ValueError(
                "Feature column mismatch between "
                "training and validation"
            )

    X_val = df_val[feature_cols].copy()

    nan_count = X_val.isnull().sum().sum()

    if nan_count > 0:

        logger.warning(
            f"Filling {nan_count} NaN values "
            f"in validation data"
        )

        X_val = X_val.fillna(0)

    inf_count = np.isinf(X_val).sum().sum()

    if inf_count > 0:

        logger.warning(
            f"Replacing {inf_count} infinite values "
            f"in validation data"
        )

        X_val = X_val.replace(
            [np.inf, -np.inf],
            0
        )

    X_val_scaled, _ = scale_features(
        X_val,
        scaler=scaler,
        fit=False
    )

    verify_features(X_val_scaled)

    return X_val_scaled


# ============================================================================
# NORMAL PROVIDER FILTERING
# ============================================================================

def filter_normal_providers(
    X_train,
    y_train,
    provider_ids_train
):
    """
    Keep only normal providers for One-Class SVM training.
    """

    logger.info("\n" + "=" * 80)
    logger.info(
        "FILTERING TO NORMAL PROVIDERS FOR ONE-CLASS SVM"
    )
    logger.info("=" * 80)

    normal_mask = (
        y_train == "No"
    )

    X_normal = X_train[normal_mask].copy()

    normal_providers = provider_ids_train[
        normal_mask
    ]

    fraud_excluded = (
        (~normal_mask).sum()
    )

    logger.info(
        f"Total training providers: {len(X_train)}"
    )

    logger.info(
        f"Normal providers for OC-SVM: "
        f"{len(X_normal)}"
    )

    logger.info(
        f"Fraud providers excluded: "
        f"{fraud_excluded}"
    )

    if len(X_normal) == 0:

        raise ValueError(
            "No normal providers in training set!"
        )

    return (
        X_normal,
        normal_providers,
        fraud_excluded
    )


# ============================================================================
# TRAIN ONE-CLASS SVM
# ============================================================================

def train_one_class_svm(
    X_normal,
    nu,
    gamma,
    kernel=KERNEL
):
    """
    Train One-Class SVM only on normal providers.
    """

    logger.info(
        f"Training One-Class SVM: "
        f"nu={nu}, gamma={gamma}, kernel={kernel}"
    )

    model = OneClassSVM(
        kernel=kernel,
        nu=nu,
        gamma=gamma
    )

    model.fit(X_normal)

    logger.info(
        "Model trained successfully"
    )

    logger.info(
        f"  Support vectors: "
        f"{len(model.support_vectors_)}"
    )

    logger.info(
        f"  Features: "
        f"{model.n_features_in_}"
    )

    return model


# ============================================================================
# PREDICTION
# ============================================================================

def predict_provider_anomalies(
    model,
    X
):
    """
    Generate One-Class SVM predictions and decision scores.
    """

    predictions = model.predict(X)

    anomaly_scores = model.decision_function(X)

    return predictions, anomaly_scores


# ============================================================================
# CONVERT PREDICTIONS
# ============================================================================

def convert_predictions(
    predictions,
    anomaly_scores
):
    """
    Convert:
        +1 = normal -> 0
        -1 = anomaly -> 1
    """

    binary_preds = (
        predictions == -1
    ).astype(int)

    return (
        binary_preds,
        anomaly_scores
    )


# ============================================================================
# RISK SCORE
# ============================================================================

def calculate_risk_score(
    anomaly_scores,
    method="minmax"
):
    """
    Convert anomaly scores into 0-100 risk scores.
    """

    if method == "minmax":

        min_score = anomaly_scores.min()
        max_score = anomaly_scores.max()

        if max_score == min_score:

            risk_scores = np.full_like(
                anomaly_scores,
                50.0,
                dtype=float
            )

        else:

            risk_scores = 100 * (
                1 -
                (
                    anomaly_scores - min_score
                )
                /
                (
                    max_score - min_score
                )
            )

    elif method == "sigmoid":

        risk_scores = (
            100 /
            (
                1 +
                np.exp(anomaly_scores)
            )
        )

    else:

        raise ValueError(
            f"Unknown risk score method: {method}"
        )

    risk_scores = np.clip(
        risk_scores,
        0,
        100
    )

    return risk_scores


# ============================================================================
# RISK LEVEL
# ============================================================================

def assign_risk_levels(
    risk_scores,
    thresholds=None
):
    """
    Assign Low / Medium / High / Critical risk levels.
    """

    if thresholds is None:

        thresholds = {
            "Low": (0, 25),
            "Medium": (26, 50),
            "High": (51, 75),
            "Critical": (76, 100)
        }

    risk_levels = []

    for score in risk_scores:

        level = "Unknown"

        for level_name, (
            min_val,
            max_val
        ) in thresholds.items():

            if (
                min_val <= score <= max_val
            ):

                level = level_name
                break

        risk_levels.append(level)

    return risk_levels


# ============================================================================
# METRIC CALCULATION
# ============================================================================

def _calculate_metrics(
    y_true,
    predictions,
    scores
):
    """
    Calculate all important validation metrics.
    """

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    try:

        negated_scores = -scores

        roc_auc = roc_auc_score(
            y_true,
            negated_scores
        )

        pr_auc = average_precision_score(
            y_true,
            negated_scores
        )

    except Exception:

        roc_auc = 0.0
        pr_auc = 0.0

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "specificity": specificity,
        "sensitivity": recall
    }


# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================

def tune_hyperparameters(
    X_train_normal,
    X_val,
    y_val_true,
    nu_values=NU_VALUES,
    gamma_values=GAMMA_VALUES
):
    """
    Tune One-Class SVM.

    Selection philosophy:

    1. Prefer models with recall >= MIN_RECALL.
    2. Among acceptable models, maximize F1.
    3. Use precision as a secondary consideration.
    4. Keep ROC-AUC and PR-AUC available for comparison.

    This prevents a model from achieving a higher F1/precision by
    completely destroying recall.
    """

    logger.info("\n" + "=" * 80)
    logger.info(
        "ONE-CLASS SVM HYPERPARAMETER TUNING"
    )
    logger.info("=" * 80)

    logger.info(
        f"nu values: {nu_values}"
    )

    logger.info(
        f"gamma values: {gamma_values}"
    )

    logger.info(
        f"kernel values: {KERNEL_VALUES}"
    )

    logger.info(
        f"Minimum required recall: {MIN_RECALL:.2%}"
    )

    results = []

    for kernel in KERNEL_VALUES:

        for nu in nu_values:

            for gamma in gamma_values:

                logger.info(
                    "\nTesting: "
                    f"kernel={kernel}, "
                    f"nu={nu}, "
                    f"gamma={gamma}"
                )

                try:

                    model = train_one_class_svm(
                        X_train_normal,
                        nu=nu,
                        gamma=gamma,
                        kernel=kernel
                    )

                    preds, scores = (
                        predict_provider_anomalies(
                            model,
                            X_val
                        )
                    )

                    binary_preds, _ = (
                        convert_predictions(
                            preds,
                            scores
                        )
                    )

                    metrics = _calculate_metrics(
                        y_val_true,
                        binary_preds,
                        scores
                    )

                    recall_ok = (
                        metrics["recall"]
                        >= MIN_RECALL
                    )

                    logger.info(
                        f"  Accuracy:  "
                        f"{metrics['accuracy']:.4f}"
                    )

                    logger.info(
                        f"  Precision: "
                        f"{metrics['precision']:.4f}"
                    )

                    logger.info(
                        f"  Recall:    "
                        f"{metrics['recall']:.4f}"
                    )

                    logger.info(
                        f"  F1-Score:  "
                        f"{metrics['f1_score']:.4f}"
                    )

                    logger.info(
                        f"  ROC-AUC:   "
                        f"{metrics['roc_auc']:.4f}"
                    )

                    logger.info(
                        f"  PR-AUC:    "
                        f"{metrics['pr_auc']:.4f}"
                    )

                    logger.info(
                        f"  Recall constraint: "
                        f"{'PASS' if recall_ok else 'FAIL'}"
                    )

                    results.append({
                        "kernel": kernel,
                        "nu": nu,
                        "gamma": gamma,

                        "accuracy":
                            metrics["accuracy"],

                        "precision":
                            metrics["precision"],

                        "recall":
                            metrics["recall"],

                        "f1_score":
                            metrics["f1_score"],

                        "roc_auc":
                            metrics["roc_auc"],

                        "pr_auc":
                            metrics["pr_auc"],

                        "tn":
                            metrics["tn"],

                        "fp":
                            metrics["fp"],

                        "fn":
                            metrics["fn"],

                        "tp":
                            metrics["tp"],

                        "false_positive_rate":
                            metrics[
                                "false_positive_rate"
                            ],

                        "false_negative_rate":
                            metrics[
                                "false_negative_rate"
                            ],

                        "specificity":
                            metrics[
                                "specificity"
                            ],

                        "sensitivity":
                            metrics[
                                "sensitivity"
                            ],

                        "recall_constraint":
                            recall_ok
                    })

                except Exception as e:

                    logger.warning(
                        f"  Combination failed: {e}"
                    )

                    continue

    results_df = pd.DataFrame(results)

    if results_df.empty:

        raise ValueError(
            "Hyperparameter tuning produced "
            "no valid results."
        )

    # ------------------------------------------------------------------------
    # Model ranking
    # ------------------------------------------------------------------------

    acceptable = results_df[
        results_df["recall_constraint"] == True
    ].copy()

    if acceptable.empty:

        logger.warning(
            "\nNo model satisfied the minimum "
            f"recall constraint of {MIN_RECALL:.2%}."
        )

        logger.warning(
            "Selecting the model with the highest F1 "
            "without the recall constraint."
        )

        ranked = results_df.sort_values(
            by=[
                "f1_score",
                "precision",
                "recall"
            ],
            ascending=False
        )

    else:

        ranked = acceptable.sort_values(
            by=[
                "f1_score",
                "precision",
                "recall",
                "pr_auc"
            ],
            ascending=False
        )

    logger.info("\n" + "=" * 80)
    logger.info(
        "TOP HYPERPARAMETER COMBINATIONS"
    )
    logger.info("=" * 80)

    logger.info(
        "\n" +
        ranked[
            [
                "kernel",
                "nu",
                "gamma",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
                "pr_auc",
                "fp",
                "fn"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    return results_df


# ============================================================================
# SELECT BEST MODEL
# ============================================================================

def select_best_model(
    results_df,
    metric="f1_score"
):
    """
    Select the best model.

    Priority:
        1. Recall >= MIN_RECALL
        2. Highest F1
        3. Highest precision
        4. Highest recall
        5. Highest PR-AUC
    """

    if results_df.empty:

        raise ValueError(
            "Cannot select model from empty results."
        )

    acceptable = results_df[
        results_df["recall_constraint"] == True
    ].copy()

    if acceptable.empty:

        logger.warning(
            "No model passed the recall constraint."
        )

        candidate_results = results_df.copy()

    else:

        candidate_results = acceptable.copy()

    ranked = candidate_results.sort_values(
        by=[
            "f1_score",
            "precision",
            "recall",
            "pr_auc"
        ],
        ascending=False
    )

    best = ranked.iloc[0]

    best_params = {
        "nu": float(best["nu"]),
        "gamma": best["gamma"],
        "kernel": best["kernel"]
    }

    logger.info("\n" + "=" * 80)
    logger.info("SELECTED BEST MODEL")
    logger.info("=" * 80)

    logger.info(
        f"Kernel:    {best_params['kernel']}"
    )

    logger.info(
        f"nu:        {best_params['nu']}"
    )

    logger.info(
        f"gamma:     {best_params['gamma']}"
    )

    logger.info(
        f"Accuracy:  {best['accuracy']:.4f}"
    )

    logger.info(
        f"Precision: {best['precision']:.4f}"
    )

    logger.info(
        f"Recall:    {best['recall']:.4f}"
    )

    logger.info(
        f"F1-Score:  {best['f1_score']:.4f}"
    )

    logger.info(
        f"ROC-AUC:   {best['roc_auc']:.4f}"
    )

    logger.info(
        f"PR-AUC:    {best['pr_auc']:.4f}"
    )

    logger.info(
        f"FP:        {int(best['fp'])}"
    )

    logger.info(
        f"FN:        {int(best['fn'])}"
    )

    return best_params