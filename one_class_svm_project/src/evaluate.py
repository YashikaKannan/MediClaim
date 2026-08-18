"""
Model evaluation and metrics calculation
"""
import pandas as pd
import numpy as np
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from config import get_logger, OUTPUTS_DIR, PROVIDER_ID_COL, FRAUD_LABEL_COL

logger = get_logger(__name__)

def evaluate_model(y_true, y_pred, y_scores=None, model_name="Model"):
    """
    Comprehensive model evaluation
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels (1=fraud/anomaly, 0=normal)
    y_pred : np.ndarray
        Predicted labels (1=fraud/anomaly, 0=normal)
    y_scores : np.ndarray, optional
        Continuous scores for AUC calculations
    model_name : str
        Name for logging
    
    Returns:
    --------
    dict
        Dictionary of all calculated metrics
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"EVALUATION: {model_name}")
    logger.info(f"{'='*80}")
    
    metrics = {}
    
    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['tn'] = int(tn)
    metrics['fp'] = int(fp)
    metrics['fn'] = int(fn)
    metrics['tp'] = int(tp)
    
    # Derived metrics
    metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # AUC metrics (if scores provided)
    if y_scores is not None:
        try:
            # Negate scores so anomalies (lower scores) become higher values
            negated_scores = -y_scores
            metrics['roc_auc'] = roc_auc_score(y_true, negated_scores)
            metrics['pr_auc'] = average_precision_score(y_true, negated_scores)
        except Exception as e:
            logger.warning(f"Could not calculate AUC metrics: {e}")
            metrics['roc_auc'] = None
            metrics['pr_auc'] = None
    else:
        metrics['roc_auc'] = None
        metrics['pr_auc'] = None
    
    # Log metrics
    logger.info(f"\nClassification Metrics:")
    logger.info(f"  Accuracy:     {metrics['accuracy']:.4f}")
    logger.info(f"  Precision:    {metrics['precision']:.4f}")
    logger.info(f"  Recall:       {metrics['recall']:.4f}")
    logger.info(f"  F1-Score:     {metrics['f1_score']:.4f}")
    
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  TN: {metrics['tn']}, FP: {metrics['fp']}")
    logger.info(f"  FN: {metrics['fn']}, TP: {metrics['tp']}")
    
    logger.info(f"\nAdditional Metrics:")
    logger.info(f"  Sensitivity (Recall):  {metrics['sensitivity']:.4f}")
    logger.info(f"  Specificity:           {metrics['specificity']:.4f}")
    logger.info(f"  False Positive Rate:   {metrics['false_positive_rate']:.4f}")
    logger.info(f"  False Negative Rate:   {metrics['false_negative_rate']:.4f}")
    
    if metrics['roc_auc'] is not None:
        logger.info(f"  ROC-AUC:               {metrics['roc_auc']:.4f}")
    if metrics['pr_auc'] is not None:
        logger.info(f"  PR-AUC:                {metrics['pr_auc']:.4f}")
    
    return metrics

def create_prediction_dataframe(provider_ids, y_true, y_pred, risk_scores, 
                                risk_levels, anomaly_scores=None):
    """
    Create predictions dataframe
    
    Parameters:
    -----------
    provider_ids : np.ndarray
        Provider identifiers
    y_true : np.ndarray
        True labels (1=fraud, 0=normal)
    y_pred : np.ndarray
        Predicted labels (1=anomaly, 0=normal)
    risk_scores : np.ndarray
        Risk scores (0-100)
    risk_levels : list
        Risk level assignments
    anomaly_scores : np.ndarray, optional
        Raw anomaly scores from model
    
    Returns:
    --------
    pd.DataFrame
        Predictions dataframe
    """
    predictions_df = pd.DataFrame({
        PROVIDER_ID_COL: provider_ids,
        'ActualFraud': y_true,
        'PredictedAnomaly': y_pred,
        'RiskScore': risk_scores,
        'RiskLevel': risk_levels,
    })
    
    if anomaly_scores is not None:
        predictions_df['DecisionScore'] = anomaly_scores
    
    return predictions_df

def save_metrics_json(metrics, filepath):
    """
    Save metrics to JSON file
    
    Parameters:
    -----------
    metrics : dict
        Metrics dictionary
    filepath : str or Path
        Output file path
    """
    # Convert numpy types to native Python types for JSON serialization
    metrics_clean = {}
    for k, v in metrics.items():
        if isinstance(v, np.integer):
            metrics_clean[k] = int(v)
        elif isinstance(v, np.floating):
            metrics_clean[k] = float(v)
        else:
            metrics_clean[k] = v
    
    with open(filepath, 'w') as f:
        json.dump(metrics_clean, f, indent=2)
    
    logger.info(f"Metrics saved to {filepath}")

def save_confusion_matrix_csv(metrics, filepath):
    """
    Save confusion matrix to CSV
    
    Parameters:
    -----------
    metrics : dict
        Metrics dictionary with tn, fp, fn, tp
    filepath : str or Path
        Output file path
    """
    cm_df = pd.DataFrame({
        'Predicted Normal': [metrics['tn'], metrics['fn']],
        'Predicted Anomaly': [metrics['fp'], metrics['tp']]
    }, index=['Actual Normal', 'Actual Anomaly'])
    
    cm_df.to_csv(filepath)
    logger.info(f"Confusion matrix saved to {filepath}")

def calculate_roc_curve(y_true, y_scores):
    """
    Calculate ROC curve
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_scores : np.ndarray
        Continuous scores
    
    Returns:
    --------
    tuple
        (fpr, tpr, thresholds)
    """
    # Negate scores for proper ROC curve
    negated_scores = -y_scores
    fpr, tpr, thresholds = roc_curve(y_true, negated_scores)
    
    return fpr, tpr, thresholds

def calculate_pr_curve(y_true, y_scores):
    """
    Calculate Precision-Recall curve
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_scores : np.ndarray
        Continuous scores
    
    Returns:
    --------
    tuple
        (precision, recall, thresholds)
    """
    # Negate scores for proper PR curve
    negated_scores = -y_scores
    precision, recall, thresholds = precision_recall_curve(y_true, negated_scores)
    
    return precision, recall, thresholds

def print_evaluation_summary(metrics):
    """
    Print a formatted evaluation summary
    
    Parameters:
    -----------
    metrics : dict
        Metrics dictionary
    """
    print(f"\n{'='*80}")
    print("FINAL EVALUATION SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"Accuracy:              {metrics['accuracy']:.4f}")
    print(f"Precision:             {metrics['precision']:.4f}")
    print(f"Recall:                {metrics['recall']:.4f}")
    print(f"F1-Score:              {metrics['f1_score']:.4f}")
    
    if metrics['roc_auc'] is not None:
        print(f"ROC-AUC:               {metrics['roc_auc']:.4f}")
    
    if metrics['pr_auc'] is not None:
        print(f"PR-AUC:                {metrics['pr_auc']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:       {metrics['tn']}")
    print(f"  False Positives:      {metrics['fp']}")
    print(f"  False Negatives:      {metrics['fn']}")
    print(f"  True Positives:       {metrics['tp']}")
    
    print(f"\nRates:")
    print(f"  False Positive Rate:  {metrics['false_positive_rate']:.4f}")
    print(f"  False Negative Rate:  {metrics['false_negative_rate']:.4f}")
    print(f"  Specificity:          {metrics['specificity']:.4f}")
    print(f"  Sensitivity:          {metrics['sensitivity']:.4f}")
