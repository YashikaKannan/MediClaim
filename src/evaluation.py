"""
Medicare Provider Anomaly Risk Engine - Evaluation
Model evaluation and performance metrics
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import config
from src.utils import logger, setup_logging, print_section, save_json

setup_logging(config.LOGGING_LEVEL)

class ModelEvaluator:
    """Model evaluation and metrics calculation"""
    
    @staticmethod
    def evaluate(predictions, anomaly_scores, true_labels=None):
        """
        Evaluate model performance
        
        Parameters:
        -----------
        predictions : np.ndarray
            Binary predictions (0=normal, 1=anomaly)
        anomaly_scores : np.ndarray
            Anomaly scores
        true_labels : np.ndarray
            True labels for evaluation (optional)
        
        Returns:
        --------
        dict
            Evaluation metrics
        """
        print_section("Model Evaluation")
        
        metrics = {
            'total_providers': len(predictions),
            'anomalies_detected': int(predictions.sum()),
            'normal_providers': int((predictions == 0).sum()),
            'anomaly_rate': float(predictions.sum() / len(predictions) * 100),
        }
        
        print(f"Total Providers: {metrics['total_providers']}")
        print(f"Anomalies Detected: {metrics['anomalies_detected']}")
        print(f"Normal Providers: {metrics['normal_providers']}")
        print(f"Anomaly Rate: {metrics['anomaly_rate']:.2f}%")
        
        # Evaluate against labels if provided
        if true_labels is not None:
            print_section("Supervised Evaluation Metrics (using PotentialFraud labels)")
            
            # Classification metrics
            metrics['precision'] = float(precision_score(true_labels, predictions, zero_division=0))
            metrics['recall'] = float(recall_score(true_labels, predictions, zero_division=0))
            metrics['f1_score'] = float(f1_score(true_labels, predictions, zero_division=0))
            
            # Confusion matrix
            cm = confusion_matrix(true_labels, predictions)
            metrics['confusion_matrix'] = cm.tolist()
            
            # ROC and PR curves
            try:
                metrics['roc_auc'] = float(roc_auc_score(true_labels, anomaly_scores))
                metrics['pr_auc'] = float(average_precision_score(true_labels, anomaly_scores))
            except:
                logger.warning("Could not calculate ROC-AUC or PR-AUC")
                metrics['roc_auc'] = None
                metrics['pr_auc'] = None
            
            # Fraud detection rate
            fraud_mask = true_labels == 1
            detected_fraud = predictions[fraud_mask].sum()
            total_fraud = fraud_mask.sum()
            
            metrics['fraud_detection_rate'] = float(
                detected_fraud / total_fraud * 100 if total_fraud > 0 else 0
            )
            
            # False positive rate
            normal_mask = true_labels == 0
            false_positives = predictions[normal_mask].sum()
            total_normal = normal_mask.sum()
            
            metrics['false_positive_rate'] = float(
                false_positives / total_normal * 100 if total_normal > 0 else 0
            )
            
            # Print metrics
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-Score: {metrics['f1_score']:.4f}")
            print(f"ROC-AUC: {metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "ROC-AUC: N/A")
            print(f"PR-AUC: {metrics['pr_auc']:.4f}" if metrics['pr_auc'] else "PR-AUC: N/A")
            print(f"\nFraud Detection Rate: {metrics['fraud_detection_rate']:.2f}%")
            print(f"False Positive Rate: {metrics['false_positive_rate']:.2f}%")
            
            print(f"\nConfusion Matrix:")
            print(f"  True Negatives:  {cm[0, 0]}")
            print(f"  False Positives: {cm[0, 1]}")
            print(f"  False Negatives: {cm[1, 0]}")
            print(f"  True Positives:  {cm[1, 1]}")
        
        return metrics
    
    @staticmethod
    def plot_confusion_matrix(true_labels, predictions, output_path):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confusion matrix to {output_path}")
    
    @staticmethod
    def plot_roc_curve(true_labels, anomaly_scores, output_path):
        """Plot and save ROC curve"""
        try:
            fpr, tpr, _ = roc_curve(true_labels, anomaly_scores)
            roc_auc = roc_auc_score(true_labels, anomaly_scores)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved ROC curve to {output_path}")
        except Exception as e:
            logger.error(f"Could not plot ROC curve: {e}")
    
    @staticmethod
    def plot_precision_recall_curve(true_labels, anomaly_scores, output_path):
        """Plot and save precision-recall curve"""
        try:
            precision, recall, _ = precision_recall_curve(true_labels, anomaly_scores)
            pr_auc = average_precision_score(true_labels, anomaly_scores)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='darkorange', lw=2,
                    label=f'PR curve (AUC = {pr_auc:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="lower left")
            plt.grid(alpha=0.3)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved PR curve to {output_path}")
        except Exception as e:
            logger.error(f"Could not plot precision-recall curve: {e}")
    
    @staticmethod
    def plot_anomaly_distribution(anomaly_scores, risk_levels, output_path):
        """Plot and save anomaly score distribution by risk level"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Distribution by risk level
        ax1 = axes[0]
        for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            mask = risk_levels == level
            if mask.sum() > 0:
                ax1.hist(anomaly_scores[mask], alpha=0.5, label=level, bins=30)
        
        ax1.set_xlabel('Anomaly Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Anomaly Score Distribution by Risk Level')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Box plot
        ax2 = axes[1]
        data_by_level = []
        labels = []
        for level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
            mask = risk_levels == level
            if mask.sum() > 0:
                data_by_level.append(anomaly_scores[mask])
                labels.append(level)
        
        ax2.boxplot(data_by_level, labels=labels)
        ax2.set_ylabel('Anomaly Score')
        ax2.set_title('Anomaly Score Distribution (Box Plot)')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved anomaly distribution plot to {output_path}")


def run_evaluation(results_df, true_labels_df=None):
    """
    Run comprehensive evaluation
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Predictions and scores
    true_labels_df : pd.DataFrame
        True labels (optional)
    
    Returns:
    --------
    dict
        Evaluation metrics
    """
    predictions = results_df['anomaly_prediction'].values
    anomaly_scores = results_df['anomaly_score_raw'].values
    
    # Get true labels if available
    true_labels = None
    if true_labels_df is not None:
        # Merge to ensure same provider order
        merged = results_df.merge(true_labels_df, on='Provider', how='left')
        if 'PotentialFraud' in merged.columns:
            # Convert string labels to binary
            fraud_col = merged['PotentialFraud'].values
            if isinstance(fraud_col[0], str):
                true_labels = (fraud_col == 'Yes').astype(int)
            else:
                true_labels = fraud_col
    
    # Evaluate
    metrics = ModelEvaluator.evaluate(predictions, anomaly_scores, true_labels)
    
    # Generate plots if true labels available
    if true_labels is not None:
        output_dir = config.PLOTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ModelEvaluator.plot_confusion_matrix(
            true_labels, predictions,
            output_dir / "confusion_matrix.png"
        )
        
        ModelEvaluator.plot_roc_curve(
            true_labels, anomaly_scores,
            output_dir / "roc_curve.png"
        )
        
        ModelEvaluator.plot_precision_recall_curve(
            true_labels, anomaly_scores,
            output_dir / "precision_recall_curve.png"
        )
    
    # Plot anomaly distribution
    risk_levels = results_df['risk_level'].values
    ModelEvaluator.plot_anomaly_distribution(
        anomaly_scores, risk_levels,
        config.PLOTS_DIR / "anomaly_distribution.png"
    )
    
    # Save metrics
    save_json(metrics, config.EVALUATION_METRICS)
    
    return metrics
