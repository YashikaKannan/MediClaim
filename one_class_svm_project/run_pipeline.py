#!/usr/bin/env python
"""
Main orchestration script for One-Class SVM Anomaly Detection Pipeline
Executes the complete pipeline from data loading to final predictions
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import (
    OUTPUTS_DIR, ARTIFACTS_DIR, REPORTS_DIR, PROVIDER_ID_COL, FRAUD_LABEL_COL,
    get_logger, RANDOM_STATE
)
from utils import print_section, print_step, create_model_metadata, log_dataset_summary
from data_loader import load_training_data, load_test_data
from preprocessing import (
    preprocess_claims, preprocess_beneficiary, preprocess_provider,
    parse_dates, handle_missing_numeric, create_scaler, scale_features
)
from provider_aggregation import create_provider_feature_table
from feature_engineering import (
    get_feature_columns, apply_log_transform, detect_skewness,
    verify_features, log_feature_statistics
)
from train import (
    train_test_split_providers, prepare_training_data, prepare_validation_data,
    filter_normal_providers, train_one_class_svm, predict_provider_anomalies,
    convert_predictions, calculate_risk_score, assign_risk_levels,
    tune_hyperparameters, select_best_model
)
from evaluate import evaluate_model, create_prediction_dataframe, save_metrics_json, save_confusion_matrix_csv
from predict import save_artifacts, predict_on_test_data, load_artifacts, get_suspicious_providers

logger = get_logger(__name__)

def main():
    """Execute complete pipeline"""
    
    start_time = datetime.now()
    
    print_section("ONE-CLASS SVM PROVIDER ANOMALY DETECTION PIPELINE")
    logger.info(f"Pipeline start time: {start_time}")
    
    total_steps = 10
    current_step = 1
    
    try:
        # =====================================================================
        # STEP 1: DATA LOADING
        # =====================================================================
        print_step(current_step, total_steps, "Loading Training & Test Data")
        current_step += 1
        
        train_data, test_data = load_training_data(), load_test_data()
        
        # =====================================================================
        # STEP 2: DATA PREPROCESSING
        # =====================================================================
        print_step(current_step, total_steps, "Preprocessing Data")
        current_step += 1
        
        logger.info("\n" + "="*80)
        logger.info("DATA PREPROCESSING")
        logger.info("="*80)
        
        # Preprocess each dataset
        train_data['inpatient'] = preprocess_claims(train_data['inpatient'], "Train Inpatient")
        train_data['outpatient'] = preprocess_claims(train_data['outpatient'], "Train Outpatient")
        train_data['beneficiary'] = preprocess_beneficiary(train_data['beneficiary'], "Train Beneficiary")
        train_data['provider'] = preprocess_provider(train_data['provider'], "Train Provider")
        
        test_data['inpatient'] = preprocess_claims(test_data['inpatient'], "Test Inpatient")
        test_data['outpatient'] = preprocess_claims(test_data['outpatient'], "Test Outpatient")
        test_data['beneficiary'] = preprocess_beneficiary(test_data['beneficiary'], "Test Beneficiary")
        test_data['provider'] = preprocess_provider(test_data['provider'], "Test Provider")
        
        # =====================================================================
        # STEP 3: PROVIDER-LEVEL FEATURE AGGREGATION
        # =====================================================================
        print_step(current_step, total_steps, "Aggregating Claims to Provider Level")
        current_step += 1
        
        logger.info("\n" + "="*80)
        logger.info("PROVIDER-LEVEL FEATURE AGGREGATION")
        logger.info("="*80)
        
        # Training providers (with labels)
        train_provider_features = create_provider_feature_table(
            train_data['inpatient'],
            train_data['outpatient'],
            train_data['provider']
        )
        
        # Test providers (no labels)
        test_provider_features = create_provider_feature_table(
            test_data['inpatient'],
            test_data['outpatient'],
            None
        )
        
        # Log feature summary
        logger.info(f"\nTrain provider features: {train_provider_features.shape}")
        logger.info(f"Test provider features: {test_provider_features.shape}")
        
        # =====================================================================
        # STEP 4: FEATURE ENGINEERING
        # =====================================================================
        print_step(current_step, total_steps, "Feature Engineering & Selection")
        current_step += 1
        
        logger.info("\n" + "="*80)
        logger.info("FEATURE ENGINEERING")
        logger.info("="*80)
        
        # Get numeric feature columns (exclude provider ID and fraud label)
        exclude_cols = [PROVIDER_ID_COL, FRAUD_LABEL_COL]
        feature_cols = get_feature_columns(train_provider_features, exclude_cols)
        
        logger.info(f"\nSelected features: {len(feature_cols)}")
        logger.info(f"Features: {feature_cols[:10]}...") # Show first 10
        
        # Detect skewness
        skewness = detect_skewness(train_provider_features, feature_cols)
        
        # Apply log transformation to skewed features
        train_provider_features = apply_log_transform(
            train_provider_features,
            columns=None
        )
        test_provider_features = apply_log_transform(
            test_provider_features,
            columns=None
        )
        
        # Log feature statistics
        log_feature_statistics(train_provider_features, feature_cols)
        
        # =====================================================================
        # STEP 5: TRAIN/VALIDATION SPLIT
        # =====================================================================
        print_step(current_step, total_steps, "Creating Train/Validation Split")
        current_step += 1
        
        df_train, df_val, train_providers, val_providers = train_test_split_providers(
            train_provider_features,
            test_size=0.2,
            random_state=RANDOM_STATE
        )
        
        # =====================================================================
        # STEP 6: FEATURE SCALING
        # =====================================================================
        print_step(current_step, total_steps, "Scaling Features")
        current_step += 1
        
        logger.info("\n" + "="*80)
        logger.info("FEATURE SCALING")
        logger.info("="*80)
        
        # Scale training data
        X_train_scaled, scaler = prepare_training_data(
            df_train,
            feature_cols,
            scaler=None,
            fit_scaler=True
        )
        
        # Scale validation data (using training scaler)
        X_val_scaled = prepare_validation_data(
            df_val,
            feature_cols,
            scaler,
            feature_order=feature_cols
        )
        
        # =====================================================================
        # STEP 7: HYPERPARAMETER TUNING
        # =====================================================================
        print_step(current_step, total_steps, "Hyperparameter Tuning")
        current_step += 1
        
        logger.info("\n" + "="*80)
        logger.info("HYPERPARAMETER TUNING")
        logger.info("="*80)
        
        # Get only normal providers for training
        X_train_normal, train_normal_providers, n_fraud_excluded = filter_normal_providers(
            X_train_scaled,
            df_train[FRAUD_LABEL_COL],
            df_train[PROVIDER_ID_COL].values
        )
        
        # Convert validation labels to binary
        y_val_binary = (df_val[FRAUD_LABEL_COL] == "Yes").astype(int).values
        
        # Tune hyperparameters
        tuning_results = tune_hyperparameters(
            X_train_normal,
            X_val_scaled,
            y_val_binary
        )
        
        # Select best parameters
        best_params = select_best_model(tuning_results, metric='f1_score')
        
        # Save tuning results
        tuning_results.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)
        logger.info(f"Tuning results saved to {OUTPUTS_DIR / 'model_comparison.csv'}")
        
        # =====================================================================
        # STEP 8: VALIDATION EVALUATION
        # =====================================================================
        print_step(current_step, total_steps, "Model Validation & Evaluation")
        current_step += 1
        
        logger.info("\n" + "="*80)
        logger.info("VALIDATION EVALUATION")
        logger.info("="*80)
        
        # Train with best parameters
        best_model = train_one_class_svm(
            X_train_normal,
            nu=best_params['nu'],
            gamma=best_params['gamma'],
            kernel=best_params['kernel']
        )
        
        # Predict on validation
        val_preds, val_scores = predict_provider_anomalies(best_model, X_val_scaled)
        val_binary_preds, _ = convert_predictions(val_preds, val_scores)
        
        # Calculate metrics
        val_metrics = evaluate_model(y_val_binary, val_binary_preds, val_scores, "Validation Set")
        
        # Calculate risk scores
        val_risk_scores = calculate_risk_score(val_scores)
        val_risk_levels = assign_risk_levels(val_risk_scores)
        
        # Create validation predictions dataframe
        val_predictions_df = create_prediction_dataframe(
            df_val[PROVIDER_ID_COL].values,
            y_val_binary,
            val_binary_preds,
            val_risk_scores,
            val_risk_levels,
            val_scores
        )
        
        # Save validation predictions
        val_predictions_df.to_csv(OUTPUTS_DIR / "validation_predictions.csv", index=False)
        logger.info(f"Validation predictions saved to {OUTPUTS_DIR / 'validation_predictions.csv'}")
        
        # Save validation metrics
        save_metrics_json(val_metrics, OUTPUTS_DIR / "validation_metrics.json")
        save_confusion_matrix_csv(val_metrics, OUTPUTS_DIR / "confusion_matrix.csv")
        
        # =====================================================================
        # STEP 9: FINAL MODEL TRAINING
        # =====================================================================
        print_step(current_step, total_steps, "Training Final Model")
        current_step += 1
        
        logger.info("\n" + "="*80)
        logger.info("FINAL MODEL TRAINING")
        logger.info("="*80)
        
        # Re-fit scaler on ALL training data (including fraud providers)
        X_train_all_scaled, scaler_final = prepare_training_data(
            df_train,
            feature_cols,
            scaler=None,
            fit_scaler=True
        )
        
        # Get normal providers for final training
        X_train_normal_final, _, _ = filter_normal_providers(
            X_train_all_scaled,
            df_train[FRAUD_LABEL_COL],
            df_train[PROVIDER_ID_COL].values
        )
        
        # Train final model
        final_model = train_one_class_svm(
            X_train_normal_final,
            nu=best_params['nu'],
            gamma=best_params['gamma'],
            kernel=best_params['kernel']
        )
        
        # Create metadata
        model_metadata = create_model_metadata(
            final_model,
            scaler_final,
            n_training_providers=len(df_train),
            n_normal_providers=len(X_train_normal_final),
            n_fraud_excluded=n_fraud_excluded,
            feature_names=feature_cols,
            nu=best_params['nu'],
            gamma=best_params['gamma']
        )
        
        # Save artifacts
        save_artifacts(final_model, scaler_final, feature_cols, model_metadata, ARTIFACTS_DIR)
        
        # =====================================================================
        # STEP 10: TEST PREDICTION
        # =====================================================================
        print_step(current_step, total_steps, "Generating Test Predictions")
        current_step += 1
        
        logger.info("\n" + "="*80)
        logger.info("TEST PREDICTION")
        logger.info("="*80)
        
        # Scale test features using final scaler
        X_test = test_provider_features[feature_cols].copy()
        
        # Handle NaN in test features
        nan_count_test = X_test.isnull().sum().sum()
        if nan_count_test > 0:
            logger.info(f"Filling {nan_count_test} NaN values in test features")
            X_test = X_test.fillna(0)
        
        # Handle inf in test features
        inf_count_test = np.isinf(X_test).sum().sum()
        if inf_count_test > 0:
            logger.info(f"Replacing {inf_count_test} infinite values in test features")
            X_test = X_test.replace([np.inf, -np.inf], 0)
        
        X_test_scaled = scaler_final.transform(X_test)
        
        # Predict
        test_preds, test_scores = predict_provider_anomalies(final_model, X_test_scaled)
        test_binary_preds, _ = convert_predictions(test_preds, test_scores)
        
        # Calculate risk scores
        test_risk_scores = calculate_risk_score(test_scores)
        test_risk_levels = assign_risk_levels(test_risk_scores)
        
        # Create test predictions dataframe
        test_predictions_df = pd.DataFrame({
            PROVIDER_ID_COL: test_provider_features[PROVIDER_ID_COL].values,
            'PredictedAnomaly': test_binary_preds,
            'DecisionScore': test_scores,
            'RiskScore': test_risk_scores,
            'RiskLevel': test_risk_levels,
        })
        
        # Save test predictions
        test_predictions_df.to_csv(OUTPUTS_DIR / "test_predictions.csv", index=False)
        logger.info(f"Test predictions saved to {OUTPUTS_DIR / 'test_predictions.csv'}")
        
        # Summary
        n_suspicious = (test_binary_preds == 1).sum()
        logger.info(f"\nTest Summary:")
        logger.info(f"  Total providers: {len(test_predictions_df)}")
        logger.info(f"  Suspicious providers: {n_suspicious} ({n_suspicious/len(test_predictions_df)*100:.2f}%)")
        
        # =====================================================================
        # FINAL REPORT
        # =====================================================================
        print_section("PIPELINE EXECUTION SUMMARY")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"\nExecution Time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        logger.info(f"\n{'='*80}")
        logger.info("FINAL RESULTS")
        logger.info(f"{'='*80}\n")
        
        logger.info(f"Training Data:")
        logger.info(f"  Total providers: {len(train_provider_features)}")
        logger.info(f"  Normal (No): {(train_provider_features[FRAUD_LABEL_COL] == 'No').sum()}")
        logger.info(f"  Fraud (Yes): {(train_provider_features[FRAUD_LABEL_COL] == 'Yes').sum()}")
        
        logger.info(f"\nTrain/Validation Split:")
        logger.info(f"  Training providers (used for fitting): {len(df_train)}")
        logger.info(f"  Validation providers: {len(df_val)}")
        logger.info(f"  Normal training providers (OC-SVM): {len(X_train_normal)}")
        
        logger.info(f"\nFeatures:")
        logger.info(f"  Total features: {len(feature_cols)}")
        logger.info(f"  Feature list: {feature_cols}")
        
        logger.info(f"\nBest Hyperparameters:")
        logger.info(f"  nu: {best_params['nu']}")
        logger.info(f"  gamma: {best_params['gamma']}")
        logger.info(f"  kernel: {best_params['kernel']}")
        
        logger.info(f"\nValidation Metrics:")
        logger.info(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {val_metrics['precision']:.4f}")
        logger.info(f"  Recall:    {val_metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {val_metrics['f1_score']:.4f}")
        if val_metrics['roc_auc'] is not None:
            logger.info(f"  ROC-AUC:   {val_metrics['roc_auc']:.4f}")
        if val_metrics['pr_auc'] is not None:
            logger.info(f"  PR-AUC:    {val_metrics['pr_auc']:.4f}")
        
        logger.info(f"\nConfusion Matrix (Validation):")
        logger.info(f"  TN: {val_metrics['tn']}, FP: {val_metrics['fp']}")
        logger.info(f"  FN: {val_metrics['fn']}, TP: {val_metrics['tp']}")
        
        logger.info(f"\nTest Predictions:")
        logger.info(f"  Total test providers: {len(test_predictions_df)}")
        logger.info(f"  Anomalous providers: {n_suspicious}")
        logger.info(f"  Anomaly rate: {n_suspicious/len(test_predictions_df)*100:.2f}%")
        
        # Risk level distribution
        test_risk_dist = test_predictions_df['RiskLevel'].value_counts().sort_index()
        logger.info(f"\nTest Risk Level Distribution:")
        for level in ['Low', 'Medium', 'High', 'Critical']:
            if level in test_risk_dist.index:
                count = test_risk_dist[level]
                logger.info(f"  {level}: {count} ({count/len(test_predictions_df)*100:.2f}%)")
        
        logger.info(f"\nArtifacts Saved:")
        logger.info(f"  - {ARTIFACTS_DIR / 'one_class_svm.pkl'}")
        logger.info(f"  - {ARTIFACTS_DIR / 'scaler.pkl'}")
        logger.info(f"  - {ARTIFACTS_DIR / 'feature_columns.pkl'}")
        logger.info(f"  - {ARTIFACTS_DIR / 'model_metadata.pkl'}")
        
        logger.info(f"\nOutput Files Generated:")
        logger.info(f"  - {OUTPUTS_DIR / 'validation_predictions.csv'}")
        logger.info(f"  - {OUTPUTS_DIR / 'test_predictions.csv'}")
        logger.info(f"  - {OUTPUTS_DIR / 'validation_metrics.json'}")
        logger.info(f"  - {OUTPUTS_DIR / 'confusion_matrix.csv'}")
        logger.info(f"  - {OUTPUTS_DIR / 'model_comparison.csv'}")
        
        logger.info(f"\n{'='*80}")
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"{'='*80}\n")
        
        return True
        
    except Exception as e:
        logger.error(f"\n{'='*80}")
        logger.error("PIPELINE FAILED WITH ERROR")
        logger.error(f"{'='*80}\n")
        logger.error(f"Error at step {current_step}: {e}")
        logger.error("", exc_info=True)
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
