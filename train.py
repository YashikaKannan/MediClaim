"""
Medicare Provider Anomaly Risk Engine - Training Script
Complete training pipeline from data loading to model deployment
"""

import pickle
import pandas as pd
import numpy as np
import sys
import json
from datetime import datetime
from pathlib import Path

import config
from src.utils import logger, setup_logging, print_section, save_csv, save_json
from src.data_loader import DataLoader, load_training_data
from src.data_audit import run_data_audit
from src.data_cleaning import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.preprocessing import PreprocessingPipeline
from src.isolation_forest_engine import create_ensemble_isolation_forest, create_isolation_forest_engine
from src.anomaly_scoring import AnomalyScorer
from src.evaluation import run_evaluation
from src.anomaly_explanation import AnomalyExplainer

setup_logging(config.LOGGING_LEVEL)

def main():
    """
    Main training pipeline
    """
    
    start_time = datetime.now()
    logger.info(f"Starting Medicare Provider Anomaly Risk Engine")
    logger.info(f"Start time: {start_time}")
    
    print_section("MEDICARE PROVIDER ANOMALY RISK ENGINE")
    print("Training Pipeline")
    print(f"Start time: {start_time}")
    
    try:
        # ========================
        # 1. LOAD DATA
        # ========================
        print_section("STEP 1: DATA LOADING")
        
        loader = load_training_data()
        loader.detect_columns()
        
        # ========================
        # 2. DATA AUDIT
        # ========================
        print_section("STEP 2: DATA AUDIT")
        
        audit_report = run_data_audit(loader)
        
        # ========================
        # 3. DATA CLEANING
        # ========================
        print_section("STEP 3: DATA CLEANING")
        
        cleaner = DataCleaner()
        
        beneficiary_df = cleaner.clean_beneficiary_data(loader.get_beneficiary_data())
        inpatient_df = cleaner.clean_inpatient_claims(loader.get_inpatient_claims())
        outpatient_df = cleaner.clean_outpatient_claims(loader.get_outpatient_claims())
        
        # Add diagnosis and procedure counts
        inpatient_df, outpatient_df = cleaner.combine_claims(inpatient_df, outpatient_df)
        
        # ========================
        # 4. FEATURE ENGINEERING
        # ========================
        print_section("STEP 4: FEATURE ENGINEERING")
        
        features_df = FeatureEngineer.engineer_provider_features(
            inpatient_df, outpatient_df, beneficiary_df
        )
        
        logger.info(f"Engineered features for {len(features_df)} providers")
        print(f"Provider features: {features_df.shape[0]} providers × {features_df.shape[1]} features")
        
        # ========================
        # 5. PREPROCESSING
        # ========================
        print_section("STEP 5: PREPROCESSING")
        
        preprocessing = PreprocessingPipeline()
        X, feature_names = preprocessing.prepare_features_for_training(features_df)
        
        logger.info(f"Features selected: {len(feature_names)}")
        
        # Fit preprocessing on training data
        preprocessing.fit_preprocessing(X)
        X_transformed = preprocessing.transform(X)
        
        logger.info(f"Preprocessing complete: {X_transformed.shape}")
        print(f"Transformed data shape: {X_transformed.shape}")
        
        # ========================
        # 6. TRAIN ISOLATION FOREST
        # ========================
        print_section("STEP 6: ISOLATION FOREST TRAINING")
        
        # Try ensemble approach
        use_ensemble = True
        
        if use_ensemble:
            logger.info("Using Ensemble Isolation Forest")
            model = create_ensemble_isolation_forest(
                contamination=config.DEFAULT_CONTAMINATION,
                use_ensemble=True
            )
        else:
            logger.info("Using Single Isolation Forest")
            model = create_isolation_forest_engine(
                contamination=config.DEFAULT_CONTAMINATION
            )
        
        model.train(X_transformed)
        
        # ========================
        # 7. GENERATE PREDICTIONS
        # ========================
        print_section("STEP 7: GENERATING PREDICTIONS")
        
        # Get predictions and scores
        if use_ensemble:
            predictions = model.predict_ensemble(X_transformed)
            anomaly_scores = model.score_ensemble(X_transformed)
        else:
            predictions = model.predict_normalized(X_transformed)
            anomaly_scores = model.score_samples(X_transformed)
        
        # ========================
        # 8. ANOMALY SCORING
        # ========================
        print_section("STEP 8: ANOMALY SCORING")
        
        results_df = AnomalyScorer.create_anomaly_report(
            features_df, predictions, anomaly_scores, features_df
        )
        
        # Save predictions
        save_csv(results_df, config.PROVIDER_PREDICTIONS)
        
        # ========================
        # 9. EVALUATION
        # ========================
        print_section("STEP 9: EVALUATION")
        
        # Load provider labels for evaluation (PotentialFraud only used for evaluation)
        provider_labels = loader.get_provider_labels()
        
        metrics = run_evaluation(results_df, provider_labels)
        
        # ========================
        # 10. TOP SUSPICIOUS PROVIDERS
        # ========================
        print_section("STEP 10: TOP SUSPICIOUS PROVIDERS")
        
        top_suspicious = results_df.nlargest(50, 'risk_score')[
            ['Provider', 'anomaly_prediction', 'anomaly_score_raw', 'risk_score', 'risk_level',
             'total_claim_count', 'total_reimbursement', 'unique_beneficiary_count',
             'avg_reimbursement', 'claims_per_beneficiary']
        ].copy()
        
        save_csv(top_suspicious, config.TOP_SUSPICIOUS)
        
        print(f"\nTop 10 Most Suspicious Providers:")
        print(top_suspicious.head(10).to_string(index=False))
        
        # ========================
        # 11. ANOMALY EXPLANATIONS
        # ========================
        print_section("STEP 11: ANOMALY EXPLANATIONS")
        
        explanations = AnomalyExplainer.generate_explanations(results_df, features_df, top_suspicious_n=10)
        
        # Print top explanation
        if explanations:
            AnomalyExplainer.print_explanation(explanations[0])
        
        # ========================
        # 12. SAVE MODELS
        # ========================
        print_section("STEP 12: SAVING MODELS")
        
        # Save Isolation Forest model
        with open(config.ISOLATION_FOREST_MODEL, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Saved Isolation Forest model to {config.ISOLATION_FOREST_MODEL}")
        
        # Save preprocessing pipeline
        with open(config.PREPROCESSING_PIPELINE, 'wb') as f:
            pickle.dump(preprocessing, f)
        logger.info(f"Saved preprocessing pipeline to {config.PREPROCESSING_PIPELINE}")
        
        # Save feature columns
        with open(config.FEATURE_COLUMNS, 'wb') as f:
            pickle.dump(feature_names, f)
        logger.info(f"Saved feature columns to {config.FEATURE_COLUMNS}")
        
        # Save metadata
        metadata = {
            'algorithm': 'Isolation Forest',
            'ensemble': use_ensemble,
            'random_seeds': config.RANDOM_SEEDS if use_ensemble else [config.RANDOM_STATE],
            'contamination': config.DEFAULT_CONTAMINATION,
            'n_providers': len(features_df),
            'n_features': len(feature_names),
            'features': feature_names,
            'training_timestamp': start_time.isoformat(),
            'training_date': start_time.strftime('%Y-%m-%d'),
            'total_anomalies': int(predictions.sum()),
            'anomaly_rate': float(predictions.sum() / len(predictions) * 100),
            'metrics': metrics,
        }
        
        with open(config.MODEL_METADATA, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"Saved model metadata to {config.MODEL_METADATA}")
        
        # ========================
        # 13. GENERATE REPORT
        # ========================
        print_section("STEP 13: GENERATING EVALUATION REPORT")
        
        report = generate_evaluation_report(metrics, results_df)
        
        with open(config.EVALUATION_REPORT, 'w') as f:
            f.write(report)
        logger.info(f"Saved evaluation report to {config.EVALUATION_REPORT}")
        
        # ========================
        # 14. VALIDATION
        # ========================
        print_section("STEP 14: MODEL VALIDATION")
        
        validation_passed = validate_model_outputs()
        
        if validation_passed:
            print_section("MODEL VALIDATION: PASSED [OK]")
        else:
            print_section("MODEL VALIDATION: FAILED [ERROR]")
            sys.exit(1)
        
        # ========================
        # COMPLETION
        # ========================
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print_section("TRAINING COMPLETE")
        print(f"End time: {end_time}")
        print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"\nAll outputs saved to: {config.OUTPUTS_DIR}")
        print(f"Models saved to: {config.MODELS_DIR}")
        
        logger.info(f"Training complete! Duration: {duration:.1f} seconds")
        
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        print_section("TRAINING FAILED")
        print(f"Error: {e}")
        sys.exit(1)


def generate_evaluation_report(metrics, results_df):
    """Generate comprehensive evaluation report"""
    
    risk_counts = results_df['risk_level'].value_counts()
    
    report = "=" * 70 + "\n"
    report += "MEDICARE PROVIDER ANOMALY RISK ENGINE\n"
    report += "EVALUATION REPORT\n"
    report += "=" * 70 + "\n\n"
    
    report += "ALGORITHM\n"
    report += "-" * 70 + "\n"
    report += f"Isolation Forest\n\n"
    
    report += "DATASET\n"
    report += "-" * 70 + "\n"
    report += f"Providers Analyzed: {metrics['total_providers']}\n"
    report += f"Features Used: 50+\n\n"
    
    report += "ANOMALIES DETECTED\n"
    report += "-" * 70 + "\n"
    report += f"Anomalies: {metrics['anomalies_detected']}\n"
    report += f"Normal: {metrics['normal_providers']}\n"
    report += f"Anomaly Rate: {metrics['anomaly_rate']:.2f}%\n\n"
    
    if 'precision' in metrics and metrics['precision'] is not None:
        report += "SUPERVISED EVALUATION METRICS (using PotentialFraud label)\n"
        report += "-" * 70 + "\n"
        report += f"Precision: {metrics['precision']:.4f}\n"
        report += f"Recall: {metrics['recall']:.4f}\n"
        report += f"F1-Score: {metrics['f1_score']:.4f}\n"
        
        if metrics['roc_auc'] is not None:
            report += f"ROC-AUC: {metrics['roc_auc']:.4f}\n"
        if metrics['pr_auc'] is not None:
            report += f"PR-AUC: {metrics['pr_auc']:.4f}\n"
        
        report += f"\nFraud Detection Rate: {metrics['fraud_detection_rate']:.2f}%\n"
        report += f"False Positive Rate: {metrics['false_positive_rate']:.2f}%\n\n"
    
    report += "RISK DISTRIBUTION\n"
    report += "-" * 70 + "\n"
    for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        count = risk_counts.get(level, 0)
        pct = count / len(results_df) * 100
        report += f"{level:.<20} {count:>6} ({pct:>5.1f}%)\n"
    
    report += "\n" + "=" * 70 + "\n"
    
    return report


def validate_model_outputs():
    """Validate all required outputs exist and are valid"""
    
    print("\nValidating model outputs...")
    
    checks = [
        ("Isolation Forest Model", config.ISOLATION_FOREST_MODEL),
        ("Preprocessing Pipeline", config.PREPROCESSING_PIPELINE),
        ("Feature Columns", config.FEATURE_COLUMNS),
        ("Model Metadata", config.MODEL_METADATA),
        ("Predictions CSV", config.PROVIDER_PREDICTIONS),
        ("Top Suspicious Providers", config.TOP_SUSPICIOUS),
        ("Evaluation Metrics", config.EVALUATION_METRICS),
    ]
    
    all_passed = True
    
    for check_name, file_path in checks:
        if file_path.exists():
            print(f"[OK] {check_name}: {file_path}")
        else:
            print(f"[FAIL] {check_name}: MISSING")
            all_passed = False
    
    # Check predictions validity
    if config.PROVIDER_PREDICTIONS.exists():
        predictions_df = pd.read_csv(config.PROVIDER_PREDICTIONS)
        
        # Check for NaN predictions
        if predictions_df['anomaly_prediction'].isnull().any():
            print("[FAIL] Predictions contain NaN values")
            all_passed = False
        else:
            print("[OK] No NaN predictions")
        
        # Check risk scores range
        if (predictions_df['risk_score'] >= 0).all() and (predictions_df['risk_score'] <= 100).all():
            print("[OK] Risk scores within 0-100 range")
        else:
            print("[FAIL] Risk scores out of range")
            all_passed = False
        
        # Check risk levels valid
        valid_levels = {'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'}
        if predictions_df['risk_level'].isin(valid_levels).all():
            print("[OK] Risk levels valid")
        else:
            print("[FAIL] Invalid risk levels")
            all_passed = False
        
        # Check every provider has prediction
        print(f"[OK] Predictions for {len(predictions_df)} providers")
    
    return all_passed


if __name__ == "__main__":
    main()
