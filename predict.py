"""
Medicare Provider Anomaly Risk Engine - Prediction Script
Generate predictions on new provider data using trained model
"""

import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

import config
from src.utils import logger, setup_logging, print_section, save_csv

setup_logging(config.LOGGING_LEVEL)

class Predictor:
    """Prediction engine using trained model"""
    
    def __init__(self):
        """Load trained models and preprocessing"""
        self.model = None
        self.preprocessing = None
        self.feature_names = None
        self.metadata = None
        
        self.load_models()
    
    def load_models(self):
        """Load trained model and preprocessing pipeline"""
        print_section("Loading Trained Models")
        
        # Load Isolation Forest model
        if not config.ISOLATION_FOREST_MODEL.exists():
            raise FileNotFoundError(f"Model not found: {config.ISOLATION_FOREST_MODEL}")
        
        with open(config.ISOLATION_FOREST_MODEL, 'rb') as f:
            self.model = pickle.load(f)
        logger.info(f"Loaded Isolation Forest model from {config.ISOLATION_FOREST_MODEL}")
        
        # Load preprocessing pipeline
        if not config.PREPROCESSING_PIPELINE.exists():
            raise FileNotFoundError(f"Pipeline not found: {config.PREPROCESSING_PIPELINE}")
        
        with open(config.PREPROCESSING_PIPELINE, 'rb') as f:
            self.preprocessing = pickle.load(f)
        logger.info(f"Loaded preprocessing pipeline from {config.PREPROCESSING_PIPELINE}")
        
        # Load feature columns
        if not config.FEATURE_COLUMNS.exists():
            raise FileNotFoundError(f"Features not found: {config.FEATURE_COLUMNS}")
        
        with open(config.FEATURE_COLUMNS, 'rb') as f:
            self.feature_names = pickle.load(f)
        logger.info(f"Loaded {len(self.feature_names)} feature names")
        
        # Load metadata
        if config.MODEL_METADATA.exists():
            with open(config.MODEL_METADATA, 'rb') as f:
                self.metadata = pickle.load(f)
            logger.info(f"Loaded model metadata")
    
    def predict(self, features_df):
        """
        Generate predictions on new data
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Provider features (same format as training)
        
        Returns:
        --------
        pd.DataFrame
            Predictions with risk scores
        """
        print_section("Generating Predictions")
        
        # Ensure required columns exist
        required_cols = set(self.feature_names)
        available_cols = set(features_df.columns) - {'Provider'}
        
        missing_cols = required_cols - available_cols
        if missing_cols:
            logger.warning(f"Missing features: {missing_cols}")
            # Fill missing features with 0
            for col in missing_cols:
                features_df[col] = 0
        
        # Select only required features in correct order
        X = features_df[['Provider'] + self.feature_names].copy()
        
        # Separate Provider ID
        provider_ids = X['Provider'].values
        X_features = X[self.feature_names]
        
        logger.info(f"Preprocessing {len(X_features)} providers")
        
        # Apply preprocessing
        X_transformed = self.preprocessing.transform(X_features)
        
        # Generate predictions
        logger.info("Generating anomaly predictions")
        
        try:
            # Try ensemble predict
            predictions = self.model.predict_ensemble(X_transformed)
            anomaly_scores = self.model.score_ensemble(X_transformed)
        except AttributeError:
            # Fall back to single model
            predictions = self.model.predict_normalized(X_transformed)
            anomaly_scores = self.model.score_samples(X_transformed)
        
        # Normalize scores to 0-100
        min_score = np.nanmin(anomaly_scores)
        max_score = np.nanmax(anomaly_scores)
        
        if max_score > min_score:
            risk_scores = (anomaly_scores - min_score) / (max_score - min_score) * 100
        else:
            risk_scores = np.zeros_like(anomaly_scores)
        
        # Assign risk levels (percentile-based)
        p90 = np.percentile(risk_scores, 90)
        p95 = np.percentile(risk_scores, 95)
        p99 = np.percentile(risk_scores, 99)
        
        risk_levels = np.full(len(risk_scores), 'LOW', dtype=object)
        risk_levels[risk_scores >= p99] = 'CRITICAL'
        risk_levels[(risk_scores >= p95) & (risk_scores < p99)] = 'HIGH'
        risk_levels[(risk_scores >= p90) & (risk_scores < p95)] = 'MEDIUM'
        
        # Create results dataframe
        results = pd.DataFrame({
            'Provider': provider_ids,
            'anomaly_prediction': predictions,
            'anomaly_score_raw': anomaly_scores,
            'risk_score': risk_scores,
            'risk_level': risk_levels,
        })
        
        logger.info(f"Predictions generated for {len(results)} providers")
        print(f"Total predictions: {len(results)}")
        print(f"Anomalies detected: {predictions.sum()} ({predictions.sum()/len(results)*100:.1f}%)")
        
        return results


def main():
    """
    Main prediction pipeline
    Loads test features and generates predictions
    """
    
    print_section("MEDICARE PROVIDER ANOMALY RISK ENGINE")
    print("Prediction Pipeline")
    
    try:
        # Load test data
        print_section("Loading Test Data")
        
        # Import DataLoader
        from src.data_loader import DataLoader
        from src.data_cleaning import DataCleaner
        from src.feature_engineering import FeatureEngineer
        
        # Load test datasets
        loader = DataLoader(use_train=False)
        loader.load_all()
        
        # Clean data
        cleaner = DataCleaner()
        
        beneficiary_df = cleaner.clean_beneficiary_data(loader.get_beneficiary_data())
        inpatient_df = cleaner.clean_inpatient_claims(loader.get_inpatient_claims())
        outpatient_df = cleaner.clean_outpatient_claims(loader.get_outpatient_claims())
        
        # Add diagnosis and procedure counts
        inpatient_df, outpatient_df = cleaner.combine_claims(inpatient_df, outpatient_df)
        
        # Engineer features
        print_section("Generating Test Features")
        
        features_df = FeatureEngineer.engineer_provider_features(
            inpatient_df, outpatient_df, beneficiary_df
        )
        
        logger.info(f"Generated features for {len(features_df)} test providers")
        
        # Load predictor
        predictor = Predictor()
        
        # Generate predictions
        results = predictor.predict(features_df)
        
        # Save predictions
        print_section("Saving Results")
        
        output_file = config.OUTPUTS_DIR / "test_provider_anomaly_predictions.csv"
        save_csv(results, output_file)
        
        # Print summary
        print_section("Prediction Summary")
        
        print(f"\nTop 10 Most Suspicious Providers:")
        top_10 = results.nlargest(10, 'risk_score')[
            ['Provider', 'risk_score', 'risk_level', 'anomaly_prediction']
        ]
        print(top_10.to_string(index=False))
        
        print_section("Prediction Complete")
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        print_section("Prediction Failed")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
