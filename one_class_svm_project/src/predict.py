"""
Prediction module for test data
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from config import (
    ARTIFACTS_DIR, OUTPUTS_DIR, PROVIDER_ID_COL,
    get_logger
)
from train import predict_provider_anomalies, convert_predictions, calculate_risk_score, assign_risk_levels
from utils import verify_dataframe

logger = get_logger(__name__)

def load_artifacts(artifact_dir=ARTIFACTS_DIR):
    """
    Load trained model artifacts
    
    Parameters:
    -----------
    artifact_dir : Path or str
        Directory containing artifacts
    
    Returns:
    --------
    dict
        Dictionary with loaded artifacts
    """
    artifact_dir = Path(artifact_dir)
    
    logger.info("Loading model artifacts...")
    
    artifacts = {}
    
    # Load model
    model_path = artifact_dir / "one_class_svm.pkl"
    if model_path.exists():
        with open(model_path, 'rb') as f:
            artifacts['model'] = pickle.load(f)
        logger.info(f"Loaded model from {model_path}")
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load scaler
    scaler_path = artifact_dir / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            artifacts['scaler'] = pickle.load(f)
        logger.info(f"Loaded scaler from {scaler_path}")
    else:
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    
    # Load feature columns
    features_path = artifact_dir / "feature_columns.pkl"
    if features_path.exists():
        with open(features_path, 'rb') as f:
            artifacts['feature_columns'] = pickle.load(f)
        logger.info(f"Loaded feature columns from {features_path}")
    else:
        raise FileNotFoundError(f"Feature columns not found: {features_path}")
    
    # Load metadata
    metadata_path = artifact_dir / "model_metadata.pkl"
    if metadata_path.exists():
        with open(metadata_path, 'rb') as f:
            artifacts['metadata'] = pickle.load(f)
        logger.info(f"Loaded metadata from {metadata_path}")
    
    return artifacts

def save_artifacts(model, scaler, feature_columns, metadata, artifact_dir=ARTIFACTS_DIR):
    """
    Save model artifacts
    
    Parameters:
    -----------
    model : OneClassSVM
        Trained model
    scaler : sklearn scaler
        Fitted scaler
    feature_columns : list
        Feature column names
    metadata : dict
        Model metadata
    artifact_dir : Path or str
        Directory to save artifacts
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(exist_ok=True)
    
    logger.info("Saving model artifacts...")
    
    # Save model
    with open(artifact_dir / "one_class_svm.pkl", 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Saved model to {artifact_dir / 'one_class_svm.pkl'}")
    
    # Save scaler
    with open(artifact_dir / "scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Saved scaler to {artifact_dir / 'scaler.pkl'}")
    
    # Save feature columns
    with open(artifact_dir / "feature_columns.pkl", 'wb') as f:
        pickle.dump(feature_columns, f)
    logger.info(f"Saved feature columns to {artifact_dir / 'feature_columns.pkl'}")
    
    # Save metadata
    with open(artifact_dir / "model_metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)
    logger.info(f"Saved metadata to {artifact_dir / 'model_metadata.pkl'}")

def predict_on_test_data(test_provider_features, artifacts, output_predictions=True):
    """
    Make predictions on test provider features
    
    CRITICAL: Use exact same feature order and preprocessing as training
    
    Parameters:
    -----------
    test_provider_features : pd.DataFrame
        Test provider features (must have all required columns)
    artifacts : dict
        Loaded model artifacts
    output_predictions : bool
        Whether to create predictions output
    
    Returns:
    --------
    pd.DataFrame
        Predictions dataframe
    """
    logger.info(f"\n{'='*80}")
    logger.info("MAKING TEST PREDICTIONS")
    logger.info(f"{'='*80}")
    
    model = artifacts['model']
    scaler = artifacts['scaler']
    feature_columns = artifacts['feature_columns']
    
    # Verify features
    missing_cols = set(feature_columns) - set(test_provider_features.columns)
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")
    
    extra_cols = set(test_provider_features.columns) - set(feature_columns) - {PROVIDER_ID_COL}
    if extra_cols:
        logger.warning(f"Extra columns in test data (will be ignored): {extra_cols}")
    
    # Extract features in exact order
    X_test = test_provider_features[feature_columns].copy()
    
    # Handle missing values
    nan_count = X_test.isnull().sum().sum()
    if nan_count > 0:
        logger.warning(f"Filling {nan_count} NaN values in test features")
        X_test = X_test.fillna(0)
    
    # Handle inf values
    inf_count = np.isinf(X_test).sum().sum()
    if inf_count > 0:
        logger.warning(f"Replacing {inf_count} infinite values in test features")
        X_test = X_test.replace([np.inf, -np.inf], 0)
    
    logger.info(f"Test providers: {len(X_test)}")
    logger.info(f"Features: {len(feature_columns)}")
    
    # Scale using training scaler
    X_test_scaled = scaler.transform(X_test)
    
    # Predict
    logger.info("Generating predictions...")
    predictions, anomaly_scores = predict_provider_anomalies(model, X_test_scaled)
    
    # Convert to binary (1=anomaly, 0=normal)
    binary_preds, _ = convert_predictions(predictions, anomaly_scores)
    
    # Calculate risk scores
    risk_scores = calculate_risk_score(anomaly_scores)
    
    # Assign risk levels
    risk_levels = assign_risk_levels(risk_scores)
    
    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        PROVIDER_ID_COL: test_provider_features[PROVIDER_ID_COL].values,
        'PredictedAnomaly': binary_preds,
        'DecisionScore': anomaly_scores,
        'RiskScore': risk_scores,
        'RiskLevel': risk_levels,
    })
    
    # Summary statistics
    n_anomalies = (binary_preds == 1).sum()
    n_normal = (binary_preds == 0).sum()
    
    logger.info(f"\nPrediction Summary:")
    logger.info(f"  Normal providers:     {n_normal}")
    logger.info(f"  Anomalous providers:  {n_anomalies}")
    logger.info(f"  Anomaly rate:         {n_anomalies/len(predictions_df)*100:.2f}%")
    
    logger.info(f"\nRisk Level Distribution:")
    risk_dist = predictions_df['RiskLevel'].value_counts()
    for level, count in risk_dist.items():
        logger.info(f"  {level}: {count} ({count/len(predictions_df)*100:.2f}%)")
    
    # Save predictions
    if output_predictions:
        output_path = OUTPUTS_DIR / "test_predictions.csv"
        predictions_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
    
    return predictions_df

def predict_with_pipeline(test_provider_features, pipeline_path=None):
    """
    Make predictions using complete pipeline
    
    Parameters:
    -----------
    test_provider_features : pd.DataFrame
        Test provider features
    pipeline_path : Path or str, optional
        Path to saved pipeline
    
    Returns:
    --------
    pd.DataFrame
        Predictions dataframe
    """
    # Load artifacts
    artifacts = load_artifacts()
    
    # Make predictions
    predictions = predict_on_test_data(test_provider_features, artifacts)
    
    return predictions

def get_suspicious_providers(predictions_df, risk_threshold=50):
    """
    Extract suspicious providers above risk threshold
    
    Parameters:
    -----------
    predictions_df : pd.DataFrame
        Predictions dataframe
    risk_threshold : float
        Risk score threshold (0-100)
    
    Returns:
    --------
    pd.DataFrame
        Filtered dataframe with suspicious providers
    """
    suspicious = predictions_df[predictions_df['RiskScore'] >= risk_threshold].copy()
    suspicious = suspicious.sort_values('RiskScore', ascending=False)
    
    logger.info(f"\nSuspicious Providers (Risk Score >= {risk_threshold}):")
    logger.info(f"  Count: {len(suspicious)}")
    logger.info(f"  Percentage: {len(suspicious)/len(predictions_df)*100:.2f}%")
    
    return suspicious

def get_high_confidence_anomalies(predictions_df, anomaly_score_threshold=None):
    """
    Extract anomalies with high confidence
    
    Parameters:
    -----------
    predictions_df : pd.DataFrame
        Predictions dataframe
    anomaly_score_threshold : float, optional
        Decision score threshold
    
    Returns:
    --------
    pd.DataFrame
        High-confidence anomalies
    """
    if anomaly_score_threshold is None:
        # Use median of negative scores (bottom 50%)
        anomaly_score_threshold = predictions_df['DecisionScore'].median()
    
    high_conf = predictions_df[predictions_df['DecisionScore'] <= anomaly_score_threshold].copy()
    high_conf = high_conf.sort_values('DecisionScore')
    
    logger.info(f"\nHigh-Confidence Anomalies (Score <= {anomaly_score_threshold:.4f}):")
    logger.info(f"  Count: {len(high_conf)}")
    
    return high_conf
