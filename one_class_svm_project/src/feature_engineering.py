"""
Feature engineering and transformation
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from config import USE_LOG_TRANSFORM, LOG_FEATURES, PROVIDER_ID_COL, FRAUD_LABEL_COL, get_logger
from utils import check_infinite_values

logger = get_logger(__name__)

def get_feature_columns(df, exclude_cols=None):
    """
    Get numeric feature columns, excluding specific columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to analyze
    exclude_cols : list
        Columns to exclude (e.g., Provider, PotentialFraud)
    
    Returns:
    --------
    list
        List of feature column names
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Select numeric columns only
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove excluded columns
    feature_cols = [col for col in feature_cols if col not in exclude_cols]
    
    logger.info(f"Selected {len(feature_cols)} numeric feature columns")
    
    return feature_cols

def apply_log_transform(df, columns=None, log_features_list=None):
    """
    Apply log1p transformation to specified columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to transform
    columns : list, optional
        Columns to transform. If None, use log_features_list
    log_features_list : list, optional
        Default list of columns to transform
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with transformed columns
    """
    if not USE_LOG_TRANSFORM:
        logger.info("Log transformation disabled in config")
        return df
    
    if columns is None:
        columns = log_features_list or LOG_FEATURES
    
    df = df.copy()
    
    # Filter to only columns that exist
    cols_to_transform = [col for col in columns if col in df.columns]
    
    if not cols_to_transform:
        logger.info("No columns to transform")
        return df
    
    logger.info(f"Applying log1p transformation to {len(cols_to_transform)} columns")
    
    for col in cols_to_transform:
        # Check if values are suitable for log transform (must be >= 0 for log1p)
        if (df[col] < 0).any():
            logger.warning(f"{col}: Contains negative values, skipping log transform")
            continue
        
        df[col] = np.log1p(df[col])
        logger.info(f"  Transformed {col}")
    
    return df

def detect_skewness(df, numeric_cols=None):
    """
    Detect and report skewed columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to analyze
    numeric_cols : list, optional
        Columns to analyze. If None, uses all numeric
    
    Returns:
    --------
    dict
        Dictionary of column names to skewness values
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    skewness_dict = {}
    
    for col in numeric_cols:
        if col not in df.columns:
            continue
        
        skew_val = df[col].skew()
        skewness_dict[col] = skew_val
        
        # Report highly skewed columns
        if abs(skew_val) > 2:
            logger.info(f"{col}: Highly skewed (skewness={skew_val:.2f})")
    
    return skewness_dict

def select_features_by_variance(df, feature_cols, threshold=0.01):
    """
    Remove features with very low variance
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing features
    feature_cols : list
        Feature column names
    threshold : float
        Variance threshold
    
    Returns:
    --------
    list
        Filtered feature columns
    """
    variances = df[feature_cols].var()
    
    low_var_cols = variances[variances < threshold].index.tolist()
    
    if low_var_cols:
        logger.warning(f"Removing {len(low_var_cols)} low-variance features:")
        for col in low_var_cols:
            logger.warning(f"  {col}: variance={variances[col]:.6f}")
    
    filtered_cols = [col for col in feature_cols if col not in low_var_cols]
    
    logger.info(f"Feature selection: {len(feature_cols)} -> {len(filtered_cols)} features")
    
    return filtered_cols

def prepare_features_for_training(df, feature_cols, scale=False, scaler=None):
    """
    Prepare feature matrix for model training
    
    Parameters:
    -----------
    df : pd.DataFrame
        Provider feature dataframe
    feature_cols : list
        List of feature column names to use
    scale : bool
        Whether to scale features
    scaler : sklearn scaler, optional
        Fitted scaler to apply
    
    Returns:
    --------
    tuple
        (feature matrix, feature columns used)
    """
    # Verify all features exist
    missing_cols = set(feature_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")
    
    # Extract features
    X = df[feature_cols].copy()
    
    # Handle any remaining NaN values
    nan_count = X.isnull().sum().sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values in features, filling with 0")
        X = X.fillna(0)
    
    # Replace inf values
    inf_count = np.isinf(X).sum().sum()
    if inf_count > 0:
        logger.warning(f"Found {inf_count} infinite values, replacing with 0")
        X = X.replace([np.inf, -np.inf], 0)
    
    # Scale if requested
    if scale and scaler is not None:
        from preprocessing import scale_features
        X, _ = scale_features(X, scaler=scaler, fit=False)
    
    logger.info(f"Features prepared: {X.shape[0]} samples × {X.shape[1]} features")
    
    return X, feature_cols

def split_features_and_target(df, feature_cols, target_col=None):
    """
    Split dataframe into features and target
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with features and target
    feature_cols : list
        Feature column names
    target_col : str, optional
        Target column name
    
    Returns:
    --------
    tuple
        (X features, y target or None)
    """
    # Verify features exist
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    
    X = df[feature_cols].copy()
    
    y = None
    if target_col is not None and target_col in df.columns:
        y = df[target_col].copy()
    
    return X, y

def verify_features(X, expected_cols=None):
    """
    Verify features are ready for model training
    
    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Feature matrix
    expected_cols : list, optional
        Expected column names (if X is DataFrame)
    
    Raises:
    -------
    ValueError : If features are invalid
    """
    # Check for NaN
    if isinstance(X, pd.DataFrame):
        nan_count = X.isnull().sum().sum()
        if nan_count > 0:
            raise ValueError(f"Features contain {nan_count} NaN values")
        
        # Check for inf
        inf_count = np.isinf(X).sum().sum()
        if inf_count > 0:
            raise ValueError(f"Features contain {inf_count} infinite values")
        
        # Check column order if expected
        if expected_cols is not None:
            if list(X.columns) != expected_cols:
                raise ValueError("Feature column order mismatch")
    else:
        # Numpy array
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            raise ValueError(f"Features contain {nan_count} NaN values")
        
        inf_count = np.isinf(X).sum()
        if inf_count > 0:
            raise ValueError(f"Features contain {inf_count} infinite values")
    
    logger.info("Features verified - ready for model training")

def create_feature_importance_df(feature_names, importance_values=None):
    """
    Create a dataframe of features and their importance
    
    Parameters:
    -----------
    feature_names : list
        List of feature names
    importance_values : list or array, optional
        Importance values for each feature
    
    Returns:
    --------
    pd.DataFrame
        Feature importance dataframe
    """
    if importance_values is None:
        importance_values = np.ones(len(feature_names))
    
    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values
    })
    
    feature_df = feature_df.sort_values('Importance', ascending=False)
    
    return feature_df

def log_feature_statistics(df, feature_cols):
    """
    Log statistics about features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing features
    feature_cols : list
        Feature column names
    """
    logger.info("Feature Statistics:")
    logger.info("="*80)
    
    for col in feature_cols:
        if col not in df.columns:
            continue
        
        logger.info(f"\n{col}:")
        logger.info(f"  Mean:   {df[col].mean():.4f}")
        logger.info(f"  Median: {df[col].median():.4f}")
        logger.info(f"  Std:    {df[col].std():.4f}")
        logger.info(f"  Min:    {df[col].min():.4f}")
        logger.info(f"  Max:    {df[col].max():.4f}")
        logger.info(f"  Skew:   {df[col].skew():.4f}")
        logger.info(f"  Kurtosis: {df[col].kurtosis():.4f}")
