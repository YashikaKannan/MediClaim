"""
Utility functions for the pipeline
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from config import get_logger, SKLEARN_VERSION, PYTHON_VERSION

logger = get_logger(__name__)

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")

def print_step(step_num, total_steps, title):
    """Print a numbered step"""
    print(f"[{step_num}/{total_steps}] {title}")

def verify_dataframe(df, name, expected_cols=None, min_rows=0):
    """
    Verify a dataframe has expected properties
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to verify
    name : str
        Name of the dataframe for error messages
    expected_cols : list, optional
        List of columns that must exist
    min_rows : int
        Minimum number of rows required
    
    Returns:
    --------
    bool
        True if valid
        
    Raises:
    -------
    ValueError : If validation fails
    """
    if df is None or df.empty:
        raise ValueError(f"{name} is empty or None")
    
    if len(df) < min_rows:
        raise ValueError(f"{name} has {len(df)} rows, expected at least {min_rows}")
    
    if expected_cols is not None:
        missing_cols = set(expected_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"{name} missing columns: {missing_cols}")
    
    logger.info(f"{name}: {len(df)} rows x {len(df.columns)} columns")
    return True

def check_no_data_leakage(df, forbidden_cols):
    """
    Check that forbidden columns are not present in dataframe
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to check
    forbidden_cols : list
        Columns that should not be present
    
    Raises:
    -------
    ValueError : If forbidden columns are found
    """
    found_cols = set(forbidden_cols) & set(df.columns)
    if found_cols:
        raise ValueError(f"Data leakage detected! Found forbidden columns: {found_cols}")

def check_duplicates(df, subset=None, name="DataFrame"):
    """
    Check for duplicate rows
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to check
    subset : list, optional
        Columns to consider for duplicate detection
    name : str
        Name for error messages
    
    Returns:
    --------
    int
        Number of duplicates found
    """
    if subset:
        dup_count = df.duplicated(subset=subset).sum()
    else:
        dup_count = df.duplicated().sum()
    
    if dup_count > 0:
        logger.warning(f"{name}: Found {dup_count} duplicate rows")
    else:
        logger.info(f"{name}: No duplicates found")
    
    return dup_count

def check_missing_values(df, critical_cols=None, name="DataFrame"):
    """
    Check for missing values
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to check
    critical_cols : list, optional
        Columns that should have no missing values
    name : str
        Name for error messages
    
    Raises:
    -------
    ValueError : If critical columns have missing values
    """
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.warning(f"{name}: Found {missing.sum()} missing values")
        logger.warning(f"Missing value distribution:\n{missing[missing > 0]}")
    
    if critical_cols:
        missing_in_critical = missing[missing.index.isin(critical_cols)]
        if missing_in_critical.sum() > 0:
            raise ValueError(f"{name}: Missing values in critical columns:\n{missing_in_critical}")

def check_infinite_values(df, numeric_only=True, name="DataFrame"):
    """
    Check for infinite values
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to check
    numeric_only : bool
        Only check numeric columns
    name : str
        Name for error messages
    
    Raises:
    -------
    ValueError : If infinite values found
    """
    if numeric_only:
        df_check = df.select_dtypes(include=[np.number])
    else:
        df_check = df
    
    inf_count = np.isinf(df_check).sum().sum()
    if inf_count > 0:
        raise ValueError(f"{name}: Found {inf_count} infinite values")

def check_range(series, min_val=None, max_val=None, name="Series"):
    """
    Check if series values are within expected range
    
    Parameters:
    -----------
    series : pd.Series
        Series to check
    min_val : float, optional
        Minimum expected value
    max_val : float, optional
        Maximum expected value
    name : str
        Name for error messages
    
    Raises:
    -------
    ValueError : If values out of range
    """
    if min_val is not None and (series < min_val).any():
        raise ValueError(f"{name}: Found values less than {min_val}")
    
    if max_val is not None and (series > max_val).any():
        raise ValueError(f"{name}: Found values greater than {max_val}")

def create_model_metadata(model, scaler, n_training_providers, n_normal_providers,
                         n_fraud_excluded, feature_names, nu, gamma):
    """
    Create metadata dictionary for the trained model
    
    Parameters:
    -----------
    model : OneClassSVM
        Trained model
    scaler : StandardScaler/RobustScaler
        Fitted scaler
    n_training_providers : int
        Total training providers
    n_normal_providers : int
        Number of normal providers used for training
    n_fraud_excluded : int
        Number of fraud providers excluded
    feature_names : list
        Names of features used
    nu : float
        Nu parameter value
    gamma : float or str
        Gamma parameter value
    
    Returns:
    --------
    dict
        Metadata dictionary
    """
    import sklearn
    
    metadata = {
        'model_name': 'One-Class SVM Provider Anomaly Detector',
        'algorithm': 'One-Class SVM',
        'kernel': model.kernel,
        'nu': nu,
        'gamma': gamma,
        'random_state': 42,
        'n_features_in': model.n_features_in_,
        'n_support_vectors': len(model.support_vectors_),
        'training_date': pd.Timestamp.now().isoformat(),
        'n_training_providers': n_training_providers,
        'n_normal_providers': n_normal_providers,
        'n_fraud_excluded': n_fraud_excluded,
        'feature_count': len(feature_names),
        'feature_names': feature_names,
        'sklearn_version': sklearn.__version__,
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    }
    
    return metadata

def log_dataset_summary(df, name):
    """Log summary statistics for a dataset"""
    print(f"\n{name}:")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"  Duplicates: {df.duplicated().sum()}")

def format_metric(value, name, percentage=False):
    """Format a metric for display"""
    if percentage:
        return f"{name}: {value*100:.2f}%"
    else:
        return f"{name}: {value:.4f}"

def calculate_metrics_display(y_true, y_pred):
    """Format all key metrics for display"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='binary', zero_division=0)
    rec = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    }
