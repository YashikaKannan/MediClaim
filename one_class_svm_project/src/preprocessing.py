"""
Data preprocessing module
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from datetime import datetime
from config import (
    CLAIM_START_DT_COL, CLAIM_END_DT_COL, ADMISSION_DT_COL, DISCHARGE_DT_COL,
    DOB_COL, REIMBURSEMENT_COL, DEDUCTIBLE_COL,
    IMPUTATION_STRATEGY, USE_ROBUST_SCALER,
    get_logger
)
from utils import check_duplicates, check_missing_values, check_infinite_values

logger = get_logger(__name__)

def parse_dates(df, date_cols):
    """
    Convert date columns to datetime
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to process
    date_cols : list
        List of column names to convert
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with converted dates
    """
    df = df.copy()
    
    for col in date_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                non_null = df[col].notna().sum()
                logger.info(f"Parsed {col}: {non_null} valid dates, {df[col].isna().sum()} NaT")
            except Exception as e:
                logger.warning(f"Error parsing {col}: {e}")
    
    return df

def validate_dates(df, date_col, name=""):
    """
    Validate date column
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to validate
    date_col : str
        Column name to validate
    name : str
        Name for logging
    
    Returns:
    --------
    int
        Number of invalid dates
    """
    if date_col not in df.columns:
        return 0
    
    invalid = df[date_col].isna().sum()
    if invalid > 0:
        logger.warning(f"{name} {date_col}: {invalid} invalid/missing dates")
    
    return invalid

def handle_missing_numeric(df, numeric_cols, strategy=IMPUTATION_STRATEGY, fit_imputer=None):
    """
    Handle missing values in numeric columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to process
    numeric_cols : list
        List of numeric column names
    strategy : str
        Imputation strategy ('mean', 'median', 'zero')
    fit_imputer : SimpleImputer, optional
        Fitted imputer to apply. If None, will fit and return new imputer
    
    Returns:
    --------
    tuple
        (df with imputed values, fitted imputer)
    """
    df = df.copy()
    
    # Filter to only columns that exist in df
    numeric_cols_present = [col for col in numeric_cols if col in df.columns]
    
    if not numeric_cols_present:
        logger.info("No numeric columns to impute")
        return df, None
    
    # Create imputer if not provided
    if fit_imputer is None:
        if strategy == 'median':
            imputer = SimpleImputer(strategy='median')
        elif strategy == 'mean':
            imputer = SimpleImputer(strategy='mean')
        elif strategy == 'zero':
            imputer = SimpleImputer(strategy='constant', fill_value=0)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        logger.info(f"Fitting imputer with strategy: {strategy}")
    else:
        imputer = fit_imputer
        logger.info(f"Using provided imputer")
    
    # Impute
    df[numeric_cols_present] = imputer.fit_transform(df[numeric_cols_present])
    
    logger.info(f"Imputed {len(numeric_cols_present)} numeric columns")
    
    return df, imputer

def handle_missing_categorical(df, categorical_cols, missing_value='Unknown', fit_imputer=None):
    """
    Handle missing values in categorical columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to process
    categorical_cols : list
        List of categorical column names
    missing_value : str
        Value to fill missing with
    fit_imputer : SimpleImputer, optional
        Fitted imputer to apply
    
    Returns:
    --------
    tuple
        (df with filled values, fitted imputer)
    """
    df = df.copy()
    
    # Filter to only columns that exist in df
    categorical_cols_present = [col for col in categorical_cols if col in df.columns]
    
    if not categorical_cols_present:
        logger.info("No categorical columns to impute")
        return df, None
    
    if fit_imputer is None:
        imputer = SimpleImputer(strategy='constant', fill_value=missing_value)
        logger.info(f"Fitting categorical imputer with value: {missing_value}")
    else:
        imputer = fit_imputer
        logger.info(f"Using provided categorical imputer")
    
    df[categorical_cols_present] = imputer.fit_transform(df[categorical_cols_present])
    
    logger.info(f"Filled missing categorical values in {len(categorical_cols_present)} columns")
    
    return df, imputer

def validate_numeric_ranges(df, numeric_cols):
    """
    Validate that numeric columns are within reasonable ranges
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to validate
    numeric_cols : dict
        Dictionary mapping column name to (min_val, max_val) tuple
    
    Returns:
    --------
    dict
        Dictionary of validation results
    """
    results = {}
    
    for col, (min_val, max_val) in numeric_cols.items():
        if col not in df.columns:
            continue
        
        out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
        results[col] = out_of_range
        
        if out_of_range > 0:
            logger.warning(f"{col}: {out_of_range} values outside range [{min_val}, {max_val}]")
    
    return results

def remove_duplicates(df, subset=None, name="DataFrame"):
    """
    Remove duplicate rows
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to process
    subset : list, optional
        Columns to consider for duplicates
    name : str
        Name for logging
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with duplicates removed
    """
    before = len(df)
    df = df.drop_duplicates(subset=subset)
    after = len(df)
    
    if before > after:
        logger.info(f"{name}: Removed {before - after} duplicate rows")
    else:
        logger.info(f"{name}: No duplicates found")
    
    return df

def clean_text_columns(df, text_cols):
    """
    Clean text columns (strip whitespace, handle NaN)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to process
    text_cols : list
        List of text column names
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with cleaned text
    """
    df = df.copy()
    
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    return df

def standardize_column_names(df):
    """
    Standardize column names (strip whitespace, etc.)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to process
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with standardized column names
    """
    df.columns = df.columns.str.strip()
    return df

def create_scaler(use_robust=USE_ROBUST_SCALER):
    """
    Create a scaler object
    
    Parameters:
    -----------
    use_robust : bool
        If True, use RobustScaler; otherwise StandardScaler
    
    Returns:
    --------
    StandardScaler or RobustScaler
    """
    if use_robust:
        logger.info("Creating RobustScaler (for outlier-rich data)")
        return RobustScaler()
    else:
        logger.info("Creating StandardScaler")
        return StandardScaler()

def scale_features(X, scaler=None, fit=True):
    """
    Scale features
    
    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Features to scale
    scaler : StandardScaler/RobustScaler, optional
        Fitted scaler to apply. If None, will create new one
    fit : bool
        If True and scaler is None, fit the scaler
    
    Returns:
    --------
    tuple
        (scaled features, scaler object)
    """
    is_df = isinstance(X, pd.DataFrame)
    
    if scaler is None:
        scaler = create_scaler()
    
    if is_df:
        feature_cols = X.columns
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X) if fit else scaler.transform(X),
            columns=feature_cols,
            index=X.index
        )
    else:
        X_scaled = scaler.fit_transform(X) if fit else scaler.transform(X)
    
    logger.info(f"Scaled features - Mean: {X_scaled.mean().mean() if is_df else X_scaled.mean(axis=0).mean():.4f}")
    
    return X_scaled, scaler

def detect_and_handle_outliers_quantile(df, numeric_cols, lower_q=0.01, upper_q=0.99):
    """
    Detect outliers using quantiles (for logging)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to analyze
    numeric_cols : list
        List of numeric column names
    lower_q : float
        Lower quantile
    upper_q : float
        Upper quantile
    
    Returns:
    --------
    dict
        Statistics about outliers
    """
    stats = {}
    
    for col in numeric_cols:
        if col not in df.columns:
            continue
        
        lower = df[col].quantile(lower_q)
        upper = df[col].quantile(upper_q)
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        
        stats[col] = {
            'lower_bound': lower,
            'upper_bound': upper,
            'n_outliers': outliers,
            'pct_outliers': (outliers / len(df)) * 100
        }
        
        if outliers > 0:
            logger.info(f"{col}: {outliers} outliers ({stats[col]['pct_outliers']:.2f}%)")
    
    return stats

def preprocess_claims(df, name="Claims"):
    """
    Preprocess claims data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Claims dataframe
    name : str
        Name for logging
    
    Returns:
    --------
    pd.DataFrame
        Preprocessed claims
    """
    logger.info(f"\nPreprocessing {name}...")
    logger.info(f"  Input: {len(df)} rows")
    
    df = df.copy()
    
    # Remove duplicates
    df = remove_duplicates(df, name=name)
    
    # Parse dates
    date_cols = [col for col in [CLAIM_START_DT_COL, CLAIM_END_DT_COL, 
                                  ADMISSION_DT_COL, DISCHARGE_DT_COL]
                 if col in df.columns]
    df = parse_dates(df, date_cols)
    
    # Validate reimbursement values (should be non-negative)
    if REIMBURSEMENT_COL in df.columns:
        negative_reimb = (df[REIMBURSEMENT_COL] < 0).sum()
        if negative_reimb > 0:
            logger.warning(f"{name}: {negative_reimb} negative reimbursement values")
            df[REIMBURSEMENT_COL] = df[REIMBURSEMENT_COL].clip(lower=0)
    
    # Validate deductible values (should be non-negative)
    if DEDUCTIBLE_COL in df.columns:
        negative_ded = (df[DEDUCTIBLE_COL] < 0).sum()
        if negative_ded > 0:
            logger.warning(f"{name}: {negative_ded} negative deductible values")
            df[DEDUCTIBLE_COL] = df[DEDUCTIBLE_COL].clip(lower=0)
    
    logger.info(f"  Output: {len(df)} rows")
    
    return df

def preprocess_beneficiary(df, name="Beneficiary"):
    """
    Preprocess beneficiary data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Beneficiary dataframe
    name : str
        Name for logging
    
    Returns:
    --------
    pd.DataFrame
        Preprocessed beneficiary data
    """
    logger.info(f"\nPreprocessing {name}...")
    logger.info(f"  Input: {len(df)} rows")
    
    df = df.copy()
    
    # Remove duplicates
    df = remove_duplicates(df, subset=[DOB_COL] if DOB_COL in df.columns else None, name=name)
    
    # Parse DOB if present
    if DOB_COL in df.columns:
        df = parse_dates(df, [DOB_COL])
    
    logger.info(f"  Output: {len(df)} rows")
    
    return df

def preprocess_provider(df, name="Provider"):
    """
    Preprocess provider data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Provider dataframe
    name : str
        Name for logging
    
    Returns:
    --------
    pd.DataFrame
        Preprocessed provider data
    """
    logger.info(f"\nPreprocessing {name}...")
    logger.info(f"  Input: {len(df)} rows")
    
    df = df.copy()
    
    # Remove duplicates
    df = remove_duplicates(df, name=name)
    
    logger.info(f"  Output: {len(df)} rows")
    
    return df
