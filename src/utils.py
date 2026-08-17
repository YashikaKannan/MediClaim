"""
Medicare Provider Anomaly Risk Engine - Utility Functions
"""

import logging
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Setup logging
def setup_logging(level="INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def save_json(data, filepath):
    """Save data to JSON file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved JSON to {filepath}")

def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_csv(df, filepath):
    """Save DataFrame to CSV"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info(f"Saved CSV to {filepath} ({len(df)} rows, {len(df.columns)} columns)")

def load_csv(filepath, **kwargs):
    """Load CSV file"""
    df = pd.read_csv(filepath, **kwargs)
    logger.info(f"Loaded CSV from {filepath} ({len(df)} rows, {len(df.columns)} columns)")
    return df

def safe_date_parse(date_val):
    """Safely parse date string or Series to datetime"""
    if isinstance(date_val, pd.Series):
        return pd.to_datetime(date_val, errors='coerce')
    if pd.isna(date_val) or date_val == '':
        return pd.NaT
    return pd.to_datetime(date_val, errors='coerce')

def calculate_duration_days(start_date, end_date):
    """Calculate duration between two dates in days"""
    if pd.isna(start_date) or pd.isna(end_date):
        return np.nan
    
    if end_date < start_date:
        return np.nan
    
    return (end_date - start_date).days

def count_non_null(series):
    """Count non-null values in a series"""
    return series.notna().sum()

def unique_count(series):
    """Count unique values in a series"""
    return series.nunique()

def safe_divide(numerator, denominator, default=0):
    """Safe division handling zero denominator"""
    if denominator == 0 or pd.isna(denominator):
        return default
    return numerator / denominator

def percentile_in_group(value, group_series):
    """Calculate percentile of value within group"""
    if pd.isna(value) or len(group_series) == 0:
        return np.nan
    
    return (group_series < value).sum() / len(group_series) * 100

def median_absolute_deviation(series):
    """Calculate median absolute deviation (robust std)"""
    if len(series) == 0:
        return np.nan
    
    median = series.median()
    mad = (series - median).abs().median()
    return mad

def robust_zscore(series):
    """Calculate robust z-score using MAD"""
    median = series.median()
    mad = median_absolute_deviation(series)
    
    if mad == 0 or pd.isna(mad):
        return np.nan
    
    return (series - median) / (1.4826 * mad)

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def print_metric(name, value, format_string=None):
    """Print a formatted metric"""
    if format_string:
        print(f"{name:.<50} {format_string.format(value)}")
    else:
        print(f"{name:.<50} {value}")

def validate_directory(directory):
    """Validate and create directory if needed"""
    Path(directory).mkdir(parents=True, exist_ok=True)
    return Path(directory)

def get_memory_usage(df):
    """Get memory usage of dataframe in MB"""
    return df.memory_usage(deep=True).sum() / 1024**2

def print_dataframe_info(df, title=None):
    """Print comprehensive dataframe information"""
    if title:
        print_section(title)
    
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Memory: {get_memory_usage(df):.2f} MB")
    print(f"\nColumns and types:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")
    
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        for col in missing[missing > 0].index:
            pct = 100 * missing[col] / len(df)
            print(f"  {col}: {missing[col]} ({pct:.1f}%)")
    else:
        print("  None")

def clip_outliers(series, lower_percentile=1, upper_percentile=99):
    """Clip outliers based on percentiles"""
    lower = series.quantile(lower_percentile / 100)
    upper = series.quantile(upper_percentile / 100)
    return series.clip(lower, upper)
