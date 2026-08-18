"""
Data loading module for CMS Healthcare Provider dataset
"""
import pandas as pd
from pathlib import Path
from config import (
    RAW_DATA_DIR,
    TRAIN_PROVIDER_FILE, TRAIN_BENEFICIARY_FILE, TRAIN_INPATIENT_FILE, TRAIN_OUTPATIENT_FILE,
    TEST_PROVIDER_FILE, TEST_BENEFICIARY_FILE, TEST_INPATIENT_FILE, TEST_OUTPATIENT_FILE,
    PROVIDER_ID_COL, FRAUD_LABEL_COL, BENE_ID_COL, CLAIM_ID_COL,
    REQUIRED_PROVIDER_COLS, REQUIRED_INPATIENT_COLS, REQUIRED_OUTPATIENT_COLS, REQUIRED_BENEFICIARY_COLS,
    get_logger
)
from utils import verify_dataframe, check_duplicates, log_dataset_summary

logger = get_logger(__name__)

def load_csv(filepath, name=None):
    """
    Load CSV file with error handling
    
    Parameters:
    -----------
    filepath : Path or str
        Path to CSV file
    name : str, optional
        Name for logging
    
    Returns:
    --------
    pd.DataFrame
        Loaded dataframe
    
    Raises:
    -------
    FileNotFoundError : If file doesn't exist
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    display_name = name or filepath.name
    logger.info(f"Loading {display_name}...")
    
    df = pd.read_csv(filepath, low_memory=False)
    log_dataset_summary(df, display_name)
    
    return df

def load_training_data():
    """
    Load all training datasets
    
    Returns:
    --------
    dict with keys: provider, beneficiary, inpatient, outpatient
        All loaded dataframes
    """
    logger.info("="*80)
    logger.info("LOADING TRAINING DATA")
    logger.info("="*80)
    
    train_data = {}
    
    # Load provider file (with labels)
    train_data['provider'] = load_csv(
        RAW_DATA_DIR / TRAIN_PROVIDER_FILE,
        "Train Provider File (with PotentialFraud labels)"
    )
    verify_dataframe(train_data['provider'], "Train Provider", 
                     expected_cols=REQUIRED_PROVIDER_COLS, min_rows=1)
    
    # Load beneficiary file
    train_data['beneficiary'] = load_csv(
        RAW_DATA_DIR / TRAIN_BENEFICIARY_FILE,
        "Train Beneficiary File"
    )
    verify_dataframe(train_data['beneficiary'], "Train Beneficiary",
                     expected_cols=REQUIRED_BENEFICIARY_COLS, min_rows=1)
    
    # Load inpatient file
    train_data['inpatient'] = load_csv(
        RAW_DATA_DIR / TRAIN_INPATIENT_FILE,
        "Train Inpatient Claims"
    )
    verify_dataframe(train_data['inpatient'], "Train Inpatient",
                     expected_cols=REQUIRED_INPATIENT_COLS, min_rows=1)
    
    # Load outpatient file
    train_data['outpatient'] = load_csv(
        RAW_DATA_DIR / TRAIN_OUTPATIENT_FILE,
        "Train Outpatient Claims"
    )
    verify_dataframe(train_data['outpatient'], "Train Outpatient",
                     expected_cols=REQUIRED_OUTPATIENT_COLS, min_rows=1)
    
    # Validate relationships
    logger.info("\nValidating data relationships...")
    
    # Check PotentialFraud values
    fraud_values = train_data['provider'][FRAUD_LABEL_COL].unique()
    logger.info(f"PotentialFraud values: {fraud_values}")
    
    fraud_counts = train_data['provider'][FRAUD_LABEL_COL].value_counts()
    logger.info(f"Fraud distribution:\n{fraud_counts}")
    
    # Check provider coverage
    providers_in_claims = set(train_data['inpatient'][PROVIDER_ID_COL].unique()) | \
                         set(train_data['outpatient'][PROVIDER_ID_COL].unique())
    providers_in_labels = set(train_data['provider'][PROVIDER_ID_COL].unique())
    
    missing_providers = providers_in_claims - providers_in_labels
    if missing_providers:
        logger.warning(f"Providers in claims but not in labels: {len(missing_providers)}")
    
    logger.info(f"Providers in labels: {len(providers_in_labels)}")
    logger.info(f"Providers in inpatient claims: {train_data['inpatient'][PROVIDER_ID_COL].nunique()}")
    logger.info(f"Providers in outpatient claims: {train_data['outpatient'][PROVIDER_ID_COL].nunique()}")
    logger.info(f"Providers in either claims: {len(providers_in_claims)}")
    
    return train_data

def load_test_data():
    """
    Load all test datasets (no labels available)
    
    Returns:
    --------
    dict with keys: provider, beneficiary, inpatient, outpatient
        All loaded dataframes
    """
    logger.info("\n" + "="*80)
    logger.info("LOADING TEST DATA")
    logger.info("="*80)
    
    test_data = {}
    
    # Load provider file (no labels in test)
    test_data['provider'] = load_csv(
        RAW_DATA_DIR / TEST_PROVIDER_FILE,
        "Test Provider File (no labels)"
    )
    verify_dataframe(test_data['provider'], "Test Provider",
                     expected_cols=[PROVIDER_ID_COL], min_rows=1)
    
    # Load beneficiary file
    test_data['beneficiary'] = load_csv(
        RAW_DATA_DIR / TEST_BENEFICIARY_FILE,
        "Test Beneficiary File"
    )
    verify_dataframe(test_data['beneficiary'], "Test Beneficiary",
                     expected_cols=REQUIRED_BENEFICIARY_COLS, min_rows=1)
    
    # Load inpatient file
    test_data['inpatient'] = load_csv(
        RAW_DATA_DIR / TEST_INPATIENT_FILE,
        "Test Inpatient Claims"
    )
    verify_dataframe(test_data['inpatient'], "Test Inpatient",
                     expected_cols=REQUIRED_INPATIENT_COLS, min_rows=1)
    
    # Load outpatient file
    test_data['outpatient'] = load_csv(
        RAW_DATA_DIR / TEST_OUTPATIENT_FILE,
        "Test Outpatient Claims"
    )
    verify_dataframe(test_data['outpatient'], "Test Outpatient",
                     expected_cols=REQUIRED_OUTPATIENT_COLS, min_rows=1)
    
    logger.info(f"\nTest providers: {test_data['provider'][PROVIDER_ID_COL].nunique()}")
    logger.info(f"Test inpatient providers: {test_data['inpatient'][PROVIDER_ID_COL].nunique()}")
    logger.info(f"Test outpatient providers: {test_data['outpatient'][PROVIDER_ID_COL].nunique()}")
    
    return test_data

def get_all_data():
    """
    Load both training and test data
    
    Returns:
    --------
    tuple
        (train_data dict, test_data dict)
    """
    train_data = load_training_data()
    test_data = load_test_data()
    
    return train_data, test_data
