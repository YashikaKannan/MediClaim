"""
Provider-level feature aggregation from claim-level data
"""
import pandas as pd
import numpy as np
from datetime import datetime
from config import (
    PROVIDER_ID_COL, BENE_ID_COL, CLAIM_ID_COL,
    CLAIM_START_DT_COL, CLAIM_END_DT_COL, ADMISSION_DT_COL, DISCHARGE_DT_COL,
    REIMBURSEMENT_COL, DEDUCTIBLE_COL,
    ATTENDING_PHYSICIAN_COL, OPERATING_PHYSICIAN_COL, OTHER_PHYSICIAN_COL,
    DIAGNOSIS_COLS_PREFIX, PROCEDURE_COLS_PREFIX,
    get_logger
)
from utils import verify_dataframe

logger = get_logger(__name__)

def aggregate_outpatient_claims(df_op):
    """
    Aggregate outpatient claims to provider level
    
    Parameters:
    -----------
    df_op : pd.DataFrame
        Outpatient claims data
    
    Returns:
    --------
    pd.DataFrame
        Provider-level outpatient aggregates
    """
    logger.info("Aggregating outpatient claims to provider level...")
    
    df = df_op.copy()
    
    # Group by provider
    agg_dict = {
        CLAIM_ID_COL: 'count',  # Number of claims
        REIMBURSEMENT_COL: ['sum', 'mean', 'median', 'max', 'std'],
        DEDUCTIBLE_COL: ['sum', 'mean'],
        BENE_ID_COL: 'nunique',  # Unique beneficiaries
        ATTENDING_PHYSICIAN_COL: 'nunique',  # Unique physicians
    }
    
    # Add diagnosis and procedure code aggregations
    diagnosis_cols = [col for col in df.columns if col.startswith(DIAGNOSIS_COLS_PREFIX)]
    for col in diagnosis_cols:
        agg_dict[col] = 'nunique'
    
    procedure_cols = [col for col in df.columns if col.startswith(PROCEDURE_COLS_PREFIX)]
    for col in procedure_cols:
        agg_dict[col] = 'nunique'
    
    # Aggregate
    agg_op = df.groupby(PROVIDER_ID_COL).agg(agg_dict)
    
    # Flatten column names
    agg_op.columns = ['_'.join(col).strip('_') for col in agg_op.columns.values]
    
    # Rename columns for clarity
    agg_op = agg_op.rename(columns={
        f'{CLAIM_ID_COL}_count': 'OP_Claim_Count',
        f'{REIMBURSEMENT_COL}_sum': 'OP_Total_Reimbursement',
        f'{REIMBURSEMENT_COL}_mean': 'OP_Avg_Reimbursement',
        f'{REIMBURSEMENT_COL}_median': 'OP_Median_Reimbursement',
        f'{REIMBURSEMENT_COL}_max': 'OP_Max_Reimbursement',
        f'{REIMBURSEMENT_COL}_std': 'OP_Std_Reimbursement',
        f'{DEDUCTIBLE_COL}_sum': 'OP_Total_Deductible',
        f'{DEDUCTIBLE_COL}_mean': 'OP_Avg_Deductible',
        f'{BENE_ID_COL}_nunique': 'OP_Unique_Beneficiaries',
        f'{ATTENDING_PHYSICIAN_COL}_nunique': 'OP_Unique_Physicians',
    })
    
    # Count unique diagnoses and procedures
    if diagnosis_cols:
        agg_op['OP_Unique_Diagnoses'] = df.groupby(PROVIDER_ID_COL)[diagnosis_cols].apply(
            lambda x: x.stack().nunique()
        )
    
    if procedure_cols:
        agg_op['OP_Unique_Procedures'] = df.groupby(PROVIDER_ID_COL)[procedure_cols].apply(
            lambda x: x.stack().nunique()
        )
    
    # Fill NaN std with 0 (single claim providers have no std)
    if 'OP_Std_Reimbursement' in agg_op.columns:
        agg_op['OP_Std_Reimbursement'] = agg_op['OP_Std_Reimbursement'].fillna(0)
    
    logger.info(f"Outpatient aggregation: {len(agg_op)} providers, {agg_op.shape[1]} features")
    
    return agg_op.reset_index()

def aggregate_inpatient_claims(df_ip):
    """
    Aggregate inpatient claims to provider level
    
    Parameters:
    -----------
    df_ip : pd.DataFrame
        Inpatient claims data
    
    Returns:
    --------
    pd.DataFrame
        Provider-level inpatient aggregates
    """
    logger.info("Aggregating inpatient claims to provider level...")
    
    df = df_ip.copy()
    
    # Calculate length of stay if admission/discharge dates are available
    if ADMISSION_DT_COL in df.columns and DISCHARGE_DT_COL in df.columns:
        df['LengthOfStay_Days'] = (
            pd.to_datetime(df[DISCHARGE_DT_COL], errors='coerce') -
            pd.to_datetime(df[ADMISSION_DT_COL], errors='coerce')
        ).dt.days
        df['LengthOfStay_Days'] = df['LengthOfStay_Days'].clip(lower=0)  # No negative LOS
    else:
        logger.warning("Could not calculate length of stay - missing date columns")
    
    # Group by provider
    agg_dict = {
        CLAIM_ID_COL: 'count',  # Number of claims
        REIMBURSEMENT_COL: ['sum', 'mean', 'median', 'max', 'std'],
        DEDUCTIBLE_COL: ['sum', 'mean'],
        BENE_ID_COL: 'nunique',  # Unique beneficiaries
        ATTENDING_PHYSICIAN_COL: 'nunique',  # Unique physicians
    }
    
    # Add length of stay metrics if available
    if 'LengthOfStay_Days' in df.columns:
        agg_dict['LengthOfStay_Days'] = ['mean', 'max', 'median']
    
    # Add diagnosis and procedure code aggregations
    diagnosis_cols = [col for col in df.columns if col.startswith(DIAGNOSIS_COLS_PREFIX)]
    for col in diagnosis_cols:
        agg_dict[col] = 'nunique'
    
    procedure_cols = [col for col in df.columns if col.startswith(PROCEDURE_COLS_PREFIX)]
    for col in procedure_cols:
        agg_dict[col] = 'nunique'
    
    # Aggregate
    agg_ip = df.groupby(PROVIDER_ID_COL).agg(agg_dict)
    
    # Flatten column names
    agg_ip.columns = ['_'.join(col).strip('_') for col in agg_ip.columns.values]
    
    # Rename columns for clarity
    rename_dict = {
        f'{CLAIM_ID_COL}_count': 'IP_Claim_Count',
        f'{REIMBURSEMENT_COL}_sum': 'IP_Total_Reimbursement',
        f'{REIMBURSEMENT_COL}_mean': 'IP_Avg_Reimbursement',
        f'{REIMBURSEMENT_COL}_median': 'IP_Median_Reimbursement',
        f'{REIMBURSEMENT_COL}_max': 'IP_Max_Reimbursement',
        f'{REIMBURSEMENT_COL}_std': 'IP_Std_Reimbursement',
        f'{DEDUCTIBLE_COL}_sum': 'IP_Total_Deductible',
        f'{DEDUCTIBLE_COL}_mean': 'IP_Avg_Deductible',
        f'{BENE_ID_COL}_nunique': 'IP_Unique_Beneficiaries',
        f'{ATTENDING_PHYSICIAN_COL}_nunique': 'IP_Unique_Physicians',
    }
    
    if 'LengthOfStay_Days_mean' in agg_ip.columns:
        rename_dict['LengthOfStay_Days_mean'] = 'IP_Avg_Stay_Days'
    if 'LengthOfStay_Days_max' in agg_ip.columns:
        rename_dict['LengthOfStay_Days_max'] = 'IP_Max_Stay_Days'
    if 'LengthOfStay_Days_median' in agg_ip.columns:
        rename_dict['LengthOfStay_Days_median'] = 'IP_Median_Stay_Days'
    
    agg_ip = agg_ip.rename(columns=rename_dict)
    
    # Count unique diagnoses and procedures
    if diagnosis_cols:
        agg_ip['IP_Unique_Diagnoses'] = df.groupby(PROVIDER_ID_COL)[diagnosis_cols].apply(
            lambda x: x.stack().nunique()
        )
    
    if procedure_cols:
        agg_ip['IP_Unique_Procedures'] = df.groupby(PROVIDER_ID_COL)[procedure_cols].apply(
            lambda x: x.stack().nunique()
        )
    
    # Fill NaN std with 0
    if 'IP_Std_Reimbursement' in agg_ip.columns:
        agg_ip['IP_Std_Reimbursement'] = agg_ip['IP_Std_Reimbursement'].fillna(0)
    
    logger.info(f"Inpatient aggregation: {len(agg_ip)} providers, {agg_ip.shape[1]} features")
    
    return agg_ip.reset_index()

def merge_inpatient_outpatient(agg_ip, agg_op):
    """
    Merge inpatient and outpatient aggregates
    
    Parameters:
    -----------
    agg_ip : pd.DataFrame
        Inpatient aggregates
    agg_op : pd.DataFrame
        Outpatient aggregates
    
    Returns:
    --------
    pd.DataFrame
        Merged provider-level data
    """
    logger.info("Merging inpatient and outpatient aggregates...")
    
    # Full outer join to include all providers
    merged = pd.merge(
        agg_ip, agg_op,
        on=PROVIDER_ID_COL,
        how='outer'
    )
    
    logger.info(f"Merged: {len(merged)} providers")
    
    return merged

def create_provider_features(merged_df):
    """
    Create derived provider-level features
    
    Parameters:
    -----------
    merged_df : pd.DataFrame
        Merged inpatient/outpatient data
    
    Returns:
    --------
    pd.DataFrame
        Provider features with derived metrics
    """
    logger.info("Creating derived provider features...")
    
    df = merged_df.copy()
    
    # Fill missing values for providers with only IP or only OP
    ip_cols = [col for col in df.columns if col.startswith('IP_')]
    op_cols = [col for col in df.columns if col.startswith('OP_')]
    
    for col in ip_cols:
        df[col] = df[col].fillna(0)
    for col in op_cols:
        df[col] = df[col].fillna(0)
    
    # Combined metrics
    df['Total_Claims'] = df.get('IP_Claim_Count', 0) + df.get('OP_Claim_Count', 0)
    df['Total_Reimbursement'] = df.get('IP_Total_Reimbursement', 0) + df.get('OP_Total_Reimbursement', 0)
    df['Total_Deductible'] = df.get('IP_Total_Deductible', 0) + df.get('OP_Total_Deductible', 0)
    
    # Unique counts (taking max since a beneficiary could have both IP and OP)
    df['Total_Unique_Beneficiaries'] = np.maximum(
        df.get('IP_Unique_Beneficiaries', 0),
        df.get('OP_Unique_Beneficiaries', 0)
    )
    
    df['Total_Unique_Physicians'] = (
        df.get('IP_Unique_Physicians', 0) + df.get('OP_Unique_Physicians', 0)
    )
    
    # Ratios (avoid division by zero)
    df['Claims_Per_Beneficiary'] = np.where(
        df['Total_Unique_Beneficiaries'] > 0,
        df['Total_Claims'] / df['Total_Unique_Beneficiaries'],
        0
    )
    
    df['Reimbursement_Per_Beneficiary'] = np.where(
        df['Total_Unique_Beneficiaries'] > 0,
        df['Total_Reimbursement'] / df['Total_Unique_Beneficiaries'],
        0
    )
    
    df['Reimbursement_Per_Claim'] = np.where(
        df['Total_Claims'] > 0,
        df['Total_Reimbursement'] / df['Total_Claims'],
        0
    )
    
    df['Deductible_Per_Claim'] = np.where(
        df['Total_Claims'] > 0,
        df['Total_Deductible'] / df['Total_Claims'],
        0
    )
    
    # Claim type ratios
    df['IP_OP_Claim_Ratio'] = np.where(
        df.get('OP_Claim_Count', 0) > 0,
        df.get('IP_Claim_Count', 0) / df.get('OP_Claim_Count', 0),
        0
    )
    
    df['IP_OP_Reimbursement_Ratio'] = np.where(
        df.get('OP_Total_Reimbursement', 0) > 0,
        df.get('IP_Total_Reimbursement', 0) / df.get('OP_Total_Reimbursement', 0),
        0
    )
    
    # Average claim cost
    df['Average_Claim_Cost'] = np.where(
        df['Total_Claims'] > 0,
        df['Total_Reimbursement'] / df['Total_Claims'],
        0
    )
    
    # Beneficiary concentration (Gini-like metric)
    # For now, use a proxy: variance in claims per beneficiary if we had that info
    # Simple proxy: claim count / beneficiary count (higher = more concentrated)
    df['Beneficiary_Concentration'] = np.where(
        df['Total_Unique_Beneficiaries'] > 0,
        df['Total_Claims'] / df['Total_Unique_Beneficiaries'],
        0
    )
    
    logger.info(f"Created derived features. Total columns: {df.shape[1]}")
    
    # Replace inf values with 0
    df = df.replace([np.inf, -np.inf], 0)
    
    return df

def create_provider_feature_table(df_inpatient, df_outpatient, df_provider_labels=None):
    """
    Create complete provider-level feature table
    
    Parameters:
    -----------
    df_inpatient : pd.DataFrame
        Inpatient claims data
    df_outpatient : pd.DataFrame
        Outpatient claims data
    df_provider_labels : pd.DataFrame, optional
        Provider labels (for training data)
    
    Returns:
    --------
    pd.DataFrame
        Provider-level feature table
    """
    # Aggregate both claim types
    agg_op = aggregate_outpatient_claims(df_outpatient)
    agg_ip = aggregate_inpatient_claims(df_inpatient)
    
    # Merge
    merged = merge_inpatient_outpatient(agg_ip, agg_op)
    
    # Create derived features
    provider_features = create_provider_features(merged)
    
    # Merge with labels if provided
    if df_provider_labels is not None:
        logger.info("Merging with provider labels...")
        provider_features = pd.merge(
            provider_features,
            df_provider_labels,
            on=PROVIDER_ID_COL,
            how='left'
        )
        logger.info(f"After merge with labels: {len(provider_features)} providers")
    
    # Validate
    verify_dataframe(provider_features, "Provider Features",
                     expected_cols=[PROVIDER_ID_COL], min_rows=1)
    
    # Check for duplicates
    if provider_features.duplicated(subset=[PROVIDER_ID_COL]).any():
        raise ValueError("Duplicate providers in feature table!")
    
    logger.info(f"Provider feature table: {provider_features.shape[0]} providers × {provider_features.shape[1]} columns")
    
    return provider_features
