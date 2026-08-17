"""
Medicare Provider Anomaly Risk Engine - Data Cleaning
Data cleaning and preprocessing
"""

import pandas as pd
import numpy as np
import config
from src.utils import logger, setup_logging, safe_date_parse, print_section

setup_logging(config.LOGGING_LEVEL)

class DataCleaner:
    """Data cleaning and preprocessing"""
    
    @staticmethod
    def clean_beneficiary_data(df):
        """Clean beneficiary data"""
        print_section("Cleaning Beneficiary Data")
        
        df = df.copy()
        initial_rows = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['BeneID'], keep='first')
        duplicates_removed = initial_rows - len(df)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate beneficiaries")
        
        # Parse dates
        logger.info("Parsing DOB and DOD")
        df['DOB'] = safe_date_parse(df['DOB'])
        df['DOD'] = safe_date_parse(df['DOD'])
        
        # Remove rows with missing critical columns
        critical_cols = ['BeneID']
        df = df.dropna(subset=critical_cols)
        
        # Ensure chronic condition columns are numeric (0/1)
        for col in config.CHRONIC_CONDITIONS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        logger.info(f"Beneficiary data cleaned: {len(df)} rows remaining")
        return df
    
    @staticmethod
    def clean_inpatient_claims(df):
        """Clean inpatient claims data"""
        print_section("Cleaning Inpatient Claims")
        
        df = df.copy()
        initial_rows = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['ClaimID'], keep='first')
        duplicates_removed = initial_rows - len(df)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate inpatient claims")
        
        # Parse dates
        logger.info("Parsing claim and admission dates")
        df['ClaimStartDt'] = safe_date_parse(df['ClaimStartDt'])
        df['ClaimEndDt'] = safe_date_parse(df['ClaimEndDt'])
        df['AdmissionDt'] = safe_date_parse(df['AdmissionDt'])
        df['DischargeDt'] = safe_date_parse(df['DischargeDt'])
        
        # Remove rows with missing Provider or BeneID
        df = df.dropna(subset=['Provider', 'BeneID', 'ClaimID'])
        
        # Convert reimbursement to numeric
        df['InscClaimAmtReimbursed'] = pd.to_numeric(df['InscClaimAmtReimbursed'], errors='coerce')
        df['DeductibleAmtPaid'] = pd.to_numeric(df['DeductibleAmtPaid'], errors='coerce')
        
        # Handle missing reimbursement (fill with 0)
        df['InscClaimAmtReimbursed'] = df['InscClaimAmtReimbursed'].fillna(0)
        df['DeductibleAmtPaid'] = df['DeductibleAmtPaid'].fillna(0)
        
        logger.info(f"Inpatient claims cleaned: {len(df)} rows remaining")
        return df
    
    @staticmethod
    def clean_outpatient_claims(df):
        """Clean outpatient claims data"""
        print_section("Cleaning Outpatient Claims")
        
        df = df.copy()
        initial_rows = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['ClaimID'], keep='first')
        duplicates_removed = initial_rows - len(df)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate outpatient claims")
        
        # Parse dates
        logger.info("Parsing claim dates")
        df['ClaimStartDt'] = safe_date_parse(df['ClaimStartDt'])
        df['ClaimEndDt'] = safe_date_parse(df['ClaimEndDt'])
        
        # Remove rows with missing Provider or BeneID
        df = df.dropna(subset=['Provider', 'BeneID', 'ClaimID'])
        
        # Convert reimbursement to numeric
        df['InscClaimAmtReimbursed'] = pd.to_numeric(df['InscClaimAmtReimbursed'], errors='coerce')
        df['DeductibleAmtPaid'] = pd.to_numeric(df['DeductibleAmtPaid'], errors='coerce')
        
        # Handle missing reimbursement (fill with 0)
        df['InscClaimAmtReimbursed'] = df['InscClaimAmtReimbursed'].fillna(0)
        df['DeductibleAmtPaid'] = df['DeductibleAmtPaid'].fillna(0)
        
        logger.info(f"Outpatient claims cleaned: {len(df)} rows remaining")
        return df
    
    @staticmethod
    def count_diagnoses(row):
        """Count non-null diagnosis codes in a row"""
        diag_cols = config.DIAGNOSIS_COLUMNS
        count = 0
        for col in diag_cols:
            if col in row and pd.notna(row[col]) and row[col] != '':
                count += 1
        return count
    
    @staticmethod
    def count_procedures(row):
        """Count non-null procedure codes in a row"""
        proc_cols = config.PROCEDURE_COLUMNS
        count = 0
        for col in proc_cols:
            if col in row and pd.notna(row[col]) and row[col] != '':
                count += 1
        return count
    
    @staticmethod
    def combine_claims(inpatient_df, outpatient_df):
        """
        Combine inpatient and outpatient claims for feature engineering
        
        Returns:
        --------
        tuple : (inpatient_df, outpatient_df) with added feature columns
        """
        logger.info("Adding diagnosis and procedure counts to claims")
        
        # Vectorized diagnosis counts
        diag_cols_ip = [c for c in config.DIAGNOSIS_COLUMNS if c in inpatient_df.columns]
        inpatient_df['diagnosis_count'] = (inpatient_df[diag_cols_ip].notna() & (inpatient_df[diag_cols_ip] != '')).sum(axis=1) if diag_cols_ip else 0
        
        diag_cols_op = [c for c in config.DIAGNOSIS_COLUMNS if c in outpatient_df.columns]
        outpatient_df['diagnosis_count'] = (outpatient_df[diag_cols_op].notna() & (outpatient_df[diag_cols_op] != '')).sum(axis=1) if diag_cols_op else 0
        
        # Vectorized procedure counts
        proc_cols_ip = [c for c in config.PROCEDURE_COLUMNS if c in inpatient_df.columns]
        inpatient_df['procedure_count'] = (inpatient_df[proc_cols_ip].notna() & (inpatient_df[proc_cols_ip] != '')).sum(axis=1) if proc_cols_ip else 0
        
        proc_cols_op = [c for c in config.PROCEDURE_COLUMNS if c in outpatient_df.columns]
        outpatient_df['procedure_count'] = (outpatient_df[proc_cols_op].notna() & (outpatient_df[proc_cols_op] != '')).sum(axis=1) if proc_cols_op else 0
        
        return inpatient_df, outpatient_df
