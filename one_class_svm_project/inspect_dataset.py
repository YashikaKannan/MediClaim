"""
Data Inspection Script
Inspect all CMS dataset files and create a comprehensive data dictionary
"""
import pandas as pd
import os
from pathlib import Path

data_dir = Path("d:/Downloads/Provider_Anomaly_Project/sythetic data")

# Define file mappings
file_groups = {
    'TRAIN': {
        'provider': 'Train-1542865627584.csv',
        'beneficiary': 'Train_Beneficiarydata-1542865627584.csv',
        'inpatient': 'Train_Inpatientdata-1542865627584.csv',
        'outpatient': 'Train_Outpatientdata-1542865627584.csv',
    },
    'TEST': {
        'provider': 'Test-1542969243754.csv',
        'beneficiary': 'Test_Beneficiarydata-1542969243754.csv',
        'inpatient': 'Test_Inpatientdata-1542969243754.csv',
        'outpatient': 'Test_Outpatientdata-1542969243754.csv',
    }
}

def inspect_file(filepath):
    """Load and inspect a single file"""
    print(f"\n{'='*80}")
    print(f"FILE: {filepath.name}")
    print(f"{'='*80}")
    
    try:
        df = pd.read_csv(filepath)
        
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"\nColumn Names & Data Types:")
        print("-" * 80)
        for col in df.columns:
            dtype = df[col].dtype
            non_null = df[col].notna().sum()
            null_count = df[col].isna().sum()
            print(f"  {col:40s} {str(dtype):15s} Non-Null: {non_null:8d} Null: {null_count:8d}")
        
        print(f"\nMissing Value Summary:")
        print("-" * 80)
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        for col, cnt in missing[missing > 0].items():
            print(f"  {col:40s}: {cnt:8d} ({missing_pct[col]:.2f}%)")
        if missing.sum() == 0:
            print("  No missing values")
        
        print(f"\nDuplicate Rows: {df.duplicated().sum()}")
        
        print(f"\nFirst 5 rows:")
        print("-" * 80)
        print(df.head())
        
        print(f"\nData Type Summary:")
        print(df.dtypes.value_counts())
        
        print(f"\nNumeric Columns Statistics:")
        print("-" * 80)
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            print(df[numeric_cols].describe())
        else:
            print("  No numeric columns")
        
        return df
        
    except Exception as e:
        print(f"ERROR loading {filepath.name}: {e}")
        return None

# Inspect all files
dfs = {}
for group_name, files in file_groups.items():
    print(f"\n\n{'#'*80}")
    print(f"# {group_name} DATASET")
    print(f"{'#'*80}")
    
    for file_type, filename in files.items():
        filepath = data_dir / filename
        if filepath.exists():
            dfs[f"{group_name}_{file_type}"] = inspect_file(filepath)
        else:
            print(f"\nERROR: File not found: {filepath}")

# Summary
print(f"\n\n{'#'*80}")
print(f"# SUMMARY")
print(f"{'#'*80}")

for key, df in dfs.items():
    if df is not None:
        print(f"{key:30s}: {df.shape[0]:8d} rows × {df.shape[1]:3d} columns")

# Check key relationships
print(f"\n\n{'#'*80}")
print(f"# KEY RELATIONSHIPS")
print(f"{'#'*80}")

# Check Provider column
print(f"\nProvider Column Check:")
for key, df in dfs.items():
    if df is not None:
        if 'Provider' in df.columns:
            print(f"  {key:30s}: Provider column exists [OK] Unique providers: {df['Provider'].nunique()}")
        else:
            print(f"  {key:30s}: Provider column NOT found [MISSING]")

# Check PotentialFraud column
print(f"\nPotentialFraud Column Check:")
for key, df in dfs.items():
    if df is not None:
        if 'PotentialFraud' in df.columns:
            fraud_counts = df['PotentialFraud'].value_counts()
            print(f"  {key:30s}: PotentialFraud exists [OK]")
            print(f"      {fraud_counts.to_dict()}")
        else:
            print(f"  {key:30s}: PotentialFraud NOT found [MISSING]")

# Check BeneID column
print(f"\nBeneID Column Check:")
for key, df in dfs.items():
    if df is not None:
        if 'BeneID' in df.columns:
            print(f"  {key:30s}: BeneID column exists [OK] Unique beneficiaries: {df['BeneID'].nunique()}")
        else:
            print(f"  {key:30s}: BeneID column NOT found [MISSING]")

# Check ClaimID
print(f"\nClaimID Column Check:")
for key, df in dfs.items():
    if df is not None:
        if 'ClaimID' in df.columns:
            print(f"  {key:30s}: ClaimID column exists [OK] Unique claims: {df['ClaimID'].nunique()}")
        else:
            print(f"  {key:30s}: ClaimID column NOT found [MISSING]")

print("\n" + "="*80)
print("Data inspection complete!")
print("="*80)
