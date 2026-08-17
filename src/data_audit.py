"""
Medicare Provider Anomaly Risk Engine - Data Audit
Comprehensive data quality assessment
"""

import pandas as pd
import numpy as np
import config
from src.utils import logger, setup_logging, print_section, save_csv

setup_logging(config.LOGGING_LEVEL)

class DataAudit:
    """Comprehensive data quality auditing"""
    
    def __init__(self):
        self.audit_results = {}
    
    def audit_dataset(self, df, dataset_name):
        """
        Perform comprehensive audit on dataset
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset to audit
        dataset_name : str
            Name of the dataset
        """
        print_section(f"Data Audit: {dataset_name}")
        
        audit = {
            "dataset_name": dataset_name,
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
        }
        
        # Missing values
        missing = df.isnull().sum()
        missing_pct = 100 * missing / len(df)
        audit["missing_values"] = missing.to_dict()
        audit["missing_percentage"] = missing_pct.to_dict()
        
        # Duplicates
        audit["duplicate_rows"] = df.duplicated().sum()
        
        # Unique values for key columns
        if "Provider" in df.columns:
            audit["unique_providers"] = df["Provider"].nunique()
        if "BeneID" in df.columns:
            audit["unique_beneficiaries"] = df["BeneID"].nunique()
        if "ClaimID" in df.columns:
            audit["unique_claims"] = df["ClaimID"].nunique()
        
        # Numerical statistics
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            audit["numerical_stats"] = {}
            for col in numerical_cols:
                if df[col].count() > 0:
                    audit["numerical_stats"][col] = {
                        "count": int(df[col].count()),
                        "mean": float(df[col].mean()),
                        "median": float(df[col].median()),
                        "std": float(df[col].std()),
                        "min": float(df[col].min()),
                        "max": float(df[col].max()),
                        "q25": float(df[col].quantile(0.25)),
                        "q75": float(df[col].quantile(0.75)),
                    }
                else:
                    audit["numerical_stats"][col] = {
                        "count": 0,
                        "mean": None,
                        "median": None,
                        "std": None,
                        "min": None,
                        "max": None,
                        "q25": None,
                        "q75": None,
                    }
        
        # Categorical statistics
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            audit["categorical_stats"] = {}
            for col in categorical_cols:
                audit["categorical_stats"][col] = {
                    "unique_values": int(df[col].nunique()),
                    "top_value": str(df[col].value_counts().index[0]) if len(df[col].value_counts()) > 0 else None,
                    "top_value_count": int(df[col].value_counts().iloc[0]) if len(df[col].value_counts()) > 0 else None,
                }
        
        self.audit_results[dataset_name] = audit
        self._print_audit(audit)
        
        return audit
    
    def _print_audit(self, audit):
        """Print audit results"""
        print(f"Rows: {audit['num_rows']}")
        print(f"Columns: {audit['num_columns']}")
        print(f"Memory: {audit['memory_mb']:.2f} MB")
        
        if audit.get('unique_providers'):
            print(f"Unique Providers: {audit['unique_providers']}")
        if audit.get('unique_beneficiaries'):
            print(f"Unique Beneficiaries: {audit['unique_beneficiaries']}")
        if audit.get('unique_claims'):
            print(f"Unique Claims: {audit['unique_claims']}")
        
        print(f"Duplicate Rows: {audit['duplicate_rows']}")
        
        # Missing values
        missing_cols = {k: v for k, v in audit['missing_values'].items() if v > 0}
        if missing_cols:
            print("\nMissing Values:")
            for col, count in missing_cols.items():
                pct = audit['missing_percentage'][col]
                print(f"  {col}: {count} ({pct:.1f}%)")
        
        # Numerical stats
        if audit.get('numerical_stats'):
            print("\nNumerical Columns:")
            for col, stats in audit['numerical_stats'].items():
                if stats['min'] is not None and stats['max'] is not None:
                    print(f"  {col}: min={stats['min']}, max={stats['max']}, mean={stats['mean']:.2f}, median={stats['median']:.2f}")

    
    def create_audit_report(self):
        """Create audit report dataframe"""
        report_data = []
        
        for dataset_name, audit in self.audit_results.items():
            for col in audit['columns']:
                row = {
                    'Dataset': dataset_name,
                    'Column': col,
                    'DataType': audit['dtypes'].get(col, 'Unknown'),
                    'NonNull': audit['num_rows'] - audit['missing_values'].get(col, 0),
                    'Null': audit['missing_values'].get(col, 0),
                    'NullPercentage': audit['missing_percentage'].get(col, 0),
                }
                
                if col in audit.get('numerical_stats', {}):
                    stats = audit['numerical_stats'][col]
                    row['Mean'] = stats['mean']
                    row['Median'] = stats['median']
                    row['Std'] = stats['std']
                    row['Min'] = stats['min']
                    row['Max'] = stats['max']
                
                report_data.append(row)
        
        return pd.DataFrame(report_data)


def run_data_audit(loader):
    """
    Run comprehensive data audit on all datasets
    
    Parameters:
    -----------
    loader : DataLoader
        Initialized data loader
    
    Returns:
    --------
    pd.DataFrame
        Audit report
    """
    print_section("COMPREHENSIVE DATA QUALITY AUDIT")
    
    audit = DataAudit()
    
    # Audit each dataset
    if loader.provider_df is not None:
        audit.audit_dataset(loader.provider_df, "Provider")
    
    if loader.beneficiary_df is not None:
        audit.audit_dataset(loader.beneficiary_df, "Beneficiary")
    
    if loader.inpatient_df is not None:
        audit.audit_dataset(loader.inpatient_df, "Inpatient Claims")
    
    if loader.outpatient_df is not None:
        audit.audit_dataset(loader.outpatient_df, "Outpatient Claims")
    
    # Generate report
    report = audit.create_audit_report()
    
    print_section("Audit Report Summary")
    print(f"Total audit entries: {len(report)}")
    
    # Save report
    save_csv(report, config.DATA_QUALITY_REPORT)
    
    return report
