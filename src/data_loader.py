"""
Medicare Provider Anomaly Risk Engine - Data Loader
Loads and manages CMS Medicare datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
import config
from src.utils import logger, setup_logging, print_section, print_dataframe_info

setup_logging(config.LOGGING_LEVEL)

class DataLoader:
    """Load and manage CMS Medicare datasets"""
    
    def __init__(self, use_train=True):
        """
        Initialize DataLoader
        
        Parameters:
        -----------
        use_train : bool
            If True, load training data; if False, load test data
        """
        self.use_train = use_train
        self.provider_file = config.TRAIN_PROVIDER_FILE if use_train else config.TEST_PROVIDER_FILE
        self.beneficiary_file = config.TRAIN_BENEFICIARY_FILE if use_train else config.TEST_BENEFICIARY_FILE
        self.inpatient_file = config.TRAIN_INPATIENT_FILE if use_train else config.TEST_INPATIENT_FILE
        self.outpatient_file = config.TRAIN_OUTPATIENT_FILE if use_train else config.TEST_OUTPATIENT_FILE
        
        self.provider_df = None
        self.beneficiary_df = None
        self.inpatient_df = None
        self.outpatient_df = None
        
        dataset_type = "TRAINING" if use_train else "TEST"
        logger.info(f"Initialized DataLoader for {dataset_type} dataset")
    
    def load_all(self):
        """Load all datasets"""
        dataset_type = "TRAINING" if self.use_train else "TEST"
        print_section(f"Loading {dataset_type} Datasets")
        
        self.load_provider()
        self.load_beneficiary()
        self.load_inpatient()
        self.load_outpatient()
        
        logger.info(f"Completed loading {dataset_type} datasets")
    
    def load_provider(self):
        """Load provider data"""
        logger.info(f"Loading provider data from {self.provider_file}")
        self.provider_df = pd.read_csv(self.provider_file)
        print_dataframe_info(self.provider_df, "Provider Data")
        return self.provider_df
    
    def load_beneficiary(self):
        """Load beneficiary data"""
        logger.info(f"Loading beneficiary data from {self.beneficiary_file}")
        self.beneficiary_df = pd.read_csv(self.beneficiary_file)
        print_dataframe_info(self.beneficiary_df, "Beneficiary Data")
        return self.beneficiary_df
    
    def load_inpatient(self):
        """Load inpatient claims"""
        logger.info(f"Loading inpatient data from {self.inpatient_file}")
        self.inpatient_df = pd.read_csv(self.inpatient_file)
        print_dataframe_info(self.inpatient_df, "Inpatient Claims Data")
        return self.inpatient_df
    
    def load_outpatient(self):
        """Load outpatient claims"""
        logger.info(f"Loading outpatient data from {self.outpatient_file}")
        self.outpatient_df = pd.read_csv(self.outpatient_file)
        print_dataframe_info(self.outpatient_df, "Outpatient Claims Data")
        return self.outpatient_df
    
    def get_inpatient_claims(self):
        """Return inpatient claims dataframe"""
        if self.inpatient_df is None:
            self.load_inpatient()
        return self.inpatient_df.copy()
    
    def get_outpatient_claims(self):
        """Return outpatient claims dataframe"""
        if self.outpatient_df is None:
            self.load_outpatient()
        return self.outpatient_df.copy()
    
    def get_beneficiary_data(self):
        """Return beneficiary dataframe"""
        if self.beneficiary_df is None:
            self.load_beneficiary()
        return self.beneficiary_df.copy()
    
    def get_provider_labels(self):
        """Return provider labels (for evaluation only)"""
        if self.provider_df is None:
            self.load_provider()
        return self.provider_df.copy()
    
    def detect_columns(self):
        """Detect and report column names in datasets"""
        print_section("Column Detection")
        
        results = {
            "provider_columns": list(self.provider_df.columns) if self.provider_df is not None else [],
            "beneficiary_columns": list(self.beneficiary_df.columns) if self.beneficiary_df is not None else [],
            "inpatient_columns": list(self.inpatient_df.columns) if self.inpatient_df is not None else [],
            "outpatient_columns": list(self.outpatient_df.columns) if self.outpatient_df is not None else [],
        }
        
        for dataset_type, columns in results.items():
            print(f"\n{dataset_type.upper()}:")
            for col in columns:
                print(f"  - {col}")
        
        return results


def load_training_data():
    """Convenience function to load training data"""
    loader = DataLoader(use_train=True)
    loader.load_all()
    return loader


def load_test_data():
    """Convenience function to load test data"""
    loader = DataLoader(use_train=False)
    loader.load_all()
    return loader
