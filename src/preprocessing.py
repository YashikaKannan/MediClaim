"""
Medicare Provider Anomaly Risk Engine - Preprocessing
Preprocessing pipeline for model training
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import config
from src.utils import logger, setup_logging, print_section

setup_logging(config.LOGGING_LEVEL)

class PreprocessingPipeline:
    """Robust preprocessing pipeline for anomaly detection"""
    
    def __init__(self):
        self.preprocessing_pipeline = None
        self.fitted = False
        self.feature_names = None
        self.excluded_features = [
            'Provider', 'BeneID', 'ClaimID', 'PotentialFraud',
            'DOB', 'DOD', 'ClaimStartDt', 'ClaimEndDt', 'AdmissionDt', 'DischargeDt'
        ]
    
    def prepare_features_for_training(self, df):
        """
        Prepare features for Isolation Forest training
        
        Parameters:
        -----------
        df : pd.DataFrame
            Provider feature dataframe
        
        Returns:
        --------
        tuple : (X, feature_names)
            Processed features and feature names
        """
        print_section("Feature Preparation and Quality Control")
        
        X = df.copy()
        
        # Remove non-feature columns
        logger.info(f"Features before filtering: {len(X.columns)}")
        print(f"Features before filtering: {len(X.columns)}")
        
        cols_to_remove = [col for col in self.excluded_features if col in X.columns]
        if cols_to_remove:
            logger.info(f"Removing columns: {cols_to_remove}")
            X = X.drop(columns=cols_to_remove)
        
        # Remove zero-variance features
        zero_var_cols = X.columns[X.var() == 0]
        if len(zero_var_cols) > 0:
            logger.info(f"Removing zero-variance features: {list(zero_var_cols)}")
            X = X.drop(columns=zero_var_cols)
        
        # Remove constant columns
        constant_cols = X.columns[(X.nunique() <= 1)]
        if len(constant_cols) > 0:
            logger.info(f"Removing constant columns: {list(constant_cols)}")
            X = X.drop(columns=constant_cols)
        
        # Remove columns with all NaN
        all_nan_cols = X.columns[X.isnull().all()]
        if len(all_nan_cols) > 0:
            logger.info(f"Removing all-NaN columns: {list(all_nan_cols)}")
            X = X.drop(columns=all_nan_cols)
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        
        logger.info(f"Features after filtering: {len(X.columns)}")
        print(f"Features after filtering: {len(X.columns)}")
        
        print("\nRemaining Features:")
        for i, col in enumerate(X.columns, 1):
            print(f"  {i}. {col}")
        
        self.feature_names = list(X.columns)
        
        return X, self.feature_names
    
    def fit_preprocessing(self, X):
        """
        Fit preprocessing pipeline on training data
        
        Parameters:
        -----------
        X : pd.DataFrame
            Training features
        """
        print_section("Fitting Preprocessing Pipeline")
        
        # Create pipeline
        self.preprocessing_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        
        # Fit on training data
        logger.info("Fitting imputer and scaler on training data")
        self.preprocessing_pipeline.fit(X)
        
        self.fitted = True
        logger.info("Preprocessing pipeline fitted successfully")
    
    def transform(self, X):
        """
        Transform features using fitted pipeline
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features to transform
        
        Returns:
        --------
        np.ndarray
            Transformed features
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        # Ensure column order matches training
        X_ordered = X[self.feature_names].copy()
        
        # Transform
        X_transformed = self.preprocessing_pipeline.transform(X_ordered)
        
        return X_transformed
    
    def fit_transform(self, X):
        """
        Fit pipeline and transform data
        
        Parameters:
        -----------
        X : pd.DataFrame
            Training features
        
        Returns:
        --------
        np.ndarray
            Transformed training data
        """
        self.fit_preprocessing(X)
        return self.transform(X)
    
    def get_pipeline(self):
        """Get the preprocessing pipeline"""
        if not self.fitted:
            raise ValueError("Pipeline must be fitted first")
        return self.preprocessing_pipeline
    
    def get_feature_names(self):
        """Get feature names after preprocessing"""
        return self.feature_names


def create_preprocessing_pipeline():
    """Create and return a preprocessing pipeline instance"""
    return PreprocessingPipeline()
