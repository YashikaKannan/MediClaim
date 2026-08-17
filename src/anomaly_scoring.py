"""
Medicare Provider Anomaly Risk Engine - Anomaly Scoring
Convert anomaly scores to risk scores and risk levels
"""

import pandas as pd
import numpy as np
import config
from src.utils import logger, setup_logging, print_section

setup_logging(config.LOGGING_LEVEL)

class AnomalyScorer:
    """Convert anomaly scores to risk scores and levels"""
    
    @staticmethod
    def generate_anomaly_scores(df_providers, predictions, anomaly_scores):
        """
        Generate comprehensive anomaly results
        
        Parameters:
        -----------
        df_providers : pd.DataFrame
            Provider dataframe with Provider ID
        predictions : np.ndarray
            Anomaly predictions (0=normal, 1=anomaly)
        anomaly_scores : np.ndarray
            Raw anomaly scores
        
        Returns:
        --------
        pd.DataFrame
            Results dataframe with predictions and scores
        """
        print_section("Generating Anomaly Scores")
        
        results = pd.DataFrame({
            'Provider': df_providers['Provider'].values,
            'anomaly_prediction': predictions,
            'anomaly_score_raw': anomaly_scores
        })
        
        logger.info(f"Generated anomaly scores for {len(results)} providers")
        print(f"Total providers scored: {len(results)}")
        print(f"Anomalies detected: {predictions.sum()} ({predictions.sum()/len(results)*100:.1f}%)")
        print(f"Normal providers: {(predictions == 0).sum()} ({(predictions == 0).sum()/len(results)*100:.1f}%)")
        
        return results
    
    @staticmethod
    def normalize_anomaly_scores(anomaly_scores):
        """
        Normalize anomaly scores to 0-100 risk score
        
        Parameters:
        -----------
        anomaly_scores : np.ndarray
            Raw anomaly scores
        
        Returns:
        --------
        np.ndarray
            Risk scores (0-100)
        """
        min_score = np.nanmin(anomaly_scores)
        max_score = np.nanmax(anomaly_scores)
        
        if max_score > min_score:
            risk_scores = (anomaly_scores - min_score) / (max_score - min_score) * 100
        else:
            risk_scores = np.zeros_like(anomaly_scores)
        
        return risk_scores
    
    @staticmethod
    def assign_risk_levels(risk_scores, thresholds=None):
        """
        Assign risk levels based on percentile thresholds
        
        Parameters:
        -----------
        risk_scores : np.ndarray
            Risk scores (0-100)
        thresholds : dict
            Risk level thresholds {level: (lower, upper)}
        
        Returns:
        --------
        np.ndarray
            Risk levels (LOW, MEDIUM, HIGH, CRITICAL)
        """
        if thresholds is None:
            thresholds = config.RISK_LEVEL_THRESHOLDS
        
        percentiles = np.percentile(risk_scores, [0, 90, 95, 99, 100])
        
        logger.info(f"Risk Score Percentiles: 0%={percentiles[0]:.2f}, 90%={percentiles[1]:.2f}, "
                   f"95%={percentiles[2]:.2f}, 99%={percentiles[3]:.2f}, 100%={percentiles[4]:.2f}")
        
        risk_levels = np.full(len(risk_scores), 'LOW', dtype=object)
        
        # Use percentile-based thresholds
        p90 = np.percentile(risk_scores, 90)
        p95 = np.percentile(risk_scores, 95)
        p99 = np.percentile(risk_scores, 99)
        
        risk_levels[risk_scores >= p99] = 'CRITICAL'
        risk_levels[(risk_scores >= p95) & (risk_scores < p99)] = 'HIGH'
        risk_levels[(risk_scores >= p90) & (risk_scores < p95)] = 'MEDIUM'
        risk_levels[risk_scores < p90] = 'LOW'
        
        return risk_levels
    
    @staticmethod
    def create_anomaly_report(df_providers, predictions, anomaly_scores, features_df=None):
        """
        Create comprehensive anomaly report
        
        Parameters:
        -----------
        df_providers : pd.DataFrame
            Provider data
        predictions : np.ndarray
            Predictions
        anomaly_scores : np.ndarray
            Anomaly scores
        features_df : pd.DataFrame
            Feature dataframe (optional)
        
        Returns:
        --------
        pd.DataFrame
            Comprehensive anomaly report
        """
        print_section("Creating Anomaly Report")
        
        # Generate scores
        results = AnomalyScorer.generate_anomaly_scores(df_providers, predictions, anomaly_scores)
        
        # Normalize to risk scores
        results['risk_score'] = AnomalyScorer.normalize_anomaly_scores(anomaly_scores)
        
        # Assign risk levels
        results['risk_level'] = AnomalyScorer.assign_risk_levels(results['risk_score'].values)
        
        # Add features if provided
        if features_df is not None:
            # Join with feature dataframe to add important behavioral features
            important_features = [
                'total_claim_count',
                'total_reimbursement',
                'unique_beneficiary_count',
                'avg_reimbursement',
                'claims_per_beneficiary',
                'inpatient_claim_count',
                'outpatient_claim_count',
                'avg_beneficiary_age',
                'high_cost_claim_pct',
            ]
            
            # Only include features that exist
            available_features = [f for f in important_features if f in features_df.columns]
            
            features_to_add = features_df[['Provider'] + available_features].copy()
            results = results.merge(features_to_add, on='Provider', how='left')
        
        logger.info(f"Anomaly report created with {len(results)} providers")
        
        # Print summary
        print(f"\nRisk Level Distribution:")
        for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            count = (results['risk_level'] == level).sum()
            pct = count / len(results) * 100
            print(f"  {level:.<20} {count:>6} ({pct:>5.1f}%)")
        
        return results
