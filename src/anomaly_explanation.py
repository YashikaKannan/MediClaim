"""
Medicare Provider Anomaly Risk Engine - Anomaly Explanation
Explain why providers are flagged as anomalies
"""

import pandas as pd
import numpy as np
import config
from src.utils import logger, setup_logging, print_section, percentile_in_group

setup_logging(config.LOGGING_LEVEL)

class AnomalyExplainer:
    """Explain anomalous provider behavior"""
    
    @staticmethod
    def explain_provider(provider_id, provider_features, all_providers_features, 
                        top_n_features=5):
        """
        Explain why a provider is anomalous
        
        Parameters:
        -----------
        provider_id : str
            Provider ID
        provider_features : dict
            Feature values for the provider
        all_providers_features : pd.DataFrame
            All provider features
        top_n_features : int
            Number of top unusual features to explain
        
        Returns:
        --------
        dict
            Explanation with unusual features
        """
        
        explanation = {
            'provider': provider_id,
            'unusual_features': []
        }
        
        # Calculate deviation metrics for each feature
        feature_deviations = {}
        
        for feature, value in provider_features.items():
            if feature == 'Provider' or pd.isna(value):
                continue
            
            if feature not in all_providers_features.columns:
                continue
            
            all_values = all_providers_features[feature]
            
            if all_values.dtype not in ['float64', 'int64']:
                continue
            
            if len(all_values) < 2:
                continue
            
            # Calculate percentile
            percentile = percentile_in_group(value, all_values)
            
            # Calculate z-score (robust)
            median = all_values.median()
            mad = (all_values - median).abs().median()
            
            if mad > 0:
                z_score = (value - median) / (1.4826 * mad)
            else:
                z_score = 0
            
            # Deviation from median
            if median > 0:
                deviation_ratio = value / median
            else:
                deviation_ratio = 1
            
            feature_deviations[feature] = {
                'value': float(value),
                'percentile': float(percentile) if not np.isnan(percentile) else None,
                'z_score': float(z_score),
                'median': float(median),
                'deviation_ratio': float(deviation_ratio),
                'deviation_pct': float((deviation_ratio - 1) * 100),
            }
        
        # Sort by percentile deviation (most extreme)
        sorted_features = sorted(
            feature_deviations.items(),
            key=lambda x: abs(x[1]['percentile'] - 50) if x[1]['percentile'] else 0,
            reverse=True
        )
        
        # Add top unusual features
        for feature, metrics in sorted_features[:top_n_features]:
            if metrics['percentile'] is None:
                continue
            
            # Determine if high or low
            if metrics['percentile'] > 50:
                direction = "HIGHER"
            else:
                direction = "LOWER"
            
            explanation['unusual_features'].append({
                'feature': feature,
                'value': metrics['value'],
                'percentile': metrics['percentile'],
                'direction': direction,
                'z_score': metrics['z_score'],
                'median': metrics['median'],
                'deviation_pct': metrics['deviation_pct'],
            })
        
        return explanation
    
    @staticmethod
    def generate_explanations(results_df, features_df, top_suspicious_n=10):
        """
        Generate explanations for top suspicious providers
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            Predictions and risk scores
        features_df : pd.DataFrame
            Provider features
        top_suspicious_n : int
            Number of top suspicious providers to explain
        
        Returns:
        --------
        list
            List of explanations
        """
        print_section("Generating Anomaly Explanations")
        
        # Get top suspicious providers
        top_suspicious = results_df.nlargest(top_suspicious_n, 'risk_score')
        
        explanations = []
        
        for _, row in top_suspicious.iterrows():
            provider_id = row['Provider']
            risk_score = row['risk_score']
            risk_level = row['risk_level']
            
            # Get provider features
            provider_row = features_df[features_df['Provider'] == provider_id]
            
            if len(provider_row) == 0:
                continue
            
            provider_features = provider_row.iloc[0].to_dict()
            
            # Generate explanation
            explanation = AnomalyExplainer.explain_provider(
                provider_id, provider_features, features_df, top_n_features=5
            )
            
            explanation['risk_score'] = float(risk_score)
            explanation['risk_level'] = risk_level
            
            explanations.append(explanation)
        
        logger.info(f"Generated {len(explanations)} explanations")
        
        return explanations
    
    @staticmethod
    def print_explanation(explanation):
        """Print readable explanation"""
        print(f"\nProvider: {explanation['provider']}")
        print(f"Risk Level: {explanation['risk_level']}")
        print(f"Risk Score: {explanation['risk_score']:.2f}")
        print(f"\nReasons for flagging (Top unusual features):")
        
        for i, feature in enumerate(explanation['unusual_features'], 1):
            feat_name = feature['feature']
            value = feature['value']
            percentile = feature['percentile']
            direction = feature['direction']
            median = feature['median']
            deviation_pct = feature['deviation_pct']
            
            print(f"\n  {i}. {feat_name}")
            print(f"     Value: {value:.2f}")
            print(f"     Percentile: {percentile:.1f}th")
            print(f"     Direction: {direction}")
            print(f"     Median (peers): {median:.2f}")
            print(f"     Deviation: {deviation_pct:+.1f}%")
    
    @staticmethod
    def create_explanation_report(explanations):
        """
        Create a text report of explanations
        
        Returns:
        --------
        str
            Formatted explanation report
        """
        report = "=" * 80 + "\n"
        report += "MEDICARE PROVIDER ANOMALY EXPLANATIONS\n"
        report += "=" * 80 + "\n\n"
        
        for explanation in explanations:
            report += f"Provider: {explanation['provider']}\n"
            report += f"Risk Level: {explanation['risk_level']}\n"
            report += f"Risk Score: {explanation['risk_score']:.2f}\n"
            report += f"Flagged because:\n"
            
            for i, feature in enumerate(explanation['unusual_features'], 1):
                feat_name = feature['feature']
                value = feature['value']
                percentile = feature['percentile']
                direction = feature['direction']
                deviation_pct = feature['deviation_pct']
                
                report += f"\n  {i}. {feat_name.upper()}\n"
                report += f"     Value: {value:.2f}\n"
                report += f"     Percentile: {percentile:.1f}th\n"
                report += f"     Direction: {direction}\n"
                report += f"     Deviation: {deviation_pct:+.1f}%\n"
            
            report += "\n" + "-" * 80 + "\n\n"
        
        return report
