"""
Medicare Provider Anomaly Risk Engine - Feature Engineering
Advanced provider-level feature engineering for anomaly detection
"""

import pandas as pd
import numpy as np
from datetime import datetime
import config
from src.utils import logger, setup_logging, safe_divide, percentile_in_group, print_section

setup_logging(config.LOGGING_LEVEL)

class FeatureEngineer:
    """Provider-level feature engineering"""
    
    @staticmethod
    def engineer_provider_features(inpatient_df, outpatient_df, beneficiary_df):
        """
        Engineer comprehensive provider-level features
        
        Parameters:
        -----------
        inpatient_df : pd.DataFrame
            Cleaned inpatient claims
        outpatient_df : pd.DataFrame
            Cleaned outpatient claims
        beneficiary_df : pd.DataFrame
            Cleaned beneficiary data
        
        Returns:
        --------
        pd.DataFrame
            Provider-level feature matrix
        """
        print_section("Feature Engineering")
        
        logger.info("Starting provider-level feature engineering")
        
        # Get unique providers from all sources
        providers_ip = set(inpatient_df['Provider'].unique()) if 'Provider' in inpatient_df.columns else set()
        providers_op = set(outpatient_df['Provider'].unique()) if 'Provider' in outpatient_df.columns else set()
        
        # Combine all providers
        all_providers = list(providers_ip | providers_op)
        
        features = pd.DataFrame({'Provider': all_providers})
        logger.info(f"Found {len(features)} unique providers")
        
        # Engineer features for each dataset
        logger.info("Engineering claim volume features")
        features = FeatureEngineer._engineer_claim_volume_features(
            features, inpatient_df, outpatient_df
        )
        
        logger.info("Engineering financial features")
        features = FeatureEngineer._engineer_financial_features(
            features, inpatient_df, outpatient_df
        )
        
        logger.info("Engineering beneficiary features")
        features = FeatureEngineer._engineer_beneficiary_features(
            features, inpatient_df, outpatient_df, beneficiary_df
        )
        
        logger.info("Engineering claim duration features")
        features = FeatureEngineer._engineer_duration_features(
            features, inpatient_df, outpatient_df
        )
        
        logger.info("Engineering diagnosis features")
        features = FeatureEngineer._engineer_diagnosis_features(
            features, inpatient_df, outpatient_df
        )
        
        logger.info("Engineering procedure features")
        features = FeatureEngineer._engineer_procedure_features(
            features, inpatient_df, outpatient_df
        )
        
        logger.info("Engineering physician features")
        features = FeatureEngineer._engineer_physician_features(
            features, inpatient_df, outpatient_df
        )
        
        logger.info("Engineering temporal features")
        features = FeatureEngineer._engineer_temporal_features(
            features, inpatient_df, outpatient_df
        )
        
        logger.info("Engineering domain fraud features")
        features = FeatureEngineer._engineer_domain_fraud_features(
            features, inpatient_df, outpatient_df, beneficiary_df
        )
        
        logger.info("Engineering peer-relative features")
        features = FeatureEngineer._engineer_peer_features(features)
        
        logger.info(f"Feature engineering complete: {len(features.columns)} features")
        
        return features
    
    @staticmethod
    def _engineer_claim_volume_features(df, inpatient_df, outpatient_df):
        """Engineer claim volume features"""
        
        ip_by_provider = inpatient_df.groupby('Provider').agg({
            'ClaimID': 'count',
            'BeneID': 'nunique'
        }).rename(columns={'ClaimID': 'inpatient_claim_count', 'BeneID': 'inpatient_unique_benes'})
        
        op_by_provider = outpatient_df.groupby('Provider').agg({
            'ClaimID': 'count',
            'BeneID': 'nunique'
        }).rename(columns={'ClaimID': 'outpatient_claim_count', 'BeneID': 'outpatient_unique_benes'})
        
        df = df.merge(ip_by_provider.reset_index(), on='Provider', how='left')
        df = df.merge(op_by_provider.reset_index(), on='Provider', how='left')
        
        df['inpatient_claim_count'] = df['inpatient_claim_count'].fillna(0).astype(int)
        df['outpatient_claim_count'] = df['outpatient_claim_count'].fillna(0).astype(int)
        df['inpatient_unique_benes'] = df['inpatient_unique_benes'].fillna(0).astype(int)
        df['outpatient_unique_benes'] = df['outpatient_unique_benes'].fillna(0).astype(int)
        
        df['total_claim_count'] = df['inpatient_claim_count'] + df['outpatient_claim_count']
        df['unique_beneficiary_count'] = (
            df[['inpatient_unique_benes', 'outpatient_unique_benes']].max(axis=1)
        )
        
        df['claims_per_beneficiary'] = df.apply(
            lambda row: safe_divide(row['total_claim_count'], row['unique_beneficiary_count']),
            axis=1
        )
        
        # Unique physicians vectorized
        ip_cols = [c for c in ['AttendingPhysician', 'OperatingPhysician', 'OtherPhysician'] if c in inpatient_df.columns]
        if ip_cols:
            ip_melted = inpatient_df[['Provider'] + ip_cols].melt(id_vars=['Provider'], value_vars=ip_cols).dropna()
            ip_phys_nunique = ip_melted.groupby('Provider')['value'].nunique()
        else:
            ip_phys_nunique = pd.Series(0, index=df['Provider'])
        
        op_cols = [c for c in ['AttendingPhysician', 'OperatingPhysician', 'OtherPhysician'] if c in outpatient_df.columns]
        if op_cols:
            op_melted = outpatient_df[['Provider'] + op_cols].melt(id_vars=['Provider'], value_vars=op_cols).dropna()
            op_phys_nunique = op_melted.groupby('Provider')['value'].nunique()
        else:
            op_phys_nunique = pd.Series(0, index=df['Provider'])
        
        ip_phys_map = df['Provider'].map(ip_phys_nunique).fillna(0).astype(int)
        op_phys_map = df['Provider'].map(op_phys_nunique).fillna(0).astype(int)
        df['unique_physician_count'] = np.maximum(ip_phys_map, op_phys_map)
        
        df['claims_per_physician'] = df.apply(
            lambda row: safe_divide(row['total_claim_count'], row['unique_physician_count']),
            axis=1
        )
        
        return df
    
    @staticmethod
    def _engineer_financial_features(df, inpatient_df, outpatient_df):
        """Engineer financial features"""
        
        # Combine claims for analysis
        all_claims = pd.concat([
            inpatient_df[['Provider', 'InscClaimAmtReimbursed']].copy(),
            outpatient_df[['Provider', 'InscClaimAmtReimbursed']].copy()
        ])
        
        # Basic financial stats
        financial_stats = all_claims.groupby('Provider')['InscClaimAmtReimbursed'].agg([
            ('total_reimbursement', 'sum'),
            ('avg_reimbursement', 'mean'),
            ('median_reimbursement', 'median'),
            ('min_reimbursement', 'min'),
            ('max_reimbursement', 'max'),
            ('std_reimbursement', 'std'),
        ])
        
        financial_stats['variance_reimbursement'] = financial_stats['std_reimbursement'] ** 2
        
        df = df.merge(financial_stats, on='Provider', how='left')
        
        # Fill NaN values
        for col in financial_stats.columns:
            df[col] = df[col].fillna(0)
        
        # Inpatient vs Outpatient reimbursement
        ip_reimb = inpatient_df.groupby('Provider')['InscClaimAmtReimbursed'].agg([
            ('inpatient_total_reimbursement', 'sum'),
            ('inpatient_avg_reimbursement', 'mean')
        ])
        
        op_reimb = outpatient_df.groupby('Provider')['InscClaimAmtReimbursed'].agg([
            ('outpatient_total_reimbursement', 'sum'),
            ('outpatient_avg_reimbursement', 'mean')
        ])
        
        df = df.merge(ip_reimb, on='Provider', how='left')
        df = df.merge(op_reimb, on='Provider', how='left')
        
        for col in ip_reimb.columns.tolist() + op_reimb.columns.tolist():
            df[col] = df[col].fillna(0)
        
        # Reimbursement per claim
        df['reimbursement_per_claim'] = df.apply(
            lambda row: safe_divide(row['total_reimbursement'], row['total_claim_count']),
            axis=1
        )
        
        # Reimbursement per beneficiary
        df['reimbursement_per_beneficiary'] = df.apply(
            lambda row: safe_divide(row['total_reimbursement'], row['unique_beneficiary_count']),
            axis=1
        )
        
        # High-cost claims (>90th percentile)
        threshold_90 = all_claims['InscClaimAmtReimbursed'].quantile(0.90)
        
        high_cost = all_claims[all_claims['InscClaimAmtReimbursed'] > threshold_90].groupby('Provider').size()
        df['high_cost_claim_count'] = df['Provider'].map(high_cost).fillna(0).astype(int)
        
        df['high_cost_claim_pct'] = df.apply(
            lambda row: safe_divide(row['high_cost_claim_count'], row['total_claim_count'], default=0) * 100,
            axis=1
        )
        
        # Top 1% and 5% percentiles
        threshold_99 = all_claims['InscClaimAmtReimbursed'].quantile(0.99)
        threshold_95 = all_claims['InscClaimAmtReimbursed'].quantile(0.95)
        
        top1_claims = all_claims[all_claims['InscClaimAmtReimbursed'] > threshold_99].groupby('Provider').size()
        top5_claims = all_claims[all_claims['InscClaimAmtReimbursed'] > threshold_95].groupby('Provider').size()
        
        df['top_1pct_claim_count'] = df['Provider'].map(top1_claims).fillna(0).astype(int)
        df['top_5pct_claim_count'] = df['Provider'].map(top5_claims).fillna(0).astype(int)
        
        df['top_1pct_claim_pct'] = df.apply(
            lambda row: safe_divide(row['top_1pct_claim_count'], row['total_claim_count'], default=0) * 100,
            axis=1
        )
        
        df['top_5pct_claim_pct'] = df.apply(
            lambda row: safe_divide(row['top_5pct_claim_count'], row['total_claim_count'], default=0) * 100,
            axis=1
        )
        
        # Reimbursement concentration (HHI-like metric) vectorized
        tot_reimb = all_claims.groupby('Provider')['InscClaimAmtReimbursed'].transform('sum')
        all_claims_sub = all_claims[['Provider', 'InscClaimAmtReimbursed']].copy()
        all_claims_sub['tot'] = tot_reimb
        all_claims_sub['share_sq'] = np.where(all_claims_sub['tot'] > 0, (all_claims_sub['InscClaimAmtReimbursed'] / all_claims_sub['tot']) ** 2, 0)
        conc = all_claims_sub.groupby('Provider')['share_sq'].sum()
        df['reimbursement_concentration'] = df['Provider'].map(conc).fillna(0)
        
        return df
    
    @staticmethod
    def _engineer_beneficiary_features(df, inpatient_df, outpatient_df, beneficiary_df):
        """Engineer beneficiary-level features"""
        
        # Get providers from claims
        providers_from_inpatient = inpatient_df[['Provider', 'BeneID']].copy()
        providers_from_outpatient = outpatient_df[['Provider', 'BeneID']].copy()
        
        providers_from_claims = pd.concat([
            providers_from_inpatient,
            providers_from_outpatient
        ]).drop_duplicates()
        
        # Merge beneficiary data with providers
        bene_provider = providers_from_claims.merge(beneficiary_df, on='BeneID', how='left')
        
        # Age calculation
        today = pd.Timestamp('2012-12-31')  # Reference date in data
        bene_provider['age'] = (today - pd.to_datetime(bene_provider['DOB'], errors='coerce')).dt.days / 365.25
        
        age_stats = bene_provider.groupby('Provider')['age'].agg([
            ('avg_beneficiary_age', 'mean'),
            ('median_beneficiary_age', 'median'),
            ('min_beneficiary_age', 'min'),
            ('max_beneficiary_age', 'max'),
            ('std_beneficiary_age', 'std'),
        ])
        
        df = df.merge(age_stats.reset_index(), on='Provider', how='left')
        
        # Gender distribution
        gender_dist = bene_provider.groupby(['Provider', 'Gender']).size().unstack(fill_value=0)
        total_benes = gender_dist.sum(axis=1)
        
        if 1 in gender_dist.columns:  # Assuming 1 = Male
            df = df.merge((gender_dist[1] / total_benes * 100).reset_index(name='male_percentage'), on='Provider', how='left')
        else:
            df['male_percentage'] = 0
        
        if 2 in gender_dist.columns:  # Assuming 2 = Female
            df = df.merge((gender_dist[2] / total_benes * 100).reset_index(name='female_percentage'), on='Provider', how='left')
        else:
            df['female_percentage'] = 0
        
        # Chronic conditions
        chronic_condition_cols = [col for col in config.CHRONIC_CONDITIONS if col in beneficiary_df.columns]
        
        if chronic_condition_cols:
            for condition in chronic_condition_cols:
                bene_provider[condition] = pd.to_numeric(bene_provider[condition], errors='coerce').fillna(0)
            
            # Count chronic conditions per beneficiary
            bene_provider['chronic_condition_count'] = bene_provider[chronic_condition_cols].sum(axis=1)
            
            chronic_stats = bene_provider.groupby('Provider')['chronic_condition_count'].agg([
                ('avg_chronic_conditions', 'mean'),
                ('median_chronic_conditions', 'median'),
            ])
            
            df = df.merge(chronic_stats.reset_index(), on='Provider', how='left')
            
            # High-risk beneficiary percentage
            high_risk_benes = bene_provider[bene_provider['chronic_condition_count'] >= 3].groupby('Provider').size()
            total_benes_per_provider = bene_provider.groupby('Provider').size()
            
            high_risk_pct = (high_risk_benes / total_benes_per_provider * 100).reindex(df['Provider']).fillna(0)
            df['high_risk_beneficiary_pct'] = high_risk_pct.values
        
        # Fill NaN values
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(0)
        
        return df
    
    @staticmethod
    def _engineer_duration_features(df, inpatient_df, outpatient_df):
        """Engineer claim duration features"""
        
        # Inpatient duration
        inpatient_df_copy = inpatient_df.copy()
        inpatient_df_copy['duration_days'] = (
            inpatient_df_copy['ClaimEndDt'] - inpatient_df_copy['ClaimStartDt']
        ).dt.days
        
        duration_stats = inpatient_df_copy.groupby('Provider')['duration_days'].agg([
            ('avg_inpatient_duration', 'mean'),
            ('median_inpatient_duration', 'median'),
            ('max_inpatient_duration', 'max'),
            ('min_inpatient_duration', 'min'),
            ('std_inpatient_duration', 'std'),
        ])
        
        df = df.merge(duration_stats, on='Provider', how='left')
        
        # Long-duration claims (>30 days)
        long_duration = inpatient_df_copy[inpatient_df_copy['duration_days'] > 30].groupby('Provider').size()
        df['long_duration_claim_count'] = df['Provider'].map(long_duration).fillna(0).astype(int)
        
        df['long_duration_claim_pct'] = df.apply(
            lambda row: safe_divide(row['long_duration_claim_count'], row['inpatient_claim_count'], default=0) * 100,
            axis=1
        )
        
        # Fill NaN values
        for col in duration_stats.columns:
            df[col] = df[col].fillna(0)
        
        return df
    
    @staticmethod
    def _engineer_diagnosis_features(df, inpatient_df, outpatient_df):
        """Engineer diagnosis-related features"""
        
        all_claims = pd.concat([inpatient_df, outpatient_df])
        
        if 'diagnosis_count' not in all_claims.columns:
            all_claims['diagnosis_count'] = 0
        
        diag_stats = all_claims.groupby('Provider')['diagnosis_count'].agg([
            ('avg_diagnoses_per_claim', 'mean'),
            ('median_diagnoses_per_claim', 'median'),
            ('max_diagnoses_per_claim', 'max'),
        ])
        
        df = df.merge(diag_stats, on='Provider', how='left')
        
        # Unique diagnosis count vectorized
        diag_cols = [col for col in config.DIAGNOSIS_COLUMNS if col in all_claims.columns]
        if diag_cols:
            melted_diag = all_claims[['Provider'] + diag_cols].melt(id_vars=['Provider'], value_vars=diag_cols).dropna()
            unique_diags = melted_diag.groupby('Provider')['value'].nunique()
        else:
            unique_diags = pd.Series(0, index=df['Provider'])
        df['unique_diagnosis_count'] = df['Provider'].map(unique_diags).fillna(0).astype(int)
        
        # Diagnosis diversity (unique diagnoses / total claims)
        df['diagnosis_diversity'] = df.apply(
            lambda row: safe_divide(row['unique_diagnosis_count'], row['total_claim_count']),
            axis=1
        )
        
        # High-diagnosis claims (>5 diagnoses)
        high_diag = all_claims[all_claims['diagnosis_count'] > 5].groupby('Provider').size()
        df['high_diagnosis_claim_count'] = df['Provider'].map(high_diag).fillna(0).astype(int)
        
        df['high_diagnosis_claim_pct'] = df.apply(
            lambda row: safe_divide(row['high_diagnosis_claim_count'], row['total_claim_count'], default=0) * 100,
            axis=1
        )
        
        # Fill NaN values
        for col in diag_stats.columns:
            df[col] = df[col].fillna(0)
        
        return df
    
    @staticmethod
    def _engineer_procedure_features(df, inpatient_df, outpatient_df):
        """Engineer procedure-related features"""
        
        all_claims = pd.concat([inpatient_df, outpatient_df])
        
        if 'procedure_count' not in all_claims.columns:
            all_claims['procedure_count'] = 0
        
        proc_stats = all_claims.groupby('Provider')['procedure_count'].agg([
            ('avg_procedures_per_claim', 'mean'),
            ('median_procedures_per_claim', 'median'),
            ('max_procedures_per_claim', 'max'),
        ])
        
        df = df.merge(proc_stats, on='Provider', how='left')
        
        # Unique procedure count vectorized
        proc_cols = [col for col in config.PROCEDURE_COLUMNS if col in all_claims.columns]
        if proc_cols:
            melted_proc = all_claims[['Provider'] + proc_cols].melt(id_vars=['Provider'], value_vars=proc_cols).dropna()
            unique_procs = melted_proc.groupby('Provider')['value'].nunique()
        else:
            unique_procs = pd.Series(0, index=df['Provider'])
        df['unique_procedure_count'] = df['Provider'].map(unique_procs).fillna(0).astype(int)
        
        # Procedure diversity
        df['procedure_diversity'] = df.apply(
            lambda row: safe_divide(row['unique_procedure_count'], row['total_claim_count']),
            axis=1
        )
        
        # High-procedure claims (>3 procedures)
        high_proc = all_claims[all_claims['procedure_count'] > 3].groupby('Provider').size()
        df['high_procedure_claim_count'] = df['Provider'].map(high_proc).fillna(0).astype(int)
        
        df['high_procedure_claim_pct'] = df.apply(
            lambda row: safe_divide(row['high_procedure_claim_count'], row['total_claim_count'], default=0) * 100,
            axis=1
        )
        
        # Fill NaN values
        for col in proc_stats.columns:
            df[col] = df[col].fillna(0)
        
        return df
    
    @staticmethod
    def _engineer_physician_features(df, inpatient_df, outpatient_df):
        """Engineer physician-related features"""
        
        all_claims = pd.concat([inpatient_df, outpatient_df])
        
        # Claims per physician concentration vectorized
        phys_cols = [c for c in ['AttendingPhysician', 'OperatingPhysician', 'OtherPhysician'] if c in all_claims.columns]
        if phys_cols:
            melted_phys = all_claims[['Provider'] + phys_cols].melt(id_vars=['Provider'], value_vars=phys_cols).dropna()
            phys_counts = melted_phys.groupby(['Provider', 'value']).size().reset_index(name='n')
            total_per_prov = phys_counts.groupby('Provider')['n'].transform('sum')
            phys_counts['share_sq'] = np.where(total_per_prov > 0, (phys_counts['n'] / total_per_prov) ** 2, 0)
            hhi = phys_counts.groupby('Provider')['share_sq'].sum()
            df['physician_concentration'] = df['Provider'].map(hhi).fillna(0)
        else:
            df['physician_concentration'] = 0
        
        # Average claims per physician
        df['avg_claims_per_physician'] = df.apply(
            lambda row: safe_divide(row['total_claim_count'], row['unique_physician_count']),
            axis=1
        )
        
        return df
    
    @staticmethod
    def _engineer_temporal_features(df, inpatient_df, outpatient_df):
        """Engineer temporal features"""
        
        all_claims = pd.concat([inpatient_df, outpatient_df])
        
        # Extract year-month
        all_claims['year_month'] = all_claims['ClaimStartDt'].dt.to_period('M')
        
        # Active months
        active_months = all_claims.groupby('Provider')['year_month'].nunique()
        df['active_months'] = df['Provider'].map(active_months).fillna(0).astype(int)
        
        # Claims per month
        df['claims_per_month'] = df.apply(
            lambda row: safe_divide(row['total_claim_count'], row['active_months']),
            axis=1
        )
        
        # Reimbursement per month
        monthly_reimb = all_claims.groupby(['Provider', 'year_month'])['InscClaimAmtReimbursed'].sum().reset_index()
        monthly_reimb = monthly_reimb.groupby('Provider')['InscClaimAmtReimbursed'].agg(['mean', 'std'])
        monthly_reimb.columns = ['reimbursement_per_month', 'monthly_reimbursement_std']
        
        df = df.merge(monthly_reimb, on='Provider', how='left')
        df['monthly_reimbursement_std'] = df['monthly_reimbursement_std'].fillna(0)
        
        return df
    
    @staticmethod
    def _engineer_peer_features(df):
        """
        Engineer peer-relative features
        Compare each provider against similar providers
        """
        
        # Define provider "peers" by common characteristics
        # Since we don't have specialty, use claims count brackets
        
        def add_peer_features(df):
            """Add percentile ranks within peer groups"""
            
            numeric_features = [
                'total_claim_count',
                'total_reimbursement',
                'avg_reimbursement',
                'reimbursement_per_beneficiary',
                'unique_beneficiary_count',
                'claims_per_beneficiary',
            ]
            
            for feature in numeric_features:
                if feature in df.columns:
                    # Percentile within all providers
                    df[f'{feature}_percentile'] = (
                        df[feature].rank(pct=True) * 100
                    )
                    
                    # Deviation from median
                    median_val = df[feature].median()
                    df[f'{feature}_vs_median'] = df[feature] / (median_val + 1)
            
            return df
        
        df = add_peer_features(df)
        
        return df
    
    @staticmethod
    def _engineer_domain_fraud_features(df, inpatient_df, outpatient_df, beneficiary_df):
        """
        Engineer specialized domain fraud indicators
        """
        all_claims = pd.concat([inpatient_df, outpatient_df], ignore_index=True)
        
        # 1. Deductible features
        if 'DeductibleAmtPaid' in all_claims.columns:
            deduct_stats = all_claims.groupby('Provider')['DeductibleAmtPaid'].agg([
                ('total_deductible', 'sum'),
                ('avg_deductible', 'mean'),
                ('std_deductible', 'std'),
                ('max_deductible', 'max')
            ]).fillna(0)
            df = df.merge(deduct_stats.reset_index(), on='Provider', how='left')
        
        # 2. Inpatient Reimbursement per Stay Day
        if len(inpatient_df) > 0 and 'DischargeDt' in inpatient_df.columns and 'AdmissionDt' in inpatient_df.columns:
            ip_copy = inpatient_df.copy()
            ip_copy['stay_days'] = np.maximum((ip_copy['DischargeDt'] - ip_copy['AdmissionDt']).dt.days, 1)
            ip_copy['reimb_per_day'] = ip_copy['InscClaimAmtReimbursed'] / ip_copy['stay_days']
            ip_day_stats = ip_copy.groupby('Provider')['reimb_per_day'].agg([
                ('avg_reimb_per_day', 'mean'),
                ('max_reimb_per_day', 'max')
            ]).fillna(0)
            df = df.merge(ip_day_stats.reset_index(), on='Provider', how='left')
        
        # 3. Same-day claims (StartDt == EndDt)
        if 'ClaimStartDt' in all_claims.columns and 'ClaimEndDt' in all_claims.columns:
            all_claims['same_day'] = (all_claims['ClaimStartDt'] == all_claims['ClaimEndDt']).astype(int)
            same_day_stats = all_claims.groupby('Provider')['same_day'].agg([
                ('same_day_claim_count', 'sum'),
                ('same_day_claim_pct', 'mean')
            ]).fillna(0)
            df = df.merge(same_day_stats.reset_index(), on='Provider', how='left')
        
        # 4. Physician role overlap (Attending == Operating)
        if 'AttendingPhysician' in all_claims.columns and 'OperatingPhysician' in all_claims.columns:
            all_claims['phys_same_att_opr'] = (
                all_claims['AttendingPhysician'].notna() & 
                (all_claims['AttendingPhysician'] == all_claims['OperatingPhysician'])
            ).astype(int)
            phys_overlap_stats = all_claims.groupby('Provider')['phys_same_att_opr'].agg([
                ('phys_same_att_opr_count', 'sum'),
                ('phys_same_att_opr_pct', 'mean')
            ]).fillna(0)
            df = df.merge(phys_overlap_stats.reset_index(), on='Provider', how='left')
        
        # 5. Beneficiary Renal Disease & Geographic Diversity
        if 'BeneID' in all_claims.columns and len(beneficiary_df) > 0:
            bene_claims = all_claims[['Provider', 'BeneID']].drop_duplicates().merge(beneficiary_df, on='BeneID', how='left')
            
            if 'RenalDiseaseIndicator' in bene_claims.columns:
                bene_claims['RenalDisease'] = (bene_claims['RenalDiseaseIndicator'] == '1').astype(int)
                renal_stats = bene_claims.groupby('Provider')['RenalDisease'].agg([
                    ('renal_disease_count', 'sum'),
                    ('renal_disease_pct', 'mean')
                ]).fillna(0)
                df = df.merge(renal_stats.reset_index(), on='Provider', how='left')
            
            geo_agg = {}
            if 'State' in bene_claims.columns:
                geo_agg['State'] = 'nunique'
            if 'County' in bene_claims.columns:
                geo_agg['County'] = 'nunique'
            
            if geo_agg:
                geo_stats = bene_claims.groupby('Provider').agg(geo_agg).rename(
                    columns={'State': 'unique_states_count', 'County': 'unique_counties_count'}
                ).fillna(0)
                df = df.merge(geo_stats.reset_index(), on='Provider', how='left')
        
        # 6. Ratios
        if 'total_deductible' in df.columns and 'total_reimbursement' in df.columns:
            df['reimb_to_deduct_ratio'] = np.where(
                df['total_deductible'] > 0,
                df['total_reimbursement'] / df['total_deductible'],
                0
            )
        
        if 'inpatient_claim_count' in df.columns and 'total_claim_count' in df.columns:
            df['inpatient_claim_ratio'] = np.where(
                df['total_claim_count'] > 0,
                df['inpatient_claim_count'] / df['total_claim_count'],
                0
            )
        
        # Fill missing numeric values with 0
        for col in df.columns:
            if col != 'Provider' and df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(0)
        
        return df
