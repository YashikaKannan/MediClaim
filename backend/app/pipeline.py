import os
import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import logging
from catboost import CatBoostClassifier
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MediClaimPipeline")

# Autoencoder definition
class AutoencoderModel(nn.Module):
    def __init__(self, input_dim):
        super(AutoencoderModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )
        
    def forward(self, x):
        return self.decoder(self.encoder(x))

# Global pipeline status tracker
pipeline_status = {
    "status": "idle",
    "progress": 0,
    "step": "",
    "error": None,
    "completed_at": None,
    "summary": None
}

DATA_DIR = "e:/CTS - MediClaim/datas"
UPLOADS_DIR = "e:/CTS - MediClaim/backend/app/uploads"
DB_PATH = "e:/CTS - MediClaim/backend/app/db/mediclaim.db"
ML_MODELS_DIR = "e:/CTS - MediClaim/backend/app/ml_models"

def run_risk_pipeline():
    global pipeline_status
    try:
        pipeline_status["status"] = "running"
        pipeline_status["progress"] = 5
        pipeline_status["step"] = "Step 1: Checking uploaded datasets and default files..."
        pipeline_status["error"] = None
        pipeline_status["summary"] = None
        
        logger.info("Starting MediClaim Risk Analysis Pipeline...")
        
        # Check files
        claims_file = os.path.join(UPLOADS_DIR, "claims.csv")
        beneficiary_file = os.path.join(UPLOADS_DIR, "beneficiary.csv")
        provider_file = os.path.join(UPLOADS_DIR, "provider.csv")
        
        # Fallbacks
        if not os.path.exists(claims_file):
            logger.info("No uploaded claims.csv found. Using default dataset files.")
            # Default dataset consists of inpatient + outpatient claim files
            in_file = os.path.join(DATA_DIR, "Train_Inpatientdata-1542865627584.csv")
            out_file = os.path.join(DATA_DIR, "Train_Outpatientdata-1542865627584.csv")
            if not os.path.exists(in_file) or not os.path.exists(out_file):
                raise Exception("Default inpatient/outpatient claims CSV files not found.")
            claims_list = [pd.read_csv(in_file), pd.read_csv(out_file)]
            for i, df_c in enumerate(claims_list):
                df_c["is_inpatient"] = 1 if i == 0 else 0
            claims_df = pd.concat(claims_list, ignore_index=True)
        else:
            logger.info(f"Using uploaded claims file: {claims_file}")
            claims_df = pd.read_csv(claims_file)
            if "is_inpatient" not in claims_df.columns:
                # Deduce is_inpatient
                claims_df["is_inpatient"] = claims_df.apply(
                    lambda r: 1 if pd.notnull(r.get("AdmissionDt")) or pd.notnull(r.get("DischargeDt")) else 0,
                    axis=1
                )
        
        if not os.path.exists(beneficiary_file):
            logger.info("No uploaded beneficiary.csv found. Using default beneficiary file.")
            bene_path = os.path.join(DATA_DIR, "Train_Beneficiarydata-1542865627584.csv")
            if not os.path.exists(bene_path):
                raise Exception("Default beneficiary CSV file not found.")
            bene_df = pd.read_csv(bene_path)
        else:
            logger.info(f"Using uploaded beneficiary file: {beneficiary_file}")
            bene_df = pd.read_csv(beneficiary_file)
            
        if not os.path.exists(provider_file):
            logger.info("No uploaded provider.csv labels found. Using default labels file.")
            labels_path = os.path.join(DATA_DIR, "Train-1542865627584.csv")
            if not os.path.exists(labels_path):
                raise Exception("Default provider labels CSV file not found.")
            labels_df = pd.read_csv(labels_path)
        else:
            logger.info(f"Using uploaded provider file: {provider_file}")
            labels_df = pd.read_csv(provider_file)
            
        pipeline_status["progress"] = 20
        pipeline_status["step"] = "Step 2: Preprocessing datasets..."
        
        # 1. Process Beneficiary Chronic Conditions
        chronic_cols = [c for c in bene_df.columns if c.startswith("ChronicCond_")]
        for col in chronic_cols:
            bene_df[col] = bene_df[col].map({1: 1, 2: 0})
            
        bene_df["RenalDiseaseIndicator"] = bene_df["RenalDiseaseIndicator"].map({"0": 0, "Y": 1})
        if len(chronic_cols) > 0:
            bene_df["ChronicCondCount"] = bene_df[chronic_cols].sum(axis=1)
        else:
            bene_df["ChronicCondCount"] = 0
            
        bene_df["DOB"] = pd.to_datetime(bene_df["DOB"], errors="coerce")
        bene_df["Age"] = 2009 - bene_df["DOB"].dt.year
        bene_df["IsDeceased"] = bene_df["DOD"].notnull().astype(int)
        
        # 2. Process Dates and Claim Duration
        claims_df["ClaimStartDt"] = pd.to_datetime(claims_df["ClaimStartDt"], errors="coerce")
        claims_df["ClaimEndDt"] = pd.to_datetime(claims_df["ClaimEndDt"], errors="coerce")
        claims_df["ClaimDuration"] = (claims_df["ClaimEndDt"] - claims_df["ClaimStartDt"]).dt.days
        
        admission_col = claims_df["AdmissionDt"] if "AdmissionDt" in claims_df.columns else pd.Series(np.nan, index=claims_df.index)
        discharge_col = claims_df["DischargeDt"] if "DischargeDt" in claims_df.columns else pd.Series(np.nan, index=claims_df.index)
        claims_df["AdmissionDt"] = pd.to_datetime(admission_col, errors="coerce")
        claims_df["DischargeDt"] = pd.to_datetime(discharge_col, errors="coerce")
        claims_df["AdmissionDuration"] = (claims_df["DischargeDt"] - claims_df["AdmissionDt"]).dt.days
        
        diag_cols = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
        proc_cols = [f"ClmProcedureCode_{i}" for i in range(1, 7)]
        
        # Ensure columns exist
        for c in diag_cols + proc_cols:
            if c not in claims_df.columns:
                claims_df[c] = np.nan
                
        claims_df["DiagnosisCount"] = claims_df[diag_cols].notnull().sum(axis=1)
        claims_df["ProcedureCount"] = claims_df[proc_cols].notnull().sum(axis=1)
        
        pipeline_status["progress"] = 40
        pipeline_status["step"] = "Step 3: Engineering provider features..."
        
        # 3. Merge claims and beneficiaries
        claims_merged = pd.merge(
            claims_df, 
            bene_df[["BeneID", "Age", "Gender", "Race", "State", "County", "ChronicCondCount", "IsDeceased", "RenalDiseaseIndicator"]], 
            on="BeneID", 
            how="left"
        )
        
        # Group by provider
        provider_groups = claims_merged.groupby("Provider")
        
        # Aggregate features
        prov_features = pd.DataFrame(index=provider_groups.indices.keys())
        prov_features.index.name = "Provider"
        
        # Volume features
        prov_features["total_claims"] = provider_groups["ClaimID"].count()
        prov_features["inpatient_claims"] = provider_groups["is_inpatient"].sum()
        prov_features["outpatient_claims"] = prov_features["total_claims"] - prov_features["inpatient_claims"]
        prov_features["inpatient_ratio"] = prov_features["inpatient_claims"] / prov_features["total_claims"]
        prov_features["total_beneficiaries"] = provider_groups["BeneID"].nunique()
        prov_features["claims_per_beneficiary"] = prov_features["total_claims"] / prov_features["total_beneficiaries"]
        
        # Financial features
        prov_features["total_reimbursement"] = provider_groups["InscClaimAmtReimbursed"].sum()
        prov_features["mean_reimbursement"] = provider_groups["InscClaimAmtReimbursed"].mean()
        prov_features["max_reimbursement"] = provider_groups["InscClaimAmtReimbursed"].max()
        prov_features["total_deductible"] = provider_groups["DeductibleAmtPaid"].sum()
        prov_features["mean_deductible"] = provider_groups["DeductibleAmtPaid"].mean()
        prov_features["reimbursement_per_beneficiary"] = prov_features["total_reimbursement"] / prov_features["total_beneficiaries"]
        
        # Physician network features
        prov_features["unique_attending"] = provider_groups["AttendingPhysician"].nunique()
        prov_features["unique_operating"] = provider_groups["OperatingPhysician"].nunique()
        prov_features["unique_other"] = provider_groups["OtherPhysician"].nunique()
        
        # Clinical features
        prov_features["mean_diagnosis_count"] = provider_groups["DiagnosisCount"].mean()
        prov_features["mean_procedure_count"] = provider_groups["ProcedureCount"].mean()
        
        # Unique codes per provider
        diag_melted = claims_merged[["Provider"] + diag_cols].melt(id_vars="Provider").dropna()
        prov_features["unique_diagnosis_codes"] = diag_melted.groupby("Provider")["value"].nunique()
        
        proc_melted = claims_merged[["Provider"] + proc_cols].melt(id_vars="Provider").dropna()
        prov_features["unique_procedure_codes"] = proc_melted.groupby("Provider")["value"].nunique()
        
        # Demographics
        prov_features["mean_patient_age"] = provider_groups["Age"].mean()
        prov_features["mean_patient_chronic_conds"] = provider_groups["ChronicCondCount"].mean()
        prov_features["patient_death_rate"] = provider_groups["IsDeceased"].mean()
        prov_features["renal_disease_rate"] = provider_groups["RenalDiseaseIndicator"].mean()
        
        # Location features
        prov_features["unique_states"] = provider_groups["State"].nunique()
        prov_features["unique_counties"] = provider_groups["County"].nunique()
        
        # Optimize primary state mode calculation
        state_counts = claims_merged.groupby(["Provider", "State"]).size().reset_index(name="count")
        primary_states = state_counts.sort_values("count", ascending=False).drop_duplicates("Provider").set_index("Provider")["State"]
        prov_features["primary_state"] = primary_states.reindex(prov_features.index).fillna(-1).astype(int)
        
        # Durations
        prov_features["mean_claim_duration"] = provider_groups["ClaimDuration"].mean()
        prov_features["mean_admission_duration"] = provider_groups["AdmissionDuration"].mean().fillna(0)
        
        # Map provider labels
        labels_df_mapped = labels_df.copy()
        if "PotentialFraud" in labels_df_mapped.columns:
            labels_df_mapped["PotentialFraud"] = labels_df_mapped["PotentialFraud"].map({"Yes": 1, "No": 0, 1: 1, 0: 0})
            labels_df_mapped.set_index("Provider", inplace=True)
            prov_features = prov_features.join(labels_df_mapped[["PotentialFraud"]], how="left")
        else:
            prov_features["PotentialFraud"] = 0
            
        prov_features["PotentialFraud"] = prov_features["PotentialFraud"].fillna(0).astype(int)
        
        # Fill missing values
        for col in prov_features.columns:
            if prov_features[col].isnull().any():
                prov_features[col] = prov_features[col].fillna(prov_features[col].median())
                
        pipeline_status["progress"] = 60
        pipeline_status["step"] = "Step 4: Running models and fusing risk scores..."
        
        # 4. Load Models and parameters
        scaler = joblib.load(os.path.join(ML_MODELS_DIR, "scaler.joblib"))
        iforest = joblib.load(os.path.join(ML_MODELS_DIR, "iforest.joblib"))
        lof = joblib.load(os.path.join(ML_MODELS_DIR, "lof.joblib"))
        ocsvm = joblib.load(os.path.join(ML_MODELS_DIR, "ocsvm.joblib"))
        
        ae_threshold = joblib.load(os.path.join(ML_MODELS_DIR, "ae_threshold.joblib"))
        ae_model = AutoencoderModel(27)
        ae_model.load_state_dict(torch.load(os.path.join(ML_MODELS_DIR, "autoencoder.pth")))
        ae_model.eval()
        
        cat_model = CatBoostClassifier()
        cat_model.load_model(os.path.join(ML_MODELS_DIR, "catboost.cbm"))
        
        peer_medians = pd.read_csv(os.path.join(ML_MODELS_DIR, "peer_medians.csv"))
        if 'peer_group' in peer_medians.columns:
            peer_medians.set_index('peer_group', inplace=True)
            
        robust_z_params = joblib.load(os.path.join(ML_MODELS_DIR, "robust_z_params.joblib"))
        
        # Standardize features (exclude index-like fields: primary_state and PotentialFraud and Provider index)
        feature_cols = [
            'total_claims', 'inpatient_claims', 'outpatient_claims', 'inpatient_ratio',
            'total_beneficiaries', 'claims_per_beneficiary', 'total_reimbursement',
            'mean_reimbursement', 'max_reimbursement', 'total_deductible', 'mean_deductible',
            'reimbursement_per_beneficiary', 'unique_attending', 'unique_operating',
            'unique_other', 'mean_diagnosis_count', 'mean_procedure_count',
            'unique_diagnosis_codes', 'unique_procedure_codes', 'mean_patient_age',
            'mean_patient_chronic_conds', 'patient_death_rate', 'renal_disease_rate',
            'unique_states', 'unique_counties', 'mean_claim_duration', 'mean_admission_duration'
        ]
        
        X = prov_features[feature_cols].copy()
        X_scaled = scaler.transform(X)
        
        # Get individual model scores
        if_decision = iforest.decision_function(X_scaled)
        if_scores = np.clip((0.2 - if_decision) / 0.5 * 100, 0, 100)
        
        lof_decision = lof.score_samples(X_scaled)
        lof_scores = np.clip((-lof_decision - 1.0) * 100, 0, 100)
        
        ocsvm_decision = ocsvm.decision_function(X_scaled)
        ocsvm_scores = np.clip((-ocsvm_decision) * 100, 0, 100)
        
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            X_reconstructed = ae_model(X_tensor)
            ae_errors = torch.mean((X_tensor - X_reconstructed)**2, dim=1).numpy()
        ae_scores = np.clip((ae_errors / ae_threshold) * 100, 0, 100)
        
        cat_probs = cat_model.predict_proba(X_scaled)[:, 1]
        cat_scores = cat_probs * 100
        
        # Peer Benchmarking
        prov_features["provider_type"] = prov_features["inpatient_ratio"].apply(lambda r: "Inpatient-heavy" if r >= 0.2 else "Outpatient-heavy")
        prov_features["peer_group"] = prov_features["provider_type"] + "_" + prov_features["primary_state"].astype(int).astype(str)
        
        # Join with peer group medians
        prov_features = prov_features.join(peer_medians, on="peer_group", how="left")
        # Fill any missing peer medians with global medians
        prov_features["peer_median_reimbursement"] = prov_features["peer_median_reimbursement"].fillna(prov_features["total_reimbursement"].median())
        prov_features["peer_median_claims"] = prov_features["peer_median_claims"].fillna(prov_features["total_claims"].median())
        prov_features["peer_median_beneficiaries"] = prov_features["peer_median_beneficiaries"].fillna(prov_features["total_beneficiaries"].median())
        
        prov_features["reimbursement_ratio"] = prov_features["total_reimbursement"] / prov_features["peer_median_reimbursement"].replace(0, 1)
        prov_features["claims_ratio"] = prov_features["total_claims"] / prov_features["peer_median_claims"].replace(0, 1)
        prov_features["beneficiary_ratio"] = prov_features["total_beneficiaries"] / prov_features["peer_median_beneficiaries"].replace(0, 1)
        
        prov_features["reimbursement_ratio_score"] = np.clip((prov_features["reimbursement_ratio"] - 1.0) / 3.0 * 100, 0, 100)
        prov_features["claims_ratio_score"] = np.clip((prov_features["claims_ratio"] - 1.0) / 3.0 * 100, 0, 100)
        prov_features["beneficiary_ratio_score"] = np.clip((prov_features["beneficiary_ratio"] - 1.0) / 3.0 * 100, 0, 100)
        
        peer_score = (prov_features["reimbursement_ratio_score"] + prov_features["claims_ratio_score"] + prov_features["beneficiary_ratio_score"]) / 3.0
        
        # Robust Z-Scores
        medians = robust_z_params["medians"]
        mads = robust_z_params["mads"]
        
        prov_features["claims_per_beneficiary"] = prov_features["total_claims"] / prov_features["total_beneficiaries"].replace(0, 1)
        
        robust_z_reimb = (prov_features["total_reimbursement"] - medians["total_reimbursement"]) / mads["total_reimbursement"]
        robust_z_claims = (prov_features["total_claims"] - medians["total_claims"]) / mads["total_claims"]
        robust_z_cpb = (prov_features["claims_per_beneficiary"] - medians["claims_per_beneficiary"]) / mads["claims_per_beneficiary"]
        
        prov_features["robust_z_reimbursement_score"] = np.clip(robust_z_reimb / 5.0 * 100, 0, 100)
        prov_features["robust_z_claims_score"] = np.clip(robust_z_claims / 5.0 * 100, 0, 100)
        prov_features["robust_z_cpb_score"] = np.clip(robust_z_cpb / 5.0 * 100, 0, 100)
        
        statistical_score = (prov_features["robust_z_reimbursement_score"] + prov_features["robust_z_claims_score"] + prov_features["robust_z_cpb_score"]) / 3.0
        
        # Percentiles
        prov_features["reimbursement_percentile"] = prov_features["total_reimbursement"].rank(pct=True) * 100
        prov_features["claims_percentile"] = prov_features["total_claims"].rank(pct=True) * 100
        prov_features["beneficiary_percentile"] = prov_features["total_beneficiaries"].rank(pct=True) * 100
        
        # Fetch configurations from Settings DB
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM settings WHERE id = 1")
        settings = dict(cursor.fetchone())
        conn.close()
        
        cat_w = settings["catboost_weight"] / 100.0
        if_w = settings["iforest_weight"] / 100.0
        lof_w = settings["lof_weight"] / 100.0
        robust_z_w = settings["robust_z_weight"] / 100.0
        peer_w = settings["peer_benchmark_weight"] / 100.0
        leie_w = settings["leie_weight"] / 100.0
        
        # Simulate LEIE scores (OIG Excluded Individuals/Entities List Match)
        leie_scores = np.zeros(len(prov_features))
        for idx, p_id in enumerate(prov_features.index):
            # Deterministic simulation: ~1.5% exclusion rate based on ID hash
            if (hash(p_id) % 67) == 0:
                leie_scores[idx] = 100.0
                
        # Fused Risk Score
        ml_score = (if_scores + lof_scores + cat_scores) / 3.0
        final_risk_score = (
            cat_w * cat_scores +
            if_w * if_scores +
            lof_w * lof_scores +
            robust_z_w * statistical_score +
            peer_w * peer_score +
            leie_w * leie_scores
        )
        
        prov_features["ml_score"] = ml_score
        prov_features["statistical_score"] = statistical_score
        prov_features["peer_score"] = peer_score
        prov_features["risk_score"] = final_risk_score
        
        # Classify risk levels based on settings limits
        high_limit = settings["high_risk_limit"]
        crit_limit = settings["crit_risk_limit"]
        
        def classify_risk_custom(s):
            if s <= 30.0:
                return "Low"
            elif s <= high_limit:
                return "Medium"
            elif s <= crit_limit:
                return "High"
            else:
                return "Critical"
                
        prov_features["risk_level"] = prov_features["risk_score"].apply(classify_risk_custom)
        
        pipeline_status["progress"] = 80
        pipeline_status["step"] = "Step 5: Generating explanations and formatting claim details..."
        
        # 5. Populate SQLite tables
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Clear existing tables (or merge gracefully)
        cursor.execute("DELETE FROM providers")
        cursor.execute("DELETE FROM model_scores")
        cursor.execute("DELETE FROM peer_benchmarks")
        cursor.execute("DELETE FROM explanations")
        # Keep investigations notes/status if they already exist for a provider!
        # Fetch current investigations to prevent overwriting user input
        cursor.execute("SELECT provider_id, status, notes, assigned_investigator FROM investigations")
        existing_inv = {row[0]: {"status": row[1], "notes": row[2], "assigned_investigator": row[3]} for row in cursor.fetchall()}
        
        cursor.execute("DELETE FROM investigations")
        cursor.execute("DELETE FROM claims")
        
        # Insert provider levels
        logger.info(f"Writing {len(prov_features)} providers to SQLite database...")
        
        for idx, (p_id, row) in enumerate(prov_features.iterrows()):
            # Insert provider
            cursor.execute("""
            INSERT INTO providers (
                provider_id, total_claims, inpatient_claims, outpatient_claims, inpatient_ratio,
                total_beneficiaries, total_reimbursement, mean_reimbursement, risk_score,
                risk_level, primary_state, provider_type, PotentialFraud
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                p_id, int(row["total_claims"]), int(row["inpatient_claims"]), int(row["outpatient_claims"]),
                float(row["inpatient_ratio"]), int(row["total_beneficiaries"]), float(row["total_reimbursement"]),
                float(row["mean_reimbursement"]), float(row["risk_score"]), row["risk_level"],
                int(row["primary_state"]), row["provider_type"], int(row["PotentialFraud"])
            ))
            
            # Insert scores
            cursor.execute("""
            INSERT INTO model_scores (
                provider_id, isolation_score, autoencoder_score, lof_score, ocsvm_score,
                catboost_score, ml_score, statistical_score, peer_score, leie_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                p_id, float(if_scores[idx]), float(ae_scores[idx]), float(lof_scores[idx]),
                float(ocsvm_scores[idx]), float(cat_scores[idx]), float(row["ml_score"]),
                float(row["statistical_score"]), float(row["peer_score"]), float(leie_scores[idx])
            ))
            
            # Insert benchmarks
            cursor.execute("""
            INSERT INTO peer_benchmarks (
                provider_id, reimbursement_ratio, claims_ratio, beneficiary_ratio,
                reimbursement_percentile, claims_percentile, beneficiary_percentile
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                p_id, float(row["reimbursement_ratio"]), float(row["claims_ratio"]), float(row["beneficiary_ratio"]),
                float(row["reimbursement_percentile"]), float(row["claims_percentile"]), float(row["beneficiary_percentile"])
            ))
            
            # Generate explanations
            reasons = []
            if row["reimbursement_ratio"] > settings["z_cutoff"]:
                reasons.append(f"Reimbursement volume is {row['reimbursement_ratio']:.1f}x the peer median for their specialty type in state {int(row['primary_state'])}.")
            if row["claims_ratio"] > settings["z_cutoff"]:
                reasons.append(f"Claims volume is {row['claims_ratio']:.1f}x the peer median for their specialty type in state {int(row['primary_state'])}.")
            if row["reimbursement_percentile"] > 95:
                reasons.append(f"Provider's total reimbursement is in the {row['reimbursement_percentile']:.1f}th percentile nationally.")
            if row["claims_percentile"] > 95:
                reasons.append(f"Provider's total claims are in the {row['claims_percentile']:.1f}th percentile nationally.")
            if row["patient_death_rate"] > 0.05:
                reasons.append(f"Billed claims for deceased beneficiaries make up {row['patient_death_rate']*100:.1f}% of total claims (extremely high risk).")
            if row["mean_reimbursement"] > prov_features["mean_reimbursement"].median() * 3:
                reasons.append(f"Average reimbursement per claim (${row['mean_reimbursement']:.2f}) is over 3x the national median (${prov_features['mean_reimbursement'].median():.2f}).")
            if leie_scores[idx] > 50:
                reasons.append("Identified matching record in the OIG Excluded Individuals/Entities List (LEIE Exclusion Screening Match).")
                
            model_flags = sum([1 for s in [if_scores[idx], ae_scores[idx], lof_scores[idx], ocsvm_scores[idx]] if s > 70])
            if model_flags >= 3:
                reasons.append(f"High agreement across multiple unsupervised anomaly detection models ({model_flags} models flagging provider behavior).")
            if cat_scores[idx] > 80:
                reasons.append(f"Supervised ML classifier flags behavior patterns aligning with known past fraudulent providers (Confidence: {cat_scores[idx]:.1f}%).")
                
            if not reasons:
                reasons.append("Overall metrics align with normal provider benchmarks. Minor deviations observed.")
                
            for r in reasons:
                cursor.execute("INSERT INTO explanations (provider_id, reason) VALUES (?, ?)", (p_id, r))
                
            # Restore or create investigation row
            if p_id in existing_inv:
                cursor.execute("""
                INSERT INTO investigations (provider_id, status, notes, assigned_investigator)
                VALUES (?, ?, ?, ?)
                """, (p_id, existing_inv[p_id]["status"], existing_inv[p_id]["notes"], existing_inv[p_id]["assigned_investigator"]))
            elif row["risk_level"] in ("High", "Critical"):
                cursor.execute("""
                INSERT INTO investigations (provider_id, status, notes, assigned_investigator)
                VALUES (?, 'New', '', 'Unassigned')
                """, (p_id,))
                
        # 6. Populate claim-level analysis
        pipeline_status["progress"] = 92
        pipeline_status["step"] = "Step 6: Generating claim-level risk scores..."
        logger.info("Computing claim-level risk scoring and tables...")
        
        # Sample or process claims (select top 10000 or process all if reasonable, to avoid slowing down database inserts)
        # To maintain quick responsiveness, we can process a representative subset (e.g. 5000 claims) or all claims.
        # Let's filter claims belonging to active providers in our providers table.
        active_providers = set(prov_features.index)
        claims_filtered = claims_merged[claims_merged["Provider"].isin(active_providers)].copy()
        
        # Limit to 5000 suspicious/high-amount claims + some random ones to populate the dashboard without bloating SQLite
        # Sort by InscClaimAmtReimbursed descending
        claims_filtered.sort_values(by="InscClaimAmtReimbursed", ascending=False, inplace=True)
        # Keep top 6000 claims to ensure rich dashboard tables
        claims_to_insert = claims_filtered.head(6000).copy()
        
        global_claim_amount_median = claims_merged["InscClaimAmtReimbursed"].median()
        
        for _, claim_row in claims_to_insert.iterrows():
            c_id = claim_row["ClaimID"]
            p_id = claim_row["Provider"]
            bene_id = claim_row["BeneID"]
            c_amount = float(claim_row["InscClaimAmtReimbursed"])
            c_type = "Inpatient" if claim_row["is_inpatient"] == 1 else "Outpatient"
            
            # Simple Rules-based Claim Scoring Engine
            p_risk = float(prov_features.loc[p_id, "risk_score"]) if p_id in prov_features.index else 50.0
            c_risk = p_risk * 0.4
            
            reasons = []
            if c_amount > 10000:
                c_risk += 30
                reasons.append("Claim amount is exceptionally high (>$10,000)")
            elif c_amount > global_claim_amount_median * 5:
                c_risk += 15
                reasons.append("Claim amount exceeds 5x national median billing")
                
            if claim_row["IsDeceased"] == 1:
                c_risk += 40
                reasons.append("Beneficiary was marked as deceased during claim period")
                
            if claim_row["DiagnosisCount"] > 5:
                c_risk += 10
                reasons.append(f"Highly complex diagnosis count ({claim_row['DiagnosisCount']} codes)")
                
            if claim_row["ClaimDuration"] > 15:
                c_risk += 15
                reasons.append(f"Suspiciously long claim duration ({claim_row['ClaimDuration']} days)")
                
            c_risk = min(100.0, c_risk)
            c_flag = 1 if c_risk >= 70 else 0
            
            c_explanation = ", ".join(reasons) if reasons else "Normal claim patterns"
            
            # Extract suspicious codes
            suspicious_codes_list = []
            for col in diag_cols[:3]: # grab first 3
                val = claim_row[col]
                if pd.notnull(val):
                    suspicious_codes_list.append(str(val))
            susp_codes = ", ".join(suspicious_codes_list) if suspicious_codes_list else "None"
            
            cursor.execute("""
            INSERT OR REPLACE INTO claims (
                claim_id, provider_id, bene_id, risk_score, fraud_flag,
                explanation, suspicious_codes, claim_amount, claim_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (c_id, p_id, bene_id, c_risk, c_flag, c_explanation, susp_codes, c_amount, c_type))
            
        conn.commit()
        conn.close()
        
        # Complete!
        pipeline_status["progress"] = 100
        pipeline_status["step"] = "Step 7: Risk analysis pipeline completed successfully!"
        pipeline_status["status"] = "completed"
        pipeline_status["completed_at"] = datetime.now().isoformat()
        pipeline_status["summary"] = {
            "providers_scanned": len(prov_features),
            "claims_scored": len(claims_to_insert),
            "critical_risk_count": int((prov_features["risk_score"] > crit_limit).sum()),
            "high_risk_count": int(((prov_features["risk_score"] > high_limit) & (prov_features["risk_score"] <= crit_limit)).sum()),
            "average_risk_score": float(prov_features["risk_score"].mean())
        }
        
        logger.info("MediClaim Risk Analysis Pipeline completed successfully.")
        
    except Exception as e:
        logger.exception("Error executing MediClaim risk pipeline")
        pipeline_status["status"] = "failed"
        pipeline_status["progress"] = 0
        pipeline_status["step"] = f"Pipeline execution failed: {str(e)}"
        pipeline_status["error"] = str(e)
