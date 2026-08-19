import os
import gc
import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import logging
import json
from catboost import CatBoostClassifier
from datetime import datetime
from pathlib import Path
from sklearn.svm import OneClassSVM
from app.explanation_service import build_claim_explanation, build_provider_explanation

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MediClaimPipeline")

# Autoencoder definition (for provider-level PyTorch AE)
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

# Claims Autoencoder definition
class ClaimAEModel(nn.Module):
    def __init__(self, input_dim=12):
        super(ClaimAEModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
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
LEIE_FILE = os.path.join(DATA_DIR, "leie_clean_specialty_filled.csv")


def normalize_npi(value):
    if pd.isna(value):
        return ""
    normalized = str(value).strip()
    if normalized.endswith(".0"):
        normalized = normalized[:-2]
    return normalized if normalized.isdigit() and int(normalized) > 0 else ""


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_leie_npi_index(filepath=LEIE_FILE):
    if not os.path.exists(filepath):
        logger.warning("LEIE source file not found: %s", filepath)
        return set()

    leie = pd.read_csv(filepath, usecols=["NPI", "IS_CURRENTLY_EXCLUDED"], low_memory=False)
    current = leie[leie["IS_CURRENTLY_EXCLUDED"].astype(str).isin({"1", "True", "true"})]
    return {npi for npi in current["NPI"].map(normalize_npi) if npi}


def calculate_leie_scores(provider_ids, labels_df, filepath=LEIE_FILE):
    leie_npis = load_leie_npi_index(filepath)
    scores = np.zeros(len(provider_ids), dtype=float)
    if not leie_npis:
        return scores

    npi_columns = {
        column.lower(): column
        for column in labels_df.columns
        if column.lower() in {"npi", "provider_npi", "providernpi", "npi_id"}
    }
    if not npi_columns or "Provider" not in labels_df.columns:
        logger.info("LEIE screening skipped: no provider NPI column is available in the active data.")
        return scores

    npi_column = next(iter(npi_columns.values()))
    npi_by_provider = (
        labels_df[["Provider", npi_column]]
        .assign(_normalized_npi=lambda frame: frame[npi_column].map(normalize_npi))
        .set_index("Provider")["_normalized_npi"]
        .to_dict()
    )
    matched = 0
    for index, provider_id in enumerate(provider_ids):
        if npi_by_provider.get(provider_id, "") in leie_npis:
            scores[index] = 100.0
            matched += 1
    logger.info("LEIE screening completed: %s exact NPI matches.", matched)
    return scores

def get_fallback_rag_pattern(category: str) -> dict:
    fallback_patterns = {
        "Excessive Claim Payment": {
            "pattern_name": "Upcoding of Services / Phantom Billing",
            "business_reason": "billing for a higher level of service than actually delivered or for services not rendered.",
            "severity": "High"
        },
        "Excessive Submitted Charges": {
            "pattern_name": "Billing Inflation",
            "business_reason": "submitting inflated charges for standard clinical encounters.",
            "severity": "Medium"
        },
        "High Charge Allowed Ratio": {
            "pattern_name": "Charge-to-Allowed Discrepancy",
            "business_reason": "gross inflation of submitted charges relative to contracted fee schedules.",
            "severity": "Medium"
        },
        "Unusual Service Intensity": {
            "pattern_name": "Unbundling / Service Fragmentation",
            "business_reason": "submitting multiple billing codes for group procedures to inflate reimbursement.",
            "severity": "High"
        },
        "Abnormally High Diagnosis Count": {
            "pattern_name": "Diagnosis Coding Intensity Inflation",
            "business_reason": "adding unrelated diagnosis codes to make a claim appear more complex and warrant higher reimbursement.",
            "severity": "High"
        }
    }
    return fallback_patterns.get(category, {
        "pattern_name": "General Billing Anomaly",
        "business_reason": "unusual billing behavior deviating from national peer baselines.",
        "severity": "Medium"
    })

def load_csv_optimized(filepath, desired_cols):
    if not os.path.exists(filepath):
        return None
    header_df = pd.read_csv(filepath, nrows=0)
    cols_to_load = [c for c in desired_cols if c in header_df.columns]
    df = pd.read_csv(filepath, usecols=cols_to_load)
    
    # Downcast types
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df

def run_risk_pipeline():
    global pipeline_status
    try:
        pipeline_status["status"] = "running"
        pipeline_status["progress"] = 5
        pipeline_status["step"] = "Step 1: Checking uploaded datasets and default files..."
        pipeline_status["error"] = None
        pipeline_status["summary"] = None
        
        logger.info("Starting MediClaim Risk Analysis Pipeline...")
        
        # Target columns to load
        claim_target_cols = [
            "ClaimID", "Provider", "BeneID", "InscClaimAmtReimbursed", "DeductibleAmtPaid",
            "ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt",
            "AttendingPhysician", "OperatingPhysician", "OtherPhysician"
        ] + [f"ClmDiagnosisCode_{i}" for i in range(1, 11)] + [f"ClmProcedureCode_{i}" for i in range(1, 7)]
        
        bene_target_cols = [
            "BeneID", "DOB", "DOD", "Gender", "Race", "State", "County", "RenalDiseaseIndicator",
            "IPAnnualReimbursementAmt", "IPAnnualDeductibleAmt", "OPAnnualReimbursementAmt", "OPAnnualDeductibleAmt"
        ] + [f"ChronicCond_{c}" for c in ["Alzheimer", "Heartfailure", "KidneyDisease", "Cancer", "ObstrPulmonary", "Depression", "Diabetes", "IschemicHeart", "Osteoporasis", "rheumatoidarthritis", "stroke"]]
        
        # Check files
        claims_file = os.path.join(UPLOADS_DIR, "claims.csv")
        beneficiary_file = os.path.join(UPLOADS_DIR, "beneficiary.csv")
        provider_file = os.path.join(UPLOADS_DIR, "provider.csv")
        
        # Load claims
        if not os.path.exists(claims_file):
            logger.info("No uploaded claims.csv found. Using default dataset files.")
            in_file = os.path.join(DATA_DIR, "Train_Inpatientdata-1542865627584.csv")
            out_file = os.path.join(DATA_DIR, "Train_Outpatientdata-1542865627584.csv")
            if not os.path.exists(in_file) or not os.path.exists(out_file):
                raise Exception("Default inpatient/outpatient claims CSV files not found.")
            
            in_df = load_csv_optimized(in_file, claim_target_cols)
            in_df["is_inpatient"] = np.int8(1)
            
            out_df = load_csv_optimized(out_file, claim_target_cols)
            out_df["is_inpatient"] = np.int8(0)
            
            claims_df = pd.concat([in_df, out_df], ignore_index=True)
            del in_df, out_df
            gc.collect()
        else:
            logger.info(f"Using uploaded claims file: {claims_file}")
            claims_df = load_csv_optimized(claims_file, claim_target_cols + ["is_inpatient"])
            if "is_inpatient" not in claims_df.columns:
                claims_df["is_inpatient"] = claims_df.apply(
                    lambda r: 1 if pd.notnull(r.get("AdmissionDt")) or pd.notnull(r.get("DischargeDt")) else 0,
                    axis=1
                ).astype('int8')
            else:
                claims_df["is_inpatient"] = claims_df["is_inpatient"].astype('int8')
        
        # Load beneficiaries
        if not os.path.exists(beneficiary_file):
            logger.info("No uploaded beneficiary.csv found. Using default beneficiary file.")
            bene_path = os.path.join(DATA_DIR, "Train_Beneficiarydata-1542865627584.csv")
            if not os.path.exists(bene_path):
                raise Exception("Default beneficiary CSV file not found.")
            bene_df = load_csv_optimized(bene_path, bene_target_cols)
        else:
            logger.info(f"Using uploaded beneficiary file: {beneficiary_file}")
            bene_df = load_csv_optimized(beneficiary_file, bene_target_cols)
            
        # Load provider labels
        if not os.path.exists(provider_file):
            logger.info("No uploaded provider.csv labels found. Using default labels file.")
            labels_path = os.path.join(DATA_DIR, "Train-1542865627584.csv")
            if not os.path.exists(labels_path):
                raise Exception("Default provider labels CSV file not found.")
            labels_df = pd.read_csv(labels_path)
        else:
            logger.info(f"Using uploaded provider file: {provider_file}")
            labels_df = pd.read_csv(provider_file)
            
        pipeline_status["progress"] = 15
        pipeline_status["step"] = "Step 2: Preprocessing datasets..."
        
        # 1. Process Beneficiary Chronic Conditions
        chronic_cols = [c for c in bene_df.columns if c.startswith("ChronicCond_")]
        for col in chronic_cols:
            bene_df[col] = bene_df[col].map({1: 1, 2: 0}).fillna(0).astype('int8')
            
        bene_df["RenalDiseaseIndicator"] = bene_df["RenalDiseaseIndicator"].map({"0": 0, "Y": 1, 0: 0, 1: 1}).fillna(0).astype('int8')
        if len(chronic_cols) > 0:
            bene_df["ChronicCondCount"] = bene_df[chronic_cols].sum(axis=1).astype('int8')
        else:
            bene_df["ChronicCondCount"] = np.int8(0)
            
        bene_df["DOB"] = pd.to_datetime(bene_df["DOB"], errors="coerce")
        bene_df["Age"] = (2009 - bene_df["DOB"].dt.year).fillna(65).astype('int16')
        bene_df["IsDeceased"] = bene_df["DOD"].notnull().astype('int8')
        
        # 2. Process Dates and Claim Duration
        claims_df["ClaimStartDt"] = pd.to_datetime(claims_df["ClaimStartDt"], errors="coerce")
        claims_df["ClaimEndDt"] = pd.to_datetime(claims_df["ClaimEndDt"], errors="coerce")
        claims_df["ClaimDuration"] = (claims_df["ClaimEndDt"] - claims_df["ClaimStartDt"]).dt.days + 1
        claims_df["ClaimDuration"] = claims_df["ClaimDuration"].fillna(1).clip(lower=1).astype('int32')
        
        admission_col = claims_df["AdmissionDt"] if "AdmissionDt" in claims_df.columns else pd.Series(np.nan, index=claims_df.index)
        discharge_col = claims_df["DischargeDt"] if "DischargeDt" in claims_df.columns else pd.Series(np.nan, index=claims_df.index)
        claims_df["AdmissionDt"] = pd.to_datetime(admission_col, errors="coerce")
        claims_df["DischargeDt"] = pd.to_datetime(discharge_col, errors="coerce")
        claims_df["AdmissionDuration"] = (claims_df["DischargeDt"] - claims_df["AdmissionDt"]).dt.days + 1
        claims_df["AdmissionDuration"] = claims_df["AdmissionDuration"].fillna(0).clip(lower=0).astype('int32')
        
        diag_cols = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
        proc_cols = [f"ClmProcedureCode_{i}" for i in range(1, 7)]
        
        # Ensure columns exist
        for c in diag_cols + proc_cols:
            if c not in claims_df.columns:
                claims_df[c] = np.nan
                
        claims_df["DiagnosisCount"] = claims_df[diag_cols].notnull().sum(axis=1).astype('int8')
        claims_df["ProcedureCount"] = claims_df[proc_cols].notnull().sum(axis=1).astype('int8')
        
        physician_fields = ["AttendingPhysician", "OperatingPhysician", "OtherPhysician"]
        present_phys_fields = [f for f in physician_fields if f in claims_df.columns]
        claims_df["PhysicianCount"] = claims_df[present_phys_fields].notnull().sum(axis=1).astype('int8')
        
        # Merge claims and beneficiaries
        logger.info("Merging claims and beneficiaries...")
        bene_cols_to_merge = ["BeneID", "Age", "Gender", "Race", "State", "County", "ChronicCondCount", "IsDeceased", "RenalDiseaseIndicator", "IPAnnualReimbursementAmt", "IPAnnualDeductibleAmt", "OPAnnualReimbursementAmt", "OPAnnualDeductibleAmt"]
        
        # Keep only the subset to save memory
        bene_df_subset = bene_df[bene_cols_to_merge].copy()
        del bene_df
        gc.collect()
        
        claims_merged = pd.merge(
            claims_df, 
            bene_df_subset, 
            on="BeneID", 
            how="left"
        )
        del bene_df_subset
        gc.collect()
        
        # Initialize RAG Engine if possible
        rag_engine = None
        try:
            from rag_engine import FraudRAGEngine
            rag_engine = FraudRAGEngine()
            kb_path = os.path.join(os.path.dirname(__file__), "db", "fraud_knowledge_base.json")
            if os.path.exists(kb_path):
                rag_engine.load_knowledge_base(Path(kb_path))
                rag_engine.build_index()
                logger.info("FAISS Cosine RAG Engine initialized successfully.")
        except Exception as e:
            logger.warning(f"Could not load RAG engine: {e}. Fallback mapping will be used.")

        # =====================================================================
        # PIPELINE 1: CLAIM-LEVEL FRAUD DETECTION
        # =====================================================================
        pipeline_status["progress"] = 30
        pipeline_status["step"] = "Step 3: Running Pipeline 1 – Claim-level fraud detection..."
        logger.info("Executing Pipeline 1 (Claim-level scoring)...")
        
        # Feature columns for claim autoencoder
        claim_feature_cols = [
            "InscClaimAmtReimbursed",
            "DeductibleAmtPaid",
            "ClaimDuration",
            "DiagnosisCount",
            "ProcedureCount",
            "PhysicianCount",
            "AdmissionDuration", # Maps to HospitalStayDays
            "IPAnnualReimbursementAmt",
            "IPAnnualDeductibleAmt",
            "OPAnnualReimbursementAmt",
            "OPAnnualDeductibleAmt",
            "ChronicCondCount" # Maps to ChronicConditionCount
        ]
        
        # Clean and construct features dataframe
        X_claims_df = claims_merged[claim_feature_cols].copy()
        X_claims_df.rename(columns={"AdmissionDuration": "HospitalStayDays", "ChronicCondCount": "ChronicConditionCount"}, inplace=True)
        
        # Handle nulls/infinities
        X_claims_df = X_claims_df.fillna(0)
        X_claims_df = X_claims_df.replace([np.inf, -np.inf], 0)
        X_claims_df = X_claims_df.clip(lower=0)
        
        # Log transformation
        X_claims_log = np.log1p(X_claims_df)
        
        # Load claim scaler and scale
        claim_scaler = joblib.load(os.path.join(ML_MODELS_DIR, "claim_scaler.pkl"))
        X_claims_scaled = claim_scaler.transform(X_claims_log)
        
        # Free up log data
        del X_claims_log
        gc.collect()
        
        # 1. Claim Autoencoder (PyTorch - Dynamic execution/training)
        # Using native PyTorch to avoid Keras 2 vs Keras 3 version compatibility issues
        logger.info("Initializing and training native PyTorch Claim Autoencoder model...")
        claim_ae_model = ClaimAEModel(input_dim=12)
        claim_ae_model.train()
        
        # Use simple SGD/Adam optimizer to fit on the dataset dynamically
        optimizer = torch.optim.Adam(claim_ae_model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        X_tensor = torch.tensor(X_claims_scaled, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X_tensor, X_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)
        
        # Train for 5 epochs (very fast, under 1 sec)
        for epoch in range(5):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                preds = claim_ae_model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                
        claim_ae_model.eval()
        with torch.no_grad():
            X_reconstructed = claim_ae_model(X_tensor).numpy()
            
        claim_ae_errors = np.mean((X_claims_scaled - X_reconstructed)**2, axis=1)
        
        # Clean up AE model / tensors
        del claim_ae_model, X_reconstructed, X_tensor, dataset, loader
        gc.collect()
        
        # Scale to 0-100 based on standard percentile thresholding
        p95_error = np.percentile(claim_ae_errors, 95) if len(claim_ae_errors) > 0 else 1.0
        claim_ae_scores = np.clip((claim_ae_errors / p95_error) * 85.0, 0, 100)
        del claim_ae_errors
        gc.collect()
        
        # 2. Claim Isolation Forest (scikit-learn)
        claim_iforest = joblib.load(os.path.join(ML_MODELS_DIR, "claim_iforest.pkl"))
        claim_if_decision = claim_iforest.decision_function(X_claims_scaled)
        claim_if_scores = np.clip((0.2 - claim_if_decision) / 0.5 * 100, 0, 100)
        del claim_iforest, claim_if_decision
        gc.collect()
        
        # 3. Claim One-Class SVM (scikit-learn - fit dynamically for current batch)
        claim_ocsvm = OneClassSVM(kernel='rbf', nu=0.03)
        # Fit on a subset to avoid slow fitting if batch is extremely large
        svm_fit_limit = min(15000, len(X_claims_scaled))
        claim_ocsvm.fit(X_claims_scaled[:svm_fit_limit])
        claim_ocsvm_decision = claim_ocsvm.decision_function(X_claims_scaled)
        claim_ocsvm_scores = np.clip((-claim_ocsvm_decision) * 100, 0, 100)
        del claim_ocsvm, claim_ocsvm_decision, X_claims_scaled
        gc.collect()
        
        # Claim Risk Fusion Score
        claims_merged["claim_risk_score"] = 0.4 * claim_ae_scores + 0.3 * claim_if_scores + 0.3 * claim_ocsvm_scores
        claims_merged["is_anomaly"] = (claims_merged["claim_risk_score"] >= 70.0).astype(int)
        
        del claim_ae_scores, claim_if_scores, claim_ocsvm_scores
        gc.collect()
        
        def classify_claim_risk(s):
            if s <= 30.0: return "Low"
            elif s <= 65.0: return "Medium"
            elif s <= 85.0: return "High"
            else: return "Critical"
            
        claims_merged["risk_category"] = claims_merged["claim_risk_score"].apply(classify_claim_risk)
        
        # Compute dynamic feature percentiles for rule explanations
        percentile_thresholds = {}
        for col in claim_feature_cols:
            percentile_thresholds[col] = {
                "p95": claims_merged[col].quantile(0.95),
                "p99": claims_merged[col].quantile(0.99)
            }
            
        # =====================================================================
        # PIPELINE 2: PROVIDER-LEVEL FRAUD DETECTION
        # =====================================================================
        pipeline_status["progress"] = 55
        pipeline_status["step"] = "Step 4: Running Pipeline 2 – Provider-level features aggregation..."
        logger.info("Executing Pipeline 2 (Provider-level scoring)...")
        
        # Group claims by provider to aggregate features
        provider_groups = claims_merged.groupby("Provider")
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
        prov_features["unique_diagnosis_codes"] = diag_melted.groupby("Provider")["value"].nunique().reindex(prov_features.index).fillna(0)
        del diag_melted
        gc.collect()
        
        proc_melted = claims_merged[["Provider"] + proc_cols].melt(id_vars="Provider").dropna()
        prov_features["unique_procedure_codes"] = proc_melted.groupby("Provider")["value"].nunique().reindex(prov_features.index).fillna(0)
        del proc_melted
        gc.collect()
        
        # Demographics
        prov_features["mean_patient_age"] = provider_groups["Age"].mean()
        prov_features["mean_patient_chronic_conds"] = provider_groups["ChronicCondCount"].mean()
        prov_features["patient_death_rate"] = provider_groups["IsDeceased"].mean()
        prov_features["renal_disease_rate"] = provider_groups["RenalDiseaseIndicator"].mean()
        
        # Location features
        prov_features["unique_states"] = provider_groups["State"].nunique()
        prov_features["unique_counties"] = provider_groups["County"].nunique()
        
        # Location Mode
        state_counts = claims_merged.groupby(["Provider", "State"]).size().reset_index(name="count")
        primary_states = state_counts.sort_values("count", ascending=False).drop_duplicates("Provider").set_index("Provider")["State"]
        prov_features["primary_state"] = primary_states.reindex(prov_features.index).fillna(-1).astype(int)
        del state_counts, primary_states
        gc.collect()
        
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
                
        # Load Provider models and scale
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
        
        # Individual provider scores
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
        
        # Peer Benchmarks
        prov_features["provider_type"] = prov_features["inpatient_ratio"].apply(lambda r: "Inpatient-heavy" if r >= 0.2 else "Outpatient-heavy")
        prov_features["peer_group"] = prov_features["provider_type"] + "_" + prov_features["primary_state"].astype(int).astype(str)
        
        prov_features = prov_features.join(peer_medians, on="peer_group", how="left")
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
        
        robust_z_reimb = (prov_features["total_reimbursement"] - medians["total_reimbursement"]) / mads["total_reimbursement"]
        robust_z_claims = (prov_features["total_claims"] - medians["total_claims"]) / mads["total_claims"]
        robust_z_cpb = (prov_features["claims_per_beneficiary"] - medians["claims_per_beneficiary"]) / mads["claims_per_beneficiary"]
        
        prov_features["robust_z_reimbursement_score"] = np.clip(robust_z_reimb / 5.0 * 100, 0, 100)
        prov_features["robust_z_claims_score"] = np.clip(robust_z_claims / 5.0 * 100, 0, 100)
        prov_features["robust_z_cpb_score"] = np.clip(robust_z_cpb / 5.0 * 100, 0, 100)
        
        statistical_score = (prov_features["robust_z_reimbursement_score"] + prov_features["robust_z_claims_score"] + prov_features["robust_z_cpb_score"]) / 3.0
        
        prov_features["reimbursement_percentile"] = prov_features["total_reimbursement"].rank(pct=True) * 100
        prov_features["claims_percentile"] = prov_features["total_claims"].rank(pct=True) * 100
        prov_features["beneficiary_percentile"] = prov_features["total_beneficiaries"].rank(pct=True) * 100
        
        # Screen provider NPIs against the supplied OIG exclusion source.
        # The default CMS provider IDs are not NPIs, so they produce no match.
        leie_scores = calculate_leie_scores(prov_features.index, labels_df)
                
        # =====================================================================
        # TEMPORAL BEHAVIORAL DRIFT MONITORING
        # =====================================================================
        pipeline_status["progress"] = 75
        pipeline_status["step"] = "Step 5: Running temporal drift tracking calculations..."
        logger.info("Computing monthly provider billing drift...")
        
        # Calculate monthly drift for each provider
        claims_merged["YearMonth"] = claims_merged["ClaimStartDt"].dt.to_period("M").astype(str)
        monthly_reimbursements = claims_merged.groupby(["Provider", "YearMonth"])["InscClaimAmtReimbursed"].sum()
        monthly_claims = claims_merged.groupby(["Provider", "YearMonth"])["ClaimID"].count()
        
        drift_records = []
        for p_id in prov_features.index:
            p_monthly_reimb = monthly_reimbursements.xs(p_id, level="Provider", drop_level=True) if p_id in monthly_reimbursements.index.levels[0] else pd.Series(dtype=float)
            p_monthly_claims = monthly_claims.xs(p_id, level="Provider", drop_level=True) if p_id in monthly_claims.index.levels[0] else pd.Series(dtype=float)
            
            # Reindex chronologically
            months_sorted = sorted(list(set(p_monthly_claims.index).union(p_monthly_reimb.index)))
            
            history_list = []
            claims_spike = 0.0
            reimb_spike = 0.0
            drift_score = 0.0
            
            if len(months_sorted) >= 2:
                # Compile history
                for m in months_sorted:
                    history_list.append({
                        "month": m,
                        "claims": int(p_monthly_claims.get(m, 0)),
                        "reimbursement": float(p_monthly_reimb.get(m, 0.0))
                    })
                
                # Compare latest month to historical median of prior months
                latest_month = months_sorted[-1]
                prior_months = months_sorted[:-1]
                
                latest_claims = p_monthly_claims.get(latest_month, 0)
                latest_reimb = p_monthly_reimb.get(latest_month, 0.0)
                
                median_prior_claims = np.median([p_monthly_claims.get(m, 0) for m in prior_months])
                median_prior_reimb = np.median([p_monthly_reimb.get(m, 0.0) for m in prior_months])
                
                claims_spike = latest_claims / max(1.0, median_prior_claims)
                reimb_spike = latest_reimb / max(1.0, median_prior_reimb)
                
                # Fuse into drift score
                drift_score = min(100.0, max(0.0, max(claims_spike, reimb_spike) - 1.0) * 15.0)
            else:
                for m in months_sorted:
                    history_list.append({
                        "month": m,
                        "claims": int(p_monthly_claims.get(m, 0)),
                        "reimbursement": float(p_monthly_reimb.get(m, 0.0))
                    })
            
            drift_level = "Low"
            if drift_score >= 85.0: drift_level = "Critical"
            elif drift_score >= 65.0: drift_level = "High"
            elif drift_score >= 35.0: drift_level = "Medium"
            
            drift_records.append({
                "provider_id": p_id,
                "drift_score": drift_score,
                "drift_level": drift_level,
                "claims_spike_ratio": claims_spike,
                "reimbursement_spike_ratio": reimb_spike,
                "coding_shift_index": 0.0,
                "historical_monthly_data": json.dumps(history_list)
            })
            
        drift_df = pd.DataFrame(drift_records).set_index("provider_id")
        
        # Load weights from Settings
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM settings WHERE id = 1")
        settings = dict(cursor.fetchone())
        conn.close()
        
        cat_w = settings["catboost_weight"]
        if_w = settings["iforest_weight"]
        lof_w = settings["lof_weight"]
        robust_z_w = settings["robust_z_weight"]
        peer_w = settings["peer_benchmark_weight"]
        leie_w = settings["leie_weight"]
        
        # Check if configured as percentages (e.g. sum is > 2.0)
        total_w = cat_w + if_w + lof_w + robust_z_w + peer_w + leie_w
        if total_w > 2.0:
            cat_w /= 100.0
            if_w /= 100.0
            lof_w /= 100.0
            robust_z_w /= 100.0
            peer_w /= 100.0
            leie_w /= 100.0
        
        # Combined Model Score
        ml_score = (if_scores + lof_scores + cat_scores) / 3.0
        
        # Combined Risk Score
        final_risk_score = (
            cat_w * cat_scores +
            if_w * if_scores +
            lof_w * lof_scores +
            robust_z_w * statistical_score +
            peer_w * peer_score +
            leie_w * leie_scores
        )
        
        # Add a billing drift penalty if temporal drift is critical/high
        drift_penalty = drift_df["drift_score"].reindex(prov_features.index).fillna(0.0).values * 0.1
        final_risk_score = np.clip(final_risk_score + drift_penalty, 0, 100)
        
        prov_features["ml_score"] = ml_score
        prov_features["statistical_score"] = statistical_score
        prov_features["peer_score"] = peer_score
        prov_features["risk_score"] = final_risk_score
        
        high_limit = settings["high_risk_limit"]
        crit_limit = settings["crit_risk_limit"]
        
        def classify_risk_custom(s):
            if s <= 30.0: return "Low"
            elif s <= high_limit: return "Medium"
            elif s <= crit_limit: return "High"
            else: return "Critical"
                
        prov_features["risk_level"] = prov_features["risk_score"].apply(classify_risk_custom)
        
        # =====================================================================
        # WRITE TO DATABASE & EXPLAINABILITY
        # =====================================================================
        pipeline_status["progress"] = 85
        pipeline_status["step"] = "Step 6: Populating SQLite database and generating explanations..."
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS provider_explanations (
            provider_id TEXT PRIMARY KEY,
            explanation_json TEXT NOT NULL,
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        try:
            cursor.execute("SELECT explanation_json FROM claims LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE claims ADD COLUMN explanation_json TEXT")
        
        # Clear provider-level tables to avoid dirty merges
        cursor.execute("DELETE FROM providers")
        cursor.execute("DELETE FROM model_scores")
        cursor.execute("DELETE FROM peer_benchmarks")
        cursor.execute("DELETE FROM explanations")
        cursor.execute("DELETE FROM provider_explanations")
        cursor.execute("DELETE FROM provider_drift")
        
        cursor.execute("SELECT provider_id, status, notes, assigned_investigator FROM investigations")
        existing_inv = {row[0]: {"status": row[1], "notes": row[2], "assigned_investigator": row[3]} for row in cursor.fetchall()}
        cursor.execute("DELETE FROM investigations")
        cursor.execute("DELETE FROM claims")
        
        # 1. Insert Providers and Provider Scores
        logger.info(f"Writing {len(prov_features)} providers and drift calculations to SQLite...")
        for idx, (p_id, row) in enumerate(prov_features.iterrows()):
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
            
            cursor.execute("""
            INSERT INTO peer_benchmarks (
                provider_id, reimbursement_ratio, claims_ratio, beneficiary_ratio,
                reimbursement_percentile, claims_percentile, beneficiary_percentile
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                p_id, float(row["reimbursement_ratio"]), float(row["claims_ratio"]), float(row["beneficiary_ratio"]),
                float(row["reimbursement_percentile"]), float(row["claims_percentile"]), float(row["beneficiary_percentile"])
            ))
            
            # Insert Provider Drift
            p_drift = drift_df.loc[p_id]
            cursor.execute("""
            INSERT INTO provider_drift (
                provider_id, drift_score, drift_level, claims_spike_ratio,
                reimbursement_spike_ratio, coding_shift_index, historical_monthly_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                p_id, float(p_drift["drift_score"]), p_drift["drift_level"],
                float(p_drift["claims_spike_ratio"]), float(p_drift["reimbursement_spike_ratio"]),
                float(p_drift["coding_shift_index"]), p_drift["historical_monthly_data"]
            ))
            
            # Generate provider explanations
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
            if p_drift["drift_score"] > 50:
                reasons.append(f"Significant billing drift detected: MoM reimbursement increased by {p_drift['reimbursement_spike_ratio']:.1f}x (Drift Score: {p_drift['drift_score']:.1f}).")
                
            model_flags = sum([1 for s in [if_scores[idx], ae_scores[idx], lof_scores[idx], ocsvm_scores[idx]] if s > 70])
            if model_flags >= 3:
                reasons.append(f"High agreement across multiple unsupervised anomaly detection models ({model_flags} models flagging provider behavior).")
            if cat_scores[idx] > 80:
                reasons.append(f"Supervised ML classifier flags behavior patterns aligning with known past fraudulent providers (Confidence: {cat_scores[idx]:.1f}%).")
                
            if not reasons:
                reasons.append("Overall metrics align with normal provider benchmarks. Minor deviations observed.")
                
            for r in reasons:
                cursor.execute("INSERT INTO explanations (provider_id, reason) VALUES (?, ?)", (p_id, r))

            provider_explanation = build_provider_explanation(
                provider={
                    "provider_id": p_id,
                    "risk_score": row["risk_score"],
                    "risk_level": row["risk_level"],
                    "total_claims": row["total_claims"],
                    "total_beneficiaries": row["total_beneficiaries"],
                    "total_reimbursement": row["total_reimbursement"],
                    "mean_reimbursement": row["mean_reimbursement"],
                },
                scores={
                    "isolation_score": if_scores[idx],
                    "lof_score": lof_scores[idx],
                    "ocsvm_score": ocsvm_scores[idx],
                    "autoencoder_score": ae_scores[idx],
                    "catboost_score": cat_scores[idx],
                    "statistical_score": row["statistical_score"],
                    "peer_score": row["peer_score"],
                    "leie_score": leie_scores[idx],
                },
                benchmarks={
                    "reimbursement_ratio": row["reimbursement_ratio"],
                    "claims_ratio": row["claims_ratio"],
                    "reimbursement_percentile": row["reimbursement_percentile"],
                    "claims_percentile": row["claims_percentile"],
                },
                drift=dict(p_drift),
                reasons=reasons,
            )
            cursor.execute(
                "INSERT INTO provider_explanations (provider_id, explanation_json) VALUES (?, ?)",
                (p_id, json.dumps(provider_explanation, default=json_default)),
            )
                
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
                """, (p_id, ))
                
        # 2. Insert Claims and Claim Explanations
        logger.info("Writing claim scoring, explainability and RAG interpretations to SQLite...")
        
        # Sort claims descending to prioritize high-risk ones
        claims_sorted = claims_merged.sort_values(by="claim_risk_score", ascending=False)
        # Process top 5000 claims to prevent DB bloating while keeping data rich
        claims_to_insert = claims_sorted.head(5000).copy()
        claim_medians = claims_to_insert.groupby("is_inpatient")["InscClaimAmtReimbursed"].median().to_dict()
        claim_counts = claims_to_insert.groupby("is_inpatient").size().to_dict()
        
        # Free memory of claims_merged and sorted claims
        del claims_merged, claims_sorted
        gc.collect()
        
        for _, claim_row in claims_to_insert.iterrows():
            c_id = claim_row["ClaimID"]
            p_id = claim_row["Provider"]
            bene_id = claim_row["BeneID"]
            c_amount = float(claim_row["InscClaimAmtReimbursed"])
            c_type = "Inpatient" if claim_row["is_inpatient"] == 1 else "Outpatient"
            c_date = str(claim_row["ClaimStartDt"].date()) if pd.notnull(claim_row["ClaimStartDt"]) else ""
            c_score = float(claim_row["claim_risk_score"])
            c_cat = claim_row["risk_category"]
            c_anomaly = int(claim_row["is_anomaly"])
            
            # Formulate claim explanations based on percentiles
            claim_reasons = []
            claim_reasons_categories = []
            
            if claim_row["InscClaimAmtReimbursed"] > percentile_thresholds["InscClaimAmtReimbursed"]["p99"]:
                claim_reasons.append(f"Claim payment amount is in the top 1% (value=${claim_row['InscClaimAmtReimbursed']:.2f}).")
                claim_reasons_categories.append("Excessive Claim Payment")
            if claim_row["DeductibleAmtPaid"] > percentile_thresholds["DeductibleAmtPaid"]["p99"]:
                claim_reasons.append(f"Deductible amount paid is in the top 1% (value=${claim_row['DeductibleAmtPaid']:.2f}).")
                claim_reasons_categories.append("Excessive Submitted Charges")
            if claim_row["ClaimDuration"] > percentile_thresholds["ClaimDuration"]["p95"]:
                claim_reasons.append(f"Claim duration exceeds 95% of claims ({int(claim_row['ClaimDuration'])} days).")
                claim_reasons_categories.append("High Charge Allowed Ratio")
            if claim_row["DiagnosisCount"] > percentile_thresholds["DiagnosisCount"]["p95"]:
                claim_reasons.append(f"Claim contains abnormally high diagnosis complexity ({int(claim_row['DiagnosisCount'])} codes).")
                claim_reasons_categories.append("Abnormally High Diagnosis Count")
            if claim_row["ProcedureCount"] > percentile_thresholds["ProcedureCount"]["p95"]:
                claim_reasons.append(f"Claim contains unusually high procedure frequency ({int(claim_row['ProcedureCount'])} codes).")
                claim_reasons_categories.append("Unusual Service Intensity")
            if claim_row["IsDeceased"] == 1:
                claim_reasons.append("Beneficiary was deceased during billing period.")
                claim_reasons_categories.append("Abnormally High Diagnosis Count")
                
            # Pad reasons
            while len(claim_reasons) < 3:
                claim_reasons.append("Clinical features align with baseline billing models.")
                claim_reasons_categories.append("Supporting Signal")
                
            # FAISS cosine RAG retrieval / fallback
            top_category = claim_reasons_categories[0]
            business_reason = ""
            rag_pattern_name = ""
            
            if rag_engine is not None and top_category != "Supporting Signal":
                try:
                    query_text = f"Anomalous claim flagged for {top_category}. Reasons: {claim_reasons[0]}"
                    rag_results = rag_engine.retrieve_top_patterns(query_text, top_k=1)
                    if rag_results:
                        rag_pattern_name = rag_results[0].pattern_name
                        business_reason = rag_results[0].business_reason
                except Exception as ex:
                    logger.warning(f"RAG search error: {ex}")
                    
            if not business_reason or not rag_pattern_name:
                # Use robust fallback
                fallback_pat = get_fallback_rag_pattern(top_category)
                rag_pattern_name = fallback_pat["pattern_name"]
                business_reason = fallback_pat["business_reason"]
                
            business_interpretation = (
                f"Claim {c_id} has a {c_score:.1f}/100 {c_cat} risk rating. "
                f"Recommended next step: "
                f"{('Review medical documentation, coding, and claim justification immediately.' if c_score >= 81 else 'Prioritize targeted documentation and coding review.' if c_score >= 61 else 'Monitor the claim and retain it for routine payment-integrity review.') }"
            )
            claim_explanation = build_claim_explanation(
                claim={
                    "claim_id": c_id,
                    "provider_id": p_id,
                    "risk_score": c_score,
                    "risk_category": c_cat,
                    "is_anomaly": c_anomaly,
                    "claim_amount": c_amount,
                    "explanation_1": claim_reasons[0],
                    "explanation_2": claim_reasons[1],
                },
                provider={
                    "risk_score": prov_features.loc[p_id, "risk_score"],
                    "risk_level": prov_features.loc[p_id, "risk_level"],
                    "reimbursement_ratio": prov_features.loc[p_id, "reimbursement_ratio"],
                },
                peer_claim_amount=claim_medians.get(claim_row["is_inpatient"], claims_to_insert["InscClaimAmtReimbursed"].median()),
                rule_reasons=claim_reasons,
                comparison_count=int(claim_counts.get(claim_row["is_inpatient"], len(claims_to_insert))),
                rag_pattern=rag_pattern_name if top_category != "Supporting Signal" else None,
            )
            
            cursor.execute("""
            INSERT INTO claims (
                claim_id, provider_id, bene_id, risk_score, risk_category,
                is_anomaly, explanation_1, explanation_2, explanation_3,
                business_interpretation, explanation_json, claim_amount, claim_type, claim_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                c_id, p_id, bene_id, c_score, c_cat, c_anomaly,
                claim_reasons[0], claim_reasons[1], claim_reasons[2],
                business_interpretation, json.dumps(claim_explanation, default=json_default), c_amount, c_type, c_date
            ))
            
        conn.commit()
        conn.close()
        
        # Complete!
        pipeline_status["progress"] = 100
        pipeline_status["step"] = "Step 7: Independent dual risk pipelines completed successfully!"
        pipeline_status["status"] = "completed"
        pipeline_status["completed_at"] = datetime.now().isoformat()
        pipeline_status["summary"] = {
            "providers_scanned": len(prov_features),
            "claims_scored": len(claims_to_insert),
            "critical_risk_count": int((prov_features["risk_score"] > crit_limit).sum()),
            "high_risk_count": int(((prov_features["risk_score"] > high_limit) & (prov_features["risk_score"] <= crit_limit)).sum()),
            "average_risk_score": float(prov_features["risk_score"].mean())
        }
        
        logger.info("MediClaim Dual-Pipeline Risk Analysis completed successfully.")
        
    except Exception as e:
        logger.exception("Error executing MediClaim dual-pipeline risk script")
        pipeline_status["status"] = "failed"
        pipeline_status["progress"] = 0
        pipeline_status["step"] = f"Pipeline execution failed: {str(e)}"
        pipeline_status["error"] = str(e)
