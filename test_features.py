import pandas as pd
import numpy as np
import os

data_dir = "e:/CTS - MediClaim/datas"

def load_and_preprocess():
    print("Loading datasets...")
    # Load Beneficiary Data
    bene_df = pd.read_csv(os.path.join(data_dir, "Train_Beneficiarydata-1542865627584.csv"))
    
    # Process Beneficiary Chronic Conditions
    # Map 1 -> 1 (Yes), 2 -> 0 (No)
    chronic_cols = [c for c in bene_df.columns if c.startswith("ChronicCond_")]
    for col in chronic_cols:
        bene_df[col] = bene_df[col].map({1: 1, 2: 0})
        
    bene_df["RenalDiseaseIndicator"] = bene_df["RenalDiseaseIndicator"].map({"0": 0, "Y": 1})
    bene_df["ChronicCondCount"] = bene_df[chronic_cols].sum(axis=1)
    
    # Age calculation
    bene_df["DOB"] = pd.to_datetime(bene_df["DOB"], errors="coerce")
    bene_df["Age"] = 2009 - bene_df["DOB"].dt.year
    
    # Is deceased
    bene_df["IsDeceased"] = bene_df["DOD"].notnull().astype(int)
    
    # Load Inpatient & Outpatient Data
    in_df = pd.read_csv(os.path.join(data_dir, "Train_Inpatientdata-1542865627584.csv"))
    out_df = pd.read_csv(os.path.join(data_dir, "Train_Outpatientdata-1542865627584.csv"))
    
    in_df["is_inpatient"] = 1
    out_df["is_inpatient"] = 0
    
    # Concatenate claims
    claims_df = pd.concat([in_df, out_df], ignore_index=True)
    
    print(f"Loaded {len(claims_df)} total claims ({len(in_df)} Inpatient, {len(out_df)} Outpatient).")
    
    # Process dates
    claims_df["ClaimStartDt"] = pd.to_datetime(claims_df["ClaimStartDt"], errors="coerce")
    claims_df["ClaimEndDt"] = pd.to_datetime(claims_df["ClaimEndDt"], errors="coerce")
    claims_df["ClaimDuration"] = (claims_df["ClaimEndDt"] - claims_df["ClaimStartDt"]).dt.days
    
    claims_df["AdmissionDt"] = pd.to_datetime(claims_df.get("AdmissionDt"), errors="coerce")
    claims_df["DischargeDt"] = pd.to_datetime(claims_df.get("DischargeDt"), errors="coerce")
    claims_df["AdmissionDuration"] = (claims_df["DischargeDt"] - claims_df["AdmissionDt"]).dt.days
    
    # Count diagnoses and procedures
    diag_cols = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
    proc_cols = [f"ClmProcedureCode_{i}" for i in range(1, 7)]
    
    claims_df["DiagnosisCount"] = claims_df[diag_cols].notnull().sum(axis=1)
    claims_df["ProcedureCount"] = claims_df[proc_cols].notnull().sum(axis=1)
    
    # Merge claims with beneficiary details
    print("Merging claims with beneficiary details...")
    claims_merged = pd.merge(claims_df, bene_df[["BeneID", "Age", "Gender", "Race", "State", "County", "ChronicCondCount", "IsDeceased", "RenalDiseaseIndicator"]], on="BeneID", how="left")
    
    print("Aggregating by Provider...")
    # Group by provider
    provider_groups = claims_merged.groupby("Provider")
    
    # Aggregate features
    prov_features = pd.DataFrame(index=provider_groups.indices.keys())
    prov_features.index.name = "Provider"
    
    # Volume Features
    prov_features["total_claims"] = provider_groups["ClaimID"].count()
    prov_features["inpatient_claims"] = provider_groups["is_inpatient"].sum()
    prov_features["outpatient_claims"] = prov_features["total_claims"] - prov_features["inpatient_claims"]
    prov_features["inpatient_ratio"] = prov_features["inpatient_claims"] / prov_features["total_claims"]
    prov_features["total_beneficiaries"] = provider_groups["BeneID"].nunique()
    prov_features["claims_per_beneficiary"] = prov_features["total_claims"] / prov_features["total_beneficiaries"]
    
    # Financial Features
    prov_features["total_reimbursement"] = provider_groups["InscClaimAmtReimbursed"].sum()
    prov_features["mean_reimbursement"] = provider_groups["InscClaimAmtReimbursed"].mean()
    prov_features["max_reimbursement"] = provider_groups["InscClaimAmtReimbursed"].max()
    
    prov_features["total_deductible"] = provider_groups["DeductibleAmtPaid"].sum()
    prov_features["mean_deductible"] = provider_groups["DeductibleAmtPaid"].mean()
    prov_features["reimbursement_per_beneficiary"] = prov_features["total_reimbursement"] / prov_features["total_beneficiaries"]
    
    # Physician Network Features
    prov_features["unique_attending"] = provider_groups["AttendingPhysician"].nunique()
    prov_features["unique_operating"] = provider_groups["OperatingPhysician"].nunique()
    prov_features["unique_other"] = provider_groups["OtherPhysician"].nunique()
    
    # Clinical Features
    prov_features["mean_diagnosis_count"] = provider_groups["DiagnosisCount"].mean()
    prov_features["mean_procedure_count"] = provider_groups["ProcedureCount"].mean()
    
    # Collect unique diagnosis codes per provider
    def count_unique_codes(df_group, prefix_cols):
        unique_vals = set()
        for c in prefix_cols:
            unique_vals.update(df_group[c].dropna().unique())
        return len(unique_vals)
    
    print("Computing unique diagnosis and procedure counts (this may take a moment)...")
    diag_counts = {}
    proc_counts = {}
    for prov, group in provider_groups:
        d_vals = set()
        for c in diag_cols:
            d_vals.update(group[c].dropna().unique())
        diag_counts[prov] = len(d_vals)
        
        p_vals = set()
        for c in proc_cols:
            p_vals.update(group[c].dropna().unique())
        proc_counts[prov] = len(p_vals)
        
    prov_features["unique_diagnosis_codes"] = pd.Series(diag_counts)
    prov_features["unique_procedure_codes"] = pd.Series(proc_counts)
    
    # Patient Demographics
    prov_features["mean_patient_age"] = provider_groups["Age"].mean()
    prov_features["mean_patient_chronic_conds"] = provider_groups["ChronicCondCount"].mean()
    prov_features["patient_death_rate"] = provider_groups["IsDeceased"].mean()
    prov_features["renal_disease_rate"] = provider_groups["RenalDiseaseIndicator"].mean()
    
    # Location Features
    prov_features["unique_states"] = provider_groups["State"].nunique()
    prov_features["unique_counties"] = provider_groups["County"].nunique()
    # Most common state (mode)
    prov_features["primary_state"] = provider_groups["State"].apply(lambda x: x.mode()[0] if not x.mode().empty else -1)
    
    # Duration
    prov_features["mean_claim_duration"] = provider_groups["ClaimDuration"].mean()
    prov_features["mean_admission_duration"] = provider_groups["AdmissionDuration"].mean().fillna(0)
    
    # Load labels
    labels = pd.read_csv(os.path.join(data_dir, "Train-1542865627584.csv"))
    labels["PotentialFraud"] = labels["PotentialFraud"].map({"Yes": 1, "No": 0})
    labels.set_index("Provider", inplace=True)
    
    # Join labels
    prov_features = prov_features.join(labels, how="inner")
    
    print(f"Finished feature engineering. Features shape: {prov_features.shape}")
    print(prov_features.head())
    
    # Save features
    prov_features.to_csv("e:/CTS - MediClaim/datas/engineered_train_features.csv")
    print("Saved engineered train features to CSV.")

if __name__ == "__main__":
    load_and_preprocess()
