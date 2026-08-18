from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datas" / "raw"
OUTPUT_DIR = ROOT / "processed"
OUTPUT_PATH = OUTPUT_DIR / "PROVIDER_ML_READY_KAGGLE.csv"


def safe_ratio(numerator: pd.Series | float, denominator: pd.Series | float) -> pd.Series | float:
    denom = pd.Series(denominator) if not isinstance(denominator, pd.Series) else denominator
    num = pd.Series(numerator) if not isinstance(numerator, pd.Series) else numerator
    result = num.divide(denom.replace(0, np.nan))
    return result.fillna(0.0)


def ensure_no_nulls(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].apply(
            lambda s: pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
        )

    object_cols = df.select_dtypes(include=[object]).columns.tolist()
    for col in object_cols:
        df[col] = df[col].fillna("UNKNOWN")

    if df.isnull().sum().sum() != 0:
        raise ValueError(f"Nulls remain after cleanup: {df.isnull().sum().sum()}")
    return df


def build_provider_dataset() -> pd.DataFrame:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    ip = pd.read_csv(DATA_DIR / "Train_Inpatientdata.csv")
    op = pd.read_csv(DATA_DIR / "Train_Outpatientdata.csv")
    bene = pd.read_csv(DATA_DIR / "Train_Beneficiarydata.csv")

    labels = train[["Provider", "PotentialFraud"]].copy()
    labels["Fraud_Label"] = labels["PotentialFraud"].str.strip().str.lower().map({"no": 0, "yes": 1}).fillna(0).astype(int)

    providers = labels[["Provider", "Fraud_Label"]].drop_duplicates().copy()

    ip = ip.copy()
    op = op.copy()
    ip["Provider"] = ip["Provider"].astype(str)
    op["Provider"] = op["Provider"].astype(str)

    ip["DeductibleAmtPaid"] = pd.to_numeric(ip["DeductibleAmtPaid"], errors="coerce")
    op["DeductibleAmtPaid"] = pd.to_numeric(op["DeductibleAmtPaid"], errors="coerce")
    ip["InscClaimAmtReimbursed"] = pd.to_numeric(ip["InscClaimAmtReimbursed"], errors="coerce")
    op["InscClaimAmtReimbursed"] = pd.to_numeric(op["InscClaimAmtReimbursed"], errors="coerce")

    ip_admit = pd.to_datetime(ip["AdmissionDt"], errors="coerce")
    ip_discharge = pd.to_datetime(ip["DischargeDt"], errors="coerce")
    ip["Length_Of_Stay"] = (ip_discharge - ip_admit).dt.days
    ip["Length_Of_Stay"] = pd.to_numeric(ip["Length_Of_Stay"], errors="coerce").fillna(0)

    ip_group = ip.groupby("Provider", as_index=False).agg(
        IP_Claim_Count=("ClaimID", "count"),
        IP_Unique_Beneficiaries=("BeneID", "nunique"),
        IP_Total_Reimbursement=("InscClaimAmtReimbursed", "sum"),
        IP_Avg_Reimbursement=("InscClaimAmtReimbursed", "mean"),
        IP_Total_Deductible=("DeductibleAmtPaid", "sum"),
        IP_Avg_Deductible=("DeductibleAmtPaid", "mean"),
        Average_Length_Of_Stay=("Length_Of_Stay", "mean"),
    )
    ip_group["Same_Day_Discharge_Count"] = ip.groupby("Provider")["Length_Of_Stay"].apply(lambda s: (s == 0).sum()).fillna(0).values
    ip_group["Same_Day_Discharge_Ratio"] = safe_ratio(ip_group["Same_Day_Discharge_Count"], ip_group["IP_Claim_Count"]).astype(float)

    op_group = op.groupby("Provider", as_index=False).agg(
        OP_Claim_Count=("ClaimID", "count"),
        OP_Unique_Beneficiaries=("BeneID", "nunique"),
        OP_Total_Reimbursement=("InscClaimAmtReimbursed", "sum"),
        OP_Avg_Reimbursement=("InscClaimAmtReimbursed", "mean"),
        OP_Total_Deductible=("DeductibleAmtPaid", "sum"),
        OP_Avg_Deductible=("DeductibleAmtPaid", "mean"),
    )

    claim_counts = providers[["Provider"]].merge(ip_group[["Provider", "IP_Claim_Count", "IP_Unique_Beneficiaries", "IP_Total_Reimbursement", "IP_Avg_Reimbursement", "IP_Total_Deductible", "IP_Avg_Deductible", "Average_Length_Of_Stay", "Same_Day_Discharge_Count", "Same_Day_Discharge_Ratio"]], on="Provider", how="left")
    claim_counts = claim_counts.merge(op_group[["Provider", "OP_Claim_Count", "OP_Unique_Beneficiaries", "OP_Total_Reimbursement", "OP_Avg_Reimbursement", "OP_Total_Deductible", "OP_Avg_Deductible"]], on="Provider", how="left")

    for col in [
        "IP_Claim_Count", "IP_Unique_Beneficiaries", "IP_Total_Reimbursement", "IP_Avg_Reimbursement",
        "IP_Total_Deductible", "IP_Avg_Deductible", "Average_Length_Of_Stay", "Same_Day_Discharge_Count",
        "OP_Claim_Count", "OP_Unique_Beneficiaries", "OP_Total_Reimbursement", "OP_Avg_Reimbursement",
        "OP_Total_Deductible", "OP_Avg_Deductible"
    ]:
        claim_counts[col] = pd.to_numeric(claim_counts[col], errors="coerce").fillna(0)

    claim_counts["Total_Claim_Count"] = claim_counts["IP_Claim_Count"].fillna(0) + claim_counts["OP_Claim_Count"].fillna(0)
    claim_counts["Total_Unique_Beneficiaries"] = claim_counts["IP_Unique_Beneficiaries"].fillna(0) + claim_counts["OP_Unique_Beneficiaries"].fillna(0)
    claim_counts["Claims_Per_Beneficiary"] = safe_ratio(claim_counts["Total_Claim_Count"], claim_counts["Total_Unique_Beneficiaries"]).astype(float)

    claim_counts["IP_Total_Reimbursement"] = claim_counts["IP_Total_Reimbursement"].fillna(0)
    claim_counts["OP_Total_Reimbursement"] = claim_counts["OP_Total_Reimbursement"].fillna(0)
    claim_counts["Total_Reimbursement"] = claim_counts["IP_Total_Reimbursement"] + claim_counts["OP_Total_Reimbursement"]
    claim_counts["IP_Avg_Reimbursement"] = claim_counts["IP_Avg_Reimbursement"].fillna(0)
    claim_counts["OP_Avg_Reimbursement"] = claim_counts["OP_Avg_Reimbursement"].fillna(0)
    claim_counts["Reimbursement_Per_Claim"] = safe_ratio(claim_counts["Total_Reimbursement"], claim_counts["Total_Claim_Count"]).astype(float)
    claim_counts["Reimbursement_Per_Beneficiary"] = safe_ratio(claim_counts["Total_Reimbursement"], claim_counts["Total_Unique_Beneficiaries"]).astype(float)

    claim_counts["IP_Total_Deductible"] = claim_counts["IP_Total_Deductible"].fillna(0)
    claim_counts["OP_Total_Deductible"] = claim_counts["OP_Total_Deductible"].fillna(0)
    claim_counts["Total_Deductible"] = claim_counts["IP_Total_Deductible"] + claim_counts["OP_Total_Deductible"]
    claim_counts["IP_Avg_Deductible"] = claim_counts["IP_Avg_Deductible"].fillna(0)
    claim_counts["OP_Avg_Deductible"] = claim_counts["OP_Avg_Deductible"].fillna(0)

    claim_counts["IP_Claim_Share"] = safe_ratio(claim_counts["IP_Claim_Count"], claim_counts["Total_Claim_Count"]).astype(float)
    claim_counts["OP_Claim_Share"] = safe_ratio(claim_counts["OP_Claim_Count"], claim_counts["Total_Claim_Count"]).astype(float)

    diag_cols = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
    proc_cols = [f"ClmProcedureCode_{i}" for i in range(1, 7)]

    diag_all = pd.concat([ip[["Provider"] + diag_cols].copy(), op[["Provider"] + diag_cols].copy()], ignore_index=True)
    diag_all = diag_all.melt(id_vars="Provider", value_vars=diag_cols, value_name="DiagnosisCode").dropna(subset=["DiagnosisCode"])
    diag_all["DiagnosisCode"] = diag_all["DiagnosisCode"].astype(str).str.strip()
    diag_all = diag_all[diag_all["DiagnosisCode"] != "nan"]
    diagnosis_summary = diag_all.groupby("Provider").agg(
        Unique_Diagnosis_Codes=("DiagnosisCode", "nunique"),
        Total_Diagnosis_Codes=("DiagnosisCode", "count"),
    ).reset_index()

    proc_all = pd.concat([ip[["Provider"] + proc_cols].copy(), op[["Provider"] + proc_cols].copy()], ignore_index=True)
    proc_all = proc_all.melt(id_vars="Provider", value_vars=proc_cols, value_name="ProcedureCode").dropna(subset=["ProcedureCode"])
    proc_all["ProcedureCode"] = proc_all["ProcedureCode"].astype(str).str.strip()
    proc_all = proc_all[proc_all["ProcedureCode"] != "nan"]
    procedure_summary = proc_all.groupby("Provider").agg(
        Unique_Procedure_Codes=("ProcedureCode", "nunique"),
        Total_Procedure_Codes=("ProcedureCode", "count"),
    ).reset_index()

    combined_features = providers[["Provider"]].merge(claim_counts, on="Provider", how="left")
    combined_features = combined_features.merge(diagnosis_summary, on="Provider", how="left")
    combined_features = combined_features.merge(procedure_summary, on="Provider", how="left")

    combined_features["Unique_Diagnosis_Codes"] = combined_features["Unique_Diagnosis_Codes"].fillna(0)
    combined_features["Diagnosis_Diversity"] = safe_ratio(combined_features["Unique_Diagnosis_Codes"], combined_features["Total_Claim_Count"]).astype(float)
    combined_features["Avg_Diagnosis_Codes_Per_Claim"] = safe_ratio(combined_features["Total_Diagnosis_Codes"], combined_features["Total_Claim_Count"]).astype(float)

    combined_features["Unique_Procedure_Codes"] = combined_features["Unique_Procedure_Codes"].fillna(0)
    combined_features["Procedure_Diversity"] = safe_ratio(combined_features["Unique_Procedure_Codes"], combined_features["Total_Claim_Count"]).astype(float)
    combined_features["Avg_Procedure_Codes_Per_Claim"] = safe_ratio(combined_features["Total_Procedure_Codes"], combined_features["Total_Claim_Count"]).astype(float)

    bene_claims = pd.concat([
        ip[["Provider", "BeneID"]].drop_duplicates(),
        op[["Provider", "BeneID"]].drop_duplicates(),
    ], ignore_index=True).drop_duplicates()

    bene_claims = bene_claims.merge(bene[[
        "BeneID", "DOB", "Gender", "ChronicCond_Alzheimer", "ChronicCond_Heartfailure",
        "ChronicCond_KidneyDisease", "ChronicCond_Cancer", "ChronicCond_Diabetes",
        "ChronicCond_IschemicHeart", "ChronicCond_ObstrPulmonary", "ChronicCond_Depression",
        "ChronicCond_Osteoporasis", "ChronicCond_rheumatoidarthritis", "ChronicCond_stroke"
    ]], on="BeneID", how="left")

    bene_claims = bene_claims.drop_duplicates(subset=["Provider", "BeneID"])
    bene_claims["DOB"] = pd.to_datetime(bene_claims["DOB"], errors="coerce")
    age_ref = pd.Timestamp("2009-12-31")
    bene_claims["Age"] = (age_ref - bene_claims["DOB"]).dt.days / 365.25
    bene_claims["Age"] = pd.to_numeric(bene_claims["Age"], errors="coerce").fillna(0)

    bene_claims["Gender"] = pd.to_numeric(bene_claims["Gender"], errors="coerce")
    provider_bene = bene_claims.groupby("Provider", as_index=False).agg(
        Average_Beneficiary_Age=("Age", "mean"),
        Male_Count=("Gender", lambda s: (s == 1).sum()),
        Female_Count=("Gender", lambda s: (s == 2).sum()),
        Total_Beneficiaries=("BeneID", "nunique"),
    )
    provider_bene["Male_Ratio"] = safe_ratio(provider_bene["Male_Count"], provider_bene["Total_Beneficiaries"]).astype(float)
    provider_bene["Female_Ratio"] = safe_ratio(provider_bene["Female_Count"], provider_bene["Total_Beneficiaries"]).astype(float)

    chronic_cols = [
        "ChronicCond_Alzheimer", "ChronicCond_Heartfailure", "ChronicCond_KidneyDisease",
        "ChronicCond_Cancer", "ChronicCond_Diabetes", "ChronicCond_IschemicHeart",
        "ChronicCond_ObstrPulmonary", "ChronicCond_Depression", "ChronicCond_Osteoporasis",
        "ChronicCond_rheumatoidarthritis", "ChronicCond_stroke"
    ]

    for col in chronic_cols:
        bene_claims[col] = pd.to_numeric(bene_claims[col], errors="coerce").fillna(0)
    bene_claims["Chronic_Condition_Count"] = (bene_claims[chronic_cols] > 0).sum(axis=1)
    chronic_provider = bene_claims.groupby("Provider", as_index=False).agg(
        Average_Chronic_Condition_Count=("Chronic_Condition_Count", "mean"),
    )

    for col in chronic_cols:
        chronic_provider[col] = bene_claims.groupby("Provider")[col].apply(lambda s: (s > 0).mean()).values

    chronic_provider = chronic_provider.rename(columns={
        "ChronicCond_Heartfailure": "Pct_HeartFailure",
        "ChronicCond_KidneyDisease": "Pct_KidneyDisease",
        "ChronicCond_Cancer": "Pct_Cancer",
        "ChronicCond_stroke": "Pct_Stroke",
        "ChronicCond_Diabetes": "Pct_Diabetes",
    })
    chronic_provider["Pct_HeartFailure"] = chronic_provider.get("Pct_HeartFailure", 0).fillna(0)
    chronic_provider["Pct_KidneyDisease"] = chronic_provider.get("Pct_KidneyDisease", 0).fillna(0)
    chronic_provider["Pct_Cancer"] = chronic_provider.get("Pct_Cancer", 0).fillna(0)
    chronic_provider["Pct_Stroke"] = chronic_provider.get("Pct_Stroke", 0).fillna(0)
    chronic_provider["Pct_Diabetes"] = chronic_provider.get("Pct_Diabetes", 0).fillna(0)

    provider_bene = provider_bene.merge(chronic_provider[["Provider", "Average_Chronic_Condition_Count", "Pct_Diabetes", "Pct_HeartFailure", "Pct_KidneyDisease", "Pct_Cancer", "Pct_Stroke"]], on="Provider", how="left")

    physician_cols = ["AttendingPhysician", "OperatingPhysician", "OtherPhysician"]
    phys = pd.concat([
        ip[["Provider"] + physician_cols].copy(),
        op[["Provider"] + physician_cols].copy(),
    ], ignore_index=True)
    phys = phys.melt(id_vars="Provider", value_vars=physician_cols, value_name="Physician").dropna(subset=["Physician"])
    phys["Physician"] = phys["Physician"].astype(str).str.strip()
    phys = phys[phys["Physician"] != "nan"]

    attending = phys[phys["variable"] == "AttendingPhysician"].groupby("Provider")["Physician"].nunique().rename("Unique_Attending_Physicians")
    operating = phys[phys["variable"] == "OperatingPhysician"].groupby("Provider")["Physician"].nunique().rename("Unique_Operating_Physicians")
    other = phys[phys["variable"] == "OtherPhysician"].groupby("Provider")["Physician"].nunique().rename("Unique_Other_Physicians")
    total_unique = phys.groupby("Provider")["Physician"].nunique().rename("Total_Unique_Physicians")

    physician_summary = pd.concat([attending, operating, other, total_unique], axis=1).reset_index().rename(columns={"Provider": "Provider"})
    physician_summary["Physician_Diversity"] = 0.0

    df = providers.merge(combined_features, on="Provider", how="left")
    df = df.merge(provider_bene, on="Provider", how="left")
    df = df.merge(physician_summary, on="Provider", how="left")

    df["Total_Unique_Beneficiaries"] = pd.to_numeric(df["Total_Unique_Beneficiaries"], errors="coerce").fillna(0)
    df["Claims_Per_Beneficiary"] = safe_ratio(df["Total_Claim_Count"], df["Total_Unique_Beneficiaries"]).astype(float)
    df["Reimbursement_Per_Beneficiary"] = safe_ratio(df["Total_Reimbursement"], df["Total_Unique_Beneficiaries"]).astype(float)
    df["Same_Day_Discharge_Ratio"] = safe_ratio(df["Same_Day_Discharge_Count"], df["IP_Claim_Count"]).astype(float)
    df["Physician_Diversity"] = safe_ratio(df["Total_Unique_Physicians"], df["Total_Claim_Count"]).astype(float)

    df = df.sort_values("Provider").reset_index(drop=True)

    if df["Provider"].nunique() != len(df):
        raise ValueError(f"Duplicate providers found: {df['Provider'].duplicated().sum()}")

    if len(df) != 5410:
        raise ValueError(f"Expected 5410 providers but found {len(df)}")

    # ensure all derived numeric columns are numeric and free of nulls
    numeric_cols = [c for c in df.columns if c not in ["Provider", "PotentialFraud"]]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

    object_cols = [c for c in df.columns if c in ["Provider"]]
    for col in object_cols:
        df[col] = df[col].fillna("UNKNOWN")

    if df.isnull().sum().sum() != 0:
        raise ValueError(f"Nulls still present after final cleaning: {df.isnull().sum().sum()}")

    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_provider_dataset()
    df = ensure_no_nulls(df)

    print(df.shape)
    print(df.isnull().sum().sum())
    print(df["Provider"].duplicated().sum())
    print(df["Fraud_Label"].value_counts())
    print(df.describe())

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved dataset: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
