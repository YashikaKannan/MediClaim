import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =====================================================================
# PATHS -- CHANGE THESE TO MATCH YOUR COMPUTER
# =====================================================================
LABELS_PATH = r"E:\MediClaim - CTS\one_class_svm_model\sythetic data\Train-1542865627584.csv"
INPATIENT_PATH = r"E:\MediClaim - CTS\one_class_svm_model\sythetic data\Train_Inpatientdata-1542865627584.csv""
OUTPATIENT_PATH = r"E:\MediClaim - CTS\one_class_svm_model\sythetic data\Train_Outpatientdata-1542865627584.csv"



OUTPUT_PATH = r"E:\MediClaim - CTS\PROVIDER_PEER_BENCHMARK_KAGGLE.csv"
N_PEER_GROUPS = 8   # number of data-driven clusters (no specialty field in this dataset)


# =====================================================================
# STEP 1: Load claims and combine inpatient + outpatient
# =====================================================================
inp = pd.read_csv(INPATIENT_PATH)
outp = pd.read_csv(OUTPATIENT_PATH)
labels = pd.read_csv(LABELS_PATH)

inp['ClaimType'] = 'Inpatient'
outp['ClaimType'] = 'Outpatient'

common_cols = ['BeneID', 'ClaimID', 'Provider', 'InscClaimAmtReimbursed',
               'DeductibleAmtPaid', 'ClaimType', 'AttendingPhysician',
               'OperatingPhysician', 'OtherPhysician',
               'ClmDiagnosisCode_1', 'ClmProcedureCode_1']
claims = pd.concat([inp[common_cols], outp[common_cols]], ignore_index=True)

print(f"STEP 1 -- Total claims combined: {len(claims)}")
print(f"STEP 1 -- Unique providers in claims: {claims['Provider'].nunique()}")
print(f"STEP 1 -- Providers in labels: {labels['Provider'].nunique()}")


# =====================================================================
# STEP 2: Build provider-level profile by aggregating claims
# =====================================================================
agg = claims.groupby('Provider').agg(
    Total_Claims=('ClaimID', 'count'),
    Total_Reimbursed=('InscClaimAmtReimbursed', 'sum'),
    Avg_Reimbursed=('InscClaimAmtReimbursed', 'mean'),
    Total_Deductible=('DeductibleAmtPaid', 'sum'),
    Unique_Beneficiaries=('BeneID', 'nunique'),
    Unique_Diagnosis_Codes=('ClmDiagnosisCode_1', 'nunique'),
    Unique_Procedure_Codes=('ClmProcedureCode_1', 'nunique'),
    Unique_Attending_Physicians=('AttendingPhysician', 'nunique'),
    Inpatient_Claims=('ClaimType', lambda x: (x == 'Inpatient').sum()),
    Outpatient_Claims=('ClaimType', lambda x: (x == 'Outpatient').sum()),
).reset_index()

agg['Services_Per_Beneficiary'] = agg['Total_Claims'] / agg['Unique_Beneficiaries']
agg['Reimbursed_Per_Beneficiary'] = agg['Total_Reimbursed'] / agg['Unique_Beneficiaries']
agg['Inpatient_Ratio'] = agg['Inpatient_Claims'] / agg['Total_Claims']

df = agg.merge(labels, on='Provider', how='left')
print(f"\nSTEP 2 -- Provider profiles built: {df.shape}")
print(f"STEP 2 -- Null values: {df.isnull().sum().sum()}")


# =====================================================================
# STEP 3: Cluster providers into data-driven peer groups
# (No specialty field exists in this dataset -- cluster on scale-of-
#  practice features only, NOT on billing amount, to avoid leaking
#  the risk signal into the peer-group assignment itself)
# =====================================================================
cluster_feats = ['Total_Claims', 'Unique_Beneficiaries', 'Unique_Procedure_Codes',
                  'Unique_Attending_Physicians', 'Inpatient_Ratio', 'Services_Per_Beneficiary']

X_log = np.log1p(df[['Total_Claims', 'Unique_Beneficiaries',
                      'Unique_Procedure_Codes', 'Unique_Attending_Physicians']])
X_log['Inpatient_Ratio'] = df['Inpatient_Ratio']
X_log['Services_Per_Beneficiary'] = np.log1p(df['Services_Per_Beneficiary'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

kmeans = KMeans(n_clusters=N_PEER_GROUPS, random_state=42, n_init=10)
df['Peer_Group'] = kmeans.fit_predict(X_scaled)

print(f"\nSTEP 3 -- Peer group sizes:")
print(df['Peer_Group'].value_counts().sort_index())


# =====================================================================
# STEP 4: Robust Z-scores within each peer group
# =====================================================================
df['Reimb_Log'] = np.log1p(df['Total_Reimbursed'])
df['Claims_Log'] = np.log1p(df['Total_Claims'])
df['Benes_Log'] = np.log1p(df['Unique_Beneficiaries'])

def robust_z(group):
    median = group.median()
    mad = (group - median).abs().median()
    if mad == 0:
        return pd.Series(0, index=group.index)
    return (group - median) / mad

df['Reimb_Z'] = df.groupby('Peer_Group')['Reimb_Log'].transform(robust_z)
df['Claims_Z'] = df.groupby('Peer_Group')['Claims_Log'].transform(robust_z)
df['Benes_Z'] = df.groupby('Peer_Group')['Benes_Log'].transform(robust_z)

df['Peer_Reimb_Median'] = df.groupby('Peer_Group')['Total_Reimbursed'].transform('median')
df['Reimb_Ratio'] = (df['Total_Reimbursed'] / df['Peer_Reimb_Median'].replace(0, np.nan)).fillna(0)


# =====================================================================
# STEP 5: Blend relative (peer) risk with absolute (magnitude) risk
# -----------------------------------------------------------------
# IMPORTANT FIX: pure peer-relative Z-score ranking lets a cluster of
# small, low-volume providers dominate the top tier -- a "big fish in
# a small pond" gets the same rank as a genuinely massive outlier.
# Blending in absolute billing magnitude (ranked across ALL providers,
# not just within-peer-group) corrects this. 60/40 split validated
# to work well on this dataset.
# =====================================================================
combined_z = (df['Reimb_Z'].clip(lower=0) + df['Claims_Z'].clip(lower=0) + df['Benes_Z'].clip(lower=0))
df['Relative_Risk'] = combined_z.rank(pct=True) * 100
df['Absolute_Risk'] = df['Total_Reimbursed'].rank(pct=True) * 100
df['RISK_SCORE'] = (0.6 * df['Relative_Risk'] + 0.4 * df['Absolute_Risk']).round(1)

def risk_level(score):
    if score <= 30:
        return "Low"
    elif score <= 60:
        return "Medium"
    elif score <= 80:
        return "High"
    else:
        return "Critical"

df['RISK_LEVEL'] = df['RISK_SCORE'].apply(risk_level)

for pct, label in [(0.99, "TOP_1PCT"), (0.97, "TOP_3PCT"), (0.95, "TOP_5PCT")]:
    df[label] = df['RISK_SCORE'] >= df['RISK_SCORE'].quantile(pct)
df['FLAG'] = df['TOP_5PCT']

print(f"\nSTEP 5 -- Risk level distribution:")
print(df['RISK_LEVEL'].value_counts())


# =====================================================================
# STEP 6: Generate explainable reasons
# =====================================================================
def build_reason(row):
    reasons = []
    if row['Reimb_Ratio'] > 1.5:
        reasons.append(f"Bills {row['Reimb_Ratio']:.1f}x higher than peer group median")
    if row['Total_Reimbursed'] > df['Total_Reimbursed'].quantile(0.99):
        reasons.append("Among top 1% by total billing volume overall")
    if row['Unique_Beneficiaries'] > 0 and row['Services_Per_Beneficiary'] > df['Services_Per_Beneficiary'].quantile(0.95):
        reasons.append(f"Serves patients {row['Services_Per_Beneficiary']:.1f} claims each, unusually high")
    if not reasons:
        reasons.append("No significant deviation detected")
    return " | ".join(reasons[:3])

df['REASON'] = df.apply(build_reason, axis=1)


# =====================================================================
# STEP 7: Save output
# =====================================================================
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSTEP 7 -- Saved to: {OUTPUT_PATH}")


# =====================================================================
# STEP 8: Validate against real fraud labels
# (PotentialFraud was NEVER used in clustering or scoring above --
#  this is a pure post-hoc validation check)
# =====================================================================
overall_fraud_rate = (df['PotentialFraud'] == 'Yes').mean()
print(f"\n{'='*50}")
print("VALIDATION AGAINST REAL FRAUD LABELS")
print(f"{'='*50}")
print(f"Overall fraud rate in population: {overall_fraud_rate*100:.1f}%")

for label, col in [("Top 1%", "TOP_1PCT"), ("Top 3%", "TOP_3PCT"), ("Top 5%", "TOP_5PCT")]:
    subset = df[df[col] == True]
    fraud_rate = (subset['PotentialFraud'] == 'Yes').mean()
    lift = fraud_rate / overall_fraud_rate
    print(f"{label} ({len(subset)} providers): fraud rate {fraud_rate*100:.1f}% -> {lift:.1f}x lift")

print(f"\nAvg RISK_SCORE -- Fraud vs Non-Fraud:")
print(df.groupby('PotentialFraud')['RISK_SCORE'].mean().round(1))
