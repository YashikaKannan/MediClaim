"""
PEER BENCHMARKING ENGINE -- KAGGLE HEALTHCARE PROVIDER FRAUD DATASET
======================================================================
CORRECTED VERSION -- peer groups are now based on each provider's
CLINICAL CASE-MIX (what conditions they treat), not billing volume.

Why this changed: grouping providers by "how big" their practice is
(claim volume, patient count) puts a small cardiology practice and a
small dermatology practice in the same peer group -- but a cardiac
procedure and a skin check have completely different normal billing
ranges. That's not a fair comparison. Specialty is supposed to prevent
exactly this apples-to-oranges problem, and this dataset has no
specialty field, so we derive the closest real substitute: each
provider's DOMINANT diagnosis category, based on standard ICD-9-CM
chapters (e.g. Circulatory, Respiratory, Musculoskeletal). Providers
who mostly treat the same category of condition are true peers.

FILES USED:
  - Train-1542865627584.csv                 (labels: Provider, PotentialFraud)
  - Train_Inpatientdata-1542865627584.csv    (inpatient claims)
  - Train_Outpatientdata-1542865627584.csv   (outpatient claims)

HOW TO RUN:
1. Update the 3 input paths and 2 output paths below
2. Run top to bottom
3. Check the validation summary at the end
"""

import pandas as pd
import numpy as np

# =====================================================================
# PATHS -- CHANGE THESE TO MATCH YOUR COMPUTER
# =====================================================================
LABELS_PATH = "Train-1542865627584.csv"
INPATIENT_PATH = "Train_Inpatientdata-1542865627584.csv"
OUTPATIENT_PATH = "Train_Outpatientdata-1542865627584.csv"

OUTPUT_SCORED_PATH = "PROVIDER_PEER_BENCHMARK_KAGGLE_FIXED.csv"
OUTPUT_MANUAL_REVIEW_PATH = "MANUAL_REVIEW_QUEUE_KAGGLE_FIXED.csv"

MIN_PEERS = 30   # minimum providers needed in a peer group to compare fairly


# =====================================================================
# STEP 1: Load and combine claims
# =====================================================================
inp = pd.read_csv(INPATIENT_PATH)
outp = pd.read_csv(OUTPATIENT_PATH)
labels = pd.read_csv(LABELS_PATH)

inp['ClaimType'] = 'Inpatient'
outp['ClaimType'] = 'Outpatient'
common_cols = ['BeneID', 'ClaimID', 'Provider', 'InscClaimAmtReimbursed',
               'DeductibleAmtPaid', 'ClaimType', 'AttendingPhysician',
               'ClmDiagnosisCode_1', 'ClmProcedureCode_1']
claims = pd.concat([inp[common_cols], outp[common_cols]], ignore_index=True)
print(f"STEP 1 -- Total claims combined: {len(claims)}")


# =====================================================================
# STEP 2: Map each claim's diagnosis code to its ICD-9-CM chapter
# (standard, well-established disease-category ranges)
# =====================================================================
def icd9_chapter(code):
    if pd.isna(code):
        return "Unknown"
    code = str(code).strip()
    if code.startswith('V'):
        return "Health_Status_V"
    if code.startswith('E'):
        return "External_Cause_E"
    try:
        n = int(code[:3])
    except ValueError:
        return "Unknown"
    if 1 <= n <= 139: return "Infectious"
    if 140 <= n <= 239: return "Neoplasms"
    if 240 <= n <= 279: return "Endocrine_Metabolic"
    if 280 <= n <= 289: return "Blood"
    if 290 <= n <= 319: return "Mental"
    if 320 <= n <= 389: return "Nervous_Sensory"
    if 390 <= n <= 459: return "Circulatory"
    if 460 <= n <= 519: return "Respiratory"
    if 520 <= n <= 579: return "Digestive"
    if 580 <= n <= 629: return "Genitourinary"
    if 630 <= n <= 679: return "Pregnancy"
    if 680 <= n <= 709: return "Skin"
    if 710 <= n <= 739: return "Musculoskeletal"
    if 740 <= n <= 759: return "Congenital"
    if 760 <= n <= 779: return "Perinatal"
    if 780 <= n <= 799: return "Symptoms_IllDefined"
    if 800 <= n <= 999: return "Injury_Poisoning"
    return "Unknown"

claims['Chapter'] = claims['ClmDiagnosisCode_1'].apply(icd9_chapter)
print(f"STEP 2 -- Diagnosis chapters found: {claims['Chapter'].nunique()}")


# =====================================================================
# STEP 3: Peer_Group = each provider's DOMINANT diagnosis chapter
# (the category of condition they treat most often -- a direct,
#  clinically meaningful proxy for specialty)
# =====================================================================
dominant_chapter = claims.groupby('Provider')['Chapter'].agg(lambda x: x.value_counts().idxmax())
dominant_chapter.name = 'Peer_Group'
print(f"\nSTEP 3 -- Peer group sizes (by dominant diagnosis chapter):")
print(dominant_chapter.value_counts())


# =====================================================================
# STEP 4: Build provider-level profile
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
df = df.merge(dominant_chapter, on='Provider', how='left')


# =====================================================================
# STEP 5: Split into scored vs manual review (small peer groups)
# =====================================================================
group_size = df['Peer_Group'].value_counts()
df['Group_Size'] = df['Peer_Group'].map(group_size)

scored = df[df['Group_Size'] >= MIN_PEERS].copy()
manual_review = df[df['Group_Size'] < MIN_PEERS].copy()
manual_review['STATUS'] = 'INSUFFICIENT_PEERS'
print(f"\nSTEP 5 -- Scored: {len(scored)} | Manual review (peer group < {MIN_PEERS}): {len(manual_review)}")


# =====================================================================
# STEP 6: Robust Z-scores within each (corrected) peer group
# =====================================================================
scored['Reimb_Log'] = np.log1p(scored['Total_Reimbursed'])
scored['Claims_Log'] = np.log1p(scored['Total_Claims'])
scored['Benes_Log'] = np.log1p(scored['Unique_Beneficiaries'])

def robust_z(g):
    med = g.median()
    mad = (g - med).abs().median()
    return pd.Series(0, index=g.index) if mad == 0 else (g - med) / mad

scored['Reimb_Z'] = scored.groupby('Peer_Group')['Reimb_Log'].transform(robust_z)
scored['Claims_Z'] = scored.groupby('Peer_Group')['Claims_Log'].transform(robust_z)
scored['Benes_Z'] = scored.groupby('Peer_Group')['Benes_Log'].transform(robust_z)
scored['Peer_Reimb_Median'] = scored.groupby('Peer_Group')['Total_Reimbursed'].transform('median')
scored['Reimb_Ratio'] = (scored['Total_Reimbursed'] / scored['Peer_Reimb_Median'].replace(0, np.nan)).fillna(0)


# =====================================================================
# STEP 7: Blend relative (peer) risk with absolute (magnitude) risk
# =====================================================================
combined_z = scored['Reimb_Z'].clip(lower=0) + scored['Claims_Z'].clip(lower=0) + scored['Benes_Z'].clip(lower=0)
scored['Relative_Risk'] = combined_z.rank(pct=True) * 100
scored['Absolute_Risk'] = scored['Total_Reimbursed'].rank(pct=True) * 100
scored['RISK_SCORE'] = (0.6 * scored['Relative_Risk'] + 0.4 * scored['Absolute_Risk']).round(1)

def risk_level(s):
    if s <= 30: return "Low"
    elif s <= 60: return "Medium"
    elif s <= 80: return "High"
    else: return "Critical"
scored['RISK_LEVEL'] = scored['RISK_SCORE'].apply(risk_level)

for pct, label in [(0.99, "TOP_1PCT"), (0.97, "TOP_3PCT"), (0.95, "TOP_5PCT")]:
    scored[label] = scored['RISK_SCORE'] >= scored['RISK_SCORE'].quantile(pct)
scored['FLAG'] = scored['TOP_5PCT']


# =====================================================================
# STEP 8: Explainable reasons
# =====================================================================
def build_reason(row):
    reasons = []
    if row['Reimb_Ratio'] > 1.5:
        reasons.append(f"Bills {row['Reimb_Ratio']:.1f}x higher than peer group ({row['Peer_Group']}) median")
    if row['Total_Reimbursed'] > scored['Total_Reimbursed'].quantile(0.99):
        reasons.append("Among top 1% by total billing volume overall")
    if row['Unique_Beneficiaries'] > 0 and row['Services_Per_Beneficiary'] > scored['Services_Per_Beneficiary'].quantile(0.95):
        reasons.append(f"Serves patients {row['Services_Per_Beneficiary']:.1f} claims each, unusually high")
    if not reasons:
        reasons.append("No significant deviation detected")
    return " | ".join(reasons[:3])

scored['REASON'] = scored.apply(build_reason, axis=1)


# =====================================================================
# STEP 9: Save outputs
# =====================================================================
scored.to_csv(OUTPUT_SCORED_PATH, index=False)
manual_review.to_csv(OUTPUT_MANUAL_REVIEW_PATH, index=False)
print(f"\nSTEP 9 -- Saved: {OUTPUT_SCORED_PATH}, {OUTPUT_MANUAL_REVIEW_PATH}")


# =====================================================================
# STEP 10: Validate against real fraud labels
# =====================================================================
overall = (scored['PotentialFraud'] == 'Yes').mean()
print(f"\n{'='*50}\nVALIDATION\n{'='*50}")
print(f"Overall fraud rate: {overall*100:.1f}%")
for label, col in [("Top 1%", "TOP_1PCT"), ("Top 3%", "TOP_3PCT"), ("Top 5%", "TOP_5PCT")]:
    sub = scored[scored[col] == True]
    fr = (sub['PotentialFraud'] == 'Yes').mean()
    print(f"{label} ({len(sub)}): fraud rate {fr*100:.1f}% -> {fr/overall:.1f}x lift")
print("\nFraud rate by RISK_LEVEL:")
print(scored.groupby('RISK_LEVEL')['PotentialFraud'].apply(lambda x: (x == 'Yes').mean() * 100).round(1))
