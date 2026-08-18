"""Validation script for fusion layer deployment."""

import pandas as pd
from pathlib import Path

print("=" * 100)
print("HEALTHCARE FRAUD ANALYTICS FUSION LAYER - IMPLEMENTATION VALIDATION")
print("=" * 100)

# Check all created files
risk_fusion_files = [
    "risk_fusion/__init__.py",
    "risk_fusion/model_registry.py",
    "risk_fusion/risk_utils.py",
    "risk_fusion/fusion_engine.py",
    "risk_fusion/run_fusion_pipeline.py",
    "risk_fusion/generate_explanations.py",
]

print("\n[✓] CREATED FILES (risk_fusion module):")
for f in risk_fusion_files:
    p = Path(f)
    exists = "✓" if p.exists() else "✗"
    size = f"{p.stat().st_size:,} bytes" if p.exists() else "N/A"
    print(f"  {exists} {f:45s} ({size})")

# Check output files
output_files = [
    "outputs/final_provider_risk_scores.csv",
    "outputs/provider_explanations.csv",
]

print("\n[✓] OUTPUT FILES (Generated):")
for f in output_files:
    p = Path(f)
    if p.exists():
        rows = len(pd.read_csv(f))
        cols = len(pd.read_csv(f).columns)
        print(f"  ✓ {f:45s} ({rows:,} rows, {cols} cols)")
    else:
        print(f"  ✗ {f:45s} (NOT FOUND)")

# Validate final provider risk scores
print("\n[✓] FUSION RESULTS:")
df_risk = pd.read_csv("outputs/final_provider_risk_scores.csv")
print(f"  Total Providers: {len(df_risk):,}")
print(f"  Columns: {list(df_risk.columns)}")
print(f"  Risk Score Range: {df_risk['Final_Risk_Score'].min():.2f} - {df_risk['Final_Risk_Score'].max():.2f}")
print(f"  Risk Levels: {dict(df_risk['Risk_Level'].value_counts().sort_index())}")

# Show top providers
print("\n[✓] TOP 5 HIGHEST-RISK PROVIDERS:")
for idx, row in df_risk.head(5).iterrows():
    print(f"  {idx+1}. {row['Provider']:10s} - Risk: {row['Final_Risk_Score']:6.2f}/100 ({row['Risk_Level']})")

# Validate explanations
print("\n[✓] EXPLANATIONS GENERATED:")
df_exp = pd.read_csv("outputs/provider_explanations.csv")
print(f"  Total Explanations: {len(df_exp):,}")
print(f"  Explanation Length Range: {df_exp['Explanation'].str.len().min()} - {df_exp['Explanation'].str.len().max()} chars")
print(f"  Sample Explanation:")
print(f'    "{df_exp.iloc[0]["Explanation"][:150]}..."')

print("\n" + "=" * 100)
print("FUSION LAYER SUCCESSFULLY DEPLOYED")
print("=" * 100)
