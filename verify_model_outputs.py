"""
Verification script for model output files
Checks structure, columns, and row counts
"""
import pandas as pd
import os

print("=" * 80)
print("MODEL OUTPUT FILES VERIFICATION")
print("=" * 80)

# Define all output files to check
outputs = {
    "Isolation Forest": "outputs/kaggle_iforest_scores.csv",
    "OCSVM": "outputs/kaggle_ocsvm_scores.csv",
    "Autoencoder": "outputs/autoencoder_provider_scores.csv",
    "CatBoost": "catboost_model/output/provider_risk_ranked.csv",
    "Claim Autoencoder": "claim_autoencoder/results/provider_investigation_queue.csv",
}

for model_name, file_path in outputs.items():
    print(f"\n{model_name.upper()}:")
    print(f"  File: {file_path}")
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Provider Column: {'Provider' in df.columns}")
        
        # Show sample data
        print(f"  Sample:")
        print(f"    {df.iloc[0].to_dict()}")
    else:
        print(f"  ❌ FILE NOT FOUND")

print("\n" + "=" * 80)
print("SPECIAL COLUMNS SEARCH")
print("=" * 80)

search_terms = ["Robust_ZScore", "robust_zscore", "Peer_", "ZScore", "MAD"]

for model_name, file_path in outputs.items():
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        found_terms = []
        for col in df.columns:
            for term in search_terms:
                if term.lower() in col.lower():
                    found_terms.append(col)
        
        if found_terms:
            print(f"\n{model_name}: {found_terms}")

print("\n" + "=" * 80)
print("CLAIM AUTOENCODER - DETAILED ANALYSIS")
print("=" * 80)

claim_ae_file = "claim_autoencoder/results/provider_investigation_queue.csv"
if os.path.exists(claim_ae_file):
    df_claim = pd.read_csv(claim_ae_file)
    print(f"\nStructure:")
    print(f"  Has Provider Column: {'Provider' in df_claim.columns}")
    print(f"  All Columns: {list(df_claim.columns)}")
    print(f"\nRow Count: {len(df_claim)}")
    print(f"\nFirst 3 rows:")
    print(df_claim.head(3).to_string(index=False))
else:
    print("FILE NOT FOUND")

print("\n" + "=" * 80)
print("MERGE COMPATIBILITY CHECK")
print("=" * 80)

base_df = pd.read_csv("outputs/kaggle_iforest_scores.csv")
print(f"\nBase DataFrame (IF): {len(base_df)} providers")

for model_name, file_path in outputs.items():
    if model_name == "Isolation Forest":
        continue
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        
        # Check merge compatibility
        has_provider = "Provider" in df.columns
        can_merge_left = len(df[df["Provider"].isin(base_df["Provider"])]) == len(base_df)
        
        print(f"\n{model_name}:")
        print(f"  Has Provider column: {has_provider}")
        print(f"  Row count: {len(df)}")
        print(f"  All providers in base: {can_merge_left}")
        print(f"  Can merge directly: {'YES' if (has_provider and len(df) == len(base_df)) else 'YES (sparse merge)' if has_provider else 'NO'}")

print("\n" + "=" * 80)
