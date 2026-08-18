import pandas as pd
import glob
import os

data_dir = "e:/CTS - MediClaim/datas"

def analyze_file(path):
    df = pd.read_csv(path)
    filename = os.path.basename(path)
    print(f"\n==================================================")
    print(f"File: {filename}")
    print(f"Shape: {df.shape}")
    print("Field | Type | Missing % | Unique Count")
    print("---|---|---|---")
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing_pct = df[col].isnull().mean() * 100
        nunique = df[col].nunique()
        print(f"{col} | {dtype} | {missing_pct:.2f}% | {nunique}")

analyze_file(os.path.join(data_dir, "Train-1542865627584.csv"))
analyze_file(os.path.join(data_dir, "Train_Beneficiarydata-1542865627584.csv"))
analyze_file(os.path.join(data_dir, "Train_Inpatientdata-1542865627584.csv"))
analyze_file(os.path.join(data_dir, "Train_Outpatientdata-1542865627584.csv"))
