import pandas as pd
import glob
import os

data_dir = "e:/CTS - MediClaim/datas"
files = glob.glob(os.path.join(data_dir, "*.csv"))

for f in files:
    filename = os.path.basename(f)
    print(f"\n==================================================")
    print(f"File: {filename}")
    try:
        # Load a small sample first to check column names and types
        df_sample = pd.read_csv(f, nrows=5)
        print("Columns:")
        for col in df_sample.columns:
            print(f"  - {col}")
        
        # Read the full dataset shape and missing values
        df = pd.read_csv(f)
        print(f"Shape: {df.shape}")
        print("Missing Values (%):")
        missing = df.isnull().mean() * 100
        for col, val in missing.items():
            if val > 0:
                print(f"  - {col}: {val:.2f}%")
        
        # If the file is a label file, show value counts
        if "Train-1542865627584" in filename or "Test-1542969243754" in filename:
            print("Value Counts for Potential Target:")
            for col in df.columns:
                if df[col].nunique() < 10:
                    print(f"Column '{col}' value counts:")
                    print(df[col].value_counts(dropna=False))
                    
    except Exception as e:
        print(f"Error reading {filename}: {e}")
