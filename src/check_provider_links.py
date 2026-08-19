import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

labels = pd.read_csv(DATA_DIR / "Train-1542865627584.csv")
inpatient = pd.read_csv(DATA_DIR / "Train_Inpatientdata-1542865627584.csv")
outpatient = pd.read_csv(DATA_DIR / "Train_Outpatientdata-1542865627584.csv")

label_providers = set(labels["Provider"].dropna().unique())
inpatient_providers = set(inpatient["Provider"].dropna().unique())
outpatient_providers = set(outpatient["Provider"].dropna().unique())

print("=" * 70)
print("PROVIDER LINKAGE CHECK")
print("=" * 70)

print("\nUnique providers:")
print("Labels     :", len(label_providers))
print("Inpatient  :", len(inpatient_providers))
print("Outpatient :", len(outpatient_providers))

print("\nProviders common with labels:")
print("Labels + Inpatient :", len(label_providers & inpatient_providers))
print("Labels + Outpatient:", len(label_providers & outpatient_providers))

print("\nProviders present in both claim datasets:")
print("Inpatient + Outpatient:",
      len(inpatient_providers & outpatient_providers))

print("\nProviders present in all three:")
print("Labels + Inpatient + Outpatient:",
      len(label_providers & inpatient_providers & outpatient_providers))

print("\nProviders with labels but NO inpatient claims:",
      len(label_providers - inpatient_providers))

print("Providers with labels but NO outpatient claims:",
      len(label_providers - outpatient_providers))