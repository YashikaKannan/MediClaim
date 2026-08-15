import pandas as pd

DATA_PATH = "CLAIM_LEVEL_AUTOENCODER_RESULTS.csv"

print("Loading final claim-level results...")

df = pd.read_csv(DATA_PATH)

print("\n================================")
print("FINAL MODEL VALIDATION")
print("================================")

print("\n1. Dataset Check")
print("Total Claims:", len(df))
print("Unique CLM_ID:", df["CLM_ID"].nunique())
print("Duplicate CLM_ID:", df["CLM_ID"].duplicated().sum())
print("Missing Values:", df.isnull().sum().sum())


print("\n2. Flag Distribution")
print(df["FLAG"].value_counts())


print("\n3. Risk Score Statistics")
print(df["RISK_SCORE"].describe())


print("\n4. Reconstruction Error Statistics")
print(df["RECONSTRUCTION_ERROR"].describe())


print("\n5. Flagged Claim Risk Statistics")

flagged = df[df["FLAG"] == True]

print(
    flagged["RISK_SCORE"].describe()
)


print("\n6. TOP_REASON_1 Distribution")

print(
    flagged["TOP_REASON_1"]
    .value_counts()
)


print("\n7. Top 20 Highest-Risk Claims")

print(
    df[
        [
            "CLM_ID",
            "BENE_ID",
            "RISK_SCORE",
            "RECONSTRUCTION_ERROR",
            "FLAG",
            "TOP_REASON_1",
            "TOP_REASON_2",
            "TOP_REASON_3"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


print("\n================================")
print("VALIDATION COMPLETED")
print("================================")