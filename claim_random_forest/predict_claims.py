import joblib
import pandas as pd
import os

model_path = os.path.join(
    os.path.dirname(__file__),
    "random_forest_claim_model.pkl"
)

model = joblib.load(model_path)

print("Model loaded successfully!")
csv_path = r"C:\Users\ranjana\OneDrive\final.processed\carrier_clean_final_perfect.csv"

df = pd.read_csv(csv_path)

print(df.shape)
claim_df = df.groupby("CLM_ID").agg({
    "BENE_ID": "first",
    "DIAGNOSIS_COUNT": "first",
    "PRVDR_STATE_CD": "first",
    "LINE_PLACE_OF_SRVC_CD": "first",
    "LINE_NUM": "count"
}).reset_index()

print("Claim-level shape:", claim_df.shape)
from sklearn.preprocessing import LabelEncoder

le_state = LabelEncoder()

claim_df["PRVDR_STATE_CD"] = le_state.fit_transform(
    claim_df["PRVDR_STATE_CD"].astype(str)
)

print(claim_df["PRVDR_STATE_CD"].head())
model_features = [
    "DIAGNOSIS_COUNT",
    "PRVDR_STATE_CD",
    "LINE_PLACE_OF_SRVC_CD",
    "LINE_NUM"
]

X_claims = claim_df[model_features]

print(X_claims.shape)
print(X_claims.head())
claim_df["RISK_PROBABILITY"] = model.predict_proba(X_claims)[:, 1]

print(claim_df[["CLM_ID", "RISK_PROBABILITY"]].head())

threshold = 0.8

claim_df["PREDICTION"] = (
    claim_df["RISK_PROBABILITY"] >= threshold
).map({
    True: "Suspicious",
    False: "Normal"
})

print(claim_df["PREDICTION"].value_counts())
claim_df["RISK_SCORE"] = (
    claim_df["RISK_PROBABILITY"] * 100
).round(2)

print(
    claim_df[
        ["CLM_ID", "RISK_SCORE", "PREDICTION"]
    ].head()
)
investigation_queue = claim_df[
    claim_df["PREDICTION"] == "Suspicious"
].sort_values(
    by="RISK_SCORE",
    ascending=False
)

print(investigation_queue[
    ["CLM_ID", "BENE_ID", "RISK_SCORE", "PREDICTION"]
].head(10))

output_path = os.path.join(
    os.path.dirname(__file__),
    "CLAIM_RANDOM_FOREST_RESULTS.csv"
)

investigation_queue.to_csv(output_path, index=False)

print("Saved:", output_path)