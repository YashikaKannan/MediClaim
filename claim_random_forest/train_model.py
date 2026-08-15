import pandas as pd

csv_path = r"C:\Users\ranjana\OneDrive\final.processed\carrier_clean_final_perfect.csv"

df = pd.read_csv(csv_path)

print("Dataset loaded:", df.shape)
claim_df = df.groupby("CLM_ID").agg({
    "BENE_ID": "first",
    "CLM_PMT_AMT": "first",
    "NCH_CARR_CLM_SBMTD_CHRG_AMT": "first",
    "LINE_SRVC_CNT": "sum",
    "DIAGNOSIS_COUNT": "first",
    "PRVDR_STATE_CD": "first",
    "LINE_PLACE_OF_SRVC_CD": "first",
    "LINE_NUM": "count"
}).reset_index()

print("Claim-level dataset:", claim_df.shape)
claim_df["RISK_CONDITION_COUNT"] = (
    (claim_df["CLM_PMT_AMT"] > 13197.3292).astype(int)
    + (claim_df["NCH_CARR_CLM_SBMTD_CHRG_AMT"] > 16084.1244).astype(int)
    + (claim_df["LINE_SRVC_CNT"] > 506).astype(int)
)

claim_df["SUSPICIOUS_FLAG"] = (
    claim_df["RISK_CONDITION_COUNT"] >= 2
).astype(int)

print(claim_df["SUSPICIOUS_FLAG"].value_counts())
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

features = [
    "DIAGNOSIS_COUNT",
    "PRVDR_STATE_CD",
    "LINE_PLACE_OF_SRVC_CD",
    "LINE_NUM"
]

X = claim_df[features]
y = claim_df["SUSPICIOUS_FLAG"]

categorical_features = ["PRVDR_STATE_CD"]
numeric_features = [
    "DIAGNOSIS_COUNT",
    "LINE_PLACE_OF_SRVC_CD",
    "LINE_NUM"
]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", "passthrough", numeric_features)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

pipeline.fit(X_train, y_train)

print("Pipeline training completed!")

import joblib
import os

model_path = os.path.join(
    os.path.dirname(__file__),
    "random_forest_pipeline.pkl"
)

joblib.dump(pipeline, model_path)

print("Pipeline model saved successfully!")
from sklearn.metrics import classification_report, confusion_matrix

y_prob = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.8).astype(int)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))