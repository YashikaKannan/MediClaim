
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

def load_model():
    path = MODEL_DIR / "isolation_forest.pkl"
    if path.exists():
        return joblib.load(path)
    return None

def load_preprocessor():
    path = MODEL_DIR / "preprocessor.pkl"
    if path.exists():
        return joblib.load(path)
    return None

def preprocess_data(df, features=None):
    data = df.copy()
    if features is None:
        features = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
    X = data[features].replace([np.inf,-np.inf], np.nan).fillna(0)
    pre = load_preprocessor()
    if pre is not None:
        X = pre.transform(X)
    return X, features

def predict(df, features=None):
    model = load_model()
    if model is None:
        # Demo adapter only; clearly label downstream UI as DEMO MODEL.
        return np.zeros(len(df), dtype=int)
    X, _ = preprocess_data(df, features)
    return model.predict(X)

def calculate_anomaly_score(df, features=None):
    model = load_model()
    if model is None:
        # Demo-only deterministic placeholder.
        numeric = df.select_dtypes(include="number")
        if numeric.empty:
            return np.zeros(len(df))
        z = (numeric - numeric.mean()) / numeric.std(ddof=0).replace(0,1)
        score = np.clip(np.abs(z).mean(axis=1).fillna(0).to_numpy(), 0, 4)
        return score
    X, _ = preprocess_data(df, features)
    return -model.score_samples(X)

def generate_risk_score(anomaly_scores):
    a = np.asarray(anomaly_scores, dtype=float)
    if len(a) == 0:
        return a
    lo, hi = np.nanmin(a), np.nanmax(a)
    if hi == lo:
        return np.full(len(a), 50.0)
    return np.round(np.clip((a-lo)/(hi-lo)*100,0,100),1)

def generate_shap_explanation(record):
    # Demo adapter until a real explainer is connected.
    # The UI explicitly labels these as DEMO EXPLANATION DATA.
    candidates = [
        ("Total Paid", float(record.get("totalPaid", record.get("paidAmount", 0))) / 100000000, "Increases Risk"),
        ("Claims Volume", float(record.get("totalClaims", record.get("units", 0))) / 10000, "Increases Risk"),
        ("Average Claim Amount", float(record.get("claimAmount",0)) / 50000, "Increases Risk"),
        ("Beneficiary Count", float(record.get("beneficiaries",120)) / 1000, "Decreases Risk"),
        ("Service Frequency", float(record.get("serviceFrequency",3.2)) / 10, "Increases Risk"),
    ]
    vals = []
    for feature, raw, direction in candidates:
        contribution = round(min(max(raw,0.02),0.48), 2)
        vals.append({"Feature": feature, "Contribution": contribution, "Direction": direction})
    return pd.DataFrame(vals)
